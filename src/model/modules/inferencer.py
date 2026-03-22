from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .feature_extractor import FeatureExtractor
from .normalizer import Normalizer
from .spatial_temporal_cnn import SpatialTemporalCNN
from src.apex.modules.v2 import ApexPhase

logger = logging.getLogger(__name__)


# ── Result dataclasses ─────────────────────────────────────────────────


@dataclass(frozen=True)
class FeatureContribution:
    """A single feature's contribution to the prediction.

    Attributes:
        name: Human-readable feature name (e.g. ``"apex1_lips_mean_mag"``).
        value: Raw (un-normalised) feature value.
        norm_value: Normalised feature value.
        saliency: Gradient × input saliency score.
        direction: ``"high"`` or ``"low"`` — the class this feature pushes toward.
    """

    name: str
    value: float
    norm_value: float
    saliency: float
    direction: str


@dataclass
class InferenceResult:
    """Structured output from a single-clip inference.

    Attributes:
        label: Predicted class label (``"high"``, ``"low"``, or ``"error"``).
        confidence: Confidence of the predicted class (0.0–1.0).
        prob_high: Softmax probability for the ``"high"`` class.
        prob_low: Softmax probability for the ``"low"`` class.
        top_features: Ranked list of feature contributions.
        feature_vector: Raw feature vector of shape ``(78,)``, or ``None`` on error.
        n_apex_detected: Number of apex frames detected in the clip.
        warning: Optional warning message about prediction reliability.
    """

    label: str
    confidence: float
    prob_high: float
    prob_low: float
    top_features: list[FeatureContribution]
    feature_vector: np.ndarray | None
    n_apex_detected: int
    warning: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to a plain dictionary for serialisation.

        Returns:
            A JSON-friendly dictionary representation.
        """
        d = asdict(self)
        # ``asdict`` converts ndarray to a nested list; keep it as-is
        d["feature_vector"] = self.feature_vector
        return d


@dataclass
class BatchResult:
    """Aggregated results from :meth:`AnxietyInferencer.predict_batch`.

    Attributes:
        succeeded: Mapping of index → result for successful predictions.
        failed: Mapping of index → exception for failed predictions.
    """

    succeeded: dict[int, InferenceResult] = field(default_factory=dict)
    failed: dict[int, Exception] = field(default_factory=dict)

    @property
    def success_count(self) -> int:
        """Number of successfully processed items."""
        return len(self.succeeded)

    @property
    def failure_count(self) -> int:
        """Number of failed items."""
        return len(self.failed)

    @property
    def all_succeeded(self) -> bool:
        """Whether all items processed without error."""
        return len(self.failed) == 0


# ── Progress callback type ─────────────────────────────────────────────

ProgressCallback = Callable[[int, int, str], None]
"""``(current_index, total, status_message) -> None``"""


# ── Feature-name builder (cached, no module-level side-effect) ─────────


@functools.lru_cache(maxsize=1)
def _build_feature_names() -> list[str]:
    """Build the canonical 78-element feature-name list.

    The result is cached so the ``FeatureExtractor`` is only
    instantiated once, and never at module-import time.

    Returns:
        List of 78 human-readable feature names.
    """
    fe = FeatureExtractor()
    names: list[str] = []
    for k in range(fe.K_APEX):
        for r in fe.ROI_ORDER:
            for f in ["mean_mag", "max_mag", "net_dx", "net_dy"]:
                names.append(f"apex{k + 1}_{r}_{f}")
    pairs = [
        ("eye", ("left_eye", "right_eye")),
        ("eyebrow", ("left_eyebrow", "right_eyebrow")),
    ]
    for k in range(fe.K_APEX):
        for pair_name, _ in pairs:
            for f in ["d_mag", "d_dx", "d_dy"]:
                names.append(f"apex{k + 1}_{pair_name}_LR_{f}")
    return names


# ── Main inferencer ────────────────────────────────────────────────────


class AnxietyInferencer:
    """Predict anxiety level (high / low) from a single video clip.

    Supports two input sources:

    1. Pre-extracted optical-flow ``.npy`` files.
    2. Raw video files (``.mp4``, ``.avi``, etc.) — extracted on the fly.

    Attributes:
        model_path: Path to the checkpoint used by this instance.
        device: Torch device for inference.
        feature_names: Canonical list of 78 feature names.

    Example:
        >>> inf = AnxietyInferencer("model.pt", "norm.npz")
        >>> result = inf.predict_npy("clip.npy")
        >>> print(result.label, result.confidence)
    """

    LABEL_MAP: dict[int, str] = {0: "low", 1: "high"}

    def __init__(
        self,
        checkpoint_path: str | Path,
        normalizer_path: str | Path | None = None,
        device: str | None = None,
        cutoff_ratio: float = 0.30,
    ) -> None:
        """Initialise the inferencer.

        Args:
            checkpoint_path: Path to a ``.pt`` model checkpoint.
            normalizer_path: Path to a ``.npz`` normaliser stats file.
                If ``None``, normalisation is skipped (not recommended
                for production).
            device: ``"cuda"`` or ``"cpu"``.  Defaults to CUDA when
                available.
            cutoff_ratio: ``ApexPhase`` onset/offset detection parameter.

        Raises:
            FileNotFoundError: If *checkpoint_path* or *normalizer_path*
                does not exist.
            ValueError: If *cutoff_ratio* is outside ``(0, 1)``.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        if not 0 < cutoff_ratio < 1:
            raise ValueError(
                f"'cutoff_ratio' must be in (0, 1), got {cutoff_ratio}"
            )

        self.model_path: str = str(checkpoint_path)
        self.cutoff_ratio: float = cutoff_ratio
        self.feature_names: list[str] = _build_feature_names()

        # Device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Model
        self._model = SpatialTemporalCNN().to(self.device)
        state = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self._model.load_state_dict(state)
        self._model.eval()

        # Prevent gradient tracking/memory leakage on weights
        for param in self._model.parameters():
            param.requires_grad = False

        # Normalizer (optional)
        self._normalizer: Normalizer | None = None
        if normalizer_path is not None:
            normalizer_path = Path(normalizer_path)
            if not normalizer_path.exists():
                raise FileNotFoundError(
                    f"Normalizer not found: {normalizer_path}"
                )
            self._normalizer = Normalizer.load(str(normalizer_path))

        # Extraction components
        self._extractor = FeatureExtractor()
        self._apex_detector = ApexPhase()

        logger.info("AnxietyInferencer ready")
        logger.info("  Checkpoint : %s", checkpoint_path.name)
        logger.info(
            "  Normalizer : %s",
            "loaded" if self._normalizer else "NOT loaded (not recommended)",
        )
        logger.info("  Device     : %s", self.device)

    # ── Public: prediction ─────────────────────────────────────────────

    def predict_npy(self, npy_path: str | Path) -> InferenceResult:
        """Predict from a pre-extracted optical-flow ``.npy`` file.

        Args:
            npy_path: Path to the ``.npy`` file containing ``frames``
                and ``magnitudes`` keys.

        Returns:
            Structured inference result.

        Raises:
            FileNotFoundError: If *npy_path* does not exist.
        """
        npy_path = Path(npy_path)
        if not npy_path.exists():
            raise FileNotFoundError(f".npy file not found: {npy_path}")

        loaded = np.load(str(npy_path), allow_pickle=True).item()
        roi_frames = loaded["frames"]
        magnitudes = np.asarray(loaded["magnitudes"], dtype=np.float32)

        return self._predict_from_frames(roi_frames, magnitudes)

    def predict_video(
        self,
        video_path: str | Path,
        roi_extractor: Callable[..., tuple[list, np.ndarray]] | None = None,
    ) -> InferenceResult:
        """Predict from a raw video file.

        Args:
            video_path: Path to a video file (``.mp4``, ``.avi``, etc.).
            roi_extractor: Optional custom function for ROI extraction.
                Signature: ``(video_path) -> (roi_frames, magnitudes)``.
                Uses the built-in TV-L1 pipeline when ``None``.

        Returns:
            Structured inference result.

        Raises:
            FileNotFoundError: If *video_path* does not exist.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if roi_extractor is not None:
            roi_frames, magnitudes = roi_extractor(str(video_path))
        else:
            roi_frames, magnitudes = self._extract_from_video(str(video_path))

        return self._predict_from_frames(roi_frames, magnitudes)

    def predict_batch(
        self,
        paths: list[str | Path],
        input_type: str = "npy",
        on_progress: ProgressCallback | None = None,
    ) -> BatchResult:
        """Predict for multiple clips, tracking per-item failures.

        Individual clip failures are captured in
        :attr:`BatchResult.failed` rather than aborting the entire batch.

        Args:
            paths: List of file paths to process.
            input_type: ``"npy"`` or ``"video"``.
            on_progress: Optional callback receiving
                ``(current, total, status)`` after each item.

        Returns:
            A :class:`BatchResult` with succeeded and failed items
            keyed by index.

        Raises:
            ValueError: If *input_type* is not ``"npy"`` or ``"video"``.
        """
        valid_types = ("npy", "video")
        if input_type not in valid_types:
            raise ValueError(
                f"'input_type' must be one of {valid_types}, "
                f"got '{input_type}'"
            )

        batch = BatchResult()
        total = len(paths)

        for idx, p in enumerate(paths):
            p = Path(p)
            if on_progress:
                on_progress(idx, total, f"Processing {p.name}")

            try:
                if input_type == "npy":
                    result = self.predict_npy(p)
                else:
                    result = self.predict_video(p)

                batch.succeeded[idx] = result
                logger.info(
                    "  [%d/%d] %s -> %s (%.3f)",
                    idx + 1, total, p.name,
                    result.label, result.confidence,
                )
            except Exception as exc:
                batch.failed[idx] = exc
                logger.error(
                    "  [%d/%d] %s -> ERROR: %s",
                    idx + 1, total, p.name, exc,
                )

        if on_progress:
            on_progress(total, total, "Complete")

        if batch.failure_count:
            logger.warning(
                "Batch completed with %d/%d failures",
                batch.failure_count, total,
            )

        return batch

    # ── Public: inspection ─────────────────────────────────────────────

    def explain(self, result: InferenceResult, top_n: int = 10) -> str:
        """Format an inference result as a human-readable report.

        Args:
            result: Output from :meth:`predict_npy` or :meth:`predict_video`.
            top_n: Number of top contributing features to include.

        Returns:
            Multi-line formatted string.
        """
        lines = [
            "=" * 55,
            "ANXIETY LEVEL PREDICTION RESULT",
            "=" * 55,
            f"  Label     : {result.label.upper()}",
            f"  Confidence: {result.confidence:.1%}",
            f"  Prob HIGH : {result.prob_high:.4f}",
            f"  Prob LOW  : {result.prob_low:.4f}",
            f"  Apex det. : {result.n_apex_detected}",
        ]

        if result.warning:
            lines.append(f"  ⚠ Warning  : {result.warning}")

        lines += [
            "",
            f"Top {top_n} Contributing Features:",
            f"  {'Feature':<35} {'Value':>8}  {'Direction'}",
            "  " + "-" * 55,
        ]
        for feat in result.top_features[:top_n]:
            arrow = "→ HIGH" if feat.direction == "high" else "→ LOW "
            lines.append(
                f"  {feat.name:<35} {feat.value:>8.4f}  {arrow}"
            )

        lines.append("=" * 55)
        return "\n".join(lines)

    # ── Private: core inference ────────────────────────────────────────

    def _predict_from_frames(
        self,
        roi_frames: list,
        magnitudes: np.ndarray,
    ) -> InferenceResult:
        """Run the full inference pipeline on pre-extracted frames.

        Args:
            roi_frames: List of per-frame ROI dicts.
            magnitudes: Per-frame magnitude signal.

        Returns:
            Structured inference result.
        """
        warning: str | None = None

        # Detect apex frames
        apex_indices = self._apex_detector.find_top_k_apex(
            signal=magnitudes, k=self._extractor.K_APEX,
        )
        n_apex = len(apex_indices)

        if n_apex == 0:
            warning = (
                "No apex frames detected. "
                "Prediction may not be reliable."
            )

        phases = self._apex_detector.find_phase(
            signal=magnitudes,
            apex_indices=apex_indices,
            cutoff_ratio=self.cutoff_ratio,
        )

        # Feature extraction
        feat_raw = self._extractor.extract(roi_frames, apex_indices, phases)

        # Normalisation
        if self._normalizer is not None:
            feat_norm = self._normalizer.transform(
                feat_raw.reshape(1, -1),
            ).squeeze(0)
        else:
            feat_norm = feat_raw
            if warning is None:
                warning = (
                    "Normalizer not loaded. "
                    "Results may be inaccurate without normalisation."
                )

        # Forward pass
        x = torch.from_numpy(feat_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        label = self.LABEL_MAP[pred_idx]
        confidence = float(probs[pred_idx])

        # Feature explanation
        top_features = self._explain_features(feat_raw, feat_norm, pred_idx)

        return InferenceResult(
            label=label,
            confidence=confidence,
            prob_high=float(probs[1]),
            prob_low=float(probs[0]),
            top_features=top_features,
            feature_vector=feat_raw,
            n_apex_detected=n_apex,
            warning=warning,
        )

    def _explain_features(
        self,
        feat_raw: np.ndarray,
        feat_norm: np.ndarray,
        pred_idx: int,
    ) -> list[FeatureContribution]:
        """Compute per-feature saliency via gradient × input.

        A lightweight interpretability method that does not require
        external libraries.

        Args:
            feat_raw: Raw feature vector of shape ``(78,)``.
            feat_norm: Normalised feature vector of shape ``(78,)``.
            pred_idx: Index of the predicted class (0 or 1).

        Returns:
            Top-10 features ranked by saliency, descending.
        """
        x = torch.from_numpy(feat_norm).unsqueeze(0).to(self.device)
        x.requires_grad_(True)

        logits = self._model(x)
        self._model.zero_grad()
        logits[0, pred_idx].backward()

        grads = x.grad.cpu().numpy()[0]            # (78,)
        saliency = np.abs(grads * feat_norm)       # gradient × input
        ranked_idx = np.argsort(saliency)[::-1]    # descending

        top_features: list[FeatureContribution] = []
        for i in ranked_idx[:10]:
            # Direction: positive gradient → pushes toward pred_idx
            if pred_idx == 1:
                direction = "high" if grads[i] > 0 else "low"
            else:
                direction = "low" if grads[i] > 0 else "high"

            top_features.append(FeatureContribution(
                name=self.feature_names[i],
                value=float(feat_raw[i]),
                norm_value=float(feat_norm[i]),
                saliency=float(saliency[i]),
                direction=direction,
            ))

        return top_features

    def _extract_from_video(self, video_path: str) -> tuple[list, np.ndarray]:
        """Extract optical flow from a raw video using ApexPhaseSpotter.

        Pipeline:
            1. Process video using ApexPhaseSpotter.
            2. Export flow data matching the .npy format.

        Args:
            video_path: Path to the video file.

        Returns:
            Tuple of ``(roi_frames, magnitudes)``.

        Raises:
            ImportError: If ApexPhaseSpotter is unavailable.
        """
        try:
            from src.apex.modules.v2.apex_phase_spotter import ApexPhaseSpotter
        except ImportError as exc:
            raise ImportError(
                f"ApexPhaseSpotter module not found: {exc}"
            ) from exc

        spotter = ApexPhaseSpotter()
        spotter.process(video_path)
        flow_data = spotter.export_flow_data()
        
        roi_frames = flow_data["frames"]
        magnitudes = np.array(flow_data["magnitudes"], dtype=np.float32)

        return roi_frames, magnitudes

    @staticmethod
    def _error_result(message: str) -> InferenceResult:
        """Build a sentinel result for a failed prediction.

        Args:
            message: Human-readable error description.

        Returns:
            An :class:`InferenceResult` with ``label="error"`` and
            zeroed probabilities.
        """
        return InferenceResult(
            label="error",
            confidence=0.0,
            prob_high=0.0,
            prob_low=0.0,
            top_features=[],
            feature_vector=None,
            n_apex_detected=0,
            warning=message,
        )

    # ── Dunder ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        norm_status = "with normalizer" if self._normalizer else "no normalizer"
        return (
            f"AnxietyInferencer("
            f"ckpt='{os.path.basename(self.model_path)}', "
            f"{norm_status}, device={self.device})"
        )
