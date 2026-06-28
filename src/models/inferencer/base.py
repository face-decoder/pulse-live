"""Abstract base class for all anxiety inference pipelines.

Pipeline (behavioural-feature models, series 01–04xx except SpatioTemporalCNN):
    NPZ file
    └─ ApexPhaseSpotter            detect apex micro-expression windows
    └─ WindowSelector              slice onset+apex frames
    └─ BehavioralFeatures          extract 47-ch statistical features
    └─ PadAndMask                  pad to max_seq_len, build mask
    └─ AugmentFlow(training=False) identity at inference time
    └─ Model.forward(x, mask)      (B, C, T) → logits
    └─ TTA averaging               average over N augmented forward passes
    └─ Threshold → InferenceResult

For SpatioTemporalCNN (raw-flow pipeline), override ``_run_pipeline``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .result import InferenceResult

logger = logging.getLogger(__name__)

# ── Labels ──────────────────────────────────────────────────────────────────

LABEL_MAP: dict[int, str] = {0: "anxiety_rendah", 1: "anxiety_tinggi"}


# ── Base class ───────────────────────────────────────────────────────────────


class BaseAnxietyInferencer(ABC):
    """Abstract base for anxiety-level inference from optical flow NPZ files.

    Subclasses only need to implement :meth:`build_model`.  All pipeline
    stages (window detection, feature extraction, TTA forward pass,
    thresholding) are handled here.

    Args:
        checkpoint_path: Path to a ``best_model.pt`` checkpoint saved by the
            combination notebooks.
        device: Torch device string or ``torch.device``.
        max_seq_len: Temporal padding length (must match training config,
            default 512).
        n_tta: Number of test-time augmentation passes (default 8).
        phases: Phase slices to include; ``["onset", "apex"]`` by default.
        detector_percentile: Legacy compatibility parameter.
        detector_prominence: Legacy compatibility parameter.
    """

    # ── Defaults matching all combination notebooks ─────────────────────────
    DEFAULT_MAX_SEQ_LEN: int = 512
    DEFAULT_N_TTA: int = 8
    DEFAULT_PHASES: tuple[str, ...] = ("onset", "apex")
    DEFAULT_DETECTOR_PERCENTILE: float = 95.0
    DEFAULT_DETECTOR_PROMINENCE: float = 0.1

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        n_tta: int = DEFAULT_N_TTA,
        phases: Sequence[str] = DEFAULT_PHASES,
        detector_percentile: float = DEFAULT_DETECTOR_PERCENTILE,
        detector_prominence: float = DEFAULT_DETECTOR_PROMINENCE,
        prefer_checkpoint_tta: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.max_seq_len = int(max_seq_len)
        self.n_tta = int(n_tta)
        self.phases = list(phases)
        self.detector_percentile = float(detector_percentile)
        self.detector_prominence = float(detector_prominence)
        self.prefer_checkpoint_tta = bool(prefer_checkpoint_tta)

        # Built lazily in _ensure_ready()
        self._model: nn.Module | None = None
        self._threshold: float = 0.5
        self._transform = None

        self._load_checkpoint()
        logger.info(
            "%s ready | ckpt=%s | device=%s | threshold=%.3f",
            self.__class__.__name__,
            self.checkpoint_path.name,
            self.device,
            self._threshold,
        )

    # ── Abstract interface ──────────────────────────────────────────────────

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Construct and return the model (un-initialised weights).

        The base class will load weights from the checkpoint automatically.
        """

    # ── Public API ──────────────────────────────────────────────────────────

    def predict_npz(self, npz_path: str | Path) -> InferenceResult:
        """Run inference on a pre-processed optical flow NPZ file.

        The NPZ must contain a ``"flow"`` key with shape
        ``(T, N_roi, 2, H, W)`` — the format produced by the v12 prefetch
        pipeline.

        Args:
            npz_path: Path to the ``.npz`` cache file.

        Returns:
            :class:`InferenceResult` with label, probabilities, and metadata.
        """
        data = np.load(npz_path, allow_pickle=False)
        flow = data["flow"].astype(np.float32)  # (T, N_roi, 2, H, W)
        return self.predict_flow(flow)

    def predict_flow(self, flow: np.ndarray) -> InferenceResult:
        """Run inference on a raw flow array ``(T, N_roi, 2, H, W)``.

        Args:
            flow: Numpy array of optical flow frames in ROI format.

        Returns:
            :class:`InferenceResult`.
        """
        self._ensure_ready()
        return self._run_pipeline(flow)

    # ── Internal pipeline ───────────────────────────────────────────────────

    def _run_pipeline(self, flow: np.ndarray) -> InferenceResult:
        """Full pipeline: window detection → features → TTA → label."""
        from src.dataset.modules.augment_flow import AugmentFlow
        from src.dataset.modules.behavioral_features import BehavioralFeatures
        from src.dataset.modules.compose import Compose
        from src.dataset.modules.temporal_transforms import PadAndMask
        from src.dataset.modules.window_selector import ApexWindowDetector, WindowSelector
        from src.dataset.modules.subject_sample import SubjectSample, TransformOutput

        import time

        # Use the same ApexWindowDetector as the training notebook
        # (percentile=95, prominence=0.5 → matching DETECTOR_PERCENTILE / DETECTOR_PROMINENCE)
        detector = ApexWindowDetector(
            percentile=self.detector_percentile,
            prominence=self.detector_prominence,
            max_window=self.max_seq_len,
        )

        # Detect apex windows — phase_mode="onset_to_apex" matches the training pipeline
        spotting_start = time.time()
        windows, meta = detector.detect_windows(flow, phase_mode="onset_to_apex")
        spotting_latency_ms = (time.time() - spotting_start) * 1000
        n_windows = len(windows)

        warning: str | None = None
        if n_windows == 0:
            # ponytail: if there is no micro-expression (no movement), the optical flow is ~zero.
            # Feeding all-zeros to the network causes its biases to default to anxiety_tinggi.
            # Fix: instantly short-circuit and assume 'normal' (anxiety_rendah) if face is perfectly still.
            return InferenceResult(
                label="anxiety_rendah",
                prob_high=0.0,
                prob_low=1.0,
                confidence=1.0,
                threshold=self._threshold,
                n_windows=0,
                warning="No apex windows detected; short-circuiting to anxiety_rendah.",
                spotting_latency_ms=spotting_latency_ms,
                model_inference_latency_ms=0.0,
            )

        # Build transform chain (identical to validation_transform in all notebooks)
        transform = Compose([
            WindowSelector(phase_includes=self.phases),
            BehavioralFeatures(),
            PadAndMask(max_len=self.max_seq_len),
            AugmentFlow(training=False),
        ])

        # Build a dummy SubjectSample (subject_id/label unused during inference)
        sample = SubjectSample(
            subject_id="inference",
            flow=flow,
            windows=windows,
            label=0,
            meta={},
        )
        out: TransformOutput = transform(sample)

        # out.x from PadAndMask is (C, T) — unsqueeze batch dim → (1, C, T)
        x = out.x.unsqueeze(0).to(self.device)  # (1, C, T)
        mask = out.mask.unsqueeze(0).to(self.device) if out.mask is not None else None

        # TTA forward pass
        inference_start = time.time()
        prob_high = self._tta_forward(x, mask)
        model_latency_ms = (time.time() - inference_start) * 1000
        prob_low = 1.0 - prob_high

        label_idx = int(prob_high >= self._threshold)
        label = LABEL_MAP[label_idx]
        confidence = prob_high if label_idx == 1 else prob_low

        return InferenceResult(
            label=label,
            prob_high=prob_high,
            prob_low=prob_low,
            confidence=confidence,
            threshold=self._threshold,
            n_windows=n_windows,
            warning=warning,
            spotting_latency_ms=spotting_latency_ms,
            model_inference_latency_ms=model_latency_ms,
        )

    def _tta_forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> float:
        """Test-time augmentation: average softmax P(high) over N passes.

        Matches ``tta_predict_positive_proba`` from the combination notebooks.

        Args:
            x: Input tensor ``(1, C, T)``.
            mask: Padding mask ``(1, T)`` or ``None``.

        Returns:
            Averaged probability for the *high-anxiety* class.
        """
        assert self._model is not None
        self._model.eval()

        total = 0.0
        with torch.no_grad():
            for _ in range(self.n_tta):
                # Mild scale + noise augmentation (matches notebook TTA)
                scale = torch.empty(1, device=x.device).uniform_(0.93, 1.07)
                scale = scale.view(1, *([1] * (x.ndim - 1)))
                x_aug = x * scale + torch.randn_like(x) * 0.02
                logits = self._model(x_aug, mask=mask)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                total += prob

        return total / float(self.n_tta)

    def _load_checkpoint(self) -> None:
        """Load model weights and best_threshold from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        ck = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        model = self.build_model().to(self.device)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        self._model = model

        self._threshold = float(ck.get("best_threshold", 0.5))
        if self.prefer_checkpoint_tta:
            ck_n_tta = ck.get("n_tta")
            if ck_n_tta is not None:
                try:
                    n_tta = int(ck_n_tta)
                    if n_tta > 0:
                        self.n_tta = n_tta
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid checkpoint n_tta=%r for %s; keeping runtime n_tta=%d",
                        ck_n_tta,
                        self.checkpoint_path.name,
                        self.n_tta,
                    )

    def _ensure_ready(self) -> None:
        """Guard: raise if model was not loaded correctly."""
        if self._model is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has no loaded model. "
                "Check that the checkpoint path is valid."
            )

    # ── Dunder ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"ckpt='{self.checkpoint_path.name}', "
            f"device={self.device}, "
            f"threshold={self._threshold:.3f})"
        )
