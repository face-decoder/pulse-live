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

LABEL_MAP: dict[int, str] = {0: "anxiety_rendah", 1: "anxiety_tinggi"}


class BaseAnxietyInferencer(ABC):
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

        self._model: nn.Module | None = None
        self._threshold: float = 0.5
        self._transform = None

        self.__load_checkpoint()
        logger.info(
            "%s ready | ckpt=%s | device=%s | threshold=%.3f",
            self.__class__.__name__,
            self.checkpoint_path.name,
            self.device,
            self._threshold,
        )

    @abstractmethod
    def build_model(self) -> nn.Module: ...

    def predict_npz(self, npz_path: str | Path) -> InferenceResult:
        data = np.load(npz_path, allow_pickle=False)
        flow = data["flow"].astype(np.float32)
        return self.predict_flow(flow)

    def predict_flow(self, flow: np.ndarray) -> InferenceResult:
        self.__ensure_ready()
        return self._run_pipeline(flow)

    def _run_pipeline(self, flow: np.ndarray) -> InferenceResult:
        import time

        from src.dataset.modules.augment_flow import AugmentFlow
        from src.dataset.modules.behavioral_features import BehavioralFeatures
        from src.dataset.modules.compose import Compose
        from src.dataset.modules.subject_sample import SubjectSample, TransformOutput
        from src.dataset.modules.temporal_transforms import PadAndMask
        from src.dataset.modules.window_selector import (
            ApexWindowDetector,
            WindowSelector,
        )

        detector = ApexWindowDetector(
            percentile=self.detector_percentile,
            prominence=self.detector_prominence,
            max_window=self.max_seq_len,
        )

        spotting_start = time.time()
        windows, meta = detector.detect_windows(flow, phase_mode="onset_to_apex")
        spotting_latency_ms = (time.time() - spotting_start) * 1000
        n_windows = len(windows)

        warning: str | None = None
        if n_windows == 0:
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

        transform = Compose(
            [
                WindowSelector(phase_includes=self.phases),
                BehavioralFeatures(),
                PadAndMask(max_len=self.max_seq_len),
                AugmentFlow(training=False),
            ]
        )

        sample = SubjectSample(
            subject_id="inference",
            flow=flow,
            windows=windows,
            label=0,
            meta={},
        )
        out: TransformOutput = transform(sample)

        x = out.x.unsqueeze(0).to(self.device)
        mask = out.mask.unsqueeze(0).to(self.device) if out.mask is not None else None

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
        mask: torch.Tensor | None = None,
    ) -> float:
        assert self._model is not None
        self._model.eval()

        total = 0.0
        with torch.no_grad():
            for _ in range(self.n_tta):
                scale = torch.empty(1, device=x.device).uniform_(0.93, 1.07)
                scale = scale.view(1, *([1] * (x.ndim - 1)))
                x_aug = x * scale + torch.randn_like(x) * 0.02
                logits = self._model(x_aug, mask=mask)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                total += prob

        return total / float(self.n_tta)

    def __load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

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

    def __ensure_ready(self) -> None:
        if self._model is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has no loaded model. "
                "Check that the checkpoint path is valid."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"ckpt='{self.checkpoint_path.name}', "
            f"device={self.device}, "
            f"threshold={self._threshold:.3f})"
        )
