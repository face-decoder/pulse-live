"""Inferencer for the SpatioTemporalCNN architecture (series 01xx, 03xx).

Unlike the other architectures, ``SpatioTemporalCNN`` operates directly on
**raw optical flow** tensors ``(T, N_roi, 2, H, W)`` — it does *not* use
``BehavioralFeatures``.  The model treats ``N_roi × 2 = 10`` as the channel
dimension and ``(H, W)`` as the spatial dimensions of a 3-D convolution.

The inference pipeline is therefore different from the base class:
    NPZ → ApexPhaseSpotter → window slicing → raw-flow stacking
    → SpatioTemporalCNN.forward() → TTA → Label

``BehavioralFeatures``, ``PadAndMask``, and ``AugmentFlow`` are **not** used.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Any

import numpy as np
import torch
import torch.nn as nn

from .base import LABEL_MAP, BaseAnxietyInferencer
from .result import InferenceResult

logger = logging.getLogger(__name__)

# SpatioTemporalCNN uses N_roi × 2 = 5 × 2 = 10 raw channels
_SPATIO_TEMPORAL_IN_CHANNELS: int = 10


class SpatioTemporalInferencer(BaseAnxietyInferencer):
    """Inferencer for :class:`SpatioTemporalCNN` (raw-flow pipeline).

    Combination codes: **0101**, **0201**, **0301**, **0401** and their
    corresponding checkpoints.

    The model receives raw flow tiles ``(B, N_roi*2, T, H, W)`` — no
    behavioural feature extraction step.

    Args:
        checkpoint_path: Path to ``best_model.pt``.
        device: Torch device.
        max_seq_len: Maximum temporal window length.
        n_tta: Number of TTA passes.
        phases: Phase slices to include.
        detector_percentile: Legacy compatibility parameter.
        detector_prominence: Legacy compatibility parameter.
        prefer_checkpoint_tta: Prefer checkpoint n_tta for notebook parity.
        tile_h: ROI tile height (must match training, default 64).
        tile_w: ROI tile width (must match training, default 64).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        max_seq_len: int = BaseAnxietyInferencer.DEFAULT_MAX_SEQ_LEN,
        n_tta: int = BaseAnxietyInferencer.DEFAULT_N_TTA,
        phases: Sequence[str] = BaseAnxietyInferencer.DEFAULT_PHASES,
        detector_percentile: float = BaseAnxietyInferencer.DEFAULT_DETECTOR_PERCENTILE,
        detector_prominence: float = BaseAnxietyInferencer.DEFAULT_DETECTOR_PROMINENCE,
        prefer_checkpoint_tta: bool = True,
        tile_h: int = 64,
        tile_w: int = 64,
        **kwargs: Any,
    ) -> None:
        self.tile_h = tile_h
        self.tile_w = tile_w
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            max_seq_len=max_seq_len,
            n_tta=n_tta,
            phases=phases,
            detector_percentile=detector_percentile,
            detector_prominence=detector_prominence,
            prefer_checkpoint_tta=prefer_checkpoint_tta,
            **kwargs,
        )

    def build_model(self) -> nn.Module:
        from src.models.modules.spatio_temporal.spatio_temporal_cnn import (
            SpatioTemporalCNN,
        )

        return SpatioTemporalCNN(
            in_channels=_SPATIO_TEMPORAL_IN_CHANNELS,
            num_classes=2,
        )

    # ── Override: raw-flow pipeline (no BehavioralFeatures) ────────────────

    def _run_pipeline(self, flow: np.ndarray) -> InferenceResult:
        """Raw-flow pipeline for SpatioTemporalCNN.

        Args:
            flow: ``(T, N_roi, 2, H, W)`` optical flow array.

        Returns:
            :class:`InferenceResult`.
        """
        from src.apex.modules import ApexPhaseSpotter
        import time

        detector = ApexPhaseSpotter(
            percentile=self.detector_percentile,
            prominence=self.detector_prominence,
        )

        spotting_start = time.time()
        windows, _ = detector.detect_windows(flow, phase_mode="full")
        spotting_latency_ms = (time.time() - spotting_start) * 1000
        n_windows = len(windows)

        warning: str | None = None
        if n_windows == 0:
            warning = "No apex windows detected; using full clip."
            T = flow.shape[0]
            apex = T // 2
            windows = [(0, apex, T)]

        # Slice and stack window clips → (T_clipped, N_roi, 2, H, W)
        slices = []
        for left, apex, right in windows:
            parts = []
            if "onset" in self.phases and apex > left:
                parts.append(flow[left:apex])
            parts.append(flow[apex: apex + 1])  # apex always included
            if "offset" in self.phases and right > apex + 1:
                parts.append(flow[apex + 1: right])
            if parts:
                slices.append(np.concatenate(parts, axis=0))

        if not slices:
            slices = [flow[:1]]

        merged = np.concatenate(slices, axis=0)[: self.max_seq_len]
        # merged: (T, N_roi, 2, H, W)
        T, N_roi, C, H, W = merged.shape

        # Reshape to (1, N_roi*C, T, H, W) for Conv3D
        x = (
            torch.from_numpy(merged.astype(np.float32))
            .permute(1, 2, 0, 3, 4)       # (N_roi, C, T, H, W)
            .reshape(N_roi * C, T, H, W)   # (N_roi*C, T, H, W)
            .unsqueeze(0)                  # (1, N_roi*C, T, H, W)
            .to(self.device)
        )

        inference_start = time.time()
        prob_high = self._tta_forward_raw(x)
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

    def _tta_forward_raw(self, x: torch.Tensor) -> float:
        """TTA for raw-flow 5D tensors (no mask support needed)."""
        assert self._model is not None
        self._model.eval()
        total = 0.0
        with torch.no_grad():
            for _ in range(self.n_tta):
                scale = torch.empty(1, device=x.device).uniform_(0.93, 1.07)
                scale = scale.view(1, *([1] * (x.ndim - 1)))
                x_aug = x * scale + torch.randn_like(x) * 0.02
                logits = self._model(x_aug)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                total += prob
        return total / float(self.n_tta)
