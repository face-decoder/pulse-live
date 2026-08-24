from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from .base import LABEL_MAP, BaseAnxietyInferencer
from .result import InferenceResult

logger = logging.getLogger(__name__)

_SPATIO_TEMPORAL_IN_CHANNELS: int = 10


class SpatioTemporalInferencer(BaseAnxietyInferencer):
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

    def _run_pipeline(self, flow: np.ndarray) -> InferenceResult:
        import time

        from src.apex.modules import ApexPhaseSpotter

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

        slices = []
        for left, apex, right in windows:
            parts = []
            if "onset" in self.phases and apex > left:
                parts.append(flow[left:apex])
            parts.append(flow[apex : apex + 1])
            if "offset" in self.phases and right > apex + 1:
                parts.append(flow[apex + 1 : right])
            if parts:
                slices.append(np.concatenate(parts, axis=0))

        if not slices:
            slices = [flow[:1]]

        merged = np.concatenate(slices, axis=0)[: self.max_seq_len]
        T, N_roi, C, H, W = merged.shape

        x = (
            torch.from_numpy(merged.astype(np.float32))
            .permute(1, 2, 0, 3, 4)
            .reshape(N_roi * C, T, H, W)
            .unsqueeze(0)
            .to(self.device)
        )

        inference_start = time.time()
        prob_high = self._tta_forward(x)
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
