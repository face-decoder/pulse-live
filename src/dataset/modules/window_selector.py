from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from ..constants.index import ROI_ORDER_DEFAULT

from .base_transform import BaseTransform
from .subject_sample import SubjectSample, TransformOutput
from ..utils.pipeline_utils import LABEL_MAP


class ApexWindowDetector:
    """
    Detect apex windows from optical flow and return with 5D (T, N_roi, 2, H, W).

    Moved here from apex_window_detector.py so the detector lives alongside the
    window selection transform.
    """

    def __init__(
        self,
        percentile: float = 70,
        prominence: float = 0.05,
        min_distance: int = 5,
        ratio: float = 0.30,
        min_window: int = 3,
        max_window: int = 200,
        context: int = 5,
        smooth_sigma: float = 1.5,
    ):
        self.percentile = float(percentile)
        self.prominence = float(prominence)
        self.min_distance = int(min_distance)
        self.ratio = float(ratio)
        self.min_window = int(min_window)
        self.max_window = int(max_window)
        self.context = int(context)
        self.smooth_sigma = float(smooth_sigma)

    def detect_windows(
        self,
        flow: np.ndarray,
        phase_mode: str = "onset_to_apex",
        selected_rois: Optional[Sequence[str]] = None,
    ) -> Tuple[List[Tuple[int, int, int]], dict]:
        T = flow.shape[0]
        dx = flow[:, :, 0, :, :].mean(axis=(2, 3))
        dy = flow[:, :, 1, :, :].mean(axis=(2, 3))

        if selected_rois is not None:
            roi_idx = np.array([ROI_ORDER_DEFAULT.index(r) for r in selected_rois])
            dx = dx[:, roi_idx]
            dy = dy[:, roi_idx]

        magnitude_1d = np.sqrt(dx**2 + dy**2).mean(axis=1)
        apex_signal = np.log1p(magnitude_1d)

        epsilon = np.percentile(apex_signal, 10)
        apex_signal_clean = apex_signal.copy()
        apex_signal_clean[apex_signal_clean < epsilon] = 0.0

        smoothed = gaussian_filter1d(apex_signal_clean, sigma=self.smooth_sigma)
        threshold = max(np.percentile(smoothed, self.percentile), np.std(smoothed) * 0.5)
        peaks, _ = find_peaks(
            smoothed,
            height=threshold,
            prominence=self.prominence,
            distance=self.min_distance,
        )

        if len(peaks) == 0:
            return [], {"valid": False, "reason": "no_peaks"}

        windows: List[Tuple[int, int, int]] = []
        for p in peaks:
            if phase_mode == "onset_to_apex":
                left = self._find_onset(smoothed, p)
                right = p + 1
            else:
                left, right = self._find_phase(smoothed, p)

            length = right - left
            if length < self.min_window:
                continue
            if length > self.max_window:
                half = self.max_window // 2
                left = max(0, p - half)
                right = min(T, p + half)

            left = max(0, left - self.context)
            if phase_mode != "onset_to_apex":
                right = min(T, right + self.context)

            windows.append((left, int(p), right))

        if not windows:
            return [], {"valid": False, "reason": "no_valid_windows", "num_peaks": len(peaks)}

        apex_vals = smoothed[[p for _, p, _ in windows]]
        confidence = float(np.mean(np.abs(apex_vals)) / (np.mean(np.abs(smoothed)) + 1e-6))
        return windows, {"valid": True, "num_peaks": len(peaks), "num_windows": len(windows), "confidence": confidence}

    def _find_onset(self, signal: np.ndarray, p: int) -> int:
        peak_val = signal[p]
        left = p
        while left > 1:
            if signal[left] < peak_val * self.ratio:
                break
            if signal[left] < signal[left - 1]:
                break
            left -= 1
        return left

    def _find_phase(self, signal: np.ndarray, p: int) -> Tuple[int, int]:
        T = len(signal)
        peak_val = signal[p]
        left = p
        while left > 1:
            if signal[left] < peak_val * self.ratio:
                break
            if signal[left] < signal[left - 1]:
                break
            left -= 1
        right = p
        while right < T - 2:
            if signal[right] < peak_val * self.ratio:
                break
            if signal[right] < signal[right + 1]:
                break
            right += 1
        return left, right


class WindowSelector(BaseTransform):
    """
    Cutting the flow based on phase windows.

    Phase windows are selected based on `phase_includes`.
        - "onset"  : from left  -> apex
        - "apex"   : apex frame itself (always included)
        - "offset" : from apex  -> right

    Output flow is concatenated along the time axis (T), in order onset->apex->offset.

    Args:
        phase_includes : subset from {"onset", "apex", "offset"}
        max_windows    : take N best apex windows (sorted by apex magnitude)
        max_len        : clip total frame after concatenation (None = no clipping)
    """

    VALID_PHASES = frozenset({"onset", "apex", "offset"})

    def __init__(
        self,
        phase_includes: Sequence[str] = ("onset", "apex"),
        max_windows: int = 10,
        max_len: Optional[int] = None,
    ):
        unknown = set(phase_includes) - self.VALID_PHASES
        if unknown:
            raise ValueError(f"Unknown phases: {unknown}. Valid: {self.VALID_PHASES}")
        if "apex" not in phase_includes:
            raise ValueError("'apex' must always be included in phase_includes.")

        self.phase_includes = list(phase_includes)
        self.max_windows = int(max_windows)
        self.max_len = max_len

    def __call__(self, sample: SubjectSample) -> TransformOutput:
        flow = sample.flow
        windows = sample.windows[: self.max_windows]

        if len(windows) == 0:
            slices = [flow]
        else:
            slices = []
            for left, apex, right in windows:
                parts = []
                if "onset" in self.phase_includes:
                    if apex > left:
                        parts.append(flow[left:apex])
                if "apex" in self.phase_includes:
                    parts.append(flow[apex : apex + 1])
                if "offset" in self.phase_includes:
                    if right > apex + 1:
                        parts.append(flow[apex + 1 : right])
                if parts:
                    slices.append(np.concatenate(parts, axis=0))

        if len(slices) == 0:
            slices = [flow[:1]]

        merged = np.concatenate(slices, axis=0)

        if self.max_len is not None:
            merged = merged[: self.max_len]

        x = torch.from_numpy(merged.astype(np.float32))
        y = torch.tensor(sample.label, dtype=torch.long)

        return TransformOutput(x=x, y=y, meta=dict(sample.meta))
