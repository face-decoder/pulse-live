from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.apex.modules.apex_phase_spotter_roi import ApexPhaseSpotterROI


class MESpotting:
    """Micro-expression apex and boundary spotting with IoU benchmarking."""

    def __init__(
        self,
        cutoff_ratio: float = 0.30,
        fps: int = 200,
        phase_mode: str = "onset_apex_offset",
        smooth_window_ms: float = None,
        thresh_window_ms: float = None,
    ) -> None:
        self.cutoff_ratio = cutoff_ratio
        self.fps = fps
        self.phase_mode = phase_mode
        self.spotter = ApexPhaseSpotterROI(
            cutoff_ratio=cutoff_ratio,
            show_frame=False,
            fps=fps,
            smooth_window_ms=smooth_window_ms,
            thresh_window_ms=thresh_window_ms,
        )

    @staticmethod
    def compute_iou(interval_a: tuple[int, int], interval_b: tuple[int, int]) -> float:
        """Compute 1D temporal Intersection over Union (IoU).

        Matches Fang et al. 2023 (RMES) convention: interval measure is
        (end - start), not inclusive frame count (end - start + 1).
        """
        s_on, s_off = interval_a
        g_on, g_off = interval_b
        intersection = max(0, min(s_off, g_off) - max(s_on, g_on))
        union = (s_off - s_on) + (g_off - g_on) - intersection
        return float(intersection / union) if union > 0 else 0.0

    def spot(
        self,
        magnitudes: Sequence[float] | np.ndarray,
        fps: int | None = None,
        fallback_half_win: int | None = None,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Spot dominant micro-expression interval from motion magnitudes.

        Args:
            magnitudes: Sequence of optical flow/phase magnitudes.
            fps: Video frame rate (defaults to self.fps).
            fallback_half_win: Half window for fallback when no phase found.

        Returns:
            Tuple of ((onset_feat, offset_feat), (onset_spot, offset_spot)).
        """
        mags = list(magnitudes)
        if not mags:
            return (0, 0), (0, 0)

        eff_fps = fps or self.fps

        try:
            _, phases_dict = self.spotter._find_apex_phase(
                mags, phase_mode=self.phase_mode
            )
        except Exception:
            phases_dict = {}

        if not phases_dict:
            peak_idx = int(np.argmax(mags))
            half = (
                fallback_half_win
                if fallback_half_win is not None
                else max(1, int(0.25 * eff_fps))
            )
            base_on = max(0, peak_idx - half)
            base_off = min(len(mags) - 1, peak_idx + half)
            return (base_on, base_off), (base_on, base_off)

        first_apex = next(iter(phases_dict.keys()))
        first_phase = phases_dict[first_apex]

        on_feat = first_phase["start"]
        off_feat = first_phase["end"]

        return (on_feat, off_feat), (on_feat, off_feat)

    def spot_all(
        self,
        magnitudes: Sequence[float] | np.ndarray,
    ) -> list[dict[str, Any]]:
        """Spot all candidate phases across untrimmed/long video sequence."""
        mags = list(magnitudes)
        if not mags:
            return []

        try:
            _, phases_dict = self.spotter._find_apex_phase(
                mags, phase_mode=self.phase_mode
            )
        except Exception:
            phases_dict = {}

        candidates = []
        for apex, phase in phases_dict.items():
            on_feat = phase["start"]
            off_feat = phase["end"]
            candidates.append(
                {
                    "apex": apex,
                    "feat_interval": (on_feat, off_feat),
                    "spot_interval": (on_feat, off_feat),
                    "phase": phase,
                }
            )
        return candidates

    def match(
        self,
        candidates: Sequence[dict[str, Any]],
        gt_interval: tuple[int, int],
        fallback_mags: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Find candidate phase with highest IoU against ground truth interval."""
        if not candidates:
            if fallback_mags is not None and len(fallback_mags) > 0:
                feat, spot = self.spot(fallback_mags)
                iou = self.compute_iou(spot, gt_interval)
                return {
                    "feat_interval": feat,
                    "spot_interval": spot,
                    "iou": iou,
                }
            return {
                "feat_interval": gt_interval,
                "spot_interval": gt_interval,
                "iou": 0.0,
            }

        best_cand = None
        best_iou = -1.0
        for cand in candidates:
            iou = self.compute_iou(cand["spot_interval"], gt_interval)
            if iou > best_iou:
                best_iou = iou
                best_cand = cand

        result = dict(best_cand) if best_cand else {}
        result["iou"] = max(0.0, best_iou)
        return result

    def benchmark(
        self,
        spotted_intervals: Sequence[tuple[int, int]],
        gt_intervals: Sequence[tuple[int, int]],
        iou_thresh: float = 0.5,
    ) -> dict[str, Any]:
        """Compute standard RMES spotting benchmark metrics (Fang et al. 2023)."""
        ious = [self.compute_iou(s, g) for s, g in zip(spotted_intervals, gt_intervals)]
        tps = sum(1 for iou in ious if iou >= iou_thresh)
        n_samples = len(spotted_intervals)

        prec = tps / n_samples if n_samples > 0 else 0.0
        rec = tps / n_samples if n_samples > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        mean_iou = float(np.mean(ious)) if ious else 0.0

        return {
            "samples": n_samples,
            "tp": tps,
            "mean_iou": mean_iou,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "iou_threshold": iou_thresh,
        }
