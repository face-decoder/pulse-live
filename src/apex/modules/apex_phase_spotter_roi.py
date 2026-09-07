from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.face.modules import FaceAligner, FaceLandmark, FaceRoiPoints
from src.optical_flow.modules import TVL1
from src.video.modules import Video

from .apex_phase import ApexPhase
from .apex_smoother import ApexSmoother
from .apex_spotter import ApexSpotter


class ApexPhaseSpotterROI(ApexSpotter):
    """
    V6-aligned apex phase detector for ROI-based analysis.

    Detects landmarks frame-by-frame (no interpolation), extracts optical flow
    for 5 ROIs (left_eye, right_eye, lips, left_eyebrow, right_eyebrow),
    and averages magnitudes per frame.
    """

    # Smoothing/threshold window sizes are tuned as real-time durations, not
    # fixed frame counts - CAS(ME)^2 (30fps) and SAMM/CASME-II (200fps) need
    # very different frame counts for the same physical window. Defaults
    # below reproduce the empirically-tuned CAS(ME)^2 values (11 and 15
    # frames at 30fps): 366.7ms and 500ms.
    SMOOTH_WINDOW_MS = 366.7
    THRESH_WINDOW_MS = 500.0

    def __init__(
        self,
        tile_size: Tuple[int, int] = (64, 64),
        margin: float = 0.05,
        distance_threshold: int = 5,
        prominence_threshold: float = 0.1,
        cutoff_ratio: float = 0.30,
        show_frame: bool = False,
        fps: float = None,
        smooth_window_ms: float = None,
        thresh_window_ms: float = None,
    ):
        """
        Initialize ROI-based apex phase spotter.

        Args:
            tile_size: Target size for each ROI.
            margin: Margin when extracting ROI (percentage).
            distance_threshold: Minimum distance between peaks.
            prominence_threshold: Minimum prominence for peaks.
            cutoff_ratio: Cutoff ratio for phase determination.
            show_frame: If True, print frame indices during processing.
            fps: Frame rate of the input signal. When None (default),
                preserves the original ApexSmoother-based, fps-agnostic
                smoothing/threshold behavior (safe default for existing
                callers - webrtc, autorunner - not yet fps-audited). Pass an
                explicit fps to opt into the fixed real-time-duration
                windows below (verified for CAS(ME)^2/CASME-II/SAMM).
            smooth_window_ms: Override the default smoothing window duration.
                Only used when fps is not None.
            thresh_window_ms: Override the default threshold window duration.
                Only used when fps is not None.
        """
        self.tile_size = tile_size
        self.tile_w, self.tile_h = tile_size
        self.margin = float(margin)
        self.show_frame = bool(show_frame)
        self.fps = float(fps) if fps is not None else None
        self.smooth_window_ms = (
            smooth_window_ms if smooth_window_ms is not None else self.SMOOTH_WINDOW_MS
        )
        self.thresh_window_ms = (
            thresh_window_ms if thresh_window_ms is not None else self.THRESH_WINDOW_MS
        )

        self.landmarker = FaceLandmark()
        self.aligner = FaceAligner()
        self.tvl1 = TVL1(fast_mode=True)

        self.apex_phase = ApexPhase(
            distance_threshold=distance_threshold,
            prominence_threshold=prominence_threshold,
            cutoff_ratio=cutoff_ratio,
        )

        self.smoothed_magnitudes: Sequence[float] = []
        self.roi_defs = [
            ("left_eye", frozenset(FaceRoiPoints.LEFT_EYE_POINTS)),
            ("right_eye", frozenset(FaceRoiPoints.RIGHT_EYE_POINTS)),
            ("lips", frozenset(FaceRoiPoints.LIPS_POINTS)),
            ("left_eyebrow", frozenset(FaceRoiPoints.LEFT_EYEBROW_POINTS)),
            ("right_eyebrow", frozenset(FaceRoiPoints.RIGHT_EYEBROW_POINTS)),
        ]

        self.cols = 3
        self.rows = math.ceil(len(self.roi_defs) / self.cols)

        self.reset()

    def process(
        self, video_path: str, phase_mode: str = "onset_to_apex"
    ) -> Tuple[List[int], dict]:
        """
        Process video to detect apex phases based on ROI.

        Args:
            video_path: Path to video file.
            phase_mode: Mode for phase determination ('onset_to_apex' or 'onset_apex_offset').

        Returns:
            Tuple of (apex_indices, phases_dict).
        """
        self.reset()
        video = Video(video_path=video_path)
        video.map(self.__process_frame__)

        return self._find_apex_phase(self.magnitudes, phase_mode=phase_mode)

    def __process_frame__(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray, frame_index: int
    ) -> None:
        """
        Process frame pair to compute ROI-based optical flow magnitude.

        Args:
            prev_frame: Previous frame.
            curr_frame: Current frame.
            frame_index: Frame index (unused).
        """
        if self.show_frame:
            try:
                print(f"Processing frame {frame_index}", end="\r", flush=True)
            except Exception:
                pass

        # Detect landmarks on both frames (no interpolation)
        prev_landmarks = self.landmarker.detect(prev_frame)
        curr_landmarks = self.landmarker.detect(curr_frame)

        # Align frames
        prev_aligned = self.aligner.align(image=prev_frame, landmarks=prev_landmarks)
        curr_aligned = self.aligner.align(image=curr_frame, landmarks=curr_landmarks)

        # Re-detect landmarks on aligned frames
        aligned_prev_landmarks = self.landmarker.detect(prev_aligned)
        aligned_curr_landmarks = self.landmarker.detect(curr_aligned)

        roi_magnitudes = []
        roi_flows_in_frame: List[Dict[str, Any]] = []
        for roi_name, roi_points in self.roi_defs:
            try:
                roi_prev, _ = self.landmarker.crop_roi(
                    image=prev_aligned,
                    landmark_result=aligned_prev_landmarks,
                    roi_points=roi_points,
                    margin=self.margin,
                    target_size=self.tile_size,
                )

                roi_next, _ = self.landmarker.crop_roi(
                    image=curr_aligned,
                    landmark_result=aligned_curr_landmarks,
                    roi_points=roi_points,
                    margin=self.margin,
                    target_size=self.tile_size,
                )

                if roi_prev is None or roi_next is None:
                    continue

                flow = self.tvl1.compute(roi_prev, roi_next, download=False)
                flow = flow.download() if hasattr(flow, "download") else flow

                dx = np.asarray(flow[..., 0], dtype=np.float32)
                dy = np.asarray(flow[..., 1], dtype=np.float32)

                # Store per-ROI flow components (matching v6 training pipeline)
                self.horizontal_magnitudes[roi_name].append(dx)
                self.vertical_magnitudes[roi_name].append(dy)

                roi_flows_in_frame.append({"roi": roi_name, "dx": dx, "dy": dy})

                mag = np.hypot(dx, dy)
                roi_magnitudes.append(float(np.mean(mag)))
            except Exception:
                continue

        if roi_magnitudes:
            frame_magnitude = float(np.mean(roi_magnitudes))
            self._detected_frames += 1
        else:
            frame_magnitude = 0.0

        self.magnitudes.append(frame_magnitude)
        self.frame_roi_flows.append(roi_flows_in_frame)

    def _find_apex_phase(
        self, magnitudes: List[float], phase_mode: str = "onset_to_apex"
    ) -> Tuple[List[int], dict]:
        """
        Detect apex and phases from magnitude signal (v6-style).

        Args:
            magnitudes: Per-frame magnitude signal.
            phase_mode: 'onset_to_apex' to return onset->apex windows, or
                        'onset_apex_offset' to return onset->offset windows.

        Returns:
            Tuple of (apex_indices, phases_dict).
        """
        if phase_mode not in ("onset_to_apex", "onset_apex_offset"):
            raise ValueError(f"Unknown phase_mode: {phase_mode}")

        if self.fps is None:
            # Original, fps-agnostic behavior - unaudited callers (webrtc,
            # autorunner) keep exactly what they had before this session.
            smoothed = ApexSmoother.smooth(signal=magnitudes)
            self.smoothed_magnitudes = smoothed
            signal_arr = np.array(smoothed)
            height_threshold = float(np.mean(signal_arr) + np.std(signal_arr))
        else:
            # ApexSmoother scales its window to 10% of video length (capped
            # 51), which for our ~2000+ frame videos means a ~51-frame savgol
            # filter smearing out a ~14-frame ME event - the two-pass
            # boundary walk then sees slow decay instead of a real valley and
            # returns windows ~2.6x too wide. Use a small fixed window
            # matched to ME duration instead, as a real-time duration
            # (self.smooth_window_ms) converted to frames via self.fps - a
            # fixed frame count would be wrong at a different frame rate
            # (e.g. SAMM/CASME-II run at 200fps, not 30fps).
            window_length = max(3, round(self.smooth_window_ms / 1000.0 * self.fps))
            window_length = min(
                window_length,
                len(magnitudes) if len(magnitudes) % 2 == 1 else len(magnitudes) - 1,
            )
            if window_length % 2 == 0:
                window_length += 1
            window_length = max(3, window_length)
            polyorder = min(3, window_length - 1)
            smoothed = savgol_filter(magnitudes, window_length, polyorder).astype(
                np.float32
            )
            self.smoothed_magnitudes = smoothed

            signal_arr = np.array(smoothed)

            # A whole-video mean+std threshold buries a short ME peak (~14
            # frames) under noise from a much longer video (~2000+ frames
            # avg): head motion, blinks, speech elsewhere in the clip skew
            # the global baseline. Use a rolling local baseline instead so
            # the threshold reflects nearby signal, not the whole video.
            # Window must roughly match the smoothing window above - too
            # much wider barely moves locally, letting noise bumps clear a
            # near-flat threshold uniformly. Real-time duration
            # (self.thresh_window_ms), scaled by fps, same reasoning as above.
            # ponytail: 500ms picked empirically for CAS(ME)^2 (plateau
            # 433-700ms all tie on F1); re-verify per-dataset if fps or event
            # duration differ.
            window = max(3, round(self.thresh_window_ms / 1000.0 * self.fps))
            series = pd.Series(signal_arr)
            local_mean = series.rolling(
                window=window, center=True, min_periods=1
            ).mean()
            local_std = (
                series.rolling(window=window, center=True, min_periods=1)
                .std()
                .fillna(0.0)
            )
            height_threshold = (local_mean + local_std).to_numpy()

        apex_indices = self.apex_phase.find_top_k_apex(
            signal=smoothed, k=10, height=height_threshold
        )
        phases = self.apex_phase.find_phase(
            signal=smoothed, apex_indices=apex_indices, phase_mode=phase_mode
        )

        return apex_indices, phases

    def reset(self) -> None:
        """Reset internal state for new video processing."""
        self.magnitudes: List[float] = []
        self._detected_frames: int = 0

        # Per-ROI flow storage (matching v6 training pipeline)
        self.horizontal_magnitudes: Dict[str, List[np.ndarray]] = {
            roi_name: [] for roi_name, _ in self.roi_defs
        }
        self.vertical_magnitudes: Dict[str, List[np.ndarray]] = {
            roi_name: [] for roi_name, _ in self.roi_defs
        }
        self.frame_roi_flows: List[List[Dict[str, Any]]] = []

    def detect_windows(
        self, flow: np.ndarray, phase_mode: str = "onset_to_apex"
    ) -> tuple:
        """
        Detect apex phase windows from ROI flow data.

        Args:
            flow: ROI optical flow with shape (T, N_roi, 2, H, W) or (T, H, W, 2)
            phase_mode: Phase extraction mode (onset_to_apex or full)

        Returns:
            Tuple of (windows, metadata)
        """
        from .apex_phase_spotter_utils import flow_to_magnitude_signal

        signal = flow_to_magnitude_signal(flow)

        return self.detect_windows_from_signal(signal, phase_mode=phase_mode)

    def detect_windows_from_signal(
        self, signal: Sequence[float], phase_mode: str = "onset_to_apex"
    ) -> tuple:
        """
        Detect windows from a magnitude signal (for webrtc compatibility).

        Uses v6-style mean+std threshold and top-10 peak selection.
        """
        from .apex_phase_spotter_utils import detect_windows_from_signal

        percentile = getattr(self, "percentile", 95.0)
        return detect_windows_from_signal(
            signal,
            percentile=percentile,
            prominence=self.apex_phase.prominence,
            min_distance=self.apex_phase.distance,
            ratio=self.apex_phase.cutoff_ratio,
            min_window=3,
            max_window=200,
            context=5,
            phase_mode=phase_mode,
        )

    def export_flow_data(self) -> dict:
        """
        Export RAW optical flow data (unprocessed, model-agnostic).

        Output format:
            {
                "flow": np.ndarray shaped (T, N_roi, 2, H, W),
                "magnitudes": np.ndarray (T,),
                "roi_order": list,
                "meta": {...}
            }

        The method returns float16-encoded flow to save memory (matching
        v6 training pipeline behaviour). All numeric magnitudes are float32.
        """
        roi_order = [roi_name for roi_name, _ in self.roi_defs]
        roi_flows = []

        for roi in roi_order:
            dx_list = self.horizontal_magnitudes[roi]
            dy_list = self.vertical_magnitudes[roi]

            if len(dx_list) == 0 or len(dy_list) == 0:
                continue

            dx = np.stack(dx_list, axis=0)  # (T, H, W)
            dy = np.stack(dy_list, axis=0)

            flow = np.stack([dx, dy], axis=1)  # (T, 2, H, W)
            roi_flows.append(flow)

        if len(roi_flows) == 0:
            raise ValueError("No valid ROI flow data.")

        flow = np.stack(roi_flows, axis=1)  # (T, N_roi, 2, H, W)
        flow = flow.astype(np.float16)

        magnitudes = np.asarray(self.magnitudes, dtype=np.float32)

        meta = {
            "frame_count": int(flow.shape[0]),
            "roi_count": int(flow.shape[1]),
            "height": int(flow.shape[3]),
            "width": int(flow.shape[4]),
            "landmark_detection_rate": float(
                self._detected_frames / len(self.magnitudes)
            )
            if len(self.magnitudes) > 0
            else 0.0,
        }

        return {
            "flow": flow,
            "magnitudes": magnitudes,
            "roi_order": roi_order,
            "meta": meta,
        }
