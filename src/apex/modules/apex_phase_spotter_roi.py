from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple, Sequence

import numpy as np

from .apex_spotter import ApexSpotter
from .apex_phase import ApexPhase
from .apex_smoother import ApexSmoother
from src.face.modules import FaceLandmark, FaceRoiPoints, FaceAligner
from src.optical_flow.modules import TVL1
from src.video.modules import Video


class ApexPhaseSpotterROI(ApexSpotter):
    """
    V6-aligned apex phase detector for ROI-based analysis.
    
    Detects landmarks frame-by-frame (no interpolation), extracts optical flow
    for 5 ROIs (left_eye, right_eye, lips, left_eyebrow, right_eyebrow),
    and averages magnitudes per frame.
    """

    def __init__(
        self,
        tile_size: Tuple[int, int] = (64, 64),
        margin: float = 0.05,
        distance_threshold: int = 5,
        prominence_threshold: float = 0.1,
        cutoff_ratio: float = 0.30,
        show_frame: bool = False,
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
        """
        self.tile_size = tile_size
        self.tile_w, self.tile_h = tile_size
        self.margin = float(margin)
        self.show_frame = bool(show_frame)

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

    def process(self, video_path: str, phase_mode: str = 'onset_to_apex') -> Tuple[List[int], dict]:
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

    def _find_apex_phase(self, magnitudes: List[float], phase_mode: str = "onset_to_apex") -> Tuple[List[int], dict]:
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

        smoothed = ApexSmoother.smooth(signal=magnitudes)
        self.smoothed_magnitudes = smoothed

        signal_arr = np.array(smoothed)
        height_threshold = float(np.mean(signal_arr) + np.std(signal_arr))

        apex_indices = self.apex_phase.find_top_k_apex(signal=smoothed, k=10, height=height_threshold)
        phases = self.apex_phase.find_phase(signal=smoothed, apex_indices=apex_indices, phase_mode=phase_mode)

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

    def detect_windows(self, flow: np.ndarray, phase_mode: str = "onset_to_apex") -> tuple:
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
