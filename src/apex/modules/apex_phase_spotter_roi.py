from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.face.modules import FaceAligner, FaceLandmark, FaceRoiPoints
from src.optical_flow.modules import TVL1
from src.video.modules import Video

from .apex_phase import ApexPhase
from .apex_smoother import ApexSmoother
from .apex_spotter import ApexSpotter


class ApexPhaseSpotterROI(ApexSpotter):
    def __init__(
        self,
        tile_size: tuple[int, int] = (64, 64),
        margin: float = 0.05,
        distance_threshold: int = 5,
        prominence_threshold: float = 0.1,
        cutoff_ratio: float = 0.30,
        show_frame: bool = False,
    ):
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

    def process(
        self, video_path: str, phase_mode: str = "onset_to_apex"
    ) -> tuple[list[int], dict]:
        self.reset()
        video = Video(video_path=video_path)
        video.map(self.__process_frame__)

        return self.__find_apex_phase(self.magnitudes, phase_mode=phase_mode)

    def __process_frame__(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray, frame_index: int
    ) -> None:
        if self.show_frame:
            try:
                print(f"Processing frame {frame_index}", end="\r", flush=True)
            except Exception:
                pass

        prev_landmarks = self.landmarker.detect(prev_frame)
        curr_landmarks = self.landmarker.detect(curr_frame)

        prev_aligned = self.aligner.align(image=prev_frame, landmarks=prev_landmarks)
        curr_aligned = self.aligner.align(image=curr_frame, landmarks=curr_landmarks)

        aligned_prev_landmarks = self.landmarker.detect(prev_aligned)
        aligned_curr_landmarks = self.landmarker.detect(curr_aligned)

        roi_magnitudes = []
        roi_flows_in_frame: list[dict[str, Any]] = []
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

    def __find_apex_phase(
        self, magnitudes: list[float], phase_mode: str = "onset_to_apex"
    ) -> tuple[list[int], dict]:
        if phase_mode not in ("onset_to_apex", "onset_apex_offset"):
            raise ValueError(f"Unknown phase_mode: {phase_mode}")

        smoothed_arr = ApexSmoother.smooth(signal=magnitudes)
        self.smoothed_magnitudes = smoothed_arr.tolist()

        height_threshold = float(np.mean(smoothed_arr) + np.std(smoothed_arr))

        apex_indices = self.apex_phase.find_top_k_apex(
            signal=smoothed_arr.tolist(), k=10, height=height_threshold
        )
        phases = self.apex_phase.find_phase(
            signal=smoothed_arr.tolist(),
            apex_indices=apex_indices,
            phase_mode=phase_mode,
        )

        return apex_indices, phases

    def reset(self) -> None:
        self.magnitudes: list[float] = []
        self._detected_frames: int = 0

        self.horizontal_magnitudes: dict[str, list[np.ndarray]] = {
            roi_name: [] for roi_name, _ in self.roi_defs
        }
        self.vertical_magnitudes: dict[str, list[np.ndarray]] = {
            roi_name: [] for roi_name, _ in self.roi_defs
        }
        self.frame_roi_flows: list[list[dict[str, Any]]] = []

    def detect_windows(
        self, flow: np.ndarray, phase_mode: str = "onset_to_apex"
    ) -> tuple:
        from .apex_phase_spotter_utils import flow_to_magnitude_signal

        signal = flow_to_magnitude_signal(flow)

        return self.detect_windows_from_signal(signal, phase_mode=phase_mode)

    def detect_windows_from_signal(
        self, signal: Sequence[float] | np.ndarray, phase_mode: str = "onset_to_apex"
    ) -> tuple:
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

    def summarize_signal(self, signal: Sequence[float]) -> dict:
        smoothed = [float(x) for x in signal]
        detected_phases: list[dict[str, int]] = []
        try:
            smoothed = [float(x) for x in ApexSmoother.smooth(signal=smoothed).tolist()]

            windows, meta = self.detect_windows_from_signal(signal)
            actual = meta.get("phases", {}) if meta.get("valid", False) else {}
            detected_phases = [
                {
                    "onset": int(actual.get(apex, {}).get("start", 0)),
                    "apex": int(apex),
                    "offset": int(actual.get(apex, {}).get("end", 0)),
                }
                for _, apex, _ in windows
            ]
        except Exception:
            detected_phases = []

        return {
            "smoothed_magnitudes": smoothed,
            "detected_phases": detected_phases,
        }

    def export_flow_data(self) -> dict:
        roi_order = [roi_name for roi_name, _ in self.roi_defs]
        roi_flows = []

        for roi in roi_order:
            dx_list = self.horizontal_magnitudes[roi]
            dy_list = self.vertical_magnitudes[roi]

            if len(dx_list) == 0 or len(dy_list) == 0:
                continue

            dx = np.stack(dx_list, axis=0)
            dy = np.stack(dy_list, axis=0)

            flow = np.stack([dx, dy], axis=1)
            roi_flows.append(flow)

        if len(roi_flows) == 0:
            raise ValueError("No valid ROI flow data.")

        flow = np.stack(roi_flows, axis=1)
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
