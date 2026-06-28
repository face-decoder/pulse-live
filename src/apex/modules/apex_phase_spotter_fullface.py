from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .apex_spotter import ApexSpotter
from .apex_phase import ApexPhase
from .apex_smoother import ApexSmoother
from src.face.modules import FaceLandmark
from src.optical_flow.modules import TVL1
from src.video.modules import Video


class ApexPhaseSpotterFullFace(ApexSpotter):
    """
    V6-aligned apex phase detector for full-face analysis.
    
    Detects landmarks frame-by-frame (no interpolation), extracts optical flow
    for the entire face, and computes per-frame magnitude.
    """

    def __init__(
        self,
        face_size: Tuple[int, int] = (240, 240),
        margin: float = 0.05,
        distance_threshold: int = 5,
        prominence_threshold: float = 0.1,
        cutoff_ratio: float = 0.30,
    ):
        """
        Initialize fullface apex phase spotter.

        Args:
            face_size: Target size for face extraction.
            margin: Margin when extracting face (percentage).
            distance_threshold: Minimum distance between peaks.
            prominence_threshold: Minimum prominence for peaks.
            cutoff_ratio: Cutoff ratio for phase determination.
        """
        self.face_size = face_size
        self.margin = float(margin)

        self.landmarker = FaceLandmark()
        self.tvl1 = TVL1(fast_mode=True)

        self.apex_phase = ApexPhase(
            distance_threshold=distance_threshold,
            prominence_threshold=prominence_threshold,
            cutoff_ratio=cutoff_ratio,
        )

        self.reset()

    def reset(self) -> None:
        """Reset internal state for new video processing."""
        self.magnitudes: List[float] = []
        self._detected_frames: int = 0

    def process(self, video_path: str, phase_mode: str = 'onset_to_apex') -> Tuple[List[int], dict]:
        """
        Process video to detect apex phases for full face.

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
        Process frame pair to compute full-face optical flow magnitude.

        Args:
            prev_frame: Previous frame.
            curr_frame: Current frame.
            frame_index: Frame index (unused).
        """
        try:
            prev_landmarks = self.landmarker.detect(prev_frame)
            curr_landmarks = self.landmarker.detect(curr_frame)

            prev_face = self.landmarker.crop(
                image=prev_frame,
                landmarks=prev_landmarks,
                margin=self.margin,
                output_size=self.face_size,
            )

            curr_face = self.landmarker.crop(
                image=curr_frame,
                landmarks=curr_landmarks,
                margin=self.margin,
                output_size=self.face_size,
            )

            flow = self.tvl1.compute(prev_face, curr_face, download=False)
            flow = flow.download() if hasattr(flow, "download") else flow

            dx = np.asarray(flow[..., 0], dtype=np.float32)
            dy = np.asarray(flow[..., 1], dtype=np.float32)

            magnitude = np.hypot(dx, dy)
            mean_magnitude = float(np.mean(magnitude))

            self.magnitudes.append(mean_magnitude)
            self._detected_frames += 1
        except Exception:
            # If landmark detection or cropping fails, use zero magnitude
            self.magnitudes.append(0.0)

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

        signal_arr = np.array(smoothed)
        height_threshold = float(np.mean(signal_arr) + np.std(signal_arr))

        # V6-style: top-k=5 peaks for fullface (less than ROI's 10)
        apex_indices = self.apex_phase.find_top_k_apex(
            signal=smoothed, k=5, height=height_threshold
        )
        phases = self.apex_phase.find_phase(signal=smoothed, apex_indices=apex_indices, phase_mode=phase_mode)

        return apex_indices, phases

    def detect_windows(self, flow: np.ndarray, phase_mode: str = "onset_to_apex") -> tuple:
        """
        Detect apex phase windows from full-face flow data.
        
        Args:
            flow: Optical flow with shape (T, H, W, 2) or (T, 2, H, W)
            phase_mode: Phase extraction mode (onset_to_apex or full)
        
        Returns:
            Tuple of (windows, metadata)
        """
        from .apex_phase_spotter_utils import flow_to_magnitude_signal
        
        # Convert flow to magnitude signal
        signal = flow_to_magnitude_signal(flow)
        
        # Detect windows from signal using v6-style thresholds
        return self.detect_windows_from_signal(signal, phase_mode=phase_mode)

    def detect_windows_from_signal(
        self, signal, phase_mode: str = "onset_to_apex"
    ) -> tuple:
        """
        Detect windows from a magnitude signal (for webrtc compatibility).
        
        Uses v6-style mean+std threshold and top-5 peak selection (fewer than ROI).
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
