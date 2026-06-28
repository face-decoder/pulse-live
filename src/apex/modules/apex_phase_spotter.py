from __future__ import annotations

from typing import List, Literal, Tuple

from .apex_spotter import ApexSpotter
from .apex_phase_spotter_fullface import ApexPhaseSpotterFullFace
from .apex_phase_spotter_roi import ApexPhaseSpotterROI

ExtractionMode = Literal["roi", "fullface"]


class ApexPhaseSpotter(ApexSpotter):
    """Backward-compatible adapter for the old combined spotter API."""

    def __init__(self, mode: ExtractionMode = "roi", **kwargs):
        object.__setattr__(self, "mode", mode)
        
        # Pop parameters that aren't accepted by ROI/FullFace constructors
        percentile = kwargs.pop("percentile", None)
        prominence = kwargs.pop("prominence", None)
        distance = kwargs.pop("distance", None)

        # Map to constructor parameters if they are not already set
        if prominence is not None and "prominence_threshold" not in kwargs:
            kwargs["prominence_threshold"] = prominence
        if distance is not None and "distance_threshold" not in kwargs:
            kwargs["distance_threshold"] = distance

        if mode == "fullface":
            impl = ApexPhaseSpotterFullFace(**kwargs)
        else:
            impl = ApexPhaseSpotterROI(**kwargs)

        # Set properties on impl for runtime reference
        if percentile is not None:
            impl.percentile = percentile
        if prominence is not None:
            impl.apex_phase.prominence = prominence
        if distance is not None:
            impl.apex_phase.distance = distance

        object.__setattr__(self, "_impl", impl)

    def process(self, video_path: str, phase_mode: str = "onset_to_apex") -> Tuple[List[int], dict]:
        # V6 implementations don't use phase_mode, ignore it
        return self._impl.process(video_path)

    def reset(self) -> None:
        self._impl.reset()

    def export_flow_data(self) -> dict:
        return self._impl.export_flow_data()

    def detect_windows(self, *args, **kwargs):
        return self._impl.detect_windows(*args, **kwargs)

    def detect_windows_from_signal(self, *args, **kwargs):
        if hasattr(self._impl, "detect_windows_from_signal"):
            return self._impl.detect_windows_from_signal(*args, **kwargs)
        raise AttributeError("Underlying spotter does not support signal detection")

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def __setattr__(self, name, value):
        if name in {"mode", "_impl"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._impl, name, value)


LazySpotter = ApexPhaseSpotter
