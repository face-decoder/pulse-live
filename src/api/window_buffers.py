from collections import deque

import numpy as np

from src.api.config import Stream, Window
from src.api.window_snapshot import WindowSnapshot


class WindowBuffers:
    def __init__(self) -> None:
        cap = int(Window.SECONDS * Stream.FPS)
        self.mags: deque[float] = deque(maxlen=cap - 1)
        self.flows: deque[np.ndarray] = deque(maxlen=cap - 1)
        self.bboxes: deque[dict | None] = deque(maxlen=cap)
        self.webrtc_ms: deque[float] = deque(maxlen=cap)
        self.landmark_ms: deque[float] = deque(maxlen=cap)
        self.flow_ms: deque[float] = deque(maxlen=cap - 1)
        self.timestamps: deque[float] = deque(maxlen=cap)
        self.history: list[float] = []

    def record_frame(
        self, bbox: dict | None, webrtc_ms: float, landmark_ms: float, at: float
    ) -> None:
        self.bboxes.append(bbox)
        self.webrtc_ms.append(webrtc_ms)
        self.landmark_ms.append(landmark_ms)
        self.timestamps.append(at)

    def record_flow(self, mag: float, canvas: np.ndarray, flow_ms: float) -> None:
        self.mags.append(mag)
        self.flows.append(canvas)
        self.flow_ms.append(flow_ms)
        self.history.append(mag)

    @property
    def ready(self) -> bool:
        return len(self.mags) >= Window.MIN_FRAMES

    def snapshot(self, received_at: float) -> WindowSnapshot:
        return WindowSnapshot(
            mags=list(self.mags),
            flows=list(self.flows),
            bboxes=list(self.bboxes),
            received_at=received_at,
            webrtc_ms=list(self.webrtc_ms),
            landmark_ms=list(self.landmark_ms),
            flow_ms=list(self.flow_ms),
            timestamps=list(self.timestamps),
        )
