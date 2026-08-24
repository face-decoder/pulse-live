from dataclasses import dataclass

import numpy as np


@dataclass
class WindowSnapshot:
    mags: list[float]
    flows: list[np.ndarray]
    bboxes: list[dict | None]
    received_at: float
    webrtc_ms: list[float]
    landmark_ms: list[float]
    flow_ms: list[float]
    timestamps: list[float]

    @property
    def n_frames(self) -> int:
        return len(self.bboxes)
