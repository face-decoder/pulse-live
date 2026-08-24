import asyncio
import time

import numpy as np
from aiortc import MediaStreamTrack

from src.api.config import Stream
from src.api.stream_processor import AnxietyStreamProcessor


class AnxietyVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(
        self,
        track: MediaStreamTrack,
        result_queue: asyncio.Queue[dict[str, object]],
        session_id: str = "unknown",
    ) -> None:
        super().__init__()
        self._track = track
        self._processor = AnxietyStreamProcessor(result_queue, session_id=session_id)
        self._window_start: float | None = None
        self._last_frame_time: float = 0.0
        self._interval: float = 1.0 / Stream.FPS

    async def recv(self) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]
        frame = await self._track.recv()
        now = time.time()

        if now - self._last_frame_time < self._interval:
            return frame
        self._last_frame_time = now

        if self._window_start is None:
            self._window_start = now - frame.time  # pyright: ignore[reportOperatorIssue, reportAttributeAccessIssue]

        img: np.ndarray = frame.to_ndarray(format="bgr24")  # pyright: ignore[reportAttributeAccessIssue]
        latency_ms = max(0.0, (now - (self._window_start + frame.time)) * 1000)  # pyright: ignore[reportOperatorIssue, reportAttributeAccessIssue]
        self._processor.push_frame(img, now, latency_ms)
        return frame

    def stop(self) -> None:
        super().stop()
        self._processor.close()
