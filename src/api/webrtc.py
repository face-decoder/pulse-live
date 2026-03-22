from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from aiortc import (
    MediaStreamTrack,
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaRelay
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.config import (
    HEARTBEAT_TIMEOUT_SECONDS,
    MIN_FRAMES,
    TARGET_FPS,
    WINDOW_SECONDS,
)
from src.model.modules import AnxietyInferencer

logger = logging.getLogger(__name__)

router = APIRouter()

relay = MediaRelay()
inferencer: AnxietyInferencer | None = None


def set_inferencer(inf: AnxietyInferencer) -> None:
    """Set the module-level inferencer instance.

    Called once from the application ``lifespan`` context manager.

    Args:
        inf: A fully-initialised :class:`AnxietyInferencer`.
    """
    global inferencer  # noqa: PLW0603
    inferencer = inf



class AnxietyVideoTrack(MediaStreamTrack):
    """Receive a WebRTC video track and run inference on buffered frames.

    Frames are accumulated for :data:`WINDOW_SECONDS` seconds at
    :data:`TARGET_FPS`.  When the window is full the TV-L1 + inference
    pipeline runs in a thread-pool so the event loop stays responsive.

    Results are pushed into *result_queue* for the WebSocket sender.

    Attributes:
        kind: Always ``"video"`` (required by aiortc).
    """

    kind = "video"

    def __init__(
        self,
        track: MediaStreamTrack,
        result_queue: asyncio.Queue[dict[str, object]],
    ) -> None:
        """Initialise the video track handler.

        Args:
            track: The relayed video track from aiortc.
            result_queue: Queue for sending prediction dicts to the
                WebSocket sender task.
        """
        super().__init__()
        self._track = track
        self._result_queue = result_queue

        from src.apex.modules.v2.apex_phase_spotter import ApexPhaseSpotter
        self._spotter = ApexPhaseSpotter(mode="batch")

        self._frame_buf: deque[np.ndarray] = deque()
        self._window_start: float = time.time()
        self._last_frame_time: float = 0.0
        self._frame_interval: float = 1.0 / TARGET_FPS

    async def recv(self) -> object:
        """Receive, buffer, and optionally trigger inference.

        Returns:
            The original ``av.VideoFrame`` (passed through unchanged).
        """
        frame = await self._track.recv()
        now = time.time()

        # Throttle — only keep frames matching TARGET_FPS
        if now - self._last_frame_time < self._frame_interval:
            return frame
        self._last_frame_time = now

        # Convert aiortc VideoFrame → numpy BGR
        img: np.ndarray = frame.to_ndarray(format="bgr24")
        self._frame_buf.append(img)

        # Check whether the window is full
        elapsed = now - self._window_start
        if elapsed >= WINDOW_SECONDS and len(self._frame_buf) >= MIN_FRAMES:
            frames = list(self._frame_buf)
            self._frame_buf.clear()
            self._window_start = now

            # Run CPU-bound inference off the event loop
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, self._run_inference, frames,
            )
            if result is not None:
                await self._result_queue.put(result)

        return frame

    # ── Private helpers ────────────────────────────────────────────────

    def _run_inference(
        self, frames: list[np.ndarray],
    ) -> dict[str, object] | None:
        """Execute TV-L1 optical flow + inference on a frame window.

        This method runs **synchronously** inside a thread-pool executor
        so it must not call any async APIs.

        Args:
            frames: BGR images accumulated over the window.

        Returns:
            A JSON-serialisable prediction dict, or ``None`` on failure.
        """
        if inferencer is None:
            logger.warning("Inferencer not loaded — skipping prediction")
            return None
        if len(frames) < 2:
            return None

        try:
            self._spotter.process_frames(frames)
            flow_data = self._spotter.export_flow_data()
        except Exception:
            logger.error("ApexPhaseSpotter failed to process frames", exc_info=True)
            return None

        roi_frames = flow_data["frames"]
        mag_array = np.array(flow_data["magnitudes"], dtype=np.float32)

        if not roi_frames or len(mag_array) == 0:
            return None

        try:
            result = inferencer._predict_from_frames(roi_frames, mag_array)
        except Exception:
            logger.error("Inference failed", exc_info=True)
            return None

        return {
            "type": "prediction",
            "label": result.label,
            "confidence": round(result.confidence, 4),
            "prob_high": round(result.prob_high, 4),
            "prob_low": round(result.prob_low, 4),
            "n_apex_detected": result.n_apex_detected,
            "n_frames": len(frames),
            "warning": result.warning,
            "top_features": [
                {
                    "name": f.name,
                    "value": round(f.value, 4),
                    "saliency": round(f.saliency, 4),
                    "direction": f.direction,
                }
                for f in result.top_features[:5]
            ],
        }

    def stop(self) -> None:
        """Release resources when the track is stopped."""
        super().stop()
        self._spotter.close()


# ── Per-session state ─────────────────────────────────────────────────


@dataclass
class _SessionState:
    """Internal bookkeeping for a single WebRTC session."""

    pc: RTCPeerConnection
    result_queue: asyncio.Queue[dict[str, object]] = field(
        default_factory=asyncio.Queue,
    )
    video_track: AnxietyVideoTrack | None = None
    result_task: asyncio.Task[None] | None = None
    consume_task: asyncio.Task[None] | None = None

    async def cleanup(self) -> None:
        """Cancel background tasks and close the peer connection."""
        if self.result_task is not None:
            self.result_task.cancel()
        if self.consume_task is not None:
            self.consume_task.cancel()
        if self.video_track is not None:
            self.video_track.stop()
        await self.pc.close()


# ── Result sender coroutine ───────────────────────────────────────────


async def _consume_track(track: AnxietyVideoTrack) -> None:
    """Consume frames from the track to prevent buffer buildup.
    
    If the track is not added to the peer connection (because we don't 
    need to send it back to the client), we must still continuously pull 
    frames from it to drive the receive pipeline.
    """
    try:
        while True:
            await track.recv()
    except Exception:
        pass


async def _send_results(
    ws: WebSocket,
    queue: asyncio.Queue[dict[str, object]],
) -> None:
    """Forward prediction results from *queue* to the WebSocket.

    Sends a heartbeat JSON when no result arrives within
    :data:`HEARTBEAT_TIMEOUT_SECONDS`.

    Args:
        ws: The open WebSocket connection.
        queue: Queue populated by :class:`AnxietyVideoTrack`.
    """
    while True:
        try:
            result = await asyncio.wait_for(
                queue.get(), timeout=HEARTBEAT_TIMEOUT_SECONDS,
            )
            raw = json.dumps(result)
            logger.info("Sending prediction to websocket: %s", raw)
            await ws.send_text(raw)
        except asyncio.TimeoutError:
            raw = json.dumps({"type": "heartbeat"})
            logger.info("Sending heartbeat to websocket: %s", raw)
            await ws.send_text(raw)
        except Exception:
            logger.warning("send_results stopped", exc_info=True)
            break


@router.websocket("/ws/rtc/{session_id}")
async def webrtc_signaling(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for WebRTC signaling and result streaming.

    Handles SDP offer/answer exchange, ICE candidate relay, and
    prediction streaming for a single recording session identified by
    *session_id*.

    Args:
        websocket: The incoming WebSocket connection.
        session_id: Unique client-assigned session identifier.
    """
    await websocket.accept()
    logger.info("Session %s connected", session_id)

    state = _SessionState(pc=RTCPeerConnection())

    # ── ICE candidate from server → browser ───────────────────────────
    @state.pc.on("connectionstatechange")
    async def _on_connectionstatechange() -> None:
        logger.info("Session %s WebRTC connection state changed to: %s", session_id, state.pc.connectionState)


    @state.pc.on("icecandidate")
    async def _on_icecandidate(candidate: object) -> None:
        if candidate is not None:
            raw = json.dumps({
                "type": "candidate",
                "candidate": {
                    "candidate": candidate.to_sdp(),
                    "sdpMid": candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex,
                },
            })
            logger.info("Sending ICE candidate to session %s: %s", session_id, raw)
            await websocket.send_text(raw)

    # ── Receive video track from browser ──────────────────────────────
    @state.pc.on("track")
    def _on_track(track: MediaStreamTrack) -> None:
        if track.kind == "video":
            # Clean up existing track resources if a renegotiation provides a new one
            if state.consume_task is not None:
                state.consume_task.cancel()
            if state.video_track is not None:
                state.video_track.stop()

            local_track = AnxietyVideoTrack(
                relay.subscribe(track), state.result_queue,
            )
            state.video_track = local_track
            # Create a task to consume frames natively instead of sending back via PC
            state.consume_task = asyncio.create_task(_consume_track(local_track))
            logger.info("Video track received for session %s", session_id)

    # ── Background: forward results to WebSocket ──────────────────────
    state.result_task = asyncio.create_task(
        _send_results(websocket, state.result_queue),
    )

    # ── Main message loop ─────────────────────────────────────────────
    try:
        async for raw in websocket.iter_text():
            msg: dict[str, object] = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "offer":
                offer = RTCSessionDescription(
                    sdp=str(msg["sdp"]),
                    type=str(msg["sdpType"]),
                )
                await state.pc.setRemoteDescription(offer)
                answer = await state.pc.createAnswer()
                await state.pc.setLocalDescription(answer)

                raw = json.dumps({
                    "type": "answer",
                    "sdp": state.pc.localDescription.sdp,
                    "sdpType": state.pc.localDescription.type,
                })
                logger.info("Sending SDP answer to session %s: %s", session_id, raw)
                await websocket.send_text(raw)

            elif msg_type == "candidate":
                c = msg["candidate"]
                if not isinstance(c, dict):
                    logger.warning(
                        "Invalid ICE candidate format from session %s",
                        session_id,
                    )
                    continue
                from aiortc.sdp import candidate_from_sdp

                c_str = str(c.get("candidate", ""))
                
                # End of candidates signal
                if not c_str:
                    logger.info("Received end of ICE candidates for session %s", session_id)
                    continue

                try:
                    if c_str.startswith("candidate:"):
                        c_str = c_str.split(":", 1)[1]
                    candidate = candidate_from_sdp(c_str)
                    candidate.sdpMid = c.get("sdpMid")
                    candidate.sdpMLineIndex = c.get("sdpMLineIndex")
                    await state.pc.addIceCandidate(candidate)
                except Exception as e:
                    logger.warning(
                        "Failed to parse ICE candidate from session %s: %s",
                        session_id, e,
                    )

            elif msg_type == "stop":
                break

    except WebSocketDisconnect:
        logger.info("Session %s disconnected", session_id)
    except Exception:
        logger.error("Session %s error", session_id, exc_info=True)
        try:
            raw = json.dumps({
                "type": "error",
                "message": "Internal server error",
            })
            logger.info("Sending error to session %s: %s", session_id, raw)
            await websocket.send_text(raw)
        except Exception:
            pass
    finally:
        await state.cleanup()
        logger.info("Session %s cleaned up", session_id)
