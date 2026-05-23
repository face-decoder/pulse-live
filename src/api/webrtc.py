from __future__ import annotations

import asyncio
import json
import logging
import threading
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



class AnxietyStreamProcessor:
    """Incremental streaming and inference processor for video frames.

    Decouples core landmark detection, optical flow transitions, and
    anxiety classifier prediction from the underlying WebRTC transport layer.
    """

    def __init__(
        self,
        result_queue: asyncio.Queue[dict[str, object]],
    ) -> None:
        self._result_queue = result_queue

        from src.apex.modules.v2.apex_phase_spotter import ApexPhaseSpotter
        self._spotter = ApexPhaseSpotter(mode="batch")

        self._inference_in_progress = False
        self._max_window_len = int(WINDOW_SECONDS * TARGET_FPS)
        self._landmark_thread_lock = threading.Lock()

        # Incremental sliding buffers
        self._last_frame: np.ndarray | None = None
        self._last_landmarks = None
        
        self._magnitudes_buf: deque[float] = deque(maxlen=self._max_window_len - 1)
        self._flows_buf: deque[list] = deque(maxlen=self._max_window_len - 1)
        self._bboxes_buf: deque[dict | None] = deque(maxlen=self._max_window_len)

        self._processing_queue = asyncio.Queue()
        self._process_loop_task = asyncio.create_task(self._process_loop())

    def push_frame(self, img: np.ndarray, received_at: float) -> None:
        """Push a new video frame to the processor queue."""
        # Discard older frames if queue is backing up to keep latency minimal
        while self._processing_queue.qsize() > 2:
            try:
                self._processing_queue.get_nowait()
                self._processing_queue.task_done()
            except asyncio.QueueEmpty:
                break

        self._processing_queue.put_nowait((img, received_at))

    async def _process_loop(self) -> None:
        """Process incoming video frames sequentially from the queue."""
        while True:
            try:
                img, received_at = await self._processing_queue.get()
            except asyncio.CancelledError:
                break
            except Exception:
                continue

            try:
                await self._process_frame(img, received_at)
            except Exception:
                logger.error("Error in process_frame background task", exc_info=True)
            finally:
                self._processing_queue.task_done()

    async def _process_frame(self, img: np.ndarray, received_at: float) -> None:
        """Perform landmark detection and optical flow transition computation for one frame."""
        loop = asyncio.get_running_loop()

        # 1. Run face landmark detection (thread-safely via executor)
        def detect_landmarks_and_bbox(image):
            with self._landmark_thread_lock:
                landmarks = self._spotter.face_landmark.detect(image)
                bbox = self._spotter._get_face_bbox(landmarks, image)
            return landmarks, bbox

        landmarks, bbox = await loop.run_in_executor(None, detect_landmarks_and_bbox, img)

        # Send the real-time bbox overlay message immediately
        latency_ms = (time.time() - received_at) * 1000
        await self._result_queue.put({
            "type": "bbox",
            "bbox": bbox,
            "latency_ms": round(latency_ms, 2)
        })

        # 2. Compute optical flow transition if we have a previous frame
        prev_img = self._last_frame
        prev_landmarks = self._last_landmarks

        if prev_img is not None and prev_landmarks is not None:
            def compute_transition(p_img, c_img, p_lm, c_lm):
                with self._landmark_thread_lock:
                    return self._spotter.process_frame_pair(p_img, c_img, p_lm, c_lm)

            mag, flow_bucket = await loop.run_in_executor(
                None, compute_transition, prev_img, img, prev_landmarks, landmarks
            )

            # Slide our buffers
            self._magnitudes_buf.append(mag)
            self._flows_buf.append(flow_bucket)
            self._bboxes_buf.append(bbox)
        else:
            self._bboxes_buf.append(bbox)

        self._last_frame = img
        self._last_landmarks = landmarks

        # 3. Trigger inference when the window is full
        if len(self._magnitudes_buf) >= self._max_window_len - 1 and not self._inference_in_progress:
            self._inference_in_progress = True
            # Copy state to avoid race conditions with next frame
            mags_copy = list(self._magnitudes_buf)
            flows_copy = list(self._flows_buf)
            bboxes_copy = list(self._bboxes_buf)
            asyncio.create_task(self._run_inference_background(mags_copy, flows_copy, bboxes_copy, received_at))

    async def _run_inference_background(
        self,
        mags: list[float],
        flows: list[list[dict]],
        bboxes: list[dict | None],
        received_at: float
    ) -> None:
        """Run the actual deep learning model inference in the background."""
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, self._run_inference, mags, flows, bboxes, received_at
            )
            if result is not None:
                await self._result_queue.put(result)
        except Exception:
            logger.error("Background inference failed", exc_info=True)
        finally:
            self._inference_in_progress = False

    def _run_inference(
        self,
        mags: list[float],
        flows: list[list[dict]],
        bboxes: list[dict | None],
        received_at: float
    ) -> dict[str, object] | None:
        """Execute prediction using pre-computed optical flow and landmarks.

        This runs synchronously in the executor.
        """
        start_time = time.time()
        if inferencer is None:
            logger.warning("Inferencer not loaded — skipping prediction")
            return None
        if len(mags) < 1:
            return None

        try:
            # Spot the onset-apex-offset phases based on magnitudes
            with self._landmark_thread_lock:
                apex_indices, phases = self._spotter.find_apex_phase(mags)
            
            # Convert phases to client representation
            detected_phases = [
                {
                    "onset": int(phase["start"]),
                    "apex": int(apex_idx),
                    "offset": int(phase["end"])
                }
                for apex_idx, phase in phases.items()
            ]

            # Run inferencer prediction from the pre-computed flows and magnitudes
            mag_array = np.array(mags, dtype=np.float32)
            result = inferencer._predict_from_frames(flows, mag_array)
        except Exception:
            logger.error("Inference pipeline failed", exc_info=True)
            return None

        smoothed_mags = getattr(self._spotter, "smoothed_magnitudes", None)
        smoothed_mags_list = [float(m) for m in smoothed_mags] if smoothed_mags is not None else []

        return {
            "type": "prediction",
            "label": result.label,
            "confidence": round(result.confidence, 4),
            "prob_high": round(result.prob_high, 4),
            "prob_low": round(result.prob_low, 4),
            "n_apex_detected": result.n_apex_detected,
            "n_frames": len(bboxes),
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
            "face_bboxes": bboxes,
            "magnitudes": mags,
            "smoothed_magnitudes": smoothed_mags_list,
            "detected_phases": detected_phases,
            "latency_ms": round((time.time() - start_time) * 1000, 2),
        }

    def close(self) -> None:
        """Release resources when the processor is stopped."""
        if hasattr(self, "_process_loop_task"):
            self._process_loop_task.cancel()
        self._spotter.close()


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

        self._processor = AnxietyStreamProcessor(result_queue)

        self._window_start: float = time.time()
        self._last_frame_time: float = 0.0
        self._frame_interval: float = 1.0 / TARGET_FPS

    @property
    def _spotter(self):
        return self._processor._spotter

    @property
    def _bboxes_buf(self):
        return self._processor._bboxes_buf

    @property
    def _processing_queue(self):
        return self._processor._processing_queue

    @property
    def _run_inference(self):
        return self._processor._run_inference

    @_run_inference.setter
    def _run_inference(self, val):
        self._processor._run_inference = val

    @property
    def _max_window_len(self):
        return self._processor._max_window_len

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

        self._processor.push_frame(img, now)
        return frame

    def stop(self) -> None:
        """Release resources when the track is stopped."""
        super().stop()
        self._processor.close()


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
            pretty_raw = json.dumps(result, indent=2)
            logger.info("Sending response to websocket:\n%s", pretty_raw)
            await ws.send_text(raw)
        except asyncio.TimeoutError:
            raw = json.dumps({"type": "heartbeat"})
            pretty_raw = json.dumps({"type": "heartbeat"}, indent=2)
            logger.info("Sending heartbeat to websocket:\n%s", pretty_raw)
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


@router.websocket("/ws/stream/{session_id}")
async def websocket_video_stream(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for raw binary video frame streaming and real-time result streaming.

    Accepts JPEG/WebP/PNG binary images, processes them, and streams back
    real-time 'bbox' and windowed 'prediction' JSON results.
    """
    await websocket.accept()
    logger.info("Streaming session %s connected", session_id)

    import cv2

    result_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    processor = AnxietyStreamProcessor(result_queue)

    # Spawn result sender task
    result_task = asyncio.create_task(
        _send_results(websocket, result_queue),
    )

    loop = asyncio.get_running_loop()

    try:
        async for data in websocket.iter_bytes():
            if not data:
                continue

            # Decode the binary image asynchronously in the thread pool executor
            def decode_and_verify(payload):
                nparr = np.frombuffer(payload, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            img = await loop.run_in_executor(None, decode_and_verify, data)
            if img is not None:
                processor.push_frame(img, time.time())
            else:
                logger.warning("Session %s: Failed to decode binary image frame", session_id)

    except WebSocketDisconnect:
        logger.info("Streaming session %s disconnected", session_id)
    except Exception:
        logger.error("Streaming session %s error", session_id, exc_info=True)
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Internal server error"
            }))
        except Exception:
            pass
    finally:
        result_task.cancel()
        processor.close()
        logger.info("Streaming session %s cleaned up", session_id)
