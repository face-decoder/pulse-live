from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from aiortc import (
    MediaStreamTrack,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.sdp import candidate_from_sdp
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.config import (
    HEARTBEAT_TIMEOUT_SECONDS,
    MIN_FRAMES,
    TARGET_FPS,
    WINDOW_SECONDS,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class AnxietyStreamProcessor:
    def __init__(
        self,
        result_queue: asyncio.Queue[dict[str, object]],
        session_id: str = "unknown",
    ) -> None:
        self._result_queue = result_queue
        self._session_id = session_id
        self._last_saved_time = 0.0

        from src.apex.modules import ApexPhaseSpotterROI
        from src.face.modules import FaceLandmark, FaceRoiPoints
        from src.face.modules.face_aligner import FaceAligner
        from src.optical_flow.modules import TVL1

        self._landmarker = FaceLandmark()
        self._aligner = FaceAligner()
        self._tvl1 = TVL1(fast_mode=True)

        self._roi_defs = [
            frozenset(FaceRoiPoints.LEFT_EYE_POINTS),
            frozenset(FaceRoiPoints.RIGHT_EYE_POINTS),
            frozenset(FaceRoiPoints.LIPS_POINTS),
            frozenset(FaceRoiPoints.LEFT_EYEBROW_POINTS),
            frozenset(FaceRoiPoints.RIGHT_EYEBROW_POINTS),
        ]
        self._tile_size = (64, 64)
        self._margin = 0.05
        self._cols = 3
        self._rows = math.ceil(len(self._roi_defs) / self._cols)

        self._phase_spotter = ApexPhaseSpotterROI()

        self._inference_in_progress = False
        self._max_window_len = int(WINDOW_SECONDS * TARGET_FPS)
        self._landmark_thread_lock = threading.Lock()

        self._last_crops: list[np.ndarray] | None = None

        self._magnitudes_buf: deque[float] = deque(maxlen=self._max_window_len - 1)
        self._flows_buf: deque[np.ndarray] = deque(maxlen=self._max_window_len - 1)
        self._bboxes_buf: deque[dict | None] = deque(maxlen=self._max_window_len)

        self._webrtc_latencies_buf: deque[float] = deque(maxlen=self._max_window_len)
        self._landmark_latencies_buf: deque[float] = deque(maxlen=self._max_window_len)
        self._flow_latencies_buf: deque[float] = deque(maxlen=self._max_window_len - 1)
        self._timestamps_buf: deque[float] = deque(maxlen=self._max_window_len)

        self.all_magnitudes: list[float] = []  # ponytail: global history for full-video graphs

        self._processing_queue = asyncio.Queue()
        self._process_loop_task = asyncio.create_task(self._process_loop())

    def _get_face_bbox(self, landmarks, image: np.ndarray) -> dict | None:
        try:
            face = (
                landmarks.face_landmarks[0]
                if landmarks and landmarks.face_landmarks
                else None
            )
            if face is None:
                return None
            xs = [lm.x for lm in face]
            ys = [lm.y for lm in face]
            return {
                "x": float(min(xs)),
                "y": float(min(ys)),
                "width": float(max(xs) - min(xs)),
                "height": float(max(ys) - min(ys)),
            }
        except Exception:
            return None

    def push_frame(
        self, img: np.ndarray, received_at: float, webrtc_latency: float = 0.0
    ) -> None:
        while self._processing_queue.qsize() > 2:
            try:
                self._processing_queue.get_nowait()
                self._processing_queue.task_done()
            except asyncio.QueueEmpty:
                break
        self._processing_queue.put_nowait((img, received_at, webrtc_latency))

    async def _process_loop(self) -> None:
        while True:
            try:
                img, received_at, webrtc_latency = await self._processing_queue.get()
            except asyncio.CancelledError:
                break
            except Exception:
                continue

            try:
                await self._process_frame(img, received_at, webrtc_latency)
            except Exception:
                logger.error("Error in process_frame background task", exc_info=True)
            finally:
                self._processing_queue.task_done()

    async def _process_frame(
        self, img: np.ndarray, received_at: float, webrtc_latency: float
    ) -> None:
        loop = asyncio.get_running_loop()

        def detect_and_crop(image):
            with self._landmark_thread_lock:
                landmarks = self._landmarker.detect(image)
                bbox = self._get_face_bbox(landmarks, image)

                try:
                    aligned = self._aligner.align(image=image, landmarks=landmarks)
                    aligned_landmarks = self._landmarker.detect(aligned)
                except Exception:
                    aligned_landmarks = landmarks

                crops = []
                for roi_points in self._roi_defs:
                    try:
                        roi, _ = self._landmarker.crop_roi(
                            image=image,
                            landmark_result=aligned_landmarks,
                            roi_points=roi_points,
                            margin=self._margin,
                            target_size=self._tile_size,
                        )
                    except Exception:
                        roi = np.zeros(
                            (self._tile_size[1], self._tile_size[0], 3), dtype=np.uint8
                        )
                    crops.append(roi)

            return landmarks, bbox, crops

        landmark_start = time.time()
        landmarks, bbox, crops = await loop.run_in_executor(None, detect_and_crop, img)
        landmark_latency_ms = (time.time() - landmark_start) * 1000

        latency_ms = (time.time() - received_at) * 1000
        logger.info(
            "Frame processing: WebRTC latency = %.2f ms | Landmark & ROI latency = %.2f ms",
            webrtc_latency,
            landmark_latency_ms,
        )
        await self._result_queue.put(
            {
                "type": "bbox",
                "bbox": bbox,
                "latency_ms": round(latency_ms, 2),
            }
        )

        self._webrtc_latencies_buf.append(webrtc_latency)
        self._landmark_latencies_buf.append(landmark_latency_ms)
        self._timestamps_buf.append(received_at)

        prev_crops = self._last_crops

        if prev_crops is not None:

            def compute_batch_flow(p_crops, c_crops):
                with self._landmark_thread_lock:
                    pairs = list(
                        zip(p_crops, c_crops)
                    )  # ponytail: zip replaces index loop
                    flows = self._tvl1.compute_batch(pairs, download=True)

                    flow_canvas = np.zeros(
                        (
                            self._rows * self._tile_size[1],
                            self._cols * self._tile_size[0],
                            2,
                        ),
                        dtype=np.float32,
                    )
                    roi_magnitudes = []
                    for idx, flow in enumerate(flows):
                        row, col = divmod(idx, self._cols)
                        y1, y2 = (
                            row * self._tile_size[1],
                            (row + 1) * self._tile_size[1],
                        )
                        x1, x2 = (
                            col * self._tile_size[0],
                            (col + 1) * self._tile_size[0],
                        )
                        flow_canvas[y1:y2, x1:x2, :] = flow
                        roi_magnitudes.append(
                            float(np.mean(np.hypot(flow[..., 0], flow[..., 1])))
                        )

                    mag = float(np.mean(roi_magnitudes))
                return mag, flow_canvas

            flow_start = time.time()
            mag, flow_canvas = await loop.run_in_executor(
                None, compute_batch_flow, prev_crops, crops
            )
            flow_latency_ms = (time.time() - flow_start) * 1000
            logger.info(
                "Optical flow (TV-L1) calculation completed. Latency: %.2f ms",
                flow_latency_ms,
            )

            self._magnitudes_buf.append(mag)
            self.all_magnitudes.append(mag)
            self._flows_buf.append(flow_canvas)
            self._bboxes_buf.append(bbox)
            self._flow_latencies_buf.append(flow_latency_ms)
        else:
            self._bboxes_buf.append(bbox)

        self._last_crops = crops

        if (
            len(self._magnitudes_buf) >= MIN_FRAMES  # ponytail: was _max_window_len-1; MIN_FRAMES cuts cold-start
            and not self._inference_in_progress
        ):
            self._inference_in_progress = True
            mags_copy = list(self._magnitudes_buf)
            flows_copy = list(self._flows_buf)
            bboxes_copy = list(self._bboxes_buf)
            webrtc_lats_copy = list(self._webrtc_latencies_buf)
            landmark_lats_copy = list(self._landmark_latencies_buf)
            flow_lats_copy = list(self._flow_latencies_buf)
            timestamps_copy = list(self._timestamps_buf)
            logger.info(
                "Triggering background model inference (window buffer full with %d flow frames)",
                len(flows_copy),
            )
            asyncio.create_task(
                self._run_inference_background(
                    mags_copy,
                    flows_copy,
                    bboxes_copy,
                    received_at,
                    webrtc_lats_copy,
                    landmark_lats_copy,
                    flow_lats_copy,
                    timestamps_copy,
                )
            )

    async def _run_inference_background(
        self,
        mags: list[float],
        flows: list[np.ndarray],
        bboxes: list[dict | None],
        received_at: float,
        webrtc_lats: list[float],
        landmark_lats: list[float],
        flow_lats: list[float],
        timestamps: list[float],
    ) -> None:
        """Run the actual deep learning model inference in the background."""
        import uuid  # ponytail: lazy import, no extra deps
        import os
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                self._run_inference,
                mags,
                flows,
                bboxes,
                received_at,
                webrtc_lats,
                landmark_lats,
                flow_lats,
            )
            if result is not None:
                if "latency_ms" in result and result["latency_ms"] > 0: # type: ignore
                    # processing fps = jumlah frame / processing_time_in_seconds
                    processing_time_sec = result["latency_ms"] / 1000.0 # type: ignore
                    true_fps = len(timestamps) / processing_time_sec
                    result["fps"] = round(true_fps, 2)
                else:
                    result["fps"] = 0.0

                current_time = time.time()
                if current_time - self._last_saved_time >= WINDOW_SECONDS:
                    self._last_saved_time = current_time
                    
                    # ponytail: minimum log locally then minio, creating new version by unique id
                    detection_id = uuid.uuid4().hex
                    result["detection_id"] = detection_id
                    
                    log_data = json.dumps(result, indent=2).encode("utf-8")
                    
                    session_dir = os.path.join("logs", self._session_id)
                    os.makedirs(session_dir, exist_ok=True)
                    local_path = os.path.join(session_dir, f"detection_{detection_id}.json")
                    with open(local_path, "wb") as f:
                        f.write(log_data)
                    
                    try:
                        from src.storage.modules import get_minio_storage
                        get_minio_storage().upload_bytes(
                            object_name=f"detections/{self._session_id}/detection_{detection_id}.json",
                            data=log_data,
                            content_type="application/json"
                        )
                    except Exception as e:
                        logger.error("Failed to upload log to MinIO: %s", e)
                    
                    result["is_logged"] = True
                else:
                    result["is_logged"] = False
                
                await self._result_queue.put(result)
        except Exception:
            logger.error("Background inference failed", exc_info=True)
        finally:
            self._inference_in_progress = False

    def _run_inference(
        self,
        mags: list[float],
        flows: list[np.ndarray],
        bboxes: list[dict | None],
        received_at: float,
        webrtc_lats: list[float],
        landmark_lats: list[float],
        flow_lats: list[float],
    ) -> dict[str, object] | None:
        from src.models.inferencer import (
            get_loaded_inferencer,
            load_inferencer_from_env,
        )

        inf = get_loaded_inferencer()
        if inf is None:
            try:
                inf = load_inferencer_from_env()
            except Exception:
                logger.warning(
                    "Inferencer not loaded and fallback env loading failed — skipping prediction",
                    exc_info=True,
                )
                return None

        if len(mags) < 1:
            return None

        smoothed_mags: list[float] = []
        try:
            from scipy.signal import savgol_filter

            from src.apex.modules import ApexSmoother

            window_length = ApexSmoother.calculate_window_length(len(mags))
            polyorder = ApexSmoother.calculate_polyorder(window_length)
            smoothed_mags = [
                float(x) for x in savgol_filter(mags, window_length, polyorder)
            ]

            # ponytail: lock dropped — detect_windows_from_signal only reads mags (a passed-in list snapshot)
            windows, meta = self._phase_spotter.detect_windows_from_signal(mags)
            actual_phases = meta.get("phases", {}) if meta.get("valid", False) else {}
            detected_phases = [
                {
                    "onset": int(actual_phases.get(apex, {}).get("start", 0)),
                    "apex": int(apex),
                    "offset": int(actual_phases.get(apex, {}).get("end", 0)),
                }
                for _, apex, _ in windows
            ]
        except Exception:
            detected_phases = []

        try:
            n_roi = len(self._roi_defs)
            tile_h, tile_w = self._tile_size

            frames = []
            for canvas in flows:
                canvas = np.asarray(
                    canvas, dtype=np.float32
                )  # ponytail: dropped _f suffix
                tiles = []

                for idx in range(n_roi):
                    row, col = divmod(idx, self._cols)
                    y1, y2 = row * tile_h, (row + 1) * tile_h
                    x1, x2 = col * tile_w, (col + 1) * tile_w
                    tiles.append(canvas[y1:y2, x1:x2, :].transpose(2, 0, 1))

                frames.append(np.stack(tiles, axis=0))

            if not frames:
                return None

            flow_array = np.stack(frames, axis=0)  # (T, N_roi, 2, tile_h, tile_w)
            result = inf.predict_flow(flow_array)

        except Exception:
            logger.error("Inference pipeline failed", exc_info=True)
            return None

        # ponytail: six single-use stat temps inlined — helper avoids repeating the guard
        def _stat(xs: list[float]) -> tuple[float, float]:
            return (float(np.mean(xs)), float(np.max(xs))) if xs else (0.0, 0.0)

        avg_webrtc, max_webrtc = _stat(webrtc_lats)
        avg_landmark, max_landmark = _stat(landmark_lats)
        avg_flow, max_flow = _stat(flow_lats)
        spotting_ms = result.spotting_latency_ms or 0.0
        model_ms = result.model_inference_latency_ms or 0.0
        total_ms = (time.time() - received_at) * 1000

        logger.info(
            "Inference completed:\n"
            "  - WebRTC: avg=%.2f ms max=%.2f ms\n"
            "  - Landmark: avg=%.2f ms max=%.2f ms\n"
            "  - TVL1 flow: avg=%.2f ms max=%.2f ms\n"
            "  - Phase spotting: %.2f ms | Model: %.2f ms\n"
            "  - Total: %.2f ms | label=%s confidence=%.4f",
            avg_webrtc,
            max_webrtc,
            avg_landmark,
            max_landmark,
            avg_flow,
            max_flow,
            spotting_ms,
            model_ms,
            total_ms,
            result.label,
            result.confidence,
        )

        return {
            "type": "prediction",
            "label": result.label,
            "confidence": round(result.confidence, 4),
            "prob_high": round(result.prob_high, 4),
            "prob_low": round(result.prob_low, 4),
            "n_windows": result.n_windows,
            "n_frames": len(bboxes),
            "warning": result.warning,
            "face_bboxes": bboxes,
            "magnitudes": mags,
            "smoothed_magnitudes": smoothed_mags,
            "detected_phases": detected_phases,
            "latency_ms": round(total_ms, 2),
            "webrtc_latency_avg_ms": round(avg_webrtc, 2),
            "webrtc_latency_max_ms": round(max_webrtc, 2),
            "landmark_latency_avg_ms": round(avg_landmark, 2),
            "landmark_latency_max_ms": round(max_landmark, 2),
            "flow_latency_avg_ms": round(avg_flow, 2),
            "flow_latency_max_ms": round(max_flow, 2),
            "spotting_latency_ms": round(spotting_ms, 2),
            "model_inference_latency_ms": round(model_ms, 2),
        }

    def close(self) -> None:
        if hasattr(self, "_process_loop_task"):
            self._process_loop_task.cancel()


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
        self._frame_interval: float = 1.0 / TARGET_FPS

    async def recv(self) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]
        frame = await self._track.recv()
        now = time.time()

        if now - self._last_frame_time < self._frame_interval:
            return frame
        self._last_frame_time = now

        if self._window_start is None:
            # Initialize base time on first frame to ignore ICE/DTLS setup delays
            self._window_start = now - frame.time

        img: np.ndarray = frame.to_ndarray(format="bgr24")  # pyright: ignore[reportAttributeAccessIssue]

        webrtc_latency = max(0.0, (now - (self._window_start + frame.time)) * 1000)  # pyright: ignore[reportOperatorIssue, reportAttributeAccessIssue]

        self._processor.push_frame(img, now, webrtc_latency)
        return frame

    def stop(self) -> None:
        super().stop()
        self._processor.close()


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


async def _consume_track(track: AnxietyVideoTrack) -> None:
    try:
        while True:
            await track.recv()
    except Exception:
        pass


async def _send_results(
    ws: WebSocket,
    queue: asyncio.Queue[dict[str, object]],
) -> None:
    while True:
        try:
            result = await asyncio.wait_for(
                queue.get(),
                timeout=HEARTBEAT_TIMEOUT_SECONDS,
            )
            is_logged = result.pop("is_logged", False)
            raw = json.dumps(result)
            if is_logged:
                logger.info(
                    "Sending response to websocket:\n%s", json.dumps(result, indent=2)
                )
            await ws.send_text(raw)
        except asyncio.TimeoutError:
            raw = json.dumps({"type": "heartbeat"})
            logger.info("Sending heartbeat to websocket")
            await ws.send_text(raw)
        except Exception:
            logger.warning("send_results stopped", exc_info=True)
            break


@router.websocket("/ws/rtc/{session_id}")
async def webrtc_signaling(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    logger.info("Session %s connected", session_id)

    state = _SessionState(pc=RTCPeerConnection())

    @state.pc.on("connectionstatechange")
    async def _on_connectionstatechange() -> None:
        logger.info(
            "Session %s WebRTC connection state changed to: %s",
            session_id,
            state.pc.connectionState,
        )

    @state.pc.on("icecandidate")
    async def _on_icecandidate(candidate: object) -> None:
        if candidate is not None:
            raw = json.dumps(
                {
                    "type": "candidate",
                    "candidate": {
                        "candidate": candidate.to_sdp(),  # pyright: ignore[reportAttributeAccessIssue]
                        "sdpMid": candidate.sdpMid,  # pyright: ignore[reportAttributeAccessIssue]
                        "sdpMLineIndex": candidate.sdpMLineIndex,  # pyright: ignore[reportAttributeAccessIssue]
                    },
                }
            )
            logger.info("Sending ICE candidate to session %s: %s", session_id, raw)
            await websocket.send_text(raw)

    @state.pc.on("track")
    def _on_track(track: MediaStreamTrack) -> None:
        if track.kind == "video":
            if state.consume_task is not None:
                state.consume_task.cancel()
            if state.video_track is not None:
                state.video_track.stop()

            local_track = AnxietyVideoTrack(track, state.result_queue, session_id=session_id)

            state.video_track = local_track
            state.consume_task = asyncio.create_task(_consume_track(local_track))
            logger.info("Video track received for session %s", session_id)

    state.result_task = asyncio.create_task(
        _send_results(websocket, state.result_queue)
    )

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

                raw = json.dumps(
                    {
                        "type": "answer",
                        "sdp": state.pc.localDescription.sdp,
                        "sdpType": state.pc.localDescription.type,
                    }
                )
                logger.info("Sending SDP answer to session %s: %s", session_id, raw)
                await websocket.send_text(raw)

            elif msg_type == "candidate":
                ice = msg["candidate"]  # ponytail: c→ice, c_str→sdp_str for clarity
                if not isinstance(ice, dict):
                    logger.warning(
                        "Invalid ICE candidate format from session %s", session_id
                    )
                    continue
                sdp_str = str(ice.get("candidate", ""))

                if not sdp_str:
                    logger.info(
                        "Received end of ICE candidates for session %s", session_id
                    )
                    continue

                try:
                    if sdp_str.startswith("candidate:"):
                        sdp_str = sdp_str.split(":", 1)[1]
                    candidate = candidate_from_sdp(sdp_str)
                    candidate.sdpMid = ice.get("sdpMid")
                    candidate.sdpMLineIndex = ice.get("sdpMLineIndex")
                    await state.pc.addIceCandidate(candidate)
                except Exception as e:
                    logger.warning(
                        "Failed to parse ICE candidate from session %s: %s",
                        session_id,
                        e,
                    )

            elif msg_type == "stop":
                break

    except WebSocketDisconnect:
        logger.info("Session %s disconnected", session_id)
    except Exception:
        logger.error("Session %s error", session_id, exc_info=True)
        try:
            raw = json.dumps(
                {
                    "type": "error",
                    "message": "Internal server error",
                }
            )
            logger.info("Sending error to session %s: %s", session_id, raw)
            await websocket.send_text(raw)
        except Exception:
            pass
    finally:
        await state.cleanup()
        logger.info("Session %s cleaned up", session_id)


@router.websocket("/ws/stream/{session_id}")
async def websocket_video_stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    logger.info("Streaming session %s connected", session_id)

    result_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    processor = AnxietyStreamProcessor(result_queue, session_id=session_id)
    result_task = asyncio.create_task(_send_results(websocket, result_queue))
    loop = asyncio.get_running_loop()

    def _decode(payload: bytes) -> np.ndarray | None:
        return cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)

    try:
        async for data in websocket.iter_bytes():
            if not data:
                continue

            img = await loop.run_in_executor(None, _decode, data)
            if img is not None:
                processor.push_frame(img, time.time())
            else:
                logger.warning(
                    "Session %s: Failed to decode binary image frame", session_id
                )

    except WebSocketDisconnect:
        logger.info("Streaming session %s disconnected", session_id)
    except Exception:
        logger.error("Streaming session %s error", session_id, exc_info=True)
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "Internal server error"})
            )
        except Exception:
            pass
    finally:
        result_task.cancel()
        processor.close()
        logger.info("Streaming session %s cleaned up", session_id)
