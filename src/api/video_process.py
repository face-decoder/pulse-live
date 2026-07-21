from __future__ import annotations

import asyncio
import json
import logging
import cv2
import numpy as np
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.webrtc import AnxietyStreamProcessor
from src.api.config import TARGET_FPS

logger = logging.getLogger(__name__)

router = APIRouter()

def _decode(payload: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)

@router.websocket("/ws/video/{session_id}")
async def websocket_video_process(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    logger.info("Video streaming session %s connected", session_id)

    result_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    processor = AnxietyStreamProcessor(result_queue, session_id=session_id)

    async def _send_results_with_summary():
        summary = {
            "total_windows": 0,
            "anxiety_detected": 0,
            "avg_confidence": 0.0,
        }
        confidences = []

        from src.api.config import HEARTBEAT_TIMEOUT_SECONDS
        while True:
            try:
                result = await asyncio.wait_for(result_queue.get(), timeout=HEARTBEAT_TIMEOUT_SECONDS)
                result.pop("is_logged", False)

                if result.get("type") in ("bbox", "prediction"):
                    if result.get("type") == "prediction":
                        summary["total_windows"] += 1
                        label = str(result.get("label", "normal")).lower()
                        if label not in ("normal", "unavailable"):
                            summary["anxiety_detected"] += 1
                        if "confidence" in result and isinstance(result["confidence"], (int, float)):
                            confidences.append(float(result["confidence"]))

                        if confidences:
                            summary["avg_confidence"] = round(sum(confidences) / len(confidences), 4)

                    mags = processor.all_magnitudes.copy()
                    if len(mags) > 0:
                        if len(mags) > 5:
                            try:
                                from scipy.signal import savgol_filter
                                from src.apex.modules import ApexSmoother
                                wl = ApexSmoother.calculate_window_length(len(mags))
                                po = ApexSmoother.calculate_polyorder(wl)
                                summary["smoothed_magnitudes"] = [float(x) for x in savgol_filter(mags, wl, po)]

                                windows, meta = processor._phase_spotter.detect_windows_from_signal(mags)
                                actual = meta.get("phases", {}) if meta.get("valid", False) else {}
                                summary["detected_phases"] = [
                                    {"onset": int(actual.get(ap, {}).get("start", 0)), "apex": int(ap), "offset": int(actual.get(ap, {}).get("end", 0))}
                                    for _, ap, _ in windows
                                ]
                            except Exception:
                                summary["smoothed_magnitudes"] = mags
                                summary["detected_phases"] = []
                        else:
                            summary["smoothed_magnitudes"] = mags
                            summary["detected_phases"] = []

                        summary["magnitudes"] = mags

                    await websocket.send_text(json.dumps(result))
                    if "magnitudes" in summary:
                        await websocket.send_text(json.dumps({"type": "summary", "data": summary}))
                    continue

                if result.get("type") == "status" and result.get("status") == "completed":
                    await websocket.send_text(json.dumps(result))
                    break

                await websocket.send_text(json.dumps(result))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
            except Exception:
                break

    result_task = asyncio.create_task(_send_results_with_summary())

    proc: asyncio.subprocess.Process | None = None
    read_task: asyncio.Task | None = None
    loop = asyncio.get_running_loop()

    try:
        raw_msg = await websocket.receive_text()
        meta = json.loads(raw_msg)

        if meta.get("type") != "start":
            await websocket.send_text(
                json.dumps({
                    "type": "error",
                    "message": "Expected initial 'start' message.",
                })
            )
            return

        filename = meta.get("filename", f"{session_id}.mp4")
        logger.info("Session %s: ready to stream '%s'", session_id, filename)
        await websocket.send_text(
            json.dumps({
                "type": "status",
                "status": "receiving",
                "message": f"Ready to receive and stream '{filename}'.",
            })
        )

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", "pipe:0", "-r", str(TARGET_FPS), "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )

        async def _read_frames():
            if not proc or not proc.stdout:
                return
            buffer = b""
            while True:
                chunk = await proc.stdout.read(8192)
                if not chunk:
                    break
                buffer += chunk
                while True:
                    start = buffer.find(b"\xff\xd8")
                    end = buffer.find(b"\xff\xd9")
                    if start != -1 and end != -1 and end > start:
                        jpg_data = buffer[start:end+2]
                        buffer = buffer[end+2:]
                        img = await loop.run_in_executor(None, _decode, jpg_data)
                        if img is not None:
                            processor.push_frame(img, time.time())
                    else:
                        break

        read_task = asyncio.create_task(_read_frames())

        while True:
            msg = await websocket.receive()

            if "bytes" in msg and msg["bytes"] is not None:
                if proc and proc.stdin:
                    proc.stdin.write(msg["bytes"])
                    try:
                        await proc.stdin.drain()
                    except ConnectionResetError:
                        break
                continue

            if "text" in msg and msg["text"] is not None:
                try:
                    text_msg = json.loads(msg["text"])
                    if text_msg.get("type") == "end":
                        break
                except json.JSONDecodeError:
                    pass

        logger.info("Session %s: upload complete, draining processor", session_id)
        if proc and proc.stdin:
            proc.stdin.close()
            await proc.wait()

        if read_task:
            await read_task

        await processor._processing_queue.join()

        if not processor._inference_in_progress and len(processor._magnitudes_buf) > 0:
            processor._inference_in_progress = True
            asyncio.create_task(
                processor._run_inference_background(
                    list(processor._magnitudes_buf),
                    list(processor._flows_buf),
                    list(processor._bboxes_buf),
                    time.time(),
                    list(processor._webrtc_latencies_buf),
                    list(processor._landmark_latencies_buf),
                    list(processor._flow_latencies_buf),
                    list(processor._timestamps_buf),
                )
            )

        while processor._inference_in_progress:
            await asyncio.sleep(0.1)

        await result_queue.put({
            "type": "status",
            "status": "completed",
            "message": "Video streaming processing completed successfully."
        })

        await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        logger.info("Video streaming session %s disconnected", session_id)
    except Exception:
        logger.error("Video streaming session %s error", session_id, exc_info=True)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": "Internal server error"}))
        except Exception:
            pass
    finally:
        if result_task:
            result_task.cancel()
        if read_task and not read_task.done():
            read_task.cancel()
        if proc:
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.kill()
            except Exception:
                pass
        processor.close()
        logger.info("Video streaming session %s cleaned up", session_id)
