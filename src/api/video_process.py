from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import cast

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.config import Stream
from src.api.stream_processor import AnxietyStreamProcessor
from src.image.modules import decode_jpeg

logger = logging.getLogger(__name__)

router = APIRouter()

_MJPEG_ARGS = (
    "ffmpeg",
    "-i",
    "pipe:0",
    "-r",
    str(Stream.FPS),
    "-f",
    "image2pipe",
    "-vcodec",
    "mjpeg",
    "pipe:1",
)


def _extract_jpeg_frames(buffer: bytes) -> tuple[list[bytes], bytes]:
    frames: list[bytes] = []
    while True:
        start = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9")
        if start == -1 or end == -1 or end <= start:
            break
        frames.append(buffer[start : end + 2])
        buffer = buffer[end + 2 :]
    return frames, buffer


async def _send_results_with_summary(
    websocket: WebSocket,
    result_queue: asyncio.Queue[dict[str, object]],
    processor: AnxietyStreamProcessor,
) -> None:
    summary: dict[str, object] = {
        "total_windows": 0,
        "anxiety_detected": 0,
        "avg_confidence": 0.0,
    }
    confidences: list[float] = []

    async def _send(payload: dict[str, object]) -> None:
        await websocket.send_text(json.dumps(payload))

    while True:
        try:
            result = await asyncio.wait_for(
                result_queue.get(), timeout=Stream.HEARTBEAT_SECONDS
            )
        except asyncio.TimeoutError:
            await _send({"type": "heartbeat"})
            continue
        except Exception:  # noqa: BLE001
            break

        result.pop("is_logged", False)
        result_type = result.get("type")

        if result_type in ("bbox", "prediction"):
            if result_type == "prediction":
                summary = _update_summary(summary, confidences, result)

            mags = processor.all_magnitudes.copy()
            if len(mags) > 0:
                summary.update(processor.summarize_signal(mags))
                summary["magnitudes"] = mags

            await _send(result)
            if "magnitudes" in summary:
                await _send({"type": "summary", "data": summary})
            continue

        if result_type == "status" and result.get("status") == "completed":
            await _send(result)
            break

        await _send(result)


def _update_summary(
    summary: dict[str, object],
    confidences: list[float],
    result: dict[str, object],
) -> dict[str, object]:
    total = cast(int, summary["total_windows"])
    summary["total_windows"] = total + 1
    label = str(result.get("label", "normal")).lower()
    if label not in ("normal", "unavailable"):
        detected = cast(int, summary["anxiety_detected"])
        summary["anxiety_detected"] = detected + 1
    confidence = result.get("confidence")
    if isinstance(confidence, (int, float)):
        confidences.append(float(confidence))
    if confidences:
        summary["avg_confidence"] = round(sum(confidences) / len(confidences), 4)
    return summary


async def _read_mjpeg_frames(
    proc: asyncio.subprocess.Process,
    processor: AnxietyStreamProcessor,
) -> None:
    loop = asyncio.get_running_loop()
    if not proc.stdout:
        return

    buffer = b""
    while True:
        chunk = await proc.stdout.read(8192)
        if not chunk:
            break
        buffer += chunk
        frames, buffer = _extract_jpeg_frames(buffer)
        for jpg_data in frames:
            img = await loop.run_in_executor(None, decode_jpeg, jpg_data)
            if img is not None:
                processor.push_frame(img, time.time())


@router.websocket("/ws/video/{session_id}")
async def websocket_video_process(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    logger.info("Video streaming session %s connected", session_id)

    result_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    processor = AnxietyStreamProcessor(result_queue, session_id=session_id)
    result_task = asyncio.create_task(
        _send_results_with_summary(websocket, result_queue, processor)
    )

    proc: asyncio.subprocess.Process | None = None
    read_task: asyncio.Task | None = None

    try:
        try:
            proc, filename = await _start_session(websocket, session_id)
        except _InvalidStartMessage:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "message": "Expected initial 'start' message.",
                    }
                )
            )
            return
        read_task = asyncio.create_task(_read_mjpeg_frames(proc, processor))

        await _consume_upload(websocket, proc, session_id)

        logger.info("Session %s: upload complete, draining processor", session_id)
        if proc.stdin:
            proc.stdin.close()
        await proc.wait()
        if read_task:
            await read_task

        await processor.flush_pending_inference()

        await result_queue.put(
            {
                "type": "status",
                "status": "completed",
                "message": "Video streaming processing completed successfully.",
            }
        )
        await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        logger.info("Video streaming session %s disconnected", session_id)
    except Exception:
        logger.error("Video streaming session %s error", session_id, exc_info=True)
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "Internal server error"})
            )
        except Exception:  # noqa: BLE001
            pass
    finally:
        result_task.cancel()
        if read_task and not read_task.done():
            read_task.cancel()
        if proc:
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        processor.close()
        logger.info("Video streaming session %s cleaned up", session_id)


class _InvalidStartMessage(Exception): ...


async def _start_session(
    websocket: WebSocket, session_id: str
) -> tuple[asyncio.subprocess.Process, str]:
    raw_msg = await websocket.receive_text()
    meta = json.loads(raw_msg)

    if meta.get("type") != "start":
        raise _InvalidStartMessage(meta.get("type"))

    filename = meta.get("filename", f"{session_id}.mp4")
    logger.info("Session %s: ready to stream '%s'", session_id, filename)
    await websocket.send_text(
        json.dumps(
            {
                "type": "status",
                "status": "receiving",
                "message": f"Ready to receive and stream '{filename}'.",
            }
        )
    )

    proc = await asyncio.create_subprocess_exec(
        *_MJPEG_ARGS,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    return proc, filename


async def _consume_upload(
    websocket: WebSocket,
    proc: asyncio.subprocess.Process,
    session_id: str,
) -> None:
    while True:
        msg = await websocket.receive()

        if "bytes" in msg and msg["bytes"] is not None:
            if proc.stdin:
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
