import asyncio
import json
import logging
import time

from aiortc import RTCPeerConnection
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.result_sender import ResultSender
from src.api.session_state import SessionState
from src.api.stream_processor import AnxietyStreamProcessor
from src.api.video_track import AnxietyVideoTrack
from src.image.modules import decode_jpeg

logger = logging.getLogger(__name__)

router = APIRouter()

__all__ = ["router", "AnxietyStreamProcessor", "AnxietyVideoTrack"]


@router.websocket("/ws/rtc/{session_id}")
async def webrtc_signaling(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    logger.info("Session %s connected", session_id)

    state = SessionState(pc=RTCPeerConnection(), ws=websocket, session_id=session_id)
    state.start()

    try:
        async for raw in websocket.iter_text():
            if not await state.dispatch(json.loads(raw)):
                break

    except WebSocketDisconnect:
        logger.info("Session %s disconnected", session_id)
    except Exception:
        logger.error("Session %s error", session_id, exc_info=True)
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "Internal server error"})
            )
        except Exception:  # noqa: BLE001
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
    result_task = asyncio.create_task(ResultSender(websocket, result_queue).run())
    loop = asyncio.get_running_loop()

    try:
        async for data in websocket.iter_bytes():
            if not data:
                continue

            img = await loop.run_in_executor(None, decode_jpeg, data)
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
        except Exception:  # noqa: BLE001
            pass
    finally:
        result_task.cancel()
        processor.close()
        logger.info("Streaming session %s cleaned up", session_id)
