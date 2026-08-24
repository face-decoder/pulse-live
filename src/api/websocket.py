import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.connection_manager import ConnectionManager

logger = logging.getLogger(__name__)

router = APIRouter()

manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            logger.info("Echo message received: %s", data)
            await manager.send_personal_message(f"You wrote: {data}", websocket)
            await manager.broadcast(f"Client says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast("A client left the chat")
