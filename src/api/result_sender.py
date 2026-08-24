import asyncio
import json
import logging

from fastapi import WebSocket

from src.api.config import Stream

logger = logging.getLogger(__name__)


class ResultSender:
    HEARTBEAT: dict[str, object] = {"type": "heartbeat"}

    def __init__(
        self,
        ws: WebSocket,
        queue: asyncio.Queue[dict[str, object]],
    ) -> None:
        self.__ws = ws
        self.__queue = queue

    async def run(self) -> None:
        while True:
            try:
                result = await asyncio.wait_for(
                    self.__queue.get(),
                    timeout=Stream.HEARTBEAT_SECONDS,
                )
            except TimeoutError:
                logger.info("Sending heartbeat to websocket")
                await self.__send(self.HEARTBEAT.copy())
                continue
            except Exception:
                logger.warning("send_results stopped", exc_info=True)
                break

            logged = result.pop("is_logged", False)
            if logged:
                logger.info(
                    "Sending response to websocket:\n%s", json.dumps(result, indent=2)
                )
            await self.__send(result)

    async def __send(self, result: dict[str, object]) -> None:
        await self.__ws.send_text(json.dumps(result))
