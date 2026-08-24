import asyncio
import json
import logging
from dataclasses import dataclass, field

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp
from fastapi import WebSocket

from src.api.result_sender import ResultSender
from src.api.video_track import AnxietyVideoTrack

logger = logging.getLogger(__name__)


def _ice_payload(candidate: object) -> dict[str, object]:
    return {
        "type": "candidate",
        "candidate": {
            "candidate": candidate.to_sdp(),  # pyright: ignore[reportAttributeAccessIssue]
            "sdpMid": candidate.sdpMid,  # pyright: ignore[reportAttributeAccessIssue]
            "sdpMLineIndex": candidate.sdpMLineIndex,  # pyright: ignore[reportAttributeAccessIssue]
        },
    }


async def _consume(track: AnxietyVideoTrack) -> None:
    try:
        while True:
            await track.recv()
    except Exception:  # noqa: BLE001
        pass


@dataclass
class SessionState:
    pc: RTCPeerConnection
    ws: WebSocket
    session_id: str
    result_queue: asyncio.Queue[dict[str, object]] = field(
        default_factory=asyncio.Queue,
    )
    video_track: AnxietyVideoTrack | None = None
    result_task: asyncio.Task[None] | None = None
    consume_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self.pc.on("connectionstatechange")(self.__on_state_change)
        self.pc.on("icecandidate")(self.__on_icecandidate)
        self.pc.on("track")(self.__on_track)
        self.result_task = asyncio.create_task(
            ResultSender(self.ws, self.result_queue).run()
        )

    async def dispatch(self, msg: dict[str, object]) -> bool:
        match msg.get("type"):
            case "offer":
                answer = await self.__answer(str(msg["sdp"]), str(msg["sdpType"]))
                raw = json.dumps(answer)
                logger.info(
                    "Sending SDP answer to session %s: %s", self.session_id, raw
                )
                await self.ws.send_text(raw)
            case "candidate":
                await self.__add_candidate(msg["candidate"])
            case "stop":
                return False
        return True

    async def cleanup(self) -> None:
        if self.result_task is not None:
            self.result_task.cancel()
        if self.consume_task is not None:
            self.consume_task.cancel()
        if self.video_track is not None:
            self.video_track.stop()
        await self.pc.close()

    async def __on_state_change(self) -> None:
        logger.info(
            "Session %s WebRTC connection state changed to: %s",
            self.session_id,
            self.pc.connectionState,
        )

    async def __on_icecandidate(self, candidate: object) -> None:
        if candidate is None:
            return
        raw = json.dumps(_ice_payload(candidate))
        logger.info("Sending ICE candidate to session %s: %s", self.session_id, raw)
        await self.ws.send_text(raw)

    def __on_track(self, track: MediaStreamTrack) -> None:
        if track.kind != "video":
            return
        if self.consume_task is not None:
            self.consume_task.cancel()
        if self.video_track is not None:
            self.video_track.stop()

        local_track = AnxietyVideoTrack(
            track, self.result_queue, session_id=self.session_id
        )
        self.video_track = local_track
        self.consume_task = asyncio.create_task(_consume(local_track))
        logger.info("Video track received for session %s", self.session_id)

    async def __answer(self, sdp: str, sdp_type: str) -> dict[str, object]:
        offer = RTCSessionDescription(sdp=sdp, type=sdp_type)
        await self.pc.setRemoteDescription(offer)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return {
            "type": "answer",
            "sdp": self.pc.localDescription.sdp,
            "sdpType": self.pc.localDescription.type,
        }

    async def __add_candidate(self, ice: object) -> None:
        if not isinstance(ice, dict):
            logger.warning(
                "Invalid ICE candidate format from session %s", self.session_id
            )
            return

        sdp_str = str(ice.get("candidate", ""))
        if not sdp_str:
            logger.info(
                "Received end of ICE candidates for session %s", self.session_id
            )
            return

        try:
            if sdp_str.startswith("candidate:"):
                sdp_str = sdp_str.split(":", 1)[1]
            candidate = candidate_from_sdp(sdp_str)
            candidate.sdpMid = ice.get("sdpMid")
            candidate.sdpMLineIndex = ice.get("sdpMLineIndex")
            await self.pc.addIceCandidate(candidate)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to parse ICE candidate from session %s: %s",
                self.session_id,
                e,
            )
