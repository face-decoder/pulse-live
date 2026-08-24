from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse

router = APIRouter(prefix="/logs", tags=["logs"])

LOG_PATH = Path("real-time.log")
_NOT_FOUND = HTTPException(status_code=404, detail="Log file not found")

_LOG_PREFIX = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - .+? - INFO - Sending response to websocket:"
)


class _PredictionParser:
    def __init__(self) -> None:
        self._buf: list[str] = []
        self._capturing = False

    def feed(self, line: str) -> dict | None:
        line = line.rstrip()

        if _LOG_PREFIX.match(line):
            self._buf = []
            self._capturing = True
            return None

        if not self._capturing:
            return None

        self._buf.append(line)
        if line != "}":
            return None

        self._capturing = False
        block, self._buf = "\n".join(self._buf), []

        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            return None

        if data.get("type") != "prediction":
            return None

        return {
            "label": data.get("label"),
            "confidence": data.get("confidence"),
            "detected_phases": data.get("detected_phases", []),
            "magnitudes": data.get("magnitudes", []),
            "smoothed_magnitudes": data.get("smoothed_magnitudes", []),
            "latency_ms": data.get("latency_ms"),
        }


def _extract_predictions(lines: Iterable[str]) -> Iterator[dict]:
    parser = _PredictionParser()
    for line in lines:
        entry = parser.feed(line)
        if entry is not None:
            yield entry


@router.get("", summary="Read log file (last N lines)")
def read_log(
    lines: int = Query(
        default=200, ge=1, le=50_000, description="Number of tail lines to return"
    ),
) -> PlainTextResponse:
    if not LOG_PATH.exists():
        raise _NOT_FOUND
    with LOG_PATH.open("r", encoding="utf-8", errors="replace") as f:
        return PlainTextResponse("\n".join(deque(f, maxlen=lines)))


@router.get("/summary", summary="Parsed prediction summary from log")
def log_summary(
    last: int | None = Query(
        default=None,
        ge=1,
        description="Return last N predictions. If not provided, returns all.",
    ),
) -> list[dict]:
    if not LOG_PATH.exists():
        raise _NOT_FOUND
    with LOG_PATH.open("r", encoding="utf-8", errors="replace") as f:
        predictions = _extract_predictions(f)
        if last is not None:
            return list(deque(predictions, maxlen=last))
        return list(predictions)


@router.get("/stream", summary="Stream prediction summaries via SSE")
async def stream_log(
    history: int = Query(
        default=5, ge=0, le=100, description="Historical predictions to send on connect"
    ),
) -> StreamingResponse:
    if not LOG_PATH.exists():
        raise _NOT_FOUND

    async def _stream():
        with LOG_PATH.open("r", encoding="utf-8", errors="replace") as f:
            historical = list(deque(_extract_predictions(f), maxlen=history))
            for entry in historical:
                yield f"data: {json.dumps(entry)}\n\n"

            parser = _PredictionParser()
            while True:
                line = f.readline()
                if not line:
                    await asyncio.sleep(0.3)
                    continue

                entry = parser.feed(line)
                if entry is not None:
                    yield f"data: {json.dumps(entry)}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")
