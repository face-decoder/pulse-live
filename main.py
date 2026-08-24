from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv(override=False)

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.history import router as history_router
from src.api.logs import router as logs_router
from src.api.video_process import router as video_process_router
from src.api.webrtc import router as webrtc_router
from src.api.websocket import router as websocket_router

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("real-time.log", mode="a", encoding="utf-8"),
    ],
    force=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from src.models.inferencer import load_inferencer_from_env

    logger.info("Loading inferencer from environment...")
    try:
        inf = load_inferencer_from_env()
        logger.info("Inferencer ready: %r", inf)
    except Exception:
        logger.error(
            "Failed to load inferencer — check MODEL_COMBINATION_ID and "
            "MODEL_CHECKPOINT_PATH in .env. Predictions will be unavailable.",
            exc_info=True,
        )

    try:
        from src.storage.modules import get_minio_storage

        get_minio_storage()
        logger.info("MinIO storage connected.")
    except Exception:
        logger.warning(
            "MinIO storage unavailable — artifact persistence disabled.", exc_info=True
        )

    yield

    logger.info("Shutting down — cleaning up resources.")


app = FastAPI(title="Pulse Live API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(history_router)
app.include_router(logs_router)
app.include_router(websocket_router)
app.include_router(webrtc_router)
app.include_router(video_process_router)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Hello from pulse live!"}


def main() -> None:
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
