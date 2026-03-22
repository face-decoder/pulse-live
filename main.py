from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import CKPT_PATH, NORM_PATH
from src.api.websocket import router as websocket_router
from src.api.webrtc import router as webrtc_router, set_inferencer

logger = logging.getLogger(__name__)

# Configure root logger so INFO logs appear in the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: load inferencer on startup, cleanup on shutdown."""
    from src.model.modules import AnxietyInferencer

    logger.info("Loading inferencer...")
    norm = NORM_PATH if Path(NORM_PATH).exists() else None
    inf = AnxietyInferencer(checkpoint_path=CKPT_PATH, normalizer_path=norm)
    set_inferencer(inf)
    logger.info("Inferencer ready.")

    yield

    logger.info("Shutting down — cleaning up resources.")


app = FastAPI(title="Pulse Live API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(websocket_router)
app.include_router(webrtc_router)


@app.get("/")
def read_root() -> dict[str, str]:
    """Health-check endpoint."""
    return {"message": "Hello from pulse live!"}


def main() -> None:
    """Run the development server."""
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
