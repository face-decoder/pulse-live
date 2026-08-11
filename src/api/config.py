from __future__ import annotations

import os

CKPT_PATH: str = os.getenv(
    "MODEL_CHECKPOINT_PATH",
    "./notebooks/checkpoints_lopo/lopo_fold00_uar1.0000.pt",
)
NORM_PATH: str = os.getenv(
    "MODEL_NORM_PATH",
    "./notebooks/checkpoints_lopo/lopo_fold00_uar1.0000_normalizer.npz",
)

WINDOW_SECONDS: float = float(os.getenv("WINDOW_SECONDS", "1.5"))

MIN_FRAMES: int = int(os.getenv("MIN_FRAMES", "20"))

TARGET_FPS: int = int(os.getenv("TARGET_FPS", "15"))

STUN_SERVERS: list[str] = []

HEARTBEAT_TIMEOUT_SECONDS: float = 30.0
