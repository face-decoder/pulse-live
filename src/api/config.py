"""Shared configuration constants for the API layer.

Centralises inference timing, model paths, and ICE server settings
so they can be imported by both the WebRTC and WebSocket modules.

Model selection is fully env-driven. Set these variables in ``.env``:

    MODEL_COMBINATION_ID=0407
    MODEL_CHECKPOINT_PATH=combinations-notebooks/checkpoints_0407-.../best_model.pt
    MODEL_DEVICE=cuda        # optional, auto-detected
    MODEL_N_TTA=8            # optional, default 8
"""

from __future__ import annotations

import os

# ── Legacy single-model paths (kept for backwards compatibility) ──────────────
# These are only used if AnxietyInferencer from src.model.modules is still
# referenced anywhere. New code should use the registry instead.
CKPT_PATH: str = os.getenv(
    "MODEL_CHECKPOINT_PATH",
    "./notebooks/checkpoints_lopo/lopo_fold00_uar1.0000.pt",
)
NORM_PATH: str = os.getenv(
    "MODEL_NORM_PATH",
    "./notebooks/checkpoints_lopo/lopo_fold00_uar1.0000_normalizer.npz",
)

WINDOW_SECONDS: float = float(os.getenv("WINDOW_SECONDS", "1.5"))
"""Accumulate frames for this many seconds before triggering inference.

Lower = faster first result, less temporal context for the model.
Set via env: WINDOW_SECONDS=1.0
"""

MIN_FRAMES: int = int(os.getenv("MIN_FRAMES", "20"))
"""Minimum flow-frame count to trigger inference (must be <= WINDOW_SECONDS * TARGET_FPS).

Set via env: MIN_FRAMES=15
"""

TARGET_FPS: int = int(os.getenv("TARGET_FPS", "15"))
"""Target FPS from the browser — excess frames are dropped.

Set via env: TARGET_FPS=10
"""

# ── WebRTC / ICE ──────────────────────────────────────────────────────
STUN_SERVERS: list[str] = []
"""STUN server URIs.  Empty = LAN-only (no STUN).

Set to ``["stun:stun.l.google.com:19302"]`` for internet use.
"""

# ── Heartbeat ─────────────────────────────────────────────────────────
HEARTBEAT_TIMEOUT_SECONDS: float = 30.0
"""Send a heartbeat when no prediction arrives within this interval."""
