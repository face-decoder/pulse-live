"""Shared configuration constants for the API layer.

Centralises inference timing, model paths, and ICE server settings
so they can be imported by both the WebRTC and WebSocket modules.
"""

from __future__ import annotations

CKPT_PATH: str = "./notebooks/checkpoints_lopo/lopo_fold00_uar1.0000.pt"
NORM_PATH: str = "./notebooks/checkpoints_lopo/lopo_fold00_uar1.0000_normalizer.npz"

WINDOW_SECONDS: float = 1.5
"""Accumulate frames for this many seconds before triggering inference."""

MIN_FRAMES: int = 20
"""Minimum frame count in a window to trigger inference."""

TARGET_FPS: int = 15
"""Target FPS from the browser — excess frames are dropped."""

# ── WebRTC / ICE ──────────────────────────────────────────────────────
STUN_SERVERS: list[str] = []
"""STUN server URIs.  Empty = LAN-only (no STUN).

Set to ``["stun:stun.l.google.com:19302"]`` for internet use.
"""

# ── Heartbeat ─────────────────────────────────────────────────────────
HEARTBEAT_TIMEOUT_SECONDS: float = 30.0
"""Send a heartbeat when no prediction arrives within this interval."""
