from __future__ import annotations

import os


class Env:
    """Static helpers for reading typed values from environment variables."""

    _TRUE_VALUES = {"1", "true", "yes", "on"}

    @staticmethod
    def get_str(name: str, default: str = "") -> str:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip()

    @staticmethod
    def get_int(name: str, default: int) -> int:
        raw = Env.get_str(name, "")
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    @staticmethod
    def get_float(name: str, default: float) -> float:
        raw = Env.get_str(name, "")
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    @staticmethod
    def get_bool(name: str, default: bool) -> bool:
        raw = Env.get_str(name, "")
        if not raw:
            return default
        return raw.lower() in Env._TRUE_VALUES

