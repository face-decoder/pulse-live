from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.utils.env import Env

from ._factory import get_inferencer
from .base import BaseAnxietyInferencer

logger = logging.getLogger(__name__)


_inferencer: BaseAnxietyInferencer | None = None


def load_inferencer_from_env() -> BaseAnxietyInferencer:
    global _inferencer  # noqa: PLW0603
    if _inferencer is not None:
        return _inferencer

    combination_id = Env.get_str("MODEL_COMBINATION_ID", "")
    checkpoint_path = Env.get_str("MODEL_CHECKPOINT_PATH", "")

    if not combination_id:
        raise ValueError(
            "Missing required env var MODEL_COMBINATION_ID. "
            "Set it to a 4-digit combination code (e.g. '0407') or a "
            "friendly architecture name (e.g. 'cnn_transformer')."
        )
    if not checkpoint_path:
        raise ValueError(
            "Missing required env var MODEL_CHECKPOINT_PATH. "
            "Set it to the path of best_model.pt for the selected combination."
        )

    ckpt = Path(checkpoint_path)
    if not ckpt.is_absolute():
        project_root = Path(__file__).resolve().parents[3]
        ckpt = project_root / ckpt

    device_str = Env.get_str("MODEL_DEVICE", "")
    if not device_str:
        try:
            import torch

            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device_str = "cpu"

    strict_parity = Env.get_bool("MODEL_STRICT_NOTEBOOK_PARITY", True)

    n_tta = Env.get_int("MODEL_N_TTA", 8)

    factory_kwargs: dict[str, Any] = {
        "device": device_str,
        "n_tta": n_tta,
        "prefer_checkpoint_tta": strict_parity,
    }

    pct_env = Env.get_str("MODEL_DETECTOR_PERCENTILE", "")
    prom_env = Env.get_str("MODEL_DETECTOR_PROMINENCE", "")
    if strict_parity and (pct_env or prom_env):
        logger.warning(
            "MODEL_STRICT_NOTEBOOK_PARITY is enabled; detector env overrides are ignored "
            "(MODEL_DETECTOR_PERCENTILE=%r, MODEL_DETECTOR_PROMINENCE=%r).",
            pct_env or None,
            prom_env or None,
        )
    else:
        if pct_env:
            factory_kwargs["detector_percentile"] = Env.get_float(
                "MODEL_DETECTOR_PERCENTILE",
                BaseAnxietyInferencer.DEFAULT_DETECTOR_PERCENTILE,
            )
        if prom_env:
            factory_kwargs["detector_prominence"] = Env.get_float(
                "MODEL_DETECTOR_PROMINENCE",
                BaseAnxietyInferencer.DEFAULT_DETECTOR_PROMINENCE,
            )

    logger.info(
        "Loading inferencer | combination=%s | ckpt=%s | device=%s | n_tta=%d | strict_parity=%s | extra=%r",
        combination_id,
        ckpt,
        device_str,
        n_tta,
        strict_parity,
        factory_kwargs,
    )

    _inferencer = get_inferencer(
        combination_id=combination_id,
        checkpoint_path=ckpt,
        **factory_kwargs,
    )

    logger.info(
        "Inferencer ready: %r",
        _inferencer,
    )
    return _inferencer


def get_loaded_inferencer() -> BaseAnxietyInferencer | None:
    return _inferencer


def reset_inferencer() -> None:
    global _inferencer  # noqa: PLW0603
    _inferencer = None
