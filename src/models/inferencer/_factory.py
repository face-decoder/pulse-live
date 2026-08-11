"""Internal factory: maps combination IDs to concrete inferencer classes.

Kept in a separate module from ``__init__`` so that ``registry.py`` can
import ``get_inferencer`` without triggering a circular import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseAnxietyInferencer
from .cnn_bi_lstm import CnnBiLstmInferencer
from .cnn_bi_lstm_attention import CnnBiLstmAttentionInferencer
from .cnn_bi_lstm_mha import CnnBiLstmMhaInferencer
from .cnn_lstm_mlp import CnnLstmMlpInferencer
from .cnn_transformer import CnnTransformerInferencer
from .spatio_temporal import SpatioTemporalInferencer
from .tcn import TcnInferencer

# ── Architecture suffix → concrete class ────────────────────────────────────

_ARCH_MAP: dict[str, type[BaseAnxietyInferencer]] = {
    # By last-two-digit suffix from combination ID
    "01": SpatioTemporalInferencer,
    "02": CnnLstmMlpInferencer,
    "03": CnnBiLstmInferencer,
    "04": CnnBiLstmAttentionInferencer,
    "05": CnnBiLstmMhaInferencer,
    "06": TcnInferencer,
    "07": CnnTransformerInferencer,
    # Also accept friendly names
    "spatio_temporal": SpatioTemporalInferencer,
    "cnn_lstm_mlp": CnnLstmMlpInferencer,
    "cnn_bi_lstm": CnnBiLstmInferencer,
    "cnn_bi_lstm_attention": CnnBiLstmAttentionInferencer,
    "cnn_bi_lstm_attn": CnnBiLstmAttentionInferencer,
    "cnn_bi_lstm_mha": CnnBiLstmMhaInferencer,
    "tcn": TcnInferencer,
    "cnn_transformer": CnnTransformerInferencer,
}


def get_inferencer(
    combination_id: str,
    checkpoint_path: str | Path,
    **kwargs: Any,
) -> BaseAnxietyInferencer:
    """Instantiate the correct inferencer for a given combination.

    Args:
        combination_id: Four-digit notebook code (e.g. ``"0407"``) **or**
            a friendly architecture name (e.g. ``"cnn_transformer"``).
            The last two digits encode the architecture; the first two are
            ignored when resolving the class.
        checkpoint_path: Path to ``best_model.pt``.
        **kwargs: Forwarded to the inferencer constructor (``device``,
            ``n_tta``, ``max_seq_len``, etc.).

    Returns:
        Concrete :class:`BaseAnxietyInferencer` subclass, fully loaded.

    Raises:
        ValueError: Unknown *combination_id*.
    """
    key = combination_id[-2:] if combination_id.isdigit() else combination_id.lower()
    cls = _ARCH_MAP.get(key)
    if cls is None:
        valid = sorted({k for k in _ARCH_MAP if not k.isdigit()})
        raise ValueError(
            f"Unknown combination_id {combination_id!r}. "
            f"Valid names: {valid!r}. "
            f"Valid numeric suffixes: {sorted(k for k in _ARCH_MAP if k.isdigit())!r}."
        )
    return cls(checkpoint_path=checkpoint_path, **kwargs)
