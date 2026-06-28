"""Public API for all anxiety inference pipelines.

Usage::

    from src.models.inferencer import get_inferencer, load_inferencer_from_env

    # Load via factory (requires checkpoint path)
    inf = get_inferencer("0407", "checkpoints_0407.../best_model.pt", device="cuda")

    # Load from .env (MODEL_COMBINATION_ID + MODEL_CHECKPOINT_PATH)
    inf = load_inferencer_from_env()

    result = inf.predict_npz("clip.npz")  # or predict_flow(flow_array)
    print(result.label, result.prob_high)

Combination IDs
---------------
    0[1-4][01-07]

    01 / spatio_temporal     → SpatioTemporalInferencer   (raw-flow)
    02 / cnn_lstm_mlp        → CnnLstmMlpInferencer
    03 / cnn_bi_lstm         → CnnBiLstmInferencer
    04 / cnn_bi_lstm_attn    → CnnBiLstmAttentionInferencer
    05 / cnn_bi_lstm_mha     → CnnBiLstmMhaInferencer
    06 / tcn                 → TcnInferencer
    07 / cnn_transformer     → CnnTransformerInferencer
"""

from __future__ import annotations

# ── Core ──────────────────────────────────────────────────────────────────────
from .base import BaseAnxietyInferencer
from .result import InferenceResult

# ── Concrete classes ──────────────────────────────────────────────────────────
from .spatio_temporal import SpatioTemporalInferencer
from .cnn_lstm_mlp import CnnLstmMlpInferencer
from .cnn_bi_lstm import CnnBiLstmInferencer
from .cnn_bi_lstm_attention import CnnBiLstmAttentionInferencer
from .cnn_bi_lstm_mha import CnnBiLstmMhaInferencer
from .tcn import TcnInferencer
from .cnn_transformer import CnnTransformerInferencer

# ── Factory (no circular dependency — lives in _factory.py) ──────────────────
from ._factory import _ARCH_MAP, get_inferencer

# ── Registry (env-driven singleton) ───────────────────────────────────────────
from .registry import get_loaded_inferencer, load_inferencer_from_env, reset_inferencer

__all__ = [
    # Factory
    "get_inferencer",
    "_ARCH_MAP",
    # Registry
    "load_inferencer_from_env",
    "get_loaded_inferencer",
    "reset_inferencer",
    # Base & result
    "BaseAnxietyInferencer",
    "InferenceResult",
    # Concrete classes
    "SpatioTemporalInferencer",
    "CnnLstmMlpInferencer",
    "CnnBiLstmInferencer",
    "CnnBiLstmAttentionInferencer",
    "CnnBiLstmMhaInferencer",
    "TcnInferencer",
    "CnnTransformerInferencer",
]
