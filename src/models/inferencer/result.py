"""Shared result dataclass for all anxiety inference models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InferenceResult:
    """Structured output of a single inference run.

    Attributes:
        label: Human-readable prediction — ``"anxiety_rendah"`` or
            ``"anxiety_tinggi"``.
        prob_high: Probability assigned to the *high-anxiety* class.
        prob_low: Probability assigned to the *low-anxiety* class.
        confidence: ``max(prob_high, prob_low)``.
        threshold: Decision boundary used to derive *label*.
        n_windows: Number of apex micro-expression windows detected.
        warning: Optional diagnostic message (e.g. *"no apex windows"*).
    """

    label: str
    prob_high: float
    prob_low: float
    confidence: float
    threshold: float
    n_windows: int
    warning: str | None = field(default=None)
    spotting_latency_ms: float | None = field(default=None)
    model_inference_latency_ms: float | None = field(default=None)

    # ── Convenience ────────────────────────────────────────────────────

    def as_dict(self) -> dict:
        """Return a JSON-serialisable dict suitable for API responses."""
        res = {
            "label": self.label,
            "prob_high": round(self.prob_high, 4),
            "prob_low": round(self.prob_low, 4),
            "confidence": round(self.confidence, 4),
            "threshold": round(self.threshold, 4),
            "n_windows": self.n_windows,
            "warning": self.warning,
        }
        if self.spotting_latency_ms is not None:
            res["spotting_latency_ms"] = round(self.spotting_latency_ms, 2)
        if self.model_inference_latency_ms is not None:
            res["model_inference_latency_ms"] = round(self.model_inference_latency_ms, 2)
        return res

    def __repr__(self) -> str:
        warn = f" | ⚠ {self.warning}" if self.warning else ""
        return (
            f"InferenceResult(label={self.label!r}, "
            f"prob_high={self.prob_high:.4f}, "
            f"threshold={self.threshold:.4f}, "
            f"n_windows={self.n_windows}{warn})"
        )
