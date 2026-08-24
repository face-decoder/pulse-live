from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class SubjectSample:
    subject_id: str
    label: int
    flow: np.ndarray
    windows: list[tuple[int, int, int]]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformOutput:
    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor | None = None
    meta: dict[str, Any] = field(default_factory=dict)
