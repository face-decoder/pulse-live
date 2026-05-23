from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class SubjectSample:
    """
    Immutable container for a subject.

    Attributes:
        subject_id  : subject identity
        label       : integer label (0 = rendah, 1 = tinggi)
        flow        : raw optical flow, shape tergantung sumber:
                        ROI      → (T, N_roi, 2, H, W)
                        FullFace → (T, 2, H, W)
        windows     : list of (left, apex, right) from ApexWindowDetector
        meta        : additional metadata (score, clip info, etc.)
    """

    subject_id: str
    label: int
    flow: np.ndarray
    windows: List[Tuple[int, int, int]]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformOutput:
    """
    Standard output from each transform, passed between compose steps.

    Attributes:
        x        : feature tensor, shape (C, T) or (T, C) depending on stage
        y        : label tensor scalar
        mask     : boolean mask (T,) — True = padding, only present if PadAndMask
        meta     : metadata (passed through as-is)
    """

    x: torch.Tensor
    y: torch.Tensor
    mask: Optional[torch.Tensor] = None
    meta: Dict[str, Any] = field(default_factory=dict)
