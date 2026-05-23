from __future__ import annotations

from typing import Dict, List, Tuple

ROI_ORDER_DEFAULT: List[str] = [
    "left_eye",
    "right_eye",
    "lips",
    "left_eyebrow",
    "right_eyebrow",
]

SYMMETRY_PAIRS_DEFAULT: List[Tuple[int, int]] = [
    (0, 1),  # left_eye ~ right_eye
    (3, 4),  # left_eyebrow ~ right_eyebrow
]

LABEL_MAP: Dict[str, int] = {
    "anxiety_rendah": 0,
    "anxiety_tinggi": 1,
}

PhaseMode = str  # "onset_to_apex" | "full"
