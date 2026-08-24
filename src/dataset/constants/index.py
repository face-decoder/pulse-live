from __future__ import annotations

ROI_ORDER_DEFAULT: list[str] = [
    "left_eye",
    "right_eye",
    "lips",
    "left_eyebrow",
    "right_eyebrow",
]

SYMMETRY_PAIRS_DEFAULT: list[tuple[int, int]] = [
    (0, 1),
    (3, 4),
]

LABEL_MAP: dict[str, int] = {
    "anxiety_rendah": 0,
    "anxiety_tinggi": 1,
}

PhaseMode = str
