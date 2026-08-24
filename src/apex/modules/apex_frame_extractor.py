from typing import List

import numpy as np


class ApexFrameExtractor:
    @staticmethod
    def extract_window(
        data: List[np.ndarray], apex_idx: int, k: int = 5
    ) -> List[np.ndarray]:
        n = len(data)
        if n == 0:
            return []

        indices = []
        for i in range(apex_idx - k, apex_idx + k + 1):
            clamped = max(0, min(i, n - 1))
            indices.append(clamped)

        return [data[i] for i in indices]

    @staticmethod
    def extract_full_phase(
        data: List[np.ndarray], onset: int, offset: int, target_length: int = 32
    ) -> List[np.ndarray]:
        n = len(data)
        if n == 0:
            return []

        onset = max(0, min(onset, n - 1))
        offset = max(0, min(offset, n - 1))

        if onset >= offset:
            onset, offset = max(0, offset), min(n - 1, onset)
            if onset >= offset:
                return [data[onset]] * target_length

        phase_length = offset - onset + 1

        if phase_length >= target_length:
            indices = np.round(np.linspace(0, phase_length - 1, target_length)).astype(
                int
            )
        else:
            indices = np.round(np.linspace(0, phase_length - 1, target_length)).astype(
                int
            )

        return [data[onset + i] for i in indices]

    @staticmethod
    def extract_hybrid(
        data: List[np.ndarray],
        onset: int,
        apex_idx: int,
        offset: int,
        target_length: int = 32,
    ) -> List[np.ndarray]:
        half = target_length // 2
        remainder = target_length - half

        seg_onset = ApexFrameExtractor.extract_full_phase(
            data, onset, apex_idx, target_length=half
        )

        seg_offset = ApexFrameExtractor.extract_full_phase(
            data, apex_idx, offset, target_length=remainder
        )

        return seg_onset + seg_offset
