import numpy as np
from typing import List, Tuple


class ApexFrameExtractor:
    """
    Utilitas untuk mengekstraksi segmen frame/optical flow dari fase apex.
    Menyediakan 3 strategi ekstraksi untuk dataset PyTorch training.
    """

    @staticmethod
    def extract_window(data: List[np.ndarray],
                       apex_idx: int,
                       k: int = 5) -> List[np.ndarray]:
        """
        Strategy 1: Apex-centered window.
        Mengambil frame di sekitar apex: [a-k, ..., a, ..., a+k] → 2k+1 frame.
        Jika dekat boundary, padding dengan edge frame terdekat.

        Args:
            data: List of arrays (flows atau frames) dari spotter.
            apex_idx: Indeks apex dalam data.
            k: Jumlah frame sebelum dan sesudah apex.

        Returns:
            List of arrays dengan panjang 2k+1.
        """
        n = len(data)
        if n == 0:
            return []

        indices = []
        for i in range(apex_idx - k, apex_idx + k + 1):
            # Clamp ke boundary jika di luar range
            clamped = max(0, min(i, n - 1))
            indices.append(clamped)

        return [data[i] for i in indices]


    @staticmethod
    def extract_full_phase(data: List[np.ndarray],
                           onset: int,
                           offset: int,
                           target_length: int = 32) -> List[np.ndarray]:
        """
        Strategy 2: Full phase with temporal normalization.
        Mengambil semua frame dari onset hingga offset, kemudian
        resample ke target_length menggunakan interpolasi indeks.

        Args:
            data: List of arrays (flows atau frames) dari spotter.
            onset: Indeks onset (awal fase).
            offset: Indeks offset (akhir fase).
            target_length: Jumlah frame output setelah normalisasi.

        Returns:
            List of arrays dengan panjang target_length.
        """
        n = len(data)
        if n == 0:
            return []

        # Clamp boundaries
        onset = max(0, min(onset, n - 1))
        offset = max(0, min(offset, n - 1))

        # Pastikan onset < offset
        if onset >= offset:
            onset, offset = max(0, offset), min(n - 1, onset)
            if onset >= offset:
                # Fase terlalu pendek, duplikasi frame
                return [data[onset]] * target_length

        phase_length = offset - onset + 1

        if phase_length >= target_length:
            # Downsample: pilih indeks yang tersebar merata
            indices = np.round(np.linspace(0, phase_length - 1, target_length)).astype(int)
        else:
            # Upsample: duplikasi frame terdekat untuk mencapai target_length
            indices = np.round(np.linspace(0, phase_length - 1, target_length)).astype(int)

        return [data[onset + i] for i in indices]


    @staticmethod
    def extract_hybrid(data: List[np.ndarray],
                       onset: int,
                       apex_idx: int,
                       offset: int,
                       target_length: int = 32) -> List[np.ndarray]:
        """
        Strategy 3: Hybrid — onset→apex + apex→offset.
        Setiap sub-segment dinormalisasi ke target_length // 2,
        lalu digabung. Mempertahankan asimetri temporal fase.

        Args:
            data: List of arrays (flows atau frames) dari spotter.
            onset: Indeks onset (awal fase).
            apex_idx: Indeks apex.
            offset: Indeks offset (akhir fase).
            target_length: Total jumlah frame output (dibagi 2 untuk tiap sub-segment).

        Returns:
            List of arrays dengan panjang target_length.
        """
        half = target_length // 2
        remainder = target_length - half  # handle odd target_length

        # Sub-segment 1: onset → apex
        seg_onset = ApexFrameExtractor.extract_full_phase(
            data, onset, apex_idx, target_length=half
        )

        # Sub-segment 2: apex → offset
        seg_offset = ApexFrameExtractor.extract_full_phase(
            data, apex_idx, offset, target_length=remainder
        )

        return seg_onset + seg_offset
