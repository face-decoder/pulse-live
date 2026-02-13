import numpy as np
from scipy.signal import find_peaks


class ApexPhase:

    # Ambang batas yang digunakan untuk menentukan jarak antar titik puncak
    DISTANCE_THRESHOLD = 5

    # Ambang batas yang digunakan untuk menentukan keunggulan puncak
    PROMINENCE_THRESHOLD = 0.01

    # Ambang batas yang digunakan untuk menentukan lebar fase puncak
    PEAK_CUTOFF_THRESHOLD = 0.35

    def __init__(self,
                 distance_threshold: int = DISTANCE_THRESHOLD,
                 prominence_threshold: float = PROMINENCE_THRESHOLD) -> None:

        self.distance = distance_threshold
        self.prominence = prominence_threshold


    def find_apex(self, signal: list) -> list:
        """
        Mendeteksi titik puncak (apex) dalam sinyal menggunakan metode find_peaks dari scipy.

        Args:
            signal (list): Sinyal input yang akan dianalisis.

        Returns:
            list: Indeks titik puncak yang terdeteksi dalam sinyal.
        """
        peaks, _ = find_peaks(signal,
                              distance=self.distance,
                              prominence=self.prominence)
        return peaks.tolist()
    

    def find_phase(self, signal: list, apex_indices: list) -> dict:
        """
        Mendeteksi fase apex berdasarkan sinyal dan indeks apex yang sudah ditemukan
        
        Args:
            signal (list): Sinyal input yang akan diproses.
            apex_indices (list): Daftar indeks apex yang sudah ditemukan.

        Returns:
            dict: Kamus yang berisi informasi fase apex.
        """
        phases = dict()

        for idx, apex_index in enumerate(apex_indices):

            left_bound = 0 if idx == 0 else (apex_indices[idx - 1] + apex_index) // 2
            right_bound = len(signal) - 1 if idx == len(apex_indices) - 1 else (apex_index + apex_indices[idx + 1]) // 2

            start_index, end_index = self.__find_phase_boundaries(signal=signal,
                                                                  apex_index=apex_index,
                                                                  cutoff_ratio=self.PEAK_CUTOFF_THRESHOLD,
                                                                  left_bound=left_bound,
                                                                  right_bound=right_bound)

            phases[apex_index] = dict(start=start_index, end=end_index)

        return phases


    def __find_phase_boundaries(self,
                                signal: list,
                                apex_index: int,
                                cutoff_ratio: float,
                                left_bound: int = 0,
                                right_bound: int = None) -> tuple:
        """
        Mendeteksi batas fase apex berdasarkan sinyal, indeks apex, dan rasio cutoff.

        Args:
            signal (list): Sinyal input yang akan dianalisis.
            apex_index (int): Indeks titik apex dalam sinyal.
            cutoff_ratio (float): Rasio cutoff untuk menentukan batas fase.
            left_bound (int): Batas kiri pencarian (mencegah tumpang tindih).
            right_bound (int): Batas kanan pencarian (mencegah tumpang tindih).

        Returns:
            tuple: Indeks batas awal dan akhir fase apex.
        """
        # Memastikan batas kanan tidak None
        # Hal ini menunjukkan bahwa pencarian dilakukan hingga akhir sinyal
        # Jika sinyal merupakan fase akhir, maka maksimal index dibuat diakhir sinyal
        if right_bound is None:
            right_bound = len(signal) - 1

        apex_value = signal[apex_index]

        # Mengambil minimal value dari keseluruhan sinyal untuk menentukan treshold
        local_min = min(np.min(signal[left_bound:apex_index + 1]),
                        np.min(signal[apex_index:right_bound + 1]))

        # Menentukan threshold berdasarkan cutoff ratio
        threshold = local_min + (apex_value - local_min) * cutoff_ratio

        # Mencari batas onset (awal) fase apex
        onset_index = left_bound
        for i in range(apex_index, left_bound, -1):
            if signal[i] <= threshold:
                onset_index = i
                break

        # Mencari batas offset (akhir) fase apex
        offset_index = right_bound
        for i in range(apex_index, right_bound + 1):
            if signal[i] <= threshold:
                offset_index = i
                break

        return onset_index, offset_index
