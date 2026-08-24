from typing import Sequence

import numpy as np
from scipy.signal import savgol_filter


class ApexSmoother:
    WINDOW_LENGTH_PERCENTAGE = 0.1

    @staticmethod
    def calculate_window_length(length: int) -> int:

        window_length = int(length * ApexSmoother.WINDOW_LENGTH_PERCENTAGE)

        if window_length % 2 == 0:
            window_length += 1

        window_length = max(5, min(window_length, 51))

        if window_length > length:
            window_length = length if length % 2 != 0 else length - 1
            if window_length < 3:
                window_length = 3

        return window_length

    @staticmethod
    def calculate_polyorder(window_length: int) -> int:

        match window_length:
            case wl if wl <= 7:
                return 2
            case wl if wl <= 15:
                return 3
            case _:
                return 4

    @staticmethod
    def smooth(signal: Sequence[float]) -> np.ndarray:
        window_length = ApexSmoother.calculate_window_length(len(signal))
        polyorder = ApexSmoother.calculate_polyorder(window_length)
        return np.asarray(
            savgol_filter(signal, window_length, polyorder), dtype=np.float32
        )
