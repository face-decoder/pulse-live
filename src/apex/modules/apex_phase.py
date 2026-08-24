import numpy as np
from scipy.signal import find_peaks


class ApexPhase:
    DISTANCE_THRESHOLD = 5

    MERGE_DISTANCE_THRESHOLD = 10

    PROMINENCE_THRESHOLD = 0.1

    PEAK_CUTOFF_THRESHOLD = 0.10

    VALLEY_UPTICK_THRESHOLD = 0.75

    MAX_SEARCH_RADIUS = 100

    def __init__(
        self,
        distance_threshold: int = DISTANCE_THRESHOLD,
        merge_distance: int = MERGE_DISTANCE_THRESHOLD,
        prominence_threshold: float = PROMINENCE_THRESHOLD,
        cutoff_ratio: float = PEAK_CUTOFF_THRESHOLD,
        valley_uptick_threshold: float = VALLEY_UPTICK_THRESHOLD,
    ) -> None:

        self.distance = distance_threshold
        self.merge_distance = merge_distance
        self.prominence = prominence_threshold
        self.cutoff_ratio = cutoff_ratio
        self.valley_uptick_threshold = valley_uptick_threshold

    def find_apex(self, signal: list, height: float = None) -> list:
        kwargs = dict(distance=self.distance, prominence=self.prominence)
        if height is not None:
            kwargs["height"] = height

        peaks, _ = find_peaks(signal, **kwargs)
        return peaks.tolist()

    def find_top_k_apex(self, signal: list, k: int = 0, height: float = None) -> list:
        kwargs = dict(distance=self.distance, prominence=self.prominence)
        if height is not None:
            kwargs["height"] = height

        peaks, _ = find_peaks(signal, **kwargs)
        peaks = peaks.tolist()

        peaks = self.merge_nearby_peaks(
            signal, peaks, merge_distance=self.merge_distance
        )

        return peaks

    def merge_nearby_peaks(
        self, signal: list, peaks: list, merge_distance: int = None
    ) -> list:
        if len(peaks) <= 1:
            return peaks

        min_dist = merge_distance if merge_distance is not None else self.distance
        signal = np.array(signal)
        merged = list(peaks)

        changed = True
        while changed:
            changed = False
            result = []
            skip = set()
            for i in range(len(merged)):
                if i in skip:
                    continue
                if i + 1 < len(merged) and (merged[i + 1] - merged[i]) < min_dist:
                    if signal[merged[i]] >= signal[merged[i + 1]]:
                        result.append(merged[i])
                    else:
                        result.append(merged[i + 1])
                    skip.add(i + 1)
                    changed = True
                else:
                    result.append(merged[i])
            merged = result

        return merged

    def find_phase(
        self,
        signal: list,
        apex_indices: list,
        cutoff_ratio: float = None,
        phase_mode: str = "onset_to_apex",
    ) -> dict:
        if phase_mode == "full":
            phase_mode = "onset_apex_offset"
        if phase_mode not in ("onset_to_apex", "onset_apex_offset"):
            raise ValueError(f"Unknown phase_mode: {phase_mode}")

        cutoff = cutoff_ratio if cutoff_ratio is not None else self.cutoff_ratio
        phases = dict()

        for idx, apex_index in enumerate(apex_indices):
            left_bound = 0 if idx == 0 else (apex_indices[idx - 1] + apex_index) // 2
            right_bound = (
                len(signal) - 1
                if idx == len(apex_indices) - 1
                else (apex_index + apex_indices[idx + 1]) // 2
            )

            start_index, end_index = self.__find_phase_boundaries(
                signal=signal,
                apex_index=apex_index,
                cutoff_ratio=cutoff,
                left_bound=left_bound,
                right_bound=right_bound,
            )

            start_index = max(start_index, left_bound)
            end_index = min(end_index, right_bound)

            if phase_mode == "onset_to_apex":
                end_index = int(apex_index)

            phases[apex_index] = dict(start=start_index, end=end_index)

        return phases

    def __find_phase_boundaries(
        self,
        signal: list,
        apex_index: int,
        cutoff_ratio: float,
        left_bound: int = 0,
        right_bound: int = None,
    ) -> tuple:
        if right_bound is None:
            right_bound = len(signal) - 1

        effective_left = max(left_bound, apex_index - self.MAX_SEARCH_RADIUS)
        effective_right = min(right_bound, apex_index + self.MAX_SEARCH_RADIUS)

        signal_arr = np.array(signal)
        apex_value = float(signal_arr[apex_index])

        run_min_val_l = apex_value
        run_min_idx_l = apex_index
        for i in range(apex_index - 1, effective_left - 1, -1):
            val = float(signal_arr[i])
            if val < run_min_val_l:
                run_min_val_l = val
                run_min_idx_l = i
            else:
                amp_range = apex_value - run_min_val_l
                if (
                    amp_range > 0
                    and (val - run_min_val_l) / amp_range > self.valley_uptick_threshold
                ):
                    break
        valley_left = run_min_idx_l

        run_min_val_r = apex_value
        run_min_idx_r = apex_index
        for i in range(apex_index + 1, effective_right + 1):
            val = float(signal_arr[i])
            if val < run_min_val_r:
                run_min_val_r = val
                run_min_idx_r = i
            else:
                amp_range = apex_value - run_min_val_r
                if (
                    amp_range > 0
                    and (val - run_min_val_r) / amp_range > self.valley_uptick_threshold
                ):
                    break
        valley_right = run_min_idx_r

        apex_value = signal_arr[apex_index]

        local_min_left = float(signal_arr[valley_left : apex_index + 1].min())
        local_min_right = float(signal_arr[apex_index : valley_right + 1].min())
        local_min = min(local_min_left, local_min_right)

        threshold = local_min + (apex_value - local_min) * cutoff_ratio

        onset_index = valley_left
        for i in range(apex_index, valley_left - 1, -1):
            if signal[i] <= threshold:
                onset_index = i
                break

        offset_index = valley_right
        for i in range(apex_index, valley_right + 1):
            if signal[i] <= threshold:
                offset_index = i
                break

        return onset_index, offset_index
