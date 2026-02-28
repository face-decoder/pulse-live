import os
import torch
import numpy as np
from typing import Tuple

from src.apex.modules import ApexFrameExtractor
from src.datasource.base_datasource import BaseDataSource


class ApexHybridDataSource(BaseDataSource):
    """Hybrid extraction onset-apex-offset dengan normalisasi per segment."""

    def __init__(self, target_length: int = 32, **kwargs):
        """Inisialisasi hybrid datasource.

        Args:
            target_length: jumlah frame target setelah resampling
        """
        super().__init__(strategy_name=f"hybrid_len{target_length}", **kwargs)
        self.target_length = target_length

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Mengambil sample berdasarkan hybrid onset-apex dan apex-offset.

        Returns:
            flows_tensor, frames_tensor, label
        """
        max_retries = len(self.datasource)

        for _ in range(max_retries):
            video_path, _ = self.datasource[index]
            cache_path = self._BaseDataSource__get_cache(video_path)

            if os.path.exists(cache_path):
                try:
                    return torch.load(cache_path, weights_only=True)
                except Exception as exc:
                    print(f"[WARN] Corrupt cache: {cache_path} - {exc}")
                    os.remove(cache_path)

            if video_path in self._blacklist:
                index = (index + 1) % len(self.datasource)
                continue

            spotter_result = self._BaseDataSource__run_spotter(index)
            if spotter_result is not None:
                break
            index = (index + 1) % len(self.datasource)
        else:
            raise RuntimeError(f"Tidak ada video valid setelah {max_retries} percobaan")

        apex_indices, phases, label = spotter_result

        if not apex_indices or not phases:
            total_frames = len(self.spotter.flows)
            midpoint = total_frames // 2
            flow_segments = ApexFrameExtractor.extract_hybrid(
                self.spotter.flows, 0, midpoint, total_frames - 1, self.target_length
            )
            frame_segments = ApexFrameExtractor.extract_hybrid(
                self.spotter.frames, 0, midpoint, total_frames - 1, self.target_length
            )
        else:
            apex_idx = apex_indices[0]
            phase = phases[apex_idx]
            onset, offset = phase["start"], phase["end"]

            flow_segments = ApexFrameExtractor.extract_hybrid(
                self.spotter.flows, onset, apex_idx, offset, self.target_length
            )
            frame_segments = ApexFrameExtractor.extract_hybrid(
                self.spotter.frames, onset, apex_idx, offset, self.target_length
            )

        flows_tensor = torch.from_numpy(np.stack(flow_segments)).float()
        frames_tensor = torch.from_numpy(np.stack(frame_segments)).float()

        torch.save((flows_tensor, frames_tensor, label), cache_path)

        return flows_tensor, frames_tensor, label
