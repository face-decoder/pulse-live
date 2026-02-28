import os
import torch
import numpy as np
from typing import Tuple

from src.apex.modules import ApexFrameExtractor
from src.datasource.base_datasource import BaseDataSource


class ApexWindowDataSource(BaseDataSource):
    """Apex-centered window extraction strategy."""

    def __init__(self, k: int = 5, **kwargs):
        """Inisialisasi window datasource.

        Args:
            k: setengah lebar window di sekitar apex
        """
        super().__init__(strategy_name=f"window_k{k}", **kwargs)
        self.k = k

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Mengambil sample berdasarkan apex-centered window.

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

            # T1-A: lewati index yang sudah di-blacklist tanpa memangil spotter
            if video_path in self._blacklist:
                index = (index + 1) % len(self.datasource)
                continue

            spotter_result = self._BaseDataSource__run_spotter(index)
            if spotter_result is not None:
                break
            # __run_spotter gagal → video sudah di-blacklist; lanjut ke berikutnya
            index = (index + 1) % len(self.datasource)
        else:
            raise RuntimeError(f"Tidak ada video valid setelah {max_retries} percobaan")

        apex_indices, phases, label = spotter_result

        if not apex_indices:
            midpoint = len(self.spotter.flows) // 2
            flow_segments = ApexFrameExtractor.extract_window(self.spotter.flows, midpoint, self.k)
            frame_segments = ApexFrameExtractor.extract_window(self.spotter.frames, midpoint, self.k)
        else:
            apex_idx = apex_indices[0]
            flow_segments = ApexFrameExtractor.extract_window(self.spotter.flows, apex_idx, self.k)
            frame_segments = ApexFrameExtractor.extract_window(self.spotter.frames, apex_idx, self.k)

        flows_tensor = torch.from_numpy(np.stack(flow_segments)).float()
        frames_tensor = torch.from_numpy(np.stack(frame_segments)).float()

        torch.save((flows_tensor, frames_tensor, label), cache_path)

        return flows_tensor, frames_tensor, label
