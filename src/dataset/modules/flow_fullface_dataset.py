from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .anxiety_dataset_base import AnxietyDatasetBase
from .compose import Compose


class FlowFullFaceDataset(AnxietyDatasetBase):
    """
    Dataset untuk flow full face.

    Mengharapkan file .npz dengan key:
      - "flow"      : (T, 2, H, W)   wajib
      - "magnitudes": (T,)            opsional

    Args:
        metadata_df   : DataFrame dengan kolom wajib
        transform     : Compose pipeline
        detector      : ApexWindowDetector instance (opsional)
        cache_dir     : direktori cache
        force_rebuild : paksa rebuild
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        transform: Optional[Compose] = None,
        detector=None,
        cache_dir: Optional[Union[str, Path]] = None,
        force_rebuild: bool = False,
    ):
        self.detector = detector
        super().__init__(
            metadata_df=metadata_df,
            transform=transform,
            cache_dir=cache_dir,
            force_rebuild=force_rebuild,
        )

    def _load_flow(self, npy_path: str) -> np.ndarray:
        data = np.load(npy_path, allow_pickle=True)

        if isinstance(data, np.lib.npyio.NpzFile):
            with data as npz:
                # Coba "flow" dulu; fallback ke "horizontal"+"vertical" stacked
                if "flow" in npz:
                    flow = npz["flow"].astype(np.float32)
                elif "horizontal_magnitudes" in npz and "vertical_magnitudes" in npz:
                    dx = npz["horizontal_magnitudes"].astype(np.float32)  # (T, H, W)
                    dy = npz["vertical_magnitudes"].astype(np.float32)
                    flow = np.stack([dx, dy], axis=1)  # (T, 2, H, W)
                else:
                    raise KeyError(f"Tidak ada key 'flow' di {npy_path}")
        elif isinstance(data, np.ndarray) and data.dtype == object:
            obj = data.item()
            flow = obj["flow"].astype(np.float32)
        else:
            raise ValueError(f"Format tidak dikenali: {npy_path}")

        # Normalisasi dimensi: (T, H, W, 2) → (T, 2, H, W)
        if flow.ndim == 4 and flow.shape[-1] == 2:
            flow = flow.transpose(0, 3, 1, 2)

        if flow.ndim != 4 or flow.shape[1] != 2:
            raise ValueError(f"Flow FullFace harus (T, 2, H, W), got {flow.shape}")
        return flow

    def _detect_windows(self, flow: np.ndarray) -> List[Tuple[int, int, int]]:
        if self.detector is None:
            T = flow.shape[0]
            apex = T // 2
            return [(0, apex, T)]

        # Buat flow dummy 5D agar kompatibel dengan ApexWindowDetector
        flow_5d = flow[:, np.newaxis, :, :, :]  # (T, 1, 2, H, W)
        windows, meta = self.detector.detect_windows(flow_5d, phase_mode="full")
        if not meta.get("valid", False):
            T = flow.shape[0]
            apex = T // 2
            return [(0, apex, T)]
        return windows
