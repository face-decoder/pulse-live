from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..constants.index import ROI_ORDER_DEFAULT
from .anxiety_dataset_base import AnxietyDatasetBase
from .compose import Compose


class FlowROIDataset(AnxietyDatasetBase):
    """
    Dataset untuk flow berbasis ROI.

    Mengharapkan file .npz dengan key:
      - "flow"      : (T, N_roi, 2, H, W)  wajib
      - "roi_order" : list nama ROI         opsional
      - "magnitudes": (T,)                  opsional (untuk window detection fallback)

    Args:
        metadata_df   : DataFrame dengan kolom wajib + "clip", "npy_path"
        transform     : Compose pipeline (wajib diisi untuk mode apapun)
        detector      : ApexWindowDetector instance
        roi_order     : urutan ROI; default ROI_ORDER_DEFAULT
        cache_dir     : direktori cache tensor
        force_rebuild : paksa rebuild cache
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        transform: Optional[Compose] = None,
        detector=None,
        roi_order: List[str] = None,
        phase_mode: str = "onset_to_apex",
        cache_dir: Optional[Union[str, Path]] = None,
        force_rebuild: bool = False,
    ):
        self.roi_order = roi_order or ROI_ORDER_DEFAULT
        self.detector = detector
        self.phase_mode = phase_mode
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
                if "flow" not in npz:
                    raise KeyError(f"'flow' key tidak ada di {npy_path}")
                flow = npz["flow"].astype(np.float32)
        elif isinstance(data, np.ndarray) and data.dtype == object:
            obj = data.item()
            flow = obj["flow"].astype(np.float32)
        else:
            raise ValueError(f"Format tidak dikenali: {npy_path}")

        if flow.ndim != 5 or flow.shape[2] != 2:
            raise ValueError(f"Flow ROI harus (T, N_roi, 2, H, W), got {flow.shape}")
        return flow

    def _detect_windows(self, flow: np.ndarray) -> List[Tuple[int, int, int]]:
        if self.detector is None:
            # Fallback: seluruh clip sebagai satu window
            T = flow.shape[0]
            apex = T // 2
            return [(0, apex, T)]

        windows, meta = self.detector.detect_windows(flow, phase_mode=self.phase_mode)
        if not meta.get("valid", False):
            T = flow.shape[0]
            apex = T // 2
            return [(0, apex, T)]
        return windows
