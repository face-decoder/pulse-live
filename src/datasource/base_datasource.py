import os
import hashlib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

from src.apex.modules import ApexPhaseSpotter, ApexFrameExtractor, ExtractionMode
from src.video.modules import LazyVideo


LABEL_MAP = {"anxiety": 1, "non-anxiety": 0}

# Versi cache: naikkan string ini setiap kali parameter ekstraksi berubah
# (TVL1 params, tile_size, mode, dll) agar file .pt lama otomatis di-skip.
CACHE_VERSION = "v1"


class BaseDataSource(Dataset):
    """Base datasource untuk micro-expression recognition pipeline."""

    DATASOURCE_ROOT_PATH = os.path.join(Path.home(), "datasets", "primary-converted")
    CACHE_ROOT_PATH = os.path.join(DATASOURCE_ROOT_PATH, ".cache")

    DATASOURCE_GROUP_SUBJECT_ONE_BEFORE = os.path.join(DATASOURCE_ROOT_PATH, "BEFORE 8-12-2025")
    DATASOURCE_GROUP_SUBJECT_ONE_AFTER  = os.path.join(DATASOURCE_ROOT_PATH, "AFTER 8-12-2025")
    DATASOURCE_GROUP_SUBJECT_TWO_BEFORE = os.path.join(DATASOURCE_ROOT_PATH, "BEFORE 9-12-2025")
    DATASOURCE_GROUP_SUBJECT_TWO_AFTER  = os.path.join(DATASOURCE_ROOT_PATH, "AFTER 9-12-2025")

    def __init__(self,
                 mode: ExtractionMode = "roi",
                 annotation_path: str = None,
                 max_subjects: int = None,
                 strategy_name: str = "base"):
        """Inisialisasi datasource dan load anotasi.

        Args:
            mode: mode ekstraksi optical flow
            annotation_path: path ke file anotasi excel
            max_subjects: batas jumlah subjek untuk diproses
            strategy_name: nama strategi untuk folder cache
        """
        self.datasource: List[Tuple[str, str]] = []
        self.spotter = ApexPhaseSpotter(mode=mode)

        self.cache_dir = os.path.join(self.CACHE_ROOT_PATH, mode, strategy_name)
        os.makedirs(self.cache_dir, exist_ok=True)

        # T1-A: Blacklist persisten — video yang selalu gagal di-skip di awal
        # sehingga WeightedRandomSampler tidak memicu O(N) retry setiap epoch.
        self._blacklist_path = os.path.join(self.cache_dir, "blacklist.txt")
        self._blacklist = self.__load_blacklist()

        if annotation_path is None:
            annotation_path = os.path.join(Path.cwd(), "..", "formatted-time-series-annotations.xlsx")

        annotation_sheets = pd.read_excel(annotation_path, sheet_name=None)

        subject_entries = (
            self.__match_annotations(annotation_sheets["before-8"], self.DATASOURCE_GROUP_SUBJECT_ONE_BEFORE) +
            self.__match_annotations(annotation_sheets["after-8"],  self.DATASOURCE_GROUP_SUBJECT_ONE_AFTER) +
            self.__match_annotations(annotation_sheets["before-9"], self.DATASOURCE_GROUP_SUBJECT_TWO_BEFORE) +
            self.__match_annotations(annotation_sheets["after-9"],  self.DATASOURCE_GROUP_SUBJECT_TWO_AFTER)
        )

        if max_subjects is not None:
            subject_entries = subject_entries[:max_subjects]

        for subject_directory, label in subject_entries:
            if not os.path.exists(subject_directory):
                continue

            for entry_name in sorted(os.listdir(subject_directory)):
                entry_path = os.path.join(subject_directory, entry_name)

                if os.path.isdir(entry_path) and entry_name.startswith("q"):
                    avi_files = [f for f in os.listdir(entry_path) if f.endswith(".avi")]

                    if avi_files:
                        video_path = os.path.join(entry_path, avi_files[0])

                        try:
                            lazy_video = LazyVideo(video_path)
                            if len(lazy_video) == 0:
                                print(f"[SKIP] Empty video: {video_path}")
                                continue
                            lazy_video.close()
                        except Exception as exc:
                            print(f"[SKIP] Unreadable video: {video_path} - {exc}")
                            continue

                        # T1-A: lewati video yang sudah masuk blacklist
                        if video_path in self._blacklist:
                            print(f"[SKIP] Blacklisted: {video_path}", flush=True)
                            continue

                        self.datasource.append((video_path, label))

    # ------------------------------------------------------------------
    # Blacklist helpers
    # ------------------------------------------------------------------

    def __load_blacklist(self) -> set:
        """Baca blacklist dari disk; kembalikan set path video yang di-skip."""
        if not os.path.exists(self._blacklist_path):
            return set()
        with open(self._blacklist_path, "r") as f:
            return set(line.strip() for line in f if line.strip())

    def _add_to_blacklist(self, video_path: str) -> None:
        """Tambahkan video ke blacklist in-memory dan tulis ke disk.

        Args:
            video_path: path absolut video yang selalu gagal diekstraksi.
        """
        if video_path not in self._blacklist:
            self._blacklist.add(video_path)
            with open(self._blacklist_path, "a") as f:
                f.write(video_path + "\n")
            print(f"[BLACKLIST] Ditambahkan ke blacklist: {video_path}", flush=True)

    def __match_annotations(self, annotation_df: pd.DataFrame, group_path: str) -> List[Tuple[str, str]]:
        """Mencocokkan anotasi dengan direktori subjek.

        Args:
            annotation_df: dataframe anotasi
            group_path: path root grup subjek

        Returns:
            list pasangan path subjek dan label
        """
        matched_subjects = []

        if not os.path.exists(group_path):
            return matched_subjects

        for _, row in annotation_df.iterrows():
            subject_name = row["name"]
            subject_label = row["label"]

            for directory_name in os.listdir(group_path):
                if subject_name.lower() in directory_name.lower() or directory_name.lower() in subject_name.lower():
                    matched_subjects.append((os.path.join(group_path, directory_name), subject_label))
                    break

        return matched_subjects

    def __len__(self) -> int:
        return len(self.datasource)

    def __run_spotter(self, index: int) -> Optional[tuple]:
        """Menjalankan apex phase spotter pada video.

        Jika proses gagal, video langsung ditambahkan ke blacklist persisten
        agar tidak diulang di run berikutnya.

        Args:
            index: indeks video dalam datasource

        Returns:
            tuple apex_indices, phases, label atau None jika gagal
        """
        video_path, label_str = self.datasource[index]
        label = LABEL_MAP.get(label_str, 0)

        print(f"Processing [{index}]: {video_path}", flush=True)

        try:
            apex_indices, phases = self.spotter.process(video_path)
        except Exception as exc:
            print(f"[SKIP] Process error: {video_path} - {exc}", flush=True)
            # T1-A: Tambahkan ke blacklist agar tidak dicoba lagi di run berikutnya
            self._add_to_blacklist(video_path)
            return None

        return apex_indices, phases, label

    def __get_cache(self, video_path: str) -> str:
        """Menghasilkan path cache untuk video.

        Path format: <cache_dir>/<CACHE_VERSION>_<name>_<hash8>.pt
        CACHE_VERSION menjamin cache lama di-skip otomatis saat parameter berubah.

        Args:
            video_path: path absolut ke file video

        Returns:
            path file cache .pt
        """
        filename = os.path.basename(video_path)
        name, _ = os.path.splitext(filename)
        video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
        return os.path.join(self.cache_dir, f"{CACHE_VERSION}_{name}_{video_hash}.pt")
