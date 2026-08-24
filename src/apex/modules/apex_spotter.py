from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple


class ApexSpotter(ABC):
    @abstractmethod
    def process(
        self, video_path: str, phase_mode: str = "onset_to_apex"
    ) -> Tuple[List[int], dict]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
