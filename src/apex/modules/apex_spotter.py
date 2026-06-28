from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple


class ApexSpotter(ABC):
    """Abstract base for apex spotters.

    Concrete implementations must reset per-video state, process a video,
    and expose their archived flow payload.
    """

    @abstractmethod
    def process(self, video_path: str, phase_mode: str = "onset_to_apex") -> Tuple[List[int], dict]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    # @abstractmethod
    # def export_flow_data(self) -> dict:
    #     raise NotImplementedError
