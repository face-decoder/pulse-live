from __future__ import annotations

from abc import ABC, abstractmethod

from .subject_sample import SubjectSample, TransformOutput


class BaseTransform(ABC):
    """Interface minimum untuk semua transform dalam pipeline ini."""

    @abstractmethod
    def __call__(self, sample: SubjectSample | TransformOutput) -> TransformOutput: ...

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_")
        )
        return f"{self.__class__.__name__}({params})"
