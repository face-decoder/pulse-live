from __future__ import annotations

from collections.abc import Callable

from .subject_sample import SubjectSample, TransformOutput


class Compose:
    def __init__(self, transforms: list[Callable]):
        self.transforms = list(transforms)

    def __call__(self, sample: SubjectSample) -> TransformOutput:
        result = None
        for i, t in enumerate(self.transforms):
            if i == 0:
                result = t(sample)
                if not isinstance(result, TransformOutput):
                    raise TypeError(
                        f"Transform[0] ({type(t).__name__}) harus return TransformOutput, "
                        f"got {type(result).__name__}"
                    )
            else:
                result = t(result)
                if not isinstance(result, TransformOutput):
                    raise TypeError(
                        f"Transform[{i}] ({type(t).__name__}) harus return TransformOutput, "
                        f"got {type(result).__name__}"
                    )
        return result

    def train(self) -> Compose:
        for t in self.transforms:
            if hasattr(t, "train"):
                t.train()
        return self

    def eval(self) -> Compose:
        for t in self.transforms:
            if hasattr(t, "eval"):
                t.eval()
        return self

    def __repr__(self) -> str:
        lines = ["Compose(["]
        for t in self.transforms:
            lines.append(f"  {t!r},")
        lines.append("])")
        return "\n".join(lines)
