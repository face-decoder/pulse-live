from __future__ import annotations

from typing import Callable, List

from .subject_sample import SubjectSample, TransformOutput


class Compose:
    """
    Chaining some transforms sequentially.

    Some of torchvision.Compose:
      - First step receives SubjectSample and outputs TransformOutput
      - Next step receives TransformOutput and outputs TransformOutput

    Usage:
        transform = Compose([
            WindowSelector(phase_includes=["onset", "apex"]),
            BehavioralFeatures(),
            PadAndMask(max_len=512),
            ChannelZScore(),
            AugmentFlow(training=True),
        ])
        output: TransformOutput = transform(subject_sample)
    """

    def __init__(self, transforms: List[Callable]):
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

    def train(self) -> "Compose":
        """Set semua transform yang memiliki .train() ke mode training."""
        for t in self.transforms:
            if hasattr(t, "train"):
                t.train()
        return self

    def eval(self) -> "Compose":
        """Set semua transform ke mode eval (disable augmentasi)."""
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
