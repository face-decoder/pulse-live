from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class MELabel:
    """Micro-expression emotion label mapper and encoder."""

    POSITIVE = {"happiness", "positive"}
    NEGATIVE = {
        "disgust",
        "fear",
        "sadness",
        "repression",
        "anger",
        "contempt",
        "negative",
    }
    SURPRISE = {"surprise"}

    @classmethod
    def map(cls, emotion: str | float | None, target: str = "2-class") -> str | None:
        """Map raw emotion string to standard target classes.

        Args:
            emotion: Raw emotion string or value.
            target: Mapping target ('2-class', '3-class', or 'raw').

        Returns:
            Mapped label string or None if excluded/invalid.
        """
        if emotion is None or pd.isna(emotion):
            return None

        emo = str(emotion).strip().lower()

        if target == "2-class":
            if emo in cls.POSITIVE:
                return "positive"
            if emo in cls.NEGATIVE:
                return "negative"
            return None

        if target == "3-class":
            if emo in cls.POSITIVE:
                return "positive"
            if emo in cls.NEGATIVE:
                return "negative"
            if emo in cls.SURPRISE:
                return "surprise"
            return "others"

        return emo

    @staticmethod
    def encode(labels: Sequence[str]) -> tuple[np.ndarray, list[str]]:
        """Encode categorical labels into numeric array.

        Args:
            labels: List of label strings.

        Returns:
            Tuple of (encoded_array, class_names).
        """
        encoder = LabelEncoder()
        y = encoder.fit_transform(labels)
        return y, list(encoder.classes_)
