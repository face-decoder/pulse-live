from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class Normalizer:
    """Per-feature Z-score normaliser for vectors of shape ``(D,)``.

    Attributes:
        eps: Small constant added to std to prevent division by zero.
        mean_: Per-feature mean computed by :meth:`fit`, or ``None``.
        std_: Per-feature std (+eps) computed by :meth:`fit`, or ``None``.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialise the normaliser.

        Args:
            eps: Small constant added to the standard deviation to
                prevent division by zero.
        """
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    # ── public ────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> Normalizer:
        """Compute mean and std from training data.

        Args:
            X: Training features of shape ``(N_samples, D_features)``.

        Returns:
            ``self``, for method chaining.

        Raises:
            ValueError: If *X* is not 2-dimensional.
        """
        self._validate_input(X)
        self.mean_ = X.mean(axis=0).astype(np.float32)
        self.std_ = (X.std(axis=0) + self.eps).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalisation using the statistics from :meth:`fit`.

        Args:
            X: Features of shape ``(N_samples, D_features)``.

        Returns:
            Normalised array of the same shape, as float32.

        Raises:
            RuntimeError: If the normaliser has not been fitted.
            ValueError: If *X* is not 2-dimensional.
        """
        self._check_fitted()
        self._validate_input(X)
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on *X* then immediately transform *X*.

        Only for training data — never use on the test set.

        Args:
            X: Training features of shape ``(N_samples, D_features)``.

        Returns:
            Normalised array of the same shape, as float32.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        """Map normalised values back to the original scale.

        Useful for feature interpretation after normalisation.

        Args:
            X_norm: Normalised features.

        Returns:
            De-normalised array as float32.

        Raises:
            RuntimeError: If the normaliser has not been fitted.
        """
        self._check_fitted()
        return (X_norm * self.std_ + self.mean_).astype(np.float32)

    def save(self, path: str) -> None:
        """Save normalisation statistics to an ``.npz`` file.

        Args:
            path: Destination file path,
                e.g. ``"normalizer_fold0.npz"``.

        Raises:
            RuntimeError: If the normaliser has not been fitted.
        """
        self._check_fitted()
        np.savez(path, mean=self.mean_, std=self.std_, eps=np.array([self.eps]))
        logger.info("Normalizer saved to %s", path)

    @classmethod
    def load(cls, path: str) -> Normalizer:
        """Load normalisation statistics from an ``.npz`` file.

        Args:
            path: Path saved by :meth:`save`.

        Returns:
            A fitted ``Normalizer`` instance.
        """
        data = np.load(path)
        normalizer = cls(eps=float(data["eps"][0]))
        normalizer.mean_ = data["mean"].astype(np.float32)
        normalizer.std_ = data["std"].astype(np.float32)
        logger.info("Normalizer loaded from %s", path)
        return normalizer

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self.mean_ is not None and self.std_ is not None

    # ── private ───────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        """Raise if the normaliser has not been fitted yet."""
        if not self.is_fitted:
            raise RuntimeError(
                "Normalizer has not been fitted. Call fit() first."
            )

    @staticmethod
    def _validate_input(X: np.ndarray) -> None:
        """Raise if *X* is not a 2-D array."""
        if X.ndim != 2:
            raise ValueError(
                f"Input must be 2-D (N_samples, D_features), "
                f"got shape {X.shape}"
            )
