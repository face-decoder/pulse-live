import torch
import torch.nn as nn

from .feature_extractor import FeatureExtractor


class SpatialTemporalCNN(nn.Module):
    """MLP classifier operating on pre-extracted feature vectors.

    Attributes:
        INPUT_DIM: Default input dimensionality (78).
        HIDDEN_1: Default first hidden layer width (128).
        HIDDEN_2: Default second hidden layer width (64).
        N_CLASSES: Number of output classes (2).
    """

    INPUT_DIM = FeatureExtractor.TOTAL_DIM   # 78
    HIDDEN_1 = 128
    HIDDEN_2 = 64
    N_CLASSES = 2

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_1: int = HIDDEN_1,
        hidden_2: int = HIDDEN_2,
        dropout_1: float = 0.4,
        dropout_2: float = 0.3,
        n_classes: int = N_CLASSES,
    ) -> None:
        """Initialise the classifier.

        Args:
            input_dim: Dimensionality of the input feature vector.
            hidden_1: Width of the first hidden layer.
            hidden_2: Width of the second hidden layer.
            dropout_1: Dropout probability after the first hidden layer.
            dropout_2: Dropout probability after the second hidden layer.
            n_classes: Number of output classes.
        """
        super().__init__()

        self.feature_dim = input_dim

        self.net = nn.Sequential(
            # Block 1
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_1),
            # Block 2
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_2),
            # Classifier head
            nn.Linear(hidden_2, n_classes),
        )

        self._init_weights()

    # ── forward ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Args:
            x: Input batch of shape ``(B, 78)`` as float32.

        Returns:
            Logits of shape ``(B, 2)``.
        """
        return self.net(x)

    # ── weight initialisation ─────────────────────────────────────────

    def _init_weights(self) -> None:
        """Apply Kaiming initialisation to Linear layers and reset BN."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── freeze / unfreeze ─────────────────────────────────────────────

    def freeze(self) -> None:
        """Freeze all layers except the classifier head.

        Useful at the start of training to stabilise batch-norm
        statistics and early layers before full fine-tuning.
        """
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze the last Linear layer only
        last_linear = [m for m in self.net if isinstance(m, nn.Linear)][-1]
        for param in last_linear.parameters():
            param.requires_grad = True

    def unfreeze(self) -> None:
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    # ── info ──────────────────────────────────────────────────────────

    def count_parameters(self) -> dict[str, int]:
        """Return total and trainable parameter counts.

        Returns:
            A dict with keys ``"total"`` and ``"trainable"``.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return {"total": total, "trainable": trainable}
