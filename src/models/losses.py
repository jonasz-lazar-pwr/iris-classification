"""Loss functions for neural network training."""

import numpy as np

from src.models.base import ILossFunction


class CrossEntropyLoss(ILossFunction):
    """Cross-entropy loss for multi-class classification."""

    def __init__(self, epsilon: float = 1e-15) -> None:
        """Initialize cross-entropy loss with numerical stability epsilon."""
        self.epsilon = epsilon

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute average cross-entropy loss over batch."""
        # Clip predictions for numerical stability
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        # Handle both 1D and 2D inputs
        if y_true.ndim == 1:
            # Convert to one-hot if needed
            num_classes = y_pred.shape[-1]
            y_true_onehot = np.eye(num_classes)[y_true]
        else:
            y_true_onehot = y_true

        # Compute cross-entropy: -Σ y_true * log(y_pred)
        loss = -np.sum(y_true_onehot * np.log(y_pred_clipped), axis=-1)

        # Return average loss over batch
        return float(np.mean(loss))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute gradient of loss with respect to predictions."""
        # Handle both 1D and 2D inputs
        if y_true.ndim == 1:
            # Convert to one-hot if needed
            num_classes = y_pred.shape[-1]
            y_true_onehot = np.eye(num_classes)[y_true]
        else:
            y_true_onehot = y_true

        # Simplified gradient for softmax + cross-entropy
        gradient = y_pred - y_true_onehot

        # Average gradient over batch
        return gradient / y_pred.shape[0]

    def __repr__(self) -> str:
        """Return string representation of loss function."""
        return f"CrossEntropyLoss(epsilon={self.epsilon})"
