"""Optimization algorithms for neural network training."""

import numpy as np

from src.models.base import IOptimizer


class SGDMomentum(IOptimizer):
    """SGD optimizer with momentum for faster convergence."""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9) -> None:
        """Initialize SGD with Momentum optimizer."""
        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {learning_rate}")
        if not 0 <= momentum < 1:
            raise ValueError(f"Momentum must be in [0, 1), got {momentum}")

        self.learning_rate = learning_rate
        self.momentum = momentum
        self._velocity: dict[str, np.ndarray] = {}

    def update(self, weights: np.ndarray, gradients: np.ndarray, layer_id: str) -> np.ndarray:
        """Update weights using momentum-based gradient descent."""
        # Initialize velocity for this layer if first time
        if layer_id not in self._velocity:
            self._velocity[layer_id] = np.zeros_like(weights)

        # Update velocity: v = momentum * v - lr * grad
        self._velocity[layer_id] = (
            self.momentum * self._velocity[layer_id] - self.learning_rate * gradients
        )

        # Update weights: w = w + v
        updated_weights = weights + self._velocity[layer_id]

        return updated_weights

    def reset(self) -> None:
        """Reset optimizer state and clear all velocity buffers."""
        self._velocity.clear()

    def get_velocity(self, layer_id: str) -> np.ndarray | None:
        """Get velocity buffer for specified layer."""
        return self._velocity.get(layer_id)

    def __repr__(self) -> str:
        """Return string representation of optimizer."""
        return f"SGDMomentum(learning_rate={self.learning_rate}, momentum={self.momentum})"
