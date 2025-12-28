"""Activation functions for neural networks."""

import numpy as np

from src.models.base import IActivation


class ReLU(IActivation):
    """ReLU activation function."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply ReLU activation element-wise."""
        return np.maximum(0, x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Compute ReLU gradient for backpropagation."""
        return (x > 0).astype(float)

    def __repr__(self) -> str:
        """Return string representation."""
        return "ReLU()"


class Tanh(IActivation):
    """Hyperbolic tangent activation function."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply Tanh activation element-wise."""
        return np.tanh(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Compute Tanh gradient for backpropagation."""
        tanh_x = np.tanh(x)
        return 1 - tanh_x**2

    def __repr__(self) -> str:
        """Return string representation."""
        return "Tanh()"


class Softmax(IActivation):
    """Softmax activation function for multi-class classification."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply Softmax activation with numerical stability."""
        # Handle empty array
        if x.size == 0:
            return x

        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Compute Softmax gradient for backpropagation."""
        softmax = self.forward(x)
        return softmax * (1 - softmax)

    def __repr__(self) -> str:
        """Return string representation."""
        return "Softmax()"
