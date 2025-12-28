"""Base interfaces for neural network components."""

from abc import ABC, abstractmethod

import numpy as np


class IActivation(ABC):
    """Interface for activation functions."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        pass

    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Compute derivative for backpropagation."""
        pass


class ILossFunction(ABC):
    """Interface for loss functions."""

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss value."""
        pass

    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute gradient for backpropagation."""
        pass


class IOptimizer(ABC):
    """Interface for optimization algorithms."""

    @abstractmethod
    def update(self, weights: np.ndarray, gradients: np.ndarray, layer_id: str) -> np.ndarray:
        """Update weights based on gradients."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset optimizer state."""
        pass


class IModel(ABC):
    """Interface for neural network models."""

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass - compute predictions."""
        pass

    @abstractmethod
    def backward(self, loss_gradient: np.ndarray) -> None:
        """Backward pass - compute gradients."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """Get model weights and biases."""
        pass

    @abstractmethod
    def set_parameters(self, params: dict) -> None:
        """Set model weights and biases."""
        pass


class IModelComponentFactory(ABC):
    """Interface for creating model components (activations, losses, optimizers)."""

    @abstractmethod
    def create_activation(self, activation_type: str, **kwargs) -> IActivation:
        """Create activation function instance."""
        pass

    @abstractmethod
    def create_loss(self, loss_type: str, **kwargs) -> ILossFunction:
        """Create loss function instance."""
        pass

    @abstractmethod
    def create_optimizer(self, optimizer_type: str, **kwargs) -> IOptimizer:
        """Create optimizer instance."""
        pass

    @abstractmethod
    def list_available_components(self) -> dict[str, list[str]]:
        """List all registered component types."""
        pass


class IMLPConfig(ABC):
    """Interface for MLP configuration."""

    @abstractmethod
    def validate(self) -> None:
        """Validate configuration parameters."""
        pass


class IModelBuilder(ABC):
    """Interface for building neural network models with fluent API."""

    @abstractmethod
    def with_input_size(self, size: int) -> "IModelBuilder":
        """Set input layer size."""
        pass

    @abstractmethod
    def add_hidden_layer(self, size: int, activation: str | IActivation) -> "IModelBuilder":
        """Add hidden layer with activation."""
        pass

    @abstractmethod
    def with_output_layer(self, size: int, activation: str | IActivation) -> "IModelBuilder":
        """Set output layer size and activation."""
        pass

    @abstractmethod
    def with_loss(self, loss: str | ILossFunction, **kwargs) -> "IModelBuilder":
        """Set loss function."""
        pass

    @abstractmethod
    def with_optimizer(self, optimizer: str | IOptimizer, **kwargs) -> "IModelBuilder":
        """Set optimizer."""
        pass

    @abstractmethod
    def with_random_state(self, random_state: int) -> "IModelBuilder":
        """Set random state for reproducibility."""
        pass

    @abstractmethod
    def build(self) -> IModel:
        """Build and return the model."""
        pass

    @abstractmethod
    def reset(self) -> "IModelBuilder":
        """Reset builder to initial state for reuse."""
        pass
