"""Multi-Layer Perceptron implementation for classification."""

import numpy as np

from src.models.base import IActivation, ILossFunction, IModel, IOptimizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLP(IModel):
    """Multi-Layer Perceptron neural network."""

    def __init__(
        self,
        layer_sizes: list[int],
        activations: list[IActivation],
        loss_function: ILossFunction,
        optimizer: IOptimizer,
        random_state: int = 42,
    ) -> None:
        """Initialize MLP with specified architecture.

        Args:
            layer_sizes: Neurons per layer including input and output [4, 64, 32, 3]
            activations: Activation function for each layer (len = len(layer_sizes) - 1)
            loss_function: Loss function for training
            optimizer: Optimizer for weight updates
            random_state: Random seed for reproducibility
        """
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(
                f"Number of activations ({len(activations)}) must equal "
                f"number of layers - 1 ({len(layer_sizes) - 1})"
            )

        self.rng = np.random.RandomState(random_state)
        self.layer_sizes = layer_sizes
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.cache: dict[str, np.ndarray] = {}

        # Initialize layers with He initialization
        self.layers: list[dict] = []
        for i in range(len(layer_sizes) - 1):
            # He initialization: std = sqrt(2 / fan_in)
            std = np.sqrt(2.0 / layer_sizes[i])
            layer = {
                "W": self.rng.randn(layer_sizes[i + 1], layer_sizes[i]) * std,  # ← self.rng
                "b": np.zeros(layer_sizes[i + 1]),
                "activation": activations[i],
            }
            self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute forward pass through network.

        Args:
            X: Input data (batch_size, input_features) or (input_features,)

        Returns:
            Network predictions (batch_size, output_classes) or (output_classes,)
        """
        # Handle both single sample and batch
        if X.ndim == 1:
            X = X.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False

        # Validate input shape
        if X.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"Input features ({X.shape[1]}) must match layer_sizes[0] ({self.layer_sizes[0]})"
            )

        # Forward through layers
        a = X.T  # Shape: (features, batch_size)
        self.cache = {"a0": a}

        for i, layer in enumerate(self.layers):
            # Linear transformation
            z = layer["W"] @ a + layer["b"].reshape(-1, 1)

            # Activation
            a = layer["activation"].forward(z.T).T  # Transpose for activation

            # Cache for backward pass
            self.cache[f"z{i + 1}"] = z
            self.cache[f"a{i + 1}"] = a

        # Return shape: (batch_size, output_classes)
        output = a.T

        if single_sample:
            return output.flatten()

        return output

    def backward(self, loss_gradient: np.ndarray) -> None:
        """Compute backward pass and update weights.

        Args:
            loss_gradient: Gradient from loss function (batch_size, output_classes)
        """
        # Handle single sample
        if loss_gradient.ndim == 1:
            loss_gradient = loss_gradient.reshape(1, -1)

        batch_size = loss_gradient.shape[0]
        dL_da = loss_gradient.T  # Shape: (output_classes, batch_size)

        # Backpropagate through layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            # Get cached values
            z = self.cache[f"z{i + 1}"]
            a_prev = self.cache[f"a{i}"]

            # Gradient through activation
            activation_grad = layer["activation"].backward(z.T).T
            dL_dz = dL_da * activation_grad

            # Gradients for weights and biases
            dL_dW = (dL_dz @ a_prev.T) / batch_size
            dL_db = np.sum(dL_dz, axis=1) / batch_size

            # Update parameters with optimizer
            layer["W"] = self.optimizer.update(layer["W"], dL_dW, f"layer{i}_W")
            layer["b"] = self.optimizer.update(layer["b"], dL_db, f"layer{i}_b")

            # Gradient for previous layer
            if i > 0:
                dL_da = layer["W"].T @ dL_dz

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input data.

        Args:
            X: Input data (batch_size, input_features) or (input_features,)

        Returns:
            Predicted class indices (batch_size,) or scalar
        """
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=-1)

    def get_parameters(self) -> dict:
        """Get all model parameters (weights and biases).

        Returns:
            Dictionary with layer parameters
        """
        params = {}
        for i, layer in enumerate(self.layers):
            params[f"layer{i}_W"] = layer["W"].copy()
            params[f"layer{i}_b"] = layer["b"].copy()
        return params

    def set_parameters(self, params: dict) -> None:
        """Set model parameters from dictionary.

        Args:
            params: Dictionary with layer parameters
        """
        for i, layer in enumerate(self.layers):
            w_key = f"layer{i}_W"
            b_key = f"layer{i}_b"

            if w_key not in params or b_key not in params:
                raise ValueError(f"Missing parameters for layer {i}")

            if params[w_key].shape != layer["W"].shape:
                raise ValueError(
                    f"Shape mismatch for {w_key}: "
                    f"expected {layer['W'].shape}, got {params[w_key].shape}"
                )

            if params[b_key].shape != layer["b"].shape:
                raise ValueError(
                    f"Shape mismatch for {b_key}: "
                    f"expected {layer['b'].shape}, got {params[b_key].shape}"
                )

            layer["W"] = params[w_key].copy()
            layer["b"] = params[b_key].copy()

    def __repr__(self) -> str:
        """Return string representation of model."""
        return (
            f"MLP(layers={self.layer_sizes}, "
            f"activations={[type(layer['activation']).__name__ for layer in self.layers]})"
        )
