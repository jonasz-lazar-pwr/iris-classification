"""Builder for constructing MLP models with fluent API."""

import logging

from typing import Any

from src.models.base import (
    IActivation,
    ILossFunction,
    IModel,
    IModelBuilder,
    IOptimizer,
)
from src.models.factory import ModelComponentFactory
from src.models.mlp import MLP

logger = logging.getLogger(__name__)


class MLPBuilder(IModelBuilder):
    """Builder for constructing Multi-Layer Perceptron models with fluent API."""

    def __init__(self, factory: ModelComponentFactory | None = None) -> None:
        """Initialize builder with optional factory."""
        self._factory = factory or ModelComponentFactory()
        self.reset()

    def with_input_size(self, size: int) -> "MLPBuilder":
        """Set input layer size."""
        if size <= 0:
            raise ValueError(f"Input size must be positive, got {size}")

        self._input_size = size
        logger.debug(f"Set input size: {size}")
        return self

    def add_hidden_layer(self, size: int, activation: str | IActivation) -> "MLPBuilder":
        """Add hidden layer with activation."""
        if size <= 0:
            raise ValueError(f"Hidden layer size must be positive, got {size}")

        if isinstance(activation, str):
            activation_obj = self._factory.create_activation(activation)
        elif isinstance(activation, IActivation):
            activation_obj = activation
        else:
            raise TypeError(f"activation must be str or IActivation, got {type(activation)}")

        self._hidden_layers.append(size)
        self._hidden_activations.append(activation_obj)

        logger.debug(f"Added hidden layer: size={size}, activation={activation}")
        return self

    def with_output_layer(self, size: int, activation: str | IActivation) -> "MLPBuilder":
        """Set output layer size and activation."""
        if size <= 0:
            raise ValueError(f"Output size must be positive, got {size}")

        if isinstance(activation, str):
            activation_obj = self._factory.create_activation(activation)
        elif isinstance(activation, IActivation):
            activation_obj = activation
        else:
            raise TypeError(f"activation must be str or IActivation, got {type(activation)}")

        self._output_size = size
        self._output_activation = activation_obj

        logger.debug(f"Set output layer: size={size}, activation={activation}")
        return self

    def with_loss(self, loss: str | ILossFunction, **kwargs: Any) -> "MLPBuilder":
        """Set loss function."""
        if isinstance(loss, str):
            loss_obj = self._factory.create_loss(loss, **kwargs)
        elif isinstance(loss, ILossFunction):
            if kwargs:
                logger.warning(
                    "Loss kwargs provided but loss is already an instance - kwargs ignored"
                )
            loss_obj = loss
        else:
            raise TypeError(f"loss must be str or ILossFunction, got {type(loss)}")

        self._loss = loss_obj
        logger.debug(f"Set loss function: {loss}")
        return self

    def with_optimizer(self, optimizer: str | IOptimizer, **kwargs: Any) -> "MLPBuilder":
        """Set optimizer."""
        if isinstance(optimizer, str):
            optimizer_obj = self._factory.create_optimizer(optimizer, **kwargs)
        elif isinstance(optimizer, IOptimizer):
            if kwargs:
                logger.warning(
                    "Optimizer kwargs provided but optimizer is already an instance - kwargs ignored"
                )
            optimizer_obj = optimizer
        else:
            raise TypeError(f"optimizer must be str or IOptimizer, got {type(optimizer)}")

        self._optimizer = optimizer_obj
        logger.debug(f"Set optimizer: {optimizer}")
        return self

    def with_random_state(self, random_state: int) -> "MLPBuilder":
        """Set random state for reproducibility."""
        if random_state < 0:
            raise ValueError(f"Random state must be non-negative, got {random_state}")

        self._random_state = random_state
        logger.debug(f"Set random state: {random_state}")
        return self

    def build(self) -> IModel:
        """Build and return the model."""
        self._validate_before_build()

        layer_sizes = [self._input_size, *self._hidden_layers, self._output_size]
        activations = [*self._hidden_activations, self._output_activation]

        model = MLP(
            layer_sizes=layer_sizes,
            activations=activations,
            loss_function=self._loss,
            optimizer=self._optimizer,
            random_state=self._random_state,
        )

        return model

    def reset(self) -> "MLPBuilder":
        """Reset builder to initial state for reuse."""
        self._input_size: int | None = None
        self._hidden_layers: list[int] = []
        self._hidden_activations: list[IActivation] = []
        self._output_size: int | None = None
        self._output_activation: IActivation | None = None
        self._loss: ILossFunction | None = None
        self._optimizer: IOptimizer | None = None
        self._random_state: int | None = None

        logger.debug("Builder reset to initial state")
        return self

    def _validate_before_build(self) -> None:
        """Validate that all required components are set."""
        errors = []

        if self._input_size is None:
            errors.append("input_size must be set")

        if not self._hidden_layers:
            errors.append("at least one hidden layer must be added")

        if len(self._hidden_layers) != len(self._hidden_activations):
            errors.append(
                f"mismatch between hidden layers ({len(self._hidden_layers)}) "
                f"and activations ({len(self._hidden_activations)})"
            )

        if self._output_size is None:
            errors.append("output_size must be set")

        if self._output_activation is None:
            errors.append("output_activation must be set")

        if self._loss is None:
            errors.append("loss function must be set")

        if self._optimizer is None:
            errors.append("optimizer must be set")

        if errors:
            raise ValueError(f"Cannot build model: {'; '.join(errors)}")

    def get_current_architecture(self) -> list[int]:
        """Get current architecture configuration."""
        if self._input_size is None or self._output_size is None:
            return []
        return [self._input_size, *self._hidden_layers, self._output_size]

    def __repr__(self) -> str:
        """String representation of builder state."""
        arch = self.get_current_architecture()
        arch_str = "->".join(map(str, arch)) if arch else "incomplete"
        return f"MLPBuilder(architecture={arch_str})"
