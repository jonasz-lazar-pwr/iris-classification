"""Configuration for MLP models."""

import logging

from dataclasses import dataclass, field

from src.models.base import IMLPConfig

logger = logging.getLogger(__name__)


@dataclass
class MLPConfig(IMLPConfig):
    """Configuration for Multi-Layer Perceptron neural network."""

    input_size: int
    hidden_layers: list[int]
    output_size: int
    hidden_activation: str = "relu"
    output_activation: str = "softmax"
    loss_function: str = "cross_entropy"
    optimizer: str = "sgd_momentum"
    learning_rate: float = 0.01
    momentum: float = 0.9
    random_state: int | None = None
    optimizer_kwargs: dict = field(default_factory=dict)
    loss_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration parameters."""
        self._validate_layer_sizes()
        self._validate_activations()
        self._validate_hyperparameters()
        self._validate_component_names()

        logger.debug(f"MLPConfig validated: {self}")

    def _validate_layer_sizes(self) -> None:
        """Validate layer size parameters."""
        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {self.input_size}")

        if self.output_size <= 0:
            raise ValueError(f"output_size must be positive, got {self.output_size}")

        if not self.hidden_layers:
            raise ValueError("hidden_layers cannot be empty")

        if not all(isinstance(size, int) for size in self.hidden_layers):
            raise TypeError("All hidden layer sizes must be integers")

        if any(size <= 0 for size in self.hidden_layers):
            raise ValueError("All hidden layer sizes must be positive")

    def _validate_activations(self) -> None:
        """Validate activation function names."""
        if not self.hidden_activation or not isinstance(self.hidden_activation, str):
            raise ValueError(
                f"hidden_activation must be a non-empty string, got {self.hidden_activation}"
            )

        if not self.output_activation or not isinstance(self.output_activation, str):
            raise ValueError(
                f"output_activation must be a non-empty string, got {self.output_activation}"
            )

    def _validate_hyperparameters(self) -> None:
        """Validate hyperparameter values."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        if not 0 <= self.momentum < 1:
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}")

        if self.random_state is not None and self.random_state < 0:
            raise ValueError(f"random_state must be non-negative or None, got {self.random_state}")

    def _validate_component_names(self) -> None:
        """Validate component name parameters."""
        if not self.loss_function or not isinstance(self.loss_function, str):
            raise ValueError(f"loss_function must be a non-empty string, got {self.loss_function}")

        if not self.optimizer or not isinstance(self.optimizer, str):
            raise ValueError(f"optimizer must be a non-empty string, got {self.optimizer}")

    def get_network_architecture(self) -> list[int]:
        """Get complete network architecture as list of layer sizes."""
        return [self.input_size, *self.hidden_layers, self.output_size]

    def get_total_layers(self) -> int:
        """Get total number of layers (input + hidden + output)."""
        return len(self.get_network_architecture())

    def get_total_parameters(self) -> int:
        """Calculate total number of trainable parameters (weights + biases)."""
        architecture = self.get_network_architecture()
        total = 0

        for i in range(len(architecture) - 1):
            weights = architecture[i] * architecture[i + 1]
            biases = architecture[i + 1]
            total += weights + biases

        return total

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "input_size": self.input_size,
            "hidden_layers": self.hidden_layers.copy(),
            "output_size": self.output_size,
            "hidden_activation": self.hidden_activation,
            "output_activation": self.output_activation,
            "loss_function": self.loss_function,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "random_state": self.random_state,
            "optimizer_kwargs": self.optimizer_kwargs.copy(),
            "loss_kwargs": self.loss_kwargs.copy(),
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "MLPConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def __repr__(self) -> str:
        """String representation of configuration."""
        arch = "->".join(map(str, self.get_network_architecture()))
        return (
            f"MLPConfig(architecture={arch}, "
            f"activations=[{self.hidden_activation}, {self.output_activation}], "
            f"optimizer={self.optimizer}, lr={self.learning_rate})"
        )


@dataclass
class MLPConfigBuilder:
    """Builder for creating MLPConfig with fluent API."""

    _input_size: int | None = None
    _hidden_layers: list[int] = field(default_factory=list)
    _output_size: int | None = None
    _hidden_activation: str = "relu"
    _output_activation: str = "softmax"
    _loss_function: str = "cross_entropy"
    _optimizer: str = "sgd_momentum"
    _learning_rate: float = 0.01
    _momentum: float = 0.9
    _random_state: int | None = None
    _optimizer_kwargs: dict = field(default_factory=dict)
    _loss_kwargs: dict = field(default_factory=dict)

    def with_input_size(self, size: int) -> "MLPConfigBuilder":
        """Set input layer size."""
        self._input_size = size
        return self

    def add_hidden_layer(self, size: int) -> "MLPConfigBuilder":
        """Add hidden layer."""
        self._hidden_layers.append(size)
        return self

    def with_hidden_layers(self, layers: list[int]) -> "MLPConfigBuilder":
        """Set all hidden layers at once."""
        self._hidden_layers = layers.copy()
        return self

    def with_output_size(self, size: int) -> "MLPConfigBuilder":
        """Set output layer size."""
        self._output_size = size
        return self

    def with_hidden_activation(self, activation: str) -> "MLPConfigBuilder":
        """Set hidden layer activation function."""
        self._hidden_activation = activation
        return self

    def with_output_activation(self, activation: str) -> "MLPConfigBuilder":
        """Set output layer activation function."""
        self._output_activation = activation
        return self

    def with_loss(self, loss: str, **kwargs) -> "MLPConfigBuilder":
        """Set loss function."""
        self._loss_function = loss
        self._loss_kwargs = kwargs
        return self

    def with_optimizer(self, optimizer: str, **kwargs) -> "MLPConfigBuilder":
        """Set optimizer."""
        self._optimizer = optimizer
        self._optimizer_kwargs = kwargs
        return self

    def with_learning_rate(self, lr: float) -> "MLPConfigBuilder":
        """Set learning rate."""
        self._learning_rate = lr
        return self

    def with_momentum(self, momentum: float) -> "MLPConfigBuilder":
        """Set momentum."""
        self._momentum = momentum
        return self

    def with_random_state(self, seed: int) -> "MLPConfigBuilder":
        """Set random state."""
        self._random_state = seed
        return self

    def build(self) -> MLPConfig:
        """Build and validate configuration."""
        if self._input_size is None:
            raise ValueError("input_size must be set")

        if self._output_size is None:
            raise ValueError("output_size must be set")

        if not self._hidden_layers:
            raise ValueError("At least one hidden layer must be added")

        return MLPConfig(
            input_size=self._input_size,
            hidden_layers=self._hidden_layers.copy(),
            output_size=self._output_size,
            hidden_activation=self._hidden_activation,
            output_activation=self._output_activation,
            loss_function=self._loss_function,
            optimizer=self._optimizer,
            learning_rate=self._learning_rate,
            momentum=self._momentum,
            random_state=self._random_state,
            optimizer_kwargs=self._optimizer_kwargs.copy(),
            loss_kwargs=self._loss_kwargs.copy(),
        )

    def reset(self) -> "MLPConfigBuilder":
        """Reset builder to initial state."""
        self._input_size = None
        self._hidden_layers = []
        self._output_size = None
        self._hidden_activation = "relu"
        self._output_activation = "softmax"
        self._loss_function = "cross_entropy"
        self._optimizer = "sgd_momentum"
        self._learning_rate = 0.01
        self._momentum = 0.9
        self._random_state = None
        self._optimizer_kwargs = {}
        self._loss_kwargs = {}
        return self
