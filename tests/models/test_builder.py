"""Comprehensive tests for MLPBuilder."""

import numpy as np
import pytest

from src.models.activations import ReLU, Softmax, Tanh
from src.models.builder import MLPBuilder
from src.models.factory import ModelComponentFactory
from src.models.losses import CrossEntropyLoss
from src.models.mlp import MLP
from src.models.optimizers import SGDMomentum


@pytest.fixture
def builder():
    """Create fresh builder instance."""
    return MLPBuilder()


@pytest.fixture
def factory():
    """Create factory instance."""
    return ModelComponentFactory()


class TestMLPBuilderInitialization:
    """Test builder initialization."""

    def test_builder_initialization(self, builder):
        """Test that builder initializes correctly."""
        assert builder is not None
        assert isinstance(builder, MLPBuilder)

    def test_builder_with_custom_factory(self, factory):
        """Test initializing builder with custom factory."""
        builder = MLPBuilder(factory=factory)
        assert builder is not None

    def test_builder_creates_default_factory(self):
        """Test that builder creates default factory if none provided."""
        builder = MLPBuilder()
        assert builder._factory is not None


class TestMLPBuilderInputSize:
    """Test input size configuration."""

    def test_with_input_size(self, builder):
        """Test setting input size."""
        result = builder.with_input_size(10)

        assert result is builder
        assert builder._input_size == 10

    def test_with_input_size_fluent_api(self, builder):
        """Test that with_input_size returns self for chaining."""
        assert builder.with_input_size(10).with_input_size(20)._input_size == 20

    def test_negative_input_size_raises_error(self, builder):
        """Test that negative input size raises error."""
        with pytest.raises(ValueError, match="Input size must be positive"):
            builder.with_input_size(-1)

    def test_zero_input_size_raises_error(self, builder):
        """Test that zero input size raises error."""
        with pytest.raises(ValueError, match="Input size must be positive"):
            builder.with_input_size(0)


class TestMLPBuilderHiddenLayers:
    """Test hidden layer configuration."""

    def test_add_hidden_layer_with_string_activation(self, builder):
        """Test adding hidden layer with string activation."""
        result = builder.add_hidden_layer(64, "relu")

        assert result is builder
        assert builder._hidden_layers == [64]
        assert len(builder._hidden_activations) == 1
        assert isinstance(builder._hidden_activations[0], ReLU)

    def test_add_hidden_layer_with_activation_object(self, builder):
        """Test adding hidden layer with activation object."""
        activation = Tanh()
        builder.add_hidden_layer(32, activation)

        assert builder._hidden_layers == [32]
        assert builder._hidden_activations[0] is activation

    def test_add_multiple_hidden_layers(self, builder):
        """Test adding multiple hidden layers."""
        builder.add_hidden_layer(128, "relu").add_hidden_layer(64, "tanh").add_hidden_layer(
            32, "relu"
        )

        assert builder._hidden_layers == [128, 64, 32]
        assert len(builder._hidden_activations) == 3
        assert isinstance(builder._hidden_activations[0], ReLU)
        assert isinstance(builder._hidden_activations[1], Tanh)
        assert isinstance(builder._hidden_activations[2], ReLU)

    def test_negative_hidden_layer_size_raises_error(self, builder):
        """Test that negative hidden layer size raises error."""
        with pytest.raises(ValueError, match="Hidden layer size must be positive"):
            builder.add_hidden_layer(-1, "relu")

    def test_zero_hidden_layer_size_raises_error(self, builder):
        """Test that zero hidden layer size raises error."""
        with pytest.raises(ValueError, match="Hidden layer size must be positive"):
            builder.add_hidden_layer(0, "relu")

    def test_invalid_activation_type_raises_error(self, builder):
        """Test that invalid activation type raises error."""
        with pytest.raises(TypeError, match="activation must be str or IActivation"):
            builder.add_hidden_layer(64, 123)  # type: ignore[arg-type]


class TestMLPBuilderOutputLayer:
    """Test output layer configuration."""

    def test_with_output_layer_string_activation(self, builder):
        """Test setting output layer with string activation."""
        result = builder.with_output_layer(3, "softmax")

        assert result is builder
        assert builder._output_size == 3
        assert isinstance(builder._output_activation, Softmax)

    def test_with_output_layer_activation_object(self, builder):
        """Test setting output layer with activation object."""
        activation = Softmax()
        builder.with_output_layer(5, activation)

        assert builder._output_size == 5
        assert builder._output_activation is activation

    def test_negative_output_size_raises_error(self, builder):
        """Test that negative output size raises error."""
        with pytest.raises(ValueError, match="Output size must be positive"):
            builder.with_output_layer(-1, "softmax")

    def test_zero_output_size_raises_error(self, builder):
        """Test that zero output size raises error."""
        with pytest.raises(ValueError, match="Output size must be positive"):
            builder.with_output_layer(0, "softmax")

    def test_invalid_output_activation_type_raises_error(self, builder):
        """Test that invalid activation type raises error."""
        with pytest.raises(TypeError, match="activation must be str or IActivation"):
            builder.with_output_layer(3, None)  # type: ignore[arg-type]


class TestMLPBuilderLoss:
    """Test loss function configuration."""

    def test_with_loss_string(self, builder):
        """Test setting loss with string."""
        result = builder.with_loss("cross_entropy")

        assert result is builder
        assert isinstance(builder._loss, CrossEntropyLoss)

    def test_with_loss_object(self, builder):
        """Test setting loss with object."""
        loss = CrossEntropyLoss()
        builder.with_loss(loss)

        assert builder._loss is loss

    def test_with_loss_string_and_kwargs(self, builder):
        """Test setting loss with string and kwargs."""
        builder.with_loss("cross_entropy", epsilon=1e-10)

        assert isinstance(builder._loss, CrossEntropyLoss)
        assert builder._loss.epsilon == 1e-10

    def test_with_loss_object_ignores_kwargs(self, builder):
        """Test that kwargs are ignored when loss is object."""
        loss = CrossEntropyLoss(epsilon=1e-7)
        builder.with_loss(loss, epsilon=1e-10)

        assert builder._loss.epsilon == 1e-7

    def test_invalid_loss_type_raises_error(self, builder):
        """Test that invalid loss type raises error."""
        with pytest.raises(TypeError, match="loss must be str or ILossFunction"):
            builder.with_loss(123)  # type: ignore[arg-type]


class TestMLPBuilderOptimizer:
    """Test optimizer configuration."""

    def test_with_optimizer_string(self, builder):
        """Test setting optimizer with string."""
        result = builder.with_optimizer("sgd_momentum")

        assert result is builder
        assert isinstance(builder._optimizer, SGDMomentum)

    def test_with_optimizer_object(self, builder):
        """Test setting optimizer with object."""
        optimizer = SGDMomentum()
        builder.with_optimizer(optimizer)

        assert builder._optimizer is optimizer

    def test_with_optimizer_string_and_kwargs(self, builder):
        """Test setting optimizer with string and kwargs."""
        builder.with_optimizer("sgd_momentum", learning_rate=0.001, momentum=0.95)

        assert isinstance(builder._optimizer, SGDMomentum)
        assert builder._optimizer.learning_rate == 0.001
        assert builder._optimizer.momentum == 0.95

    def test_with_optimizer_object_ignores_kwargs(self, builder):
        """Test that kwargs are ignored when optimizer is object."""
        optimizer = SGDMomentum(learning_rate=0.01)
        builder.with_optimizer(optimizer, learning_rate=0.001)

        assert builder._optimizer.learning_rate == 0.01

    def test_invalid_optimizer_type_raises_error(self, builder):
        """Test that invalid optimizer type raises error."""
        with pytest.raises(TypeError, match="optimizer must be str or IOptimizer"):
            builder.with_optimizer([])  # type: ignore[arg-type]


class TestMLPBuilderRandomState:
    """Test random state configuration."""

    def test_with_random_state(self, builder):
        """Test setting random state."""
        result = builder.with_random_state(42)

        assert result is builder
        assert builder._random_state == 42

    def test_with_random_state_zero(self, builder):
        """Test that zero random state is valid."""
        builder.with_random_state(0)
        assert builder._random_state == 0

    def test_negative_random_state_raises_error(self, builder):
        """Test that negative random state raises error."""
        with pytest.raises(ValueError, match="Random state must be non-negative"):
            builder.with_random_state(-1)


class TestMLPBuilderBuild:
    """Test model building."""

    def test_build_complete_model(self, builder):
        """Test building complete model."""
        model = (
            builder.with_input_size(10)
            .add_hidden_layer(64, "relu")
            .add_hidden_layer(32, "relu")
            .with_output_layer(3, "softmax")
            .with_loss("cross_entropy")
            .with_optimizer("sgd_momentum")
            .build()
        )

        assert isinstance(model, MLP)
        assert model.layer_sizes == [10, 64, 32, 3]

    def test_build_with_random_state(self, builder):
        """Test building model with random state."""
        model = (
            builder.with_input_size(4)
            .add_hidden_layer(16, "relu")
            .with_output_layer(3, "softmax")
            .with_loss("cross_entropy")
            .with_optimizer("sgd_momentum")
            .with_random_state(42)
            .build()
        )

        assert isinstance(model, MLP)

    def test_build_without_input_size_raises_error(self, builder):
        """Test that building without input size raises error."""
        with pytest.raises(ValueError, match="input_size must be set"):
            (
                builder.add_hidden_layer(64, "relu")
                .with_output_layer(3, "softmax")
                .with_loss("cross_entropy")
                .with_optimizer("sgd_momentum")
                .build()
            )

    def test_build_without_hidden_layers_raises_error(self, builder):
        """Test that building without hidden layers raises error."""
        with pytest.raises(ValueError, match="at least one hidden layer must be added"):
            (
                builder.with_input_size(10)
                .with_output_layer(3, "softmax")
                .with_loss("cross_entropy")
                .with_optimizer("sgd_momentum")
                .build()
            )

    def test_build_without_output_layer_raises_error(self, builder):
        """Test that building without output layer raises error."""
        with pytest.raises(ValueError, match="output_size must be set"):
            (
                builder.with_input_size(10)
                .add_hidden_layer(64, "relu")
                .with_loss("cross_entropy")
                .with_optimizer("sgd_momentum")
                .build()
            )

    def test_build_without_loss_raises_error(self, builder):
        """Test that building without loss raises error."""
        with pytest.raises(ValueError, match="loss function must be set"):
            (
                builder.with_input_size(10)
                .add_hidden_layer(64, "relu")
                .with_output_layer(3, "softmax")
                .with_optimizer("sgd_momentum")
                .build()
            )

    def test_build_without_optimizer_raises_error(self, builder):
        """Test that building without optimizer raises error."""
        with pytest.raises(ValueError, match="optimizer must be set"):
            (
                builder.with_input_size(10)
                .add_hidden_layer(64, "relu")
                .with_output_layer(3, "softmax")
                .with_loss("cross_entropy")
                .build()
            )


class TestMLPBuilderReset:
    """Test builder reset functionality."""

    def test_reset_clears_configuration(self, builder):
        """Test that reset clears all configuration."""
        builder.with_input_size(10).add_hidden_layer(64, "relu").with_output_layer(3, "softmax")

        builder.reset()

        assert builder._input_size is None
        assert builder._hidden_layers == []
        assert builder._hidden_activations == []
        assert builder._output_size is None
        assert builder._output_activation is None

    def test_reset_returns_self(self, builder):
        """Test that reset returns self for chaining."""
        result = builder.reset()
        assert result is builder

    def test_build_after_reset(self, builder):
        """Test building after reset."""
        builder.with_input_size(10).add_hidden_layer(64, "relu").with_output_layer(
            3, "softmax"
        ).with_loss("cross_entropy").with_optimizer("sgd_momentum").build()

        model = (
            builder.reset()
            .with_input_size(5)
            .add_hidden_layer(32, "tanh")
            .with_output_layer(2, "softmax")
            .with_loss("cross_entropy")
            .with_optimizer("sgd_momentum")
            .build()
        )

        assert isinstance(model, MLP)
        assert model.layer_sizes == [5, 32, 2]


class TestMLPBuilderUtilities:
    """Test utility methods."""

    def test_get_current_architecture_incomplete(self, builder):
        """Test getting architecture when incomplete."""
        assert builder.get_current_architecture() == []

    def test_get_current_architecture_partial(self, builder):
        """Test getting architecture when partially complete."""
        builder.with_input_size(10).add_hidden_layer(64, "relu")

        assert builder.get_current_architecture() == []

    def test_get_current_architecture_complete(self, builder):
        """Test getting complete architecture."""
        builder.with_input_size(10).add_hidden_layer(64, "relu").add_hidden_layer(
            32, "relu"
        ).with_output_layer(3, "softmax")

        assert builder.get_current_architecture() == [10, 64, 32, 3]

    def test_repr_incomplete(self, builder):
        """Test string representation when incomplete."""
        repr_str = repr(builder)

        assert "MLPBuilder" in repr_str
        assert "incomplete" in repr_str

    def test_repr_complete(self, builder):
        """Test string representation when complete."""
        builder.with_input_size(10).add_hidden_layer(64, "relu").with_output_layer(3, "softmax")

        repr_str = repr(builder)

        assert "MLPBuilder" in repr_str
        assert "10->64->3" in repr_str


class TestMLPBuilderIntegration:
    """Integration tests with real model."""

    def test_build_and_predict(self, builder):
        """Test building and using model for prediction."""
        model = (
            builder.with_input_size(4)
            .add_hidden_layer(16, "relu")
            .with_output_layer(3, "softmax")
            .with_loss("cross_entropy")
            .with_optimizer("sgd_momentum")
            .with_random_state(42)
            .build()
        )

        X = np.random.randn(5, 4)
        predictions = model.predict(X)

        assert predictions.shape == (5,)
        assert all(0 <= p < 3 for p in predictions)

    def test_builder_reuse(self, builder):
        """Test reusing builder for multiple models."""
        model1 = (
            builder.with_input_size(4)
            .add_hidden_layer(16, "relu")
            .with_output_layer(3, "softmax")
            .with_loss("cross_entropy")
            .with_optimizer("sgd_momentum")
            .with_random_state(42)
            .build()
        )

        model2 = (
            builder.reset()
            .with_input_size(10)
            .add_hidden_layer(32, "tanh")
            .with_output_layer(5, "softmax")
            .with_loss("cross_entropy")
            .with_optimizer("sgd_momentum")
            .with_random_state(123)
            .build()
        )

        assert model1.layer_sizes == [4, 16, 3]
        assert model2.layer_sizes == [10, 32, 5]
