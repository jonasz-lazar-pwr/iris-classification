"""Comprehensive tests for ModelComponentFactory."""

import numpy as np
import pytest

from src.models.activations import ReLU, Softmax, Tanh
from src.models.base import IActivation, ILossFunction, IOptimizer
from src.models.factory import (
    ModelComponentFactory,
    register_activation,
    register_loss,
    register_optimizer,
)
from src.models.losses import CrossEntropyLoss
from src.models.optimizers import SGDMomentum


@pytest.fixture
def factory():
    """Create fresh factory instance."""
    return ModelComponentFactory()


class TestModelComponentFactoryInitialization:
    """Test factory initialization."""

    def test_factory_initialization(self, factory):
        """Test that factory initializes correctly."""
        assert factory is not None
        assert isinstance(factory, ModelComponentFactory)

    def test_defaults_are_registered(self, factory):
        """Test that default components are registered on init."""
        components = factory.list_available_components()

        assert "relu" in components["activations"]
        assert "tanh" in components["activations"]
        assert "softmax" in components["activations"]
        assert "cross_entropy" in components["losses"]
        assert "sgd_momentum" in components["optimizers"]

    def test_defaults_registered_only_once(self):
        """Test that defaults are registered only once across multiple instances."""
        factory1 = ModelComponentFactory()
        factory2 = ModelComponentFactory()

        assert factory1.list_available_components() == factory2.list_available_components()


class TestModelComponentFactoryActivations:
    """Test activation creation."""

    def test_create_relu(self, factory):
        """Test creating ReLU activation."""
        activation = factory.create_activation("relu")

        assert isinstance(activation, ReLU)
        assert isinstance(activation, IActivation)

    def test_create_tanh(self, factory):
        """Test creating Tanh activation."""
        activation = factory.create_activation("tanh")

        assert isinstance(activation, Tanh)
        assert isinstance(activation, IActivation)

    def test_create_softmax(self, factory):
        """Test creating Softmax activation."""
        activation = factory.create_activation("softmax")

        assert isinstance(activation, Softmax)
        assert isinstance(activation, IActivation)

    def test_create_unknown_activation_raises_error(self, factory):
        """Test that creating unknown activation raises error."""
        with pytest.raises(ValueError, match="Unknown activation type"):
            factory.create_activation("unknown_activation")

    def test_create_activation_error_shows_available(self, factory):
        """Test that error message shows available activations."""
        with pytest.raises(ValueError, match="Available:"):
            factory.create_activation("invalid")


class TestModelComponentFactoryLosses:
    """Test loss function creation."""

    def test_create_cross_entropy(self, factory):
        """Test creating CrossEntropyLoss."""
        loss = factory.create_loss("cross_entropy")

        assert isinstance(loss, CrossEntropyLoss)
        assert isinstance(loss, ILossFunction)

    def test_create_loss_with_kwargs(self, factory):
        """Test creating loss with custom arguments."""
        loss = factory.create_loss("cross_entropy", epsilon=1e-10)

        assert isinstance(loss, CrossEntropyLoss)
        assert loss.epsilon == 1e-10

    def test_create_unknown_loss_raises_error(self, factory):
        """Test that creating unknown loss raises error."""
        with pytest.raises(ValueError, match="Unknown loss type"):
            factory.create_loss("mse")


class TestModelComponentFactoryOptimizers:
    """Test optimizer creation."""

    def test_create_sgd_momentum(self, factory):
        """Test creating SGDMomentum optimizer."""
        optimizer = factory.create_optimizer("sgd_momentum")

        assert isinstance(optimizer, SGDMomentum)
        assert isinstance(optimizer, IOptimizer)

    def test_create_optimizer_with_kwargs(self, factory):
        """Test creating optimizer with custom arguments."""
        optimizer = factory.create_optimizer("sgd_momentum", learning_rate=0.001, momentum=0.95)

        assert isinstance(optimizer, SGDMomentum)
        assert optimizer.learning_rate == 0.001
        assert optimizer.momentum == 0.95

    def test_create_optimizer_returns_new_instance(self, factory):
        """Test that factory creates new optimizer instances."""
        optimizer1 = factory.create_optimizer("sgd_momentum")
        optimizer2 = factory.create_optimizer("sgd_momentum")

        assert optimizer1 is not optimizer2
        assert isinstance(optimizer1, SGDMomentum)
        assert isinstance(optimizer2, SGDMomentum)

    def test_create_unknown_optimizer_raises_error(self, factory):
        """Test that creating unknown optimizer raises error."""
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            factory.create_optimizer("adam")


class TestModelComponentFactoryRegistration:
    """Test component registration."""

    def test_register_custom_activation(self, factory):
        """Test registering custom activation class."""

        class CustomActivation(IActivation):
            def forward(self, x: np.ndarray) -> np.ndarray:
                return x * 2

            def backward(self, x: np.ndarray) -> np.ndarray:
                return np.ones_like(x) * 2

        ModelComponentFactory.register_activation("custom_test_activation", CustomActivation)

        activation = factory.create_activation("custom_test_activation")
        assert isinstance(activation, CustomActivation)

    def test_register_custom_loss(self, factory):
        """Test registering custom loss class."""

        class CustomLoss(ILossFunction):
            def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
                return 0.0

            def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
                return np.zeros_like(y_pred)

        ModelComponentFactory.register_loss("custom_test_loss", CustomLoss)

        loss = factory.create_loss("custom_test_loss")
        assert isinstance(loss, CustomLoss)

    def test_register_custom_optimizer(self, factory):
        """Test registering custom optimizer class."""

        class CustomOptimizer(IOptimizer):
            def update(
                self, weights: np.ndarray, gradients: np.ndarray, layer_id: str
            ) -> np.ndarray:
                return weights - gradients

            def reset(self) -> None:
                pass

        ModelComponentFactory.register_optimizer("custom_test_optimizer", CustomOptimizer)

        optimizer = factory.create_optimizer("custom_test_optimizer")
        assert isinstance(optimizer, CustomOptimizer)

    def test_register_non_interface_raises_error(self, factory):
        """Test that registering non-interface class raises error."""

        class NotAnActivation:
            pass

        with pytest.raises(TypeError, match="must implement IActivation"):
            ModelComponentFactory.register_activation("bad", NotAnActivation)  # type: ignore[arg-type]


class TestModelComponentFactoryDecorators:
    """Test decorator-based registration."""

    def test_register_activation_decorator(self, factory):
        """Test @register_activation decorator."""

        @register_activation("sigmoid_test")
        class SigmoidActivation(IActivation):
            def forward(self, x: np.ndarray) -> np.ndarray:
                return 1 / (1 + np.exp(-x))

            def backward(self, x: np.ndarray) -> np.ndarray:
                s = self.forward(x)
                return s * (1 - s)

        activation = factory.create_activation("sigmoid_test")
        assert isinstance(activation, SigmoidActivation)

    def test_register_loss_decorator(self, factory):
        """Test @register_loss decorator."""

        @register_loss("mse_test")
        class MSELoss(ILossFunction):
            def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
                return float(np.mean((y_true - y_pred) ** 2))

            def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
                return 2 * (y_pred - y_true) / y_true.size

        loss = factory.create_loss("mse_test")
        assert isinstance(loss, MSELoss)

    def test_register_optimizer_decorator(self, factory):
        """Test @register_optimizer decorator."""

        @register_optimizer("adam_test")
        class AdamOptimizer(IOptimizer):
            def update(
                self, weights: np.ndarray, gradients: np.ndarray, layer_id: str
            ) -> np.ndarray:
                return weights - 0.001 * gradients

            def reset(self) -> None:
                pass

        optimizer = factory.create_optimizer("adam_test")
        assert isinstance(optimizer, AdamOptimizer)


class TestModelComponentFactoryUtilities:
    """Test utility methods."""

    def test_list_available_components(self, factory):
        """Test listing available components."""
        components = factory.list_available_components()

        assert "activations" in components
        assert "losses" in components
        assert "optimizers" in components

        assert isinstance(components["activations"], list)
        assert len(components["activations"]) >= 3
        assert len(components["losses"]) >= 1
        assert len(components["optimizers"]) >= 1

    def test_is_registered_activation(self, factory):
        """Test checking if activation is registered."""
        assert factory.is_registered("activation", "relu") is True
        assert factory.is_registered("activation", "tanh") is True
        assert factory.is_registered("activation", "unknown") is False

    def test_is_registered_loss(self, factory):
        """Test checking if loss is registered."""
        assert factory.is_registered("loss", "cross_entropy") is True
        assert factory.is_registered("loss", "unknown_loss") is False

    def test_is_registered_optimizer(self, factory):
        """Test checking if optimizer is registered."""
        assert factory.is_registered("optimizer", "sgd_momentum") is True
        assert factory.is_registered("optimizer", "unknown_optimizer") is False

    def test_is_registered_invalid_type_raises_error(self, factory):
        """Test that checking invalid component type raises error."""
        with pytest.raises(ValueError, match="Invalid component type"):
            factory.is_registered("invalid_type", "anything")

    def test_repr(self, factory):
        """Test string representation."""
        repr_str = repr(factory)

        assert "ModelComponentFactory" in repr_str
        assert "total_registered" in repr_str


class TestModelComponentFactoryIntegration:
    """Integration tests with real components."""

    def test_create_and_use_activation(self, factory):
        """Test creating and using an activation."""
        activation = factory.create_activation("relu")

        X = np.array([-2, -1, 0, 1, 2])
        output = activation.forward(X)

        assert np.array_equal(output, np.array([0, 0, 0, 1, 2]))

    def test_create_and_use_loss(self, factory):
        """Test creating and using a loss function."""
        loss = factory.create_loss("cross_entropy")

        y_true = np.array([0, 1, 2])
        y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

        loss_value = loss.compute(y_true, y_pred)

        assert isinstance(loss_value, float)
        assert loss_value > 0

    def test_create_and_use_optimizer(self, factory):
        """Test creating and using an optimizer."""
        optimizer = factory.create_optimizer("sgd_momentum", learning_rate=0.1, momentum=0.9)

        weights = np.array([1.0, 2.0, 3.0])
        gradients = np.array([0.1, 0.2, 0.3])

        updated_weights = optimizer.update(weights, gradients, "layer0")

        assert not np.array_equal(weights, updated_weights)
        assert updated_weights.shape == weights.shape

    def test_factory_creates_independent_instances(self, factory):
        """Test that factory creates independent optimizer instances."""
        optimizer1 = factory.create_optimizer("sgd_momentum")
        optimizer2 = factory.create_optimizer("sgd_momentum")

        assert isinstance(optimizer1, SGDMomentum)
        assert isinstance(optimizer2, SGDMomentum)

        weights = np.array([1.0, 2.0])
        gradients = np.array([0.1, 0.2])

        optimizer1.update(weights, gradients, "layer0")
        optimizer2.update(weights * 2, gradients * 2, "layer0")

        # Optimizers should have independent state
        vel1 = optimizer1.get_velocity("layer0")
        vel2 = optimizer2.get_velocity("layer0")

        assert vel1 is not None
        assert vel2 is not None
        assert not np.array_equal(vel1, vel2)
