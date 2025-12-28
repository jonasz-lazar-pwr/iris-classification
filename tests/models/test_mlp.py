"""Unit tests for MLP model."""

import numpy as np
import pytest

from src.models.activations import ReLU, Softmax, Tanh
from src.models.base import IModel
from src.models.losses import CrossEntropyLoss
from src.models.mlp import MLP
from src.models.optimizers import SGDMomentum


class TestMLPInitialization:
    """Test MLP initialization."""

    def test_implements_interface(self):
        """Test that MLP implements IModel interface."""
        model = MLP(
            layer_sizes=[4, 8, 3],
            activations=[ReLU(), Softmax()],
            loss_function=CrossEntropyLoss(),
            optimizer=SGDMomentum(),
        )
        assert isinstance(model, IModel)

    def test_simple_architecture(self):
        """Test initialization with simple architecture."""
        model = MLP(
            layer_sizes=[4, 8, 3],
            activations=[ReLU(), Softmax()],
            loss_function=CrossEntropyLoss(),
            optimizer=SGDMomentum(),
        )
        assert len(model.layers) == 2
        assert model.layer_sizes == [4, 8, 3]

    def test_deep_architecture(self):
        """Test initialization with deep architecture."""
        model = MLP(
            layer_sizes=[4, 64, 32, 16, 3],
            activations=[ReLU(), Tanh(), ReLU(), Softmax()],
            loss_function=CrossEntropyLoss(),
            optimizer=SGDMomentum(),
        )
        assert len(model.layers) == 4
        assert model.layers[0]["W"].shape == (64, 4)
        assert model.layers[1]["W"].shape == (32, 64)
        assert model.layers[2]["W"].shape == (16, 32)
        assert model.layers[3]["W"].shape == (3, 16)

    def test_weight_initialization_shape(self):
        """Test that weights have correct shapes."""
        model = MLP(
            layer_sizes=[10, 20, 5],
            activations=[ReLU(), Softmax()],
            loss_function=CrossEntropyLoss(),
            optimizer=SGDMomentum(),
        )
        assert model.layers[0]["W"].shape == (20, 10)
        assert model.layers[0]["b"].shape == (20,)
        assert model.layers[1]["W"].shape == (5, 20)
        assert model.layers[1]["b"].shape == (5,)

    def test_bias_initialization(self):
        """Test that biases are initialized to zero."""
        model = MLP(
            layer_sizes=[4, 8, 3],
            activations=[ReLU(), Softmax()],
            loss_function=CrossEntropyLoss(),
            optimizer=SGDMomentum(),
        )
        for layer in model.layers:
            np.testing.assert_array_equal(layer["b"], np.zeros_like(layer["b"]))

    def test_random_state_reproducibility(self):
        """Test that same random_state gives same initialization."""
        model1 = MLP(
            layer_sizes=[4, 8, 3],
            activations=[ReLU(), Softmax()],
            loss_function=CrossEntropyLoss(),
            optimizer=SGDMomentum(),
            random_state=42,
        )
        model2 = MLP(
            layer_sizes=[4, 8, 3],
            activations=[ReLU(), Softmax()],
            loss_function=CrossEntropyLoss(),
            optimizer=SGDMomentum(),
            random_state=42,
        )
        np.testing.assert_array_equal(model1.layers[0]["W"], model2.layers[0]["W"])

    def test_invalid_activation_count(self):
        """Test error when activation count doesn't match layers."""
        with pytest.raises(ValueError, match="Number of activations"):
            MLP(
                layer_sizes=[4, 8, 3],
                activations=[ReLU()],  # Should be 2 activations
                loss_function=CrossEntropyLoss(),
                optimizer=SGDMomentum(),
            )


class TestMLPForward:
    """Test MLP forward pass."""

    @pytest.fixture
    def simple_model(self) -> MLP:
        """Create simple MLP for testing."""
        return MLP(
            layer_sizes=[4, 8, 3],
            activations=[ReLU(), Softmax()],
            loss_function=CrossEntropyLoss(),
            optimizer=SGDMomentum(),
            random_state=42,
        )

    def test_forward_single_sample(self, simple_model: MLP):
        """Test forward pass with single sample."""
        X = np.random.randn(4)
        output = simple_model.forward(X)
        assert output.shape == (3,)
        assert np.isclose(np.sum(output), 1.0)  # Softmax sums to 1

    def test_forward_batch(self, simple_model: MLP):
        """Test forward pass with batch."""
        X = np.random.randn(10, 4)
        output = simple_model.forward(X)
        assert output.shape == (10, 3)
        # Each sample should sum to 1 (softmax)
        np.testing.assert_array_almost_equal(np.sum(output, axis=1), np.ones(10))

    def test_forward_output_range(self, simple_model: MLP):
        """Test that forward output is in valid probability range."""
        X = np.random.randn(5, 4)
        output = simple_model.forward(X)
        assert np.all(output >= 0)
        assert np.all(output <= 1)

    def test_forward_caches_values(self, simple_model: MLP):
        """Test that forward pass caches intermediate values."""
        X = np.random.randn(4)
        simple_model.forward(X)
        assert "a0" in simple_model.cache
        assert "z1" in simple_model.cache
        assert "a1" in simple_model.cache

    def test_forward_invalid_input_shape(self, simple_model: MLP):
        """Test error with invalid input shape."""
        X = np.random.randn(5)  # Should be 4 features
        with pytest.raises(ValueError, match="Input features"):
            simple_model.forward(X)


class TestMLPBackward:
    """Test MLP backward pass."""

    @pytest.fixture
    def simple_model(self) -> MLP:
        """Create simple MLP for testing."""
        return MLP(
            layer_sizes=[4, 8, 3],
            activations=[ReLU(), Softmax()],
            loss_function=CrossEntropyLoss(),
            optimizer=SGDMomentum(learning_rate=0.01),
            random_state=42,
        )

    def test_backward_updates_weights(self, simple_model: MLP):
        """Test that backward pass updates weights."""
        X = np.random.randn(2, 4)
        y_true = np.array([[1, 0, 0], [0, 1, 0]])

        # Get initial parameters
        params_before = simple_model.get_parameters()

        # Forward + backward
        y_pred = simple_model.forward(X)
        loss_grad = simple_model.loss_function.gradient(y_true, y_pred)
        simple_model.backward(loss_grad)

        # Parameters should have changed
        params_after = simple_model.get_parameters()
        for key in params_before:
            assert not np.array_equal(params_before[key], params_after[key])

    def test_backward_single_sample(self, simple_model: MLP):
        """Test backward with single sample."""
        X = np.random.randn(4)
        y_true = np.array([1, 0, 0])

        params_before = simple_model.get_parameters()

        y_pred = simple_model.forward(X)
        loss_grad = simple_model.loss_function.gradient(
            y_true.reshape(1, -1), y_pred.reshape(1, -1)
        )
        simple_model.backward(loss_grad)

        params_after = simple_model.get_parameters()
        assert not np.array_equal(params_before["layer0_W"], params_after["layer0_W"])

    def test_backward_batch(self, simple_model: MLP):
        """Test backward with batch."""
        X = np.random.randn(10, 4)
        y_true = np.eye(3)[np.random.randint(0, 3, 10)]

        y_pred = simple_model.forward(X)
        loss_grad = simple_model.loss_function.gradient(y_true, y_pred)
        simple_model.backward(loss_grad)

        # Should not raise errors


class TestMLPPredict:
    """Test MLP prediction."""

    @pytest.fixture
    def simple_model(self) -> MLP:
        """Create simple MLP for testing."""
        return MLP(
            layer_sizes=[4, 8, 3],
            activations=[ReLU(), Softmax()],
            loss_function=CrossEntropyLoss(),
            optimizer=SGDMomentum(),
            random_state=42,
        )

    def test_predict_single_sample(self, simple_model: MLP):
        """Test prediction for single sample."""
        X = np.random.randn(4)
        prediction = simple_model.predict(X)
        assert isinstance(prediction, (int, np.integer)) or prediction.shape == ()
        assert 0 <= prediction < 3

    def test_predict_batch(self, simple_model: MLP):
        """Test prediction for batch."""
        X = np.random.randn(10, 4)
        predictions = simple_model.predict(X)
        assert predictions.shape == (10,)
        assert np.all((predictions >= 0) & (predictions < 3))

    def test_predict_returns_class_index(self, simple_model: MLP):
        """Test that predict returns argmax of probabilities."""
        X = np.random.randn(5, 4)
        predictions = simple_model.predict(X)
        probabilities = simple_model.forward(X)
        expected = np.argmax(probabilities, axis=1)
        np.testing.assert_array_equal(predictions, expected)


class TestMLPParameters:
    """Test parameter management."""

    @pytest.fixture
    def simple_model(self) -> MLP:
        """Create simple MLP for testing."""
        return MLP(
            layer_sizes=[4, 8, 3],
            activations=[ReLU(), Softmax()],
            loss_function=CrossEntropyLoss(),
            optimizer=SGDMomentum(),
            random_state=42,
        )

    def test_get_parameters(self, simple_model: MLP):
        """Test getting parameters."""
        params = simple_model.get_parameters()
        assert "layer0_W" in params
        assert "layer0_b" in params
        assert "layer1_W" in params
        assert "layer1_b" in params

    def test_set_parameters(self, simple_model: MLP):
        """Test setting parameters."""
        params = simple_model.get_parameters()

        # Modify parameters
        params["layer0_W"] = np.ones_like(params["layer0_W"])

        # Set back
        simple_model.set_parameters(params)

        # Verify
        new_params = simple_model.get_parameters()
        np.testing.assert_array_equal(new_params["layer0_W"], np.ones_like(params["layer0_W"]))

    def test_set_parameters_creates_copy(self, simple_model: MLP):
        """Test that set_parameters creates copies."""
        params = simple_model.get_parameters()
        original_W = params["layer0_W"].copy()

        simple_model.set_parameters(params)

        # Modify external params
        params["layer0_W"][0, 0] = 999

        # Model parameters should not change
        model_params = simple_model.get_parameters()
        np.testing.assert_array_equal(model_params["layer0_W"], original_W)

    def test_set_parameters_invalid_shape(self, simple_model: MLP):
        """Test error with invalid parameter shapes."""
        params = simple_model.get_parameters()
        params["layer0_W"] = np.random.randn(10, 10)  # Wrong shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            simple_model.set_parameters(params)

    def test_set_parameters_missing_key(self, simple_model: MLP):
        """Test error with missing parameter keys."""
        params = {"layer0_W": np.random.randn(8, 4)}  # Missing layer0_b

        with pytest.raises(ValueError, match="Missing parameters"):
            simple_model.set_parameters(params)


class TestMLPTraining:
    """Integration tests for MLP training."""

    def test_simple_training_loop(self):
        """Test simple training loop reduces loss."""
        # Create simple XOR-like problem
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

        model = MLP(
            layer_sizes=[2, 4, 2],
            activations=[Tanh(), Softmax()],
            loss_function=CrossEntropyLoss(),
            optimizer=SGDMomentum(learning_rate=0.1),
            random_state=42,
        )

        # Initial loss
        y_pred = model.forward(X)
        loss_initial = model.loss_function.compute(y, y_pred)

        # Train for a few iterations
        for _ in range(50):
            y_pred = model.forward(X)
            loss_grad = model.loss_function.gradient(y, y_pred)
            model.backward(loss_grad)

        # Final loss
        y_pred = model.forward(X)
        loss_final = model.loss_function.compute(y, y_pred)

        # Loss should decrease
        assert loss_final < loss_initial

    def test_iris_like_training(self):
        """Test training on Iris-like data."""
        np.random.seed(42)

        # Generate synthetic data
        n_samples = 100
        X = np.random.randn(n_samples, 4)
        y = np.eye(3)[np.random.randint(0, 3, n_samples)]

        model = MLP(
            layer_sizes=[4, 16, 8, 3],
            activations=[ReLU(), ReLU(), Softmax()],
            loss_function=CrossEntropyLoss(),
            optimizer=SGDMomentum(learning_rate=0.01, momentum=0.9),
            random_state=42,
        )

        losses = []
        for _ in range(20):
            y_pred = model.forward(X)
            loss = model.loss_function.compute(y, y_pred)
            losses.append(loss)
            loss_grad = model.loss_function.gradient(y, y_pred)
            model.backward(loss_grad)

        # Loss should generally decrease
        assert losses[-1] < losses[0]


class TestMLPRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        model = MLP(
            layer_sizes=[4, 8, 3],
            activations=[ReLU(), Softmax()],
            loss_function=CrossEntropyLoss(),
            optimizer=SGDMomentum(),
        )
        repr_str = repr(model)
        assert "MLP" in repr_str
        assert "[4, 8, 3]" in repr_str
        assert "ReLU" in repr_str
        assert "Softmax" in repr_str
