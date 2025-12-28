"""Unit tests for loss functions."""

import numpy as np
import pytest

from src.models.activations import Softmax
from src.models.base import ILossFunction
from src.models.losses import CrossEntropyLoss


class TestCrossEntropyLoss:
    """Test suite for CrossEntropyLoss."""

    @pytest.fixture
    def loss_fn(self) -> CrossEntropyLoss:
        """Create CrossEntropyLoss instance."""
        return CrossEntropyLoss()

    def test_implements_interface(self, loss_fn: CrossEntropyLoss):
        """Test that CrossEntropyLoss implements ILossFunction."""
        assert isinstance(loss_fn, ILossFunction)

    def test_perfect_prediction(self, loss_fn: CrossEntropyLoss):
        """Test loss with perfect prediction."""
        y_true = np.array([[0, 1, 0]])
        y_pred = np.array([[0.0, 1.0, 0.0]])
        loss = loss_fn.compute(y_true, y_pred)
        # Loss should be very close to 0
        assert loss < 0.01

    def test_worst_prediction(self, loss_fn: CrossEntropyLoss):
        """Test loss with worst prediction."""
        y_true = np.array([[0, 1, 0]])
        y_pred = np.array([[0.0, 0.0, 1.0]])  # completely wrong
        loss = loss_fn.compute(y_true, y_pred)
        # Loss should be very high
        assert loss > 10.0

    def test_compute_single_sample(self, loss_fn: CrossEntropyLoss):
        """Test loss computation for single sample."""
        y_true = np.array([[0, 1, 0]])
        y_pred = np.array([[0.1, 0.7, 0.2]])
        loss = loss_fn.compute(y_true, y_pred)
        # Expected: -log(0.7) ≈ 0.357
        expected = -np.log(0.7)
        assert np.isclose(loss, expected, rtol=1e-3)

    def test_compute_batch(self, loss_fn: CrossEntropyLoss):
        """Test loss computation for batch."""
        y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        y_pred = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
        loss = loss_fn.compute(y_true, y_pred)

        # Expected: mean([-log(0.7), -log(0.8), -log(0.7)])
        individual_losses = [-np.log(0.7), -np.log(0.8), -np.log(0.7)]
        expected = np.mean(individual_losses)
        assert np.isclose(loss, expected, rtol=1e-3)

    def test_compute_with_label_indices(self, loss_fn: CrossEntropyLoss):
        """Test loss with label indices instead of one-hot."""
        y_true = np.array([1, 0, 2])  # class indices
        y_pred = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
        loss = loss_fn.compute(y_true, y_pred)

        # Expected: mean([-log(0.7), -log(0.8), -log(0.7)])
        individual_losses = [-np.log(0.7), -np.log(0.8), -np.log(0.7)]
        expected = np.mean(individual_losses)
        assert np.isclose(loss, expected, rtol=1e-3)

    def test_numerical_stability_zero(self, loss_fn: CrossEntropyLoss):
        """Test numerical stability with zero predictions."""
        y_true = np.array([[0, 1, 0]])
        y_pred = np.array([[0.0, 0.0, 1.0]])
        # Should not raise error or return inf/nan
        loss = loss_fn.compute(y_true, y_pred)
        assert np.isfinite(loss)
        assert loss > 0

    def test_numerical_stability_one(self, loss_fn: CrossEntropyLoss):
        """Test numerical stability with predictions close to 1."""
        y_true = np.array([[0, 1, 0]])
        y_pred = np.array([[0.0, 1.0, 0.0]])
        loss = loss_fn.compute(y_true, y_pred)
        assert np.isfinite(loss)
        assert loss < 1.0

    def test_gradient_perfect_prediction(self, loss_fn: CrossEntropyLoss):
        """Test gradient with perfect prediction."""
        y_true = np.array([[0, 1, 0]])
        y_pred = np.array([[0.0, 1.0, 0.0]])
        grad = loss_fn.gradient(y_true, y_pred)
        # Gradient should be zero for perfect prediction
        expected = np.array([[0.0, 0.0, 0.0]])
        np.testing.assert_array_almost_equal(grad, expected)

    def test_gradient_single_sample(self, loss_fn: CrossEntropyLoss):
        """Test gradient computation for single sample."""
        y_true = np.array([[0, 1, 0]])
        y_pred = np.array([[0.1, 0.7, 0.2]])
        grad = loss_fn.gradient(y_true, y_pred)

        # Expected: (y_pred - y_true) / batch_size
        expected = np.array([[0.1, -0.3, 0.2]])
        np.testing.assert_array_almost_equal(grad, expected)

    def test_gradient_batch(self, loss_fn: CrossEntropyLoss):
        """Test gradient computation for batch."""
        y_true = np.array([[0, 1, 0], [1, 0, 0]])
        y_pred = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
        grad = loss_fn.gradient(y_true, y_pred)

        # Expected: (y_pred - y_true) / batch_size
        expected = (y_pred - y_true) / 2
        np.testing.assert_array_almost_equal(grad, expected)

    def test_gradient_with_label_indices(self, loss_fn: CrossEntropyLoss):
        """Test gradient with label indices instead of one-hot."""
        y_true = np.array([1, 0])  # class indices
        y_pred = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
        grad = loss_fn.gradient(y_true, y_pred)

        # Convert to one-hot
        y_true_onehot = np.array([[0, 1, 0], [1, 0, 0]])
        expected = (y_pred - y_true_onehot) / 2
        np.testing.assert_array_almost_equal(grad, expected)

    def test_gradient_shape(self, loss_fn: CrossEntropyLoss):
        """Test gradient has same shape as predictions."""
        y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        y_pred = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.2, 0.2, 0.6]])
        grad = loss_fn.gradient(y_true, y_pred)
        assert grad.shape == y_pred.shape

    def test_loss_monotonicity(self, loss_fn: CrossEntropyLoss):
        """Test that loss increases as prediction confidence decreases."""
        y_true = np.array([[0, 1, 0]])

        # High confidence correct prediction
        y_pred_high = np.array([[0.1, 0.8, 0.1]])
        loss_high = loss_fn.compute(y_true, y_pred_high)

        # Medium confidence correct prediction
        y_pred_medium = np.array([[0.2, 0.6, 0.2]])
        loss_medium = loss_fn.compute(y_true, y_pred_medium)

        # Low confidence correct prediction
        y_pred_low = np.array([[0.3, 0.4, 0.3]])
        loss_low = loss_fn.compute(y_true, y_pred_low)

        # Loss should increase as confidence decreases
        assert loss_high < loss_medium < loss_low

    def test_custom_epsilon(self):
        """Test custom epsilon value."""
        loss_fn = CrossEntropyLoss(epsilon=1e-10)
        assert loss_fn.epsilon == 1e-10

        y_true = np.array([[0, 1, 0]])
        y_pred = np.array([[0.0, 1.0, 0.0]])
        loss = loss_fn.compute(y_true, y_pred)
        assert np.isfinite(loss)

    def test_repr(self, loss_fn: CrossEntropyLoss):
        """Test string representation."""
        assert "CrossEntropyLoss" in repr(loss_fn)
        assert "epsilon" in repr(loss_fn)


class TestCrossEntropyLossEdgeCases:
    """Test edge cases for CrossEntropyLoss."""

    def test_binary_classification(self):
        """Test with binary classification (2 classes)."""
        loss_fn = CrossEntropyLoss()
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])

        loss = loss_fn.compute(y_true, y_pred)
        assert loss > 0
        assert np.isfinite(loss)

        grad = loss_fn.gradient(y_true, y_pred)
        assert grad.shape == y_pred.shape

    def test_many_classes(self):
        """Test with many classes (10 classes)."""
        loss_fn = CrossEntropyLoss()
        y_true = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        y_pred = np.array([[0.1, 0.05, 0.6, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01]])

        loss = loss_fn.compute(y_true, y_pred)
        assert loss > 0
        assert np.isfinite(loss)

    def test_large_batch(self):
        """Test with large batch size."""
        loss_fn = CrossEntropyLoss()
        batch_size = 1000
        num_classes = 3

        y_true = np.eye(num_classes)[np.random.randint(0, num_classes, batch_size)]
        y_pred = np.random.rand(batch_size, num_classes)
        y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)  # normalize

        loss = loss_fn.compute(y_true, y_pred)
        assert np.isfinite(loss)
        assert loss > 0

        grad = loss_fn.gradient(y_true, y_pred)
        assert grad.shape == y_pred.shape

    def test_uniform_predictions(self):
        """Test with uniform predictions (maximum uncertainty)."""
        loss_fn = CrossEntropyLoss()
        y_true = np.array([[0, 1, 0]])
        y_pred = np.array([[1 / 3, 1 / 3, 1 / 3]])  # uniform

        loss = loss_fn.compute(y_true, y_pred)
        # Loss should be -log(1/3) ≈ 1.099
        expected = -np.log(1 / 3)
        assert np.isclose(loss, expected, rtol=1e-3)


class TestCrossEntropyLossIntegration:
    """Integration tests for CrossEntropyLoss with softmax."""

    def test_softmax_cross_entropy_gradient(self):
        """Test that softmax + cross-entropy gradient simplifies correctly."""
        loss_fn = CrossEntropyLoss()
        softmax = Softmax()

        # Logits before softmax
        logits = np.array([[2.0, 1.0, 0.1]])
        y_true = np.array([[0, 1, 0]])

        # Forward pass
        y_pred = softmax.forward(logits)
        loss = loss_fn.compute(y_true, y_pred)

        # Backward pass
        grad = loss_fn.gradient(y_true, y_pred)

        # Gradient should equal (y_pred - y_true) / batch_size
        expected = (y_pred - y_true) / 1
        np.testing.assert_array_almost_equal(grad, expected)

        assert loss > 0
        assert np.isfinite(loss)

    def test_iris_example(self):
        """Test with Iris-like example."""
        loss_fn = CrossEntropyLoss()

        # 3 samples, 3 classes (Iris)
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # one of each class
        y_pred = np.array(
            [[0.85, 0.10, 0.05], [0.15, 0.75, 0.10], [0.10, 0.15, 0.75]]
        )  # good predictions

        loss = loss_fn.compute(y_true, y_pred)
        grad = loss_fn.gradient(y_true, y_pred)

        # Loss should be relatively low (good predictions)
        assert loss < 1.0
        assert grad.shape == y_pred.shape
