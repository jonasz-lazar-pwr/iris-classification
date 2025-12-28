"""Unit tests for optimization algorithms."""

import numpy as np
import pytest

from src.models.base import IOptimizer
from src.models.optimizers import SGDMomentum


class TestSGDMomentum:
    """Test suite for SGDMomentum optimizer."""

    @pytest.fixture
    def optimizer(self) -> SGDMomentum:
        """Create SGDMomentum instance with default parameters."""
        return SGDMomentum(learning_rate=0.01, momentum=0.9)

    def test_implements_interface(self, optimizer: SGDMomentum):
        """Test that SGDMomentum implements IOptimizer."""
        assert isinstance(optimizer, IOptimizer)

    def test_initialization_default(self):
        """Test optimizer initialization with default parameters."""
        optimizer = SGDMomentum()
        assert optimizer.learning_rate == 0.01
        assert optimizer.momentum == 0.9

    def test_initialization_custom(self):
        """Test optimizer initialization with custom parameters."""
        optimizer = SGDMomentum(learning_rate=0.1, momentum=0.95)
        assert optimizer.learning_rate == 0.1
        assert optimizer.momentum == 0.95

    def test_initialization_invalid_lr(self):
        """Test that invalid learning rate raises error."""
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            SGDMomentum(learning_rate=0.0)
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            SGDMomentum(learning_rate=-0.01)

    def test_initialization_invalid_momentum(self):
        """Test that invalid momentum raises error."""
        with pytest.raises(ValueError, match="Momentum must be in"):
            SGDMomentum(momentum=-0.1)
        with pytest.raises(ValueError, match="Momentum must be in"):
            SGDMomentum(momentum=1.0)
        with pytest.raises(ValueError, match="Momentum must be in"):
            SGDMomentum(momentum=1.5)

    def test_first_update_no_momentum(self, optimizer: SGDMomentum):
        """Test first update (velocity starts at zero)."""
        weights = np.array([1.0, 2.0, 3.0])
        gradients = np.array([0.1, 0.2, 0.3])
        layer_id = "test_layer"

        updated = optimizer.update(weights, gradients, layer_id)

        # First update: v = 0 - lr * grad, w = w + v
        expected_velocity = -0.01 * gradients
        expected_weights = weights + expected_velocity

        np.testing.assert_array_almost_equal(updated, expected_weights)

    def test_second_update_with_momentum(self, optimizer: SGDMomentum):
        """Test second update (momentum kicks in)."""
        weights = np.array([1.0, 2.0, 3.0])
        gradients = np.array([0.1, 0.2, 0.3])
        layer_id = "test_layer"

        # First update
        weights_1 = optimizer.update(weights, gradients, layer_id)
        velocity_1 = optimizer.get_velocity(layer_id)

        # Second update with same gradient
        weights_2 = optimizer.update(weights_1, gradients, layer_id)

        # Expected: v2 = 0.9 * v1 - 0.01 * grad
        expected_velocity_2 = 0.9 * velocity_1 - 0.01 * gradients
        expected_weights_2 = weights_1 + expected_velocity_2

        np.testing.assert_array_almost_equal(weights_2, expected_weights_2)

    def test_momentum_accumulation(self, optimizer: SGDMomentum):
        """Test that momentum accumulates over multiple updates."""
        weights = np.array([1.0])
        gradients = np.array([1.0])
        layer_id = "test_layer"

        # Perform multiple updates
        for _ in range(5):
            weights = optimizer.update(weights, gradients, layer_id)

        # Velocity should have accumulated
        velocity = optimizer.get_velocity(layer_id)
        assert np.abs(velocity[0]) > 0.01  # Greater than single update

    def test_different_layers_separate_velocity(self, optimizer: SGDMomentum):
        """Test that different layers have separate velocity buffers."""
        weights_1 = np.array([1.0, 2.0])
        weights_2 = np.array([3.0, 4.0])
        gradients = np.array([0.1, 0.2])

        # Update two different layers
        optimizer.update(weights_1, gradients, "layer1")
        optimizer.update(weights_2, gradients, "layer2")

        # Both should have velocities
        velocity_1 = optimizer.get_velocity("layer1")
        velocity_2 = optimizer.get_velocity("layer2")

        assert velocity_1 is not None
        assert velocity_2 is not None
        # Velocities should be different objects
        assert velocity_1 is not velocity_2

    def test_reset_clears_velocity(self, optimizer: SGDMomentum):
        """Test that reset clears all velocity buffers."""
        weights = np.array([1.0, 2.0, 3.0])
        gradients = np.array([0.1, 0.2, 0.3])

        # Perform update to create velocity
        optimizer.update(weights, gradients, "layer1")
        assert optimizer.get_velocity("layer1") is not None

        # Reset
        optimizer.reset()

        # Velocity should be cleared
        assert optimizer.get_velocity("layer1") is None

    def test_reset_multiple_layers(self, optimizer: SGDMomentum):
        """Test that reset clears velocities for all layers."""
        weights = np.array([1.0])
        gradients = np.array([0.1])

        # Update multiple layers
        optimizer.update(weights, gradients, "layer1")
        optimizer.update(weights, gradients, "layer2")
        optimizer.update(weights, gradients, "layer3")

        # Reset
        optimizer.reset()

        # All velocities should be cleared
        assert optimizer.get_velocity("layer1") is None
        assert optimizer.get_velocity("layer2") is None
        assert optimizer.get_velocity("layer3") is None

    def test_update_preserves_shape(self, optimizer: SGDMomentum):
        """Test that update preserves weight shape."""
        weights = np.random.randn(10, 5)
        gradients = np.random.randn(10, 5)

        updated = optimizer.update(weights, gradients, "layer")

        assert updated.shape == weights.shape

    def test_zero_gradient(self, optimizer: SGDMomentum):
        """Test update with zero gradient."""
        weights = np.array([1.0, 2.0, 3.0])
        gradients = np.zeros(3)

        # First update with zero gradient
        updated = optimizer.update(weights, gradients, "layer")

        # Should not change weights
        np.testing.assert_array_almost_equal(updated, weights)

    def test_negative_gradients(self, optimizer: SGDMomentum):
        """Test update with negative gradients."""
        weights = np.array([1.0, 2.0, 3.0])
        gradients = np.array([-0.1, -0.2, -0.3])

        updated = optimizer.update(weights, gradients, "layer")

        # Negative gradients should increase weights
        assert np.all(updated > weights)

    def test_positive_gradients(self, optimizer: SGDMomentum):
        """Test update with positive gradients."""
        weights = np.array([1.0, 2.0, 3.0])
        gradients = np.array([0.1, 0.2, 0.3])

        updated = optimizer.update(weights, gradients, "layer")

        # Positive gradients should decrease weights
        assert np.all(updated < weights)

    def test_learning_rate_effect(self):
        """Test that higher learning rate leads to larger updates."""
        weights = np.array([1.0])
        gradients = np.array([1.0])

        # Small learning rate
        opt_small = SGDMomentum(learning_rate=0.01, momentum=0.0)
        updated_small = opt_small.update(weights.copy(), gradients, "layer")

        # Large learning rate
        opt_large = SGDMomentum(learning_rate=0.1, momentum=0.0)
        updated_large = opt_large.update(weights.copy(), gradients, "layer")

        # Larger LR should lead to larger change
        change_small = np.abs(updated_small - weights)
        change_large = np.abs(updated_large - weights)
        assert change_large > change_small

    def test_momentum_effect(self):
        """Test that higher momentum leads to faster acceleration."""
        weights = np.array([1.0])
        gradients = np.array([1.0])

        # No momentum
        opt_no_momentum = SGDMomentum(learning_rate=0.01, momentum=0.0)
        weights_no_mom = weights.copy()
        for _ in range(3):
            weights_no_mom = opt_no_momentum.update(weights_no_mom, gradients, "layer")

        # High momentum
        opt_momentum = SGDMomentum(learning_rate=0.01, momentum=0.9)
        weights_mom = weights.copy()
        for _ in range(3):
            weights_mom = opt_momentum.update(weights_mom, gradients, "layer")

        # Momentum should lead to larger total change
        change_no_mom = np.abs(weights_no_mom - weights)
        change_mom = np.abs(weights_mom - weights)
        assert change_mom > change_no_mom

    def test_repr(self, optimizer: SGDMomentum):
        """Test string representation."""
        repr_str = repr(optimizer)
        assert "SGDMomentum" in repr_str
        assert "learning_rate" in repr_str
        assert "momentum" in repr_str


class TestSGDMomentumEdgeCases:
    """Test edge cases for SGDMomentum."""

    def test_very_small_learning_rate(self):
        """Test with very small learning rate."""
        optimizer = SGDMomentum(learning_rate=1e-10, momentum=0.9)
        weights = np.array([1.0, 2.0])
        gradients = np.array([1.0, 1.0])

        updated = optimizer.update(weights, gradients, "layer")

        # Change should be extremely small
        change = np.abs(updated - weights)
        assert np.all(change < 1e-9)

    def test_zero_momentum(self):
        """Test with zero momentum (pure SGD)."""
        optimizer = SGDMomentum(learning_rate=0.1, momentum=0.0)
        weights = np.array([1.0])
        gradients = np.array([1.0])

        # First update
        weights_1 = optimizer.update(weights, gradients, "layer")

        # Second update with same gradient
        weights_2 = optimizer.update(weights_1, gradients, "layer")

        # Changes should be identical (no momentum)
        change_1 = weights_1 - weights
        change_2 = weights_2 - weights_1
        np.testing.assert_array_almost_equal(change_1, change_2)

    def test_large_batch_update(self):
        """Test with large weight matrix (typical neural network layer)."""
        optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9)
        weights = np.random.randn(1000, 500)
        gradients = np.random.randn(1000, 500)

        updated = optimizer.update(weights, gradients, "large_layer")

        assert updated.shape == weights.shape
        assert not np.array_equal(updated, weights)

    def test_convergence_simulation(self):
        """Simulate convergence to minimum."""
        optimizer = SGDMomentum(learning_rate=0.1, momentum=0.9)

        # Simple 1D optimization: minimize f(x) = x^2
        # Gradient: df/dx = 2x
        x = np.array([10.0])  # Start far from minimum (x=0)

        for _ in range(100):  # ← Zmiana: 50 → 100
            gradient = 2 * x  # Gradient of x^2
            x = optimizer.update(x, gradient, "param")

        # Should converge close to 0
        assert np.abs(x[0]) < 0.1


class TestSGDMomentumIntegration:
    """Integration tests for SGDMomentum with other components."""

    def test_with_weights_and_biases(self):
        """Test updating both weights and biases separately."""
        optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9)

        # Layer weights and biases
        weights = np.random.randn(10, 5)
        biases = np.random.randn(5)

        # Gradients
        grad_weights = np.random.randn(10, 5)
        grad_biases = np.random.randn(5)

        # Update both
        new_weights = optimizer.update(weights, grad_weights, "layer1_W")
        new_biases = optimizer.update(biases, grad_biases, "layer1_b")

        # Both should be updated
        assert not np.array_equal(new_weights, weights)
        assert not np.array_equal(new_biases, biases)

        # Should have separate velocities
        assert optimizer.get_velocity("layer1_W") is not None
        assert optimizer.get_velocity("layer1_b") is not None

    def test_multiple_epochs(self):
        """Test optimizer across multiple training epochs."""
        optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9)
        weights = np.array([1.0, 2.0, 3.0])

        # Simulate 3 epochs with different gradients
        gradients_per_epoch = [
            np.array([0.5, 0.5, 0.5]),
            np.array([0.3, 0.3, 0.3]),
            np.array([0.1, 0.1, 0.1]),
        ]

        for epoch_grads in gradients_per_epoch:
            weights = optimizer.update(weights, epoch_grads, "layer")

        # Weights should have changed
        assert not np.array_equal(weights, np.array([1.0, 2.0, 3.0]))

        # Velocity should exist
        assert optimizer.get_velocity("layer") is not None
