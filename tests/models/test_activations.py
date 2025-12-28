"""Unit tests for activation functions."""

import numpy as np
import pytest

from src.models.activations import ReLU, Softmax, Tanh
from src.models.base import IActivation


class TestReLU:
    """Test suite for ReLU activation."""

    @pytest.fixture
    def relu(self) -> ReLU:
        """Create ReLU instance."""
        return ReLU()

    def test_implements_interface(self, relu: ReLU):
        """Test that ReLU implements IActivation."""
        assert isinstance(relu, IActivation)

    def test_forward_positive_values(self, relu: ReLU):
        """Test ReLU forward with positive values."""
        x = np.array([1.0, 2.0, 3.0])
        result = relu.forward(x)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_forward_negative_values(self, relu: ReLU):
        """Test ReLU forward with negative values."""
        x = np.array([-1.0, -2.0, -3.0])
        result = relu.forward(x)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_forward_mixed_values(self, relu: ReLU):
        """Test ReLU forward with mixed positive and negative values."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = relu.forward(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(result, expected)

    def test_forward_zero(self, relu: ReLU):
        """Test ReLU forward with zero."""
        x = np.array([0.0])
        result = relu.forward(x)
        expected = np.array([0.0])
        np.testing.assert_array_equal(result, expected)

    def test_forward_2d_array(self, relu: ReLU):
        """Test ReLU forward with 2D array."""
        x = np.array([[-1.0, 2.0], [3.0, -4.0]])
        result = relu.forward(x)
        expected = np.array([[0.0, 2.0], [3.0, 0.0]])
        np.testing.assert_array_equal(result, expected)

    def test_backward_positive_values(self, relu: ReLU):
        """Test ReLU backward with positive values."""
        x = np.array([1.0, 2.0, 3.0])
        result = relu.backward(x)
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_backward_negative_values(self, relu: ReLU):
        """Test ReLU backward with negative values."""
        x = np.array([-1.0, -2.0, -3.0])
        result = relu.backward(x)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_backward_mixed_values(self, relu: ReLU):
        """Test ReLU backward with mixed values."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = relu.backward(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_backward_2d_array(self, relu: ReLU):
        """Test ReLU backward with 2D array."""
        x = np.array([[-1.0, 2.0], [3.0, -4.0]])
        result = relu.backward(x)
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_array_equal(result, expected)

    def test_repr(self, relu: ReLU):
        """Test string representation."""
        assert repr(relu) == "ReLU()"


class TestTanh:
    """Test suite for Tanh activation."""

    @pytest.fixture
    def tanh(self) -> Tanh:
        """Create Tanh instance."""
        return Tanh()

    def test_implements_interface(self, tanh: Tanh):
        """Test that Tanh implements IActivation."""
        assert isinstance(tanh, IActivation)

    def test_forward_zero(self, tanh: Tanh):
        """Test Tanh forward with zero."""
        x = np.array([0.0])
        result = tanh.forward(x)
        expected = np.array([0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_forward_positive_values(self, tanh: Tanh):
        """Test Tanh forward with positive values."""
        x = np.array([1.0, 2.0])
        result = tanh.forward(x)
        expected = np.tanh(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_forward_negative_values(self, tanh: Tanh):
        """Test Tanh forward with negative values."""
        x = np.array([-1.0, -2.0])
        result = tanh.forward(x)
        expected = np.tanh(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_forward_range(self, tanh: Tanh):
        """Test Tanh output is in range [-1, 1]."""
        x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        result = tanh.forward(x)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_forward_saturation(self, tanh: Tanh):
        """Test Tanh saturation at extreme values."""
        x = np.array([10.0, -10.0])
        result = tanh.forward(x)
        np.testing.assert_array_almost_equal(result, [1.0, -1.0], decimal=5)

    def test_forward_2d_array(self, tanh: Tanh):
        """Test Tanh forward with 2D array."""
        x = np.array([[0.0, 1.0], [-1.0, 2.0]])
        result = tanh.forward(x)
        expected = np.tanh(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_backward_zero(self, tanh: Tanh):
        """Test Tanh backward at zero."""
        x = np.array([0.0])
        result = tanh.backward(x)
        expected = np.array([1.0])  # derivative at 0 is 1
        np.testing.assert_array_almost_equal(result, expected)

    def test_backward_values(self, tanh: Tanh):
        """Test Tanh backward gradient computation."""
        x = np.array([0.5, 1.0, 2.0])
        result = tanh.backward(x)
        tanh_x = np.tanh(x)
        expected = 1 - tanh_x**2
        np.testing.assert_array_almost_equal(result, expected)

    def test_backward_range(self, tanh: Tanh):
        """Test Tanh derivative is in range (0, 1]."""
        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        result = tanh.backward(x)
        assert np.all(result > 0.0)
        assert np.all(result <= 1.0)

    def test_backward_2d_array(self, tanh: Tanh):
        """Test Tanh backward with 2D array."""
        x = np.array([[0.0, 1.0], [-1.0, 2.0]])
        result = tanh.backward(x)
        tanh_x = np.tanh(x)
        expected = 1 - tanh_x**2
        np.testing.assert_array_almost_equal(result, expected)

    def test_repr(self, tanh: Tanh):
        """Test string representation."""
        assert repr(tanh) == "Tanh()"


class TestSoftmax:
    """Test suite for Softmax activation."""

    @pytest.fixture
    def softmax(self) -> Softmax:
        """Create Softmax instance."""
        return Softmax()

    def test_implements_interface(self, softmax: Softmax):
        """Test that Softmax implements IActivation."""
        assert isinstance(softmax, IActivation)

    def test_forward_uniform_input(self, softmax: Softmax):
        """Test Softmax forward with uniform input."""
        x = np.array([1.0, 1.0, 1.0])
        result = softmax.forward(x)
        expected = np.array([1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_array_almost_equal(result, expected)

    def test_forward_sums_to_one(self, softmax: Softmax):
        """Test Softmax output sums to 1."""
        x = np.array([1.0, 2.0, 3.0])
        result = softmax.forward(x)
        assert np.isclose(np.sum(result), 1.0)

    def test_forward_different_scales(self, softmax: Softmax):
        """Test Softmax with different input scales."""
        x = np.array([0.1, 0.2, 0.7])
        result = softmax.forward(x)
        assert np.isclose(np.sum(result), 1.0)
        assert result[2] > result[1] > result[0]  # order preserved

    def test_forward_negative_values(self, softmax: Softmax):
        """Test Softmax with negative values."""
        x = np.array([-1.0, 0.0, 1.0])
        result = softmax.forward(x)
        assert np.isclose(np.sum(result), 1.0)
        assert np.all(result > 0)

    def test_forward_numerical_stability(self, softmax: Softmax):
        """Test Softmax numerical stability with large values."""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax.forward(x)
        assert np.isclose(np.sum(result), 1.0)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_forward_2d_batch(self, softmax: Softmax):
        """Test Softmax with batch input (2D array)."""
        x = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        result = softmax.forward(x)
        # Each row should sum to 1
        row_sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0])

    def test_forward_output_range(self, softmax: Softmax):
        """Test Softmax output is in range [0, 1]."""
        x = np.array([-10.0, 0.0, 10.0])
        result = softmax.forward(x)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_backward_uniform_input(self, softmax: Softmax):
        """Test Softmax backward with uniform input."""
        x = np.array([1.0, 1.0, 1.0])
        result = softmax.backward(x)
        # For uniform input, softmax = [1/3, 1/3, 1/3]
        # Derivative = softmax * (1 - softmax) = 1/3 * 2/3 = 2/9
        expected = np.array([2 / 9, 2 / 9, 2 / 9])
        np.testing.assert_array_almost_equal(result, expected)

    def test_backward_values(self, softmax: Softmax):
        """Test Softmax backward gradient computation."""
        x = np.array([1.0, 2.0, 3.0])
        result = softmax.backward(x)
        softmax_out = softmax.forward(x)
        expected = softmax_out * (1 - softmax_out)
        np.testing.assert_array_almost_equal(result, expected)

    def test_backward_2d_batch(self, softmax: Softmax):
        """Test Softmax backward with batch input."""
        x = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        result = softmax.backward(x)
        softmax_out = softmax.forward(x)
        expected = softmax_out * (1 - softmax_out)
        np.testing.assert_array_almost_equal(result, expected)

    def test_repr(self, softmax: Softmax):
        """Test string representation."""
        assert repr(softmax) == "Softmax()"


class TestActivationComparison:
    """Comparative tests between activations."""

    def test_relu_vs_tanh_on_positive(self):
        """Test ReLU preserves positive values better than Tanh."""
        x = np.array([5.0, 10.0, 15.0])
        relu = ReLU()
        tanh = Tanh()

        relu_out = relu.forward(x)
        tanh_out = tanh.forward(x)

        # ReLU preserves values, Tanh saturates to ~1
        assert np.all(relu_out == x)
        assert np.all(tanh_out < 1.0)

    def test_gradient_flow(self):
        """Test gradient flow through different activations."""
        x = np.array([0.5, 1.0, 1.5])
        relu = ReLU()
        tanh = Tanh()

        relu_grad = relu.backward(x)
        tanh_grad = tanh.backward(x)

        # ReLU has constant gradient for positive values
        assert np.all(relu_grad == 1.0)
        # Tanh gradient decreases with larger inputs
        assert np.all(tanh_grad < 1.0)
        assert tanh_grad[0] > tanh_grad[1] > tanh_grad[2]


class TestActivationEdgeCases:
    """Test edge cases for all activations."""

    def test_empty_array(self):
        """Test activations with empty array."""
        x = np.array([])
        relu = ReLU()
        tanh = Tanh()
        softmax = Softmax()

        assert relu.forward(x).shape == (0,)
        assert tanh.forward(x).shape == (0,)
        assert softmax.forward(x).shape == (0,)

    def test_single_value(self):
        """Test activations with single value."""
        x = np.array([2.0])
        relu = ReLU()
        tanh = Tanh()
        softmax = Softmax()

        assert relu.forward(x).shape == (1,)
        assert tanh.forward(x).shape == (1,)
        assert softmax.forward(x).shape == (1,)
        assert np.isclose(softmax.forward(x)[0], 1.0)

    def test_large_arrays(self):
        """Test activations with large arrays."""
        x = np.random.randn(1000, 100)
        relu = ReLU()
        tanh = Tanh()
        softmax = Softmax()

        relu_out = relu.forward(x)
        tanh_out = tanh.forward(x)
        softmax_out = softmax.forward(x)

        assert relu_out.shape == x.shape
        assert tanh_out.shape == x.shape
        assert softmax_out.shape == x.shape
