import numpy as np
import pytest

from src.data.scalers import MinMaxScaler, StandardScaler


class TestStandardScaler:
    """Test suite for StandardScaler."""

    def test_init(self):
        """Test StandardScaler initialization."""
        scaler = StandardScaler()

        assert scaler.mean_ is None
        assert scaler.std_ is None
        assert scaler._fitted is False

    def test_fit(self):
        """Test fitting StandardScaler."""
        x = np.array([[1, 2], [3, 4], [5, 6]])

        scaler = StandardScaler()
        result = scaler.fit(x)

        assert result is scaler
        assert scaler._fitted is True
        assert scaler.mean_ is not None
        assert scaler.std_ is not None
        np.testing.assert_array_almost_equal(scaler.mean_, [3.0, 4.0])

    def test_transform(self):
        """Test transforming data with StandardScaler."""
        x = np.array([[1, 2], [3, 4], [5, 6]])

        scaler = StandardScaler()
        scaler.fit(x)
        x_scaled = scaler.transform(x)

        assert x_scaled.shape == x.shape
        np.testing.assert_array_almost_equal(np.mean(x_scaled, axis=0), [0.0, 0.0])
        np.testing.assert_array_almost_equal(np.std(x_scaled, axis=0), [1.0, 1.0])

    def test_fit_transform(self):
        """Test fit_transform method."""
        x = np.array([[1, 2], [3, 4], [5, 6]])

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        assert scaler._fitted is True
        assert x_scaled.shape == x.shape
        np.testing.assert_array_almost_equal(np.mean(x_scaled, axis=0), [0.0, 0.0])

    def test_transform_before_fit_raises_error(self):
        """Test that transform before fit raises ValueError."""
        scaler = StandardScaler()
        x = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="must be fitted before transform"):
            scaler.transform(x)

    def test_fit_empty_array_raises_error(self):
        """Test that fitting on empty array raises ValueError."""
        scaler = StandardScaler()
        x = np.array([])

        with pytest.raises(ValueError, match="Cannot fit on empty array"):
            scaler.fit(x)

    def test_transform_empty_array(self):
        """Test transforming empty array after fit."""
        x_train = np.array([[1, 2], [3, 4]])
        x_test = np.array([])

        scaler = StandardScaler()
        scaler.fit(x_train)
        result = scaler.transform(x_test)

        assert result.size == 0

    def test_constant_feature(self):
        """Test StandardScaler with constant feature."""
        x = np.array([[1, 5], [1, 10], [1, 15]])

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        assert x_scaled[0, 0] == 0.0
        assert x_scaled[1, 0] == 0.0
        assert x_scaled[2, 0] == 0.0

    def test_single_sample(self):
        """Test StandardScaler with single sample."""
        x = np.array([[1, 2]])

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        assert x_scaled.shape == (1, 2)

    def test_preserves_data_leakage_prevention(self):
        """Test that train and test use same parameters."""
        x_train = np.array([[1, 2], [3, 4], [5, 6]])
        x_test = np.array([[7, 8], [9, 10]])

        scaler = StandardScaler()
        scaler.fit(x_train)

        train_mean = scaler.mean_.copy()
        train_std = scaler.std_.copy()

        x_test_scaled = scaler.transform(x_test)

        np.testing.assert_array_equal(scaler.mean_, train_mean)
        np.testing.assert_array_equal(scaler.std_, train_std)
        assert x_test_scaled.shape == x_test.shape


class TestMinMaxScaler:
    """Test suite for MinMaxScaler."""

    def test_init(self):
        """Test MinMaxScaler initialization."""
        scaler = MinMaxScaler()

        assert scaler.min_ is None
        assert scaler.max_ is None
        assert scaler._fitted is False

    def test_fit(self):
        """Test fitting MinMaxScaler."""
        x = np.array([[1, 2], [3, 4], [5, 6]])

        scaler = MinMaxScaler()
        result = scaler.fit(x)

        assert result is scaler
        assert scaler._fitted is True
        assert scaler.min_ is not None
        assert scaler.max_ is not None
        np.testing.assert_array_equal(scaler.min_, [1, 2])
        np.testing.assert_array_equal(scaler.max_, [5, 6])

    def test_transform(self):
        """Test transforming data with MinMaxScaler."""
        x = np.array([[1, 2], [3, 4], [5, 6]])

        scaler = MinMaxScaler()
        scaler.fit(x)
        x_scaled = scaler.transform(x)

        assert x_scaled.shape == x.shape
        np.testing.assert_array_almost_equal(x_scaled[0], [0.0, 0.0])
        np.testing.assert_array_almost_equal(x_scaled[-1], [1.0, 1.0])
        assert np.all(x_scaled >= 0.0)
        assert np.all(x_scaled <= 1.0)

    def test_fit_transform(self):
        """Test fit_transform method."""
        x = np.array([[1, 2], [3, 4], [5, 6]])

        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x)

        assert scaler._fitted is True
        assert x_scaled.shape == x.shape
        assert np.all(x_scaled >= 0.0)
        assert np.all(x_scaled <= 1.0)

    def test_transform_before_fit_raises_error(self):
        """Test that transform before fit raises ValueError."""
        scaler = MinMaxScaler()
        x = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="must be fitted before transform"):
            scaler.transform(x)

    def test_fit_empty_array_raises_error(self):
        """Test that fitting on empty array raises ValueError."""
        scaler = MinMaxScaler()
        x = np.array([])

        with pytest.raises(ValueError, match="Cannot fit on empty array"):
            scaler.fit(x)

    def test_transform_empty_array(self):
        """Test transforming empty array after fit."""
        x_train = np.array([[1, 2], [3, 4]])
        x_test = np.array([])

        scaler = MinMaxScaler()
        scaler.fit(x_train)
        result = scaler.transform(x_test)

        assert result.size == 0

    def test_constant_feature(self):
        """Test MinMaxScaler with constant feature."""
        x = np.array([[1, 5], [1, 10], [1, 15]])

        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x)

        assert x_scaled[0, 0] == 0.0
        assert x_scaled[1, 0] == 0.0
        assert x_scaled[2, 0] == 0.0

    def test_single_sample(self):
        """Test MinMaxScaler with single sample."""
        x = np.array([[1, 2]])

        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x)

        assert x_scaled.shape == (1, 2)

    def test_preserves_data_leakage_prevention(self):
        """Test that train and test use same parameters."""
        x_train = np.array([[1, 2], [3, 4], [5, 6]])
        x_test = np.array([[7, 8], [9, 10]])

        scaler = MinMaxScaler()
        scaler.fit(x_train)

        train_min = scaler.min_.copy()
        train_max = scaler.max_.copy()

        x_test_scaled = scaler.transform(x_test)

        np.testing.assert_array_equal(scaler.min_, train_min)
        np.testing.assert_array_equal(scaler.max_, train_max)
        assert x_test_scaled.shape == x_test.shape

    def test_out_of_range_test_data(self):
        """Test MinMaxScaler with test data outside training range."""
        x_train = np.array([[1, 2], [3, 4], [5, 6]])
        x_test = np.array([[0, 1], [10, 20]])

        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_test_scaled = scaler.transform(x_test)

        assert x_test_scaled[0, 0] < 0.0
        assert x_test_scaled[1, 0] > 1.0


class TestPreprocessorsComparison:
    """Comparison tests between preprocessors."""

    def test_both_scalers_on_same_data(self):
        """Test that both scalers work on same data."""
        x = np.array([[1, 2], [3, 4], [5, 6]])

        standard_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()

        x_standard = standard_scaler.fit_transform(x)
        x_minmax = minmax_scaler.fit_transform(x)

        assert x_standard.shape == x_minmax.shape
        assert not np.array_equal(x_standard, x_minmax)

    @pytest.mark.parametrize("shape", [(10, 2), (50, 4), (100, 5)])
    def test_scalers_with_various_shapes(self, shape: tuple):
        """Test scalers with different data shapes."""
        x = np.random.randn(*shape)

        standard_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()

        x_standard = standard_scaler.fit_transform(x)
        x_minmax = minmax_scaler.fit_transform(x)

        assert x_standard.shape == shape
        assert x_minmax.shape == shape
