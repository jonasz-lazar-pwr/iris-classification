import numpy as np
import pytest

from src.data.splitter import DataSplitter


class TestDataSplitter:
    """Test suite for DataSplitter."""

    def test_split_basic(self):
        """Test basic data splitting."""
        X = np.arange(100).reshape(100, 1)
        y = np.array([0] * 50 + [1] * 50)

        splitter = DataSplitter()
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
            X=X, y=y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )

        assert X_train.shape[0] == 70
        assert X_val.shape[0] == 15
        assert X_test.shape[0] == 15
        assert y_train.shape[0] == 70
        assert y_val.shape[0] == 15
        assert y_test.shape[0] == 15

    def test_split_with_stratification(self):
        """Test that stratification preserves class distribution."""
        X = np.arange(150).reshape(150, 1)
        y = np.array([0] * 50 + [1] * 50 + [2] * 50)

        splitter = DataSplitter()
        _, _, _, y_train, y_val, y_test = splitter.split(
            X=X, y=y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, stratify=True
        )

        train_counts = np.bincount(y_train)
        val_counts = np.bincount(y_val)
        test_counts = np.bincount(y_test)

        assert len(train_counts) == 3
        assert len(val_counts) == 3
        assert len(test_counts) == 3

        assert np.all(train_counts >= 30)
        assert np.all(val_counts >= 5)
        assert np.all(test_counts >= 5)

    def test_split_without_stratification(self):
        """Test splitting without stratification."""
        X = np.arange(100).reshape(100, 1)
        y = np.array([0] * 50 + [1] * 50)

        splitter = DataSplitter()
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
            X=X, y=y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, stratify=False
        )

        assert len(X_train) + len(X_val) + len(X_test) == 100
        assert len(y_train) + len(y_val) + len(y_test) == 100

    def test_split_reproducibility(self):
        """Test that same random_state produces same splits."""
        X = np.arange(100).reshape(100, 1)
        y = np.random.randint(0, 3, 100)

        splitter = DataSplitter()

        X_train1, X_val1, X_test1, y_train1, y_val1, y_test1 = splitter.split(
            X=X, y=y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
        )

        X_train2, X_val2, X_test2, y_train2, y_val2, y_test2 = splitter.split(
            X=X, y=y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
        )

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(X_val1, X_val2)
        np.testing.assert_array_equal(y_val1, y_val2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_test1, y_test2)

    def test_split_different_random_states(self):
        """Test that different random_state produces different splits."""
        X = np.arange(100).reshape(100, 1)
        y = np.random.randint(0, 3, 100)

        splitter = DataSplitter()

        X_train1, _, _, _, _, _ = splitter.split(
            X=X, y=y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
        )

        X_train2, _, _, _, _, _ = splitter.split(
            X=X, y=y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=123
        )

        assert not np.array_equal(X_train1, X_train2)

    def test_split_with_zero_val_ratio(self):
        """Test splitting with zero validation ratio."""
        X = np.arange(100).reshape(100, 1)
        y = np.array([0] * 50 + [1] * 50)

        splitter = DataSplitter()
        X_train, X_val, X_test, _, y_val, _ = splitter.split(
            X=X, y=y, train_ratio=0.8, val_ratio=0.0, test_ratio=0.2
        )

        assert X_train.shape[0] == 80
        assert X_val.shape[0] == 0
        assert X_test.shape[0] == 20
        assert y_val.shape[0] == 0

    def test_split_multidimensional_features(self):
        """Test splitting with multidimensional features."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)

        splitter = DataSplitter()
        X_train, X_val, X_test, _, _, _ = splitter.split(
            X=X, y=y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )

        assert X_train.shape[1] == 5
        assert X_val.shape[1] == 5
        assert X_test.shape[1] == 5

    @pytest.mark.parametrize(
        "train_ratio,val_ratio,test_ratio",
        [
            (0.6, 0.2, 0.2),
            (0.7, 0.15, 0.15),
            (0.8, 0.1, 0.1),
            (0.5, 0.25, 0.25),
        ],
    )
    def test_split_various_ratios(self, train_ratio: float, val_ratio: float, test_ratio: float):
        """Test splitting with various ratio combinations."""
        X = np.arange(100).reshape(100, 1)
        y = np.random.randint(0, 2, 100)

        splitter = DataSplitter()
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
            X=X, y=y, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
        )

        total_samples = len(X_train) + len(X_val) + len(X_test)
        assert total_samples == 100

        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)


class TestDataSplitterValidation:
    """Test suite for DataSplitter input validation."""

    def test_validate_mismatched_lengths(self):
        """Test validation fails when X and y have different lengths."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(50)

        splitter = DataSplitter()

        with pytest.raises(ValueError, match="X and y must have same length"):
            splitter.split(X=X, y=y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    def test_validate_empty_arrays(self):
        """Test validation fails on empty arrays."""
        X = np.array([])
        y = np.array([])

        splitter = DataSplitter()

        with pytest.raises(ValueError, match="Cannot split empty arrays"):
            splitter.split(X=X, y=y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    def test_validate_train_ratio_zero(self):
        """Test validation fails when train_ratio is zero."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        splitter = DataSplitter()

        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            splitter.split(X=X, y=y, train_ratio=0.0, val_ratio=0.2, test_ratio=0.8)

    def test_validate_train_ratio_one(self):
        """Test validation fails when train_ratio is one."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        splitter = DataSplitter()

        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            splitter.split(X=X, y=y, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

    def test_validate_negative_val_ratio(self):
        """Test validation fails when val_ratio is negative."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        splitter = DataSplitter()

        with pytest.raises(ValueError, match="val_ratio must be between 0 and 1"):
            splitter.split(X=X, y=y, train_ratio=0.7, val_ratio=-0.1, test_ratio=0.4)

    def test_validate_test_ratio_zero(self):
        """Test validation fails when test_ratio is zero."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        splitter = DataSplitter()

        with pytest.raises(ValueError, match="test_ratio must be between 0 and 1"):
            splitter.split(X=X, y=y, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)

    def test_validate_ratios_sum_not_one(self):
        """Test validation fails when ratios don't sum to 1.0."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        splitter = DataSplitter()

        with pytest.raises(ValueError, match=r"Ratios must sum to 1\.0"):
            splitter.split(X=X, y=y, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_validate_ratios_sum_with_float_precision(self):
        """Test validation handles floating point precision."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        splitter = DataSplitter()

        X_train, X_val, X_test, _, _, _ = splitter.split(
            X=X, y=y, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, stratify=False
        )

        assert len(X_train) + len(X_val) + len(X_test) == 100

    def test_validate_very_small_test_set_warning(self, caplog):
        """Test warning is logged for very small test sets."""
        X = np.arange(10).reshape(10, 1)
        y = np.arange(10)

        splitter = DataSplitter()

        splitter.split(X=X, y=y, train_ratio=0.95, val_ratio=0.0, test_ratio=0.05, stratify=False)

        assert "very few samples" in caplog.text.lower()


class TestDataSplitterStratification:
    """Test suite for stratification functionality."""

    def test_stratification_preserves_class_proportions(self):
        """Test that stratification maintains class proportions across splits."""
        X = np.arange(150).reshape(150, 1)
        y = np.array([0] * 50 + [1] * 50 + [2] * 50)

        splitter = DataSplitter()
        _, _, _, y_train, y_val, y_test = splitter.split(
            X=X, y=y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, stratify=True
        )

        original_proportions = np.bincount(y) / len(y)
        train_proportions = np.bincount(y_train) / len(y_train)
        val_proportions = np.bincount(y_val) / len(y_val)
        test_proportions = np.bincount(y_test) / len(y_test)

        np.testing.assert_allclose(train_proportions, original_proportions, atol=0.05)
        np.testing.assert_allclose(val_proportions, original_proportions, atol=0.1)
        np.testing.assert_allclose(test_proportions, original_proportions, atol=0.1)

    def test_stratification_with_imbalanced_classes(self):
        """Test stratification with imbalanced class distribution."""
        X = np.arange(100).reshape(100, 1)
        y = np.array([0] * 80 + [1] * 20)

        splitter = DataSplitter()
        _, _, _, y_train, y_val, y_test = splitter.split(
            X=X, y=y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, stratify=True
        )

        train_class_0_ratio = np.sum(y_train == 0) / len(y_train)
        val_class_0_ratio = np.sum(y_val == 0) / len(y_val)
        test_class_0_ratio = np.sum(y_test == 0) / len(y_test)

        assert abs(train_class_0_ratio - 0.8) < 0.1
        assert abs(val_class_0_ratio - 0.8) < 0.15
        assert abs(test_class_0_ratio - 0.8) < 0.15

    def test_no_stratification_allows_imbalance(self):
        """Test that without stratification, splits can be imbalanced."""
        X = np.arange(100).reshape(100, 1)
        y = np.array([0] * 50 + [1] * 50)

        splitter = DataSplitter()

        results = []
        for seed in range(10):
            _, _, _, y_train, _, _ = splitter.split(
                X=X,
                y=y,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                stratify=False,
                random_state=seed,
            )
            results.append(np.sum(y_train == 0))

        assert len(set(results)) > 1

    def test_stratification_logs_distribution(self, caplog):
        """Test that class distribution is logged when stratify=True."""
        X = np.arange(150).reshape(150, 1)
        y = np.array([0] * 50 + [1] * 50 + [2] * 50)

        splitter = DataSplitter()
        splitter.split(X=X, y=y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, stratify=True)

        assert "Class distribution per split" in caplog.text


class TestDataSplitterIntegration:
    """Integration tests for DataSplitter."""

    def test_split_iris_like_dataset(self):
        """Test splitting Iris-like dataset."""
        X = np.random.randn(150, 4)
        y = np.array([0] * 50 + [1] * 50 + [2] * 50)

        splitter = DataSplitter()
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
            X=X, y=y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, stratify=True
        )

        assert X_train.shape == (105, 4)
        assert X_val.shape[0] + X_test.shape[0] == 45
        assert len(y_train) == 105
        assert len(y_val) + len(y_test) == 45

        for cls in [0, 1, 2]:
            assert np.sum(y_train == cls) >= 30
            assert np.sum(y_val == cls) >= 5
            assert np.sum(y_test == cls) >= 5

    def test_split_preserves_no_data_leakage(self):
        """Test that splits don't overlap (no data leakage)."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        splitter = DataSplitter()
        X_train, X_val, X_test, _, _, _ = splitter.split(
            X=X, y=y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, stratify=False
        )

        train_indices = set(X_train.flatten())
        val_indices = set(X_val.flatten())
        test_indices = set(X_test.flatten())

        assert len(train_indices & val_indices) == 0
        assert len(train_indices & test_indices) == 0
        assert len(val_indices & test_indices) == 0

    def test_split_returns_numpy_arrays(self):
        """Test that split returns numpy arrays."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        splitter = DataSplitter()
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
            X=X, y=y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, stratify=False
        )

        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_val, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_val, np.ndarray)
        assert isinstance(y_test, np.ndarray)
