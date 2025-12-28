"""Tests for MetricsCalculator with simplified API."""

import numpy as np
import pytest

from src.evaluation.metrics import MetricsCalculator


@pytest.fixture
def perfect_predictions():
    """Perfect predictions (100% accuracy)."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    return y_true, y_pred


@pytest.fixture
def imperfect_predictions():
    """Imperfect predictions with known metrics."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 2, 0])
    return y_true, y_pred


class TestMetricsCalculatorInitialization:
    """Test MetricsCalculator initialization."""

    def test_init_without_class_names(self):
        """Test initialization without class names."""
        calc = MetricsCalculator()
        assert calc.class_names is None

    def test_init_with_class_names(self):
        """Test initialization with class names."""
        class_names = ["setosa", "versicolor", "virginica"]
        calc = MetricsCalculator(class_names=class_names)
        assert calc.class_names == class_names

    def test_repr(self):
        """Test string representation."""
        calc = MetricsCalculator(class_names=["A", "B"])
        assert "MetricsCalculator" in repr(calc)
        assert "['A', 'B']" in repr(calc)


class TestAccuracy:
    """Test accuracy metric."""

    def test_perfect_accuracy(self, perfect_predictions):
        """Test accuracy with perfect predictions."""
        y_true, y_pred = perfect_predictions
        calc = MetricsCalculator()

        accuracy = calc.accuracy(y_true, y_pred)
        assert accuracy == 1.0

    def test_zero_accuracy(self):
        """Test accuracy with completely wrong predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        calc = MetricsCalculator()

        accuracy = calc.accuracy(y_true, y_pred)
        assert accuracy == 0.0

    def test_partial_accuracy(self, imperfect_predictions):
        """Test accuracy with partial correct predictions."""
        y_true, y_pred = imperfect_predictions
        calc = MetricsCalculator()

        accuracy = calc.accuracy(y_true, y_pred)
        expected = 6 / 9  # 6 correct out of 9
        assert np.isclose(accuracy, expected)


class TestConfusionMatrix:
    """Test confusion matrix."""

    def test_perfect_predictions_matrix(self, perfect_predictions):
        """Test confusion matrix with perfect predictions."""
        y_true, y_pred = perfect_predictions
        calc = MetricsCalculator()

        cm = calc.confusion_matrix(y_true, y_pred)
        expected = np.array(
            [
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 2],
            ]
        )

        assert np.array_equal(cm, expected)

    def test_confusion_matrix_auto_detection(self):
        """Test that confusion matrix auto-detects number of classes."""
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        calc = MetricsCalculator()

        cm = calc.confusion_matrix(y_true, y_pred)
        assert cm.shape == (4, 4)

    def test_confusion_matrix_shape(self, imperfect_predictions):
        """Test confusion matrix shape."""
        y_true, y_pred = imperfect_predictions
        calc = MetricsCalculator()

        cm = calc.confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)

    def test_confusion_matrix_values(self, imperfect_predictions):
        """Test confusion matrix values."""
        y_true, y_pred = imperfect_predictions
        calc = MetricsCalculator()

        cm = calc.confusion_matrix(y_true, y_pred)

        # Check diagonal (correct predictions)
        assert cm[0, 0] == 2  # Class 0 correct
        assert cm[1, 1] == 2  # Class 1 correct
        assert cm[2, 2] == 2  # Class 2 correct


class TestPrecision:
    """Test precision metric."""

    def test_perfect_precision(self, perfect_predictions):
        """Test precision with perfect predictions."""
        y_true, y_pred = perfect_predictions
        calc = MetricsCalculator()

        precision = calc.precision(y_true, y_pred, average="macro")
        assert precision == 1.0

    def test_precision_per_class(self, imperfect_predictions):
        """Test per-class precision."""
        y_true, y_pred = imperfect_predictions
        calc = MetricsCalculator()

        precision = calc.precision(y_true, y_pred, average=None)
        assert isinstance(precision, np.ndarray)
        assert len(precision) == 3

    def test_precision_weighted(self, imperfect_predictions):
        """Test weighted average precision."""
        y_true, y_pred = imperfect_predictions
        calc = MetricsCalculator()

        precision = calc.precision(y_true, y_pred, average="weighted")
        assert isinstance(precision, float)
        assert 0 <= precision <= 1


class TestRecall:
    """Test recall metric."""

    def test_perfect_recall(self, perfect_predictions):
        """Test recall with perfect predictions."""
        y_true, y_pred = perfect_predictions
        calc = MetricsCalculator()

        recall = calc.recall(y_true, y_pred, average="macro")
        assert recall == 1.0

    def test_recall_per_class(self, imperfect_predictions):
        """Test per-class recall."""
        y_true, y_pred = imperfect_predictions
        calc = MetricsCalculator()

        recall = calc.recall(y_true, y_pred, average=None)
        assert isinstance(recall, np.ndarray)
        assert len(recall) == 3

    def test_recall_weighted(self, imperfect_predictions):
        """Test weighted average recall."""
        y_true, y_pred = imperfect_predictions
        calc = MetricsCalculator()

        recall = calc.recall(y_true, y_pred, average="weighted")
        assert isinstance(recall, float)
        assert 0 <= recall <= 1


class TestF1Score:
    """Test F1 score metric."""

    def test_perfect_f1(self, perfect_predictions):
        """Test F1 score with perfect predictions."""
        y_true, y_pred = perfect_predictions
        calc = MetricsCalculator()

        f1 = calc.f1_score(y_true, y_pred, average="macro")
        assert f1 == 1.0

    def test_f1_per_class(self, imperfect_predictions):
        """Test per-class F1 score."""
        y_true, y_pred = imperfect_predictions
        calc = MetricsCalculator()

        f1 = calc.f1_score(y_true, y_pred, average=None)
        assert isinstance(f1, np.ndarray)
        assert len(f1) == 3

    def test_f1_weighted(self, imperfect_predictions):
        """Test weighted average F1 score."""
        y_true, y_pred = imperfect_predictions
        calc = MetricsCalculator()

        f1 = calc.f1_score(y_true, y_pred, average="weighted")
        assert isinstance(f1, float)
        assert 0 <= f1 <= 1


class TestComputeAll:
    """Test compute_all method for ExperimentResult compatibility."""

    def test_compute_all_returns_dict(self, perfect_predictions):
        """Test that compute_all returns dictionary."""
        y_true, y_pred = perfect_predictions
        calc = MetricsCalculator()

        result = calc.compute_all(y_true, y_pred)
        assert isinstance(result, dict)

    def test_compute_all_contains_required_keys(self, perfect_predictions):
        """Test that result contains all required keys."""
        y_true, y_pred = perfect_predictions
        calc = MetricsCalculator()

        result = calc.compute_all(y_true, y_pred)

        required_keys = ["accuracy", "precision", "recall", "f1", "confusion_matrix"]
        for key in required_keys:
            assert key in result

    def test_compute_all_metrics_are_floats(self, perfect_predictions):
        """Test that metrics are float values."""
        y_true, y_pred = perfect_predictions
        calc = MetricsCalculator()

        result = calc.compute_all(y_true, y_pred)

        assert isinstance(result["accuracy"], float)
        assert isinstance(result["precision"], float)
        assert isinstance(result["recall"], float)
        assert isinstance(result["f1"], float)

    def test_compute_all_confusion_matrix_is_list(self, perfect_predictions):
        """Test that confusion matrix is converted to list."""
        y_true, y_pred = perfect_predictions
        calc = MetricsCalculator()

        result = calc.compute_all(y_true, y_pred)

        assert isinstance(result["confusion_matrix"], list)
        assert all(isinstance(row, list) for row in result["confusion_matrix"])

    def test_compute_all_metrics_in_range(self, imperfect_predictions):
        """Test that all metrics are in valid range [0, 1]."""
        y_true, y_pred = imperfect_predictions
        calc = MetricsCalculator()

        result = calc.compute_all(y_true, y_pred)

        assert 0 <= result["accuracy"] <= 1
        assert 0 <= result["precision"] <= 1
        assert 0 <= result["recall"] <= 1
        assert 0 <= result["f1"] <= 1


class TestComputeDetailed:
    """Test compute_detailed method for comprehensive reports."""

    def test_compute_detailed_returns_dict(self, perfect_predictions):
        """Test that compute_detailed returns dictionary."""
        y_true, y_pred = perfect_predictions
        calc = MetricsCalculator()

        result = calc.compute_detailed(y_true, y_pred)
        assert isinstance(result, dict)

    def test_compute_detailed_contains_sections(self, perfect_predictions):
        """Test that detailed report contains all sections."""
        y_true, y_pred = perfect_predictions
        calc = MetricsCalculator()

        result = calc.compute_detailed(y_true, y_pred)

        assert "confusion_matrix" in result
        assert "accuracy" in result
        assert "macro_avg" in result
        assert "weighted_avg" in result
        assert "per_class" in result

    def test_compute_detailed_with_class_names(self, perfect_predictions):
        """Test detailed report with custom class names."""
        y_true, y_pred = perfect_predictions
        calc = MetricsCalculator(class_names=["A", "B", "C"])

        result = calc.compute_detailed(y_true, y_pred)

        assert "A" in result["per_class"]
        assert "B" in result["per_class"]
        assert "C" in result["per_class"]

    def test_compute_detailed_auto_generates_class_names(self, perfect_predictions):
        """Test that class names are auto-generated if not provided."""
        y_true, y_pred = perfect_predictions
        calc = MetricsCalculator()

        result = calc.compute_detailed(y_true, y_pred)

        assert "Class 0" in result["per_class"]
        assert "Class 1" in result["per_class"]
        assert "Class 2" in result["per_class"]
