"""Comprehensive tests for ExperimentResult."""

import pytest

from src.experiment.experiment_result import ExperimentResult


@pytest.fixture
def valid_result_data():
    """Valid experiment result data."""
    return {
        "test_accuracy": 0.9500,
        "test_precision": 0.9400,
        "test_recall": 0.9500,
        "test_f1": 0.9450,
        "confusion_matrix": [[10, 0, 0], [0, 9, 1], [0, 1, 9]],
    }


class TestExperimentResultInitialization:
    """Test ExperimentResult initialization."""

    def test_init_with_valid_data(self, valid_result_data):
        """Test initialization with valid data."""
        result = ExperimentResult(**valid_result_data)

        assert result.test_accuracy == 0.9500
        assert result.test_precision == 0.9400
        assert result.test_recall == 0.9500
        assert result.test_f1 == 0.9450
        assert result.confusion_matrix == [[10, 0, 0], [0, 9, 1], [0, 1, 9]]

    def test_init_rounds_to_4_decimals(self):
        """Test that metrics are rounded to 4 decimal places."""
        result = ExperimentResult(
            test_accuracy=0.123456789,
            test_precision=0.987654321,
            test_recall=0.555555555,
            test_f1=0.111111111,
            confusion_matrix=[[5, 0], [0, 5]],
        )

        assert result.test_accuracy == 0.1235
        assert result.test_precision == 0.9877
        assert result.test_recall == 0.5556
        assert result.test_f1 == 0.1111

    def test_init_accepts_integers(self):
        """Test that integer metrics are accepted."""
        result = ExperimentResult(
            test_accuracy=1,
            test_precision=1,
            test_recall=1,
            test_f1=1,
            confusion_matrix=[[10, 0], [0, 10]],
        )

        assert result.test_accuracy == 1.0
        assert result.test_precision == 1.0


class TestExperimentResultValidation:
    """Test metric validation."""

    def test_init_with_negative_accuracy_raises_error(self):
        """Test that negative accuracy raises error."""
        with pytest.raises(ValueError, match="test_accuracy must be in range"):
            ExperimentResult(
                test_accuracy=-0.1,
                test_precision=0.9,
                test_recall=0.9,
                test_f1=0.9,
                confusion_matrix=[[5, 0], [0, 5]],
            )

    def test_init_with_accuracy_greater_than_1_raises_error(self):
        """Test that accuracy > 1.0 raises error."""
        with pytest.raises(ValueError, match="test_accuracy must be in range"):
            ExperimentResult(
                test_accuracy=1.1,
                test_precision=0.9,
                test_recall=0.9,
                test_f1=0.9,
                confusion_matrix=[[5, 0], [0, 5]],
            )

    def test_init_with_non_numeric_metric_raises_error(self):
        """Test that non-numeric metric raises error."""
        with pytest.raises(ValueError, match="test_accuracy must be numeric"):
            ExperimentResult(
                test_accuracy="0.95",
                test_precision=0.9,
                test_recall=0.9,
                test_f1=0.9,
                confusion_matrix=[[5, 0], [0, 5]],
            )

    def test_init_with_invalid_precision_raises_error(self):
        """Test that invalid precision raises error."""
        with pytest.raises(ValueError, match="test_precision must be in range"):
            ExperimentResult(
                test_accuracy=0.9,
                test_precision=1.5,
                test_recall=0.9,
                test_f1=0.9,
                confusion_matrix=[[5, 0], [0, 5]],
            )


class TestExperimentResultConfusionMatrixValidation:
    """Test confusion matrix validation."""

    def test_init_with_non_list_matrix_raises_error(self):
        """Test that non-list matrix raises error."""
        with pytest.raises(ValueError, match="Confusion matrix must be a list"):
            ExperimentResult(
                test_accuracy=0.9,
                test_precision=0.9,
                test_recall=0.9,
                test_f1=0.9,
                confusion_matrix="not a list",
            )

    def test_init_with_empty_matrix_raises_error(self):
        """Test that empty matrix raises error."""
        with pytest.raises(ValueError, match="Confusion matrix cannot be empty"):
            ExperimentResult(
                test_accuracy=0.9,
                test_precision=0.9,
                test_recall=0.9,
                test_f1=0.9,
                confusion_matrix=[],
            )

    def test_init_with_non_square_matrix_raises_error(self):
        """Test that non-square matrix raises error."""
        with pytest.raises(ValueError, match="Confusion matrix must be square"):
            ExperimentResult(
                test_accuracy=0.9,
                test_precision=0.9,
                test_recall=0.9,
                test_f1=0.9,
                confusion_matrix=[[5, 0], [0, 5], [1, 1]],
            )

    def test_init_with_irregular_matrix_raises_error(self):
        """Test that irregular matrix raises error."""
        with pytest.raises(ValueError, match=r"All rows .* must have same length"):
            ExperimentResult(
                test_accuracy=0.9,
                test_precision=0.9,
                test_recall=0.9,
                test_f1=0.9,
                confusion_matrix=[[5, 0], [0, 5, 1]],
            )

    def test_init_with_non_integer_values_raises_error(self):
        """Test that non-integer values raise error."""
        with pytest.raises(ValueError, match="Confusion matrix values must be integers"):
            ExperimentResult(
                test_accuracy=0.9,
                test_precision=0.9,
                test_recall=0.9,
                test_f1=0.9,
                confusion_matrix=[[5.5, 0], [0, 5]],
            )

    def test_init_with_negative_values_raises_error(self):
        """Test that negative values raise error."""
        with pytest.raises(ValueError, match="Confusion matrix values must be non-negative"):
            ExperimentResult(
                test_accuracy=0.9,
                test_precision=0.9,
                test_recall=0.9,
                test_f1=0.9,
                confusion_matrix=[[5, -1], [0, 5]],
            )


class TestExperimentResultGetters:
    """Test getter methods."""

    def test_get_accuracy(self, valid_result_data):
        """Test get_accuracy method."""
        result = ExperimentResult(**valid_result_data)
        assert result.get_accuracy() == 0.9500

    def test_get_precision(self, valid_result_data):
        """Test get_precision method."""
        result = ExperimentResult(**valid_result_data)
        assert result.get_precision() == 0.9400

    def test_get_recall(self, valid_result_data):
        """Test get_recall method."""
        result = ExperimentResult(**valid_result_data)
        assert result.get_recall() == 0.9500

    def test_get_f1(self, valid_result_data):
        """Test get_f1 method."""
        result = ExperimentResult(**valid_result_data)
        assert result.get_f1() == 0.9450

    def test_get_confusion_matrix(self, valid_result_data):
        """Test get_confusion_matrix method."""
        result = ExperimentResult(**valid_result_data)
        assert result.get_confusion_matrix() == [[10, 0, 0], [0, 9, 1], [0, 1, 9]]


class TestExperimentResultToDict:
    """Test to_dict method."""

    def test_to_dict_returns_correct_structure(self, valid_result_data):
        """Test that to_dict returns correct dictionary."""
        result = ExperimentResult(**valid_result_data)
        result_dict = result.to_dict()

        assert result_dict == valid_result_data

    def test_to_dict_includes_all_fields(self, valid_result_data):
        """Test that to_dict includes all fields."""
        result = ExperimentResult(**valid_result_data)
        result_dict = result.to_dict()

        required_keys = [
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "confusion_matrix",
        ]
        assert set(result_dict.keys()) == set(required_keys)


class TestExperimentResultFromDict:
    """Test from_dict class method."""

    def test_from_dict_with_valid_data(self, valid_result_data):
        """Test creating result from valid dictionary."""
        result = ExperimentResult.from_dict(valid_result_data)

        assert result.test_accuracy == 0.9500
        assert result.test_precision == 0.9400
        assert result.confusion_matrix == [[10, 0, 0], [0, 9, 1], [0, 1, 9]]

    def test_from_dict_with_missing_key_raises_error(self):
        """Test that missing key raises KeyError."""
        incomplete_data = {"test_accuracy": 0.9, "test_precision": 0.9}

        with pytest.raises(KeyError, match="Missing required keys"):
            ExperimentResult.from_dict(incomplete_data)

    def test_from_dict_with_invalid_values_raises_error(self):
        """Test that invalid values raise ValueError."""
        invalid_data = {
            "test_accuracy": 1.5,
            "test_precision": 0.9,
            "test_recall": 0.9,
            "test_f1": 0.9,
            "confusion_matrix": [[5, 0], [0, 5]],
        }

        with pytest.raises(ValueError):
            ExperimentResult.from_dict(invalid_data)

    def test_from_dict_roundtrip(self, valid_result_data):
        """Test that to_dict and from_dict are inverses."""
        result1 = ExperimentResult(**valid_result_data)
        result_dict = result1.to_dict()
        result2 = ExperimentResult.from_dict(result_dict)

        assert result1 == result2


class TestExperimentResultRepr:
    """Test string representation."""

    def test_repr(self, valid_result_data):
        """Test __repr__ method."""
        result = ExperimentResult(**valid_result_data)
        repr_str = repr(result)

        assert "ExperimentResult" in repr_str
        assert "accuracy=0.9500" in repr_str
        assert "precision=0.9400" in repr_str
        assert "recall=0.9500" in repr_str
        assert "f1=0.9450" in repr_str


class TestExperimentResultEquality:
    """Test equality comparison."""

    def test_equal_results(self, valid_result_data):
        """Test that identical results are equal."""
        result1 = ExperimentResult(**valid_result_data)
        result2 = ExperimentResult(**valid_result_data)

        assert result1 == result2

    def test_unequal_accuracy(self, valid_result_data):
        """Test that different accuracy makes results unequal."""
        result1 = ExperimentResult(**valid_result_data)

        data2 = valid_result_data.copy()
        data2["test_accuracy"] = 0.8000
        result2 = ExperimentResult(**data2)

        assert result1 != result2

    def test_unequal_confusion_matrix(self, valid_result_data):
        """Test that different confusion matrix makes results unequal."""
        result1 = ExperimentResult(**valid_result_data)

        data2 = valid_result_data.copy()
        data2["confusion_matrix"] = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
        result2 = ExperimentResult(**data2)

        assert result1 != result2

    def test_not_equal_to_other_types(self, valid_result_data):
        """Test that result is not equal to other types."""
        result = ExperimentResult(**valid_result_data)

        assert result != valid_result_data
        assert result != "ExperimentResult"
        assert result != 0.95


class TestExperimentResultEdgeCases:
    """Test edge cases."""

    def test_perfect_scores(self):
        """Test with perfect scores (all 1.0)."""
        result = ExperimentResult(
            test_accuracy=1.0,
            test_precision=1.0,
            test_recall=1.0,
            test_f1=1.0,
            confusion_matrix=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        )

        assert result.test_accuracy == 1.0
        assert result.test_precision == 1.0

    def test_zero_scores(self):
        """Test with zero scores."""
        result = ExperimentResult(
            test_accuracy=0.0,
            test_precision=0.0,
            test_recall=0.0,
            test_f1=0.0,
            confusion_matrix=[[0, 5], [5, 0]],
        )

        assert result.test_accuracy == 0.0
        assert result.test_precision == 0.0

    def test_large_confusion_matrix(self):
        """Test with larger confusion matrix."""
        large_matrix = [[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]]

        result = ExperimentResult(
            test_accuracy=1.0,
            test_precision=1.0,
            test_recall=1.0,
            test_f1=1.0,
            confusion_matrix=large_matrix,
        )

        assert result.confusion_matrix == large_matrix
