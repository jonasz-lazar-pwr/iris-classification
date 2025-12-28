import numpy as np
import pandas as pd
import pytest

from src.data.validator import DataValidator


class TestDataValidator:
    """Test suite for DataValidator."""

    def test_init(self):
        """Test DataValidator initialization."""
        validator = DataValidator()

        assert validator._report == {}

    def test_validate_clean_data(self):
        """Test validation on clean data."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [10, 20, 30, 40, 50],
                "c": ["x", "y", "z", "w", "v"],
            }
        )

        validator = DataValidator()
        result = validator.validate(df)

        assert len(result) == 5
        assert list(result.columns) == ["a", "b", "c"]

    def test_validate_with_missing_values(self):
        """Test validation removes rows with NaN values."""
        df = pd.DataFrame(
            {
                "a": [1, 2, np.nan, 4, 5],
                "b": [10, np.nan, 30, 40, 50],
                "c": ["x", "y", "z", "w", "v"],
            }
        )

        validator = DataValidator()
        result = validator.validate(df)

        assert len(result) == 3
        assert result["a"].isnull().sum() == 0
        assert result["b"].isnull().sum() == 0

    def test_validate_with_duplicates(self):
        """Test validation removes duplicate rows."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 2, 3, 3, 3],
                "b": [10, 20, 20, 30, 30, 30],
            }
        )

        validator = DataValidator()
        result = validator.validate(df)

        assert len(result) == 3
        assert result["a"].tolist() == [1, 2, 3]

    def test_validate_with_mixed_issues(self):
        """Test validation handles missing values and duplicates together."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 2, np.nan, 3, 3],
                "b": [10, 20, 20, 40, 30, 30],
            }
        )

        validator = DataValidator()
        result = validator.validate(df)

        assert len(result) == 3
        assert result["a"].isnull().sum() == 0

    def test_get_validation_report_missing_values(self):
        """Test validation report for missing values."""
        df = pd.DataFrame(
            {
                "a": [1, np.nan, 3],
                "b": [10, 20, np.nan],
                "c": ["x", "y", "z"],
            }
        )

        validator = DataValidator()
        validator.validate(df)
        report = validator.get_validation_report()

        assert "missing_values" in report
        assert "a" in report["missing_values"]
        assert "b" in report["missing_values"]
        assert "c" not in report["missing_values"]

    def test_get_validation_report_duplicates(self):
        """Test validation report for duplicates."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 2, 3],
                "b": [10, 20, 20, 30],
            }
        )

        validator = DataValidator()
        validator.validate(df)
        report = validator.get_validation_report()

        assert "duplicates" in report
        assert report["duplicates"] == 1

    def test_get_validation_report_outliers(self):
        """Test validation report for outliers."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 100],
                "b": [10, 20, 30, 40, 50, 60],
            }
        )

        validator = DataValidator()
        validator.validate(df)
        report = validator.get_validation_report()

        assert "outliers" in report
        assert "a" in report["outliers"]
        assert report["outliers"]["a"] > 0

    def test_get_validation_report_rows_removed(self):
        """Test validation report tracks rows removed."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 2, np.nan, 3],
                "b": [10, 20, 20, 40, 30],
            }
        )

        validator = DataValidator()
        validator.validate(df)
        report = validator.get_validation_report()

        assert "rows_removed" in report
        assert report["rows_removed"] == 2

    def test_validate_empty_dataframe(self):
        """Test validation on empty DataFrame."""
        df = pd.DataFrame({"a": [], "b": []})

        validator = DataValidator()
        result = validator.validate(df)

        assert len(result) == 0

    def test_validate_no_numeric_columns(self):
        """Test validation on DataFrame with no numeric columns."""
        df = pd.DataFrame(
            {
                "a": ["x", "y", "z"],
                "b": ["p", "q", "r"],
            }
        )

        validator = DataValidator()
        result = validator.validate(df)
        report = validator.get_validation_report()

        assert len(result) == 3
        assert "outliers" in report
        assert len(report["outliers"]) == 0

    def test_validate_multiple_calls(self):
        """Test validator can be reused for multiple validations."""
        df1 = pd.DataFrame({"a": [1, 2, np.nan], "b": [10, 20, 30]})
        df2 = pd.DataFrame({"x": [5, 5, 6], "y": [50, 60, 70]})

        validator = DataValidator()

        result1 = validator.validate(df1)
        report1 = validator.get_validation_report()

        result2 = validator.validate(df2)
        report2 = validator.get_validation_report()

        assert len(result1) == 2
        assert len(result2) == 3
        assert report1 != report2

    def test_outlier_detection_iqr_method(self):
        """Test IQR outlier detection method."""
        df = pd.DataFrame(
            {
                "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
            }
        )

        validator = DataValidator()
        validator.validate(df)
        report = validator.get_validation_report()

        assert report["outliers"]["values"] == 1

    def test_report_is_copy(self):
        """Test get_validation_report returns a copy."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        validator = DataValidator()
        validator.validate(df)

        report1 = validator.get_validation_report()
        report1["custom_key"] = "modified"

        report2 = validator.get_validation_report()

        assert "custom_key" not in report2

    @pytest.mark.parametrize("missing_count", [0, 1, 5, 10])
    def test_validate_various_missing_counts(self, missing_count: int):
        """Test validation with varying amounts of missing values."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for i in range(missing_count):
            data[i] = np.nan

        df = pd.DataFrame({"a": data})

        validator = DataValidator()
        result = validator.validate(df)

        assert len(result) == 10 - missing_count


class TestDataValidatorIntegration:
    """Integration tests for DataValidator."""

    def test_validate_iris_like_data(self):
        """Test validation on Iris-like dataset."""
        df = pd.DataFrame(
            {
                "sepal_length": [5.1, 4.9, np.nan, 4.6, 5.0, 5.0],
                "sepal_width": [3.5, 3.0, 3.2, 3.1, 3.6, 3.6],
                "petal_length": [1.4, 1.4, 1.3, 1.5, 1.4, 1.4],
                "petal_width": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                "species": ["setosa", "setosa", "setosa", "setosa", "setosa", "setosa"],
            }
        )

        validator = DataValidator()
        result = validator.validate(df)
        report = validator.get_validation_report()

        assert len(result) == 4
        assert "missing_values" in report
        assert "duplicates" in report
        assert report["duplicates"] == 1
        assert report["rows_removed"] == 2
