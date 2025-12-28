"""Tests for DataPreprocessor orchestration and integration."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from src.data.base import IDataLoader, IDataPreprocessor, IDataSplitter, IDataValidator, IScaler
from src.data.config import PreprocessConfig
from src.data.loaders import IrisDataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.scalers import MinMaxScaler, StandardScaler
from src.data.splitter import DataSplitter
from src.data.validator import DataValidator


class TestDataPreprocessorInitialization:
    """Test suite for DataPreprocessor initialization."""

    def test_init_with_all_components(self):
        """Test initialization with all components."""
        loader = Mock(spec=IDataLoader)
        validator = Mock(spec=IDataValidator)
        splitter = Mock(spec=IDataSplitter)
        scaler = Mock(spec=IScaler)

        preprocessor = DataPreprocessor(
            loader=loader, validator=validator, splitter=splitter, scaler=scaler
        )

        assert preprocessor._loader is loader
        assert preprocessor._validator is validator
        assert preprocessor._splitter is splitter
        assert preprocessor._scaler is scaler

    def test_init_without_scaler(self):
        """Test initialization without scaler."""
        loader = Mock(spec=IDataLoader)
        validator = Mock(spec=IDataValidator)
        splitter = Mock(spec=IDataSplitter)

        preprocessor = DataPreprocessor(loader=loader, validator=validator, splitter=splitter)

        assert preprocessor._scaler is None

    def test_repr(self):
        """Test string representation."""
        loader = IrisDataLoader()
        validator = DataValidator()
        splitter = DataSplitter()
        scaler = StandardScaler()

        preprocessor = DataPreprocessor(
            loader=loader, validator=validator, splitter=splitter, scaler=scaler
        )

        repr_str = repr(preprocessor)
        assert "DataPreprocessor" in repr_str
        assert "IrisDataLoader" in repr_str
        assert "DataValidator" in repr_str
        assert "DataSplitter" in repr_str
        assert "StandardScaler" in repr_str

    def test_repr_without_scaler(self):
        """Test string representation without scaler."""
        loader = IrisDataLoader()
        validator = DataValidator()
        splitter = DataSplitter()

        preprocessor = DataPreprocessor(loader=loader, validator=validator, splitter=splitter)

        repr_str = repr(preprocessor)
        assert "scaler=None" in repr_str

    def test_interface_compliance(self):
        """Test that DataPreprocessor implements IDataPreprocessor."""
        preprocessor = DataPreprocessor(
            loader=Mock(spec=IDataLoader),
            validator=Mock(spec=IDataValidator),
            splitter=Mock(spec=IDataSplitter),
        )

        assert isinstance(preprocessor, IDataPreprocessor)


class TestDataPreprocessorProcess:
    """Test suite for DataPreprocessor.process method."""

    @pytest.fixture
    def sample_iris_csv(self, tmp_path: Path) -> str:
        """Create sample Iris CSV file."""
        csv_file = tmp_path / "iris.csv"
        csv_content = (
            "5.1,3.5,1.4,0.2,Iris-setosa\n"
            "4.9,3.0,1.4,0.2,Iris-setosa\n"
            "7.0,3.2,4.7,1.4,Iris-versicolor\n"
            "6.4,3.2,4.5,1.5,Iris-versicolor\n"
            "6.3,3.3,6.0,2.5,Iris-virginica\n"
            "5.8,2.7,5.1,1.9,Iris-virginica\n"
        )
        csv_file.write_text(csv_content)
        return str(csv_file)

    def test_process_full_pipeline(self, sample_iris_csv: str):
        """Test full preprocessing pipeline."""
        preprocessor = DataPreprocessor(
            loader=IrisDataLoader(),
            validator=DataValidator(),
            splitter=DataSplitter(),
            scaler=StandardScaler(),
        )

        config = PreprocessConfig(
            feature_columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            target_column="species",
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            stratify=False,
            random_state=42,
            scale_features=True,
        )

        result = preprocessor.process(sample_iris_csv, config)

        assert "X_train" in result
        assert "X_val" in result
        assert "X_test" in result
        assert "y_train" in result
        assert "y_val" in result
        assert "y_test" in result
        assert "validation_report" in result
        assert "feature_columns" in result
        assert "target_column" in result
        assert "scaler" in result

        assert len(result["X_train"]) + len(result["X_val"]) + len(result["X_test"]) == 6

    def test_process_returns_correct_shapes(self, sample_iris_csv: str):
        """Test that returned arrays have correct shapes."""
        preprocessor = DataPreprocessor(
            loader=IrisDataLoader(),
            validator=DataValidator(),
            splitter=DataSplitter(),
            scaler=StandardScaler(),
        )

        config = PreprocessConfig(
            feature_columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            target_column="species",
            stratify=False,
        )

        result = preprocessor.process(sample_iris_csv, config)

        assert result["X_train"].shape[1] == 4  # 4 features
        assert result["X_val"].shape[1] == 4
        assert result["X_test"].shape[1] == 4
        assert len(result["y_train"].shape) == 1  # 1D array
        assert len(result["y_val"].shape) == 1
        assert len(result["y_test"].shape) == 1

    def test_process_without_scaling(self, sample_iris_csv: str):
        """Test preprocessing without feature scaling."""
        preprocessor = DataPreprocessor(
            loader=IrisDataLoader(),
            validator=DataValidator(),
            splitter=DataSplitter(),
            scaler=StandardScaler(),
        )

        config = PreprocessConfig(
            feature_columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            target_column="species",
            scale_features=False,
            stratify=False,
        )

        result = preprocessor.process(sample_iris_csv, config)

        # Unscaled data should not be centered around 0
        assert result["X_train"].mean() != pytest.approx(0.0, abs=0.1)

    def test_process_without_scaler_component(self, sample_iris_csv: str):
        """Test preprocessing when scaler is None."""
        preprocessor = DataPreprocessor(
            loader=IrisDataLoader(),
            validator=DataValidator(),
            splitter=DataSplitter(),
            scaler=None,
        )

        config = PreprocessConfig(
            feature_columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            target_column="species",
            scale_features=True,
            stratify=False,
        )

        result = preprocessor.process(sample_iris_csv, config)

        assert result["scaler"] is None

    def test_process_with_minmax_scaler(self, sample_iris_csv: str):
        """Test preprocessing with MinMaxScaler."""
        preprocessor = DataPreprocessor(
            loader=IrisDataLoader(),
            validator=DataValidator(),
            splitter=DataSplitter(),
            scaler=MinMaxScaler(),
        )

        config = PreprocessConfig(
            feature_columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            target_column="species",
            scale_features=True,
            stratify=False,
        )

        result = preprocessor.process(sample_iris_csv, config)

        assert result["X_train"].min() >= 0.0
        assert result["X_train"].max() <= 1.0

    def test_process_with_zero_val_ratio(self, sample_iris_csv: str):
        """Test preprocessing with zero validation ratio."""
        preprocessor = DataPreprocessor(
            loader=IrisDataLoader(),
            validator=DataValidator(),
            splitter=DataSplitter(),
            scaler=StandardScaler(),
        )

        config = PreprocessConfig(
            feature_columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            target_column="species",
            train_ratio=0.7,
            val_ratio=0.0,
            test_ratio=0.3,
            stratify=False,
        )

        result = preprocessor.process(sample_iris_csv, config)

        assert len(result["X_val"]) == 0
        assert len(result["y_val"]) == 0

    def test_process_missing_feature_column(self, sample_iris_csv: str):
        """Test error when feature column is missing."""
        preprocessor = DataPreprocessor(
            loader=IrisDataLoader(),
            validator=DataValidator(),
            splitter=DataSplitter(),
            scaler=StandardScaler(),
        )

        config = PreprocessConfig(
            feature_columns=["nonexistent_column"],
            target_column="species",
        )

        with pytest.raises(ValueError, match="Missing feature columns"):
            preprocessor.process(sample_iris_csv, config)

    def test_process_missing_target_column(self, sample_iris_csv: str):
        """Test error when target column is missing."""
        preprocessor = DataPreprocessor(
            loader=IrisDataLoader(),
            validator=DataValidator(),
            splitter=DataSplitter(),
            scaler=StandardScaler(),
        )

        config = PreprocessConfig(
            feature_columns=["sepal_length"],
            target_column="nonexistent_target",
        )

        with pytest.raises(ValueError, match=r"Target column .* not found"):
            preprocessor.process(sample_iris_csv, config)

    def test_process_with_validation_report(self, sample_iris_csv: str):
        """Test that validation report is included in result."""
        preprocessor = DataPreprocessor(
            loader=IrisDataLoader(),
            validator=DataValidator(),
            splitter=DataSplitter(),
            scaler=StandardScaler(),
        )

        config = PreprocessConfig(
            feature_columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            target_column="species",
            stratify=False,
        )

        result = preprocessor.process(sample_iris_csv, config)

        assert isinstance(result["validation_report"], dict)
        assert "rows_removed" in result["validation_report"]
        assert "missing_values" in result["validation_report"]
        assert "duplicates" in result["validation_report"]


class TestDataPreprocessorIntegration:
    """Integration tests for DataPreprocessor."""

    @pytest.fixture
    def iris_csv_with_issues(self, tmp_path: Path) -> str:
        """Create Iris CSV with data quality issues."""
        csv_file = tmp_path / "iris_dirty.csv"
        csv_content = (
            "5.1,3.5,1.4,0.2,Iris-setosa\n"
            "5.1,3.5,1.4,0.2,Iris-setosa\n"  # Duplicate
            "7.0,3.2,4.7,1.4,Iris-versicolor\n"
            ",,4.5,1.5,Iris-versicolor\n"  # Missing values
            "6.3,3.3,6.0,2.5,Iris-virginica\n"
            "5.8,2.7,5.1,1.9,Iris-virginica\n"
        )
        csv_file.write_text(csv_content)
        return str(csv_file)

    def test_process_with_data_cleaning(self, iris_csv_with_issues: str):
        """Test full pipeline with data cleaning."""
        preprocessor = DataPreprocessor(
            loader=IrisDataLoader(),
            validator=DataValidator(),
            splitter=DataSplitter(),
            scaler=StandardScaler(),
        )

        config = PreprocessConfig(
            feature_columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            target_column="species",
            stratify=False,
        )

        result = preprocessor.process(iris_csv_with_issues, config)

        assert result["validation_report"]["rows_removed"] >= 2

    def test_process_reproducibility(self, tmp_path: Path):
        """Test that same random_state produces same splits."""
        csv_file = tmp_path / "iris.csv"
        csv_content = "\n".join(
            [
                f"{5.0 + i * 0.1},{3.0 + i * 0.05},{1.0 + i * 0.1},{0.2 + i * 0.05},Iris-setosa"
                for i in range(20)
            ]
        )
        csv_file.write_text(csv_content)

        preprocessor = DataPreprocessor(
            loader=IrisDataLoader(),
            validator=DataValidator(),
            splitter=DataSplitter(),
            scaler=StandardScaler(),
        )

        config = PreprocessConfig(
            feature_columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            target_column="species",
            random_state=42,
            stratify=False,
        )

        result1 = preprocessor.process(str(csv_file), config)
        result2 = preprocessor.process(str(csv_file), config)

        np.testing.assert_array_equal(result1["X_train"], result2["X_train"])
        np.testing.assert_array_equal(result1["y_train"], result2["y_train"])

    def test_process_with_different_random_states(self, tmp_path: Path):
        """Test that different random_state produces different splits."""
        csv_file = tmp_path / "iris.csv"
        csv_content = "\n".join(
            [
                f"{5.0 + i * 0.1},{3.0 + i * 0.05},{1.0 + i * 0.1},{0.2 + i * 0.05},Iris-setosa"
                for i in range(20)
            ]
        )
        csv_file.write_text(csv_content)

        preprocessor = DataPreprocessor(
            loader=IrisDataLoader(),
            validator=DataValidator(),
            splitter=DataSplitter(),
            scaler=StandardScaler(),
        )

        config1 = PreprocessConfig(
            feature_columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            target_column="species",
            random_state=42,
            stratify=False,
        )

        config2 = PreprocessConfig(
            feature_columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            target_column="species",
            random_state=999,
            stratify=False,
        )

        result1 = preprocessor.process(str(csv_file), config1)
        result2 = preprocessor.process(str(csv_file), config2)

        # Splits should be different with different random states
        assert not np.array_equal(result1["X_train"], result2["X_train"])


class TestDataPreprocessorEdgeCases:
    """Test edge cases for DataPreprocessor."""

    def test_process_with_single_feature(self, tmp_path: Path):
        """Test preprocessing with single feature column."""
        csv_file = tmp_path / "iris_single.csv"
        csv_content = (
            "5.1,3.5,1.4,0.2,Iris-setosa\n"
            "4.9,3.0,1.4,0.2,Iris-setosa\n"
            "7.0,3.2,4.7,1.4,Iris-versicolor\n"
            "6.4,3.2,4.5,1.5,Iris-versicolor\n"
        )
        csv_file.write_text(csv_content)

        preprocessor = DataPreprocessor(
            loader=IrisDataLoader(),
            validator=DataValidator(),
            splitter=DataSplitter(),
            scaler=StandardScaler(),
        )

        config = PreprocessConfig(
            feature_columns=["sepal_length"],
            target_column="species",
            stratify=False,
        )

        result = preprocessor.process(str(csv_file), config)

        assert result["X_train"].shape[1] == 1

    def test_process_custom_ratios(self, tmp_path: Path):
        """Test preprocessing with custom train/val/test ratios."""
        csv_file = tmp_path / "iris.csv"
        csv_content = "\n".join([f"{5.0 + i * 0.1},3.0,1.0,0.2,Iris-setosa" for i in range(30)])
        csv_file.write_text(csv_content)

        preprocessor = DataPreprocessor(
            loader=IrisDataLoader(),
            validator=DataValidator(),
            splitter=DataSplitter(),
            scaler=None,
        )

        config = PreprocessConfig(
            feature_columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            target_column="species",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            stratify=False,
        )

        result = preprocessor.process(str(csv_file), config)

        total = len(result["X_train"]) + len(result["X_val"]) + len(result["X_test"])
        assert total == 30
        assert len(result["X_train"]) == 18
        assert len(result["X_val"]) == 6
        assert len(result["X_test"]) == 6

    def test_process_stores_config_metadata(self, tmp_path: Path):
        """Test that config metadata is stored in result."""
        csv_file = tmp_path / "iris.csv"
        csv_content = "\n".join(
            [
                f"{5.0 + i * 0.1},{3.0 + i * 0.05},{1.4 + i * 0.1},{0.2 + i * 0.05},Iris-setosa"
                for i in range(10)
            ]
        )
        csv_file.write_text(csv_content)

        preprocessor = DataPreprocessor(
            loader=IrisDataLoader(),
            validator=DataValidator(),
            splitter=DataSplitter(),
            scaler=StandardScaler(),
        )

        config = PreprocessConfig(
            feature_columns=["sepal_length", "sepal_width"],
            target_column="species",
            stratify=False,
        )

        result = preprocessor.process(str(csv_file), config)

        assert result["feature_columns"] == ["sepal_length", "sepal_width"]
        assert result["target_column"] == "species"
