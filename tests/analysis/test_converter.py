"""Tests for JSONToCSVConverter."""

import copy
import json

from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

from src.analysis.converter import JSONToCSVConverter


@pytest.fixture
def sample_result() -> Dict:
    """Sample experiment result."""
    return {
        "test_accuracy": 0.9733,
        "test_precision": 0.9750,
        "test_recall": 0.9680,
        "test_f1": 0.9700,
        "confusion_matrix": [[5, 0, 0], [0, 4, 1], [0, 0, 5]],
        "experiment_id": 1,
        "config": {
            "data": {
                "path": "data/raw/iris.data",
                "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
                "target_column": "species_encoded",
            },
            "preprocessing": {
                "scale_features": True,
                "scaler_type": "standard",
                "split_strategy": {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
                "stratify": True,
                "random_state": 42,
            },
            "model": {
                "architecture": {"hidden_layers": [64, 32], "output_size": 3},
                "activations": {"hidden": "relu", "output": "softmax"},
            },
            "training": {
                "optimizer": {"type": "sgd_momentum", "learning_rate": 0.01, "momentum": 0.9},
                "loss_function": "cross_entropy",
                "epochs": 100,
                "batch_size": 16,
                "patience": 30,
                "random_state": 42,
            },
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1_score", "confusion_matrix"],
                "class_names": ["Setosa", "Versicolor", "Virginica"],
            },
        },
        "training_history": {"best_epoch": 48, "total_epochs_trained": 78},
    }


@pytest.fixture
def sample_results(sample_result: Dict) -> List[Dict]:
    """List of sample experiment results."""
    result2 = copy.deepcopy(sample_result)
    result2["experiment_id"] = 2
    result2["test_accuracy"] = 0.9600
    result2["config"]["model"]["activations"]["hidden"] = "tanh"
    result2["config"]["training"]["batch_size"] = 32

    return [sample_result, result2]


@pytest.fixture
def temp_json_file(tmp_path: Path, sample_results: List[Dict]) -> Path:
    """Create temporary JSON file with sample results."""
    json_file = tmp_path / "test_results.json"
    with json_file.open("w") as f:
        json.dump(sample_results, f)
    return json_file


@pytest.fixture
def converter() -> JSONToCSVConverter:
    """Create converter instance."""
    return JSONToCSVConverter()


class TestJSONToCSVConverter:
    """Tests for JSONToCSVConverter class."""

    def test_load_success(self, converter: JSONToCSVConverter, temp_json_file: Path):
        """Test loading results from JSON file."""
        results = converter.load(str(temp_json_file))

        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["experiment_id"] == 1
        assert results[1]["experiment_id"] == 2

    def test_load_file_not_found(self, converter: JSONToCSVConverter):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            converter.load("non_existent_file.json")

    def test_convert_creates_dataframe(
        self, converter: JSONToCSVConverter, sample_results: List[Dict]
    ):
        """Test converting results to DataFrame."""
        df = converter.convert(sample_results)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_convert_has_required_columns(
        self, converter: JSONToCSVConverter, sample_results: List[Dict]
    ):
        """Test DataFrame has all required columns."""
        df = converter.convert(sample_results)

        # Configuration columns
        assert "experiment_id" in df.columns
        assert "scaler_type" in df.columns
        assert "split_strategy" in df.columns
        assert "layers" in df.columns
        assert "activation" in df.columns
        assert "learning_rate" in df.columns
        assert "momentum" in df.columns
        assert "batch_size" in df.columns
        assert "max_epochs" in df.columns

        # Metrics columns
        assert "test_accuracy" in df.columns
        assert "test_f1" in df.columns
        assert "test_precision" in df.columns
        assert "test_recall" in df.columns

        # Training history columns
        assert "best_epoch" in df.columns
        assert "total_epochs_trained" in df.columns

        # Confusion matrix columns (9 total)
        assert "cm_setosa_setosa" in df.columns
        assert "cm_versicolor_versicolor" in df.columns
        assert "cm_virginica_virginica" in df.columns

    def test_convert_correct_values(
        self, converter: JSONToCSVConverter, sample_results: List[Dict]
    ):
        """Test DataFrame contains correct values."""
        df = converter.convert(sample_results)

        # First row
        assert df.iloc[0]["experiment_id"] == 1
        assert df.iloc[0]["scaler_type"] == "standard"
        assert df.iloc[0]["split_strategy"] == "0.8/0.1/0.1"
        assert df.iloc[0]["layers"] == "[64, 32]"
        assert df.iloc[0]["activation"] == "relu"
        assert df.iloc[0]["learning_rate"] == 0.01
        assert df.iloc[0]["momentum"] == 0.9
        assert df.iloc[0]["batch_size"] == 16
        assert df.iloc[0]["max_epochs"] == 100
        assert df.iloc[0]["test_accuracy"] == 0.9733
        assert df.iloc[0]["best_epoch"] == 48
        assert df.iloc[0]["total_epochs_trained"] == 78

        # Second row
        assert df.iloc[1]["experiment_id"] == 2
        assert df.iloc[1]["activation"] == "tanh"
        assert df.iloc[1]["batch_size"] == 32

    def test_save_creates_file(
        self, converter: JSONToCSVConverter, sample_results: List[Dict], tmp_path: Path
    ):
        """Test saving DataFrame to CSV file."""
        df = converter.convert(sample_results)
        output_file = tmp_path / "output.csv"

        converter.save(df, str(output_file))

        assert output_file.exists()

    def test_save_creates_parent_directory(
        self, converter: JSONToCSVConverter, sample_results: List[Dict], tmp_path: Path
    ):
        """Test saving creates parent directories if needed."""
        df = converter.convert(sample_results)
        output_file = tmp_path / "subdir" / "output.csv"

        converter.save(df, str(output_file))

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_save_correct_csv_format(
        self, converter: JSONToCSVConverter, sample_results: List[Dict], tmp_path: Path
    ):
        """Test saved CSV has correct format."""
        df = converter.convert(sample_results)
        output_file = tmp_path / "output.csv"

        converter.save(df, str(output_file))

        # Read back and verify
        df_loaded = pd.read_csv(output_file)
        assert len(df_loaded) == 2
        assert list(df_loaded.columns) == list(df.columns)
        assert df_loaded.iloc[0]["experiment_id"] == 1

    def test_format_split(self, converter: JSONToCSVConverter):
        """Test split strategy formatting."""
        split = {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1}

        result = converter._format_split(split)

        assert result == "0.8/0.1/0.1"

    def test_format_split_different_ratios(self, converter: JSONToCSVConverter):
        """Test split strategy formatting with different ratios."""
        split = {"train_ratio": 0.7, "val_ratio": 0.2, "test_ratio": 0.1}

        result = converter._format_split(split)

        assert result == "0.7/0.2/0.1"

    def test_flatten_confusion_matrix(self, converter: JSONToCSVConverter):
        """Test confusion matrix flattening."""
        cm = [[5, 0, 0], [0, 4, 1], [0, 0, 5]]

        result = converter._flatten_confusion_matrix(cm)

        assert len(result) == 9
        assert result["cm_setosa_setosa"] == 5
        assert result["cm_setosa_versicolor"] == 0
        assert result["cm_setosa_virginica"] == 0
        assert result["cm_versicolor_setosa"] == 0
        assert result["cm_versicolor_versicolor"] == 4
        assert result["cm_versicolor_virginica"] == 1
        assert result["cm_virginica_setosa"] == 0
        assert result["cm_virginica_versicolor"] == 0
        assert result["cm_virginica_virginica"] == 5

    def test_flatten_confusion_matrix_empty(self, converter: JSONToCSVConverter):
        """Test flattening empty confusion matrix."""
        result = converter._flatten_confusion_matrix([])

        assert result == {}

    def test_extract_row_complete(self, converter: JSONToCSVConverter, sample_result: Dict):
        """Test extracting complete row from result."""
        row = converter._extract_row(sample_result)

        # Verify all expected keys present
        assert "experiment_id" in row
        assert "scaler_type" in row
        assert "split_strategy" in row
        assert "layers" in row
        assert "activation" in row
        assert "learning_rate" in row
        assert "momentum" in row
        assert "batch_size" in row
        assert "max_epochs" in row
        assert "test_accuracy" in row
        assert "test_f1" in row
        assert "test_precision" in row
        assert "test_recall" in row
        assert "best_epoch" in row
        assert "total_epochs_trained" in row
        assert "cm_setosa_setosa" in row

    def test_extract_row_missing_training_history(
        self, converter: JSONToCSVConverter, sample_result: Dict
    ):
        """Test extracting row when training history is missing."""
        del sample_result["training_history"]

        row = converter._extract_row(sample_result)

        assert row["best_epoch"] is None
        assert row["total_epochs_trained"] is None

    def test_repr(self, converter: JSONToCSVConverter):
        """Test string representation."""
        assert repr(converter) == "JSONToCSVConverter()"

    def test_full_pipeline(
        self, converter: JSONToCSVConverter, temp_json_file: Path, tmp_path: Path
    ):
        """Test complete load -> convert -> save pipeline."""
        output_file = tmp_path / "output.csv"

        # Load
        results = converter.load(str(temp_json_file))

        # Convert
        df = converter.convert(results)

        # Save
        converter.save(df, str(output_file))

        # Verify
        assert output_file.exists()
        df_loaded = pd.read_csv(output_file)
        assert len(df_loaded) == 2
        assert df_loaded.iloc[0]["test_accuracy"] == 0.9733
