"""Comprehensive tests for ExperimentRunner."""

import json

from pathlib import Path

import pytest
import yaml

from src.experiment.config_expander import ConfigExpander
from src.experiment.config_loader import ConfigLoader
from src.experiment.config_validator import ConfigValidator
from src.experiment.experiment_runner import ExperimentRunner


@pytest.fixture
def runner(tmp_path):
    """Create ExperimentRunner with temporary directory."""
    return ExperimentRunner(output_dir=str(tmp_path / "results"))


@pytest.fixture
def sample_results():
    """Sample results for testing."""
    return [
        {
            "id": 1,
            "config": {"model": {"hidden_layers": [32, 16]}, "training": {"learning_rate": 0.01}},
            "results": {
                "test_accuracy": 0.9500,
                "test_precision": 0.9400,
                "test_recall": 0.9500,
                "test_f1": 0.9450,
                "confusion_matrix": [[10, 0, 0], [0, 9, 1], [0, 1, 9]],
            },
        },
        {
            "id": 2,
            "config": {"model": {"hidden_layers": [64, 32]}, "training": {"learning_rate": 0.05}},
            "results": {
                "test_accuracy": 0.9800,
                "test_precision": 0.9750,
                "test_recall": 0.9800,
                "test_f1": 0.9775,
                "confusion_matrix": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
            },
        },
        {
            "id": 3,
            "config": {"model": {"hidden_layers": [128, 64]}, "training": {"learning_rate": 0.1}},
            "results": {
                "test_accuracy": 0.9200,
                "test_precision": 0.9100,
                "test_recall": 0.9200,
                "test_f1": 0.9150,
                "confusion_matrix": [[10, 0, 0], [0, 8, 2], [0, 1, 9]],
            },
        },
    ]


class TestExperimentRunnerInitialization:
    """Test runner initialization."""

    def test_init_creates_output_directory(self, tmp_path):
        """Test that output directory is created."""
        output_dir = tmp_path / "test_results"
        _ = ExperimentRunner(output_dir=str(output_dir))

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_init_with_nested_directory(self, tmp_path):
        """Test initialization with nested directory path."""
        output_dir = tmp_path / "nested" / "path" / "results"
        _ = ExperimentRunner(output_dir=str(output_dir))

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_init_with_existing_directory(self, tmp_path):
        """Test initialization with already existing directory."""
        output_dir = tmp_path / "existing"
        output_dir.mkdir()

        runner = ExperimentRunner(output_dir=str(output_dir))

        assert output_dir.exists()
        assert runner.output_dir == output_dir

    def test_init_creates_loader_expander_validator(self, runner):
        """Test that loader, expander, and validator are created."""
        assert isinstance(runner.loader, ConfigLoader)
        assert isinstance(runner.expander, ConfigExpander)
        assert isinstance(runner.validator, ConfigValidator)


class TestExperimentRunnerSaveResults:
    """Test saving results to JSON."""

    def test_save_results_creates_file(self, runner, sample_results):
        """Test that save_results creates JSON file."""
        output_path = runner.save_results(sample_results)

        assert output_path.exists()
        assert output_path.is_file()
        assert output_path.suffix == ".json"

    def test_save_results_correct_content(self, runner, sample_results):
        """Test that saved JSON contains correct data."""
        output_path = runner.save_results(sample_results)

        with output_path.open("r") as f:
            saved_data = json.load(f)

        assert saved_data == sample_results

    def test_save_results_custom_filename(self, runner, sample_results):
        """Test saving with custom filename."""
        custom_name = "custom_results.json"
        output_path = runner.save_results(sample_results, filename=custom_name)

        assert output_path.name == custom_name
        assert output_path.exists()

    def test_save_results_overwrites_existing(self, runner, sample_results):
        """Test that save_results overwrites existing file."""
        # Save first time
        output_path = runner.save_results(sample_results)

        # Modify results and save again
        modified_results = sample_results.copy()
        modified_results[0]["results"]["test_accuracy"] = 0.9999

        runner.save_results(modified_results)

        # Read and verify
        with output_path.open("r") as f:
            saved_data = json.load(f)

        assert saved_data[0]["results"]["test_accuracy"] == 0.9999

    def test_save_results_empty_list(self, runner):
        """Test saving empty results list."""
        output_path = runner.save_results([])

        assert output_path.exists()

        with output_path.open("r") as f:
            saved_data = json.load(f)

        assert saved_data == []

    def test_save_results_returns_path(self, runner, sample_results):
        """Test that save_results returns Path object."""
        output_path = runner.save_results(sample_results)

        assert isinstance(output_path, Path)
        assert output_path == runner.output_dir / "results.json"


class TestExperimentRunnerFindBest:
    """Test finding best configuration."""

    def test_find_best_by_accuracy(self, runner, sample_results):
        """Test finding best configuration by test_accuracy."""
        best = runner.find_best(sample_results, metric="test_accuracy")

        assert best["id"] == 2
        assert best["results"]["test_accuracy"] == 0.9800

    def test_find_best_by_f1(self, runner, sample_results):
        """Test finding best configuration by test_f1."""
        best = runner.find_best(sample_results, metric="test_f1")

        assert best["id"] == 2
        assert best["results"]["test_f1"] == 0.9775

    def test_find_best_by_precision(self, runner, sample_results):
        """Test finding best configuration by test_precision."""
        best = runner.find_best(sample_results, metric="test_precision")

        assert best["id"] == 2
        assert best["results"]["test_precision"] == 0.9750

    def test_find_best_by_recall(self, runner, sample_results):
        """Test finding best configuration by test_recall."""
        best = runner.find_best(sample_results, metric="test_recall")

        assert best["id"] == 2
        assert best["results"]["test_recall"] == 0.9800

    def test_find_best_single_result(self, runner):
        """Test finding best with single result."""
        single_result = [
            {
                "id": 1,
                "config": {},
                "results": {"test_accuracy": 0.9500, "test_f1": 0.9400},
            }
        ]

        best = runner.find_best(single_result)

        assert best["id"] == 1
        assert best["results"]["test_accuracy"] == 0.9500

    def test_find_best_empty_results_raises_error(self, runner):
        """Test that find_best raises error on empty results."""
        with pytest.raises(ValueError, match="Cannot find best configuration from empty results"):
            runner.find_best([])

    def test_find_best_invalid_metric_raises_error(self, runner, sample_results):
        """Test that find_best raises error on invalid metric."""
        with pytest.raises(ValueError, match="Metric 'invalid_metric' not found"):
            runner.find_best(sample_results, metric="invalid_metric")

    def test_find_best_returns_complete_result(self, runner, sample_results):
        """Test that find_best returns complete result dict."""
        best = runner.find_best(sample_results)

        assert "id" in best
        assert "config" in best
        assert "results" in best
        assert isinstance(best["config"], dict)
        assert isinstance(best["results"], dict)


class TestExperimentRunnerRunSweep:
    """Test full sweep execution (integration tests)."""

    @pytest.fixture
    def minimal_config_file(self, tmp_path):
        """Create minimal valid config file for testing."""
        config_content = """
data:
  path: "data/raw/iris.data"
  feature_columns:
    - sepal_length
    - sepal_width
    - petal_length
    - petal_width
  target_column: species_encoded

preprocessing:
  scale_features: true
  scaler_type: [standard, minmax]
  split_strategy:
    - train_ratio: 0.7
      val_ratio: 0.15
      test_ratio: 0.15
  stratify: true
  random_state: 42

model:
  architecture:
    hidden_layers: [[16], [32]]
    output_size: 3
  activations:
    hidden: relu
    output: softmax

training:
  optimizer:
    type: sgd_momentum
    learning_rate: 0.01
    momentum: 0.9
  loss_function: cross_entropy
  epochs: 2
  batch_size: 16
  random_state: 42

evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score
    - confusion_matrix
  class_names:
    - Setosa
    - Versicolor
    - Virginica
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        return str(config_file)

    def test_run_sweep_returns_results_list(self, runner, minimal_config_file):
        """Test that run_sweep returns list of results."""
        results = runner.run_sweep(minimal_config_file)

        assert isinstance(results, list)
        assert len(results) > 0

    def test_run_sweep_correct_number_of_experiments(self, runner, minimal_config_file):
        """Test that run_sweep runs correct number of experiments."""
        results = runner.run_sweep(minimal_config_file)

        # scaler_type: 2, hidden_layers: 2 = 4 total
        assert len(results) == 4

    def test_run_sweep_results_structure(self, runner, minimal_config_file):
        """Test that each result has correct structure."""
        results = runner.run_sweep(minimal_config_file)

        for result in results:
            assert "id" in result
            assert "config" in result
            assert "results" in result
            assert isinstance(result["id"], int)
            assert isinstance(result["config"], dict)
            assert isinstance(result["results"], dict)

    def test_run_sweep_results_contain_metrics(self, runner, minimal_config_file):
        """Test that results contain all expected metrics."""
        results = runner.run_sweep(minimal_config_file)

        for result in results:
            metrics = result["results"]
            assert "test_accuracy" in metrics
            assert "test_precision" in metrics
            assert "test_recall" in metrics
            assert "test_f1" in metrics
            assert "confusion_matrix" in metrics

    def test_run_sweep_creates_results_file(self, runner, minimal_config_file):
        """Test that run_sweep creates results.json file."""
        runner.run_sweep(minimal_config_file)

        results_file = runner.output_dir / "results.json"
        assert results_file.exists()

    def test_run_sweep_results_file_readable(self, runner, minimal_config_file):
        """Test that results file is valid JSON."""
        runner.run_sweep(minimal_config_file)

        results_file = runner.output_dir / "results.json"
        with results_file.open("r") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) > 0

    def test_run_sweep_sequential_ids(self, runner, minimal_config_file):
        """Test that experiment IDs are sequential."""
        results = runner.run_sweep(minimal_config_file)

        ids = [r["id"] for r in results]
        assert ids == list(range(1, len(results) + 1))


class TestExperimentRunnerErrorHandling:
    """Test error handling in runner."""

    def test_run_sweep_nonexistent_file_raises_error(self, runner):
        """Test that run_sweep raises error for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            runner.run_sweep("nonexistent_config.yaml")

    def test_run_sweep_invalid_yaml_raises_error(self, runner, tmp_path):
        """Test that run_sweep raises error for invalid YAML."""
        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text("key: [unclosed list")

        with pytest.raises(yaml.YAMLError):
            runner.run_sweep(str(invalid_file))


class TestExperimentRunnerRepr:
    """Test string representation."""

    def test_repr(self, runner):
        """Test __repr__ method."""
        repr_str = repr(runner)

        assert "ExperimentRunner" in repr_str
        assert "output_dir" in repr_str
        assert str(runner.output_dir) in repr_str
