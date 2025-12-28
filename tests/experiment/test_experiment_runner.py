"""Tests for ExperimentRunner orchestration."""

import json

from pathlib import Path

import pytest

from src.experiment.base import IExperimentRunner
from src.experiment.experiment_runner import ExperimentRunner


class TestExperimentRunnerInitialization:
    """Test ExperimentRunner initialization."""

    def test_init_default_output_dir(self):
        """Test initialization with default output directory."""
        runner = ExperimentRunner()

        assert runner.output_dir == Path("results")
        assert runner.output_dir.exists()

    def test_init_custom_output_dir(self, tmp_path: Path):
        """Test initialization with custom output directory."""
        output_dir = tmp_path / "custom_results"
        runner = ExperimentRunner(output_dir=str(output_dir))

        assert runner.output_dir == output_dir
        assert runner.output_dir.exists()

    def test_init_creates_directory(self, tmp_path: Path):
        """Test that initialization creates output directory."""
        output_dir = tmp_path / "new_dir" / "nested"
        runner = ExperimentRunner(output_dir=str(output_dir))

        assert runner.output_dir.exists()

    def test_init_components(self):
        """Test that components are initialized."""
        runner = ExperimentRunner()

        assert runner.loader is not None
        assert runner.expander is not None
        assert runner.validator is not None

    def test_repr(self, tmp_path: Path):
        """Test string representation."""
        output_dir = tmp_path / "test_results"
        runner = ExperimentRunner(output_dir=str(output_dir))

        repr_str = repr(runner)
        assert "ExperimentRunner" in repr_str
        assert "test_results" in repr_str

    def test_interface_compliance(self):
        """Test that ExperimentRunner implements IExperimentRunner."""
        runner = ExperimentRunner()

        assert isinstance(runner, IExperimentRunner)


class TestExperimentRunnerSaveResults:
    """Test save_results method."""

    @pytest.fixture
    def sample_results(self) -> list[dict]:
        """Sample experiment results."""
        return [
            {
                "experiment_id": 1,
                "test_accuracy": 0.95,
                "test_f1": 0.94,
                "config": {"learning_rate": 0.01},
            },
            {
                "experiment_id": 2,
                "test_accuracy": 0.92,
                "test_f1": 0.91,
                "config": {"learning_rate": 0.001},
            },
        ]

    def test_save_results_default_filename(self, tmp_path: Path, sample_results: list[dict]):
        """Test saving results with default filename."""
        runner = ExperimentRunner(output_dir=str(tmp_path))
        output_path = runner.save_results(sample_results)

        assert output_path == tmp_path / "results.json"
        assert output_path.exists()

    def test_save_results_custom_filename(self, tmp_path: Path, sample_results: list[dict]):
        """Test saving results with custom filename."""
        runner = ExperimentRunner(output_dir=str(tmp_path))
        output_path = runner.save_results(sample_results, filename="custom.json")

        assert output_path == tmp_path / "custom.json"
        assert output_path.exists()

    def test_save_results_content(self, tmp_path: Path, sample_results: list[dict]):
        """Test that saved content is correct."""
        runner = ExperimentRunner(output_dir=str(tmp_path))
        output_path = runner.save_results(sample_results)

        with output_path.open() as f:
            loaded_results = json.load(f)

        assert loaded_results == sample_results
        assert len(loaded_results) == 2
        assert loaded_results[0]["experiment_id"] == 1

    def test_save_results_returns_path(self, tmp_path: Path, sample_results: list[dict]):
        """Test that save_results returns Path object."""
        runner = ExperimentRunner(output_dir=str(tmp_path))
        output_path = runner.save_results(sample_results)

        assert isinstance(output_path, Path)

    def test_save_empty_results(self, tmp_path: Path):
        """Test saving empty results list."""
        runner = ExperimentRunner(output_dir=str(tmp_path))
        output_path = runner.save_results([])

        assert output_path.exists()
        with output_path.open() as f:
            loaded = json.load(f)
        assert loaded == []


class TestExperimentRunnerFindBest:
    """Test find_best method."""

    @pytest.fixture
    def sample_results(self) -> list[dict]:
        """Sample experiment results with different metrics."""
        return [
            {
                "experiment_id": 1,
                "test_accuracy": 0.85,
                "test_f1": 0.90,
                "config": {"lr": 0.01},
            },
            {
                "experiment_id": 2,
                "test_accuracy": 0.95,
                "test_f1": 0.88,
                "config": {"lr": 0.001},
            },
            {
                "experiment_id": 3,
                "test_accuracy": 0.90,
                "test_f1": 0.92,
                "config": {"lr": 0.005},
            },
        ]

    def test_find_best_default_metric(self, sample_results: list[dict]):
        """Test finding best by default metric (test_accuracy)."""
        runner = ExperimentRunner()
        best = runner.find_best(sample_results)

        assert best["experiment_id"] == 2
        assert best["test_accuracy"] == 0.95

    def test_find_best_custom_metric(self, sample_results: list[dict]):
        """Test finding best by custom metric (test_f1)."""
        runner = ExperimentRunner()
        best = runner.find_best(sample_results, metric="test_f1")

        assert best["experiment_id"] == 3
        assert best["test_f1"] == 0.92

    def test_find_best_empty_results(self):
        """Test that empty results raises ValueError."""
        runner = ExperimentRunner()

        with pytest.raises(ValueError, match="results list is empty"):
            runner.find_best([])

    def test_find_best_invalid_metric(self, sample_results: list[dict]):
        """Test that invalid metric raises ValueError."""
        runner = ExperimentRunner()

        with pytest.raises(ValueError, match="Metric 'nonexistent' not found"):
            runner.find_best(sample_results, metric="nonexistent")

    def test_find_best_single_result(self):
        """Test finding best with single result."""
        runner = ExperimentRunner()
        results = [{"experiment_id": 1, "test_accuracy": 0.8}]

        best = runner.find_best(results)
        assert best["experiment_id"] == 1

    def test_find_best_returns_dict(self, sample_results: list[dict]):
        """Test that find_best returns dictionary."""
        runner = ExperimentRunner()
        best = runner.find_best(sample_results)

        assert isinstance(best, dict)
        assert "experiment_id" in best
        assert "config" in best


class TestExperimentRunnerRunSweep:
    """Test run_sweep method with real config files."""

    @pytest.fixture
    def simple_yaml(self, tmp_path: Path) -> Path:
        """Create simple YAML config without sweep."""
        yaml_file = tmp_path / "simple.yaml"
        yaml_content = """
    data:
      path: "data/raw/iris.data"
      feature_columns: ["sepal_length", "sepal_width", "petal_length", "petal_width"]
      target_column: "species"

    preprocessing:
      scaler_type: "standard"
      split_strategy:
        train_ratio: 0.7
        val_ratio: 0.15
        test_ratio: 0.15

    model:
      architecture:
        hidden_layers: [[16]]
        output_size: 3
      activations:
        hidden: "relu"
        output: "softmax"

    training:
      optimizer:
        type: "sgd_momentum"
        learning_rate: 0.01
        momentum: 0.9
      loss_function: "cross_entropy"
      epochs: 5
      batch_size: 16
    """
        yaml_file.write_text(yaml_content)
        return yaml_file

    @pytest.fixture
    def sweep_yaml(self, tmp_path: Path) -> Path:
        """Create YAML config with sweep parameters."""
        yaml_file = tmp_path / "sweep.yaml"
        yaml_content = """
    data:
      path: "data/raw/iris.data"
      feature_columns: ["sepal_length", "sepal_width", "petal_length", "petal_width"]
      target_column: "species"

    preprocessing:
      scaler_type: "standard"
      split_strategy:
        train_ratio: 0.7
        val_ratio: 0.15
        test_ratio: 0.15

    model:
      architecture:
        hidden_layers: [[16], [32]]
        output_size: 3
      activations:
        hidden: "relu"
        output: "softmax"

    training:
      optimizer:
        type: "sgd_momentum"
        learning_rate: [0.01, 0.001]
        momentum: 0.9
      loss_function: "cross_entropy"
      epochs: 3
      batch_size: 16
    """
        yaml_file.write_text(yaml_content)
        return yaml_file

    @pytest.fixture
    def mini_sweep_yaml(self, tmp_path: Path) -> Path:
        """Create minimal sweep config for fast testing."""
        yaml_file = tmp_path / "mini_sweep.yaml"
        yaml_content = """
    data:
      path: "data/raw/iris.data"
      feature_columns: ["sepal_length", "sepal_width"]
      target_column: "species"

    preprocessing:
      scaler_type: "standard"
      split_strategy:
        train_ratio: 0.7
        val_ratio: 0.15
        test_ratio: 0.15

    model:
      architecture:
        hidden_layers: [[8], [16]]
        output_size: 3
      activations:
        hidden: "relu"
        output: "softmax"

    training:
      optimizer:
        type: "sgd_momentum"
        learning_rate: 0.01
        momentum: 0.9
      loss_function: "cross_entropy"
      epochs: 2
      batch_size: 16
    """
        yaml_file.write_text(yaml_content)
        return yaml_file

    def test_run_sweep_simple_config(self, tmp_path: Path, simple_yaml: Path):
        """Test running sweep with single configuration."""
        runner = ExperimentRunner(output_dir=str(tmp_path / "results"))
        results = runner.run_sweep(str(simple_yaml))

        assert len(results) == 1
        assert results[0]["experiment_id"] == 1
        assert "test_accuracy" in results[0]
        assert "config" in results[0]

    def test_run_sweep_with_sweep_params(self, tmp_path: Path, sweep_yaml: Path):
        """Test running sweep with multiple configurations."""
        runner = ExperimentRunner(output_dir=str(tmp_path / "results"))
        results = runner.run_sweep(str(sweep_yaml))

        # 2 hidden_layers x 2 learning_rates = 4 configs
        assert len(results) == 4
        assert all("test_accuracy" in r for r in results)
        assert all("experiment_id" in r for r in results)

    def test_run_sweep_file_not_found(self, tmp_path: Path):
        """Test that non-existent file raises FileNotFoundError."""
        runner = ExperimentRunner(output_dir=str(tmp_path))

        with pytest.raises(FileNotFoundError):
            runner.run_sweep("nonexistent.yaml")

    def test_run_sweep_invalid_config(self, tmp_path: Path):
        """Test that invalid config raises ValueError."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_content = """
data:
  path: "data/raw/iris.data"
  # Missing required fields
"""
        yaml_file.write_text(yaml_content)

        runner = ExperimentRunner(output_dir=str(tmp_path / "results"))

        with pytest.raises(ValueError, match="validation errors"):
            runner.run_sweep(str(yaml_file))

    def test_run_sweep_stores_config_in_results(self, tmp_path: Path, simple_yaml: Path):
        """Test that each result contains its configuration."""
        runner = ExperimentRunner(output_dir=str(tmp_path / "results"))
        results = runner.run_sweep(str(simple_yaml))

        assert "config" in results[0]
        assert "data" in results[0]["config"]
        assert "model" in results[0]["config"]
        assert "training" in results[0]["config"]


class TestExperimentRunnerIntegration:
    """Integration tests for full sweep workflow."""

    @pytest.fixture
    def mini_sweep_yaml(self, tmp_path: Path) -> Path:
        """Create minimal sweep config for fast testing."""
        yaml_file = tmp_path / "mini_sweep.yaml"
        yaml_content = """
    data:
      path: "data/raw/iris.data"
      feature_columns: ["sepal_length", "sepal_width"]
      target_column: "species"

    preprocessing:
      scaler_type: "standard"
      split_strategy:
        train_ratio: 0.7
        val_ratio: 0.15
        test_ratio: 0.15

    model:
      architecture:
        hidden_layers: [[8], [16]]
        output_size: 3
      activations:
        hidden: "relu"
        output: "softmax"

    training:
      optimizer:
        type: "sgd_momentum"
        learning_rate: 0.01
        momentum: 0.9
      loss_function: "cross_entropy"
      epochs: 2
      batch_size: 16
    """
        yaml_file.write_text(yaml_content)
        return yaml_file

    def test_full_workflow(self, tmp_path: Path, mini_sweep_yaml: Path):
        """Test complete workflow: run sweep -> save results -> find best."""
        runner = ExperimentRunner(output_dir=str(tmp_path / "results"))

        # Run sweep
        results = runner.run_sweep(str(mini_sweep_yaml))
        assert len(results) == 2

        # Save results
        output_path = runner.save_results(results, filename="results.json")
        assert output_path.exists()

        # Find best
        best = runner.find_best(results)
        assert best["experiment_id"] in [1, 2]
        assert 0 <= best["test_accuracy"] <= 1

    def test_results_are_comparable(self, tmp_path: Path, mini_sweep_yaml: Path):
        """Test that all results have same structure for comparison."""
        runner = ExperimentRunner(output_dir=str(tmp_path / "results"))
        results = runner.run_sweep(str(mini_sweep_yaml))

        # All results should have same keys
        keys_0 = set(results[0].keys())
        for result in results[1:]:
            assert set(result.keys()) == keys_0

    def test_saved_results_can_be_reloaded(self, tmp_path: Path, mini_sweep_yaml: Path):
        """Test that saved results can be loaded and used."""
        runner = ExperimentRunner(output_dir=str(tmp_path / "results"))

        # Run and save
        results = runner.run_sweep(str(mini_sweep_yaml))
        output_path = runner.save_results(results)

        # Load and verify
        with output_path.open() as f:
            loaded_results = json.load(f)

        assert len(loaded_results) == len(results)
        best_original = runner.find_best(results)
        best_loaded = runner.find_best(loaded_results)
        assert best_original["experiment_id"] == best_loaded["experiment_id"]


class TestExperimentRunnerEdgeCases:
    """Test edge cases and error handling."""

    def test_run_sweep_with_empty_sweep(self, tmp_path: Path):
        """Test config with empty list (should return single config with empty list)."""
        yaml_file = tmp_path / "empty_sweep.yaml"
        yaml_content = """
    data:
      path: "data/raw/iris.data"
      feature_columns: ["sepal_length"]
      target_column: "species"

    preprocessing:
      scaler_type: "standard"
      split_strategy:
        train_ratio: 0.7
        val_ratio: 0.15
        test_ratio: 0.15

    model:
      architecture:
        hidden_layers: []
        output_size: 3
      activations:
        hidden: "relu"
        output: "softmax"

    training:
      optimizer:
        type: "sgd_momentum"
        learning_rate: 0.01
        momentum: 0.9
      loss_function: "cross_entropy"
      epochs: 2
      batch_size: 16
    """
        yaml_file.write_text(yaml_content)

        runner = ExperimentRunner(output_dir=str(tmp_path / "results"))

        # Empty list = fixed value, not sweep, so should run but fail validation
        with pytest.raises(ValueError, match="validation errors"):
            runner.run_sweep(str(yaml_file))

    def test_save_results_overwrites_existing(self, tmp_path: Path):
        """Test that saving overwrites existing file."""
        runner = ExperimentRunner(output_dir=str(tmp_path))

        results1 = [{"experiment_id": 1, "test_accuracy": 0.8}]
        results2 = [{"experiment_id": 2, "test_accuracy": 0.9}]

        runner.save_results(results1, filename="test.json")
        runner.save_results(results2, filename="test.json")

        with (tmp_path / "test.json").open() as f:
            loaded = json.load(f)

        assert loaded == results2
        assert len(loaded) == 1
