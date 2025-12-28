"""Comprehensive tests for Experiment class with full integration."""

import copy
import json

import numpy as np
import pytest

from src.experiment.experiment import Experiment


@pytest.fixture
def valid_config():
    """Valid experiment configuration."""
    return {
        "data": {
            "path": "data/raw/iris.data",
            "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "target_column": "species",
        },
        "preprocessing": {
            "scaler_type": "standard",
            "split_strategy": {
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
            },
        },
        "model": {
            "architecture": {
                "hidden_layers": [32, 16],
                "output_size": 3,
            },
            "activations": {
                "hidden": "relu",
                "output": "softmax",
            },
        },
        "training": {
            "optimizer": {
                "type": "sgd_momentum",
                "learning_rate": 0.01,
                "momentum": 0.9,
            },
            "loss_function": "cross_entropy",
            "epochs": 10,
            "batch_size": 16,
            "patience": 10,
        },
    }


class TestExperimentInitialization:
    """Test Experiment initialization."""

    def test_experiment_initialization(self, valid_config):
        """Test that experiment initializes correctly."""
        exp = Experiment(valid_config, experiment_id=1)

        assert exp.config == valid_config
        assert exp.id == 1
        assert exp.data_factory is not None
        assert exp.model_factory is not None

    def test_experiment_get_config(self, valid_config):
        """Test getting experiment configuration."""
        exp = Experiment(valid_config, experiment_id=1)

        assert exp.get_config() == valid_config

    def test_experiment_get_id(self, valid_config):
        """Test getting experiment ID."""
        exp = Experiment(valid_config, experiment_id=42)

        assert exp.get_id() == 42

    def test_experiment_repr(self, valid_config):
        """Test string representation."""
        exp = Experiment(valid_config, experiment_id=5)

        assert repr(exp) == "Experiment(id=5)"


class TestExperimentRun:
    """Test complete experiment execution."""

    def test_run_returns_dict(self, valid_config):
        """Test that run returns dictionary."""
        exp = Experiment(valid_config, experiment_id=1)
        result = exp.run()

        assert isinstance(result, dict)

    def test_run_contains_required_keys(self, valid_config):
        """Test that result contains all required keys."""
        exp = Experiment(valid_config, experiment_id=1)
        result = exp.run()

        required_keys = [
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "confusion_matrix",
            "config",
            "experiment_id",
            "training_history",
        ]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_run_metrics_are_floats(self, valid_config):
        """Test that metrics are float values."""
        exp = Experiment(valid_config, experiment_id=1)
        result = exp.run()

        assert isinstance(result["test_accuracy"], float)
        assert isinstance(result["test_precision"], float)
        assert isinstance(result["test_recall"], float)
        assert isinstance(result["test_f1"], float)

    def test_run_metrics_in_valid_range(self, valid_config):
        """Test that metrics are in valid range [0, 1]."""
        exp = Experiment(valid_config, experiment_id=1)
        result = exp.run()

        assert 0 <= result["test_accuracy"] <= 1
        assert 0 <= result["test_precision"] <= 1
        assert 0 <= result["test_recall"] <= 1
        assert 0 <= result["test_f1"] <= 1

    def test_run_confusion_matrix_is_list(self, valid_config):
        """Test that confusion matrix is a list."""
        exp = Experiment(valid_config, experiment_id=1)
        result = exp.run()

        assert isinstance(result["confusion_matrix"], list)
        assert len(result["confusion_matrix"]) == 3  # 3 classes for Iris

    def test_run_stores_experiment_id(self, valid_config):
        """Test that result stores experiment ID."""
        exp = Experiment(valid_config, experiment_id=7)
        result = exp.run()

        assert result["experiment_id"] == 7

    def test_run_stores_config(self, valid_config):
        """Test that result stores configuration."""
        exp = Experiment(valid_config, experiment_id=1)
        result = exp.run()

        assert result["config"] == valid_config

    def test_run_stores_training_history(self, valid_config):
        """Test that result stores minimal training history."""
        exp = Experiment(valid_config, experiment_id=1)
        result = exp.run()

        history = result["training_history"]

        # Check only minimal fields are present
        assert "best_epoch" in history
        assert "total_epochs_trained" in history

        # Check removed fields are NOT present
        assert "final_train_loss" not in history
        assert "final_train_accuracy" not in history
        assert "final_val_loss" not in history
        assert "final_val_accuracy" not in history
        assert "best_val_accuracy" not in history
        assert "stopped_early" not in history

        # Validate values
        assert isinstance(history["best_epoch"], int)
        assert isinstance(history["total_epochs_trained"], int)
        assert history["best_epoch"] > 0
        assert history["total_epochs_trained"] > 0


class TestExperimentDifferentConfigurations:
    """Test experiment with different configurations."""

    def test_run_with_different_architecture(self, valid_config):
        """Test running with different architecture."""
        config = copy.deepcopy(valid_config)
        config["model"]["architecture"]["hidden_layers"] = [64, 32]

        exp = Experiment(config, experiment_id=1)
        result = exp.run()

        assert result["test_accuracy"] >= 0

    def test_run_with_different_learning_rate(self, valid_config):
        """Test running with different learning rate."""
        config = copy.deepcopy(valid_config)
        config["training"]["optimizer"]["learning_rate"] = 0.001

        exp = Experiment(config, experiment_id=1)
        result = exp.run()

        assert result["test_accuracy"] >= 0

    def test_run_with_different_batch_size(self, valid_config):
        """Test running with different batch size."""
        config = copy.deepcopy(valid_config)
        config["training"]["batch_size"] = 32

        exp = Experiment(config, experiment_id=1)
        result = exp.run()

        assert result["test_accuracy"] >= 0

    def test_run_with_minmax_scaler(self, valid_config):
        """Test running with MinMax scaler."""
        config = copy.deepcopy(valid_config)
        config["preprocessing"]["scaler_type"] = "minmax"

        exp = Experiment(config, experiment_id=1)
        result = exp.run()

        assert result["test_accuracy"] >= 0

    def test_run_with_different_epochs(self, valid_config):
        """Test running with different number of epochs."""
        config = copy.deepcopy(valid_config)
        config["training"]["epochs"] = 5

        exp = Experiment(config, experiment_id=1)
        result = exp.run()

        assert result["test_accuracy"] >= 0


class TestExperimentIntegration:
    """Integration tests for complete pipeline."""

    def test_experiment_produces_json_serializable_output(self, valid_config):
        """Test that experiment output is JSON serializable."""
        exp = Experiment(valid_config, experiment_id=1)
        result = exp.run()

        # Remove config from result for simpler test
        result_without_config = {k: v for k, v in result.items() if k != "config"}

        try:
            json.dumps(result_without_config)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Result is not JSON serializable: {e}")

    def test_metrics_have_4_decimal_places(self, valid_config):
        """Test that metrics are rounded to 4 decimal places."""
        exp = Experiment(valid_config, experiment_id=1)
        result = exp.run()

        # Check that metrics have at most 4 decimal places
        for metric in ["test_accuracy", "test_precision", "test_recall", "test_f1"]:
            value = result[metric]
            assert round(value, 4) == value, f"{metric} not rounded to 4 decimals"

    def test_training_improves_accuracy(self, valid_config):
        """Test that training improves accuracy over epochs."""
        config = copy.deepcopy(valid_config)
        config["training"]["epochs"] = 50  # More epochs to see improvement

        exp = Experiment(config, experiment_id=1)
        result = exp.run()

        # Test accuracy should be reasonable for Iris dataset
        # With 3 classes, random baseline is ~33%. With early stopping, expect at least 40%.
        assert result["test_accuracy"] > 0.40, (
            f"Test accuracy {result['test_accuracy']:.4f} is too low (expected > 0.40). "
            f"This may indicate training issues or unlucky early stopping."
        )

    def test_reproducibility_with_same_config(self, valid_config):
        """Test that same config produces similar results."""
        exp1 = Experiment(valid_config, experiment_id=1)
        result1 = exp1.run()

        exp2 = Experiment(valid_config, experiment_id=2)
        result2 = exp2.run()

        # Results should be similar (not exact due to data splitting randomness)
        assert abs(result1["test_accuracy"] - result2["test_accuracy"]) < 0.2


class TestExperimentPrivateMethods:
    """Test private helper methods."""

    def test_to_onehot_encoding(self, valid_config):
        """Test one-hot encoding conversion."""
        exp = Experiment(valid_config, experiment_id=1)

        y = np.array([0, 1, 2, 0, 1])
        onehot = exp._to_onehot(y, n_classes=3)

        expected = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ]
        )

        assert np.array_equal(onehot, expected)

    def test_to_onehot_shape(self, valid_config):
        """Test one-hot encoding shape."""
        exp = Experiment(valid_config, experiment_id=1)

        y = np.array([0, 1, 2, 0, 1, 2, 0])
        onehot = exp._to_onehot(y, n_classes=3)

        assert onehot.shape == (7, 3)

    def test_preprocess_data_returns_dict(self, valid_config):
        """Test that preprocessing returns dictionary with all splits."""
        exp = Experiment(valid_config, experiment_id=1)
        preprocessed = exp._preprocess_data()

        assert isinstance(preprocessed, dict)
        assert "X_train" in preprocessed
        assert "X_val" in preprocessed
        assert "X_test" in preprocessed
        assert "y_train" in preprocessed
        assert "y_val" in preprocessed
        assert "y_test" in preprocessed
