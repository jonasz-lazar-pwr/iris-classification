"""Comprehensive tests for ConfigValidator."""

import copy

import pytest

from src.experiment.config_validator import ConfigValidator


@pytest.fixture
def validator():
    """Create ConfigValidator instance."""
    return ConfigValidator()


@pytest.fixture
def valid_config():
    """Valid expanded configuration."""
    return {
        "data": {
            "path": "data/iris.data",
            "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "target_column": "species",
        },
        "preprocessing": {
            "scaler_type": "standard",
            "split_strategy": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
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
            "epochs": 50,
            "batch_size": 16,
            "patience": 10,
            "random_state": 42,
        },
        "evaluation": {
            "metrics": ["accuracy", "precision", "recall"],
            "class_names": ["Setosa", "Versicolor", "Virginica"],
        },
    }


class TestConfigValidatorValid:
    """Test validation of valid configurations."""

    def test_validate_valid_config(self, validator, valid_config):
        """Test that valid config passes validation."""
        is_valid, errors = validator.validate(valid_config)

        assert is_valid
        assert errors == []

    def test_validate_minimal_valid_config(self, validator):
        """Test minimal valid configuration."""
        config = {
            "data": {
                "path": "data.csv",
                "feature_columns": ["feature1"],
                "target_column": "target",
            },
            "preprocessing": {
                "scaler_type": "standard",
                "split_strategy": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
            },
            "model": {
                "architecture": {"hidden_layers": [32], "output_size": 2},
                "activations": {"hidden": "relu", "output": "softmax"},
            },
            "training": {
                "optimizer": {"type": "sgd", "learning_rate": 0.01},
                "loss_function": "cross_entropy",
                "epochs": 10,
                "batch_size": 8,
            },
        }

        is_valid, errors = validator.validate(config)

        assert is_valid
        assert errors == []


class TestConfigValidatorDataSection:
    """Test validation of data section."""

    def test_missing_data_section(self, validator, valid_config):
        """Test that missing data section is detected."""
        del valid_config["data"]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert "Data section is missing" in errors

    def test_missing_data_path(self, validator, valid_config):
        """Test that missing data.path is detected."""
        del valid_config["data"]["path"]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("data.path is required" in e for e in errors)

    def test_missing_feature_columns(self, validator, valid_config):
        """Test that missing feature_columns is detected."""
        del valid_config["data"]["feature_columns"]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("data.feature_columns is required" in e for e in errors)

    def test_empty_feature_columns(self, validator, valid_config):
        """Test that empty feature_columns is detected."""
        valid_config["data"]["feature_columns"] = []

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("data.feature_columns cannot be empty" in e for e in errors)

    def test_feature_columns_not_list(self, validator, valid_config):
        """Test that feature_columns must be a list."""
        valid_config["data"]["feature_columns"] = "not_a_list"

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("data.feature_columns must be a list" in e for e in errors)

    def test_missing_target_column(self, validator, valid_config):
        """Test that missing target_column is detected."""
        del valid_config["data"]["target_column"]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("data.target_column is required" in e for e in errors)

    def test_target_column_not_string(self, validator, valid_config):
        """Test that target_column must be a string."""
        valid_config["data"]["target_column"] = 123

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("data.target_column must be a string" in e for e in errors)


class TestConfigValidatorPreprocessingSection:
    """Test validation of preprocessing section."""

    def test_missing_preprocessing_section(self, validator, valid_config):
        """Test that missing preprocessing section is detected."""
        del valid_config["preprocessing"]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert "Preprocessing section is missing" in errors

    def test_invalid_scaler_type(self, validator, valid_config):
        """Test that invalid scaler_type is detected."""
        valid_config["preprocessing"]["scaler_type"] = "invalid_scaler"

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("scaler_type 'invalid_scaler' not in allowed" in e for e in errors)

    def test_scaler_type_as_list_not_expanded(self, validator, valid_config):
        """Test that scaler_type as list (not expanded) is detected."""
        valid_config["preprocessing"]["scaler_type"] = ["standard", "minmax"]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("scaler_type should be a string, not list" in e for e in errors)

    def test_split_strategy_as_list_not_expanded(self, validator, valid_config):
        """Test that split_strategy as list (not expanded) is detected."""
        valid_config["preprocessing"]["split_strategy"] = [
            {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15}
        ]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("split_strategy should be a dict, not list" in e for e in errors)

    def test_split_ratios_missing_keys(self, validator, valid_config):
        """Test that missing split ratio keys are detected."""
        valid_config["preprocessing"]["split_strategy"] = {"train_ratio": 0.7, "val_ratio": 0.15}

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("split_strategy missing keys" in e for e in errors)

    def test_split_ratios_dont_sum_to_one(self, validator, valid_config):
        """Test that split ratios not summing to 1.0 are detected."""
        valid_config["preprocessing"]["split_strategy"] = {
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.1,
        }

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("split_strategy ratios must sum to 1.0" in e for e in errors)

    def test_split_ratios_negative(self, validator, valid_config):
        """Test that negative split ratios are detected."""
        valid_config["preprocessing"]["split_strategy"] = {
            "train_ratio": -0.7,
            "val_ratio": 1.0,
            "test_ratio": 0.7,
        }

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("must be between 0 and 1" in e for e in errors)

    def test_split_ratios_greater_than_one(self, validator, valid_config):
        """Test that split ratios > 1 are detected."""
        valid_config["preprocessing"]["split_strategy"] = {
            "train_ratio": 1.5,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        }

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("must be between 0 and 1" in e for e in errors)


class TestConfigValidatorModelSection:
    """Test validation of model section."""

    def test_missing_model_section(self, validator, valid_config):
        """Test that missing model section is detected."""
        del valid_config["model"]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert "Model section is missing" in errors

    def test_missing_architecture(self, validator, valid_config):
        """Test that missing architecture is detected."""
        del valid_config["model"]["architecture"]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("model.architecture is required" in e for e in errors)

    def test_missing_hidden_layers(self, validator, valid_config):
        """Test that missing hidden_layers is detected."""
        del valid_config["model"]["architecture"]["hidden_layers"]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("model.architecture.hidden_layers is required" in e for e in errors)

    def test_hidden_layers_nested_list_not_expanded(self, validator, valid_config):
        """Test that nested list hidden_layers (not expanded) is detected."""
        valid_config["model"]["architecture"]["hidden_layers"] = [[32, 16], [64, 32]]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("hidden_layers should be a single list, not nested" in e for e in errors)

    def test_hidden_layers_empty(self, validator, valid_config):
        """Test that empty hidden_layers is detected."""
        valid_config["model"]["architecture"]["hidden_layers"] = []

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("hidden_layers cannot be empty" in e for e in errors)

    def test_hidden_layers_non_integer(self, validator, valid_config):
        """Test that non-integer layer sizes are detected."""
        valid_config["model"]["architecture"]["hidden_layers"] = [64, "32", 16]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("hidden_layers[1] must be int" in e for e in errors)

    def test_hidden_layers_negative_size(self, validator, valid_config):
        """Test that negative layer sizes are detected."""
        valid_config["model"]["architecture"]["hidden_layers"] = [64, -32, 16]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("hidden_layers[1] must be positive" in e for e in errors)

    def test_missing_output_size(self, validator, valid_config):
        """Test that missing output_size is detected."""
        del valid_config["model"]["architecture"]["output_size"]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("model.architecture.output_size is required" in e for e in errors)

    def test_output_size_not_integer(self, validator, valid_config):
        """Test that non-integer output_size is detected."""
        valid_config["model"]["architecture"]["output_size"] = "3"

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("output_size must be int" in e for e in errors)

    def test_output_size_negative(self, validator, valid_config):
        """Test that negative output_size is detected."""
        valid_config["model"]["architecture"]["output_size"] = -3

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("output_size must be positive" in e for e in errors)

    def test_missing_activations(self, validator, valid_config):
        """Test that missing activations is detected."""
        del valid_config["model"]["activations"]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("model.activations is required" in e for e in errors)

    def test_invalid_hidden_activation(self, validator, valid_config):
        """Test that invalid hidden activation is detected."""
        valid_config["model"]["activations"]["hidden"] = "invalid_activation"

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("activations.hidden 'invalid_activation' not in allowed" in e for e in errors)

    def test_hidden_activation_as_list_not_expanded(self, validator, valid_config):
        """Test that hidden activation as list (not expanded) is detected."""
        valid_config["model"]["activations"]["hidden"] = ["relu", "tanh"]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("activations.hidden should be a string, not list" in e for e in errors)

    def test_invalid_output_activation(self, validator, valid_config):
        """Test that invalid output activation is detected."""
        valid_config["model"]["activations"]["output"] = "invalid_activation"

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("activations.output 'invalid_activation' not in allowed" in e for e in errors)


class TestConfigValidatorTrainingSection:
    """Test validation of training section."""

    def test_missing_training_section(self, validator, valid_config):
        """Test that missing training section is detected."""
        del valid_config["training"]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert "Training section is missing" in errors

    def test_invalid_optimizer_type(self, validator, valid_config):
        """Test that invalid optimizer type is detected."""
        valid_config["training"]["optimizer"]["type"] = "invalid_optimizer"

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("optimizer.type 'invalid_optimizer' not in allowed" in e for e in errors)

    def test_learning_rate_as_list_not_expanded(self, validator, valid_config):
        """Test that learning_rate as list (not expanded) is detected."""
        valid_config["training"]["optimizer"]["learning_rate"] = [0.01, 0.05]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("learning_rate should be a number, not list" in e for e in errors)

    def test_learning_rate_negative(self, validator, valid_config):
        """Test that negative learning_rate is detected."""
        valid_config["training"]["optimizer"]["learning_rate"] = -0.01

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("learning_rate must be positive" in e for e in errors)

    def test_learning_rate_zero(self, validator, valid_config):
        """Test that zero learning_rate is detected."""
        valid_config["training"]["optimizer"]["learning_rate"] = 0

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("learning_rate must be positive" in e for e in errors)

    def test_momentum_as_list_not_expanded(self, validator, valid_config):
        """Test that momentum as list (not expanded) is detected."""
        valid_config["training"]["optimizer"]["momentum"] = [0.9, 0.95]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("momentum should be a number, not list" in e for e in errors)

    def test_momentum_out_of_range(self, validator, valid_config):
        """Test that momentum outside [0, 1] is detected."""
        valid_config["training"]["optimizer"]["momentum"] = 1.5

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("momentum must be between 0 and 1" in e for e in errors)

    def test_invalid_loss_function(self, validator, valid_config):
        """Test that invalid loss_function is detected."""
        valid_config["training"]["loss_function"] = "invalid_loss"

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("loss_function 'invalid_loss' not in allowed" in e for e in errors)

    def test_epochs_as_list_not_expanded(self, validator, valid_config):
        """Test that epochs as list (not expanded) is detected."""
        valid_config["training"]["epochs"] = [30, 50, 100]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("epochs should be an int, not list" in e for e in errors)

    def test_epochs_negative(self, validator, valid_config):
        """Test that negative epochs is detected."""
        valid_config["training"]["epochs"] = -50

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("epochs must be positive" in e for e in errors)

    def test_batch_size_as_list_not_expanded(self, validator, valid_config):
        """Test that batch_size as list (not expanded) is detected."""
        valid_config["training"]["batch_size"] = [16, 32]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("batch_size should be an int, not list" in e for e in errors)

    def test_batch_size_negative(self, validator, valid_config):
        """Test that negative batch_size is detected."""
        valid_config["training"]["batch_size"] = -16

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("batch_size must be positive" in e for e in errors)

    # ← NOWE TESTY DLA PATIENCE:

    def test_patience_as_list_not_expanded(self, validator, valid_config):
        """Test that patience as list (not expanded) is detected."""
        valid_config["training"]["patience"] = [5, 10, 20]

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("patience should be an int, not list" in e for e in errors)

    def test_patience_not_integer(self, validator, valid_config):
        """Test that non-integer patience is detected."""
        valid_config["training"]["patience"] = 10.5

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("patience must be int" in e for e in errors)

    def test_patience_zero(self, validator, valid_config):
        """Test that zero patience is detected."""
        valid_config["training"]["patience"] = 0

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("patience must be at least 1" in e for e in errors)

    def test_patience_negative(self, validator, valid_config):
        """Test that negative patience is detected."""
        valid_config["training"]["patience"] = -5

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("patience must be at least 1" in e for e in errors)

    def test_patience_optional(self, validator, valid_config):
        """Test that patience is optional."""
        # Remove patience from config
        if "patience" in valid_config["training"]:
            del valid_config["training"]["patience"]

        is_valid, errors = validator.validate(valid_config)

        # Should still be valid if patience is missing
        assert is_valid
        assert errors == []

    def test_patience_valid_values(self, validator, valid_config):
        """Test that valid patience values pass validation."""
        for patience in [1, 5, 10, 20, 100]:
            config = copy.deepcopy(valid_config)
            config["training"]["patience"] = patience

            is_valid, errors = validator.validate(config)

            assert is_valid, f"Patience {patience} should be valid but got errors: {errors}"


class TestConfigValidatorEvaluationSection:
    """Test validation of evaluation section."""

    def test_evaluation_section_optional(self, validator, valid_config):
        """Test that evaluation section is optional."""
        del valid_config["evaluation"]

        is_valid, _ = validator.validate(valid_config)

        # Should still be valid (other sections are fine)
        # Only fails if other errors exist
        assert is_valid

    def test_metrics_not_list(self, validator, valid_config):
        """Test that metrics must be a list."""
        valid_config["evaluation"]["metrics"] = "accuracy"

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("evaluation.metrics must be a list" in e for e in errors)

    def test_metrics_empty(self, validator, valid_config):
        """Test that empty metrics list is detected."""
        valid_config["evaluation"]["metrics"] = []

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("evaluation.metrics cannot be empty" in e for e in errors)

    def test_class_names_not_list(self, validator, valid_config):
        """Test that class_names must be a list."""
        valid_config["evaluation"]["class_names"] = "Setosa"

        is_valid, errors = validator.validate(valid_config)

        assert not is_valid
        assert any("evaluation.class_names must be a list" in e for e in errors)


class TestConfigValidatorValidateAll:
    """Test validation of multiple configurations."""

    def test_validate_all_valid_configs(self, validator, valid_config):
        """Test validating multiple valid configs."""
        configs = [copy.deepcopy(valid_config) for _ in range(3)]

        invalid = validator.validate_all(configs)

        assert invalid == {}

    def test_validate_all_with_some_invalid(self, validator, valid_config):
        """Test validating mix of valid and invalid configs."""
        invalid_config1 = copy.deepcopy(valid_config)
        invalid_config1["training"]["epochs"] = -50

        invalid_config2 = copy.deepcopy(valid_config)
        invalid_config2["preprocessing"]["scaler_type"] = "invalid"

        configs = [
            copy.deepcopy(valid_config),
            invalid_config1,
            copy.deepcopy(valid_config),
            invalid_config2,
        ]

        invalid = validator.validate_all(configs)

        assert len(invalid) == 2
        assert 1 in invalid  # Second config (index 1)
        assert 3 in invalid  # Fourth config (index 3)
        assert any("epochs must be positive" in e for e in invalid[1])
        assert any("scaler_type 'invalid' not in allowed" in e for e in invalid[3])

    def test_validate_all_empty_list(self, validator):
        """Test validating empty list of configs."""
        invalid = validator.validate_all([])

        assert invalid == {}


class TestConfigValidatorIntegration:
    """Integration tests with realistic scenarios."""

    def test_validate_unexpanded_config(self, validator):
        """Test that unexpanded config (with lists) is detected."""
        unexpanded_config = {
            "data": {
                "path": "data.csv",
                "feature_columns": ["f1", "f2"],
                "target_column": "target",
            },
            "preprocessing": {
                "scaler_type": ["standard", "minmax"],  # Not expanded!
                "split_strategy": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
            },
            "model": {
                "architecture": {
                    "hidden_layers": [[32, 16], [64, 32]],
                    "output_size": 3,
                },  # Not expanded!
                "activations": {"hidden": ["relu", "tanh"], "output": "softmax"},  # Not expanded!
            },
            "training": {
                "optimizer": {"type": "sgd", "learning_rate": [0.01, 0.05]},  # Not expanded!
                "loss_function": "cross_entropy",
                "epochs": [30, 50],  # Not expanded!
                "batch_size": 16,
            },
        }

        is_valid, errors = validator.validate(unexpanded_config)

        assert not is_valid
        # Should detect multiple "not expanded" errors
        not_expanded_errors = [e for e in errors if "not expanded" in e]
        assert len(not_expanded_errors) >= 4

    def test_validate_all_allowed_values(self, validator):
        """Test all allowed values for enums."""
        base_config = {
            "data": {"path": "data.csv", "feature_columns": ["f1"], "target_column": "target"},
            "preprocessing": {
                "scaler_type": "standard",
                "split_strategy": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
            },
            "model": {
                "architecture": {"hidden_layers": [32], "output_size": 2},
                "activations": {"hidden": "relu", "output": "softmax"},
            },
            "training": {
                "optimizer": {"type": "sgd", "learning_rate": 0.01},
                "loss_function": "cross_entropy",
                "epochs": 10,
                "batch_size": 8,
            },
        }

        # Test all allowed scalers
        for scaler in ["standard", "minmax", "robust", "normalizer"]:
            config = base_config.copy()
            config["preprocessing"]["scaler_type"] = scaler
            is_valid, _ = validator.validate(config)
            assert is_valid, f"Scaler '{scaler}' should be valid"

        # Test all allowed activations
        for activation in ["relu", "tanh", "sigmoid", "softmax", "linear"]:
            config = base_config.copy()
            config["model"]["activations"]["hidden"] = activation
            is_valid, _ = validator.validate(config)
            assert is_valid, f"Activation '{activation}' should be valid"

        # Test all allowed optimizers
        for optimizer in ["sgd", "sgd_momentum", "adam", "rmsprop"]:
            config = base_config.copy()
            config["training"]["optimizer"]["type"] = optimizer
            is_valid, _ = validator.validate(config)
            assert is_valid, f"Optimizer '{optimizer}' should be valid"
