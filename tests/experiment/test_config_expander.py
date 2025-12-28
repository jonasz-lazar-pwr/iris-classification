"""Comprehensive tests for ConfigExpander."""

import pytest

from src.experiment.config_expander import ConfigExpander


@pytest.fixture
def expander():
    """Create ConfigExpander instance."""
    return ConfigExpander()


@pytest.fixture
def simple_config():
    """Simple config with one sweep parameter."""
    return {
        "preprocessing": {"scaler_type": ["standard", "minmax"]},
        "training": {"epochs": 50},
    }


@pytest.fixture
def iris_config():
    """Realistic Iris experiment configuration."""
    return {
        "data": {
            "path": "data/raw/iris.data",
            "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "target_column": "species_encoded",
        },
        "preprocessing": {
            "scale_features": True,
            "scaler_type": ["standard", "minmax"],
            "split_strategy": [
                {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
                {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2},
            ],
            "stratify": True,
            "random_state": 42,
        },
        "model": {
            "architecture": {"hidden_layers": [[32, 16], [64, 32], [64, 32, 16]], "output_size": 3},
            "activations": {"hidden": ["relu", "tanh"], "output": "softmax"},
        },
        "training": {
            "optimizer": {
                "type": "sgd_momentum",
                "learning_rate": [0.01, 0.05],
                "momentum": [0.9, 0.95],
            },
            "loss_function": "cross_entropy",
            "epochs": [30, 50, 100],
            "batch_size": [16, 32],
            "random_state": 42,
        },
        "evaluation": {
            "metrics": ["accuracy", "precision", "recall", "f1_score", "confusion_matrix"],
            "class_names": ["Setosa", "Versicolor", "Virginica"],
        },
    }


class TestConfigExpanderBasic:
    """Test basic expansion functionality."""

    def test_expand_single_parameter(self, expander, simple_config):
        """Test expanding config with single sweep parameter."""
        configs = expander.expand(simple_config)

        assert len(configs) == 2
        assert configs[0]["preprocessing"]["scaler_type"] == "standard"
        assert configs[1]["preprocessing"]["scaler_type"] == "minmax"

    def test_expand_no_sweep_params(self, expander):
        """Test config with no sweep parameters returns single config."""
        config = {"preprocessing": {"scaler_type": "standard"}, "training": {"epochs": 50}}

        configs = expander.expand(config)

        assert len(configs) == 1
        assert configs[0] == config
        assert configs[0] is not config  # Should be deep copy

    def test_expand_multiple_parameters(self, expander):
        """Test expanding config with multiple sweep parameters."""
        config = {
            "preprocessing": {"scaler_type": ["standard", "minmax"]},
            "training": {"epochs": [50, 100], "batch_size": [16, 32]},
        }

        configs = expander.expand(config)

        # 2 * 2 * 2 = 8 combinations
        assert len(configs) == 8

        # Check first combination
        assert configs[0]["preprocessing"]["scaler_type"] == "standard"
        assert configs[0]["training"]["epochs"] == 50
        assert configs[0]["training"]["batch_size"] == 16

        # Check last combination
        assert configs[7]["preprocessing"]["scaler_type"] == "minmax"
        assert configs[7]["training"]["epochs"] == 100
        assert configs[7]["training"]["batch_size"] == 32

    def test_expand_preserves_non_sweep_params(self, expander):
        """Test that non-sweep parameters are preserved."""
        config = {
            "preprocessing": {"scaler_type": ["standard", "minmax"], "random_state": 42},
            "training": {"epochs": 50},
        }

        configs = expander.expand(config)

        assert len(configs) == 2
        assert all(c["preprocessing"]["random_state"] == 42 for c in configs)
        assert all(c["training"]["epochs"] == 50 for c in configs)

    def test_expand_nested_dict_parameters(self, expander):
        """Test expanding nested dict sweep parameters."""
        config = {
            "preprocessing": {
                "split_strategy": [
                    {"train": 0.7, "val": 0.15, "test": 0.15},
                    {"train": 0.6, "val": 0.2, "test": 0.2},
                ]
            }
        }

        configs = expander.expand(config)

        assert len(configs) == 2
        assert configs[0]["preprocessing"]["split_strategy"]["train"] == 0.7
        assert configs[1]["preprocessing"]["split_strategy"]["train"] == 0.6

    def test_expand_list_of_lists(self, expander):
        """Test expanding list of lists (like hidden_layers)."""
        config = {"model": {"hidden_layers": [[32, 16], [64, 32], [128, 64, 32]]}}

        configs = expander.expand(config)

        assert len(configs) == 3
        assert configs[0]["model"]["hidden_layers"] == [32, 16]
        assert configs[1]["model"]["hidden_layers"] == [64, 32]
        assert configs[2]["model"]["hidden_layers"] == [128, 64, 32]


class TestConfigExpanderNonSweepPaths:
    """Test that non-sweep paths are not expanded."""

    def test_data_section_not_expanded(self, expander):
        """Test that data section lists are not treated as sweep params."""
        config = {
            "data": {
                "feature_columns": ["sepal_length", "sepal_width", "petal_length"],
                "target_column": "species",
            },
            "training": {"epochs": [50, 100]},
        }

        configs = expander.expand(config)

        # Only epochs should be expanded (2 configs)
        assert len(configs) == 2
        # feature_columns should remain as list in all configs
        assert all(
            c["data"]["feature_columns"] == ["sepal_length", "sepal_width", "petal_length"]
            for c in configs
        )

    def test_evaluation_metrics_not_expanded(self, expander):
        """Test that evaluation.metrics is not treated as sweep param."""
        config = {
            "evaluation": {"metrics": ["accuracy", "precision", "recall"]},
            "training": {"batch_size": [16, 32]},
        }

        configs = expander.expand(config)

        assert len(configs) == 2
        assert all(
            c["evaluation"]["metrics"] == ["accuracy", "precision", "recall"] for c in configs
        )

    def test_evaluation_class_names_not_expanded(self, expander):
        """Test that evaluation.class_names is not treated as sweep param."""
        config = {
            "evaluation": {"class_names": ["Setosa", "Versicolor", "Virginica"]},
            "training": {"epochs": [30, 50]},
        }

        configs = expander.expand(config)

        assert len(configs) == 2
        assert all(
            c["evaluation"]["class_names"] == ["Setosa", "Versicolor", "Virginica"] for c in configs
        )


class TestConfigExpanderCountCombinations:
    """Test combination counting."""

    def test_count_no_sweep_params(self, expander):
        """Test counting with no sweep parameters."""
        config = {"training": {"epochs": 50, "batch_size": 32}}

        assert expander.count_combinations(config) == 1

    def test_count_single_parameter(self, expander, simple_config):
        """Test counting with single sweep parameter."""
        assert expander.count_combinations(simple_config) == 2

    def test_count_multiple_parameters(self, expander):
        """Test counting with multiple sweep parameters."""
        config = {
            "preprocessing": {"scaler_type": ["standard", "minmax"]},
            "training": {"epochs": [30, 50, 100], "batch_size": [16, 32]},
        }

        # 2 * 3 * 2 = 12
        assert expander.count_combinations(config) == 12

    def test_count_matches_expand_length(self, expander, iris_config):
        """Test that count matches actual expanded configs length."""
        count = expander.count_combinations(iris_config)
        configs = expander.expand(iris_config)

        assert count == len(configs)

    def test_count_iris_config(self, expander, iris_config):
        """Test counting realistic Iris config."""
        # preprocessing: 2 * 2 = 4
        # model: 3 * 2 = 6
        # training: 2 * 2 * 3 * 2 = 24
        # Total: 4 * 6 * 24 = 576
        assert expander.count_combinations(iris_config) == 576


class TestConfigExpanderDeepCopy:
    """Test that original config is not modified."""

    def test_expand_does_not_modify_original(self, expander):
        """Test that expand does not modify original config."""
        original = {
            "preprocessing": {"scaler_type": ["standard", "minmax"]},
            "training": {"epochs": [50, 100]},
        }
        original_copy = {
            "preprocessing": {"scaler_type": ["standard", "minmax"]},
            "training": {"epochs": [50, 100]},
        }

        expander.expand(original)

        assert original == original_copy

    def test_expanded_configs_are_independent(self, expander):
        """Test that expanded configs are independent objects."""
        config = {
            "preprocessing": {"scaler_type": ["standard", "minmax"]},
            "training": {"epochs": 50},
        }

        configs = expander.expand(config)

        # Modify first config
        configs[0]["training"]["epochs"] = 999

        # Second config should be unaffected
        assert configs[1]["training"]["epochs"] == 50


class TestConfigExpanderEdgeCases:
    """Test edge cases and special scenarios."""

    def test_expand_empty_config(self, expander):
        """Test expanding empty config."""
        configs = expander.expand({})

        assert len(configs) == 1
        assert configs[0] == {}

    def test_expand_deeply_nested_config(self, expander):
        """Test expanding deeply nested configuration."""
        config = {
            "level1": {
                "level2": {
                    "level3": {"level4": {"param": ["value1", "value2"]}, "other": "fixed"},
                    "param2": ["a", "b"],
                }
            }
        }

        configs = expander.expand(config)

        # 2 * 2 = 4 combinations
        assert len(configs) == 4

    def test_expand_mixed_types_in_list(self, expander):
        """Test expanding lists with mixed types."""
        config = {"training": {"learning_rate": [0.001, 0.01, 0.1]}}

        configs = expander.expand(config)

        assert len(configs) == 3
        assert configs[0]["training"]["learning_rate"] == 0.001
        assert configs[1]["training"]["learning_rate"] == 0.01
        assert configs[2]["training"]["learning_rate"] == 0.1

    def test_expand_preserves_types(self, expander):
        """Test that data types are preserved after expansion."""
        config = {
            "training": {
                "epochs": [30, 50],  # ints
                "learning_rate": [0.01, 0.05],  # floats
                "optimizer": ["sgd", "adam"],  # strings
                "use_momentum": [True, False],  # bools
            }
        }

        configs = expander.expand(config)

        # Check first config types
        assert isinstance(configs[0]["training"]["epochs"], int)
        assert isinstance(configs[0]["training"]["learning_rate"], float)
        assert isinstance(configs[0]["training"]["optimizer"], str)
        assert isinstance(configs[0]["training"]["use_momentum"], bool)


class TestConfigExpanderIrisIntegration:
    """Integration tests with full Iris config."""

    def test_expand_full_iris_config(self, expander, iris_config):
        """Test expanding complete Iris configuration."""
        configs = expander.expand(iris_config)

        # Should generate 576 configs (2*2*3*2*2*2*3*2)
        assert len(configs) == 576

        # Check that all configs are valid
        for config in configs:
            assert "data" in config
            assert "preprocessing" in config
            assert "model" in config
            assert "training" in config
            assert "evaluation" in config

    def test_iris_configs_have_no_lists_in_sweep_params(self, expander, iris_config):
        """Test that expanded configs have no lists in sweep param locations."""
        configs = expander.expand(iris_config)

        for config in configs:
            # All sweep params should be concrete values, not lists
            assert isinstance(config["preprocessing"]["scaler_type"], str)
            assert isinstance(config["preprocessing"]["split_strategy"], dict)
            assert isinstance(
                config["model"]["architecture"]["hidden_layers"], list
            )  # But not nested
            assert isinstance(config["model"]["activations"]["hidden"], str)
            assert isinstance(config["training"]["optimizer"]["learning_rate"], float)
            assert isinstance(config["training"]["optimizer"]["momentum"], float)
            assert isinstance(config["training"]["epochs"], int)
            assert isinstance(config["training"]["batch_size"], int)

    def test_iris_configs_preserve_data_section(self, expander, iris_config):
        """Test that data section is preserved in all configs."""
        configs = expander.expand(iris_config)

        expected_data = iris_config["data"]

        for config in configs:
            assert config["data"] == expected_data

    def test_iris_configs_preserve_evaluation_section(self, expander, iris_config):
        """Test that evaluation section is preserved in all configs."""
        configs = expander.expand(iris_config)

        expected_evaluation = iris_config["evaluation"]

        for config in configs:
            assert config["evaluation"] == expected_evaluation

    def test_iris_first_and_last_config(self, expander, iris_config):
        """Test specific values of first and last configs."""
        configs = expander.expand(iris_config)

        # First config (all first values)
        first = configs[0]
        assert first["preprocessing"]["scaler_type"] == "standard"
        assert first["preprocessing"]["split_strategy"]["train_ratio"] == 0.7
        assert first["model"]["architecture"]["hidden_layers"] == [32, 16]
        assert first["model"]["activations"]["hidden"] == "relu"
        assert first["training"]["optimizer"]["learning_rate"] == 0.01
        assert first["training"]["optimizer"]["momentum"] == 0.9
        assert first["training"]["epochs"] == 30
        assert first["training"]["batch_size"] == 16

        # Last config (all last values)
        last = configs[-1]
        assert last["preprocessing"]["scaler_type"] == "minmax"
        assert last["preprocessing"]["split_strategy"]["train_ratio"] == 0.6
        assert last["model"]["architecture"]["hidden_layers"] == [64, 32, 16]
        assert last["model"]["activations"]["hidden"] == "tanh"
        assert last["training"]["optimizer"]["learning_rate"] == 0.05
        assert last["training"]["optimizer"]["momentum"] == 0.95
        assert last["training"]["epochs"] == 100
        assert last["training"]["batch_size"] == 32
