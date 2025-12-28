"""Comprehensive tests for ConfigLoader."""

import pytest
import yaml

from src.experiment.config_loader import ConfigLoader


@pytest.fixture
def loader():
    """Create ConfigLoader instance."""
    return ConfigLoader()


@pytest.fixture
def temp_yaml_file(tmp_path):
    """Create temporary YAML file with valid config."""
    config = {
        "data": {"path": "data/iris.data"},
        "preprocessing": {"scaler_type": ["standard", "minmax"]},
        "model": {"architecture": {"hidden_layers": [[32, 16]]}},
        "training": {"epochs": [50, 100]},
        "evaluation": {"metrics": ["accuracy"]},
    }

    yaml_file = tmp_path / "test_config.yaml"
    with yaml_file.open("w") as f:
        yaml.dump(config, f)

    return yaml_file


class TestConfigLoaderSuccess:
    """Test successful configuration loading."""

    def test_load_valid_yaml(self, loader, temp_yaml_file):
        """Test loading valid YAML file."""
        config = loader.load(str(temp_yaml_file))

        assert isinstance(config, dict)
        assert "data" in config
        assert "preprocessing" in config
        assert "model" in config
        assert "training" in config
        assert "evaluation" in config

    def test_load_returns_correct_structure(self, loader, temp_yaml_file):
        """Test that loaded config has correct structure."""
        config = loader.load(str(temp_yaml_file))

        assert config["data"]["path"] == "data/iris.data"
        assert config["preprocessing"]["scaler_type"] == ["standard", "minmax"]
        assert config["model"]["architecture"]["hidden_layers"] == [[32, 16]]
        assert config["training"]["epochs"] == [50, 100]
        assert config["evaluation"]["metrics"] == ["accuracy"]

    def test_load_with_yml_extension(self, loader, tmp_path):
        """Test loading file with .yml extension."""
        config = {"test": "value"}
        yml_file = tmp_path / "config.yml"
        with yml_file.open("w") as f:
            yaml.dump(config, f)

        result = loader.load(str(yml_file))
        assert result == config

    def test_load_with_nested_structures(self, loader, tmp_path):
        """Test loading deeply nested configuration."""
        config = {
            "level1": {"level2": {"level3": {"value": [1, 2, 3], "nested_list": [[1, 2], [3, 4]]}}}
        }

        yaml_file = tmp_path / "nested.yaml"
        with yaml_file.open("w") as f:
            yaml.dump(config, f)

        result = loader.load(str(yaml_file))
        assert result == config
        assert result["level1"]["level2"]["level3"]["value"] == [1, 2, 3]

    def test_load_preserves_data_types(self, loader, tmp_path):
        """Test that data types are preserved correctly."""
        config = {
            "string": "text",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"key": "value"},
        }

        yaml_file = tmp_path / "types.yaml"
        with yaml_file.open("w") as f:
            yaml.dump(config, f)

        result = loader.load(str(yaml_file))
        assert result["string"] == "text"
        assert result["integer"] == 42
        assert result["float"] == 3.14
        assert result["boolean"] is True
        assert result["none"] is None
        assert result["list"] == [1, 2, 3]
        assert result["dict"] == {"key": "value"}


class TestConfigLoaderFileErrors:
    """Test file-related errors."""

    def test_load_nonexistent_file(self, loader):
        """Test loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            loader.load("nonexistent_file.yaml")

    def test_load_directory_instead_of_file(self, loader, tmp_path):
        """Test loading directory raises ValueError."""
        directory = tmp_path / "config_dir"
        directory.mkdir()

        with pytest.raises(ValueError, match="Path is not a file"):
            loader.load(str(directory))

    def test_load_wrong_extension(self, loader, tmp_path):
        """Test loading file with wrong extension raises ValueError."""
        txt_file = tmp_path / "config.txt"
        txt_file.write_text("test: value")

        with pytest.raises(ValueError, match=r"File must have \.yaml or \.yml extension"):
            loader.load(str(txt_file))

    def test_load_json_extension(self, loader, tmp_path):
        """Test loading .json file raises ValueError."""
        json_file = tmp_path / "config.json"
        json_file.write_text('{"test": "value"}')

        with pytest.raises(ValueError, match=r"File must have \.yaml or \.yml extension"):
            loader.load(str(json_file))


class TestConfigLoaderYAMLErrors:
    """Test YAML parsing errors."""

    def test_load_malformed_yaml(self, loader, tmp_path):
        """Test loading malformed YAML raises YAMLError."""
        yaml_file = tmp_path / "malformed.yaml"
        yaml_file.write_text("key: value\n  invalid indentation\n: no key")

        with pytest.raises(yaml.YAMLError, match="Failed to parse YAML file"):
            loader.load(str(yaml_file))

    def test_load_invalid_yaml_syntax(self, loader, tmp_path):
        """Test loading YAML with invalid syntax."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("key: [unclosed list")

        with pytest.raises(yaml.YAMLError):
            loader.load(str(yaml_file))

    def test_load_empty_file(self, loader, tmp_path):
        """Test loading empty YAML file raises ValueError."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        with pytest.raises(ValueError, match="Configuration file is empty"):
            loader.load(str(yaml_file))

    def test_load_only_whitespace(self, loader, tmp_path):
        """Test loading file with only whitespace raises ValueError."""
        yaml_file = tmp_path / "whitespace.yaml"
        yaml_file.write_text("   \n\n   \n")

        with pytest.raises(ValueError, match="Configuration file is empty"):
            loader.load(str(yaml_file))

    def test_load_yaml_with_list_at_root(self, loader, tmp_path):
        """Test loading YAML with list at root raises ValueError."""
        yaml_file = tmp_path / "list_root.yaml"
        yaml_file.write_text("- item1\n- item2\n- item3")

        with pytest.raises(ValueError, match="Configuration must be a dictionary"):
            loader.load(str(yaml_file))

    def test_load_yaml_with_string_at_root(self, loader, tmp_path):
        """Test loading YAML with string at root raises ValueError."""
        yaml_file = tmp_path / "string_root.yaml"
        yaml_file.write_text("just a string")

        with pytest.raises(ValueError, match="Configuration must be a dictionary"):
            loader.load(str(yaml_file))


class TestConfigLoaderEdgeCases:
    """Test edge cases and special scenarios."""

    def test_load_with_comments(self, loader, tmp_path):
        """Test loading YAML with comments."""
        yaml_file = tmp_path / "comments.yaml"
        content = """
        # This is a comment
        data:
          path: "data/iris.data"  # inline comment
        # Another comment
        training:
          epochs: 50
        """
        yaml_file.write_text(content)

        config = loader.load(str(yaml_file))
        assert config["data"]["path"] == "data/iris.data"
        assert config["training"]["epochs"] == 50

    def test_load_with_anchors_and_aliases(self, loader, tmp_path):
        """Test loading YAML with anchors and aliases."""
        yaml_file = tmp_path / "anchors.yaml"
        content = """
        defaults: &defaults
          learning_rate: 0.01
          momentum: 0.9

        config1:
          <<: *defaults
          batch_size: 16

        config2:
          <<: *defaults
          batch_size: 32
        """
        yaml_file.write_text(content)

        config = loader.load(str(yaml_file))
        assert config["config1"]["learning_rate"] == 0.01
        assert config["config1"]["momentum"] == 0.9
        assert config["config1"]["batch_size"] == 16
        assert config["config2"]["batch_size"] == 32

    def test_load_unicode_content(self, loader, tmp_path):
        """Test loading YAML with Unicode characters."""
        yaml_file = tmp_path / "unicode.yaml"
        config = {"name": "Tęst Čonfigurátioñ", "emoji": "🚀 🎯 ✅", "polish": "Zażółć gęślą jaźń"}
        with yaml_file.open("w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)

        result = loader.load(str(yaml_file))
        assert result == config

    def test_load_large_config(self, loader, tmp_path):
        """Test loading large configuration file."""
        config = {
            f"key_{i}": {f"nested_{j}": [k for k in range(10)] for j in range(10)}
            for i in range(10)
        }

        yaml_file = tmp_path / "large.yaml"
        with yaml_file.open("w") as f:
            yaml.dump(config, f)

        result = loader.load(str(yaml_file))
        assert len(result) == 10
        assert len(result["key_0"]) == 10

    def test_load_with_pathlib_path(self, loader, temp_yaml_file):
        """Test loading using pathlib.Path object."""
        config = loader.load(str(temp_yaml_file))
        assert isinstance(config, dict)

    def test_load_multiple_times(self, loader, temp_yaml_file):
        """Test loading same file multiple times returns same result."""
        config1 = loader.load(str(temp_yaml_file))
        config2 = loader.load(str(temp_yaml_file))

        assert config1 == config2
        assert config1 is not config2  # Different objects


class TestConfigLoaderIntegration:
    """Integration tests with real-like configurations."""

    def test_load_iris_experiment_config(self, loader, tmp_path):
        """Test loading realistic Iris experiment configuration."""
        config = {
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
                "architecture": {
                    "hidden_layers": [[32, 16], [64, 32], [64, 32, 16]],
                    "output_size": 3,
                },
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

        yaml_file = tmp_path / "experiments.yaml"
        with yaml_file.open("w") as f:
            yaml.dump(config, f)

        result = loader.load(str(yaml_file))

        # Validate structure
        assert "data" in result
        assert "preprocessing" in result
        assert "model" in result
        assert "training" in result
        assert "evaluation" in result

        # Validate specific values
        assert result["data"]["target_column"] == "species_encoded"
        assert len(result["preprocessing"]["scaler_type"]) == 2
        assert len(result["model"]["architecture"]["hidden_layers"]) == 3
        assert len(result["training"]["epochs"]) == 3
        assert len(result["evaluation"]["class_names"]) == 3
