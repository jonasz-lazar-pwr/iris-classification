"""Comprehensive tests for MLPConfig."""

import pytest

from src.models.config import MLPConfig, MLPConfigBuilder


class TestMLPConfigInitialization:
    """Test MLPConfig initialization."""

    def test_valid_config_initialization(self):
        """Test creating valid configuration."""
        config = MLPConfig(
            input_size=10,
            hidden_layers=[64, 32],
            output_size=3,
        )

        assert config.input_size == 10
        assert config.hidden_layers == [64, 32]
        assert config.output_size == 3
        assert config.hidden_activation == "relu"
        assert config.output_activation == "softmax"
        assert config.loss_function == "cross_entropy"
        assert config.optimizer == "sgd_momentum"

    def test_config_with_custom_parameters(self):
        """Test creating configuration with custom parameters."""
        config = MLPConfig(
            input_size=5,
            hidden_layers=[128],
            output_size=2,
            hidden_activation="tanh",
            output_activation="softmax",
            learning_rate=0.001,
            momentum=0.95,
            random_state=42,
        )

        assert config.input_size == 5
        assert config.hidden_layers == [128]
        assert config.output_size == 2
        assert config.hidden_activation == "tanh"
        assert config.learning_rate == 0.001
        assert config.momentum == 0.95
        assert config.random_state == 42

    def test_config_with_kwargs(self):
        """Test configuration with optimizer and loss kwargs."""
        config = MLPConfig(
            input_size=10,
            hidden_layers=[64],
            output_size=3,
            optimizer_kwargs={"nesterov": True},
            loss_kwargs={"epsilon": 1e-10},
        )

        assert config.optimizer_kwargs == {"nesterov": True}
        assert config.loss_kwargs == {"epsilon": 1e-10}


class TestMLPConfigValidation:
    """Test configuration validation."""

    def test_negative_input_size_raises_error(self):
        """Test that negative input size raises error."""
        with pytest.raises(ValueError, match="input_size must be positive"):
            MLPConfig(input_size=-1, hidden_layers=[64], output_size=3)

    def test_zero_input_size_raises_error(self):
        """Test that zero input size raises error."""
        with pytest.raises(ValueError, match="input_size must be positive"):
            MLPConfig(input_size=0, hidden_layers=[64], output_size=3)

    def test_negative_output_size_raises_error(self):
        """Test that negative output size raises error."""
        with pytest.raises(ValueError, match="output_size must be positive"):
            MLPConfig(input_size=10, hidden_layers=[64], output_size=-1)

    def test_zero_output_size_raises_error(self):
        """Test that zero output size raises error."""
        with pytest.raises(ValueError, match="output_size must be positive"):
            MLPConfig(input_size=10, hidden_layers=[64], output_size=0)

    def test_empty_hidden_layers_raises_error(self):
        """Test that empty hidden layers raises error."""
        with pytest.raises(ValueError, match="hidden_layers cannot be empty"):
            MLPConfig(input_size=10, hidden_layers=[], output_size=3)

    def test_negative_hidden_layer_size_raises_error(self):
        """Test that negative hidden layer size raises error."""
        with pytest.raises(ValueError, match="All hidden layer sizes must be positive"):
            MLPConfig(input_size=10, hidden_layers=[64, -32], output_size=3)

    def test_zero_hidden_layer_size_raises_error(self):
        """Test that zero hidden layer size raises error."""
        with pytest.raises(ValueError, match="All hidden layer sizes must be positive"):
            MLPConfig(input_size=10, hidden_layers=[64, 0], output_size=3)

    def test_non_integer_hidden_layers_raises_error(self):
        """Test that non-integer hidden layer sizes raise error."""
        with pytest.raises(TypeError, match="All hidden layer sizes must be integers"):
            MLPConfig(input_size=10, hidden_layers=[64, 32.5], output_size=3)  # type: ignore[list-item]

    def test_negative_learning_rate_raises_error(self):
        """Test that negative learning rate raises error."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            MLPConfig(input_size=10, hidden_layers=[64], output_size=3, learning_rate=-0.01)

    def test_zero_learning_rate_raises_error(self):
        """Test that zero learning rate raises error."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            MLPConfig(input_size=10, hidden_layers=[64], output_size=3, learning_rate=0.0)

    def test_negative_momentum_raises_error(self):
        """Test that negative momentum raises error."""
        with pytest.raises(ValueError, match="momentum must be in"):
            MLPConfig(input_size=10, hidden_layers=[64], output_size=3, momentum=-0.1)

    def test_momentum_equal_one_raises_error(self):
        """Test that momentum equal to 1 raises error."""
        with pytest.raises(ValueError, match="momentum must be in"):
            MLPConfig(input_size=10, hidden_layers=[64], output_size=3, momentum=1.0)

    def test_momentum_greater_than_one_raises_error(self):
        """Test that momentum greater than 1 raises error."""
        with pytest.raises(ValueError, match="momentum must be in"):
            MLPConfig(input_size=10, hidden_layers=[64], output_size=3, momentum=1.5)

    def test_negative_random_state_raises_error(self):
        """Test that negative random state raises error."""
        with pytest.raises(ValueError, match="random_state must be non-negative"):
            MLPConfig(input_size=10, hidden_layers=[64], output_size=3, random_state=-1)

    def test_empty_activation_raises_error(self):
        """Test that empty activation string raises error."""
        with pytest.raises(ValueError, match="hidden_activation must be a non-empty string"):
            MLPConfig(input_size=10, hidden_layers=[64], output_size=3, hidden_activation="")

    def test_empty_loss_function_raises_error(self):
        """Test that empty loss function raises error."""
        with pytest.raises(ValueError, match="loss_function must be a non-empty string"):
            MLPConfig(input_size=10, hidden_layers=[64], output_size=3, loss_function="")

    def test_empty_optimizer_raises_error(self):
        """Test that empty optimizer raises error."""
        with pytest.raises(ValueError, match="optimizer must be a non-empty string"):
            MLPConfig(input_size=10, hidden_layers=[64], output_size=3, optimizer="")


class TestMLPConfigArchitecture:
    """Test architecture-related methods."""

    def test_get_network_architecture(self):
        """Test getting complete network architecture."""
        config = MLPConfig(input_size=10, hidden_layers=[64, 32], output_size=3)

        architecture = config.get_network_architecture()

        assert architecture == [10, 64, 32, 3]

    def test_get_total_layers(self):
        """Test getting total number of layers."""
        config = MLPConfig(input_size=10, hidden_layers=[64, 32], output_size=3)

        assert config.get_total_layers() == 4

    def test_get_total_parameters_simple(self):
        """Test calculating total parameters for simple network."""
        config = MLPConfig(input_size=2, hidden_layers=[3], output_size=2)

        total_params = config.get_total_parameters()

        assert total_params == (2 * 3 + 3) + (3 * 2 + 2)
        assert total_params == 17

    def test_get_total_parameters_complex(self):
        """Test calculating total parameters for complex network."""
        config = MLPConfig(input_size=10, hidden_layers=[64, 32], output_size=3)

        total_params = config.get_total_parameters()

        layer1 = 10 * 64 + 64
        layer2 = 64 * 32 + 32
        layer3 = 32 * 3 + 3
        expected = layer1 + layer2 + layer3

        assert total_params == expected


class TestMLPConfigSerialization:
    """Test configuration serialization."""

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = MLPConfig(
            input_size=10,
            hidden_layers=[64, 32],
            output_size=3,
            learning_rate=0.001,
            random_state=42,
        )

        config_dict = config.to_dict()

        assert config_dict["input_size"] == 10
        assert config_dict["hidden_layers"] == [64, 32]
        assert config_dict["output_size"] == 3
        assert config_dict["learning_rate"] == 0.001
        assert config_dict["random_state"] == 42

    def test_from_dict(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            "input_size": 10,
            "hidden_layers": [64, 32],
            "output_size": 3,
            "hidden_activation": "tanh",
            "learning_rate": 0.001,
            "random_state": 42,
        }

        config = MLPConfig.from_dict(config_dict)

        assert config.input_size == 10
        assert config.hidden_layers == [64, 32]
        assert config.output_size == 3
        assert config.hidden_activation == "tanh"
        assert config.learning_rate == 0.001
        assert config.random_state == 42

    def test_to_dict_from_dict_roundtrip(self):
        """Test that to_dict and from_dict are inverses."""
        original_config = MLPConfig(
            input_size=10,
            hidden_layers=[128, 64],
            output_size=5,
            learning_rate=0.005,
            momentum=0.95,
        )

        config_dict = original_config.to_dict()
        restored_config = MLPConfig.from_dict(config_dict)

        assert restored_config.to_dict() == original_config.to_dict()


class TestMLPConfigBuilder:
    """Test MLPConfigBuilder fluent API."""

    def test_builder_basic_usage(self):
        """Test basic builder usage."""
        config = (
            MLPConfigBuilder()
            .with_input_size(10)
            .add_hidden_layer(64)
            .add_hidden_layer(32)
            .with_output_size(3)
            .build()
        )

        assert config.input_size == 10
        assert config.hidden_layers == [64, 32]
        assert config.output_size == 3

    def test_builder_with_all_options(self):
        """Test builder with all configuration options."""
        config = (
            MLPConfigBuilder()
            .with_input_size(10)
            .add_hidden_layer(128)
            .with_output_size(5)
            .with_hidden_activation("tanh")
            .with_output_activation("softmax")
            .with_loss("cross_entropy", epsilon=1e-10)
            .with_optimizer("sgd_momentum", nesterov=True)
            .with_learning_rate(0.001)
            .with_momentum(0.95)
            .with_random_state(42)
            .build()
        )

        assert config.input_size == 10
        assert config.hidden_layers == [128]
        assert config.output_size == 5
        assert config.hidden_activation == "tanh"
        assert config.learning_rate == 0.001
        assert config.momentum == 0.95
        assert config.random_state == 42
        assert config.loss_kwargs == {"epsilon": 1e-10}
        assert config.optimizer_kwargs == {"nesterov": True}

    def test_builder_with_hidden_layers_method(self):
        """Test setting all hidden layers at once."""
        config = (
            MLPConfigBuilder()
            .with_input_size(10)
            .with_hidden_layers([128, 64, 32])
            .with_output_size(3)
            .build()
        )

        assert config.hidden_layers == [128, 64, 32]

    def test_builder_without_input_size_raises_error(self):
        """Test that building without input size raises error."""
        with pytest.raises(ValueError, match="input_size must be set"):
            MLPConfigBuilder().add_hidden_layer(64).with_output_size(3).build()

    def test_builder_without_output_size_raises_error(self):
        """Test that building without output size raises error."""
        with pytest.raises(ValueError, match="output_size must be set"):
            MLPConfigBuilder().with_input_size(10).add_hidden_layer(64).build()

    def test_builder_without_hidden_layers_raises_error(self):
        """Test that building without hidden layers raises error."""
        with pytest.raises(ValueError, match="At least one hidden layer must be added"):
            MLPConfigBuilder().with_input_size(10).with_output_size(3).build()

    def test_builder_reset(self):
        """Test resetting builder."""
        builder = MLPConfigBuilder()

        config1 = builder.with_input_size(10).add_hidden_layer(64).with_output_size(3).build()

        builder.reset()

        config2 = builder.with_input_size(20).add_hidden_layer(128).with_output_size(5).build()

        assert config1.input_size == 10
        assert config2.input_size == 20
        assert config1.hidden_layers == [64]
        assert config2.hidden_layers == [128]


class TestMLPConfigRepresentation:
    """Test configuration string representation."""

    def test_repr(self):
        """Test string representation."""
        config = MLPConfig(
            input_size=10, hidden_layers=[64, 32], output_size=3, learning_rate=0.001
        )

        repr_str = repr(config)

        assert "MLPConfig" in repr_str
        assert "10->64->32->3" in repr_str
        assert "relu" in repr_str
        assert "softmax" in repr_str
        assert "sgd_momentum" in repr_str
        assert "0.001" in repr_str
