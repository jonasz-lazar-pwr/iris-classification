"""Tests for PreprocessConfig validation and behavior."""

import pytest

from src.data.config import PreprocessConfig


class TestPreprocessConfigDefaults:
    """Test PreprocessConfig default values."""

    def test_config_default_values(self):
        """Test config with default values."""
        config = PreprocessConfig(
            feature_columns=["a", "b"],
            target_column="target",
        )

        assert config.train_ratio == 0.7
        assert config.val_ratio == 0.15
        assert config.test_ratio == 0.15
        assert config.stratify is True
        assert config.random_state == 42
        assert config.scale_features is True

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = PreprocessConfig(
            feature_columns=["x", "y"],
            target_column="z",
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            stratify=False,
            random_state=123,
            scale_features=False,
        )

        assert config.train_ratio == 0.8
        assert config.val_ratio == 0.1
        assert config.test_ratio == 0.1
        assert config.stratify is False
        assert config.random_state == 123
        assert config.scale_features is False

    def test_config_with_all_custom_values(self):
        """Test creating config with all custom values."""
        config = PreprocessConfig(
            feature_columns=["x1", "x2", "x3", "x4"],
            target_column="y",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            stratify=False,
            random_state=999,
            scale_features=False,
        )

        assert config.feature_columns == ["x1", "x2", "x3", "x4"]
        assert config.target_column == "y"
        assert config.train_ratio == 0.6
        assert config.val_ratio == 0.2
        assert config.test_ratio == 0.2
        assert config.stratify is False
        assert config.random_state == 999
        assert config.scale_features is False


class TestPreprocessConfigRatioValidation:
    """Test ratio validation in PreprocessConfig."""

    def test_valid_ratios(self):
        """Test that valid ratios pass validation."""
        config = PreprocessConfig(
            feature_columns=["a", "b", "c"],
            target_column="target",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        assert config.train_ratio == 0.7
        assert config.val_ratio == 0.15
        assert config.test_ratio == 0.15

    def test_ratios_must_sum_to_one(self):
        """Test that ratios must sum to 1.0."""
        with pytest.raises(ValueError, match=r"Ratios must sum to 1\.0"):
            PreprocessConfig(
                feature_columns=["a"],
                target_column="target",
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.1,  # Sum = 0.9
            )

    def test_ratios_sum_slightly_off(self):
        """Test that ratios slightly off 1.0 raise error."""
        with pytest.raises(ValueError, match=r"Ratios must sum to 1\.0"):
            PreprocessConfig(
                feature_columns=["a"],
                target_column="target",
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.21,  # Sum = 1.01
            )

    def test_train_ratio_must_be_positive(self):
        """Test that train_ratio must be > 0."""
        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            PreprocessConfig(
                feature_columns=["a"],
                target_column="target",
                train_ratio=0.0,
                val_ratio=0.2,
                test_ratio=0.8,
            )

    def test_train_ratio_cannot_be_one(self):
        """Test that train_ratio cannot be 1.0."""
        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            PreprocessConfig(
                feature_columns=["a"],
                target_column="target",
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )

    def test_train_ratio_cannot_be_negative(self):
        """Test that train_ratio cannot be negative."""
        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            PreprocessConfig(
                feature_columns=["a"],
                target_column="target",
                train_ratio=-0.1,
                val_ratio=0.5,
                test_ratio=0.6,
            )

    def test_val_ratio_can_be_zero(self):
        """Test that val_ratio can be 0."""
        config = PreprocessConfig(
            feature_columns=["a"],
            target_column="target",
            train_ratio=0.8,
            val_ratio=0.0,
            test_ratio=0.2,
        )

        assert config.val_ratio == 0.0

    def test_val_ratio_cannot_be_negative(self):
        """Test that val_ratio cannot be negative."""
        with pytest.raises(ValueError, match="val_ratio must be between 0"):
            PreprocessConfig(
                feature_columns=["a"],
                target_column="target",
                train_ratio=0.6,
                val_ratio=-0.1,
                test_ratio=0.5,
            )

    def test_val_ratio_upper_bound(self):
        """Test that val_ratio cannot be >= 1.0."""
        with pytest.raises(ValueError, match=r"Ratios must sum to 1\.0"):
            PreprocessConfig(
                feature_columns=["a"],
                target_column="target",
                train_ratio=0.1,
                val_ratio=1.0,
                test_ratio=0.1,
            )

    def test_test_ratio_must_be_positive(self):
        """Test that test_ratio must be > 0."""
        with pytest.raises(ValueError, match="test_ratio must be between 0 and 1"):
            PreprocessConfig(
                feature_columns=["a"],
                target_column="target",
                train_ratio=0.9,
                val_ratio=0.1,
                test_ratio=0.0,
            )

    def test_test_ratio_cannot_be_negative(self):
        """Test that test_ratio cannot be negative."""
        with pytest.raises(ValueError, match="test_ratio must be between 0 and 1"):
            PreprocessConfig(
                feature_columns=["a"],
                target_column="target",
                train_ratio=0.7,
                val_ratio=0.5,
                test_ratio=-0.2,
            )


class TestPreprocessConfigColumnValidation:
    """Test column validation in PreprocessConfig."""

    def test_valid_columns(self):
        """Test that valid columns pass validation."""
        config = PreprocessConfig(
            feature_columns=["feat1", "feat2", "feat3"],
            target_column="label",
        )

        assert config.feature_columns == ["feat1", "feat2", "feat3"]
        assert config.target_column == "label"

    def test_feature_columns_cannot_be_empty(self):
        """Test that feature_columns cannot be empty."""
        with pytest.raises(ValueError, match="feature_columns cannot be empty"):
            PreprocessConfig(
                feature_columns=[],
                target_column="target",
            )

    def test_feature_columns_must_be_list(self):
        """Test that feature_columns must be a list."""
        with pytest.raises(TypeError, match="feature_columns must be a list"):
            PreprocessConfig(
                feature_columns="not_a_list",  # type: ignore[arg-type]
                target_column="target",
            )

    def test_feature_columns_tuple_fails(self):
        """Test that feature_columns as tuple fails."""
        with pytest.raises(TypeError, match="feature_columns must be a list"):
            PreprocessConfig(
                feature_columns=("a", "b"),  # type: ignore[arg-type]
                target_column="target",
            )

    def test_target_column_cannot_be_empty(self):
        """Test that target_column cannot be empty."""
        with pytest.raises(ValueError, match="target_column cannot be empty"):
            PreprocessConfig(
                feature_columns=["a"],
                target_column="",
            )

    def test_target_column_must_be_string(self):
        """Test that target_column must be a string."""
        with pytest.raises(TypeError, match="target_column must be a string"):
            PreprocessConfig(
                feature_columns=["a"],
                target_column=123,  # type: ignore[arg-type]
            )

    def test_target_column_as_list_fails(self):
        """Test that target_column as list fails."""
        with pytest.raises(TypeError, match="target_column must be a string"):
            PreprocessConfig(
                feature_columns=["a", "b"],
                target_column=["target"],  # type: ignore[arg-type]
            )

    def test_target_cannot_be_in_features(self):
        """Test that target_column cannot be in feature_columns."""
        with pytest.raises(ValueError, match=r"target_column .* cannot be in feature_columns"):
            PreprocessConfig(
                feature_columns=["a", "b", "target"],
                target_column="target",
            )

    def test_target_in_middle_of_features(self):
        """Test detection when target is in middle of feature list."""
        with pytest.raises(ValueError, match=r"target_column .* cannot be in feature_columns"):
            PreprocessConfig(
                feature_columns=["a", "label", "b"],
                target_column="label",
            )


class TestPreprocessConfigRandomStateValidation:
    """Test random_state validation in PreprocessConfig."""

    def test_valid_random_state(self):
        """Test that valid random_state passes."""
        config = PreprocessConfig(
            feature_columns=["a"],
            target_column="target",
            random_state=42,
        )

        assert config.random_state == 42

    def test_random_state_zero_is_valid(self):
        """Test that random_state=0 is valid."""
        config = PreprocessConfig(
            feature_columns=["a"],
            target_column="target",
            random_state=0,
        )

        assert config.random_state == 0

    def test_random_state_must_be_non_negative(self):
        """Test that random_state must be >= 0."""
        with pytest.raises(ValueError, match="random_state must be non-negative"):
            PreprocessConfig(
                feature_columns=["a"],
                target_column="target",
                random_state=-1,
            )

    def test_random_state_large_value(self):
        """Test that large random_state values work."""
        config = PreprocessConfig(
            feature_columns=["a"],
            target_column="target",
            random_state=999999,
        )

        assert config.random_state == 999999

    def test_random_state_must_be_integer(self):
        """Test that random_state must be an integer."""
        with pytest.raises(TypeError, match="random_state must be an integer"):
            PreprocessConfig(
                feature_columns=["a"],
                target_column="target",
                random_state=42.5,  # type: ignore[arg-type]
            )

    def test_random_state_as_string_fails(self):
        """Test that random_state as string fails."""
        with pytest.raises(TypeError, match="random_state must be an integer"):
            PreprocessConfig(
                feature_columns=["a"],
                target_column="target",
                random_state="42",  # type: ignore[arg-type]
            )


class TestPreprocessConfigValidateMethod:
    """Test the validate() method behavior."""

    def test_validate_is_called_on_init(self):
        """Test that validate is automatically called during __post_init__."""
        # Should raise error immediately during initialization
        with pytest.raises(ValueError, match="feature_columns cannot be empty"):
            PreprocessConfig(
                feature_columns=[],
                target_column="target",
            )

    def test_validate_can_be_called_explicitly(self):
        """Test that validate() can be called explicitly."""
        config = PreprocessConfig(
            feature_columns=["a", "b"],
            target_column="target",
        )

        # Should not raise
        config.validate()

    def test_epsilon_constant(self):
        """Test that EPSILON constant exists and has reasonable value."""
        assert hasattr(PreprocessConfig, "EPSILON")
        assert PreprocessConfig.EPSILON == 1e-6


class TestPreprocessConfigEdgeCases:
    """Test edge cases for PreprocessConfig."""

    def test_single_feature_column(self):
        """Test config with single feature column."""
        config = PreprocessConfig(
            feature_columns=["only_feature"],
            target_column="target",
        )

        assert len(config.feature_columns) == 1
        assert config.feature_columns[0] == "only_feature"

    def test_many_feature_columns(self):
        """Test config with many feature columns."""
        features = [f"feat_{i}" for i in range(100)]
        config = PreprocessConfig(
            feature_columns=features,
            target_column="target",
        )

        assert len(config.feature_columns) == 100

    def test_ratios_with_floating_point_precision(self):
        """Test ratios that sum to 1.0 within epsilon."""
        config = PreprocessConfig(
            feature_columns=["a"],
            target_column="target",
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
        )

        assert abs((config.train_ratio + config.val_ratio + config.test_ratio) - 1.0) < 1e-6

    def test_special_characters_in_column_names(self):
        """Test column names with special characters."""
        config = PreprocessConfig(
            feature_columns=["feat_1", "feat-2", "feat.3"],
            target_column="target_label",
        )

        assert "feat_1" in config.feature_columns
        assert "feat-2" in config.feature_columns
        assert "feat.3" in config.feature_columns
