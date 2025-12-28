"""Configuration validator for checking config correctness."""

from typing import ClassVar, Dict, List, Tuple

from .base import IConfigValidator


class ConfigValidator(IConfigValidator):
    """Validates configuration correctness."""

    # Tolerance for floating point comparison
    SPLIT_RATIO_TOLERANCE: ClassVar[float] = 0.001

    # Allowed values for specific parameters
    ALLOWED_SCALERS: ClassVar[set[str]] = {"standard", "minmax", "robust", "normalizer"}
    ALLOWED_ACTIVATIONS: ClassVar[set[str]] = {"relu", "tanh", "sigmoid", "softmax", "linear"}
    ALLOWED_OPTIMIZERS: ClassVar[set[str]] = {"sgd", "sgd_momentum", "adam", "rmsprop"}
    ALLOWED_LOSS_FUNCTIONS: ClassVar[set[str]] = {
        "cross_entropy",
        "mse",
        "mae",
        "binary_crossentropy",
    }

    def validate(self, config: Dict) -> Tuple[bool, List[str]]:
        """Validate single configuration, returns (is_valid, list_of_errors)."""
        errors = []

        # Validate each section
        errors.extend(self._validate_data_section(config.get("data", {})))
        errors.extend(self._validate_preprocessing_section(config.get("preprocessing", {})))
        errors.extend(self._validate_model_section(config.get("model", {})))
        errors.extend(self._validate_training_section(config.get("training", {})))
        errors.extend(self._validate_evaluation_section(config.get("evaluation", {})))

        return len(errors) == 0, errors

    def validate_all(self, configs: List[Dict]) -> Dict[int, List[str]]:
        """Validate multiple configurations, returns dict mapping config_index to errors."""
        invalid_configs = {}

        for idx, config in enumerate(configs):
            is_valid, errors = self.validate(config)
            if not is_valid:
                invalid_configs[idx] = errors

        return invalid_configs

    def _validate_data_section(self, data: Dict) -> List[str]:
        """Validate data section."""
        errors = []

        if not data:
            errors.append("Data section is missing")
            return errors

        # Check required fields
        if "path" not in data:
            errors.append("data.path is required")

        if "feature_columns" not in data:
            errors.append("data.feature_columns is required")
        elif not isinstance(data["feature_columns"], list):
            errors.append("data.feature_columns must be a list")
        elif len(data["feature_columns"]) == 0:
            errors.append("data.feature_columns cannot be empty")

        if "target_column" not in data:
            errors.append("data.target_column is required")
        elif not isinstance(data["target_column"], str):
            errors.append("data.target_column must be a string")

        return errors

    def _validate_preprocessing_section(self, preprocessing: Dict) -> List[str]:
        """Validate preprocessing section."""
        errors = []

        if not preprocessing:
            errors.append("Preprocessing section is missing")
            return errors

        # Validate scaler_type
        if "scaler_type" in preprocessing:
            scaler = preprocessing["scaler_type"]
            if isinstance(scaler, list):
                errors.append(
                    "preprocessing.scaler_type should be a string, not list (config not expanded?)"
                )
            elif scaler not in self.ALLOWED_SCALERS:
                errors.append(
                    f"preprocessing.scaler_type '{scaler}' not in allowed: {self.ALLOWED_SCALERS}"
                )

        # Validate split_strategy
        if "split_strategy" in preprocessing:
            split = preprocessing["split_strategy"]
            if isinstance(split, list):
                errors.append(
                    "preprocessing.split_strategy should be a dict, not list (config not expanded?)"
                )
            elif isinstance(split, dict):
                errors.extend(self._validate_split_ratios(split))

        return errors

    def _validate_split_ratios(self, split: Dict) -> List[str]:
        """Validate train/val/test split ratios."""
        errors = []

        required_keys = {"train_ratio", "val_ratio", "test_ratio"}
        missing_keys = required_keys - set(split.keys())
        if missing_keys:
            errors.append(f"split_strategy missing keys: {missing_keys}")
            return errors

        # Check that all ratios are positive
        for key in required_keys:
            ratio = split[key]
            if not isinstance(ratio, (int, float)):
                errors.append(f"split_strategy.{key} must be a number, got {type(ratio).__name__}")
            elif ratio <= 0 or ratio >= 1:
                errors.append(f"split_strategy.{key} must be between 0 and 1, got {ratio}")

        # Check that ratios sum to 1.0 (with tolerance for floating point)
        if all(isinstance(split[k], (int, float)) for k in required_keys):
            total = split["train_ratio"] + split["val_ratio"] + split["test_ratio"]
            if abs(total - 1.0) > self.SPLIT_RATIO_TOLERANCE:
                errors.append(f"split_strategy ratios must sum to 1.0, got {total:.4f}")

        return errors

    def _validate_model_section(self, model: Dict) -> List[str]:  # noqa: PLR0912
        """Validate model section."""
        errors = []

        if not model:
            errors.append("Model section is missing")
            return errors

        # Validate architecture
        if "architecture" not in model:
            errors.append("model.architecture is required")
        else:
            arch = model["architecture"]

            if "hidden_layers" not in arch:
                errors.append("model.architecture.hidden_layers is required")
            else:
                hidden = arch["hidden_layers"]
                if isinstance(hidden, list):
                    # Check if it's a list of lists (not expanded)
                    if len(hidden) > 0 and isinstance(hidden[0], list):
                        errors.append(
                            "model.architecture.hidden_layers should be a single list, not nested "
                            "(config not expanded?)"
                        )
                    elif len(hidden) == 0:
                        errors.append("model.architecture.hidden_layers cannot be empty")
                    else:
                        # Validate each layer size
                        for i, size in enumerate(hidden):
                            if not isinstance(size, int):
                                errors.append(
                                    f"model.architecture.hidden_layers[{i}] must be int, got "
                                    f"{type(size).__name__}"
                                )
                            elif size <= 0:
                                errors.append(
                                    f"model.architecture.hidden_layers[{i}] must be positive, got {size}"
                                )

            if "output_size" not in arch:
                errors.append("model.architecture.output_size is required")
            elif not isinstance(arch["output_size"], int):
                errors.append(
                    f"model.architecture.output_size must be int, got {type(arch['output_size']).__name__}"
                )
            elif arch["output_size"] <= 0:
                errors.append(
                    f"model.architecture.output_size must be positive, got {arch['output_size']}"
                )

        # Validate activations
        if "activations" not in model:
            errors.append("model.activations is required")
        else:
            activations = model["activations"]

            if "hidden" in activations:
                hidden_act = activations["hidden"]
                if isinstance(hidden_act, list):
                    errors.append(
                        "model.activations.hidden should be a string, not list (config not expanded?)"
                    )
                elif hidden_act not in self.ALLOWED_ACTIVATIONS:
                    errors.append(
                        f"model.activations.hidden '{hidden_act}' not in allowed: "
                        f"{self.ALLOWED_ACTIVATIONS}"
                    )

            if "output" in activations:
                output_act = activations["output"]
                if output_act not in self.ALLOWED_ACTIVATIONS:
                    errors.append(
                        f"model.activations.output '{output_act}' not in allowed: "
                        f"{self.ALLOWED_ACTIVATIONS}"
                    )

        return errors

    def _validate_training_section(self, training: Dict) -> List[str]:  # noqa: PLR0912, PLR0915
        """Validate training section."""
        errors = []

        if not training:
            errors.append("Training section is missing")
            return errors

        # Validate optimizer
        if "optimizer" in training:
            optimizer = training["optimizer"]
            if not isinstance(optimizer, dict):
                errors.append("training.optimizer must be a dict")
            else:
                if "type" in optimizer:
                    opt_type = optimizer["type"]
                    if opt_type not in self.ALLOWED_OPTIMIZERS:
                        errors.append(
                            f"training.optimizer.type '{opt_type}' not in allowed: "
                            f"{self.ALLOWED_OPTIMIZERS}"
                        )

                if "learning_rate" in optimizer:
                    lr = optimizer["learning_rate"]
                    if isinstance(lr, list):
                        errors.append(
                            "training.optimizer.learning_rate should be a number, not list "
                            "(config not expanded?)"
                        )
                    elif not isinstance(lr, (int, float)):
                        errors.append(
                            f"training.optimizer.learning_rate must be a number, got {type(lr).__name__}"
                        )
                    elif lr <= 0:
                        errors.append(
                            f"training.optimizer.learning_rate must be positive, got {lr}"
                        )

                if "momentum" in optimizer:
                    momentum = optimizer["momentum"]
                    if isinstance(momentum, list):
                        errors.append(
                            "training.optimizer.momentum should be a number, not list "
                            "(config not expanded?)"
                        )
                    elif not isinstance(momentum, (int, float)):
                        errors.append(
                            f"training.optimizer.momentum must be a number, got {type(momentum).__name__}"
                        )
                    elif momentum < 0 or momentum > 1:
                        errors.append(
                            f"training.optimizer.momentum must be between 0 and 1, got {momentum}"
                        )

        # Validate loss_function
        if "loss_function" in training:
            loss = training["loss_function"]
            if loss not in self.ALLOWED_LOSS_FUNCTIONS:
                errors.append(
                    f"training.loss_function '{loss}' not in allowed: {self.ALLOWED_LOSS_FUNCTIONS}"
                )

        # Validate epochs
        if "epochs" in training:
            epochs = training["epochs"]
            if isinstance(epochs, list):
                errors.append("training.epochs should be an int, not list (config not expanded?)")
            elif not isinstance(epochs, int):
                errors.append(f"training.epochs must be int, got {type(epochs).__name__}")
            elif epochs <= 0:
                errors.append(f"training.epochs must be positive, got {epochs}")

        # Validate batch_size
        if "batch_size" in training:
            batch_size = training["batch_size"]
            if isinstance(batch_size, list):
                errors.append(
                    "training.batch_size should be an int, not list (config not expanded?)"
                )
            elif not isinstance(batch_size, int):
                errors.append(f"training.batch_size must be int, got {type(batch_size).__name__}")
            elif batch_size <= 0:
                errors.append(f"training.batch_size must be positive, got {batch_size}")

        if "patience" in training:
            patience = training["patience"]
            if isinstance(patience, list):
                errors.append("training.patience should be an int, not list (config not expanded?)")
            elif not isinstance(patience, int):
                errors.append(f"training.patience must be int, got {type(patience).__name__}")
            elif patience < 1:
                errors.append(f"training.patience must be at least 1, got {patience}")

        return errors

    def _validate_evaluation_section(self, evaluation: Dict) -> List[str]:
        """Validate evaluation section."""
        errors = []

        if not evaluation:
            # Evaluation section is optional
            return errors

        if "metrics" in evaluation:
            metrics = evaluation["metrics"]
            if not isinstance(metrics, list):
                errors.append("evaluation.metrics must be a list")
            elif len(metrics) == 0:
                errors.append("evaluation.metrics cannot be empty")

        if "class_names" in evaluation:
            class_names = evaluation["class_names"]
            if not isinstance(class_names, list):
                errors.append("evaluation.class_names must be a list")

        return errors
