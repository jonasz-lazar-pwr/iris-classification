"""Single experiment execution with full integration."""

from typing import Dict

import numpy as np

from src.data.config import PreprocessConfig
from src.data.factory import DataComponentFactory
from src.data.loaders import IrisDataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.splitter import DataSplitter
from src.data.validator import DataValidator
from src.evaluation.metrics import MetricsCalculator
from src.experiment.base import IExperiment
from src.experiment.experiment_result import ExperimentResult
from src.models.builder import MLPBuilder
from src.models.factory import ModelComponentFactory
from src.training.trainer import Trainer


class Experiment(IExperiment):
    """Single experiment with full pipeline: preprocess -> build -> train -> evaluate."""

    def __init__(self, config: Dict, experiment_id: int) -> None:
        """Initialize experiment with configuration and ID."""
        super().__init__(config, experiment_id)
        self.data_factory = DataComponentFactory()
        self.model_factory = ModelComponentFactory()

    def run(self) -> Dict:
        """Execute complete experiment pipeline."""

        # 1. Preprocess data
        preprocessed = self._preprocess_data()

        # 2. Build model
        model = self._build_model(input_size=preprocessed["X_train"].shape[1])

        # 3. Train model
        history = self._train_model(
            model,
            preprocessed["X_train"],
            preprocessed["y_train"],
            preprocessed["X_val"],
            preprocessed["y_val"],
        )

        # 4. Evaluate model
        result = self._evaluate_model(
            model,
            preprocessed["X_test"],
            preprocessed["y_test"],
        )

        # Add config and history to result
        result_dict = result.to_dict()
        result_dict["experiment_id"] = self.id
        result_dict["config"] = self.config
        result_dict["training_history"] = {
            "best_epoch": history["best_epoch"],
            "total_epochs_trained": history["total_epochs_trained"],
        }

        return result_dict

    def _preprocess_data(self) -> Dict[str, np.ndarray]:
        """Preprocess data using DataPreprocessor."""
        data_config = self.config["data"]
        preprocessing_config = self.config["preprocessing"]

        # Create scaler using factory
        scaler = self.data_factory.create_scaler(preprocessing_config["scaler_type"])

        # Create preprocessor with dependencies
        preprocessor = DataPreprocessor(
            loader=IrisDataLoader(),
            validator=DataValidator(),
            splitter=DataSplitter(),
            scaler=scaler,
        )

        # Create preprocessing config
        preprocess_config = PreprocessConfig(
            feature_columns=data_config["feature_columns"],
            target_column=data_config["target_column"],
            train_ratio=preprocessing_config["split_strategy"]["train_ratio"],
            val_ratio=preprocessing_config["split_strategy"]["val_ratio"],
            test_ratio=preprocessing_config["split_strategy"]["test_ratio"],
            stratify=preprocessing_config.get("stratify", True),
            random_state=preprocessing_config.get("random_state", 42),
            scale_features=preprocessing_config.get("scale_features", True),
        )

        # Run preprocessing with config
        return preprocessor.process(data_config["path"], preprocess_config)

    def _build_model(self, input_size: int):
        """Build MLP model using MLPBuilder and ModelComponentFactory."""
        model_config = self.config["model"]
        training_config = self.config["training"]

        arch = model_config["architecture"]
        activations = model_config["activations"]

        # Build model using builder pattern
        builder = MLPBuilder(factory=self.model_factory)

        # Set input size
        builder.with_input_size(input_size)

        # Add hidden layers
        for layer_size in arch["hidden_layers"]:
            builder.add_hidden_layer(layer_size, activations["hidden"])

        # Set output layer
        builder.with_output_layer(arch["output_size"], activations["output"])

        # Set loss function
        builder.with_loss(training_config["loss_function"])

        # Set optimizer
        optimizer_config = training_config["optimizer"]
        builder.with_optimizer(
            optimizer_config["type"],
            learning_rate=optimizer_config["learning_rate"],
            momentum=optimizer_config.get("momentum", 0.9),
        )

        # Set random state for reproducibility - FROM CONFIG
        builder.with_random_state(training_config.get("random_state", 42))

        return builder.build()

    def _train_model(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict:
        """Train model using Trainer with early stopping."""
        training_config = self.config["training"]
        n_classes = self.config["model"]["architecture"]["output_size"]

        # Convert y to one-hot encoding for training
        y_train_onehot = self._to_onehot(y_train, n_classes)
        y_val_onehot = self._to_onehot(y_val, n_classes)

        # Create trainer with composition pattern - ALL FROM CONFIG
        trainer = Trainer(
            model=model,
            epochs=training_config["epochs"],
            batch_size=training_config["batch_size"],
            patience=training_config.get("patience", 30),
            random_state=training_config.get("random_state", 42),
        )

        # Train model
        history = trainer.fit(
            X_train=X_train,
            y_train=y_train_onehot,
            X_val=X_val,
            y_val=y_val_onehot,
        )

        return history

    def _evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ExperimentResult:
        """Evaluate model and return ExperimentResult."""
        # Get predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.compute_all(y_test, y_pred)

        # Create ExperimentResult
        return ExperimentResult(
            test_accuracy=metrics["accuracy"],
            test_precision=metrics["precision"],
            test_recall=metrics["recall"],
            test_f1=metrics["f1"],
            confusion_matrix=metrics["confusion_matrix"],
        )

    def _to_onehot(self, y: np.ndarray, n_classes: int) -> np.ndarray:
        """Convert class indices to one-hot encoding."""
        onehot = np.zeros((len(y), n_classes))
        onehot[np.arange(len(y)), y] = 1
        return onehot

    def get_config(self) -> Dict:
        """Get the configuration used for this experiment."""
        return self.config

    def get_id(self) -> int:
        """Get the experiment ID."""
        return self.id

    def __repr__(self) -> str:
        """String representation of experiment."""
        return f"Experiment(id={self.id})"
