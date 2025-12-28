"""Model training implementation with composition pattern and early stopping."""

import logging

import numpy as np

from src.models.base import IModel
from src.training.base import ITrainer

logger = logging.getLogger(__name__)


class Trainer(ITrainer):
    """Trainer for neural network models using composition pattern."""

    def __init__(
        self,
        model: IModel,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int | None = 10,
        random_state: int | None = None,
    ) -> None:
        """Initialize trainer with model and training parameters.

        Args:
            model: Neural network model to train
            epochs: Maximum number of training epochs
            batch_size: Size of mini-batches for training
            patience: Number of epochs without improvement before early stopping (None = disabled)
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.history: dict | None = None

        # Create own RNG generator instead of setting global seed
        if random_state is not None:
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = np.random.RandomState()

        logger.debug(
            f"Trainer initialized: epochs={epochs}, batch_size={batch_size}, "
            f"patience={patience}, random_state={random_state}"
        )

    def fit(  # noqa: PLR0915
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict:
        """Train model using mini-batch gradient descent with early stopping."""
        # Validate inputs
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train and y_train must have same number of samples: "
                f"{X_train.shape[0]} != {y_train.shape[0]}"
            )

        has_validation = X_val is not None and y_val is not None

        if has_validation and X_val.shape[0] != y_val.shape[0]:
            raise ValueError(
                f"X_val and y_val must have same number of samples: "
                f"{X_val.shape[0]} != {y_val.shape[0]}"
            )

        # Early stopping setup
        use_early_stopping = has_validation and self.patience is not None
        best_val_acc = 0.0
        best_weights = None
        best_epoch = 0
        patience_counter = 0
        stopped_early = False

        # Use model's loss function
        loss_fn = self.model.loss_function

        # Initialize history
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
        }

        if has_validation:
            self.history["val_loss"] = []
            self.history["val_accuracy"] = []

        n_samples = X_train.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        for epoch in range(self.epochs):
            # Shuffle training data
            indices = self.rng.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            # Mini-batch training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)

                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                # Forward pass
                y_pred = self.model.forward(X_batch)

                # Backward pass
                loss_grad = loss_fn.gradient(y_batch, y_pred)
                self.model.backward(loss_grad)

            # Evaluate on train set
            train_metrics = self.evaluate(X_train, y_train)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_accuracy"].append(train_metrics["accuracy"])

            # Evaluate on validation set if provided
            if has_validation:
                val_metrics = self.evaluate(X_val, y_val)
                val_acc = val_metrics["accuracy"]
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_accuracy"].append(val_acc)

                # Early stopping logic
                if use_early_stopping:
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_weights = self.model.get_parameters()
                        best_epoch = epoch + 1
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Check if should stop
                    if patience_counter >= self.patience:
                        stopped_early = True
                        break

        # Restore best weights if early stopping was used
        if use_early_stopping and best_weights is not None:
            self.model.set_parameters(best_weights)

        # Add metadata to history
        self.history["best_epoch"] = (
            best_epoch if use_early_stopping else len(self.history["train_loss"])
        )
        self.history["best_val_accuracy"] = best_val_acc if use_early_stopping else None
        self.history["stopped_early"] = stopped_early
        self.history["total_epochs_trained"] = epoch + 1

        return self.history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluate model on provided dataset."""
        # Forward pass
        y_pred = self.model.forward(X)

        # Compute loss
        loss = self.model.loss_function.compute(y, y_pred)

        # Compute accuracy
        y_pred_classes = self.model.predict(X)
        y_true_classes = np.argmax(y, axis=1) if y.ndim > 1 else y
        accuracy = np.mean(y_pred_classes == y_true_classes)

        return {"loss": float(loss), "accuracy": float(accuracy)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained model."""
        return self.model.predict(X)

    def get_model(self) -> IModel:
        """Get the underlying model."""
        return self.model

    def get_history(self) -> dict | None:
        """Get training history if available."""
        return self.history

    def __repr__(self) -> str:
        """Return string representation of trainer."""
        return (
            f"Trainer(model={type(self.model).__name__}, "
            f"epochs={self.epochs}, batch_size={self.batch_size}, patience={self.patience})"
        )
