"""Base interfaces for training components."""

from abc import ABC, abstractmethod

import numpy as np

from src.models.base import IModel


class ITrainer(ABC):
    """Interface for model training with composition pattern."""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict:
        """Train model and return history with metrics."""
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluate model and return metrics dictionary."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained model."""
        pass

    @abstractmethod
    def get_model(self) -> IModel:
        """Get the underlying model."""
        pass

    @abstractmethod
    def get_history(self) -> dict | None:
        """Get training history if available."""
        pass
