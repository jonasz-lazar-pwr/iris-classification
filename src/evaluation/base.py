"""Base interfaces for evaluation components."""

from abc import ABC, abstractmethod

import numpy as np


class IMetrics(ABC):
    """Interface for classification metrics."""

    @abstractmethod
    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy score."""
        pass

    @abstractmethod
    def precision(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str | None = "macro"
    ) -> float | np.ndarray:
        """Compute precision score."""
        pass

    @abstractmethod
    def recall(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str | None = "macro"
    ) -> float | np.ndarray:
        """Compute recall score."""
        pass

    @abstractmethod
    def f1_score(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str | None = "macro"
    ) -> float | np.ndarray:
        """Compute F1 score."""
        pass

    @abstractmethod
    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute confusion matrix."""
        pass

    @abstractmethod
    def compute_all(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute all metrics and return comprehensive report."""
        pass
