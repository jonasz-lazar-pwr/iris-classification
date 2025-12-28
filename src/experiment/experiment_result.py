"""Experiment result data structure implementation."""

from typing import Any, Dict, List

from src.experiment.base import IExperimentResult


class ExperimentResult(IExperimentResult):
    """Data class for experiment results with type safety and validation."""

    def __init__(
        self,
        test_accuracy: float,
        test_precision: float,
        test_recall: float,
        test_f1: float,
        confusion_matrix: List[List[int]],
        training_history: Dict[str, Any] | None = None,
    ):
        """Initialize experiment result with metrics rounded to 4 decimal places."""
        # Validate metric ranges
        self._validate_metric(test_accuracy, "test_accuracy")
        self._validate_metric(test_precision, "test_precision")
        self._validate_metric(test_recall, "test_recall")
        self._validate_metric(test_f1, "test_f1")

        # Validate confusion matrix
        self._validate_confusion_matrix(confusion_matrix)

        # Store metrics rounded to 4 decimal places
        self.test_accuracy = round(test_accuracy, 4)
        self.test_precision = round(test_precision, 4)
        self.test_recall = round(test_recall, 4)
        self.test_f1 = round(test_f1, 4)
        self.confusion_matrix = confusion_matrix
        self.training_history = training_history or {}

    def to_dict(self) -> Dict:
        """Convert result to dictionary for JSON serialization."""
        result = {
            "test_accuracy": self.test_accuracy,
            "test_precision": self.test_precision,
            "test_recall": self.test_recall,
            "test_f1": self.test_f1,
            "confusion_matrix": self.confusion_matrix,
        }

        # Save only best_epoch and total_epochs_trained
        if self.training_history:
            result["training_history"] = {
                "best_epoch": self.training_history.get("best_epoch"),
                "total_epochs_trained": self.training_history.get("total_epochs_trained"),
            }

        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "ExperimentResult":
        """Create ExperimentResult instance from dictionary."""
        required_keys = [
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "confusion_matrix",
        ]

        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise KeyError(f"Missing required keys: {missing_keys}")

        return cls(
            test_accuracy=data["test_accuracy"],
            test_precision=data["test_precision"],
            test_recall=data["test_recall"],
            test_f1=data["test_f1"],
            confusion_matrix=data["confusion_matrix"],
            training_history=data.get("training_history"),
        )

    def get_accuracy(self) -> float:
        """Return test accuracy metric."""
        return self.test_accuracy

    def get_precision(self) -> float:
        """Return test precision metric."""
        return self.test_precision

    def get_recall(self) -> float:
        """Return test recall metric."""
        return self.test_recall

    def get_f1(self) -> float:
        """Return test F1 score metric."""
        return self.test_f1

    def get_confusion_matrix(self) -> List[List[int]]:
        """Return confusion matrix."""
        return self.confusion_matrix

    def get_training_history(self) -> Dict[str, Any]:
        """Return minimal training history metadata."""
        return self.training_history

    def _validate_metric(self, value: float, name: str) -> None:
        """Validate that metric value is in valid range [0.0, 1.0]."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric, got {type(value).__name__}")

        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in range [0.0, 1.0], got {value}")

    def _validate_confusion_matrix(self, matrix: List[List[int]]) -> None:
        """Validate confusion matrix structure and values."""
        if not isinstance(matrix, list):
            raise ValueError(f"Confusion matrix must be a list, got {type(matrix).__name__}")

        if not matrix:
            raise ValueError("Confusion matrix cannot be empty")

        # Check that all rows are lists
        if not all(isinstance(row, list) for row in matrix):
            raise ValueError("All rows in confusion matrix must be lists")

        # Check that matrix is square
        n_rows = len(matrix)
        n_cols = len(matrix[0])

        if n_rows != n_cols:
            raise ValueError(f"Confusion matrix must be square, got {n_rows}x{n_cols}")

        # Check that all rows have same length
        if not all(len(row) == n_cols for row in matrix):
            raise ValueError("All rows in confusion matrix must have same length")

        # Check that all values are integers
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if not isinstance(val, int):
                    raise ValueError(
                        f"Confusion matrix values must be integers, "
                        f"got {type(val).__name__} at position [{i}][{j}]"
                    )

                if val < 0:
                    raise ValueError(
                        f"Confusion matrix values must be non-negative, "
                        f"got {val} at position [{i}][{j}]"
                    )

    def __repr__(self) -> str:
        """Return string representation of experiment result."""
        return (
            f"ExperimentResult("
            f"accuracy={self.test_accuracy:.4f}, "
            f"precision={self.test_precision:.4f}, "
            f"recall={self.test_recall:.4f}, "
            f"f1={self.test_f1:.4f})"
        )

    def __eq__(self, other) -> bool:
        """Check equality with another ExperimentResult instance."""
        if not isinstance(other, ExperimentResult):
            return False

        return (
            self.test_accuracy == other.test_accuracy
            and self.test_precision == other.test_precision
            and self.test_recall == other.test_recall
            and self.test_f1 == other.test_f1
            and self.confusion_matrix == other.confusion_matrix
        )

    # Explicitly disable hashing since we implement __eq__ and have mutable data
    __hash__ = None
