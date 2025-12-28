"""Classification metrics implementation with simplified API."""

import numpy as np

from src.evaluation.base import IMetrics


class MetricsCalculator(IMetrics):
    """Calculator for multi-class classification metrics with auto-detection."""

    def __init__(self, class_names: list[str] | None = None) -> None:
        """Initialize metrics calculator.

        Args:
            class_names: Optional list of class names for display
        """
        self.class_names = class_names

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute classification accuracy."""
        return float(np.mean(y_true == y_pred))

    def precision(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str | None = "macro"
    ) -> float | np.ndarray:
        """Compute precision score (TP / (TP + FP))."""
        cm = self.confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]

        # Per-class precision
        precisions = np.zeros(n_classes)
        for i in range(n_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp

            if tp + fp > 0:
                precisions[i] = tp / (tp + fp)
            else:
                precisions[i] = 0.0

        if average == "macro":
            return float(np.mean(precisions))
        elif average == "weighted":
            support = np.sum(cm, axis=1)
            return float(np.sum(precisions * support) / np.sum(support))
        elif average is None:
            return precisions
        else:
            raise ValueError(f"Unknown average method: {average}")

    def recall(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str | None = "macro"
    ) -> float | np.ndarray:
        """Compute recall score (TP / (TP + FN))."""
        cm = self.confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]

        # Per-class recall
        recalls = np.zeros(n_classes)
        for i in range(n_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp

            if tp + fn > 0:
                recalls[i] = tp / (tp + fn)
            else:
                recalls[i] = 0.0

        if average == "macro":
            return float(np.mean(recalls))
        elif average == "weighted":
            support = np.sum(cm, axis=1)
            return float(np.sum(recalls * support) / np.sum(support))
        elif average is None:
            return recalls
        else:
            raise ValueError(f"Unknown average method: {average}")

    def f1_score(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str | None = "macro"
    ) -> float | np.ndarray:
        """Compute F1 score (harmonic mean of precision and recall)."""
        prec = self.precision(y_true, y_pred, average=None)
        rec = self.recall(y_true, y_pred, average=None)
        n_classes = len(prec)

        # Per-class F1
        f1_scores = np.zeros(n_classes)
        for i in range(n_classes):
            if prec[i] + rec[i] > 0:
                f1_scores[i] = 2 * (prec[i] * rec[i]) / (prec[i] + rec[i])
            else:
                f1_scores[i] = 0.0

        if average == "macro":
            return float(np.mean(f1_scores))
        elif average == "weighted":
            cm = self.confusion_matrix(y_true, y_pred)
            support = np.sum(cm, axis=1)
            return float(np.sum(f1_scores * support) / np.sum(support))
        elif average is None:
            return f1_scores
        else:
            raise ValueError(f"Unknown average method: {average}")

    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute confusion matrix for multi-class classification."""
        # Auto-detect number of classes
        n_classes = max(y_true.max(), y_pred.max()) + 1
        matrix = np.zeros((n_classes, n_classes), dtype=int)

        for true_class, pred_class in zip(y_true, y_pred, strict=True):
            matrix[true_class, pred_class] += 1

        return matrix

    def compute_all(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute all metrics for ExperimentResult compatibility.

        Returns dict with keys: accuracy, precision, recall, f1, confusion_matrix
        (using macro averaging for scalar metrics).
        """
        cm = self.confusion_matrix(y_true, y_pred)

        return {
            "accuracy": self.accuracy(y_true, y_pred),
            "precision": self.precision(y_true, y_pred, average="macro"),
            "recall": self.recall(y_true, y_pred, average="macro"),
            "f1": self.f1_score(y_true, y_pred, average="macro"),
            "confusion_matrix": cm.tolist(),  # Convert to list for JSON serialization
        }

    def compute_detailed(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute comprehensive metrics report with per-class breakdown."""
        cm = self.confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]

        # Generate class names if not provided
        class_names = self.class_names or [f"Class {i}" for i in range(n_classes)]

        prec_per_class = self.precision(y_true, y_pred, average=None)
        rec_per_class = self.recall(y_true, y_pred, average=None)
        f1_per_class = self.f1_score(y_true, y_pred, average=None)
        support = np.sum(cm, axis=1)

        report = {
            "confusion_matrix": cm.tolist(),
            "accuracy": self.accuracy(y_true, y_pred),
            "macro_avg": {
                "precision": float(np.mean(prec_per_class)),
                "recall": float(np.mean(rec_per_class)),
                "f1_score": float(np.mean(f1_per_class)),
            },
            "weighted_avg": {
                "precision": self.precision(y_true, y_pred, average="weighted"),
                "recall": self.recall(y_true, y_pred, average="weighted"),
                "f1_score": self.f1_score(y_true, y_pred, average="weighted"),
            },
            "per_class": {},
        }

        for i, class_name in enumerate(class_names):
            report["per_class"][class_name] = {
                "precision": float(prec_per_class[i]),
                "recall": float(rec_per_class[i]),
                "f1_score": float(f1_per_class[i]),
                "support": int(support[i]),
            }

        return report

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MetricsCalculator(class_names={self.class_names})"
