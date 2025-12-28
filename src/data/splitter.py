"""Data splitting for train/validation/test sets."""

from typing import ClassVar

import numpy as np

from sklearn.model_selection import train_test_split

from src.data.base import IDataSplitter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataSplitter(IDataSplitter):
    """Split data into train/validation/test sets with stratification support."""

    EPSILON: ClassVar[float] = 1e-6
    MIN_SAMPLES_PER_CLASS: ClassVar[int] = 2

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        stratify: bool = True,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/validation/test sets."""
        self._validate_inputs(X, y, train_ratio, val_ratio, test_ratio, stratify)

        stratify_labels = y if stratify else None

        temp_ratio = val_ratio + test_ratio
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=temp_ratio,
            stratify=stratify_labels,
            random_state=random_state,
        )

        if val_ratio > self.EPSILON:
            val_test_ratio = test_ratio / temp_ratio

            X_val, X_test, y_val, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=val_test_ratio,
                stratify=y_temp if stratify else None,
                random_state=random_state,
            )
        else:
            X_val = np.empty((0, X.shape[1]), dtype=X.dtype)
            y_val = np.array([], dtype=y.dtype)
            X_test, y_test = X_temp, y_temp

        if stratify:
            self._log_class_distribution(y_train, y_val, y_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _validate_inputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        stratify: bool,
    ) -> None:
        """Validate input parameters."""
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same length: {X.shape[0]} vs {y.shape[0]}")

        if X.shape[0] == 0:
            raise ValueError("Cannot split empty arrays")

        if train_ratio <= 0 or train_ratio >= 1:
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

        if val_ratio < 0 or val_ratio >= 1:
            raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

        if test_ratio <= 0 or test_ratio >= 1:
            raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")

        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > self.EPSILON:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

        if stratify:
            _, counts = np.unique(y, return_counts=True)
            min_count = counts.min()
            if min_count < self.MIN_SAMPLES_PER_CLASS:
                raise ValueError(
                    f"Stratification requires at least {self.MIN_SAMPLES_PER_CLASS} samples per class, "
                    f"but found class with only {min_count} sample(s). "
                    f"Consider using stratify=False or providing more data."
                )

        min_samples_per_split = 1
        if X.shape[0] * test_ratio < min_samples_per_split:
            logger.warning(f"Test set will have very few samples: {int(X.shape[0] * test_ratio)}")

    def _log_class_distribution(
        self, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray
    ) -> None:
        """Log class distribution across splits."""
        unique_classes = np.unique(np.concatenate([y_train, y_val, y_test]))

        logger.debug("Class distribution per split:")
        for split_name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
            if len(y_split) > 0:
                distribution = {int(cls): int(np.sum(y_split == cls)) for cls in unique_classes}
                logger.debug(f"  {split_name}: {distribution}")
