"""Configuration classes for data preprocessing."""

from dataclasses import dataclass
from typing import ClassVar

from src.data.base import IPreprocessConfig


@dataclass
class PreprocessConfig(IPreprocessConfig):
    """Configuration for data preprocessing with validation."""

    feature_columns: list[str]
    target_column: str
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify: bool = True
    random_state: int = 42
    scale_features: bool = True

    EPSILON: ClassVar[float] = 1e-6

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration parameters."""
        self._validate_ratios()
        self._validate_columns()
        self._validate_random_state()

    def _validate_ratios(self) -> None:
        """Validate train/val/test ratios."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > self.EPSILON:
            raise ValueError(
                f"Ratios must sum to 1.0, got {total:.6f} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )

        if not (0 < self.train_ratio < 1):
            raise ValueError(
                f"train_ratio must be between 0 and 1 (exclusive), got {self.train_ratio}"
            )

        if not (0 <= self.val_ratio < 1):
            raise ValueError(
                f"val_ratio must be between 0 (inclusive) and 1 (exclusive), got {self.val_ratio}"
            )

        if not (0 < self.test_ratio < 1):
            raise ValueError(
                f"test_ratio must be between 0 and 1 (exclusive), got {self.test_ratio}"
            )

    def _validate_columns(self) -> None:
        """Validate feature and target columns."""
        if not self.feature_columns:
            raise ValueError("feature_columns cannot be empty")

        if not isinstance(self.feature_columns, list):
            raise TypeError(
                f"feature_columns must be a list, got {type(self.feature_columns).__name__}"
            )

        if not self.target_column:
            raise ValueError("target_column cannot be empty")

        if not isinstance(self.target_column, str):
            raise TypeError(
                f"target_column must be a string, got {type(self.target_column).__name__}"
            )

        if self.target_column in self.feature_columns:
            raise ValueError(f"target_column '{self.target_column}' cannot be in feature_columns")

    def _validate_random_state(self) -> None:
        """Validate random state."""
        if not isinstance(self.random_state, int):
            raise TypeError(
                f"random_state must be an integer, got {type(self.random_state).__name__}"
            )

        if self.random_state < 0:
            raise ValueError(f"random_state must be non-negative, got {self.random_state}")
