"""Base interfaces for data processing components."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class IDataLoader(ABC):
    """Interface for data loading."""

    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        """Load data from file."""
        pass


class IDataValidator(ABC):
    """Interface for data validation."""

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data."""
        pass

    @abstractmethod
    def get_validation_report(self) -> dict[str, Any]:
        """Get validation statistics."""
        pass


class IScaler(ABC):
    """Interface for data scaling."""

    @abstractmethod
    def fit(self, x: np.ndarray) -> "IScaler":
        """Fit scaler on training data."""
        pass

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform data."""
        pass

    @abstractmethod
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        pass


class IDataSplitter(ABC):
    """Interface for data splitting."""

    @abstractmethod
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
        """Split data into train/val/test."""
        pass


class IDataPreprocessor(ABC):
    """Interface for data preprocessing orchestration."""

    @abstractmethod
    def process(self, data_path: str, config: Any) -> dict[str, Any]:
        """Execute full preprocessing pipeline with given config."""
        pass


class IDataComponentFactory(ABC):
    """Interface for creating data processing components."""

    @abstractmethod
    def create_loader(self, loader_type: str, **kwargs) -> IDataLoader:
        """Create data loader instance."""
        pass

    @abstractmethod
    def create_scaler(self, scaler_type: str, **kwargs) -> IScaler:
        """Create scaler instance."""
        pass

    @abstractmethod
    def create_validator(self, validator_type: str, **kwargs) -> IDataValidator:
        """Create validator instance."""
        pass

    @abstractmethod
    def create_splitter(self, splitter_type: str, **kwargs) -> IDataSplitter:
        """Create splitter instance."""
        pass

    @abstractmethod
    def list_available_components(self) -> dict[str, list[str]]:
        """List all available component types."""
        pass


class IPreprocessConfig(ABC):
    """Interface for preprocessing configuration."""

    @abstractmethod
    def validate(self) -> None:
        """Validate configuration parameters."""
        pass
