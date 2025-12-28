"""Data preprocessing orchestrator with dependency injection."""

from typing import Any

import numpy as np
import pandas as pd

from src.data.base import (
    IDataLoader,
    IDataPreprocessor,
    IDataSplitter,
    IDataValidator,
    IPreprocessConfig,
    IScaler,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor(IDataPreprocessor):
    """Orchestrates data loading, validation, scaling, and splitting using DI."""

    def __init__(
        self,
        loader: IDataLoader,
        validator: IDataValidator,
        splitter: IDataSplitter,
        scaler: IScaler | None = None,
    ) -> None:
        """Initialize preprocessor with injected dependencies."""
        self._loader = loader
        self._validator = validator
        self._splitter = splitter
        self._scaler = scaler

    def process(self, data_path: str, config: IPreprocessConfig) -> dict[str, Any]:
        """Execute full preprocessing pipeline."""
        config.validate()

        df = self._load_data(data_path)
        df, validation_report = self._validate_data(df)
        X, y = self._extract_features_target(df, config.feature_columns, config.target_column)
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(
            X,
            y,
            config.train_ratio,
            config.val_ratio,
            config.test_ratio,
            config.stratify,
            config.random_state,
        )
        X_train, X_val, X_test = self._scale_features(X_train, X_val, X_test, config.scale_features)

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "validation_report": validation_report,
            "feature_columns": config.feature_columns,
            "target_column": config.target_column,
            "scaler": self._scaler,
        }

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data using injected loader."""
        return self._loader.load(data_path)

    def _validate_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Validate data using injected validator."""
        df = self._validator.validate(df)
        report = self._validator.get_validation_report()
        return df, report

    def _extract_features_target(
        self, df: pd.DataFrame, feature_columns: list[str], target_column: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features and target arrays from DataFrame."""
        missing_features = set(feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        X = df[feature_columns].values
        y = df[target_column].values

        logger.debug(f"Features shape: {X.shape}, target shape: {y.shape}")

        return X, y

    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        stratify: bool,
        random_state: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data using injected splitter."""
        return self._splitter.split(
            X=X,
            y=y,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stratify=stratify,
            random_state=random_state,
        )

    def _scale_features(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        scale_features: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scale features using injected scaler if enabled."""
        if not scale_features or self._scaler is None:
            return X_train, X_val, X_test

        X_train = self._scaler.fit_transform(X_train)
        X_val = self._scaler.transform(X_val) if len(X_val) > 0 else X_val
        X_test = self._scaler.transform(X_test)

        return X_train, X_val, X_test

    def __repr__(self) -> str:
        """String representation of the preprocessor."""
        return (
            f"DataPreprocessor("
            f"loader={type(self._loader).__name__}, "
            f"validator={type(self._validator).__name__}, "
            f"splitter={type(self._splitter).__name__}, "
            f"scaler={type(self._scaler).__name__ if self._scaler else None})"
        )
