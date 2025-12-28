"""Feature scaling implementations."""

import numpy as np

from src.data.base import IScaler
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StandardScaler(IScaler):
    """Standardize features by removing mean and scaling to unit variance."""

    def __init__(self) -> None:
        """Initialize StandardScaler."""
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self._fitted: bool = False

    def fit(self, x: np.ndarray) -> "StandardScaler":
        """Fit scaler on training data."""
        if x.size == 0:
            raise ValueError("Cannot fit on empty array")

        if x.ndim == 1:
            x = x.reshape(-1, 1)
            logger.debug("Reshaped 1D array to 2D for fitting")

        self.n_features_in_ = x.shape[1]
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0)

        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)

        self._fitted = True

        logger.debug(f"Mean: {self.mean_}")
        logger.debug(f"Std: {self.std_}")

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        if not self._fitted:
            raise ValueError("Scaler must be fitted before transform")

        if x.size == 0:
            return x

        original_ndim = x.ndim
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            logger.debug("Reshaped 1D array to 2D for transformation")

        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {x.shape[1]} features, but scaler is fitted on "
                f"{self.n_features_in_} features"
            )

        x_scaled = (x - self.mean_) / self.std_

        if original_ndim == 1:
            x_scaled = x_scaled.ravel()

        logger.debug(f"Transformed data with shape {x.shape}")

        return x_scaled

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data in one step."""
        return self.fit(x).transform(x)

    def __repr__(self) -> str:
        """String representation of the scaler."""
        if self._fitted:
            return f"StandardScaler(fitted=True, n_features={self.n_features_in_})"
        return "StandardScaler(fitted=False)"


class MinMaxScaler(IScaler):
    """Scale features to a [0, 1] range."""

    def __init__(self) -> None:
        """Initialize MinMaxScaler."""
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self._fitted: bool = False

    def fit(self, x: np.ndarray) -> "MinMaxScaler":
        """Fit scaler on training data."""
        if x.size == 0:
            raise ValueError("Cannot fit on empty array")

        if x.ndim == 1:
            x = x.reshape(-1, 1)
            logger.debug("Reshaped 1D array to 2D for fitting")

        self.n_features_in_ = x.shape[1]
        self.min_ = np.min(x, axis=0)
        self.max_ = np.max(x, axis=0)

        self._fitted = True

        logger.debug(f"Min: {self.min_}")
        logger.debug(f"Max: {self.max_}")

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        if not self._fitted:
            raise ValueError("Scaler must be fitted before transform")

        if x.size == 0:
            return x

        original_ndim = x.ndim
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            logger.debug("Reshaped 1D array to 2D for transformation")

        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {x.shape[1]} features, but scaler is fitted on "
                f"{self.n_features_in_} features"
            )

        range_ = self.max_ - self.min_

        range_ = np.where(range_ == 0, 1.0, range_)

        x_scaled = (x - self.min_) / range_

        if original_ndim == 1:
            x_scaled = x_scaled.ravel()

        logger.debug(f"Transformed data with shape {x.shape}")

        return x_scaled

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data in one step."""
        return self.fit(x).transform(x)

    def __repr__(self) -> str:
        """String representation of the scaler."""
        if self._fitted:
            return f"MinMaxScaler(fitted=True, n_features={self.n_features_in_})"
        return "MinMaxScaler(fitted=False)"
