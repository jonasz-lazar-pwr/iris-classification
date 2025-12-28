"""Data validation and cleaning utilities."""

from typing import Any

import numpy as np
import pandas as pd

from src.data.base import IDataValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator(IDataValidator):
    """Data validator for cleaning and quality checks."""

    def __init__(self) -> None:
        """Initialize data validator."""
        self._report: dict[str, Any] = {}

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean DataFrame."""
        self._report = {}

        if df.empty:
            logger.warning("Empty DataFrame provided - skipping validation")
            self._report = {
                "rows_removed": 0,
                "missing_values": {},
                "duplicates": 0,
                "outliers": {},
            }
            return df

        original_size = len(df)

        df = self._remove_missing_values(df)
        df = self._remove_duplicates(df)
        self._detect_outliers(df)

        rows_removed = original_size - len(df)
        self._report["rows_removed"] = rows_removed

        return df

    def get_validation_report(self) -> dict[str, Any]:
        """Get validation statistics report."""
        return self._report.copy()

    def _remove_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing values."""
        missing = df.isnull().sum()
        missing_dict = {col: int(count) for col, count in missing.items() if count > 0}

        self._report["missing_values"] = missing_dict

        total_missing = missing.sum()
        if total_missing > 0:
            logger.warning(f"Found {total_missing} missing values in {len(missing_dict)} columns")
            df = df.dropna()

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        duplicates = df.duplicated().sum()
        self._report["duplicates"] = int(duplicates)

        if duplicates > 0:
            df = df.drop_duplicates()

        return df

    def _detect_outliers(self, df: pd.DataFrame) -> None:
        """Detect outliers using IQR method (detection only, does not remove)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            logger.warning("No numeric columns for outlier detection")
            self._report["outliers"] = {}
            return

        outliers_dict = {}

        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outliers_dict[col] = int(outliers)

        self._report["outliers"] = outliers_dict

        # total_outliers = sum(outliers_dict.values())
        # if total_outliers > 0:
        #     logger.info(
        #         f"Detected {total_outliers} outliers across {len(numeric_cols)} columns "
        #         "(not removed)"
        #     )
