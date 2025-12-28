from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import numpy as np
import pandas as pd

from src.analysis.base import IResultsAnalyzer


@dataclass(frozen=True)
class BestConfigsDistributionConfig:
    """Configuration for analyzing distributions among top-performing configs."""

    acc_threshold: float = 1.0
    acc_column: str = "test_accuracy"
    columns: List[str] = field(
        default_factory=lambda: [
            "split_strategy",
            "scaler_type",
            "layers",
            "activation",
            "learning_rate",
            "momentum",
            "batch_size",
        ]
    )
    strict_columns: bool = True
    atol: float = 1e-12  # tolerance for float comparisons


class BestConfigsDistributionAnalyzer(IResultsAnalyzer):
    """
    Logs frequency distributions of selected hyperparameters among experiments
    with test_accuracy == acc_threshold (within tolerance).
    """

    def __init__(self, config: BestConfigsDistributionConfig, logger) -> None:
        self._config = config
        self._logger = logger

    def analyze(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        # Validate required accuracy column
        if self._config.acc_column not in df.columns:
            raise ValueError(f"Missing required column '{self._config.acc_column}' in DataFrame")

        # Validate hyperparameter columns
        missing = [c for c in self._config.columns if c not in df.columns]
        if missing:
            msg = f"Missing required hyperparameter columns: {missing}"
            if self._config.strict_columns:
                raise ValueError(msg)
            self._logger.warning(msg + " (skipping missing columns)")

        # Robust numeric conversion for accuracy
        acc = pd.to_numeric(df[self._config.acc_column], errors="coerce").to_numpy()
        mask = np.isclose(acc, self._config.acc_threshold, atol=self._config.atol, rtol=0.0)

        best_df = df.loc[mask].copy()
        total_best = len(best_df)
        total_all = len(df)

        self._logger.info(
            f"Perfect configs filter: {self._config.acc_column} ~= {self._config.acc_threshold}"
        )
        self._logger.info(f"  Total experiments: {total_all}")
        self._logger.info(f"  Perfect experiments: {total_best}")

        if total_best == 0:
            self._logger.warning("No perfect configurations found. Nothing to summarize.")
            return

        # Log distributions
        for col in self._config.columns:
            if col not in best_df.columns:
                continue

            counts = best_df[col].value_counts(dropna=False)
            self._logger.info(f"\nDistribution for '{col}' (n={total_best}):")

            for value, count in counts.items():
                pct = (count / total_best) * 100.0
                value_str = self._format_value(value)
                self._logger.info(f"  - {value_str}: {count} ({pct:.1f}%)")

    @staticmethod
    def _format_value(value: Any) -> str:
        if pd.isna(value):
            return "<NA>"
        # Keep floats readable (e.g., 0.05 instead of 0.0500000001)
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)
