"""Base interfaces for analysis components."""

from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd

from src.analysis.plot_config import PlotConfig


class IResultsConverter(ABC):
    """Interface for converting experiment results to different formats."""

    @abstractmethod
    def load(self, input_path: str) -> List[Dict]:
        """Load results from file."""
        pass

    @abstractmethod
    def convert(self, results: List[Dict]) -> pd.DataFrame:
        """Convert results to DataFrame."""
        pass

    @abstractmethod
    def save(self, df: pd.DataFrame, output_path: str) -> None:
        """Save DataFrame to file."""
        pass


class IResultsAnalyzer(ABC):
    """Interface for analyzing experiment results."""

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> None:
        """Run analysis on DataFrame (side-effect: logs/outputs)."""
        pass


class ITableGenerator(ABC):
    """Interface for generating tables from experiment results."""

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> str:
        """Generate table from DataFrame."""
        pass

    @abstractmethod
    def save(self, table: str, output_path: str) -> None:
        """Save generated table to file."""
        pass


class IPlotGenerator(ABC):
    """Interface for generating plots from experiment results."""

    @abstractmethod
    def generate(self, df: pd.DataFrame, config: PlotConfig) -> None:
        """Generate plot from DataFrame according to config."""
        pass

    @abstractmethod
    def save(self, output_path: str) -> None:
        """Save generated plot to file."""
        pass
