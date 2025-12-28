"""Data loaders for various file formats."""

from pathlib import Path
from typing import ClassVar

import pandas as pd

from src.data.base import IDataLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CSVDataLoader(IDataLoader):
    """CSV data loader implementation."""

    def __init__(self, column_names: list[str] | None = None, encoding: str = "utf-8") -> None:
        """Initialize CSV loader with optional column names and encoding."""
        self._column_names = column_names
        self._encoding = encoding

    def load(self, path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        file_path = Path(path)
        if not file_path.exists():
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"Data file not found: {path}")

        try:
            df = pd.read_csv(path, header=None, names=self._column_names, encoding=self._encoding)
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise

        return df


class IrisDataLoader(CSVDataLoader):
    """Specialized loader for Iris dataset."""

    COLUMN_NAMES: ClassVar[list[str]] = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]

    SPECIES_MAP: ClassVar[dict[str, int]] = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2,
    }

    def __init__(self) -> None:
        """Initialize Iris dataset loader."""
        super().__init__(column_names=self.COLUMN_NAMES)

    def load(self, path: str) -> pd.DataFrame:
        """Load Iris dataset with label encoding."""
        df = super().load(path)

        original_size = len(df)

        # Create encoded species column
        df["species_encoded"] = df["species"].map(self.SPECIES_MAP)

        unmapped_count = df["species_encoded"].isnull().sum()
        if unmapped_count > 0:
            logger.warning(
                f"Found {unmapped_count} rows with unknown species labels - removing them"
            )
            df = df.dropna(subset=["species_encoded"])

        df["species_encoded"] = df["species_encoded"].astype(int)

        rows_removed = original_size - len(df)
        if rows_removed > 0:
            logger.info(f"Removed {rows_removed} rows with invalid species labels")

        # Replace string column with encoded integers for downstream processing
        df["species"] = df["species_encoded"]

        return df
