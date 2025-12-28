import json

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.analysis.base import IResultsConverter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class JSONToCSVConverter(IResultsConverter):
    """Convert experiment results from JSON to CSV format."""

    def load(self, input_path: str) -> List[Dict]:
        """Load results from JSON file."""
        path = Path(input_path)

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with path.open("r") as f:
            results = json.load(f)

        logger.info(f"Loaded {len(results)} experiments from {input_path}")
        return results

    def convert(self, results: List[Dict]) -> pd.DataFrame:
        """Convert results to DataFrame with flattened structure."""
        rows = []

        for result in results:
            row = self._extract_row(result)
            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(f"Converted {len(df)} experiments to DataFrame")

        return df

    def save(self, df: pd.DataFrame, output_path: str) -> None:
        """Save DataFrame to CSV file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(path, index=False)
        logger.info(f"Saved results to {output_path}")

    def _extract_row(self, result: Dict) -> Dict:
        """Extract single row from experiment result."""
        config = result["config"]

        # Configuration parameters
        row = {
            "experiment_id": result["experiment_id"],
            "scaler_type": config["preprocessing"]["scaler_type"],
            "split_strategy": self._format_split(config["preprocessing"]["split_strategy"]),
            "layers": str(config["model"]["architecture"]["hidden_layers"]),
            "activation": config["model"]["activations"]["hidden"],
            "learning_rate": config["training"]["optimizer"]["learning_rate"],
            "momentum": config["training"]["optimizer"]["momentum"],
            "batch_size": config["training"]["batch_size"],
            "max_epochs": config["training"]["epochs"],
        }

        # Test metrics
        row.update(
            {
                "test_accuracy": result["test_accuracy"],
                "test_f1": result["test_f1"],
                "test_precision": result["test_precision"],
                "test_recall": result["test_recall"],
            }
        )

        # Training history
        history = result.get("training_history", {})
        row.update(
            {
                "best_epoch": history.get("best_epoch"),
                "total_epochs_trained": history.get("total_epochs_trained"),
            }
        )

        # Confusion matrix (flatten to separate columns)
        cm = result.get("confusion_matrix", [])
        row.update(self._flatten_confusion_matrix(cm))

        return row

    def _format_split(self, split_strategy: Dict) -> str:
        """Format split strategy as string '0.8/0.1/0.1'."""
        train = split_strategy["train_ratio"]
        val = split_strategy["val_ratio"]
        test = split_strategy["test_ratio"]
        return f"{train}/{val}/{test}"

    def _flatten_confusion_matrix(self, cm: List[List[int]]) -> Dict:
        """Flatten confusion matrix to individual columns."""
        if not cm:
            return {}

        flat = {}
        class_names = ["setosa", "versicolor", "virginica"]

        for i, true_class in enumerate(class_names):
            for j, pred_class in enumerate(class_names):
                key = f"cm_{true_class}_{pred_class}"
                flat[key] = cm[i][j] if i < len(cm) and j < len(cm[i]) else 0

        return flat

    def __repr__(self) -> str:
        return "JSONToCSVConverter()"
