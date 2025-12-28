"""Generate LaTeX tables from experiment results."""

from pathlib import Path

import pandas as pd

from src.analysis.base import ITableGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LaTeXTableGenerator(ITableGenerator):
    """Generate LaTeX table with top configurations."""

    # Formatting constants
    LR_PRECISION_THRESHOLD = 0.01  # Threshold for LR decimal precision

    def __init__(
        self,
        top_n: int = 40,
        sort_by: str = "test_accuracy",
        ascending: bool = False,
    ) -> None:
        """Initialize table generator.

        Args:
            top_n: Number of top configurations to include
            sort_by: Column to sort by
            ascending: Sort order (False = descending, best first)
        """
        self.top_n = top_n
        self.sort_by = sort_by
        self.ascending = ascending

    def generate(self, df: pd.DataFrame) -> str:
        """Generate LaTeX table from DataFrame.

        Args:
            df: DataFrame with experiment results

        Returns:
            LaTeX table as string
        """
        # Sort and select top N
        df_sorted = df.sort_values(by=self.sort_by, ascending=self.ascending)
        df_top = df_sorted.head(self.top_n).copy()

        # Add ranking column
        df_top.insert(0, "rank", range(1, len(df_top) + 1))

        logger.info(f"Generating LaTeX table for top {len(df_top)} configurations")

        # Generate table
        table = self._generate_table_header()
        table += self._generate_table_rows(df_top)
        table += self._generate_table_footer()

        return table

    def save(self, table: str, output_path: str) -> None:
        """Save LaTeX table to file.

        Args:
            table: LaTeX table string
            output_path: Output file path
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            f.write(table)

        logger.info(f"LaTeX table saved to {output_path}")

    def _generate_table_header(self) -> str:
        """Generate LaTeX table header."""
        return """\\begin{table}[H]
\\centering
\\caption{Najlepsze konfiguracje modelu MLP dla klasyfikacji Iris}
\\label{tab:best_configs}
\\begin{tabular}{c l c l c c c c c c}
\\toprule
No & Split & Scal & Layers & Act & LR & Mom & Batch & Epochs & Acc \\\\
\\midrule
"""

    def _generate_table_rows(self, df: pd.DataFrame) -> str:
        """Generate table rows from DataFrame.

        Args:
            df: DataFrame with top configurations

        Returns:
            LaTeX table rows
        """
        rows = []

        for _, row in df.iterrows():
            latex_row = self._format_row(row)
            rows.append(latex_row)

        return "\n".join(rows) + "\n"

    def _format_row(self, row: pd.Series) -> str:
        """Format single row as LaTeX."""
        # Format values in algorithm flow order
        rank = int(row["rank"])
        split = self._format_split(row["split_strategy"])
        scal = self._format_scaler(row["scaler_type"])
        layers = self._format_layers(row["layers"])
        act = row["activation"]
        lr = self._format_lr(row["learning_rate"])
        mom = self._format_momentum(row["momentum"])
        batch = int(row["batch_size"])

        # Use total_epochs_trained instead of max_epochs
        epochs = self._format_epochs(
            row["best_epoch"], row.get("total_epochs_trained", row.get("max_epochs", 100))
        )

        acc = f"{row['test_accuracy']:.1%}".replace("%", "\\%")

        # Build LaTeX row (10 columns) - flow order
        return (
            f"{rank} & {split} & {scal} & {layers} & {act} & {lr} & "
            f"{mom} & {batch} & {epochs} & {acc} \\\\"
        )

    def _format_layers(self, layers: str) -> str:
        """Format layers string for LaTeX.

        Args:
            layers: String representation of layers (e.g., "[64, 32]")

        Returns:
            Formatted layers string
        """
        # Remove spaces for compactness: [64, 32] -> [64,32]
        return layers.replace(", ", ",")

    def _format_lr(self, lr: float) -> str:
        """Format learning rate.

        Args:
            lr: Learning rate value

        Returns:
            Formatted learning rate (e.g., ".01")
        """
        if lr >= self.LR_PRECISION_THRESHOLD:
            return f"{lr:.2f}".lstrip("0")  # .01, .05
        else:
            return f"{lr:.3f}".lstrip("0")  # .001

    def _format_momentum(self, momentum: float) -> str:
        """Format momentum value.

        Args:
            momentum: Momentum value

        Returns:
            Formatted momentum (e.g., ".90")
        """
        return f"{momentum:.2f}".lstrip("0")  # .90, .95

    def _format_scaler(self, scaler: str) -> str:
        """Format scaler name.

        Args:
            scaler: Scaler type (e.g., "standard", "minmax")

        Returns:
            Short scaler name
        """
        return "std" if scaler == "standard" else "mm"

    def _format_split(self, split: str) -> str:
        """Format split strategy.

        Args:
            split: Split strategy (e.g., "0.8/0.1/0.1")

        Returns:
            Formatted split (e.g., "80/10/10")
        """
        # Convert 0.8/0.1/0.1 to 80/10/10
        parts = split.split("/")
        formatted = [str(int(float(p) * 100)) for p in parts]
        return "/".join(formatted)

    def _format_epochs(self, best_epoch, total_trained: int) -> str:
        """Format epochs as best/total_trained.

        Args:
            best_epoch: Best epoch number (or NaN)
            total_trained: Total number of epochs actually trained

        Returns:
            Formatted epochs (e.g., "24/30")
        """
        if pd.notna(best_epoch):
            return f"{int(best_epoch)}/{total_trained}"
        else:
            return f"--/{total_trained}"

    def _generate_table_footer(self) -> str:
        """Generate LaTeX table footer."""
        return """\\bottomrule
\\end{tabular}
\\end{table}
"""

    def __repr__(self) -> str:
        return f"LaTeXTableGenerator(top_n={self.top_n}, sort_by='{self.sort_by}')"
