"""Bar plot generator for hyperparameter analysis."""

from pathlib import Path
from typing import ClassVar, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.analysis.base import IPlotGenerator
from src.analysis.plot_config import PlotConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BarPlotGenerator(IPlotGenerator):
    """Generate bar plots for hyperparameter impact analysis."""

    # Professional color palette (similar to your GA project)
    _PALETTE: ClassVar[list[str]] = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
    ]

    def __init__(self):
        """Initialize bar plot generator."""
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self._aggregated_data: Optional[pd.DataFrame] = None

    def generate(self, df: pd.DataFrame, config: PlotConfig) -> None:
        """Generate bar plot from DataFrame.

        Args:
            df: DataFrame with experiment results
            config: Plot configuration
        """
        logger.info(f"Generating bar plot: {config.hyperparameter} vs {config.metric}")

        # Aggregate data
        self._aggregated_data = self._aggregate_data(df, config)

        # Create plot
        self._create_plot(config)

        logger.info("Bar plot generated successfully")

    def save(self, output_path: str) -> None:
        """Save generated plot to file.

        Args:
            output_path: Path where to save the plot
        """
        if self.fig is None:
            raise ValueError("No plot generated. Call generate() first.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.fig.savefig(path, bbox_inches="tight", dpi=300, format=path.suffix[1:])

        logger.info(f"Plot saved to {output_path}")
        plt.close(self.fig)

    def _aggregate_data(self, df: pd.DataFrame, config: PlotConfig) -> pd.DataFrame:
        """Aggregate data by hyperparameter.

        Args:
            df: Input DataFrame
            config: Plot configuration

        Returns:
            Aggregated DataFrame with mean, std, count
        """
        grouped = (
            df.groupby(config.hyperparameter)[config.metric]
            .agg([("mean", "mean"), ("std", "std"), ("count", "count")])
            .reset_index()
        )

        # Use specified aggregation method
        grouped["value"] = grouped["mean"]

        # Sort if requested
        if config.sort_by == "value":
            grouped = grouped.sort_values("value", ascending=False)
        elif config.sort_by == "name":
            grouped = grouped.sort_values(config.hyperparameter)

        return grouped

    def _create_plot(self, config: PlotConfig) -> None:
        """Create the bar plot.

        Args:
            config: Plot configuration
        """
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=config.figsize)

        # Determine colors - cycle through palette
        n_bars = len(self._aggregated_data)
        colors = [self._PALETTE[i % len(self._PALETTE)] for i in range(n_bars)]

        # Convert hyperparameter values to strings for categorical axis
        x_values = self._aggregated_data[config.hyperparameter].astype(str)

        # Create bars (no error bars - removed)
        bars = self.ax.bar(
            x_values,
            self._aggregated_data["value"],
            color=colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )

        # Add values on bars
        if config.show_values:
            self._add_value_labels(bars, config)

        # Set labels (NO TITLE - removed plt.title)
        self.ax.set_xlabel(config.xlabel, fontsize=12)
        self.ax.set_ylabel(config.ylabel, fontsize=12)

        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha="right")

        # Grid
        self.ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
        self.ax.set_axisbelow(True)

        # Clean style
        sns.despine(left=False, bottom=False)

        # Tight layout
        plt.tight_layout()

    def _add_value_labels(self, bars, config: PlotConfig) -> None:
        """Add value labels on bars.

        Args:
            bars: Bar container
            config: Plot configuration
        """
        for bar, row in zip(bars, self._aggregated_data.itertuples()):
            value = row.value

            # Format value based on metric
            if config.metric == "test_accuracy":
                label = f"{value:.1%}"
            elif "epoch" in config.metric.lower():
                label = f"{value:.1f}"
            else:
                label = f"{value:.3f}"

            # Position label on top of bar
            self.ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="normal",
            )

    def __repr__(self) -> str:
        return "BarPlotGenerator()"
