"""Configuration for plot generation."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PlotConfig:
    """Configuration for generating a single plot."""

    # Data configuration
    hyperparameter: str  # Column name to group by
    metric: str  # Column name to aggregate
    aggregation: str = "mean"  # mean, median, sum

    # Plot appearance
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    figsize: Tuple[float, float] = (10, 6)
    color_palette: Optional[list] = None

    # Output configuration
    output_path: str = ""
    file_format: str = "png"  # Changed from pdf to png
    dpi: int = 300

    # Optional features
    show_values: bool = True
    show_error_bars: bool = True
    sort_by: str = "value"  # value, name, none
    horizontal: bool = False

    def __post_init__(self):
        """Set default values based on metric."""
        if not self.xlabel:
            self.xlabel = self.hyperparameter.replace("_", " ").title()

        if not self.ylabel:
            if self.metric == "test_accuracy":
                self.ylabel = "Test Accuracy"
            elif self.metric == "best_epoch":
                self.ylabel = "Best Epoch"
            else:
                self.ylabel = self.metric.replace("_", " ").title()
