"""Factory for creating plot configurations."""

from typing import List

from src.analysis.plot_config import PlotConfig


class PlotConfigFactory:
    """Factory for creating standard plot configurations."""

    # Mapping of hyperparameters to display names
    PARAM_LABELS = {
        "split_strategy": "Strategia podziału danych",
        "scaler_type": "Typ skalowania",
        "layers": "Architektura sieci",
        "activation": "Funkcja aktywacji",
        "learning_rate": "Współczynnik uczenia (LR)",
        "momentum": "Momentum",
        "batch_size": "Rozmiar paczki",
        "max_epochs": "Maksymalna liczba epok",
    }

    METRIC_LABELS = {
        "test_accuracy": "Test Accuracy",
        "best_epoch": "Best Epoch",
    }

    @staticmethod
    def create_config(
        hyperparameter: str,
        metric: str,
        output_dir: str = "figures",
    ) -> PlotConfig:
        """Create a plot configuration.

        Args:
            hyperparameter: Name of hyperparameter column
            metric: Name of metric column
            output_dir: Directory where to save plots

        Returns:
            PlotConfig instance
        """
        # Generate title
        param_label = PlotConfigFactory.PARAM_LABELS.get(
            hyperparameter, hyperparameter.replace("_", " ").title()
        )
        metric_label = PlotConfigFactory.METRIC_LABELS.get(metric, metric.replace("_", " ").title())

        title = f"Wpływ {param_label.lower()} na {metric_label}"

        # Generate output filename
        filename = f"{hyperparameter}_{metric}_barplot.png"
        output_path = f"{output_dir}/{filename}"

        return PlotConfig(
            hyperparameter=hyperparameter,
            metric=metric,
            title=title,
            output_path=output_path,
            aggregation="mean",
            show_values=True,
            show_error_bars=True,
            sort_by="value",
        )

    @staticmethod
    def create_all_configs(
        hyperparameters: List[str],
        metrics: List[str],
        output_dir: str = "figures",
    ) -> List[PlotConfig]:
        """Create configurations for all hyperparameter-metric combinations.

        Args:
            hyperparameters: List of hyperparameter column names
            metrics: List of metric column names
            output_dir: Directory where to save plots

        Returns:
            List of PlotConfig instances
        """
        configs = []
        for param in hyperparameters:
            for metric in metrics:
                config = PlotConfigFactory.create_config(param, metric, output_dir)
                configs.append(config)
        return configs

    @staticmethod
    def create_standard_configs(output_dir: str = "figures") -> List[PlotConfig]:
        """Create standard set of 16 plot configurations.

        Args:
            output_dir: Directory where to save plots

        Returns:
            List of 16 PlotConfig instances (8 params × 2 metrics)
        """
        hyperparameters = [
            "split_strategy",
            "scaler_type",
            "layers",
            "activation",
            "learning_rate",
            "momentum",
            "batch_size",
            "max_epochs",
        ]

        metrics = ["test_accuracy", "best_epoch"]

        return PlotConfigFactory.create_all_configs(hyperparameters, metrics, output_dir)
