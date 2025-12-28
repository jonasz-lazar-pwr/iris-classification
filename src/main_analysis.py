"""Entry point for experiment analysis and result processing."""

from pathlib import Path

from src.analysis.bar_plot_generator import BarPlotGenerator
from src.analysis.best_configs_distribution_analyze import (
    BestConfigsDistributionAnalyzer,
    BestConfigsDistributionConfig,
)
from src.analysis.converter import JSONToCSVConverter
from src.analysis.plot_config_factory import PlotConfigFactory
from src.analysis.table_generator import LaTeXTableGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Convert experiment results from JSON to CSV and generate LaTeX tables."""

    # Paths
    input_path = Path("results/results.json")
    csv_path = Path("results/results.csv")
    table_path = Path("results/top_40_table.tex")
    figures_dir = Path("figures")

    # Validate input exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Run experiments first: python main.py")
        return

    try:
        logger.info("Experiment results conversion started...")

        # ========== Step 1: Convert JSON to CSV ==========
        logger.info("\n[Step 1/3] Converting JSON to CSV...")

        converter = JSONToCSVConverter()

        # Load
        results = converter.load(str(input_path))

        # Convert
        df = converter.convert(results)

        # Save
        converter.save(df, str(csv_path))

        logger.info(f"CSV saved: {csv_path.name}")
        logger.info(f"  Total experiments: {len(df)}")

        # ========== Step 2: Generate LaTeX Table ==========
        logger.info("\n[Step 2/3] Generating LaTeX table...")

        table_gen = LaTeXTableGenerator(
            top_n=40,
            sort_by="test_accuracy",
            ascending=False,  # Best first
        )

        # Generate and save
        latex_table = table_gen.generate(df)
        table_gen.save(latex_table, str(table_path))

        logger.info(f"LaTeX table saved: {table_path.name}")

        # ========== Step 3: Generate Hyperparameter Analysis Plots ==========
        logger.info("\n[Step 3/3] Generating hyperparameter analysis plots...")

        # Create plot configurations (16 total: 8 params × 2 metrics)
        configs = PlotConfigFactory.create_standard_configs(output_dir=str(figures_dir))

        logger.info(f"  Generating {len(configs)} plots...")

        # Generate plots
        plot_generator = BarPlotGenerator()

        for i, config in enumerate(configs, 1):
            logger.info(f"  [{i}/{len(configs)}] {config.hyperparameter} vs {config.metric}")

            # Generate and save
            plot_generator.generate(df, config)
            plot_generator.save(config.output_path)

        logger.info(f"All plots saved to: {figures_dir}/")

        # ========== Step 4: Analyze distributions among perfect configs ==========
        logger.info(
            "\n[Step 4/4] Summarizing distributions for perfect configurations (Acc=1.0)..."
        )

        analyzer_config = BestConfigsDistributionConfig(
            acc_threshold=1.0,
            columns=[
                "split_strategy",
                "scaler_type",
                "layers",
                "activation",
                "learning_rate",
                "momentum",
                "batch_size",
            ],
            strict_columns=True,
        )

        analyzer = BestConfigsDistributionAnalyzer(config=analyzer_config, logger=logger)
        analyzer.analyze(df)

        logger.info("\nAnalysis completed successfully!")

    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
