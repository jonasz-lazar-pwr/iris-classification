"""Main entry point for Iris Classification hyperparameter sweep."""

import sys

from pathlib import Path

from src.experiment.experiment_runner import ExperimentRunner
from src.utils.logger import disable_file_logging, get_logger

# Disable file logging globally
disable_file_logging()

logger = get_logger(__name__)


def main() -> int:
    """Execute complete hyperparameter sweep workflow."""
    config_path = "config/experiments.yaml"
    output_dir = "results"

    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1

    try:
        runner = ExperimentRunner(output_dir=output_dir)

        # Run sweep with minimal console output
        results = runner.run_sweep(config_path)

        # Save results
        runner.save_results(results, filename="results.json")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration validation error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
