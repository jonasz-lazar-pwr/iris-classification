"""Orchestrator for running hyperparameter sweeps and managing experiments."""

import json
import logging

from pathlib import Path
from typing import Dict, List

from src.experiment.base import IExperimentRunner
from src.experiment.config_expander import ConfigExpander
from src.experiment.config_loader import ConfigLoader
from src.experiment.config_validator import ConfigValidator
from src.experiment.experiment import Experiment

logger = logging.getLogger(__name__)


class ExperimentRunner(IExperimentRunner):
    """Orchestrates multiple experiments for hyperparameter sweeps."""

    def __init__(self, output_dir: str = "results") -> None:
        """Initialize runner with output directory and components."""
        super().__init__(output_dir)
        self.loader = ConfigLoader()
        self.expander = ConfigExpander()
        self.validator = ConfigValidator()

        logger.info(f"ExperimentRunner initialized - output_dir={self.output_dir}")

    def run_sweep(self, yaml_path: str) -> List[Dict]:
        """Run full hyperparameter sweep from YAML config."""
        logger.info(f"Starting sweep from: {yaml_path}")

        base_config = self.loader.load(yaml_path)

        configs = self.expander.expand(base_config)
        total_configs = len(configs)
        logger.info(f"Expanded to {total_configs} configurations")

        validation_errors = self.validator.validate_all(configs)
        if validation_errors:
            error_msg = f"Found validation errors in {len(validation_errors)} configurations"
            logger.error(error_msg)
            for idx, errors in validation_errors.items():
                logger.error(f"Config {idx}: {errors}")
            raise ValueError(f"{error_msg}. Check logs for details.")

        logger.info("All configurations validated successfully")

        results = []
        for exp_id, config in enumerate(configs, start=1):
            try:
                experiment = Experiment(config=config, experiment_id=exp_id)
                result = experiment.run()
                results.append(result)

                # Log compact experiment result
                self._log_experiment_result(exp_id, total_configs, config, result)

            except Exception as e:
                logger.error(f"Experiment {exp_id} failed: {e}")
                raise

        logger.info(f"Sweep completed: {len(results)}/{total_configs} experiments successful")
        return results

    def save_results(self, results: List[Dict], filename: str = "results.json") -> Path:
        """Save experiment results to JSON file."""
        output_path = self.output_dir / filename

        with output_path.open("w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved successfully to {output_path}")
        return output_path

    def find_best(self, results: List[Dict], metric: str = "test_accuracy") -> Dict:
        """Find best configuration by specified metric."""
        if not results:
            raise ValueError("Cannot find best result: results list is empty")

        logger.info(f"Finding best configuration by metric: {metric}")

        if metric not in results[0]:
            available_metrics = [k for k in results[0].keys() if k.startswith("test_")]
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {available_metrics}")

        best_result = max(results, key=lambda r: r[metric])
        best_value = best_result[metric]
        best_id = best_result["experiment_id"]

        logger.info(f"Best configuration: experiment_id={best_id}, {metric}={best_value:.4f}")

        return best_result

    def _log_experiment_result(self, exp_id: int, total: int, config: Dict, result: Dict) -> None:
        """Log experiment result in compact format with full labels."""
        arch = config["model"]["architecture"]["hidden_layers"]
        act = config["model"]["activations"]["hidden"]
        lr = config["training"]["optimizer"]["learning_rate"]
        batch = config["training"]["batch_size"]
        split = config["preprocessing"]["split_strategy"]
        split_str = f"{split['train_ratio']:.0%}/{split['val_ratio']:.0%}/{split['test_ratio']:.0%}"

        acc = result["test_accuracy"]
        f1 = result["test_f1"]

        history = result.get("training_history", {})
        best_epoch = history.get("best_epoch", "?")
        total_epochs = history.get("total_epochs_trained", "?")

        logger.info(
            f"[Exp {exp_id}/{total}] "
            f"hidden={arch} | act={act} | lr={lr} | batch={batch} | split={split_str} → "
            f"acc={acc:.1%} | f1={f1:.2f} | epoch={best_epoch}/{total_epochs}"
        )

    def __repr__(self) -> str:
        """String representation of ExperimentRunner."""
        return f"ExperimentRunner(output_dir={self.output_dir})"
