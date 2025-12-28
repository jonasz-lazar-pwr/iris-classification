"""Configuration expander for generating sweep combinations."""

import copy

from itertools import product
from typing import Any, ClassVar, Dict, List

from .base import IConfigExpander


class ConfigExpander(IConfigExpander):
    """Expands sweep parameters into concrete configurations."""

    NON_SWEEP_PATHS: ClassVar[set[str]] = {
        "data.path",
        "data.feature_columns",
        "data.target_column",
        "evaluation.metrics",
        "evaluation.class_names",
    }

    def expand(self, config: Dict) -> List[Dict]:
        """Expand configuration with sweep parameters into list of concrete configs."""
        sweep_params = self._find_sweep_params(config)

        if not sweep_params:
            return [copy.deepcopy(config)]

        paths = list(sweep_params.keys())
        value_lists = list(sweep_params.values())

        combinations = list(product(*value_lists))

        expanded_configs = []
        for combination in combinations:
            concrete_config = copy.deepcopy(config)

            for path, value in zip(paths, combination, strict=True):
                self._set_nested_value(concrete_config, path, value)

            expanded_configs.append(concrete_config)

        return expanded_configs

    def count_combinations(self, config: Dict) -> int:
        """Count total number of configurations without expanding."""
        sweep_params = self._find_sweep_params(config)

        if not sweep_params:
            return 1

        total = 1
        for values in sweep_params.values():
            total *= len(values)

        return total

    def _find_sweep_params(self, config: Dict, parent_path: str = "") -> Dict[str, List]:
        """Recursively find all sweep parameters in config."""
        sweep_params = {}

        for key, value in config.items():
            current_path = f"{parent_path}.{key}" if parent_path else key

            if self._is_non_sweep_path(current_path):
                continue

            if isinstance(value, list):
                if len(value) >= 1:
                    sweep_params[current_path] = value
                # Lista z 1 elementem = fixed value, nie sweep

            elif isinstance(value, dict):
                nested_params = self._find_sweep_params(value, current_path)
                sweep_params.update(nested_params)

        return sweep_params

    def _is_non_sweep_path(self, path: str) -> bool:
        """Check if path should not be treated as sweep parameter."""
        if path in self.NON_SWEEP_PATHS:
            return True

        for non_sweep_path in self.NON_SWEEP_PATHS:
            if path.startswith(f"{non_sweep_path}."):
                return True

        return False

    def _set_nested_value(self, config: Dict, path: str, value: Any) -> None:
        """Set value in nested dict using dot-notation path."""
        keys = path.split(".")
        current = config

        for key in keys[:-1]:
            current = current[key]

        current[keys[-1]] = value

    def _get_nested_value(self, config: Dict, path: str) -> Any:
        """Get value from nested dict using dot-notation path."""
        keys = path.split(".")
        current = config

        for key in keys:
            current = current[key]

        return current
