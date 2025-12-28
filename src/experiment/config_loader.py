"""Configuration loader implementation."""

from pathlib import Path
from typing import Dict

import yaml

from .base import IConfigLoader


class ConfigLoader(IConfigLoader):
    """Loads YAML configuration files."""

    def load(self, yaml_path: str) -> Dict:
        """Load configuration from YAML file."""
        path = Path(yaml_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {yaml_path}")

        if path.suffix not in [".yaml", ".yml"]:
            raise ValueError(f"File must have .yaml or .yml extension: {yaml_path}")

        try:
            with path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file: {yaml_path}\n{e}") from e
        except Exception as e:
            raise OSError(f"Failed to read file: {yaml_path}\n{e}") from e

        if config is None:
            raise ValueError(f"Configuration file is empty: {yaml_path}")

        if not isinstance(config, dict):
            raise ValueError(f"Configuration must be a dictionary, got {type(config).__name__}")

        return config
