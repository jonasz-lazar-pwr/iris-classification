"""Abstract base classes defining interfaces for experiment components."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple


class IConfigLoader(ABC):
    """Interface for loading configuration files."""

    @abstractmethod
    def load(self, yaml_path: str) -> Dict:
        """Load configuration from YAML file."""
        pass


class IConfigExpander(ABC):
    """Interface for expanding sweep parameters into concrete configurations."""

    @abstractmethod
    def expand(self, config: Dict) -> List[Dict]:
        """Expand configuration with sweep parameters into list of concrete configs."""
        pass

    @abstractmethod
    def count_combinations(self, config: Dict) -> int:
        """Count total number of configurations without expanding."""
        pass


class IConfigValidator(ABC):
    """Interface for validating configuration correctness."""

    @abstractmethod
    def validate(self, config: Dict) -> Tuple[bool, List[str]]:
        """Validate single configuration, returns (is_valid, list_of_errors)."""
        pass

    @abstractmethod
    def validate_all(self, configs: List[Dict]) -> Dict[int, List[str]]:
        """Validate multiple configurations, returns dict mapping config_index to errors."""
        pass


class IExperiment(ABC):
    """Interface for running a single experiment."""

    def __init__(self, config: Dict, experiment_id: int):
        """Initialize experiment with configuration and ID."""
        self.config = config
        self.id = experiment_id

    @abstractmethod
    def run(self) -> Dict:
        """Execute experiment: preprocess -> train -> evaluate, returns metrics dict."""
        pass

    @abstractmethod
    def get_config(self) -> Dict:
        """Get the configuration used for this experiment."""
        pass

    @abstractmethod
    def get_id(self) -> int:
        """Get the experiment ID."""
        pass


class IExperimentRunner(ABC):
    """Interface for orchestrating multiple experiments (sweep)."""

    def __init__(self, output_dir: str = "results"):
        """Initialize runner with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    @abstractmethod
    def run_sweep(self, yaml_path: str) -> List[Dict]:
        """Run full hyperparameter sweep and return list of results."""
        pass

    @abstractmethod
    def save_results(self, results: List[Dict], filename: str = "results.json") -> Path:
        """Save results to JSON file and return path."""
        pass

    @abstractmethod
    def find_best(self, results: List[Dict], metric: str = "test_accuracy") -> Dict:
        """Find best configuration by metric."""
        pass


class IExperimentResult(ABC):
    """Interface for experiment result data structure."""

    @abstractmethod
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict) -> "IExperimentResult":
        """Create ExperimentResult from dictionary."""
        pass

    @abstractmethod
    def get_accuracy(self) -> float:
        """Get test accuracy metric."""
        pass

    @abstractmethod
    def get_precision(self) -> float:
        """Get test precision metric."""
        pass

    @abstractmethod
    def get_recall(self) -> float:
        """Get test recall metric."""
        pass

    @abstractmethod
    def get_f1(self) -> float:
        """Get test F1 score metric."""
        pass

    @abstractmethod
    def get_confusion_matrix(self) -> List[List[int]]:
        """Get confusion matrix."""
        pass
