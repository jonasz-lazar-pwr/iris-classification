from pathlib import Path

import pytest

from src.utils.logger import disable_file_logging


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Disable file logging for all tests."""
    disable_file_logging()


@pytest.fixture(scope="session")
def base_dir() -> Path:
    """Return base project directory."""
    return Path.cwd()


@pytest.fixture(scope="session")
def data_dir(base_dir: Path) -> Path:
    """Return data directory."""
    return base_dir / "data"


@pytest.fixture(scope="session")
def config_dir(base_dir: Path) -> Path:
    """Return config directory."""
    return base_dir / "config"


@pytest.fixture
def iris_data_path(data_dir: Path) -> Path:
    """Return Iris dataset path."""
    return data_dir / "raw" / "iris.data"
