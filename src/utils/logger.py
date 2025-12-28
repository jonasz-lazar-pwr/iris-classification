import logging

from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "iris.log"

FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

MAX_LOG_SIZE = 5_000_000  # 5 MB
BACKUP_COUNT = 5

_file_handler: RotatingFileHandler | None = None
_console_handler: RichHandler | None = None
_initialized: bool = False


def _initialize_logging() -> None:
    """Initialize logging handlers and root logger."""
    global _file_handler, _console_handler, _initialized

    if _initialized:
        return

    # Create log directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # File handler (DEBUG)
    _file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))

    # Console handler (INFO)
    _console_handler = RichHandler(
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        show_time=True,
        show_level=True,
        show_path=False,
    )
    _console_handler.setLevel(logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(_console_handler)
    root_logger.addHandler(_file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced logger for the given module."""
    if not _initialized:
        _initialize_logging()

    return logging.getLogger(name)


def disable_file_logging() -> None:
    """Disable file logging (useful for tests)."""
    if not _initialized:
        _initialize_logging()

    if _file_handler is None:
        return

    root = logging.getLogger()
    if _file_handler in root.handlers:
        root.removeHandler(_file_handler)
        logging.debug("File logging disabled")


def enable_file_logging() -> None:
    """Re-enable file logging if previously disabled."""
    if not _initialized:
        _initialize_logging()

    if _file_handler is None:
        return

    root = logging.getLogger()
    if _file_handler not in root.handlers:
        root.addHandler(_file_handler)
        logging.debug("File logging enabled")


def set_console_level(level: str) -> None:
    """Change console logging level (DEBUG, INFO, WARNING, ERROR)."""
    if not _initialized:
        _initialize_logging()

    if _console_handler is None:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    _console_handler.setLevel(numeric_level)
    logging.info(f"Console level changed to: {level}")


def set_file_level(level: str) -> None:
    """Change file logging level (DEBUG, INFO, WARNING, ERROR)."""
    if not _initialized:
        _initialize_logging()

    if _file_handler is None:
        return

    numeric_level = getattr(logging, level.upper(), logging.DEBUG)
    _file_handler.setLevel(numeric_level)
    logging.debug(f"File level changed to: {level}")
