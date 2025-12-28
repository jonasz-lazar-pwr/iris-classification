"""Factory for creating data processing components with registry pattern."""

from typing import Callable, ClassVar

from src.data.base import (
    IDataComponentFactory,
    IDataLoader,
    IDataSplitter,
    IDataValidator,
    IScaler,
)
from src.data.loaders import CSVDataLoader, IrisDataLoader
from src.data.scalers import MinMaxScaler, StandardScaler
from src.data.splitter import DataSplitter
from src.data.validator import DataValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataComponentFactory(IDataComponentFactory):
    """Factory for creating data processing components with extensible registry."""

    _loaders: ClassVar[dict[str, type[IDataLoader]]] = {}
    _scalers: ClassVar[dict[str, type[IScaler]]] = {}
    _validators: ClassVar[dict[str, type[IDataValidator]]] = {}
    _splitters: ClassVar[dict[str, type[IDataSplitter]]] = {}
    _defaults_registered: ClassVar[bool] = False

    def __init__(self) -> None:
        """Initialize factory and register default components."""
        if not DataComponentFactory._defaults_registered:
            self._register_defaults()
            DataComponentFactory._defaults_registered = True

    @classmethod
    def _register_defaults(cls) -> None:
        """Register default built-in components."""
        cls.register_loader("csv", CSVDataLoader)
        cls.register_loader("iris", IrisDataLoader)
        cls.register_scaler("standard", StandardScaler)
        cls.register_scaler("minmax", MinMaxScaler)
        cls.register_validator("default", DataValidator)
        cls.register_splitter("default", DataSplitter)

    @classmethod
    def register_loader(cls, name: str, loader_class: type[IDataLoader]) -> None:
        """Register a new data loader type."""
        if not issubclass(loader_class, IDataLoader):
            raise TypeError(f"{loader_class.__name__} must implement IDataLoader")

        if name in cls._loaders:
            logger.warning(f"Overwriting existing loader registration: {name}")

        cls._loaders[name] = loader_class
        logger.debug(f"Registered loader: {name} -> {loader_class.__name__}")

    @classmethod
    def register_scaler(cls, name: str, scaler_class: type[IScaler]) -> None:
        """Register a new scaler type."""
        if not issubclass(scaler_class, IScaler):
            raise TypeError(f"{scaler_class.__name__} must implement IScaler")

        if name in cls._scalers:
            logger.warning(f"Overwriting existing scaler registration: {name}")

        cls._scalers[name] = scaler_class
        logger.debug(f"Registered scaler: {name} -> {scaler_class.__name__}")

    @classmethod
    def register_validator(cls, name: str, validator_class: type[IDataValidator]) -> None:
        """Register a new validator type."""
        if not issubclass(validator_class, IDataValidator):
            raise TypeError(f"{validator_class.__name__} must implement IDataValidator")

        if name in cls._validators:
            logger.warning(f"Overwriting existing validator registration: {name}")

        cls._validators[name] = validator_class
        logger.debug(f"Registered validator: {name} -> {validator_class.__name__}")

    @classmethod
    def register_splitter(cls, name: str, splitter_class: type[IDataSplitter]) -> None:
        """Register a new splitter type."""
        if not issubclass(splitter_class, IDataSplitter):
            raise TypeError(f"{splitter_class.__name__} must implement IDataSplitter")

        if name in cls._splitters:
            logger.warning(f"Overwriting existing splitter registration: {name}")

        cls._splitters[name] = splitter_class
        logger.debug(f"Registered splitter: {name} -> {splitter_class.__name__}")

    def create_loader(self, loader_type: str, **kwargs) -> IDataLoader:  # type: ignore[misc]
        """Create data loader instance."""
        if loader_type not in self._loaders:
            available = list(self._loaders.keys())
            raise ValueError(f"Unknown loader type: '{loader_type}'. Available: {available}")

        loader_class = self._loaders[loader_type]
        logger.debug(f"Creating loader: {loader_type} ({loader_class.__name__})")

        return loader_class(**kwargs)  # type: ignore[operator]

    def create_scaler(self, scaler_type: str, **kwargs) -> IScaler:  # type: ignore[misc]
        """Create scaler instance."""
        if scaler_type not in self._scalers:
            available = list(self._scalers.keys())
            raise ValueError(f"Unknown scaler type: '{scaler_type}'. Available: {available}")

        scaler_class = self._scalers[scaler_type]
        logger.debug(f"Creating scaler: {scaler_type} ({scaler_class.__name__})")

        return scaler_class(**kwargs)  # type: ignore[operator]

    def create_validator(self, validator_type: str = "default", **kwargs) -> IDataValidator:  # type: ignore[misc]
        """Create validator instance."""
        if validator_type not in self._validators:
            available = list(self._validators.keys())
            raise ValueError(f"Unknown validator type: '{validator_type}'. Available: {available}")

        validator_class = self._validators[validator_type]
        logger.debug(f"Creating validator: {validator_type} ({validator_class.__name__})")

        return validator_class(**kwargs)  # type: ignore[operator]

    def create_splitter(self, splitter_type: str = "default", **kwargs) -> IDataSplitter:  # type: ignore[misc]
        """Create splitter instance."""
        if splitter_type not in self._splitters:
            available = list(self._splitters.keys())
            raise ValueError(f"Unknown splitter type: '{splitter_type}'. Available: {available}")

        splitter_class = self._splitters[splitter_type]
        logger.debug(f"Creating splitter: {splitter_type} ({splitter_class.__name__})")

        return splitter_class(**kwargs)  # type: ignore[operator]

    def list_available_components(self) -> dict[str, list[str]]:
        """List all available component types."""
        return {
            "loaders": list(self._loaders.keys()),
            "scalers": list(self._scalers.keys()),
            "validators": list(self._validators.keys()),
            "splitters": list(self._splitters.keys()),
        }

    def is_registered(self, component_type: str, name: str) -> bool:
        """Check if a component is registered."""
        registries = {
            "loader": self._loaders,
            "scaler": self._scalers,
            "validator": self._validators,
            "splitter": self._splitters,
        }

        if component_type not in registries:
            raise ValueError(
                f"Invalid component type: '{component_type}'. "
                f"Must be one of: {list(registries.keys())}"
            )

        return name in registries[component_type]

    def __repr__(self) -> str:
        """String representation of factory."""
        components = self.list_available_components()
        total = sum(len(v) for v in components.values())
        return f"DataComponentFactory(total_registered={total})"


def register_loader(name: str) -> Callable:
    """Decorator for registering custom loaders."""

    def decorator(cls: type[IDataLoader]) -> type[IDataLoader]:
        DataComponentFactory.register_loader(name, cls)
        return cls

    return decorator


def register_scaler(name: str) -> Callable:
    """Decorator for registering custom scalers."""

    def decorator(cls: type[IScaler]) -> type[IScaler]:
        DataComponentFactory.register_scaler(name, cls)
        return cls

    return decorator


def register_validator(name: str) -> Callable:
    """Decorator for registering custom validators."""

    def decorator(cls: type[IDataValidator]) -> type[IDataValidator]:
        DataComponentFactory.register_validator(name, cls)
        return cls

    return decorator


def register_splitter(name: str) -> Callable:
    """Decorator for registering custom splitters."""

    def decorator(cls: type[IDataSplitter]) -> type[IDataSplitter]:
        DataComponentFactory.register_splitter(name, cls)
        return cls

    return decorator
