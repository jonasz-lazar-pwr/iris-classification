"""Factory for creating model components (activations, losses, optimizers)."""

import logging

from typing import Any, ClassVar, Type

from src.models.activations import ReLU, Softmax, Tanh
from src.models.base import (
    IActivation,
    ILossFunction,
    IModelComponentFactory,
    IOptimizer,
)
from src.models.losses import CrossEntropyLoss
from src.models.optimizers import SGDMomentum

logger = logging.getLogger(__name__)


class ModelComponentFactory(IModelComponentFactory):
    """Factory for creating neural network components with registry pattern."""

    _activations: ClassVar[dict[str, Type[IActivation]]] = {}
    _losses: ClassVar[dict[str, Type[ILossFunction]]] = {}
    _optimizers: ClassVar[dict[str, Type[IOptimizer]]] = {}
    _defaults_registered: ClassVar[bool] = False

    def __init__(self) -> None:
        """Initialize factory and register default components."""
        if not ModelComponentFactory._defaults_registered:
            self._register_defaults()
            ModelComponentFactory._defaults_registered = True

    @classmethod
    def _register_defaults(cls) -> None:
        """Register built-in components."""
        logger.debug("Registering default model components")

        cls._activations["relu"] = ReLU
        cls._activations["tanh"] = Tanh
        cls._activations["softmax"] = Softmax

        cls._losses["cross_entropy"] = CrossEntropyLoss

        cls._optimizers["sgd_momentum"] = SGDMomentum

        logger.debug(
            f"Registered {len(cls._activations)} activations, "
            f"{len(cls._losses)} losses, "
            f"{len(cls._optimizers)} optimizers"
        )

    @classmethod
    def register_activation(cls, name: str, activation_class: Type[IActivation]) -> None:
        """Register custom activation function."""
        if not issubclass(activation_class, IActivation):
            raise TypeError(f"{activation_class.__name__} must implement IActivation interface")

        if name in cls._activations:
            logger.warning(f"Overwriting existing activation: '{name}'")

        cls._activations[name] = activation_class
        logger.debug(f"Registered activation: {name} ({activation_class.__name__})")

    @classmethod
    def register_loss(cls, name: str, loss_class: Type[ILossFunction]) -> None:
        """Register custom loss function."""
        if not issubclass(loss_class, ILossFunction):
            raise TypeError(f"{loss_class.__name__} must implement ILossFunction interface")

        if name in cls._losses:
            logger.warning(f"Overwriting existing loss: '{name}'")

        cls._losses[name] = loss_class
        logger.debug(f"Registered loss: {name} ({loss_class.__name__})")

    @classmethod
    def register_optimizer(cls, name: str, optimizer_class: Type[IOptimizer]) -> None:
        """Register custom optimizer."""
        if not issubclass(optimizer_class, IOptimizer):
            raise TypeError(f"{optimizer_class.__name__} must implement IOptimizer interface")

        if name in cls._optimizers:
            logger.warning(f"Overwriting existing optimizer: '{name}'")

        cls._optimizers[name] = optimizer_class
        logger.debug(f"Registered optimizer: {name} ({optimizer_class.__name__})")

    def create_activation(self, activation_type: str, **kwargs: Any) -> IActivation:
        """Create activation function instance."""
        if activation_type not in self._activations:
            available = list(self._activations.keys())
            raise ValueError(
                f"Unknown activation type: '{activation_type}'. Available: {available}"
            )

        activation_class = self._activations[activation_type]
        logger.debug(f"Creating activation: {activation_type} ({activation_class.__name__})")

        return activation_class(**kwargs)  # type: ignore[return-value]

    def create_loss(self, loss_type: str, **kwargs: Any) -> ILossFunction:
        """Create loss function instance."""
        if loss_type not in self._losses:
            available = list(self._losses.keys())
            raise ValueError(f"Unknown loss type: '{loss_type}'. Available: {available}")

        loss_class = self._losses[loss_type]
        logger.debug(f"Creating loss: {loss_type} ({loss_class.__name__})")

        return loss_class(**kwargs)  # type: ignore[return-value]

    def create_optimizer(self, optimizer_type: str, **kwargs: Any) -> IOptimizer:
        """Create optimizer instance."""
        if optimizer_type not in self._optimizers:
            available = list(self._optimizers.keys())
            raise ValueError(f"Unknown optimizer type: '{optimizer_type}'. Available: {available}")

        optimizer_class = self._optimizers[optimizer_type]
        logger.debug(f"Creating optimizer: {optimizer_type} ({optimizer_class.__name__})")

        return optimizer_class(**kwargs)  # type: ignore[return-value]

    def list_available_components(self) -> dict[str, list[str]]:
        """List all registered component types."""
        return {
            "activations": sorted(self._activations.keys()),
            "losses": sorted(self._losses.keys()),
            "optimizers": sorted(self._optimizers.keys()),
        }

    def is_registered(self, component_type: str, name: str) -> bool:
        """Check if a component is registered."""
        if component_type == "activation":
            return name in self._activations
        elif component_type == "loss":
            return name in self._losses
        elif component_type == "optimizer":
            return name in self._optimizers
        else:
            raise ValueError(
                f"Invalid component type: '{component_type}'. "
                "Must be 'activation', 'loss', or 'optimizer'"
            )

    def __repr__(self) -> str:
        """String representation of factory."""
        total = len(self._activations) + len(self._losses) + len(self._optimizers)
        return f"ModelComponentFactory(total_registered={total})"


def register_activation(name: str):
    """Decorator to register activation function."""

    def decorator(cls: Type[IActivation]) -> Type[IActivation]:
        ModelComponentFactory.register_activation(name, cls)
        return cls

    return decorator


def register_loss(name: str):
    """Decorator to register loss function."""

    def decorator(cls: Type[ILossFunction]) -> Type[ILossFunction]:
        ModelComponentFactory.register_loss(name, cls)
        return cls

    return decorator


def register_optimizer(name: str):
    """Decorator to register optimizer."""

    def decorator(cls: Type[IOptimizer]) -> Type[IOptimizer]:
        ModelComponentFactory.register_optimizer(name, cls)
        return cls

    return decorator
