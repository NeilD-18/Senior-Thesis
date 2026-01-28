"""Registry for corruption functions."""
from typing import Callable, Dict

_CORRUPTION_REGISTRY: Dict[str, Callable] = {}


def register_corruption(name: str):
    """Decorator to register a corruption function."""
    def decorator(func: Callable):
        _CORRUPTION_REGISTRY[name] = func
        return func
    return decorator


def get_corruption(name: str) -> Callable:
    """Get a registered corruption function by name."""
    if name not in _CORRUPTION_REGISTRY:
        raise ValueError(
            f"Unknown corruption: {name}. "
            f"Available corruptions: {list(_CORRUPTION_REGISTRY.keys())}"
        )
    return _CORRUPTION_REGISTRY[name]


def list_corruptions() -> list:
    """List all registered corruptions."""
    return list(_CORRUPTION_REGISTRY.keys())
