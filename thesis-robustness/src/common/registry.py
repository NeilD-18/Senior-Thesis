"""Registry for models and datasets."""
from typing import Dict, Callable, Any

# Model registry
MODELS: Dict[str, Callable] = {}

# Dataset registry
DATASETS: Dict[str, Callable] = {}


def register_model(name: str):
    """Decorator to register a model factory."""
    def decorator(func: Callable):
        MODELS[name] = func
        return func
    return decorator


def register_dataset(name: str):
    """Decorator to register a dataset loader."""
    def decorator(func: Callable):
        DATASETS[name] = func
        return func
    return decorator


def get_model(name: str, **kwargs):
    """Get a model by name."""
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
    return MODELS[name](**kwargs)


def get_dataset(name: str, **kwargs):
    """Get a dataset by name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return DATASETS[name](**kwargs)
