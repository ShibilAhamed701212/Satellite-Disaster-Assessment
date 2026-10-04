"""
Model backbone registry for pluggable architectures.

Supports registration and lookup of different backbone architectures.
"""

from typing import Dict, List, Optional



MODEL_REGISTRY: Dict[str, type] = {}


def register_model(name: str):
    """Decorator to register a model class in the global registry.

    Usage:
        @register_model("my_model")
        class MyModel(nn.Module):
            ...
    """
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str) -> Optional[type]:
    """Look up a model class by name."""
    return MODEL_REGISTRY.get(name)


def list_models() -> List[str]:
    """List all registered model names."""
    return sorted(MODEL_REGISTRY.keys())


def get_model_status() -> Dict[str, str]:
    """Get status of all registered models."""
    statuses = {}
    for name, cls in MODEL_REGISTRY.items():
        if hasattr(cls, 'get_status'):
            statuses[name] = "AVAILABLE"
        else:
            statuses[name] = "REGISTERED"
    return statuses
