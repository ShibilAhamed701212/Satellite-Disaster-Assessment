"""
Pluggable backbone architecture for model registry.
"""

from .base import BackboneBase
from .registry import MODEL_REGISTRY, register_model, get_model, list_models

# Import backbone modules to trigger registration decorators
from . import unet_backbone
from . import foundation_backbone

__all__ = ["BackboneBase", "MODEL_REGISTRY", "register_model", "get_model", "list_models"]
