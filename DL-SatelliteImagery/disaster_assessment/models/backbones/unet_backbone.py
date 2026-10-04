"""
UNet backbone adapter for the registry.
"""

from typing import List

import torch.nn as nn

from .registry import register_model


class UNetBackbone:
    """Adapter wrapping the existing LightweightEncoder as a registry backbone."""

    def __init__(self):
        from ..base_unet import LightweightEncoder
        self.encoder = LightweightEncoder(in_channels=3)

    @staticmethod
    def create(in_channels: int = 3) -> nn.Module:
        from ..base_unet import LightweightEncoder
        return LightweightEncoder(in_channels=in_channels)


def get_unet_channels() -> List[int]:
    """Get standard UNet channel configuration."""
    return [16, 32, 64, 128, 256]


# Register built-in backbone
@register_model("base_unet")
class BaseUNetRegistration:
    """Registry entry for the base lightweight UNet backbone."""
    channels = [16, 32, 64, 128, 256]
    params = "~1.18M"
    description = "Lightweight CNN U-Net encoder (16→32→64→128→256)"
    available = True
