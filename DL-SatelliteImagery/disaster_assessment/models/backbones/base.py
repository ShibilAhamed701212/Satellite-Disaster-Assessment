"""
Base class for pluggable model backbones.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn


class BackboneBase(ABC):
    """Abstract base for backbone models.

    All backbone adapters must implement these methods.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Return the unique name of this backbone."""
        ...

    @abstractmethod
    def get_output_channels(self) -> List[int]:
        """Return the channel dimensions at each encoder level."""
        ...

    @abstractmethod
    def get_total_parameters(self) -> int:
        """Return total number of trainable parameters."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backbone is usable (weights loaded, etc.)."""
        ...

    @abstractmethod
    def get_status(self) -> str:
        """Return current status string."""
        ...


class PyTorchBackboneWrapper(nn.Module, BackboneBase):
    """Wrap a PyTorch module as a backbone."""

    def __init__(self, name: str, module: nn.Module, channels: List[int]):
        super().__init__()
        self._name = name
        self._module = module
        self._channels = channels

    def get_name(self) -> str:
        return self._name

    def get_output_channels(self) -> List[int]:
        return self._channels

    def get_total_parameters(self) -> int:
        return sum(p.numel() for p in self._module.parameters())

    def is_available(self) -> bool:
        return True

    def get_status(self) -> str:
        return "READY"

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return self._module(x)
