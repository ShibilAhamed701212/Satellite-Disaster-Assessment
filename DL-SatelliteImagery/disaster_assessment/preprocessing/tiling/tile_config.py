"""Tiling configuration for large raster processing."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TilingConfig:
    """Configuration for sliding-window tiling of large rasters.

    Attributes:
        tile_size: Size of each tile (width, height) in pixels.
        overlap: Fractional overlap between adjacent tiles (0.0 - 0.5).
        blend_mode: How to blend overlapping predictions: 'uniform', 'gaussian', 'linear'.
        output_channels: Number of output prediction channels.
        nodata_value: Value representing nodata in source raster.
    """
    tile_size: int = 512
    overlap: float = 0.25
    blend_mode: str = "gaussian"
    output_channels: int = 1
    nodata_value: Optional[float] = None

    def __post_init__(self):
        if not (0.0 <= self.overlap <= 0.5):
            raise ValueError(f"overlap must be 0.0-0.5, got {self.overlap}")
        if self.tile_size < 64:
            raise ValueError(f"tile_size must be >= 64, got {self.tile_size}")
        if self.blend_mode not in ("uniform", "gaussian", "linear"):
            raise ValueError(f"blend_mode must be 'uniform', 'gaussian', or 'linear', got {self.blend_mode}")

    @property
    def stride(self) -> int:
        """Effective stride between tile origins."""
        return max(1, int(self.tile_size * (1.0 - self.overlap)))
