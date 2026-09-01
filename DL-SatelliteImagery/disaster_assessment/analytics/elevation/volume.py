"""
Volume calculation from depth rasters.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VolumeResult:
    """Volume calculation result."""
    total_volume_m3: Optional[float] = None
    total_volume_km3: Optional[float] = None
    flooded_area_m2: Optional[float] = None
    mean_depth: Optional[float] = None
    max_depth: Optional[float] = None
    volume_status: str = "NOT_CALCULATED"


class VolumeCalculator:
    """Calculate water volume from depth rasters and DEM resolution."""

    def calculate(
        self,
        depth_raster: np.ndarray,
        flood_mask: np.ndarray,
        pixel_resolution_m: Optional[float] = None,
    ) -> VolumeResult:
        """Calculate volume from depth and area.

        volume = sum(depth) * pixel_area

        Args:
            depth_raster: (H, W) depth in meters.
            flood_mask: (H, W) binary flood mask.
            pixel_resolution_m: Pixel size in meters (for area calculation).

        Returns:
            VolumeResult.
        """
        result = VolumeResult()

        if depth_raster is None or flood_mask is None:
            result.volume_status = "INSUFFICIENT_INPUT"
            return result

        flood_bool = flood_mask.astype(bool)
        valid_depths = depth_raster[flood_bool]

        if valid_depths.size == 0:
            result.total_volume_m3 = 0.0
            result.flooded_area_m2 = 0.0
            result.mean_depth = 0.0
            result.max_depth = 0.0
            result.volume_status = "COMPLETED"
            return result

        result.mean_depth = float(np.mean(valid_depths))
        result.max_depth = float(np.max(valid_depths))

        if pixel_resolution_m is not None and pixel_resolution_m > 0:
            pixel_area = pixel_resolution_m ** 2
            result.flooded_area_m2 = float(np.sum(flood_bool)) * pixel_area
            result.total_volume_m3 = float(np.sum(valid_depths)) * pixel_area
            result.total_volume_km3 = result.total_volume_m3 / 1e9
            result.volume_status = "COMPLETED"
        else:
            result.volume_status = "AREA_RESOLUTION_MISSING"

        return result
