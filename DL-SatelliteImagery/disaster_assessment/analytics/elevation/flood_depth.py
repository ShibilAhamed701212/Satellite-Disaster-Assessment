"""
Flood depth estimation framework.

IMPORTANT SCIENTIFIC DISCLAIMER:
    Flood depth estimation from a binary flood mask alone is physically
    insufficient. Accurate flood depth requires:
    - Known water surface elevation (WSE)
    - Boundary water levels or gauge data
    - Hydraulic model outputs
    - High-resolution DEM

    This module provides a framework that returns INSUFFICIENT_INPUT when
    the required reference data is not available.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FloodDepthResult:
    """Flood depth estimation result."""
    depth_raster: Optional[np.ndarray] = None   # (H, W) depth in meters
    max_depth: Optional[float] = None
    mean_depth: Optional[float] = None
    estimated_volume_m3: Optional[float] = None
    depth_estimation_status: str = "NOT_ATTEMPTED"
    assumptions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class FloodDepthEstimator:
    """Flood depth estimation with scientific honesty.

    Requires water surface elevation or reference boundary data for
    physically meaningful depth estimation.
    """

    def estimate_from_boundary(
        self,
        flood_mask: np.ndarray,
        dem: np.ndarray,
        water_surface_elevation: float,
        dem_resolution: Optional[float] = None,
    ) -> FloodDepthResult:
        """Estimate flood depth using a known water surface elevation.

        depth = max(0, WSE - DEM_elevation) for flooded pixels

        This is the simplest physically-grounded approach and requires
        a known water surface elevation (from gauge data, hydraulic model,
        or satellite altimetry).

        Args:
            flood_mask: Binary flood mask (H, W), 1 = flooded.
            dem: (H, W) DEM elevation array.
            water_surface_elevation: Known water surface elevation in same units as DEM.
            dem_resolution: Pixel size in meters for volume calculation.

        Returns:
            FloodDepthResult with depth raster and volume estimate.
        """
        result = FloodDepthResult()

        if flood_mask is None or dem is None:
            result.depth_estimation_status = "INSUFFICIENT_INPUT"
            result.warnings.append("Missing flood_mask or DEM data")
            return result

        if flood_mask.shape != dem.shape:
            result.depth_estimation_status = "INSUFFICIENT_INPUT"
            result.warnings.append(f"Shape mismatch: flood {flood_mask.shape} vs DEM {dem.shape}")
            return result

        # Compute depth: WSE - DEM_elevation at flooded pixels
        flood_bool = flood_mask.astype(bool)
        depth = np.zeros_like(dem, dtype=np.float32)

        depth[flood_bool] = np.maximum(
            0.0,
            water_surface_elevation - dem[flood_bool].astype(np.float32)
        )

        # Zero out non-flooded areas
        depth[~flood_bool] = 0.0

        valid_depths = depth[flood_bool]
        if valid_depths.size == 0:
            result.depth_estimation_status = "COMPLETED"
            result.depth_raster = depth
            result.max_depth = 0.0
            result.mean_depth = 0.0
            return result

        result.depth_raster = depth
        result.max_depth = float(np.max(valid_depths))
        result.mean_depth = float(np.mean(valid_depths))
        result.depth_estimation_status = "COMPLETED"

        # Compute volume if resolution is available
        if dem_resolution is not None and dem_resolution > 0:
            pixel_area_m2 = dem_resolution ** 2
            flooded_volume = float(np.sum(valid_depths)) * pixel_area_m2
            result.estimated_volume_m3 = flooded_volume

        result.assumptions.append(
            f"Water surface elevation = {water_surface_elevation} m (uniform assumption)"
        )
        result.assumptions.append(
            "Depth = max(0, WSE - DEM_elevation) for flooded pixels"
        )

        return result

    def estimate_insufficient(
        self,
        flood_mask: Optional[np.ndarray] = None,
    ) -> FloodDepthResult:
        """Return INSUFFICIENT_INPUT when required reference data is unavailable.

        Call this when:
        - No DEM is available
        - No water surface elevation is known
        - No boundary gauge data is available
        """
        return FloodDepthResult(
            depth_estimation_status="INSUFFICIENT_INPUT",
            warnings=[
                "Flood depth estimation requires water surface elevation data "
                "or boundary reference levels, which are not available.",
                "Provide water_surface_elevation parameter or DEM + gauge data."
            ],
        )
