"""
Terrain analysis from DEM data.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class TerrainResult:
    """Terrain analysis results."""
    slope: Optional[np.ndarray] = None       # degrees
    aspect: Optional[np.ndarray] = None      # degrees from north
    hillshade: Optional[np.ndarray] = None   # 0-255
    mean_slope: float = 0.0
    max_slope: float = 0.0
    mean_elevation: float = 0.0
    elevation_range: Tuple[float, float] = (0.0, 0.0)


class TerrainAnalyzer:
    """Compute terrain features from DEM data."""

    def analyze(
        self,
        elevation: np.ndarray,
        resolution: Optional[float] = None,
    ) -> TerrainResult:
        """Compute terrain features from elevation data.

        Args:
            elevation: (H, W) float32 elevation array.
            resolution: Pixel size in meters. If None, slope values are approximate.

        Returns:
            TerrainResult with slope, aspect, hillshade.
        """
        if elevation is None or elevation.size == 0:
            return TerrainResult()

        elev = elevation.astype(np.float64)

        # Compute gradient for slope
        dy, dx = np.gradient(elev, resolution or 1.0)

        # Slope in degrees
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)

        # Aspect (direction of steepest slope, measured from north clockwise)
        aspect_rad = np.arctan2(-dx, dy)
        aspect_deg = np.degrees(aspect_rad) % 360

        # Simple hillshade
        azimuth = 315.0
        altitude = 45.0
        az_rad = np.radians(azimuth)
        alt_rad = np.radians(altitude)

        hillshade = (
            np.cos(alt_rad) * np.cos(slope_rad) +
            np.sin(alt_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect_rad)
        )
        hillshade = np.clip(hillshade * 255, 0, 255).astype(np.uint8)

        valid_elev = elev[~np.isnan(elev)]

        return TerrainResult(
            slope=slope_deg.astype(np.float32),
            aspect=aspect_deg.astype(np.float32),
            hillshade=hillshade,
            mean_slope=float(np.nanmean(slope_deg)) if valid_elev.size > 0 else 0.0,
            max_slope=float(np.nanmax(slope_deg)) if valid_elev.size > 0 else 0.0,
            mean_elevation=float(np.nanmean(valid_elev)) if valid_elev.size > 0 else 0.0,
            elevation_range=(
                float(np.nanmin(valid_elev)) if valid_elev.size > 0 else 0.0,
                float(np.nanmax(valid_elev)) if valid_elev.size > 0 else 0.0,
            ),
        )
