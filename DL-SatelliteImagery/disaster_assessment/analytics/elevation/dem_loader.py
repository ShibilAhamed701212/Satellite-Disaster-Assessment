"""
DEM (Digital Elevation Model) loading and preprocessing.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class DEMData:
    """Container for DEM raster data."""
    elevation: np.ndarray  # (H, W) float32
    transform: Optional[object] = None
    crs: Optional[str] = None
    resolution: Optional[float] = None  # meters
    bounds: Optional[Tuple[float, float, float, float]] = None
    nodata: Optional[float] = None


class DEMLoader:
    """Load and validate DEM rasters.

    Supports GeoTIFF format via rasterio.
    """

    def __init__(self):
        self._rasterio = None
        try:
            import rasterio
            self._rasterio = rasterio
        except ImportError:
            pass

    def load(self, dem_path: str, target_crs: Optional[str] = None) -> Optional[DEMData]:
        """Load DEM from file.

        Args:
            dem_path: Path to DEM GeoTIFF.
            target_crs: Optional target CRS for reprojection.

        Returns:
            DEMData or None if loading fails.
        """
        if not os.path.isfile(dem_path):
            return None

        if self._rasterio is None:
            return None

        try:
            with self._rasterio.open(dem_path) as src:
                # Read elevation data
                elevation = src.read(1).astype(np.float32)

                # Extract metadata
                transform = src.transform
                crs = src.crs.to_string() if src.crs else None
                bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                nodata = src.nodata

                # Compute resolution
                res_x, res_y = src.res
                if src.crs and src.crs.is_projected:
                    resolution = abs(float(res_x))
                else:
                    resolution = None  # Geographic CRS, resolution in degrees

                # Mask nodata
                if nodata is not None:
                    elevation[elevation == nodata] = np.nan

                return DEMData(
                    elevation=elevation,
                    transform=transform,
                    crs=crs,
                    resolution=resolution,
                    bounds=bounds,
                    nodata=nodata,
                )
        except Exception:
            return None

    def resample_to_match(
        self,
        dem_data: DEMData,
        target_height: int,
        target_width: int,
    ) -> DEMData:
        """Resample DEM to match target dimensions.

        Args:
            dem_data: Source DEM data.
            target_height: Target height in pixels.
            target_width: Target width in pixels.

        Returns:
            Resampled DEMData.
        """
        from scipy.ndimage import zoom

        h, w = dem_data.elevation.shape
        zoom_y = target_height / h
        zoom_x = target_width / w

        resampled = zoom(dem_data.elevation, (zoom_y, zoom_x), order=1)
        resampled = resampled.astype(np.float32)

        return DEMData(
            elevation=resampled,
            transform=dem_data.transform,
            crs=dem_data.crs,
            resolution=dem_data.resolution,
            bounds=dem_data.bounds,
            nodata=dem_data.nodata,
        )
