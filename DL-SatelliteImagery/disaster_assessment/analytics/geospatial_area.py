"""
Georeferenced Flood and Disaster Area Calculator.

Supports 3 distinct operational modes:
    1. MODE 1 — GeoTIFF with Georeferenced Metadata:
       - Extracts CRS, Affine transform, pixel dimensions, and scene boundaries.
       - Correctly handles Projected CRS (direct meter-based area) and
         Geographic CRS (EPSG:4326 with geodesic latitude-adjusted area).
       - Never blindly treats degree coordinates as meters.

    2. MODE 2 — User-Supplied Ground Resolution (meters_per_pixel):
       - Computes pixel_area_m2 = meters_per_pixel^2 for standard PNG/JPG files.

    3. MODE 3 — No Georeference Available:
       - Reports pixel counts and percentages only.
       - Never invents km^2 without geospatial resolution.

Also includes patch-based full-resolution mask reconstruction to ensure area
measurements correspond to original raster dimensions.
"""

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class GeoreferenceInfo:
    """Metadata extracted from geospatial raster files."""

    has_georeference: bool = False
    source_type: str = "none"  # "geotiff", "user_supplied", or "none"
    crs_name: Optional[str] = None
    is_projected: bool = False
    pixel_width_m: Optional[float] = None
    pixel_height_m: Optional[float] = None
    pixel_area_m2: Optional[float] = None
    bounds: Optional[Tuple[float, float, float, float]] = None  # (left, bottom, right, top)
    raster_width: int = 0
    raster_height: int = 0
    warnings: List[str] = field(default_factory=list)


@dataclass
class GeospatialFloodResult:
    """Comprehensive flood area measurement result."""

    flooded_pixels: int = 0
    total_pixels: int = 0
    flood_percentage: float = 0.0
    flood_area_m2: Optional[float] = None
    flood_area_km2: Optional[float] = None
    area_status: str = "Geospatial resolution unavailable"
    georeference: GeoreferenceInfo = field(default_factory=GeoreferenceInfo)

    def to_dict(self) -> dict:
        return {
            "flooded_pixels": int(self.flooded_pixels),
            "total_pixels": int(self.total_pixels),
            "flood_percentage": round(self.flood_percentage, 2),
            "flood_area_m2": round(self.flood_area_m2, 2) if self.flood_area_m2 is not None else None,
            "flood_area_km2": round(self.flood_area_km2, 6) if self.flood_area_km2 is not None else None,
            "area_status": self.area_status,
            "source_type": self.georeference.source_type,
            "crs": self.georeference.crs_name,
        }

    def summary(self) -> str:
        lines = [
            "Geospatial Flood Area Assessment:",
            f"  • Flooded Pixels:   {self.flooded_pixels:,} / {self.total_pixels:,} ({self.flood_percentage:.2f}%)",
        ]
        if self.flood_area_m2 is not None:
            lines.append(f"  • Flood Area (m²):  {self.flood_area_m2:,.2f} m²")
        if self.flood_area_km2 is not None:
            lines.append(f"  • Flood Area (km²): {self.flood_area_km2:,.6f} km²")
        lines.append(f"  • Status / Basis:   {self.area_status}")
        return "\n".join(lines)


class GeospatialAreaCalculator:
    """
    Geospatial Area Engine with GeoTIFF parsing and geodesic area calculations.
    """

    @staticmethod
    def extract_geotiff_metadata(file_path: str) -> GeoreferenceInfo:
        """
        Extract geospatial metadata from GeoTIFF file.
        Uses rasterio if available, otherwise gracefully informs user.
        """
        info = GeoreferenceInfo(raster_width=0, raster_height=0)

        if not os.path.isfile(file_path):
            info.warnings.append(f"File not found: {file_path}")
            return info

        try:
            import rasterio

            with rasterio.open(file_path) as src:
                info.raster_width = src.width
                info.raster_height = src.height
                info.bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)

                if src.crs is not None:
                    info.has_georeference = True
                    info.source_type = "geotiff"
                    info.crs_name = src.crs.to_string()
                    info.is_projected = src.crs.is_projected

                    res_x, res_y = src.res  # (pixel_width, pixel_height)

                    if info.is_projected:
                        # Projected CRS: resolution is already in meters (e.g. UTM)
                        info.pixel_width_m = abs(float(res_x))
                        info.pixel_height_m = abs(float(res_y))
                        info.pixel_area_m2 = info.pixel_width_m * info.pixel_height_m
                    else:
                        # Geographic CRS (e.g. EPSG:4326): resolution is in degrees.
                        # Calculate geodesic ground resolution at the scene's central latitude.
                        center_lat = (src.bounds.bottom + src.bounds.top) / 2.0
                        info.pixel_width_m, info.pixel_height_m, info.pixel_area_m2 = (
                            GeospatialAreaCalculator._degrees_to_meters(
                                abs(float(res_x)), abs(float(res_y)), center_lat
                            )
                        )
                else:
                    info.warnings.append("GeoTIFF contains no Coordinate Reference System (CRS).")

        except ImportError:
            info.warnings.append(
                "Optional dependency 'rasterio' is not available. "
                "Install rasterio for automated GeoTIFF parsing, or provide meters_per_pixel."
            )
        except Exception as e:
            info.warnings.append(f"Failed to read GeoTIFF metadata: {str(e)}")

        return info

    @staticmethod
    def _degrees_to_meters(
        dx_deg: float, dy_deg: float, center_lat_deg: float
    ) -> Tuple[float, float, float]:
        """
        Convert angular degree resolution to metric ground distance (meters)
        using the WGS-84 ellipsoidal model at the given latitude.
        """
        lat_rad = math.radians(center_lat_deg)

        # WGS-84 metric meters per degree
        meters_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
        meters_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)

        width_m = dx_deg * meters_per_deg_lon
        height_m = dy_deg * meters_per_deg_lat
        area_m2 = width_m * height_m
        return width_m, height_m, area_m2

    @classmethod
    def calculate_flood_area(
        cls,
        flood_mask: np.ndarray,
        geotiff_path: Optional[str] = None,
        meters_per_pixel: Optional[float] = None,
        source_shape: Optional[Tuple[int, int]] = None,
    ) -> GeospatialFloodResult:
        """
        Calculate flood area with support for GeoTIFF, meters_per_pixel, or pixel fallback.

        The mask may be a resized copy of the source image (the models run at a
        fixed target size). Ground resolution always describes the source image,
        so each mask pixel covers (source pixels / mask pixels) source pixels.

        Args:
            flood_mask: 2D binary numpy array (1 = flooded, 0 = non-flooded).
            geotiff_path: Optional path to GeoTIFF file. The mask is assumed to
                cover the whole raster.
            meters_per_pixel: Optional user-supplied ground resolution of the
                source image.
            source_shape: (height, width) of the source image the mask was
                resized from. Used with meters_per_pixel; defaults to the mask
                shape (no resampling).

        Returns:
            GeospatialFloodResult with exact area and measurement status.
        """
        total_pixels = int(flood_mask.size)
        flooded_pixels = int((flood_mask == 1).sum())
        flood_percentage = (flooded_pixels / max(total_pixels, 1)) * 100.0

        result = GeospatialFloodResult(
            flooded_pixels=flooded_pixels,
            total_pixels=total_pixels,
            flood_percentage=flood_percentage,
        )

        # MODE 1: GeoTIFF metadata extraction
        geotiff_warnings: List[str] = []
        if geotiff_path and not os.path.isfile(geotiff_path):
            geotiff_warnings.append(f"GeoTIFF not found: {geotiff_path}")
        if geotiff_path and os.path.isfile(geotiff_path):
            geo_info = cls.extract_geotiff_metadata(geotiff_path)
            result.georeference = geo_info
            geotiff_warnings = list(geo_info.warnings)

            if geo_info.has_georeference and geo_info.pixel_area_m2 is not None:
                scale = cls._resample_factor(
                    flood_mask.shape, (geo_info.raster_height, geo_info.raster_width)
                )
                result.flood_area_m2 = flooded_pixels * geo_info.pixel_area_m2 * scale
                result.flood_area_km2 = result.flood_area_m2 / 1_000_000.0
                crs_label = geo_info.crs_name or "Unknown CRS"
                result.area_status = (
                    f"Geospatially calculated from GeoTIFF ({crs_label}, "
                    f"Pixel Area = {geo_info.pixel_area_m2:.3f} m²)"
                )
                return result

        # MODE 2: User-supplied meters_per_pixel
        if meters_per_pixel is not None and meters_per_pixel > 0:
            scale = cls._resample_factor(flood_mask.shape, source_shape)
            pixel_area_m2 = float(meters_per_pixel) ** 2 * scale
            result.flood_area_m2 = flooded_pixels * pixel_area_m2
            result.flood_area_km2 = result.flood_area_m2 / 1_000_000.0
            result.georeference = GeoreferenceInfo(
                has_georeference=True,
                source_type="user_supplied",
                pixel_width_m=float(meters_per_pixel),
                pixel_height_m=float(meters_per_pixel),
                pixel_area_m2=pixel_area_m2,
                warnings=geotiff_warnings,
            )
            result.area_status = (
                f"Calculated from user ground resolution (GSD = {meters_per_pixel:.3f} m/pixel, "
                f"Pixel Area = {pixel_area_m2:.3f} m²)"
            )
            if scale != 1.0:
                result.area_status += (
                    f" — mask resampled from {source_shape[1]}×{source_shape[0]} source image"
                )
            return result

        # MODE 3: No geospatial resolution available
        result.area_status = "Geospatial resolution unavailable — reporting pixel and percentage area only."
        result.georeference = GeoreferenceInfo(source_type="none", warnings=geotiff_warnings)
        return result

    @staticmethod
    def _resample_factor(
        mask_shape: Tuple[int, ...], source_shape: Optional[Tuple[int, int]]
    ) -> float:
        """Source pixels represented by one mask pixel (1.0 when not resampled)."""
        if not source_shape or source_shape[0] <= 0 or source_shape[1] <= 0:
            return 1.0
        mask_pixels = int(mask_shape[0]) * int(mask_shape[1])
        if mask_pixels <= 0:
            return 1.0
        return (int(source_shape[0]) * int(source_shape[1])) / mask_pixels

    @staticmethod
    def reconstruct_full_mask(
        patches: List[np.ndarray],
        grid_rows: int,
        grid_cols: int,
        original_h: int,
        original_w: int,
        patch_size: int = 256,
    ) -> np.ndarray:
        """
        Reconstruct a stitched full-resolution mask from tiled model prediction patches.
        """
        if not patches:
            return np.zeros((original_h, original_w), dtype=np.uint8)

        canvas = np.zeros((grid_rows * patch_size, grid_cols * patch_size), dtype=patches[0].dtype)
        idx = 0
        for r in range(grid_rows):
            for c in range(grid_cols):
                if idx < len(patches):
                    canvas[
                        r * patch_size : (r + 1) * patch_size,
                        c * patch_size : (c + 1) * patch_size,
                    ] = patches[idx]
                    idx += 1

        # Crop to exact original dimensions
        return canvas[:original_h, :original_w]
