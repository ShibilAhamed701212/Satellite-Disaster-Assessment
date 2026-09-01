"""
Multispectral remote-sensing indices for vegetation, water, and burn analysis.

Supports configurable band mappings for Sentinel-2, Landsat, and custom GeoTIFF.
"""

from .indices import compute_ndvi, compute_ndwi, compute_mndwi, compute_nbr, compute_dnbr
from .band_mapping import BandMapping, SENTINEL2_BANDS, LANDSAT_BANDS
from .validators import validate_bands

__all__ = [
    "compute_ndvi", "compute_ndwi", "compute_mndwi", "compute_nbr", "compute_dnbr",
    "BandMapping", "SENTINEL2_BANDS", "LANDSAT_BANDS",
    "validate_bands",
]
