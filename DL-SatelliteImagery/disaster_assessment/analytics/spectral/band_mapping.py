"""
Configurable band mappings for multispectral satellite sensors.

Different satellites have different band orders and names.
This module provides standard mappings and a configuration system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BandMapping:
    """Maps spectral band names to raster band indices (1-based).

    Attributes:
        sensor: Name of the satellite sensor.
        bands: Dict mapping band_name -> band_index (1-based).
        rgb: Tuple of (R, G, B) band names for true-color composites.
        nir: NIR band name.
        swir: SWIR band name(s).
    """
    sensor: str
    bands: Dict[str, int]
    rgb: tuple = ("red", "green", "blue")
    nir: str = "nir"
    swir: str = "swir"
    swir2: Optional[str] = None

    def get_band_indices(self, band_names: List[str]) -> List[int]:
        """Convert band name list to 1-based indices."""
        indices = []
        for name in band_names:
            if name in self.bands:
                indices.append(self.bands[name])
            else:
                raise ValueError(f"Band '{name}' not found for sensor '{self.sensor}'")
        return indices

    def has_bands(self, band_names: List[str]) -> bool:
        """Check if all requested bands are available."""
        return all(name in self.bands for name in band_names)


# Sentinel-2 L2A band mapping
SENTINEL2_BANDS = BandMapping(
    sensor="Sentinel-2",
    bands={
        "blue": 2,    # B02 - 490nm
        "green": 3,   # B03 - 560nm
        "red": 4,     # B04 - 665nm
        "rededge1": 5,  # B05 - 705nm
        "rededge2": 6,  # B06 - 740nm
        "rededge3": 7,  # B07 - 783nm
        "nir": 8,     # B08 - 842nm
        "nir_narrow": 8,  # B8A - 865nm
        "swir": 11,   # B11 - 1610nm
        "swir2": 12,  # B12 - 2190nm
        "cloud": 1,
    },
    rgb=("red", "green", "blue"),
    nir="nir",
    swir="swir",
    swir2="swir2",
)

# Landsat 8/9 OLI band mapping
LANDSAT_BANDS = BandMapping(
    sensor="Landsat-8/9",
    bands={
        "blue": 2,    # B2 - 482nm
        "green": 3,   # B3 - 561nm
        "red": 4,     # B4 - 655nm
        "nir": 5,     # B5 - 865nm
        "swir": 6,    # B6 - 1609nm
        "swir2": 7,   # B7 - 2201nm
        "pan": 8,     # B8 - 590nm (panchromatic)
        "cirrus": 9,  # B9 - 1373nm
    },
    rgb=("red", "green", "blue"),
    nir="nir",
    swir="swir",
    swir2="swir2",
)

# Generic RGB GeoTIFF (3-band)
RGB_BANDS = BandMapping(
    sensor="RGB",
    bands={
        "red": 1,
        "green": 2,
        "blue": 3,
    },
    rgb=("red", "green", "blue"),
)

# Registry of known sensor mappings
SENSOR_REGISTRY: Dict[str, BandMapping] = {
    "sentinel-2": SENTINEL2_BANDS,
    "sentinel2": SENTINEL2_BANDS,
    "landsat-8": LANDSAT_BANDS,
    "landsat-9": LANDSAT_BANDS,
    "landsat": LANDSAT_BANDS,
    "rgb": RGB_BANDS,
}


def get_band_mapping(sensor: str) -> Optional[BandMapping]:
    """Look up a band mapping by sensor name (case-insensitive)."""
    return SENSOR_REGISTRY.get(sensor.lower().strip())
