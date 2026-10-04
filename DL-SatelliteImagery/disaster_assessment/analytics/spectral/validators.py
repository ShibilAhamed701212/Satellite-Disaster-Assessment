"""
Validators for spectral band inputs.
"""

from typing import List, Tuple

import numpy as np

from .band_mapping import BandMapping


def validate_bands(
    data: np.ndarray,
    required_bands: List[str],
    band_mapping: BandMapping,
) -> Tuple[bool, str]:
    """Validate that required spectral bands are available in the input.

    Args:
        data: Input array of shape (bands, H, W) or (H, W, bands).
        required_bands: List of required band names (e.g., ["nir", "red"]).
        band_mapping: BandMapping describing the sensor's band layout.

    Returns:
        Tuple of (is_valid, message).
    """
    # Determine number of bands
    if data.ndim == 3:
        # Could be (C, H, W) or (H, W, C)
        if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
            num_bands = data.shape[0]  # (C, H, W)
        else:
            num_bands = data.shape[2]  # (H, W, C)
    elif data.ndim == 2:
        num_bands = 1
    else:
        return False, f"Invalid data shape: {data.shape}"

    # Check band availability
    missing = []
    for band_name in required_bands:
        if band_name not in band_mapping.bands:
            missing.append(band_name)
            continue
        band_idx = band_mapping.bands[band_name]
        if isinstance(band_idx, int) and band_idx > num_bands:
            missing.append(f"{band_name} (index {band_idx} > available bands {num_bands})")

    if missing:
        return False, f"Missing required bands: {', '.join(missing)}"

    # Check for NaN/inf
    if np.any(np.isinf(data.astype(np.float64))):
        return True, "Warning: Data contains infinite values"

    return True, "All required bands available"


def extract_band(
    data: np.ndarray,
    band_name: str,
    band_mapping: BandMapping,
) -> np.ndarray:
    """Extract a single band from multispectral data.

    Args:
        data: (C, H, W) or (H, W, C) array.
        band_name: Name of band to extract.
        band_mapping: Sensor band mapping.

    Returns:
        2D array for the requested band.
    """
    if band_name not in band_mapping.bands:
        raise ValueError(f"Band '{band_name}' not available for sensor '{band_mapping.sensor}'")

    band_idx = band_mapping.bands[band_name]
    if not isinstance(band_idx, int):
        raise ValueError(f"Band index for '{band_name}' must be an integer, got {type(band_idx)}")

    # Determine axis layout
    if data.ndim == 3:
        if data.shape[0] < data.shape[1]:
            # (C, H, W) layout
            if band_idx - 1 >= data.shape[0]:
                raise ValueError(f"Band index {band_idx} out of range for {data.shape[0]} channels")
            return data[band_idx - 1]
        else:
            # (H, W, C) layout
            if band_idx - 1 >= data.shape[2]:
                raise ValueError(f"Band index {band_idx} out of range for {data.shape[2]} channels")
            return data[:, :, band_idx - 1]

    raise ValueError("Cannot extract band from 2D data")
