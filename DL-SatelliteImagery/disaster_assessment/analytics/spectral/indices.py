"""
Spectral index computations for multispectral remote sensing.

All functions handle nodata, NaN, and division by zero safely.
"""

from typing import Optional, Tuple

import numpy as np


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Safe division avoiding division by zero."""
    return np.where(
        np.abs(denominator) > epsilon,
        numerator / np.where(np.abs(denominator) > epsilon, denominator, epsilon),
        0.0,
    )


def compute_ndvi(
    nir: np.ndarray,
    red: np.ndarray,
    nodata: Optional[float] = None,
) -> np.ndarray:
    """Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)

    Values range from -1 to 1. Healthy vegetation typically > 0.3.

    Args:
        nir: Near-infrared band.
        red: Red band.
        nodata: Optional nodata value to mask.

    Returns:
        NDVI array in range [-1, 1].
    """
    nir_f = nir.astype(np.float32)
    red_f = red.astype(np.float32)

    if nodata is not None:
        mask = (nir_f == nodata) | (red_f == nodata)
    else:
        mask = np.zeros_like(nir_f, dtype=bool)

    numerator = nir_f - red_f
    denominator = nir_f + red_f

    result = _safe_divide(numerator, denominator)
    result[mask] = np.nan

    return np.clip(result, -1.0, 1.0)


def compute_ndwi(
    green: np.ndarray,
    nir: np.ndarray,
    nodata: Optional[float] = None,
) -> np.ndarray:
    """Normalized Difference Water Index (McFeeters 1996).

    NDWI = (Green - NIR) / (Green + NIR)

    Values > 0 typically indicate water bodies.

    Args:
        green: Green band.
        nir: Near-infrared band.
        nodata: Optional nodata value to mask.

    Returns:
        NDWI array in range [-1, 1].
    """
    green_f = green.astype(np.float32)
    nir_f = nir.astype(np.float32)

    if nodata is not None:
        mask = (green_f == nodata) | (nir_f == nodata)
    else:
        mask = np.zeros_like(green_f, dtype=bool)

    numerator = green_f - nir_f
    denominator = green_f + nir_f

    result = _safe_divide(numerator, denominator)
    result[mask] = np.nan

    return np.clip(result, -1.0, 1.0)


def compute_mndwi(
    green: np.ndarray,
    swir: np.ndarray,
    nodata: Optional[float] = None,
) -> np.ndarray:
    """Modified NDWI (Xu 2006), using SWIR instead of NIR.

    MNDWI = (Green - SWIR) / (Green + SWIR)

    Better at distinguishing water from built-up areas.

    Args:
        green: Green band.
        swir: Short-wave infrared band.
        nodata: Optional nodata value to mask.

    Returns:
        MNDWI array in range [-1, 1].
    """
    green_f = green.astype(np.float32)
    swir_f = swir.astype(np.float32)

    if nodata is not None:
        mask = (green_f == nodata) | (swir_f == nodata)
    else:
        mask = np.zeros_like(green_f, dtype=bool)

    numerator = green_f - swir_f
    denominator = green_f + swir_f

    result = _safe_divide(numerator, denominator)
    result[mask] = np.nan

    return np.clip(result, -1.0, 1.0)


def compute_nbr(
    nir: np.ndarray,
    swir: np.ndarray,
    nodata: Optional[float] = None,
) -> np.ndarray:
    """Normalized Burn Ratio.

    NBR = (NIR - SWIR) / (NIR + SWIR)

    Pre-fire NBR helps identify burn severity when compared to post-fire.

    Args:
        nir: Near-infrared band.
        swir: Short-wave infrared band.
        nodata: Optional nodata value to mask.

    Returns:
        NBR array in range [-1, 1].
    """
    nir_f = nir.astype(np.float32)
    swir_f = swir.astype(np.float32)

    if nodata is not None:
        mask = (nir_f == nodata) | (swir_f == nodata)
    else:
        mask = np.zeros_like(nir_f, dtype=bool)

    numerator = nir_f - swir_f
    denominator = nir_f + swir_f

    result = _safe_divide(numerator, denominator)
    result[mask] = np.nan

    return np.clip(result, -1.0, 1.0)


def compute_dnbr(
    nbr_pre: np.ndarray,
    nbr_post: np.ndarray,
    nodata: Optional[float] = None,
) -> Tuple[np.ndarray, dict]:
    """Delta Normalized Burn Ratio (dNBR).

    dNBR = NBR_pre - NBR_post

    Higher values indicate greater burn severity.

    Severity thresholds (USGS):
        < 0.1:   Unburned
        0.1-0.27: Low severity
        0.27-0.44: Moderate-low severity
        0.44-0.66: Moderate-high severity
        > 0.66:   High severity

    Args:
        nbr_pre: Pre-fire NBR.
        nbr_post: Post-fire NBR.
        nodata: Optional nodata value to mask.

    Returns:
        Tuple of (dNBR array, metadata dict).
    """
    nbr_pre_f = nbr_pre.astype(np.float32)
    nbr_post_f = nbr_post.astype(np.float32)

    if nodata is not None:
        mask = np.isnan(nbr_pre_f) | np.isnan(nbr_post_f) | \
               (nbr_pre_f == nodata) | (nbr_post_f == nodata)
    else:
        mask = np.isnan(nbr_pre_f) | np.isnan(nbr_post_f)

    dnbr = nbr_pre_f - nbr_post_f
    dnbr[mask] = np.nan

    dnbr = np.clip(dnbr, -2.0, 2.0)

    # Classify severity
    severity_map = np.full_like(dnbr, -1, dtype=np.int32)
    valid = ~mask
    severity_map[valid & (dnbr < 0.1)] = 0      # Unburned
    severity_map[valid & (dnbr >= 0.1) & (dnbr < 0.27)] = 1  # Low
    severity_map[valid & (dnbr >= 0.27) & (dnbr < 0.44)] = 2  # Moderate-low
    severity_map[valid & (dnbr >= 0.44) & (dnbr < 0.66)] = 3  # Moderate-high
    severity_map[valid & (dnbr >= 0.66)] = 4    # High

    valid_pixels = int(valid.sum())
    metadata = {
        "total_pixels": int(valid.size),
        "valid_pixels": valid_pixels,
        "mean_dnbr": float(np.nanmean(dnbr)) if valid_pixels > 0 else 0.0,
        "max_dnbr": float(np.nanmax(dnbr)) if valid_pixels > 0 else 0.0,
        "unburned_pixels": int((severity_map == 0).sum()),
        "low_severity_pixels": int((severity_map == 1).sum()),
        "mod_low_severity_pixels": int((severity_map == 2).sum()),
        "mod_high_severity_pixels": int((severity_map == 3).sum()),
        "high_severity_pixels": int((severity_map == 4).sum()),
    }

    return dnbr, metadata
