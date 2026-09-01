"""
SAR intensity normalization and scaling utilities.
"""

import numpy as np


def normalize_sar(
    data: np.ndarray,
    method: str = "minmax",
    percent_clip: tuple = (2, 98),
) -> np.ndarray:
    """Normalize SAR intensity data.

    Args:
        data: SAR intensity or amplitude data.
        method: Normalization method - 'minmax', 'minmax_clip', or 'robust'.
        percent_clip: Percentile range for robust normalization.

    Returns:
        Normalized array in [0, 1] range.
    """
    data_f = data.astype(np.float64)

    # Handle nodata/negative
    valid_mask = data_f > 0

    if not np.any(valid_mask):
        return np.zeros_like(data, dtype=np.float32)

    if method == "minmax":
        dmin = np.min(data_f[valid_mask])
        dmax = np.max(data_f[valid_mask])
        denom = dmax - dmin
        if denom < 1e-10:
            denom = 1.0
        result = (data_f - dmin) / denom
        result = np.clip(result, 0.0, 1.0)

    elif method == "minmax_clip":
        p_low = np.percentile(data_f[valid_mask], percent_clip[0])
        p_high = np.percentile(data_f[valid_mask], percent_clip[1])
        denom = p_high - p_low
        if denom < 1e-10:
            denom = 1.0
        result = (data_f - p_low) / denom
        result = np.clip(result, 0.0, 1.0)

    elif method == "robust":
        median = np.median(data_f[valid_mask])
        mad = np.median(np.abs(data_f[valid_mask] - median))
        if mad < 1e-10:
            mad = 1.0
        result = (data_f - median) / (1.4826 * mad)
        result = np.clip(result, -3.0, 3.0)
        result = (result + 3.0) / 6.0  # Map to [0, 1]
    else:
        result = np.zeros_like(data_f, dtype=np.float32)

    return result.astype(np.float32)


def to_db_scale(
    data: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Convert SAR intensity to decibel (dB) scale.

    dB = 10 * log10(intensity)

    Args:
        data: SAR intensity data (linear power).
        epsilon: Small value to avoid log(0).

    Returns:
        dB-scaled array.
    """
    data_f = data.astype(np.float64)
    data_f = np.maximum(data_f, epsilon)
    return (10.0 * np.log10(data_f)).astype(np.float32)


def from_db_scale(data: np.ndarray) -> np.ndarray:
    """Convert dB scale back to linear power."""
    return np.power(10.0, data.astype(np.float64) / 10.0).astype(np.float32)
