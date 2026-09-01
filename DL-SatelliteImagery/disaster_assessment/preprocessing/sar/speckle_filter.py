"""
SAR speckle noise filtering.
"""

import numpy as np
from scipy.ndimage import median_filter, uniform_filter


def median_filter_sar(
    data: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Apply median filter for speckle reduction.

    Simple and effective for preserving edges while reducing speckle.

    Args:
        data: SAR intensity or normalized data.
        kernel_size: Filter kernel size (must be odd).

    Returns:
        Filtered array.
    """
    if kernel_size < 1:
        return data.copy()
    if kernel_size % 2 == 0:
        kernel_size += 1

    return median_filter(data.astype(np.float32), size=kernel_size)


def lee_filter_sar(
    data: np.ndarray,
    kernel_size: int = 5,
    sigma_noise: float = 0.04,
) -> np.ndarray:
    """Apply Lee speckle filter.

    Adaptive filter that preserves edges and reduces speckle in homogeneous regions.

    Args:
        data: SAR intensity data.
        kernel_size: Filter window size (odd).
        sigma_noise: Estimated noise variance (speckle noise level).

    Returns:
        Filtered array.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    data_f = data.astype(np.float64)
    result = data_f.copy()
    half_k = kernel_size // 2
    h, w = data_f.shape

    # Local statistics
    local_mean = uniform_filter(data_f, size=kernel_size)
    local_sq_mean = uniform_filter(data_f ** 2, size=kernel_size)
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 0)

    # Coefficient of variation
    cv_signal_sq = np.maximum(local_var - sigma_noise, 0) / (local_mean ** 2 + 1e-10)
    cv_noise_sq = sigma_noise / (local_mean ** 2 + 1e-10)

    # Weight: small weight in edges (high variance), large in homogeneous areas
    weight = np.where(
        cv_signal_sq > 0,
        np.minimum(1.0, cv_noise_sq / (cv_signal_sq + 1e-10)),
        1.0,
    )

    result = local_mean + weight * (data_f - local_mean)
    return result.astype(np.float32)
