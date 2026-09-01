"""
SAR change feature computation for pre/post disaster analysis.
"""

import numpy as np


def compute_sar_change_features(
    pre: np.ndarray,
    post: np.ndarray,
    method: str = "difference",
) -> np.ndarray:
    """Compute change features from pre/post SAR bands.

    Args:
        pre: Pre-disaster SAR band.
        post: Post-disaster SAR band.
        method: Change metric - 'difference', 'ratio', or 'log_ratio'.

    Returns:
        Change feature array.
    """
    pre_f = pre.astype(np.float64)
    post_f = post.astype(np.float64)

    if method == "difference":
        return (post_f - pre_f).astype(np.float32)

    elif method == "ratio":
        # log(ratio) is more common in SAR literature
        epsilon = 1e-10
        ratio = (post_f + epsilon) / (pre_f + epsilon)
        return np.log(ratio).astype(np.float32)

    elif method == "log_ratio":
        epsilon = 1e-10
        return (np.log(post_f + epsilon) - np.log(pre_f + epsilon)).astype(np.float32)

    else:
        raise ValueError(f"Unknown SAR change method: {method}")
