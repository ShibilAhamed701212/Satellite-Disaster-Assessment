"""
Sentinel-1 SAR data processor.

Handles VV/VH polarization inputs, pre/post disaster analysis,
and SAR-specific change features for flood detection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .normalization import normalize_sar, to_db_scale
from .speckle_filter import median_filter_sar
from .coherence import compute_sar_change_features


@dataclass
class SARInput:
    """Container for SAR input data."""
    vv: Optional[np.ndarray] = None
    vh: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class SARChangeResult:
    """Results from pre/post SAR change analysis."""
    vv_difference: Optional[np.ndarray] = None
    vh_difference: Optional[np.ndarray] = None
    vv_ratio: Optional[np.ndarray] = None
    vh_ratio: Optional[np.ndarray] = None
    flood_mask: Optional[np.ndarray] = None
    change_magnitude: Optional[np.ndarray] = None
    features_computed: List[str] = field(default_factory=list)
    status: str = "COMPLETED"


class Sentinel1Processor:
    """Process Sentinel-1 SAR data for disaster analysis.

    Status: IMPLEMENTED ARCHITECTURE
    Requires real SAR data for validation.
    """

    def __init__(
        self,
        enable_speckle_filter: bool = True,
        speckle_kernel_size: int = 5,
        flood_threshold_db: float = -2.0,
    ):
        self.enable_speckle_filter = enable_speckle_filter
        self.speckle_kernel_size = speckle_kernel_size
        self.flood_threshold_db = flood_threshold_db

    def preprocess(self, sar_input: SARInput) -> SARInput:
        """Preprocess SAR data: normalize, filter speckle.

        Args:
            sar_input: Raw SAR input with VV and/or VH bands.

        Returns:
            Preprocessed SARInput.
        """
        result = SARInput(metadata=dict(sar_input.metadata))

        if sar_input.vv is not None:
            vv = normalize_sar(sar_input.vv)
            if self.enable_speckle_filter:
                vv = median_filter_sar(vv, kernel_size=self.speckle_kernel_size)
            result.vv = vv

        if sar_input.vh is not None:
            vh = normalize_sar(sar_input.vh)
            if self.enable_speckle_filter:
                vh = median_filter_sar(vh, kernel_size=self.speckle_kernel_size)
            result.vh = vh

        return result

    def compute_change(
        self,
        pre: SARInput,
        post: SARInput,
        include_ratio: bool = True,
    ) -> SARChangeResult:
        """Compute pre/post SAR change features.

        Args:
            pre: Pre-disaster SAR input.
            post: Post-disaster SAR input.
            include_ratio: Whether to compute VV/VH ratios.

        Returns:
            SARChangeResult with change features.
        """
        result = SARChangeResult()

        # Preprocess both inputs
        pre_proc = self.preprocess(pre)
        post_proc = self.preprocess(post)

        # Compute VV change features
        if pre_proc.vv is not None and post_proc.vv is not None:
            result.vv_difference = compute_sar_change_features(pre_proc.vv, post_proc.vv, "difference")
            result.features_computed.append("vv_difference")

            if include_ratio:
                result.vv_ratio = compute_sar_change_features(pre_proc.vv, post_proc.vv, "ratio")
                result.features_computed.append("vv_ratio")

        # Compute VH change features
        if pre_proc.vh is not None and post_proc.vh is not None:
            result.vh_difference = compute_sar_change_features(pre_proc.vh, post_proc.vh, "difference")
            result.features_computed.append("vh_difference")

            if include_ratio:
                result.vh_ratio = compute_sar_change_features(pre_proc.vh, post_proc.vh, "ratio")
                result.features_computed.append("vh_ratio")

        # Flood mask from VV decrease (water causes strong backscatter reduction)
        if result.vv_difference is not None:
            # Convert pre/post to dB scale for threshold comparison
            # The normalized difference is not in dB, so we compute dB difference directly
            pre_db = to_db_scale(np.maximum(pre_proc.vv, 1e-10))
            post_db = to_db_scale(np.maximum(post_proc.vv, 1e-10))
            db_diff = post_db - pre_db
            # Flood: VV backscatter drops by > threshold in dB
            result.flood_mask = (db_diff < self.flood_threshold_db).astype(np.uint8)
            result.features_computed.append("flood_mask")

            # Also include change magnitude
            result.change_magnitude = np.abs(result.vv_difference)
            if result.vh_difference is not None:
                result.change_magnitude = np.maximum(result.change_magnitude, np.abs(result.vh_difference))
            result.features_computed.append("change_magnitude")

        if not result.features_computed:
            result.status = "INSUFFICIENT_INPUT"
        else:
            result.status = "COMPLETED"

        return result

    def estimate_coherence(
        self,
        pre_vv: np.ndarray,
        post_vv: np.ndarray,
        window_size: int = 5,
    ) -> Optional[np.ndarray]:
        """Estimate interferometric coherence between pre/post SAR.

        WARNING: True interferometric coherence requires complex (phase) data.
        This implementation estimates magnitude-based coherence from intensity
        data only and should NOT be claimed as true interferometric coherence.

        Args:
            pre_vv: Pre-disaster VV intensity.
            post_vv: Post-disaster VV intensity.
            window_size: Sliding window size for estimation.

        Returns:
            Estimated coherence map, or None if inputs invalid.
        """
        # Validate inputs
        if pre_vv is None or post_vv is None:
            return None
        if pre_vv.shape != post_vv.shape:
            return None

        # Magnitude coherence estimate over a spatial window
        # Not true interferometric coherence (requires complex data)
        # Uses windowed correlation coefficient as proxy
        from scipy.ndimage import uniform_filter

        pre_f = pre_vv.astype(np.float64)
        post_f = post_vv.astype(np.float64)

        # Windowed cross-correlation numerator: sum(pre * post)
        cross = uniform_filter(pre_f * post_f, size=window_size)
        # Windowed power
        pre_power = uniform_filter(pre_f ** 2, size=window_size)
        post_power = uniform_filter(post_f ** 2, size=window_size)

        # Coherence = |cross| / sqrt(pre_power * post_power)
        coherence = np.abs(cross) / np.sqrt(np.maximum(pre_power * post_power, 1e-20))

        return np.clip(coherence, 0.0, 1.0)
