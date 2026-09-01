"""
Area calculation utilities for disaster assessment.

Supports:
    - Pixel-based area (count and percentage)
    - Geospatially-accurate area when GSD (Ground Sample Distance) is provided
    - Clear labeling of measurement basis
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AreaResult:
    """Structured area measurement result."""

    pixel_count: int
    total_pixels: int
    percentage: float
    area_m2: Optional[float] = None
    area_km2: Optional[float] = None
    measurement_basis: str = "pixel-based estimate"

    def summary(self) -> str:
        lines = [
            f"  Pixels:     {self.pixel_count:,} / {self.total_pixels:,}",
            f"  Percentage: {self.percentage:.2f}%",
        ]
        if self.area_m2 is not None:
            lines.append(f"  Area (m²):  {self.area_m2:,.2f}")
        if self.area_km2 is not None:
            lines.append(f"  Area (km²): {self.area_km2:,.6f}")
        lines.append(f"  Basis:      {self.measurement_basis}")
        return "\n".join(lines)


class AreaCalculator:
    """
    Calculates area from binary or multi-class masks.

    If GSD (ground sample distance in meters/pixel) is provided,
    real-world area is computed. Otherwise, only pixel counts and
    percentages are reported.

    Important:
        - Pixel percentage is ALWAYS reported.
        - m²/km² are ONLY reported when GSD is provided.
        - The output clearly labels whether the area is pixel-based or geospatial.
    """

    def __init__(self, gsd_meters: Optional[float] = None):
        """
        Args:
            gsd_meters: Ground Sample Distance in meters/pixel.
                        If None, only pixel-based measurements are computed.
        """
        self.gsd_meters = gsd_meters
        self.pixel_area_m2 = gsd_meters ** 2 if gsd_meters is not None else None

    def calculate_mask_area(
        self,
        mask: np.ndarray,
        target_value: int = 1,
    ) -> AreaResult:
        """
        Calculate area for a specific class in a mask.

        Args:
            mask: 2D array (H, W) with integer class labels.
            target_value: Class value to measure.

        Returns:
            AreaResult with pixel counts, percentage, and optionally real area.
        """
        total_pixels = mask.size
        pixel_count = int(np.sum(mask == target_value))
        percentage = (pixel_count / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        result = AreaResult(
            pixel_count=pixel_count,
            total_pixels=total_pixels,
            percentage=percentage,
        )

        if self.pixel_area_m2 is not None:
            result.area_m2 = pixel_count * self.pixel_area_m2
            result.area_km2 = result.area_m2 / 1_000_000.0
            result.measurement_basis = (
                f"geospatially calculated (GSD={self.gsd_meters:.2f} m/pixel)"
            )
        else:
            result.measurement_basis = "pixel-based estimate (no GSD provided)"

        return result

    def calculate_change_area(
        self,
        pre_mask: np.ndarray,
        post_mask: np.ndarray,
        class_value: int,
    ) -> dict:
        """
        Calculate area changes for a specific class between pre and post masks.

        Args:
            pre_mask: Pre-disaster class mask (H, W).
            post_mask: Post-disaster class mask (H, W).
            class_value: Class label to analyze.

        Returns:
            Dictionary with pre_area, post_area, change (gain/loss), and percentage change.
        """
        pre_area = self.calculate_mask_area(pre_mask, class_value)
        post_area = self.calculate_mask_area(post_mask, class_value)

        pixel_change = post_area.pixel_count - pre_area.pixel_count
        pct_change = (
            (pixel_change / pre_area.pixel_count * 100.0)
            if pre_area.pixel_count > 0
            else 0.0
        )

        result = {
            "pre_area": pre_area,
            "post_area": post_area,
            "pixel_change": pixel_change,
            "percentage_change": pct_change,
            "direction": "gain" if pixel_change > 0 else ("loss" if pixel_change < 0 else "no change"),
        }

        if self.pixel_area_m2 is not None:
            result["area_change_m2"] = pixel_change * self.pixel_area_m2
            result["area_change_km2"] = result["area_change_m2"] / 1_000_000.0

        return result

    def total_changed_area(
        self,
        change_mask: np.ndarray,
    ) -> AreaResult:
        """
        Calculate total changed area from a binary change mask.

        Args:
            change_mask: Binary mask (H, W) where 1 = changed.

        Returns:
            AreaResult for changed pixels.
        """
        return self.calculate_mask_area(change_mask, target_value=1)
