"""
Land cover change metrics between pre and post disaster images.

Uses the existing project's 6-class land cover labels:
    0 = Water
    1 = Land
    2 = Road
    3 = Building
    4 = Vegetation
    5 = Unlabeled
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


# Class labels matching the existing project
CLASS_LABELS = {
    0: "Water",
    1: "Land",
    2: "Road",
    3: "Building",
    4: "Vegetation",
    5: "Unlabeled",
}

NUM_CLASSES = len(CLASS_LABELS)


@dataclass
class LandChangeResult:
    """Structured land cover change analysis result."""

    # Per-class pixel counts
    pre_class_counts: Dict[str, int] = field(default_factory=dict)
    post_class_counts: Dict[str, int] = field(default_factory=dict)
    class_changes: Dict[str, Dict] = field(default_factory=dict)

    # Key derived metrics
    vegetation_loss_percentage: float = 0.0
    water_expansion_percentage: float = 0.0
    building_change_percentage: float = 0.0
    road_change_percentage: float = 0.0
    total_changed_percentage: float = 0.0

    # Transition matrix (class_from → class_to counts)
    transition_matrix: Optional[np.ndarray] = None

    total_pixels: int = 0

    def summary(self) -> str:
        lines = [
            "Land Cover Change Analysis:",
            f"  Total Pixels: {self.total_pixels:,}",
            "",
            "  Per-Class Changes:",
        ]
        for class_name, change_info in self.class_changes.items():
            direction = change_info.get("direction", "n/a")
            pct = change_info.get("percentage_change", 0.0)
            pre = change_info.get("pre_count", 0)
            post = change_info.get("post_count", 0)
            lines.append(
                f"    {class_name:<12}: {pre:>8,} → {post:>8,}  "
                f"({direction} {abs(pct):.1f}%)"
            )
        lines.extend([
            "",
            f"  Vegetation Loss:    {self.vegetation_loss_percentage:.2f}%",
            f"  Water Expansion:    {self.water_expansion_percentage:.2f}%",
            f"  Building Change:    {self.building_change_percentage:.2f}%",
            f"  Total Changed Area: {self.total_changed_percentage:.2f}%",
        ])
        return "\n".join(lines)


class LandChangeMetrics:
    """
    Computes land cover change metrics from pre/post segmentation masks.

    Uses the existing project's 6-class labels to analyze:
        - Per-class pixel gains and losses
        - Vegetation loss
        - Water expansion
        - Building change
        - Full class-to-class transition matrix
    """

    def __init__(self, class_labels: Dict[int, str] = None):
        """
        Args:
            class_labels: Override class label mapping.
                          Default uses the existing project's 6-class mapping.
        """
        self.class_labels = class_labels or CLASS_LABELS.copy()
        self.num_classes = len(self.class_labels)

    def compute(
        self,
        pre_segmentation: np.ndarray,
        post_segmentation: np.ndarray,
    ) -> LandChangeResult:
        """
        Compute comprehensive land cover change metrics.

        Args:
            pre_segmentation: Pre-disaster class mask (H, W) with integer labels.
            post_segmentation: Post-disaster class mask (H, W) with integer labels.

        Returns:
            LandChangeResult with per-class changes, key metrics, and transition matrix.
        """
        result = LandChangeResult()
        result.total_pixels = pre_segmentation.size

        # Per-class counts
        for class_id, class_name in self.class_labels.items():
            pre_count = int(np.sum(pre_segmentation == class_id))
            post_count = int(np.sum(post_segmentation == class_id))
            pixel_change = post_count - pre_count
            pct_change = (
                (pixel_change / pre_count * 100.0) if pre_count > 0 else 0.0
            )

            result.pre_class_counts[class_name] = pre_count
            result.post_class_counts[class_name] = post_count
            result.class_changes[class_name] = {
                "pre_count": pre_count,
                "post_count": post_count,
                "pixel_change": pixel_change,
                "percentage_change": pct_change,
                "direction": "gain" if pixel_change > 0 else ("loss" if pixel_change < 0 else "stable"),
            }

        # Key derived metrics
        result.vegetation_loss_percentage = self._calculate_loss(
            pre_segmentation, post_segmentation, class_id=4  # Vegetation
        )
        result.water_expansion_percentage = self._calculate_gain(
            pre_segmentation, post_segmentation, class_id=0  # Water
        )
        result.building_change_percentage = self._calculate_absolute_change(
            pre_segmentation, post_segmentation, class_id=3  # Building
        )
        result.road_change_percentage = self._calculate_absolute_change(
            pre_segmentation, post_segmentation, class_id=2  # Road
        )

        # Total changed pixels
        changed_pixels = int(np.sum(pre_segmentation != post_segmentation))
        result.total_changed_percentage = (
            (changed_pixels / result.total_pixels * 100.0)
            if result.total_pixels > 0
            else 0.0
        )

        # Transition matrix
        result.transition_matrix = self._compute_transition_matrix(
            pre_segmentation, post_segmentation
        )

        return result

    def _calculate_loss(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        class_id: int,
    ) -> float:
        """Calculate percentage of a class that was lost (pre had it, post doesn't)."""
        pre_pixels = np.sum(pre == class_id)
        if pre_pixels == 0:
            return 0.0
        lost_pixels = np.sum((pre == class_id) & (post != class_id))
        return float(lost_pixels / pre_pixels * 100.0)

    def _calculate_gain(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        class_id: int,
    ) -> float:
        """Calculate percentage gain of a class relative to its pre-disaster extent."""
        pre_pixels = np.sum(pre == class_id)
        post_pixels = np.sum(post == class_id)
        if pre_pixels == 0:
            return 100.0 if post_pixels > 0 else 0.0
        gained = post_pixels - pre_pixels
        if gained <= 0:
            return 0.0
        return float(gained / pre_pixels * 100.0)

    def _calculate_absolute_change(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        class_id: int,
    ) -> float:
        """Calculate absolute change percentage for a class."""
        pre_pixels = np.sum(pre == class_id)
        post_pixels = np.sum(post == class_id)
        if pre_pixels == 0 and post_pixels == 0:
            return 0.0
        denominator = max(pre_pixels, 1)
        return float(abs(post_pixels - pre_pixels) / denominator * 100.0)

    def _compute_transition_matrix(
        self,
        pre: np.ndarray,
        post: np.ndarray,
    ) -> np.ndarray:
        """
        Compute class-to-class transition matrix.

        Returns:
            (num_classes, num_classes) array where [i, j] is the count of
            pixels that changed from class i to class j.
        """
        matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                matrix[i, j] = int(np.sum((pre == i) & (post == j)))
        return matrix
