"""
Building damage estimation metrics.

Compares pre/post building segmentation masks to estimate damage levels.

IMPORTANT DISCLAIMER:
    All outputs are labeled as "Estimated Damage" unless a properly trained
    damage classification model is provided. Simple pixel-level comparison
    is NOT a scientifically validated damage classifier.

Damage Classes:
    0 = Undamaged
    1 = Possible Damage (partial overlap loss)
    2 = Severe Damage (near-complete loss)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Damage class constants
UNDAMAGED = 0
POSSIBLE_DAMAGE = 1
SEVERE_DAMAGE = 2

DAMAGE_LABELS = {
    UNDAMAGED: "Undamaged",
    POSSIBLE_DAMAGE: "Possible Damage",
    SEVERE_DAMAGE: "Severe Damage",
}


@dataclass
class BuildingDamageResult:
    """Structured building damage estimation result."""

    total_building_pixels_pre: int
    total_building_pixels_post: int
    undamaged_pixels: int
    possible_damage_pixels: int
    severe_damage_pixels: int
    undamaged_percentage: float
    possible_damage_percentage: float
    severe_damage_percentage: float
    overall_damage_score: float  # 0-100
    damage_map: Optional[np.ndarray] = None
    disclaimer: str = (
        "ESTIMATED DAMAGE: Based on pixel-level building mask comparison. "
        "This is NOT a validated damage classifier. Results should be "
        "verified by domain experts with ground-truth data."
    )

    def summary(self) -> str:
        lines = [
            "Building Damage Assessment (ESTIMATED):",
            f"  Pre-disaster building pixels:  {self.total_building_pixels_pre:,}",
            f"  Post-disaster building pixels: {self.total_building_pixels_post:,}",
            f"  Undamaged:       {self.undamaged_percentage:.1f}% ({self.undamaged_pixels:,} px)",
            f"  Possible Damage: {self.possible_damage_percentage:.1f}% ({self.possible_damage_pixels:,} px)",
            f"  Severe Damage:   {self.severe_damage_percentage:.1f}% ({self.severe_damage_pixels:,} px)",
            f"  Overall Score:   {self.overall_damage_score:.1f}/100",
            f"",
            f"  ⚠ {self.disclaimer}",
        ]
        return "\n".join(lines)


class DamageMetrics:
    """
    Estimates building damage from pre/post building segmentation masks.

    Pipeline:
        1. Extract building pixels from pre and post land-cover masks
        2. Compare overlap at the pixel level
        3. Classify into Undamaged / Possible Damage / Severe Damage
        4. Compute damage percentages and overall score

    The building class value should match the existing project's mapping:
        Building = class 3

    The design allows future replacement of the pixel-comparison logic
    with a trained damage classification model.
    """

    def __init__(
        self,
        building_class: int = 3,
        severe_threshold: float = 0.7,
        possible_threshold: float = 0.3,
        patch_size: int = 16,
    ):
        """
        Args:
            building_class: Class label for buildings in segmentation masks.
            severe_threshold: Minimum fraction of building loss in a patch for "Severe Damage".
            possible_threshold: Minimum fraction of building loss in a patch for "Possible Damage".
            patch_size: Size of patches for local damage assessment.
        """
        self.building_class = building_class
        self.severe_threshold = severe_threshold
        self.possible_threshold = possible_threshold
        self.patch_size = patch_size

    def estimate_damage(
        self,
        pre_segmentation: np.ndarray,
        post_segmentation: np.ndarray,
    ) -> BuildingDamageResult:
        """
        Estimate building damage from pre/post land-cover segmentation masks.

        Args:
            pre_segmentation: Pre-disaster class mask (H, W) with integer labels.
            post_segmentation: Post-disaster class mask (H, W) with integer labels.

        Returns:
            BuildingDamageResult with damage estimates.
        """
        # Extract building masks
        pre_buildings = (pre_segmentation == self.building_class).astype(np.uint8)
        post_buildings = (post_segmentation == self.building_class).astype(np.uint8)

        total_pre = int(pre_buildings.sum())
        total_post = int(post_buildings.sum())

        # If no buildings in pre image, can't assess damage
        if total_pre == 0:
            return BuildingDamageResult(
                total_building_pixels_pre=0,
                total_building_pixels_post=total_post,
                undamaged_pixels=0,
                possible_damage_pixels=0,
                severe_damage_pixels=0,
                undamaged_percentage=0.0,
                possible_damage_percentage=0.0,
                severe_damage_percentage=0.0,
                overall_damage_score=0.0,
                damage_map=np.zeros_like(pre_segmentation, dtype=np.uint8),
            )

        # Compute damage map using local patch comparison
        damage_map = self._compute_damage_map(pre_buildings, post_buildings)

        # Count damage categories (only in pre-building regions)
        building_region_damage = damage_map[pre_buildings == 1]
        undamaged = int(np.sum(building_region_damage == UNDAMAGED))
        possible = int(np.sum(building_region_damage == POSSIBLE_DAMAGE))
        severe = int(np.sum(building_region_damage == SEVERE_DAMAGE))

        total_assessed = undamaged + possible + severe
        if total_assessed == 0:
            total_assessed = 1  # Prevent division by zero

        undamaged_pct = (undamaged / total_assessed) * 100.0
        possible_pct = (possible / total_assessed) * 100.0
        severe_pct = (severe / total_assessed) * 100.0

        # Overall damage score: weighted combination
        # Severe contributes more to the score than possible damage
        damage_score = min(
            100.0,
            (possible_pct * 0.3) + (severe_pct * 1.0),
        )

        return BuildingDamageResult(
            total_building_pixels_pre=total_pre,
            total_building_pixels_post=total_post,
            undamaged_pixels=undamaged,
            possible_damage_pixels=possible,
            severe_damage_pixels=severe,
            undamaged_percentage=undamaged_pct,
            possible_damage_percentage=possible_pct,
            severe_damage_percentage=severe_pct,
            overall_damage_score=damage_score,
            damage_map=damage_map,
        )

    def _compute_damage_map(
        self,
        pre_buildings: np.ndarray,
        post_buildings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-pixel damage classification using local patch analysis.

        For each pixel that was a building in pre-image:
            - If building still present in post → Undamaged
            - If local building loss > severe_threshold → Severe Damage
            - If local building loss > possible_threshold → Possible Damage
            - Otherwise → Undamaged

        Returns:
            Damage map (H, W) with values 0/1/2.
        """
        h, w = pre_buildings.shape
        damage_map = np.zeros((h, w), dtype=np.uint8)
        ps = self.patch_size

        for y in range(0, h, ps):
            for x in range(0, w, ps):
                y_end = min(y + ps, h)
                x_end = min(x + ps, w)

                pre_patch = pre_buildings[y:y_end, x:x_end]
                post_patch = post_buildings[y:y_end, x:x_end]

                pre_count = pre_patch.sum()
                if pre_count == 0:
                    continue

                post_count = post_patch.sum()
                loss_ratio = 1.0 - (post_count / pre_count)

                # Assign damage level to all building pixels in this patch
                mask = pre_patch == 1
                if loss_ratio >= self.severe_threshold:
                    damage_map[y:y_end, x:x_end][mask] = SEVERE_DAMAGE
                elif loss_ratio >= self.possible_threshold:
                    damage_map[y:y_end, x:x_end][mask] = POSSIBLE_DAMAGE
                else:
                    damage_map[y:y_end, x:x_end][mask] = UNDAMAGED

        return damage_map
