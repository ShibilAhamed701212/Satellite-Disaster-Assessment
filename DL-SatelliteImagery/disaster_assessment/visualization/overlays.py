"""
Overlay rendering for disaster assessment visualizations.

Generates color-coded overlays on satellite images:
    - Land-cover segmentation overlays (reuses existing hex colors)
    - Binary change overlays
    - Flood overlays
    - Building damage overlays
    - Vegetation loss overlays
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# Existing project's class color mapping (RGB)
# Matches: Satellite_Imagery_Segmentation.ipynb
LANDCOVER_COLORS = {
    0: (226, 169, 41),   # Water    - #E2A929
    1: (132, 41, 246),   # Land     - #8429F6
    2: (110, 193, 228),  # Road     - #6EC1E4
    3: (60, 16, 152),    # Building - #3C1098
    4: (254, 221, 58),   # Vegetation - #FEDD3A
    5: (155, 155, 155),  # Unlabeled - #9B9B9B
}

LANDCOVER_LABELS = {
    0: "Water",
    1: "Land",
    2: "Road",
    3: "Building",
    4: "Vegetation",
    5: "Unlabeled",
}

# Damage overlay colors (RGB) - Phase 1 (Estimated)
DAMAGE_COLORS = {
    0: (0, 200, 0),     # Undamaged - Green
    1: (255, 200, 0),   # Possible Damage - Yellow
    2: (255, 0, 0),     # Severe Damage - Red
}

# 4-Class Trained Damage Colors & Labels - Phase 2
TRAINED_DAMAGE_COLORS = {
    0: (0, 200, 0),     # Undamaged - Green
    1: (255, 215, 0),   # Minor Damage - Gold/Yellow
    2: (255, 140, 0),   # Major Damage - Dark Orange
    3: (255, 0, 0),     # Destroyed - Red
}

TRAINED_DAMAGE_LABELS = {
    0: "Undamaged",
    1: "Minor Damage",
    2: "Major Damage",
    3: "Destroyed",
}


class OverlayRenderer:
    """
    Renders color overlays on satellite images for various disaster masks.

    All overlays preserve the original image dimensions and use
    clear color coding with optional transparency.
    """

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Transparency for overlay blending (0=transparent, 1=opaque).
        """
        self.alpha = alpha

    def landcover_overlay(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        colors: Dict[int, Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Render land-cover segmentation overlay on the original image.

        Args:
            image: Original image (H, W, 3) uint8.
            segmentation: Class mask (H, W) with integer labels.
            colors: Override color mapping {class_id: (R, G, B)}.

        Returns:
            Blended overlay image (H, W, 3) uint8.
        """
        if colors is None:
            colors = LANDCOVER_COLORS

        overlay = self._colorize_mask(segmentation, colors)
        return self._blend(image, overlay)

    def landcover_colorized(
        self,
        segmentation: np.ndarray,
        colors: Dict[int, Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Create a standalone colorized land-cover map (no image blending).

        Args:
            segmentation: Class mask (H, W) with integer labels.
            colors: Override color mapping.

        Returns:
            Colorized mask (H, W, 3) uint8.
        """
        if colors is None:
            colors = LANDCOVER_COLORS
        return self._colorize_mask(segmentation, colors)

    def change_overlay(
        self,
        image: np.ndarray,
        change_mask: np.ndarray,
        color: Tuple[int, int, int] = (255, 0, 0),
    ) -> np.ndarray:
        """
        Overlay binary change mask on image (changed regions highlighted).

        Args:
            image: Original image (H, W, 3) uint8.
            change_mask: Binary mask (H, W) where 1 = changed.
            color: RGB color for changed regions.

        Returns:
            Blended overlay (H, W, 3) uint8.
        """
        overlay = np.zeros_like(image)
        overlay[change_mask == 1] = color
        return self._blend(image, overlay, mask=(change_mask == 1))

    def flood_overlay(
        self,
        image: np.ndarray,
        flood_mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 100, 255),
    ) -> np.ndarray:
        """
        Overlay flood mask on image (flooded regions in blue).

        Args:
            image: Original image (H, W, 3) uint8.
            flood_mask: Binary mask (H, W) where 1 = flooded.
            color: RGB color for flooded regions.

        Returns:
            Blended overlay (H, W, 3) uint8.
        """
        overlay = np.zeros_like(image)
        overlay[flood_mask == 1] = color
        return self._blend(image, overlay, mask=(flood_mask == 1))

    def damage_overlay(
        self,
        image: np.ndarray,
        damage_map: np.ndarray,
        colors: Dict[int, Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Overlay building damage map on image (Phase 1 3-class estimated).
        """
        if colors is None:
            colors = DAMAGE_COLORS

        overlay = self._colorize_mask(damage_map, colors)
        active_mask = damage_map > 0
        return self._blend(image, overlay, mask=active_mask)

    def trained_damage_overlay(
        self,
        image: np.ndarray,
        damage_map: np.ndarray,
        colors: Dict[int, Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Overlay 4-class trained building damage map on image (Phase 2).
        0=Undamaged (Green), 1=Minor (Yellow), 2=Major (Orange), 3=Destroyed (Red).
        """
        if colors is None:
            colors = TRAINED_DAMAGE_COLORS

        overlay = self._colorize_mask(damage_map, colors)
        return self._blend(image, overlay)

    def vegetation_loss_overlay(
        self,
        image: np.ndarray,
        pre_segmentation: np.ndarray,
        post_segmentation: np.ndarray,
        vegetation_class: int = 4,
        loss_color: Tuple[int, int, int] = (255, 100, 0),
    ) -> np.ndarray:
        """
        Highlight vegetation loss between pre and post images.

        Args:
            image: Post-disaster image (H, W, 3) uint8.
            pre_segmentation: Pre-disaster class mask.
            post_segmentation: Post-disaster class mask.
            vegetation_class: Class label for vegetation.
            loss_color: RGB color for vegetation loss regions.

        Returns:
            Blended overlay (H, W, 3) uint8.
        """
        # Vegetation lost: was vegetation in pre, is NOT vegetation in post
        veg_loss_mask = (pre_segmentation == vegetation_class) & (
            post_segmentation != vegetation_class
        )
        overlay = np.zeros_like(image)
        overlay[veg_loss_mask] = loss_color
        return self._blend(image, overlay, mask=veg_loss_mask)

    def _blend(
        self,
        image: np.ndarray,
        overlay: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Blend overlay onto image with transparency."""
        result = image.copy()
        if mask is not None:
            # Only blend where mask is True
            mask_3d = np.stack([mask] * 3, axis=-1) if mask.ndim == 2 else mask
            result = np.where(
                mask_3d,
                (image * (1 - self.alpha) + overlay * self.alpha).astype(np.uint8),
                result,
            )
        else:
            result = cv2.addWeighted(image, 1 - self.alpha, overlay, self.alpha, 0)
        return result

    @staticmethod
    def _colorize_mask(
        mask: np.ndarray,
        colors: Dict[int, Tuple[int, int, int]],
    ) -> np.ndarray:
        """Convert integer class mask to RGB color image."""
        h, w = mask.shape[:2]
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in colors.items():
            colored[mask == class_id] = color
        return colored

    @staticmethod
    def create_legend(
        labels: Dict[int, str],
        colors: Dict[int, Tuple[int, int, int]],
        cell_size: int = 30,
        font_scale: float = 0.6,
    ) -> np.ndarray:
        """
        Create a color legend image.

        Returns:
            Legend image (H, W, 3) uint8.
        """
        num_items = len(labels)
        legend_width = 250
        legend_height = num_items * (cell_size + 5) + 10

        legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255

        y_offset = 10
        for class_id in sorted(labels.keys()):
            label = labels[class_id]
            color = colors.get(class_id, (128, 128, 128))

            # Draw color swatch
            cv2.rectangle(
                legend,
                (10, y_offset),
                (10 + cell_size, y_offset + cell_size),
                color[::-1],  # BGR for OpenCV
                -1,
            )
            cv2.rectangle(
                legend,
                (10, y_offset),
                (10 + cell_size, y_offset + cell_size),
                (0, 0, 0),
                1,
            )

            # Draw label text
            cv2.putText(
                legend,
                label,
                (10 + cell_size + 10, y_offset + cell_size - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            y_offset += cell_size + 5

        # Convert BGR → RGB
        legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        return legend
