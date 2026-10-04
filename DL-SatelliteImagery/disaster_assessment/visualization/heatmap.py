"""
Disaster heatmap generation.

Creates combined disaster severity heatmaps by fusing multiple
disaster indicators (change, flood, damage) into a single
color-coded intensity map.
"""

from typing import Dict, Optional

import cv2
import numpy as np


class HeatmapGenerator:
    """
    Generates combined disaster severity heatmaps.

    Fuses multiple binary/categorical masks into a single intensity map,
    applies Gaussian smoothing, and renders with a color map.
    """

    def __init__(
        self,
        colormap: int = cv2.COLORMAP_JET,
        gaussian_kernel: int = 15,
        alpha: float = 0.6,
    ):
        """
        Args:
            colormap: OpenCV colormap for intensity rendering.
            gaussian_kernel: Kernel size for Gaussian smoothing (must be odd).
            alpha: Transparency for overlay blending.
        """
        self.colormap = colormap
        # Ensure kernel is odd
        self.gaussian_kernel = gaussian_kernel if gaussian_kernel % 2 == 1 else gaussian_kernel + 1
        self.alpha = alpha

    def combined_disaster_heatmap(
        self,
        change_mask: Optional[np.ndarray] = None,
        flood_mask: Optional[np.ndarray] = None,
        damage_map: Optional[np.ndarray] = None,
        vegetation_loss_mask: Optional[np.ndarray] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Create a combined disaster severity heatmap from multiple indicators.

        Args:
            change_mask: Binary change mask (H, W), 1 = changed.
            flood_mask: Binary flood mask (H, W), 1 = flooded.
            damage_map: Damage class mask (H, W), values 0/1/2.
            vegetation_loss_mask: Binary vegetation loss (H, W), 1 = lost.
            weights: Weight for each indicator {name: weight}.

        Returns:
            Heatmap image (H, W, 3) uint8 with color-coded intensity.
        """
        if weights is None:
            weights = {
                "change": 0.25,
                "flood": 0.35,
                "damage": 0.30,
                "vegetation": 0.10,
            }

        # Determine output shape from first available mask
        shape = None
        for m in [change_mask, flood_mask, damage_map, vegetation_loss_mask]:
            if m is not None:
                shape = m.shape[:2]
                break
        if shape is None:
            raise ValueError("At least one mask must be provided for heatmap generation.")

        # Build intensity map (0.0 - 1.0)
        intensity = np.zeros(shape, dtype=np.float32)

        if change_mask is not None:
            intensity += change_mask.astype(np.float32) * weights.get("change", 0.25)

        if flood_mask is not None:
            intensity += flood_mask.astype(np.float32) * weights.get("flood", 0.35)

        if damage_map is not None:
            # Normalize damage (0, 1, 2) to (0, 0.5, 1.0)
            damage_norm = damage_map.astype(np.float32) / 2.0
            intensity += damage_norm * weights.get("damage", 0.30)

        if vegetation_loss_mask is not None:
            intensity += vegetation_loss_mask.astype(np.float32) * weights.get("vegetation", 0.10)

        # Normalize to 0-1 range
        max_val = intensity.max()
        if max_val > 0:
            intensity = intensity / max_val

        # Apply Gaussian smoothing for visual appeal
        intensity = cv2.GaussianBlur(
            intensity, (self.gaussian_kernel, self.gaussian_kernel), 0
        )

        # Re-normalize after blur
        max_val = intensity.max()
        if max_val > 0:
            intensity = intensity / max_val

        # Convert to uint8 and apply colormap
        intensity_uint8 = (intensity * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(intensity_uint8, self.colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        return heatmap_rgb

    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
    ) -> np.ndarray:
        """
        Overlay a heatmap on an original image.

        Args:
            image: Original image (H, W, 3) uint8.
            heatmap: Heatmap image (H, W, 3) uint8.

        Returns:
            Blended result (H, W, 3) uint8.
        """
        # Resize heatmap to match image if needed
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(
                heatmap,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        return cv2.addWeighted(image, 1 - self.alpha, heatmap, self.alpha, 0)

    def change_intensity_heatmap(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
    ) -> np.ndarray:
        """
        Create a pixel-level change intensity heatmap by computing
        absolute difference between pre and post images.

        Args:
            pre_image: Pre-disaster image (H, W, 3) uint8.
            post_image: Post-disaster image (H, W, 3) uint8.

        Returns:
            Change intensity heatmap (H, W, 3) uint8.
        """
        # Compute per-channel absolute difference
        diff = np.abs(
            pre_image.astype(np.float32) - post_image.astype(np.float32)
        )
        # Average across channels
        intensity = diff.mean(axis=2)

        # Normalize to 0-1
        max_val = intensity.max()
        if max_val > 0:
            intensity = intensity / max_val

        # Smooth
        intensity = cv2.GaussianBlur(
            intensity, (self.gaussian_kernel, self.gaussian_kernel), 0
        )
        max_val = intensity.max()
        if max_val > 0:
            intensity = intensity / max_val

        # Apply colormap
        intensity_uint8 = (intensity * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(intensity_uint8, self.colormap)
        return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
