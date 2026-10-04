"""
Pre/post comparison visualization utilities.

Creates side-by-side and grid layout comparison panels
for disaster assessment outputs.
"""

from typing import List, Tuple

import cv2
import numpy as np


class ComparisonRenderer:
    """
    Renders comparison visualizations for pre/post disaster images
    and their associated analysis outputs.
    """

    def __init__(
        self,
        padding: int = 5,
        title_height: int = 30,
        font_scale: float = 0.6,
        font_thickness: int = 1,
        bg_color: Tuple[int, int, int] = (30, 30, 30),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ):
        self.padding = padding
        self.title_height = title_height
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.bg_color = bg_color
        self.text_color = text_color

    def side_by_side(
        self,
        image_left: np.ndarray,
        image_right: np.ndarray,
        title_left: str = "Pre-Disaster",
        title_right: str = "Post-Disaster",
    ) -> np.ndarray:
        """
        Create a side-by-side comparison of two images with titles.

        Args:
            image_left: Left image (H, W, 3) uint8.
            image_right: Right image (H, W, 3) uint8.
            title_left: Title for the left image.
            title_right: Title for the right image.

        Returns:
            Combined image (H+title, W*2+padding, 3) uint8.
        """
        # Resize to same height
        h = max(image_left.shape[0], image_right.shape[0])
        left = self._resize_to_height(image_left, h)
        right = self._resize_to_height(image_right, h)

        # Add titles
        left_titled = self._add_title(left, title_left)
        right_titled = self._add_title(right, title_right)

        # Create separator
        sep = np.full(
            (left_titled.shape[0], self.padding, 3),
            self.bg_color,
            dtype=np.uint8,
        )

        return np.hstack([left_titled, sep, right_titled])

    def grid(
        self,
        images: List[np.ndarray],
        titles: List[str],
        columns: int = 3,
        cell_size: Tuple[int, int] = (256, 256),
    ) -> np.ndarray:
        """
        Create a grid layout of multiple images with titles.

        Args:
            images: List of images (H, W, 3) uint8.
            titles: List of titles for each image.
            columns: Number of columns in the grid.
            cell_size: (width, height) for each cell.

        Returns:
            Grid image uint8.
        """
        if len(images) == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        cell_w, cell_h = cell_size
        rows = (len(images) + columns - 1) // columns
        total_h = rows * (cell_h + self.title_height + self.padding) + self.padding
        total_w = columns * (cell_w + self.padding) + self.padding

        canvas = np.full((total_h, total_w, 3), self.bg_color, dtype=np.uint8)

        for idx, (img, title) in enumerate(zip(images, titles)):
            row = idx // columns
            col = idx % columns

            # Resize image to cell size
            resized = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            titled = self._add_title(resized, title)

            y_start = row * (cell_h + self.title_height + self.padding) + self.padding
            x_start = col * (cell_w + self.padding) + self.padding

            # Ensure we don't overflow canvas
            actual_h = min(titled.shape[0], canvas.shape[0] - y_start)
            actual_w = min(titled.shape[1], canvas.shape[1] - x_start)

            if actual_h > 0 and actual_w > 0:
                canvas[y_start:y_start + actual_h, x_start:x_start + actual_w] = (
                    titled[:actual_h, :actual_w]
                )

        return canvas

    def difference_highlight(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        threshold: int = 30,
    ) -> np.ndarray:
        """
        Create a difference image highlighting changed pixels.

        Args:
            pre_image: Pre-disaster image (H, W, 3) uint8.
            post_image: Post-disaster image (H, W, 3) uint8.
            threshold: Minimum per-channel difference to highlight.

        Returns:
            Difference visualization (H, W, 3) uint8.
        """
        # Ensure same size
        if pre_image.shape != post_image.shape:
            h = min(pre_image.shape[0], post_image.shape[0])
            w = min(pre_image.shape[1], post_image.shape[1])
            pre_image = cv2.resize(pre_image, (w, h))
            post_image = cv2.resize(post_image, (w, h))

        diff = np.abs(
            pre_image.astype(np.int16) - post_image.astype(np.int16)
        ).astype(np.uint8)

        # Create mask of significant changes
        change_mask = diff.max(axis=2) > threshold

        # Create output: grayscale base with red highlights
        gray = cv2.cvtColor(post_image, cv2.COLOR_RGB2GRAY)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Highlight changed areas in red
        result[change_mask] = [255, 0, 0]

        return result

    def _add_title(self, image: np.ndarray, title: str) -> np.ndarray:
        """Add a title bar above an image."""
        h, w = image.shape[:2]
        title_bar = np.full(
            (self.title_height, w, 3), self.bg_color, dtype=np.uint8
        )

        # Center the text
        text_size = cv2.getTextSize(
            title,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            self.font_thickness,
        )[0]
        text_x = max(5, (w - text_size[0]) // 2)
        text_y = self.title_height - 8

        cv2.putText(
            title_bar,
            title,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            self.text_color,
            self.font_thickness,
            cv2.LINE_AA,
        )

        return np.vstack([title_bar, image])

    @staticmethod
    def _resize_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
        """Resize image to target height while preserving aspect ratio."""
        h, w = image.shape[:2]
        if h == target_h:
            return image
        scale = target_h / h
        new_w = int(w * scale)
        return cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_AREA)
