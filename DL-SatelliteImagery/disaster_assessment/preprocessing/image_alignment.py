"""
Image alignment and preprocessing for pre/post disaster image pairs.

Handles:
    - Dimension matching (resize to common size)
    - Optional ORB-based feature alignment for slight misalignment
    - Brightness / histogram normalization
    - Alignment quality assessment
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class AlignmentReport:
    """Structured report from image alignment."""

    pre_original_shape: Tuple[int, ...] = (0, 0, 0)
    post_original_shape: Tuple[int, ...] = (0, 0, 0)
    output_shape: Tuple[int, ...] = (0, 0, 0)
    was_resized: bool = False
    was_aligned: bool = False
    was_normalized: bool = False
    alignment_quality: float = 1.0  # 0.0 = poor, 1.0 = perfect / not needed
    num_feature_matches: int = 0
    warnings: list = field(default_factory=list)

    def add_warning(self, message: str):
        self.warnings.append(message)

    def summary(self) -> str:
        """Return a human-readable alignment summary."""
        lines = [
            "Image Alignment Report:",
            f"  PRE  original: {self.pre_original_shape[1]}×{self.pre_original_shape[0]}",
            f"  POST original: {self.post_original_shape[1]}×{self.post_original_shape[0]}",
            f"  Output:        {self.output_shape[1]}×{self.output_shape[0]}",
            f"  Resized:       {self.was_resized}",
            f"  Aligned:       {self.was_aligned}",
            f"  Normalized:    {self.was_normalized}",
            f"  Quality:       {self.alignment_quality:.2f}",
        ]
        if self.num_feature_matches > 0:
            lines.append(f"  Feature Matches: {self.num_feature_matches}")
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    ⚠ {w}")
        return "\n".join(lines)


class ImageAligner:
    """
    Aligns and preprocesses pre/post disaster satellite image pairs.

    Processing pipeline:
        1. Resize both images to a common target size
        2. Optionally align post image to pre image using ORB features
        3. Optionally normalize brightness/color
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        enable_feature_alignment: bool = False,
        enable_color_normalization: bool = True,
        min_feature_matches: int = 10,
    ):
        """
        Args:
            target_size: (height, width) to resize both images to.
            enable_feature_alignment: Whether to attempt ORB-based alignment.
            enable_color_normalization: Whether to normalize brightness/color.
            min_feature_matches: Minimum feature matches for acceptable alignment.
        """
        self.target_size = target_size
        self.enable_feature_alignment = enable_feature_alignment
        self.enable_color_normalization = enable_color_normalization
        self.min_feature_matches = min_feature_matches

    def align(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, AlignmentReport]:
        """
        Align and preprocess a pre/post image pair.

        Args:
            pre_image:  Pre-disaster image (H, W, 3) uint8.
            post_image: Post-disaster image (H, W, 3) uint8.

        Returns:
            Tuple of (aligned_pre, aligned_post, AlignmentReport).
        """
        report = AlignmentReport(
            pre_original_shape=pre_image.shape,
            post_original_shape=post_image.shape,
        )

        # Step 1: Resize to common target size
        pre_resized, post_resized, resized = self._resize_pair(
            pre_image, post_image, report
        )
        report.was_resized = resized

        # Step 2: Optional feature-based alignment
        if self.enable_feature_alignment:
            post_aligned, aligned = self._feature_align(
                pre_resized, post_resized, report
            )
            report.was_aligned = aligned
        else:
            post_aligned = post_resized

        # Step 3: Optional color normalization
        if self.enable_color_normalization:
            pre_normalized, post_normalized = self._normalize_colors(
                pre_resized, post_aligned
            )
            report.was_normalized = True
        else:
            pre_normalized = pre_resized
            post_normalized = post_aligned

        report.output_shape = pre_normalized.shape
        return pre_normalized, post_normalized, report

    def _resize_pair(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        report: AlignmentReport,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Resize both images to target_size."""
        target_h, target_w = self.target_size
        resized = False

        pre_h, pre_w = pre.shape[:2]
        post_h, post_w = post.shape[:2]

        if pre_h != target_h or pre_w != target_w:
            pre = cv2.resize(pre, (target_w, target_h), interpolation=cv2.INTER_AREA)
            resized = True
            report.add_warning(
                f"PRE image resized from {pre_w}×{pre_h} to {target_w}×{target_h}"
            )

        if post_h != target_h or post_w != target_w:
            post = cv2.resize(post, (target_w, target_h), interpolation=cv2.INTER_AREA)
            resized = True
            report.add_warning(
                f"POST image resized from {post_w}×{post_h} to {target_w}×{target_h}"
            )

        return pre, post, resized

    def _feature_align(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        report: AlignmentReport,
    ) -> Tuple[np.ndarray, bool]:
        """
        Attempt ORB-based feature alignment of post image to pre image.

        Returns:
            Tuple of (aligned_post, was_successfully_aligned).
        """
        try:
            # Convert to grayscale for feature detection
            pre_gray = cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY)
            post_gray = cv2.cvtColor(post, cv2.COLOR_RGB2GRAY)

            # ORB feature detector
            orb = cv2.ORB_create(nfeatures=500)
            kp1, desc1 = orb.detectAndCompute(pre_gray, None)
            kp2, desc2 = orb.detectAndCompute(post_gray, None)

            if desc1 is None or desc2 is None:
                report.add_warning(
                    "Feature alignment skipped: insufficient features detected."
                )
                report.alignment_quality = 0.5
                return post, False

            # Brute-force matching with Hamming distance
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(desc1, desc2)
            matches = sorted(matches, key=lambda m: m.distance)

            report.num_feature_matches = len(matches)

            if len(matches) < self.min_feature_matches:
                report.add_warning(
                    f"Only {len(matches)} feature matches found "
                    f"(minimum {self.min_feature_matches} required). "
                    f"Alignment quality may be poor."
                )
                report.alignment_quality = len(matches) / self.min_feature_matches
                return post, False

            # Compute homography
            src_pts = np.float32(
                [kp2[m.trainIdx].pt for m in matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp1[m.queryIdx].pt for m in matches]
            ).reshape(-1, 1, 2)

            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if homography is None:
                report.add_warning("Homography computation failed.")
                report.alignment_quality = 0.3
                return post, False

            # Warp post image to align with pre
            h, w = pre.shape[:2]
            aligned_post = cv2.warpPerspective(post, homography, (w, h))

            # Calculate alignment quality from inlier ratio
            if mask is not None:
                inlier_ratio = float(mask.sum()) / len(mask)
                report.alignment_quality = inlier_ratio
            else:
                report.alignment_quality = 0.5

            if report.alignment_quality < 0.3:
                report.add_warning(
                    f"Low alignment quality ({report.alignment_quality:.2f}). "
                    f"The images may not be of the same location. "
                    f"Results should be interpreted with caution."
                )

            return aligned_post, True

        except Exception as e:
            report.add_warning(f"Feature alignment failed: {e}")
            report.alignment_quality = 0.0
            return post, False

    @staticmethod
    def _normalize_colors(
        pre: np.ndarray,
        post: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize brightness/color by matching histograms.

        Uses CLAHE (Contrast Limited Adaptive Histogram Equalization)
        applied to the L channel in LAB color space.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        def normalize_single(img: np.ndarray) -> np.ndarray:
            # Convert RGB → LAB
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            # Apply CLAHE to L channel
            l_normalized = clahe.apply(l_channel)
            # Merge back
            lab_normalized = cv2.merge([l_normalized, a_channel, b_channel])
            return cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2RGB)

        return normalize_single(pre), normalize_single(post)

    @staticmethod
    def prepare_for_model(
        image: np.ndarray,
        target_size: Tuple[int, int] = (256, 256),
    ) -> np.ndarray:
        """
        Prepare a single image for model inference.

        Resizes to target_size and normalizes to [0, 1] float32.

        Args:
            image: (H, W, 3) uint8 image.
            target_size: (height, width) for the model.

        Returns:
            (H, W, 3) float32 array normalized to [0, 1].
        """
        h, w = target_size
        if image.shape[0] != h or image.shape[1] != w:
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

        return image.astype(np.float32) / 255.0
