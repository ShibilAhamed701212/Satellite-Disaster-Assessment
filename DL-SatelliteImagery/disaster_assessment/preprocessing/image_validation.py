"""
Image validation utilities for satellite disaster assessment.

Validates image format, dimensions, channels, and integrity
before feeding into the disaster analysis pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Minimum dimension for processing (must accommodate 256×256 patches)
MIN_DIMENSION = 64
# Recommended dimension
RECOMMENDED_DIMENSION = 256


@dataclass
class ValidationReport:
    """Structured report from image validation."""

    is_valid: bool = True
    image_path: str = ""
    height: int = 0
    width: int = 0
    channels: int = 0
    dtype: str = ""
    file_size_bytes: int = 0
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def add_warning(self, message: str):
        self.warnings.append(message)

    def add_error(self, message: str):
        self.errors.append(message)
        self.is_valid = False

    def summary(self) -> str:
        """Return a human-readable validation summary."""
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        lines = [
            f"Validation: {status}",
            f"  Path:     {self.image_path}",
            f"  Size:     {self.width}×{self.height} × {self.channels}ch",
            f"  Dtype:    {self.dtype}",
            f"  File:     {self.file_size_bytes:,} bytes",
        ]
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    ⚠ {w}")
        if self.errors:
            lines.append("  Errors:")
            for e in self.errors:
                lines.append(f"    ✗ {e}")
        return "\n".join(lines)


class ImageValidator:
    """
    Validates satellite images for disaster assessment processing.

    Checks:
        1. File existence and format
        2. Image dimensions (≥ MIN_DIMENSION)
        3. Channel count (must be 3 for RGB)
        4. Data type and range
        5. Non-empty content (not all-zero or all-constant)
    """

    def __init__(
        self,
        min_dimension: int = MIN_DIMENSION,
        required_channels: int = 3,
    ):
        self.min_dimension = min_dimension
        self.required_channels = required_channels

    def validate_file(self, file_path: str) -> ValidationReport:
        """
        Validate an image file on disk.

        Args:
            file_path: Absolute or relative path to the image file.

        Returns:
            ValidationReport with errors/warnings.
        """
        report = ValidationReport(image_path=file_path)

        # Check file exists
        if not os.path.isfile(file_path):
            report.add_error(f"File not found: {file_path}")
            return report

        # Check extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in SUPPORTED_EXTENSIONS:
            report.add_error(
                f"Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
            return report

        # Check file size
        report.file_size_bytes = os.path.getsize(file_path)
        if report.file_size_bytes == 0:
            report.add_error("File is empty (0 bytes)")
            return report

        # Try loading the image
        try:
            from PIL import Image

            img = Image.open(file_path)
            img.verify()  # Verify integrity without fully loading
            img = Image.open(file_path)  # Re-open after verify()
            img_array = np.array(img)
        except Exception as e:
            report.add_error(f"Failed to load image: {e}")
            return report

        return self.validate_array(img_array, report=report)

    def validate_array(
        self,
        image: np.ndarray,
        report: Optional[ValidationReport] = None,
    ) -> ValidationReport:
        """
        Validate an image as a numpy array.

        Args:
            image: Image as numpy array (H, W, C) or (H, W).
            report: Existing report to append to, or create new.

        Returns:
            ValidationReport with errors/warnings.
        """
        if report is None:
            report = ValidationReport(image_path="<array>")

        # Check it's a valid numpy array
        if not isinstance(image, np.ndarray):
            report.add_error(f"Expected numpy array, got {type(image).__name__}")
            return report

        if image.size == 0:
            report.add_error("Image array is empty (zero elements)")
            return report

        # Check dimensionality
        if image.ndim == 2:
            report.height, report.width = image.shape
            report.channels = 1
            report.add_warning(
                "Image is grayscale (2D). Expected 3-channel RGB. "
                "Will be converted to RGB by repeating across 3 channels."
            )
        elif image.ndim == 3:
            report.height, report.width, report.channels = image.shape
        else:
            report.add_error(
                f"Invalid dimensions: ndim={image.ndim}. Expected 2D or 3D array."
            )
            return report

        report.dtype = str(image.dtype)

        # Check channel count
        if report.channels != self.required_channels and image.ndim == 3:
            if report.channels == 4:
                report.add_warning(
                    "Image has 4 channels (RGBA). Alpha channel will be discarded."
                )
            elif report.channels == 1:
                report.add_warning(
                    "Image has 1 channel (grayscale). Will be converted to 3-channel RGB."
                )
            else:
                report.add_error(
                    f"Expected {self.required_channels} channels, got {report.channels}."
                )

        # Check minimum dimensions
        if report.height < self.min_dimension:
            report.add_error(
                f"Height {report.height}px is below minimum {self.min_dimension}px."
            )
        if report.width < self.min_dimension:
            report.add_error(
                f"Width {report.width}px is below minimum {self.min_dimension}px."
            )

        # Check recommended dimensions
        if report.height < RECOMMENDED_DIMENSION or report.width < RECOMMENDED_DIMENSION:
            report.add_warning(
                f"Image dimensions ({report.width}×{report.height}) are below "
                f"recommended {RECOMMENDED_DIMENSION}×{RECOMMENDED_DIMENSION}. "
                f"Results may be less accurate."
            )

        # Check for degenerate content
        if image.ndim >= 2:
            if np.all(image == image.flat[0]):
                report.add_warning(
                    "Image is constant (all pixels identical). "
                    "This may produce meaningless predictions."
                )
            elif np.std(image.astype(np.float32)) < 1.0:
                report.add_warning(
                    "Image has very low variance (near-constant). "
                    "This may produce unreliable predictions."
                )

        return report

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image to 3-channel uint8 RGB.

        Handles:
            - Grayscale → RGB conversion
            - RGBA → RGB conversion
            - Float [0,1] → uint8 [0,255] conversion

        Args:
            image: Input image array.

        Returns:
            Normalized (H, W, 3) uint8 array.
        """
        if image is None or image.size == 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)

        # Handle grayscale
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]  # Drop alpha

        # Handle float images
        if image.dtype in (np.float32, np.float64):
            if image.size > 0 and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        return image
