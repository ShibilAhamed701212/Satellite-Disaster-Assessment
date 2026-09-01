"""
Tests for preprocessing modules (validation and alignment).
"""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disaster_assessment.preprocessing.image_validation import (
    ImageValidator,
    ValidationReport,
)
from disaster_assessment.preprocessing.image_alignment import (
    ImageAligner,
    AlignmentReport,
)


class TestImageValidator:
    def setup_method(self):
        self.validator = ImageValidator()

    def test_valid_rgb_image(self):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        report = self.validator.validate_array(img)
        assert report.is_valid
        assert report.height == 256
        assert report.width == 256
        assert report.channels == 3
        assert len(report.errors) == 0

    def test_empty_array(self):
        img = np.array([])
        report = self.validator.validate_array(img)
        assert not report.is_valid
        assert len(report.errors) > 0

    def test_grayscale_warning(self):
        img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        report = self.validator.validate_array(img)
        assert report.is_valid  # Grayscale is valid but warns
        assert len(report.warnings) > 0
        assert "grayscale" in report.warnings[0].lower()

    def test_rgba_warning(self):
        img = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
        report = self.validator.validate_array(img)
        assert report.is_valid
        assert any("alpha" in w.lower() for w in report.warnings)

    def test_too_small(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        report = self.validator.validate_array(img)
        assert not report.is_valid
        assert any("below minimum" in e.lower() for e in report.errors)

    def test_constant_image_warning(self):
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        report = self.validator.validate_array(img)
        assert report.is_valid
        assert any("constant" in w.lower() for w in report.warnings)

    def test_not_numpy(self):
        report = self.validator.validate_array("not an array")
        assert not report.is_valid

    def test_summary(self):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        report = self.validator.validate_array(img)
        summary = report.summary()
        assert "VALID" in summary

    def test_normalize_grayscale(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        normalized = ImageValidator.normalize_image(img)
        assert normalized.shape == (100, 100, 3)
        assert normalized.dtype == np.uint8

    def test_normalize_rgba(self):
        img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        normalized = ImageValidator.normalize_image(img)
        assert normalized.shape == (100, 100, 3)

    def test_normalize_float(self):
        img = np.random.rand(100, 100, 3).astype(np.float32)
        normalized = ImageValidator.normalize_image(img)
        assert normalized.dtype == np.uint8
        assert normalized.max() <= 255


class TestImageAligner:
    def setup_method(self):
        self.aligner = ImageAligner(target_size=(256, 256))

    def test_same_size_images(self):
        pre = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        pre_out, post_out, report = self.aligner.align(pre, post)

        assert pre_out.shape == (256, 256, 3)
        assert post_out.shape == (256, 256, 3)
        assert not report.was_resized

    def test_different_size_images(self):
        pre = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        pre_out, post_out, report = self.aligner.align(pre, post)

        assert pre_out.shape[:2] == (256, 256)
        assert post_out.shape[:2] == (256, 256)
        assert report.was_resized

    def test_color_normalization(self):
        aligner = ImageAligner(
            target_size=(256, 256),
            enable_color_normalization=True,
        )
        pre = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        pre_out, post_out, report = aligner.align(pre, post)
        assert report.was_normalized

    def test_no_normalization(self):
        aligner = ImageAligner(
            target_size=(256, 256),
            enable_color_normalization=False,
        )
        pre = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        _, _, report = aligner.align(pre, post)
        assert not report.was_normalized

    def test_report_summary(self):
        pre = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        _, _, report = self.aligner.align(pre, post)
        summary = report.summary()
        assert "Alignment Report" in summary

    def test_prepare_for_model(self):
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        prepared = ImageAligner.prepare_for_model(img, target_size=(256, 256))
        assert prepared.shape == (256, 256, 3)
        assert prepared.dtype == np.float32
        assert prepared.min() >= 0.0
        assert prepared.max() <= 1.0
