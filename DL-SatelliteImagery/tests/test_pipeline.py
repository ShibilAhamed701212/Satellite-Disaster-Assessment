"""
Tests for the full DisasterAnalyzer pipeline.

Uses synthetic images — no real dataset required.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disaster_assessment.pipeline.disaster_analyzer import DisasterAnalyzer, DisasterReport
from disaster_assessment.visualization.overlays import OverlayRenderer, LANDCOVER_COLORS, LANDCOVER_LABELS
from disaster_assessment.visualization.heatmap import HeatmapGenerator
from disaster_assessment.visualization.comparison import ComparisonRenderer


class TestDisasterAnalyzer:
    def setup_method(self):
        self.analyzer = DisasterAnalyzer(
            target_size=(256, 256),
        )

    def test_full_pipeline_synthetic(self):
        """Test the full pipeline with synthetic RGB images."""
        pre = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        report = self.analyzer.analyze(pre, post)

        assert isinstance(report, DisasterReport)
        assert 0 <= report.change_percentage <= 100
        assert 0 <= report.flood_percentage <= 100
        assert report.building_damage is not None
        assert report.land_change is not None
        assert report.severity is not None
        assert report.severity.severity_level in ["LOW", "MODERATE", "HIGH", "CRITICAL"]

        # Check maps were generated
        assert "pre_image" in report.maps
        assert "post_image" in report.maps
        assert "change_map" in report.maps
        assert "flood_map" in report.maps
        assert "disaster_heatmap" in report.maps

    def test_to_dict(self):
        pre = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        report = self.analyzer.analyze(pre, post)

        d = report.to_dict()
        assert "change_percentage" in d
        assert "flood_percentage" in d
        assert "severity_score" in d
        assert "severity_level" in d

    def test_summary(self):
        pre = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        report = self.analyzer.analyze(pre, post)

        summary = report.summary()
        assert "DISASTER ASSESSMENT REPORT" in summary
        assert "Severity" in summary

    def test_invalid_pre_image(self):
        pre = np.array([])  # Invalid
        post = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        report = self.analyzer.analyze(pre, post)
        assert len(report.warnings) > 0

    def test_single_image_analysis(self):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = self.analyzer.analyze_single(img)

        assert "original" in result
        assert "segmentation" in result
        assert "overlay" in result
        assert "legend" in result
        assert "class_percentages" in result
        assert "class_counts" in result
        assert "summary" in result
        assert result["original"].shape == (256, 256, 3)
        assert len(result["class_percentages"]) == 6

    def test_different_sized_inputs(self):
        """Pipeline should handle different sized pre/post images."""
        pre = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        report = self.analyzer.analyze(pre, post)
        assert isinstance(report, DisasterReport)
        assert report.change_percentage >= 0

    def test_measurement_basis_pixel(self):
        analyzer = DisasterAnalyzer(gsd_meters=None)
        pre = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        report = analyzer.analyze(pre, post)
        assert "pixel-based" in report.measurement_basis

    def test_measurement_basis_gsd(self):
        analyzer = DisasterAnalyzer(gsd_meters=0.5)
        pre = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        report = analyzer.analyze(pre, post)
        assert "geospatially" in report.measurement_basis


class TestOverlayRenderer:
    def setup_method(self):
        self.renderer = OverlayRenderer()

    def test_landcover_overlay(self):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        seg = np.random.randint(0, 6, (256, 256), dtype=np.uint8)
        overlay = self.renderer.landcover_overlay(img, seg)
        assert overlay.shape == (256, 256, 3)

    def test_change_overlay(self):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
        overlay = self.renderer.change_overlay(img, mask)
        assert overlay.shape == (256, 256, 3)

    def test_flood_overlay(self):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
        overlay = self.renderer.flood_overlay(img, mask)
        assert overlay.shape == (256, 256, 3)

    def test_create_legend(self):
        legend = OverlayRenderer.create_legend(LANDCOVER_LABELS, LANDCOVER_COLORS)
        assert legend.ndim == 3
        assert legend.shape[2] == 3


class TestHeatmapGenerator:
    def setup_method(self):
        self.gen = HeatmapGenerator()

    def test_combined_heatmap(self):
        change = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
        flood = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
        heatmap = self.gen.combined_disaster_heatmap(
            change_mask=change, flood_mask=flood
        )
        assert heatmap.shape == (256, 256, 3)

    def test_overlay_heatmap(self):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        heatmap = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = self.gen.overlay_heatmap(img, heatmap)
        assert result.shape == (256, 256, 3)

    def test_no_masks_raises(self):
        with pytest.raises(ValueError):
            self.gen.combined_disaster_heatmap()


class TestComparisonRenderer:
    def setup_method(self):
        self.renderer = ComparisonRenderer()

    def test_side_by_side(self):
        left = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = self.renderer.side_by_side(left, right)
        assert result.ndim == 3
        assert result.shape[2] == 3
        # Width should be > original (two images + padding)
        assert result.shape[1] > 256

    def test_grid(self):
        images = [
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            for _ in range(6)
        ]
        titles = [f"Image {i}" for i in range(6)]
        result = self.renderer.grid(images, titles, columns=3)
        assert result.ndim == 3
        assert result.shape[2] == 3
