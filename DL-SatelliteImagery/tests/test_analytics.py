"""
Tests for analytics modules (area, damage, land change, severity).
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disaster_assessment.analytics.area_calculator import AreaCalculator, AreaResult
from disaster_assessment.analytics.damage_metrics import DamageMetrics, UNDAMAGED, POSSIBLE_DAMAGE, SEVERE_DAMAGE
from disaster_assessment.analytics.land_change_metrics import LandChangeMetrics
from disaster_assessment.analytics.severity_engine import SeverityEngine


class TestAreaCalculator:
    def test_pixel_based(self):
        calc = AreaCalculator(gsd_meters=None)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:60] = 1  # 20×30 = 600 pixels

        result = calc.calculate_mask_area(mask, target_value=1)
        assert result.pixel_count == 600
        assert result.total_pixels == 10000
        assert abs(result.percentage - 6.0) < 0.01
        assert result.area_m2 is None
        assert "pixel-based" in result.measurement_basis

    def test_gsd_based(self):
        calc = AreaCalculator(gsd_meters=0.5)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:50, :50] = 1  # 2500 pixels

        result = calc.calculate_mask_area(mask, target_value=1)
        assert result.pixel_count == 2500
        assert result.area_m2 == 2500 * 0.25  # 0.5^2 = 0.25 m²/pixel
        assert result.area_km2 is not None
        assert "geospatially" in result.measurement_basis

    def test_change_area(self):
        calc = AreaCalculator()
        pre = np.zeros((100, 100), dtype=np.uint8)
        post = np.zeros((100, 100), dtype=np.uint8)
        pre[:30, :30] = 1  # 900 pixels of class 1
        post[:50, :50] = 1  # 2500 pixels of class 1

        result = calc.calculate_change_area(pre, post, class_value=1)
        assert result["pixel_change"] == 1600
        assert result["direction"] == "gain"

    def test_empty_mask(self):
        calc = AreaCalculator()
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = calc.calculate_mask_area(mask, target_value=1)
        assert result.pixel_count == 0
        assert result.percentage == 0.0

    def test_full_mask(self):
        calc = AreaCalculator()
        mask = np.ones((100, 100), dtype=np.uint8)
        result = calc.calculate_mask_area(mask, target_value=1)
        assert result.pixel_count == 10000
        assert result.percentage == 100.0


class TestDamageMetrics:
    def setup_method(self):
        self.metrics = DamageMetrics(building_class=3)

    def test_no_damage(self):
        # Same building footprint pre and post
        pre = np.zeros((256, 256), dtype=np.uint8)
        post = np.zeros((256, 256), dtype=np.uint8)
        pre[100:150, 100:150] = 3  # Building
        post[100:150, 100:150] = 3  # Building (same)

        result = self.metrics.estimate_damage(pre, post)
        assert result.severe_damage_percentage == 0.0
        assert result.undamaged_percentage > 0

    def test_severe_damage(self):
        # Buildings completely gone in post
        pre = np.zeros((256, 256), dtype=np.uint8)
        post = np.zeros((256, 256), dtype=np.uint8)
        pre[100:150, 100:150] = 3  # Building exists
        # Post has no buildings → severe damage

        result = self.metrics.estimate_damage(pre, post)
        assert result.severe_damage_percentage > 0
        assert result.total_building_pixels_pre > 0
        assert result.total_building_pixels_post == 0

    def test_no_buildings(self):
        pre = np.zeros((256, 256), dtype=np.uint8)
        post = np.zeros((256, 256), dtype=np.uint8)
        result = self.metrics.estimate_damage(pre, post)
        assert result.total_building_pixels_pre == 0
        assert result.overall_damage_score == 0.0

    def test_disclaimer_present(self):
        pre = np.zeros((256, 256), dtype=np.uint8)
        post = np.zeros((256, 256), dtype=np.uint8)
        result = self.metrics.estimate_damage(pre, post)
        assert "ESTIMATED" in result.disclaimer


class TestLandChangeMetrics:
    def setup_method(self):
        self.metrics = LandChangeMetrics()

    def test_no_change(self):
        seg = np.zeros((256, 256), dtype=np.uint8)
        seg[:128, :] = 0  # Water
        seg[128:, :] = 4  # Vegetation
        result = self.metrics.compute(seg, seg)  # Same pre and post
        assert result.total_changed_percentage == 0.0
        assert result.vegetation_loss_percentage == 0.0

    def test_vegetation_loss(self):
        pre = np.full((256, 256), 4, dtype=np.uint8)  # All vegetation
        post = np.full((256, 256), 1, dtype=np.uint8)  # All land
        result = self.metrics.compute(pre, post)
        assert result.vegetation_loss_percentage == 100.0

    def test_water_expansion(self):
        pre = np.full((256, 256), 1, dtype=np.uint8)  # All land
        post = np.full((256, 256), 0, dtype=np.uint8)  # All water
        result = self.metrics.compute(pre, post)
        assert result.water_expansion_percentage > 0

    def test_transition_matrix(self):
        pre = np.zeros((100, 100), dtype=np.uint8)
        post = np.ones((100, 100), dtype=np.uint8)  # All changed from 0→1
        result = self.metrics.compute(pre, post)

        assert result.transition_matrix is not None
        assert result.transition_matrix[0, 1] == 10000  # All water→land
        assert result.transition_matrix.shape == (6, 6)

    def test_partial_change(self):
        pre = np.zeros((100, 100), dtype=np.uint8)
        post = np.zeros((100, 100), dtype=np.uint8)
        post[:50, :] = 1  # Half changed
        result = self.metrics.compute(pre, post)
        assert abs(result.total_changed_percentage - 50.0) < 0.01


class TestSeverityEngine:
    def setup_method(self):
        self.engine = SeverityEngine(config_path=None)  # Use defaults

    def test_zero_severity(self):
        result = self.engine.calculate_severity(
            flood_percentage=0,
            building_damage_score=0,
            changed_area_percentage=0,
            vegetation_loss_percentage=0,
        )
        assert result.total_score == 0.0
        assert result.severity_level == "LOW"

    def test_critical_severity(self):
        result = self.engine.calculate_severity(
            flood_percentage=100,
            building_damage_score=100,
            changed_area_percentage=100,
            vegetation_loss_percentage=100,
        )
        assert result.total_score == 100.0
        assert result.severity_level == "CRITICAL"

    def test_moderate_severity(self):
        result = self.engine.calculate_severity(
            flood_percentage=30,
            building_damage_score=20,
            changed_area_percentage=40,
            vegetation_loss_percentage=25,
        )
        assert 20 < result.total_score < 50
        assert result.severity_level in ["MODERATE", "HIGH"]

    def test_component_breakdown(self):
        result = self.engine.calculate_severity(
            flood_percentage=50,
            building_damage_score=30,
            changed_area_percentage=20,
            vegetation_loss_percentage=10,
        )
        assert "flood" in result.component_scores
        assert "building_damage" in result.component_scores
        assert result.component_scores["flood"]["weight"] == 0.40

    def test_explanation(self):
        result = self.engine.calculate_severity(
            flood_percentage=50,
        )
        assert "Score =" in result.explanation

    def test_config_loading(self):
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "disaster_assessment",
            "configs",
            "severity_config.yaml",
        )
        if os.path.isfile(config_path):
            engine = SeverityEngine(config_path=config_path)
            result = engine.calculate_severity(flood_percentage=50)
            assert result.total_score > 0
