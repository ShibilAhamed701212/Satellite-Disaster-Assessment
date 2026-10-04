"""
Tests for DEM loading, terrain analysis, and flood depth estimation.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disaster_assessment.analytics.elevation.terrain_analysis import TerrainAnalyzer
from disaster_assessment.analytics.elevation.flood_depth import FloodDepthEstimator
from disaster_assessment.analytics.elevation.volume import VolumeCalculator


class TestTerrainAnalyzer:
    def test_flat_terrain(self):
        analyzer = TerrainAnalyzer()
        elevation = np.ones((100, 100), dtype=np.float32) * 100.0
        result = analyzer.analyze(elevation)
        assert result.mean_slope == pytest.approx(0.0, abs=0.1)
        assert result.slope is not None
        assert result.aspect is not None
        assert result.hillshade is not None

    def test_sloped_terrain(self):
        analyzer = TerrainAnalyzer()
        elevation = np.linspace(0, 100, 100).reshape(10, 10).astype(np.float32)
        result = analyzer.analyze(elevation)
        assert result.mean_slope > 0
        assert result.max_slope > 0

    def test_elevation_stats(self):
        analyzer = TerrainAnalyzer()
        elevation = np.random.rand(50, 50).astype(np.float32) * 500
        result = analyzer.analyze(elevation)
        assert result.mean_elevation > 0
        assert result.elevation_range[1] > result.elevation_range[0]

    def test_resolution_effect(self):
        analyzer = TerrainAnalyzer()
        elevation = np.linspace(0, 100, 100).reshape(10, 10).astype(np.float32)
        result_1m = analyzer.analyze(elevation, resolution=1.0)
        result_10m = analyzer.analyze(elevation, resolution=10.0)
        # Higher resolution => lower slope in degrees
        assert result_1m.mean_slope > result_10m.mean_slope

    def test_empty_dem(self):
        analyzer = TerrainAnalyzer()
        result = analyzer.analyze(np.array([]))
        assert result.mean_slope == 0.0


class TestFloodDepthEstimator:
    def test_estimate_with_wse(self):
        estimator = FloodDepthEstimator()
        flood_mask = np.zeros((100, 100), dtype=np.uint8)
        flood_mask[20:80, 20:80] = 1
        dem = np.ones((100, 100), dtype=np.float32) * 10.0  # 10m elevation

        result = estimator.estimate_from_boundary(
            flood_mask, dem, water_surface_elevation=15.0
        )
        assert result.depth_estimation_status == "COMPLETED"
        assert result.max_depth == pytest.approx(5.0, abs=0.01)
        assert result.mean_depth == pytest.approx(5.0, abs=0.01)
        assert result.depth_raster is not None
        assert result.assumptions

    def test_estimate_with_volume(self):
        estimator = FloodDepthEstimator()
        flood_mask = np.ones((10, 10), dtype=np.uint8)
        dem = np.zeros((10, 10), dtype=np.float32)

        result = estimator.estimate_from_boundary(
            flood_mask, dem, water_surface_elevation=5.0, dem_resolution=10.0
        )
        # 10x10 = 100 pixels, 10m resolution => pixel_area=100m², depth=5m => 100*100*5=50000
        assert result.estimated_volume_m3 == pytest.approx(50000.0, abs=0.1)

    def test_estimate_insufficient_no_dem(self):
        estimator = FloodDepthEstimator()
        result = estimator.estimate_insufficient()
        assert result.depth_estimation_status == "INSUFFICIENT_INPUT"

    def test_estimate_shape_mismatch(self):
        estimator = FloodDepthEstimator()
        flood_mask = np.zeros((100, 100), dtype=np.uint8)
        dem = np.zeros((50, 50), dtype=np.float32)
        result = estimator.estimate_from_boundary(flood_mask, dem, 10.0)
        assert result.depth_estimation_status == "INSUFFICIENT_INPUT"

    def test_no_flood(self):
        estimator = FloodDepthEstimator()
        flood_mask = np.zeros((50, 50), dtype=np.uint8)
        dem = np.ones((50, 50), dtype=np.float32) * 10.0
        result = estimator.estimate_from_boundary(flood_mask, dem, 15.0)
        assert result.depth_estimation_status == "COMPLETED"
        assert result.max_depth == 0.0


class TestVolumeCalculator:
    def test_calculate_volume(self):
        calc = VolumeCalculator()
        depth = np.ones((10, 10), dtype=np.float32) * 2.0  # 2m depth
        flood = np.ones((10, 10), dtype=np.uint8)
        result = calc.calculate(depth, flood, pixel_resolution_m=10.0)
        # 10x10 = 100 pixels, 10m resolution => pixel_area=100m², depth=2m => 100*100*2=20000
        assert result.total_volume_m3 == pytest.approx(20000.0, abs=0.1)
        assert result.mean_depth == pytest.approx(2.0, abs=0.01)
        assert result.flooded_area_m2 == pytest.approx(10000.0, abs=0.1)

    def test_no_resolution(self):
        calc = VolumeCalculator()
        depth = np.ones((10, 10), dtype=np.float32) * 2.0
        flood = np.ones((10, 10), dtype=np.uint8)
        result = calc.calculate(depth, flood)
        assert result.volume_status == "AREA_RESOLUTION_MISSING"

    def test_no_flood(self):
        calc = VolumeCalculator()
        depth = np.zeros((10, 10), dtype=np.float32)
        flood = np.zeros((10, 10), dtype=np.uint8)
        result = calc.calculate(depth, flood, pixel_resolution_m=10.0)
        assert result.total_volume_m3 == 0.0

    def test_missing_inputs(self):
        calc = VolumeCalculator()
        result = calc.calculate(None, None)
        assert result.volume_status == "INSUFFICIENT_INPUT"
