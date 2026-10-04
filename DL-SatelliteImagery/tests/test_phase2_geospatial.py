"""
Tests for Phase 2: Georeferenced Flood Area Calculation & GeoTIFF Parsing.
"""

import os
import shutil
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disaster_assessment.analytics.geospatial_area import (
    GeospatialAreaCalculator,
)
from disaster_assessment.pipeline.disaster_analyzer import DisasterAnalyzer


class TestGeospatialAreaCalculator:
    def test_user_supplied_resolution(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:20, :50] = 1  # 1000 flooded pixels
        meters_per_pixel = 0.5  # 0.25 m² per pixel

        result = GeospatialAreaCalculator.calculate_flood_area(
            flood_mask=mask,
            meters_per_pixel=meters_per_pixel,
        )

        assert result.flooded_pixels == 1000
        assert result.total_pixels == 10000
        assert result.flood_percentage == 10.0
        assert result.flood_area_m2 == 250.0  # 1000 * 0.25 m²
        assert result.flood_area_km2 == 0.00025
        assert "Calculated from user ground resolution" in result.area_status

    def test_no_resolution_fallback(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[:50, :50] = 1  # 2500 flooded pixels

        result = GeospatialAreaCalculator.calculate_flood_area(
            flood_mask=mask,
            geotiff_path=None,
            meters_per_pixel=None,
        )

        assert result.flooded_pixels == 2500
        assert result.flood_percentage == 6.25
        assert result.flood_area_m2 is None
        assert result.flood_area_km2 is None
        assert "Geospatial resolution unavailable" in result.area_status

    def test_degrees_to_meters(self):
        # At equator (lat=0): 1 deg lon ~ 111.4 km, 1 deg lat ~ 110.5 km
        w_m, h_m, area_m2 = GeospatialAreaCalculator._degrees_to_meters(
            dx_deg=0.001, dy_deg=0.001, center_lat_deg=0.0
        )
        assert 100 < w_m < 120
        assert 100 < h_m < 120
        assert area_m2 > 10000

        # At high latitude (lat=60): lon width is halved (cos(60)=0.5)
        w_m60, h_m60, area_m60 = GeospatialAreaCalculator._degrees_to_meters(
            dx_deg=0.001, dy_deg=0.001, center_lat_deg=60.0
        )
        assert abs(w_m60 - (w_m * 0.5)) < 15.0

    def test_reconstruct_full_mask(self):
        patches = [np.full((256, 256), i, dtype=np.uint8) for i in range(4)]
        reconstructed = GeospatialAreaCalculator.reconstruct_full_mask(
            patches=patches,
            grid_rows=2,
            grid_cols=2,
            original_h=500,
            original_w=500,
            patch_size=256,
        )
        assert reconstructed.shape == (500, 500)
        assert reconstructed[0, 0] == 0
        assert reconstructed[0, 300] == 1
        assert reconstructed[300, 0] == 2
        assert reconstructed[300, 300] == 3


class TestGeoTIFFMetadataExtraction:
    @pytest.fixture(autouse=True)
    def setup_geotiff(self):
        self.temp_dir = tempfile.mkdtemp()
        self.projected_tif = os.path.join(self.temp_dir, "projected.tif")
        self.geographic_tif = os.path.join(self.temp_dir, "geographic.tif")

        try:
            import rasterio
            from rasterio.crs import CRS
            from rasterio.transform import from_origin

            # 1. Create a synthetic Projected GeoTIFF (UTM Zone 33N, resolution 10m)
            transform_proj = from_origin(500000, 5000000, 10, 10)
            data = np.zeros((1, 100, 100), dtype=np.uint8)
            with rasterio.open(
                self.projected_tif,
                "w",
                driver="GTiff",
                height=100,
                width=100,
                count=1,
                dtype=data.dtype,
                crs=CRS.from_epsg(32633),
                transform=transform_proj,
            ) as dst:
                dst.write(data)

            # 2. Create a synthetic Geographic GeoTIFF (EPSG:4326, resolution 0.0001 deg)
            transform_geo = from_origin(10.0, 45.0, 0.0001, 0.0001)
            with rasterio.open(
                self.geographic_tif,
                "w",
                driver="GTiff",
                height=100,
                width=100,
                count=1,
                dtype=data.dtype,
                crs=CRS.from_epsg(4326),
                transform=transform_geo,
            ) as dst:
                dst.write(data)

            self.rasterio_available = True
        except ImportError:
            self.rasterio_available = False

        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_projected_geotiff_extraction(self):
        if not self.rasterio_available:
            pytest.skip("rasterio not available")

        meta = GeospatialAreaCalculator.extract_geotiff_metadata(self.projected_tif)
        assert meta.has_georeference
        assert meta.is_projected
        assert meta.pixel_width_m == 10.0
        assert meta.pixel_height_m == 10.0
        assert meta.pixel_area_m2 == 100.0

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:10, :10] = 1  # 100 pixels
        res = GeospatialAreaCalculator.calculate_flood_area(mask, geotiff_path=self.projected_tif)
        assert res.flood_area_m2 == 10000.0  # 100 * 100 m²
        assert res.flood_area_km2 == 0.01

    def test_geographic_geotiff_extraction(self):
        if not self.rasterio_available:
            pytest.skip("rasterio not available")

        meta = GeospatialAreaCalculator.extract_geotiff_metadata(self.geographic_tif)
        assert meta.has_georeference
        assert not meta.is_projected
        assert meta.pixel_area_m2 is not None
        assert meta.pixel_area_m2 > 0


class TestDisasterAnalyzerPhase2Integration:
    def setup_method(self):
        self.analyzer = DisasterAnalyzer()

    def test_analyze_with_resolution_parameter(self):
        pre = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        report = self.analyzer.analyze(
            pre_image=pre,
            post_image=post,
            damage_mode="estimated",
            meters_per_pixel=1.0,
        )

        assert report.damage_mode == "estimated"
        assert report.flood_area_m2 is not None
        assert "meters_per_pixel" in report.area_status or "user" in report.area_status.lower()

    def test_analyze_auto_damage_mode(self):
        pre = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        report = self.analyzer.analyze(
            pre_image=pre,
            post_image=post,
            damage_mode="auto",
        )

        # Without trained weights, auto falls back to estimated
        assert report.damage_mode == "estimated"
        assert report.damage_output is not None
        assert report.damage_output.overall_damage_score >= 0
