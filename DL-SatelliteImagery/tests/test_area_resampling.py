"""
Regression tests: flood area must be measured at the input image's ground
resolution, even though the models run on a resized (target_size) copy.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disaster_assessment.analytics.geospatial_area import GeospatialAreaCalculator
from disaster_assessment.pipeline.disaster_analyzer import DisasterAnalyzer


class _AllFloodModel:
    """Stand-in for a validated FloodUNet that marks every pixel as flooded."""

    def predict(self, x):
        return torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]))


def _analyzer_with_full_flood():
    analyzer = DisasterAnalyzer(target_size=(256, 256), device=torch.device("cpu"))
    analyzer._flood_model = _AllFloodModel()
    return analyzer


def test_gsd_area_uses_original_image_resolution():
    """512x512 image at 10 m/px fully flooded = 512*512*100 m2, not 256*256*100 m2."""
    analyzer = _analyzer_with_full_flood()
    pre = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    post = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    report = analyzer.analyze(pre, post, meters_per_pixel=10.0)

    assert report.flood_mode == "trained"
    assert report.flood_percentage == 100.0
    assert report.flood_area_m2 == 512 * 512 * 100.0
    assert report.to_dict()["flood_area_km2"] == 26.2144


def test_gsd_area_non_square_resize():
    analyzer = _analyzer_with_full_flood()
    pre = np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8)
    post = np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8)

    report = analyzer.analyze(pre, post, meters_per_pixel=2.0)

    assert report.flood_area_m2 == 300 * 600 * 4.0


def test_source_shape_scales_pixel_area():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:10, :10] = 1  # 1% of the scene

    result = GeospatialAreaCalculator.calculate_flood_area(
        flood_mask=mask, meters_per_pixel=1.0, source_shape=(200, 400)
    )

    # Scene is 200x400 m; 1% of it is 800 m2.
    assert result.flood_area_m2 == 800.0
    assert result.georeference.pixel_area_m2 == 8.0


def test_geotiff_area_scales_to_raster_dimensions(tmp_path):
    import pytest

    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    path = str(tmp_path / "scene.tif")
    data = np.zeros((1, 200, 200), dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", height=200, width=200, count=1,
        dtype=data.dtype, crs=CRS.from_epsg(32633),
        transform=from_origin(500000, 5000000, 10, 10),
    ) as dst:
        dst.write(data)

    # Model ran on a 100x100 copy of the 200x200 raster; whole mask flooded.
    mask = np.ones((100, 100), dtype=np.uint8)
    result = GeospatialAreaCalculator.calculate_flood_area(mask, geotiff_path=path)

    assert result.flood_area_m2 == 200 * 200 * 100.0


def test_unusable_geotiff_warning_reaches_report(tmp_path):
    analyzer = DisasterAnalyzer(target_size=(64, 64), device=torch.device("cpu"))
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    report = analyzer.analyze(img, img, geotiff_path=str(tmp_path / "missing.tif"))

    assert any(w.startswith("GeoTIFF:") for w in report.warnings)
