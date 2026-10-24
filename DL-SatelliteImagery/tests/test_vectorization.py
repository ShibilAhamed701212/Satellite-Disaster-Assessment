"""
Tests for vectorization, GeoJSON export.
"""

import pytest
import numpy as np
import json
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disaster_assessment.analytics.vectorization.polygonizer import mask_to_polygons, export_predictions
from disaster_assessment.analytics.vectorization.geojson_export import export_geojson, export_predictions_geojson


class TestMaskToPolygons:
    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        features = mask_to_polygons(mask)
        assert len(features) == 0

    def test_single_class(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        features = mask_to_polygons(mask)
        assert len(features) >= 1
        assert features[0].class_id == 1

    def test_multiple_classes(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:50, :50] = 1
        mask[50:, 50:] = 2
        features = mask_to_polygons(mask)
        class_ids = {f.class_id for f in features}
        assert 1 in class_ids
        assert 2 in class_ids

    def test_class_labels(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:50, 10:50] = 1
        labels = {1: "Flood"}
        features = mask_to_polygons(mask, class_labels=labels)
        assert len(features) >= 1
        assert features[0].class_label == "Flood"

    def test_min_area_filter(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0, 0] = 1  # Single pixel
        features = mask_to_polygons(mask, min_area_pixels=10)
        assert len(features) == 0

    def test_geometry_is_valid(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:90, 10:90] = 1
        features = mask_to_polygons(mask)
        assert len(features) >= 1
        assert features[0].geometry.is_valid

    def test_properties(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:50, 10:50] = 3
        features = mask_to_polygons(mask, class_labels={3: "Building"})
        assert len(features) >= 1
        assert "class_id" in features[0].properties
        assert "class_label" in features[0].properties


class TestGeoJSONExport:
    def test_export_empty(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            output_path = os.path.join(tmp_dir, "empty.geojson")
            result = export_geojson([], output_path)
            assert os.path.isfile(result)
            with open(result) as f:
                data = json.load(f)
            assert data["type"] == "FeatureCollection"
            assert len(data["features"]) == 0
        finally:
            shutil.rmtree(tmp_dir)

    def test_export_with_features(self):
        from disaster_assessment.analytics.vectorization.polygonizer import PolygonFeature
        from shapely.geometry import box

        features = [
            PolygonFeature(
                geometry=box(0, 0, 10, 10),
                class_id=1,
                class_label="Flood",
                area_pixels=100,
            )
        ]

        tmp_dir = tempfile.mkdtemp()
        try:
            output_path = os.path.join(tmp_dir, "test.geojson")
            result = export_geojson(features, output_path)
            assert os.path.isfile(result)
            with open(result) as f:
                data = json.load(f)
            assert len(data["features"]) == 1
            assert data["features"][0]["properties"]["class_label"] == "Flood"
        finally:
            shutil.rmtree(tmp_dir)

    def test_export_predictions_geojson(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1

        tmp_dir = tempfile.mkdtemp()
        try:
            output_path = os.path.join(tmp_dir, "pred.geojson")
            result = export_predictions_geojson(mask, output_path)
            assert os.path.isfile(result)
        finally:
            shutil.rmtree(tmp_dir)

    def test_export_predictions(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:90, 10:90] = 1

        tmp_dir = tempfile.mkdtemp()
        try:
            output_path = os.path.join(tmp_dir, "export.geojson")
            result = export_predictions(mask, output_path=output_path, format="geojson")
            assert os.path.isfile(result)
        finally:
            shutil.rmtree(tmp_dir)


class TestGeoreferencedExport:
    """raster_metadata['transform'] must place polygons in map coordinates."""

    def test_transform_applied_to_polygons(self):
        from rasterio.transform import from_origin

        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:4, 5:8] = 1  # rows 2-3, cols 5-7
        transform = from_origin(500000.0, 4000000.0, 10.0, 10.0)

        features = mask_to_polygons(mask, transform=transform)

        assert len(features) == 1
        minx, miny, maxx, maxy = features[0].geometry.bounds
        assert (minx, maxx) == (500050.0, 500080.0)
        assert (miny, maxy) == (3999960.0, 3999980.0)
        # Pixel area is still reported in pixels.
        assert features[0].area_pixels == 6

    def test_export_predictions_uses_metadata_transform(self, tmp_path):
        import json
        from rasterio.transform import from_origin

        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:2, 0:2] = 1
        out = tmp_path / "geo.geojson"
        meta = {"crs": "EPSG:32633", "transform": from_origin(500000.0, 4000000.0, 10.0, 10.0)}

        export_predictions(mask, raster_metadata=meta, output_path=str(out), simplify_tolerance=0)

        doc = json.loads(out.read_text())
        xs = [pt[0] for pt in doc["features"][0]["geometry"]["coordinates"][0]]
        ys = [pt[1] for pt in doc["features"][0]["geometry"]["coordinates"][0]]
        # UTM 33N easting 500000 is the 15 deg E central meridian; ~36 deg N.
        assert min(xs) == pytest.approx(15.0, abs=1e-6)
        assert 15.0 < max(xs) < 15.001
        assert 36.0 < min(ys) < max(ys) < 36.2
        assert doc["metadata"]["crs"] == "EPSG:4326"
        assert doc["metadata"]["source_crs"] == "EPSG:32633"

    def test_export_predictions_kml_in_wgs84(self, tmp_path):
        from rasterio.transform import from_origin

        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:2, 0:2] = 1
        out = tmp_path / "geo.kml"
        meta = {"crs": "EPSG:32633", "transform": from_origin(500000.0, 4000000.0, 10.0, 10.0)}

        export_predictions(mask, raster_metadata=meta, output_path=str(out), format="kml")

        text = out.read_text()
        assert "500000" not in text
        assert "15.0" in text

    def test_resized_mask_scales_transform(self):
        from rasterio.transform import from_origin

        from disaster_assessment.analytics.vectorization.polygonizer import mask_transform

        meta = {"transform": from_origin(0.0, 100.0, 1.0, 1.0), "width": 40, "height": 20}
        t = mask_transform(meta, (10, 20))  # mask is half the source size
        assert t.a == pytest.approx(2.0) and t.e == pytest.approx(-2.0)
        assert (t.c, t.f) == (0.0, 100.0)
        # Same size, or no size info: transform is unchanged.
        assert mask_transform({"transform": meta["transform"]}, (10, 20)).a == 1.0
        assert mask_transform(None, (10, 20)) is None


def test_kml_defines_each_class_style_once(tmp_path):
    from disaster_assessment.analytics.vectorization.kml_export import export_kml

    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:3, 0:3] = 1
    mask[10:13, 10:13] = 1  # second polygon, same class
    out = tmp_path / "out.kml"
    export_kml(mask_to_polygons(mask), str(out))

    assert out.read_text().count('id="style_1"') == 1
