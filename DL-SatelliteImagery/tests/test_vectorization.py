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
