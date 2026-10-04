"""
Tests for the Gradio dashboard wiring (no server is started).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

gr = pytest.importorskip("gradio")

import disaster_gradio_app as app_module  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GEOTIFF = os.path.join(REPO_ROOT, "sample_images", "geotiff_rasters", "shade_topography.tif")


def test_sample_root_points_at_repository_samples():
    assert os.path.isfile(os.path.join(app_module.SAMPLE_ROOT, "pre_disaster.png"))
    assert os.path.isfile(os.path.join(app_module.SAMPLE_ROOT, "post_disaster.png"))


def test_upload_path_accepts_gradio4_string_and_legacy_wrapper():
    class Legacy:
        name = "/tmp/legacy.tif"

    assert app_module._upload_path(None) is None
    assert app_module._upload_path("/tmp/x.tif") == "/tmp/x.tif"
    assert app_module._upload_path(Legacy()) == "/tmp/legacy.tif"


def test_geotiff_upload_is_used_for_area():
    pytest.importorskip("rasterio")
    pre = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    post = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    # Gradio 4+ hands the handler a plain filepath string.
    outputs = app_module.analyze_disaster(pre, post, "Auto", None, GEOTIFF)
    metrics_text = outputs[-1]

    assert "Error" not in metrics_text
    assert "GeoTIFF" in metrics_text


def test_create_app_builds_with_examples():
    demo = app_module.create_app()
    examples = [b for b in demo.blocks.values() if isinstance(b, gr.Dataset)]
    assert len(examples) >= 3
