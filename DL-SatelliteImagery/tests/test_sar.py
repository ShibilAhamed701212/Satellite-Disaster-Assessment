"""
Tests for SAR preprocessing modules.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disaster_assessment.preprocessing.sar.normalization import normalize_sar, to_db_scale, from_db_scale
from disaster_assessment.preprocessing.sar.speckle_filter import median_filter_sar, lee_filter_sar
from disaster_assessment.preprocessing.sar.coherence import compute_sar_change_features
from disaster_assessment.preprocessing.sar.sentinel1 import Sentinel1Processor, SARInput


class TestSARNormalization:
    def test_minmax_normalize(self):
        data = np.array([[0.0, 0.5], [1.0, 2.0]])
        result = normalize_sar(data, method="minmax")
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_clip_normalize(self):
        data = np.array([[0.0, 0.5], [1.0, 2.0]])
        result = normalize_sar(data, method="minmax_clip")
        assert result.dtype == np.float32

    def test_robust_normalize(self):
        data = np.array([[0.0, 0.5], [1.0, 2.0]])
        result = normalize_sar(data, method="robust")
        assert result.dtype == np.float32

    def test_zero_data(self):
        data = np.zeros((10, 10))
        result = normalize_sar(data)
        assert result.shape == (10, 10)

    def test_db_scale(self):
        data = np.array([[1.0, 10.0], [100.0, 1000.0]])
        result = to_db_scale(data)
        assert result.dtype == np.float32
        assert result[0, 0] == pytest.approx(0.0, abs=0.01)
        assert result[0, 1] == pytest.approx(10.0, abs=0.01)
        assert result[1, 0] == pytest.approx(20.0, abs=0.01)

    def test_from_db_scale_roundtrip(self):
        data = np.array([[1.0, 10.0], [100.0, 1000.0]])
        db = to_db_scale(data)
        recovered = from_db_scale(db)
        np.testing.assert_allclose(recovered, data, rtol=0.01)

    def test_negative_db(self):
        data = np.array([[0.01, 0.1]])
        result = to_db_scale(data)
        assert result[0, 0] < 0  # Negative dB for values < 1


class TestSpeckleFilter:
    def test_median_filter(self):
        data = np.random.rand(64, 64).astype(np.float32)
        data[32, 32] = 100.0  # Spike
        result = median_filter_sar(data, kernel_size=3)
        assert result.shape == data.shape
        # Spike should be reduced
        assert result[32, 32] < 100.0

    def test_median_filter_preserves_shape(self):
        data = np.random.rand(128, 128).astype(np.float32)
        result = median_filter_sar(data, kernel_size=5)
        assert result.shape == data.shape

    def test_lee_filter(self):
        data = np.random.rand(64, 64).astype(np.float32) * 0.5
        result = lee_filter_sar(data, kernel_size=3)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_filter_no_change_small_kernel(self):
        data = np.random.rand(32, 32).astype(np.float32)
        result = median_filter_sar(data, kernel_size=1)
        np.testing.assert_array_equal(result, data)


class TestSARChange:
    def test_difference(self):
        pre = np.array([[1.0, 2.0], [3.0, 4.0]])
        post = np.array([[4.0, 3.0], [2.0, 1.0]])
        result = compute_sar_change_features(pre, post, "difference")
        expected = np.array([[3.0, 1.0], [-1.0, -3.0]])
        np.testing.assert_allclose(result, expected)

    def test_ratio(self):
        pre = np.array([[1.0, 2.0]])
        post = np.array([[2.0, 2.0]])
        result = compute_sar_change_features(pre, post, "ratio")
        assert result[0, 0] > 0  # ln(2)
        assert abs(result[0, 1]) < 0.01  # ln(1) = 0

    def test_log_ratio(self):
        pre = np.array([[1.0, 2.0]])
        post = np.array([[4.0, 2.0]])
        result = compute_sar_change_features(pre, post, "log_ratio")
        np.testing.assert_allclose(result[0, 0], np.log(4.0), atol=0.01)
        np.testing.assert_allclose(result[0, 1], 0.0, atol=0.01)


class TestSentinel1Processor:
    def test_preprocess(self):
        proc = Sentinel1Processor(enable_speckle_filter=True)
        vv = np.random.rand(64, 64).astype(np.float32) * 0.5
        sar_input = SARInput(vv=vv)
        result = proc.preprocess(sar_input)
        assert result.vv is not None
        assert result.vv.shape == vv.shape

    def test_preprocess_no_data(self):
        proc = Sentinel1Processor()
        sar_input = SARInput()
        result = proc.preprocess(sar_input)
        assert result.vv is None
        assert result.vh is None

    def test_compute_change(self):
        proc = Sentinel1Processor(flood_threshold_db=-2.0)
        pre = SARInput(vv=np.random.rand(64, 64).astype(np.float32) * 0.5)
        post = SARInput(vv=np.random.rand(64, 64).astype(np.float32) * 0.5)
        result = proc.compute_change(pre, post)
        assert result.status == "COMPLETED"
        assert result.vv_difference is not None
        assert result.flood_mask is not None
        assert result.change_magnitude is not None

    def test_compute_change_insufficient(self):
        proc = Sentinel1Processor()
        pre = SARInput()
        post = SARInput()
        result = proc.compute_change(pre, post)
        assert result.status == "INSUFFICIENT_INPUT"

    def test_estimate_coherence(self):
        proc = Sentinel1Processor()
        pre = np.random.rand(64, 64).astype(np.float32)
        post = np.random.rand(64, 64).astype(np.float32)
        coherence = proc.estimate_coherence(pre, post)
        assert coherence is not None
        assert coherence.shape == (64, 64)
        assert coherence.min() >= 0
        assert coherence.max() <= 1
