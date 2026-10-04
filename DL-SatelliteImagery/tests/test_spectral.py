"""
Tests for multispectral indices and band mapping.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disaster_assessment.analytics.spectral.indices import (
    compute_ndvi, compute_ndwi, compute_mndwi, compute_nbr, compute_dnbr,
)
from disaster_assessment.analytics.spectral.band_mapping import (
    SENTINEL2_BANDS, LANDSAT_BANDS, RGB_BANDS, get_band_mapping,
)
from disaster_assessment.analytics.spectral.validators import validate_bands, extract_band


class TestNDVI:
    def test_basic_ndvi(self):
        nir = np.array([[0.8, 0.6], [0.3, 0.1]])
        red = np.array([[0.1, 0.2], [0.3, 0.1]])
        result = compute_ndvi(nir, red)
        assert result.shape == (2, 2)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_ndvi_healthy_vegetation(self):
        nir = np.full((10, 10), 0.8)
        red = np.full((10, 10), 0.1)
        result = compute_ndvi(nir, red)
        expected = (0.8 - 0.1) / (0.8 + 0.1)
        assert abs(result.mean() - expected) < 0.01

    def test_ndvi_water(self):
        nir = np.full((10, 10), 0.1)
        red = np.full((10, 10), 0.05)
        result = compute_ndvi(nir, red)
        # Water has low NDVI
        assert result.mean() < 0.5

    def test_ndvi_zero_division(self):
        nir = np.zeros((5, 5))
        red = np.zeros((5, 5))
        result = compute_ndvi(nir, red)
        assert not np.any(np.isinf(result))

    def test_ndvi_nodata(self):
        nir = np.array([[0.8, 0.6], [0.3, 0.1]])
        red = np.array([[0.1, 0.2], [-9999, 0.1]])
        result = compute_ndvi(nir, red, nodata=-9999)
        assert np.isnan(result[1, 0])

    def test_ndvi_clipping(self):
        nir = np.array([[2.0, 0.0]])
        red = np.array([[0.0, 2.0]])
        result = compute_ndvi(nir, red)
        assert result.min() >= -1.0
        assert result.max() <= 1.0


class TestNDWI:
    def test_basic_ndwi(self):
        green = np.array([[0.5, 0.3]])
        nir = np.array([[0.1, 0.6]])
        result = compute_ndwi(green, nir)
        assert result.min() >= -1.0
        assert result.max() <= 1.0
        # Water pixel (high green, low NIR)
        assert result[0, 0] > 0

    def test_ndwi_zero_division(self):
        green = np.zeros((5, 5))
        nir = np.zeros((5, 5))
        result = compute_ndwi(green, nir)
        assert not np.any(np.isinf(result))


class TestMNDWI:
    def test_basic_mndwi(self):
        green = np.array([[0.5, 0.3]])
        swir = np.array([[0.1, 0.6]])
        result = compute_mndwi(green, swir)
        assert result.min() >= -1.0
        assert result.max() <= 1.0


class TestNBR:
    def test_basic_nbr(self):
        nir = np.array([[0.8, 0.3]])
        swir = np.array([[0.1, 0.6]])
        result = compute_nbr(nir, swir)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_nbr_healthy_vegetation(self):
        nir = np.full((10, 10), 0.8)
        swir = np.full((10, 10), 0.2)
        result = compute_nbr(nir, swir)
        expected = (0.8 - 0.2) / (0.8 + 0.2)
        assert abs(result.mean() - expected) < 0.01


class TestDNBR:
    def test_basic_dnbr(self):
        nbr_pre = np.full((10, 10), 0.5)
        nbr_post = np.full((10, 10), 0.1)
        dnbr, meta = compute_dnbr(nbr_pre, nbr_post)
        assert dnbr.shape == (10, 10)
        assert abs(dnbr.mean() - 0.4) < 0.01
        assert "mean_dnbr" in meta

    def test_dnbr_severity_classification(self):
        # No change
        nbr_pre = np.full((10, 10), 0.5)
        nbr_post = np.full((10, 10), 0.5)
        dnbr, meta = compute_dnbr(nbr_pre, nbr_post)
        assert meta["unburned_pixels"] == 100

    def test_dnbr_high_severity(self):
        nbr_pre = np.full((10, 10), 0.7)
        nbr_post = np.full((10, 10), -0.1)
        dnbr, meta = compute_dnbr(nbr_pre, nbr_post)
        assert meta["high_severity_pixels"] == 100

    def test_dnbr_nan_handling(self):
        nbr_pre = np.array([[0.5, np.nan]])
        nbr_post = np.array([[0.1, 0.3]])
        dnbr, meta = compute_dnbr(nbr_pre, nbr_post)
        assert meta["valid_pixels"] == 1


class TestBandMapping:
    def test_sentinel2_mapping(self):
        assert SENTINEL2_BANDS.sensor == "Sentinel-2"
        assert SENTINEL2_BANDS.bands["red"] == 4
        assert SENTINEL2_BANDS.bands["nir"] == 8
        assert SENTINEL2_BANDS.bands["swir"] == 11

    def test_landsat_mapping(self):
        assert LANDSAT_BANDS.sensor == "Landsat-8/9"
        assert LANDSAT_BANDS.bands["nir"] == 5
        assert LANDSAT_BANDS.bands["swir"] == 6

    def test_rgb_mapping(self):
        assert RGB_BANDS.sensor == "RGB"
        assert RGB_BANDS.bands["red"] == 1

    def test_get_band_mapping(self):
        mapping = get_band_mapping("sentinel-2")
        assert mapping is not None
        assert mapping.sensor == "Sentinel-2"

        mapping2 = get_band_mapping("rgb")
        assert mapping2 is not None

    def test_get_band_mapping_invalid(self):
        mapping = get_band_mapping("nonexistent")
        assert mapping is None

    def test_has_bands(self):
        assert SENTINEL2_BANDS.has_bands(["red", "nir"])
        assert not SENTINEL2_BANDS.has_bands(["nonexistent"])

    def test_get_band_indices(self):
        indices = SENTINEL2_BANDS.get_band_indices(["red", "nir"])
        assert indices == [4, 8]


class TestValidators:
    def test_validate_bands_rgb(self):
        data = np.random.rand(3, 64, 64).astype(np.float32)
        valid, msg = validate_bands(data, ["red", "green", "blue"], RGB_BANDS)
        assert valid

    def test_validate_bands_missing(self):
        data = np.random.rand(3, 64, 64).astype(np.float32)
        valid, msg = validate_bands(data, ["nir"], RGB_BANDS)
        assert not valid

    def test_extract_band(self):
        data = np.random.rand(3, 64, 64).astype(np.float32)
        band = extract_band(data, "red", RGB_BANDS)
        assert band.shape == (64, 64)
        np.testing.assert_array_equal(band, data[0])

    def test_extract_band_hwc(self):
        data = np.random.rand(64, 64, 3).astype(np.float32)
        band = extract_band(data, "red", RGB_BANDS)
        assert band.shape == (64, 64)
