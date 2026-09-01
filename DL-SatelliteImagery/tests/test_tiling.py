"""
Tests for the large GeoTIFF tiling engine.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disaster_assessment.preprocessing.tiling.tile_config import TilingConfig
from disaster_assessment.preprocessing.tiling.tile_generator import TileGenerator, TileInfo
from disaster_assessment.preprocessing.tiling.blending import PredictionBlender


class TestTilingConfig:
    def test_default_config(self):
        config = TilingConfig()
        assert config.tile_size == 512
        assert config.overlap == 0.25
        assert config.blend_mode == "gaussian"
        assert config.stride == 384

    def test_custom_config(self):
        config = TilingConfig(tile_size=256, overlap=0.5, blend_mode="uniform")
        assert config.stride == 128

    def test_invalid_overlap(self):
        with pytest.raises(ValueError):
            TilingConfig(overlap=0.8)

    def test_invalid_tile_size(self):
        with pytest.raises(ValueError):
            TilingConfig(tile_size=32)

    def test_invalid_blend_mode(self):
        with pytest.raises(ValueError):
            TilingConfig(blend_mode="invalid")


class TestTileGenerator:
    def test_generate_grid_small(self):
        gen = TileGenerator(TilingConfig(tile_size=256, overlap=0.25))
        tiles = gen.generate_grid(256, 256)
        assert len(tiles) == 1
        assert tiles[0].tile_id == 0
        assert tiles[0].x_offset == 0
        assert tiles[0].y_offset == 0
        assert tiles[0].width == 256
        assert tiles[0].height == 256

    def test_generate_grid_large(self):
        gen = TileGenerator(TilingConfig(tile_size=256, overlap=0.25))
        tiles = gen.generate_grid(1024, 1024)
        assert len(tiles) > 1
        # All tiles should be within bounds
        for t in tiles:
            assert t.x_offset >= 0
            assert t.y_offset >= 0

    def test_tile_coordinates(self):
        gen = TileGenerator(TilingConfig(tile_size=100, overlap=0.2))
        tiles = gen.generate_grid(300, 300)
        assert len(tiles) > 0
        # All tiles should have valid dimensions
        for t in tiles:
            assert t.width > 0
            assert t.height > 0
            assert t.width <= 100
            assert t.height <= 100

    def test_edge_tiles(self):
        gen = TileGenerator(TilingConfig(tile_size=256, overlap=0.25))
        tiles = gen.generate_grid(500, 500)
        edge_tiles = [t for t in tiles if t.is_edge]
        assert len(edge_tiles) > 0

    def test_overlap_correctness(self):
        config = TilingConfig(tile_size=100, overlap=0.5)
        gen = TileGenerator(config)
        tiles = gen.generate_grid(200, 200)
        # With 50% overlap and stride=50, should have more tiles
        assert len(tiles) >= 4

    def test_tile_count(self):
        gen = TileGenerator(TilingConfig(tile_size=256, overlap=0.25))
        count = gen.tile_count(1024, 1024)
        assert count > 0
        assert count == len(gen.generate_grid(1024, 1024))

    def test_weight_masks(self):
        gen = TileGenerator(TilingConfig(tile_size=256, blend_mode="gaussian"))
        tile = TileInfo(tile_id=0, x_offset=0, y_offset=0, width=256, height=256)
        mask, _ = gen.get_overlap_masks(tile)
        assert mask.shape == (256, 256)
        assert mask.min() > 0
        # Center should have higher weight than edges
        center_val = mask[128, 128]
        corner_val = mask[0, 0]
        assert center_val > corner_val

    def test_gaussian_blending_mask(self):
        gen = TileGenerator(TilingConfig(tile_size=128, overlap=0.25, blend_mode="gaussian"))
        tile = TileInfo(tile_id=0, x_offset=0, y_offset=0, width=128, height=128)
        mask, _ = gen.get_overlap_masks(tile)
        assert mask.shape == (128, 128)
        # Check that it's symmetric
        np.testing.assert_allclose(mask, mask.T, atol=1e-5)

    def test_uniform_mask(self):
        gen = TileGenerator(TilingConfig(tile_size=128, overlap=0.25, blend_mode="uniform"))
        tile = TileInfo(tile_id=0, x_offset=0, y_offset=0, width=128, height=128)
        mask, _ = gen.get_overlap_masks(tile)
        np.testing.assert_array_equal(mask, np.ones((128, 128), dtype=np.float32))


class TestPredictionBlender:
    def test_initialize(self):
        blender = PredictionBlender()
        blender.initialize(512, 512, num_channels=1)
        assert blender.num_accumulated == 0

    def test_accumulate_and_blend_single(self):
        blender = PredictionBlender(TilingConfig(blend_mode="uniform"))
        blender.initialize(512, 512, num_channels=1)

        tile = TileInfo(tile_id=0, x_offset=0, y_offset=0, width=256, height=256)
        pred = np.ones((1, 256, 256), dtype=np.float32) * 0.8
        blender.accumulate(tile, pred)

        result = blender.blend()
        assert result.shape == (1, 512, 512)
        assert blender.num_accumulated == 1

    def test_accumulate_multiple_tiles(self):
        blender = PredictionBlender(TilingConfig(tile_size=256, overlap=0.0, blend_mode="uniform"))
        blender.initialize(512, 512, num_channels=1)

        # Tile 0: top-left
        tile0 = TileInfo(tile_id=0, x_offset=0, y_offset=0, width=256, height=256)
        blender.accumulate(tile0, np.ones((1, 256, 256)) * 0.5)

        # Tile 1: top-right
        tile1 = TileInfo(tile_id=1, x_offset=256, y_offset=0, width=256, height=256)
        blender.accumulate(tile1, np.ones((1, 256, 256)) * 0.9)

        result = blender.blend()
        assert result.shape == (1, 512, 512)
        assert blender.num_accumulated == 2

    def test_blend_mask(self):
        blender = PredictionBlender(TilingConfig(blend_mode="uniform"))
        blender.initialize(256, 256, num_channels=1)

        tile = TileInfo(tile_id=0, x_offset=0, y_offset=0, width=256, height=256)
        pred = np.ones((1, 256, 256), dtype=np.float32)
        pred[:, :128, :128] = 0.0
        blender.accumulate(tile, pred)

        mask = blender.blend_mask()
        assert mask.shape == (256, 256)
        assert mask[0, 0] == 0
        assert mask[200, 200] == 1

    def test_multiclass_blend(self):
        blender = PredictionBlender(TilingConfig(blend_mode="uniform"))
        blender.initialize(128, 128, num_channels=4)

        tile = TileInfo(tile_id=0, x_offset=0, y_offset=0, width=128, height=128)
        pred = np.zeros((4, 128, 128), dtype=np.float32)
        pred[0, :, :] = 1.0  # Class 0 everywhere
        blender.accumulate(tile, pred)

        mask = blender.blend_mask()
        assert mask.shape == (128, 128)
        assert np.all(mask == 0)

    def test_reset(self):
        blender = PredictionBlender()
        blender.initialize(256, 256)
        assert blender.num_accumulated == 0

        tile = TileInfo(tile_id=0, x_offset=0, y_offset=0, width=256, height=256)
        blender.accumulate(tile, np.ones((1, 256, 256)))
        assert blender.num_accumulated == 1

        blender.reset()
        assert blender.num_accumulated == 0

    def test_edge_tile_blending(self):
        """Test that edge tiles with smaller dimensions blend correctly."""
        blender = PredictionBlender(TilingConfig(tile_size=256, overlap=0.0, blend_mode="uniform"))
        blender.initialize(300, 300, num_channels=1)

        # Full tile
        tile0 = TileInfo(tile_id=0, x_offset=0, y_offset=0, width=256, height=256)
        blender.accumulate(tile0, np.ones((1, 256, 256)) * 0.5)

        # Edge tile (smaller)
        tile1 = TileInfo(tile_id=1, x_offset=256, y_offset=0, width=44, height=256)
        blender.accumulate(tile1, np.ones((1, 256, 44)) * 0.9)

        result = blender.blend()
        assert result.shape == (1, 300, 300)
