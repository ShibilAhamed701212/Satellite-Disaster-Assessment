"""
Sliding-window tile generator for processing large GeoTIFF files.

Streams tiles from disk without loading the entire raster into memory.
Each tile retains geospatial metadata (transform, CRS, bounds).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .tile_config import TilingConfig


@dataclass
class TileInfo:
    """Metadata for a single tile within a larger raster.

    Attributes:
        tile_id: Unique integer identifier.
        x_offset: Left pixel offset in source raster.
        y_offset: Top pixel offset in source raster.
        width: Tile width in pixels.
        height: Tile height in pixels.
        transform: Affine transform for this tile in source CRS.
        crs: Coordinate Reference System identifier string.
        source_bounds: (left, bottom, right, top) of tile in source CRS.
        is_edge: Whether this tile is on the boundary of the raster.
    """
    tile_id: int
    x_offset: int
    y_offset: int
    width: int
    height: int
    transform: Optional[object] = None
    crs: Optional[str] = None
    source_bounds: Optional[Tuple[float, float, float, float]] = None
    is_edge: bool = False


class TileGenerator:
    """Generate tiles from a large raster using a sliding window.

    Usage:
        gen = TileGenerator(config=TilingConfig(tile_size=512, overlap=0.25))
        tiles = gen.generate_grid(raster_height=4096, raster_width=4096)
        for tile_info in tiles:
            # Read tile from disk using tile_info.x_offset, tile_info.y_offset
            tile_data = read_tile(raster_path, tile_info)
            prediction = model(tile_data)
            blender.accumulate(tile_info, prediction)
        result = blender.blend()
    """

    def __init__(self, config: Optional[TilingConfig] = None):
        self.config = config or TilingConfig()

    def generate_grid(
        self,
        raster_height: int,
        raster_width: int,
        transform: Optional[object] = None,
        crs: Optional[str] = None,
    ) -> List[TileInfo]:
        """Generate all tile positions for a raster of given dimensions.

        Args:
            raster_height: Total raster height in pixels.
            raster_width: Total raster width in pixels.
            transform: Affine transform of the source raster.
            crs: CRS string of the source raster.

        Returns:
            List of TileInfo objects covering the raster.
        """
        stride = self.config.stride
        ts = self.config.tile_size

        # Calculate grid dimensions
        cols = max(1, math.ceil((raster_width - ts) / stride) + 1)
        rows = max(1, math.ceil((raster_height - ts) / stride) + 1)

        tiles = []
        tile_id = 0

        for row in range(rows):
            for col in range(cols):
                # Compute offset, clamping to stay within raster bounds
                x_off = min(col * stride, max(0, raster_width - ts))
                y_off = min(row * stride, max(0, raster_height - ts))

                # Actual tile dimensions (may be smaller at edges)
                w = min(ts, raster_width - x_off)
                h = min(ts, raster_height - y_off)

                is_edge = (x_off == 0 and w < ts) or \
                          (y_off == 0 and h < ts) or \
                          (x_off + w == raster_width) or \
                          (y_off + h == raster_height)

                # Compute tile transform if source transform provided
                tile_transform = None
                tile_bounds = None
                if transform is not None:
                    try:
                        from rasterio.transform import Affine
                        if isinstance(transform, Affine):
                            tile_transform = transform * Affine.translation(x_off, y_off)
                    except (ImportError, Exception):
                        pass

                tiles.append(TileInfo(
                    tile_id=tile_id,
                    x_offset=x_off,
                    y_offset=y_off,
                    width=w,
                    height=h,
                    transform=tile_transform,
                    crs=crs,
                    source_bounds=tile_bounds,
                    is_edge=is_edge,
                ))
                tile_id += 1

        return tiles

    def get_overlap_masks(self, tile_info: TileInfo) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 1D weight masks for the overlap region of a tile.

        For gaussian blend, returns a 2D weight mask.
        For uniform blend, returns ones.
        For linear blend, returns distance-from-center weights.

        Returns:
            (h_mask, w_mask) tuple of 1D weight arrays.
        """
        ts = self.config.tile_size
        h, w = tile_info.height, tile_info.width

        if self.config.blend_mode == "uniform":
            return np.ones((h, w), dtype=np.float32), None

        if self.config.blend_mode == "gaussian":
            # Gaussian weights centered on tile
            sigma = ts / 4.0
            y_coords = np.arange(h, dtype=np.float32)
            x_coords = np.arange(w, dtype=np.float32)
            y_center = h / 2.0
            x_center = w / 2.0

            y_weights = np.exp(-0.5 * ((y_coords - y_center) / sigma) ** 2)
            x_weights = np.exp(-0.5 * ((x_coords - x_center) / sigma) ** 2)

            mask = np.outer(y_weights, x_weights).astype(np.float32)
            return mask, None

        elif self.config.blend_mode == "linear":
            # Linear falloff from center
            y_coords = np.arange(h, dtype=np.float32)
            x_coords = np.arange(w, dtype=np.float32)
            y_center = h / 2.0
            x_center = w / 2.0

            y_weights = 1.0 - np.abs(y_coords - y_center) / y_center
            x_weights = 1.0 - np.abs(x_coords - x_center) / x_center

            y_weights = np.clip(y_weights, 0.01, 1.0)
            x_weights = np.clip(x_weights, 0.01, 1.0)

            mask = np.outer(y_weights, x_weights).astype(np.float32)
            return mask, None

        return np.ones((h, w), dtype=np.float32), None

    def tile_count(self, raster_height: int, raster_width: int) -> int:
        """Return the number of tiles that would be generated."""
        return len(self.generate_grid(raster_height, raster_width))
