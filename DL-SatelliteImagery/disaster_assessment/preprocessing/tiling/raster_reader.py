"""
Efficient raster reader for streaming tiles from large GeoTIFF files.

Avoids loading the entire raster into memory.
Uses rasterio for GeoTIFF I/O with fallback to PIL for standard images.
"""

import os
from typing import Optional, Tuple

import numpy as np

from .tile_generator import TileInfo


class RasterReader:
    """Read tiles from a raster file on disk.

    Supports GeoTIFF via rasterio and standard images via PIL.
    """

    def __init__(self, file_path: str):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Raster file not found: {file_path}")
        self.file_path = file_path
        self._rasterio_src = None

    def open(self):
        """Open the raster file for reading."""
        try:
            import rasterio
            self._rasterio_src = rasterio.open(self.file_path)
        except ImportError:
            self._rasterio_src = None
        except Exception:
            self._rasterio_src = None

    def close(self):
        """Close the raster file."""
        if self._rasterio_src is not None:
            self._rasterio_src.close()
            self._rasterio_src = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def is_georeferenced(self) -> bool:
        if self._rasterio_src is not None and self._rasterio_src.crs is not None:
            return True
        return False

    @property
    def crs(self) -> Optional[str]:
        if self._rasterio_src is not None and self._rasterio_src.crs is not None:
            return self._rasterio_src.crs.to_string()
        return None

    @property
    def transform(self):
        if self._rasterio_src is not None:
            return self._rasterio_src.transform
        return None

    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        if self._rasterio_src is not None:
            b = self._rasterio_src.bounds
            return (b.left, b.bottom, b.right, b.top)
        return None

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return (bands, height, width)."""
        if self._rasterio_src is not None:
            return (self._rasterio_src.count, self._rasterio_src.height, self._rasterio_src.width)
        return (0, 0, 0)

    @property
    def height(self) -> int:
        return self.shape[1]

    @property
    def width(self) -> int:
        return self.shape[2]

    @property
    def num_bands(self) -> int:
        return self.shape[0]

    def read_tile(self, tile_info: TileInfo, num_bands: Optional[int] = None) -> np.ndarray:
        """Read a single tile from the raster.

        Args:
            tile_info: Tile metadata with offset and dimensions.
            num_bands: Number of bands to read (None = all).

        Returns:
            Array of shape (bands, height, width) or (height, width) for single band.
        """
        if self._rasterio_src is not None:
            return self._read_tile_rasterio(tile_info, num_bands)
        return self._read_tile_pil(tile_info)

    def _read_tile_rasterio(self, tile_info: TileInfo, num_bands: Optional[int] = None) -> np.ndarray:
        """Read tile using rasterio windowed reading."""
        from rasterio.windows import Window

        count = num_bands or self._rasterio_src.count
        window = Window(
            col_off=tile_info.x_offset,
            row_off=tile_info.y_offset,
            width=tile_info.width,
            height=tile_info.height,
        )
        data = self._rasterio_src.read(window=window, indexes=list(range(1, count + 1)))
        return data  # (bands, H, W)

    def _read_tile_pil(self, tile_info: TileInfo) -> np.ndarray:
        """Read tile using PIL (for non-GeoTIFF images)."""
        from PIL import Image

        img = Image.open(self.file_path)
        img_array = np.array(img)

        if img_array.ndim == 2:
            img_array = img_array[:, :, np.newaxis]

        # Crop to tile region
        y1 = tile_info.y_offset
        y2 = y1 + tile_info.height
        x1 = tile_info.x_offset
        x2 = x1 + tile_info.width

        tile = img_array[y1:y2, x1:x2]

        # Convert to (C, H, W)
        if tile.ndim == 3:
            tile = tile.transpose(2, 0, 1)
        else:
            tile = tile[np.newaxis, :, :]

        return tile

    def read_full(self) -> np.ndarray:
        """Read the entire raster (use only for small files!).

        Returns:
            Array of shape (bands, height, width).
        """
        if self._rasterio_src is not None:
            return self._rasterio_src.read()
        else:
            from PIL import Image
            img = Image.open(self.file_path)
            arr = np.array(img)
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            return arr.transpose(2, 0, 1)
