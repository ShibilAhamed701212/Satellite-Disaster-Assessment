"""
Overlap-aware blending engine for reconstructing full predictions from tiled outputs.

Supports uniform averaging, Gaussian weighting, and linear weighting.
Handles edge tiles and avoids division by zero.
"""

from typing import Dict, Optional

import numpy as np

from .tile_config import TilingConfig
from .tile_generator import TileInfo


class PredictionBlender:
    """Accumulate tiled predictions and blend them into a full-resolution output.

    Usage:
        blender = PredictionBlender(TilingConfig(tile_size=512, overlap=0.25))
        blender.initialize(canvas_height, canvas_width, num_channels)

        for tile_info, prediction in tile_predictions:
            blender.accumulate(tile_info, prediction)

        result = blender.blend()
    """

    def __init__(self, config: Optional[TilingConfig] = None):
        self.config = config or TilingConfig()
        self._accumulator: Optional[np.ndarray] = None
        self._weight_map: Optional[np.ndarray] = None
        self._num_accumulated: int = 0

    def initialize(self, height: int, width: int, num_channels: int = 1):
        """Initialize accumulation buffers.

        Args:
            height: Full raster height.
            width: Full raster width.
            num_channels: Number of output channels (1 for binary, N for multiclass).
        """
        self._accumulator = np.zeros((num_channels, height, width), dtype=np.float64)
        self._weight_map = np.zeros((height, width), dtype=np.float64)
        self._num_accumulated = 0

    @property
    def num_accumulated(self) -> int:
        return self._num_accumulated

    def accumulate(self, tile_info: TileInfo, prediction: np.ndarray, weight_mask: Optional[np.ndarray] = None):
        """Add a tile's prediction to the accumulator.

        Args:
            tile_info: Tile metadata with offset information.
            prediction: Predicted output (C, H, W) or (H, W).
            weight_mask: Optional per-pixel weight mask for this tile.
        """
        if self._accumulator is None:
            raise RuntimeError("Call initialize() before accumulate()")

        y_off = tile_info.y_offset
        x_off = tile_info.x_offset
        h = tile_info.height
        w = tile_info.width

        # Ensure prediction is (C, H, W)
        if prediction.ndim == 2:
            prediction = prediction[np.newaxis, :, :]

        num_channels = prediction.shape[0]

        # Generate weight mask if not provided
        if weight_mask is None:
            weight_mask = self._generate_weight_mask(tile_info)

        # Clip prediction to valid range
        prediction = np.clip(prediction[:, :h, :w], 0.0, None)

        # Accumulate weighted predictions
        for c in range(min(num_channels, self._accumulator.shape[0])):
            self._accumulator[c, y_off:y_off+h, x_off:x_off+w] += prediction[c, :h, :w].astype(np.float64) * weight_mask[:h, :w]

        self._weight_map[y_off:y_off+h, x_off:x_off+w] += weight_mask[:h, :w].astype(np.float64)
        self._num_accumulated += 1

    def blend(self) -> np.ndarray:
        """Normalize accumulated predictions by accumulated weights.

        Returns:
            Blended prediction array (C, H, W) in float32.
        """
        if self._accumulator is None or self._num_accumulated == 0:
            return np.array([])

        # Avoid division by zero
        safe_weights = np.maximum(self._weight_map, 1e-8)

        # Broadcast weight map to channels
        result = np.zeros_like(self._accumulator, dtype=np.float32)
        for c in range(self._accumulator.shape[0]):
            result[c] = (self._accumulator[c] / safe_weights).astype(np.float32)

        return result

    def blend_mask(self) -> np.ndarray:
        """For binary/multiclass predictions, blend and return class indices.

        Returns:
            Integer class mask (H, W).
        """
        result = self.blend()
        if result.size == 0:
            return np.array([], dtype=np.uint8)

        if result.shape[0] == 1:
            # Binary: threshold at 0.5
            return (result[0] >= 0.5).astype(np.uint8)
        else:
            # Multiclass: argmax
            return np.argmax(result, axis=0).astype(np.uint8)

    def _generate_weight_mask(self, tile_info: TileInfo) -> np.ndarray:
        """Generate a weight mask for the given tile based on blend mode."""
        h, w = tile_info.height, tile_info.width

        if self.config.blend_mode == "uniform":
            return np.ones((h, w), dtype=np.float32)

        ts = self.config.tile_size

        if self.config.blend_mode == "gaussian":
            sigma = ts / 4.0
            y_coords = np.arange(h, dtype=np.float32)
            x_coords = np.arange(w, dtype=np.float32)
            y_center = h / 2.0
            x_center = w / 2.0

            y_w = np.exp(-0.5 * ((y_coords - y_center) / sigma) ** 2)
            x_w = np.exp(-0.5 * ((x_coords - x_center) / sigma) ** 2)
            return np.outer(y_w, x_w).astype(np.float32)

        elif self.config.blend_mode == "linear":
            y_coords = np.arange(h, dtype=np.float32)
            x_coords = np.arange(w, dtype=np.float32)
            y_center = h / 2.0
            x_center = w / 2.0

            y_w = np.clip(1.0 - np.abs(y_coords - y_center) / max(y_center, 1), 0.01, 1.0)
            x_w = np.clip(1.0 - np.abs(x_coords - x_center) / max(x_center, 1), 0.01, 1.0)
            return np.outer(y_w, x_w).astype(np.float32)

        return np.ones((h, w), dtype=np.float32)

    def reset(self):
        """Reset the blender for a new raster."""
        self._accumulator = None
        self._weight_map = None
        self._num_accumulated = 0
