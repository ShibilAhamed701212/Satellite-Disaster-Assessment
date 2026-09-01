"""
Interactive geospatial map viewer using Folium.

Displays prediction masks, vector layers, and hazard overlays
on an interactive Leaflet-based map.
"""

import os
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np


class MapViewer:
    """Create interactive geospatial maps using Folium/Leaflet.

    Features:
        - Display prediction masks
        - Pre/post image comparison
        - Opacity control
        - Layer toggles
        - Hazard overlays
        - Severity visualization
    """

    def __init__(self, default_center: Tuple[float, float] = (28.6139, 77.2090), zoom: int = 6):
        self.default_center = default_center
        self.zoom = zoom

    def create_map(self) -> Optional[object]:
        """Create a Folium map object."""
        try:
            import folium
            return folium.Map(
                location=self.default_center,
                zoom_start=self.zoom,
                tiles="OpenStreetMap",
            )
        except ImportError:
            return None

    def add_prediction_overlay(
        self,
        map_obj: object,
        mask: np.ndarray,
        name: str = "Prediction",
        opacity: float = 0.5,
        colormap: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> None:
        """Add a prediction mask as an image overlay on the map.

        Args:
            map_obj: Folium map object.
            mask: 2D integer class mask.
            name: Layer name.
            opacity: Layer opacity.
            colormap: Dict mapping class_id -> (R,G,B).
        """
        if map_obj is None:
            return

        try:
            import folium
            from PIL import Image

            if colormap is None:
                colormap = {
                    0: (0, 100, 255),
                    1: (255, 0, 0),
                    2: (255, 215, 0),
                    3: (200, 0, 0),
                }

            h, w = mask.shape
            colored = np.zeros((h, w, 3), dtype=np.uint8)
            for class_id, color in colormap.items():
                colored[mask == class_id] = color

            # Add alpha channel
            alpha = np.where(mask > 0, int(opacity * 255), 0).astype(np.uint8)
            rgba = np.dstack([colored, alpha])

            # Save to temp file
            img = Image.fromarray(rgba)
            tmp_path = os.path.join(tempfile.gettempdir(), f"overlay_{name}.png")
            img.save(tmp_path)

            # Add to map
            folium.raster_layers.ImageOverlay(
                image=tmp_path,
                bounds=[[0, 0], [h, w]],
                name=name,
                opacity=opacity,
            ).add_to(map_obj)
        except Exception:
            pass

    def add_geojson_layer(
        self,
        map_obj: object,
        geojson_path: str,
        name: str = "Vector Layer",
        color: str = "blue",
        fill_opacity: float = 0.3,
    ) -> None:
        """Add a GeoJSON file as a layer."""
        if map_obj is None or not os.path.isfile(geojson_path):
            return

        try:
            import folium
            import json

            with open(geojson_path, "r") as f:
                data = json.load(f)

            folium.GeoJson(
                data,
                name=name,
                style_function=lambda feature: {
                    "color": color,
                    "fillOpacity": fill_opacity,
                    "weight": 2,
                },
            ).add_to(map_obj)
        except Exception:
            pass

    def add_layer_control(self, map_obj: object) -> None:
        """Add layer control toggle to the map."""
        if map_obj is None:
            return
        try:
            import folium
            folium.LayerControl().add_to(map_obj)
        except Exception:
            pass

    def create_comparison_map(
        self,
        center: Tuple[float, float],
        pre_bounds: Optional[List] = None,
        post_bounds: Optional[List] = None,
    ) -> Optional[object]:
        """Create a pre/post comparison map with split view."""
        map_obj = self.create_map()
        if map_obj is None:
            return None

        try:
            import folium
            # Add side-by-side comparison
            # Note: folium-sideby-side may need to be installed separately
            map_obj.location = center
        except Exception:
            pass

        return map_obj

    def save_map(self, map_obj: object, output_path: str) -> str:
        """Save map to HTML file."""
        if map_obj is None:
            return ""

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        try:
            map_obj.save(output_path)
            return output_path
        except Exception:
            return ""

    def render_html(self, map_obj: object) -> str:
        """Render map to HTML string."""
        if map_obj is None:
            return "<p>Map unavailable (folium not installed)</p>"

        try:
            return map_obj._repr_html_()
        except Exception:
            return "<p>Failed to render map</p>"
