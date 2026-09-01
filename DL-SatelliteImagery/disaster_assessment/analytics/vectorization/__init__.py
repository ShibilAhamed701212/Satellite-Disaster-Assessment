"""
Raster-to-vector engine for converting prediction masks to geospatial polygons.

Supports flood extent, building damage, land-cover, and change detection regions.
"""

from .polygonizer import mask_to_polygons, PolygonFeature
from .geojson_export import export_geojson, export_predictions_geojson
from .shapefile_export import export_shapefile
from .kml_export import export_kml

__all__ = [
    "mask_to_polygons",
    "PolygonFeature",
    "export_geojson",
    "export_predictions_geojson",
    "export_shapefile",
    "export_kml",
]
