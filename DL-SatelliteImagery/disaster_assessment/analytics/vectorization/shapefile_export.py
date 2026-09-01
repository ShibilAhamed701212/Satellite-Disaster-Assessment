"""
ESRI Shapefile export for vector features.
Requires geopandas and fiona/pyshp.
"""

import os
from typing import Dict, List, Optional

from .polygonizer import PolygonFeature


def export_shapefile(
    features: List[PolygonFeature],
    output_path: str,
    raster_metadata: Optional[Dict] = None,
) -> str:
    """Export polygon features to ESRI Shapefile.

    Args:
        features: List of PolygonFeature objects.
        output_path: Path for the output .shp file.
        raster_metadata: Optional CRS/transform metadata.

    Returns:
        Path to exported file.
    """
    try:
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import mapping
    except ImportError:
        raise ImportError("geopandas is required for Shapefile export. "
                          "Install with: pip install geopandas")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    if not features:
        # Create empty shapefile
        gdf = gpd.GeoDataFrame(columns=["geometry", "class_id", "class_label", "confidence", "area_pixels"], crs=None)
        gdf.to_file(output_path, driver="ESRI Shapefile")
        return output_path

    records = []
    for feat in features:
        records.append({
            "geometry": feat.geometry,
            "class_id": feat.class_id,
            "class_label": feat.class_label,
            "confidence": feat.confidence,
            "area_pixels": feat.area_pixels,
        })

    gdf = gpd.GeoDataFrame(records, geometry="geometry")

    # Set CRS if available
    crs_str = None
    if raster_metadata:
        crs_str = raster_metadata.get("crs") or raster_metadata.get("crs_name")
    if crs_str:
        try:
            gdf.crs = crs_str
        except Exception:
            pass

    gdf.to_file(output_path, driver="ESRI Shapefile")
    return output_path
