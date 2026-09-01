"""
GeoJSON export for vector features.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np

from .polygonizer import PolygonFeature


def export_geojson(
    features: List[PolygonFeature],
    output_path: str,
    raster_metadata: Optional[Dict] = None,
) -> str:
    """Export polygon features to GeoJSON file.

    Args:
        features: List of PolygonFeature objects.
        output_path: Path for the output .geojson file.
        raster_metadata: Optional CRS/transform metadata.

    Returns:
        Path to exported file.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    geojson_features = []
    for feat in features:
        try:
            geom_dict = feat.geometry.__geo_interface__ if hasattr(feat.geometry, '__geo_interface__') else {}
        except Exception:
            geom_dict = {}

        geojson_feature = {
            "type": "Feature",
            "geometry": geom_dict,
            "properties": {
                "class_id": feat.class_id,
                "class_label": feat.class_label,
                "area_pixels": feat.area_pixels,
                **feat.properties,
                "confidence": feat.confidence,
            },
        }
        geojson_features.append(geojson_feature)

    geojson = {
        "type": "FeatureCollection",
        "features": geojson_features,
    }

    # Add metadata
    if raster_metadata:
        geojson["metadata"] = {
            k: str(v) for k, v in raster_metadata.items()
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, default=str)

    return output_path


def export_predictions_geojson(
    prediction: np.ndarray,
    output_path: str,
    class_labels: Optional[Dict[int, str]] = None,
    raster_metadata: Optional[Dict] = None,
) -> str:
    """Quick export: mask -> GeoJSON in one call."""
    from .polygonizer import mask_to_polygons

    polygons = mask_to_polygons(prediction, class_labels=class_labels)
    return export_geojson(polygons, output_path, raster_metadata)
