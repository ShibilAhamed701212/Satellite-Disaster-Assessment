"""
Convert raster prediction masks into vector polygon features.

Uses rasterio.features for contour extraction and shapely for geometry operations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class PolygonFeature:
    """A single vector polygon feature from a raster mask."""
    geometry: object  # shapely geometry
    class_id: int
    class_label: str = ""
    confidence: float = 1.0
    area_pixels: int = 0
    properties: Dict = field(default_factory=dict)


def mask_to_polygons(
    mask: np.ndarray,
    class_labels: Optional[Dict[int, str]] = None,
    min_area_pixels: int = 4,
    simplify_tolerance: float = 0.0,
) -> List[PolygonFeature]:
    """Convert a 2D integer class mask to polygon features.

    Args:
        mask: 2D array of integer class labels (H, W).
        class_labels: Optional mapping of class_id -> label string.
        min_area_pixels: Minimum polygon area in pixels to keep.
        simplify_tolerance: Shapely simplification tolerance (0 = no simplification).

    Returns:
        List of PolygonFeature objects.
    """
    try:
        from rasterio import features
        from shapely.geometry import shape, mapping
        from shapely.ops import unary_union
    except ImportError:
        raise ImportError("rasterio and shapely are required for vectorization. "
                          "Install with: pip install rasterio shapely")

    features_list = []
    class_ids = np.unique(mask)

    for class_id in class_ids:
        if class_id == 0:
            continue  # Skip background

        # Create binary mask for this class
        class_mask = (mask == class_id).astype(np.uint8)

        # Extract contours/polygons using rasterio
        contours = features.shapes(
            class_mask,
            mask=class_mask,
            connectivity=8,
        )

        polygons_for_class = []
        for geom, value in contours:
            if value == 0:
                continue
            shp = shape(geom)

            if simplify_tolerance > 0:
                shp = shp.simplify(simplify_tolerance, preserve_topology=True)

            # Calculate area
            area = int(np.sum(class_mask[
                max(0, int(shp.bounds[1])):min(mask.shape[0], int(shp.bounds[3]) + 1),
                max(0, int(shp.bounds[0])):min(mask.shape[1], int(shp.bounds[2]) + 1)
            ])) if shp.is_valid else 0

            if area < min_area_pixels and shp.area < min_area_pixels:
                continue

            label = class_labels.get(class_id, f"class_{class_id}") if class_labels else f"class_{class_id}"

            features_list.append(PolygonFeature(
                geometry=shp,
                class_id=int(class_id),
                class_label=label,
                area_pixels=area,
                properties={
                    "class_id": int(class_id),
                    "class_label": label,
                    "area_pixels": area,
                },
            ))

    return features_list


def export_predictions(
    prediction: np.ndarray,
    raster_metadata: Optional[Dict] = None,
    output_path: str = "predictions.geojson",
    format: str = "geojson",
    class_labels: Optional[Dict[int, str]] = None,
    min_area_pixels: int = 4,
    simplify_tolerance: float = 0.0001,
) -> str:
    """Export prediction mask to geospatial vector format.

    Args:
        prediction: 2D integer class mask (H, W).
        raster_metadata: Optional dict with 'crs', 'transform' for georeferencing.
        output_path: Output file path.
        format: 'geojson', 'shapefile', or 'kml'.
        class_labels: Optional class_id -> label mapping.
        min_area_pixels: Minimum polygon area to keep.
        simplify_tolerance: Geometry simplification tolerance.

    Returns:
        Path to the exported file.
    """
    polygons = mask_to_polygons(
        prediction,
        class_labels=class_labels,
        min_area_pixels=min_area_pixels,
        simplify_tolerance=simplify_tolerance,
    )

    if format == "geojson":
        from .geojson_export import export_geojson
        return export_geojson(polygons, output_path, raster_metadata)
    elif format in ("shapefile", "shp"):
        from .shapefile_export import export_shapefile
        return export_shapefile(polygons, output_path, raster_metadata)
    elif format == "kml":
        from .kml_export import export_kml
        return export_kml(polygons, output_path)
    else:
        raise ValueError(f"Unsupported export format: {format}")
