"""
KML export for vector features.
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

from .polygonizer import PolygonFeature


def export_kml(
    features: List[PolygonFeature],
    output_path: str,
) -> str:
    """Export polygon features to KML file.

    Args:
        features: List of PolygonFeature objects.
        output_path: Path for the output .kml file.

    Returns:
        Path to exported file.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = ET.SubElement(kml, "Document")

    doc_name = ET.SubElement(document, "name")
    doc_name.text = "Disaster Assessment Vector Export"

    # Color scheme for classes
    class_colors = {
        0: "ff00c800",  # Green
        1: "ff00d7ff",  # Yellow
        2: "ff0082ff",  # Orange
        3: "ff0000ff",  # Red
        4: "ff00ff00",  # Light green
        5: "ff9b9b9b",  # Gray
    }

    for feat in features:
        placemark = ET.SubElement(document, "Placemark")

        name_el = ET.SubElement(placemark, "name")
        name_el.text = feat.class_label or f"Class {feat.class_id}"

        style_url = ET.SubElement(placemark, "styleUrl")
        style_url.text = f"#style_{feat.class_id}"

        # Add style if not defined
        style = ET.SubElement(document, "Style", id=f"style_{feat.class_id}")
        line_style = ET.SubElement(style, "LineStyle")
        color = ET.SubElement(line_style, "color")
        color.text = class_colors.get(feat.class_id, "ffffffff")
        poly_style = ET.SubElement(style, "PolyStyle")
        fill_color = ET.SubElement(poly_style, "color")
        fill_color.text = class_colors.get(feat.class_id, "ffffffff")

        # Geometry
        try:
            geom_dict = feat.geometry.__geo_interface__ if hasattr(feat.geometry, '__geo_interface__') else {}
            geom_type = geom_dict.get("type", "")
            coords = geom_dict.get("coordinates", [])

            if geom_type == "Polygon" and coords:
                polygon_el = ET.SubElement(placemark, "Polygon")
                outer = ET.SubElement(polygon_el, "outerBoundaryIs")
                ring = ET.SubElement(outer, "LinearRing")
                coords_el = ET.SubElement(ring, "coordinates")

                # KML coordinates: lon,lat[,alt]
                coord_strs = []
                for pt in coords[0]:
                    if len(pt) >= 2:
                        coord_strs.append(f"{pt[0]},{pt[1]}")
                    if len(pt) >= 3:
                        coord_strs[-1] += f",{pt[2]}"
                coords_el.text = " ".join(coord_strs)

        except Exception:
            pass

    # Write KML
    tree = ET.ElementTree(kml)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path
