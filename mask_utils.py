"""Convert xView2 label JSON (WKT building polygons) into a per-pixel label mask.

Label encoding (single-channel, integer per pixel):
    0  no-damage
    1  minor-damage
    2  major-damage
    3  destroyed
    4  background  (no building)
    255 un-classified / unknown subtype  -> ignore_index for the loss
"""

import json

import cv2
import numpy as np
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon

CLASS_MAP = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}

BACKGROUND_CLASS = 4
IGNORE_INDEX = 255
NUM_CLASSES = 5  # classes 0..4; 255 is excluded from the loss


def _polygon_contours(geom):
    """Yield int32 (N, 2) exterior-ring arrays for a Polygon or MultiPolygon."""
    if geom.is_empty:
        return
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for poly in polys:
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        yield np.array(poly.exterior.coords, dtype=np.int32)


def polygons_to_mask(label_json_path, image_size=(1024, 1024)):
    """Build an (H, W) uint8 label mask from an xView2 post-disaster label JSON.

    Un-classified polygons are painted first so that any overlapping classified
    building always wins; pixels touched by no polygon stay BACKGROUND_CLASS.
    """
    height, width = image_size
    mask = np.full((height, width), BACKGROUND_CLASS, dtype=np.uint8)

    with open(label_json_path, "r") as f:
        data = json.load(f)

    features = data.get("features", {}).get("xy", [])

    # (class_value, [contour, ...]) collected then drawn ignore-first.
    ignore_contours = []
    class_contours = []  # list of (class_value, contour)

    for obj in features:
        raw_wkt = obj.get("wkt")
        if not raw_wkt:
            continue
        geom = wkt.loads(raw_wkt)
        subtype = obj.get("properties", {}).get("subtype", "un-classified")
        class_value = CLASS_MAP.get(subtype, IGNORE_INDEX)
        for contour in _polygon_contours(geom):
            if contour.shape[0] < 3:
                continue
            if class_value == IGNORE_INDEX:
                ignore_contours.append(contour)
            else:
                class_contours.append((class_value, contour))

    for contour in ignore_contours:
        cv2.fillPoly(mask, [contour], IGNORE_INDEX)
    for class_value, contour in class_contours:
        cv2.fillPoly(mask, [contour], class_value)

    return mask
