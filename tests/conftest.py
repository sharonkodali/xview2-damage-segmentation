"""Shared helpers: tiny synthetic xBD-style label JSON and image fixtures."""

import json

import cv2
import numpy as np
import pytest


def square_wkt(x0, y0, x1, y1):
    return f"POLYGON (({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"


def write_label_json(path, features):
    """Write an xView2-style label file. ``features`` is a list of (wkt, subtype)."""
    xy = []
    for wkt_str, subtype in features:
        obj = {"wkt": wkt_str, "properties": {"feature_type": "building"}}
        if subtype is not None:
            obj["properties"]["subtype"] = subtype
        xy.append(obj)
    with open(path, "w") as f:
        json.dump({"features": {"lng_lat": [], "xy": xy}, "metadata": {}}, f)
    return path


@pytest.fixture
def label_file(tmp_path):
    """Factory: label_file(features) -> path to a written label JSON."""
    counter = iter(range(10_000))

    def make(features):
        return write_label_json(tmp_path / f"label_{next(counter)}.json", features)

    return make


@pytest.fixture
def xbd_dir(tmp_path):
    """A tiny xBD-like tree: images/<id>_{pre,post}_disaster.png + labels/<id>_post_disaster.json.

    Each tile is 64x64. The pre image is black; the post image is white in the
    top-left quadrant, which is also where the single building polygon sits
    (in xBD's 1024x1024 label coordinates), so image/mask alignment can be
    checked after augmentation.
    """
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()

    ids = [
        "hurricane-michael_00000001",
        "hurricane-michael_00000002",
        "santa-rosa-wildfire_00000001",
        "santa-rosa-wildfire_00000002",
    ]
    pre = np.zeros((64, 64, 3), np.uint8)
    post = np.zeros((64, 64, 3), np.uint8)
    post[:32, :32] = 255
    for image_id in ids:
        cv2.imwrite(str(images / f"{image_id}_pre_disaster.png"), pre)
        cv2.imwrite(str(images / f"{image_id}_post_disaster.png"), post)
        write_label_json(
            labels / f"{image_id}_post_disaster.json",
            [(square_wkt(0, 0, 511, 511), "destroyed")],
        )

    # A tile with no label JSON must be skipped by list_ids.
    cv2.imwrite(str(images / "hurricane-michael_00000099_pre_disaster.png"), pre)
    cv2.imwrite(str(images / "hurricane-michael_00000099_post_disaster.png"), post)

    return {"images": str(images), "labels": str(labels), "ids": ids}
