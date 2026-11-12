import random

import numpy as np
import torch

from dataset import XView2Dataset, list_ids, make_splits
from mask_utils import BACKGROUND_CLASS


def test_list_ids_requires_pre_post_and_label(xbd_dir):
    ids = list_ids(xbd_dir["images"], xbd_dir["labels"])
    assert ids == sorted(xbd_dir["ids"])  # the tile with no JSON is dropped


def test_list_ids_filters_by_event(xbd_dir):
    ids = list_ids(xbd_dir["images"], xbd_dir["labels"], events=["santa-rosa-wildfire"])
    assert ids == ["santa-rosa-wildfire_00000001", "santa-rosa-wildfire_00000002"]


def test_make_splits_is_deterministic_and_disjoint(xbd_dir):
    a = make_splits(xbd_dir["images"], xbd_dir["labels"], val_frac=0.25, seed=42)
    b = make_splits(xbd_dir["images"], xbd_dir["labels"], val_frac=0.25, seed=42)
    assert a == b
    train_ids, val_ids = a
    assert len(val_ids) == 1 and len(train_ids) == 3
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids) | set(val_ids) == set(xbd_dir["ids"])


def test_item_shapes_dtypes_and_ranges(xbd_dir):
    ds = XView2Dataset(xbd_dir["images"], xbd_dir["labels"], xbd_dir["ids"], img_size=64)
    assert len(ds) == 4
    image, mask = ds[0]

    assert image.shape == (6, 64, 64)
    assert image.dtype == torch.float32
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert mask.shape == (64, 64)
    assert mask.dtype == torch.int64


def test_channels_are_pre_then_post_and_mask_is_resized(xbd_dir):
    ds = XView2Dataset(xbd_dir["images"], xbd_dir["labels"], xbd_dir["ids"], img_size=64)
    image, mask = ds[0]

    assert torch.all(image[:3] == 0)  # pre image is black
    assert torch.all(image[3:, :32, :32] == 1)  # post image is white top-left
    assert torch.all(image[3:, 32:, :] == 0)

    # Polygon covers label pixels 0..511 of 1024 -> rows/cols 0..31 at 64x64.
    expected = torch.full((64, 64), BACKGROUND_CLASS, dtype=torch.int64)
    expected[:32, :32] = 3
    assert torch.equal(mask, expected)


def test_images_are_resized_to_img_size(xbd_dir):
    ds = XView2Dataset(xbd_dir["images"], xbd_dir["labels"], xbd_dir["ids"], img_size=32)
    image, mask = ds[0]
    assert image.shape == (6, 32, 32)
    assert mask.shape == (32, 32)


def test_augmentation_flips_image_and_mask_together(xbd_dir):
    ds = XView2Dataset(
        xbd_dir["images"], xbd_dir["labels"], xbd_dir["ids"], img_size=64, augment=True
    )
    random.seed(0)
    seen_corners = set()
    for _ in range(20):
        image, mask = ds[0]
        building = mask != BACKGROUND_CLASS
        bright = image[3] > 0.5
        assert torch.equal(building, bright)
        assert torch.all(image[:3] == 0)  # pre/post channel order preserved
        assert building.sum() == 32 * 32
        rows, cols = np.nonzero(building.numpy())
        seen_corners.add((int(rows.min()), int(cols.min())))
    assert len(seen_corners) > 1  # flips actually happened
