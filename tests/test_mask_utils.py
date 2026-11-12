import numpy as np
import pytest

from conftest import square_wkt
from mask_utils import (
    BACKGROUND_CLASS,
    CLASS_MAP,
    IGNORE_INDEX,
    NUM_CLASSES,
    polygons_to_mask,
)


def test_label_encoding_constants():
    assert CLASS_MAP == {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}
    assert BACKGROUND_CLASS == 4
    assert NUM_CLASSES == 5
    assert IGNORE_INDEX == 255
    assert IGNORE_INDEX >= NUM_CLASSES  # never collides with a real class


def test_no_features_gives_all_background(label_file):
    mask = polygons_to_mask(label_file([]), image_size=(32, 48))
    assert mask.shape == (32, 48)
    assert mask.dtype == np.uint8
    assert (mask == BACKGROUND_CLASS).all()


def test_square_is_rasterised_inclusive_of_boundary(label_file):
    # fillPoly fills the closed polygon, so x,y in [10, 19] -> a 10x10 block.
    mask = polygons_to_mask(label_file([(square_wkt(10, 10, 19, 19), "no-damage")]), (64, 64))
    expected = np.full((64, 64), BACKGROUND_CLASS, np.uint8)
    expected[10:20, 10:20] = 0
    np.testing.assert_array_equal(mask, expected)


def test_wkt_x_is_column_and_y_is_row(label_file):
    # A wide, short rectangle: x spans 0..29, y spans 0..4.
    mask = polygons_to_mask(label_file([(square_wkt(0, 0, 29, 4), "destroyed")]), (64, 64))
    rows, cols = np.nonzero(mask == 3)
    assert rows.max() == 4
    assert cols.max() == 29


@pytest.mark.parametrize("subtype, value", list(CLASS_MAP.items()))
def test_each_damage_subtype_maps_to_its_class(label_file, subtype, value):
    mask = polygons_to_mask(label_file([(square_wkt(2, 2, 5, 5), subtype)]), (16, 16))
    assert mask[3, 3] == value
    assert set(np.unique(mask)) == {value, BACKGROUND_CLASS}


@pytest.mark.parametrize("subtype", ["un-classified", "something-new", None])
def test_unknown_or_missing_subtype_is_ignored(label_file, subtype):
    mask = polygons_to_mask(label_file([(square_wkt(2, 2, 5, 5), subtype)]), (16, 16))
    assert mask[3, 3] == IGNORE_INDEX


@pytest.mark.parametrize("order", ["ignore_first", "ignore_last"])
def test_classified_building_wins_over_overlapping_unclassified(label_file, order):
    feats = [
        (square_wkt(0, 0, 9, 9), "un-classified"),
        (square_wkt(5, 5, 14, 14), "major-damage"),
    ]
    if order == "ignore_last":
        feats = feats[::-1]
    mask = polygons_to_mask(label_file(feats), (16, 16))
    assert mask[7, 7] == 2  # overlap region -> classified
    assert mask[1, 1] == IGNORE_INDEX  # un-classified only
    assert mask[13, 13] == 2  # classified only
    assert mask[15, 0] == BACKGROUND_CLASS


def test_multipolygon_fills_every_part(label_file):
    multi = "MULTIPOLYGON (((0 0, 3 0, 3 3, 0 3, 0 0)), ((10 10, 13 10, 13 13, 10 13, 10 10)))"
    mask = polygons_to_mask(label_file([(multi, "minor-damage")]), (16, 16))
    assert (mask[0:4, 0:4] == 1).all()
    assert (mask[10:14, 10:14] == 1).all()
    assert (mask == 1).sum() == 32


def test_features_without_wkt_are_skipped(label_file):
    mask = polygons_to_mask(label_file([("", "destroyed")]), (8, 8))
    assert (mask == BACKGROUND_CLASS).all()


def test_default_size_matches_xbd_tiles(label_file):
    mask = polygons_to_mask(label_file([]))
    assert mask.shape == (1024, 1024)
