"""Metric code in evaluate.py, checked against hand-computed answers."""

import json
import os

import numpy as np
import pytest

from evaluate import confusion, summarize
from mask_utils import BACKGROUND_CLASS, IGNORE_INDEX, NUM_CLASSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 12 building pixels (classes 0-3) and 2 background pixels (class 4).
GT = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4])
PRED = np.array([0, 0, 0, 1, 1, 1, 0, 4, 2, 3, 3, 3, 4, 0])


def test_confusion_counts_and_excludes_ignore():
    t = np.array([0, 1, 4, IGNORE_INDEX, IGNORE_INDEX])
    p = np.array([0, 2, 4, 3, 0])
    conf = confusion(t, p)
    assert conf.shape == (NUM_CLASSES, NUM_CLASSES)
    assert conf.sum() == 3
    assert conf[0, 0] == 1 and conf[1, 2] == 1 and conf[4, 4] == 1


def test_perfect_prediction():
    r = summarize(confusion(GT, GT))
    assert r["building_damage_macro_f1"] == pytest.approx(1.0)
    assert r["building_damage_acc"] == pytest.approx(1.0)
    assert r["pixel_acc"] == pytest.approx(1.0)
    assert r["mean_iou"] == pytest.approx(1.0)


def test_hand_computed_damage_f1():
    # Precision is computed among ground-truth building pixels only, so the
    # background pixel predicted as no-damage (last entry) does not count.
    #   no-damage: P=3/4 R=3/4 -> F1 3/4
    #   minor:     P=2/3 R=2/4 -> F1 4/7
    #   major:     P=1/1 R=1/2 -> F1 2/3
    #   destroyed: P=2/3 R=2/2 -> F1 4/5
    r = summarize(confusion(GT, PRED))
    f1 = r["building_damage_f1"]
    assert f1["no-damage"] == pytest.approx(3 / 4)
    assert f1["minor"] == pytest.approx(4 / 7)
    assert f1["major"] == pytest.approx(2 / 3)
    assert f1["destroyed"] == pytest.approx(4 / 5)
    assert r["building_damage_macro_f1"] == pytest.approx((3 / 4 + 4 / 7 + 2 / 3 + 4 / 5) / 4)
    assert r["building_damage_acc"] == pytest.approx(8 / 12)
    assert r["pixel_acc"] == pytest.approx(9 / 14)
    # background: TP=1, FP=1 (minor px predicted bg), FN=1 (bg px predicted no-damage)
    assert r["per_class_iou"]["background"] == pytest.approx(1 / 3)


def test_ignore_pixels_do_not_change_metrics():
    gt = np.concatenate([GT, np.full(50, IGNORE_INDEX)])
    pred = np.concatenate([PRED, np.random.default_rng(0).integers(0, 5, 50)])
    assert summarize(confusion(gt, pred)) == summarize(confusion(GT, PRED))


def test_all_background_baseline_scores_zero_damage_f1():
    r = summarize(confusion(GT, np.full_like(GT, BACKGROUND_CLASS)))
    assert r["building_damage_macro_f1"] == 0.0
    assert r["building_damage_acc"] == 0.0
    assert r["pixel_acc"] == pytest.approx(2 / 14)


def test_oracle_no_damage_baseline_closed_form():
    # Every building predicted no-damage: P0 = n0/N, R0 = 1 -> F1_0 = 2*n0/(n0+N),
    # other classes 0. Here n0 = 4, N = 12 -> F1_0 = 0.5, macro = 0.125.
    oracle = np.where(GT <= 3, 0, BACKGROUND_CLASS)
    r = summarize(confusion(GT, oracle))
    assert r["building_damage_f1"]["no-damage"] == pytest.approx(0.5)
    assert r["building_damage_macro_f1"] == pytest.approx(0.125)
    assert r["per_class_iou"]["background"] == pytest.approx(1.0)


def test_absent_class_counts_as_zero_in_macro_f1():
    gt = np.array([0, 0, 1, 1])
    r = summarize(confusion(gt, gt))
    assert r["building_damage_f1"]["major"] == 0.0
    assert r["building_damage_macro_f1"] == pytest.approx(0.5)
    assert np.isnan(r["per_class_iou"]["major"])  # no support, no predictions
    assert r["mean_iou"] == pytest.approx(1.0)  # NaN classes skipped


def test_committed_metrics_are_internally_consistent():
    """docs/metrics.json backs the README table; guard against drift."""
    with open(os.path.join(ROOT, "docs", "metrics.json")) as f:
        results = json.load(f)["results"]
    for r in results.values():
        assert r["building_damage_macro_f1"] == pytest.approx(
            np.mean(list(r["building_damage_f1"].values()))
        )
    assert round(results["UNet (ours)"]["building_damage_macro_f1"], 3) == 0.475
    assert round(results["oracle footprints + no-damage"]["building_damage_macro_f1"], 3) == 0.185
    assert round(results["UNet (ours)"]["mean_iou"], 3) == 0.337
