"""Tests for src/analysis/uncertainty.py (split-conformal intervals)."""

import json
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "analysis"))

from uncertainty import (conformal_half_width, coverage, interval,  # noqa: E402
                         max_tanimoto, morgan_fp, split_cal_eval)


def test_interval_width_positive_and_symmetric():
    hw = conformal_half_width(np.array([3.0, -1.0, 7.0, 2.0, -5.0, 4.0]))
    assert hw > 0
    lo, hi = interval(142.0, hw)
    assert hi > lo
    assert math.isclose(142.0 - lo, hi - 142.0)


def test_conformal_quantile_uses_finite_sample_rank():
    # n = 9: rank ceil(10 * 0.9) = 9 -> the largest of the 9 residuals
    assert conformal_half_width(np.arange(1, 10), alpha=0.1) == 9
    # n = 19: rank ceil(20 * 0.9) = 18 -> the 18th smallest
    assert conformal_half_width(np.arange(1, 20), alpha=0.1) == 18
    # too few points for a 90% guarantee -> infinite width, never a fake number
    assert conformal_half_width(np.arange(1, 5), alpha=0.1) == float("inf")


def test_coverage_on_toy_example():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([0.5, -1.0, 2.0, 3.0])
    assert coverage(y_true, y_pred, half_width=1.0) == 0.5   # |err| 0.5 and 1.0 are inside
    assert coverage(y_true, y_pred, half_width=10.0) == 1.0
    assert coverage(y_true, y_pred, half_width=0.1) == 0.0


def test_conformal_interval_covers_about_90_percent_on_synthetic_data():
    rng = np.random.default_rng(0)
    y = rng.normal(size=4000)
    pred = y + rng.normal(scale=2.0, size=4000)
    cal, ev = split_cal_eval(len(y), seed=1)
    hw = conformal_half_width(np.abs(y[cal] - pred[cal]))
    assert 0.87 <= coverage(y[ev], pred[ev], hw) <= 0.93


def test_split_is_disjoint_and_complete():
    cal, ev = split_cal_eval(737, seed=0)
    assert len(set(cal) & set(ev)) == 0
    assert sorted(np.concatenate([cal, ev])) == list(range(737))


def test_tanimoto_identical_is_one_and_invalid_is_none():
    fp = morgan_fp("CC(c1ccccc1)C")
    assert max_tanimoto(fp, [morgan_fp("CC"), fp]) == 1.0
    assert morgan_fp("not a smiles") is None


def test_saved_uncertainty_results_are_consistent():
    path = os.path.join(ROOT, "results", "uncertainty.json")
    if not os.path.exists(path):
        pytest.skip("run src/analysis/uncertainty.py first")
    with open(path) as f:
        res = json.load(f)
    assert res["half_width_C"] > 0
    assert res["n_cal"] + res["n_eval"] == 737
    assert 0.0 <= res["empirical_coverage_eval"] <= 1.0
