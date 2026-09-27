"""Conformal prediction intervals."""

from __future__ import annotations

import numpy as np
import pytest

import hessboost
from conftest import FRAME_TYPES, frame, numpy_codes, reorder_colors
from hessboost import DMatrix, HessboostError
from hessboost.conformal import ConformalizedQuantile, SplitConformal


def heteroscedastic(rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(rows, 2))
    return x, 2 * x[:, 1] + rng.normal(0, 0.1 + np.abs(x[:, 0]))


def coverage(intervals: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((y >= intervals[:, 0]) & (y <= intervals[:, 1])))


def test_split_conformal_covers_new_rows() -> None:
    x, y = heteroscedastic(3000, 0)
    booster = hessboost.train({"max_depth": 3}, DMatrix(x[:1000], y[:1000]), 50)
    calibrated = SplitConformal.calibrate(booster, x[1000:2000], y[1000:2000], alpha=0.1)
    assert calibrated.alpha == 0.1
    assert calibrated.n_calibration == 1000
    intervals = calibrated.predict_interval(x[2000:])
    assert intervals.shape == (1000, 2)
    np.testing.assert_allclose(intervals.mean(axis=1), booster.predict(x[2000:]), atol=1e-5)
    np.testing.assert_allclose(
        intervals[:, 1] - intervals[:, 0], 2 * calibrated.half_width, rtol=1e-5
    )
    assert 0.86 < coverage(intervals, y[2000:]) < 0.94
    # A labelled DMatrix calibrates the same way.
    again = SplitConformal.calibrate(booster, DMatrix(x[1000:2000], y[1000:2000]), alpha=0.1)
    assert again.half_width == calibrated.half_width
    assert "half_width" in repr(again)


def test_tiny_calibration_sets_give_unbounded_intervals() -> None:
    x, y = heteroscedastic(200, 1)
    booster = hessboost.train({}, DMatrix(x[:100], y[:100]), 5)
    calibrated = SplitConformal.calibrate(booster, x[100:105], y[100:105], alpha=0.01)
    assert calibrated.half_width == np.inf
    assert np.all(np.isinf(calibrated.predict_interval(x[:3])))


def test_cqr_from_two_models_one_model_and_a_distribution() -> None:
    x, y = heteroscedastic(4000, 2)
    dtrain = DMatrix(x[:2000], y[:2000])
    calibration = (x[2000:3000], y[2000:3000])
    test_x, test_y = x[3000:], y[3000:]
    lower = hessboost.train({"objective": "reg:quantileerror", "quantile_alpha": 0.05}, dtrain, 100)
    upper = hessboost.train({"objective": "reg:quantileerror", "quantile_alpha": 0.95}, dtrain, 100)
    band = hessboost.train(
        {"objective": "reg:quantileerror", "quantile_alpha": [0.05, 0.95]}, dtrain, 100
    )
    dist = hessboost.train({"objective": "dist:normal", "max_depth": 2, "eta": 0.1}, dtrain, 200)
    calibrators = [
        ConformalizedQuantile.calibrate(lower, upper, *calibration, alpha=0.1),
        ConformalizedQuantile.calibrate_outputs(band, *calibration, alpha=0.1),
        ConformalizedQuantile.calibrate_distribution(dist, *calibration, alpha=0.1),
    ]
    for calibrated in calibrators:
        assert calibrated.n_calibration == 1000
        intervals = calibrated.predict_interval(test_x)
        assert 0.86 < coverage(intervals, test_y) < 0.94
        # Adaptive: wider where the noise is larger.
        width = intervals[:, 1] - intervals[:, 0]
        noisy = np.abs(test_x[:, 0]) > 0.5
        assert width[noisy].mean() > 1.5 * width[~noisy].mean()
    raw = band.predict(test_x)
    np.testing.assert_allclose(
        calibrators[1].predict_interval(test_x),
        raw + np.array([-1, 1]) * calibrators[1].correction,
        atol=1e-5,
    )
    swapped = ConformalizedQuantile.calibrate_outputs(band, *calibration, alpha=0.1, outputs=(1, 0))
    assert swapped.correction != calibrators[1].correction


def test_calibration_refusals() -> None:
    x, y = heteroscedastic(200, 3)
    booster = hessboost.train({}, DMatrix(x, y), 3)
    with pytest.raises(HessboostError, match="alpha"):
        SplitConformal.calibrate(booster, x, y, alpha=1.5)
    with pytest.raises(TypeError, match="alpha"):
        SplitConformal.calibrate(booster, x, y, alpha="0.1")  # ty: ignore[invalid-argument-type]
    with pytest.raises(HessboostError, match="labels"):
        SplitConformal.calibrate(booster, x, alpha=0.1)
    with pytest.raises(HessboostError, match="weights"):
        SplitConformal.calibrate(booster, DMatrix(x, y, weight=np.arange(1, 201)), alpha=0.1)
    with pytest.raises(HessboostError, match="dist"):
        ConformalizedQuantile.calibrate_distribution(booster, x, y, alpha=0.1)
    with pytest.raises(TypeError, match="calibrate"):
        SplitConformal()


def test_calibration_recodes_categories_and_refuses_mismatched_bands() -> None:
    df, y = frame(rows=900)
    fit, calibration, test = slice(0, 300), slice(300, 600), slice(600, 900)
    booster = hessboost.train({"max_depth": 3}, DMatrix(df[fit], y[fit]), 20)
    right = SplitConformal.calibrate(booster, df[calibration], y[calibration], alpha=0.1)
    swapped = SplitConformal.calibrate(
        booster, reorder_colors(df[calibration]), y[calibration], alpha=0.1
    )
    assert swapped.half_width == right.half_width
    np.testing.assert_array_equal(
        swapped.predict_interval(reorder_colors(df[test])), right.predict_interval(df[test])
    )
    # A band's two models are evaluated on one matrix, so they must share
    # their categories.
    quantile = {"objective": "reg:quantileerror", "max_depth": 3}
    lower = hessboost.train({**quantile, "quantile_alpha": 0.05}, DMatrix(df[fit], y[fit]), 20)
    upper = hessboost.train(
        {**quantile, "quantile_alpha": 0.95}, DMatrix(reorder_colors(df[fit]), y[fit]), 20
    )
    with pytest.raises(HessboostError, match="upper: the categories of feature 'color' differ"):
        ConformalizedQuantile.calibrate(lower, upper, df[calibration], y[calibration], alpha=0.1)
    # A lower model trained on numpy codes (here upper's) records no
    # categories, so frames are re-coded to upper's and checked against each
    # model.
    numpy_lower = hessboost.train(
        {**quantile, "quantile_alpha": 0.05},
        DMatrix(numpy_codes(reorder_colors(df[fit])), y[fit], feature_types=FRAME_TYPES),
        20,
    )
    band = ConformalizedQuantile.calibrate(
        numpy_lower, upper, reorder_colors(df[calibration]), y[calibration], alpha=0.1
    )
    again = ConformalizedQuantile.calibrate(
        numpy_lower, upper, df[calibration], y[calibration], alpha=0.1
    )
    assert band.correction == again.correction
    np.testing.assert_array_equal(
        band.predict_interval(df[test]), band.predict_interval(reorder_colors(df[test]))
    )
    with pytest.raises(HessboostError, match="the data: the categories of feature 'color'"):
        band.predict_interval(DMatrix(df[test]))
