"""Conformal prediction intervals."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import hessboost
from conftest import FRAME_TYPES, frame, numpy_codes, reorder_colors
from hessboost import DMatrix, HessboostError
from hessboost.conformal import (
    CalibrationForecast,
    ConformalizedQuantile,
    OnlineCalibrator,
    SplitConformal,
)


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


def test_online_feedback_uses_issued_threshold_and_covariates() -> None:
    cal = OnlineCalibrator(0.5, 1.0, lags=1, covariates=1, coefficient_bound=10.0)
    features = np.array([2.0])
    first = cal.predict(features)
    features[0] = 99.0
    second = cal.predict([3.0])
    assert (first.index, second.index, cal.pending) == (0, 1, 2)
    assert (first.threshold, second.threshold) == (0.0, 0.0)
    cal.update(first, 1.0)
    np.testing.assert_array_equal(cal.coefficients, [0.0, 1.0])
    # The issued threshold is zero; recomputing with the new state would
    # incorrectly count this as covered and use a newly observed lag.
    cal.update(second, 1.0)
    np.testing.assert_array_equal(cal.coefficients, [0.0, 2.5])
    assert (cal.intercept, cal.n_updates, cal.pending) == (1.0, 2, 0)
    third = cal.predict([0.0])
    cal.update(third, 2.0)
    assert cal.predict([0.0]).threshold == 2.5
    copied = cal.coefficients
    copied[:] = 0.0
    np.testing.assert_array_equal(cal.coefficients, [0.5, 2.5])


def test_online_receipts_refuse_replay_reordering_and_foreign_calibrators() -> None:
    cal = OnlineCalibrator(0.25, 0.5)
    other = OnlineCalibrator(0.25, 0.5)
    first, second = cal.predict(), cal.predict()
    with pytest.raises(HessboostError):
        cal.update(second, 1.0)
    with pytest.raises(HessboostError):
        other.update(first, 1.0)
    with pytest.raises(hessboost.InvalidDataError):
        cal.update(first, np.nan)
    assert (cal.intercept, cal.n_updates, cal.pending) == (0.0, 0, 2)
    cal.update(first, 0.0)  # A tie is covered; the next threshold can be negative.
    assert cal.intercept == -0.125
    with pytest.raises(HessboostError):
        cal.update(first, 1.0)
    cal.update(second, 1.0)
    assert (cal.intercept, cal.n_updates, cal.pending) == (0.25, 2, 0)


def test_online_projection_and_numerical_refusals() -> None:
    cal = OnlineCalibrator(0.5, 2.0, covariates=2, coefficient_bound=5.0)
    first = cal.predict([3.0, 4.0])
    cal.update(first, 1.0)
    second = cal.predict([3.0, 4.0])
    cal.update(second, 100.0)
    np.testing.assert_array_equal(cal.coefficients, [3.0, 4.0])
    assert cal.intercept == 2.0
    for values in ([1.0], [[1.0, 2.0]], [np.inf, 0.0]):
        with pytest.raises(HessboostError):
            cal.predict(values)
    assert cal.pending == 0
    enormous = OnlineCalibrator(0.25, np.finfo(np.float64).max, covariates=1)
    issued = enormous.predict([np.finfo(np.float64).max])
    with pytest.raises(HessboostError):
        enormous.update(issued, 1.0)
    assert enormous.intercept == 0.0
    assert (enormous.n_updates, enormous.pending) == (0, 1)


def test_online_numeric_and_receipt_types() -> None:
    with pytest.raises(TypeError):
        OnlineCalibrator("0.1", 0.1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        OnlineCalibrator(0.1, True)
    with pytest.raises(TypeError):
        OnlineCalibrator(0.1, 0.1, lags=1.5)  # ty: ignore[invalid-argument-type]
    with pytest.raises(HessboostError):
        OnlineCalibrator(0.1, 0.1, lags=-1)
    for alpha in (0.0, 1.0, np.nan):
        with pytest.raises(HessboostError):
            OnlineCalibrator(alpha, 0.1)
    with pytest.raises(HessboostError):
        OnlineCalibrator(0.1, 0.0)
    with pytest.raises(TypeError):
        CalibrationForecast()
    cal = OnlineCalibrator(0.1, 0.1, covariates=1)
    with pytest.raises(TypeError):
        cal.predict(["1"])
    with pytest.raises(TypeError):
        cal.predict([1j])
    with pytest.raises(TypeError):
        cal.update(1.0, 1.0)  # ty: ignore[invalid-argument-type]


def test_online_state_and_receipts_are_not_copied_or_pickled() -> None:
    import copy
    import pickle

    cal = OnlineCalibrator(0.1, 0.1)
    issued = cal.predict()
    for operation in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError):
            operation(cal)
    with pytest.raises(TypeError):
        pickle.dumps(issued)


def test_online_concurrent_predictions_preserve_all_receipts() -> None:
    cal = OnlineCalibrator(0.1, 0.01, covariates=256)
    features = np.linspace(-1.0, 1.0, 256)
    features.flags.writeable = False
    with ThreadPoolExecutor(max_workers=8) as pool:
        forecasts = list(pool.map(lambda _: cal.predict(features), range(512)))
    forecasts.sort(key=lambda issued: issued.index)
    assert [issued.index for issued in forecasts] == list(range(512))
    assert cal.pending == 512
    for issued in forecasts:
        cal.update(issued, 1.0)
    assert cal.n_updates == 512
    assert cal.pending == 0
    assert cal.intercept == pytest.approx(512 * 0.009)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_online_fork_refuses_inherited_state_but_new_state_works() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
from hessboost import HessboostError
from hessboost.conformal import OnlineCalibrator
parent = OnlineCalibrator(0.1, 0.01)
issued = parent.predict()
pid = os.fork()
if pid == 0:
    try:
        try:
            parent.predict()
        except HessboostError:
            pass
        else:
            os._exit(1)
        child = OnlineCalibrator(0.1, 0.01)
        child.update(child.predict(), 1.0)
        assert child.n_updates == 1
    except BaseException:
        os._exit(1)
    os._exit(0)
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0
parent.update(issued, 1.0)
assert parent.n_updates == 1
""",
        ],
        check=True,
        timeout=30,
    )
