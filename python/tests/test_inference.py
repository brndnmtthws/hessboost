"""Boulevard inference through the Python API."""

from __future__ import annotations

import numpy as np
import pytest

import hessboost
from conftest import additive
from hessboost.inference import BoulevardInference, BoulevardInfo, honest_refit

PARAMS = {
    "booster": "boulevard",
    "eta": 0.8,
    "boulevard_dropout": 0.5,
    "subsample": 0.8,
    "max_depth": 3,
    "min_child_weight": 5,
}


def test_intervals_nest_and_match_the_standard_errors() -> None:
    x, y = additive(400, 0, features=2, noise=0.2)
    xc, yc = additive(200, 1, features=2, noise=0.2)
    xt, _ = additive(30, 2, features=2, noise=0.2)
    booster = hessboost.train(PARAMS, hessboost.DMatrix(x, label=y), 60)
    inference = BoulevardInference.fit(booster, x, holdout=xc, holdout_label=yc)
    se = inference.standard_errors(xt)
    ci = inference.confidence_intervals(xt, alpha=0.05)
    pi = inference.prediction_intervals(xt, alpha=0.05)
    ri = inference.reproduction_intervals(xt, alpha=0.05)
    cal = inference.calibrated_prediction_intervals(xt, alpha=0.05)
    assert se.dtype == np.float64
    assert se.shape == (30,)
    assert ci.shape == pi.shape == ri.shape == cal.shape == (30, 2)
    np.testing.assert_allclose((ci[:, 1] - ci[:, 0]) / 2, 1.959964 * se, rtol=1e-5)
    assert np.all(pi[:, 0] < ci[:, 0])
    assert np.all(ci[:, 1] < pi[:, 1])
    assert np.all(ri[:, 1] - ri[:, 0] > ci[:, 1] - ci[:, 0])
    assert inference.noise_variance > 0


def test_nystrom_on_every_row_matches_the_exact_solver() -> None:
    x, y = additive(200, 3, features=2, noise=0.2)
    booster = hessboost.train(PARAMS, hessboost.DMatrix(x, label=y), 20)
    exact = BoulevardInference.fit(booster, x, noise_variance=0.04)
    nystrom = BoulevardInference.fit(booster, x, noise_variance=0.04, landmarks=200)
    np.testing.assert_allclose(nystrom.standard_errors(x), exact.standard_errors(x), rtol=1e-8)


def test_honest_refit_and_the_boulevard_record() -> None:
    x, y = additive(300, 4, features=2, noise=0.2)
    xv, yv = additive(300, 5, features=2, noise=0.2)
    booster = hessboost.train(PARAMS, hessboost.DMatrix(x, label=y), 20)
    info = booster.boulevard
    assert isinstance(info, BoulevardInfo)
    assert info.dropout == 0.5
    assert info.learning_rate == 0.8
    assert info.intercept_from_labels
    # No truncation is None; the flat `boulevard_truncation = 0` is none too.
    assert info.truncation is None
    for level, expected in ((0.0, None), (0.5, 0.5)):
        clipped = hessboost.train(
            {**PARAMS, "boulevard_truncation": level}, hessboost.DMatrix(x, label=y), 2
        )
        assert clipped.boulevard is not None
        assert clipped.boulevard.truncation == expected
    refit = honest_refit(booster, xv, yv)
    assert refit.boulevard == info
    BoulevardInference.fit(refit, xv, label=yv)  # training-residual noise
    plain = hessboost.train({"max_depth": 2}, hessboost.DMatrix(x, label=y), 5)
    assert plain.boulevard is None
    with pytest.raises(hessboost.HessboostError, match="not a Boulevard fit"):
        BoulevardInference.fit(plain, x, noise_variance=1.0)


def test_refusals() -> None:
    x, y = additive(100, 6, features=2, noise=0.2)
    booster = hessboost.train(PARAMS, hessboost.DMatrix(x, label=y), 5)
    with pytest.raises(hessboost.HessboostError, match="not both"):
        BoulevardInference.fit(booster, x, holdout=x, holdout_label=y, noise_variance=1.0)
    with pytest.raises(TypeError):
        BoulevardInference.fit(booster, x, noise_variance="1")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        BoulevardInference()
    with pytest.raises(hessboost.HessboostError, match="objective"):
        hessboost.train(
            {**PARAMS, "objective": "reg:pseudohubererror"}, hessboost.DMatrix(x, label=y), 2
        )
    # Class-balanced bagging samples rows by label (and needs a `binary:*`
    # objective, which it refuses first otherwise).
    balanced = {**PARAMS, "subsample": 1.0, "objective": "binary:logistic"}
    with pytest.raises(hessboost.HessboostError, match="linear smoother"):
        hessboost.train(
            {**balanced, "neg_bagging_fraction": 0.5},
            hessboost.DMatrix(x, label=(y > 0).astype(float)),
            2,
        )
    with pytest.raises(hessboost.HessboostError, match="binary"):
        hessboost.train(
            {**balanced, "objective": "reg:squarederror", "neg_bagging_fraction": 0.5},
            hessboost.DMatrix(x, label=(y > 0).astype(float)),
            2,
        )


def test_sglb_and_virtual_ensembles_are_refused() -> None:
    x, y = additive(200, 7, features=2, noise=0.2)
    dtrain = hessboost.DMatrix(x, label=y)
    for key, value in (
        ("langevin", True),
        ("posterior_sampling", True),
        ("model_shrink_rate", 0.01),
    ):
        with pytest.raises(hessboost.HessboostError, match="gbtree"):
            hessboost.train({**PARAMS, key: value}, dtrain, 2)
    booster = hessboost.train(PARAMS, dtrain, 20)
    with pytest.raises(hessboost.HessboostError, match="Boulevard"):
        booster.predict_uncertainty(x, 2)
    with pytest.raises(hessboost.HessboostError, match="Boulevard"):
        booster.predict_virtual_ensembles(x, 2)
