"""Explainable boosting machines through the Python API."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hessboost
from hessboost.ebm import CategoricalAxis, EbmInfo, NumericAxis, shape_functions
from hessboost.inference import EbmInference, TermBands

CLASSIC = {"booster": "ebm", "eta": 0.1, "max_leaves": 3, "grow_policy": "lossguide"}
BOULEVARD = {
    "booster": "ebm",
    "ebm_boulevard": True,
    "eta": 1.0,
    "subsample": 0.8,
    "max_leaves": 8,
    "grow_policy": "lossguide",
    "min_child_weight": 5,
}


def data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.random((n, 3))
    y = np.sin(6 * x[:, 0]) + (x[:, 1] - 0.5) ** 2 + (x[:, 2] > 0.5) + 0.1 * rng.standard_normal(n)
    return x, y


def test_shape_functions_add_up_to_the_margin() -> None:
    x, y = data(500, 0)
    booster = hessboost.train({**CLASSIC, "ebm_interactions": 1}, hessboost.DMatrix(x, label=y), 40)
    shapes = shape_functions(booster)
    assert len(shapes.terms) == 4
    main, pair = shapes.terms[0], shapes.terms[3]
    assert isinstance(main.axes[0], NumericAxis)
    assert main.values.shape == (main.axes[0].cells,)
    assert pair.values.shape == tuple(axis.cells for axis in pair.axes)
    margins = booster.predict(x[:50], output_margin=True)
    for row, m in zip(x[:50], margins, strict=True):
        total = shapes.intercept + sum(t.value(row[list(t.features)]) for t in shapes.terms)
        assert total == pytest.approx(float(m), abs=1e-4)
    assert main.cell([np.nan]) == (main.axes[0].cells - 1,)
    info = booster.ebm
    assert isinstance(info, EbmInfo) and info.boulevard is None and len(info.terms) == 4
    with pytest.raises(hessboost.HessboostError):
        main.value([0.5, 0.5])


def test_categorical_terms_report_per_category_cells() -> None:
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 4, 600)
    b = rng.random(600)
    effect = np.array([-1.0, 0.5, 0.0, 2.0])
    frame = pd.DataFrame({"c": pd.Categorical(codes), "b": b})
    y = effect[codes] + np.sin(6 * b)
    booster = hessboost.train(CLASSIC, hessboost.DMatrix(frame, label=y, enable_categorical=True), 60)
    shape = shape_functions(booster).terms[0]
    axis = shape.axes[0]
    assert isinstance(axis, CategoricalAxis) and shape.values.shape == (axis.cells,)
    zero = shape.value([2.0])
    for code in (0, 1, 3):
        assert shape.value([float(code)]) - zero == pytest.approx(effect[code], abs=0.3)


def test_boulevard_bands() -> None:
    x, y = data(400, 2)
    booster = hessboost.train(BOULEVARD, hessboost.DMatrix(x, label=y), 30)
    inference = EbmInference.fit(booster, x, label=y)
    bands = inference.term_bands(0, alpha=0.05)
    assert isinstance(bands, TermBands)
    assert bands.standard_errors.shape == bands.shape.values.shape
    assert np.all(bands.lower <= bands.shape.values) and np.all(bands.shape.values <= bands.upper)
    assert inference.term_standard_errors(1, x[:10]).shape == (10,)
    ci = inference.confidence_intervals(x[:10], alpha=0.1)
    pi = inference.prediction_intervals(x[:10], alpha=0.1)
    assert np.all(pi[:, 0] < ci[:, 0]) and inference.standard_errors(x[:10]).shape == (10,)
    assert inference.intercept_standard_error > 0 and inference.noise_variance > 0
    assert booster.ebm is not None and booster.ebm.boulevard is not None


def test_refusals() -> None:
    x, y = data(100, 3)
    classic = hessboost.train(CLASSIC, hessboost.DMatrix(x, label=y), 5)
    with pytest.raises(hessboost.HessboostError, match="Boulevard EBM"):
        EbmInference.fit(classic, x, noise_variance=1.0)
    plain = hessboost.train({"max_depth": 2}, hessboost.DMatrix(x, label=y), 2)
    with pytest.raises(hessboost.HessboostError, match="not an EBM"):
        shape_functions(plain)
    assert plain.ebm is None
    with pytest.raises(hessboost.HessboostError):
        hessboost.train({**CLASSIC, "colsample_bytree": 0.5}, hessboost.DMatrix(x, label=y), 2)
    # Class-balanced bagging samples rows by label, which the Boulevard EBM's
    # kernel cannot represent, whatever the objective.
    # (A non-binary objective is refused first by balanced bagging itself.)
    labels = (y > np.median(y)).astype(float)
    for objective, reason in (("binary:logistic", "linear smoother"), ("reg:squarederror", "binary")):
        balanced = {**BOULEVARD, "subsample": 1.0, "objective": objective}
        with pytest.raises(hessboost.HessboostError, match=reason):
            hessboost.train(
                {**balanced, "neg_bagging_fraction": 0.5}, hessboost.DMatrix(x, label=labels), 2
            )


def test_classic_ebms_bag_rows_by_class() -> None:
    x, y = data(600, 5)
    dtrain = hessboost.DMatrix(x, label=(y > np.median(y)).astype(float))
    params = {**CLASSIC, "objective": "binary:logistic"}
    plain = hessboost.train(params, dtrain, 40).predict(dtrain)
    bagged = hessboost.train({**params, "neg_bagging_fraction": 0.1}, dtrain, 40).predict(dtrain)
    # Keeping a tenth of the negatives pulls every tree toward the positives.
    assert bagged.mean() > plain.mean() + 0.05


def test_classic_ebms_bag_whole_queries() -> None:
    x, y = data(400, 6)
    relevance = np.minimum(np.floor(3 * (x[:, 0] + 0.5 * x[:, 1])), 3)
    dtrain = hessboost.DMatrix(x, label=relevance, group=[10] * 40)
    params = {**CLASSIC, "objective": "rank:ndcg"}
    plain = hessboost.train(params, dtrain, 10).predict(dtrain, output_margin=True)
    bagged = {**params, "bagging_by_query": True, "subsample": 0.3}
    first = hessboost.train(bagged, dtrain, 10).predict(dtrain, output_margin=True)
    again = hessboost.train(bagged, dtrain, 10).predict(dtrain, output_margin=True)
    assert not np.array_equal(first, plain)
    np.testing.assert_array_equal(first, again)


def test_sglb_and_virtual_ensembles_are_refused() -> None:
    x, y = data(200, 7)
    dtrain = hessboost.DMatrix(x, label=y)
    with pytest.raises(hessboost.HessboostError, match="gbtree"):
        hessboost.train({**CLASSIC, "posterior_sampling": True}, dtrain, 2)
    booster = hessboost.train(CLASSIC, dtrain, 10)
    with pytest.raises(hessboost.HessboostError, match="EBM"):
        booster.predict_virtual_ensembles(x, 2)


def test_early_stopping_rounds_zero_is_off() -> None:
    x, y = data(200, 3)
    dtrain = hessboost.DMatrix(x, label=y)
    off = hessboost.train({**CLASSIC, "ebm_early_stopping_rounds": 0}, dtrain, 5)
    plain = hessboost.train(CLASSIC, dtrain, 5)
    np.testing.assert_array_equal(off.predict(x), plain.predict(x))
    with pytest.raises(hessboost.HessboostError, match="ebm_early_stopping_tolerance"):
        hessboost.train({**CLASSIC, "ebm_early_stopping_tolerance": 0.01}, dtrain, 5)
    stopped = {**CLASSIC, "ebm_bag_fraction": 0.8, "ebm_early_stopping_rounds": 2}
    hessboost.train({**stopped, "ebm_early_stopping_tolerance": 0.01}, dtrain, 5)
