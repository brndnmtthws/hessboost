"""Single-row prediction and the in-place path of ``predict`` on numpy
arrays, each bit for bit the DMatrix path."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import hessboost
from conftest import classes, regression
from hessboost import DMatrix, HessboostError, InvalidDataError

MISSING = -999.0


def with_gaps(x: NDArray[np.float64], seed: int = 1) -> NDArray[np.float64]:
    x = x.copy()
    x[np.random.default_rng(seed).random(x.shape) < 0.15] = np.nan
    return x


def models() -> list[tuple[hessboost.Booster, NDArray[np.float64]]]:
    """Regression with missing values, binary, softprob, softmax (one class
    index per row), a multi-target model, and one with categorical splits,
    each with its data."""
    x, y = regression(rows=300)
    x = with_gaps(x)
    xc, labels = classes(rows=300, n_classes=3)
    binary = (labels > 0).astype(float)
    cats = np.column_stack([np.random.default_rng(2).integers(0, 6, 300), x[:, :2]])
    return [
        (hessboost.train({"max_depth": 3}, DMatrix(x, y), 8), x),
        (hessboost.train({"objective": "binary:logistic"}, DMatrix(xc, binary), 6), xc),
        (
            hessboost.train(
                {"objective": "multi:softprob", "num_class": 3}, DMatrix(xc, labels), 5
            ),
            xc,
        ),
        (
            hessboost.train({"objective": "multi:softmax", "num_class": 3}, DMatrix(xc, labels), 5),
            xc,
        ),
        (hessboost.train({}, DMatrix(x, np.column_stack([y, -y])), 5), x),
        (
            hessboost.train(
                {"max_depth": 3},
                DMatrix(cats, cats[:, 0] % 3 + y, feature_types=["c", "q", "q"]),
                6,
            ),
            cats,
        ),
    ]


@pytest.mark.parametrize("output_margin", [False, True])
def test_predict_row_equals_the_batch_row_bit_for_bit(output_margin: bool) -> None:
    for booster, x in models():
        batch = booster.predict(DMatrix(x[:20]), output_margin=output_margin)
        for index in range(20):
            row = booster.predict_row(x[index], output_margin=output_margin)
            assert row.dtype == np.float32
            assert row.ndim == 1
            np.testing.assert_array_equal(row, np.atleast_1d(batch[index]))
        ranged = booster.predict(
            DMatrix(x[:1]), output_margin=output_margin, iteration_range=(0, 2)
        )
        np.testing.assert_array_equal(
            booster.predict_row(x[0], output_margin=output_margin, iteration_range=(0, 2)),
            ranged.reshape(-1),
        )


def test_predict_row_lengths() -> None:
    xc, labels = classes(rows=200, n_classes=3)
    softmax = hessboost.train(
        {"objective": "multi:softmax", "num_class": 3}, DMatrix(xc, labels), 3
    )
    assert softmax.predict_row(xc[0]).shape == (1,)
    assert softmax.predict_row(xc[0], output_margin=True).shape == (3,)


def test_predict_row_writes_into_out() -> None:
    booster, x = models()[2]
    out = np.full(3, np.nan, dtype=np.float32)
    returned = booster.predict_row(x[4], out=out)
    assert returned is out
    np.testing.assert_array_equal(out, booster.predict(x[4:5])[0])
    with pytest.raises(TypeError, match="float32"):
        booster.predict_row(x[4], out=np.zeros(3))  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="C-contiguous"):
        booster.predict_row(x[4], out=np.zeros(6, dtype=np.float32)[::2])
    with pytest.raises(TypeError):
        booster.predict_row(x[4], out=[0.0, 0.0, 0.0])  # ty: ignore[invalid-argument-type]
    with pytest.raises(HessboostError):
        booster.predict_row(x[4], out=np.zeros(2, dtype=np.float32))
    frozen = np.zeros(3, dtype=np.float32)
    frozen.flags.writeable = False
    with pytest.raises(HessboostError, match="writeable"):
        booster.predict_row(x[4], out=frozen)


def test_predict_row_missing_and_refusals() -> None:
    booster, x = models()[0]
    marked = np.where(np.isnan(x), MISSING, x)
    np.testing.assert_array_equal(
        booster.predict_row(marked[3], missing=MISSING), booster.predict_row(x[3])
    )
    with pytest.raises(HessboostError, match="1-D"):
        booster.predict_row(x[:1])
    with pytest.raises(HessboostError, match="feature count"):
        booster.predict_row(x[0, :3])
    with pytest.raises(InvalidDataError, match="NaN"):
        booster.predict_row(x[np.isnan(x).any(axis=1)][0], missing=MISSING)
    with pytest.raises(InvalidDataError):
        booster.predict_row(np.full(x.shape[1], np.inf))


def inputs(x: NDArray[np.float64]) -> list[tuple[NDArray[Any], float]]:
    """``x`` as float64, float32 and Fortran-ordered arrays with NaN
    missing, and with a non-NaN marker."""
    return [
        (x, np.nan),
        (x.astype(np.float32), np.nan),
        (np.asfortranarray(x), np.nan),
        (np.where(np.isnan(x), MISSING, x), MISSING),
        (np.asfortranarray(np.where(np.isnan(x), MISSING, x)), MISSING),
    ]


@pytest.mark.parametrize("output_margin", [False, True])
def test_predict_on_arrays_equals_the_dmatrix_path(output_margin: bool) -> None:
    for booster, x in models():
        for data, missing in inputs(x):
            fast = booster.predict(data, output_margin=output_margin, missing=missing)
            matrix = booster.predict(DMatrix(data, missing=missing), output_margin=output_margin)
            assert fast.dtype == matrix.dtype
            assert fast.shape == matrix.shape
            np.testing.assert_array_equal(fast, matrix)
    booster, x = models()[0]
    ints = np.arange(50, dtype=np.int64).reshape(10, 5)
    np.testing.assert_array_equal(booster.predict(ints), booster.predict(DMatrix(ints)))


def test_predict_on_arrays_refuses_what_the_dmatrix_path_refuses() -> None:
    booster, x = models()[0]
    with pytest.raises(InvalidDataError):
        booster.predict(x, missing=MISSING)  # a NaN that is not missing
    with pytest.raises(InvalidDataError):
        booster.predict(np.full((2, x.shape[1]), np.inf))
    with pytest.raises(HessboostError, match="feature count"):
        booster.predict(x[:, :4])
    with pytest.raises(HessboostError):
        booster.predict(np.empty((0, x.shape[1])))


def test_transform_margins_is_predicts_transform() -> None:
    xc, labels = classes(rows=300, n_classes=3)
    x, y = regression(rows=300)
    boosters = [
        (hessboost.train({"objective": "binary:logistic"}, DMatrix(xc, labels > 0), 5), xc),
        (
            hessboost.train(
                {"objective": "multi:softprob", "num_class": 3}, DMatrix(xc, labels), 5
            ),
            xc,
        ),
        (
            hessboost.train({"objective": "multi:softmax", "num_class": 3}, DMatrix(xc, labels), 5),
            xc,
        ),
        (
            hessboost.train(
                {"objective": "reg:quantileerror", "quantile_alpha": [0.1, 0.5, 0.9]},
                DMatrix(x, y),
                5,
            ),
            x,
        ),
        (hessboost.train({"objective": "count:poisson"}, DMatrix(x, np.abs(y).round()), 5), x),
    ]
    for booster, data in boosters:
        margins = booster.predict(data, output_margin=True)
        predictions = booster.predict(data)
        transformed = booster.transform_margins(margins)
        assert transformed.dtype == np.float32
        assert transformed.shape == predictions.shape
        np.testing.assert_array_equal(transformed, predictions)
        out = np.empty_like(predictions, dtype=np.float32)
        assert booster.transform_margins(margins.astype(np.float64), out=out) is out
        np.testing.assert_array_equal(out, predictions)
    binary, data = boosters[0]
    for row in data[:10]:
        margin = binary.predict_row(row, output_margin=True)[0]
        value = binary.transform_margin(margin)
        assert isinstance(value, float)
        assert np.float32(value) == binary.predict_row(row)[0]


def test_transform_margins_refusals() -> None:
    xc, labels = classes(rows=200, n_classes=3)
    softprob = hessboost.train(
        {"objective": "multi:softprob", "num_class": 3}, DMatrix(xc, labels), 2
    )
    margins = softprob.predict(xc[:4], output_margin=True)
    with pytest.raises(hessboost.IncompatibleModelError, match="outputs"):
        softprob.transform_margin(0.5)
    with pytest.raises(HessboostError, match="shape"):
        softprob.transform_margins(margins.reshape(-1))
    with pytest.raises(HessboostError, match="shape"):
        softprob.transform_margins(margins[:, :2])
    with pytest.raises(HessboostError, match="shape"):
        softprob.transform_margins(margins, out=np.empty((4, 2), dtype=np.float32))
    float64 = np.empty((4, 3))
    with pytest.raises(TypeError, match="float32"):
        softprob.transform_margins(margins, out=float64)  # ty: ignore[invalid-argument-type]


def materialized(
    base: NDArray[np.float64], features: list[int], values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The rows ``base`` with ``features`` set to each row of ``values``."""
    rows = np.repeat(base[None, :], len(values), axis=0)
    rows[:, features] = values
    return rows


@pytest.mark.parametrize("output_margin", [False, True])
def test_predict_variants_equals_predict_of_the_rows(output_margin: bool) -> None:
    rng = np.random.default_rng(5)
    for booster, x in models():
        features = [x.shape[1] - 1, 0]
        # 70 variants: past one 64-variant walk; missing values included.
        values = x[rng.integers(0, len(x), 70)][:, features]
        rows = materialized(x[1], features, values)
        for iteration_range in [None, (0, 2), (1, 3)]:
            variants = booster.predict_variants(
                x[1], features, values, output_margin=output_margin, iteration_range=iteration_range
            )
            expected = booster.predict(
                rows, output_margin=output_margin, iteration_range=iteration_range
            ).reshape(70, -1)
            assert variants.dtype == np.float32
            assert variants.shape == expected.shape
            np.testing.assert_array_equal(variants, expected)


def test_predict_variants_by_feature_name() -> None:
    x, y = regression(rows=200)
    names = [f"f{i}" for i in range(x.shape[1])]
    booster = hessboost.train({}, DMatrix(x, y, feature_names=names), 4)
    values = x[:10, [3, 1]]
    by_index = booster.predict_variants(x[0], [3, 1], values)
    np.testing.assert_array_equal(booster.predict_variants(x[0], ["f3", "f1"], values), by_index)
    np.testing.assert_array_equal(booster.predict_variants(x[0], ["f3", 1], values), by_index)
    np.testing.assert_array_equal(
        by_index, booster.predict(materialized(x[0], [3, 1], values)).reshape(10, 1)
    )
    with pytest.raises(HessboostError, match="unknown feature"):
        booster.predict_variants(x[0], ["nope"], values[:, :1])
    unnamed = hessboost.train({}, DMatrix(x, y), 2)
    with pytest.raises(HessboostError, match="no feature names"):
        unnamed.predict_variants(x[0], ["f3"], values[:, :1])


def test_predict_variants_refusals() -> None:
    booster, x = models()[0]
    values = x[:4, :2]
    with pytest.raises(HessboostError, match="1-D"):
        booster.predict_variants(x[:2], [0, 1], values)
    with pytest.raises(HessboostError, match="feature count"):
        booster.predict_variants(x[0, :3], [0, 1], values)
    with pytest.raises(HessboostError, match="shape"):
        booster.predict_variants(x[0], [0], values)
    with pytest.raises(HessboostError, match="shape"):
        booster.predict_variants(x[0], [0, 1], values.reshape(-1))
    with pytest.raises(HessboostError, match="twice"):
        booster.predict_variants(x[0], [1, 1], values)
    with pytest.raises(HessboostError, match="at least one"):
        booster.predict_variants(x[0], [], np.empty((3, 0)))
    with pytest.raises(HessboostError, match="negative"):
        booster.predict_variants(x[0], [-1, 1], values)
    with pytest.raises(HessboostError, match="out of range"):
        booster.predict_variants(x[0], [0, x.shape[1]], values)
    with pytest.raises(InvalidDataError):
        booster.predict_variants(x[0], [0, 1], np.full((2, 2), np.inf))
    with pytest.raises(InvalidDataError):
        booster.predict_variants(np.full(x.shape[1], np.inf), [0, 1], values)
    with pytest.raises(TypeError, match="sequence"):
        booster.predict_variants(x[0], "f0", values[:, :1])  # a str is not a list of names
    assert booster.predict_variants(x[0], [0, 1], np.empty((0, 2))).shape == (0, 1)


def test_margin_bounds_contain_sampled_predictions() -> None:
    rng = np.random.default_rng(9)
    for booster, x in models():
        categorical = [i for i, t in enumerate(booster.feature_types or []) if t == "c"]
        lower = np.nanquantile(x, 0.2, axis=0)
        upper = np.nanquantile(x, 0.8, axis=0)
        lower[categorical] = upper[categorical] = 2.0
        low, high = booster.margin_bounds(lower, upper)
        assert low.dtype == high.dtype == np.float32
        assert low.shape == high.shape == (booster.num_outputs,)
        rows = lower + (upper - lower) * rng.random((200, x.shape[1]))
        rows[:, categorical] = 2.0
        margins = booster.predict(rows, output_margin=True).reshape(200, -1)
        assert (low <= margins).all()
        assert (margins <= high).all()
        # Unbounded numeric features hold every row of the training data.
        lower = np.full(x.shape[1], -np.inf)
        upper = np.full(x.shape[1], np.inf)
        lower[categorical] = upper[categorical] = 2.0
        low, high = booster.margin_bounds(lower, upper, iteration_range=(0, 2))
        inside = x[(x[:, categorical] == 2.0).all(axis=1)]
        margins = booster.predict(inside, output_margin=True, iteration_range=(0, 2))
        margins = margins.reshape(len(inside), -1)
        assert (low <= margins).all()
        assert (margins <= high).all()
        # A box of one row (missing values as NaN in both bounds) is its margin.
        for row in x[:5]:
            low, high = booster.margin_bounds(row, row)
            exact = booster.predict_row(row, output_margin=True)
            np.testing.assert_array_equal(low, exact)
            np.testing.assert_array_equal(high, exact)


def test_margin_bounds_refusals() -> None:
    booster, x = models()[0]
    lower, upper = np.zeros(x.shape[1]), np.ones(x.shape[1])
    with pytest.raises(HessboostError, match="feature count"):
        booster.margin_bounds(lower[:3], upper[:3])
    with pytest.raises(HessboostError, match="1-D"):
        booster.margin_bounds(lower[None, :], upper)
    with pytest.raises(HessboostError, match="above"):
        booster.margin_bounds(upper, lower)
    one_sided = upper.copy()
    one_sided[1] = np.nan
    with pytest.raises(HessboostError, match="NaN"):
        booster.margin_bounds(lower, one_sided)
    cats, data = models()[5]
    with pytest.raises(HessboostError, match="categorical"):
        cats.margin_bounds(np.zeros(data.shape[1]), np.full(data.shape[1], 5.0))
    xs, y = regression(rows=200)
    linear = hessboost.train({"linear_tree": True}, DMatrix(xs, y), 3)
    with pytest.raises(hessboost.IncompatibleModelError, match="linear leaves"):
        linear.margin_bounds(np.zeros(xs.shape[1]), np.ones(xs.shape[1]))
