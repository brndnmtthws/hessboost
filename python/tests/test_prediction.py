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
