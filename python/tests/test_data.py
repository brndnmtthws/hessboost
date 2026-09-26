"""Input conversion: dtypes and layouts, missing values, pandas frames and
categoricals, scipy sparse matrices, and DMatrix metadata."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from conftest import regression

import hessboost
from hessboost import DMatrix, HessboostError


def test_any_dtype_and_layout_trains_the_same_model() -> None:
    x, y = regression(rows=200)
    reference = hessboost.train({}, DMatrix(x.astype(np.float32), y), 5).save_raw()
    for variant in [
        x,
        np.asfortranarray(x),
        np.ascontiguousarray(np.repeat(x, 2, axis=1)[:, ::2]),
        np.repeat(x, 2, axis=1)[:, ::2],
        x.tolist(),
        pd.DataFrame(x),
    ]:
        assert hessboost.train({}, DMatrix(variant, y.tolist()), 5).save_raw() == reference


def test_nan_is_missing_and_other_sentinels_are_honored() -> None:
    x, y = regression(rows=200)
    holes = x.copy()
    holes[::3, 0] = np.nan
    sentinel = x.copy()
    sentinel[::3, 0] = -999.0
    nan_model = hessboost.train({}, DMatrix(holes, y), 5)
    sentinel_model = hessboost.train({}, DMatrix(sentinel, y, missing=-999.0), 5)
    assert nan_model.save_raw() == sentinel_model.save_raw()
    np.testing.assert_array_equal(
        nan_model.predict(holes), sentinel_model.predict(sentinel, missing=-999.0)
    )


def test_invalid_values_are_refused() -> None:
    x, y = regression(rows=20)
    bad = x.copy()
    bad[0, 0] = np.inf
    with pytest.raises(HessboostError, match="finite"):
        DMatrix(bad, y)
    labels = y.copy()
    labels[3] = np.nan
    with pytest.raises(HessboostError, match="labels must be finite"):
        DMatrix(x, labels)
    with pytest.raises(HessboostError, match="labels"):
        DMatrix(x, y[:-1])
    with pytest.raises(HessboostError, match="weights"):
        DMatrix(x, y, weight=-np.ones(20))
    with pytest.raises(ValueError, match="2-D"):
        DMatrix(x[:, 0], y)
    with pytest.raises(HessboostError, match="empty"):
        DMatrix(np.empty((0, 3)))
    with pytest.raises(ValueError, match="complex"):
        DMatrix(x.astype(np.complex128))
    with pytest.raises(HessboostError, match="2 feature names for 5"):
        DMatrix(x, feature_names=["a", "b"])
    with pytest.raises(HessboostError, match="unique"):
        DMatrix(x, feature_names=["a"] * 5)
    with pytest.raises(HessboostError, match="feature type 'x'"):
        DMatrix(x, feature_types=["x"] * 5)
    with pytest.raises(HessboostError, match="together"):
        DMatrix(x, label_lower_bound=y)
    with pytest.raises(HessboostError, match="group or qid"):
        DMatrix(x, y, group=[20], qid=np.zeros(20))


def test_metadata_getters_and_set_info() -> None:
    x, y = regression(rows=10)
    matrix = DMatrix(x, y, feature_names=list("abcde"))
    assert (matrix.num_row(), matrix.num_col()) == (10, 5)
    np.testing.assert_array_equal(matrix.get_label(), y.astype(np.float32))
    assert matrix.get_weight().size == 0
    assert matrix.get_base_margin().size == 0
    assert matrix.get_group().size == 0
    matrix.set_info(weight=np.arange(1, 11), base_margin=np.ones(10), group=[4, 6])
    np.testing.assert_array_equal(matrix.get_weight(), np.arange(1, 11, dtype=np.float32))
    np.testing.assert_array_equal(matrix.get_base_margin(), np.ones(10, np.float32))
    np.testing.assert_array_equal(matrix.get_group(), [4, 6])
    np.testing.assert_array_equal(matrix.get_label(), y.astype(np.float32))
    with pytest.raises(HessboostError):
        matrix.set_info(label=np.ones(3))
    np.testing.assert_array_equal(matrix.get_label(), y.astype(np.float32))
    targets = DMatrix(x, np.column_stack([y, y]))
    assert targets.get_label().shape == (10, 2)
    matrix.feature_names = ["v", "w", "x", "y", "z"]
    assert matrix.feature_names == ["v", "w", "x", "y", "z"]
    part = matrix.slice([9, 0])
    np.testing.assert_array_equal(part.get_label(), y[[9, 0]].astype(np.float32))
    np.testing.assert_array_equal(part.get_weight(), [10.0, 1.0])
    assert part.feature_names == matrix.feature_names
    assert part.get_group().size == 0
    with pytest.raises(HessboostError, match="out of range"):
        matrix.slice([10])
    assert repr(matrix) == "DMatrix(rows=10, features=5)"


def test_scipy_sparse_matrices_equal_dense_with_nan() -> None:
    rng = np.random.default_rng(2)
    dense = rng.normal(size=(300, 6))
    dense[rng.random(dense.shape) < 0.6] = 0.0
    y = dense[:, 0] - dense[:, 1] + rng.normal(0, 0.1, 300)
    with_nan = np.where(dense == 0.0, np.nan, dense)
    reference = hessboost.train({}, DMatrix(with_nan, y), 5)
    for sparse in [
        scipy.sparse.csr_matrix(dense),
        scipy.sparse.csc_matrix(dense),
        scipy.sparse.coo_array(dense),
    ]:
        booster = hessboost.train({}, DMatrix(sparse, y), 5)
        assert booster.save_raw() == reference.save_raw()
        np.testing.assert_array_equal(booster.predict(sparse), reference.predict(with_nan))
    # Uncanonical CSR (duplicate entries) is summed first, as scipy does.
    coo = scipy.sparse.coo_matrix(
        (np.array([1.0, 2.0]), (np.array([0, 0]), np.array([1, 1]))), shape=(2, 3)
    )
    np.testing.assert_array_equal(DMatrix(coo.tocsr(), [0, 1]).num_col(), 3)


def frame(rows: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    colors = np.array(["red", "green", "blue", "cyan", "plum"])
    color = colors[rng.integers(0, 5, rows)]
    effect = {"red": 0.0, "green": 3.0, "blue": -2.0, "cyan": 1.0, "plum": 5.0}
    size = rng.normal(size=rows)
    y = np.array([effect[c] for c in color]) + size
    df = pd.DataFrame(
        {
            "color": pd.Categorical(color),
            "size": size,
            "count": pd.array(rng.integers(0, 5, rows), dtype="Int64"),
            "flag": rng.random(rows) < 0.5,
        }
    )
    return df, y


def test_pandas_frames_keep_names_and_categories() -> None:
    df, y = frame()
    df.loc[3, "count"] = pd.NA
    matrix = DMatrix(df, y)
    assert matrix.feature_names == ["color", "size", "count", "flag"]
    assert matrix.feature_types == ["c", "q", "q", "q"]
    booster = hessboost.train({"max_depth": 3}, matrix, 50)
    assert booster.feature_names == ["color", "size", "count", "flag"]
    assert booster.feature_types == ["c", "q", "q", "q"]
    predictions = booster.predict(df)
    assert np.sqrt(np.mean((predictions - y) ** 2)) < 0.5
    assert set(booster.get_score()) <= {"color", "size", "count", "flag"}
    assert "color" in booster.get_score("total_gain")


def test_prediction_recodes_categories_to_the_training_ones() -> None:
    df, y = frame()
    booster = hessboost.train({"max_depth": 3}, DMatrix(df, y), 30)
    expected = booster.predict(df)
    # Same values, categories listed in another order: different codes.
    reordered = df.copy()
    reordered["color"] = reordered["color"].cat.reorder_categories(
        ["plum", "cyan", "blue", "green", "red"]
    )
    assert not np.array_equal(reordered["color"].cat.codes, df["color"].cat.codes)
    np.testing.assert_array_equal(booster.predict(reordered), expected)
    # A DMatrix built from it cannot be re-coded, so it is refused.
    with pytest.raises(HessboostError, match="categories of feature 0 differ"):
        booster.predict(DMatrix(reordered))
    # An unseen category is missing.
    unseen = df.head(3).copy()
    unseen["color"] = pd.Categorical(["mauve"] * 3)
    missing = df.head(3).copy()
    missing["color"] = pd.Categorical([None] * 3, categories=["red"])
    np.testing.assert_array_equal(booster.predict(unseen), booster.predict(missing))


def test_frames_need_numeric_or_category_columns() -> None:
    df, y = frame(rows=20)
    df["name"] = "x"
    with pytest.raises(TypeError, match="column 'name' has dtype"):
        DMatrix(df, y)
    with pytest.raises(HessboostError, match="enable_categorical"):
        DMatrix(df.drop(columns="name"), y, enable_categorical=False)


def test_numpy_category_codes_with_feature_types() -> None:
    df, y = frame()
    codes = np.column_stack([df["color"].cat.codes, df["size"]]).astype(np.float64)
    booster = hessboost.train({"max_depth": 3}, DMatrix(codes, y, feature_types=["c", "q"]), 30)
    assert booster.feature_types == ["c", "q"]
    frame_model = hessboost.train({"max_depth": 3}, DMatrix(df[["color", "size"]], y), 30)
    assert booster.save_raw() == frame_model.save_raw()
    bad = codes.copy()
    bad[0, 0] = 1.5
    with pytest.raises(HessboostError, match="invalid category value 1.5"):
        DMatrix(bad, y, feature_types=["c", "q"])


def test_prediction_validates_feature_names() -> None:
    df, y = frame()
    booster = hessboost.train({}, DMatrix(df, y), 3)
    renamed = df.rename(columns={"size": "SIZE"})
    with pytest.raises(HessboostError, match="feature names differ"):
        booster.predict(renamed)
    np.testing.assert_array_equal(
        booster.predict(renamed, validate_features=False), booster.predict(df)
    )
    # Plain arrays carry no names.
    codes = np.column_stack(
        [df["color"].cat.codes, df["size"], df["count"].astype(float), df["flag"]]
    )
    np.testing.assert_array_equal(booster.predict(codes), booster.predict(df))
    with pytest.raises(HessboostError, match="feature count"):
        booster.predict(np.zeros((2, 3)))
