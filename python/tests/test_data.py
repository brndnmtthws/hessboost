"""Input conversion: dtypes and layouts, missing values, pandas frames and
categoricals, scipy sparse matrices, and DMatrix metadata."""

from __future__ import annotations

import array
import ctypes
import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

import hessboost
from conftest import FRAME_TYPES, frame, numpy_codes, regression, reorder_colors, rmse
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
    part = matrix.slice([4, 5, 6, 7, 8, 9, 0, 1, 2, 3])
    order = [4, 5, 6, 7, 8, 9, 0, 1, 2, 3]
    np.testing.assert_array_equal(part.get_label(), y[order].astype(np.float32))
    np.testing.assert_array_equal(part.get_weight()[:2], [5.0, 6.0])
    assert part.feature_names == matrix.feature_names
    np.testing.assert_array_equal(part.get_group(), [6, 4])
    with pytest.raises(HessboostError, match="query group 1"):
        matrix.slice([9, 0])
    with pytest.raises(HessboostError, match="out of range"):
        matrix.slice([10])
    assert repr(matrix) == "DMatrix(rows=10, features=5)"


def test_slices_train_as_copies_or_on_their_sources_bins() -> None:
    x, y = regression(rows=600)
    x[::7, 1] = np.nan
    matrix = DMatrix(x, y)
    rows = np.r_[np.arange(599, 0, -3), np.repeat(np.arange(50, 90), 2)]
    params = {"max_depth": 4, "subsample": 0.8, "seed": 3}
    copy = hessboost.train(params, DMatrix(x[rows], y[rows]), 6)
    view = hessboost.train(params, matrix.slice(rows), 6)
    assert view.save_raw() == copy.save_raw()
    with pytest.raises(HessboostError, match="cuts must be"):
        matrix.slice(rows, cuts="parent")  # ty: ignore[invalid-argument-type]

    # A step at 5 on a feature of the values 0..9: a view without the rows
    # at 5 cuts at the next value it holds, a view with its source's cuts
    # at 5 itself.
    steps = np.repeat(np.arange(10, dtype=np.float32), 30)[:, None]
    labels = (steps[:, 0] >= 5).astype(np.float32)
    without_five = np.flatnonzero(steps[:, 0] != 5)
    stepped = DMatrix(steps, labels)
    stump = {"max_depth": 1}
    own = hessboost.train(stump, stepped.slice(without_five), 1).model_info()
    source = hessboost.train(stump, stepped.slice(without_five, cuts="source"), 1).model_info()
    assert own.trees[0].threshold[0] == 6.0
    assert source.trees[0].threshold[0] == 5.0


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
    assert rmse(predictions, y) < 0.5
    assert set(booster.get_score()) <= {"color", "size", "count", "flag"}
    assert "color" in booster.get_score("total_gain")


def test_prediction_recodes_categories_to_the_training_ones() -> None:
    df, y = frame()
    booster = hessboost.train({"max_depth": 3}, DMatrix(df, y), 30)
    expected = booster.predict(df)
    # Same values, categories listed in another order: different codes.
    reordered = reorder_colors(df)
    np.testing.assert_array_equal(booster.predict(reordered), expected)
    # A DMatrix built from it cannot be re-coded, so it is refused.
    with pytest.raises(HessboostError, match="categories of feature 'color' differ"):
        booster.predict(DMatrix(reordered))
    # An unseen category is missing.
    unseen = df.head(3).copy()
    unseen["color"] = pd.Categorical(["mauve"] * 3)
    missing = df.head(3).copy()
    missing["color"] = pd.Categorical([None] * 3, categories=["red"])
    np.testing.assert_array_equal(booster.predict(unseen), booster.predict(missing))


def test_eval_sets_must_share_the_training_categories() -> None:
    df, y = frame()
    dtrain = DMatrix(df[:300], y[:300])
    params = {"max_depth": 3}
    with pytest.raises(
        HessboostError, match="eval set 'valid': the categories of feature 'color' differ"
    ):
        hessboost.train(
            params, dtrain, 3, evals=[(DMatrix(reorder_colors(df[300:]), y[300:]), "valid")]
        )
    # Codes without recorded categories (numpy with feature_types) are taken
    # as dtrain's codes; only which features are categorical is checked.
    codes = numpy_codes(df)[300:]
    expected: hessboost.EvalsResult = {}
    hessboost.train(
        params, dtrain, 3, evals=[(DMatrix(df[300:], y[300:]), "valid")], evals_result=expected
    )
    history: hessboost.EvalsResult = {}
    hessboost.train(
        params,
        dtrain,
        3,
        evals=[(DMatrix(codes, y[300:], feature_types=FRAME_TYPES), "valid")],
        evals_result=history,
    )
    assert history == expected
    numerical = DMatrix(codes, y[300:], feature_types=["q"] * 4)
    with pytest.raises(HessboostError, match="eval set 'valid': categorical features differ"):
        hessboost.train(params, dtrain, 3, evals=[(numerical, "valid")])


def test_continuation_and_refresh_refuse_data_with_other_categories() -> None:
    df, y = frame()
    first = hessboost.train({"max_depth": 3}, DMatrix(df, y), 5)
    reordered = DMatrix(reorder_colors(df), y)
    settings: list[dict[str, Any]] = [
        {"max_depth": 3},
        {"process_type": "update", "refresh_leaf": True},
    ]
    for params in settings:
        with pytest.raises(
            HessboostError, match="dtrain: the categories of feature 'color' differ"
        ):
            hessboost.train(params, reordered, 5, xgb_model=first)
        # The training order is accepted.
        hessboost.train(params, DMatrix(df, y), 5, xgb_model=first)


def test_numpy_codes_do_not_bridge_other_categories_into_a_continued_model() -> None:
    # dtrain without recorded categories matches both sides, so the model
    # being continued is checked against the eval sets itself and keeps its
    # categories for prediction.
    df, y = frame()
    first = hessboost.train({"max_depth": 3}, DMatrix(df, y), 5)
    dtrain = DMatrix(numpy_codes(df), y, feature_types=FRAME_TYPES)
    dvalid = DMatrix(reorder_colors(df), y)
    with pytest.raises(
        HessboostError,
        match="eval set 'valid': the categories of feature 'color' differ from xgb_model's",
    ):
        hessboost.train({}, dtrain, 5, evals=[(dvalid, "valid")], xgb_model=first)
    continued = hessboost.train({}, dtrain, 5, xgb_model=first)
    assert continued.feature_names == first.feature_names
    np.testing.assert_array_equal(continued.predict(reorder_colors(df)), continued.predict(df))


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
    with pytest.raises(HessboostError, match=r"invalid category value 1\.5"):
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


def test_native_base_margin_rank_is_checked() -> None:
    # `as_float32` already promotes a scalar to 1-D; the extension must
    # refuse other ranks itself.
    x, y = regression(rows=1)
    native = DMatrix(x, y)._core
    for margin in (np.array(0.5, np.float32), np.zeros((1, 1, 1), np.float32)):
        with pytest.raises(HessboostError, match="1-D or 2-D"):
            native.with_info({"base_margin": margin, "categorical": None})


def test_read_only_float32_arrays_are_borrowed_with_copy_false(tmp_path: Path) -> None:
    x, y = regression(rows=500)
    x = np.ascontiguousarray(x, dtype=np.float32)
    x[::7, 1] = np.nan
    params = {"max_depth": 3, "subsample": 0.8, "seed": 2}
    reference = hessboost.train(params, DMatrix(x, y), 5).save_raw()
    np.save(tmp_path / "x.npy", x)
    frozen = x.copy()
    frozen.setflags(write=False)
    borrowed = [
        DMatrix(frozen, y, copy=False),
        DMatrix(np.load(tmp_path / "x.npy", mmap_mode="r"), y, copy=False),
        # A view of a read-only buffer.
        DMatrix(np.frombuffer(x.tobytes(), np.float32).reshape(x.shape), y, copy=False),
    ]
    # The matrices keep the arrays alive: the memory map has no other owner.
    del frozen
    gc.collect()
    for matrix in borrowed:
        assert hessboost.train(params, matrix, 5).save_raw() == reference
        rows = np.arange(0, 500, 3)
        copy = DMatrix(x[rows], y[rows])
        assert (
            hessboost.train(params, matrix.slice(rows), 5).save_raw()
            == hessboost.train(params, copy, 5).save_raw()
        )


def test_copy_false_refuses_what_it_cannot_borrow() -> None:
    x, y = regression(rows=50)
    x = np.ascontiguousarray(x, dtype=np.float32)
    with pytest.raises(HessboostError, match="read-only"):
        DMatrix(x, y, copy=False)
    strided = x[:, ::2]
    strided.setflags(write=False)
    with pytest.raises(HessboostError, match="C-contiguous"):
        DMatrix(strided, y, copy=False)
    view = x.view()
    view.setflags(write=False)
    with pytest.raises(HessboostError, match="base can still write it"):
        DMatrix(view, y, copy=False)
    wrapped = np.asarray(memoryview(x))
    wrapped.setflags(write=False)
    with pytest.raises(HessboostError, match="base can still write it"):
        DMatrix(wrapped, y, copy=False)

    class HidesBase(np.ndarray):
        @property
        def base(self) -> None:
            return None

    hidden = x.view(HidesBase)
    hidden.setflags(write=False)
    with pytest.raises(HessboostError, match="base can still write it"):
        DMatrix(hidden, y, copy=False)
    for owner in (bytearray(32), array.array("f", [0.0] * 8)):
        over_owner = np.ndarray((4, 2), dtype=np.float32, buffer=owner)
        over_owner.setflags(write=False)
        with pytest.raises(HessboostError, match="base can still write it"):
            DMatrix(over_owner, copy=False)
    # A base exposing only an array interface hides whether its memory (here
    # `x`'s) can be written.
    interfaced = np.lib.stride_tricks.as_strided(x, writeable=False)
    with pytest.raises(HessboostError, match="exposes no buffer to ask"):
        DMatrix(interfaced, y, copy=False)
    with pytest.raises(TypeError, match="float32"):
        DMatrix(x.astype(np.float64), y, copy=False)
    with pytest.raises(TypeError, match="numpy array"):
        DMatrix(pd.DataFrame(x), y, copy=False)


def test_copy_false_refuses_memory_no_python_object_owns() -> None:
    # A memory view without an exporter (`ctypes.memoryview_at`'s) keeps no
    # owner alive, so a matrix borrowing an array over it could outlive the
    # memory. `owner` lives through the test: nothing here reads freed memory.
    if sys.version_info < (3, 14):
        pytest.skip("ctypes.memoryview_at is new in Python 3.14")
    owner = (ctypes.c_float * 12)(*range(12))
    view = ctypes.memoryview_at(ctypes.addressof(owner), ctypes.sizeof(owner), readonly=True)
    unowned = np.frombuffer(view, dtype=np.float32).reshape(4, 3)
    assert view.obj is None
    assert not unowned.flags.writeable
    with pytest.raises(HessboostError, match="no Python object owns"):
        DMatrix(unowned, copy=False)
    # The same values over memory a Python object owns are borrowed.
    owned = np.frombuffer(bytes(owner), dtype=np.float32).reshape(4, 3)
    assert DMatrix(owned, copy=False).num_row() == 4
