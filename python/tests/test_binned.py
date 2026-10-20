"""BinnedMatrix: training on a matrix's bins is training on the matrix, and
everything that would read the values the bins replaced is refused."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

import hessboost
from conftest import regression
from hessboost import BinnedMatrix, DMatrix, HessboostError, InvalidDataError


def test_training_on_bins_is_training_on_their_matrix() -> None:
    x, y = regression(rows=800)
    x[::5, 1] = np.nan
    weights = np.linspace(0.5, 2.0, 800, dtype=np.float32)
    matrix = DMatrix(x, y, weight=weights)
    binned = BinnedMatrix(matrix, max_bin=64)
    assert (binned.num_row(), binned.num_col(), binned.max_bin) == (800, x.shape[1], 64)
    np.testing.assert_array_equal(binned.get_weight(), weights)
    for params in [
        {"max_depth": 4, "max_bin": 64},
        {"max_depth": 4, "max_bin": 64, "subsample": 0.6, "colsample_bynode": 0.7, "seed": 3},
        {
            "max_bin": 64,
            "objective": "reg:quantileerror",
            "quantile_alpha": 0.3,
            "leaf_estimation_method": "exact",
            "subsample": 0.7,
        },
    ]:
        evals: list[tuple[DMatrix, str]] = [(matrix, "train")]
        from_bins: hessboost.EvalsResult = {}
        from_values: hessboost.EvalsResult = {}
        a = hessboost.train(
            params, binned, 6, evals=evals, evals_result=from_bins, verbose_eval=False
        )
        b = hessboost.train(
            params, matrix, 6, evals=evals, evals_result=from_values, verbose_eval=False
        )
        assert a.save_raw() == b.save_raw()
        assert from_bins == from_values


def test_binned_matrices_keep_the_frame_schema_and_take_new_labels() -> None:
    x, y = regression(rows=400)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    frame["kind"] = pd.Categorical(np.array(["a", "b", "c", "d"])[np.arange(400) % 4])
    matrix = DMatrix(frame, y)
    binned = BinnedMatrix(matrix)
    assert binned.feature_names == matrix.feature_names
    assert binned.feature_types == matrix.feature_types
    flipped = -y
    binned.set_info(label=flipped)
    matrix.set_info(label=flipped)
    np.testing.assert_array_equal(binned.get_label(), flipped.astype(np.float32))
    a = hessboost.train({"max_depth": 3}, binned, 5)
    b = hessboost.train({"max_depth": 3}, matrix, 5)
    assert a.save_raw() == b.save_raw()
    # The booster reads the frame as the matrix's training would.
    np.testing.assert_array_equal(a.predict(frame), b.predict(frame))
    assert a.feature_names == matrix.feature_names


def test_custom_objectives_read_the_binned_matrix() -> None:
    x, y = regression(rows=300)
    matrix = DMatrix(x, y)

    def squared_error(
        margins: NDArray[np.float32], dtrain: DMatrix | BinnedMatrix
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        return margins - dtrain.get_label(), np.ones_like(margins)

    a = hessboost.train({"max_depth": 3}, BinnedMatrix(matrix), 4, obj=squared_error)
    b = hessboost.train({"max_depth": 3}, matrix, 4, obj=squared_error)
    assert a.save_raw() == b.save_raw()


def test_what_reads_values_is_refused() -> None:
    x, y = regression(rows=300)
    matrix = DMatrix(x, y)
    binned = BinnedMatrix(matrix, max_bin=32)
    with pytest.raises(HessboostError, match="max_bin = 32"):
        hessboost.train({}, binned, 2)
    for params in [
        {"max_bin": 32, "tree_method": "approx"},
        {"max_bin": 32, "booster": "dart"},
        {"max_bin": 32, "linear_tree": True},
    ]:
        with pytest.raises(HessboostError, match="holds no feature values"):
            hessboost.train(params, binned, 2)
    model = hessboost.train({"max_bin": 32}, matrix, 2)
    with pytest.raises(HessboostError, match="cannot continue"):
        hessboost.train({"max_bin": 32}, binned, 2, xgb_model=model)
    with pytest.raises(TypeError, match="predict on a DMatrix"):
        model.predict(binned)
    with pytest.raises(TypeError, match="eval set 'bins' must be a DMatrix"):
        hessboost.train(
            {"max_bin": 32},
            binned,
            2,
            evals=[(binned, "bins")],  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(TypeError, match="must be a DMatrix"):
        hessboost.cv({"max_bin": 32}, binned, 2)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="data must be a DMatrix"):
        BinnedMatrix(x)  # ty: ignore[invalid-argument-type]


def test_binning_needs_every_row_in_the_cuts() -> None:
    x, y = regression(rows=300)
    weights = np.ones(300, dtype=np.float32)
    weights[7] = 0.0
    with pytest.raises(InvalidDataError, match="zero sample weight"):
        BinnedMatrix(DMatrix(x, y, weight=weights))
    with pytest.raises(HessboostError, match="max_bin"):
        BinnedMatrix(DMatrix(x, y), max_bin=1)
    binned = BinnedMatrix(DMatrix(x, y))
    with pytest.raises(TypeError):
        binned.set_info(weight=weights)  # ty: ignore[unknown-argument]
