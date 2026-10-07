"""``Booster.model_info``: margins recomputed in numpy from the arrays equal
the booster's own, for every tree layout."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import hessboost
from conftest import classes, regression
from hessboost import DMatrix, ModelInfo, TreeInfo


def float32_rows(x: NDArray[np.float64]) -> list[NDArray[np.float32]]:
    """The rows of ``x`` as the model reads them."""
    return [np.asarray(row, dtype=np.float32) for row in x]


def leaf(tree: TreeInfo, row: NDArray[np.float32]) -> int:
    """The leaf ``row`` reaches in ``tree``."""
    node = 0
    while tree.left[node] != -1:
        value = row[tree.feature[node]]
        if np.isnan(value):
            left = bool(tree.default_left[node])
        elif tree.categorical[node]:
            left = int(value) in tree.categories[node].tolist()
        else:
            left = bool(value < tree.threshold[node])
        node = int(tree.left[node] if left else tree.right[node])
    return node


def margins(info: ModelInfo, x: NDArray[np.float64]) -> NDArray[np.float32]:
    """Every row's margins from ``info``: the intercepts plus each tree's
    weighted leaf, added in ``float32`` in tree order as prediction adds
    them, so the result is exact."""
    out = np.empty((len(x), info.num_outputs), dtype=np.float32)
    for r, row in enumerate(float32_rows(x)):
        margin = info.base_margins.copy()
        for t, tree in enumerate(info.trees):
            weight = info.tree_weights[t]
            values = tree.value[leaf(tree, row)]
            if info.vector_leaves:
                margin += weight * values
            else:
                k = info.tree_outputs[t]
                margin[k] = margin[k] + weight * values
        out[r] = margin
    return out


def assert_recomputed(booster: hessboost.Booster, x: NDArray[np.float64]) -> ModelInfo:
    info = booster.model_info()
    expected = booster.predict(x, output_margin=True).reshape(len(x), -1)
    np.testing.assert_array_equal(margins(info, x), expected)
    return info


def test_scalar_trees_with_categorical_splits_and_missing_values() -> None:
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 8, 400).astype(float)
    effect = rng.normal(size=8) * 2
    x = np.column_stack([codes, rng.normal(size=(400, 2))])
    y = effect[codes.astype(int)] + x[:, 1] + rng.normal(0, 0.1, 400)
    x[rng.random(x.shape) < 0.1] = np.nan
    booster = hessboost.train({"max_depth": 4}, DMatrix(x, y, feature_types=["c", "q", "q"]), 12)
    info = assert_recomputed(booster, x[:60])
    assert (info.objective, info.num_features, info.num_outputs) == ("reg:squarederror", 3, 1)
    assert (info.num_class, info.num_parallel_tree, info.trees_per_iteration) == (0, 1, 1)
    assert info.num_boosted_rounds == len(info.trees) == 12
    assert info.best_iteration is None
    assert (info.vector_leaves, info.linear_leaves) == (False, False)
    assert (info.gblinear, info.shrinkage) == (None, None)
    assert info.base_margins.dtype == np.float32
    np.testing.assert_array_equal(info.tree_weights, np.ones(12, dtype=np.float32))
    assert any(tree.categorical.any() for tree in info.trees)
    tree = info.trees[0]
    leaves = tree.left == -1
    assert tree.left.dtype == tree.right.dtype == tree.feature.dtype == np.int32
    assert tree.value.dtype == tree.cover.dtype == tree.gain.dtype == np.float32
    np.testing.assert_array_equal(tree.feature[leaves], -1)
    assert (tree.feature[~leaves] >= 0).all()
    assert np.isnan(tree.threshold[leaves | tree.categorical]).all()
    assert np.isnan(tree.value[~leaves]).all()
    assert not np.isnan(tree.value[leaves]).any()
    assert len(tree.categories) == len(tree.left)
    for node, categories in enumerate(tree.categories):
        assert categories.dtype == np.int32
        assert (len(categories) > 0) == bool(tree.categorical[node])
    assert tree.linear is None
    assert (tree.cover > 0).all()


def test_forest_of_a_multiclass_model() -> None:
    x, y = classes(rows=300, n_classes=3)
    booster = hessboost.train(
        {"objective": "multi:softprob", "num_class": 3, "num_parallel_tree": 2},
        DMatrix(x, y),
        4,
    )
    info = assert_recomputed(booster, x[:40])
    assert (info.num_class, info.num_parallel_tree, info.trees_per_iteration) == (3, 2, 6)
    assert info.tree_outputs.dtype == np.int32
    np.testing.assert_array_equal(info.tree_outputs[:6], [0, 0, 1, 1, 2, 2])
    assert len(info.trees) == 24


def test_dart_tree_weights() -> None:
    x, y = regression(rows=300)
    booster = hessboost.train({"booster": "dart", "rate_drop": 0.3, "seed": 2}, DMatrix(x, y), 15)
    info = assert_recomputed(booster, x[:40])
    assert (info.tree_weights != 1).any()


def test_vector_leaf_trees() -> None:
    x, y = regression(rows=300)
    targets = np.column_stack([y, -y, 2 * y])
    booster = hessboost.train(
        {"multi_strategy": "multi_output_tree", "tree_method": "hist"}, DMatrix(x, targets), 6
    )
    info = assert_recomputed(booster, x[:40])
    assert info.vector_leaves
    assert (info.num_outputs, info.num_targets, info.trees_per_iteration) == (3, 3, 1)
    np.testing.assert_array_equal(info.tree_outputs, np.zeros(6, dtype=np.int32))
    assert info.trees[0].value.shape == (len(info.trees[0].left), 3)


def test_gblinear_record() -> None:
    x, y = regression(rows=200)
    booster = hessboost.train({"booster": "gblinear"}, DMatrix(x, y), 5)
    info = booster.model_info()
    assert info.trees == ()
    assert info.gblinear is not None
    weights, bias = info.gblinear.weights, info.gblinear.bias
    assert weights.shape == (5, 1)
    assert weights.dtype == np.float32
    assert bias.shape == (1,)
    # base + bias, then each feature's product formed in float64 and added
    # in float32, in feature order: exact.
    expected = booster.predict(x[:10], output_margin=True)
    for row, want in zip(float32_rows(x[:10]), expected, strict=True):
        margin = np.float32(info.base_margins[0] + bias[0])
        for f, value in enumerate(row):
            margin = np.float32(margin + np.float32(np.float64(weights[f, 0]) * np.float64(value)))
        assert margin == want


def test_shrinkage_record() -> None:
    x, y = regression(rows=200)
    booster = hessboost.train(
        {"model_shrink_rate": 0.2, "model_shrink_mode": "constant", "max_depth": 2},
        DMatrix(x, y),
        6,
    )
    info = booster.model_info()
    assert info.shrinkage is not None
    factors = info.shrinkage.factors
    assert factors.dtype == np.float64
    assert factors.shape == (6,)
    assert factors[0] == 1
    assert (factors[1:] != 1).any()
    # Training's recurrence: shrink, then add the iteration's trees.
    expected = booster.predict(x[:20], output_margin=True)
    for row, want in zip(float32_rows(x[:20]), expected, strict=True):
        margin = info.shrinkage.base_margins.copy()
        for i, factor in enumerate(factors):
            if factor != 1:
                margin = (margin.astype(np.float64) * factor).astype(np.float32)
            tree = info.trees[i]
            margin[0] = margin[0] + tree.value[leaf(tree, row)]
        assert margin[0] == want


def test_linear_leaves_record() -> None:
    x, y = regression(rows=300)
    booster = hessboost.train({"linear_tree": True, "max_depth": 2}, DMatrix(x, y), 3)
    info = booster.model_info()
    assert info.linear_leaves
    # The first boosting round keeps constant leaves.
    assert info.trees[0].linear is None
    for tree in info.trees[1:]:
        linear = tree.linear
        assert linear is not None
        nodes = len(tree.left)
        assert linear.intercept.shape == (nodes,)
        assert linear.offsets.shape == (nodes + 1,)
        assert linear.offsets.dtype == np.int64
        assert linear.features.dtype == np.int32
        assert linear.offsets[0] == 0
        assert linear.offsets[-1] == len(linear.features) == len(linear.coefficients)
        internal = np.flatnonzero(tree.left != -1)
        np.testing.assert_array_equal(linear.offsets[internal + 1], linear.offsets[internal])
    # A leaf predicts its linear model on rows with every term present.
    expected = booster.predict(x[:10], output_margin=True)
    for row, want in zip(float32_rows(x[:10]), expected, strict=True):
        margin = float(info.base_margins[0])
        for tree in info.trees:
            n = leaf(tree, row)
            if tree.linear is None:
                margin += float(tree.value[n])
                continue
            linear = tree.linear
            terms = slice(linear.offsets[n], linear.offsets[n + 1])
            values = row[linear.features[terms]].astype(np.float64)
            margin += float(linear.intercept[n] + linear.coefficients[terms] @ values)
        # Summed in float64 here, in float32 per tree by the model.
        assert margin == pytest.approx(float(want), rel=1e-5, abs=1e-5)
