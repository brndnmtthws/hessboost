"""Ordered target statistics: fitting, encoding new data by name or index,
explicit labels for multi-target matrices, and per-fold encoding in cv."""

from __future__ import annotations

import copy as copy_module
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import ArrayLike, NDArray

import hessboost
from hessboost import DMatrix, HessboostError, InvalidDataError, ModelFormatError
from hessboost.target_stats import FittedTargetEncoder, OrderedTargetEncoder


def categorical_data(
    rows: int = 300,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Column 0: one of 20 category codes with its own label mean; column 1
    numeric noise."""
    rng = np.random.default_rng(5)
    codes = rng.integers(0, 20, rows)
    effect = rng.normal(size=20)
    x = np.column_stack([codes, rng.normal(size=rows)])
    return x, effect[codes] + 0.1 * rng.normal(size=rows)


def matrix(x: NDArray[np.float64], label: ArrayLike) -> DMatrix:
    return DMatrix(x, label, feature_names=["city", "noise"], feature_types=["c", "q"])


def test_fit_transform_encodes_by_name_and_index_alike() -> None:
    x, y = categorical_data()
    dtrain = matrix(x, y)
    encoder = OrderedTargetEncoder(seed=3, prior_weight=2.0)
    by_name, stats = encoder.fit_transform(dtrain, ["city"])
    by_index, _ = encoder.fit_transform(dtrain, [0])
    assert by_name.feature_types == ["q", "q"]
    assert by_name.feature_names == dtrain.feature_names
    np.testing.assert_array_equal(by_name.get_label(), dtrain.get_label())
    assert stats.columns == [0]
    assert stats.prior == pytest.approx(y.astype(np.float32).mean(), rel=1e-5)

    # Inference encodes each category by its smoothed mean over every row.
    labels = y.astype(np.float32).astype(np.float64)
    in_three = labels[x[:, 0] == 3]
    want = (in_three.sum() + 2.0 * stats.prior) / (in_three.size + 2.0)
    assert stats.encode("city", 3) == pytest.approx(want, rel=1e-6)
    assert stats.encode(0, 99) == stats.prior
    with pytest.raises(HessboostError, match="not target-encoded"):
        stats.encode("noise", 0)

    # A numpy array takes the training matrix's feature types.
    test = stats.transform(x[:4])
    assert test.feature_types == ["q", "q"]
    trained = hessboost.train({"max_depth": 3}, by_name, 10)
    assert trained.predict(test).shape == (4,)
    booster = hessboost.train({}, by_index, 5)
    assert booster.save_raw() == hessboost.train({}, by_name, 5).save_raw()


def test_label_encodes_one_target_of_a_multi_target_matrix() -> None:
    x, y = categorical_data()
    multi = matrix(x, np.column_stack([y, -y]))
    encoder = OrderedTargetEncoder(seed=1)
    with pytest.raises(InvalidDataError, match="one label per row"):
        encoder.fit_transform(multi, ["city"])
    encoded, stats = encoder.fit_transform(multi, ["city"], label=-y)
    _, single_stats = encoder.fit_transform(matrix(x, -y), ["city"])
    assert encoded.get_label().shape == (300, 2)
    for code in range(20):
        assert stats.encode("city", code) == single_stats.encode("city", code)
    booster = hessboost.train({}, encoded, 3)
    assert booster.predict(stats.transform(x)).shape == (300, 2)
    with pytest.raises(HessboostError, match="one value per row"):
        encoder.fit_transform(multi, ["city"], label=np.column_stack([y, y]))
    with pytest.raises(HessboostError, match="labels"):
        encoder.fit_transform(multi, ["city"], label=y[:10])


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_frame_transform_keeps_unseen_categories_apart_from_nulls(library: str) -> None:
    cities = ["oslo", "rome", "lima", "oslo", "rome", "oslo"]
    y = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 0.5])
    new = ["paris", None, "rome"]
    if library == "pandas":
        import pandas as pd

        train = pd.DataFrame({"city": pd.Categorical(cities), "x": np.arange(6.0)})
        test = pd.DataFrame({"city": pd.Categorical(new), "x": np.arange(3.0)})
        # No categories at all: every code is null.
        nulls = pd.DataFrame({"city": pd.Categorical([None, None]), "x": [0.0, 1.0]})
    else:
        pl = pytest.importorskip("polars")
        train = pl.DataFrame({"city": cities, "x": np.arange(6.0)}).with_columns(
            pl.col("city").cast(pl.Categorical)
        )
        test = pl.DataFrame({"city": new, "x": np.arange(3.0)}).with_columns(
            pl.col("city").cast(pl.Categorical)
        )
        nulls = pl.DataFrame(
            {"city": pl.Series([None, None], dtype=pl.String), "x": [0.0, 1.0]}
        ).with_columns(pl.col("city").cast(pl.Categorical))
    stats = OrderedTargetEncoder().fit_transform(DMatrix(train, y), ["city"])[1]
    rome = stats.encode("city", sorted(set(cities)).index("rome"))
    # A linear model reads the encoded values back: a missing entry adds
    # nothing, any value adds its weighted value.
    rng = np.random.default_rng(0)
    fit = rng.normal(size=(50, 2))
    probe = hessboost.train(
        {"booster": "gblinear", "eta": 0.5}, DMatrix(fit, fit @ [2.0, -1.0] + 0.3), 20
    )
    got = probe.predict(stats.transform(test))
    unseen_prior = [[stats.prior, 0.0], [np.nan, 1.0], [rome, 2.0]]
    np.testing.assert_array_equal(got, probe.predict(np.array(unseen_prior)))
    assert got[0] != probe.predict(np.array([[np.nan, 0.0]]))[0]
    np.testing.assert_array_equal(
        probe.predict(stats.transform(nulls)),
        probe.predict(np.array([[np.nan, 0.0], [np.nan, 1.0]])),
    )


def test_refusals() -> None:
    x, y = categorical_data()
    dtrain = matrix(x, y)
    with pytest.raises(HessboostError, match="not categorical"):
        OrderedTargetEncoder().fit_transform(dtrain, ["noise"])
    with pytest.raises(HessboostError, match="unknown feature"):
        OrderedTargetEncoder().fit_transform(dtrain, ["town"])
    with pytest.raises(TypeError, match="sequence"):
        OrderedTargetEncoder().fit_transform(dtrain, "city")
    with pytest.raises(HessboostError, match="prior_weight"):
        OrderedTargetEncoder(prior_weight=0.0)
    with pytest.raises(HessboostError, match="target"):
        OrderedTargetEncoder(target="multiclass")  # ty: ignore[invalid-argument-type]
    with pytest.raises(InvalidDataError, match="0/1"):
        OrderedTargetEncoder(target="binary").fit_transform(dtrain, ["city"])
    with pytest.raises(TypeError):
        FittedTargetEncoder()
    _, stats = OrderedTargetEncoder().fit_transform(dtrain, ["city"])
    with pytest.raises(HessboostError, match="feature names differ"):
        stats.transform(DMatrix(x, feature_names=["a", "b"], feature_types=["c", "q"]))


def test_cv_fits_the_encoder_on_each_folds_training_rows() -> None:
    x, y = categorical_data()
    dtrain = matrix(x, y)
    encoder = OrderedTargetEncoder(seed=2)
    folds = hessboost.folds.k_fold(300, 3, seed=4)
    result = hessboost.cv(
        {"max_depth": 3}, dtrain, 6, folds=folds, target_stats=["city"], target_encoder=encoder
    )
    # The same folds encoded by hand from their training rows alone.
    scores = []
    for train_rows, test_rows in folds:
        encoded, stats = encoder.fit_transform(dtrain.slice(train_rows), ["city"])
        history: hessboost.EvalsResult = {}
        hessboost.train(
            {"max_depth": 3},
            encoded,
            6,
            evals=[(stats.transform(dtrain.slice(test_rows)), "test")],
            evals_result=history,
            verbose_eval=False,
        )
        scores.append(history["test"]["rmse"][-1])
    assert result["test-rmse-mean"][-1] == pytest.approx(np.mean(scores), rel=1e-12)
    with pytest.raises(HessboostError, match="target_stats"):
        hessboost.cv({}, dtrain, 2, target_encoder=encoder)


def test_cv_target_stats_label_splits_with_the_folds() -> None:
    x, y = categorical_data()
    multi = matrix(x, np.column_stack([y, -y]))
    encoder = OrderedTargetEncoder(seed=6)
    folds = hessboost.folds.k_fold(300, 3, seed=2)
    result = hessboost.cv(
        {"max_depth": 3},
        multi,
        5,
        folds=folds,
        target_stats=["city"],
        target_encoder=encoder,
        target_stats_label=-y,
    )
    scores = []
    for train_rows, test_rows in folds:
        encoded, stats = encoder.fit_transform(
            multi.slice(train_rows), ["city"], label=-y[train_rows]
        )
        history: hessboost.EvalsResult = {}
        hessboost.train(
            {"max_depth": 3},
            encoded,
            5,
            evals=[(stats.transform(multi.slice(test_rows)), "test")],
            evals_result=history,
            verbose_eval=False,
        )
        scores.append(history["test"]["rmse"])
    np.testing.assert_allclose(result["test-rmse-mean"], np.mean(scores, axis=0), rtol=1e-12)
    with pytest.raises(HessboostError, match="target_stats_label"):
        hessboost.cv({}, multi, 2, target_stats_label=-y)
    with pytest.raises(HessboostError):
        hessboost.cv({}, multi, 2, target_stats=["city"], target_stats_label=y[:10])


def test_cv_refit_returns_the_encoder_fitted_on_every_row() -> None:
    x, y = categorical_data()
    dtrain = matrix(x, y)
    encoder = OrderedTargetEncoder(seed=2)
    refit = hessboost.cv(
        {"max_depth": 3},
        dtrain,
        30,
        early_stopping_rounds=2,
        target_stats=["city"],
        target_encoder=encoder,
        refit=True,
    )
    stats = refit.target_encoder
    assert isinstance(stats, FittedTargetEncoder)
    assert stats.columns == [0]
    booster = refit.booster
    assert booster.feature_types == ["q", "q"]
    assert booster.feature_names == ["city", "noise"]
    encoded, by_hand = encoder.fit_transform(dtrain, ["city"])
    trained = hessboost.train({"max_depth": 3}, encoded, refit.num_boost_round)
    np.testing.assert_array_equal(
        booster.predict(stats.transform(x)), trained.predict(by_hand.transform(x))
    )
    # A continued model was trained on its own encoding of the columns.
    init = hessboost.train({}, dtrain, 2)
    with pytest.raises(HessboostError, match="target_stats"):
        hessboost.cv({}, dtrain, 2, target_stats=["city"], xgb_model=init)


def frame_encoder() -> tuple[FittedTargetEncoder, pd.DataFrame, NDArray[np.float64]]:
    """An encoder fitted on a frame, a frame of its and unseen categories
    in another order, and numpy data."""
    x, y = categorical_data()
    cities = np.array(["a", "b", "c", "d"])[x[:, 0].astype(int) % 4]
    train = pd.DataFrame({"city": pd.Categorical(cities), "noise": x[:, 1]})
    _, stats = OrderedTargetEncoder(seed=1).fit_transform(DMatrix(train, y), ["city"])
    test = pd.DataFrame(
        {"city": pd.Categorical(["d", "z", "a", None], categories=["z", "d", "a"]), "noise": 0.5}
    )
    return stats, test, np.column_stack([np.arange(4.0), np.ones(4)])


def assert_same_encoding(a: FittedTargetEncoder, b: FittedTargetEncoder, data: object) -> None:
    """A linear probe's per-feature contributions (weight times value) agree."""
    rng = np.random.default_rng(0)
    fit = rng.normal(size=(50, 2))
    probe = hessboost.train(
        {"booster": "gblinear", "eta": 0.5}, DMatrix(fit, fit @ [2.0, -1.0] + 0.3), 20
    )
    np.testing.assert_array_equal(
        probe.predict(a.transform(data), pred_contribs=True),
        probe.predict(b.transform(data), pred_contribs=True),
    )


def test_persistence_round_trips(tmp_path: Path) -> None:
    stats, test, x = frame_encoder()
    restored = FittedTargetEncoder.from_bytes(memoryview(stats.to_bytes()))
    stats.save(tmp_path / "stats.json")
    loaded = FittedTargetEncoder.load(tmp_path / "stats.json")
    # Without the schema, encoded columns must be marked categorical and a
    # frame must carry the training categories in their order.
    typed = DMatrix(x, feature_types=["c", "q"])
    same_order = test.assign(city=test["city"].cat.set_categories(["a", "b", "c", "d"]))
    for copy in (restored, loaded):
        assert copy.to_bytes() == stats.to_bytes()
        assert_same_encoding(stats, copy, typed)
        assert_same_encoding(stats, copy, same_order)
        assert copy.encode(0, 2) == stats.encode("city", 2)
    for copy in (pickle.loads(pickle.dumps(stats)), copy_module.deepcopy(stats)):
        assert_same_encoding(stats, copy, x)
        assert_same_encoding(stats, copy, test)
        assert copy.encode("city", 2) == stats.encode("city", 2)


def test_corrupt_bytes_raise_model_format_error() -> None:
    stats, _, _ = frame_encoder()
    data = stats.to_bytes()
    for corrupt in (data[:-3], b"\xff", data.replace(b'"prior":', b'"prior":1e999,"x":')):
        with pytest.raises(ModelFormatError):
            FittedTargetEncoder.from_bytes(corrupt)
