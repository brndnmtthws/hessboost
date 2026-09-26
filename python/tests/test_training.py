"""Training on every objective family, the xgboost.train options, and the
determinism invariants."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import classes, regression
from numpy.typing import NDArray

import hessboost
from hessboost import DMatrix, HessboostError


def rmse(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - b) ** 2)))


def test_regression_learns_and_reports_the_history() -> None:
    x, y = regression()
    dtrain = DMatrix(x[:300], y[:300])
    dvalid = DMatrix(x[300:], y[300:])
    history: hessboost.EvalsResult = {}
    booster = hessboost.train(
        {"max_depth": 3, "learning_rate": 0.2},
        dtrain,
        60,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        evals_result=history,
        verbose_eval=False,
    )
    assert set(history) == {"train", "valid"}
    valid = history["valid"]["rmse"]
    assert len(valid) == 60
    assert valid[-1] < 0.5 * valid[0]
    predictions = booster.predict(x[300:])
    assert predictions.dtype == np.float32
    assert predictions.shape == (100,)
    assert rmse(predictions, y[300:]) == pytest.approx(valid[-1], rel=1e-5)


def test_verbose_eval_prints_every_nth_round_and_the_last(capsys: pytest.CaptureFixture[str]) -> None:
    x, y = regression(rows=100)
    dtrain = DMatrix(x, y)
    hessboost.train({}, dtrain, 7, evals=[(dtrain, "train")], verbose_eval=3)
    lines = capsys.readouterr().out.splitlines()
    assert [line.split("\t")[0] for line in lines] == ["[0]", "[3]", "[6]"]
    assert lines[0].split("\t")[1].startswith("train-rmse:")


def test_binary_classification_predicts_probabilities_and_margins() -> None:
    x, y = classes()
    dtrain = DMatrix(x, y)
    booster = hessboost.train({"objective": "binary:logistic", "max_depth": 3}, dtrain, 30)
    p = booster.predict(x)
    assert p.shape == (400,)
    assert np.all((p > 0) & (p < 1))
    assert np.mean((p > 0.5) == y) > 0.85
    margin = booster.predict(x, output_margin=True)
    np.testing.assert_allclose(1 / (1 + np.exp(-margin.astype(np.float64))), p, rtol=1e-5)


def test_multiclass_softprob_and_softmax_agree() -> None:
    x, y = classes(n_classes=3)
    dtrain = DMatrix(x, y)
    params = {"num_class": 3, "max_depth": 3, "seed": 1}
    prob = hessboost.train({**params, "objective": "multi:softprob"}, dtrain, 20)
    soft = hessboost.train({**params, "objective": "multi:softmax"}, dtrain, 20)
    p = prob.predict(x)
    assert p.shape == (400, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(soft.predict(x), p.argmax(axis=1).astype(np.float32))
    assert np.mean(p.argmax(axis=1) == y) > 0.7
    assert prob.predict(x, output_margin=True).shape == (400, 3)


def test_multiclass_without_num_class_is_refused() -> None:
    x, y = classes(n_classes=3)
    with pytest.raises(HessboostError, match="num_class"):
        hessboost.train({"objective": "multi:softprob"}, DMatrix(x, y), 2)


def ranking_data() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(3)
    groups = 30
    x = rng.normal(size=(groups * 10, 3))
    relevance = np.clip(np.round(x[:, 0] + rng.normal(0, 0.3, len(x)) + 1), 0, 3)
    qid = np.repeat(np.arange(groups), 10)
    return x, relevance, qid


@pytest.mark.parametrize("objective", ["rank:ndcg", "rank:pairwise", "rank:map"])
def test_ranking_orders_documents_within_queries(objective: str) -> None:
    x, relevance, qid = ranking_data()
    by_group = DMatrix(x, relevance, group=[10] * 30)
    booster = hessboost.train({"objective": objective, "eval_metric": "ndcg@5"}, by_group, 20)
    scores = booster.predict(x)
    first = slice(0, 10)
    assert np.corrcoef(scores[first], relevance[first])[0, 1] > 0.5
    by_qid = DMatrix(x, relevance, qid=qid)
    np.testing.assert_array_equal(by_qid.get_group(), [10] * 30)
    again = hessboost.train({"objective": objective, "eval_metric": "ndcg@5"}, by_qid, 20)
    assert again.save_raw() == booster.save_raw()


def test_ranking_takes_one_weight_per_query() -> None:
    x, relevance, qid = ranking_data()
    plain = hessboost.train({"objective": "rank:ndcg"}, DMatrix(x, relevance, qid=qid), 5)
    weights = np.linspace(0.1, 3.0, 30)
    weighted = DMatrix(x, relevance, qid=qid, weight=weights)
    np.testing.assert_array_equal(weighted.get_weight(), np.repeat(weights, 10).astype(np.float32))
    booster = hessboost.train({"objective": "rank:ndcg"}, weighted, 5)
    assert booster.save_raw() != plain.save_raw()


def test_unsorted_qid_is_refused() -> None:
    x, relevance, _ = ranking_data()
    with pytest.raises(HessboostError, match="sorted"):
        DMatrix(x[:4], relevance[:4], qid=[1, 0, 0, 1])


def test_survival_cox_and_aft() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, 3))
    time = np.exp(0.8 * x[:, 0] + rng.normal(0, 0.2, 300))
    censored = rng.random(300) < 0.2
    cox = hessboost.train(
        {"objective": "survival:cox"}, DMatrix(x, np.where(censored, -time, time)), 20
    )
    hazard = cox.predict(x)
    assert np.all(hazard > 0)
    # Higher risk means shorter survival.
    assert np.corrcoef(np.log(hazard), np.log(time))[0, 1] < -0.5
    upper = np.where(censored, np.inf, time)
    aft = DMatrix(x, label_lower_bound=time, label_upper_bound=upper)
    lower, upper_back = aft.get_label_bounds()
    np.testing.assert_array_equal(lower, time.astype(np.float32))
    assert np.isinf(upper_back[censored]).all()
    booster = hessboost.train(
        {"objective": "survival:aft", "aft_loss_distribution": "logistic"}, aft, 30
    )
    assert np.corrcoef(np.log(booster.predict(x)), np.log(time))[0, 1] > 0.8


def test_quantile_and_multi_target_models_predict_one_column_per_output() -> None:
    rng = np.random.default_rng(0)
    noisy = rng.normal(size=(2000, 3))
    target = noisy[:, 0] + rng.normal(size=2000)
    quantiles = hessboost.train(
        {"objective": "reg:quantileerror", "quantile_alpha": [0.1, 0.9]},
        DMatrix(noisy, target),
        100,
    )
    band = quantiles.predict(noisy)
    assert band.shape == (2000, 2)
    assert 0.05 < np.mean(target < band[:, 0]) < 0.15
    assert 0.05 < np.mean(target > band[:, 1]) < 0.15
    x, y = regression()
    targets = np.column_stack([y, -y, 2 * y])
    for strategy in ["one_output_per_tree", "multi_output_tree"]:
        booster = hessboost.train(
            {"multi_strategy": strategy, "tree_method": "hist"}, DMatrix(x, targets), 30
        )
        predictions = booster.predict(x)
        assert predictions.shape == (400, 3)
        assert rmse(predictions[:, 1], -y) < 0.5


def test_gblinear_and_dart() -> None:
    x, y = regression()
    linear = hessboost.train(
        {"booster": "gblinear", "updater": "coord_descent", "feature_selector": "cyclic"},
        DMatrix(x, y),
        50,
    )
    assert rmse(linear.predict(x), y) < 1.0
    dart = hessboost.train({"booster": "dart", "rate_drop": 0.1, "seed": 2}, DMatrix(x, y), 20)
    assert rmse(dart.predict(x), y) < 0.5


def test_distributional_objective_predicts_distributions() -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(-1, 1, size=(600, 2))
    scale = 0.2 + np.abs(x[:, 0])
    y = 3 * x[:, 1] + rng.normal(0, scale)
    booster = hessboost.train(
        {"objective": "dist:normal", "max_depth": 2, "eta": 0.1}, DMatrix(x, y), 300
    )
    dists = booster.predict_distribution(x)
    assert isinstance(dists, hessboost.Distributions)
    assert len(dists) == 600
    assert dists.family == "dist:normal"
    assert dists.param_names == ["mu", "sigma"]
    params = dists.params
    assert params.shape == (600, 2)
    np.testing.assert_array_equal(dists.mean(), params[:, 0])
    np.testing.assert_allclose(dists.std(), params[:, 1])
    np.testing.assert_allclose(dists.variance(), params[:, 1] ** 2)
    # The spread follows the noise scale.
    assert np.corrcoef(dists.std(), scale)[0, 1] > 0.8
    interval = dists.interval(0.9)
    assert interval.shape == (600, 2)
    assert 0.8 < np.mean((y >= interval[:, 0]) & (y <= interval[:, 1])) < 0.97
    np.testing.assert_allclose(dists.quantile(0.5), dists.mean())
    np.testing.assert_allclose(dists.cdf(dists.mean().astype(np.float32)), 0.5, atol=1e-6)
    assert np.all(np.isfinite(dists.log_prob(y))) and np.all(dists.crps(y) >= 0)
    with pytest.raises(HessboostError, match="600 distributions"):
        dists.cdf(y[:5])
    point = hessboost.train({}, DMatrix(x, y), 5)
    with pytest.raises(HessboostError, match="dist"):
        point.predict_distribution(x)


def test_early_stopping_records_the_best_iteration() -> None:
    x, y = regression(rows=300)
    dtrain = DMatrix(x[:200], y[:200])
    dvalid = DMatrix(x[200:], y[200:])
    history: hessboost.EvalsResult = {}
    booster = hessboost.train(
        {"eta": 0.5, "max_depth": 6},
        dtrain,
        500,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=5,
        evals_result=history,
        verbose_eval=False,
    )
    best = booster.best_iteration
    assert best is not None
    assert booster.num_boosted_rounds() == best + 6
    scores = history["valid"]["rmse"]
    assert booster.best_score == scores[best] == min(scores)
    # Plain prediction stops at the best iteration.
    np.testing.assert_array_equal(
        booster.predict(x), booster.predict(x, iteration_range=(0, best + 1))
    )
    assert not np.array_equal(booster.predict(x), booster.predict(x, iteration_range=(0, 0)))


def test_early_stopping_needs_an_eval_set() -> None:
    x, y = regression(rows=50)
    with pytest.raises(HessboostError):
        hessboost.train({}, DMatrix(x, y), 5, early_stopping_rounds=2)


def test_continued_training_matches_one_run() -> None:
    x, y = regression()
    dtrain = DMatrix(x, y)
    params = {"subsample": 0.7, "colsample_bytree": 0.8, "seed": 11}
    whole = hessboost.train(params, dtrain, 15)
    first = hessboost.train(params, dtrain, 10)
    continued = hessboost.train(params, dtrain, 5, xgb_model=first)
    assert continued.num_boosted_rounds() == 15
    assert first.num_boosted_rounds() == 10
    np.testing.assert_array_equal(continued.predict(x), whole.predict(x))


def test_continued_training_from_a_model_file(tmp_path: object) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    x, y = regression()
    dtrain = DMatrix(x, y)
    first = hessboost.train({}, dtrain, 4)
    first.save_model(tmp_path / "first.bin")
    continued = hessboost.train({}, dtrain, 3, xgb_model=tmp_path / "first.bin")
    np.testing.assert_array_equal(
        continued.predict(x), hessboost.train({}, dtrain, 3, xgb_model=first).predict(x)
    )


def test_refresh_updates_leaves_on_new_data() -> None:
    x, y = regression()
    old = hessboost.train({"max_depth": 3}, DMatrix(x[:200], y[:200]), 10)
    refreshed = hessboost.train(
        {"max_depth": 3, "process_type": "update", "refresh_leaf": True},
        DMatrix(x[200:], y[200:] + 5.0),
        10,
        xgb_model=old,
    )
    assert refreshed.num_boosted_rounds() == 10
    # Same splits, leaves moved toward the shifted labels.
    np.testing.assert_array_equal(refreshed.predict(x, pred_leaf=True), old.predict(x, pred_leaf=True))
    assert np.mean(refreshed.predict(x) - old.predict(x)) > 1.0


def test_base_margin_offsets_training_and_prediction() -> None:
    x, y = regression()
    offset = np.full(len(y), 10.0)
    booster = hessboost.train({"base_score": 0.0}, DMatrix(x, y + offset, base_margin=offset), 20)
    with_margin = booster.predict(x, base_margin=offset)
    assert rmse(with_margin, y + offset) < 0.5
    dtest = DMatrix(x, base_margin=offset)
    np.testing.assert_array_equal(booster.predict(dtest), with_margin)
    # Without the offset, the zero intercept predicts the residual part.
    assert rmse(booster.predict(x), y) < 0.5
    with pytest.raises(HessboostError, match="base_margin"):
        booster.predict(dtest, base_margin=offset)


def test_weights_change_the_fit() -> None:
    x, y = regression()
    w = np.where(x[:, 0] > 0, 10.0, 0.1)
    plain = hessboost.train({}, DMatrix(x, y), 5).predict(x)
    weighted = hessboost.train({}, DMatrix(x, y, weight=w), 5).predict(x)
    heavy = x[:, 0] > 0
    assert rmse(weighted[heavy], y[heavy]) < rmse(plain[heavy], y[heavy])


def test_custom_objective_reproduces_squared_error() -> None:
    x, y = regression()
    dtrain = DMatrix(x, y)
    seen: list[tuple[int, ...]] = []

    def squared_error(
        margins: NDArray[np.float32], data: DMatrix
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        seen.append(margins.shape)
        return margins - data.get_label(), np.ones_like(margins)

    custom = hessboost.train({"base_score": 0.0}, dtrain, 10, obj=squared_error)
    builtin = hessboost.train({"base_score": 0.0}, dtrain, 10)
    assert seen == [(400,)] * 10
    np.testing.assert_array_equal(custom.predict(x), builtin.predict(x))


def test_custom_objective_errors_propagate() -> None:
    x, y = regression(rows=50)

    def broken(margins: NDArray[np.float32], data: DMatrix) -> tuple[list[float], list[float]]:
        raise RuntimeError("objective exploded")

    with pytest.raises(RuntimeError, match="objective exploded"):
        hessboost.train({}, DMatrix(x, y), 3, obj=broken)

    def short(margins: NDArray[np.float32], data: DMatrix) -> tuple[list[float], list[float]]:
        return [0.0], [1.0]

    with pytest.raises(HessboostError, match="1 gradients and 1 hessians for 50"):
        hessboost.train({}, DMatrix(x, y), 3, obj=short)


def test_custom_metric_drives_early_stopping() -> None:
    x, y = regression(rows=300)
    dvalid = DMatrix(x[200:], y[200:])
    calls: list[int] = []

    def mae(
        predictions: NDArray[np.float32],
        labels: NDArray[np.float32],
        weights: NDArray[np.float32] | None,
    ) -> float:
        calls.append(len(predictions))
        assert weights is None
        return float(np.mean(np.abs(predictions - labels)))

    history: hessboost.EvalsResult = {}
    booster = hessboost.train(
        {"eta": 0.5},
        DMatrix(x[:200], y[:200]),
        300,
        evals=[(dvalid, "valid")],
        custom_metric=mae,
        early_stopping_rounds=4,
        evals_result=history,
        verbose_eval=False,
    )
    assert list(history["valid"]) == ["mae"]
    assert set(calls) == {100}
    best = booster.best_iteration
    assert best is not None and booster.best_score == min(history["valid"]["mae"])

    with pytest.raises(HessboostError, match="maximize"):
        hessboost.train({}, DMatrix(x, y), 2, maximize=True)


def test_training_is_identical_at_any_thread_count() -> None:
    x, y = regression(rows=3000, features=8)
    dtrain = DMatrix(x, y)
    params = {"subsample": 0.8, "colsample_bynode": 0.7, "seed": 5, "max_depth": 5}
    models = {
        hessboost.train({**params, "nthread": n}, dtrain, 20).save_raw() for n in (1, 2, 4, 0)
    }
    assert len(models) == 1
    for method in ["exact", "approx"]:
        pair = {
            hessboost.train({**params, "tree_method": method, "nthread": n}, dtrain, 5).save_raw()
            for n in (1, 3)
        }
        assert len(pair) == 1


def test_cv_reports_test_metrics_per_round() -> None:
    x, y = regression()
    dtrain = DMatrix(x, y)
    result = hessboost.cv({"max_depth": 3, "eval_metric": ["mae", "rmse"]}, dtrain, 8, nfold=4)
    assert set(result) == {"test-mae-mean", "test-mae-std", "test-rmse-mean", "test-rmse-std"}
    assert result["test-rmse-mean"].shape == (8,)
    assert result["test-rmse-mean"][-1] < result["test-rmse-mean"][0]
    assert np.all(result["test-rmse-std"] >= 0)
    stopped = hessboost.cv({"eta": 1.0}, dtrain, 200, early_stopping_rounds=3)
    assert len(stopped["test-rmse-mean"]) < 200
    assert np.argmin(stopped["test-rmse-mean"]) == len(stopped["test-rmse-mean"]) - 1


def test_cv_accepts_explicit_folds_and_splitters() -> None:
    from sklearn.model_selection import KFold  # type: ignore[import-untyped]

    x, y = regression()
    dtrain = DMatrix(x, y)
    chained = hessboost.cv({}, dtrain, 5, folds=hessboost.folds.forward_chaining(400, 3, gap=5))
    assert chained["test-rmse-mean"].shape == (5,)
    kfold = KFold(3, shuffle=True, random_state=0)
    by_splitter = hessboost.cv({}, dtrain, 5, folds=kfold)
    by_pairs = hessboost.cv({}, dtrain, 5, folds=list(kfold.split(x)))
    np.testing.assert_array_equal(by_splitter["test-rmse-mean"], by_pairs["test-rmse-mean"])


def test_folds() -> None:
    folds = hessboost.folds.k_fold(10, 3, seed=1)
    assert len(folds) == 3
    tests = np.concatenate([test for _, test in folds])
    np.testing.assert_array_equal(np.sort(tests), np.arange(10))
    for train_rows, test_rows in folds:
        assert train_rows.dtype == np.int64
        assert not set(train_rows) & set(test_rows)
    assert [t.tolist() for t in hessboost.folds.k_fold(10, 3, seed=1)[0]] == [
        t.tolist() for t in folds[0]
    ]
    chained = hessboost.folds.forward_chaining(20, 3, gap=2)
    assert [(tr.tolist(), te.tolist()) for tr, te in chained] == [
        (list(range(3)), list(range(5, 10))),
        (list(range(8)), list(range(10, 15))),
        (list(range(13)), list(range(15, 20))),
    ]
    with pytest.raises(HessboostError, match="gap"):
        hessboost.folds.forward_chaining(20, 3, gap=5)
    with pytest.raises(HessboostError, match="nfold"):
        hessboost.folds.k_fold(10, 1)



def test_purged_forward_folds_purge_by_each_rows_label_window() -> None:
    day = 86_400
    # Three rows per day for 10 days; labels end two days after decision.
    decision_at = np.repeat(np.arange(10) * day, 3)
    label_end = decision_at + 2 * day
    (train_rows, test_rows), = hessboost.folds.purged_forward(
        decision_at, label_end, validation_fraction=0.2
    )
    # Days 8 and 9 test; day 7's labels reach past day 8's start and are
    # purged, day 6's end exactly at it and train.
    np.testing.assert_array_equal(decision_at[test_rows], np.repeat([8 * day, 9 * day], 3))
    assert decision_at[train_rows].max() == 6 * day
    assert len(train_rows) == 21
    # Row-varying windows: one short-label row per day survives the purge.
    short = label_end.copy()
    short[::3] = decision_at[::3] + 1
    (train_rows, _), = hessboost.folds.purged_forward(decision_at, short, validation_fraction=0.2)
    np.testing.assert_array_equal(np.sort(train_rows)[-1:], [21])
    assert set(decision_at[train_rows] // day) == set(range(8))
    # One tick past the block start purges day 6 too.
    later = label_end + 1
    (train_rows, _), = hessboost.folds.purged_forward(decision_at, later, validation_fraction=0.2)
    assert decision_at[train_rows].max() == 5 * day
    # datetime64 works, rows in any order, several blocks.
    stamps = np.datetime64("2026-01-01") + decision_at.astype("timedelta64[s]")
    ends = np.datetime64("2026-01-01") + label_end.astype("timedelta64[s]")
    order = np.random.default_rng(0).permutation(len(stamps))
    folds = hessboost.folds.purged_forward(
        stamps[order], ends[order], validation_fraction=0.4, blocks=2
    )
    assert len(folds) == 2
    for train_rows, test_rows in folds:
        assert stamps[order][train_rows].max() < stamps[order][test_rows].min()
    with pytest.raises(HessboostError, match="before its decision"):
        hessboost.folds.purged_forward([5, 6], [4, 7], validation_fraction=0.5)
    with pytest.raises(HessboostError, match="min_train|purged training rows"):
        hessboost.folds.purged_forward(decision_at, label_end, validation_fraction=0.2, min_train=100)
    with pytest.raises(TypeError, match="integer times"):
        hessboost.folds.purged_forward([0.5], [1.0], validation_fraction=0.5)


class Recorder(hessboost.TrainingCallback):
    """Records every call; stops at ``stop_at``."""

    def __init__(self, stop_at: int | None = None) -> None:
        self.calls: list[tuple[int, dict[str, dict[str, int]]]] = []
        self.stop_at = stop_at

    def after_iteration(self, iteration: int, evals_log: hessboost.EvalsResult) -> bool:
        lengths = {data: {m: len(v) for m, v in metrics.items()} for data, metrics in evals_log.items()}
        self.calls.append((iteration, lengths))
        return iteration == self.stop_at


def test_callbacks_see_every_round_and_can_stop_training() -> None:
    x, y = regression()
    dtrain = DMatrix(x[:300], y[:300])
    dvalid = DMatrix(x[300:], y[300:])
    watcher, stopper = Recorder(), Recorder(stop_at=6)
    history: hessboost.EvalsResult = {}
    booster = hessboost.train(
        {"subsample": 0.8, "seed": 2},
        dtrain,
        1000,
        evals=[(dvalid, "valid")],
        evals_result=history,
        verbose_eval=False,
        callbacks=[watcher, stopper],
    )
    assert booster.num_boosted_rounds() == 7
    assert [iteration for iteration, _ in watcher.calls] == list(range(7))
    # The log grows by one value per round and is complete when called.
    assert [lengths["valid"]["rmse"] for _, lengths in watcher.calls] == list(range(1, 8))
    assert len(history["valid"]["rmse"]) == 7
    # Stopping keeps the rounds so far: the same model as 7 rounds.
    short = hessboost.train({"subsample": 0.8, "seed": 2}, dtrain, 7, verbose_eval=False)
    assert booster.save_raw() == short.save_raw()
    # Without eval sets callbacks still run every round.
    quiet = Recorder(stop_at=2)
    assert hessboost.train({}, dtrain, 50, callbacks=[quiet]).num_boosted_rounds() == 3
    assert quiet.calls == [(0, {}), (1, {}), (2, {})]


def test_stopping_under_early_stopping_keeps_the_best_round_so_far() -> None:
    x, y = regression(rows=300)
    dtrain = DMatrix(x[:200], y[:200])
    dvalid = DMatrix(x[200:], y[200:])
    history: hessboost.EvalsResult = {}
    booster = hessboost.train(
        {"eta": 0.5},
        dtrain,
        500,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=50,
        evals_result=history,
        verbose_eval=False,
        callbacks=[Recorder(stop_at=8)],
    )
    scores = history["valid"]["rmse"]
    assert booster.num_boosted_rounds() == len(scores) == 9
    assert booster.best_iteration == int(np.argmin(scores))
    assert booster.best_score == min(scores)


def test_callback_exceptions_stop_training_and_propagate() -> None:
    x, y = regression()
    calls: list[int] = []

    class Broken(hessboost.TrainingCallback):
        def after_iteration(self, iteration: int, evals_log: hessboost.EvalsResult) -> bool:
            calls.append(iteration)
            if iteration == 2:
                raise ValueError("callback exploded")
            return False

    with pytest.raises(ValueError, match="callback exploded"):
        hessboost.train({}, DMatrix(x, y), 1000, callbacks=[Broken()])
    assert calls == [0, 1, 2]


def test_verbose_eval_prints_each_round_as_it_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    import sys

    x, y = regression(rows=100)
    dtrain = DMatrix(x, y)
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    printed: list[str] = []

    class Snapshot(hessboost.TrainingCallback):
        def after_iteration(self, iteration: int, evals_log: hessboost.EvalsResult) -> bool:
            printed.append(out.getvalue().splitlines()[-1].split("\t")[0])
            return False

    hessboost.train({}, dtrain, 4, evals=[(dtrain, "train")], callbacks=[Snapshot()])
    # Each round's line is out before the round's callbacks run.
    assert printed == ["[0]", "[1]", "[2]", "[3]"]


@pytest.mark.parametrize("nthread", [0, 2])
def test_keyboard_interrupt_stops_training_promptly(nthread: int) -> None:
    import _thread
    import threading
    import time

    x, y = regression(rows=2000)
    dtrain = DMatrix(x, y)
    watcher = Recorder()
    timer = threading.Timer(0.3, _thread.interrupt_main)
    started = time.perf_counter()
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            hessboost.train(
                {"max_depth": 6, "nthread": nthread}, dtrain, 1_000_000, callbacks=[watcher]
            )
    finally:
        timer.cancel()
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0
    assert 0 < len(watcher.calls) < 1_000_000
    # Rounds ran until the interrupt, then stopped within about one poll.
    assert elapsed > 0.25
