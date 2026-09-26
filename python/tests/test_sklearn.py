"""The scikit-learn estimators: scikit-learn's own estimator checks, label
encoding, eval sets, pandas input, and model selection."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import classes, regression
from sklearn.base import clone  # type: ignore[import-untyped]
from sklearn.model_selection import GridSearchCV  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from sklearn.utils.estimator_checks import parametrize_with_checks  # type: ignore[import-untyped]

import hessboost
from hessboost import HessboostError
from hessboost.sklearn import (
    HessboostClassifier,
    HessboostDistributionRegressor,
    HessboostRanker,
    HessboostRegressor,
)

ZERO_WEIGHTS = (
    "the refusal of all-zero weights says 'at least one weight must be positive', not the "
    "message the check matches"
)
SPLIT_CANDIDATES = (
    "zero-weight rows still place split candidates (as in XGBoost), so a zero-weight row can "
    "fall on the other side of a threshold than when it is removed"
)


def expected_failures(estimator: Any) -> dict[str, str]:
    failures = {"check_all_zero_sample_weights_error": ZERO_WEIGHTS}
    if isinstance(estimator, HessboostRegressor):
        failures["check_regressor_multioutput"] = (
            "predictions are float32, the precision hessboost (like XGBoost) computes in"
        )
    if isinstance(estimator, HessboostClassifier):
        failures["check_sample_weight_equivalence_on_dense_data"] = SPLIT_CANDIDATES
        failures["check_sample_weight_equivalence_on_sparse_data"] = SPLIT_CANDIDATES
    return failures


@parametrize_with_checks(  # type: ignore[untyped-decorator]
    [
        HessboostRegressor(n_estimators=20),
        HessboostClassifier(n_estimators=20),
        HessboostDistributionRegressor(n_estimators=50),
    ],
    expected_failed_checks=expected_failures,
)
def test_scikit_learn_estimator_checks(estimator: Any, check: Any) -> None:
    check(estimator)


def test_classifier_encodes_arbitrary_labels() -> None:
    x, y = classes(n_classes=3)
    names = np.array(["setosa", "versicolor", "virginica"])[y]
    model = HessboostClassifier(n_estimators=30, max_depth=3).fit(x, names)
    np.testing.assert_array_equal(model.classes_, ["setosa", "versicolor", "virginica"])
    assert model.n_classes_ == 3
    assert model.get_booster().objective == "multi:softprob"
    probabilities = model.predict_proba(x)
    assert probabilities.shape == (400, 3)
    np.testing.assert_array_equal(model.predict(x), model.classes_[probabilities.argmax(axis=1)])
    assert model.score(x, names) > 0.8
    binary = HessboostClassifier(n_estimators=10).fit(x, y > 0)
    assert binary.get_booster().objective == "binary:logistic"
    np.testing.assert_allclose(binary.predict_proba(x).sum(axis=1), 1.0, rtol=1e-6)
    assert binary.predict(x).dtype == bool


def test_classifier_objectives_without_probabilities() -> None:
    x, y = classes()
    hinge = HessboostClassifier(n_estimators=10, objective="binary:hinge").fit(x, y)
    assert set(np.unique(hinge.predict(x))) <= {0, 1}
    with pytest.raises(HessboostError, match="probabilities"):
        hinge.predict_proba(x)


def test_early_stopping_with_eval_sets() -> None:
    x, y = regression(rows=600)
    model = HessboostRegressor(
        n_estimators=1000, learning_rate=0.5, early_stopping_rounds=5, eval_metric="mae"
    )
    model.fit(x[:400], y[:400], eval_set=[(x[:400], y[:400]), (x[400:], y[400:])])
    history = model.evals_result()
    assert set(history) == {"validation_0", "validation_1"}
    assert list(history["validation_1"]) == ["mae"]
    best = model.best_iteration
    assert best is not None and best < 999
    assert model.best_score == min(history["validation_1"]["mae"])
    classifier = HessboostClassifier(n_estimators=100, early_stopping_rounds=3)
    x, labels = classes()
    words = np.array(["no", "yes"])[labels]
    classifier.fit(x[:300], words[:300], eval_set=[(x[300:], words[300:])])
    assert classifier.best_iteration is not None
    with pytest.raises(HessboostError, match="not in the training classes"):
        classifier.fit(x[:300], words[:300], eval_set=[(x[300:], np.full(100, "maybe"))])


def test_pandas_input_keeps_names_and_categories() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "kind": pd.Categorical(rng.choice(["a", "b", "c"], 300)),
            "value": rng.normal(size=300),
        }
    )
    y = df["kind"].map({"a": 0.0, "b": 5.0, "c": -5.0}).to_numpy() + df["value"].to_numpy()
    model = HessboostRegressor(n_estimators=50, max_depth=3).fit(df, y)
    np.testing.assert_array_equal(model.feature_names_in_, ["kind", "value"])
    assert model.n_features_in_ == 2
    assert model.get_booster().feature_types == ["c", "q"]
    assert model.score(df, y) > 0.95
    importances = model.feature_importances_
    assert importances.shape == (2,)
    assert importances.sum() == pytest.approx(1.0)
    assert importances[0] > importances[1]


def test_regressor_multi_target_and_continued_fitting() -> None:
    x, y = regression()
    targets = np.column_stack([y, -y])
    model = HessboostRegressor(n_estimators=20).fit(x, targets)
    assert model.predict(x).shape == (400, 2)
    first = HessboostRegressor(n_estimators=10, random_state=0).fit(x, y)
    continued = HessboostRegressor(n_estimators=5, random_state=0).fit(x, y, xgb_model=first)
    assert continued.get_booster().num_boosted_rounds() == 15
    whole = HessboostRegressor(n_estimators=15, random_state=0).fit(x, y)
    np.testing.assert_array_equal(continued.predict(x), whole.predict(x))


def test_params_passes_other_options_and_refuses_conflicts() -> None:
    x, y = regression()
    smooth = HessboostRegressor(n_estimators=5, params={"path_smooth": 5.0}).fit(x, y)
    plain = HessboostRegressor(n_estimators=5).fit(x, y)
    assert not np.array_equal(smooth.predict(x), plain.predict(x))
    with pytest.raises(HessboostError, match="both"):
        HessboostRegressor(max_depth=3, params={"max_depth": 4}).fit(x, y)
    with pytest.raises(HessboostError, match="unknown parameter"):
        HessboostRegressor(params={"max_dept": 4}).fit(x, y)


def test_ranker() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 3))
    relevance = np.clip(np.round(x[:, 0] + 1), 0, 3)
    qid = np.repeat(np.arange(30), 10)
    ranker = HessboostRanker(n_estimators=20, eval_metric="ndcg@5", early_stopping_rounds=5)
    ranker.fit(x[:200], relevance[:200], qid=qid[:200], eval_set=[(x[200:], relevance[200:])],
               eval_qid=[qid[200:]])
    assert list(ranker.evals_result()["validation_0"]) == ["ndcg@5"]
    scores = ranker.predict(x)
    assert np.corrcoef(scores, relevance)[0, 1] > 0.8
    by_group = HessboostRanker(n_estimators=20).fit(x, relevance, group=[10] * 30)
    by_qid = HessboostRanker(n_estimators=20).fit(x, relevance, qid=qid)
    np.testing.assert_array_equal(by_group.predict(x), by_qid.predict(x))
    with pytest.raises(HessboostError, match="group or qid"):
        HessboostRanker().fit(x, relevance)


def test_distribution_regressor() -> None:
    rng = np.random.default_rng(4)
    x = rng.uniform(-1, 1, size=(500, 2))
    y = np.exp(x[:, 0] + rng.normal(0, 0.3, 500))
    model = HessboostDistributionRegressor(objective="dist:lognormal", n_estimators=100)
    model.fit(x, y)
    dists = model.predict_distribution(x)
    assert dists.family == "dist:lognormal"
    np.testing.assert_array_equal(model.predict(x), dists.mean())
    with pytest.raises(HessboostError, match="dist"):
        HessboostDistributionRegressor(objective="reg:squarederror").fit(x, y)


def test_estimators_work_in_model_selection_and_pickle() -> None:
    x, y = regression()
    search = GridSearchCV(
        make_pipeline(StandardScaler(), HessboostRegressor(n_estimators=10)),
        {"hessboostregressor__max_depth": [1, 4]},
        cv=3,
    ).fit(x, y)
    assert search.best_params_ == {"hessboostregressor__max_depth": 4}
    model = HessboostClassifier(n_estimators=5, max_depth=2).fit(*classes())
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict_proba(classes()[0]), model.predict_proba(classes()[0]))
    assert clone(model).get_params() == model.get_params()
    assert "max_depth=2" in repr(model)


def test_callbacks_and_verbose_pass_through(capsys: pytest.CaptureFixture[str]) -> None:
    class StopAt(hessboost.TrainingCallback):
        def after_iteration(self, iteration: int, evals_log: hessboost.EvalsResult) -> bool:
            return iteration == 4

    x, y = regression()
    model = HessboostRegressor(n_estimators=100, callbacks=[StopAt()])
    model.fit(x, y, eval_set=[(x, y)], verbose=2)
    assert model.get_booster().num_boosted_rounds() == 5
    assert [line.split("\t")[0] for line in capsys.readouterr().out.splitlines()] == [
        "[0]",
        "[2]",
        "[4]",
    ]


def test_unfitted_estimators_raise_not_fitted() -> None:
    from sklearn.exceptions import NotFittedError  # type: ignore[import-untyped]

    with pytest.raises(NotFittedError):
        HessboostRegressor().predict(np.zeros((1, 2)))
    with pytest.raises(NotFittedError):
        HessboostRegressor().get_booster()


def test_booster_from_estimator_saves_like_any_booster() -> None:
    x, y = regression()
    model = HessboostRegressor(n_estimators=5).fit(x, y)
    booster = hessboost.Booster(model.get_booster().save_raw())
    np.testing.assert_array_equal(booster.predict(x), model.predict(x))
