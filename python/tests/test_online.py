"""In-place updates (``hessboost.online``): the exact mode equals retraining
bit for bit, the approximate mode reports what it did, and refused, stopped,
or interrupted updates change nothing."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import hessboost
from conftest import classes, regression
from hessboost import DMatrix, HessboostError
from hessboost.online import OnlineModel, UpdateReport

PARAMS = {"tree_method": "hist", "max_depth": 4, "eta": 0.3}
ROUNDS = 12


def binary(rows: int, seed: int) -> DMatrix:
    x, y = classes(rows=rows, seed=seed)
    return DMatrix(x, y)


def state(online: OnlineModel) -> tuple[bytes, int]:
    return online.model.save_raw(), online.num_row()


@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic"])
def test_exact_updates_equal_training_on_the_data(objective: str) -> None:
    params = {**PARAMS, "objective": objective}
    if objective == "binary:logistic":
        dtrain, added = binary(300, 0), binary(20, 1)
    else:
        x, y = regression(rows=320)
        dtrain, added = DMatrix(x[:300], y[:300]), DMatrix(x[300:], y[300:])
    online = OnlineModel.train(params, dtrain, ROUNDS, tolerance=0.0)
    assert online.tolerance == 0.0
    changes: list[tuple[DMatrix | None, list[int] | NDArray[np.int64]]] = [
        (added, [0, 7, 299]),
        (None, np.array([3, 4, 5])),
        (added, []),
    ]
    for additions, deletions in changes:
        report = online.update(additions, deletions)
        assert report.subtrees_regrown == ROUNDS
        retrained = hessboost.train(params, online.data, ROUNDS)
        assert online.model.save_raw() == retrained.save_raw()
        assert online.model.save_raw("json") == retrained.save_raw("json")
    assert online.num_row() == online.data.num_row() == 300 - 3 + 20 - 3 + 20


def test_approximate_updates_report_and_stay_close_to_retraining() -> None:
    x, y = regression(rows=2400)
    dtrain, test = DMatrix(x[:2000], y[:2000]), x[2000:2360]
    added = DMatrix(x[2360:], y[2360:])
    online = OnlineModel.train(PARAMS, dtrain, ROUNDS)
    assert online.tolerance == 0.1
    report = online.update(added, deletions=list(range(0, 400, 10)))
    assert isinstance(report, UpdateReport)
    assert report.nodes_kept > 0
    assert 0 < report.rows_refreshed <= online.num_row() == 2000
    retrained = hessboost.train(PARAMS, online.data, ROUNDS)
    updated, reference = online.model.predict(test), retrained.predict(test)
    target = y[2000:2360]
    rmse = [float(np.sqrt(np.mean((p - target) ** 2))) for p in (updated, reference)]
    assert rmse[0] == pytest.approx(rmse[1], rel=0.05)


def test_the_model_is_a_snapshot() -> None:
    online = OnlineModel.train(PARAMS, binary(200, 2), ROUNDS)
    before = online.model
    saved = before.save_raw()
    online.update(binary(10, 3), [0])
    assert before.save_raw() == saved != online.model.save_raw()


@pytest.mark.parametrize("tolerance", [0.1, 0.0])
def test_from_model_resumes_like_the_trained_online_model(tolerance: float) -> None:
    dtrain, added = binary(300, 4), binary(15, 5)
    trained = OnlineModel.train(PARAMS, dtrain, ROUNDS, tolerance)
    loaded = hessboost.Booster(trained.model.save_raw())
    resumed = OnlineModel.from_model(loaded, PARAMS, dtrain, tolerance)
    for online in (trained, resumed):
        online.update(added, [1, 2])
    assert trained.model.save_raw() == resumed.model.save_raw()
    with pytest.raises(HessboostError, match="model"):
        OnlineModel.from_model(loaded, {**PARAMS, "objective": "reg:logistic"}, dtrain)


def test_from_model_refuses_an_early_stopped_model() -> None:
    x, y = regression(rows=400)
    dtrain, dvalid = DMatrix(x[:300], y[:300]), DMatrix(x[300:], y[300:])
    stopped = hessboost.train(
        PARAMS,
        dtrain,
        200,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=2,
        verbose_eval=False,
    )
    best = stopped.best_iteration
    assert best is not None
    with pytest.raises(HessboostError, match="early-stopped"):
        OnlineModel.from_model(stopped, PARAMS, dtrain)
    resumed = OnlineModel.from_model(stopped[: best + 1], PARAMS, dtrain)
    assert resumed.model.num_boosted_rounds() == best + 1


@pytest.mark.parametrize("tolerance", [0.1, 0.0])
def test_a_stopping_callback_abandons_the_update(tolerance: float) -> None:
    online = OnlineModel.train(PARAMS, binary(300, 6), ROUNDS, tolerance)
    before = state(online)
    calls: list[int] = []

    def stop_at_3(iteration: int) -> bool:
        calls.append(iteration)
        return iteration == 3

    assert online.update(binary(10, 7), [0, 1], callback=stop_at_3) is None
    assert calls == [0, 1, 2, 3]
    assert state(online) == before

    def explode(iteration: int) -> bool:
        if iteration == 2:
            raise ValueError("callback exploded")
        return False

    with pytest.raises(ValueError, match="callback exploded"):
        online.update(binary(10, 7), [0, 1], callback=explode)
    assert state(online) == before
    # A callback that never stops sees every iteration, and the update then
    # equals an uninterrupted one.
    calls.clear()

    def watch(iteration: int) -> bool:
        calls.append(iteration)
        return False

    report = online.update(binary(10, 7), [0, 1], callback=watch)
    assert report is not None
    assert calls == list(range(ROUNDS))
    fresh = OnlineModel.train(PARAMS, binary(300, 6), ROUNDS, tolerance)
    fresh.update(binary(10, 7), [0, 1])
    assert state(online) == state(fresh)


@pytest.mark.parametrize("tolerance", [0.1, 0.0])
@pytest.mark.parametrize("at", [2, ROUNDS - 1])
@pytest.mark.parametrize("wait", [0.3, 0.0])
def test_keyboard_interrupt_abandons_the_update(tolerance: float, at: int, wait: float) -> None:
    """Ctrl-C during any iteration's callback, the last included, and
    whether or not the waiting caller sees it before the callback returns,
    raises with the state unchanged."""
    import _thread
    import time

    online = OnlineModel.train(PARAMS, binary(300, 8), ROUNDS, tolerance)
    before = state(online)

    def interrupt(iteration: int) -> bool:
        if iteration == at:
            _thread.interrupt_main()
            time.sleep(wait)
        return False

    with pytest.raises(KeyboardInterrupt):
        online.update(binary(10, 9), [5], callback=interrupt)
    assert state(online) == before
    assert online.update(binary(10, 9), [5]).rows_refreshed > 0
    assert online.num_row() == 309


_REENTRANT = """
import numpy as np
import hessboost
from hessboost import DMatrix, HessboostError
from hessboost.online import OnlineModel

rng = np.random.default_rng(0)
x = rng.normal(size=(300, 4))
y = x[:, 0] - x[:, 1]
online = OnlineModel.train(
    {"tree_method": "hist", "max_depth": 3}, DMatrix(x, y), 6, tolerance=TOLERANCE
)
seen = []

def callback(iteration):
    seen.append((online.num_row(), online.tolerance, repr(online)))
    for access in (lambda: online.model, lambda: online.data, lambda: online.update(None, [0])):
        try:
            access()
        except HessboostError as error:
            assert "being updated" in str(error), error
        else:
            raise AssertionError("reentrant access was not refused")
    return False

report = online.update(DMatrix(x[:5], y[:5]), [1, 2], callback=callback)
assert report is not None
assert [rows for rows, _, _ in seen] == [300] * 6, seen
assert online.num_row() == 303 == online.data.num_row()
assert online.model.num_boosted_rounds() == 6
print("ok")
"""


@pytest.mark.parametrize("tolerance", [0.1, 0.0])
def test_access_from_the_update_callback_fails_fast(tolerance: float) -> None:
    """Reading the model or data, or updating again, from an update's own
    callback raises instead of deadlocking; the row count and tolerance stay
    readable. Run in a subprocess, so a deadlock fails the test by timeout."""
    import subprocess
    import sys

    code = _REENTRANT.replace("TOLERANCE", repr(tolerance))
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


@pytest.mark.parametrize("tolerance", [0.1, 0.0])
def test_labels_retraining_refuses_are_refused(tolerance: float) -> None:
    params = {**PARAMS, "objective": "binary:logistic"}
    x, y = classes(rows=300, seed=10)
    online = OnlineModel.train(params, DMatrix(x, y), ROUNDS, tolerance)
    before = state(online)
    bad = DMatrix(x[:1], [2.0])
    with pytest.raises(HessboostError, match="labels"):
        hessboost.train(params, bad, ROUNDS)
    with pytest.raises(HessboostError, match="labels"):
        online.update(bad, [0])
    assert state(online) == before
    online.update(DMatrix(x[:3], y[:3]), [0])
    assert online.num_row() == 302


def test_approximate_updates_refuse_values_beyond_the_training_bins() -> None:
    x = np.array([[0.0], [1.0], [np.nan], [np.nan]], dtype=np.float32)
    params = {**PARAMS, "max_depth": 1}
    approximate = OnlineModel.train(params, DMatrix(x, [0.0, 0.0, 1.0, 1.0]), 1)
    before = state(approximate)
    with pytest.raises(HessboostError, match="additions"):
        approximate.update(DMatrix([[3.0]], [0.0]), [0])
    assert state(approximate) == before
    exact = OnlineModel.train(params, DMatrix(x, [0.0, 0.0, 1.0, 1.0]), 1, 0.0)
    exact.update(DMatrix([[3.0]], [0.0]), [0])
    assert exact.num_row() == 4


def test_an_update_that_overflows_is_refused() -> None:
    top = float(np.finfo(np.float32).max)
    online = OnlineModel.train({"base_score": top}, DMatrix([[0.0]], [top]), 2)
    before = state(online)
    with pytest.raises(hessboost.ModelFormatError):
        online.update(DMatrix([[0.0]], [-top]))
    assert state(online) == before


def test_unsupported_configurations_and_changes_are_refused() -> None:
    x, y = regression(rows=200)
    dtrain = DMatrix(x, y, feature_names=[f"f{i}" for i in range(5)])
    refused: list[dict[str, object]] = [
        {"subsample": 0.8},
        {"booster": "dart"},
        {"objective": "reg:absoluteerror"},
    ]
    for params in refused:
        with pytest.raises(HessboostError):
            OnlineModel.train({**PARAMS, **params}, dtrain, 3)
    with pytest.raises(HessboostError, match="tolerance"):
        OnlineModel.train(PARAMS, dtrain, 3, tolerance=1.5)
    with pytest.raises(HessboostError, match="weights"):
        OnlineModel.train(PARAMS, DMatrix(x, y, weight=np.ones(200)), 3)
    online = OnlineModel.train(PARAMS, dtrain, 3)
    before = state(online)
    for deletions in ([200], [4, 4], list(range(200)), [-1]):
        with pytest.raises(HessboostError):
            online.update(None, deletions)
    with pytest.raises(HessboostError, match="feature names"):
        online.update(DMatrix(x[:2], y[:2], feature_names=list("abcde")))
    with pytest.raises(HessboostError, match="additions"):
        online.update(DMatrix(x[:2]))
    assert state(online) == before
    with pytest.raises(TypeError):
        online.update(x[:2])  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        online.update(None, [0.5])  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        OnlineModel.train(PARAMS, dtrain, 3, tolerance="0.1")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        OnlineModel()


@pytest.mark.parametrize("tolerance", [0.1, 0.0])
def test_an_interrupted_update_keeps_the_update_state(tolerance: float) -> None:
    """Ctrl-C at the last iteration (refused at the commit gate) after an
    earlier update restores the exact update state: later updates equal
    those of a model that never tried the interrupted one."""
    import _thread

    dtrain, a, b, c = binary(400, 20), binary(20, 21), binary(15, 22), binary(25, 23)
    online, control = (OnlineModel.train(PARAMS, dtrain, ROUNDS, tolerance) for _ in range(2))
    for model in (online, control):
        model.update(a, [0, 5, 9])

    def interrupt_at_last(iteration: int) -> bool:
        if iteration == ROUNDS - 1:
            _thread.interrupt_main()
        return False

    with pytest.raises(KeyboardInterrupt):
        online.update(b, [1, 2], callback=interrupt_at_last)
    assert state(online) == state(control)
    for additions, deletions in [(c, [3, 7]), (None, [0, 1, 40])]:
        assert online.update(additions, deletions) == control.update(additions, deletions)
        assert state(online) == state(control)
