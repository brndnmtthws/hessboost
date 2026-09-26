"""Thread safety and GIL release: concurrent training and prediction on
shared objects (free-threaded builds included) give the serial results."""

from __future__ import annotations

import sys
import sysconfig
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from conftest import regression

import hessboost
from hessboost import DMatrix


def test_concurrent_prediction_on_one_booster_matches_serial() -> None:
    x, y = regression(rows=2000)
    booster = hessboost.train({"max_depth": 6}, DMatrix(x, y), 50)
    expected = booster.predict(x)
    contribs = booster.predict(x[:200], pred_contribs=True)
    with ThreadPoolExecutor(8) as pool:
        predictions = list(pool.map(lambda _: booster.predict(x), range(16)))
        shap = list(pool.map(lambda _: booster.predict(x[:200], pred_contribs=True), range(8)))
    for prediction in predictions:
        np.testing.assert_array_equal(prediction, expected)
    for values in shap:
        np.testing.assert_array_equal(values, contribs)


def test_concurrent_training_on_one_matrix_is_deterministic() -> None:
    x, y = regression(rows=2000)
    dtrain = DMatrix(x, y)
    params = {"subsample": 0.8, "seed": 3, "nthread": 2}
    expected = hessboost.train(params, dtrain, 10).save_raw()
    with ThreadPoolExecutor(6) as pool:
        models = list(pool.map(lambda _: hessboost.train(params, dtrain, 10).save_raw(), range(6)))
    assert set(models) == {expected}


def test_training_releases_the_gil() -> None:
    x, y = regression(rows=40_000, features=20)
    dtrain = DMatrix(x, y)
    done = threading.Event()
    stamps: list[float] = []

    def count() -> None:
        ticks = 0
        while not done.is_set():
            ticks += 1
            if ticks % 1000 == 0:
                stamps.append(time.perf_counter())

    counter = threading.Thread(target=count)
    counter.start()
    try:
        started = time.perf_counter()
        hessboost.train({"max_depth": 8, "nthread": 1}, dtrain, 30)
        elapsed = time.perf_counter() - started
    finally:
        done.set()
        counter.join()
    # Had the call held the GIL, the counter could not have run in the
    # middle of it.
    middle = [s for s in stamps if started + elapsed / 4 < s < started + 3 * elapsed / 4]
    assert elapsed > 0.05
    assert middle


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="not a free-threaded build"
)
def test_importing_the_extension_keeps_the_gil_disabled() -> None:
    assert not sys._is_gil_enabled()
