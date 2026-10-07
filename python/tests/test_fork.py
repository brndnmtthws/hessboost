"""Fork safety: a child forked after the parent ran parallel native work
predicts and trains instead of waiting on the parent's thread pool, whose
threads do not exist in the child."""

from __future__ import annotations

import os
import signal
import sys
import time
import warnings
from collections.abc import Callable

import numpy as np
import pytest

import hessboost
from conftest import regression
from hessboost import DMatrix

pytestmark = [
    pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork is unavailable"),
    pytest.mark.skipif(
        sys.platform == "darwin", reason="forking with Metal and system frameworks loaded"
    ),
]

TIMEOUT = 60.0


def _fork(child: Callable[[], None]) -> int:
    """Forks, runs `child` in the child (exit 0 on success, 1 on an
    exception), and returns the child's pid."""
    with warnings.catch_warnings():
        # Python 3.12+ warns about forking a process with threads.
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        code = 1
        try:
            child()
            code = 0
        finally:
            os._exit(code)
    return pid


def _wait(pid: int) -> int:
    """The child's exit code; kills it and fails after `TIMEOUT`."""
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    pytest.fail(f"the forked child hung for {TIMEOUT} s")


def test_forked_child_predicts_and_trains() -> None:
    x, y = regression(rows=4096)
    booster = hessboost.train({"max_depth": 4}, DMatrix(x, y), 10)
    expected = booster.predict(x)

    def child() -> None:
        np.testing.assert_array_equal(booster.predict(x), expected)
        hessboost.train({"max_depth": 4}, DMatrix(x, y), 3)

    assert _wait(_fork(child)) == 0
    # The parent's pool still works after the fork.
    np.testing.assert_array_equal(booster.predict(x), expected)
