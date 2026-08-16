#!/usr/bin/env python3
"""Generate XGBoost parity fixtures for sequoia-boost.

Trains real XGBoost on standardized synthetic datasets across several
objectives and writes each dataset, the exact training parameters, and
XGBoost's predictions to ``fixtures/*.json``. The Rust integration test
``tests/parity.rs`` (run with ``cargo test -- --ignored``) loads these and
asserts sequoia-boost matches within tolerance.

Usage:
    pip install xgboost numpy
    python scripts/gen_fixtures.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import xgboost as xgb

FIX_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")

# Parameters shared by every fixture. `tree_method=hist` and a fixed max_bin
# give sequoia-boost's histogram method the best chance of exact agreement.
COMMON = dict(
    tree_method="hist",
    max_depth=4,
    eta=0.1,
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0.0,
    min_child_weight=1.0,
    max_bin=256,
    subsample=1.0,
    colsample_bytree=1.0,
    base_score=0.5,
)
NUM_ROUND = 60


def _rng():
    return np.random.default_rng(20260720)


def _write(name, x_train, y_train, x_test, y_test, params, num_class, preds):
    x_train = np.ascontiguousarray(x_train, dtype=np.float32)
    x_test = np.ascontiguousarray(x_test, dtype=np.float32)
    fixture = {
        "name": name,
        "objective": params["objective"],
        "num_class": num_class,
        "num_round": NUM_ROUND,
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_cols": int(x_train.shape[1]),
        "x_train": x_train.reshape(-1).tolist(),
        "y_train": np.asarray(y_train, dtype=np.float32).tolist(),
        "x_test": x_test.reshape(-1).tolist(),
        "y_test": np.asarray(y_test, dtype=np.float32).tolist(),
        "params": {
            "max_depth": params["max_depth"],
            "eta": params["eta"],
            "lambda": params["reg_lambda"],
            "alpha": params["reg_alpha"],
            "gamma": params["gamma"],
            "min_child_weight": params["min_child_weight"],
            "max_bin": params["max_bin"],
            "base_score": params["base_score"],
        },
        "xgb_pred": np.asarray(preds, dtype=np.float32).reshape(-1).tolist(),
    }
    os.makedirs(FIX_DIR, exist_ok=True)
    path = os.path.join(FIX_DIR, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(fixture, fh)
    print(
        f"wrote {path}  "
        f"({fixture['n_train']} train, {fixture['n_test']} test, {fixture['n_cols']} features)"
    )


def _train(x_train, y_train, x_test, params, num_class):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test)
    p = dict(COMMON, **params)
    if num_class:
        p["num_class"] = num_class
    booster = xgb.train(p, dtrain, num_boost_round=NUM_ROUND)
    return booster.predict(dtest)


def regression():
    rng = _rng()
    x = rng.random((2500, 8), dtype=np.float32)
    y = 2 * x[:, 0] - 3 * x[:, 1] ** 2 + 0.5 * x[:, 2] + 0.1 * rng.standard_normal(2500)
    preds = _train(x[:2000], y[:2000], x[2000:], dict(objective="reg:squarederror"), 0)
    _write("regression", x[:2000], y[:2000], x[2000:], y[2000:], dict(COMMON, objective="reg:squarederror"), 0, preds)


def binary():
    rng = _rng()
    x = rng.random((2500, 8), dtype=np.float32)
    logit = 3 * x[:, 0] - 2 * x[:, 1]
    y = (1 / (1 + np.exp(-logit)) > rng.random(2500)).astype(np.float32)
    preds = _train(x[:2000], y[:2000], x[2000:], dict(objective="binary:logistic"), 0)
    _write("binary", x[:2000], y[:2000], x[2000:], y[2000:], dict(COMMON, objective="binary:logistic"), 0, preds)


def multiclass():
    rng = _rng()
    x = rng.random((2500, 8), dtype=np.float32)
    y = (x[:, 0] * 3).astype(int).clip(0, 2).astype(np.float32)
    preds = _train(x[:2000], y[:2000], x[2000:], dict(objective="multi:softprob"), 3)
    _write("multiclass", x[:2000], y[:2000], x[2000:], y[2000:], dict(COMMON, objective="multi:softprob"), 3, preds)


if __name__ == "__main__":
    regression()
    binary()
    multiclass()
    print("done. run: cargo test -p sequoia-boost --test parity -- --ignored")
