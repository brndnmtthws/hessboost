#!/usr/bin/env python3
"""Export parity: reload sequoia-boost's XGBoost-JSON exports in real XGBoost.

`crates/sequoia-boost/tests/parity.rs` writes `fixtures/exports/<name>.model.json`
(sequoia's `to_xgboost_json`) and `<name>.pred.json` (sequoia's predictions on
the case's `x_test`). This script loads each model with `xgb.Booster.load_model`,
predicts the same `x_test` from `fixtures/<name>.json`, and requires
max |delta| <= the fixture's `tol.import`.

Usage:
    uv run --with xgboost==3.4.1 --with numpy python scripts/check_exports.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import xgboost as xgb

FIX_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
EXPORT_DIR = os.path.join(FIX_DIR, "exports")


def _dense(values: list, n_rows: int, n_cols: int) -> np.ndarray:
    x = np.array([np.nan if v is None else v for v in values], dtype=np.float32)
    return x.reshape(n_rows, n_cols)


def check_case(model_path: str) -> tuple[str, str]:
    """Return (name, verdict). verdict is 'OK' or a failure description."""
    name = os.path.basename(model_path)[: -len(".model.json")]
    with open(os.path.join(FIX_DIR, f"{name}.json")) as fh:
        fx = json.load(fh)
    with open(os.path.join(EXPORT_DIR, f"{name}.pred.json")) as fh:
        sequoia_pred = np.asarray(json.load(fh), dtype=np.float32)

    booster = xgb.Booster()
    try:
        booster.load_model(model_path)
    except xgb.core.XGBoostError as e:
        return name, f"load_model failed: {str(e).splitlines()[0]}"

    x_test = _dense(fx["x_test"], fx["n_test"], fx["n_cols"])
    dtest = xgb.DMatrix(x_test, nthread=1, feature_types=fx.get("feature_types"))
    xgb_pred = np.asarray(booster.predict(dtest), dtype=np.float32).reshape(-1)
    if xgb_pred.shape != sequoia_pred.shape:
        return name, f"length mismatch: xgboost {xgb_pred.size} vs sequoia {sequoia_pred.size}"

    delta = np.abs(xgb_pred.astype(np.float64) - sequoia_pred.astype(np.float64))
    max_delta = float(np.max(delta)) if delta.size else 0.0
    tol = fx["tol"]["import"]
    ok = np.isfinite(max_delta) and max_delta <= tol
    print(f"{name:<26} n={xgb_pred.size:<5} max|d|={max_delta:.3e} tol={tol:.0e} {'OK' if ok else 'FAIL'}")
    return name, "OK" if ok else f"max|d|={max_delta:.3e} > tol {tol:.0e}"


def main() -> int:
    models = sorted(glob.glob(os.path.join(EXPORT_DIR, "*.model.json")))
    if not models:
        print(
            f"no exports in {os.path.abspath(EXPORT_DIR)}; run "
            "`cargo test -p sequoia-boost --test parity --release -- --ignored` first"
        )
        return 2
    failures = [(name, why) for name, why in map(check_case, models) if why != "OK"]
    print(f"{len(models) - len(failures)}/{len(models)} exports reload in xgboost {xgb.__version__}")
    if failures:
        print("export parity failures:")
        for name, why in failures:
            print(f"  {name:<26} {why}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
