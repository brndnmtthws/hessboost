#!/usr/bin/env python3
"""Export parity: reload hessboost's XGBoost-format exports in real XGBoost.

`tests/parity.rs` writes, per case, `fixtures/exports/<name>.model.json`
(hessboost's `to_xgboost_json`), `<name>.model.ubj` (`to_xgboost_ubjson`), and
`<name>.pred.json` (hessboost's predictions on the case's `x_test`). This
script loads each model with `xgb.Booster.load_model`, predicts the same
`x_test` from `fixtures/<name>.json`, and requires max |delta| <= the
fixture's `tol.import`.

For UBJSON exports it also requires the encoding to mirror XGBoost's: after
loading, XGBoost re-saves the model (`save_raw("ubj")`), and every array both
files share must use the same container form (the typed-array element marker,
or a generic array).

Usage:
    uv run --with-requirements scripts/requirements-xgboost.txt python scripts/check_exports.py
"""

from __future__ import annotations

import glob
import json
import os
import struct
import sys

import numpy as np
import xgboost as xgb

FIX_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
EXPORT_DIR = os.path.join(FIX_DIR, "exports")
SUFFIXES = (".model.json", ".model.ubj")

# Payload widths of the UBJSON scalar markers XGBoost writes.
_WIDTH = {b"i": 1, b"U": 1, b"C": 1, b"I": 2, b"l": 4, b"d": 4, b"L": 8, b"D": 8}
_INT = {b"i": ">b", b"U": ">B", b"I": ">h", b"l": ">i", b"L": ">q"}


def _array_forms(data: bytes) -> dict[str, str]:
    """Map the path of every array in a UBJSON document to its container form:
    the typed element marker (`d`, `l`, `U`, ...) or `generic`."""
    forms: dict[str, str] = {}
    pos = 0

    def take(n: int) -> bytes:
        nonlocal pos
        chunk = data[pos : pos + n]
        if len(chunk) != n:
            raise ValueError(f"truncated UBJSON at byte {pos}")
        pos += n
        return chunk

    def length() -> int:
        marker = take(1)
        return struct.unpack(_INT[marker], take(_WIDTH[marker]))[0]

    def value(path: str, marker: bytes) -> None:
        if marker in (b"Z", b"T", b"F", b"N"):
            return
        if marker in _WIDTH:
            take(_WIDTH[marker])
        elif marker in (b"S", b"H"):
            take(length())
        elif marker == b"[":
            array(path)
        elif marker == b"{":
            obj(path)
        else:
            raise ValueError(f"unexpected marker {marker!r} at byte {pos - 1}")

    def array(path: str) -> None:
        nonlocal pos
        if data[pos : pos + 1] == b"$":
            take(1)
            element = take(1)
            take(1)  # '#'
            n = length()
            forms[path] = element.decode()
            take(n * _WIDTH[element])
            return
        forms[path] = "generic"
        if data[pos : pos + 1] == b"#":
            take(1)
            for i in range(length()):
                value(f"{path}/{i}", take(1))
            return
        i = 0
        while data[pos : pos + 1] != b"]":
            value(f"{path}/{i}", take(1))
            i += 1
        take(1)

    def obj(path: str) -> None:
        while data[pos : pos + 1] != b"}":
            key = take(length()).decode()
            value(f"{path}/{key}", take(1))
        take(1)

    value("", take(1))
    return forms


def _check_ubj_forms(model_path: str, booster: xgb.Booster) -> str | None:
    """Compare the container form of every array hessboost's UBJSON export
    shares with XGBoost's own re-save of the loaded model."""
    with open(model_path, "rb") as fh:
        ours = _array_forms(fh.read())
    theirs = _array_forms(bytes(booster.save_raw("ubj")))
    shared = sorted(set(ours) & set(theirs))
    if not any(path.endswith("/split_conditions") for path in shared):
        return "no tree arrays shared with XGBoost's re-save"
    mismatched = [f"{p} {ours[p]}!={theirs[p]}" for p in shared if ours[p] != theirs[p]]
    if mismatched:
        return f"array encodings differ from XGBoost: {', '.join(mismatched[:3])}"
    return None


def _dense(values: list, n_rows: int, n_cols: int) -> np.ndarray:
    x = np.array([np.nan if v is None else v for v in values], dtype=np.float32)
    return x.reshape(n_rows, n_cols)


def check_case(model_path: str) -> tuple[str, str]:
    """Return (label, verdict). verdict is 'OK' or a failure description."""
    base = os.path.basename(model_path)
    suffix = next(s for s in SUFFIXES if base.endswith(s))
    name = base[: -len(suffix)]
    label = f"{name} [{suffix.rsplit('.', 1)[-1]}]"
    with open(os.path.join(FIX_DIR, f"{name}.json")) as fh:
        fx = json.load(fh)
    with open(os.path.join(EXPORT_DIR, f"{name}.pred.json")) as fh:
        hessboost_pred = np.asarray(json.load(fh), dtype=np.float32)

    booster = xgb.Booster()
    try:
        booster.load_model(model_path)
    except xgb.core.XGBoostError as e:
        return label, f"load_model failed: {str(e).splitlines()[0]}"

    x_test = _dense(fx["x_test"], fx["n_test"], fx["n_cols"])
    dtest = xgb.DMatrix(x_test, nthread=1, feature_types=fx.get("feature_types"))
    xgb_pred = np.asarray(booster.predict(dtest), dtype=np.float32).reshape(-1)
    if xgb_pred.shape != hessboost_pred.shape:
        return label, f"length mismatch: xgboost {xgb_pred.size} vs hessboost {hessboost_pred.size}"

    delta = np.abs(xgb_pred.astype(np.float64) - hessboost_pred.astype(np.float64))
    max_delta = float(np.max(delta)) if delta.size else 0.0
    tol = fx["tol"]["import"]
    ok = np.isfinite(max_delta) and max_delta <= tol
    form_error = _check_ubj_forms(model_path, booster) if suffix == ".model.ubj" else None
    verdict = "OK" if ok and form_error is None else "FAIL"
    print(f"{label:<33} n={xgb_pred.size:<5} max|d|={max_delta:.3e} tol={tol:.0e} {verdict}")
    if not ok:
        return label, f"max|d|={max_delta:.3e} > tol {tol:.0e}"
    return label, form_error or "OK"


def main() -> int:
    models = sorted(
        path for suffix in SUFFIXES for path in glob.glob(os.path.join(EXPORT_DIR, f"*{suffix}"))
    )
    if not models:
        print(
            f"no exports in {os.path.abspath(EXPORT_DIR)}; run "
            "`cargo test -p hessboost --test parity --release -- --ignored` first"
        )
        return 2
    failures = [(label, why) for label, why in map(check_case, models) if why != "OK"]
    print(f"{len(models) - len(failures)}/{len(models)} exports reload in xgboost {xgb.__version__}")
    if failures:
        print("export parity failures:")
        for label, why in failures:
            print(f"  {label:<33} {why}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
