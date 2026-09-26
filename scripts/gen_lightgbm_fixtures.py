#!/usr/bin/env python3
"""Generate LightGBM parity fixtures for hessboost's LightGBM model import.

Trains real LightGBM (single thread, deterministic) on synthetic datasets, one
case per mapped feature, and writes for each case `fixtures/lightgbm/<name>.txt`
(the `Booster.save_model` text model) and `fixtures/lightgbm/<name>.json`:

  name, lightgbm_version, n_cols, n_test, x_test (NaN -> null),
  expect      "import", or "refuse" for models the importer must refuse,
  error       (refuse) a substring of the expected ModelFormat message,
  objective   (import) the hessboost objective the model maps to,
  n_outputs   (import) outputs per row (LightGBM's num_tree_per_iteration),
  raw         Booster.predict(raw_score=True), [row][output],
  pred        Booster.predict(), [row][output],
  contribs    Booster.predict(pred_contrib=True), [row][output][n_cols + 1],
              null for linear_tree models (LightGBM refuses them),
  leaf        Booster.predict(pred_leaf=True), [row][tree],
  slice_iterations, raw_slice
              raw scores of the first `slice_iterations` iterations,
  xgboost_export
              whether the imported model has an XGBoost JSON encoding.

`tests/lightgbm_parity.rs` consumes them. With `--test-data` the script
instead rewrites the small always-on models under `tests/data/`
(`lightgbm-<version>-*.txt` plus the same fields in `*.expected.json`;
checked in, regenerate only when moving to a new LightGBM release).
"""

from __future__ import annotations

import json
import os
import sys
import zlib

import lightgbm as lgb
import numpy as np

# The pinned release (scripts/requirements-lightgbm.txt).
LIGHTGBM_VERSION = "4.7.0"

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIX_DIR = os.path.join(ROOT, "fixtures", "lightgbm")
TEST_DATA_DIR = os.path.join(ROOT, "tests", "data")

N_TRAIN = 2000
N_TEST = 400
N_COLS = 6
NUM_ROUND = 40
BASE_SEED = 20260926

# Every case starts from these; `deterministic` with one thread makes the
# fixtures reproducible.
BASE_PARAMS = dict(
    num_leaves=15,
    learning_rate=0.1,
    min_data_in_leaf=10,
    num_threads=1,
    deterministic=True,
    force_row_wise=True,
    seed=7,
    verbose=-1,
)


def _seed(name: str) -> int:
    return BASE_SEED ^ zlib.crc32(name.encode())


def _floats(a) -> list:
    """Array -> flat list of Python floats, NaN -> None (JSON null)."""
    flat = np.asarray(a, dtype=np.float64).ravel()
    return [None if np.isnan(v) else float(v) for v in flat]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def features(rng, n, nan=0.0, zeros=0.0, categorical=()):
    """Normal features as f32 (the precision hessboost predicts at), with a
    fraction of exact zeros and NaNs, and integer-coded categorical columns
    (`categorical` holds (column, category count) pairs)."""
    x = rng.standard_normal((n, N_COLS)).astype(np.float32)
    for col, count in categorical:
        x[:, col] = rng.integers(0, count, n)
    if zeros:
        x[rng.random(x.shape) < zeros] = 0.0
    if nan:
        x[rng.random(x.shape) < nan] = np.nan
    return x


def signal(x):
    """A smooth numeric signal that treats NaN as 0."""
    z = np.nan_to_num(x.astype(np.float64))
    return 1.5 * z[:, 0] - z[:, 1] ** 2 + 0.5 * z[:, 2] * z[:, 3]


def y_regression(x, rng):
    return signal(x) + 0.1 * rng.standard_normal(x.shape[0])


def y_positive(x, rng):
    return np.exp(0.3 * signal(x)) * rng.gamma(2.0, 0.5, x.shape[0]) + 0.05


def y_counts(x, rng):
    return rng.poisson(np.exp(0.3 * signal(x))).astype(np.float64)


def y_binary(x, rng):
    return (1 / (1 + np.exp(-signal(x))) > rng.random(x.shape[0])).astype(np.float64)


def y_probability(x, rng):
    del rng
    return 1 / (1 + np.exp(-signal(x)))


def y_multiclass(x, rng):
    score = signal(x) + 0.3 * rng.standard_normal(x.shape[0])
    return np.digitize(score, [-0.8, 0.6]).astype(np.float64)


def y_relevance(x, rng):
    score = signal(x) + 0.3 * rng.standard_normal(x.shape[0])
    return np.clip(np.rint(score + 1), 0, 3).astype(np.float64)


def y_categorical(x, rng):
    effect0 = rng.standard_normal(64)
    effect1 = rng.standard_normal(64)
    c0 = np.nan_to_num(x[:, 0], nan=63).astype(int)
    c1 = np.nan_to_num(x[:, 1], nan=63).astype(int)
    return effect0[c0] + 0.7 * effect1[c1] + 0.5 * np.nan_to_num(x[:, 2])


def y_zero_nonnegative(x, rng):
    """Concave in the non-negative x0 with exact zeros behaving like the
    smallest values, so zero-as-missing splits send zeros with the small
    side (on this data every split of the small model does)."""
    z = np.nan_to_num(x.astype(np.float64))
    return 3 * np.log1p(z[:, 0]) + z[:, 1] + 0.05 * rng.standard_normal(x.shape[0])


def y_zero_middle(x, rng):
    """Zeros behave unlike their neighbours on both sides, which a threshold
    cannot express once zero is missing."""
    z = np.nan_to_num(x.astype(np.float64))
    return np.where(z[:, 0] == 0, 3.0, z[:, 0]) + 0.1 * rng.standard_normal(x.shape[0])


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

# name -> dict(objective params, target, data options, expectation). `expect`
# is "import" (with the hessboost objective) or ("refuse", message substring).
CASES = {
    "regression_numeric": dict(params=dict(objective="regression"), y=y_regression, objective="reg:squarederror"),
    "regression_nan": dict(params=dict(objective="regression"), y=y_regression, nan=0.15, zeros=0.1,
                           objective="reg:squarederror"),
    "binary_nan": dict(params=dict(objective="binary"), y=y_binary, nan=0.1, zeros=0.1, objective="binary:logistic"),
    "zero_as_missing_nonnegative": dict(
        params=dict(objective="regression", zero_as_missing=True, num_leaves=4),
        y=y_zero_nonnegative, abs=True, zeros=0.25, rounds=20, objective="reg:squarederror"),
    "zero_as_missing_refused": dict(
        params=dict(objective="regression", zero_as_missing=True),
        y=y_zero_middle, zeros=0.25, refuse="zero_as_missing"),
    "categorical": dict(
        params=dict(objective="regression", max_cat_to_onehot=4, cat_smooth=1.0, min_data_per_group=5),
        y=y_categorical, categorical=((0, 40), (1, 3)), nan=0.05, objective="reg:squarederror"),
    "categorical_binary": dict(
        params=dict(objective="binary", max_cat_threshold=64, cat_smooth=1.0, min_data_per_group=5),
        y=lambda x, rng: (y_categorical(x, rng) > 0).astype(np.float64),
        categorical=((0, 60), (1, 5)), nan=0.05, objective="binary:logistic"),
    "multiclass": dict(params=dict(objective="multiclass", num_class=3), y=y_multiclass, nan=0.1,
                       objective="multi:softprob"),
    "multiclassova": dict(params=dict(objective="multiclassova", num_class=3), y=y_multiclass, nan=0.1,
                          objective="binary:logistic"),
    "lambdarank": dict(params=dict(objective="lambdarank"), y=y_relevance, ranking=True, nan=0.05,
                       objective="rank:ndcg"),
    "rank_xendcg": dict(params=dict(objective="rank_xendcg"), y=y_relevance, ranking=True, objective="rank:ndcg"),
    "linear_tree": dict(params=dict(objective="regression", linear_tree=True, linear_lambda=0.1), y=y_regression,
                        nan=0.1, objective="reg:squarederror"),
    "linear_tree_binary_categorical": dict(
        params=dict(objective="binary", linear_tree=True, max_cat_to_onehot=4, min_data_per_group=5),
        y=lambda x, rng: (y_categorical(x, rng) > 0).astype(np.float64),
        categorical=((0, 12),), nan=0.05, objective="binary:logistic"),
    "regression_l1": dict(params=dict(objective="regression_l1"), y=y_regression, objective="reg:absoluteerror"),
    "huber": dict(params=dict(objective="huber", alpha=1.3), y=y_regression, objective="reg:pseudohubererror"),
    "fair": dict(params=dict(objective="fair"), y=y_regression, objective="reg:squarederror"),
    "quantile": dict(params=dict(objective="quantile", alpha=0.8), y=y_regression, objective="reg:quantileerror"),
    "mape": dict(params=dict(objective="mape"), y=y_positive, objective="reg:absoluteerror"),
    "poisson": dict(params=dict(objective="poisson"), y=y_counts, objective="count:poisson"),
    "gamma": dict(params=dict(objective="gamma"), y=y_positive, objective="reg:gamma"),
    "tweedie": dict(params=dict(objective="tweedie", tweedie_variance_power=1.3), y=y_positive,
                    objective="reg:tweedie"),
    "cross_entropy": dict(params=dict(objective="cross_entropy"), y=y_probability, objective="reg:logistic"),
    "dart": dict(params=dict(objective="regression", boosting="dart", drop_seed=3), y=y_regression, nan=0.1,
                 objective="reg:squarederror"),
    "goss_bagging": dict(params=dict(objective="binary", data_sample_strategy="goss", feature_fraction=0.7),
                         y=y_binary, objective="binary:logistic"),
    "binary_sigmoid_2": dict(params=dict(objective="binary", sigmoid=2.0), y=y_binary, refuse="sigmoid"),
    "multiclassova_sigmoid_2": dict(params=dict(objective="multiclassova", num_class=3, sigmoid=2.0),
                                    y=y_multiclass, refuse="sigmoid"),
    "regression_sqrt": dict(params=dict(objective="regression", reg_sqrt=True), y=y_regression, refuse="sqrt"),
    "cross_entropy_lambda": dict(params=dict(objective="cross_entropy_lambda"), y=y_probability,
                                 refuse="cross_entropy_lambda"),
    "random_forest": dict(params=dict(objective="regression", boosting="rf", bagging_fraction=0.7, bagging_freq=1),
                          y=y_regression, refuse="average_output"),
}


def make_data(spec, rng, n):
    x = features(rng, n, nan=spec.get("nan", 0.0), zeros=spec.get("zeros", 0.0),
                 categorical=spec.get("categorical", ()))
    if spec.get("abs"):
        x = np.abs(x)
    return x


def train(name, spec, n_train=N_TRAIN, n_test=N_TEST, num_round=NUM_ROUND, overrides=None):
    rng = np.random.default_rng(_seed(name))
    x_train = make_data(spec, rng, n_train)
    y_train = spec["y"](x_train, rng)
    x_test = make_data(spec, rng, n_test)
    params = dict(BASE_PARAMS, **spec["params"], **(overrides or {}))
    categorical = [col for col, _ in spec.get("categorical", ())]
    dataset = lgb.Dataset(
        x_train,
        label=y_train,
        categorical_feature=categorical or "auto",
        group=[20] * (n_train // 20) if spec.get("ranking") else None,
        free_raw_data=False,
    )
    booster = lgb.train(params, dataset, num_boost_round=num_round)
    return booster, x_test


def boundary_rows(booster, x_test, limit=40):
    """Test rows placing one feature on a numeric threshold, rounded to f32,
    and on the neighbouring f32 values: the conversion of LightGBM's
    `x <= threshold` on doubles must route them exactly. Thresholds within
    the zero band are skipped: LightGBM's dense-matrix predict zeroes
    inputs with |x| <= 1e-35 before the trees see them."""
    splits = []
    stack = [tree["tree_structure"] for tree in booster.dump_model()["tree_info"]]
    while stack and len(splits) < limit:
        node = stack.pop()
        if "split_index" not in node:
            continue
        stack += [node["left_child"], node["right_child"]]
        threshold = node["threshold"]
        if node["decision_type"] == "<=" and 1e-34 < abs(threshold) < 3e38:
            splits.append((node["split_feature"], threshold))
    rows = []
    for i, (feature, threshold) in enumerate(splits):
        at = np.float32(threshold)
        for value in (np.nextafter(at, np.float32(-np.inf)), at, np.nextafter(at, np.float32(np.inf))):
            row = x_test[i % len(x_test)].copy()
            row[feature] = value
            rows.append(row)
    return np.vstack([x_test, np.array(rows, dtype=np.float32).reshape(-1, x_test.shape[1])])


def fixture(name, spec, booster, x_test):
    if "refuse" not in spec:
        x_test = boundary_rows(booster, x_test)
    out = dict(name=name, lightgbm_version=lgb.__version__, n_cols=N_COLS, n_test=len(x_test),
               x_test=_floats(x_test))
    if "refuse" in spec:
        out.update(expect="refuse", error=spec["refuse"])
        return out
    linear = bool(spec["params"].get("linear_tree"))
    iterations = booster.current_iteration()
    out.update(
        expect="import",
        objective=spec["objective"],
        n_outputs=booster.num_model_per_iteration(),
        raw=_floats(booster.predict(x_test, raw_score=True)),
        pred=_floats(booster.predict(x_test)),
        contribs=None if linear else _floats(booster.predict(x_test, pred_contrib=True)),
        leaf=np.asarray(booster.predict(x_test, pred_leaf=True)).astype(int).ravel().tolist(),
        slice_iterations=iterations // 2,
        raw_slice=_floats(booster.predict(x_test, raw_score=True, num_iteration=iterations // 2)),
        xgboost_export=not linear,
    )
    return out


def write(directory, name, booster, data, suffix=".json"):
    booster.save_model(os.path.join(directory, f"{name}.txt"))
    with open(os.path.join(directory, f"{name}{suffix}"), "w") as fh:
        json.dump(data, fh, separators=(",", ":"))


# The always-on models checked in under tests/data/ (small on purpose):
# a binary model with NaN, zero and categorical splits, and a linear_tree
# model with missing values.
TEST_DATA = {
    f"lightgbm-{LIGHTGBM_VERSION}-binary": ("categorical_binary", dict(num_leaves=6)),
    f"lightgbm-{LIGHTGBM_VERSION}-linear": ("linear_tree", dict(num_leaves=4)),
}


def write_test_data():
    for file_name, (case, overrides) in TEST_DATA.items():
        spec = CASES[case]
        booster, x_test = train(case, spec, n_train=600, n_test=24, num_round=6, overrides=overrides)
        data = fixture(file_name, spec, booster, x_test)
        write(TEST_DATA_DIR, file_name, booster, data, suffix=".expected.json")
        print(f"wrote tests/data/{file_name}.{{txt,expected.json}}")


def main() -> None:
    if lgb.__version__ != LIGHTGBM_VERSION:
        sys.exit(f"expected LightGBM {LIGHTGBM_VERSION}, found {lgb.__version__}")
    if "--test-data" in sys.argv[1:]:
        write_test_data()
        return
    os.makedirs(FIX_DIR, exist_ok=True)
    for name, spec in CASES.items():
        booster, x_test = train(name, spec, num_round=spec.get("rounds", NUM_ROUND))
        write(FIX_DIR, name, booster, fixture(name, spec, booster, x_test))
        print(f"wrote fixtures/lightgbm/{name}")


if __name__ == "__main__":
    main()
