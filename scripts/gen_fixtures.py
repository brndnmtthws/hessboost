#!/usr/bin/env python3
"""Generate XGBoost 3.4.1 parity fixtures for sequoia-boost.

Trains real XGBoost (single thread) on deterministic synthetic datasets, one
case per supported feature, and writes `fixtures/<name>.json` holding the data,
the exact `xgb.train` parameter dict, XGBoost's test-set predictions (transformed,
raw margin, SHAP contributions on the first 50 rows) and the saved model JSON.
`crates/sequoia-boost/tests/parity.rs` consumes these; the fixture schema is the
contract between the two.

Tiers:
  exact    every assertion is pointwise (train, import, export).
  quality  the model is RNG-driven (subsample, colsample, DART) or the objective
           is not yet measured for pointwise agreement (rank:*), so training is
           held to a quality band instead: RMSE <= band * xgb RMSE, or
           accuracy / NDCG@20 >= xgb - band. Import/export stay pointwise.

Usage:
    uv run --with xgboost==3.4.1 --with numpy python scripts/gen_fixtures.py
"""

from __future__ import annotations

import json
import os
import tempfile
import zlib

import numpy as np
import xgboost as xgb

FIX_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")

N_TRAIN = 2000
N_TEST = 500
N_COLS = 8
N_CONTRIB_ROWS = 50
GROUP_SIZE = 20
NUM_ROUND = 50
BASE_SEED = 20260915

# Tolerances (max |delta|) for the exact tier.
TOL_TRAIN = 1e-4
TOL_TRAIN_PROB = 1e-5  # binary probabilities / softprob
TOL_IMPORT = 1e-5
TOL_CONTRIBS = 1e-4
# Quality-tier bands: relative RMSE factor for regression, absolute accuracy
# slack for classification.
BAND_RMSE = 1.08
BAND_ACC = 0.02

# Baseline tree-booster parameters; every case starts from these.
TREE_BASE = dict(
    tree_method="hist",
    max_depth=6,
    eta=0.1,
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0.0,
    min_child_weight=1.0,
    max_bin=256,
    base_score=0.5,
)


def _seed(name: str) -> int:
    return BASE_SEED ^ zlib.crc32(name.encode())


# ---------------------------------------------------------------------------
# Synthetic targets. Each takes the feature matrix and the case RNG.
# ---------------------------------------------------------------------------


def y_regression(x, rng):
    n = x.shape[0]
    return 2 * x[:, 0] - 3 * x[:, 1] ** 2 + 0.5 * x[:, 2] + 0.1 * rng.standard_normal(n)


def y_heavy_tail(x, rng):
    """Regression signal with heavy-tailed noise (for pseudo-Huber)."""
    n = x.shape[0]
    return 2 * x[:, 0] - 3 * x[:, 1] ** 2 + 0.5 * x[:, 2] + 0.3 * rng.standard_t(2, n)


def y_binary(x, rng):
    logit = 3 * x[:, 0] - 2 * x[:, 1]
    return (1 / (1 + np.exp(-logit)) > rng.random(x.shape[0])).astype(np.float32)


def y_multiclass(x, rng):
    del rng
    return (x[:, 0] * 3).astype(int).clip(0, 2).astype(np.float32)


def y_poisson(x, rng):
    return rng.poisson(np.exp(0.5 + x[:, 0] - 1.5 * x[:, 1])).astype(np.float32)


def y_gamma(x, rng):
    return rng.gamma(2.0, np.exp(0.5 * x[:, 0] - x[:, 1]) / 2.0).astype(np.float32)


def y_tweedie(x, rng):
    """Zero-inflated positive target (compound Poisson-gamma shape)."""
    n = x.shape[0]
    positive = rng.gamma(2.0, np.exp(x[:, 0] - x[:, 1]) / 2.0)
    return np.where(rng.random(n) < 0.3, 0.0, positive).astype(np.float32)


def y_relevance(x, rng):
    """Graded relevance 0..3 for NDCG / pairwise ranking."""
    n = x.shape[0]
    return np.clip(np.rint(3 * x[:, 0] - x[:, 1] + 0.3 * rng.standard_normal(n)), 0, 3).astype(
        np.float32
    )


def y_relevance_binary(x, rng):
    """Binary relevance (rank:map requires 0/1 labels)."""
    return (y_relevance(x, rng) >= 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Case matrix
# ---------------------------------------------------------------------------

# name -> (target fn, param overrides, options)
# options: tier, num_round, missing (fraction of NaN features), weighted, ranking,
#          tol_train (override), drop (params removed from TREE_BASE)
CASES = {
    # tree_method / grow policy / constraints on reg:squarederror
    "exact_reg_d6": (y_regression, dict(tree_method="exact"), {}),
    "exact_reg_missing_d4": (y_regression, dict(tree_method="exact", max_depth=4), dict(missing=0.3)),
    "hist_reg_d1_r1": (y_regression, dict(max_depth=1), dict(num_round=1)),
    "hist_reg_d6_r50": (y_regression, {}, {}),
    "hist_reg_bin32_d6": (y_regression, dict(max_bin=32), {}),
    "hist_reg_missing_d6": (y_regression, {}, dict(missing=0.3)),
    "approx_reg_d6": (y_regression, dict(tree_method="approx"), {}),
    "lossguide_reg_l15": (y_regression, dict(grow_policy="lossguide", max_leaves=15, max_depth=0), {}),
    "monotone_reg_d6": (y_regression, dict(monotone_constraints="(1,-1,0,0,0,0,0,0)"), {}),
    "interaction_reg_d6": (y_regression, dict(interaction_constraints="[[0,1],[2,3,4],[5,6,7]]"), {}),
    "gamma_mcw_alpha_reg_d6": (y_regression, dict(gamma=0.05, min_child_weight=10, reg_alpha=0.5), {}),
    "interaction_overlap_reg_d6": (
        y_regression,
        dict(interaction_constraints="[[0,1],[1,2],[3,4,5,6,7]]"),
        {},
    ),
    "weighted_reg_d6": (y_regression, {}, dict(weighted=True)),
    "categorical_reg_d6": (
        y_regression,
        {},
        dict(categorical=True),
    ),
    # objectives
    "binary_d6": (y_binary, dict(objective="binary:logistic"), dict(tol_train=TOL_TRAIN_PROB)),
    "binary_spw3_d6": (
        y_binary,
        dict(objective="binary:logistic", scale_pos_weight=3.0),
        dict(tol_train=TOL_TRAIN_PROB),
    ),
    # reg:logistic is binary:logistic's loss reported as a probability
    # regression (rmse); XGBoost saves the name as-is.
    "reg_logistic_d6": (y_binary, dict(objective="reg:logistic"), dict(tol_train=TOL_TRAIN_PROB)),
    # Deprecated alias: XGBoost 3.4.1 trains it as reg:squarederror (with a
    # warning) and saves the model objective as reg:squarederror.
    "reg_linear_d6": (y_regression, dict(objective="reg:linear"), {}),
    "softprob_d4": (
        y_multiclass,
        dict(objective="multi:softprob", num_class=3, max_depth=4),
        dict(tol_train=TOL_TRAIN_PROB),
    ),
    "softmax_d4": (y_multiclass, dict(objective="multi:softmax", num_class=3, max_depth=4), {}),
    "poisson_d4": (y_poisson, dict(objective="count:poisson", max_depth=4), {}),
    "gamma_d4": (y_gamma, dict(objective="reg:gamma", max_depth=4), {}),
    "tweedie_d4": (
        y_tweedie,
        dict(objective="reg:tweedie", tweedie_variance_power=1.5, max_depth=4),
        {},
    ),
    "huber_d4": (y_heavy_tail, dict(objective="reg:pseudohubererror", huber_slope=1.0, max_depth=4), {}),
    # ranking: groups of 20 and topk=20 enumerate every unordered pair. The
    # Rust objective reproduces XGBoost's top-k accumulation/normalization.
    "rank_ndcg_d4": (
        y_relevance,
        dict(
            objective="rank:ndcg",
            max_depth=4,
            lambdarank_pair_method="topk",
            lambdarank_num_pair_per_sample=GROUP_SIZE,
        ),
        dict(ranking=True),
    ),
    "rank_ndcg_top20_g50_d4": (
        y_relevance,
        dict(
            objective="rank:ndcg",
            max_depth=4,
            lambdarank_pair_method="topk",
            lambdarank_num_pair_per_sample=20,
        ),
        dict(ranking=True, group_size=50),
    ),
    "rank_pairwise_d4": (
        y_relevance,
        dict(
            objective="rank:pairwise",
            max_depth=4,
            lambdarank_pair_method="topk",
            lambdarank_num_pair_per_sample=GROUP_SIZE,
        ),
        dict(ranking=True),
    ),
    "rank_map_d4": (
        y_relevance_binary,
        dict(
            objective="rank:map",
            max_depth=4,
            lambdarank_pair_method="topk",
            lambdarank_num_pair_per_sample=GROUP_SIZE,
        ),
        dict(ranking=True),
    ),
    # linear booster (coord_descent is the deterministic updater; shotgun is not)
    "gblinear_r50": (
        y_regression,
        dict(
            booster="gblinear",
            updater="coord_descent",
            feature_selector="cyclic",
            eta=0.5,
            reg_lambda=0.0,
            reg_alpha=0.0,
        ),
        dict(tier="trainonly", drop=("tree_method", "max_depth", "gamma", "min_child_weight", "max_bin")),
    ),
    # no base_score -> XGBoost estimates the intercept from the labels
    "nobs_reg_d4": (y_regression, dict(max_depth=4), dict(drop=("base_score",))),
    "nobs_binary_d4": (
        y_binary,
        dict(objective="binary:logistic", max_depth=4),
        dict(drop=("base_score",), tol_train=TOL_TRAIN_PROB),
    ),
    "nobs_softprob_d4": (
        y_multiclass,
        dict(objective="multi:softprob", num_class=3, max_depth=4),
        dict(drop=("base_score",), tol_train=TOL_TRAIN_PROB),
    ),
    "nobs_poisson_d4": (y_poisson, dict(objective="count:poisson", max_depth=4), dict(drop=("base_score",))),
    "nobs_huber_d4": (
        y_heavy_tail,
        dict(objective="reg:pseudohubererror", huber_slope=1.0, max_depth=4),
        dict(drop=("base_score",)),
    ),
    # quality tier: RNG-driven sampling, pointwise agreement is not expected
    "subsample_0p8_d6": (y_regression, dict(subsample=0.8, seed=42), dict(tier="quality")),
    "colsample_bytree_0p5_d6": (y_regression, dict(colsample_bytree=0.5, seed=42), dict(tier="quality")),
    "dart_d4": (
        y_regression,
        dict(booster="dart", rate_drop=0.1, skip_drop=0.5, seed=42, max_depth=4),
        dict(tier="quality"),
    ),
}


def _params(overrides: dict, drop: tuple) -> dict:
    p = dict(TREE_BASE)
    for key in drop:
        p.pop(key, None)
    p.update(overrides)
    p.setdefault("objective", "reg:squarederror")
    return p


def _quality_band(params: dict) -> float:
    """Quality-tier band: relative RMSE factor for regression, absolute slack
    for accuracy (classification) and NDCG@GROUP_SIZE (ranking)."""
    obj = params["objective"]
    if obj.startswith(("binary:", "multi:", "rank:")):
        return BAND_ACC
    return BAND_RMSE


def _to_json_floats(a: np.ndarray) -> list:
    """f32 array -> list of Python floats, NaN -> None (JSON null)."""
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    out = a.astype(np.float64).astype(object)
    out[np.isnan(a)] = None
    return out.tolist()


def _inject_missing(x: np.ndarray, frac: float, rng) -> np.ndarray:
    x = x.copy()
    x[rng.random(x.shape) < frac] = np.nan
    return x


def _save_model_json(booster: xgb.Booster) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.json")
        booster.save_model(path)
        with open(path) as fh:
            return json.load(fh)


def build_case(name: str) -> dict:
    target, overrides, opts = CASES[name]
    tier = opts.get("tier", "exact")
    num_round = opts.get("num_round", NUM_ROUND)
    params = _params(overrides, opts.get("drop", ()))
    rng = np.random.default_rng(_seed(name))

    x = rng.random((N_TRAIN + N_TEST, N_COLS), dtype=np.float32)
    feature_types = None
    if opts.get("categorical"):
        # Two integer-coded categorical columns plus six numeric columns.
        x[:, 0] = rng.integers(0, 5, x.shape[0]).astype(np.float32)
        x[:, 1] = rng.integers(0, 9, x.shape[0]).astype(np.float32)
        feature_types = ["c", "c"] + ["q"] * (N_COLS - 2)
    y = np.asarray(target(x, rng), dtype=np.float32)
    if "missing" in opts:
        x = _inject_missing(x, opts["missing"], rng)
    x_train, x_test = x[:N_TRAIN], x[N_TRAIN:]
    y_train, y_test = y[:N_TRAIN], y[N_TRAIN:]

    weights = rng.uniform(0.5, 2.0, N_TRAIN).astype(np.float32) if opts.get("weighted") else None
    group_sizes = test_group_sizes = None
    if opts.get("ranking"):
        group_size = opts.get("group_size", GROUP_SIZE)
        assert N_TRAIN % group_size == 0 and N_TEST % group_size == 0
        group_sizes = [group_size] * (N_TRAIN // group_size)
        test_group_sizes = [group_size] * (N_TEST // group_size)

    dtrain = xgb.DMatrix(
        x_train, label=y_train, weight=weights, nthread=1, feature_types=feature_types
    )
    if group_sizes is not None:
        dtrain.set_group(group_sizes)
    dtest = xgb.DMatrix(x_test, nthread=1, feature_types=feature_types)
    dcontrib = xgb.DMatrix(x_test[:N_CONTRIB_ROWS], nthread=1, feature_types=feature_types)

    booster = xgb.train(dict(params, nthread=1), dtrain, num_boost_round=num_round)
    pred = booster.predict(dtest)
    margin = booster.predict(dtest, output_margin=True)
    contribs = booster.predict(dcontrib, pred_contribs=True)

    tol_train = _quality_band(params) if tier == "quality" else opts.get("tol_train", TOL_TRAIN)

    return {
        "xgboost_version": xgb.__version__,
        "name": name,
        "tier": tier,
        "params": params,
        "num_class": int(params.get("num_class", 0)),
        "num_round": num_round,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "n_cols": N_COLS,
        "x_train": _to_json_floats(x_train),
        "y_train": _to_json_floats(y_train),
        "x_test": _to_json_floats(x_test),
        "y_test": _to_json_floats(y_test),
        "weights": None if weights is None else _to_json_floats(weights),
        "group_sizes": group_sizes,
        "test_group_sizes": test_group_sizes,
        "feature_types": feature_types,
        "xgb_pred": _to_json_floats(pred),
        "xgb_margin": _to_json_floats(margin),
        "xgb_contribs": _to_json_floats(contribs),
        "xgb_model": _save_model_json(booster),
        "tol": {"train": tol_train, "import": TOL_IMPORT, "contribs": TOL_CONTRIBS},
    }


# ---------------------------------------------------------------------------
# Quantile-cut oracle: XGBoost's hist cuts for a matrix, bit-for-bit.
# ---------------------------------------------------------------------------

CUT_DIR = os.path.join(FIX_DIR, "cuts")


def x_uniform(rng, n, cols):
    return rng.random((n, cols), dtype=np.float32)


def x_few_distinct(rng, n, cols):
    return rng.integers(0, 5, size=(n, cols)).astype(np.float32)


def x_normal(rng, n, cols):
    return rng.standard_normal((n, cols), dtype=np.float32)


def x_missing(rng, n, cols):
    return _inject_missing(x_uniform(rng, n, cols), 0.4, rng)


def w_sparse(rng, n):
    """uniform*3 weights with every 7th row weighted zero."""
    w = (rng.random(n) * 3).astype(np.float32)
    w[::7] = 0.0
    return w


# name -> (feature generator, n_rows, n_cols, max_bin, weight generator | None,
#          tree_method, objective)
# `approx` cuts are Hessian-weighted: reg:squarederror has a constant Hessian
# (XGBoost keeps the round-0 streaming sketch), binary:logistic does not
# (XGBoost rebuilds a sorted-column summary each round).
CUT_CASES = {
    "uniform_2000_b256": (x_uniform, 2000, 4, 256, None, "hist", "reg:squarederror"),
    "uniform_2000_b16": (x_uniform, 2000, 4, 16, None, "hist", "reg:squarederror"),
    "uniform_200_b256": (x_uniform, 200, 4, 256, None, "hist", "reg:squarederror"),
    "few_distinct_b256": (x_few_distinct, 2000, 4, 256, None, "hist", "reg:squarederror"),
    "few_distinct_b3": (x_few_distinct, 2000, 4, 3, None, "hist", "reg:squarederror"),
    "large_120k_b256": (x_uniform, 120_000, 2, 256, None, "hist", "reg:squarederror"),
    "large_120k_b64": (x_uniform, 120_000, 2, 64, None, "hist", "reg:squarederror"),
    "missing_5000_b256": (x_missing, 5000, 3, 256, None, "hist", "reg:squarederror"),
    "weighted_4000_b256": (x_uniform, 4000, 4, 256, w_sparse, "hist", "reg:squarederror"),
    "weighted_4000_b32": (x_uniform, 4000, 4, 32, w_sparse, "hist", "reg:squarederror"),
    "normal_50k_b256": (x_normal, 50_000, 3, 256, None, "hist", "reg:squarederror"),
    "approx_uniform_2000_b256": (x_uniform, 2000, 4, 256, None, "approx", "reg:squarederror"),
    "approx_uniform_2000_b16": (x_uniform, 2000, 4, 16, None, "approx", "reg:squarederror"),
    "approx_missing_5000_b256": (x_missing, 5000, 3, 256, None, "approx", "reg:squarederror"),
    "approx_weighted_4000_b32": (x_uniform, 4000, 4, 32, w_sparse, "approx", "reg:squarederror"),
    "approx_large_120k_b256": (x_uniform, 120_000, 2, 256, None, "approx", "reg:squarederror"),
    "approx_logit_uniform_2000_b256": (x_uniform, 2000, 4, 256, None, "approx", "binary:logistic"),
    "approx_logit_uniform_2000_b16": (x_uniform, 2000, 4, 16, None, "approx", "binary:logistic"),
    "approx_logit_missing_5000_b256": (x_missing, 5000, 3, 256, None, "approx", "binary:logistic"),
    "approx_logit_weighted_4000_b32": (x_uniform, 4000, 4, 32, w_sparse, "approx", "binary:logistic"),
    "approx_logit_large_120k_b256": (x_uniform, 120_000, 2, 256, None, "approx", "binary:logistic"),
}


def build_cut_case(name: str) -> dict:
    features, n_rows, n_cols, max_bin, weight_fn, tree_method, objective = CUT_CASES[name]
    rng = np.random.default_rng(_seed("cuts/" + name))
    x = features(rng, n_rows, n_cols)
    w = None if weight_fn is None else weight_fn(rng, n_rows)

    # Labels are all zero; with base_score=0.5 the round-0 Hessian is 1 for
    # squared error and 0.25 for logistic, times the sample weight.
    d = xgb.DMatrix(x, label=np.zeros(n_rows, dtype=np.float32), weight=w, nthread=1)
    params = dict(tree_method=tree_method, max_bin=max_bin, max_depth=1, nthread=1,
                  objective=objective, base_score=0.5)
    xgb.train(params, d, 1)
    indptr, cuts = d.get_quantile_cut()
    assert indptr.shape == (n_cols + 1,) and cuts.shape == (indptr[-1],)
    # Each feature's block opens with -inf (its minimum bound); the Rust side
    # compares only the real cut values that follow, so -inf becomes null.
    cuts = np.where(np.isneginf(cuts), np.nan, cuts)

    return {
        "xgboost_version": xgb.__version__,
        "name": name,
        "tree_method": tree_method,
        "objective": objective,
        "max_bin": max_bin,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "x": _to_json_floats(x),
        "w": None if w is None else _to_json_floats(w),
        "indptr": indptr.astype(int).tolist(),
        "cuts": _to_json_floats(cuts),
    }


def _clear_json(directory: str) -> None:
    """Remove stale `*.json` so a regeneration never leaves outputs of removed cases."""
    os.makedirs(directory, exist_ok=True)
    for entry in os.listdir(directory):
        if entry.endswith(".json"):
            os.remove(os.path.join(directory, entry))


def main() -> None:
    if xgb.__version__ != "3.4.1":
        raise SystemExit(f"fixtures target xgboost 3.4.1, found {xgb.__version__}")
    _clear_json(FIX_DIR)
    _clear_json(CUT_DIR)
    _clear_json(os.path.join(FIX_DIR, "exports"))
    for name in CASES:
        fixture = build_case(name)
        path = os.path.join(FIX_DIR, f"{name}.json")
        with open(path, "w") as fh:
            json.dump(fixture, fh, separators=(",", ":"))
        print(
            f"{name:<26} {fixture['tier']:<7} {fixture['params']['objective']:<22} "
            f"rounds={fixture['num_round']:<3} pred={len(fixture['xgb_pred'])} "
            f"contribs={len(fixture['xgb_contribs'])}"
        )
    print(f"wrote {len(CASES)} fixtures to {os.path.abspath(FIX_DIR)}")

    for name in CUT_CASES:
        fixture = build_cut_case(name)
        path = os.path.join(CUT_DIR, f"{name}.json")
        with open(path, "w") as fh:
            json.dump(fixture, fh, separators=(",", ":"))
        print(
            f"{name:<26} cuts    rows={fixture['n_rows']:<7} cols={fixture['n_cols']} "
            f"max_bin={fixture['max_bin']:<4} n_cuts={len(fixture['cuts'])}"
        )
    print(f"wrote {len(CUT_CASES)} cut oracles to {os.path.abspath(CUT_DIR)}")


if __name__ == "__main__":
    main()
