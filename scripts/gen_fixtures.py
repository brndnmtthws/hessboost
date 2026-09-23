#!/usr/bin/env python3
"""Generate XGBoost 3.4.2 parity fixtures for hessboost.

Trains real XGBoost (single thread) on deterministic synthetic datasets, one
case per supported feature, and writes `fixtures/<name>.json` holding the data,
the exact `xgb.train` parameter dict, XGBoost's test-set predictions (transformed,
raw margin, SHAP contributions on the first 50 rows) and the saved model JSON,
plus the same model's UBJSON encoding (`save_raw("ubj")`) as the sidecar
`fixtures/<name>.ubj` named by the fixture's `xgb_model_ubj`.
`tests/parity.rs` consumes these; the fixture schema is the
contract between the two.

Tiers:
  exact    every assertion is pointwise (train, import, export).
  quality  the model is RNG-driven (subsample, colsample, DART) or the objective
           is not yet measured for pointwise agreement (rank:*), so training is
           held to a quality band instead: RMSE <= band * xgb RMSE, or
           accuracy / NDCG@20 >= xgb - band. Import/export stay pointwise.

Usage:
    uv run --with-requirements scripts/requirements-xgboost.txt python scripts/gen_fixtures.py
"""

from __future__ import annotations

import json
import os
import tempfile
import zlib

import numpy as np
import xgboost as xgb

# The pinned release (scripts/requirements-xgboost.txt).
XGBOOST_VERSION = "3.4.2"

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
# Per-round eval-metric oracles: |hessboost - xgboost| <= TOL_EVALS * max(1, |xgboost|).
TOL_EVALS = 1e-5
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


def y_multi_regression(x, rng):
    """Three regression targets (a label matrix) of different shapes."""
    n = x.shape[0]
    noise = 0.1 * rng.standard_normal((n, 3))
    return np.stack(
        [
            2 * x[:, 0] - 3 * x[:, 1] ** 2,
            np.sin(6 * x[:, 2]) + x[:, 3],
            4 * x[:, 4] * x[:, 5] - 1,
        ],
        axis=1,
    ) + noise


def y_multi_label(x, rng):
    """Three independent binary labels (multi-label classification)."""
    logits = np.stack([3 * x[:, 0] - 2 * x[:, 1], 4 * x[:, 2] - 2, 2 * x[:, 3] - 3 * x[:, 4] + 1], axis=1)
    return (1 / (1 + np.exp(-logits)) > rng.random(logits.shape)).astype(np.float32)


def y_multi_heavy_tail(x, rng):
    """Two regression targets with heavy-tailed noise (for pseudo-Huber)."""
    n = x.shape[0]
    return np.stack(
        [
            2 * x[:, 0] - 3 * x[:, 1] ** 2 + 0.3 * rng.standard_t(2, n),
            x[:, 2] + 0.5 * x[:, 3] + 0.3 * rng.standard_t(2, n),
        ],
        axis=1,
    )


# ---------------------------------------------------------------------------
# Case matrix
# ---------------------------------------------------------------------------

# name -> (target fn, param overrides, options)
# A target fn returns the label vector, an (n, K) array to train on a label
# matrix (K targets), or a `(labels | None, lower, upper)` tuple for survival
# cases that carry `label_lower_bound`/`label_upper_bound` (labels `None` when
# the objective reads the bounds only, e.g. survival:aft).
# options: tier, num_round, missing (fraction of NaN features), weighted, ranking,
#          tol_train (override), drop (params removed from TREE_BASE),
#          feature_weights (per-column DMatrix weights for column sampling),
#          evals (record per-round metrics on the labeled test set; the
#          params' `eval_metric` list, or XGBoost's default metric when absent),
#          test_weighted (per-row test-set weights; constant within each
#          query group and passed to XGBoost per group for ranking cases)
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
    # per-round metric oracles on a weighted test set
    "evals_reg_d4": (
        y_regression,
        dict(max_depth=4, eval_metric=["rmse", "mae"]),
        dict(evals=True, test_weighted=True),
    ),
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
    # Deprecated alias: XGBoost 3.4.2 trains it as reg:squarederror (with a
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
    # evals: the default metric is `mphe` (pseudo-Huber without factor 2)
    "huber_d4": (
        y_heavy_tail,
        dict(objective="reg:pseudohubererror", huber_slope=1.0, max_depth=4),
        dict(evals=True),
    ),
    # small objectives; `evals` checks each default metric (rmsle, logloss on
    # raw margins, error on the 0/1 hinge output) round by round
    "squaredlog_d4": (y_gamma, dict(objective="reg:squaredlogerror", max_depth=4), dict(evals=True)),
    "squaredlog_exact_d4": (
        y_gamma,
        dict(objective="reg:squaredlogerror", tree_method="exact", max_depth=4),
        dict(evals=True),
    ),
    "nobs_squaredlog_weighted_d4": (
        y_gamma,
        dict(objective="reg:squaredlogerror", max_depth=4),
        dict(drop=("base_score",), weighted=True, evals=True, test_weighted=True),
    ),
    "logitraw_d6": (y_binary, dict(objective="binary:logitraw"), dict(evals=True)),
    "logitraw_exact_d4": (
        y_binary,
        dict(objective="binary:logitraw", tree_method="exact", max_depth=4),
        dict(evals=True),
    ),
    "nobs_logitraw_weighted_d4": (
        y_binary,
        dict(objective="binary:logitraw", max_depth=4),
        dict(drop=("base_score",), weighted=True, evals=True),
    ),
    "nobs_logitraw_spw3_d4": (
        y_binary,
        dict(objective="binary:logitraw", scale_pos_weight=3.0, max_depth=4),
        dict(drop=("base_score",), evals=True),
    ),
    "hinge_d4": (y_binary, dict(objective="binary:hinge", max_depth=4), dict(evals=True)),
    "hinge_exact_d4": (
        y_binary,
        dict(objective="binary:hinge", tree_method="exact", max_depth=4),
        dict(evals=True),
    ),
    "nobs_hinge_weighted_d4": (
        y_binary,
        dict(objective="binary:hinge", max_depth=4),
        dict(drop=("base_score",), weighted=True, evals=True, test_weighted=True),
    ),
    # metric oracles: rmsle / mape / mphe (non-default slope) on a positive
    # target, and pre / pre@k on weighted query groups
    "evals_metrics_reg_d4": (
        y_gamma,
        dict(max_depth=4, huber_slope=0.7, eval_metric=["rmsle", "mape", "mphe"]),
        dict(evals=True, test_weighted=True),
    ),
    "evals_pre_rank_d4": (
        y_relevance_binary,
        dict(
            objective="rank:ndcg",
            max_depth=4,
            lambdarank_pair_method="topk",
            lambdarank_num_pair_per_sample=GROUP_SIZE,
            eval_metric=["pre", "pre@5"],
        ),
        dict(ranking=True, evals=True, test_weighted=True),
    ),
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
    # multi-target labels (a label matrix, one output per column) on the
    # default one_output_per_tree strategy; intercepts are estimated per target.
    "multi_reg3_d6": (y_multi_regression, {}, {}),
    "multi_reg3_nobs_d6": (y_multi_regression, {}, dict(drop=("base_score",))),
    "multi_reg3_exact_nobs_d6": (y_multi_regression, dict(tree_method="exact"), dict(drop=("base_score",))),
    "multi_label_binary_d4": (
        y_multi_label,
        dict(objective="binary:logistic", max_depth=4),
        dict(drop=("base_score",), tol_train=TOL_TRAIN_PROB),
    ),
    "multi_label_spw3_d4": (
        y_multi_label,
        dict(objective="binary:logistic", scale_pos_weight=3.0, max_depth=4),
        dict(drop=("base_score",), tol_train=TOL_TRAIN_PROB),
    ),
    "multi_huber_weighted_d4": (
        y_multi_heavy_tail,
        dict(objective="reg:pseudohubererror", huber_slope=1.0, max_depth=4),
        dict(drop=("base_score",), weighted=True),
    ),
    # alpha-list objectives: one output per alpha (quantile_alpha /
    # expectile_alpha lists become one scalar tree per alpha and round) and
    # the smoothed MAE. `nobs_*` cases exercise the per-output intercepts
    # (label quantiles, the MAE Newton step from the mean, the monotone
    # expectile step); the others broadcast base_score through ProbToMargin.
    "quantile_d4": (y_heavy_tail, dict(objective="reg:quantileerror", quantile_alpha=0.5, max_depth=4), {}),
    "nobs_quantile_multi_d4": (
        y_heavy_tail,
        dict(objective="reg:quantileerror", quantile_alpha=[0.1, 0.5, 0.9], max_depth=4),
        dict(drop=("base_score",)),
    ),
    "nobs_quantile_multi_exact_d4": (
        y_heavy_tail,
        dict(objective="reg:quantileerror", quantile_alpha=[0.1, 0.5, 0.9], tree_method="exact", max_depth=4),
        dict(drop=("base_score",)),
    ),
    "nobs_quantile_weighted_d4": (
        y_heavy_tail,
        dict(objective="reg:quantileerror", quantile_alpha=[0.2, 0.8], max_depth=4),
        dict(drop=("base_score",), weighted=True),
    ),
    "nobs_mae_d4": (y_heavy_tail, dict(objective="reg:absoluteerror", max_depth=4), dict(drop=("base_score",))),
    "nobs_mae_weighted_exact_d4": (
        y_heavy_tail,
        dict(objective="reg:absoluteerror", tree_method="exact", max_depth=4),
        dict(drop=("base_score",), weighted=True),
    ),
    "nobs_expectile_d4": (
        y_heavy_tail,
        dict(objective="reg:expectileerror", expectile_alpha=0.3, max_depth=4),
        dict(drop=("base_score",)),
    ),
    "nobs_expectile_multi_d4": (
        y_heavy_tail,
        dict(objective="reg:expectileerror", expectile_alpha=[0.1, 0.5, 0.9], max_depth=4),
        dict(drop=("base_score",)),
    ),
    "expectile_multi_weighted_d4": (
        y_heavy_tail,
        dict(objective="reg:expectileerror", expectile_alpha=[0.2, 0.8], max_depth=4),
        dict(weighted=True),
    ),
    # multi-target smoothed MAE: a label matrix with per-target intercepts.
    "multi_mae_weighted_d4": (
        y_multi_heavy_tail,
        dict(objective="reg:absoluteerror", max_depth=4),
        dict(drop=("base_score",), weighted=True),
    ),
    # quality tier: RNG-driven sampling, pointwise agreement is not expected
    "subsample_0p8_d6": (y_regression, dict(subsample=0.8, seed=42), dict(tier="quality")),
    "colsample_bytree_0p5_d6": (y_regression, dict(colsample_bytree=0.5, seed=42), dict(tier="quality")),
    # sampling_method=gradient_based: XGBoost's CPU MVS row sampler (hist and approx)
    "gradient_based_0p3_d6": (
        y_regression,
        dict(sampling_method="gradient_based", subsample=0.3, seed=42),
        dict(tier="quality"),
    ),
    "gradient_based_binary_0p5_d6": (
        y_binary,
        dict(objective="binary:logistic", sampling_method="gradient_based", subsample=0.5, seed=42),
        dict(tier="quality"),
    ),
    "gradient_based_approx_0p4_d6": (
        y_regression,
        dict(tree_method="approx", sampling_method="gradient_based", subsample=0.4, seed=42),
        dict(tier="quality"),
    ),
    # feature-weighted column sampling: skewed weights favor the noise columns
    "feature_weights_bynode_0p5_d6": (
        y_regression,
        dict(colsample_bynode=0.5, seed=42),
        dict(tier="quality", num_round=100, feature_weights=[0.2, 3.0, 0.5, 1.0, 1.0, 2.0, 4.0, 0.1]),
    ),
    "feature_weights_bytree_bylevel_d6": (
        y_regression,
        dict(colsample_bytree=0.75, colsample_bylevel=0.5, seed=42),
        dict(tier="quality", num_round=100, feature_weights=[4.0, 2.0, 1.0, 0.5, 0.0, 0.5, 0.25, 8.0]),
    ),
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


def _to_json_bounds(a: np.ndarray) -> list:
    """f32 label bounds -> list of Python floats; +/-inf -> "inf"/"-inf" strings
    (JSON has no infinity; bounds are never NaN)."""
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    assert not np.isnan(a).any(), "label bounds must not be NaN"
    out = a.astype(np.float64).astype(object)
    out[np.isposinf(a)] = "inf"
    out[np.isneginf(a)] = "-inf"
    return out.tolist()


def _inject_missing(x: np.ndarray, frac: float, rng) -> np.ndarray:
    x = x.copy()
    x[rng.random(x.shape) < frac] = np.nan
    return x


def _save_model_ubj(booster: xgb.Booster, name: str) -> str:
    """Write the model's UBJSON encoding (`save_raw("ubj")`, the bytes of
    `save_model("m.ubj")`) next to the fixture; returns the file name."""
    file_name = f"{name}.ubj"
    with open(os.path.join(FIX_DIR, file_name), "wb") as fh:
        fh.write(booster.save_raw("ubj"))
    return file_name


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
    target_out = target(x, rng)
    lower = upper = None
    if isinstance(target_out, tuple):
        y, lower, upper = target_out
    else:
        y = target_out
    if "missing" in opts:
        x = _inject_missing(x, opts["missing"], rng)
    x_train, x_test = x[:N_TRAIN], x[N_TRAIN:]
    y_train = y_test = lo_train = lo_test = hi_train = hi_test = None
    if y is not None:
        y = np.asarray(y, dtype=np.float32)
        y_train, y_test = y[:N_TRAIN], y[N_TRAIN:]
    if lower is not None:
        lower = np.asarray(lower, dtype=np.float32)
        upper = np.asarray(upper, dtype=np.float32)
        lo_train, lo_test = lower[:N_TRAIN], lower[N_TRAIN:]
        hi_train, hi_test = upper[:N_TRAIN], upper[N_TRAIN:]

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
    feature_weights = opts.get("feature_weights")
    if feature_weights is not None:
        assert len(feature_weights) == N_COLS
        dtrain.set_info(feature_weights=np.asarray(feature_weights, dtype=np.float32))
    if lo_train is not None:
        dtrain.set_float_info("label_lower_bound", lo_train)
        dtrain.set_float_info("label_upper_bound", hi_train)
    dtest = xgb.DMatrix(x_test, nthread=1, feature_types=feature_types)
    dcontrib = xgb.DMatrix(x_test[:N_CONTRIB_ROWS], nthread=1, feature_types=feature_types)

    # Test-set weights are drawn after every other draw so that cases without
    # them keep their data.
    test_weights = None
    if opts.get("test_weighted"):
        if test_group_sizes is not None:
            per_group = rng.uniform(0.5, 2.0, len(test_group_sizes)).astype(np.float32)
            test_weights = np.repeat(per_group, test_group_sizes)
        else:
            test_weights = rng.uniform(0.5, 2.0, N_TEST).astype(np.float32)

    evals_result: dict = {}
    evals = []
    if opts.get("evals"):
        deval = xgb.DMatrix(x_test, label=y_test, nthread=1, feature_types=feature_types)
        if test_group_sizes is not None:
            deval.set_group(test_group_sizes)
            if test_weights is not None:
                deval.set_weight(per_group)
        elif test_weights is not None:
            deval.set_weight(test_weights)
        if lo_test is not None:
            deval.set_float_info("label_lower_bound", lo_test)
            deval.set_float_info("label_upper_bound", hi_test)
        evals = [(deval, "test")]

    booster = xgb.train(
        dict(params, nthread=1),
        dtrain,
        num_boost_round=num_round,
        evals=evals,
        evals_result=evals_result,
        verbose_eval=False,
    )
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
        # Label columns: y_train / y_test are row-major [row][target].
        "n_targets": 1 if y is None or y.ndim == 1 else int(y.shape[1]),
        "x_train": _to_json_floats(x_train),
        "y_train": [] if y_train is None else _to_json_floats(y_train),
        "x_test": _to_json_floats(x_test),
        "y_test": [] if y_test is None else _to_json_floats(y_test),
        "label_lower_bound": None if lo_train is None else _to_json_bounds(lo_train),
        "label_upper_bound": None if hi_train is None else _to_json_bounds(hi_train),
        "test_label_lower_bound": None if lo_test is None else _to_json_bounds(lo_test),
        "test_label_upper_bound": None if hi_test is None else _to_json_bounds(hi_test),
        "test_weights": None if test_weights is None else _to_json_floats(test_weights),
        "xgb_evals": evals_result["test"] if evals else None,
        "weights": None if weights is None else _to_json_floats(weights),
        "feature_weights": feature_weights,
        "group_sizes": group_sizes,
        "test_group_sizes": test_group_sizes,
        "feature_types": feature_types,
        "xgb_pred": _to_json_floats(pred),
        "xgb_margin": _to_json_floats(margin),
        "xgb_contribs": _to_json_floats(contribs),
        "xgb_model": _save_model_json(booster),
        "xgb_model_ubj": _save_model_ubj(booster, name),
        "tol": {
            "train": tol_train,
            "import": TOL_IMPORT,
            "contribs": TOL_CONTRIBS,
            "evals": TOL_EVALS,
        },
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


def _clear_outputs(directory: str) -> None:
    """Remove stale `*.json` / `*.ubj` so a regeneration never leaves outputs of removed cases."""
    os.makedirs(directory, exist_ok=True)
    for entry in os.listdir(directory):
        if entry.endswith((".json", ".ubj")):
            os.remove(os.path.join(directory, entry))


def main() -> None:
    if xgb.__version__ != XGBOOST_VERSION:
        raise SystemExit(f"fixtures target xgboost {XGBOOST_VERSION}, found {xgb.__version__}")
    _clear_outputs(FIX_DIR)
    _clear_outputs(CUT_DIR)
    _clear_outputs(os.path.join(FIX_DIR, "exports"))
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
