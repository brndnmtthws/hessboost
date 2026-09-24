# hessboost

[![crates.io](https://img.shields.io/crates/v/hessboost.svg)](https://crates.io/crates/hessboost)
[![docs.rs](https://img.shields.io/docsrs/hessboost)](https://docs.rs/hessboost)
[![CI](https://github.com/brndnmtthws/hessboost/actions/workflows/ci.yml/badge.svg)](https://github.com/brndnmtthws/hessboost/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A faithful, fast Rust reimplementation of [XGBoost](https://github.com/dmlc/xgboost)
gradient boosting. Its only C dependency is the official zstd library, which
compresses native model files.

`hessboost` re-implements XGBoost's algorithms from scratch in idiomatic
Rust. It includes the regularized second-order boosting objective, exact,
histogram, and approximate tree construction, XGBoost's CPU objectives and
nearly all of its metrics, multi-target labels and vector-leaf trees,
monotone and interaction constraints, categorical splits, DART, gblinear,
and random-forest boosters, continued training, QuadratureTreeSHAP, and
XGBoost JSON/UBJSON model interop, with multi-core (`rayon`) acceleration.
Opt-in extensions beyond XGBoost (conformal intervals, distributional
boosting, compact models, and more) never change default training.

Objective, metric, and parameter names mirror XGBoost, so configurations
transfer directly.

> Using AI coding agents? See [`AGENTS.md`](AGENTS.md) for a task-oriented guide.

## Quick start

```rust
use hessboost::prelude::*;

fn main() -> Result<()> {
    // Dense features (row-major) + labels.
    let x: Vec<f32> = /* n_rows * n_cols values */ vec![0.0; 400];
    let y: Vec<f32> = vec![0.0; 100];

    let dtrain = DMatrix::from_dense(&x, 100, 4)?.with_labels(&y)?;

    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .tree_method(TreeMethod::Hist)   // fast histogram method
        .max_depth(6)
        .eta(0.1)
        .subsample(0.9)
        .lambda(1.0)
        .build()?;

    let model = train(&params, &dtrain, 200)?;
    let preds = model.predict(&dtrain)?;

    model.save_binary("model.bin")?;
    Ok(())
}
```

## Examples

Runnable, self-contained examples live in
[`examples/`](examples). Run any with
`cargo run --release --example <name>`:

| Example | Shows |
|---|---|
| `binary_classification` | `binary:logistic`, watched eval set, early stopping, AUC |
| `multiclass` | `multi:softprob`, per-class probabilities, `predict_class` |
| `ranking` | LambdaMART `rank:ndcg` over query groups |
| `shap` | `predict_contribs` and `predict_interactions` (QuadratureTreeSHAP) |
| `model_io` | native binary / JSON and XGBoost JSON / UBJSON model save & load |
| `custom_objective` | custom loss and custom eval-metric hooks |
| `constraints` | monotone + interaction constraints and categorical features |
| `train_regression` | end-to-end regression with feature importance |
| `conformal` | split-conformal and conformalized-quantile (CQR) prediction intervals |
| `pfn_boost` | boosting from a pretrained prior's logits via `base_margin` (PFN-Boost) |
| `ordered_target_stats` | opt-in CatBoost-style ordered target statistics for a high-cardinality categorical |
| `compact_model` | opt-in feature/threshold reuse penalties and the bit-packed compact model layout (Trees on a Diet) |
| `budget` | budget-mode training (PerpetualBooster) across budgets vs default and validation-tuned training |
| `distributional` | `dist:normal` predictive distributions, intervals, and NLL vs a homoscedastic baseline |

`bench_compare` is the Rust side of the XGBoost timing harness
(`scripts/bench_xgb.py`), not a standalone example.

### Boosting from a pretrained prior

PFN-Boost and LLM-Boost ([Jayawardhana et al., 2025](https://arxiv.org/abs/2502.02672))
seed a GBDT with the per-row logits of a pretrained transformer (TabPFN or an
LLM) so the trees learn the residual of that prior: the initial margin is
`s * score + C` instead of a constant, with the scale `s` tuned on validation
data (`s = 0` is plain boosting). In hessboost that margin is
`DMatrix::with_base_margin`, attached to the train, eval, **and** test
matrices: `base_margin` replaces the model intercept and is not saved with the
model, so predicting on a matrix without it falls back to the intercept. The
`pfn_boost` example runs the method against boosting from scratch across
training-set sizes with a stand-in prior, or on TabPFN logits exported from
Python (`cargo run --release --example pfn_boost -- <dir>`; see its docs).

### Training with a budget instead of tuning

`train_with_budget(&params, &dtrain, &BudgetConfig::new(1.0))` implements
[PerpetualBooster](https://github.com/perpetual-ml/perpetual)'s algorithm: one
`budget` number replaces the learning rate, tree-size limits, and round
count. The budget sets the learning rate (smaller for larger budgets), a
per-tree target loss reduction, and the iteration cap; each split must pass a
five-fold generalization check (its in-fold improvement has to hold up out
of fold), and boosting stops by itself once trees stop generalizing. Larger
budgets train more trees and fit held-out data more closely at a higher
cost; 0.5 (Perpetual's default) to 1.5 is the useful range. The result is an
ordinary gbtree model (every prediction, SHAP, and model-I/O path applies,
XGBoost export included). Settings budget mode derives itself (`eta`,
`max_depth`, `lambda`, subsampling, ...) are refused rather than ignored;
supported objectives are the single-output ones with a pointwise loss
(squared error, pseudo-Huber, logistic, Poisson, Gamma, Tweedie). On
synthetic Friedman #1 data (`budget` example), budget 1.0 is within 1% of,
and budget 1.5 better than, a round count tuned by early stopping on a
validation set for regression, and within 4-10% for binary classification.
Perpetual's dataset-regime heuristics (automatic subsampling, class
reweighting, leaf refinement, and objective/shape-specific schedule
adjustments) are not reproduced; the `hessboost::learner::budget` rustdoc
gives the exact learning-rate, loss-target, split, and stopping rules.

## Feature status

### XGBoost parity (implemented & tested)

- **Boosters:** `gbtree`, **`dart`** (tree dropout), and **`gblinear`**
  (coordinate descent); **boosted random forests** with `num_parallel_tree`
  (each iteration grows that many trees per output from the same gradients,
  each with its own row/column sample and `eta / num_parallel_tree`
  shrinkage, as XGBoost does).
- **Training lifecycle:** **continued training** from an existing model
  (`train_continue` / `train_continue_with_eval`, XGBoost's `xgb_model=`),
  which keeps the model's intercept and continues its RNG stream, so `a + b`
  rounds in two calls grow the same trees as one run; **`process_type =
  update`**, XGBoost's `refresh` updater, recomputing an existing gbtree
  model's statistics and (with `refresh_leaf`) leaf values on new data
  (refused for DART, monotone constraints, vector leaves, linear leaves,
  `path_smooth`, `use_quantized_grad`, `extra_trees`, and reuse penalties);
  **model slicing** by boosting iteration
  (`BoostedModel::slice(begin, end, step)`, XGBoost's `booster[a:b:c]`); and
  **`iteration_range`** prediction (`predict_range`, `predict_margin_range`,
  `predict_leaf_range`, `predict_contribs_range`,
  `predict_interactions_range`).
- **Trees:** `tree_method = exact | hist | approx` (approx uses
  hessian-weighted per-round binning), `grow_policy = depthwise | lossguide`,
  histogram binning with the parent−child subtraction trick, sparsity-aware
  missing-value handling, row subsampling (`subsample`, with `sampling_method
  = uniform` or XGBoost's minimal-variance **`gradient_based`** sampler for
  `hist`/`approx`), and column subsampling (`colsample_bytree`/`bylevel`/
  `bynode`), optionally weighted per feature (`DMatrix::with_feature_weights`,
  XGBoost's weighted draw without replacement).
- **Regularization and constraints:** `lambda`, `alpha`, `gamma`,
  `min_child_weight`, `max_delta_step`, `max_depth`, `max_leaves`, `max_bin`;
  monotone and **interaction constraints** in both the `hist` and `exact`
  builders.
- **Objectives:** `reg:squarederror` (alias `reg:linear`), `reg:logistic`,
  `reg:pseudohubererror` (`huber_slope`), `reg:squaredlogerror`,
  `reg:absoluteerror` (XGBoost 3.4's smoothed MAE), `reg:quantileerror`
  (`quantile_alpha` list, one non-crossing output per quantile),
  `reg:expectileerror` (`expectile_alpha` list, one increasing output per
  expectile), `binary:logistic`, `binary:logitraw` (raw-margin output),
  `binary:hinge`, `multi:softmax`, `multi:softprob`, `count:poisson`,
  `reg:gamma`, `reg:tweedie` (`tweedie_variance_power`), learning-to-rank
  (`rank:pairwise`, `rank:ndcg`, `rank:map`, LambdaMART with
  `lambdarank_num_pair_per_sample`), survival analysis (`survival:cox` on
  signed right-censored times with Breslow ties; `survival:aft` on
  interval-censored label bounds with `normal`/`logistic`/`extreme` noise,
  `aft_loss_distribution[_scale]`), and a user **custom-objective hook**.
  Intercepts are estimated per output exactly as XGBoost 3.4.2 does (label
  mean, class log-frequencies, label quantiles, or a Newton step).
- **Metrics:** `rmse`, `rmsle`, `mae`, `mape`, `mphe` (`huber_slope`),
  `logloss`, `error`, `auc`, `aucpr`, `mlogloss`, `merror`,
  `poisson/gamma/tweedie-nloglik`, `ndcg`, `map`, `pre` (with `@k`; plain
  `pre` cuts at 32 like XGBoost), `quantile` and `expectile` (averaged over
  the configured alphas), `cox-nloglik`, `aft-nloglik`,
  `interval-regression-accuracy`, and a **custom-metric hook**. Default
  metrics follow XGBoost (`ndcg@k`/`map@k` for ranking, `tweedie-nloglik@rho`
  for Tweedie, `mphe` for pseudo-Huber, `rmsle` for squared log error,
  `logloss` on raw margins for `binary:logitraw`, `error` for hinge; the
  default `aft-nloglik` uses the objective's distribution at scale 1, as
  XGBoost's does).
- **Dataset metadata:** labels (`with_labels`, or a row-major multi-target
  `with_label_matrix`), instance and group weights, `base_margin`, query
  groups, label bounds for censored targets (`with_label_bounds`), feature
  types, and per-feature column-sampling weights (`with_feature_weights`).
  Objectives and metrics see a dataset through `MetaInfo`, and each
  objective validates its own label domain.
- **Multi-target labels:** `reg:squarederror`, `reg:pseudohubererror`,
  `reg:absoluteerror`, `reg:logistic`, and `binary:logistic` (multi-label)
  train on a label matrix like XGBoost's default `one_output_per_tree`
  strategy: one tree per target per round, per-target intercepts,
  `[row][target]` predictions and a target axis in SHAP output, and
  `num_target` models in XGBoost JSON/UBJSON. Elementwise metrics average
  over every row and target (row weights repeated per target); `auc`/`aucpr`
  macro-average the targets. Other objectives and the ranking/multiclass
  metrics reject label matrices.
- **Vector-leaf trees:** `multi_strategy = multi_output_tree` (`tree_method =
  hist`) grows one tree per round (per parallel tree) whose leaves hold a
  weight per output, for every multi-output objective (label matrices,
  `multi:softprob`/`softmax`, `reg:quantileerror`/`reg:expectileerror` alpha
  lists, custom objectives), with XGBoost 3.4.2's vector split gain
  (target-summed scores, `min_child_weight` on the mean Hessian), depthwise
  and lossguide growth, missing values, categorical partition splits,
  monotone (pooled weights) and interaction constraints, row/column sampling
  (including `gradient_based` and feature weights), DART, `num_parallel_tree`
  forests, continued training, iteration ranges and slicing. Vector-leaf
  models predict, explain (QuadratureTreeSHAP contributions and interactions
  per output), report feature importance, and round-trip XGBoost's
  `MultiTargetTree` JSON/UBJSON layout (`size_leaf_vector`, `leaf_weights`).
  Custom objectives may grow the structure from **reduced split gradients**
  (`Objective::split_gradient` / `CustomObjective::with_split_gradient`,
  XGBoost's `TreeObjective.split_grad` / SketchBoost) while leaves are refit
  from the full gradients. As in XGBoost, the refresh updater is refused for
  vector leaves, and so are the opt-in extensions below that bypass the
  vector split search (`grow_policy = symmetric`, reuse penalties, quantized
  gradients, `extra_trees`/`path_smooth`/`linear_tree`, budget mode) and the
  compact model format.
- **Modeling:** **native categorical splits** (hist, approx, and exact),
  per-instance `base_margin` (warm-start), SHAP contributions
  (`predict_contribs`) and **interaction values** (`predict_interactions`)
  via XGBoost 3.4's **QuadratureTreeSHAP** (8-point Gauss–Legendre rule, same
  `f32` arithmetic and accumulation order, so imported models reproduce
  XGBoost's values), early stopping, feature importance (weight / gain /
  cover / totals), leaf-index and margin prediction, and k-fold
  cross-validation (`cv`).
- **Model I/O:** libsvm & CSV loaders; native binary (a zstd-compressed,
  column-wise section container) and JSON model I/O, with models from any
  earlier release loading in later ones; and
  **XGBoost-format import/export** of `gbtree`/DART models (numeric and
  categorical splits, forests, multi-output and vector-leaf trees) in both
  XGBoost encodings: JSON (`save_xgboost_json` / `load_xgboost_json`,
  `to_`/`from_xgboost_json`) and **UBJSON** `.ubj` (`save_xgboost_ubjson` /
  `load_xgboost_ubjson`, `to_`/`from_xgboost_ubjson`), written with
  XGBoost's typed arrays. gblinear models, `dist:*` models, and linear-leaf
  trees have no XGBoost encoding and are refused.

### Beyond XGBoost (opt-in)

None of these change default training; each is off unless requested.

- **Prediction intervals** (`hessboost::learner::conformal`):
  distribution-free intervals with finite-sample marginal coverage
  `P(Y ∈ C(X)) ≥ 1 − alpha`: **split conformal** (`SplitConformal`, absolute
  residuals around a point model) and **conformalized quantile regression**
  (`ConformalizedQuantile`, Romano et al. 2019) over two quantile models, two
  outputs of one model, or a `dist:*` model's central band.
- **Ordered target statistics** (`hessboost::data::OrderedTargetEncoder`):
  CatBoost-style encoding of categorical columns as smoothed target means,
  where each training row only sees the rows before it in a seeded random
  permutation, so the preceding rows' statistics never include its own label.
  The default prior is the mean of all training labels, through which each
  row's label still enters its own encoding (with a small weight); set a fixed,
  label-independent `.prior(...)` for strict independence. The fitted
  encoder applies full-training-set statistics to new data, maps unseen
  categories to the prior, and is serde-serializable. Regression and binary
  labels; dense and CSR input.
- **LightGBM tree options** (histogram builder, `hist`/`approx`):
  **`extra_trees`** scores each feature at one random threshold per node
  (drawn inside the node's occupied bin range, seeded by `extra_seed` and the
  per-tree seed); **`path_smooth`** pulls each child's output toward its
  parent's, `w·(n/s)/(n/s+1) + w_parent/(n/s+1)`, and scores splits at the
  smoothed outputs; **`linear_tree`** fits a ridge linear model
  (`linear_lambda` on the slopes) in every leaf on the numerical features
  split on along its path, falling back to the constant leaf for rows with a
  missing model feature (first-round trees stay constant, as in LightGBM;
  refused with the adaptive-leaf `reg:absoluteerror`/`reg:quantileerror`).
  Linear-leaf models round-trip through the native binary and JSON formats;
  XGBoost export, the compact format, and `predict_contribs` /
  `predict_interactions` refuse them (TreeSHAP is undefined for linear
  leaves; LightGBM refuses too). `extra_trees` and `path_smooth` act in the
  per-node split search and are refused with `grow_policy = symmetric`;
  linear leaves apply to symmetric trees.
- **Symmetric (oblivious) trees** (`grow_policy = symmetric`, hist/approx):
  CatBoost-style level-wise growth where every node of a level shares one
  split (feature, threshold, missing direction), chosen to maximize the gain
  summed over the level. `lambda`, `alpha`, `max_delta_step`, monotone and
  interaction constraints apply per node; a node whose share of the level
  split fails `min_child_weight`, `gamma`, or a monotone constraint stays a
  leaf. Numerical features only; `max_depth` in `1..=16`. The trees are
  ordinary trees, so SHAP and XGBoost JSON/UBJSON export work unchanged
  (XGBoost 3.4.2 loads them). Batch prediction recognizes symmetric trees
  and routes rows by the bit pattern of their level comparisons: margins are
  bit-identical to the generic walk, which is 7.5× slower on one core for
  100 depth-6 trees (see `docs/performance.md`).
- **Compact models** ("Trees on a Diet", [Herrmann et al., ICLR 2026](https://arxiv.org/abs/2510.26557)):
  `toad_penalty_feature` (`ι`) and `toad_penalty_threshold` (`ξ`) subtract a
  penalty from the loss change of every split candidate that uses a feature,
  or a threshold of a feature, not yet used anywhere in the ensemble
  (`Δ − s_f·ι − s_t·ξ`, in `gamma`'s units; hist, approx and exact
  builders; refused with `extra_trees`, `path_smooth`, and `grow_policy =
  symmetric`). `BoostedModel::to_compact_bytes` / `CompactModel` store a
  tree ensemble in the paper's layout: a used-feature map, per-feature
  threshold dictionaries at the narrowest exact width (1–32-bit integers,
  binary16 or binary32), a global leaf-value table and pointer-free heap
  trees (a preorder layout for deep unbalanced trees) with bit-packed
  references. `CompactModel::predict_margin` is bit-identical to the source
  model; `BoostedModel::size_report` compares native and compact bytes.
  Forests and scalar multi-output models are supported; gblinear,
  linear-leaf, and vector-leaf models are refused, and XGBoost cannot read
  the format. On the `compact_model` example (binary classification, 16
  sensor features, 100 depth-3 trees, 8000 rows) the compact layout alone is
  2.8x smaller than the zstd-compressed native format (6370 vs 17730 bytes,
  94.98% test accuracy); `ι = ξ = 4` keeps accuracy (95.05%) with 9 of 16
  features and 67 instead of 379 thresholds at 5204 bytes (3.3x), and
  `ι = ξ = 16` reaches 4976 bytes (3.3x, 93.75%). The `f32` leaf table (one
  value per leaf) bounds the savings.
- **Quantized-gradient training** (`use_quantized_grad`, LightGBM's
  quantized training, NeurIPS 2022): each tree's gradients and Hessians are
  rounded to `num_grad_quant_bins` integer levels (stochastic rounding by
  default, seeded and thread-count independent). Histograms then accumulate
  packed integers whose width (32/64/128-bit) follows the node's row count.
  `quant_train_renew_leaf` refits leaf values from the full-precision
  gradients (refused with `path_smooth`, whose leaves keep the outputs
  their splits recorded). Supports `hist`/`approx` with the depthwise and lossguide
  growth policies. Trees differ from full-precision training (test loss is
  within a few percent on the synthetic suites) and are ordinary trees for
  every model format. The speedup is largest when histogram building
  dominates: 1.5× (1 thread) and 1.85× (16 threads) for a 1M × 50 depth-8
  tree. On 50k-row training it is within ±7% (see `docs/performance.md`).
- **Budget-mode training** (`train_with_budget`, `BudgetConfig`):
  PerpetualBooster's hyperparameter-free boosting; see
  [Training with a budget](#training-with-a-budget-instead-of-tuning).
  Deterministic and thread-count independent.
- **Distributional boosting** (NGBoost, [Duan et al. 2020](https://arxiv.org/abs/1910.03225);
  XGBoostLSS, [März 2019](https://arxiv.org/abs/1907.03178)): objectives
  `dist:normal` (`μ`, `ln σ`), `dist:lognormal`, `dist:gamma` (`ln` mean,
  `ln` shape), `dist:poisson` (`ln λ`; equals `count:poisson` without its
  `max_delta_step` Hessian inflation) and `dist:negbinomial` (`ln` mean,
  `ln` size) fit one tree per distribution parameter on the negative
  log-likelihood. `dist_gradient` picks the second-order statistic:
  `fisher` (default; the diagonal Fisher information, which is the full
  Fisher matrix for these orthogonal parameterizations, so leaf steps are
  natural-gradient steps), `hessian` (the exact Hessian diagonal, floored at
  `1e-16`), or `natural` (NGBoost's natural gradient with unit Hessian). The
  intercept is the MLE of the marginal label distribution.
  `BoostedModel::predict_distribution` returns one `Dist` per row with
  `mean`, `variance`, `cdf`, `quantile`, `log_prob`, `crps`, central
  `interval`s and seeded inverse-CDF `sample`; `predict` reports the natural
  parameters `[row][parameter]`. Metrics `nll` (default) and `crps` (closed
  form for Normal/LogNormal/Gamma, exact step sums for the count families).
  With `multi_strategy = multi_output_tree` one shared vector-leaf tree per
  round fits every parameter: `dist_split_direction = random` (default) or
  `cyclic` is parallel gradient boosting
  ([Chapelle et al. 2026](https://arxiv.org/abs/2607.13550), Algorithm 1:
  the structure is grown from one parameter's gradients `e_m` per round, the
  leaves take every parameter's Newton step), `all` the plain vector-leaf
  gain. Native binary/JSON only: XGBoost JSON/UBJSON export and import
  refuse `dist:*`. On the `distributional` example (heteroscedastic Normal
  noise, early stopping) the held-out NLL is 0.768 against 1.014 for a
  squared-error model with one global deviation (0.774 with parallel
  gradient boosting's 112 shared trees instead of 2 × 80), with 90%
  intervals covering 0.891 (0.906 after CQR).

### Not implemented

A GPU backend, distributed or external-memory training, and Python/CLI/C-ABI
wrappers. Some XGBoost options exist only at one setting: gblinear is
`updater=coord_descent` with `feature_selector=cyclic`, LambdaMART is
`lambdarank_pair_method=topk` (no `lambdarank_unbiased` or `ndcg_exp_gain`),
DART has no `sample_type`/`normalize_type`/`one_drop`, and categorical
splits use XGBoost's defaults `max_cat_to_onehot = 4` and
`max_cat_threshold = 64` (not configurable). The metrics
`gamma-deviance` and `ndcg-`/`map-` are missing, and `@` cutoffs are read by
`tweedie-nloglik`, `ndcg`, `map`, and `pre` only (`error@t` evaluates plain
`error`).

## Performance

AArch64 builds use runtime-detected NEON kernels for objective gradients,
probability transforms, metric reductions, and quantile bin search. x86-64
builds use AVX2+FMA for exponential/sigmoid transforms and logistic and
short-softmax gradients, and SSE2 for quantile bin search. Scalar fallbacks
cover other CPUs, short inputs, and values outside the approximation ranges.
Split search is scalar and follows XGBoost's `f32` gain arithmetic exactly.

In the CPU `hist` timing harness (`scripts/bench_xgb.py` with
`bench_compare`: regression, binary, and 4-class workloads at 1, 4, and 16
threads), hessboost's median fit time is lower than XGBoost's in every
configuration, with matching held-out scores. See
[Performance](docs/performance.md) for the measurements and their
provenance, workload definitions, kernel and tree-building benchmarks, and
reproduction commands.

## Testing & parity

```sh
cargo test                          # unit + integration tests
cargo clippy --all-targets -- -D warnings
```

Numerical parity with **XGBoost 3.4.2** is checked in CI by a fixture harness
(`scripts/gen_fixtures.py`, `tests/parity.rs`,
`scripts/check_exports.py`). Each case is checked three ways: **train parity**
(same data and parameters, compare predictions), **import parity**
(`from_xgboost_json` on the XGBoost model: predictions, margins, SHAP
contributions, and SHAP interaction values; `from_xgboost_ubjson` on XGBoost's UBJSON save of the same model
must give the identical model) and **export parity** (`to_xgboost_json` and
`to_xgboost_ubjson` reloaded by XGBoost, with the UBJSON array encodings
matching XGBoost's own re-save).
`exact`-tier cases (every XGBoost objective and deterministic tree method,
categorical splits, label matrices, vector-leaf trees, forests, continued
training, refresh, iteration ranges and slices, and per-round metric
oracles) must agree pointwise (1e-4 / 1e-5). `quality` cases (row/column
subsampling, `gradient_based` sampling, feature weights, random forests,
and DART) use an RMSE or accuracy band because their RNG streams differ. The
`trainonly` gblinear case is pointwise; gblinear XGBoost-JSON import/export
remains unsupported and is asserted explicitly. Histogram and approximate
quantile cuts are compared bit-for-bit.

```sh
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/gen_fixtures.py
cargo test --test parity --release -- --ignored --nocapture
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/check_exports.py
```

XGBoost 3.4.2 ships only as a source tarball (its release notes call it
identical to 3.4.1 apart from Python 3.11 support), so uv builds it with CMake
and a C++ compiler on first use.

See `scripts/README.md` for the case matrix and tolerances.

## License and attribution

Licensed under the **Apache License, Version 2.0**; see [`LICENSE`](LICENSE).
Copyright 2026 Brenden Matthews.

hessboost is a fork of
[sequoia-boost](https://github.com/pgarrett-scripps/sequoia-boost)
(Copyright 2026 Patrick Garrett, Apache-2.0).

It is an independent reimplementation of
[XGBoost](https://github.com/dmlc/xgboost) (Copyright the XGBoost Contributors,
Apache-2.0) built from XGBoost's public descriptions and papers; it contains no
XGBoost source code. "XGBoost" is used descriptively, for algorithmic lineage and
result compatibility. This project is not affiliated with or endorsed by the
XGBoost project.
Budget-mode training re-implements the algorithm of
[PerpetualBooster](https://github.com/perpetual-ml/perpetual) (Copyright 2024
Perpetual ML, Apache-2.0) from its published description and Rust source; no
Perpetual code is copied.
