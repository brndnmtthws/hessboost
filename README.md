# hessboost

[![crates.io](https://img.shields.io/crates/v/hessboost.svg)](https://crates.io/crates/hessboost)
[![docs.rs](https://img.shields.io/docsrs/hessboost)](https://docs.rs/hessboost)
[![CI](https://github.com/brndnmtthws/hessboost/actions/workflows/ci.yml/badge.svg)](https://github.com/brndnmtthws/hessboost/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A faithful, fast Rust reimplementation of
[XGBoost](https://github.com/dmlc/xgboost) gradient boosting. Its only C
dependency is the official zstd library, which compresses native model
files.

hessboost implements XGBoost's algorithms from scratch: the regularized
second-order objective; `exact`, `hist`, and `approx` tree construction;
the `gbtree`, `dart`, and `gblinear` boosters; XGBoost's CPU objectives and
metrics; multi-target and vector-leaf models; constraints; native
categorical splits; QuadratureTreeSHAP; and XGBoost JSON/UBJSON model
interchange. Training runs on all cores (`rayon`) with runtime-detected SIMD
kernels. Parity with **XGBoost 3.4.2** is tested in CI.

Objective, metric, and parameter names mirror XGBoost. A setting hessboost
does not support is refused with an error, never silently ignored.

Opt-in extensions beyond XGBoost (conformal prediction intervals,
distributional boosting, compact models, budget training, and more) are off
by default and never change default training.

## Installation

```sh
cargo add hessboost
```

Requires Rust 1.93 or newer (edition 2024) and a C compiler, which the
`zstd` crate uses to build libzstd.

## Quick start

```rust
use hessboost::config::TreeMethod;
use hessboost::prelude::*;

fn main() -> Result<()> {
    // 100 rows × 4 features, row-major, and one label per row.
    let (n_rows, n_cols) = (100, 4);
    let x: Vec<f32> = (0..n_rows * n_cols).map(|i| (i % 17) as f32 / 17.0).collect();
    let y: Vec<f32> = x.chunks(n_cols).map(|row| 2.0 * row[0] - row[1]).collect();

    let dtrain = DMatrix::from_dense(&x, n_rows, n_cols)?.with_labels(&y)?;

    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .tree_method(TreeMethod::Hist)
        .max_depth(6)
        .eta(0.1)
        .subsample(0.9)
        .build()?;

    let model = train(&params, &dtrain, 200)?;
    let preds = model.predict(&dtrain)?;
    println!("first prediction: {}", preds[0]);

    model.save_binary("model.bin")?;
    let reloaded = BoostedModel::load_binary("model.bin")?;
    assert_eq!(reloaded.predict(&dtrain)?, preds);
    Ok(())
}
```

`hessboost::prelude` holds only the train-and-predict workflow
(`TrainingParams`, `DMatrix`, `train`, `Trainer`, `BoostedModel`, and the
error types); everything else is imported from its module, for example
`hessboost::config::TreeMethod` or `hessboost::data::load_csv`. The
[API documentation](https://docs.rs/hessboost) covers every type and option.

`train(&params, &dtrain, rounds)` is the plain run; `Trainer` adds the
optional arguments of XGBoost's `xgb.train` as builder methods:

```rust
let result = Trainer::new(&params, &dtrain, 1000)
    .eval(&dvalid, "valid")         // watched eval set (repeatable)
    .early_stopping_rounds(20)      // on the last metric of the last eval set
    .train()?;                      // TrainResult { model, history }
let continued = Trainer::new(&params, &dtrain, 50)
    .init_model(&result.model)      // continued training (xgb_model=)
    .train()?
    .model;
```

## Examples

Self-contained programs in [`examples/`](examples); run one with
`cargo run --release --example <name>`.

| Example | Shows |
|---|---|
| `train_regression` | end-to-end regression with feature importance |
| `binary_classification` | `binary:logistic`, a watched eval set, early stopping, AUC |
| `multiclass` | `multi:softprob`, per-class probabilities, `predict_class` |
| `ranking` | LambdaMART `rank:ndcg` over query groups |
| `constraints` | monotone and interaction constraints, categorical features |
| `custom_objective` | custom loss and custom eval-metric hooks |
| `shap` | `predict_contribs` and `predict_interactions` |
| `model_io` | native binary/JSON and XGBoost JSON/UBJSON save and load |
| `pfn_boost` | boosting from a pretrained model's logits via `base_margin` |
| `conformal` | split-conformal and conformalized-quantile (CQR) intervals |
| `distributional` | `dist:normal` predictive distributions, intervals, and NLL |
| `ordered_target_stats` | ordered target statistics for a high-cardinality categorical |
| `compact_model` | reuse penalties and the bit-packed compact model format |
| `budget` | budget-mode training against default and tuned training |

`bench_compare` is the Rust half of the XGBoost timing harness
(`scripts/bench_xgb.py`), not a standalone example.

## XGBoost compatibility

Everything in this section follows XGBoost 3.4.2 and is covered by the
[parity suite](#testing-and-parity).

### Boosters and training

- **Boosters:** `gbtree`, `dart`, and `gblinear` (coordinate descent).
  Boosted random forests with `num_parallel_tree`: each iteration grows
  that many trees per output from the same gradients, each with its own
  row/column sample and `eta / num_parallel_tree` shrinkage.
- **Entry points:** `train` and the `Trainer` builder (watched eval sets
  via `.eval`, `.early_stopping_rounds`, custom `.objective` and
  `.custom_metric` hooks, `.init_model`), and `cv` (k-fold
  cross-validation), all in `hessboost::training`.
- **Continued training:** `Trainer::init_model` (XGBoost's `xgb_model=`)
  keeps the model's intercept and continues its RNG stream, so `a` rounds
  followed by `b` rounds grow the same trees as one run of `a + b`.
- **Refresh:** `process_type = update` (XGBoost's `refresh` updater, via
  `Trainer::init_model`) recomputes an existing gbtree model's node
  statistics and, with `refresh_leaf`, its leaf values on new data. It is
  refused for DART-weighted and vector-leaf models, monotone constraints,
  linear leaves, `path_smooth`, `use_quantized_grad`, `extra_trees`, reuse
  penalties, and more rounds than the model has.
- **Slicing and iteration ranges:** `BoostedModel::slice(0..4, 1)`
  (XGBoost's `booster[a:b:c]`) and the `iteration_range` predictions
  `predict_range`, `predict_margin_range`, `predict_leaf_range`,
  `predict_contribs_range`, `predict_interactions_range`, and
  `predict_distribution_range`, which take a Rust range of iterations
  (`..` for all, `..n`, `2..5`) where XGBoost takes `(begin, end)`.

### Trees

- `tree_method = auto | exact | hist | approx` (`auto` selects `hist`;
  `approx` rebuilds Hessian-weighted cuts every round) and
  `grow_policy = depthwise | lossguide`.
- Sparsity-aware missing-value handling and native categorical splits in
  every tree method, with XGBoost's one-hot and partition search.
- Row subsampling (`subsample`) with `sampling_method = uniform`, or
  XGBoost's `gradient_based` sampler under `hist`/`approx`.
- Column subsampling (`colsample_bytree`, `colsample_bylevel`,
  `colsample_bynode`), optionally weighted per feature
  (`DMatrix::with_feature_weights`).
- Regularization: `lambda`, `alpha`, `gamma`, `min_child_weight`,
  `max_delta_step`, `max_depth`, `max_leaves`, `max_bin`.
- Monotone and interaction constraints in the `hist`, `approx`, and
  `exact` builders.

### Objectives

| Task | Objectives |
|---|---|
| Regression | `reg:squarederror` (alias `reg:linear`), `reg:squaredlogerror`, `reg:logistic`, `reg:pseudohubererror` (`huber_slope`), `reg:absoluteerror`, `reg:gamma`, `reg:tweedie` (`tweedie_variance_power`) |
| Quantiles | `reg:quantileerror` (`quantile_alpha` list), `reg:expectileerror` (`expectile_alpha` list); one output per alpha |
| Classification | `binary:logistic`, `binary:logitraw`, `binary:hinge`, `multi:softmax`, `multi:softprob` (`num_class`) |
| Counts | `count:poisson` |
| Ranking | `rank:pairwise`, `rank:ndcg`, `rank:map` (LambdaMART, `lambdarank_num_pair_per_sample`) |
| Survival | `survival:cox` (negative labels are right-censored), `survival:aft` (interval-censored label bounds, `aft_loss_distribution` `normal`/`logistic`/`extreme`) |
| Custom | `CustomObjective` via `Trainer::objective` |

Intercepts are estimated per output as XGBoost 3.4.2 does.
`reg:absoluteerror` and `reg:quantileerror` use XGBoost 3.4's smoothed
losses.

### Metrics

`rmse`, `rmsle`, `mae`, `mape`, `mphe`, `logloss`, `error`, `auc`, `aucpr`,
`mlogloss`, `merror`, `poisson-nloglik`, `gamma-nloglik`, `tweedie-nloglik`,
`ndcg`, `map`, `pre`, `quantile`, `expectile`, `cox-nloglik`, `aft-nloglik`,
`interval-regression-accuracy`, and custom metrics (`CustomMetric`) via
`Trainer::custom_metric`. Ranking metrics take `@k` cutoffs (plain `pre`
cuts at 32) and `tweedie-nloglik@rho` a variance power. Each objective's
default metric is XGBoost's.

### Multi-output models

- **Label matrices** (`DMatrix::with_label_matrix(y, k)`, row-major) train
  like XGBoost's default `multi_strategy = one_output_per_tree`: one tree
  per target per round and per-target intercepts. Built-in support:
  `reg:squarederror`, `reg:pseudohubererror`, `reg:absoluteerror`,
  `reg:logistic`, and `binary:logistic` (multi-label); custom objectives
  accept a label matrix as wide as their outputs. Elementwise metrics
  average over every cell; `auc`/`aucpr` macro-average the targets;
  ranking, multiclass, survival, and distributional metrics refuse label
  matrices.
- **Vector-leaf trees** (`multi_strategy = multi_output_tree`, `hist`) grow
  one tree per round whose leaves hold a value per output, for every
  multi-output objective (label matrices, `multi:softprob`/`softmax`,
  quantile and expectile alpha lists, custom objectives). They support both
  grow policies, missing values, categorical splits, monotone and
  interaction constraints, row/column sampling, DART, forests, continued
  training, slicing, SHAP, and XGBoost's `MultiTargetTree` JSON/UBJSON
  layout. Custom objectives can grow the tree structure from reduced split
  gradients (`CustomObjective::with_split_gradient`, XGBoost's
  `split_grad`) while leaves are fit from the full gradients; this cannot be
  combined with monotone constraints. The refresh updater, the compact
  format, and the opt-in extensions that replace the split search (symmetric
  trees, reuse penalties, quantized gradients, the LightGBM options, budget
  mode) refuse vector-leaf models.

Predictions for multi-output models are row-major, `[row][output]`.

### Prediction and explainability

- `predict`, `predict_margin`, `predict_class`, `predict_leaf`.
- Per-row `base_margin` (`DMatrix::with_base_margin`) replaces the model
  intercept for that matrix's rows in training and prediction.
- SHAP contributions (`predict_contribs`) and interaction values
  (`predict_interactions`) use XGBoost 3.4's QuadratureTreeSHAP with its
  arithmetic, so imported models reproduce XGBoost's values. One deliberate
  difference: where an intermediate `f32` value overflows and XGBoost
  returns `NaN`, hessboost recomputes in `f64` and returns a finite value.
- Feature importance (`feature_importance(ImportanceType::…)`): `Weight`,
  `Gain`, `Cover`, `TotalGain`, `TotalCover`.

### Data and model formats

- **Input:** dense or CSR `DMatrix`, libsvm and CSV loaders
  (`hessboost::data::{load_libsvm, load_csv}`), and metadata: instance
  weights, query groups (`with_group_sizes`) and group weights, base
  margins, label bounds for censored targets, feature types, and feature
  weights.
- **Native formats:** a checksummed, zstd-compressed binary format
  (`save_binary` / `load_binary`) and JSON (`save_json` / `load_json`),
  both covering every model hessboost trains. Files written by 0.2.0
  and later load in every later release; native binaries from 0.1.x are
  refused.
- **XGBoost interchange:** import and export `gbtree` and DART models
  (numeric and categorical splits, forests, multi-output and vector-leaf
  trees) as XGBoost JSON (`save_xgboost_json` / `load_xgboost_json`) or
  UBJSON (`save_xgboost_ubjson` / `load_xgboost_ubjson`), plus in-memory
  `to_*` / `from_*` variants. gblinear models, custom-objective models,
  `dist:*` models, and linear-leaf trees have no XGBoost encoding and are
  refused.

## Beyond XGBoost (opt-in)

Each extension is off unless requested, and none is part of the parity
suite. Unsupported combinations are refused with an error.

### Prediction intervals

`SplitConformal` (absolute residuals around a point model) and
`ConformalizedQuantile` (CQR, Romano et al. 2019), in
`hessboost::conformal`, calibrate a fitted model on held-out data and
return `(lower, upper)` intervals from `predict_interval`. CQR calibrates
two single-output models, two outputs of one model, or a `dist:*` model's
central band. When calibration and test rows are exchangeable and unseen
in training, coverage is at least `1 − alpha` in finite samples.

### Distributional boosting

After NGBoost ([Duan et al. 2020](https://arxiv.org/abs/1910.03225)) and
XGBoostLSS ([März 2019](https://arxiv.org/abs/1907.03178)): the objectives
`dist:normal`, `dist:lognormal`, `dist:gamma`, `dist:poisson`, and
`dist:negbinomial` fit every distribution parameter (one output each) by
negative log-likelihood.

- `dist_gradient = fisher` (default; natural-gradient Newton steps),
  `hessian`, or `natural` (NGBoost's unit-Hessian natural gradient).
- `predict_distribution` / `predict_distribution_range` return a `Dist`
  (`hessboost::objective::distributional`) per row with `mean`,
  `variance`, `std_dev`, `cdf`, `quantile`, `log_prob`, `crps`, central
  `interval`, and inverse-CDF `sample`; `predict` returns the parameters
  `[row][parameter]`.
- Metrics: `nll` (default) and `crps`.
- With `multi_strategy = multi_output_tree`, one shared tree per round fits
  every parameter: `dist_split_direction = random` (default) or `cyclic`
  grows the structure from one parameter's gradients per round (parallel
  gradient boosting, [Chapelle et al. 2026](https://arxiv.org/abs/2607.13550));
  `all` uses the plain vector-leaf gain.
- Native formats only; XGBoost import and export refuse `dist:*`.

On the `distributional` example (heteroscedastic Normal noise), held-out NLL
is 0.768 against 1.014 for a squared-error model with one global deviation,
and 90% intervals cover 0.891 of test rows (0.906 after CQR).

### Budget training

`train_with_budget(&params, &dtrain, &BudgetConfig::new(1.0))`
(`hessboost::training::budget`) reimplements
[PerpetualBooster](https://github.com/perpetual-ml/perpetual)'s algorithm:
one `budget` number replaces the learning rate, tree-size limits, and round
count. Below the root, every split must pass a five-fold generalization
check, and boosting stops by itself when trees stop improving or
generalizing. Larger budgets train more trees and fit more closely; 0.5
(Perpetual's default) to 1.5 is the useful range. The result is an ordinary
gbtree model, and training is deterministic.

Supported objectives: `reg:squarederror`, `reg:pseudohubererror`,
`reg:logistic`, `binary:logistic`, `binary:logitraw`, `count:poisson`,
`reg:gamma`, and `reg:tweedie`. Parameters that budget mode derives itself
(`eta`, `max_depth`, `lambda`, sampling, ...) are refused. Perpetual's
dataset-specific heuristics are not reproduced; the
[`training::budget`](https://docs.rs/hessboost/latest/hessboost/training/budget/)
docs give the exact rules.

On Friedman #1 data (`budget` example), budget 1.0 comes within 1% of the
test RMSE of an early-stopping-tuned model and budget 1.5 beats it; for
binary classification, budgets 1.0–1.5 are 3–9% behind the tuned logloss.

### Tree options from LightGBM and CatBoost

These need a tree booster with `hist` or `approx` and one output per tree.

- **`extra_trees`** scores one random candidate per feature and node (a bin
  boundary for numerical features, a category prefix for categorical ones),
  seeded by `extra_seed`.
- **`path_smooth`** shrinks each child's value toward its parent's and
  scores splits at the smoothed values.
- **`linear_tree`** fits a ridge model (`linear_lambda`) in every leaf on
  the numerical features split on along its path, falling back to the
  constant leaf for rows missing one of them. Not available with
  `reg:absoluteerror` or `reg:quantileerror`. Linear-leaf models use the
  native formats only; SHAP, XGBoost export, and the compact format refuse
  them.
- **Symmetric (oblivious) trees** (`grow_policy = symmetric`): every node of
  a level shares one split, chosen by the gain summed over the level.
  Numerical features only, `max_depth` in `1..=16`, `max_leaves = 0`;
  `extra_trees` and `path_smooth` are refused. The results are ordinary
  trees (SHAP and XGBoost export work), and batch prediction routes rows by
  bit pattern: on one core, 7.5× faster than the generic walk for 100
  depth-6 trees, with bit-identical margins.

### Quantized-gradient training

`use_quantized_grad` (LightGBM's quantized training, NeurIPS 2022) rounds
each tree's gradients to `num_grad_quant_bins` levels (2–127, default 4;
`stochastic_rounding` by default, seeded and thread-count independent) and
accumulates integer histograms. `quant_train_renew_leaf` refits leaves from
the full-precision gradients. Supports `hist`/`approx` with depthwise or
lossguide growth and one output per tree. Trees differ from full-precision
training; the gain is largest when histogram building dominates, up to
1.85× on a 1M × 50, depth-8 tree at 16 threads, while 50k-row training is
within ±7% ([measurements](docs/performance.md)).

### Compact models

After *Boosted Trees on a Diet* ([Herrmann et al., ICLR 2026](https://arxiv.org/abs/2510.26557)):

- **Reuse penalties** `toad_penalty_feature` and `toad_penalty_threshold`
  (in `gamma`'s units) penalize split candidates that use a feature or
  threshold not yet used anywhere in the ensemble. Available in every tree
  method; refused with `extra_trees`, `path_smooth`, and symmetric trees.
- **Compact format:** `BoostedModel::to_compact_bytes` / `to_compact` give a
  `CompactModel` (`hessboost::model::compact`) with deduplicated,
  bit-packed feature, threshold, and leaf tables. Its `predict_margin` is
  bit-identical to the source model's; `BoostedModel::size_report` compares
  native and compact sizes. Forests and scalar multi-output models are
  supported; gblinear, linear-leaf, and vector-leaf models are refused.
  XGBoost cannot read the format.

On the `compact_model` example (100 depth-3 trees, 16 features), the compact
format is 2.8× smaller than the zstd-compressed native binary (6370 vs
17730 bytes, 94.98% test accuracy); with both penalties at 4 it is 3.3×
smaller (9 features, 67 thresholds) at 95.05% accuracy.

### Ordered target statistics

`OrderedTargetEncoder` (`hessboost::data::target_stats`) encodes
categorical columns as smoothed target means, CatBoost-style: each training
row sees only the rows before it in a seeded random permutation. The
default prior is the training-label mean; set a fixed `.prior(...)` to keep
every row's label out of its own encoding. The `FittedTargetEncoder`
applies full-training statistics to new data, maps unseen categories to the
prior, and is serde-serializable. Regression and binary labels; dense and
CSR input.

### Boosting from a pretrained prior

PFN-Boost and LLM-Boost ([Jayawardhana et al., 2025](https://arxiv.org/abs/2502.02672))
start boosting from the per-row logits of a pretrained model (TabPFN or an
LLM), so the trees learn its residual. In hessboost the scaled logits are a
`base_margin` on the train, eval, and test matrices. `base_margin` is not
saved with the model, so predicting on a matrix without it falls back to
the model intercept. The `pfn_boost` example compares this with boosting
from scratch, using a synthetic prior or TabPFN logits exported from Python
(`cargo run --release --example pfn_boost -- <dir>`).

## Not implemented

- GPU training, distributed or external-memory training, and Python, CLI,
  or C-ABI bindings.
- XGBoost options available at one setting only: gblinear uses
  `updater = coord_descent` with `feature_selector = cyclic`; LambdaMART
  uses `lambdarank_pair_method = topk` (no `lambdarank_unbiased` or
  `ndcg_exp_gain`); DART has no `sample_type`, `normalize_type`, or
  `one_drop`; categorical splits use XGBoost's defaults
  `max_cat_to_onehot = 4` and `max_cat_threshold = 64`.
- The metrics `gamma-deviance` and `ndcg-`/`map-`. Only ranking metrics and
  `tweedie-nloglik` read an `@` suffix; `error@t` evaluates plain `error`.
- gblinear models cannot be imported from or exported to XGBoost formats.

## Performance

On an Apple M3 Max against XGBoost 3.4.1 (numerically identical to 3.4.2),
with CPU `hist`, 100 rounds, and depth 6, hessboost's median fit time was
lower in all 12 measured configurations (regression, wide regression,
binary, and 4-class workloads at 1, 4, and 16 threads): 2.3–2.8× faster
single-threaded and 1.4–1.6× at 16 threads, with matching held-out scores.

AArch64 builds use NEON kernels for objective gradients, probability
transforms, metric reductions, and quantile bin search. x86-64 builds use
AVX2+FMA for exponential and sigmoid transforms and logistic and 2-/4-class
softmax gradients, and SSE2 for bin search. Features are detected at
runtime, with scalar fallbacks. Split search is scalar and matches
XGBoost's `f32` gain arithmetic exactly.

[`docs/performance.md`](docs/performance.md) has the measurements,
workload definitions, kernel and tree-building benchmarks, and reproduction
commands.

## Testing and parity

```sh
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

Parity with XGBoost 3.4.2 is checked in CI by a fixture harness. Fixtures
are generated locally into the gitignored `fixtures/` directory; XGBoost
3.4.2 ships as a source tarball, so the first run builds it with CMake and a
C++ compiler.

```sh
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/gen_fixtures.py
cargo test --test parity --release -- --ignored --nocapture
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/check_exports.py
```

Each case checks **train parity** (same data and parameters, compare
predictions), **import parity** (load XGBoost's JSON and UBJSON saves and
compare predictions, margins, and SHAP values), and **export parity**
(XGBoost reloads hessboost's exports). Deterministic cases must match
pointwise (training predictions within `1e-4`, probabilities `1e-5`), and
histogram and `approx` cuts must match bit for bit. RNG-driven cases
(row/column sampling, feature weights, forests, DART) must stay within a
quality band, because the random streams differ. See
[`scripts/README.md`](scripts/README.md) for the case matrix and
tolerances.

## License and attribution

Licensed under the [Apache License, Version 2.0](LICENSE). Copyright 2026
Brenden Matthews.

hessboost is a fork of
[sequoia-boost](https://github.com/pgarrett-scripps/sequoia-boost)
(Copyright 2026 Patrick Garrett, Apache-2.0).

It is an independent reimplementation of
[XGBoost](https://github.com/dmlc/xgboost) (Copyright the XGBoost
Contributors, Apache-2.0), built from XGBoost's public descriptions and
papers; it contains no XGBoost source code. "XGBoost" is used descriptively,
for algorithmic lineage and result compatibility; this project is not
affiliated with or endorsed by the XGBoost project.

Budget-mode training reimplements the algorithm of
[PerpetualBooster](https://github.com/perpetual-ml/perpetual) (Copyright 2024
Perpetual ML, Apache-2.0) from its published description and Rust source; no
Perpetual code is copied.
