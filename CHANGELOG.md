# Changelog

All notable changes to hessboost are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate
follows [Semantic Versioning](https://semver.org/) (pre-1.0: a minor release
may break the API).

## [0.2.0] - Unreleased

### Upgrading from 0.1

- **Saved models:** 0.2.0 refuses both native binary (`save_binary`) and
  native JSON (`save_json`) files written by 0.1.x. To keep a model, export
  it with 0.1.1's `save_xgboost_json` and load it with
  `BoostedModel::load_xgboost_json`: gbtree and DART models (numeric or
  categorical splits, multiclass) reload with identical predictions. 0.1.1
  cannot export gblinear or custom-objective models, and exports an
  early-stopped model only up to its best iteration, without
  `best_iteration`; retrain those. Files written by 0.2.0 load in every
  later release.
- **Retrained models differ:** seeded sampling (rows, columns, DART, `cv`
  folds) draws from a new in-crate RNG, and categorical splits now follow
  XGBoost's one-hot and partition search, so such models differ from 0.1.1's
  for the same seed. SHAP values follow XGBoost 3.4's QuadratureTreeSHAP.
- **Build:** a C compiler is now required: native models are compressed
  with libzstd (`zstd` crate).
- **Modules:** `learner` and `booster` are gone. Training is in `training`
  (`train`, `Trainer`, `TrainResult`, `RoundEval`, `cv`, `CvResult`), the
  model in `model` (`BoostedModel`, `ImportanceType`). Implementation
  modules are private: `tree::{builder, constraints, gain, hist, sampler}`
  (with `GradStats`, `RegParams`, `calc_gain`, `calc_weight`),
  `data::{ghist, quantile}` (`GHistIndex`, `HistCuts`), `data::{Entry,
  CscView}`, `booster::gblinear`, and `EvalSet`.
- **Prelude:** only `TrainingParams`, `DMatrix`, `train`, `Trainer`,
  `BoostedModel`, `HessboostError`, and `Result`. Import the parameter enums
  from `config`, `CsvOptions`/`FeatureType` from `data`, `cv`/`CvResult`/
  `TrainResult` from `training`, `ImportanceType` from `model`, and the
  objective and metric hooks from `objective`/`metric`.
- **Training entry points:** `train_with_eval`, `train_with_objective`, and
  `train_with_custom_metric` are replaced by the `Trainer` builder:
  `Trainer::new(&params, &dtrain, rounds).eval(&dvalid, "valid")
  .early_stopping_rounds(k).objective(&obj).custom_metric(metric).train()`.
- **Objectives:** the built-ins lost their `Objective` suffix
  (`LogisticObjective` → `Logistic`, `SquaredErrorObjective` →
  `SquaredError`, `LambdaMartObjective` → `LambdaMart`, ...), and
  `create_objective(params)` is `create_objective(params, n_targets)`.
- **`Objective` trait:** `base_margins(labels, weights)` is replaced by
  `base_margins_info(&MetaInfo)` (build one with `MetaInfo::new(labels,
  weights, group)`), the only intercept hook; `prob_to_margin(f32)` by the
  row-wise `probs_to_margins(&mut [f32])`, the only link hook (identity by
  default).
- **Metrics:** `create_metric(name, num_class)` is `create_metric(name,
  &params)`; `create_metrics` is private. Metric names with a suffix
  hessboost does not implement (`error@0.7`, `ndcg@3-`, `ndcg@2.9`) are
  refused instead of ignored or truncated.
- **Unit-struct built-ins** (`SquaredError`, `Gamma`, `Rmse`, `Mae`,
  `LogLoss`, `ErrorRate`, `Auc`, `AucPr`, ...) are `#[non_exhaustive]`:
  write `Rmse::default()`, or use `create_objective`/`create_metric`.
- **`#[non_exhaustive]`:** the enums `BoosterKind`, `TreeMethod`,
  `GrowPolicy`, `FeatureType`, `ImportanceType`, and `HessboostError` (and
  its `Parse` variant) need a `_` arm in exhaustive matches; the structs
  `TrainingParams`, `ObjectiveParams`, `CsvOptions`, `GroupInfo`,
  `TrainResult`, `RoundEval`, `CvResult`, and `tree::Node` cannot be built
  with a struct literal or `..Default::default()`: start from `Default`, a
  builder, or a constructor and assign fields (`let mut p =
  TrainingParams::default(); p.eta = 0.1;`). `Monotone` and `GradPair`
  stay exhaustive.
- **Prediction:** `predict_margin_limited(data, ntree_limit)` (a raw tree
  count) is `predict_margin_range(data, iterations)`, which counts boosting
  iterations and takes any `impl RangeBounds<usize>`: `..` is every
  iteration regardless of early stopping, `..n` the first `n`, and `0..0`
  the intercept alone. gblinear models accept only `..`. `predict_range`
  takes the same ranges; `predict_leaf_range`, `predict_contribs_range`, and
  `predict_interactions_range` take ranges starting at 0.
- **XGBoost interchange:** the free functions
  `model::{export_xgboost_json, import_xgboost_json}` are gone; use
  `BoostedModel::{to,from,save,load}_xgboost_json`.
- **Trees:** `RegTree::predict_row` and `RegTree::scale_leaves` are private
  and `RegTree` has no `Default`; a row's output in a tree is
  `leaf_vector(leaf_id_dense(row, missing))` (or `leaf_id_with`).
- **Data:** `DMatrix::row_into` and `DMatrix::to_csc` are private.

### Added

- **XGBoost 3.4.2 parity:** deterministic configurations match XGBoost
  pointwise, with bit-identical `hist`/`approx` cuts, categorical one-hot and
  partition search, and QuadratureTreeSHAP contributions and interactions
  ([parity][testing]).
- **Objectives:** `reg:squaredlogerror`, `reg:absoluteerror`,
  `reg:quantileerror`, `reg:expectileerror`, `binary:logitraw`,
  `binary:hinge`, `survival:cox`, and `survival:aft` ([objectives]).
- **Metrics:** `rmsle`, `mape`, `mphe`, `pre@k`, `quantile`, `expectile`,
  `cox-nloglik`, `aft-nloglik`, and `interval-regression-accuracy`
  ([metrics]).
- **Training:** continued training (`Trainer::init_model`), the refresh
  updater (`process_type = update`), `num_parallel_tree` forests, model
  slicing and `iteration_range` predictions, `gradient_based` row
  sampling, feature-weighted column sampling, and label bounds
  ([boosters and training][boosters]).
- **Multi-output models:** label matrices (`one_output_per_tree`) and
  vector-leaf trees (`multi_output_tree`), with reduced split gradients for
  custom objectives ([multi-output]).
- **Objective and metric hooks** read the whole `MetaInfo` (label bounds,
  label matrices, groups): `Objective::gradient_info`, `validate_info`,
  `split_gradient`; `Metric::eval_info`, `validate_info`,
  `prediction_width`.
- **Formats:** a native binary container (magic `HBM\0`, version 3: a
  section table, zstd-compressed, with an XXH64 checksum and an optional
  `model.writer` section naming the release that wrote it; readers refuse
  undefined flag bits and trailing bytes, and name 0.1.x files as such),
  the compact bit-packed `HBTD` format (`model::compact`), XGBoost UBJSON
  import and export, and XGBoost JSON for categorical splits, forests, and
  multi-output and vector-leaf models ([formats]).
- **Opt-in extensions** (off by default, never change default training):
  conformal prediction intervals, distributional `dist:*` boosting, budget
  training, ordered target statistics, `linear_tree` leaves,
  quantized-gradient training, reuse penalties, symmetric trees, and
  `extra_trees`/`path_smooth` ([beyond XGBoost][beyond]).
- **Metal backend** (`metal` feature, macOS 10.15+): GPU prediction
  (`BoostedModel::to_gpu`) and `device = metal` histogram training, both
  identical to the CPU bit for bit ([GPU acceleration][metal]).
- **Performance:** batched split search, parallel split scans and row
  routing, faster histogram construction, and concurrent growth of an
  iteration's trees, with unchanged models ([performance]).

### Changed

- **Refusals** (errors instead of silently ignored settings):
  - `booster = gblinear` refuses row and column sampling,
    `gradient_based` sampling, `num_parallel_tree > 1`, monotone and
    interaction constraints, and training-matrix feature weights.
  - `process_type = update` refuses feature weights and non-default
    settings it does not read (sampling, symmetric growth, DART dropout,
    the opt-in tree options); XGBoost's tree-shape settings stay accepted.
  - `num_parallel_tree` must be in `1..=65536`.
  - `num_class >= 2` with a built-in non-multiclass objective is refused in
    training, at load, and on XGBoost export.
  - Budget training refuses `num_class >= 2`.
  - XGBoost import refuses invalid objective parameters, non-string booster
    or objective names, a single `base_score` for more than 65,536 outputs,
    and categorical segments larger than the category array.
  - gblinear models with a stored `best_iteration` are refused at load.
  - `OrderedTargetEncoder` refuses priors beyond ±`f32::MAX`.
- **Metal:** `device = metal` histograms, which could differ from the
  CPU's, are exact 64-bit integer sums for every node whose sums the CPU's
  `f64` adds compute exactly (`n * max <= 2^53` grains); other nodes run
  on the CPU. The backend now needs macOS 10.15 (Metal 2.2).
- **Serde:** deserializing a `BoostedModel`, `RegTree`, or `LinearLeaves`
  validates it and refuses inconsistent data; the native JSON format is
  unchanged.
- **Custom objectives:** the default intercept (`base_margins_info`) takes
  its Newton step on the full metadata (label bounds, label matrices).
- The erf port in `survival:aft` carries the Sun Microsystems fdlibm
  notice.

### Fixed

- **Training:**
  - `hist`, `approx`, and `exact` pick XGBoost's split when tiny gradients
    meet very large Hessians (e.g. weights near `1e38`).
  - Multi-output `approx` training with a constant-Hessian custom objective
    no longer depends on the thread count.
  - CPU histograms of nodes with 8,192 or more rows on sparse data (or row
    subsets of dense data past 2^18 rows) no longer depend on the thread
    count: the rows are summed in fixed blocks reduced in block order, at
    every thread count (single-threaded training on such data can differ
    in the last bits from before).
  - Gradient-based sampling handles gradients whose `f32` squares overflow
    (up to `f32::MAX`) and reports non-finite gradients as an error.
  - `reg:absoluteerror`/`reg:quantileerror` scales stay finite when
    zero-weight rows have overflowing residuals.
  - A structurally invalid init model is a `ModelFormat` error, not a
    panic.
- **Budget training:**
  - Splits whose gain or child statistics overflow `f32` are skipped; a
    root whose Hessian sum overflows is an error, and the returned model is
    validated (previously it could save but not load).
  - The small-root fallback (8 rows or fewer) considers categorical splits
    and missing-value directions.
  - With `extra_trees` or `path_smooth`, overflowing categorical candidates
    are skipped like numeric ones.
- **Metrics and objectives:**
  - Ranking metrics score 0 on empty input or empty groups instead of
    panicking; `LambdaMart::validate_info` errors on mismatched groups or
    weights.
  - Inconsistent `MetaInfo` lengths (including a label matrix with no
    labels) give NaN metrics and hook errors instead of allocation panics.
  - `nll`/`crps` ignore zero-weight rows whose score overflows.
  - `Dist::quantile` for Poisson and negative-binomial means beyond `2^53`
    returns instead of looping forever.
- **Models:**
  - Very large, highly compressible native saves (e.g. gblinear with ~67M
    zero weights) are written uncompressed and now load.
  - SHAP contributions and interactions stay finite when a repeated
    feature's path probability overflows `f32`.
  - A deserialized `RegTree` whose leaf-vector size overflows is refused
    instead of panicking.
- **Metal:** buffer writes are bounds-checked, inputs are validated before
  dispatch, gradient staging cannot race a running build, and failed
  command buffers fall back to the CPU (histograms) or return an error
  (prediction).

[testing]: https://github.com/brndnmtthws/hessboost/blob/v0.2.0/README.md#testing-and-parity
[objectives]: https://github.com/brndnmtthws/hessboost/blob/v0.2.0/README.md#objectives
[metrics]: https://github.com/brndnmtthws/hessboost/blob/v0.2.0/README.md#metrics
[boosters]: https://github.com/brndnmtthws/hessboost/blob/v0.2.0/README.md#boosters-and-training
[multi-output]: https://github.com/brndnmtthws/hessboost/blob/v0.2.0/README.md#multi-output-models
[formats]: https://github.com/brndnmtthws/hessboost/blob/v0.2.0/README.md#data-and-model-formats
[beyond]: https://github.com/brndnmtthws/hessboost/blob/v0.2.0/README.md#beyond-xgboost-opt-in
[metal]: https://github.com/brndnmtthws/hessboost/blob/v0.2.0/README.md#gpu-acceleration-macos-metal
[performance]: https://github.com/brndnmtthws/hessboost/blob/v0.2.0/README.md#performance

## [0.1.1] - 2026-09-23

First supported release.

## [0.1.0] - 2026-09-23 [YANKED]

[0.2.0]: https://github.com/brndnmtthws/hessboost/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/brndnmtthws/hessboost/releases/tag/v0.1.1
[0.1.0]: https://github.com/brndnmtthws/hessboost/tree/v0.1.0
