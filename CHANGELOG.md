# Changelog

All notable changes to `hessboost` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/). hessboost starts at 0.1.0. The
sequoia-boost releases it was forked from are kept at the end for reference.

## [Unreleased]

Changes since sequoia-boost 0.2.0.

### Added

- Runtime-dispatched AArch64 NEON kernels for objective gradients, prediction
  transforms, and metric reductions. Scalar fallbacks handle other CPUs, short
  inputs, and values outside the approximation ranges.
- Runtime-dispatched x86-64 kernels: AVX2 exponential, sigmoid, logistic
  gradient, and short softmax (2 and 4 classes) kernels, and an SSE2 16-cut
  search for quantile binning. On x86-64 the histogram bin sums keep the scalar
  summation order.
- Kernel and tree-building benchmarks with reproducible comparisons and
  [documented results](docs/performance.md).
- Benchmark charts in the [performance guide](docs/performance.md): XGBoost
  speedup and fit time by thread count, plus scalar-baseline time cuts for
  full training, tree builds, and pointwise gradients, transforms, and
  metrics. The SVGs are rendered by `docs/benchmarks/charts.gp` from
  `docs/benchmarks/xgboost.dat` and `docs/benchmarks/optimization.dat`.
- XGBoost 3.4.1 parity harness: `scripts/gen_fixtures.py` writes 37 pointwise,
  banded, and train-only cases (every supported objective, including
  `reg:logistic` and the `reg:linear` alias) plus 21 histogram/approximate
  cut oracles; `tests/parity.rs` checks train, import (predictions, margins,
  SHAP) and cut parity, and `scripts/check_exports.py` reloads supported
  exported models in XGBoost. CI runs all three via `uv`.
- `TrainingParams::tweedie_variance_power` (default 1.5, in `[1, 2)`) and
  `TrainingParams::huber_slope` (default 1.0), wired into `reg:tweedie` and
  `reg:pseudohubererror`; the pseudo-Huber gradient now honors the slope like
  XGBoost's `PseudoHuberRegression`.
- `Objective::const_hess` (true only for `reg:squarederror`).
- XGBoost JSON interop with 3.4.1's DART layout: `model.weight_drop` is read
  as the per-tree DART weights and written back for models with non-unit tree
  weights (which previously failed to export).
- `reg:logistic` as its own objective: the `binary:logistic` loss, reported
  under XGBoost's name and default metric (`rmse`), and saved as
  `reg:logistic`. `reg:linear` models are saved as `reg:squarederror`, as
  XGBoost does.
- `BoostedModel::objective_params()` and `ObjectiveParams`: the objective
  hyper-parameters (`scale_pos_weight`, effective `max_delta_step`,
  `tweedie_variance_power`, `huber_slope`, `lambdarank_num_pair_per_sample`)
  a model was trained with. They are stored in native models, used to rebuild
  the objective for prediction, and written to / read from the XGBoost JSON
  objective block instead of hard-coded defaults.
- `TrainingParams::effective_max_delta_step()`: the `max_delta_step` in
  effect, resolving an unset value to XGBoost's objective-dependent default.

### Changed

- Forked from [sequoia-boost](https://github.com/pgarrett-scripps/sequoia-boost)
  and renamed to `hessboost`, restarting at version 0.1.0. The crate, library
  path (`hessboost::prelude`), and error type (`HessboostError`) are renamed.
  The native binary format is unchanged, so models saved by sequoia-boost 0.2.0
  still load.
- The crate now lives at the repository root instead of a one-member workspace
  under `crates/`.
- Removed the upstream citation and Zenodo metadata (`CITATION.cff`,
  `.zenodo.json`) and the CI job that validated them.
- Removed the unused `num-traits` and `rand_pcg` dependencies and the empty
  `gpu` feature.
- Skip child histograms and split searches for depthwise leaves at `max_depth`,
  preserving leaf statistics, monotone bounds, and column-sampling order.
- Parallelize quantile cuts, bin assignment, and depthwise node construction.
  Size histogram tasks to the available rows and reuse final training-row
  partitions in smaller thread pools.
- Rewrite prediction around a lazily built branch-free tree layout: breadth-first
  node arena, single-compare split encoding with mirrored default-right
  children, sixteen-row (or sixteen-tree) lockstep traversal, block-parallel
  batches, and per-block densification of sparse rows. `predict`,
  `predict_margin`, `predict_leaf`, and single-row prediction are 10× to 20× faster
  with identical results.
- Speed up TreeSHAP contributions and interactions 8× with an arena-backed
  decision path, precomputed cover fractions, hoisted divisions, a shared
  unwound sum for off-path elements, and row-parallel evaluation.
- Compare prediction splits on monotone `u32` keys (NaN maps to the missing
  direction, mirrored nodes read a precomputed negated key) loaded together
  with the node's feature slot in one 64-bit load. Batch prediction is
  25-35% faster with identical results.
- Build the two children's split evaluations concurrently on large nodes and
  reduce partial histograms in parallel by bin range, keeping the per-bin
  addition order. Root histograms over contiguous row ranges are built
  column-wise with one writer per feature. Sibling partitions are written into
  spare capacity instead of zero-filled buffers.
- Compute built-in row-independent gradients (squared error, logistic, softmax)
  over fixed row chunks in parallel, bit-identical to the whole-batch result.
- Evaluate TreeSHAP hot path elements in monomorphized lanes, skip the unit
  `one_fraction` division, and reuse the parent's path region for the cold
  child, for a further ~5% reduction with per-element results unchanged.
- Consolidate shared helpers across the crate (single source for SIMD
  dispatch, quantile cut building, tree split acceptance and leaf
  finalization, gradient-statistic accumulation, linear-margin traversal,
  TreeSHAP scaffolding, metric ideal-DCG and tie-run helpers, dataset
  parsing, and the example/benchmark data generators). Behavior is unchanged;
  verified end-to-end by an all-subsystem equivalence probe hashing every
  output against the pre-refactor revision.
- Per-output intercepts: `BoostedModel::base_score()` remains the scalar first
  output, while `base_scores()` returns one margin-space value per output/class.
  `Objective::base_margin` became `base_margins(labels, weights, group) ->
  Vec<f32>`, reproducing XGBoost 3.4.1's `InitEstimation` bit for bit: the
  (weighted) label mean for squared error, logistic (`scale_pos_weight == 1`),
  Poisson, Gamma and Tweedie; centered class log-frequencies for softmax; and
  a Newton step through the link for everything else (pseudo-Huber, ranking,
  reweighted logistic). A user `base_score` is broadcast to every output
  through the link, as XGBoost does. Native JSON/binary models store the
  intercept as a vector.
- XGBoost JSON export now targets 3.4.1: `version [3,4,1]`, vector
  `base_score` (`"[v0,v1,...]"`), `boost_from_average`, and each objective's
  parameter block (`reg_loss_param`, `poisson_regression_param`,
  `tweedie_regression_param`, `pseudo_huber_param`, `softmax_multiclass_param`,
  `lambdarank_param`). Import requires the 3.x vector `base_score` (one entry
  applies to every output) and rejects the pre-3.x scalar form.
- `TrainingParams::max_delta_step` is `Option<f64>`. `None` (the default)
  keeps XGBoost's behavior of injecting `0.7` for `count:poisson`; an explicit
  `0` now disables the constraint for both the Poisson Hessian and the tree
  regularizer instead of being treated as unset.
- `Objective::default_metric` returns an owned `String` so it can carry
  configuration: LambdaRank reports `ndcg@k` (`rank:pairwise`, `rank:ndcg`)
  or `map@k` (`rank:map`) with `k = lambdarank_num_pair_per_sample`, and
  `reg:tweedie` reports `tweedie-nloglik@<tweedie_variance_power>`, matching
  XGBoost's `DefaultEvalMetric`; early stopping therefore tracks the same
  score as XGBoost.
- Native models record `n_outputs` (raw outputs per instance) separately from
  `num_class`, so a custom objective with several outputs keeps its intercept
  and tree layout through prediction and serialization. Native JSON requires
  the field.
- XGBoost JSON import honors `tree_info` (with `iteration_indptr` /
  `num_parallel_tree`): grouped multiclass forests are reordered into the
  round-robin layout together with their DART `weight_drop` entries, and
  layouts that cannot be mapped losslessly are rejected. Export refuses
  objectives XGBoost cannot load (custom objectives) instead of writing a
  bogus `reg_loss_param` block.
- Logistic labels are validated like XGBoost's `LogisticRegression::CheckLabel`
  (`binary:logistic` and `reg:logistic` accept probabilities in `[0, 1]`).
- `TrainingParams::validate` rejects `lambdarank_num_pair_per_sample == 0`
  (XGBoost's lower bound is 1; zero would silently train zero gradients),
  `tweedie_variance_power` values that round to `2.0` in `f32`, and
  `huber_slope` values whose `f32` square overflows or vanishes.

### Removed

- `DMatrix::with_feature_names` and `DMatrix::feature_names`, which were
  unreferenced anywhere in the crate, examples, tests, or benchmarks.
- The free function `tree::split_gain` and the NEON/AVX2 dense split-gain
  prefilter kernels; split evaluation is the scalar XGBoost `f32` gain
  arithmetic in the builders.

### Fixed

- Reserve sparse bin chunks by their stored entry count instead of the dense
  row width, so CSR training no longer requests dense-sized allocations that
  can fail for matrices with few stored entries.
- Accumulate TreeSHAP contributions one tree at a time, matching the
  interaction path, so a later constant tree with very large leaf values can
  no longer round away an earlier tree's contribution.
- Check the compact split encoding's feature bound when the prediction layout
  is built, so a feature index that would wrap the slot field fails loudly
  instead of silently addressing another feature's keys.
- Evaluate XGBoost's final backward histogram candidate: missing values alone
  on the left, every present bin on the right (`default_left`, threshold
  `f32::MIN` standing in for XGBoost's `-inf`). It is distinct under monotone
  constraints, where the mirrored forward endpoint can violate the required
  direction. The missing-value test now uses the forward pass's completed
  accumulator exactly as XGBoost's `SplitContainsMissingValues` does.
- Round split gains where XGBoost does: the parent gain is evaluated at the
  `f32` weight and each child's gain is rounded to `f32` before the two are
  added, so near-tied candidates resolve the same way as upstream.
- Keep exact-method split thresholds finite for feature values near
  `±f32::MAX`: the XGBoost endpoint (`last ± (|last| + eps)`) and midpoint
  arithmetic are used unchanged, and only an overflowing result falls back to
  the finite value that induces the same partition (or the endpoint is
  skipped when none exists), so prediction no longer hits a non-finite
  `split_cond`.
- Seed the SIMD softmax gradient shift with `f32::MIN_POSITIVE` like XGBoost's
  `SoftmaxMultiClassObj` and the scalar path, so rows whose margins are all
  far below zero produce the same result on every architecture and batch
  size instead of NaN scalarly and finite gradients through NEON/AVX2.
- LambdaRank: rank documents with a stable numeric sort in which `+0.0` and
  `-0.0` tie (XGBoost `ArgSort` with `std::greater`), and apply the
  normalization, query weight and weight normalization as three separate
  `f32` multiplications in XGBoost's order instead of one pre-combined scale.

## [sequoia-boost 0.2.0] - 2026-08-16

### Fixed

- Correct DART tree normalization to use the learning rate in the normalization
  denominator.
- Enforce interaction constraints in exact trees and isolate features omitted
  from configured interaction groups.
- Make `multi:softmax` return one class label per row while preserving the
  probability matrix returned by `multi:softprob`.
- Include DART weights, early-stopping limits, dataset base margins, and linear
  boosters in SHAP contributions and interaction values.
- Reject malformed datasets, invalid objective labels, incompatible evaluation
  sets, invalid early-stopping settings, and prediction feature mismatches with
  structured errors instead of panics or silently ignored inputs.
- Honor per-row base margins in `gblinear` and honor `nthread` with a scoped
  Rayon pool.
- Interpret explicit count, Gamma, and Tweedie base scores in reported-value
  space and pass configured Poisson `max_delta_step` into its Hessian.
- Add query-level ranking weights and use them when averaging NDCG and MAP,
  matching XGBoost's ranking-weight semantics.
- Support XGBoost 3.4.1 categorical-tree JSON import/export by translating its
  right-set representation to sequoia's left-set representation; gblinear
  model JSON remains unsupported.
- Replace the unmaintained bincode native serializer with Postcard. Native
  binary blobs from 0.1.0 are not compatible with 0.2.0. JSON remains the
  portable native interchange format across these versions.
- Return `Result` from margin-limited and leaf prediction APIs so feature-count
  and base-margin mismatches cannot silently produce predictions.

### Changed

- Prepare version 0.2.0 citation, Zenodo, and Cargo release metadata. No DOI is
  claimed until the first Zenodo archive has been published.

## [sequoia-boost 0.1.0]

Initial release: a faithful, pure-Rust reimplementation of XGBoost gradient
boosting.

### Boosters & tree construction
- Boosters: `gbtree`, `dart` (tree dropout), `gblinear` (coordinate descent).
- Tree methods: `exact`, `hist` (histogram + subtraction trick), `approx`
  (hessian-weighted per-round binning), plus `depthwise` and `lossguide` growth.
- Sparsity-aware missing-value handling, row and column subsampling
  (`bytree` / `bylevel` / `bynode`), and multi-core histogram construction (`rayon`).

### Objectives & metrics
- Objectives: `reg:squarederror`, `reg:pseudohubererror`, `reg:gamma`,
  `reg:tweedie`, `count:poisson`, `binary:logistic`, `multi:softmax`,
  `multi:softprob`, learning-to-rank (`rank:pairwise` / `rank:ndcg` / `rank:map`,
  LambdaMART), plus a custom-objective hook.
- Metrics: `rmse`, `mae`, `logloss`, `error`, `auc`, `aucpr`, `mlogloss`,
  `merror`, `ndcg`, `map` (with `@k`), `poisson/gamma/tweedie-nloglik`, plus a
  custom-metric hook.

### Modeling
- Monotone constraints and interaction constraints (both `hist` and `exact`).
- Native categorical splits (both `hist` and `exact`).
- Per-instance `base_margin` (warm-start / stacking), early stopping, feature
  importance (weight / gain / cover / totals).
- TreeSHAP feature contributions (`predict_contribs`) and interaction values
  (`predict_interactions`).

### I/O & tooling
- Loaders: libsvm, CSV, dense/CSR in-memory.
- Model I/O: native binary + JSON, and XGBoost-format JSON import/export.
- K-fold cross-validation.

### Quality
- Unit, property (`proptest`), and doc tests. XGBoost model-quality parity is
  verified in CI against real XGBoost.

[Unreleased]: https://github.com/brndnmtthws/hessboost/commits/main
[sequoia-boost 0.2.0]: https://github.com/pgarrett-scripps/sequoia-boost/compare/v0.1.0...v0.2.0
[sequoia-boost 0.1.0]: https://github.com/pgarrett-scripps/sequoia-boost/releases/tag/v0.1.0
