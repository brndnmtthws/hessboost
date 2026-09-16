# Changelog

All notable changes to `sequoia-boost` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Runtime-dispatched AArch64 NEON kernels for objective gradients, prediction
  transforms, metric reductions, and dense numeric histogram split evaluation.
  Scalar fallbacks handle other CPUs, short inputs, and values outside the
  approximation ranges.
- Runtime-dispatched x86-64 kernels: an AVX2+FMA dense split-gain scan whose
  division-free prefilter hands surviving candidates to the exact scalar test
  (split choices are bit-identical), AVX2 exponential, sigmoid, logistic
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

### Changed

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

### Removed

- `DMatrix::with_feature_names` and `DMatrix::feature_names`, which were
  unreferenced anywhere in the crate, examples, tests, or benchmarks.
- The free function `tree::split_gain`; split evaluation keeps using the
  inlined `calc_gain` arithmetic in the builders and SIMD kernels.

### Fixed

- Make the AArch64 NEON dense split scan bit-identical to the scalar scan. Its
  vector prefilter compared a bound that could overflow to infinity (where
  `inf > inf` reads false) or round subnormal, silently dropping candidates
  the scalar scan accepts. The prefilter now saturates its bound and routes
  subnormal products and intermediates to the exact check. Surviving
  candidates are also accepted with the scalar `calc_gain` arithmetic, which
  scores zero-Hessian children with a zero gain instead of rejecting them, so
  the chosen split, statistics, and loss match the scalar scan and the x86-64
  kernel bit for bit.

- Continue the vector split scan from the node-wide incumbent gain. Each
  feature's scan previously started from zero, so within the 1e-6 gain
  epsilon it could accept a weaker candidate and discard the stronger one
  the sequential scalar scan picks, moving the split to another feature on
  both NEON and AVX2. The scan now receives the incumbent, restoring
  bit-identical split choices.
- Reserve sparse bin chunks by their stored entry count instead of the dense
  row width, so CSR training no longer requests dense-sized allocations that
  can fail for matrices with few stored entries.
- Accumulate TreeSHAP contributions one tree at a time, matching the
  interaction path, so a later constant tree with very large leaf values can
  no longer round away an earlier tree's contribution.
- Gate the adversarial split-scan test to the architectures that provide the
  vector kernels, so the test target compiles on scalar-only architectures.
- Check the compact split encoding's feature bound when the prediction layout
  is built, so a feature index that would wrap the slot field fails loudly
  instead of silently addressing another feature's keys.

## [0.2.0] - 2026-08-16

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
- Reject XGBoost JSON export for DART, `gblinear`, and categorical models that
  cannot be represented correctly by the current exporter.
- Replace the unmaintained bincode native serializer with Postcard. Native
  binary blobs from 0.1.0 are not compatible with 0.2.0. JSON remains the
  portable native interchange format across these versions.
- Return `Result` from margin-limited and leaf prediction APIs so feature-count
  and base-margin mismatches cannot silently produce predictions.

### Changed

- Prepare version 0.2.0 citation, Zenodo, and Cargo release metadata. No DOI is
  claimed until the first Zenodo archive has been published.

## [0.1.0]

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

[Unreleased]: https://github.com/pgarrett-scripps/sequoia-boost/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/pgarrett-scripps/sequoia-boost/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/pgarrett-scripps/sequoia-boost/releases/tag/v0.1.0
