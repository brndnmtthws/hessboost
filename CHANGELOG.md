# Changelog

All notable changes to `sequoia-boost` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
  (hessian-weighted per-round binning); `depthwise` and `lossguide` growth.
- Sparsity-aware missing-value handling; row and column subsampling
  (`bytree` / `bylevel` / `bynode`); multi-core histogram construction (`rayon`).

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
- Unit, property (`proptest`), and doc tests; XGBoost model-quality parity is
  verified in CI against real XGBoost.

[Unreleased]: https://github.com/pgarrett-scripps/sequoia-boost/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pgarrett-scripps/sequoia-boost/releases/tag/v0.1.0
