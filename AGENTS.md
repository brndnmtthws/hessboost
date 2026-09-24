# AGENTS.md

hessboost is a Rust reimplementation of XGBoost gradient boosting: one
library crate, with no C/C++ or FFI apart from the `zstd` crate (the
official libzstd, compressing native model files) and, on macOS with the
opt-in `metal` feature, the `objc2-metal` bindings to Apple's Metal
framework. User docs are `README.md`, the rustdoc (`src/lib.rs` and module
docs), `examples/`, and `docs/performance.md`. This file covers working on
the code.

## Toolchain

`mise.toml` pins Rust (1.98.1, with clippy and rustfmt), mr-boxington (`mbx`,
a build cache), and uv; run `mise install`. Edition 2024, MSRV 1.93
(`rust-version` in `Cargo.toml`). `Cargo.lock` is gitignored, so never pass
`--locked`. Building needs a C compiler for libzstd (`zstd-sys`); a
cross-target build needs one for that target.

## Commands

```sh
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
MISE_RUST_VERSION=1.93.0 mise exec -- cargo build --all-features   # MSRV
```

XGBoost parity (needs uv, CMake, and a C++ compiler: XGBoost 3.4.2 is
built from its source tarball on first use). Fixtures are generated into the
gitignored `fixtures/` and never committed:

```sh
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/gen_fixtures.py
cargo test --test parity --release -- --ignored --nocapture
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/check_exports.py
```

CI (`.github/workflows/ci.yml`) runs each of these with `mbx` in place of
`cargo` (same arguments; rustfmt runs as plain `cargo`) and
`RUSTFLAGS=-D warnings`. Tests run on x86_64 Linux, aarch64 Linux, and
aarch64 macOS (with `--all-features`, so the `metal` backend's tests run
there; the device-dependent ones skip on runners without a Metal device,
and an always-on guard test fails if the kernels fail to compile). Clippy
runs on x86_64 Linux and aarch64 macOS, so the aarch64 SIMD kernels and
the darwin-only `backend/metal.rs` are linted. Everything else runs on
x86_64 Linux. `all-checks-passed` gates merges to `main`.

When touching `simd/` or `cfg(target_arch)` code, still cross-check the
architecture CI does not build:

```sh
cargo clippy --all-targets --all-features --target x86_64-unknown-linux-gnu -- -D warnings   # from aarch64
cargo clippy --all-targets --all-features --target aarch64-unknown-linux-gnu -- -D warnings  # from x86_64
```

`scripts/README.md` documents the fixture case matrix, tiers, tolerances, and
the benchmark harnesses.

## Lints

`[lints.clippy]` in `Cargo.toml` enables `pedantic` and
`undocumented_unsafe_blocks`, and allows casts, `float_cmp`, short/similar
names, `too_many_lines`, `missing_errors_doc`/`missing_panics_doc`,
`must_use_candidate`, and `inline_always`. Fix new warnings. A new local
`#[allow(...)]` needs `reason = "..."` and is acceptable only where the lint
is wrong for that site. Never `#[allow(clippy::too_many_arguments)]`: group
the parameters into a well-named struct of arguments that belong together
(e.g. per-call context vs. per-node state; references and `Copy` values
passed by reference in hot loops). `clippy.toml` lists the doc identifiers
(`XGBoost`, `TreeSHAP`, ...) exempt from `doc_markdown`; add new proper
nouns there.

## Layout (`src/`)

| Path | Contents |
|---|---|
| `lib.rs` | crate docs (module map, "What's here"), module list, `prelude`, hidden `internals` |
| `error.rs` | `HessboostError`, `Result` |
| `rng.rs` | `Rng` (xoshiro256++) and the SplitMix64 mixing for counter-based streams |
| `data/` | `DMatrix` (dense/CSR, labels or a label matrix, label bounds, weights, groups, base margins, feature types and weights), `meta` (`MetaInfo`, the view objectives and metrics read), `loaders` (libsvm/CSV), `sketch`/`quantile` (quantile sketch, `HistCuts`), `ghist` (`GHistIndex` binning), `target_stats` (public; opt-in ordered target statistics) |
| `config/` | `params.rs`: `TrainingParams`, its builder and `validate`, the parameter enums, `ObjectiveParams`; names mirror XGBoost |
| `objective/` | Losses by XGBoost name: `regression`, `classification`, `multiclass`, `count`, `ranking`, `quantile` (quantile/expectile alpha lists), `absolute` (smoothed MAE), `survival` (Cox/AFT; `erf` ported from glibc), `multi_target` (label-matrix wrapper), `custom`; `distributional/` (public; opt-in `dist:*` families, `Dist`, `special` functions) |
| `metric/` | Eval metrics by XGBoost name: `mod.rs` (factory, defaults, rmse, mae, logloss, error, auc/aucpr, multiclass, count, ndcg/map, custom), `elementwise` (rmsle, mape, mphe), `ranking` (`pre@k`), `quantile`, `survival` (cox/aft-nloglik, interval accuracy), `distributional` (`nll`, `crps`) |
| `tree/` | `regtree` (`RegTree`, scalar or vector leaves), `gain`, `constraints` (monotone/interaction), `sampler` (colsample bytree/bylevel/bynode, optionally feature-weighted), `builder/` (see below), `hist/` (histogram accumulation; `hist/quantized` for quantized gradients), `compact` (prediction-optimized layout, scalar and vector leaves), `oblivious` (bit-pattern prediction for symmetric trees), `linear` (opt-in `linear_tree` leaves, `LinearLeaves`), `reuse` (opt-in Trees-on-a-Diet reuse penalties); only `RegTree`, `Node`, `LinearLeaves` are public |
| `tree/builder/` | `mod.rs` (split enumeration shared by all builders, incl. `sweep_categorical`), `exact`, `hist` (also `approx`), `multi` (vector-leaf trees), and the opt-in `oblivious` (symmetric growth), `lightgbm` (`extra_trees`/`path_smooth` search), `budget` (generalization-gated grower) |
| `training/` | `train` (`train`, `Trainer`, `TrainResult`; gbtree, DART, gblinear; `approx` = hist builder with per-round weighted cuts; `num_parallel_tree` forests), `gblinear` (coordinate descent), `multi_output` (vector-leaf rounds, reduced split gradients), `sampling` (gradient-based row sampling), `continuation` (continued training / `process_type=update` checks), `refresh` (refresh updater), `cv`, `budget` (public; opt-in PerpetualBooster-style training) |
| `model/` | `mod.rs` (`BoostedModel`: iteration layout, slicing, `iteration_range` prediction, save/load entry points; user docs for XGBoost interchange), `native` (native binary container), `sections` (section table shared by the native and compact formats), `shap` (QuadratureTreeSHAP), `compact` (public; `CompactModel`, `HBTD` format), `xgboost` (XGBoost JSON/UBJSON schema mapping), `ubjson` (UBJSON codec over `serde_json::Value`) |
| `conformal.rs` | split-conformal / CQR intervals (`SplitConformal`, `ConformalizedQuantile`) |
| `backend/` | opt-in compute backends: `metal/` (macOS, `metal` feature; `MetalHistBackend` for GPU histograms, `GpuModel` for GPU prediction, runtime-compiled MSL kernels, exact double-float accumulation) |
| `simd/` | private runtime-dispatched kernels: `scalar`, `aarch64` (NEON), `x86_64` (AVX2/FMA, SSE2), `tests` |
| `test_support.rs` | unit-test helpers (`cfg(test)`) |

### Tests, examples, benches

- `tests/`: `parity.rs` (ignored by default; needs fixtures), `properties.rs`
  (proptest), `shap_accumulation.rs`, `sampling.rs` (row/column sampling),
  `target_stats.rs`, `continuation.rs` (continued training, refresh, forests,
  slicing, iteration ranges), `quantized.rs` (thread-count determinism,
  quality band, leaf renewal), `tree_options.rs` (`extra_trees`,
  `path_smooth`, `linear_tree`), `budget.rs`, `multi_output.rs` (vector-leaf
  trees), `distributional.rs` (`dist:*`: calibration, NLL, serialization,
  CQR), `native_format.rs` (native round trips, refused versions and corrupt
  payloads, saved models of every release), and `metal.rs` (macOS, `metal`
  feature; GPU==CPU bit-exactness, determinism, refusals, `to_gpu`
  predictions; device-dependent tests skip without a Metal device).
- `tests/common/` and `examples/common/` hold shared helpers.
- `tests/data/saved/<version>/` holds the models each release saved (`.bin`,
  `.json`, compact `.hbtd` where supported, and `.margins`);
  `tests/data/xgboost-3.4.2-categorical.{json,ubj}` are XGBoost saves imported
  by the `model/xgboost.rs` unit tests.
- `benches/training.rs` is the Criterion suite; `docs/performance.md` records
  its results and the XGBoost comparison.

## Invariants

- **Errors:** public fallible APIs return `hessboost::error::Result<T>`
  (`HessboostError`). No `unwrap`/`expect` or other panics on
  user-controlled input (including NaN) in library code.
- **Determinism:** identical params, data, and seed give identical
  predictions (`tests/properties.rs`), independent of the thread count:
  every grow policy must grow the same tree serially and in parallel, and
  parallel reductions keep a fixed order. Sequential sampling (rows,
  columns, DART, folds, target-stat permutations) draws from `rng::Rng`
  (the same stream on every platform). Keyed draws (`extra_trees` node
  seeds, the `dist:*` random split direction, quantized stochastic rounding,
  per-block row-sampling seeds) use counter-based SplitMix64 streams keyed
  by seed and index (`rng.rs`), so they do not depend on scheduling.
  Quantized histograms sum integers exactly. `rand` is a dev-dependency
  only. A `device = metal` training run reproduces the single-threaded CPU
  model bit for bit: the GPU kernels accumulate exact double-float
  (two-sum) partials in a fixed chunk/slice/merge order, with no atomics,
  and every constant is machine-independent (`src/backend/metal.rs`).
- **Unsafe:** confined to `simd/`, the hot loops in `tree/compact.rs`,
  `tree/hist/`, and `tree/builder/hist.rs`, and the Metal FFI in
  `backend/metal.rs`. Every block needs a `// SAFETY:` comment;
  `unsafe_op_in_unsafe_fn` is forbidden (`lib.rs`).
- **SIMD:** kernels cover objective gradients, exp/sigmoid/softmax
  transforms, metric sums, and cut search (`count_le`). Dispatch checks CPU
  features at runtime and falls back to `simd/scalar.rs` (also below
  minimum lengths and outside approximation ranges). Cut search must match
  scalar exactly; transcendental kernels stay within the tolerances in
  `simd/tests.rs`. Split search, histogram accumulation, and prediction are
  scalar and follow XGBoost's `f32` arithmetic and operation order; their
  optimized paths must stay bit-identical to the plain ones.
- **Parity:** the target is XGBoost 3.4.2 (whose release notes call it
  numerically identical to 3.4.1). Fixture tiers: `exact` (pointwise
  train/import/export), `quality` (RNG-driven cases, i.e. subsampling,
  column sampling, and DART, plus `rank:*` objectives; training within a
  quality band, import/export still pointwise), and `trainonly` (gblinear).
  Two sampling structures deliberately differ from XGBoost and are covered
  only by the band: under `hist`, a multi-output model's outputs share each
  parallel tree's uniform row sample (XGBoost draws one per output group),
  and under `approx` with uniform sampling the per-round cuts weight
  unsampled rows by their Hessian (XGBoost gives them zero). Categorical
  splits follow XGBoost's `HistEvaluator` (`EnumerateOneHot` below
  `max_cat_to_onehot = 4` categories, otherwise `EnumeratePart` scanned in
  both directions up to `max_cat_threshold = 64`) in
  `tree/builder/mod.rs::sweep_categorical`, shared by the histogram and
  exact builders. Beyond-XGBoost features are opt-in, leave default training
  unchanged, and stay out of the parity fixtures.
- **Formats:** from 0.2.0 on, files written by a release keep loading in
  every later one (0.1.x native binaries are refused).
  - Native binary (`model/native.rs`): a zstd frame holding `SQB\0`, a
    container version byte (`CONTAINER_VERSION`, currently 3), a section
    table (`model/sections.rs`), and an XXH64 checksum of the preceding
    bytes. Sections hold model scalars (`model.*`), objective parameters
    (`objective.*`), trees column-wise (`tree.*`, `node.*`,
    `leaf_linear.*`), and gblinear weights (`gblinear.*`). A new stored
    field is a new section: flag it `REQUIRED` when readers that do not know
    it must refuse the file rather than skip it, and read its absence with a
    default that reproduces older files (for objective parameters, the
    objective's defaults). Changing an existing section's meaning or the
    container layout bumps `CONTAINER_VERSION`; the reader accepts only the
    current version, so such a change must add a reader for the previous
    one.
  - Native JSON (`BoostedModel`'s serde fields by name): a new field needs
    `#[serde(default)]` reproducing older files.
  - Compact (`HBTD`, documented in `model/compact.rs`): metadata is a
    section table like the native one; a change to the bit stream bumps its
    version byte (currently 1).
  - Before each release, `cargo test --test native_format -- --ignored
    save_models_of_this_version` writes `tests/data/saved/<Cargo
    version>/` (it refuses to overwrite). Commit it; never regenerate an
    earlier version's directory.
- **Tree layout:** as in XGBoost, iteration `i` owns trees
  `i * trees_per_iteration ..`. Scalar-leaf models have `trees_per_iteration
  = n_outputs × num_parallel_tree`, grouped by output, and tree `t` feeds
  output `(t / num_parallel_tree) % n_outputs` (`BoostedModel::tree_output`).
  Vector-leaf models have `num_parallel_tree` trees per iteration, each
  feeding every output (`tree_info` 0). Iteration counts, `best_iteration`,
  slicing, and `iteration_range` count iterations, never raw trees.
- **Prediction layout:** single-output and `multi:softmax` give `n_rows`
  values; `multi:softprob` gives `n_rows * num_class`, and other
  multi-output models (label matrix, quantile/expectile alpha lists,
  `dist:*` parameters) `n_rows * n_outputs`, all row-major `[row][output]`.
  Multi-target `predict_class` thresholds every target (`[row][target]`).
  SHAP contributions are `[row][n_features + 1]` (bias last), interactions
  `[row][(n_features + 1)^2]`, with an extra output axis for multi-output
  models. `BoostedModel::n_targets` counts label columns; XGBoost's
  `num_target` counts outputs (label columns or alphas), and is 1 for
  multiclass, which uses `num_class`.
- **Objective/metric hooks:** training and evaluation read data only through
  `MetaInfo` hooks: `Objective::gradient_info`, `base_margins_info`,
  `eval_transform`, `validate_info`, `requires_labels`;
  `Metric::eval_info`, `validate_info`, `prediction_width`,
  `supports_label_matrix`. Specialized paths add `Objective::split_gradient`
  (vector-leaf structure search), `pointwise_loss` (budget mode), and
  `probs_to_margins`/`margins_to_probs` (XGBoost `base_score` import/export;
  `margins_to_probs` defaults to `pred_transform`, and `binary:hinge`
  overrides it because its threshold is not its link).
  - Label-domain checks live in each objective's `validate_info`.
  - `create_objective(params, n_targets)` wraps the objectives in
    `MULTI_TARGET_OBJECTIVES` (`objective/mod.rs`) in
    `multi_target::MultiTarget` for a label matrix (row weights broadcast per
    cell, intercepts per column); `reg:absoluteerror` handles label matrices
    itself; other built-in objectives refuse them.
  - The default `Metric::eval_info` reduces a label matrix elementwise
    (every cell weighted by its row weight); non-elementwise metrics
    override it or return `supports_label_matrix() == false`, which training
    refuses. Before training, every eval set is checked with
    `Metric::validate_info` (labels by default; interval metrics accept
    label bounds) and `prediction_width` (must equal the model's outputs;
    `None`, as for custom metrics, needs a whole number of outputs per label
    column). `Metric::eval` returns NaN for inconsistent lengths.
- **Refusals:** unsupported parameters and combinations fail with an error,
  never silently ignored. Checks live in `TrainingParams::validate`
  (`device = metal` needs the `metal` feature on macOS, `tree_method =
  hist`/`auto`, and a tree booster, and refuses `use_quantized_grad` and
  `process_type = update`), `training/multi_output.rs::validate`
  (`multi_output_tree` needs `hist` or `auto`, and refuses vector-leaf
  trees on a GPU device), `training/continuation.rs` (`process_type=update`),
  and `training/budget.rs`. Budget mode compares the serialized params
  against the defaults plus its allow-list, so a non-default value of any
  field it does not read, including one added later, is refused
  automatically.
- **Parity-fixed options:** options that XGBoost has but hessboost supports
  at one setting (README, "Not implemented") are not `TrainingParams`
  fields. `tests/parity.rs` (`expect_fixed`) fails a fixture that sets
  `updater`, `feature_selector`, or `lambdarank_pair_method` to anything
  else.

## Public API

The crate root exports only modules. Rules:

- `hessboost::prelude` holds only the train-and-predict workflow:
  `TrainingParams`, `DMatrix`, `train`, `Trainer`, `BoostedModel`,
  `HessboostError`, `Result`. Parameter enums, objectives, metrics, and
  opt-in features are imported from their modules.
- Each item has exactly one public path (the prelude re-exports are the only
  second path): no flat re-exports of a public submodule's items, no
  aliases.
- Opt-in subsystems with substantial docs get their own public submodule:
  `data::target_stats`, `training::budget`, `model::compact`,
  `objective::distributional`, and the top-level `conformal`.
- Implementation modules are crate-private (`pub(crate)` or private); the
  benches and parity tests reach internals (`HistCuts`, `GHistIndex`,
  `HistTreeBuilder`, `CpuBackend`, `HistogramBackend`, `zeroed`,
  `ColumnSampler`) through the `#[doc(hidden)] pub mod internals` in
  `lib.rs`, which is not public API.

Paths: `config` (`TrainingParams`, `TrainingParamsBuilder`, the parameter
enums, `ObjectiveParams`, `MAX_SYMMETRIC_DEPTH`); `data` (`DMatrix`,
`FeatureType`, `MetaInfo`, `GroupInfo`, `CsvOptions`,
`{load,read}_{csv,libsvm}`); `training` (`train`, `Trainer`, `TrainResult`,
`RoundEval`, `cv`, `CvResult`); `model` (`BoostedModel`, `ImportanceType`);
`objective` (`Objective`, `GradPair`, `SplitGradient`, `PointwiseLoss`,
`CustomObjective`, `create_objective`, built-ins named without an
`Objective` suffix: `SquaredError`, `Logistic`, `Softmax`, `LambdaMart`,
`Aft`, ...); `metric` (`Metric`, `CustomMetric`, the built-in metrics,
`create_metric(name, &params)`); `tree` (`RegTree`, `Node`,
`LinearLeaves`); `error`.

- Training (`rounds: usize`):
  - `train(&params, &dtrain, rounds)` → `BoostedModel`.
  - `Trainer::new(&params, &dtrain, rounds)` with optional
    `.eval(&data, "name")` (repeatable), `.early_stopping_rounds(k)`,
    `.objective(&dyn Objective)`, `.custom_metric(Box<dyn Metric>)`,
    `.init_model(&model)` (continued training; with
    `process_type(ProcessType::Update)` the refresh updater), then
    `.train()` → `TrainResult { model, history }`.
  - `cv(&params, &data, rounds, nfold, seed: u64)` → `Vec<CvResult>`.
  - `training::budget::train_with_budget(&params, &dtrain,
    &BudgetConfig::new(budget))` → `BudgetResult { model, eta, stop }` (no
    round count).
- `DMatrix::from_dense` / `from_dense_with_missing` / `from_csr`, then
  `.with_labels` / `.with_label_matrix(y, k)` / `.with_label_bounds` /
  `.with_weights` / `.with_base_margin` / `.with_group_sizes` /
  `.with_group_weights` / `.with_feature_types` / `.with_feature_weights`;
  `.info()` gives the `MetaInfo`.
- `BoostedModel`:
  - Prediction: `predict`, `predict_margin`, `predict_class`, `predict_leaf`,
    `predict_contribs`, `predict_interactions`, `predict_distribution`
    (`dist:*`, → `Vec<Dist>`), and the `iteration_range` variants
    `predict_range`, `predict_margin_range`, `predict_leaf_range`,
    `predict_contribs_range`, `predict_interactions_range`,
    `predict_distribution_range` taking `impl RangeBounds<usize>` of
    iterations (`..`, `..n`, `2..5`); leaf, contribution, and interaction
    ranges must start at 0.
  - `slice(iterations: impl RangeBounds<usize>, step)`, `num_boost_rounds`,
    `num_parallel_tree`, `feature_importance(ImportanceType)`.
  - Formats: native binary `to_bytes`/`from_bytes`,
    `save_binary`/`load_binary`; native JSON `to_json`/`from_json`,
    `save_json`/`load_json`; XGBoost `{to,from,save,load}_xgboost_json` and
    `{to,from,save,load}_xgboost_ubjson`.
  - Compact: `to_compact()` / `to_compact_bytes()` → `model::compact::CompactModel`
    (`from_bytes`/`load`, `to_bytes`/`save`, `predict_margin` bit-identical
    to the source, `predict`); `size_report()` → `ModelSizeReport`.
- `conformal`: `SplitConformal::calibrate(&model, &dcal, alpha)`;
  `ConformalizedQuantile::calibrate(&lower, &upper, &dcal, alpha)` /
  `calibrate_outputs(&model, lower_output, upper_output, &dcal, alpha)` /
  `calibrate_distribution(&model, &dcal, alpha)`; then
  `.predict_interval(&data)` → `Vec<(f32, f32)>`.

Usage requirements that are easy to miss: multiclass objectives need
`.num_class(k)`; ranking objectives need `.with_group_sizes`;
`survival:aft` needs `.with_label_bounds` (labels optional);
`survival:cox` reads non-positive labels as right-censored times; custom
objectives supply reduced split gradients via `Objective::split_gradient`
(vector-leaf only, not with monotone constraints). Opt-in option
requirements (`grow_policy = symmetric`, `extra_trees`, `path_smooth`,
`linear_tree`, `use_quantized_grad`, reuse penalties) are enforced in
`TrainingParams::validate` and documented on each `TrainingParams` field.
Linear-leaf trees predict through `tree::linear` instead of the compact
forest; XGBoost export, SHAP, and the compact format refuse them.

## When changing behavior

Update, in the same change: the rustdoc of the touched items, the
README feature lists (and "Not implemented"), the `lib.rs` "What's here"
list, this file's layout and invariants, and the examples that exercise it.
New options need a `TrainingParams` field, builder setter, and validation;
beyond-XGBoost options must default to off.

## Releases

Bump `version` in `Cargo.toml`, write `tests/data/saved/<version>/` (see
Formats), merge, then push a `v<version>` tag: `publish.yml` checks the tag
against the crate version, runs the tests on all three platforms, publishes
to crates.io, and creates the GitHub release (notes grouped by
`.github/release.yml`) with a discussion.
