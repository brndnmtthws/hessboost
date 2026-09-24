# AGENTS.md

hessboost is a pure-Rust reimplementation of XGBoost gradient boosting: one
library crate, no C/C++, no FFI. User docs live in `README.md`, the rustdoc
(`src/lib.rs`), and `examples/`. This file covers working on the code.

## Toolchain

`mise.toml` pins Rust, mbx, and uv; run `mise install`. Edition 2024, MSRV
1.93 (`rust-version` in `Cargo.toml`). CI runs cargo through `mbx` (a build
cache); locally, plain `cargo` works the same.

## Commands

```sh
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo clippy --all-targets --all-features --target x86_64-unknown-linux-gnu -- -D warnings  # from an aarch64 host
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
MISE_RUST_VERSION=1.93.0 mise exec -- cargo check --all-targets --all-features             # MSRV
```

XGBoost parity (needs uv; fixtures are generated into the gitignored
`fixtures/`, never committed):

```sh
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/gen_fixtures.py
cargo test --test parity --release -- --ignored --nocapture
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/check_exports.py
```

CI (`.github/workflows/ci.yml`) runs all of these; tests run on x86_64 Linux,
aarch64 Linux, and aarch64 macOS. `scripts/README.md` documents the fixture
case matrix, tolerances, and benchmark harnesses.

## Lints

Clippy `pedantic` is on via `[lints.clippy]` in `Cargo.toml`, with a short
allow-list for lints that fight numeric code (casts, float equality, short
names, `inline(always)`, per-function `# Errors`/`# Panics` docs). CI denies
all warnings. Fix new warnings; a local `#[allow(..., reason = "...")]` is
acceptable only where the lint is wrong for that site. `clippy.toml` lists
doc identifiers (`XGBoost`, `TreeSHAP`, ...) exempt from `doc_markdown`.

## Layout (`src/`)

| Path | Contents |
|---|---|
| `data/` | `DMatrix` (dense/CSR, labels or a label matrix, label bounds, weights, groups, feature types and weights), `meta` (`MetaInfo`, the view objectives and metrics read), `loaders` (libsvm/CSV), `sketch`/`quantile` (quantile sketch, `HistCuts`), `ghist` (`GHistIndex` binning), `target_stats` (opt-in ordered target statistics, beyond XGBoost) |
| `config/` | `TrainingParams`, its builder and `validate`; names mirror XGBoost |
| `objective/` | Losses by XGBoost name: `regression`, `classification`, `multiclass`, `count`, `ranking`, `quantile` (quantile/expectile alpha lists), `absolute` (smoothed MAE), `survival` (Cox/AFT, glibc-exact `erf`), `multi_target` (label-matrix wrapper), `custom` hook; `distributional/` (opt-in `dist:*` families, `Dist`, `special` functions; beyond XGBoost) |
| `metric/` | Eval metrics by XGBoost name: `mod.rs` (rmse, mae, logloss, error, auc/aucpr, multiclass, count, ndcg/map, custom hook), `elementwise` (rmsle, mape, mphe), `ranking` (`pre@k`), `quantile` (quantile/expectile), `survival` (cox/aft-nloglik, interval accuracy), `distributional` (`nll`, `crps`; beyond XGBoost) |
| `tree/` | `RegTree` (scalar or vector leaves), `gain`, `constraints` (monotone/interaction), `sampler` (colsample bytree/bylevel/bynode, optionally feature-weighted), `builder/` (`exact`, `hist`, `multi` vector-leaf hist trees for `multi_output_tree`, and the opt-in `oblivious` symmetric growth, `lightgbm` `extra_trees`/`path_smooth` search, `budget` generalization-gated grower), `hist/` accumulation (`hist/quantized`: opt-in quantized-gradient histograms), `compact` (prediction layout, constant leaves only), `oblivious` (bit-pattern tables for symmetric trees), `linear` (opt-in `linear_tree` leaves: fit, storage, prediction), `reuse` (opt-in Trees-on-a-Diet reuse penalties) |
| `booster/` | `gblinear` |
| `learner/` | `train` (gbtree, DART, gblinear; `approx` = hist builder with per-round weighted cuts; `num_parallel_tree` forests), `model` (`BoostedModel`: iteration layout, slicing, `iteration_range` prediction, native formats), `multi_output` (vector-leaf rounds, reduced split gradients), `sampling` (gradient-based row sampling), `continuation` (continued training / `process_type=update` checks), `refresh` (refresh updater), `cv`, `shap` (QuadratureTreeSHAP), `conformal` (split-conformal / CQR intervals), `compact_model` (`CompactModel`, bit-packed Trees-on-a-Diet format), `budget` (opt-in PerpetualBooster-style training) |
| `model/` | XGBoost model import/export: `xgboost_json` (schema mapping, JSON and UBJSON entry points), `ubjson` (UBJSON codec over `serde_json::Value`) |
| `simd/` | Private runtime-dispatched kernels: `scalar`, `aarch64` (NEON), `x86_64` (AVX2/FMA, SSE2) |

`tests/`: `parity.rs` (ignored by default; needs fixtures), `properties.rs`
(proptest), `shap_accumulation.rs`, `sampling.rs` (row/column sampling
contracts), `target_stats.rs`, `continuation.rs` (continued training,
refresh, forests, slicing, iteration ranges), `quantized.rs`
(quantized-gradient training: thread-count determinism, quality band, leaf
renewal), `tree_options.rs` (`extra_trees`, `path_smooth`, `linear_tree`),
`budget.rs`, `multi_output.rs` (vector-leaf trees), `distributional.rs`
(`dist:*` objectives: interval calibration, NLL vs a homoscedastic baseline,
serialization, CQR), `native_format.rs` (native binary/JSON round trips,
refused versions and corrupt payloads, and the models saved by each release
in `tests/data/saved/<version>/`, which must keep loading with their
recorded margins).
`tests/data/xgboost-3.4.2-categorical.{json,ubj}` are committed XGBoost saves
that `model/xgboost_json.rs` unit tests import. `tests/common/` and
`examples/common/` hold the helpers shared by the integration tests and by
the examples. `benches/training.rs` is the Criterion suite;
`docs/performance.md` records its results.

## Invariants

- **Errors:** public fallible APIs return `hessboost::error::Result<T>`
  (`HessboostError`). No `unwrap`/`expect` on user-controlled input in library
  code.
- **Determinism:** identical params, data, and seed give identical
  predictions (property-tested), and the hist builder (every grow policy)
  grows the same tree serially and in parallel. Parallel reductions keep a fixed order.
  Quantized training (`use_quantized_grad`) draws its stochastic rounding
  from a counter-based stream keyed by row index and sums integers exactly,
  so it keeps the same guarantee.
- **Unsafe:** confined to `simd/` and the hot loops in `tree/compact.rs`,
  `tree/hist/`, and `tree/builder/hist.rs`. Every block needs a `// SAFETY:`
  comment (`undocumented_unsafe_blocks`); `unsafe_op_in_unsafe_fn` is
  forbidden.
- **SIMD:** dispatch checks CPU features at runtime and falls back to
  `simd/scalar.rs`. Split search, histogram sums, and prediction must match
  the scalar path bit for bit; transcendental kernels stay within the
  tolerances in `simd/tests.rs`. Code for one architecture only compiles on
  it, so lint the other target explicitly.
- **Parity:** the target is XGBoost 3.4.2 behavior (numerically identical to
  3.4.1). `exact`-tier fixtures match pointwise; RNG-driven cases
  (subsampling, DART) only match within a quality band because the RNG
  streams differ. Two sampling structures also differ from XGBoost and are
  covered only by that band: under `hist`, a multi-output model's outputs
  share each parallel tree's uniform row sample (XGBoost draws one per
  output group), and under `approx` with uniform sampling, the per-round
  cuts weight unsampled rows by their Hessian (XGBoost gives them zero
  weight). Categorical splits follow XGBoost's `HistEvaluator`
  (`EnumerateOneHot` below `max_cat_to_onehot = 4` categories, otherwise
  `EnumeratePart` scanned in both directions up to `max_cat_threshold = 64`)
  in `tree/builder/mod.rs::sweep_categorical`, shared by the histogram and
  exact builders.
- **Formats:** files written by a release keep loading in every later one.
  - Native binary (`learner/native.rs`): a zstd frame holding `SQB\0`, a
    container version byte, a table of named, typed sections
    (`learner/sections.rs`), and an XXH64 checksum of the preceding bytes.
    The sections hold model scalars (`model.*`), objective parameters
    (`objective.*`), and the trees column-wise (`node.*`, `tree.*`,
    `leaf_linear.*`). A new stored field is a new section, written with
    `REQUIRED` when older readers must refuse rather than skip it, and read
    with a default that reproduces files written before it existed (for
    objective parameters, the objective's defaults). Changing or removing an
    existing section's meaning, or the container layout, bumps
    `CONTAINER_VERSION` and keeps reading the previous version.
  - Native JSON (`BoostedModel`'s fields by name): a new field needs
    `#[serde(default)]` reproducing older files.
  - Compact (`HBTD`, documented in `learner/compact_model.rs`): its metadata
    is a section table like the native one; a change to the bit stream bumps
    its version byte.
  - Before each release, `cargo test --test native_format -- --ignored
    save_models_of_this_version` writes `tests/data/saved/<version>/`;
    commit it and never rewrite an earlier version's directory.
- **Tree layout:** as in XGBoost, iteration `i` owns trees
  `i * trees_per_iteration ..` (`trees_per_iteration = n_outputs ×
  num_parallel_tree`), grouped by output; tree `t` feeds output
  `BoostedModel::tree_output(t) = (t / num_parallel_tree) % n_outputs`.
  Iteration counts, `best_iteration`, slicing and `iteration_range` are
  iteration-based, never raw tree counts.
- **Prediction layout:** single-output and `multi:softmax` give `n_rows`
  values; `multi:softprob` gives `n_rows * num_class` and the other
  multi-output models (a multi-target label matrix, the `reg:quantileerror` /
  `reg:expectileerror` alpha lists, the `dist:*` parameters)
  `n_rows * n_outputs`, all row-major `[row][output]` (vector-leaf models
  too: each of their trees feeds every output, `num_parallel_tree` trees per
  iteration, `tree_info` 0); multi-target `predict_class` thresholds every
  target (`[row][target]`). SHAP contributions are `[row][n_features + 1]`
  (bias last), interactions `[row][(n_features + 1)^2]`, with an extra
  output axis for multi-output models. XGBoost's `num_target` counts outputs
  (one per label column, or one per alpha), while `BoostedModel::n_targets`
  counts label columns.
- **Objective/metric hooks:** training reaches objectives and metrics only
  through the `MetaInfo` hooks (`Objective::gradient_info`,
  `base_margins_info`, `eval_transform`, `probs_to_margins`, `validate_info`,
  `requires_labels`; `Metric::eval_info`, `validate_info`, `prediction_width`,
  `supports_label_matrix`).
  Label-domain checks live in each objective's `validate_info`;
  `create_objective(params, n_targets)` wraps the elementwise objectives
  listed in `MULTI_TARGET_OBJECTIVES` (`objective/mod.rs`) in
  `multi_target::MultiTarget` for a label matrix (row weights broadcast per
  cell, intercepts per column); `reg:absoluteerror` models label matrices
  itself (per-target intercepts), and label matrices any other objective
  cannot model are rejected. The default `Metric::eval_info` reduces a label
  matrix elementwise (every cell weighted by its row weight); non-elementwise
  metrics override it or return `supports_label_matrix() == false`, which
  training rejects; `Metric::validate_info` (labels required by default, the
  survival interval metrics accept label bounds instead) and
  `Metric::prediction_width` (predictions per row the metric reads, which
  must equal the model's outputs) are checked on every eval set before
  training, and `Metric::eval` returns NaN for inconsistent lengths. Unsupported parameter combinations are refused, never
  ignored: `TrainingParams::validate`, `multi_output::validate`
  (`multi_output_tree` needs `hist`), and `continuation.rs`
  (`process_type=update`). XGBoost-JSON import maps `base_score` with
  `probs_to_margins` and export with `margins_to_probs` (defaults to
  `pred_transform`; `binary:hinge` overrides it because its threshold is not
  its link). Budget
  mode (`learner/budget.rs`) instead refuses every `TrainingParams` field it
  does not read by diffing the serialized params against the defaults, so a
  new field is refused there automatically; it needs an objective's
  `Objective::pointwise_loss` (per-row loss matching its gradients).

## Public API at a glance

Everything below is in `hessboost::prelude`. The crate root exports only
modules; `prelude` is the one place that re-exports items. Other public items
are reached through their module (e.g. `hessboost::tree::RegTree`).

- `train(&params, &dtrain, rounds)`, `train_with_eval(.., &[(&DMatrix, "name")], early_stopping: Option<usize>)`,
  `train_with_objective(.., &dyn Objective)`, `train_with_custom_metric(.., Box<dyn Metric>)`,
  `train_continue(.., rounds, &model)` / `train_continue_with_eval(.., evals, early_stopping, &model)`
  (continued training; with `process_type(ProcessType::Update)` the refresh updater),
  `cv(&params, &data, rounds, nfold, seed)`.
- `train_with_budget(&params, &dtrain, &BudgetConfig::new(budget))` → `BudgetResult { model, eta, stop }`
  (opt-in, beyond XGBoost; no round count).
- `DMatrix::from_dense`/`from_csr`, `.with_labels`/`.with_label_matrix`/`.with_label_bounds`/
  `.with_weights`/`.with_base_margin`/`.with_group_sizes`/`.with_group_weights`/`.with_feature_types`/`.with_feature_weights`,
  `.info()` (`MetaInfo`); file loaders are `hessboost::data::{load_csv, load_libsvm}`.
- `BoostedModel::predict`/`predict_margin`/`predict_class`/`predict_leaf`/
  `predict_contribs`/`predict_interactions`, the `iteration_range` variants
  `predict_range`/`predict_margin_range`/`predict_leaf_range`/
  `predict_contribs_range`/`predict_interactions_range(.., (begin, end))`
  (leaf/contribs/interactions need `begin == 0`), `slice(begin, end, step)`,
  `num_boost_rounds`, `num_parallel_tree`, `feature_importance`, and
  `save_*`/`load_*` + `to_*`/`from_*` for native binary, JSON, XGBoost JSON,
  and XGBoost UBJSON (`*_xgboost_ubjson`).
- `model.to_compact_bytes()` / `model.to_compact()` → `CompactModel` (`from_bytes`/`load`,
  `to_bytes`/`save`, `predict_margin` bit-identical to the source model, `predict`),
  `model.size_report()` → `ModelSizeReport`; train with
  `toad_penalty_feature`/`toad_penalty_threshold` to shrink its dictionaries.
  gblinear, linear-leaf, and vector-leaf models have no compact encoding.
- `SplitConformal::calibrate(&model, &dcal, alpha)` and
  `ConformalizedQuantile::calibrate(&lo, &hi, ..)` / `calibrate_outputs(&model, lo, hi, ..)` /
  `calibrate_distribution(&model, ..)`, then `.predict_interval(&data)` → `Vec<(lower, upper)>`.
- `BoostedModel::predict_distribution(&data)` / `predict_distribution_range` → `Vec<Dist>`
  for `dist:*` models (`DistFamily`, `Dist`, `DistGradient`, `DistSplitDirection` in the prelude).

Opt-in quantized training: `.use_quantized_grad(true)` with
`num_grad_quant_bins`, `stochastic_rounding`, `quant_train_renew_leaf`
(`hist`/`approx` only; validated in `TrainingParams::validate`).

Multiclass objectives need `.num_class(k)`; ranking objectives need
`.with_group_sizes`; multi-target training takes `.with_label_matrix(y, k)`
and gives `k` outputs (`multi_strategy = one_output_per_tree`, or vector-leaf trees with
`multi_output_tree`; custom objectives can supply reduced split gradients
via `Objective::split_gradient`);
`survival:aft` needs `.with_label_bounds` (labels are optional), and
`survival:cox` reads negative labels as right-censored times.
`GrowPolicy::Symmetric` (opt-in, beyond XGBoost) needs `hist`/`approx`,
numerical features, and `1 <= max_depth <= config::MAX_SYMMETRIC_DEPTH`.

Opt-in LightGBM tree options (`extra_trees`/`extra_seed`, `path_smooth`,
`linear_tree`/`linear_lambda`) require the histogram builder and are refused
elsewhere in `TrainingParams::validate`. Linear-leaf trees
(`RegTree::linear_leaves`) predict through `tree::linear::accumulate_forest`
instead of the compact forest, and XGBoost export and TreeSHAP refuse them.

## Not implemented

See [`README.md`](README.md#not-implemented). Options that exist at only
one setting there are not parameters; `tests/parity.rs` fails a fixture
that sets them any other way.
