# AGENTS.md

hessboost is a pure-Rust reimplementation of XGBoost gradient boosting: one
library crate, no C/C++, no FFI. User docs live in `README.md`, the rustdoc
(`src/lib.rs`), and `examples/`. This file covers working on the code.

The code is largely AI-generated. Treat unlisted behavior as unverified and
prove changes with the commands below.

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
| `data/` | `DMatrix` (dense/CSR, labels, weights, groups, feature types), libsvm/CSV loaders, quantile sketch and `HistCuts`, `GHistIndex` binning, opt-in ordered target statistics (`target_stats`, beyond XGBoost) |
| `config/` | `TrainingParams` and its builder; names mirror XGBoost |
| `objective/`, `metric/` | Losses and eval metrics by XGBoost name (`survival.rs` in each: Cox/AFT and their metrics, with a glibc-exact `erf`), plus custom hooks |
| `tree/` | `RegTree`, split gain, monotone/interaction constraints, column sampler (`sampler`: bytree/bylevel-per-depth/bynode, optionally feature-weighted), `builder/{exact,hist,oblivious}` (`oblivious`: opt-in `grow_policy = symmetric` level-wise growth, beyond XGBoost), `builder/budget` (five-fold generalization-gated grower for budget mode), `builder/lightgbm` (opt-in `extra_trees` / `path_smooth` split search, beyond XGBoost), `linear` (opt-in `linear_tree` leaf models: fit, storage, slow prediction path), `hist/` accumulation (`hist/quantized`: opt-in LightGBM quantized-gradient histograms), `compact` (prediction layout, constant leaves only), `oblivious` (bit-pattern tables `compact` uses for symmetric trees), `reuse` (opt-in Trees-on-a-Diet feature/threshold reuse penalties) |
| `booster/` | `gblinear` |
| `learner/` | Training loop (gbtree, DART, gblinear; `approx` = hist builder with per-round weighted cuts; `num_parallel_tree` forests), `budget` (opt-in PerpetualBooster-style budget training, beyond XGBoost), `sampling` (gradient-based/MVS row sampling per tree), `continuation` (continued-training / `process_type=update` checks), `refresh` (the refresh updater), `BoostedModel` (iteration layout, slicing, `iteration_range` prediction), cv, QuadratureTreeSHAP (`shap.rs`), `conformal` (split-conformal / CQR intervals), `compact_model` (bit-packed Trees-on-a-Diet format, `CompactModel`) |
| `model/` | XGBoost model import/export: `xgboost_json` (schema mapping, JSON and UBJSON entry points), `ubjson` (UBJSON codec over `serde_json::Value`) |
| `simd/` | Private runtime-dispatched kernels: `scalar`, `aarch64` (NEON), `x86_64` (AVX2/FMA, SSE2) |

`tests/`: `parity.rs` (ignored by default; needs fixtures), `properties.rs`
(proptest), `shap_accumulation.rs`, `sampling.rs` (row/column sampling
contracts), `target_stats.rs`, `continuation.rs` (continued training,
refresh, forests, slicing, iteration ranges), `quantized.rs` (quantized-gradient
training: thread-count determinism, quality band, leaf renewal), `budget.rs`. `benches/training.rs` is the Criterion
suite; `docs/performance.md` records its results.

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
  streams differ.
- **Formats:** the native binary magic (`SQB\0`), the JSON layouts and the
  compact layout (`HBTD`, version byte, documented in
  `learner/compact_model.rs`) are compatibility contracts; do not change them
  without a migration. The compact metadata embeds `ObjectiveParams` as
  postcard, so changing that struct changes the compact format too.
- **Tree layout:** as in XGBoost, iteration `i` owns trees
  `i * trees_per_iteration ..` (`trees_per_iteration = n_outputs ×
  num_parallel_tree`), grouped by output; tree `t` feeds output
  `BoostedModel::tree_output(t) = (t / num_parallel_tree) % n_outputs`.
  Iteration counts, `best_iteration`, slicing and `iteration_range` are
  iteration-based, never raw tree counts.
- **Prediction layout:** single-output and `multi:softmax` give `n_rows`
  values; `multi:softprob` gives `n_rows * num_class` and the other
  multi-output models (a multi-target label matrix, the `reg:quantileerror` /
  `reg:expectileerror` alpha lists) `n_rows * n_outputs`, all row-major
  `[row][output]`; multi-target `predict_class` thresholds every target
  (`[row][target]`). SHAP contributions are `[row][n_features + 1]` (bias
  last), interactions `[row][(n_features + 1)^2]`, with an extra output axis
  for multi-output models. XGBoost's `num_target` counts outputs (one per
  label column, or one per alpha), while `BoostedModel::n_targets` counts
  label columns.
- **Objective/metric hooks:** training reaches objectives and metrics only
  through the `MetaInfo` hooks (`Objective::gradient_info`,
  `base_margins_info`, `eval_transform`, `probs_to_margins`, `validate_info`,
  `requires_labels`; `Metric::eval_info`, `supports_label_matrix`).
  Label-domain checks live in each objective's `validate_info`;
  `create_objective(params, n_targets)` wraps the elementwise objectives in
  `objective::MULTI_TARGET_OBJECTIVES` in `objective/multi_target.rs` for a
  label matrix (row weights broadcast per cell, intercepts per column);
  `reg:absoluteerror` models label matrices itself (per-target intercepts),
  and label matrices any other objective cannot model are rejected. The default
  `Metric::eval_info` reduces a label matrix elementwise (every cell weighted
  by its row weight); non-elementwise metrics override it or return
  `supports_label_matrix() == false`, which training rejects. Parameters the
  training loop does not act on yet are refused in
  `train.rs::reject_unimplemented`. XGBoost-JSON import maps `base_score` with
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
  `.with_weights`/`.with_base_margin`/`.with_group_sizes`/`.with_feature_types`/`.with_feature_weights`,
  `.info()` (`MetaInfo`); file loaders are `hessboost::data::{load_csv, load_libsvm}`.
- `BoostedModel::predict`/`predict_margin`/`predict_class`/`predict_leaf`/
  `predict_contribs`/`predict_interactions`, their `*_range(.., (begin, end))`
  variants (XGBoost `iteration_range`; leaf/contribs/interactions need
  `begin == 0`), `slice(begin, end, step)`, `num_boost_rounds`,
  `num_parallel_tree`, `feature_importance`, and
  `save_*`/`load_*` for native binary, JSON, XGBoost JSON, and XGBoost UBJSON
  (`*_xgboost_ubjson`).
- `model.to_compact_bytes()` / `model.to_compact()` → `CompactModel` (`from_bytes`,
  `predict_margin` bit-identical to the source model, `predict`), `model.size_report()`
  → `ModelSizeReport`; train with `toad_penalty_feature`/`toad_penalty_threshold` to
  shrink its dictionaries.
- `SplitConformal::calibrate(&model, &dcal, alpha)` and
  `ConformalizedQuantile::calibrate(&lo, &hi, ..)` / `calibrate_outputs(&model, lo, hi, ..)`,
  then `.predict_interval(&data)` → `Vec<(lower, upper)>`.

Opt-in quantized training: `.use_quantized_grad(true)` with
`num_grad_quant_bins`, `stochastic_rounding`, `quant_train_renew_leaf`
(`hist`/`approx` only; validated in `TrainingParams::validate`).

Multiclass objectives need `.num_class(k)`; ranking objectives need
`.with_group_sizes`; multi-target training takes `.with_label_matrix(y, k)`
and gives `k` outputs (`multi_strategy = one_output_per_tree`);
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

GPU training, distributed or external-memory training, and Python/CLI/C-ABI
bindings.
