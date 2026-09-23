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
| `objective/`, `metric/` | Losses and eval metrics by XGBoost name, plus custom hooks |
| `tree/` | `RegTree`, split gain, monotone/interaction constraints, column sampler, `builder/{exact,hist}`, `hist/` accumulation (`hist/quantized`: opt-in LightGBM-style quantized-gradient histograms, `use_quantized_grad`, beyond XGBoost), `compact` (prediction layout) |
| `booster/` | `gblinear` |
| `learner/` | Training loop (gbtree, DART, gblinear; `approx` = hist builder with per-round weighted cuts), `BoostedModel`, cv, TreeSHAP, `conformal` (split-conformal / CQR intervals) |
| `model/` | XGBoost model import/export: `xgboost_json` (schema mapping, JSON and UBJSON entry points), `ubjson` (UBJSON codec over `serde_json::Value`) |
| `simd/` | Private runtime-dispatched kernels: `scalar`, `aarch64` (NEON), `x86_64` (AVX2/FMA, SSE2) |

`tests/`: `parity.rs` (ignored by default; needs fixtures), `properties.rs`
(proptest), `shap_accumulation.rs`, `quantized.rs` (quantized-gradient
training: thread-count determinism, quality band, leaf renewal). `benches/training.rs` is the Criterion
suite; `docs/performance.md` records its results.

## Invariants

- **Errors:** public fallible APIs return `hessboost::error::Result<T>`
  (`HessboostError`). No `unwrap`/`expect` on user-controlled input in library
  code.
- **Determinism:** identical params, data, and seed give identical
  predictions (property-tested), and the hist builder grows the same tree
  serially and in parallel. Parallel reductions keep a fixed order.
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
- **Formats:** the native binary magic (`SQB\0`) and the JSON layouts are
  compatibility contracts; do not change them without a migration.
- **Prediction layout:** single-output and `multi:softmax` give `n_rows`
  values; `multi:softprob` gives `n_rows * num_class`, row-major. SHAP
  contributions are `[row][n_features + 1]` (bias last), interactions
  `[row][(n_features + 1)^2]`, with an extra output axis for multiclass.
- **Objective/metric hooks:** training reaches objectives and metrics only
  through the `MetaInfo` hooks (`Objective::gradient_info`,
  `base_margins_info`, `eval_transform`, `probs_to_margins`, `validate_info`,
  `requires_labels`; `Metric::eval_info`). Label-domain checks live in each
  objective's `validate_info`; `create_objective(params, n_targets)` rejects
  label matrices an objective cannot model. Parameters the training loop does
  not act on yet are refused in `train.rs::reject_unimplemented`.

## Public API at a glance

Everything below is in `hessboost::prelude`. The crate root exports only
modules; `prelude` is the one place that re-exports items. Other public items
are reached through their module (e.g. `hessboost::tree::RegTree`).

- `train(&params, &dtrain, rounds)`, `train_with_eval(.., &[(&DMatrix, "name")], early_stopping: Option<usize>)`,
  `train_with_objective(.., &dyn Objective)`, `train_with_custom_metric(.., Box<dyn Metric>)`,
  `cv(&params, &data, rounds, nfold, seed)`.
- `DMatrix::from_dense`/`from_csr`, `.with_labels`/`.with_label_matrix`/`.with_label_bounds`/
  `.with_weights`/`.with_base_margin`/`.with_group_sizes`/`.with_feature_types`/`.with_feature_weights`,
  `.info()` (`MetaInfo`); file loaders are `hessboost::data::{load_csv, load_libsvm}`.
- `BoostedModel::predict`/`predict_margin`/`predict_class`/`predict_leaf`/
  `predict_contribs`/`predict_interactions`, `feature_importance`, and
  `save_*`/`load_*` for native binary, JSON, XGBoost JSON, and XGBoost UBJSON
  (`*_xgboost_ubjson`).
- `SplitConformal::calibrate(&model, &dcal, alpha)` and
  `ConformalizedQuantile::calibrate(&lo, &hi, ..)` / `calibrate_outputs(&model, lo, hi, ..)`,
  then `.predict_interval(&data)` → `Vec<(lower, upper)>`.

Opt-in quantized training: `.use_quantized_grad(true)` with
`num_grad_quant_bins`, `stochastic_rounding`, `quant_train_renew_leaf`
(`hist`/`approx` only; validated in `TrainingParams::validate`).

Multiclass objectives need `.num_class(k)`; ranking objectives need
`.with_group_sizes`.

## Not implemented

GPU training, distributed or external-memory training, and Python/CLI/C-ABI
bindings.
