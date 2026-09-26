# AGENTS.md

hessboost reimplements XGBoost in Rust as one library crate. No C/C++ or FFI
besides `zstd` (libzstd, for native model files) and, with the macOS-only
`metal` feature, `objc2-metal`. User docs: `README.md` (overview only;
details belong in rustdoc), rustdoc (`src/lib.rs`, module docs),
`examples/`, `docs/performance.md`. No changelog: release notes are written
at release time.

## Toolchain

`mise install` provides the pinned Rust 1.98.1, `mbx` (build cache),
cargo-nextest, and uv; after changing a version, refresh `mise.lock` with
`mise lock`. MSRV 1.93. `Cargo.lock` is gitignored: never pass `--locked`.
libzstd needs a C compiler for every build target. docs.rs builds only
Linux (no Apple SDK for `zstd-sys`), so the Metal API renders only in a
local macOS `cargo doc --features metal`. `include` in `Cargo.toml` lists
what the published crate ships.

## Commands

```sh
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo nextest run --all-features
cargo test --doc --all-features   # nextest skips doctests
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
MISE_RUST_VERSION=1.93.0 mise exec -- cargo build --all-features   # MSRV
cargo semver-checks   # API vs. latest crates.io release; Cargo.toml's version must be a large enough bump
```

XGBoost parity needs uv, CMake, and a C++ compiler (the first run builds
XGBoost 3.4.2 from source). Fixtures go to the gitignored `fixtures/`;
never commit them. `scripts/README.md` documents the case matrix, tiers,
tolerances, and benchmark harnesses.

```sh
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/gen_fixtures.py
cargo nextest run --test parity --release --run-ignored only --no-capture
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/check_exports.py
```

CI (`.github/workflows/ci.yml`) runs these through `mbx` with
`RUSTFLAGS=-D warnings`. mise-action caches mise's tools; `MISE_ENV=ci`
loads `mise.ci.toml`, which moves rustup's toolchains into that cache.
Tests run on x86_64 Linux, aarch64 Linux, and
aarch64 macOS (Metal tests needing a device skip without one; a guard test
still fails if the kernels do not compile). Clippy runs on x86_64 Linux and
aarch64 macOS; everything else on x86_64 Linux only. `all-checks-passed`
gates merges. After touching `simd/` or `cfg(target_arch)` code, lint the
architecture your host is not:

```sh
cargo clippy --all-targets --all-features --target x86_64-unknown-linux-gnu -- -D warnings
cargo clippy --all-targets --all-features --target aarch64-unknown-linux-gnu -- -D warnings
```

Fuzzing: run from `fuzz/` (its own crate; its `mise.toml` adds nightly and
cargo-fuzz). `./run.sh [seconds] [target...]` rebuilds seeds from
`tests/data/saved/` and `fuzz/fixed-seeds/`, then runs each target (CI: 10
s). A crash is saved in `fuzz/artifacts/<target>/`; replay with
`cargo fuzz run <target> <file>`. Pass `--target <host triple>` as `run.sh`
does: prebuilt x86_64 cargo-fuzz defaults to musl, which the sanitizers
reject. After changing `train.rs`'s input layout, re-check
`fixed-seeds/train/*` with `cargo fuzz fmt train <file>`. Targets:
`native-model`, `json-model`, `xgboost-json-model`, `xgboost-ubjson-model`,
`compact-model` (accepted models must predict and round-trip), `loaders`,
`train` (valid params must train or error, identically across thread
counts).

Python bindings: `python/` is its own crate (like `fuzz/`), built by
maturin through uv. Unlike the root, its `Cargo.lock` and `uv.lock` are
committed and every build is locked; after changing the root crate's
dependencies run `cargo update --manifest-path python/Cargo.toml
--workspace`, after changing `python/pyproject.toml` run `uv lock`. From
`python/`:

```sh
uv sync --locked
uv run --locked pytest
uv run --locked python -m mypy.stubtest hessboost._hessboost
uv run --locked mypy --strict
uv run --locked pyright --verifytypes hessboost --ignoreexternal
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## Lints

Clippy `pedantic` is on (`Cargo.toml` lists the allowed lints). A new local
`#[allow]` needs `reason = "..."` and only where the lint is wrong for that
site. Never allow `clippy::too_many_arguments`: group parameters into a
named struct of things that belong together (e.g. per-call context vs.
per-node state). Add new proper nouns in docs to `clippy.toml`.

## Layout (`src/`)

|Path|Non-obvious contents|
|---|---|
|`lib.rs`|crate docs ("What's here", "Not implemented"), `prelude`, hidden `internals`|
|`rng.rs`|`Rng` (xoshiro256++), SplitMix64 counter-based streams|
|`data/`|`meta` (`MetaInfo`), `sketch`/`quantile` (`HistCuts`), `ghist` (`GHistIndex`), `target_stats` (public, opt-in)|
|`config/params.rs`|`TrainingParams`, builder, `validate`, parameter enums, `ObjectiveParams`|
|`objective/`|files by XGBoost family; `absolute` (smoothed MAE), `survival` (`erf` from glibc), `multi_target` (label-matrix wrapper), `distributional/` (public, `dist:*`)|
|`metric/`|`mod.rs` holds the factory, defaults, and most metrics; the rest by family|
|`tree/`|`regtree`, `gain`, `constraints`, `sampler` (colsample), `hist/` (accumulation; `quantized`), `compact`, `oblivious` (symmetric-tree prediction), `linear` (`linear_tree` leaves), `reuse` (Trees-on-a-Diet penalties); public: `RegTree`, `Node`, `LinearLeaves`|
|`tree/builder/`|`mod.rs`: split enumeration for all builders, `sweep_categorical`, `scan_numeric_splits` with the `f32` prefilter (`approx_run`, `APPROX_MARGIN`) and exact's `ScreenBound` screen (`Screen::bound`, `rules_out`), both proven to keep the sequential choice. `hist` (also `approx`; speculative parallel loss-guide), `exact`, `multi` (vector leaves), `oblivious`, `lightgbm` (`extra_trees`/`path_smooth`), `budget`|
|`training/`|`train` (gbtree, DART, gblinear, forests; `approx` = hist with per-round weighted cuts), `gblinear`, `multi_output`, `sampling` (gradient-based), `continuation`, `refresh`, `cv` (`Fold` builders incl. `purged_forward`), `budget` (public)|
|`model/`|`mod.rs` (`BoostedModel`; XGBoost interchange docs), `native`, `sections` (shared by native and compact), `shap` (QuadratureTreeSHAP), `compact` (public, `HBTD`), `xgboost` (JSON/UBJSON schema), `ubjson` (codec over `serde_json::Value`)|
|`backend/`|`metal.rs` (GPU histograms and prediction, runtime-compiled MSL), `exact_sum.rs` (`SumDomain` and its proof; built on every platform)|
|`simd/`|`scalar`, `aarch64` (NEON), `x86_64` (AVX2/FMA, SSE2), `tests`|

Tests: `tests/parity.rs` is ignored without fixtures; `properties.rs` is
proptest; shared helpers are in `tests/common/` and `examples/common/`.
`tests/data/saved/<version>/` holds each release's saved models (`.bin`,
`.json`, `.hbtd`, `.margins`); `tests/data/xgboost-3.4.2-categorical.*` are
XGBoost saves for `model/xgboost.rs` tests. `benches/training.rs`
(Criterion) results go in `docs/performance.md`.

## Layout (`python/`)

|Path|Non-obvious contents|
|---|---|
|`Cargo.toml`|`hessboost-python`, version = root's (the wheel's); `include` is the sdist; `metal` on macOS|
|`src/`|private extension `hessboost._hessboost`: `data` (`DMatrix`, metadata dict → setters), `params` (mapping → `TrainingParams`), `booster` (predict variants, formats, format detection), `train` (`Trainer` on a signal-polled worker thread, `cv`, folds, Python callbacks), `conformal` (calibrators owning their model via `self_cell`), `dist`|
|`python/hessboost/`|the public API, pure Python: `_core` (`DMatrix`, `Booster`), `_data` (numpy/pandas/scipy conversion, category re-coding), `_training` (`train`, `cv`), `sklearn`, `conformal`, `folds`; `_hessboost.pyi` (native stub), `_sklearn_base.pyi` (typed scikit-learn bases)|
|`tests/`|pytest; `test_model_io.py` checks the root's `tests/data/saved/` margins bit for bit|

## Invariants

- **Errors:** public fallible APIs return `error::Result`. No panics
  (`unwrap`, `expect`, ...) on user input, NaN included, in library code.
- **Determinism:** same params, data, and seed give the same predictions
  at any thread count. Every grow policy grows the same tree serially and
  in parallel; parallel reductions keep a fixed order. CPU `f64`
  histograms (`tree/hist/mod.rs`) add each bin's rows in row order; a large
  node splits into fixed row blocks reduced in block order, partitioned by
  the data, never the thread count, and the serial build sums the same
  blocks. Sequential draws (rows, columns, DART, folds, target-stat
  permutations) use `rng::Rng`. Keyed draws (`extra_trees` node seeds,
  `dist:*` split direction, quantized stochastic rounding, per-block
  row-sampling seeds) use SplitMix64 streams keyed by seed and index.
  Quantized histograms sum integers. `rand` stays a dev-dependency.
  `Trainer::on_round` only observes: a hook that always continues leaves
  the model byte-identical, and a `Break` after round `k` gives the
  `k + 1`-round model (`tests/round_hook.rs`).
- **Metal:** `device = metal` reproduces the single-threaded CPU model bit
  for bit: gradients are staged as integer multiples of a per-component
  grain and summed in 64-bit integers (order-free, no atomics), and a node
  goes to the GPU only where the CPU's `f64` sums are also exact
  (`n * max <= 2^53` grains, `backend/exact_sum.rs`). Everything else
  (small nodes, non-finite gradients, failed command buffers) runs on CPU.
- **Unsafe:** only in `simd/`, hot loops of `tree/compact.rs`, `tree/hist/`,
  `tree/builder/hist.rs`, and `backend/metal.rs`. Each block needs
  `// SAFETY:`.
- **SIMD:** covers objective gradients, exp/sigmoid/softmax, metric sums,
  cut search (`count_le`), and SHAP's per-lane kernels (return-edge terms
  `shap_edge_terms`, child basis `shap_scaled_basis`/`shap_divided_basis`),
  with runtime dispatch falling back to
  `simd/scalar.rs` (also below minimum lengths and outside approximation
  ranges). Cut search and the SHAP kernels match scalar exactly;
  transcendentals stay within `simd/tests.rs` tolerances. Split search,
  histograms, and prediction are scalar, follow XGBoost's `f32` operation
  order, and their optimized paths must stay bit-identical to the plain
  ones. Data-dependent tree-walk steps go through `step_if_greater`
  (AArch64 `cmp` + `cinc`): LLVM lowers the plain select to a branch that
  random rows mispredict.
- **Parity (XGBoost 3.4.2):** fixture tiers `exact` (pointwise
  train/import/export, incl. `rank:*`), `quality` (RNG-driven: subsampling,
  colsample, forests, DART; training within a quality band, import/export
  pointwise), `trainonly` (gblinear). Two deliberate differences, covered
  only by the band: under `hist`, a multi-output model's outputs share each
  parallel tree's row sample (XGBoost draws per output group); under
  `approx` with uniform sampling, per-round cuts weight unsampled rows by
  Hessian (XGBoost: zero). Categorical splits follow XGBoost's
  `HistEvaluator` (one-hot below 4 categories, else partition scanned both
  ways up to 64) for the hist and exact builders. Beyond-XGBoost features are
  opt-in, default off, and absent from fixtures.
- **Parity-fixed options:** XGBoost options supported at one setting only
  ("Not implemented" in `lib.rs`) are not `TrainingParams` fields;
  `tests/parity.rs` (`expect_fixed`) fails fixtures that set `updater`,
  `feature_selector`, or `lambdarank_pair_method` otherwise.
- **Formats:** every file written since 0.2.0 loads in every later release;
  0.1.x files are refused.
  - Native binary (`model/native.rs`): zstd frame of magic `HBM\0` (0.1.x's
    `SQB\0` is refused by name), `CONTAINER_VERSION` byte (3), section table
    (`model/sections.rs`), XXH64 of the preceding bytes. A new stored field
    is a new section: flag it `REQUIRED` if unaware readers must refuse
    rather than skip it, and default its absence to reproduce older files
    (objective parameters: the objective's defaults). Readers refuse
    anything undefined inside known sections (unknown `node.flags` bits,
    `tree.has_linear` not 0/1, trailing bytes), which is what makes new flag
    bits safe. Changing a section's meaning or the container layout bumps
    `CONTAINER_VERSION` and needs a reader for the previous version (only
    the current one is accepted). A save whose frame would exceed the
    reader's decompression bound (`ALWAYS_ALLOWED`, `MAX_EXPANSION`) is
    written uncompressed, so every save loads.
  - Native JSON: `BoostedModel`, `RegTree`, and `LinearLeaves` deserialize
    via `Unchecked…` mirrors (`#[serde(try_from)]`) and validate. A new
    field goes in the type and its mirror, with a `#[serde(default)]` on the
    mirror that reproduces older files. A new `ObjectiveParams` field goes
    in the `objective_param_mirrors!` list (`config/params.rs`); missing
    objective fields take `ObjectiveParams::defaults_for(objective)`.
    Everything predictions depend on is required, nullable ones via
    `deserialize_with = "Option::deserialize"` (a plain `Option` would
    default when absent); exceptions: a tree may omit `size_leaf_vector`
    (0) except in multi-output models, and `leaf_vectors`; an absent
    `best_iteration` means none. Writers emit every field.
  - Compact (`HBTD`, `model/compact.rs`): section-table metadata; a bit
    stream change bumps its version byte (1).
- **Tree layout:** iteration `i` owns trees `i * trees_per_iteration ..`.
  Scalar leaves: `n_outputs × num_parallel_tree` per iteration, grouped by
  output; tree `t` feeds output `(t / num_parallel_tree) % n_outputs`.
  Vector leaves: `num_parallel_tree` per iteration, each feeding all
  outputs. Counts, `best_iteration`, slicing, and ranges are in iterations,
  never trees.
- **Prediction layout:** single-output and `multi:softmax` give `n_rows`
  values; `multi:softprob` `n_rows * num_class`; other multi-output models
  `n_rows * n_outputs`, row-major. Multi-target `predict_class` thresholds
  each target. SHAP: `[row][n_features + 1]` (bias last), interactions
  `[row][(n_features + 1)^2]`, plus an output axis for multi-output.
  `n_targets` counts label columns; XGBoost's `num_target` counts outputs
  (columns or alphas) and is 1 for multiclass.
- **Objective/metric hooks:** training and evaluation read data only via
  `MetaInfo` hooks (`Objective::gradient_info`, `base_margins_info`,
  `eval_transform`, `validate_info`, `requires_labels`;
  `Metric::eval_info`, `validate_info`, `prediction_width`,
  `supports_label_matrix`). `base_margins_info` is the only intercept hook;
  `probs_to_margins` is the only link hook, applied to user, imported, and
  Newton-default `base_score`. `margins_to_probs` exports `base_score`
  (default `pred_transform`; `binary:hinge` and `reg:quantileerror`
  override it). Label-domain checks go in each objective's `validate_info`.
  Label matrices: `create_objective` wraps `MULTI_TARGET_OBJECTIVES` in
  `MultiTarget` (row weight per cell, per-column intercepts);
  `reg:absoluteerror` handles them itself; other built-ins refuse them. The
  default `Metric::eval_info` reduces them elementwise; other metrics
  override it or report `supports_label_matrix() == false`, which training
  refuses. Every eval set is checked before training (`validate_info`;
  `prediction_width` equal to the model's outputs, or for `None` a whole
  number per label column); `Metric::eval` returns NaN on length mismatch.
  `Metric::name` is XGBoost's `evals_result` key, suffix included
  (`ndcg@5`, `pre@3`, `tweedie-nloglik@1.5`), so parity compares names.
- **Refusals:** unsupported parameters or combinations error, never get
  ignored. Checks live in `TrainingParams::validate` (static),
  `validate_request` in `training/train.rs` (data-dependent),
  `training/multi_output.rs::validate`, `training/continuation.rs`,
  `metric/mod.rs::build` (metric suffixes), and `training/budget.rs`.
  Budget mode and refresh compare params against defaults plus an
  allow-list (`TrainingParams::refuse_changes_from`), so any new field is
  refused there automatically.
- **Python:** `python/` uses only the crate's public API. The public
  Python API is pure Python; the extension is private, fully stubbed
  (stubtest), `unsafe`-free (`forbid`), declares `gil_used = false`, keeps
  every class `frozen`, and releases the GIL around matrix construction,
  training, prediction, and model encode/decode. Parameter mappings
  deserialize through `TrainingParams`' serde (XGBoost) names plus the
  aliases and one-setting options in `python/src/params.rs`, so a new
  field is accepted automatically and unknown keys are refused. `train`
  runs on a worker thread while the caller polls for signals; Python
  callbacks (objective, metric, per-round) re-attach to the interpreter,
  and the first exception (or Ctrl-C's `KeyboardInterrupt`) stops
  training through `Trainer::on_round` at the end of the round and is
  re-raised. Crate errors map to `HessboostError` (a `ValueError`),
  `ModelFormatError`, and `OSError`; wrong Python types raise `TypeError`.

## Public API

- The crate root exports only modules. The prelude holds the
  train-and-predict workflow and types its everyday methods take; the rest
  is imported from its module.
- One public path per item (plus the prelude): no flat re-exports, no
  aliases.
- Opt-in subsystems with substantial docs get their own public module
  (`data::target_stats`, `training::budget`, `model::compact`,
  `objective::distributional`, `conformal`).
- Implementation modules are crate-private; benches and parity tests reach
  internals through `#[doc(hidden)] pub mod internals` in `lib.rs`, which
  is not public API.
- `#[non_exhaustive]` on every public enum, struct with public fields, and
  unit-struct built-in that could grow; users build them via `Default`, a
  builder, or a constructor. Only closed sets stay exhaustive (`Monotone`,
  `GradPair`, `Dist`'s variants).

Easy-to-miss requirements: multiclass needs `.num_class(k)`; ranking needs
`.with_group_sizes`; `survival:aft` needs `.with_label_bounds`;
`survival:cox` reads non-positive labels as right-censored;
`Objective::split_gradient` serves vector-leaf trees only, not with
monotone constraints. Linear-leaf models predict through `tree::linear`;
XGBoost export, SHAP, and compact refuse them.

## When changing behavior

In the same change, update the touched items' rustdoc, the README's
feature lists and caveats, `lib.rs` "What's here" and "Not implemented",
this file, and affected examples. New options need a `TrainingParams`
field, builder setter, and validation.

## Releases

Bump `version` in `Cargo.toml`; write `tests/data/saved/<version>/` with
`cargo nextest run --test native_format --run-ignored only
save_models_of_this_version` and commit it (never regenerate an older
version's directory); merge; push tag `v<version>`. `publish.yml` checks
the tag, tests on all three platforms, publishes to crates.io, and creates
the GitHub release (with a discussion), its notes seeded from PRs since the
last tag (grouped by `.github/release.yml`); rewrite them by hand then.
