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

LightGBM import parity (LightGBM 4.7.0 wheels; fixtures in
`fixtures/lightgbm/`, pointwise within `1e-5` relative, refusals by message):

```sh
uv run --with-requirements scripts/requirements-lightgbm.txt python scripts/gen_lightgbm_fixtures.py
cargo nextest run --test lightgbm_parity --release --run-ignored only --no-capture
```

CI (`.github/workflows/ci.yml`) runs the Rust checks through `mbx` with
`RUSTFLAGS=-D warnings`. mise-action caches mise's tools; `MISE_ENV=ci`
loads `mise.ci.toml`, which moves rustup's toolchains into that cache. Rust
tests run on x86_64 Linux, aarch64 Linux, and aarch64 macOS (Metal tests
needing a device skip without one; a guard test still fails if the kernels
do not compile). Its Python jobs build and test the extension on
x86_64/aarch64 Linux, aarch64 macOS, and x86_64 Windows across CPython 3.11,
latest Python 3.x, and free-threaded 3.14t, plus pyright's public-type
check, a ruff/ty lint job over all of the repository's Python, and an sdist
round trip. Root fmt also checks `python/Cargo.toml`; Python
clippy runs in both the x86_64-linux and aarch64-macOS lint jobs (the
latter checks the Metal feature). `all-checks-passed` gates merges. After
touching `simd/` or `cfg(target_arch)` code, lint the architecture your
host is not:

```sh
cargo clippy --all-targets --all-features --target x86_64-unknown-linux-gnu -- -D warnings
cargo clippy --all-targets --all-features --target aarch64-unknown-linux-gnu -- -D warnings
```

Fuzzing: run from `fuzz/` (its own crate; its `mise.toml` adds nightly and
cargo-fuzz). `./run.sh [seconds] [target...]` rebuilds seeds from
`tests/data/` and `fuzz/fixed-seeds/`, then runs each target (CI: 10
s). A crash is saved in `fuzz/artifacts/<target>/`; replay with
`cargo fuzz run <target> <file>`. Pass `--target <host triple>` as `run.sh`
does: prebuilt x86_64 cargo-fuzz defaults to musl, which the sanitizers
reject. After changing `train.rs`'s input layout, re-check
`fixed-seeds/train/*` with `cargo fuzz fmt train <file>`. Targets:
`native-model`, `json-model`, `xgboost-json-model`, `xgboost-ubjson-model`,
`lightgbm-model`, `compact-model` (accepted models must predict and
round-trip), `diffusion-model` (binary and JSON; accepted models must
sample and round-trip), `loaders`, `train` (valid params must train or
error, identically across thread counts).

Python bindings: `python/` is its own crate (like `fuzz/`), built by
maturin through uv. Unlike the root, its `Cargo.lock` and `uv.lock` are
committed and every build is locked; after changing the root crate's
dependencies run `cargo update --manifest-path python/Cargo.toml
--workspace`, after changing `python/pyproject.toml` run `uv lock`. From
`python/`:

```sh
uv sync --locked
uv run --locked pytest
uv run --locked pyright --verifytypes hessboost --ignoreexternal
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Python lint and types, for `python/`, `scripts/`, `release.py`, and
Markdown's Python snippets (`ruff.toml`, `ty.toml`), from the root:

```sh
uv run --project python --locked ruff check
uv run --project python --locked ruff format --check
uv run --project python --locked ty check
```

## Lints

Clippy `pedantic` is on (`Cargo.toml` lists the allowed lints). A new local
`#[allow]` needs `reason = "..."` and only where the lint is wrong for that
site. Never allow `clippy::too_many_arguments`: group parameters into a
named struct of things that belong together (e.g. per-call context vs.
per-node state). Add new proper nouns in docs to `clippy.toml`.

Python: `ruff.toml` selects the rule families (and says why naming and
signature rules are off); `ty.toml` fails on warnings and honors only
`# ty: ignore[rule]`, which is for tests that pass a wrong type on purpose.
Fix findings rather than suppress them.

## Layout (`src/`)

|Path|Non-obvious contents|
|---|---|
|`lib.rs`|crate docs ("What's here", "Not implemented"), `prelude`, hidden `internals`|
|`rng.rs`|`Rng` (xoshiro256++), SplitMix64 counter-based streams (`stream_key`, `keyed_normal`)|
|`data/`|`meta` (`MetaInfo`), `sketch`/`quantile` (`HistCuts`), `ghist` (`GHistIndex`), `target_stats` (public, opt-in)|
|`config/params.rs`|`TrainingParams`, builder, `validate`, `loss` (the loss a configuration trains with), parameter enums|
|`config/groups.rs`|option groups a switch owns: `Dart` (`BoosterKind::Dart`), `Boulevard` (`BoosterKind::Boulevard`), `Ebm` (`BoosterKind::Ebm`), `Refresh` (`ProcessType::Update`), `QuantizedGrad`, `ExtraTrees`, `LinearTree`, `BalancedBagging`, `QueryBagging`, `Langevin`, `ModelShrink` (`Option` fields); each validates when built|
|`config/xgboost.rs`|XGBoost's flat parameter form: `TrainingParams::from_xgboost`/`to_xgboost` (keys, aliases, value spellings, one-setting options), `changed_keys`|
|`objective/`|`spec` (`Objective`: one exhaustive match per property, `build_loss`, `ObjectiveParts`/`from_parts`/`parts`, the flat keys by XGBoost name), `params` (the validated parameter structs, shared with `EvalMetric`); losses by XGBoost family (crate-private): `absolute` (smoothed MAE), `survival` (`erf` from glibc), `xendcg` (LightGBM XE-NDCG; its own keyed RNG stream), `multi_target` (label-matrix wrapper), `distributional/` (public, `dist:*`, `Distributional`)|
|`metric/`|`mod.rs` holds `EvalMetric` (the typed metrics; `from_xgboost` reads XGBoost names with the flat parameters they borrow) and most metrics; the rest by family (built-in metric structs are crate-private)|
|`tree/`|`regtree`, `gain`, `constraints`, `sampler` (colsample), `hist/` (accumulation; `quantized`), `compact`, `oblivious` (symmetric-tree prediction), `linear` (`linear_tree` leaves), `reuse` (Trees-on-a-Diet penalties); public: `RegTree`, `Node`, `LinearLeaves`|
|`tree/builder/`|`mod.rs`: split enumeration for all builders, `sweep_categorical`, `scan_numeric_splits` with the `f32` prefilter (`approx_run`, `APPROX_MARGIN`) and exact's `ScreenBound` screen (`Screen::bound`, `rules_out`), both proven to keep the sequential choice. `hist` (also `approx`; speculative parallel loss-guide), `exact`, `multi` (vector leaves), `oblivious`, `lightgbm` (`extra_trees`/`path_smooth`), `budget`, `online` (split ranking for `training::online`)|
|`training/`|`train` (gbtree, DART, gblinear, forests; `approx` = hist with per-round weighted cuts; uniform, class-balanced, and query-level row sampling), `boulevard` (BRAT-D/BRAT-P `Recursion`, shared with the honest refit), `ebm` (classic cyclic EBM with outer bags, Boulevard EBM stages on the same `Recursion`, FAST pair ranking), `gblinear`, `multi_output`, `sampling` (gradient-based), `sglb` (Langevin noise, leaf re-estimation, shrink schedule), `continuation`, `refresh`, `cv` (`Fold` builders incl. `purged_forward`), `budget` (public), `online` (public: in-place row addition/deletion; cached per-node histograms, split robustness tolerance, lazy gradients; exact mode = retraining)|
|`inference/`|public: Boulevard inference (`BoulevardInfo`, `BoulevardInference`, `EbmInference`, `TermBands`, `honest_refit`); `kernel` (the `Kernel` trait the solvers read; leaf kernel over the training rows), `term_kernel` (a Boulevard EBM stage's centered additive kernel, computed on term grids), `solver` (exact Cholesky or Nyström ridge solves, Gram or solution vectors), `linalg` (blocked and pivoted Cholesky, triangular solves), `refit` (`honest_refit`, Boulevard models and Boulevard EBMs), `ebm` (shape-function bands)|
|`ebm/`|public: `EbmInfo` (terms, tree→term map, term means), `shape_functions`, `TermShape`; `grid` (a term's cell grid from its trees' thresholds, leaves as boxes, difference arrays)|
|`model/`|`mod.rs` (`BoostedModel`; XGBoost interchange docs), `objective` (`ModelObjective`; `StoredObjectiveParams`, the stored objective-parameter record), `native`, `sections` (shared by native and compact), `shap` (QuadratureTreeSHAP), `shrinkage` (per-iteration record; training's shrink step, shared by prediction), `uncertainty` (public, virtual ensembles), `compact` (public, `HBTD`), `xgboost` (JSON/UBJSON schema), `ubjson` (codec over `serde_json::Value`), `lightgbm` (LightGBM text import; mapping docs in `mod.rs`, "LightGBM import")|
|`diffusion/`|public, opt-in: `mod.rs` (params, `DiffusionModel`), `process` (SDE kernels, flow paths, time sampling, Box–Muller and keyed normal draws), `fit` (standardization, cross-fitted residualizer, noisy training set), `sample` (reverse SDE/ODE, `Samples`, `Quantiles`), `format` (`HBDM` container embedding native GBDT containers; JSON)|
|`backend/`|`metal.rs` (GPU histograms and prediction, runtime-compiled MSL), `exact_sum.rs` (`SumDomain` and its proof; built on every platform)|
|`simd/`|`scalar`, `aarch64` (NEON), `x86_64` (AVX2/FMA, SSE2), `tests`|

Tests: `tests/parity.rs` and `tests/lightgbm_parity.rs` are ignored without fixtures; `properties.rs` is
proptest; shared helpers are in `tests/common/` and `examples/common/`.
`tests/data/saved/<version>/` holds each release's saved models (`.bin`,
`.json`, `.hbtd`, `.margins`); `tests/data/xgboost-3.4.2-categorical.*` are
XGBoost saves for `model/xgboost.rs` tests, and `tests/data/lightgbm-4.7.0-*`
LightGBM saves (with LightGBM's predictions in `*.expected.json`, written by
`gen_lightgbm_fixtures.py --test-data`) for `model/lightgbm.rs` tests. `benches/training.rs`
(Criterion) results go in `docs/performance.md`.

## Layout (`python/`)

|Path|Non-obvious contents|
|---|---|
|`Cargo.toml`|`hessboost-python`, version = root's (the wheel's); `include` is the sdist; `metal` on macOS|
|`src/`|private extension `hessboost._hessboost`: `data` (`DMatrix`, metadata dict → setters), `params` (mapping → `TrainingParams`), `booster` (predict variants, formats, format detection), `train` (`Trainer` on a signal-polled worker thread via `run_hooked`, `cv`, folds, Python callbacks), `conformal` (calibrators owning their model via `self_cell`), `inference` (`BoulevardInference` owning its model and holdout rows via `self_cell`, `honest_refit`), `ebm` (`TermShape`, `shape_functions`, `EbmInference`), `online` (`OnlineModel`: the one mutable class, its state behind a mutex locked only detached; updates through `run_hooked`), `dist`, `diffusion` (`DiffusionParams` from a request dict with the `Method` as its serde JSON, `DiffusionModel`, `Samples` summaries of draw arrays; `fit` has no round hook, so it is not interruptible)|
|`python/hessboost/`|the public API, pure Python: `_core` (`DMatrix`, `Booster`, `_check_schema`: the feature-name/categorical/category-order check every pairing of data with a model or `dtrain` goes through), `_data` (numpy/pandas/scipy conversion, category re-coding), `_training` (`train`, `cv`), `sklearn`, `conformal`, `diffusion` (frozen dataclasses mirroring `hessboost::diffusion`, presets read from the crate, `DiffusionModel`, `mean`/`quantiles`/`crps`), `inference` (`BoulevardInference`, `BoulevardInfo`, `EbmInference`, `TermBands`, `honest_refit`), `ebm` (`shape_functions`, `TermShape`, axes, `EbmInfo`), `folds`, `online` (`OnlineModel`, `UpdateReport`); `_hessboost.pyi` (native stub), `_sklearn_base.pyi` (typed scikit-learn bases)|
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
  permutations, diffusion training noise, splits and folds) use
  `rng::Rng`. Keyed draws (`extra_trees` node seeds, `dist:*` split
  direction, quantized stochastic rounding, per-block row-sampling seeds,
  Langevin noise, diffusion sampler noise keyed by row and sample) use
  SplitMix64 streams keyed by seed and index. Boulevard rounds (dropout
  sets, row samples) and Nyström landmarks draw from `rng::Rng` seeded per
  round; the inference's parallel loops split by rows or fixed row blocks,
  never by thread. EBM rounds draw from `Rng` keyed by seed, bag, stage,
  and round (bags' rows from SplitMix64 keyed by bag); bags and a Boulevard
  round's per-term trees may grow in parallel and are combined in bag and
  term order.
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
  `TrainingParams::from_xgboost` (`FIXED` in `config/xgboost.rs`) accepts
  `updater`, `feature_selector`, `lambdarank_pair_method`,
  `max_cat_to_onehot`, and `max_cat_threshold` at that setting only, so
  parity fixtures that set them otherwise fail.
- **Formats:** every file written since 0.2.0 loads in every later release;
  0.1.x files are refused.
  - Native binary (`model/native.rs`): zstd frame of magic `HBM\0` (0.1.x's
    `SQB\0` is refused by name), `CONTAINER_VERSION` byte (3), section table
    (`model/sections.rs`), XXH64 of the preceding bytes. A new stored field
    is a new section: flag it `REQUIRED` if unaware readers must refuse
    rather than skip it, and default its absence to reproduce older files
    (objective parameters: the objective's defaults; the optional
    `boulevard.*` sections: not a Boulevard fit; the optional `ebm.*`
    sections: not an EBM). Readers refuse
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
    mirror that reproduces older files. A model's objective is stored as
    its name plus `StoredObjectiveParams` (`model/objective.rs`): every
    built-in objective's parameters, the defaults for those the objective
    does not read, so saves stay byte-identical to 0.2.0's. A new objective
    parameter goes in `StoredObjectiveParams` and
    `PartialStoredObjectiveParams` (each member a `Stored`, so `null` stays
    an error), and missing members take
    `StoredObjectiveParams::defaults_for(objective)`. Loading maps the
    record to `ModelObjective::from_stored` (an unknown name is recorded
    as a name alone, a built-in name always loads as that objective;
    parameters the objective does not read are dropped). `ModelObjective`
    keeps its representation private so a recorded name is never a
    built-in objective's and a built-in objective never a custom loss.
    A missing `boulevard` or `ebm` is `None`.
    Everything predictions depend on is required, nullable ones via
    `deserialize_with = "Option::deserialize"` (a plain `Option` would
    default when absent); exceptions: a tree may omit `size_leaf_vector`
    (0) except in multi-output models, and `leaf_vectors`; an absent
    `best_iteration` means none, and an absent `shrinkage` means no model
    shrinkage. Writers emit every field.
  - Compact (`HBTD`, `model/compact.rs`): section-table metadata; a bit
    stream change bumps its version byte (1).
  - Diffusion (`HBDM`, `diffusion/format.rs`): the native container framing
    (`model::native::frame`, `section_table`) with its own magic and version
    byte (1), the same section rules, and the GBDTs embedded as uncompressed
    native containers; its JSON validates through `UncheckedDiffusionModel`.
- **Tree layout:** iteration `i` owns trees `i * trees_per_iteration ..`.
  Scalar leaves: `n_outputs × num_parallel_tree` per iteration, grouped by
  output; tree `t` feeds output `(t / num_parallel_tree) % n_outputs`.
  Vector leaves: `num_parallel_tree` per iteration, each feeding all
  outputs. Counts, `best_iteration`, slicing, and ranges are in iterations,
  never trees. Tree weights are DART's or model shrinkage's: a shrunk
  model stores unscaled trees, per-iteration coefficients, and the
  unshrunk intercepts (`model/shrinkage.rs`), from which its closed-form
  tree weights and intercepts derive bit for bit (SHAP and XGBoost export
  use them). Its predictions (native, compact, `..k` ranges,
  `slice(..k, 1)`, virtual ensembles) repeat training's shrink-then-add
  recurrence (`shrink_margins`), so they are the training margins and the
  `k`-round model bit for bit; later starts are refused, and early
  stopping truncates it to the best iteration.
- **Prediction layout:** predictions return `model::Predictions` (row-major
  `n_rows × width`, owning the computed buffer without a copy): width
  `n_outputs` (`num_class` for `multi:softprob`), 1 for `multi:softmax`, tree
  count for leaves. Multi-target `predict_class` thresholds each target. SHAP:
  `Contributions` `[row][output][n_features + 1]` (bias last),
  `Interactions` `[row][output][(n_features + 1)^2]`. No `Deref` to slices:
  callers take the flat buffer (`as_slice`/`into_vec`) only where it is flat
  (numpy, metrics). `n_targets` counts label columns; XGBoost's `num_target`
  counts outputs (columns or alphas) and is 1 for multiclass.
- **Loss/metric hooks:** training and evaluation read data only via
  `MetaInfo` hooks (`Loss::gradient_info`, `base_margins_info`,
  `eval_transform`, `validate_info`, `requires_labels`;
  `Metric::eval_info`, `validate_info`, `prediction_width`,
  `supports_label_matrix`). `base_margins_info` is the only intercept hook;
  `probs_to_margins` is the only link hook, applied to user, imported, and
  Newton-default `base_score`; a user `base_score` is first checked by
  `validate_base_score` of the loss being trained (never by the configured
  objective's name). `margins_to_probs` exports `base_score`
  (default `pred_transform`; `binary:hinge` and `reg:quantileerror`
  override it). Label-domain checks go in each loss's `validate_info`.
  Label matrices: `Objective::build_loss` wraps the `LabelMatrix::PerColumn`
  objectives in `MultiTarget` (row weight per cell, per-column intercepts);
  `reg:absoluteerror` and custom losses handle them themselves; other
  built-ins refuse them. The
  default `Metric::eval_info` reduces them elementwise; other metrics
  override it or report `supports_label_matrix() == false`, which training
  refuses. Every eval set is checked before training (`validate_info`;
  `prediction_width` equal to the model's outputs, or for `None` a whole
  number per label column); `Metric::eval` returns NaN on length mismatch.
  `Metric::name` (and `EvalMetric::name`, equal to it) is XGBoost's
  `evals_result` key, suffix included (`ndcg@5`, `pre@3`,
  `tweedie-nloglik@1.5`), so parity compares names. Every `EvalMetric`
  carries its own parameters; only `from_xgboost` fills them from the
  flat keys XGBoost's metrics borrow (`huber_slope`, the alpha lists, the
  AFT noise, the `dist:*` family), and `Loss::default_metric` gives the
  loss's own (AFT's at scale 1, as XGBoost). `mlogloss`/`merror` read the
  class count from the model's outputs. A `Trainer::custom_metric` is
  evaluated after the configured or default metrics, as XGBoost's
  `xgb.train` does.
- **Refusals:** unsupported parameters or combinations error, never get
  ignored. Checks live in the parameter structs' constructors (ranges),
  `TrainingParams::validate` (static combinations), the builder (a
  setter value its field cannot hold, e.g. `max_depth(0)`, is reported by
  `build()` under its key; limits are `Option<NonZeroUsize>`, never a `0`
  sentinel),
  `TrainingParams::from_xgboost` (keys and value spellings; an objective
  parameter key that neither the objective nor a listed metric reads is
  refused, `OBJECTIVE_KEYS` in `config/xgboost.rs`, as is a dependent key
  without its switch, e.g. `rate_drop` without `booster=dart`),
  `validate_request` in `training/train.rs` (data-dependent),
  `validate_boulevard_request` and `validate_ebm_request` there too,
  `training/multi_output.rs::validate`, `training/continuation.rs`,
  `EvalMetric::from_xgboost` (metric names and suffixes), `training/budget.rs`,
  and `training/online.rs::check_supported`; SGLB and model shrinkage in
  `TrainingParams::validate_sglb` (static) and `training/sglb.rs::Sglb::resolve`
  (posterior sampling's row count).
  Budget mode and refresh compare params against defaults plus an
  allow-list (`TrainingParams::refuse_changes_from`, over `changed_keys`,
  which destructures every field), so any new field is refused there
  automatically.
- **One objective:** `TrainingParams::objective` is the only source of
  the trained loss (`TrainingParams::loss`); a custom loss is
  `Objective::Custom`, so every property (base-score domain, default
  metric, default `max_delta_step`, adaptive leaves, output count) comes
  from the loss being trained. A custom loss may not take a built-in
  objective's name: the model records it by name (`ModelObjective::name`,
  no `built_in` objective).
- **Python:** `python/` uses only the crate's public API. The public
  Python API is pure Python; the extension is private, fully stubbed
  (`_hessboost.pyi`, which ty checks callers against; nothing checks it
  against the built module, so change both together), `unsafe`-free
  (`forbid`), declares `gil_used = false`, keeps
  every class `frozen`, and releases the GIL around matrix construction,
  training, prediction, and model encode/decode. Parameter mappings go
  through `TrainingParams::from_xgboost`, the crate's one XGBoost
  boundary (also used by `tests/parity.rs` and the `train` fuzz target),
  so unknown keys are refused. `train(obj=...)` trains
  `Objective::Custom` (the mapping may not set `objective`; its
  `num_class` is the custom loss's output count, XGBoost's convention).
  `train`
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
  `objective::distributional`, `conformal`, `model::uncertainty`,
  `inference`, `ebm`, `diffusion`, `training::online`).
- Implementation modules are crate-private; benches and parity tests reach
  internals through `#[doc(hidden)] pub mod internals` in `lib.rs`, which
  is not public API.
- `#[non_exhaustive]` on every public enum, struct with public fields, and
  unit-struct built-in that could grow; users build them via `Default`, a
  builder, or a constructor. Only closed sets stay exhaustive (`Monotone`,
  `GradPair`, `Dist`'s variants).

Easy-to-miss requirements: multiclass needs `Multiclass::new(k)`; ranking needs
`.with_group_sizes`; `survival:aft` needs `.with_label_bounds`;
`survival:cox` reads non-positive labels as right-censored;
losses randomized per round (XE-NDCG) take the round from
`Loss::gradient_info_at`, so every training loop calls it;
`Loss::split_gradient` serves vector-leaf trees only, not with
monotone constraints. `booster = boulevard` is squared error only, refuses
nonlinear-leaf options, label-dependent row sampling (gradient-based,
class-balanced), weights, base margins, early stopping, continuation, and
online updates, and needs `eta = 1` with `num_parallel_tree > 1` (BRAT-P); its
leaves carry the final `1/B` scale, so exports and SHAP see a plain gbtree
ensemble, and slices drop the `BoulevardInfo`. `booster = ebm` counts every
tree as an iteration (`num_boost_round` counts EBM rounds, one tree per term
each, and caps each stage under `ebm_early_stopping_rounds`, which stops
every bag on its held-out rows), needs one output, refuses eval sets,
`Trainer::early_stopping_rounds`, continuation, online updates, column sampling,
interaction constraints, forests, feature weights, and base margins, draws
each classic tree's rows from its outer bag (by class under balanced
bagging, by query under query bagging), and calls `on_round` after every round of both stages; `ebm_boulevard` adds
Boulevard's refusals plus outer bags, early stopping, and `base_score`, and
its loaders check the stage-contiguous round-robin tree layout. Slices and exports drop the `EbmInfo`. Linear-leaf models predict through `tree::linear`;
XGBoost export, SHAP, and compact refuse them.

## When changing behavior

In the same change, update the touched items' rustdoc, the README's
feature lists and caveats, `lib.rs` "What's here" and "Not implemented",
this file, and affected examples. New options need a `TrainingParams`
field, builder setter, validation, and their key in `config/xgboost.rs`
(`flat_params!`, `into_params`, `to_xgboost`, `changed_keys`; each
destructures the struct, so a missing one does not compile). A new
objective is an `Objective` variant (the compiler then asks for it in
every property match of `objective/spec.rs`, `objective_to_json`, ...),
and a new objective parameter a field of its parameter struct plus its
flat key (`ObjectiveParts`, `OBJECTIVE_KEYS`) and stored member.

## Releases

Bump releases with `./release.py bump major|minor|patch` (or an explicit
SemVer such as `1.2.3-rc.1`). This creates a release branch, updates the crate
and Python versions and lockfiles, refreshes current dependency snippets,
saves `tests/data/saved/<version>/`, pushes the branch, and opens a PR.
Review and merge that PR; then, on `main`, run `./release.py --dry-run` and
`./release.py` to create and push the annotated release tag. The tag runs
`.github/workflows/publish.yml`, which verifies the Rust tests and Python
wheels/sdist before publishing to crates.io and PyPI and creating a GitHub
release with a discussion and generated notes. The notes are seeded from PRs
since the last tag and grouped by `.github/release.yml`; rewrite them by hand
afterward.

Use `./release.py bump <part> --dry-run` to inspect the bump plan without
changes, `--yes` to skip confirmation, or `--no-pr` to push without opening
a PR.

One-time setup: configure PyPI's pending trusted publisher for owner
`brndnmtthws`, repository `hessboost`, workflow `publish.yml`, environment
`pypi`; create the GitHub `pypi` environment.
