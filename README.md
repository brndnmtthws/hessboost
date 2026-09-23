# hessboost

[![crates.io](https://img.shields.io/crates/v/hessboost.svg)](https://crates.io/crates/hessboost)
[![docs.rs](https://img.shields.io/docsrs/hessboost)](https://docs.rs/hessboost)
[![CI](https://github.com/brndnmtthws/hessboost/actions/workflows/ci.yml/badge.svg)](https://github.com/brndnmtthws/hessboost/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A faithful, fast, pure-Rust reimplementation of [XGBoost](https://github.com/dmlc/xgboost)
gradient boosting with no C/C++ dependency and no FFI.

`hessboost` re-implements XGBoost's algorithms from scratch in idiomatic
Rust. It includes the regularized second-order boosting objective, exact,
histogram, and approximate tree construction, the full objective and metric
catalog, monotone and interaction constraints, categorical splits, DART and
gblinear boosters, TreeSHAP, and XGBoost-format model interop (numeric and
categorical trees) with
multi-core (`rayon`) acceleration.

Objective, metric, and parameter names mirror XGBoost, so configurations
transfer directly.

> **Built with AI.** The implementation was generated with **Claude** (Anthropic's
> AI coding assistant) under human direction and review. It is **AI-generated
> code**: it is covered by unit, property, and doc tests plus CI-checked
> XGBoost 3.4.2 parity, but it may still contain bugs, subtle numerical errors, or
> wrong edge-case behavior. **Review and validate it for your own use case. It is
> provided as-is, without warranty** (see [LICENSE](LICENSE)). Issue reports and
> fixes are welcome.

> Using AI coding agents? See [`AGENTS.md`](AGENTS.md) for a task-oriented guide.

## Quick start

```rust
use hessboost::prelude::*;

fn main() -> Result<()> {
    // Dense features (row-major) + labels.
    let x: Vec<f32> = /* n_rows * n_cols values */ vec![0.0; 400];
    let y: Vec<f32> = vec![0.0; 100];

    let dtrain = DMatrix::from_dense(&x, 100, 4)?.with_labels(&y)?;

    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .tree_method(TreeMethod::Hist)   // fast histogram method
        .max_depth(6)
        .eta(0.1)
        .subsample(0.9)
        .lambda(1.0)
        .build()?;

    let model = train(&params, &dtrain, 200)?;
    let preds = model.predict(&dtrain)?;

    model.save_binary("model.bin")?;
    Ok(())
}
```

## Examples

Runnable, self-contained examples live in
[`examples/`](examples). Run any with
`cargo run --release --example <name>`:

| Example | Shows |
|---|---|
| `binary_classification` | `binary:logistic`, watched eval set, early stopping, AUC |
| `multiclass` | `multi:softprob`, per-class probabilities, `predict_class` |
| `ranking` | LambdaMART `rank:ndcg` over query groups |
| `shap` | `predict_contribs` and `predict_interactions` (TreeSHAP) |
| `model_io` | native binary / JSON and XGBoost-format model save & load |
| `custom_objective` | custom loss and custom eval-metric hooks |
| `constraints` | monotone + interaction constraints and categorical features |
| `train_regression` | end-to-end regression with feature importance |
| `pfn_boost` | boosting from a pretrained prior's logits via `base_margin` (PFN-Boost) |

### Boosting from a pretrained prior

PFN-Boost and LLM-Boost ([Jayawardhana et al., 2025](https://arxiv.org/abs/2502.02672))
seed a GBDT with the per-row logits of a pretrained transformer (TabPFN or an
LLM) so the trees learn the residual of that prior: the initial margin is
`s * score + C` instead of a constant, with the scale `s` tuned on validation
data (`s = 0` is plain boosting). In hessboost that margin is
`DMatrix::with_base_margin`, attached to the train, eval, **and** test
matrices: `base_margin` replaces the model intercept and is not saved with the
model, so predicting on a matrix without it falls back to the intercept. The
`pfn_boost` example runs the method against boosting from scratch across
training-set sizes with a stand-in prior, or on TabPFN logits exported from
Python (`cargo run --release --example pfn_boost -- <dir>`; see its docs).

## Feature status

**Implemented & tested**

- **Boosters:** `gbtree`, **`dart`** (tree dropout), and **`gblinear`** (linear
  model via coordinate descent).
- **Trees:** `tree_method = exact | hist | approx` (approx uses hessian-weighted
  per-round binning), `grow_policy = depthwise | lossguide`, histogram binning
  with the parent−child subtraction trick, sparsity-aware missing-value handling,
  row/column subsampling (`bytree`/`bylevel`/`bynode`).
- **Regularization:** `lambda`, `alpha`, `gamma`, `min_child_weight`,
  `max_delta_step`, `max_depth`, `max_leaves`, `max_bin`.
- **Objectives:** `reg:squarederror` (alias `reg:linear`), `reg:logistic`,
  `reg:pseudohubererror` (`huber_slope`), `binary:logistic`, `multi:softmax`,
  `multi:softprob`, `count:poisson`, `reg:gamma`, `reg:tweedie`
  (`tweedie_variance_power`), learning-to-rank (`rank:pairwise`, `rank:ndcg`,
  `rank:map`, LambdaMART with `lambdarank_num_pair_per_sample`), and a user
  **custom-objective hook**. Intercepts are estimated per output exactly as
  XGBoost 3.4.1 does (label mean, class log-frequencies, or a Newton step).
- **Metrics:** `rmse`, `mae`, `logloss`, `error`, `auc`, `aucpr`, `mlogloss`,
  `merror`, `poisson/gamma/tweedie-nloglik`, `ndcg`, `map` (with `@k`), and a
  **custom-metric hook**. Default metrics follow XGBoost (`ndcg@k`/`map@k` for
  ranking, `tweedie-nloglik@rho` for Tweedie), except `reg:pseudohubererror`,
  which reports `mae` because XGBoost's `mphe` is not implemented.
- **Constraints:** monotone constraints and **interaction constraints**,
  supported in **both** the `hist` and `exact` builders.
- **Modeling:** **native categorical splits** (hist and exact), per-instance
  `base_margin` (warm-start), **TreeSHAP** contributions (`predict_contribs`) and
  **interaction values** (`predict_interactions`), early stopping, feature
  importance (weight / gain / cover / totals), leaf-index and margin prediction.
- **Ecosystem:** libsvm & CSV loaders, native binary + JSON model I/O,
  **XGBoost-format JSON model import/export** for `gbtree`/DART ensembles
  with numeric and categorical splits,
  k-fold cross-validation,
  multi-core histogram construction, and runtime-detected SIMD kernels:
  **AArch64 NEON** for objective and metric kernels, prediction transforms,
  multiclass operations, and histogram split evaluation. **x86-64 AVX2/FMA**
  for objective gradients, prediction transforms, and histogram split
  evaluation.

**Not implemented:** the UBJSON (binary) XGBoost model format, a GPU backend,
distributed or external-memory training, and Python/CLI/C-ABI wrappers.

## Performance

AArch64 builds use runtime-detected NEON kernels for objective gradients,
probability transforms, and metric reductions. x86-64 builds use AVX2+FMA for
exponential/sigmoid transforms, logistic and short-softmax gradients, and SSE2
for quantile bin search. Scalar fallbacks cover other CPUs, short inputs, and
values outside the approximation ranges. Split search is scalar and follows
XGBoost's `f32` gain arithmetic exactly. Histogram training parallelizes data
preparation and independent nodes, scales histogram tasks to node size, and
reuses training-row partitions when that reduces prediction work. Leaves at
`max_depth` skip histograms and split searches.

### Compared with XGBoost

Measured on **Apple M3 Max** against [XGBoost 3.4.1](https://pypi.org/project/xgboost/3.4.1/),
the latest stable PyPI release checked on **2026-09-14 UTC**. Both engines use the
same dense `f32` data and CPU `hist` parameters: 100 boosting rounds, depth 6,
256 bins, `eta=0.1`, and `lambda=1`. Times include fresh training-matrix
preparation and training, and report the median of six fits after warmup.

| Workload | Threads | hessboost | XGBoost 3.4.1 |
|---|---:|---:|---:|
| Regression, 100k × 30 | 1 | 0.458 s | 1.054 s |
| Regression, 100k × 30 | 4 | 0.201 s | 0.366 s |
| Regression, 100k × 30 | 16 | 0.264 s | 0.362 s |
| Regression, 50k × 128 | 1 | 1.153 s | 3.200 s |
| Regression, 50k × 128 | 4 | 0.452 s | 0.994 s |
| Regression, 50k × 128 | 16 | 0.428 s | 0.669 s |
| Binary, 100k × 30 | 1 | 0.459 s | 1.044 s |
| Binary, 100k × 30 | 4 | 0.197 s | 0.366 s |
| Binary, 100k × 30 | 16 | 0.256 s | 0.361 s |
| 4-class, 50k × 30 | 1 | 1.094 s | 2.506 s |
| 4-class, 50k × 30 | 4 | 0.523 s | 0.981 s |
| 4-class, 50k × 30 | 16 | 0.746 s | 1.219 s |

hessboost has lower median fit time in all 12 configurations in this run.
Single-thread speedups are 2.28× to 2.77×, four-thread speedups are 1.82× to
2.20×, and sixteen-thread speedups are 1.37× to 1.63×.

See [Performance](docs/performance.md) for held-out quality, workload
definitions, kernel benchmarks, and reproduction commands.

## Testing & parity

```sh
cargo test                          # unit + integration tests
cargo clippy --all-targets -- -D warnings
```

Numerical parity with **XGBoost 3.4.2** is checked in CI by a fixture harness
(`scripts/gen_fixtures.py`, `tests/parity.rs`,
`scripts/check_exports.py`). Each case is checked three ways: **train parity**
(same data and parameters, compare predictions), **import parity**
(`from_xgboost_json` on the XGBoost model: predictions, margins, SHAP
contributions) and **export parity** (`to_xgboost_json` reloaded by XGBoost).
`exact`-tier cases — including every objective and deterministic tree method,
plus categorical splits — must agree pointwise (1e-4 / 1e-5). `quality` cases
(row/column subsampling and DART) use an RMSE band because their RNG streams
differ. The `train-only` gblinear case is pointwise; gblinear XGBoost-JSON
import/export remains unsupported and is asserted explicitly. Histogram and
approximate quantile cuts are compared bit-for-bit.

```sh
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/gen_fixtures.py
cargo test --test parity --release -- --ignored --nocapture
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/check_exports.py
```

XGBoost 3.4.2 ships only as a source tarball (its release notes call it
identical to 3.4.1 apart from Python 3.11 support), so uv builds it with CMake
and a C++ compiler on first use.

See `scripts/README.md` for the case matrix and tolerances.

## License and attribution

Licensed under the **Apache License, Version 2.0**; see [`LICENSE`](LICENSE).
Copyright 2026 Brenden Matthews.

hessboost is a fork of
[sequoia-boost](https://github.com/pgarrett-scripps/sequoia-boost)
(Copyright 2026 Patrick Garrett, Apache-2.0).

It is an independent reimplementation of
[XGBoost](https://github.com/dmlc/xgboost) (Copyright the XGBoost Contributors,
Apache-2.0) built from XGBoost's public descriptions and papers; it contains no
XGBoost source code. "XGBoost" is used descriptively, for algorithmic lineage and
result compatibility. This project is not affiliated with or endorsed by the
XGBoost project.
