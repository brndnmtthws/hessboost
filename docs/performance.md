# Performance

sequoia-boost combines AArch64 NEON numerical kernels with parallel histogram
training. Data preparation and independent depthwise nodes share the worker
pool; histogram task sizes follow node size. Training reuses row partitions
where they reduce prediction work, and terminal leaves skip histogram and split
searches. Scalar Rust handles other architectures and inputs outside the vector
paths. This guide compares CPU training with XGBoost and measures the numerical
and tree-building optimizations within sequoia-boost.

## XGBoost comparison

Measured on **Apple M3 Max** against [XGBoost 3.4.1](https://pypi.org/project/xgboost/3.4.1/),
the latest stable PyPI release checked on **2026-09-05 UTC**. Both engines use the
same dense `f32` data and CPU `hist` parameters: 100 boosting rounds, depth 6,
256 bins, `eta=0.1`, and `lambda=1`. Times include fresh training-matrix
preparation and training, and report the median of six fits after warmup.

| Workload | Threads | sequoia-boost | XGBoost 3.4.1 |
|---|---:|---:|---:|
| Regression, 100k × 30 | 1 | 0.947 s | 1.130 s |
| Regression, 100k × 30 | 4 | 0.392 s | 0.417 s |
| Regression, 100k × 30 | 16 | 0.421 s | 0.451 s |
| Regression, 50k × 128 | 1 | 2.268 s | 3.398 s |
| Regression, 50k × 128 | 4 | 0.841 s | 1.085 s |
| Regression, 50k × 128 | 16 | 0.796 s | 0.859 s |
| Binary, 100k × 30 | 1 | 0.917 s | 1.109 s |
| Binary, 100k × 30 | 4 | 0.379 s | 0.406 s |
| Binary, 100k × 30 | 16 | 0.410 s | 0.446 s |
| 4-class, 50k × 30 | 1 | 1.962 s | 2.661 s |
| 4-class, 50k × 30 | 4 | 0.851 s | 1.019 s |
| 4-class, 50k × 30 | 16 | 0.960 s | 1.246 s |

Sequoia has lower median fit time in all 12 configurations in this run.
Single-thread speedups range from 1.19× on 30-feature regression to 1.50× on
wide regression. At four threads, wide regression and multiclass reach 1.29×
and 1.20× respectively; sixteen-thread multiclass reaches 1.30×. The remaining
multithread differences are 6–9%. Treat those smaller differences as near
parity given the uncontrolled background activity on this workstation.

### CPU scheduling

XGBoost distributes histogram work across nodes and row blocks, and split
searches across nodes and features. Its updater also produces final row
positions. These are useful places to look when a faster numerical kernel has
little effect on total training time. See the upstream
[histogram builder](https://github.com/dmlc/xgboost/blob/v3.4.1/src/tree/hist/histogram.h),
[split evaluator](https://github.com/dmlc/xgboost/blob/v3.4.1/src/tree/hist/evaluate_splits.h),
and [hist updater](https://github.com/dmlc/xgboost/blob/v3.4.1/src/tree/updater_quantile_hist.cc).

Sequoia's scheduler overlaps independent depthwise nodes and limits histogram
allocations by the rows available. Data preparation and margin updates also
share the worker pool. The measured benefit depends on tree shape, feature
count, sampling, and worker count; more workers do not guarantee a faster fit.

### Model quality

Both engines receive the same held-out rows, generated independently of the
training rows from the same distribution. The following scores are from the
single-thread fits; lower is better. These synthetic tasks measure comparable
fit quality under identical hyperparameters, not quality across all datasets.

| Workload | Held-out rows | Metric | sequoia-boost | XGBoost 3.4.1 |
|---|---:|---|---:|---:|
| Regression, 100k × 30 | 20,000 | rmse | 0.060635 | 0.060965 |
| Regression, 50k × 128 | 10,000 | rmse | 0.063959 | 0.064210 |
| Binary, 100k × 30 | 20,000 | logloss | 0.515638 | 0.516273 |
| 4-class, 50k × 30 | 10,000 | mlogloss | 0.150499 | 0.150027 |

### Workloads and method

- Regression predicts `2*x0 - 3*x1² + 0.5*x2 + x3*x4` with Gaussian noise
  of standard deviation 0.05. The 128-feature case adds irrelevant features.
- Binary labels are Bernoulli draws with log-odds
  `4*(x0-0.5) - 3*(x1-0.5) + 2*(x2-0.5)`.
- Four-class labels select the largest of `3*xi - x((i+1) mod 4)` plus Gaussian
  noise of standard deviation 0.1. Each boosting round constructs four trees.

Features are uniform on `[0, 1)`. The NumPy generator and training seed are
1234. Held-out sets have one fifth as many rows as their training sets. Both
engines use depthwise growth, a fixed `base_score=0.5`, `alpha=0`, `gamma=0`,
`min_child_weight=1`, and full row/column sampling. No early stopping or eval
callbacks run inside the timer.

XGBoost uses its macOS ARM64 PyPI wheel with OpenMP enabled and
`QuantileDMatrix`, with quantile construction inside the timer. sequoia-boost
constructs a fresh `DMatrix` inside the timer and bins during training. Both
read identical binary data before timing. Test-matrix preparation, file I/O,
process startup, prediction, scoring, and model destruction are excluded.

The runtime is macOS 26.6.2, Rust 1.98.1 / LLVM 22.1.8, uv-managed Python
3.14.5, NumPy 2.5.2, and XGBoost 3.4.1. Rust uses the release profile
(`opt-level=3`, thin LTO, one codegen unit) without extra `RUSTFLAGS`.
XGBoost's native build reports Clang 15 and OpenMP support. Compilation finishes
before measurements; the two engines run sequentially. This is an interactive
workstation with unrelated CPU activity, so small differences should be treated
as near parity rather than an isolated-machine result.

For each workload and thread count, batches run in
XGBoost/sequoia/sequoia/XGBoost order. Each batch discards one warmup fit and
records three fits. The table uses the median of all six recorded fits per
engine. The [complete results](benchmarks/xgboost.json) include every sample,
minimum/maximum times, held-out scores, native build configuration, dataset and
source hashes, and the executable hash. These numbers describe this CPU and
these synthetic workloads; they do not establish GPU or other-platform
performance.

The result also includes the Cargo lockfile used for these measurements.

Reproduce from the repository root, using a new output directory:

```sh
cargo build --release --example bench_compare
uv run --with xgboost==3.4.1 --with numpy==2.5.2 python scripts/bench_xgb.py \
  --sequoia target/release/examples/bench_compare \
  --output /tmp/sequoia-xgb --threads 1 4 16
```

The [script documentation](../scripts/README.md#xgboost-comparison) describes
workload selection and a quick harness check.

## Optimization benchmarks

These measurements compare the optimized implementation with the scalar
baseline, using the same benchmark source and compiler.

Measured on **Apple M3 Max**, 16 physical cores,
macOS 26.6.2, with **Rust 1.98.1 / LLVM 22.1.8**, on 2026-09-05 UTC.
Both builds use `opt-level=3`, thin LTO, one codegen unit, and no `RUSTFLAGS`.
`RAYON_NUM_THREADS` is fixed per comparison. Compilation and tests finish before
timing begins; the benchmark executables run sequentially. Unrelated workstation
activity remains uncontrolled; alternating run order reduces timing bias but
does not eliminate it.

Each value is the mean of two Criterion run medians, collected in
baseline/optimized/optimized/baseline order. Each run requests 0.5 seconds of
warmup, 1 second of measurement, 20 samples, and 10,000 bootstrap resamples.
Full-training groups use 10 samples; Criterion extends measurement time when
needed to collect them. **Less time** is `100 × (1 − optimized / baseline)`.
These are results for the specified machine and workloads, not a guarantee for
every dataset or AArch64 CPU.

### Full training

All cases train 50 depth-six trees on 50,000 rows × 20 features. Data creation
is outside the timer; training includes quantile preparation, gradients,
tree construction, and training-prediction updates.

| Workload | Threads | Baseline (ms) | Optimized (ms) | Less time |
|---|---:|---:|---:|---:|
| Regression, 256 bins | 1 | 414.739 | 208.003 | 49.8% |
| Regression, 256 bins | 4 | 396.480 | 94.607 | 76.1% |
| Regression, L1 = 1 | 1 | 414.339 | 203.823 | 50.8% |
| Regression, L1 = 1 | 4 | 393.711 | 93.414 | 76.3% |
| Regression, 16 bins | 1 | 244.064 | 158.713 | 35.0% |
| Regression, 16 bins | 4 | 227.739 | 75.711 | 66.8% |
| Binary classification | 1 | 427.917 | 214.212 | 49.9% |
| Binary classification | 4 | 411.213 | 101.709 | 75.3% |

### Single histogram tree

Cuts, binned data, and gradients are prepared outside the timer. These cases
isolate tree construction, including the column sampler, histogram building,
row partitioning, and split evaluation.

| Workload | Threads | Baseline (ms) | Optimized (ms) | Less time |
|---|---:|---:|---:|---:|
| Depth 1 | 1 | 0.875 | 0.463 | 47.0% |
| Depth 1 | 4 | 0.645 | 0.189 | 70.7% |
| Depth 6 | 1 | 7.127 | 4.192 | 41.2% |
| Depth 6 | 4 | 7.013 | 1.703 | 75.7% |
| Depth 10 | 1 | 53.372 | 18.229 | 65.8% |
| Depth 10 | 4 | 53.444 | 5.493 | 89.7% |
| 128 features | 1 | 26.409 | 8.441 | 68.0% |
| 128 features | 4 | 26.555 | 3.404 | 87.2% |
| Missing values | 1 | 9.868 | 6.477 | 34.4% |
| Missing values | 4 | 9.572 | 2.710 | 71.7% |
| Monotone constraint | 1 | 9.649 | 6.263 | 35.1% |
| Monotone constraint | 4 | 9.462 | 2.252 | 76.2% |
| Loss-guide growth | 1 | 7.302 | 5.718 | 21.7% |
| Loss-guide growth | 4 | 6.947 | 4.992 | 28.1% |

Depth cases use 50,000 rows × 20 features. The wide case uses 10,000 rows × 128
features at depth six. Missing, monotone, and loss-guide cases use the depth-six
dataset; missing values occupy 2 of every 11 feature entries, the first feature
has an increasing constraint in the monotone case. All cases use 256 bins.
Loss-guide growth uses a 64-leaf limit; depthwise growth stops at its configured
depth.

### Numerical kernels

Pointwise cases process one million predictions. Multiclass cases contain
`floor(1,000,000 / classes)` rows, keeping the output count near one million.
Transforms include copying the input into the reusable output buffer. Metrics
include final normalization. These single-thread measurements use prepared
inputs, not tree training.

| Workload | Threads | Baseline (ms) | Optimized (ms) | Less time |
|---|---:|---:|---:|---:|
| Logistic gradient | 1 | 2.080 | 0.837 | 59.8% |
| Logistic gradient, weighted | 1 | 2.105 | 0.835 | 60.3% |
| Poisson gradient | 1 | 2.688 | 0.977 | 63.7% |
| Gamma gradient | 1 | 1.424 | 0.600 | 57.9% |
| Tweedie gradient | 1 | 2.713 | 1.102 | 59.4% |
| Softmax gradient, 2 classes | 1 | 4.301 | 0.867 | 79.8% |
| Softmax gradient, 3 classes | 1 | 6.124 | 0.841 | 86.3% |
| Softmax gradient, 4 classes | 1 | 3.633 | 0.966 | 73.4% |
| Softmax gradient, 8 classes | 1 | 2.792 | 1.353 | 51.5% |
| Softmax gradient, 32 classes | 1 | 2.324 | 1.051 | 54.8% |
| Softmax gradient, 128 classes | 1 | 2.293 | 0.993 | 56.7% |
| Sigmoid transform | 1 | 1.526 | 0.603 | 60.5% |
| Exponential transform | 1 | 1.319 | 0.514 | 61.0% |
| Softmax transform, 2 classes | 1 | 2.415 | 0.607 | 74.9% |
| Softmax transform, 3 classes | 1 | 2.464 | 0.570 | 76.9% |
| Softmax transform, 4 classes | 1 | 2.080 | 0.604 | 71.0% |
| Softmax transform, 8 classes | 1 | 1.979 | 0.745 | 62.4% |
| Softmax transform, 32 classes | 1 | 1.942 | 0.604 | 68.9% |
| Softmax transform, 128 classes | 1 | 2.005 | 0.624 | 68.9% |
| RMSE, weighted | 1 | 0.799 | 0.281 | 64.9% |
| MAE, weighted | 1 | 0.795 | 0.281 | 64.7% |
| Binary error, weighted | 1 | 1.397 | 0.216 | 84.5% |
| Log loss, weighted | 1 | 5.082 | 2.871 | 43.5% |
| Poisson NLL, weighted | 1 | 2.696 | 1.599 | 40.7% |
| Gamma NLL, weighted | 1 | 2.733 | 1.619 | 40.8% |
| Tweedie NLL, weighted | 1 | 14.220 | 4.670 | 67.2% |
| Multiclass log loss, 32 classes, weighted | 1 | 0.104 | 0.064 | 38.0% |
| Multiclass error, 32 classes, weighted | 1 | 1.655 | 0.254 | 84.7% |

The [complete results](benchmarks/performance.json) include all 73
single-thread cases and 11 four-thread cases, including unweighted metrics,
additional class counts, and histogram-accumulation controls. Histogram
accumulation controls exercise the scalar accumulation loop and task scheduling;
tree construction also measures split evaluation, parallel nodes, and avoiding
unnecessary child work. The
[compressed samples](benchmarks/performance-samples.json.gz) contain Criterion
estimates, confidence intervals, raw samples, logs, the shared benchmark source,
lockfile, and a patch that reconstructs the measured optimized source. The
result file records executable and source SHA-256 hashes.

## Implementation

The private `simd` module owns dispatch and numerical kernels. AArch64 checks
NEON support once per process and caches the result. Other architectures use
scalar Rust; no target-specific build flags are required. Dispatch checks the
slice lengths before entering an unsafe kernel, and vector loads and stores
stay within complete blocks. Scalar formulas are shared by whole-input
fallbacks, exceptional blocks, and tails.

| Operation | NEON path |
|---|---|
| Logistic, Poisson, Gamma, Tweedie gradients | Four `f32` predictions per block, with weighted and unweighted inputs |
| Sigmoid and exponential transforms | Four `f32` predictions per block |
| Softmax gradients and transforms, 2–4 classes | Four rows at a time using interleaved loads and stores |
| Softmax gradients and transforms, 8+ classes | Vector blocks within each row |
| RMSE, MAE, binary error, log loss, count metrics | `f64` reductions with optional weights |
| Multiclass log loss | Gathered label probabilities with `f64` logarithms |
| Multiclass error, 8+ classes | Vector row maxima |
| Histogram statistics and dense numeric split gains | Two `f64` lanes |

Most kernels require at least 16 input elements. Softmax with 5–7 classes uses
the scalar path. The `f32` exponential uses range reduction and a degree-seven
polynomial for finite inputs in `[-80, 80]`; other inputs use `f32::exp`.
Softmax subtracts the row maximum and falls back for nonfinite rows or a margin
spread above 80. The exponential uses Estrin evaluation for gradients and
small-class softmax, and Horner evaluation for wide in-place softmax. Metric
logarithms and Tweedie exponentials use `f64` throughout.

The histogram builder uses NEON to evaluate pairs of gains for dense numeric features without monotone constraints or a
nonzero `max_delta_step`. Prefix statistics and accepted-candidate comparisons
remain sequential, preserving split order, ties, and the gain epsilon. Bins
that cannot improve the best gain are rejected before extracting scalar lanes.
Missing-value and constrained split searches retain their scalar evaluation.

Depthwise growth expands nodes and draws child feature samples in traversal
order, then partitions rows, builds histograms, and evaluates the independent
nodes in parallel. Candidate scans within a node retain their original order.
Loss-guide growth retains its priority-queue ordering. Histogram accumulation
limits the task count to one per 4,096 rows, capped at the worker count,
so a smaller node does not allocate a full histogram for every worker.

Leaves at `max_depth` need no histograms or split searches. With full row
sampling and at most four workers, training retains their final row partitions
and adds the finalized leaf values directly to cached training margins. Larger
pools, sampled training, and evaluation datasets update independent rows in
parallel, skipping small inputs where task overhead would dominate. Leaf
statistics, monotone bounds, and column-sampler draws are preserved.

Quantile cuts are sorted independently by feature, and rows are binned in
parallel chunks. Ordered collection preserves the cut layout, row order,
missing-value handling, categorical bins, and the choice of 16- or 32-bit bin
storage. These scheduling changes also work on other CPU architectures.

## Numerical behavior and validation

Objective outputs remain `f32`; metrics and histogram statistics accumulate
in `f64`. SIMD reductions and polynomial evaluation can change rounding, so
cross-architecture predictions are not promised to be bit-identical. Repeated
training with the same inputs, parameters, seed, and execution configuration
remains deterministic.

The test suite compares kernels against scalar formulas, including short
inputs, vector tails, optional weights, saturation, NaNs, infinities, and
softmax ties. Gradient checks use `1e-6 * max(1, |reference|)`; metric checks use
relative tolerances between `1e-12` and `3e-12` for their finite test datasets.
Split tests check candidate order and the sequential gain epsilon. These are
test tolerances, not universal error bounds for arbitrary inputs.

Validation for the recorded build includes debug and release tests, Clippy,
XGBoost regression/binary/multiclass quality parity, a Rust 1.86 build, and an
x86_64 cross-check. A separate 108-case equivalence check covers dense, missing,
categorical, sampled, constrained, and multiclass trees. It verifies serialized models and
predictions with one, four, and sixteen threads; its source and result hashes are
included with the benchmark samples.

```sh
cargo test --workspace --locked
cargo test --workspace --locked --release
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo fmt --all --check
cargo test --locked --test parity -- --ignored
cargo +1.86.0 build --locked
cargo check --locked --all-targets --all-features --target x86_64-pc-windows-gnu
```

Generate the parity fixtures first using the
[fixture instructions](../scripts/README.md#parity-fixtures). The cross-check
requires the corresponding Rust target to be installed and checks compilation
only.

## Reproduce the measurements

The benchmark definitions live in
[`benches/training.rs`](../crates/sequoia-boost/benches/training.rs). To run the
suite on the current checkout:

```sh
RAYON_NUM_THREADS=1 cargo bench --locked -p sequoia-boost --bench training
```

For an exact source comparison, reconstruct both trees from the recorded
baseline revision and source archive. Run from the repository root:

```sh
export BENCH_WORKDIR="$(mktemp -d)"
uv run python - <<'PY'
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

root = Path(os.environ['BENCH_WORKDIR'])
record = json.loads(gzip.decompress(
    Path('docs/benchmarks/performance-samples.json.gz').read_bytes()))['source']
archive = subprocess.check_output(['git', 'archive', record['baseline_revision']])
for name in ['baseline', 'optimized']:
    tree = root / name
    tree.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(tree, filter='data')
    if name == 'optimized':
        subprocess.run(['git', 'apply', '--no-index', '-'], cwd=tree,
                       input=record['optimized_patch'].encode(), check=True)
    (tree / 'Cargo.lock').write_text(record['lockfile'])
    (tree / 'crates/sequoia-boost/benches/training.rs').write_text(
        record['benchmark_source'])
    build = subprocess.check_output([
        'cargo', '+1.98.1', 'bench', '--locked', '--no-run',
        '-p', 'sequoia-boost', '--bench', 'training', '--message-format=json',
    ], cwd=tree, env=dict(os.environ,
                         CARGO_TARGET_DIR=str(root / f'{name}-target')))
    artifacts = [json.loads(line) for line in build.splitlines()]
    executable = next(Path(item['executable']) for item in artifacts
                      if item.get('reason') == 'compiler-artifact'
                      and item['target']['name'] == 'training'
                      and item.get('executable'))
    (root / f'{name}-bench').symlink_to(executable)
PY

uv run python scripts/compare_benchmarks.py \
  --baseline "$BENCH_WORKDIR/baseline-bench" \
  --optimized "$BENCH_WORKDIR/optimized-bench" \
  --output "$BENCH_WORKDIR/results-t1" --threads 1 \
  --warmup 0.5 --measurement 1 --samples 20 --resamples 10000 \
  --filter 'objective_gradient|prediction_transform/.*automatic|prediction_transform_multiclass|pointwise_metric|log_metric|multiclass_metric|hist_tree_build|histogram_build|train_50k_x20_50rounds/Hist|train_binary_50k_x20_50rounds/automatic'

uv run python scripts/compare_benchmarks.py \
  --baseline "$BENCH_WORKDIR/baseline-bench" \
  --optimized "$BENCH_WORKDIR/optimized-bench" \
  --output "$BENCH_WORKDIR/results-t4" --threads 4 \
  --warmup 0.5 --measurement 1 --samples 20 --resamples 10000 \
  --filter 'hist_tree_build|train_50k_x20_50rounds/Hist|train_binary_50k_x20_50rounds/automatic'
```

The reconstruction snippet requires Python 3.12+ and Rust 1.98.1. Keep
`RUSTFLAGS`, `CARGO_ENCODED_RUSTFLAGS`, `CARGO_BUILD_TARGET`, and Cargo profile
overrides unset to match the recorded build. The comparison runner itself uses
only the Python standard library. It creates separate Criterion directories
for each run and refuses to overwrite an existing output directory.

For XGBoost quality fixtures and the separate XGBoost timing harness, see
[Development scripts](../scripts/README.md).
