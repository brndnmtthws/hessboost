# Performance

hessboost combines runtime-detected SIMD numerical kernels — NEON on
AArch64; AVX2+FMA (exponential and sigmoid transforms, logistic and
short-softmax gradients) and SSE2 (quantile bin search) on x86-64 — with
parallel histogram training. Split search is scalar. Data preparation and independent
depthwise nodes share the worker pool; histogram task sizes follow node size,
the two children of a large node are evaluated concurrently, and partial
histograms are reduced in parallel by bin range. Training reuses row partitions
where they reduce prediction work, and terminal leaves skip histogram and split
searches. Prediction compares monotone integer keys in a branch-free lockstep
walk. Scalar Rust handles other architectures and inputs outside the vector
paths. Split choices, histogram sums, and prediction results match the scalar
path exactly; the transcendental kernels stay within a few f32 ULPs of the
scalar library functions. This guide compares CPU training with XGBoost and
measures the numerical and tree-building optimizations within hessboost.

## XGBoost comparison

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
Single-thread speedups range from 2.28× on binary classification to 2.77× on
wide regression; 30-feature regression reaches 2.30× and multiclass 2.29×. At
four threads, speedups range from 1.82× to 2.20×, and at sixteen threads from
1.37× to 1.63×.

![hessboost speedup over XGBoost 3.4.1 by workload and thread count](benchmarks/xgboost-speedup.svg)

![Median fit time by workload and thread count, log scale](benchmarks/xgboost-threads.svg)

The charts in this guide are rendered by
[`benchmarks/charts.gp`](benchmarks/charts.gp) from the table values recorded
in [`benchmarks/xgboost.dat`](benchmarks/xgboost.dat) and
[`benchmarks/optimization.dat`](benchmarks/optimization.dat). Regenerate
them with `gnuplot -c docs/benchmarks/charts.gp` after updating a data
file. The data files also carry the measurement provenance for each chart.

### CPU scheduling

XGBoost distributes histogram work across nodes and row blocks, and split
searches across nodes and features. Its updater also produces final row
positions. These are useful places to look when a faster numerical kernel has
little effect on total training time. See the upstream
[histogram builder](https://github.com/dmlc/xgboost/blob/v3.4.1/src/tree/hist/histogram.h),
[split evaluator](https://github.com/dmlc/xgboost/blob/v3.4.1/src/tree/hist/evaluate_splits.h),
and [hist updater](https://github.com/dmlc/xgboost/blob/v3.4.1/src/tree/updater_quantile_hist.cc).

hessboost's scheduler overlaps independent depthwise nodes and limits histogram
allocations by the rows available. Data preparation and margin updates also
share the worker pool. The measured benefit depends on tree shape, feature
count, sampling, and worker count; more workers do not guarantee a faster fit.

### Model quality

Both engines receive the same held-out rows, generated independently of the
training rows from the same distribution. The following scores are from the
single-thread fits; lower is better. These synthetic tasks measure comparable
fit quality under identical hyperparameters, not quality across all datasets.

| Workload | Held-out rows | Metric | hessboost | XGBoost 3.4.1 |
|---|---:|---|---:|---:|
| Regression, 100k × 30 | 20,000 | rmse | 0.060965 | 0.060965 |
| Regression, 50k × 128 | 10,000 | rmse | 0.064210 | 0.064210 |
| Binary, 100k × 30 | 20,000 | logloss | 0.516273 | 0.516273 |
| 4-class, 50k × 30 | 10,000 | mlogloss | 0.150027 | 0.150027 |

The XGBoost column is from the M3 Max run above. The hessboost column was
re-measured on 2026-09-23 on the AWS Neoverse-V3 host (aarch64 Linux 6.12)
with `bench_xgb.py --threads 1` against XGBoost 3.4.2 built from source
(`scripts/requirements-xgboost.txt`): XGBoost 3.4.2 reproduced the M3 3.4.1
scores to all six digits, and hessboost at release 1af4e21 and on the
roadmap branch matched XGBoost to within 1e-9 on every workload. The
hessboost scores previously listed here (0.060635, 0.063959, 0.515638,
0.150499) reproduce with neither build and have no recorded provenance.
hessboost's quality was not re-measured on the M3 Max (both hosts are
AArch64 and dispatch the same NEON kernels).

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
`QuantileDMatrix`, with quantile construction inside the timer. hessboost
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
XGBoost/hessboost/hessboost/XGBoost order. Each batch discards one warmup fit and
records three fits. The table uses the median of all six recorded fits per
engine. These numbers describe this CPU and these synthetic workloads; they do
not establish GPU or other-platform performance.

Reproduce from the repository root, using a new output directory:

```sh
cargo build --release --example bench_compare
uv run --with xgboost==3.4.1 --with numpy==2.5.2 python scripts/bench_xgb.py \
  --hessboost target/release/examples/bench_compare \
  --output /tmp/hessboost-xgb --threads 1 4 16
```

The quality re-check used the parity pin instead of the 3.4.1 wheel (a source
build of XGBoost 3.4.2; one measured fit per batch is enough for scores):

```sh
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/bench_xgb.py \
  --hessboost target/release/examples/bench_compare \
  --output /tmp/hessboost-xgb-quality --threads 1 --repeats 1
```

The [script documentation](../scripts/README.md#xgboost-comparison) describes
workload selection and a quick harness check.

## Optimization benchmarks

These measurements compare the optimized implementation with the scalar
baseline, using the same benchmark source and compiler.

Measured on **Apple M3 Max**, 16 physical cores,
macOS 26.6.2, with **Rust 1.98.1 / LLVM 22.1.8**, on 2026-09-14 UTC.
Both builds use `opt-level=3`, thin LTO, one codegen unit, and no `RUSTFLAGS`.
`RAYON_NUM_THREADS` is fixed per comparison. Compilation and tests finish before
timing begins; the benchmark executables run sequentially. Unrelated workstation
activity remains uncontrolled; alternating run order reduces timing bias but
does not eliminate it.

Each value is the mean of two Criterion run medians, collected in
baseline/optimized/optimized/baseline order. Each run requests 0.5 seconds of
warmup, 1 second of measurement, 20 samples, and 10,000 bootstrap resamples.
Criterion extends measurement time when
needed to collect them. **Less time** is `100 × (1 − optimized / baseline)`.
These are results for the specified machine and workloads, not a guarantee for
every dataset or AArch64 CPU.

### Full training

All cases train 50 depth-six trees on 50,000 rows × 20 features. Data creation
is outside the timer; training includes quantile preparation, gradients,
tree construction, and training-prediction updates.

| Workload | Threads | Baseline (ms) | Optimized (ms) | Less time |
|---|---:|---:|---:|---:|
| Regression, 256 bins | 1 | 398.027 | 110.772 | 72.2% |
| Regression, 256 bins | 4 | 381.738 | 54.300 | 85.8% |
| Regression, L1 = 1 | 1 | 392.963 | 107.278 | 72.7% |
| Regression, L1 = 1 | 4 | 379.340 | 53.641 | 85.9% |
| Regression, 16 bins | 1 | 231.349 | 82.752 | 64.2% |
| Regression, 16 bins | 4 | 220.740 | 43.935 | 80.1% |
| Binary classification | 1 | 408.616 | 114.892 | 71.9% |
| Binary classification | 4 | 396.398 | 57.560 | 85.5% |

![Full-training time cut vs scalar baseline](benchmarks/training-optimization.svg)

### Single histogram tree

Cuts, binned data, and gradients are prepared outside the timer. These cases
isolate tree construction, including the column sampler, histogram building,
row partitioning, and split evaluation.

| Workload | Threads | Baseline (ms) | Optimized (ms) | Less time |
|---|---:|---:|---:|---:|
| Depth 1 | 1 | 0.855 | 0.361 | 57.8% |
| Depth 1 | 4 | 0.618 | 0.220 | 64.3% |
| Depth 6 | 1 | 6.866 | 2.155 | 68.6% |
| Depth 6 | 4 | 6.726 | 0.908 | 86.5% |
| Depth 10 | 1 | 49.609 | 12.373 | 75.1% |
| Depth 10 | 4 | 49.458 | 3.926 | 92.1% |
| 128 features | 1 | 24.854 | 5.742 | 76.9% |
| 128 features | 4 | 24.886 | 2.310 | 90.7% |
| Missing values | 1 | 9.409 | 5.938 | 36.9% |
| Missing values | 4 | 9.287 | 2.438 | 73.7% |
| Monotone constraint | 1 | 9.193 | 4.096 | 55.4% |
| Monotone constraint | 4 | 9.008 | 1.475 | 83.6% |
| Loss-guide growth | 1 | 6.843 | 3.077 | 55.0% |
| Loss-guide growth | 4 | 6.680 | 2.924 | 56.2% |

Depth cases use 50,000 rows × 20 features. The wide case uses 10,000 rows × 128
features at depth six. Missing, monotone, and loss-guide cases use the depth-six
dataset; missing values occupy 2 of every 11 feature entries, the first feature
has an increasing constraint in the monotone case. All cases use 256 bins.
Loss-guide growth uses a 64-leaf limit; depthwise growth stops at its configured
depth.

![Histogram tree-build time cut vs scalar baseline](benchmarks/tree-optimization.svg)

### Numerical kernels

Pointwise cases process one million predictions. Multiclass cases contain
`floor(1,000,000 / classes)` rows, keeping the output count near one million.
Transforms include copying the input into the reusable output buffer. Metrics
include final normalization. These single-thread measurements use prepared
inputs, not tree training.

![Objective gradient time cut vs scalar baseline](benchmarks/gradient-optimization.svg)

![Prediction transform time cut vs scalar baseline](benchmarks/transform-optimization.svg)

![Metric time cut vs scalar baseline](benchmarks/metric-optimization.svg)

| Workload | Threads | Baseline (ms) | Optimized (ms) | Less time |
|---|---:|---:|---:|---:|
| Logistic gradient | 1 | 2.041 | 0.809 | 60.4% |
| Logistic gradient, weighted | 1 | 2.038 | 0.811 | 60.2% |
| Poisson gradient | 1 | 2.608 | 0.932 | 64.2% |
| Gamma gradient | 1 | 1.380 | 0.579 | 58.0% |
| Tweedie gradient | 1 | 2.630 | 1.056 | 59.8% |
| Softmax gradient, 2 classes | 1 | 4.210 | 0.833 | 80.2% |
| Softmax gradient, 3 classes | 1 | 5.890 | 0.810 | 86.2% |
| Softmax gradient, 4 classes | 1 | 3.574 | 0.901 | 74.8% |
| Softmax gradient, 8 classes | 1 | 2.675 | 1.309 | 51.1% |
| Softmax gradient, 32 classes | 1 | 2.176 | 1.012 | 53.5% |
| Softmax gradient, 128 classes | 1 | 2.188 | 0.948 | 56.6% |
| Sigmoid transform | 1 | 1.448 | 0.573 | 60.4% |
| Exponential transform | 1 | 1.280 | 0.473 | 63.1% |
| Softmax transform, 2 classes | 1 | 2.339 | 0.564 | 75.9% |
| Softmax transform, 3 classes | 1 | 2.378 | 0.543 | 77.2% |
| Softmax transform, 4 classes | 1 | 1.997 | 0.581 | 70.9% |
| Softmax transform, 8 classes | 1 | 1.896 | 0.712 | 62.4% |
| Softmax transform, 32 classes | 1 | 1.877 | 0.581 | 69.1% |
| Softmax transform, 128 classes | 1 | 1.927 | 0.589 | 69.4% |
| RMSE, weighted | 1 | 0.776 | 0.268 | 65.4% |
| MAE, weighted | 1 | 0.768 | 0.268 | 65.1% |
| Binary error, weighted | 1 | 1.338 | 0.204 | 84.8% |
| Log loss, weighted | 1 | 4.930 | 2.738 | 44.5% |
| Poisson NLL, weighted | 1 | 2.598 | 1.545 | 40.5% |
| Gamma NLL, weighted | 1 | 2.633 | 1.575 | 40.2% |
| Tweedie NLL, weighted | 1 | 13.797 | 4.521 | 67.2% |
| Multiclass log loss, 32 classes, weighted | 1 | 0.102 | 0.062 | 39.5% |
| Multiclass error, 32 classes, weighted | 1 | 1.616 | 0.244 | 84.9% |

The benchmark suite also covers unweighted metrics, additional class counts,
and histogram-accumulation controls. Histogram accumulation controls exercise
the scalar accumulation loop and task scheduling; tree construction also
measures split evaluation, parallel nodes, and avoiding unnecessary child work.

### Quantized-gradient training (opt-in)

`use_quantized_grad` (LightGBM's quantized training; not XGBoost behavior)
replaces each bin's two `f64` sums with one packed integer. Its width follows
the node's row count (32-bit up to `32767 / Q` rows, else 64-bit). Nodes too
large for 32 bits still accumulate runs of rows in a 32-bit scratch histogram,
which fits in L1, before adding them into the node's bins. Rows are read as
one `i32` instead of an 8-byte gradient pair. Split evaluation is unchanged:
it reads the dequantized sums, and each split pays one extra pass to
dequantize both child histograms.

Measured on **AWS Neoverse-V3** (192 cores, Linux 6.12) with **Rust 1.98.1**,
`opt-level=3`, thin LTO, one codegen unit, on 2026-09-23 UTC. Each value is the
Criterion median (2 s warm-up, 6 s measurement). The host also ran other
agents' builds, so treat differences under about 3% as noise. Q = 4 levels,
stochastic rounding, no leaf renewal. The full-precision column re-measures
the cases above on this machine in the same run, plus a new 1M × 50 case
(`large_depth8`).

| Workload | Threads | Full precision (ms) | Quantized (ms) | Speedup |
|---|---:|---:|---:|---:|
| Tree, depth 6, 50k × 20 | 1 | 5.935 | 5.982 | 0.99× |
| Tree, depth 6, 50k × 20 | 16 | 1.387 | 1.314 | 1.06× |
| Tree, depth 10, 50k × 20 | 1 | 64.10 | 65.70 | 0.98× |
| Tree, depth 10, 50k × 20 | 16 | 5.197 | 5.195 | 1.00× |
| Tree, 128 features, 10k rows | 1 | 26.35 | 26.77 | 0.98× |
| Tree, 128 features, 10k rows | 16 | 5.527 | 5.298 | 1.04× |
| Tree, missing values | 1 | 11.50 | 11.32 | 1.02× |
| Tree, missing values | 16 | 2.875 | 2.825 | 1.02× |
| Tree, depth 8, 1M × 50 | 1 | 274.9 | 181.7 | **1.51×** |
| Tree, depth 8, 1M × 50 | 16 | 29.81 | 16.09 | **1.85×** |
| Training, regression, 50 rounds | 1 | 308.8 | 318.5 | 0.97× |
| Training, regression, 50 rounds | 16 | 77.22 | 71.92 | 1.07× |
| Training, binary, 50 rounds | 1 | 309.4 | 323.7 | 0.96× |
| Training, binary, 50 rounds | 16 | 80.24 | 75.01 | 1.07× |

Quantization pays off only when histogram accumulation dominates the tree
build, as in the 1M-row case, where accumulation takes about half the
full-precision profile. On the 50k-row and 128-feature cases, the per-bin gain
evaluation takes most of the time. It is the same code in both modes. The
integer histograms save about a third of the smaller accumulation share, and
the per-tree quantization pass plus the per-split dequantization give most of
that back. Single-threaded training on 50k rows is therefore 3–4% *slower*,
and 16 threads gain about 7%. The integer loops are scalar: widths are chosen
per node, and serial and parallel builds agree bit for bit. There is no SIMD
path to keep in sync.

### Roadmap regression check

The roadmap branch adds many opt-in features on top of release 1af4e21, the
build the XGBoost comparison above measured. This check confirms that the
default training, prediction, and SHAP paths did not get slower along the way.

Measured on **AWS Neoverse-V3** (aarch64 Linux 6.12, 192 cores in two NUMA
nodes) with **Rust 1.98.1**, `opt-level=3`, thin LTO, one codegen unit, on
2026-09-23 UTC. Every run is pinned to the same 16 cores of one NUMA node
(`taskset -c 100-115`), with `RAYON_NUM_THREADS` fixed per row. Other agents'
builds shared the host, on other cores. Criterion rows come from
`scripts/compare_benchmarks.py` (release/roadmap/roadmap/release order,
0.5 s warm-up, 2 s measurement, 20 samples; the training groups keep their
own 10). Each value is the mean of the two run medians. Both executables
compile the same bench source. The release checkout got the depthwise
`predict_100k_x30_100trees_depth6` and the `shap_x20_100trees_depth6` cases,
neither of which it has, copied verbatim for this run. **Change** is
`roadmap / release − 1`. For this machine and these workloads only.

The first run found one real regression. `sum_rows`, the root gradient sum,
had been inlined into the roadmap's larger `HistTreeBuilder::build_inner`.
There LLVM kept the running sum in the caller's stack slot, so every row paid
a store-to-load round trip. Each tree took about 90 µs longer, a fixed serial
cost: depth-1 trees were 16% slower on one thread and 45% slower on 16.
Training got 4–5% slower on one thread and 7–10% slower on 16. Keeping
`sum_rows` out of line (`#[inline(never)]`) fixes it. The arithmetic and the
models are unchanged, and the parity suite stays bit-identical. The
"before fix" column is the roadmap at `ca37b7c`. The last column is the
roadmap after the fix, merged with `5779ed8`.

| Workload | Threads | 1af4e21 (ms) | Roadmap, before fix (ms) | Roadmap (ms) | Change |
|---|---:|---:|---:|---:|---:|
| Tree, depth 1 | 1 | 0.575 | 0.666 | 0.576 | +0.1% |
| Tree, depth 1 | 16 | 0.206 | 0.300 | 0.212 | +2.6% |
| Tree, depth 6 | 1 | 5.856 | 6.017 | 5.939 | +1.4% |
| Tree, depth 6 | 16 | 1.231 | 1.321 | 1.234 | +0.3% |
| Tree, depth 10 | 1 | 64.335 | 65.539 | 65.508 | +1.8% |
| Tree, depth 10 | 16 | 4.939 | 5.130 | 4.978 | +0.8% |
| Tree, 128 features | 1 | 26.304 | 27.047 | 26.954 | +2.5% |
| Tree, 128 features | 16 | 5.327 | 5.430 | 5.429 | +1.9% |
| Tree, missing values | 1 | 11.352 | 11.538 | 11.368 | +0.1% |
| Tree, missing values | 16 | 2.757 | 2.819 | 2.726 | -1.1% |
| Tree, monotone | 1 | 5.879 | 6.121 | 5.980 | +1.7% |
| Tree, monotone | 16 | 1.230 | 1.322 | 1.237 | +0.6% |
| Tree, loss-guide | 1 | 9.749 | 10.097 | 9.922 | +1.8% |
| Tree, loss-guide | 16 | 8.994 | 9.276 | 9.153 | +1.8% |
| Training, regression | 1 | 302.9 | 316.0 | 308.6 | +1.9% |
| Training, regression | 16 | 70.895 | 75.324 | 70.877 | -0.0% |
| Training, L1 = 1 | 1 | 303.3 | 314.0 | 308.3 | +1.6% |
| Training, L1 = 1 | 16 | 70.898 | 75.185 | 70.729 | -0.2% |
| Training, 16 bins | 1 | 130.3 | 137.0 | 131.6 | +1.0% |
| Training, 16 bins | 16 | 43.596 | 47.324 | 43.093 | -1.2% |
| Training, exact | 1 | 6486.3 | 6387.4 | 6380.0 | -1.6% |
| Training, exact | 16 | 6473.7 | 6301.4 | 6352.6 | -1.9% |
| Training, binary | 1 | 304.4 | 318.4 | 309.9 | +1.8% |
| Training, binary | 16 | 72.441 | 76.965 | 72.320 | -0.2% |
| Training, binary, scalar objective | 1 | 315.4 | 329.4 | 320.8 | +1.7% |
| Training, binary, scalar objective | 16 | 85.330 | 90.102 | 85.567 | +0.3% |
| Prediction, 100k rows | 1 | 136.2 | 136.3 | 132.8 | -2.5% |
| Prediction, 100k rows | 16 | 8.892 | 8.877 | 8.658 | -2.6% |
| SHAP contributions, 2,000 rows | 1 | 510.0 | 354.9 | 359.1 | -29.6% |
| SHAP contributions, 2,000 rows | 16 | 32.342 | 22.602 | 22.785 | -29.5% |
| SHAP interactions, 200 rows | 1 | 2008.8 | 77.360 | 77.394 | -96.1% |
| SHAP interactions, 200 rows | 16 | 130.8 | 5.268 | 5.272 | -96.0% |

End-to-end fits use `examples/bench_compare.rs` on the `bench_xgb.py`
datasets: 100 rounds, depth 6, 256 bins, fresh `DMatrix` per fit. Runs go in
release/roadmap/roadmap/release order, each with one warm-up fit and five
recorded fits. The table shows the median of the ten recorded fits per build.
Both builds reach identical held-out scores.

| Workload | Threads | 1af4e21 (s) | Roadmap (s) | Change |
|---|---:|---:|---:|---:|
| Regression, 100k × 30 | 1 | 1.091 | 1.107 | +1.5% |
| Regression, 100k × 30 | 16 | 0.213 | 0.214 | +0.6% |
| Binary, 100k × 30 | 1 | 1.033 | 1.049 | +1.5% |
| Binary, 100k × 30 | 16 | 0.206 | 0.208 | +1.1% |
| 4-class, 50k × 30 | 1 | 2.803 | 2.845 | +1.5% |
| 4-class, 50k × 30 | 16 | 0.722 | 0.727 | +0.7% |

What remains is under 3% everywhere. It sits in split evaluation: on one
thread, depth-6 trees are 1–2% slower, and so is training that consists of
them. The profile shares are the same in both builds (split evaluation about
1.55× histogram accumulation). The `histogram_build` cases also match the
release to within 0.5%. `HistTreeBuilder::evaluate` gained two never-taken
branches for the opt-in reuse penalties. Removing them made the depth-6 case
4% *slower*, not faster, so the remaining difference follows code layout, not
extra work. At 16 threads, training matches the release. Prediction is 2.5%
faster. QuadratureTreeSHAP cuts contribution time by 30% and interaction time
26-fold compared with the classic TreeSHAP the release shipped.

Reproduce with the commands in [Development scripts](../scripts/README.md),
building the release in a separate worktree:

```sh
git worktree add /tmp/hb-base 1af4e21   # add the predict/SHAP cases to its bench
python3 scripts/compare_benchmarks.py --baseline <release bench> --optimized <roadmap bench> \
  --output /tmp/hb-cmp-t1 --threads 1 \
  --filter '^(hist_tree_build/(depth1|depth6|depth10|wide128|missing|monotone|lossguide)|train_50k_x20_50rounds/(Hist|Hist_l1|Hist_16bins|Exact)|train_binary_50k_x20_50rounds/.*_(dispatch|reference)|predict_100k_x30_100trees_depth6/depthwise|shap_x20_100trees_depth6/.*)$'
```

## Implementation

The private `simd` module owns dispatch and numerical kernels. AArch64 checks
NEON support and x86-64 checks AVX2+FMA support once per process and caches
the result. Other architectures use scalar Rust; no target-specific build
flags are required. Dispatch checks the
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

Most kernels require at least 16 input elements. Softmax with 5–7 classes uses
the scalar path. The `f32` exponential uses range reduction and a degree-seven
polynomial for finite inputs in `[-80, 80]`; other inputs use `f32::exp`.
Softmax subtracts the row maximum and falls back for nonfinite rows or a margin
spread above 80. The exponential uses Estrin evaluation for gradients and
small-class softmax, and Horner evaluation for wide in-place softmax. Metric
logarithms and Tweedie exponentials use `f64` throughout.

Depthwise growth expands nodes and draws child feature samples in traversal
order, then partitions rows, builds histograms, and evaluates the independent
nodes in parallel. Candidate scans within a node retain their original order.
Loss-guide growth retains its priority-queue ordering. Histogram accumulation
limits the task count to one per 4,096 rows, capped at the worker count,
so a smaller node does not allocate a full histogram for every worker.

Leaves at `max_depth` need no histograms or split searches. With full row
sampling, training retains their final row partitions and adds the finalized
leaf values directly to cached training margins. Sampled training and
evaluation datasets update independent rows in parallel, skipping small inputs
where task overhead would dominate. Leaf statistics, monotone bounds, and
column-sampler draws are preserved.

Quantile cuts are sorted independently by feature, and rows are binned in
parallel chunks. Ordered collection preserves the cut layout, row order,
missing-value handling, categorical bins, and the choice of 16- or 32-bit bin
storage. These scheduling changes also work on other CPU architectures.

### Prediction

`BoostedModel` lazily derives a prediction layout of the ensemble on first use
and drops it whenever a tree is appended; it is never serialized. Every tree is
renumbered breadth-first into one 16-byte-node arena so that the two children
of a node are adjacent, and each numeric split is stored as a single ordered
compare that is false for missing values: splits whose missing values go left
store the next-lower threshold, splits whose missing values go right store
mirrored children, a negated threshold, and a sign mask applied to the feature
value. Leaves point at themselves. A traversal step is therefore a node load, a
feature load, an XOR, a compare, and an add, with no data-dependent branch.
Sixteen rows are walked in lockstep for a fixed number of levels (the tree
depth), and batches of fewer than sixteen rows walk sixteen trees in lockstep
instead. Rows are processed in 256-row blocks in parallel; each block stores
its full sixteen-row groups feature-major (`[group][feature][lane]`) so a
lane's value is an immediate offset from the group's feature base and the
kernel needs no per-lane address registers. Within a block the trees are
summed in order, so results match the sequential sum bit for bit. CSR rows and
dense matrices with a non-`NaN` missing sentinel are scattered straight into
that layout per block; matrices wider than 4,096 sparse columns use per-lookup
access. Categorical splits and trees deeper than 16 levels use an early-exit
walk. The kernel runs at roughly six instructions per cycle on a Neoverse V3
and is bound by instruction issue, not memory.

#### Symmetric trees

Trees in which every internal node of a level carries the same split (grown
with `grow_policy = symmetric`, or any imported tree of that shape) skip the
node walk for full sixteen-row groups. The layout records each level's split
as one `(slot, key)` compare and a `2^depth` table of leaf ids and values
indexed by the bit pattern of the level outcomes (root most significant);
collapsed subtrees fill every slot below them. A level is one contiguous
16-lane key load compared against a single threshold, which vectorizes and
carries no dependent load chain. The compares and the leaves reached are the
generic walk's, so margins and leaf indices are bit-identical (unit-tested);
tail rows, single-row batches, trees shallower than two levels or deeper than
16, and tables that would exceed four slots per leaf keep the generic walk.

`cargo bench --bench training -- predict_100k` predicts 100,000 × 30 rows with
100 depth-6 trees (`eta = 0.1`, other parameters default). Criterion medians
on a 192-core **Neoverse V3**, Rust 1.98.1, 2026-09-23, `RAYON_NUM_THREADS`
fixed per row. The middle column is the same symmetric model with the table
path disabled (a one-line local change), isolating the kernel:

| Threads | Depthwise model (ms) | Symmetric model, generic walk (ms) | Symmetric model, bit pattern (ms) | Speedup, same model |
|---:|---:|---:|---:|---:|
| 1 | 132.13 | 99.64 | 13.37 | 7.5× |
| 16 | 8.63 | 6.57 | 1.03 | 6.4× |
| 192 | 1.66 | 1.42 | 0.70 | 2.0× |

At full width the per-block row loading and scheduling, which both paths
share, dominate.

On a Neoverse V3 core these prediction changes cut prediction time by 10–20×
for dense, sparse, and multiclass batches relative to the per-node traversal.
That figure is an informal spot measurement from a separate machine. It is not
part of the recorded artifacts in this document.

SHAP values use XGBoost 3.4's QuadratureTreeSHAP. One recursive walk per tree
carries an 8-lane quadrature basis in `f32` and extracts each return edge's
contribution from its subtree's return, so contributions cost `O(L · D)` per
tree and row (`L` leaves, `D` depth) and interactions `O(L · D²)`. Classic
path-dependent TreeSHAP needed `O(L · D²)` for contributions and repeated a
conditioned walk per feature for interactions. Each tree's precomputed nodes
hold both child branch weights, only the tree's split features are cleared
and accumulated per tree, and rows are processed in parallel. Spot
measurements on the 192-core Neoverse V3 host (hist, 20 features, 100 trees
trained on 20,000 rows; mean of 3–5 calls; not part of the Criterion
artifacts):

| Workload | Threads | Classic TreeSHAP | QuadratureTreeSHAP | Speedup |
|---|---:|---:|---:|---:|
| contributions, depth 6, 2,000 rows | 192 | 4.2 ms | 3.8 ms | 1.1× |
| contributions, depth 10, 2,000 rows | 192 | 46.1 ms | 29.9 ms | 1.5× |
| interactions, depth 6, 200 rows | 192 | 19.7 ms | 1.9 ms | 10.6× |
| interactions, depth 10, 200 rows | 192 | 285 ms | 14.4 ms | 19.8× |
| contributions, depth 6, 2,000 rows | 1 | 486 ms | 385 ms | 1.3× |
| contributions, depth 10, 2,000 rows | 1 | 7.49 s | 4.47 s | 1.7× |
| interactions, depth 6, 200 rows | 1 | 1.90 s | 77 ms | 24.6× |
| interactions, depth 10, 200 rows | 1 | 29.6 s | 1.14 s | 26.1× |

## Numerical behavior and validation

Objective outputs remain `f32`; metrics and histogram statistics accumulate
in `f64`. SIMD reductions and polynomial evaluation can change rounding, so
cross-architecture predictions are not promised to be bit-identical. Repeated
training with the same inputs, parameters, seed, and execution configuration
remains deterministic.

SHAP values follow XGBoost 3.4.2's arithmetic: the quadrature rule is built in
`f64` and stored as `f32`, the recurrence and every accumulation are `f32` in
XGBoost's order (categorical children are walked in XGBoost's orientation),
and each tree's expected value is summed in `f64` and rounded once. XGBoost's
aarch64 builds contract `a * b + c` into fused multiply-adds while its x86_64
wheels do not, and hessboost mirrors this per target, so imported models
reproduce XGBoost's contributions and interaction values bit for bit on the
parity fixtures (checked on aarch64 Linux). The unfused arithmetic stays
within 2e-5 of the fused one on the same fixtures. The 8-point rule is exact
for paths with at most seven distinct features; longer paths are the same
quadrature approximation XGBoost computes.

The test suite compares kernels against scalar formulas, including short
inputs, vector tails, optional weights, saturation, NaNs, infinities, and
softmax ties. Gradient checks use `1e-6 * max(1, |reference|)`; metric checks use
relative tolerances between `1e-12` and `3e-12` for their finite test datasets.
Split tests check candidate order and the sequential gain epsilon. These are
test tolerances, not universal error bounds for arbitrary inputs.

## Reproduce the measurements

The benchmark definitions live in
[`benches/training.rs`](../benches/training.rs). To run the
suite on the current checkout:

```sh
RAYON_NUM_THREADS=1 cargo bench --bench training
```

The quantized-gradient rows use the `*_quantized` cases of
`hist_tree_build` and the `Hist_quantized` / `quantized` training cases.

To regenerate the charts in this guide from the `.dat` files after updating
the tables, run:

```sh
gnuplot -c docs/benchmarks/charts.gp
```

To compare two builds of the benchmark binary, use
`scripts/compare_benchmarks.py`; see
[Development scripts](../scripts/README.md) for it, the XGBoost quality
fixtures, and the XGBoost timing harness.
