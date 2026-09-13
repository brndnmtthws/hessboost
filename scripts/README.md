# Development scripts

## Parity fixtures

`gen_fixtures.py` trains **real XGBoost** on standardized synthetic datasets and
writes disjoint training and test datasets, parameters, and XGBoost test-set
predictions to `../fixtures/*.json`. The Rust test
`crates/sequoia-boost/tests/parity.rs` trains on the same training rows and
asserts that held-out RMSE or accuracy remains close to XGBoost.

```sh
uv run --with xgboost --with numpy python scripts/gen_fixtures.py
cargo test -p sequoia-boost --test parity -- --ignored
```

Fixtures are intentionally not checked in and are regenerated in CI. The
datasets use `tree_method=hist` with a fixed `max_bin`. Pointwise prediction
differences are reported for diagnosis, while held-out model quality determines
pass or failure because independent histogram implementations need not select
identical split points.

## Criterion comparisons

`compare_benchmarks.py` runs two compiled Criterion executables in
baseline/optimized/optimized/baseline order with a fixed Rayon thread count.
Each run gets its own result directory. The script writes `comparison.json`
with medians and confidence intervals, and `samples.json.gz` with raw Criterion
samples and console output. It requires only the Python standard library.

```sh
python3 scripts/compare_benchmarks.py \
  --baseline /path/to/baseline-training-bench \
  --optimized /path/to/optimized-training-bench \
  --output /tmp/sequoia-comparison \
  --threads 1 --filter 'hist_tree_build|train_50k_x20_50rounds/Hist'
```

The output directory must not already exist. Build both executables with the
same benchmark source, lockfile, compiler, and release settings before running
the comparison. See [Performance](../docs/performance.md) for recorded results,
workload definitions, and complete reproduction commands.

## XGBoost comparison

`bench_xgb.py` generates shared training and held-out datasets, then benchmarks
XGBoost and the compiled `bench_compare` Rust example. Both engines read the
same little-endian `f32` bytes. Each timed fit constructs a fresh training
matrix and trains the model. File I/O, test-data preparation, evaluation, model
destruction, and process startup are outside the timer. XGBoost uses
`QuantileDMatrix` with CPU `hist`. sequoia-boost builds its `DMatrix` and performs
binning during training.

The default suite covers regression, wide regression, binary classification,
and four-class classification, with 100 boosting rounds at 1, 4, and 16
threads. Each comparison runs in XGBoost/sequoia/sequoia/XGBoost order. Every
batch discards one warmup fit and records three fits. The report uses the
median of all six measurements for each engine. Held-out RMSE or log loss
checks model quality alongside timing.

```sh
cargo build --release --example bench_compare
uv run --with xgboost==3.4.1 --with numpy==2.5.2 python scripts/bench_xgb.py \
  --sequoia target/release/examples/bench_compare \
  --output /tmp/sequoia-xgb --threads 1 4 16
```

Use a new output directory for each comparison. It contains the shared binary
datasets and `comparison.json`, including every timing sample, held-out score,
training parameters, native XGBoost build information, package versions, and
source, executable, and dataset hashes. Build before timing and avoid running
other benchmarks or compiler jobs concurrently.

For a quick harness check, add `--rows 512 --rounds 3 --repeats 1`. Use
`--workloads regression` to select one dataset or `--threads 1` for a
single-thread comparison. Remove the package version constraints to benchmark
the latest releases available through `uv`. The output records the versions
actually used.

See [Performance](../docs/performance.md#xgboost-comparison) for the recorded
comparison and workload definitions.
