# Development scripts

## Parity fixtures

`gen_fixtures.py` trains **real XGBoost 3.4.2** (single thread) on deterministic
synthetic datasets, one case per supported feature (tree methods, missing values,
constraints, every objective — including `reg:logistic` and the `reg:linear`
alias — sample weights, ranking groups, gblinear, DART, intercept estimation,
and per-round metric oracles for `rmsle`, `mape`, `mphe`, `pre`/`pre@k` and
each objective's default metric; 50 cases in total), and writes each case to
`../fixtures/<name>.json`: data,
the exact `xgb.train` parameter dict, XGBoost's test-set predictions (transformed,
raw margin, SHAP contributions on the first 50 rows) and the saved model JSON,
plus the model's UBJSON encoding (`save_raw("ubj")`) as `../fixtures/<name>.ubj`.
It also writes `../fixtures/cuts/<name>.json`, XGBoost's `hist` and `approx`
quantile cuts (`DMatrix.get_quantile_cut`) for a set of matrices.

`tests/parity.rs` runs a three-way check per case:

1. **Train parity** - train on the fixture data, compare `predict(x_test)` with
   XGBoost's predictions.
2. **Import parity** - `BoostedModel::from_xgboost_json` on the embedded model,
   compare predictions, raw margins, and SHAP contributions;
   `from_xgboost_ubjson` on the `.ubj` sidecar must yield the identical model
   (column `ubj`).
3. **Export parity** - write `to_xgboost_json` (`<name>.model.json`),
   `to_xgboost_ubjson` (`<name>.model.ubj`) and hessboost's predictions to
   `../fixtures/exports/`; `check_exports.py` reloads each model in XGBoost and
   compares predictions, and for UBJSON also requires every array to use the
   same container form (typed element marker or generic) as XGBoost's own
   `save_raw("ubj")` of the loaded model.

`quantile_cuts_match_xgboost` compares `HistCuts::from_dmatrix` bit-for-bit with
the cut oracles.

Cases are tiered. `exact` cases are pointwise: max |delta| within `tol.train`
(1e-4; 1e-5 for probabilities), `tol.import` (1e-5) and `tol.contribs` (1e-4).
`quality` cases are RNG-driven (`subsample`, `colsample_bytree`, DART); training
uses a regression RMSE <= 1.08x XGBoost band while import/export remain
pointwise. The `train-only` gblinear case validates training pointwise; its
unsupported XGBoost-JSON import/export path is visibly reported as
`n/a`/`skipped` and required to return `ModelFormat` on import. Unknown XGBoost
parameters fail the test.

Optional fixture fields extend the schema for metadata beyond plain labels:

- `label_lower_bound` / `label_upper_bound` (training rows) and
  `test_label_lower_bound` / `test_label_upper_bound` (test rows): survival
  label bounds, attached with `DMatrix::with_label_bounds`. JSON has no
  infinity, so `+inf`/`-inf` are the strings `"inf"`/`"-inf"`; bounds are never
  NaN. A case whose objective reads the bounds only (`survival:aft`) has empty
  `y_train`/`y_test`, and no labels are attached.
- `test_weights`: per-row test-set weights (constant within a query group;
  XGBoost receives one weight per group for ranking cases).
- `xgb_evals`: `{metric: [value per round]}`, XGBoost's `evals_result()` on the
  labeled test set (labels, bounds, groups, `test_weights`) for cases built
  with the `evals` option. The metrics are the params' `eval_metric` list, or
  the objective's default metric when it is absent. The Rust side trains with
  `train_with_eval` on the same set and requires the same metric names and,
  every round, `|hessboost - xgboost| <= tol.evals * max(1, |xgboost|)`
  (`tol.evals` = 1e-5; column `evals`, `-` for cases without oracles).

```sh
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/gen_fixtures.py
cargo test --test parity --release -- --ignored --nocapture
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/check_exports.py
```

Fixtures are not checked in; CI regenerates them (`.github/workflows/ci.yml`,
job `parity`). The generator refuses any XGBoost version other than 3.4.2, pinned in
`requirements-xgboost.txt` (a source build: 3.4.2 has no PyPI wheel).

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
  --output /tmp/hessboost-comparison \
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
`QuantileDMatrix` with CPU `hist`. hessboost builds its `DMatrix` and performs
binning during training.

The default suite covers regression, wide regression, binary classification,
and four-class classification, with 100 boosting rounds at 1, 4, and 16
threads. Each comparison runs in XGBoost/hessboost/hessboost/XGBoost order. Every
batch discards one warmup fit and records three fits. The report uses the
median of all six measurements for each engine. Held-out RMSE or log loss
checks model quality alongside timing.

```sh
cargo build --release --example bench_compare
uv run --with xgboost==3.4.1 --with numpy==2.5.2 python scripts/bench_xgb.py \
  --hessboost target/release/examples/bench_compare \
  --output /tmp/hessboost-xgb --threads 1 4 16
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
