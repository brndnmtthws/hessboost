# Development scripts

## Existing-hook synthetic experiments

Run the diagnostics without Python or external datasets:

```sh
RAYON_NUM_THREADS=1 cargo run --release --example research_hooks -- small 3
RAYON_NUM_THREADS=1 cargo run --release --example research_hooks -- large 3
cargo run --release --example research_hooks -- calibration 3
```

The training comparisons use a seeded uniform random quantile coordinate per
round (8/32/128 outputs), fold-local 1D k-means numeric codes against 16/256
quantile bins, and frozen warmup-gain column sampling against all features
and uniform sampling. Both sampled variants retain 50% and continue the same
four-round all-feature warmup. Sampling uses dense 128-column (`small`) or
500-column (`large`) data; other cases use 32 columns and 2% missingness.
Prepared cuts are shared across sampling continuations, with initial binning
and warmup costs charged once to each standalone fit. This is not EMA
sampling, and projected Newton leaves are not exact-leaf PGB.

Each variant independently chooses its best round by validation early stopping
with patience 50. The safety cap is 1,024 rounds (`small`) or 4,096 (`large`);
a capped search with unexpired patience is flagged and cannot qualify as a
quality match. Timing and test labels never choose rounds. Three separate
fixed-count refits then time precisely the chosen number of rounds without
validation, reporting all samples and their median. Variant order rotates
across seeds. Built-in and all-output custom quantile controls must have the
same selected count, margin hash, and transformed-prediction hash.

JSONL `benchmark` and `comparison` rows report validation/test errors, stopping
status, fit counts, hashes, tuning time, initial cut construction, growth
samples, and end-to-end cold-fit time. Quality-matched means both searches
finished their patience and the variant's validation error is within 1% of the
baseline's; it is not a claim that test quality is identical. Generation and
evaluation are excluded from fixed-count fitting; k-means encoding of all three
splits is included, while ordinary matrix construction is excluded.

For controlled timing, finish compilation, linting and tests first, then pin
the executable to an otherwise idle allowed CPU with `taskset -c CPU`.
Pinning does not reserve a CPU or isolate memory bandwidth from other users.
Do not infer speedups on a busy shared host. The harness records descriptive
times but does not certify host exclusivity or production benefit.

The `calibration` mode uses identical shifted score streams and delays 0/5
for fixed thresholds, scalar/linear tracking, and rolling empirical quantiles
over the latest 50/200 observed scores. Rolling thresholds use the
`ceil((W + 1) * (1 - alpha))` order statistic. It reports coverage, width and
score pinball loss; all settings are preselected. These synthetic comparisons
do not establish market quality or conditional coverage.

## Real hourly forecasting comparison

Prepare the pinned public UCI [Bike Sharing](https://doi.org/10.24432/C5W894)
and [Beijing PM2.5](https://doi.org/10.24432/C5JS49) archives, both CC BY 4.0:

```sh
python3 scripts/prepare_forecasting.py --output fixtures/research-real
RAYON_NUM_THREADS=1 cargo run --release --example research_forecasting -- fixtures/research-real/bike-h1.json 32 17
```

The fixed study is both datasets × horizons 1/6 hours × quantile counts
8/32/128 × seeds 17/29/43. Splits span 50/20/10/20% of chronological time
for training, validation, calibration and untouched testing. Every earlier
split purges labels available at or after the next split boundary. Only
origin weather, exact-hour target lags and known future calendar enter the
features. Missing numeric features get a zero plus a missingness indicator;
label RMS scaling uses training only. Bike times use `America/New_York`,
Beijing `Asia/Shanghai`; measurements are available after their recorded
hour. Ambiguous Bike fall-back hours are omitted, not assigned an invented
fold. Source hashes, clocks, boundaries and prepared-file hashes are printed
by preparation; no datasets or fitted models are committed.

The executable compares default one-output-per-tree quantiles, shared vector
trees, a delegated custom-all control, and random-coordinate projected
Newton trees. Common settings are depth 4, 256 bins, eta 0.1, and base score
zero. Each strategy selects its prefix on validation with patience 50 and a
4,096-round cap. Capped searches disqualify quality matches. The three timed
refits are interleaved across strategies; the common bin cache, timing
exclusions and tuning time are reported explicitly.

Reports include test pinball loss over the central 5–95% quantile grid (not
full CRPS), batch and single-row prediction latency including the common
quantile transform, bit-identity checks, native encoded size, and tree/node
counts. Both existing strategies are baselines. A training speedup alone is
insufficient: prediction cost and size may grow with the projected model's
round count. Experimental custom-loss saves require the external transform.

Calibration uses original forecast bands and actual delayed label arrival:
fixed correction, rolling windows 50/200, and scalar/linear online trackers.
Online rates are chosen from 0.005/0.02/0.1 inside calibration, never test.
Coverage, interval width, endpoint penalties, empty intervals and score
pinball are all reported. This does not establish trading profit or
conditional coverage.

For a hosted run, dispatch `ci.yml` on the research branch with
`forecasting_research=true`. Each dataset/horizon/seed gets a separate
GitHub-hosted VM. Compilation finishes before its sequential measured runs;
host metadata and JSON reports are uploaded for 14 days. Normal CI leaves
this job off and uploads no research artifacts. Hosted VMs isolate this job
from the developer machine's workloads, not from all possible cloud noise.

## Parity fixtures

`gen_fixtures.py` trains **real XGBoost 3.4.2** (single thread) on deterministic
synthetic datasets, one case per supported feature (tree methods, missing values,
constraints, every objective — including `reg:logistic`, the `reg:linear`
alias, the alpha-list objectives with one and three alphas (list-valued
`quantile_alpha` / `expectile_alpha` params), and the survival objectives
with censored and tied times — sample weights, `scale_pos_weight` on every
objective XGBoost reads it in (`binary:logistic`, `binary:logitraw`,
`reg:logistic`, `reg:squarederror`, `reg:linear`, `reg:gamma`: the
`*_spw*` cases, whose regression labels are exactly 1 on about a third of
the rows), ranking groups, gblinear, DART, intercept estimation, row/column
sampling including `sampling_method=gradient_based` and DMatrix
`feature_weights`, multi-target
label matrices, per-round metric oracles for `rmsle`, `mape`, `mphe`,
`pre`/`pre@k`, the survival metrics and each objective's default metric,
continued training, `process_type=update`, `num_parallel_tree` forests,
`iteration_range` and slicing), and writes each case to
`../fixtures/<name>.json`: data, the exact `xgb.train` parameter dict,
XGBoost's test-set predictions (transformed, raw margin, SHAP contributions on
the first 50 rows, and for the cases in `INTERACTION_CASES` SHAP interaction
values on the first 5 rows: numeric, missing-value, categorical, multiclass,
DART, and the multi-output layouts, i.e. label matrix, alpha list,
parallel-tree forests, and vector-leaf trees) and the saved model JSON, plus
the model's UBJSON encoding (`save_raw("ubj")`) as `../fixtures/<name>.ubj`,
named by the `xgb_model_ubj` field. Case names say what they vary: `exact_` /
`approx_` / `hist_` tree methods, `_d<k>` the depth, and `nobs_` cases drop
`base_score` so the intercept is estimated. `weighted` cases carry
`weights`, ranking cases `group_sizes` / `test_group_sizes`, and categorical
cases (two integer-coded categorical columns) `feature_types`;
`categorical_missing_reg_d6` (6 and 80 categories with missing values, so the
forward and backward partition scans differ and `max_cat_threshold` binds)
and `categorical_onehot_reg_d6` (3 and 2 categories, one-hot splits) cover
the categorical split search. `n_targets`
gives the label columns: the `multi_*` cases (3-target `reg:squarederror`
on hist and exact and with `scale_pos_weight`, multi-label
`binary:logistic` with and without `scale_pos_weight`, weighted 2-target
`reg:pseudohubererror` and
`reg:absoluteerror`) store `y_train`/`y_test` row-major `[row][target]`, and
their predictions, margins, and contributions carry the target axis. The
`mot_*` cases train `multi_strategy=multi_output_tree` (vector-leaf trees):
3-target `reg:squarederror` depthwise, lossguide, with missing values,
regularized (`gamma`, `min_child_weight`, `reg_alpha`, `reg_lambda`,
`max_delta_step`), monotone, interaction-constrained and categorical;
multi-label `binary:logistic`; `multi:softprob` / `multi:softmax`; weighted
`reg:pseudohubererror`; `reg:quantileerror` / `reg:expectileerror` alpha
lists and a weighted `reg:absoluteerror` label matrix; a `num_parallel_tree=3`
forest with iteration ranges and slices; continued `multi:softprob` training
with ranges; plus quality-tier subsampling and DART. Their contributions
(and, for several, interactions) exercise vector-leaf QuadratureTreeSHAP.
The `forest_*`, `rf_*`, and `boosted_rf_*` cases cover `num_parallel_tree`
on scalar, multiclass, label-matrix, and alpha-list models.

It also writes `../fixtures/cuts/<name>.json`, XGBoost's `hist` and `approx`
quantile cuts (`DMatrix.get_quantile_cut`) for a set of matrices (uniform,
few distinct values, normal, missing values, sparse weights; `approx` cases
with squared-error and logistic round-0 Hessians).

`tests/parity.rs` runs these checks per case:

1. **Train parity** - train on the fixture data, compare `predict(x_test)` with
   XGBoost's predictions.
2. **Import parity** - `BoostedModel::decode` (`ModelFormat::XgboostJson`) on the embedded model,
   compare predictions, raw margins, SHAP contributions, and (where recorded)
   SHAP interaction values; decoding the `.ubj` sidecar (`XgboostUbjson`) must
   yield the identical model (column `ubj`).
3. **Export parity** - encode `XgboostJson` (`<name>.model.json`),
   `XgboostUbjson` (`<name>.model.ubj`) and hessboost's predictions to
   `../fixtures/exports/`; `check_exports.py` reloads each model in XGBoost and
   compares predictions, and for UBJSON also requires every array to use the
   same container form (typed element marker or generic) as XGBoost's own
   `save_raw("ubj")` of the loaded model.

4. **Feature extras** (column `extra`: largest delta / number of checks, `-`
   when the case has none), from optional fixture fields:
   - `continuation` (`first_rounds`, `xgb_model_initial`): XGBoost trained
     `first_rounds`, saved the model, and continued to `num_round` with
     `xgb_model=`. Train parity then runs `train` + `Trainer::init_model`
     the same way, and the imported initial model is continued and compared
     too.
   - `refresh` (`n_rows`, `y`, `rounds`, `refresh_leaf`, `xgb_pred`):
     `process_type=update` + `updater=refresh` of the final model on the first
     `n_rows` training rows relabelled `y`; hessboost refreshes the imported
     model (and, for `exact`-tier cases, its own) with `Trainer::init_model`.
   - `ranges` / `range_contribs` / `slices`: `iteration_range=(begin, end)`
     margins (hessboost's `predict_margin(data, begin..end)`), prefix-range
     contributions and leaf indices on the contribution rows, and
     `booster[begin:end:step]` margins (`slice(begin..end, step)`). Checked
     on the imported model (leaf ids included) and, for `exact`-tier cases,
     the trained model.

`quantile_cuts_match_xgboost` compares the cut oracles bit-for-bit with
`HistCuts::from_dmatrix` (`hist`) and `HistCuts::from_dmatrix_weighted` with
the round-0 Hessians (`approx`).

Cases are tiered. `exact` cases are pointwise: max |delta| within `tol.train`
(1e-4; 1e-5 for probabilities), `tol.import` (1e-5), `tol.contribs` (1e-4) and
`tol.interactions` (1e-4).
`quality` cases are RNG-driven (`subsample`, `sampling_method=gradient_based`,
`colsample_*` with or without `feature_weights`, DART, random forests);
training uses a regression RMSE <= 1.08x XGBoost band (accuracy >= XGBoost -
0.02 for classification) while import/export remain pointwise. A case's
optional `feature_weights` array (one weight per column) is set on the
training DMatrix on both sides. The `trainonly`-tier gblinear case validates
training pointwise; its unsupported XGBoost-JSON import/export path is
visibly reported as `n/a`/`skipped` and required to return `ModelFormat` on
import. Unknown XGBoost
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
  a `Trainer` watching the same set and requires the same metric names and,
  every round, `|hessboost - xgboost| <= tol.evals * max(1, |xgboost|)`
  (`tol.evals` = 1e-5; column `evals`, `-` for cases without oracles).

```sh
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/gen_fixtures.py
cargo nextest run --test parity --release --run-ignored only --no-capture
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/check_exports.py
```

Fixtures are not checked in; CI regenerates them (`.github/workflows/ci.yml`,
job `parity`). The generator refuses any XGBoost version other than 3.4.2, pinned in
`requirements-xgboost.txt` (a source build: 3.4.2 has no PyPI wheel; CI
caches the built wheel in uv's cache).

Fixture generators share deterministic seed, dtype-explicit JSON-float, dense-array, and compact JSON-writing helpers in `_fixture_common.py`; each keeps its own case matrices and data distributions. `gen_lightgbm_fixtures.py --test-data` is parsed as a command-line option and regenerates the checked-in small models under `tests/data/`.

## LightGBM import parity

`gen_lightgbm_fixtures.py` trains **real LightGBM 4.7.0**
(`requirements-lightgbm.txt`; single thread, `deterministic`) and writes each
case to `../fixtures/lightgbm/<name>.txt` (the `save_model` text model) and
`<name>.json` (test rows, LightGBM's raw scores, predictions,
`pred_contrib` SHAP values, `pred_leaf` indices, raw scores of a half-length
slice, and what the import must do). The test rows include, for up to 40
numeric splits, the threshold rounded to `f32` and its two `f32`
neighbours, so the `<=`-on-doubles conversion is checked at its edge. Cases:
numeric, `NaN` and zero missing values, `zero_as_missing` (a small model
whose splits all map, and one that must be refused), categorical splits
with multi-word bitsets, multiclass and one-vs-all, `lambdarank` and
`rank_xendcg`, `linear_tree` (with categorical routing), every other
objective, DART and GOSS; refusals: `sigmoid` other than 1, `reg_sqrt`,
`cross_entropy_lambda`, and random forests (`average_output`).

`tests/lightgbm_parity.rs` imports each model and compares pointwise within
`1e-5` relative (`max(1, |LightGBM|)`), checks leaf indices, the slice, and
native binary/JSON and XGBoost JSON round trips; refused cases must fail
with a `ModelFormat` error naming the reason. `--test-data` instead
rewrites the small checked-in `tests/data/lightgbm-4.7.0-*` models and
their `*.expected.json` predictions.

```sh
uv run --with-requirements scripts/requirements-lightgbm.txt python scripts/gen_lightgbm_fixtures.py
cargo nextest run --test lightgbm_parity --release --run-ignored only --no-capture
```

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
the comparison. Its results document a change and belong in that change's
pull request; [Performance](../docs/performance.md) records only comparisons
between alternatives that exist side by side.

The CUDA Criterion groups (`--filter '^cuda_'`, executables built with
`--features cuda`) time the CPU and CUDA engines side by side: one-node
histogram builds, dense, categorical, loss-guided, and CSR training, and
reused-forest prediction (row transfers included; forest upload and the
driver's first PTX compilation excluded). The executable embeds the
kernels' PTX, so its hash in `comparison.json` identifies the CUDA code
too.

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
source, executable (which embeds the CUDA kernels' PTX), and dataset hashes.
Build before timing and avoid running other benchmarks or compiler jobs
concurrently.

For a quick harness check, add `--rows 512 --rounds 3 --repeats 1`. Use
`--workloads regression` to select one dataset or `--threads 1` for a
single-thread comparison. Remove the package version constraints to benchmark
the latest releases available through `uv`. The output records the versions
actually used. To compare against the parity pin instead, replace the two
`--with` options with `--with-requirements scripts/requirements-xgboost.txt`
(XGBoost 3.4.2, a source build).

`--device cuda` trains both engines on the GPU (Linux with an NVIDIA GPU):
build the example with `--features cuda`, and use the PyPI `xgboost` wheel
(built with CUDA 13, so the driver must support CUDA 13), since the parity
pin is a CPU-only source build. The report records the GPU name, driver, and
ECC mode; the warmup fit absorbs CUDA context creation and hessboost's
kernel compilation. XGBoost's GPU sketch and gradient quantization give a
different model than its CPU `hist`, so compare the held-out scores too.

```sh
cargo build --release --features cuda --example bench_compare
uv run --with xgboost==3.4.1 --with numpy==2.5.2 python scripts/bench_xgb.py \
  --hessboost target/release/examples/bench_compare \
  --output /tmp/hessboost-xgb-cuda --threads 16 --device cuda
```

See [Performance](../docs/performance.md#xgboost-comparison) for the recorded
comparison and workload definitions.


## CPU baseline/candidate comparison

`examples/bench_cpu.rs` generates deterministic synthetic regression data and
measures warm-bin-cache training, cold training (fresh matrix construction and
binning included), margin prediction, cut construction (`cuts`), or bin
assignment with precomputed cuts (`bins`, including the cuts clone). Arguments:

```sh
RAYON_NUM_THREADS=1 cargo run --release --example bench_cpu -- \
  predict 8 30 100 6 1 5 dense 10000
# operation rows features rounds depth colsample repeats [dense|sparse] [training_rows]
```

Prediction training size is independent of batch size (at least 10,000 by
default). `sparse` means a NaN-filled dense matrix, not CSR. Small-batch
prediction below 16 rows records the per-call time over 1,000 calls; 16–1,023
rows averages 100 calls, and larger batches time one call. One warmup measurement
is discarded per process. Prediction and serialized-model checksums are
checked on every repeat, outside the timer. The warm training timer excludes
matrix construction and reuses the prebuilt bins; `train-cold` includes both.
Model destruction and checksum/quality validation are outside fit timing.
For `cuts` and `bins`, `checksum` hashes the cut values or assigned bins,
respectively, and the legacy `model_checksum` field holds the cuts hash, not
a model hash. Cut construction is outside the `bins` timer.

Build this same example against both source revisions, using separate target
directories (or force rebuilding the library when switching checkouts), then:

```sh
python scripts/bench_cpu.py --baseline /tmp/baseline-bench_cpu \
  --candidate /tmp/candidate-bench_cpu --output /tmp/cpu-comparison \
  --threads 1 4 --repeats 5
```

The runner executes baseline/candidate/candidate/baseline serially for each
case and thread count, checks identical model and prediction checksums across
both binaries and thread counts, and saves all samples, median/min/max,
binary SHA-256s and host information. Each result contains ten recorded
samples per binary with the command above. Select cases with `--cases`.
Avoid competing workloads during the run; shared/virtualized hosts still
introduce noise. These synthetic cases test specific paths, not general model
quality or hardware-wide performance.
