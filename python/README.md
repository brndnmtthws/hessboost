# hessboost for Python

**XGBoost's gradient boosting, reimplemented in Rust**, with XGBoost's
Python API: `DMatrix`, `train`, `cv`, `Booster`, and scikit-learn
estimators. hessboost takes XGBoost's parameter names, reproduces XGBoost
3.4.2's predictions (checked against XGBoost in CI), and reads and writes
XGBoost JSON and UBJSON model files.

- **Strict.** An unknown parameter, a value of the wrong type or range, or
  a combination hessboost does not implement raises an error; nothing is
  silently ignored.
- **Deterministic.** The same parameters, data, and seed give the same
  model at any `nthread`.
- **Typed** (`py.typed`, complete type information), with the GIL released
  while training and predicting, and free-threaded CPython supported.
- **More than XGBoost, opt-in.** Conformal prediction intervals,
  distributional boosting (a predictive distribution per row), and
  LightGBM/CatBoost tree options.

## Installation

```sh
uv add hessboost        # or: pip install hessboost
```

Extras: `hessboost[pandas]` and `hessboost[scikit-learn]`. numpy is the only
required dependency.

Prebuilt wheels are published for Linux x86_64 and aarch64 (manylinux),
macOS arm64 (with Metal support, `device="metal"`), and Windows x86_64: one
`abi3` wheel per platform for CPython 3.11 and newer, plus a wheel for
free-threaded CPython 3.14t. Elsewhere the installer builds from the source
distribution, which needs Rust 1.93 or newer and a C compiler (for libzstd).

## Quick start

```python
import numpy as np
import hessboost

rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 5))
y = (X[:, 0] + X[:, 1] ** 2 > 1).astype(int)

dtrain = hessboost.DMatrix(X[:800], label=y[:800])
dvalid = hessboost.DMatrix(X[800:], label=y[800:])

booster = hessboost.train(
    {"objective": "binary:logistic", "max_depth": 4, "eta": 0.1, "eval_metric": "auc"},
    dtrain,
    num_boost_round=500,
    evals=[(dvalid, "valid")],
    early_stopping_rounds=20,
    verbose_eval=False,
)
probabilities = booster.predict(X[800:])       # through booster.best_iteration
shap = booster.predict(X[800:], pred_contribs=True)  # (rows, features + 1)
print(booster.best_iteration, booster.get_score(importance_type="gain"))
```

`DMatrix` takes numpy arrays of any numeric dtype and memory layout (a
C-contiguous `float32` array is used without a copy), pandas DataFrames,
scipy sparse matrices, and anything `numpy.asarray` accepts. NaN is missing
(or pass `missing=`); ranking data takes `group=` sizes or `qid=`, and
`survival:aft` takes `label_lower_bound=`/`label_upper_bound=`.
`Booster.predict` accepts the same inputs directly.

`train` supports XGBoost's everyday arguments: `evals`, `evals_result`,
`early_stopping_rounds`, `verbose_eval` (printed live, every round or
every `n`-th), `xgb_model` (continued training, or tree refresh with
`process_type="update"`), a custom objective `obj`, a `custom_metric`,
and `callbacks`. Ctrl-C stops training at the end of the current round
and raises `KeyboardInterrupt`. `hessboost.cv` cross-validates over
shuffled folds, explicit folds, or a scikit-learn splitter.

```python
class StopAtTarget(hessboost.TrainingCallback):
    def after_iteration(self, iteration, evals_log):
        return evals_log["valid"]["auc"][-1] > 0.99   # True stops training

hessboost.train(params, dtrain, 1000, evals=[(dvalid, "valid")],
                callbacks=[StopAtTarget()])
```

### pandas and categorical features

DataFrame column names become feature names, and `category` columns become
native categorical features. The booster remembers each column's
categories, so predicting on a frame whose categories are ordered
differently (or include unseen values, which count as missing) re-codes
them first:

```python
import pandas as pd

df = pd.DataFrame({"color": pd.Categorical(["red", "blue", "red"] * 100),
                   "size": np.arange(300.0)})
booster = hessboost.train({}, hessboost.DMatrix(df, label=np.arange(300.0)), 20)
booster.predict(df)
```

## scikit-learn

```python
from hessboost.sklearn import HessboostClassifier

model = HessboostClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                            early_stopping_rounds=20)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
model.predict_proba(X_test)
model.feature_importances_
```

`HessboostRegressor` (multi-target with a 2-D `y`), `HessboostClassifier`
(any class labels), `HessboostRanker` (`group=` or `qid=`), and
`HessboostDistributionRegressor` (`dist:*` objectives; `predict` returns
means, `predict_distribution` the distributions) take XGBoost's
scikit-learn parameter names, plus `callbacks`. Parameters left at `None`
keep hessboost's defaults, and `params={...}` passes any other training
parameter; `fit(..., verbose=True)` (or a period) prints rounds live. They
work with pipelines, `clone`, grid search, and pickling, and pass
scikit-learn's estimator checks (except three documented deviations).
`import hessboost` does not import scikit-learn; `hessboost.sklearn` needs
it.

## Model files

| Method | Formats |
|---|---|
| `save_model(path, format=None)` | by extension: `.json` native JSON, `.ubj` XGBoost UBJSON, anything else native binary; or `format="binary" \| "json" \| "xgboost-json" \| "xgboost-ubjson"` |
| `Booster(path_or_bytes)`, `load_model(...)` | any of the four, detected from the content |
| `save_raw(format="binary")` | the same formats as bytes |
| `pickle` / `copy` | native binary plus feature names, categories, and `best_score` |

The native binary format is compressed, checksummed, and lossless; files
written by 0.2.0 or later load in every later release. Use
`format="xgboost-json"` or `.ubj` for a file XGBoost loads. `booster[a:b]`
slices boosting iterations.

## Beyond XGBoost

```python
from hessboost.conformal import ConformalizedQuantile

band = hessboost.train({"objective": "reg:quantileerror", "quantile_alpha": [0.05, 0.95]},
                       hessboost.DMatrix(X_train, y_train), 200)
cqr = ConformalizedQuantile.calibrate_outputs(band, X_cal, y_cal, alpha=0.1)
lower, upper = cqr.predict_interval(X_test).T   # >= 90% coverage, finite-sample

dist = hessboost.train({"objective": "dist:normal"}, hessboost.DMatrix(X_train, y_train), 300)
d = dist.predict_distribution(X_test)
d.mean(), d.std(), d.interval(0.9), d.log_prob(y_test), d.crps(y_test)
```

- `hessboost.conformal`: `SplitConformal` and `ConformalizedQuantile`
  (from two quantile models, two outputs of one, or a `dist:*` model).
- `hessboost.folds`: `k_fold`, `forward_chaining` (expanding-window,
  purged by a row `gap`), and `purged_forward` (timestamped rows, purged
  by each row's own label window, for overlapping or irregular horizons)
  folds for `cv` or your own validation loops.
- Every hessboost training option (`path_smooth`, `extra_trees`,
  `linear_tree`, `grow_policy="symmetric"`, `use_quantized_grad`, the
  `dist:*` objectives and their `dist_gradient`, ...) is a `params` key.

## Differences from XGBoost's Python package

- Errors: `hessboost.HessboostError` (a `ValueError`; `ModelFormatError`
  for model files) for refused inputs, `TypeError` for wrong types.
  Unsupported parameters are refused, including `verbosity`; `missing`
  belongs to `DMatrix`.
- `TrainingCallback.after_iteration(iteration, evals_log) -> bool` sees the
  round and the evaluation history, not the model (XGBoost's also gets the
  booster and has `before_training`/`after_training` hooks); returning
  `True` stops training with the rounds so far. `hessboost.cv` has no
  callbacks and finishes its folds before Ctrl-C takes effect.
- `custom_metric(predictions, labels, weights)` returns a float and is named
  by the function's `__name__` (XGBoost passes a `DMatrix` and returns
  `(name, value)`). `obj(margins, dtrain)` matches XGBoost; the model then
  predicts margins from a zero intercept (or `base_score`).
- `cv` returns a dict of numpy arrays (`test-<metric>-mean`/`-std`) with
  held-out metrics only; there is no `stratified` or `as_pandas`.
- `predict` defaults to the iterations through `best_iteration` (XGBoost's
  scikit-learn behavior); pass `iteration_range=(0, 0)` for all.
  `pred_leaf` returns `int32`.
- Model files do not store feature names or categories (pickles do).
- Not available: `DMatrix` from files or `QuantileDMatrix`, `inplace_predict`
  (`predict` takes arrays directly), `Booster.get_dump`/`trees_to_dataframe`
  /`dump_model`, attributes (`set_attr`), plotting, distributed (Dask/Spark)
  and GPU (CUDA) training, `approx_contribs`, and `strict_shape`.

## Development

From `python/` in the [repository](https://github.com/brndnmtthws/hessboost),
with [uv](https://docs.astral.sh/uv/):

```sh
uv sync                        # build the extension and install dev tools
uv run pytest
uv run python -m mypy.stubtest hessboost._hessboost
uv run mypy --strict
uv run pyright --verifytypes hessboost --ignoreexternal
```

## License

Apache-2.0.
