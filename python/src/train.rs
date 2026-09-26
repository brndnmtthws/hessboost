//! Training (`Trainer`), cross-validation, and fold construction.

use crate::booster::Booster;
use crate::data::{DMatrix, row_major, to_numpy};
use crate::errors::{OrRaise, refuse};
use crate::params::Params;
use hessboost::metric::CustomMetric;
use hessboost::objective::{CustomObjective, GradPair};
use hessboost::training::{CrossValidation, Fold, RoundEval, Trainer};
use numpy::ndarray::{ArrayView1, ArrayView2};
use numpy::{PyArrayDyn, PyReadonlyArray1, ToPyArray};
use pyo3::panic::PanicException;
use pyo3::prelude::*;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

/// The first exception a Python callback raised during training (or the
/// caller's `KeyboardInterrupt`), re-raised once training returns. The round
/// hook then stops training at the end of the round; until then later
/// callbacks are skipped: the objective returns zero gradients and the
/// metric NaN.
#[derive(Clone, Default)]
struct Failure(Arc<Mutex<Option<PyErr>>>);

impl Failure {
    fn failed(&self) -> bool {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
    }

    fn record(&self, error: PyErr) {
        let mut slot = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.is_none() {
            *slot = Some(error);
        }
    }

    fn take(&self) -> Option<PyErr> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).take()
    }
}

/// A `(rows,)` or `(rows, width)` view of row-major `values` as numpy.
fn rows_array<'py>(py: Python<'py>, values: &[f32], width: usize) -> PyResult<Bound<'py, PyAny>> {
    if width <= 1 {
        return Ok(ArrayView1::from(values).to_pyarray(py).into_any());
    }
    let view = ArrayView2::from_shape((values.len() / width, width), values)
        .map_err(|error| refuse(format!("unexpected prediction layout: {error}")))?;
    Ok(view.to_pyarray(py).into_any())
}

/// The custom objective: `function(margins) -> (grad, hess)`, both
/// `float32` arrays of `rows × outputs` values.
fn custom_objective(
    function: Py<PyAny>,
    outputs: usize,
    base: f32,
    failure: Failure,
) -> CustomObjective {
    CustomObjective::new(
        "custom",
        outputs,
        base,
        "rmse",
        move |margins, _labels, _weights, out: &mut [GradPair]| {
            out.fill(GradPair::default());
            if failure.failed() {
                return;
            }
            let result = Python::attach(|py| -> PyResult<()> {
                let returned = function.call1(py, (rows_array(py, margins, outputs)?,))?;
                let (grad, hess): (PyReadonlyArray1<'_, f32>, PyReadonlyArray1<'_, f32>) =
                    returned.extract(py)?;
                let (grad, hess) = (row_major(&grad, "grad")?, row_major(&hess, "hess")?);
                if grad.len() != out.len() || hess.len() != out.len() {
                    return Err(refuse(format!(
                        "the custom objective returned {} gradients and {} hessians for {} \
                         predictions",
                        grad.len(),
                        hess.len(),
                        out.len()
                    )));
                }
                for ((pair, &g), &h) in out.iter_mut().zip(grad).zip(hess) {
                    *pair = GradPair::new(g, h);
                }
                Ok(())
            });
            if let Err(error) = result {
                out.fill(GradPair::default());
                failure.record(error);
            }
        },
    )
}

/// The custom metric: `function(predictions, labels, weights) -> float`.
/// Labels hold `targets` values per row; predictions a whole number per row.
fn custom_metric(metric: MetricRequest, targets: usize, failure: Failure) -> CustomMetric {
    let MetricRequest {
        function,
        name,
        maximize,
    } = metric;
    CustomMetric::new(name, maximize, move |predictions, labels, weights| {
        if failure.failed() {
            return f64::NAN;
        }
        let result = Python::attach(|py| -> PyResult<f64> {
            let rows = (labels.len() / targets.max(1)).max(1);
            let width = predictions.len() / rows;
            let predictions = rows_array(py, predictions, width)?;
            let labels = rows_array(py, labels, targets)?;
            let weights = weights.map_or_else(
                || py.None().into_bound(py),
                |weights| ArrayView1::from(weights).to_pyarray(py).into_any(),
            );
            function
                .call1(py, (predictions, labels, weights))?
                .extract::<f64>(py)
        });
        result.unwrap_or_else(|error| {
            failure.record(error);
            f64::NAN
        })
    })
}

/// How often a waiting caller checks for signals (Ctrl-C).
const SIGNAL_POLL: Duration = Duration::from_millis(50);

/// The per-round hook: stops training once a callback failed or the caller
/// was interrupted, else calls `function(iteration, scores)` (if any), whose
/// truthy result stops training and whose exception is recorded and stops
/// it.
fn round_hook(
    function: Option<Py<PyAny>>,
    failure: Failure,
    interrupted: Arc<AtomicBool>,
) -> impl FnMut(&RoundEval) -> ControlFlow<()> + Send {
    move |round| {
        if interrupted.load(Ordering::Relaxed) || failure.failed() {
            return ControlFlow::Break(());
        }
        let Some(function) = &function else {
            return ControlFlow::Continue(());
        };
        let stop = Python::attach(|py| {
            function
                .call1(py, (round.iteration, round.scores.clone()))?
                .is_truthy(py)
        });
        match stop {
            Ok(false) => ControlFlow::Continue(()),
            Ok(true) => ControlFlow::Break(()),
            Err(error) => {
                failure.record(error);
                ControlFlow::Break(())
            }
        }
    }
}

/// Runs `work` on a worker thread while the caller, detached, wakes every
/// [`SIGNAL_POLL`] to run the interpreter's signal handlers (only the main
/// thread's do anything), passing a raised exception (`KeyboardInterrupt`)
/// to `on_signal`. `work` sees the interruption through its round hook and
/// stops at the end of the round.
fn interruptible<T: Send>(
    py: Python<'_>,
    work: impl FnOnce() -> T + Send,
    on_signal: impl Fn(PyErr),
) -> PyResult<T> {
    let caller = std::thread::current();
    std::thread::scope(|scope| {
        let worker = scope.spawn(move || {
            let out = work();
            caller.unpark();
            out
        });
        while !worker.is_finished() {
            py.detach(|| std::thread::park_timeout(SIGNAL_POLL));
            if let Err(error) = py.check_signals() {
                on_signal(error);
            }
        }
        worker
            .join()
            .map_err(|_| PanicException::new_err("the training thread panicked"))
    })
}

/// A custom metric callback with its reported name and direction.
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct MetricRequest {
    function: Py<PyAny>,
    name: String,
    maximize: bool,
}

/// One training run, passed from Python as a `dict`.
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct TrainRequest {
    params: Py<Params>,
    dtrain: Py<DMatrix>,
    num_boost_round: usize,
    #[pyo3(default)]
    evals: Vec<(Py<DMatrix>, String)>,
    #[pyo3(default)]
    early_stopping_rounds: Option<usize>,
    #[pyo3(default)]
    init_model: Option<Py<Booster>>,
    #[pyo3(default)]
    obj: Option<Py<PyAny>>,
    #[pyo3(default)]
    custom_metric: Option<MetricRequest>,
    /// `on_round(iteration, scores) -> stop`, called after every round.
    #[pyo3(default)]
    on_round: Option<Py<PyAny>>,
}

/// Trains a model; returns it with the best score (with early stopping).
/// The evaluation history reaches Python through `on_round`.
#[pyfunction]
pub(crate) fn train(py: Python<'_>, request: TrainRequest) -> PyResult<(Booster, Option<f64>)> {
    let params = &request.params.get().inner;
    let dtrain = &request.dtrain.get().inner;
    let evals: Vec<(&hessboost::data::DMatrix, &str)> = request
        .evals
        .iter()
        .map(|(data, name)| (&data.get().inner, name.as_str()))
        .collect();
    let init = request
        .init_model
        .as_ref()
        .map(|booster| &*booster.get().model);
    let failure = Failure::default();
    let outputs = if params.num_class > 0 {
        params.num_class
    } else {
        dtrain.n_targets()
    };
    let objective = request.obj.map(|function| {
        let base = params.base_score.unwrap_or(0.0) as f32;
        custom_objective(function, outputs, base, failure.clone())
    });
    let targets = dtrain.n_targets();
    let metric = request
        .custom_metric
        .map(|metric| custom_metric(metric, targets, failure.clone()));
    let interrupted = Arc::new(AtomicBool::new(false));
    let hook = round_hook(request.on_round, failure.clone(), Arc::clone(&interrupted));
    let result = interruptible(
        py,
        || {
            let mut trainer = Trainer::new(params, dtrain, request.num_boost_round).on_round(hook);
            for (data, name) in &evals {
                trainer = trainer.eval(data, name);
            }
            if let Some(rounds) = request.early_stopping_rounds {
                trainer = trainer.early_stopping_rounds(rounds);
            }
            if let Some(model) = init {
                trainer = trainer.init_model(model);
            }
            if let Some(objective) = &objective {
                trainer = trainer.objective(objective);
            }
            if let Some(metric) = metric {
                trainer = trainer.custom_metric(Box::new(metric));
            }
            trainer.train()
        },
        |error| {
            interrupted.store(true, Ordering::Relaxed);
            failure.record(error);
        },
    )?;
    if let Some(error) = failure.take() {
        return Err(error);
    }
    let result = result.or_raise()?;
    Ok((Booster::new(result.model), result.best_score))
}

/// `(metric, per-round test means, per-round test standard deviations)`.
type CvHistory = Vec<(String, Vec<f64>, Vec<f64>)>;

/// Cross-validates over explicit `(train rows, test rows)` folds.
#[pyfunction]
#[pyo3(signature = (params, data, num_boost_round, folds, early_stopping_rounds=None))]
pub(crate) fn cv(
    py: Python<'_>,
    params: &Params,
    data: &DMatrix,
    num_boost_round: usize,
    folds: Vec<(Vec<usize>, Vec<usize>)>,
    early_stopping_rounds: Option<usize>,
) -> PyResult<CvHistory> {
    let folds = folds
        .into_iter()
        .map(|(train, test)| Fold::new(train, test))
        .collect();
    let results = py
        .detach(|| {
            let mut cv = CrossValidation::new(&params.inner, &data.inner, num_boost_round, folds);
            if let Some(rounds) = early_stopping_rounds {
                cv = cv.early_stopping_rounds(rounds);
            }
            cv.run()
        })
        .or_raise()?;
    Ok(results
        .into_iter()
        .map(|result| (result.metric, result.test_mean, result.test_std))
        .collect())
}

type FoldArrays<'py> = Vec<(Bound<'py, PyArrayDyn<i64>>, Bound<'py, PyArrayDyn<i64>>)>;

fn fold_arrays(py: Python<'_>, folds: Vec<Fold>) -> PyResult<FoldArrays<'_>> {
    let rows = |rows: Vec<usize>| {
        let count = rows.len();
        to_numpy(
            py,
            rows.into_iter().map(|row| row as i64).collect(),
            &[count],
        )
    };
    folds
        .into_iter()
        .map(|fold| Ok((rows(fold.train)?, rows(fold.test)?)))
        .collect()
}

/// Shuffled k-fold splits of `n_rows` rows.
#[pyfunction]
pub(crate) fn k_fold(
    py: Python<'_>,
    n_rows: usize,
    nfold: usize,
    seed: u64,
) -> PyResult<FoldArrays<'_>> {
    fold_arrays(py, Fold::k_fold(n_rows, nfold, seed).or_raise()?)
}

/// Forward-chaining (expanding-window) splits of `n_rows` time-ordered rows,
/// purging `gap` rows before each test block.
#[pyfunction]
pub(crate) fn forward_chaining(
    py: Python<'_>,
    n_rows: usize,
    n_splits: usize,
    gap: usize,
) -> PyResult<FoldArrays<'_>> {
    fold_arrays(
        py,
        Fold::forward_chaining(n_rows, n_splits, gap).or_raise()?,
    )
}

/// Forward folds over timestamped rows, purging every training row whose
/// label window reaches into its fold's test block.
#[pyfunction]
pub(crate) fn purged_forward<'py>(
    py: Python<'py>,
    decision_at: PyReadonlyArray1<'_, i64>,
    label_end: PyReadonlyArray1<'_, i64>,
    validation_fraction: f64,
    blocks: usize,
    min_train: usize,
) -> PyResult<FoldArrays<'py>> {
    let (decision_at, label_end) = (
        row_major(&decision_at, "decision_at")?,
        row_major(&label_end, "label_end")?,
    );
    let folds = py
        .detach(|| {
            Fold::purged_forward(
                decision_at,
                label_end,
                validation_fraction,
                blocks,
                min_train,
            )
        })
        .or_raise()?;
    fold_arrays(py, folds)
}
