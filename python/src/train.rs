//! Training (`Trainer`, budget mode), cross-validation, and fold
//! construction.

use crate::booster::Booster;
use crate::data::{DMatrix, row_major, to_numpy};
use crate::errors::{DetachExt, OrRaise, refuse};
use crate::params::Params;
use crate::target_stats::OrderedTargetEncoder;
use hessboost::metric::CustomMetric;
use hessboost::objective::{CustomLoss, GradPair, Objective};
use hessboost::training::budget::{self, BudgetConfig};
use hessboost::training::{CrossValidation, Fold, RoundEval, Trainer};
use numpy::ndarray::{ArrayView1, ArrayView2};
use numpy::{PyArrayDyn, PyReadonlyArray1, ToPyArray};
use pyo3::panic::PanicException;
use pyo3::prelude::*;
use std::num::NonZeroUsize;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::Thread;
use std::time::Duration;

/// The first exception a Python callback raised during training (or the
/// caller's `KeyboardInterrupt`), re-raised once training returns. The round
/// hook then stops training at the end of the round; until then later
/// callbacks are skipped: the objective returns zero gradients and the
/// metric NaN.
#[derive(Clone, Default)]
pub(crate) struct Failure(Arc<Mutex<Option<PyErr>>>);

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
) -> CustomLoss {
    CustomLoss::new(
        "custom",
        outputs,
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
    .with_base_margin(base)
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
/// it. The interruption is checked again after `function` returns, so one
/// that arrived while it ran stops the work even after the last round.
fn round_hook(
    function: Option<Py<PyAny>>,
    failure: Failure,
    interrupted: Arc<AtomicBool>,
) -> impl FnMut(RoundEval<'_>) -> ControlFlow<()> + Send {
    move |round| {
        let stopped = || interrupted.load(Ordering::Relaxed) || failure.failed();
        if stopped() {
            return ControlFlow::Break(());
        }
        let Some(function) = &function else {
            return ControlFlow::Continue(());
        };
        // Python sees XGBoost's `(dataset, metric, value)` triples.
        let scores: Vec<(&str, &str, f64)> = round.scores().collect();
        let stop = Python::attach(|py| {
            function
                .call1(py, (round.iteration(), scores))?
                .is_truthy(py)
        });
        match stop {
            Ok(false) if !stopped() => ControlFlow::Continue(()),
            Ok(_) => ControlFlow::Break(()),
            Err(error) => {
                failure.record(error);
                ControlFlow::Break(())
            }
        }
    }
}

/// Where a [`CommitGate`] is.
enum Phase {
    Working,
    Asking,
    Answered(bool),
}

/// The worker's last question to its waiting caller: may the finished work
/// be applied? The caller answers after one more signal check and checks no
/// more signals after answering, so an interruption either reaches the work
/// before it is applied or is left for the interpreter to raise after the
/// call returns, never raised over applied work.
#[derive(Clone)]
pub(crate) struct CommitGate(Arc<GateState>);

struct GateState {
    phase: Mutex<Phase>,
    answered: Condvar,
    caller: Thread,
}

impl CommitGate {
    fn new(caller: Thread) -> Self {
        Self(Arc::new(GateState {
            phase: Mutex::new(Phase::Working),
            answered: Condvar::new(),
            caller,
        }))
    }

    /// Worker side: wakes the caller and waits for its answer.
    pub(crate) fn confirm(&self) -> ControlFlow<()> {
        let state = &self.0;
        let mut phase = state.phase.lock().unwrap_or_else(PoisonError::into_inner);
        *phase = Phase::Asking;
        state.caller.unpark();
        loop {
            match *phase {
                Phase::Answered(true) => return ControlFlow::Continue(()),
                Phase::Answered(false) => return ControlFlow::Break(()),
                Phase::Working | Phase::Asking => {
                    phase = state
                        .answered
                        .wait(phase)
                        .unwrap_or_else(PoisonError::into_inner);
                }
            }
        }
    }

    /// Caller side: answers a pending question with `commit()`; whether it
    /// has been answered.
    fn answer(&self, commit: impl FnOnce() -> bool) -> bool {
        let state = &self.0;
        let mut phase = state.phase.lock().unwrap_or_else(PoisonError::into_inner);
        match *phase {
            Phase::Working => false,
            Phase::Asking => {
                *phase = Phase::Answered(commit());
                state.answered.notify_all();
                true
            }
            Phase::Answered(_) => true,
        }
    }
}

/// Runs `work` on a worker thread, inside the extension's rayon pool, while
/// the caller, detached, wakes every [`SIGNAL_POLL`] to run the interpreter's signal handlers (only the main
/// thread's do anything), passing a raised exception (`KeyboardInterrupt`)
/// to `on_signal`, and answers `gate` with `may_commit` after a signal
/// check. `work` sees the interruption through its round hook and stops at
/// the end of the round.
fn interruptible<T: Send>(
    py: Python<'_>,
    work: impl FnOnce() -> T + Send,
    on_signal: impl Fn(PyErr),
    gate: &CommitGate,
    may_commit: impl Fn() -> bool,
) -> PyResult<T> {
    let caller = std::thread::current();
    std::thread::scope(|scope| {
        let worker = scope.spawn(move || {
            let out = crate::pool::install(work);
            caller.unpark();
            out
        });
        let mut answered = false;
        while !worker.is_finished() {
            py.detach(|| std::thread::park_timeout(SIGNAL_POLL));
            if answered {
                continue;
            }
            if let Err(error) = py.check_signals() {
                on_signal(error);
            }
            answered = gate.answer(&may_commit);
        }
        worker
            .join()
            .map_err(|_| PanicException::new_err("the training thread panicked"))
    })
}

/// The round hook [`run_hooked`] hands its work.
pub(crate) type RoundHook = Box<dyn FnMut(RoundEval<'_>) -> ControlFlow<()> + Send>;

/// Runs `work` [`interruptible`] with a [`round_hook`] calling `on_round`
/// and a [`CommitGate`] that lets it apply its result only if nothing
/// failed or interrupted it, and raises the first exception a callback
/// recorded in `failure`, or the caller's `KeyboardInterrupt`, in place of
/// `work`'s result.
pub(crate) fn run_hooked<T: Send>(
    py: Python<'_>,
    on_round: Option<Py<PyAny>>,
    failure: &Failure,
    work: impl FnOnce(RoundHook, CommitGate) -> T + Send,
) -> PyResult<T> {
    let interrupted = Arc::new(AtomicBool::new(false));
    let hook = Box::new(round_hook(
        on_round,
        failure.clone(),
        Arc::clone(&interrupted),
    ));
    let gate = CommitGate::new(std::thread::current());
    let worker_gate = gate.clone();
    let out = interruptible(
        py,
        move || work(hook, worker_gate),
        |error| {
            interrupted.store(true, Ordering::Relaxed);
            failure.record(error);
        },
        &gate,
        || !interrupted.load(Ordering::Relaxed) && !failure.failed(),
    )?;
    match failure.take() {
        Some(error) => Err(error),
        None => Ok(out),
    }
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
    /// The custom objective, `function(margins) -> (grad, hess)`.
    #[pyo3(default)]
    obj: Option<Py<PyAny>>,
    /// The custom objective's output count (default: one per label column
    /// of `dtrain`).
    #[pyo3(default)]
    outputs: Option<usize>,
    #[pyo3(default)]
    custom_metric: Option<MetricRequest>,
    /// `on_round(iteration, scores) -> stop`, called after every round.
    #[pyo3(default)]
    on_round: Option<Py<PyAny>>,
}

/// `early_stopping_rounds` as Python passes it: `0` is refused, as the
/// Rust API's `NonZeroUsize` cannot express it.
fn patience(rounds: Option<usize>) -> PyResult<Option<NonZeroUsize>> {
    rounds
        .map(|rounds| {
            NonZeroUsize::new(rounds).ok_or_else(|| {
                hessboost::error::HessboostError::invalid_param(
                    "early_stopping_rounds",
                    "must be greater than zero",
                )
            })
        })
        .transpose()
        .or_raise()
}

/// Trains a model; returns it with the best score (with early stopping).
/// The evaluation history reaches Python through `on_round`.
#[pyfunction]
pub(crate) fn train(py: Python<'_>, request: TrainRequest) -> PyResult<(Booster, Option<f64>)> {
    let parsed = &request.params.get().inner;
    let dtrain = &request.dtrain.get().inner;
    let early_stopping_rounds = patience(request.early_stopping_rounds)?;
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
    let custom;
    let params = match request.obj {
        Some(function) => {
            let outputs = request.outputs.unwrap_or_else(|| dtrain.n_targets());
            let base = parsed.base_score.unwrap_or(0.0) as f32;
            let mut params = parsed.clone();
            params.objective =
                Objective::custom(custom_objective(function, outputs, base, failure.clone()));
            custom = params;
            &custom
        }
        None => parsed,
    };
    let targets = dtrain.n_targets();
    let metric = request
        .custom_metric
        .map(|metric| custom_metric(metric, targets, failure.clone()));
    let result = run_hooked(py, request.on_round, &failure, |hook, _gate| {
        let mut trainer = Trainer::new(params, dtrain, request.num_boost_round).on_round(hook);
        for (data, name) in &evals {
            trainer = trainer.eval(data, name);
        }
        if let Some(rounds) = early_stopping_rounds {
            trainer = trainer.early_stopping_rounds(rounds);
        }
        if let Some(model) = init {
            trainer = trainer.init_model(model);
        }
        if let Some(metric) = metric {
            trainer = trainer.custom_metric(Box::new(metric));
        }
        trainer.train()
    })?;
    let result = result.or_raise()?;
    Ok((Booster::new(result.model), result.best_score))
}

/// Trains in budget mode (`hessboost::training::budget`). There is no
/// round hook, so it is not interruptible.
#[pyfunction]
#[pyo3(signature = (params, dtrain, budget, iteration_limit=None, stopping_rounds=None))]
pub(crate) fn train_with_budget(
    py: Python<'_>,
    params: &Params,
    dtrain: &DMatrix,
    budget: f64,
    iteration_limit: Option<usize>,
    stopping_rounds: Option<usize>,
) -> PyResult<Booster> {
    let mut config = BudgetConfig::new(budget);
    if let Some(limit) = iteration_limit {
        config = config.iteration_limit(limit);
    }
    if let Some(rounds) = stopping_rounds {
        config = config.stopping_rounds(rounds);
    }
    let result =
        py.detached(|| budget::train_with_budget(&params.inner, &dtrain.inner, &config))?;
    Ok(Booster::new(result.model))
}

/// `(metric, per-round test means, per-round test standard deviations)`.
type CvHistory = Vec<(String, Vec<f64>, Vec<f64>)>;

/// Cross-validates over explicit `(train rows, test rows)` folds, with
/// `target_stats = (encoder, columns)` fitted inside each fold.
#[pyfunction]
#[pyo3(signature = (
    params, data, num_boost_round, folds, early_stopping_rounds=None, target_stats=None
))]
pub(crate) fn cv(
    py: Python<'_>,
    params: &Params,
    data: &DMatrix,
    num_boost_round: usize,
    folds: Vec<(Vec<usize>, Vec<usize>)>,
    early_stopping_rounds: Option<usize>,
    target_stats: Option<(Py<OrderedTargetEncoder>, Vec<usize>)>,
) -> PyResult<CvHistory> {
    let early_stopping_rounds = patience(early_stopping_rounds)?;
    let folds = folds
        .into_iter()
        .map(|(train, test)| Fold::new(train, test))
        .collect();
    let target_stats =
        target_stats.map(|(encoder, columns)| (encoder.get().inner.clone(), columns));
    let results = py.detached(|| {
        let mut cv = CrossValidation::new(&params.inner, &data.inner, num_boost_round, folds);
        if let Some(rounds) = early_stopping_rounds {
            cv = cv.early_stopping_rounds(rounds);
        }
        if let Some((encoder, columns)) = target_stats {
            cv = cv.target_stats(encoder, columns);
        }
        cv.run()
    })?;
    Ok(results
        .into_iter()
        .map(|result| {
            let (means, stds) = result
                .rounds
                .iter()
                .map(|round| (round.mean, round.std))
                .unzip();
            (result.metric, means, stds)
        })
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
    let folds = py.detached(|| {
        Fold::purged_forward(
            decision_at,
            label_end,
            validation_fraction,
            blocks,
            min_train,
        )
    })?;
    fold_arrays(py, folds)
}
