//! In-place row addition and deletion (`hessboost::training::online`).

use crate::booster::Booster;
use crate::data::{DMatrix, row_major};
use crate::errors::{DetachExt, OrRaise, refuse};
use crate::params::Params;
use crate::train::{Failure, run_hooked};
use hessboost::training::online;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, TryLockError};

/// `(nodes_kept, subtrees_regrown, rows_refreshed)`.
type Report = (usize, usize, usize);

/// A model with its training data and update state. The one class with
/// mutable state: an update changes it in place (copying the per-node
/// histograms would cost more than the update) under a mutex. An update is
/// admitted on its caller's thread, before any of its work is scheduled
/// (`updating`); while one is admitted, reading the model or data and
/// starting another update fail fast, from any thread or the update's own
/// callback, instead of waiting behind it. The admitted update waits only
/// for a read in progress, which holds the mutex briefly. The row count is
/// readable throughout.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct OnlineModel {
    state: Mutex<online::OnlineModel>,
    /// Whether an update is admitted: set before its work is scheduled,
    /// cleared once it returns.
    updating: AtomicBool,
    /// The committed data's row count.
    rows: AtomicUsize,
}

/// An admitted update; dropping it ends the admission.
struct Admission<'a>(&'a AtomicBool);

impl Drop for Admission<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// The refusal of an access while an update is admitted.
fn being_updated() -> PyErr {
    refuse(
        "the online model is being updated (from another thread or this update's callback); \
         its model, data and updates are available once the update returns",
    )
}

/// The refusal of every access after an update panicked.
fn poisoned() -> PyErr {
    refuse("an earlier update of this online model panicked; its state is unknown")
}

/// An update mode (`online::OnlineParams`): `exact()` or `approximate(tolerance)`.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct OnlineParams {
    inner: online::OnlineParams,
}

#[pymethods]
impl OnlineParams {
    #[staticmethod]
    fn exact() -> Self {
        Self {
            inner: online::OnlineParams::exact(),
        }
    }

    #[staticmethod]
    fn approximate(tolerance: f64) -> PyResult<Self> {
        let inner = online::OnlineParams::approximate(tolerance).or_raise()?;
        Ok(Self { inner })
    }
}

impl OnlineModel {
    fn new(online: online::OnlineModel) -> Self {
        Self {
            rows: AtomicUsize::new(online.data().n_rows()),
            updating: AtomicBool::new(false),
            state: Mutex::new(online),
        }
    }

    /// Admits an update unless one is admitted already, deciding on the
    /// caller's thread before any of the update's work is scheduled: a
    /// second update is refused at once instead of queued behind the first.
    fn admit(&self) -> PyResult<Admission<'_>> {
        self.updating
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| being_updated())?;
        Ok(Admission(&self.updating))
    }

    /// The state for a read, unless an update is admitted or holds it. A
    /// poisoned lock (a panic mid-update) leaves the state unknown, so it is
    /// refused from then on.
    fn try_state(&self) -> PyResult<MutexGuard<'_, online::OnlineModel>> {
        if self.updating.load(Ordering::Acquire) {
            return Err(being_updated());
        }
        self.state.try_lock().map_err(|error| match error {
            TryLockError::WouldBlock => being_updated(),
            TryLockError::Poisoned(_) => poisoned(),
        })
    }

    /// The state for the admitted update, after any read in progress.
    fn admitted_state(&self) -> PyResult<MutexGuard<'_, online::OnlineModel>> {
        self.state.lock().map_err(|_| poisoned())
    }

    /// `read` of the state, detached.
    fn with<T: Send>(
        &self,
        py: Python<'_>,
        read: impl FnOnce(&online::OnlineModel) -> T + Send,
    ) -> PyResult<T> {
        py.detach(|| Ok(read(&*self.try_state()?)))
    }
}

#[pymethods]
impl OnlineModel {
    /// Trains `num_boost_round` iterations on `dtrain`, keeping what updates
    /// need.
    #[staticmethod]
    fn train(
        py: Python<'_>,
        params: &Params,
        dtrain: &DMatrix,
        num_boost_round: usize,
        mode: &OnlineParams,
    ) -> PyResult<Self> {
        let online = mode.inner;
        let failure = Failure::default();
        let trained = run_hooked(py, None, &failure, |hook, _gate| {
            online::OnlineModel::train_with(
                &params.inner,
                &dtrain.inner,
                num_boost_round,
                online,
                hook,
            )
        })?
        .or_raise()?;
        Ok(Self::new(trained))
    }

    /// Resumes from `booster`, trained with `params` on `dtrain`.
    #[staticmethod]
    fn from_model(
        py: Python<'_>,
        booster: &Booster,
        params: &Params,
        dtrain: &DMatrix,
        mode: &OnlineParams,
    ) -> PyResult<Self> {
        let mode = mode.inner;
        let online = py.detached(|| {
            online::OnlineModel::from_model(
                (*booster.model).clone(),
                &params.inner,
                &dtrain.inner,
                mode,
            )
        })?;
        Ok(Self::new(online))
    }

    /// Adds `additions`' rows and deletes the rows `deletions`, calling
    /// `on_round(iteration, scores) -> stop` after every updated iteration.
    /// Returns `None` when `on_round` stopped the update, which then changed
    /// nothing, as does an exception or `KeyboardInterrupt` (re-raised): the
    /// update is applied only once the caller has checked for signals after
    /// the last iteration.
    #[pyo3(signature = (additions, deletions, on_round=None))]
    fn update(
        &self,
        py: Python<'_>,
        additions: Option<&DMatrix>,
        deletions: PyReadonlyArray1<'_, i64>,
        on_round: Option<Py<PyAny>>,
    ) -> PyResult<Option<Report>> {
        let deletions = row_major(&deletions, "deletions")?
            .iter()
            .map(|&row| {
                usize::try_from(row)
                    .map_err(|_| refuse(format!("deletions: negative row index {row}")))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let additions = additions.map(|matrix| &matrix.inner);
        let failure = Failure::default();
        let _admitted = self.admit()?;
        let (result, stopped) = run_hooked(py, on_round, &failure, |mut hook, gate| {
            let mut state = match self.admitted_state() {
                Ok(state) => state,
                Err(error) => return (Err(error), false),
            };
            let mut stopped = false;
            let result = state.update_with_commit(
                additions,
                &deletions,
                |round| {
                    let flow = hook(round);
                    stopped |= flow.is_break();
                    flow
                },
                || gate.confirm(),
            );
            if result.is_ok() {
                self.rows.store(state.data().n_rows(), Ordering::Relaxed);
            }
            (result.or_raise(), stopped)
        })?;
        if stopped {
            return Ok(None);
        }
        let report = result?;
        Ok(Some((
            report.nodes_kept,
            report.subtrees_regrown,
            report.rows_refreshed,
        )))
    }

    /// The current model (a copy).
    #[getter]
    fn model(&self, py: Python<'_>) -> PyResult<Booster> {
        let model = self.with(py, |state| state.model().clone())?;
        Ok(Booster::new(model))
    }

    /// The current training data (a copy).
    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<DMatrix> {
        let inner = self.with(py, |state| state.data().clone())?;
        Ok(DMatrix { inner })
    }

    #[getter]
    fn num_row(&self) -> usize {
        self.rows.load(Ordering::Relaxed)
    }
}
