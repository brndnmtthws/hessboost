//! In-place row addition and deletion (`hessboost::training::online`).

use crate::booster::Booster;
use crate::data::{DMatrix, row_major};
use crate::errors::{OrRaise, refuse};
use crate::params::Params;
use crate::train::{Failure, run_hooked};
use hessboost::training::online::{self, OnlineParams};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::sync::Mutex;

/// `(nodes_kept, subtrees_regrown, rows_refreshed)`.
type Report = (usize, usize, usize);

/// A model with its training data and update state. The one class with
/// mutable state: an update changes it in place (copying the per-node
/// histograms would cost more than the update), under a mutex that is only
/// ever locked detached, so an update's callback can re-attach while another
/// thread waits for it.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct OnlineModel {
    state: Mutex<online::OnlineModel>,
}

impl OnlineModel {
    fn new(online: online::OnlineModel) -> Self {
        Self {
            state: Mutex::new(online),
        }
    }

    /// `read` of the state, detached. A poisoned lock (a panic mid-update)
    /// leaves the state unknown, so it is refused from then on.
    fn with<T: Send>(
        &self,
        py: Python<'_>,
        read: impl FnOnce(&online::OnlineModel) -> T + Send,
    ) -> PyResult<T> {
        py.detach(|| {
            let state = self.state.lock().map_err(|_| poisoned())?;
            Ok(read(&state))
        })
    }
}

fn poisoned() -> PyErr {
    refuse("an earlier update of this online model panicked; its state is unknown")
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
        tolerance: f64,
    ) -> PyResult<Self> {
        let online = OnlineParams::with_tolerance(tolerance);
        let failure = Failure::default();
        let trained = run_hooked(py, None, &failure, |hook| {
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
        tolerance: f64,
    ) -> PyResult<Self> {
        let online = py
            .detach(|| {
                online::OnlineModel::from_model(
                    (*booster.model).clone(),
                    &params.inner,
                    &dtrain.inner,
                    OnlineParams::with_tolerance(tolerance),
                )
            })
            .or_raise()?;
        Ok(Self::new(online))
    }

    /// Adds `additions`' rows and deletes the rows `deletions`, calling
    /// `on_round(iteration, scores) -> stop` after every updated iteration.
    /// Returns `None` when `on_round` stopped the update, which then changed
    /// nothing, as does an exception or `KeyboardInterrupt` (re-raised).
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
        let (result, stopped) = run_hooked(py, on_round, &failure, |mut hook| {
            let Ok(mut state) = self.state.lock() else {
                return (Err(poisoned()), false);
            };
            let mut stopped = false;
            let result = state.update_with(additions, &deletions, |round| {
                let flow = hook(round);
                stopped |= flow.is_break();
                flow
            });
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
    fn num_row(&self, py: Python<'_>) -> PyResult<usize> {
        self.with(py, |state| state.data().n_rows())
    }

    #[getter]
    fn tolerance(&self, py: Python<'_>) -> PyResult<f64> {
        self.with(py, |state| state.online_params().tolerance)
    }
}
