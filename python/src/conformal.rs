//! Conformal prediction intervals (`hessboost::conformal`) around a model
//! the calibrator keeps alive.

use crate::booster::Booster;
use crate::data::{DMatrix, to_numpy};
use crate::errors::OrRaise;
use hessboost::conformal;
use hessboost::model::BoostedModel;
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use self_cell::self_cell;
use std::sync::Arc;

/// The model(s) a calibrator borrows: a band's lower and upper model, or
/// one model twice.
struct Models {
    lower: Arc<BoostedModel>,
    upper: Arc<BoostedModel>,
}

type SplitRef<'a> = conformal::SplitConformal<'a>;
type QuantileRef<'a> = conformal::ConformalizedQuantile<'a>;

self_cell!(
    struct SplitCell {
        owner: Arc<BoostedModel>,
        #[covariant]
        dependent: SplitRef,
    }
);

self_cell!(
    struct QuantileCell {
        owner: Models,
        #[covariant]
        dependent: QuantileRef,
    }
);

/// `(rows, 2)` `[lower, upper]` bounds.
fn intervals(py: Python<'_>, bounds: Vec<(f32, f32)>) -> PyResult<Bound<'_, PyArrayDyn<f32>>> {
    let rows = bounds.len();
    let values = bounds
        .into_iter()
        .flat_map(|(lower, upper)| [lower, upper])
        .collect();
    to_numpy(py, values, &[rows, 2])
}

/// Split-conformal intervals `[f(x) - Q, f(x) + Q]` around a single-output
/// model.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct SplitConformal {
    cell: SplitCell,
}

#[pymethods]
impl SplitConformal {
    /// Calibrates `booster` on the labelled rows of `calibration` at
    /// miscoverage `alpha`.
    #[staticmethod]
    fn calibrate(
        py: Python<'_>,
        booster: &Booster,
        calibration: &DMatrix,
        alpha: f64,
    ) -> PyResult<Self> {
        let model = Arc::clone(&booster.model);
        let cell = py
            .detach(|| {
                SplitCell::try_new(model, |model| {
                    conformal::SplitConformal::calibrate(model, &calibration.inner, alpha)
                })
            })
            .or_raise()?;
        Ok(Self { cell })
    }

    fn predict_interval<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let bounds = py
            .detach(|| self.cell.borrow_dependent().predict_interval(&data.inner))
            .or_raise()?;
        intervals(py, bounds)
    }

    #[getter]
    fn half_width(&self) -> f64 {
        self.cell.borrow_dependent().half_width()
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.cell.borrow_dependent().alpha()
    }

    #[getter]
    fn n_calibration(&self) -> usize {
        self.cell.borrow_dependent().n_calibration()
    }
}

/// Conformalized quantile regression: a band `[q_lo(x), q_hi(x)]` adjusted
/// to `[q_lo(x) - Q, q_hi(x) + Q]`.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct ConformalizedQuantile {
    cell: QuantileCell,
}

/// Where the band comes from.
#[derive(Clone, Copy)]
enum Band {
    /// Two single-output models.
    Models,
    /// Two outputs of one model.
    Outputs(usize, usize),
    /// A `dist:*` model's central quantiles.
    Distribution,
}

impl ConformalizedQuantile {
    fn build(
        py: Python<'_>,
        models: Models,
        band: Band,
        calibration: &DMatrix,
        alpha: f64,
    ) -> PyResult<Self> {
        let data = &calibration.inner;
        let cell = py
            .detach(|| {
                QuantileCell::try_new(models, |models| match band {
                    Band::Models => conformal::ConformalizedQuantile::calibrate(
                        &models.lower,
                        &models.upper,
                        data,
                        alpha,
                    ),
                    Band::Outputs(lower, upper) => {
                        conformal::ConformalizedQuantile::calibrate_outputs(
                            &models.lower,
                            lower,
                            upper,
                            data,
                            alpha,
                        )
                    }
                    Band::Distribution => conformal::ConformalizedQuantile::calibrate_distribution(
                        &models.lower,
                        data,
                        alpha,
                    ),
                })
            })
            .or_raise()?;
        Ok(Self { cell })
    }

    fn single(booster: &Booster) -> Models {
        Models {
            lower: Arc::clone(&booster.model),
            upper: Arc::clone(&booster.model),
        }
    }
}

#[pymethods]
impl ConformalizedQuantile {
    /// A band from two single-output quantile models.
    #[staticmethod]
    fn calibrate(
        py: Python<'_>,
        models: (PyRef<'_, Booster>, PyRef<'_, Booster>),
        calibration: &DMatrix,
        alpha: f64,
    ) -> PyResult<Self> {
        let models = Models {
            lower: Arc::clone(&models.0.model),
            upper: Arc::clone(&models.1.model),
        };
        Self::build(py, models, Band::Models, calibration, alpha)
    }

    /// A band from outputs `outputs = (lower, upper)` of one model.
    #[staticmethod]
    fn calibrate_outputs(
        py: Python<'_>,
        booster: &Booster,
        outputs: (usize, usize),
        calibration: &DMatrix,
        alpha: f64,
    ) -> PyResult<Self> {
        let band = Band::Outputs(outputs.0, outputs.1);
        Self::build(py, Self::single(booster), band, calibration, alpha)
    }

    /// A band from a `dist:*` model's `alpha / 2` and `1 - alpha / 2`
    /// quantiles.
    #[staticmethod]
    fn calibrate_distribution(
        py: Python<'_>,
        booster: &Booster,
        calibration: &DMatrix,
        alpha: f64,
    ) -> PyResult<Self> {
        Self::build(
            py,
            Self::single(booster),
            Band::Distribution,
            calibration,
            alpha,
        )
    }

    fn predict_interval<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let bounds = py
            .detach(|| self.cell.borrow_dependent().predict_interval(&data.inner))
            .or_raise()?;
        intervals(py, bounds)
    }

    #[getter]
    fn correction(&self) -> f64 {
        self.cell.borrow_dependent().correction()
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.cell.borrow_dependent().alpha()
    }

    #[getter]
    fn n_calibration(&self) -> usize {
        self.cell.borrow_dependent().n_calibration()
    }
}
