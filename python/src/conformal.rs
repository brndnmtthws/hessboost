//! Conformal prediction intervals (`hessboost::conformal`) around a model
//! the calibrator keeps alive.

use crate::booster::Booster;
use crate::data::{DMatrix, intervals_to_numpy, row_major};
use crate::errors::{DetachExt, OrRaise, refuse};
use hessboost::conformal;
use hessboost::model::BoostedModel;
use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArray1};
use pyo3::prelude::*;
use self_cell::self_cell;
use std::sync::Arc;
use std::sync::{Mutex, MutexGuard};

/// An immutable forecast-time receipt, accepted only by its creating calibrator.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct CalibrationForecast {
    inner: conformal::online::Forecast,
}

#[pymethods]
impl CalibrationForecast {
    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold()
    }

    #[getter]
    fn index(&self) -> u64 {
        self.inner.index()
    }
}

/// Online score calibration; mutable state is accessed only while detached.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct OnlineCalibrator {
    state: Mutex<conformal::online::OnlineCalibrator>,
    owner: u32,
}

impl OnlineCalibrator {
    fn state(&self) -> PyResult<MutexGuard<'_, conformal::online::OnlineCalibrator>> {
        // No callbacks or Rayon work run under this lock. Block only while
        // detached, but never wait on a mutex inherited across a fork.
        if self.owner != std::process::id() {
            return Err(refuse("create a new online calibrator in the forked child"));
        }
        self.state
            .lock()
            .map_err(|_| refuse("an earlier online calibration call panicked"))
    }
}

#[pymethods]
impl OnlineCalibrator {
    #[new]
    fn new(
        py: Python<'_>,
        alpha: f64,
        learning_rate: f64,
        initial_threshold: f64,
        dimensions: (usize, usize),
        coefficient_bound: f64,
    ) -> PyResult<Self> {
        let state = py
            .detach(|| {
                let params = conformal::online::OnlineParams::builder(alpha, learning_rate)
                    .initial_threshold(initial_threshold)
                    .lags(dimensions.0)
                    .covariates(dimensions.1)
                    .coefficient_bound(coefficient_bound)
                    .build()?;
                conformal::online::OnlineCalibrator::new(params)
            })
            .or_raise()?;
        Ok(Self {
            state: Mutex::new(state),
            owner: std::process::id(),
        })
    }

    fn predict(
        &self,
        py: Python<'_>,
        covariates: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<CalibrationForecast> {
        let covariates = row_major(&covariates, "covariates")?;
        py.detach(|| {
            let inner = self.state()?.predict(covariates).or_raise()?;
            Ok(CalibrationForecast { inner })
        })
    }

    fn update(&self, py: Python<'_>, forecast: &CalibrationForecast, score: f64) -> PyResult<()> {
        py.detach(|| self.state()?.update(&forecast.inner, score).or_raise())
    }

    #[getter]
    fn alpha(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| Ok(self.state()?.params().alpha()))
    }

    #[getter]
    fn learning_rate(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| Ok(self.state()?.params().learning_rate()))
    }

    #[getter]
    fn n_updates(&self, py: Python<'_>) -> PyResult<u64> {
        py.detach(|| Ok(self.state()?.n_updates()))
    }

    #[getter]
    fn pending(&self, py: Python<'_>) -> PyResult<u64> {
        py.detach(|| Ok(self.state()?.pending()))
    }

    #[getter]
    fn intercept(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| Ok(self.state()?.intercept()))
    }

    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values: Vec<f64> =
            py.detach(|| Ok::<_, PyErr>(self.state()?.coefficients().to_vec()))?;
        Ok(values.into_pyarray(py))
    }
}
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
        let cell = py.detached(|| {
            SplitCell::try_new(model, |model| {
                conformal::SplitConformal::calibrate(model, &calibration.inner, alpha)
            })
        })?;
        Ok(Self { cell })
    }

    fn predict_interval<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let bounds = py.detached(|| self.cell.borrow_dependent().predict_interval(&data.inner))?;
        intervals_to_numpy(py, bounds)
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
        let cell = py.detached(|| {
            QuantileCell::try_new(models, |models| match band {
                Band::Models => conformal::ConformalizedQuantile::calibrate(
                    &models.lower,
                    &models.upper,
                    data,
                    alpha,
                ),
                Band::Outputs(lower, upper) => conformal::ConformalizedQuantile::calibrate_outputs(
                    &models.lower,
                    lower,
                    upper,
                    data,
                    alpha,
                ),
                Band::Distribution => conformal::ConformalizedQuantile::calibrate_distribution(
                    &models.lower,
                    data,
                    alpha,
                ),
            })
        })?;
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
        let bounds = py.detached(|| self.cell.borrow_dependent().predict_interval(&data.inner))?;
        intervals_to_numpy(py, bounds)
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
