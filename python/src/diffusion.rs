//! Conditional diffusion and flow matching (`hessboost::diffusion`): the fit
//! configuration, the fitted model, and Monte Carlo summaries of its draws.
//!
//! The method's settings cross the boundary as the JSON of
//! `hessboost::diffusion::Method` (the form diffusion model files store),
//! which the Python layer's dataclasses map onto one to one; the GBDTs'
//! training parameters as validated `Params`.

use crate::data::{DMatrix, row_major, to_numpy};
use crate::errors::{OrRaise, refuse};
use crate::params::{Params, to_python};
use hessboost::config::TrainingParams;
use hessboost::diffusion::{self, EarlyStopping, Method, Quantiles, Residualizer, Samples};
use hessboost::model::Predictions;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use serde_json::Value;
use std::num::NonZeroUsize;

/// A fit configuration, passed from Python as a `dict`.
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct ParamsRequest {
    /// The JSON of a [`Method`].
    method: String,
    n_repeats: usize,
    n_steps: usize,
    training: Py<Params>,
    num_boost_round: usize,
    /// `(rounds, eval_fraction)`.
    early_stopping: Option<(usize, f64)>,
    /// `(folds, training, num_boost_round)`.
    residualizer: Option<(usize, Py<Params>, usize)>,
    seed: u64,
}

/// `value` as a positive count, refused under `name` when it is `0`.
fn positive(name: &str, value: usize) -> PyResult<NonZeroUsize> {
    NonZeroUsize::new(value).ok_or_else(|| refuse(format!("{name} must be at least 1, got 0")))
}

fn method_from_json(json: &str) -> PyResult<Method> {
    serde_json::from_str(json)
        .map_err(|error| refuse(format!("invalid diffusion method {json}: {error}")))
}

fn method_json(method: &Method) -> PyResult<String> {
    serde_json::to_string(method)
        .map_err(|error| refuse(format!("cannot describe the diffusion method: {error}")))
}

/// `params` in XGBoost's flat form, as a `dict`.
fn training_dict<'py>(py: Python<'py>, params: &TrainingParams) -> PyResult<Bound<'py, PyAny>> {
    to_python(py, &Value::Object(params.to_xgboost().or_raise()?))
}

/// A validated fit configuration.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct DiffusionParams {
    inner: diffusion::DiffusionParams,
}

#[pymethods]
impl DiffusionParams {
    /// Builds and validates a configuration from `request`.
    #[new]
    fn new(request: ParamsRequest) -> PyResult<Self> {
        let mut inner = diffusion::DiffusionParams::default();
        inner.method = method_from_json(&request.method)?;
        inner.n_repeats = positive("n_repeats", request.n_repeats)?;
        inner.n_steps = positive("n_steps", request.n_steps)?;
        inner.training = request.training.get().inner.clone();
        inner.num_boost_round = positive("num_boost_round", request.num_boost_round)?;
        inner.early_stopping = match request.early_stopping {
            Some((rounds, eval_fraction)) => {
                let mut stop = EarlyStopping::default();
                stop.rounds = positive("early_stopping.rounds", rounds)?;
                stop.eval_fraction = eval_fraction;
                Some(stop)
            }
            None => None,
        };
        inner.residualizer = match request.residualizer {
            Some((folds, training, num_boost_round)) => {
                let mut residualizer = Residualizer::default();
                residualizer.folds = folds;
                residualizer.training = training.get().inner.clone();
                residualizer.num_boost_round =
                    positive("residualizer.num_boost_round", num_boost_round)?;
                Some(residualizer)
            }
            None => None,
        };
        inner.seed = request.seed;
        inner.validate().or_raise()?;
        Ok(Self { inner })
    }

    /// The preset `name` (`"default"`, `"treeffuser"`, `"flow_matching"`) as
    /// the `dict` the constructor takes, with every training configuration
    /// as its XGBoost `dict` instead of `Params`.
    #[staticmethod]
    fn preset<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
        let params = match name {
            "default" => diffusion::DiffusionParams::default(),
            "treeffuser" => diffusion::DiffusionParams::treeffuser(),
            "flow_matching" => diffusion::DiffusionParams::flow_matching(),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown diffusion preset {other:?}; expected \"default\", \"treeffuser\" \
                     or \"flow_matching\""
                )));
            }
        };
        let dict = PyDict::new(py);
        dict.set_item("method", method_json(&params.method)?)?;
        dict.set_item("n_repeats", params.n_repeats.get())?;
        dict.set_item("n_steps", params.n_steps.get())?;
        dict.set_item("training", training_dict(py, &params.training)?)?;
        dict.set_item("num_boost_round", params.num_boost_round.get())?;
        dict.set_item(
            "early_stopping",
            params
                .early_stopping
                .map(|stop| (stop.rounds.get(), stop.eval_fraction)),
        )?;
        let residualizer = match &params.residualizer {
            Some(r) => Some((
                r.folds,
                training_dict(py, &r.training)?,
                r.num_boost_round.get(),
            )),
            None => None,
        };
        dict.set_item("residualizer", residualizer)?;
        dict.set_item("seed", params.seed)?;
        Ok(dict)
    }
}

/// A fitted conditional diffusion or flow-matching model.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct DiffusionModel {
    inner: diffusion::DiffusionModel,
}

#[pymethods]
impl DiffusionModel {
    /// Fits a model of `p(y | x)` to the features and labels of `data`.
    #[staticmethod]
    fn fit(py: Python<'_>, params: &DiffusionParams, data: &DMatrix) -> PyResult<Self> {
        let inner = py
            .detach(|| diffusion::DiffusionModel::fit(&params.inner, &data.inner))
            .or_raise()?;
        Ok(Self { inner })
    }

    /// `n_samples` draws for every row of `data`, `(rows, n_samples,
    /// outputs)`.
    fn sample<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        n_samples: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let samples = py
            .detach(|| self.inner.sample(&data.inner, n_samples, seed))
            .or_raise()?;
        let shape = [samples.n_rows(), samples.n_samples(), samples.n_outputs()];
        to_numpy(py, samples.into_vec(), &shape)
    }

    /// A copy of the model that samples with `n_steps` integration steps.
    fn with_n_steps(&self, py: Python<'_>, n_steps: usize) -> PyResult<Self> {
        let n_steps = positive("n_steps", n_steps)?;
        let mut inner = py.detach(|| self.inner.clone());
        inner.set_n_steps(n_steps);
        Ok(Self { inner })
    }

    /// Decodes the native binary format.
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        let inner = py
            .detach(|| diffusion::DiffusionModel::from_bytes(data))
            .or_raise()?;
        Ok(Self { inner })
    }

    /// Decodes the JSON format.
    #[staticmethod]
    fn from_json(py: Python<'_>, json: &str) -> PyResult<Self> {
        let inner = py
            .detach(|| diffusion::DiffusionModel::from_json(json))
            .or_raise()?;
        Ok(Self { inner })
    }

    /// The model in the native binary format.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| self.inner.to_bytes()).or_raise()?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// The model as JSON.
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.to_json()).or_raise()
    }

    /// The JSON of the model's [`Method`].
    #[getter]
    fn method(&self) -> PyResult<String> {
        method_json(self.inner.method())
    }

    #[getter]
    fn n_steps(&self) -> usize {
        self.inner.n_steps().get()
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.inner.n_features()
    }

    #[getter]
    fn n_outputs(&self) -> usize {
        self.inner.n_outputs()
    }

    #[getter]
    fn is_residualized(&self) -> bool {
        self.inner.is_residualized()
    }
}

/// Runs `summary` on the `(rows, samples, outputs)` draws of `samples`, with
/// the GIL released.
fn summarize<T: Send>(
    py: Python<'_>,
    samples: &PyReadonlyArrayDyn<'_, f32>,
    summary: impl FnOnce(&Samples) -> hessboost::error::Result<T> + Send,
) -> PyResult<(T, [usize; 2])> {
    let &[rows, n_samples, outputs] = samples.shape() else {
        return Err(refuse(format!(
            "samples must be a (rows, samples, outputs) array, got shape {:?}",
            samples.shape()
        )));
    };
    let values = row_major(samples, "samples")?;
    let out = py
        .detach(|| summary(&Samples::new(values.to_vec(), n_samples, outputs)?))
        .or_raise()?;
    Ok((out, [rows, outputs]))
}

/// The Monte Carlo mean of each row's draws, `(rows, outputs)`.
#[pyfunction]
pub(crate) fn samples_mean<'py>(
    py: Python<'py>,
    samples: PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let (mean, shape) = summarize(py, &samples, |s| Ok(s.mean().into_vec()))?;
    to_numpy(py, mean, &shape)
}

/// Empirical quantiles of each row's draws, `(rows, levels, outputs)`.
#[pyfunction]
pub(crate) fn samples_quantiles<'py>(
    py: Python<'py>,
    samples: PyReadonlyArrayDyn<'_, f32>,
    levels: Vec<f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let k = levels.len();
    let (quantiles, [rows, outputs]) = summarize(py, &samples, |s| {
        s.quantiles(&levels).map(Quantiles::into_vec)
    })?;
    to_numpy(py, quantiles, &[rows, k, outputs])
}

/// The ensemble CRPS of each label under its row's draws, `(rows, outputs)`.
#[pyfunction]
pub(crate) fn samples_crps<'py>(
    py: Python<'py>,
    samples: PyReadonlyArrayDyn<'_, f32>,
    labels: PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let labels = row_major(&labels, "labels")?;
    let (crps, shape) = summarize(py, &samples, |s| s.crps(labels).map(Predictions::into_vec))?;
    to_numpy(py, crps, &shape)
}
