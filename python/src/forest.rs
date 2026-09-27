//! ForestFlow and ForestDiffusion (`hessboost::diffusion::forest`): tabular
//! generation and imputation with per-noise-level GBDTs.
//!
//! As in `diffusion`, the method crosses the boundary as the JSON of
//! `ForestMethod` (the form forest model files store).

use crate::data::{DMatrix, row_major, to_numpy};
use crate::diffusion::positive;
use crate::errors::{OrRaise, refuse};
use crate::params::{Params, to_python};
use hessboost::diffusion::forest::{self, ColumnKind, ForestMethod, Repaint, Synthetic};
use numpy::{PyArrayDyn, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use serde_json::Value;

/// A fit configuration, passed from Python as a `dict`.
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct ForestRequest {
    /// The JSON of a [`ForestMethod`].
    method: String,
    n_t: usize,
    duplicate_k: usize,
    /// `"continuous"`, `"integer"` or `"categorical"` per column, or `None`
    /// for all continuous.
    column_kinds: Option<Vec<String>>,
    training: Py<Params>,
    num_boost_round: usize,
    seed: u64,
}

fn method_from_json(json: &str) -> PyResult<ForestMethod> {
    serde_json::from_str(json)
        .map_err(|error| refuse(format!("invalid forest method {json}: {error}")))
}

fn method_json(method: ForestMethod) -> PyResult<String> {
    serde_json::to_string(&method)
        .map_err(|error| refuse(format!("cannot describe the forest method: {error}")))
}

/// A column kind from its crate (serde) name.
fn column_kind(name: &str) -> PyResult<ColumnKind> {
    serde_json::from_value(Value::String(name.to_owned()))
        .map_err(|error| refuse(format!("invalid column kind {name:?}: {error}")))
}

/// The crate (serde) name of `kind`.
fn column_kind_name(kind: ColumnKind) -> PyResult<String> {
    match serde_json::to_value(kind) {
        Ok(Value::String(name)) => Ok(name),
        _ => Err(refuse(format!("cannot name the column kind {kind:?}"))),
    }
}

/// A validated fit configuration.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct ForestParams {
    inner: forest::ForestParams,
}

#[pymethods]
impl ForestParams {
    /// Builds and validates a configuration from `request`.
    #[new]
    fn new(request: ForestRequest) -> PyResult<Self> {
        let mut inner = forest::ForestParams::default();
        inner.method = method_from_json(&request.method)?;
        inner.n_t = request.n_t;
        inner.duplicate_k = positive("duplicate_k", request.duplicate_k)?;
        inner.column_kinds = request
            .column_kinds
            .map(|names| names.iter().map(|name| column_kind(name)).collect())
            .transpose()?;
        inner.training = request.training.get().inner.clone();
        inner.num_boost_round = positive("num_boost_round", request.num_boost_round)?;
        inner.seed = request.seed;
        inner.validate().or_raise()?;
        Ok(Self { inner })
    }

    /// The preset `name` (`"default"`, `"diffusion"`) as the `dict` the
    /// constructor takes, with the training configuration as its XGBoost
    /// `dict` instead of `Params`.
    #[staticmethod]
    fn preset<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
        let params = match name {
            "default" => forest::ForestParams::default(),
            "diffusion" => forest::ForestParams::diffusion(),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown forest preset {other:?}; expected \"default\" or \"diffusion\""
                )));
            }
        };
        let dict = PyDict::new(py);
        dict.set_item("method", method_json(params.method)?)?;
        dict.set_item("n_t", params.n_t)?;
        dict.set_item("duplicate_k", params.duplicate_k.get())?;
        let kinds = params
            .column_kinds
            .map(|kinds| {
                kinds
                    .into_iter()
                    .map(column_kind_name)
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        dict.set_item("column_kinds", kinds)?;
        let training = Value::Object(params.training.to_xgboost().or_raise()?);
        dict.set_item("training", to_python(py, &training)?)?;
        dict.set_item("num_boost_round", params.num_boost_round.get())?;
        dict.set_item("seed", params.seed)?;
        Ok(dict)
    }
}

/// `(rows, columns)` values and `(rows,)` labels (class-conditional models).
type SyntheticArrays<'py> = (
    Bound<'py, PyArrayDyn<f32>>,
    Option<Bound<'py, PyArrayDyn<f32>>>,
);

fn synthetic_arrays(py: Python<'_>, synthetic: Synthetic) -> PyResult<SyntheticArrays<'_>> {
    let (rows, columns) = (synthetic.n_rows(), synthetic.n_columns());
    let (values, labels) = synthetic.into_parts();
    let labels = labels
        .map(|labels| to_numpy(py, labels, &[rows]))
        .transpose()?;
    Ok((to_numpy(py, values, &[rows, columns])?, labels))
}

/// A fitted ForestFlow / ForestDiffusion model.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct ForestModel {
    inner: forest::ForestModel,
}

#[pymethods]
impl ForestModel {
    /// Fits a model of the rows of `data`, per class of its labels if any.
    #[staticmethod]
    fn fit(py: Python<'_>, params: &ForestParams, data: &DMatrix) -> PyResult<Self> {
        let inner = py
            .detach(|| forest::ForestModel::fit(&params.inner, &data.inner))
            .or_raise()?;
        Ok(Self { inner })
    }

    /// `n_rows` synthetic rows, with labels drawn from the training
    /// proportions for a class-conditional model.
    fn generate<'py>(
        &self,
        py: Python<'py>,
        n_rows: usize,
        seed: u64,
    ) -> PyResult<SyntheticArrays<'py>> {
        let synthetic = py.detach(|| self.inner.generate(n_rows, seed)).or_raise()?;
        synthetic_arrays(py, synthetic)
    }

    /// One synthetic row per label, from that class's model.
    fn generate_for_labels<'py>(
        &self,
        py: Python<'py>,
        labels: PyReadonlyArray1<'_, f32>,
        seed: u64,
    ) -> PyResult<SyntheticArrays<'py>> {
        let labels = row_major(&labels, "labels")?;
        let synthetic = py
            .detach(|| self.inner.generate_for_labels(labels, seed))
            .or_raise()?;
        synthetic_arrays(py, synthetic)
    }

    /// The missing entries of `data` imputed `n_imputations` times,
    /// `(n_imputations, rows, columns)`; `repaint` is `(resample, jump)`.
    #[pyo3(signature = (data, n_imputations, repaint, seed))]
    fn impute<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        n_imputations: usize,
        repaint: Option<(usize, f64)>,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let repaint = match repaint {
            Some((resample, jump)) => {
                let mut repaint = Repaint::default();
                repaint.resample = positive("repaint.resample", resample)?;
                repaint.jump = jump;
                Some(repaint)
            }
            None => None,
        };
        let imputations = py
            .detach(|| self.inner.impute(&data.inner, n_imputations, repaint, seed))
            .or_raise()?;
        let shape = [
            imputations.n_imputations(),
            imputations.n_rows(),
            imputations.n_columns(),
        ];
        to_numpy(py, imputations.into_vec(), &shape)
    }

    /// Decodes the binary format.
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        let inner = py
            .detach(|| forest::ForestModel::from_bytes(data))
            .or_raise()?;
        Ok(Self { inner })
    }

    /// Decodes the JSON format.
    #[staticmethod]
    fn from_json(py: Python<'_>, json: &str) -> PyResult<Self> {
        let inner = py
            .detach(|| forest::ForestModel::from_json(json))
            .or_raise()?;
        Ok(Self { inner })
    }

    /// The model in the binary format.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| self.inner.to_bytes()).or_raise()?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// The model as JSON.
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.to_json()).or_raise()
    }

    /// The JSON of the model's [`ForestMethod`].
    #[getter]
    fn method(&self) -> PyResult<String> {
        method_json(self.inner.method())
    }

    #[getter]
    fn n_t(&self) -> usize {
        self.inner.n_t()
    }

    #[getter]
    fn n_columns(&self) -> usize {
        self.inner.n_columns()
    }

    /// The sorted class labels (empty for an unconditional model).
    #[getter]
    fn classes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let classes = self.inner.classes();
        to_numpy(py, classes.to_vec(), &[classes.len()])
    }
}
