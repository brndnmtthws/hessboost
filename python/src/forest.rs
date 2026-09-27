//! ForestFlow and ForestDiffusion (`hessboost::diffusion::forest`): tabular
//! generation and imputation with per-noise-level GBDTs.
//!
//! As in `diffusion`, the method crosses the boundary as the JSON of
//! `ForestMethod` (the form forest model files store).

use crate::codec::{encode_bytes, from_json, to_json};
use crate::data::{DMatrix, row_major, to_numpy};
use crate::diffusion::positive;
use crate::errors::{DetachExt, OrRaise, refuse};
use crate::params::{Params, to_python};
use hessboost::diffusion::forest::{
    self, ColumnKind, ImputeOptions, NoiseLevels, Repaint, Synthetic,
};
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

const METHOD: &str = "forest method";

/// A column kind from its crate (serde) name.
fn column_kind(name: &str) -> PyResult<ColumnKind> {
    match name {
        "continuous" => Ok(ColumnKind::Continuous),
        "integer" => Ok(ColumnKind::Integer),
        "categorical" => Ok(ColumnKind::Categorical),
        other => Err(refuse(format!(
            "invalid column kind {other:?}; expected \"continuous\", \"integer\" or \"categorical\""
        ))),
    }
}

/// The crate (serde) name of `kind`.
fn column_kind_name(kind: ColumnKind) -> PyResult<&'static str> {
    match kind {
        ColumnKind::Continuous => Ok("continuous"),
        ColumnKind::Integer => Ok("integer"),
        ColumnKind::Categorical => Ok("categorical"),
        // `ColumnKind` is `#[non_exhaustive]`: a kind added in the crate
        // is refused until it has a name here.
        other => Err(refuse(format!("cannot name the column kind {other:?}"))),
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
        inner.method = from_json(&request.method, METHOD)?;
        inner.n_t = NoiseLevels::try_from(request.n_t).or_raise()?;
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

    /// The preset `name` (`"forest_flow"`, `"forest_diffusion"`) as the `dict` the
    /// constructor takes, with the training configuration as its XGBoost
    /// `dict` instead of `Params`.
    #[staticmethod]
    fn preset<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
        let params = match name {
            "forest_flow" => forest::ForestParams::forest_flow(),
            "forest_diffusion" => forest::ForestParams::forest_diffusion(),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown forest preset {other:?}; expected \"forest_flow\" or \
                     \"forest_diffusion\""
                )));
            }
        };
        let dict = PyDict::new(py);
        dict.set_item("method", to_json(&params.method, METHOD)?)?;
        dict.set_item("n_t", params.n_t.get())?;
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
        let inner = py.detached(|| forest::ForestModel::fit(&params.inner, &data.inner))?;
        Ok(Self { inner })
    }

    /// `n_rows` synthetic rows, with labels drawn from the training
    /// proportions for a class-conditional model.
    fn sample<'py>(
        &self,
        py: Python<'py>,
        n_rows: usize,
        seed: u64,
    ) -> PyResult<SyntheticArrays<'py>> {
        let synthetic = py.detached(|| self.inner.sample(n_rows, seed))?;
        synthetic_arrays(py, synthetic)
    }

    /// One synthetic row per label, from that class's model.
    fn sample_for_labels<'py>(
        &self,
        py: Python<'py>,
        labels: PyReadonlyArray1<'_, f32>,
        seed: u64,
    ) -> PyResult<SyntheticArrays<'py>> {
        let labels = row_major(&labels, "labels")?;
        let synthetic = py.detached(|| self.inner.sample_for_labels(labels, seed))?;
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
        let mut options = ImputeOptions::seeded(seed);
        if let Some((resample, jump)) = repaint {
            let mut repaint = Repaint::default();
            repaint.resample = positive("repaint.resample", resample)?;
            repaint.jump = jump;
            options = options.with_repaint(repaint);
        }
        let imputations =
            py.detached(|| self.inner.impute(&data.inner, n_imputations, &options))?;
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
        let inner = py.detached(|| forest::ForestModel::from_bytes(data))?;
        Ok(Self { inner })
    }

    /// Decodes the JSON format.
    #[staticmethod]
    fn from_json(py: Python<'_>, json: &str) -> PyResult<Self> {
        let inner = py.detached(|| forest::ForestModel::from_json(json))?;
        Ok(Self { inner })
    }

    /// The model in the binary format.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        encode_bytes(py, || self.inner.to_bytes())
    }

    /// The model as JSON.
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detached(|| self.inner.to_json())
    }

    /// The JSON of the model's [`ForestMethod`].
    #[getter]
    fn method(&self) -> PyResult<String> {
        to_json(&self.inner.method(), METHOD)
    }

    #[getter]
    fn n_t(&self) -> usize {
        self.inner.n_t().get()
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
