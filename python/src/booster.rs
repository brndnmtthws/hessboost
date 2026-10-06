//! `Booster`: a trained or loaded model, its prediction variants, feature
//! importance, slicing, and every model format.

use crate::codec;
use crate::compact::CompactModel;
use crate::data::{DMatrix, to_numpy};
use crate::dist::Distributions;
use crate::errors::{DetachExt, refuse};
use crate::gpu::GpuModel;
use hessboost::model::{
    BoostedModel, Contributions, ImportanceType, Interactions, Iterations, Predictions,
};
use numpy::PyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::sync::Arc;

/// What a prediction call returns.
#[derive(Clone, Copy)]
enum Kind {
    Value,
    Margin,
    Contribs,
    Interactions,
}

/// `(rows,)` for one value per row, else `(rows, width)`.
pub(crate) fn dense<T>(predictions: Predictions<T>) -> (Vec<T>, Vec<usize>) {
    let (rows, width) = (predictions.n_rows(), predictions.width());
    let shape = if width == 1 {
        vec![rows]
    } else {
        vec![rows, width]
    };
    (predictions.into_vec(), shape)
}

/// `(rows, features + 1)`, with an output axis for multi-output models.
fn contributions(contribs: Contributions) -> (Vec<f32>, Vec<usize>) {
    let (rows, outputs, width) = (
        contribs.n_rows(),
        contribs.n_outputs(),
        contribs.n_features() + 1,
    );
    let shape = if outputs == 1 {
        vec![rows, width]
    } else {
        vec![rows, outputs, width]
    };
    (contribs.into_vec(), shape)
}

/// `(rows, features + 1, features + 1)`, with an output axis for
/// multi-output models.
fn interactions(interactions: Interactions) -> (Vec<f32>, Vec<usize>) {
    let (rows, outputs, width) = (
        interactions.n_rows(),
        interactions.n_outputs(),
        interactions.n_features() + 1,
    );
    let shape = if outputs == 1 {
        vec![rows, width, width]
    } else {
        vec![rows, outputs, width, width]
    };
    (interactions.into_vec(), shape)
}

/// A trained model, shared read-only.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct Booster {
    pub(crate) model: Arc<BoostedModel>,
}

impl Booster {
    pub(crate) fn new(model: BoostedModel) -> Self {
        Self {
            model: Arc::new(model),
        }
    }
}

/// XGBoost's `iteration_range` as iterations of `model`: `(begin, 0)` runs
/// through the last iteration, and `None` is the method's `default`.
pub(crate) fn iterations(
    model: &BoostedModel,
    range: Option<(usize, usize)>,
    default: Iterations,
) -> Iterations {
    range.map_or(default, |(begin, end)| {
        let end = if end == 0 {
            model.num_boost_rounds()
        } else {
            end
        };
        (begin..end).into()
    })
}

#[pymethods]
impl Booster {
    /// Decodes a model in `format` (`"auto"` detects it from the bytes).
    #[staticmethod]
    fn load(py: Python<'_>, data: &[u8], format: &str) -> PyResult<Self> {
        codec::decode(py, data, format, BoostedModel::decode).map(Self::new)
    }

    /// The model encoded in `format`.
    fn save<'py>(&self, py: Python<'py>, format: &str) -> PyResult<Bound<'py, PyBytes>> {
        codec::encode(py, format, |format| self.model.encode(format))
    }

    /// Predictions of `kind` (`value`, `margin`, `contribs`,
    /// `interactions`) shaped as XGBoost's Python package shapes them.
    #[pyo3(signature = (data, kind, iteration_range=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        kind: &str,
        iteration_range: Option<(usize, usize)>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let kind = match kind {
            "value" => Kind::Value,
            "margin" => Kind::Margin,
            "contribs" => Kind::Contribs,
            "interactions" => Kind::Interactions,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown prediction kind {other:?}"
                )));
            }
        };
        let model = &*self.model;
        let matrix = &data.inner;
        // `None`: through `best_iteration` after early stopping.
        let iterations = iterations(model, iteration_range, Iterations::Best);
        let (values, shape) = py.detached(|| -> hessboost::error::Result<_> {
            Ok(match kind {
                Kind::Value => dense(model.predict(matrix, iterations)?),
                Kind::Margin => dense(model.predict_margin(matrix, iterations)?),
                Kind::Contribs => contributions(model.predict_contribs(matrix, iterations)?),
                Kind::Interactions => interactions(model.predict_interactions(matrix, iterations)?),
            })
        })?;
        to_numpy(py, values, &shape)
    }

    /// The leaf each row reaches in every tree, `(rows, trees)`; ranges
    /// start at iteration 0 and default to every iteration.
    #[pyo3(signature = (data, iteration_range=None))]
    fn predict_leaf<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        iteration_range: Option<(usize, usize)>,
    ) -> PyResult<Bound<'py, PyArrayDyn<i32>>> {
        // `None`: every iteration, regardless of early stopping.
        let iterations = iterations(&self.model, iteration_range, (..).into());
        let leaves = py.detached(|| self.model.predict_leaf(&data.inner, iterations))?;
        let (rows, trees) = (leaves.n_rows(), leaves.width());
        // The array stays `int32`, as it always was; a leaf id past it (a
        // tree of more than 2^31 nodes) is refused rather than clipped.
        let leaves = leaves
            .into_vec()
            .into_iter()
            .map(i32::try_from)
            .collect::<Result<_, _>>()
            .map_err(|_| refuse("a leaf id exceeds the int32 range of predict_leaf's array"))?;
        to_numpy(py, leaves, &[rows, trees])
    }

    /// The distribution a `dist:*` model predicts for every row.
    #[pyo3(signature = (data, iteration_range=None))]
    fn predict_distribution(
        &self,
        py: Python<'_>,
        data: &DMatrix,
        iteration_range: Option<(usize, usize)>,
    ) -> PyResult<Distributions> {
        // `None`: through `best_iteration` after early stopping.
        let iterations = iterations(&self.model, iteration_range, Iterations::Best);
        let dists = py.detached(|| self.model.predict_distribution(&data.inner, iterations))?;
        Distributions::new(dists)
    }

    /// The predictions (`output_margin`: raw margins) of a virtual ensemble
    /// of `count` members, `(count, rows)` or `(count, rows, width)`, and
    /// each member's iteration count.
    fn predict_virtual_ensembles<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        count: usize,
        output_margin: bool,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, Vec<usize>)> {
        let ensembles = py.detached(|| self.model.predict_virtual_ensembles(&data.inner, count))?;
        let (members, rows) = (ensembles.n_members(), ensembles.n_rows());
        let iterations = ensembles.iterations().to_vec();
        let (width, values) = if output_margin {
            (ensembles.margin_width(), ensembles.into_margins())
        } else {
            (ensembles.width(), ensembles.into_predictions())
        };
        let shape = if width == 1 {
            vec![members, rows]
        } else {
            vec![members, rows, width]
        };
        Ok((to_numpy(py, values, &shape)?, iterations))
    }

    /// A virtual ensemble's `(mean, knowledge, data, total)` uncertainty,
    /// each `(rows,)` or `(rows, width)` with its own width (a multiclass
    /// `mean` per class, its uncertainties per row); `data` and `total`
    /// `None` for plain regression.
    #[allow(
        clippy::type_complexity,
        reason = "the tuple is the extension's return shape; the public class wraps it"
    )]
    fn predict_uncertainty<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        count: usize,
    ) -> PyResult<(
        Bound<'py, PyArrayDyn<f64>>,
        Bound<'py, PyArrayDyn<f64>>,
        Option<Bound<'py, PyArrayDyn<f64>>>,
        Option<Bound<'py, PyArrayDyn<f64>>>,
    )> {
        let uncertainty = py.detached(|| self.model.predict_uncertainty(&data.inner, count))?;
        let array = |values: Predictions<f64>| {
            let (values, shape) = dense(values);
            to_numpy(py, values, &shape)
        };
        Ok((
            array(uncertainty.mean)?,
            array(uncertainty.knowledge)?,
            uncertainty.data.map(array).transpose()?,
            uncertainty.total.map(array).transpose()?,
        ))
    }

    /// `{feature index: score}` for every feature used in a split.
    fn feature_importance<'py>(
        &self,
        py: Python<'py>,
        importance_type: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let kind = match importance_type {
            "weight" => ImportanceType::Weight,
            "gain" => ImportanceType::Gain,
            "total_gain" => ImportanceType::TotalGain,
            "cover" => ImportanceType::Cover,
            "total_cover" => ImportanceType::TotalCover,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown importance_type {other:?}; expected \"weight\", \"gain\", \
                     \"total_gain\", \"cover\" or \"total_cover\""
                )));
            }
        };
        let dict = PyDict::new(py);
        for (feature, score) in self.model.feature_importance(kind) {
            dict.set_item(feature, score)?;
        }
        Ok(dict)
    }

    /// Every `step`-th iteration of `begin..end`.
    fn slice(&self, py: Python<'_>, begin: usize, end: usize, step: usize) -> PyResult<Self> {
        let model = py.detached(|| self.model.slice(begin..end, step))?;
        Ok(Self::new(model))
    }

    /// Lays this model out for GPU batch prediction on `device` (`"metal"`
    /// on macOS, `"wgpu"` anywhere; `None`: Metal on macOS, wgpu
    /// elsewhere). `gblinear` and `linear_tree` models are refused, as they
    /// do not predict through the forest.
    fn to_gpu(&self, py: Python<'_>, device: Option<&str>) -> PyResult<GpuModel> {
        GpuModel::build(py, &self.model, device)
    }

    /// This model in the bit-packed compact layout (the trees prediction
    /// uses by default).
    fn to_compact(&self, py: Python<'_>) -> PyResult<CompactModel> {
        py.detached(|| self.model.to_compact())
            .map(CompactModel::new)
    }

    /// Native versus compact size and the compact dictionary statistics.
    fn size_report<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let report = py.detached(|| self.model.size_report())?;
        let dict = PyDict::new(py);
        dict.set_item("native_bytes", report.native_bytes)?;
        dict.set_item("compact_bytes", report.compact_bytes)?;
        dict.set_item("trees", report.trees)?;
        dict.set_item("splits", report.splits)?;
        dict.set_item("leaves", report.leaves)?;
        dict.set_item("used_features", report.used_features)?;
        dict.set_item("thresholds", report.thresholds)?;
        dict.set_item("leaf_values", report.leaf_values)?;
        Ok(dict)
    }

    #[getter]
    fn objective(&self) -> &str {
        self.model.objective().name()
    }

    #[getter]
    fn num_features(&self) -> usize {
        self.model.n_features()
    }

    #[getter]
    fn num_outputs(&self) -> usize {
        self.model.n_outputs()
    }

    #[getter]
    fn num_targets(&self) -> usize {
        self.model.n_targets()
    }

    #[getter]
    fn num_boosted_rounds(&self) -> usize {
        self.model.num_boost_rounds()
    }

    #[getter]
    fn num_trees(&self) -> usize {
        self.model.num_trees()
    }

    #[getter]
    fn num_parallel_tree(&self) -> usize {
        self.model.num_parallel_tree()
    }

    #[getter]
    fn best_iteration(&self) -> Option<usize> {
        self.model.best_iteration()
    }

    /// Per-output intercepts in margin space.
    #[getter]
    fn base_margins(&self) -> Vec<f32> {
        self.model.base_scores().to_vec()
    }

    #[getter]
    fn vector_leaves(&self) -> bool {
        self.model.has_vector_leaves()
    }

    /// The Boulevard record of a `booster = boulevard` model, else `None`.
    #[getter]
    fn boulevard<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        crate::inference::boulevard_info(py, &self.model)
    }

    /// The EBM record of a `booster = ebm` model, else `None`.
    #[getter]
    fn ebm<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        crate::ebm::ebm_info(py, &self.model)
    }
}
