//! Boulevard inference (`hessboost::inference`) around a model the object
//! keeps alive: standard errors and intervals for `f(x)`, and the honest
//! leaf refit.

use crate::booster::Booster;
use crate::data::{DMatrix, to_numpy};
use crate::errors::{OrRaise, refuse};
use hessboost::conformal::Interval;
use hessboost::data::DMatrix as RustMatrix;
use hessboost::inference::{self, KernelSolver, NoiseVariance};
use hessboost::model::BoostedModel;
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use self_cell::self_cell;
use std::sync::Arc;

/// What a fitted inference borrows: the model, and the holdout rows its
/// noise variance (and calibrated intervals) come from, if any.
struct Owner {
    model: Arc<BoostedModel>,
    holdout: Option<RustMatrix>,
}

type InferenceRef<'a> = inference::BoulevardInference<'a>;

self_cell!(
    struct Cell {
        owner: Owner,
        #[covariant]
        dependent: InferenceRef,
    }
);

/// `(rows, 2)` `[lower, upper]` bounds.
pub(crate) fn intervals(
    py: Python<'_>,
    bounds: Vec<Interval<f64>>,
) -> PyResult<Bound<'_, PyArrayDyn<f64>>> {
    let rows = bounds.len();
    let values = bounds
        .into_iter()
        .flat_map(|iv| [iv.lower, iv.upper])
        .collect();
    to_numpy(py, values, &[rows, 2])
}

/// The kernel solver: Nyström with `landmarks`, else exact.
fn solver(landmarks: Option<usize>, seed: u64) -> KernelSolver {
    landmarks.map_or(KernelSolver::Exact, |landmarks| KernelSolver::Nystrom {
        landmarks,
        seed,
    })
}

/// The leaf-kernel variance machinery of one Boulevard model.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct BoulevardInference {
    cell: Cell,
}

/// Which interval a method returns.
#[derive(Clone, Copy)]
enum Kind {
    Confidence,
    Prediction,
    Reproduction,
    Calibrated,
}

impl BoulevardInference {
    fn bounds<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        alpha: f64,
        kind: Kind,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let inner = self.cell.borrow_dependent();
        let bounds = py
            .detach(|| match kind {
                Kind::Confidence => inner.confidence_intervals(&data.inner, alpha),
                Kind::Prediction => inner.prediction_intervals(&data.inner, alpha),
                Kind::Reproduction => inner.reproduction_intervals(&data.inner, alpha),
                Kind::Calibrated => inner.calibrated_prediction_intervals(&data.inner, alpha),
            })
            .or_raise()?;
        intervals(py, bounds)
    }
}

#[pymethods]
impl BoulevardInference {
    /// Fits the leaf kernel of `booster` over `train` (its training rows,
    /// or the rows of an honest refit). The noise variance comes from the
    /// labelled `holdout` rows, a known `noise_variance`, or (neither) the
    /// training residuals; `landmarks` selects the Nyström solver.
    #[staticmethod]
    #[pyo3(signature = (booster, train, holdout, noise_variance, landmarks, seed))]
    fn fit(
        py: Python<'_>,
        booster: &Booster,
        train: &DMatrix,
        holdout: Option<&DMatrix>,
        noise_variance: Option<f64>,
        landmarks: Option<usize>,
        seed: u64,
    ) -> PyResult<Self> {
        if holdout.is_some() && noise_variance.is_some() {
            return Err(refuse(
                "pass either holdout rows or a known noise_variance, not both",
            ));
        }
        let owner = Owner {
            model: Arc::clone(&booster.model),
            holdout: holdout.map(|h| h.inner.clone()),
        };
        let train = &train.inner;
        let cell = py
            .detach(|| {
                Cell::try_new(owner, |owner| {
                    let noise = match (&owner.holdout, noise_variance) {
                        (Some(holdout), _) => NoiseVariance::Holdout(holdout),
                        (None, Some(v)) => NoiseVariance::Known(v),
                        (None, None) => NoiseVariance::TrainingResiduals,
                    };
                    inference::BoulevardInference::fit(
                        &owner.model,
                        train,
                        noise,
                        solver(landmarks, seed),
                    )
                })
            })
            .or_raise()?;
        Ok(Self { cell })
    }

    #[getter]
    fn noise_variance(&self) -> f64 {
        self.cell.borrow_dependent().noise_variance()
    }

    fn standard_errors<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let se = py
            .detach(|| self.cell.borrow_dependent().standard_errors(&data.inner))
            .or_raise()?;
        let rows = se.n_rows();
        to_numpy(py, se.into_vec(), &[rows])
    }

    fn confidence_intervals<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        alpha: f64,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        self.bounds(py, data, alpha, Kind::Confidence)
    }

    fn prediction_intervals<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        alpha: f64,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        self.bounds(py, data, alpha, Kind::Prediction)
    }

    fn reproduction_intervals<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        alpha: f64,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        self.bounds(py, data, alpha, Kind::Reproduction)
    }

    fn calibrated_prediction_intervals<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        alpha: f64,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        self.bounds(py, data, alpha, Kind::Calibrated)
    }
}

/// Refits every leaf of the Boulevard model `booster` on the labelled rows
/// `values`, keeping its tree structures.
#[pyfunction]
pub fn honest_refit(py: Python<'_>, booster: &Booster, values: &DMatrix) -> PyResult<Booster> {
    let model = py
        .detach(|| inference::honest_refit(&booster.model, &values.inner))
        .or_raise()?;
    Ok(Booster::new(model))
}

/// The model's Boulevard record as a `dict`, or `None`.
pub(crate) fn boulevard_info<'py>(
    py: Python<'py>,
    model: &BoostedModel,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let Some(info) = model.boulevard() else {
        return Ok(None);
    };
    let dict = PyDict::new(py);
    dict.set_item("dropout", info.dropout)?;
    dict.set_item("learning_rate", info.learning_rate)?;
    dict.set_item("subsample", info.subsample)?;
    dict.set_item("reg_lambda", info.reg_lambda)?;
    dict.set_item("truncation", info.truncation)?;
    dict.set_item("seed", info.seed)?;
    dict.set_item("intercept_from_labels", info.intercept_from_labels)?;
    Ok(Some(dict))
}
