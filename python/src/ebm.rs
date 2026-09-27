//! Explainable boosting machines: shape functions (`hessboost::ebm`) and
//! their Boulevard confidence bands (`hessboost::inference::EbmInference`).

use crate::booster::Booster;
use crate::data::{DMatrix, to_numpy};
use crate::errors::{OrRaise, refuse};
use crate::inference::{intervals, solver};
use hessboost::data::DMatrix as RustMatrix;
use hessboost::ebm::{self, TermAxis};
use hessboost::inference::{self, NoiseVariance};
use hessboost::model::{BoostedModel, Predictions};
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use self_cell::self_cell;
use std::sync::Arc;

/// One term's shape function.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct TermShape {
    inner: ebm::TermShape,
}

/// The number of cells along every axis of `shape`.
fn grid_shape(shape: &ebm::TermShape) -> Vec<usize> {
    shape.axes().iter().map(TermAxis::cells).collect()
}

/// `values` laid out on `shape`'s grid.
fn on_grid<'py>(
    py: Python<'py>,
    shape: &ebm::TermShape,
    values: Vec<f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    to_numpy(py, values, &grid_shape(shape))
}

#[pymethods]
impl TermShape {
    #[getter]
    fn features(&self) -> Vec<u32> {
        self.inner.features().to_vec()
    }

    /// Every axis as `("numeric", edges)` or `("categorical", codes)`.
    #[getter]
    fn axes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for axis in self.inner.axes() {
            let item = match axis {
                TermAxis::Numeric { edges, .. } => {
                    let edges = to_numpy(py, edges.clone(), &[edges.len()])?;
                    PyTuple::new(
                        py,
                        ["numeric".into_pyobject(py)?.into_any(), edges.into_any()],
                    )?
                }
                TermAxis::Categorical { categories, .. } => PyTuple::new(
                    py,
                    [
                        "categorical".into_pyobject(py)?.into_any(),
                        categories.clone().into_pyobject(py)?.into_any(),
                    ],
                )?,
                _ => return Err(refuse("an axis kind this version does not know")),
            };
            list.append(item)?;
        }
        Ok(list)
    }

    /// The values on the grid, one array axis per feature (missing cell
    /// last along each).
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        on_grid(py, &self.inner, self.inner.values().to_vec())
    }

    /// The flat index into `values` of the cell holding `x`.
    fn cell(&self, x: Vec<f32>) -> PyResult<usize> {
        self.inner.cell(&x).or_raise()
    }

    fn value(&self, x: Vec<f32>) -> PyResult<f64> {
        self.inner.value(&x).or_raise()
    }
}

/// The intercept and every term's shape function of the EBM `booster`.
#[pyfunction]
pub fn shape_functions(booster: &Booster) -> PyResult<(f64, Vec<TermShape>)> {
    let shapes = ebm::shape_functions(&booster.model).or_raise()?;
    Ok((
        shapes.intercept,
        shapes
            .terms
            .into_iter()
            .map(|inner| TermShape { inner })
            .collect(),
    ))
}

/// The model's EBM record as a `dict`, or `None`.
pub(crate) fn ebm_info<'py>(
    py: Python<'py>,
    model: &BoostedModel,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let Some(info) = model.ebm() else {
        return Ok(None);
    };
    let dict = PyDict::new(py);
    dict.set_item("terms", info.terms.clone())?;
    dict.set_item("tree_terms", info.tree_terms.clone())?;
    dict.set_item("term_means", info.term_means.clone())?;
    let boulevard = match info.boulevard {
        Some(b) => {
            let d = PyDict::new(py);
            d.set_item("learning_rate", b.learning_rate)?;
            d.set_item("subsample", b.subsample)?;
            d.set_item("reg_lambda", b.reg_lambda)?;
            Some(d)
        }
        None => None,
    };
    dict.set_item("boulevard", boulevard)?;
    Ok(Some(dict))
}

struct Owner {
    model: Arc<BoostedModel>,
    holdout: Option<RustMatrix>,
}

type InferenceRef<'a> = inference::EbmInference<'a>;

self_cell!(
    struct Cell {
        owner: Owner,
        #[covariant]
        dependent: InferenceRef,
    }
);

/// The Boulevard EBM's variance machinery: bands on its shape functions.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct EbmInference {
    cell: Cell,
}

/// A 1-D array of one value per row.
fn column(py: Python<'_>, values: Predictions<f64>) -> PyResult<Bound<'_, PyArrayDyn<f64>>> {
    let n = values.n_rows();
    to_numpy(py, values.into_vec(), &[n])
}

/// A term's shape with its bands: `(shape, standard errors, lower, upper)`,
/// the last three on the shape's grid.
type Bands<'py> = (
    TermShape,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
);

#[pymethods]
impl EbmInference {
    /// As `BoulevardInference.fit`, for a Boulevard EBM.
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
        let solver = solver(landmarks, seed)?;
        let cell = py
            .detach(|| {
                Cell::try_new(owner, |owner| {
                    let noise = match (&owner.holdout, noise_variance) {
                        (Some(holdout), _) => NoiseVariance::Holdout(holdout),
                        (None, Some(v)) => NoiseVariance::Known(v),
                        (None, None) => NoiseVariance::TrainingResiduals,
                    };
                    inference::EbmInference::fit(&owner.model, train, noise, solver)
                })
            })
            .or_raise()?;
        Ok(Self { cell })
    }

    #[getter]
    fn noise_variance(&self) -> f64 {
        self.cell.borrow_dependent().noise_variance()
    }

    #[getter]
    fn intercept_standard_error(&self) -> f64 {
        self.cell.borrow_dependent().intercept_standard_error()
    }

    fn term_bands<'py>(&self, py: Python<'py>, term: usize, alpha: f64) -> PyResult<Bands<'py>> {
        let bands = py
            .detach(|| self.cell.borrow_dependent().term_bands(term, alpha))
            .or_raise()?;
        let shape = &bands.shape;
        let se = on_grid(py, shape, bands.standard_errors)?;
        let lower = on_grid(py, shape, bands.lower)?;
        let upper = on_grid(py, shape, bands.upper)?;
        Ok((TermShape { inner: bands.shape }, se, lower, upper))
    }

    fn term_standard_errors<'py>(
        &self,
        py: Python<'py>,
        term: usize,
        data: &DMatrix,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let se = py
            .detach(|| {
                self.cell
                    .borrow_dependent()
                    .term_standard_errors(term, &data.inner)
            })
            .or_raise()?;
        column(py, se)
    }

    fn standard_errors<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let se = py
            .detach(|| self.cell.borrow_dependent().standard_errors(&data.inner))
            .or_raise()?;
        column(py, se)
    }

    fn confidence_intervals<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        alpha: f64,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let bounds = py
            .detach(|| {
                self.cell
                    .borrow_dependent()
                    .confidence_intervals(&data.inner, alpha)
            })
            .or_raise()?;
        intervals(py, bounds)
    }

    fn prediction_intervals<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        alpha: f64,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let bounds = py
            .detach(|| {
                self.cell
                    .borrow_dependent()
                    .prediction_intervals(&data.inner, alpha)
            })
            .or_raise()?;
        intervals(py, bounds)
    }
}
