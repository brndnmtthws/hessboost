//! `Distributions`: the per-row distributions a `dist:*` model predicts,
//! with vectorized summaries.

use crate::data::to_numpy;
use crate::errors::refuse;
use hessboost::objective::distributional::Dist;
use numpy::{AllowTypeChange, PyArrayDyn, PyArrayLike1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// The conditional distributions a ``dist:*`` model predicts, one per row
/// (returned by ``Booster.predict_distribution``).
///
/// Every summary is vectorized over the rows and returns a ``float64``
/// array; methods taking ``y`` expect one value per row.
#[pyclass(frozen, module = "hessboost", sequence)]
pub struct Distributions {
    dists: Vec<Dist>,
    family: &'static str,
    param_names: &'static [&'static str],
}

impl Distributions {
    pub(crate) fn new(dists: Vec<Dist>) -> PyResult<Self> {
        let family = dists
            .first()
            .ok_or_else(|| refuse("no rows to predict distributions for"))?
            .family();
        Ok(Self {
            dists,
            family: family.objective_name(),
            param_names: family.param_names(),
        })
    }

    fn map<'py>(
        &self,
        py: Python<'py>,
        f: impl Fn(&Dist) -> f64,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        to_numpy(py, self.dists.iter().map(f).collect(), &[self.dists.len()])
    }

    /// `f(dist, y)` per row; `y` holds one value per row.
    fn map_y<'py>(
        &self,
        py: Python<'py>,
        y: &PyReadonlyArray1<'_, f64>,
        f: impl Fn(&Dist, f64) -> f64,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        if y.len() != self.dists.len() {
            return Err(refuse(format!(
                "y has {} values for {} distributions",
                y.len(),
                self.dists.len()
            )));
        }
        let values = self
            .dists
            .iter()
            .zip(y.as_array())
            .map(|(dist, &y)| f(dist, y))
            .collect();
        to_numpy(py, values, &[self.dists.len()])
    }
}

#[pymethods]
impl Distributions {
    /// The objective naming the family, e.g. `"dist:normal"`.
    #[getter]
    fn family(&self) -> &'static str {
        self.family
    }

    /// The natural parameters' names, in column order of `params`.
    #[getter]
    fn param_names(&self) -> Vec<&'static str> {
        self.param_names.to_vec()
    }

    /// The natural parameters, `(rows, len(param_names))`.
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let values = self.dists.iter().flat_map(Dist::params).collect();
        to_numpy(py, values, &[self.dists.len(), self.param_names.len()])
    }

    /// The mean of every row's distribution, ``(rows,)``.
    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        self.map(py, Dist::mean)
    }

    /// The variance of every row's distribution, ``(rows,)``.
    fn variance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        self.map(py, Dist::variance)
    }

    /// The standard deviation of every row's distribution, ``(rows,)``.
    fn std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        self.map(py, Dist::std_dev)
    }

    /// The ``q``-quantile of every row's distribution, ``(rows,)``: the
    /// smallest ``y`` with ``cdf(y) >= q`` (an integer for count families);
    /// NaN outside ``[0, 1]``.
    fn quantile<'py>(&self, py: Python<'py>, q: f64) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        self.map(py, |dist| dist.quantile(q))
    }

    /// The central interval holding probability ``coverage``, ``(rows, 2)``
    /// as ``[lower, upper]``.
    fn interval<'py>(
        &self,
        py: Python<'py>,
        coverage: f64,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let values = self
            .dists
            .iter()
            .flat_map(|dist| {
                let (lower, upper) = dist.interval(coverage);
                [lower, upper]
            })
            .collect();
        to_numpy(py, values, &[self.dists.len(), 2])
    }

    /// ``P(Y <= y)`` for every row, ``(rows,)``.
    fn cdf<'py>(
        &self,
        py: Python<'py>,
        y: PyArrayLike1<'_, f64, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        self.map_y(py, &y, Dist::cdf)
    }

    /// The log density (log mass for count families) at ``y``, ``(rows,)``.
    fn log_prob<'py>(
        &self,
        py: Python<'py>,
        y: PyArrayLike1<'_, f64, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        self.map_y(py, &y, Dist::log_prob)
    }

    /// The continuous ranked probability score of ``y``, ``(rows,)``.
    fn crps<'py>(
        &self,
        py: Python<'py>,
        y: PyArrayLike1<'_, f64, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        self.map_y(py, &y, Dist::crps)
    }

    /// The number of rows.
    fn __len__(&self) -> usize {
        self.dists.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Distributions(family='{}', rows={})",
            self.family,
            self.dists.len()
        )
    }
}
