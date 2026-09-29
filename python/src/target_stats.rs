//! Ordered target statistics (`hessboost::data::target_stats`): the unfitted
//! encoder and the fitted statistics it produces.

use crate::data::{DMatrix, row_major};
use crate::errors::{DetachExt, OrRaise, refuse};
use hessboost::data::target_stats::{self, TargetKind};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// An unfitted ordered target-statistics encoder.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct OrderedTargetEncoder {
    pub(crate) inner: target_stats::OrderedTargetEncoder,
}

#[pymethods]
impl OrderedTargetEncoder {
    /// An encoder with prior weight `a`, a fixed prior (`None`: the mean
    /// training label), `permutations` averaged permutations drawn from
    /// `seed`, and `target` (`"regression"` or `"binary"`) labels.
    #[new]
    #[pyo3(signature = (*, prior_weight, prior, permutations, seed, target))]
    fn new(
        prior_weight: f64,
        prior: Option<f64>,
        permutations: usize,
        seed: u64,
        target: &str,
    ) -> PyResult<Self> {
        let target = match target {
            "regression" => TargetKind::Regression,
            "binary" => TargetKind::Binary,
            other => {
                return Err(refuse(format!(
                    "target must be \"regression\" or \"binary\", got {other:?}"
                )));
            }
        };
        let mut builder = target_stats::OrderedTargetEncoder::builder()
            .prior_weight(prior_weight)
            .permutations(permutations)
            .seed(seed)
            .target(target);
        if let Some(prior) = prior {
            builder = builder.prior(prior);
        }
        Ok(Self {
            inner: builder.build().or_raise()?,
        })
    }

    /// Fits on `data`'s labels (or `label`, one per row) and encodes
    /// `columns`; returns the encoded matrix and the fitted statistics.
    #[pyo3(signature = (data, columns, label=None))]
    fn fit_transform(
        &self,
        py: Python<'_>,
        data: &DMatrix,
        columns: Vec<usize>,
        label: Option<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<(DMatrix, FittedTargetEncoder)> {
        let label = label
            .as_ref()
            .map(|label| row_major(label, "label"))
            .transpose()?;
        let (inner, fitted) = py.detached(|| match label {
            Some(labels) => self
                .inner
                .fit_transform_with_labels(&data.inner, &columns, labels),
            None => self.inner.fit_transform(&data.inner, &columns),
        })?;
        Ok((DMatrix { inner }, FittedTargetEncoder { inner: fitted }))
    }
}

/// Fitted ordered target statistics over a whole training matrix.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct FittedTargetEncoder {
    inner: target_stats::FittedTargetEncoder,
}

#[pymethods]
impl FittedTargetEncoder {
    /// `data` with the encoded columns replaced by their statistics.
    fn transform(&self, py: Python<'_>, data: &DMatrix) -> PyResult<DMatrix> {
        let inner = py.detached(|| self.inner.transform(&data.inner))?;
        Ok(DMatrix { inner })
    }

    /// The encoding of category `code` in `column`, or `None` when `column`
    /// is not encoded.
    fn encode(&self, column: usize, code: u32) -> Option<f32> {
        self.inner.encode(column, code)
    }

    #[getter]
    fn prior(&self) -> f32 {
        self.inner.prior()
    }

    #[getter]
    fn columns(&self) -> Vec<usize> {
        self.inner.columns().collect()
    }
}
