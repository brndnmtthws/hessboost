//! `CompactModel`: the bit-packed *Trees on a Diet* layout
//! (`hessboost::model::compact`), predicting a booster's margins bit for bit.

use crate::booster::dense;
use crate::data::{DMatrix, to_numpy};
use crate::errors::DetachExt;
use hessboost::model::compact;
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// A tree ensemble in the bit-packed compact layout.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct CompactModel {
    inner: compact::CompactModel,
}

impl CompactModel {
    pub(crate) fn new(inner: compact::CompactModel) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl CompactModel {
    /// Decodes a compact model's bytes.
    #[staticmethod]
    fn load(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        py.detached(|| compact::CompactModel::decode(data))
            .map(Self::new)
    }

    /// The serialized model.
    fn save<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.encode())
    }

    /// Predictions (`margin`: raw margins), `(rows,)` or `(rows, width)`.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        margin: bool,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let predictions = py.detached(|| {
            if margin {
                self.inner.predict_margin(&data.inner)
            } else {
                self.inner.predict(&data.inner)
            }
        })?;
        let (values, shape) = dense(predictions);
        to_numpy(py, values, &shape)
    }

    #[getter]
    fn objective(&self) -> &str {
        self.inner.objective().name()
    }

    #[getter]
    fn num_trees(&self) -> usize {
        self.inner.num_trees()
    }

    #[getter]
    fn num_features(&self) -> usize {
        self.inner.n_features()
    }

    #[getter]
    fn num_outputs(&self) -> usize {
        self.inner.n_outputs()
    }

    #[getter]
    fn size_bytes(&self) -> usize {
        self.inner.size_bytes()
    }

    #[getter]
    fn used_features(&self) -> Vec<usize> {
        self.inner.used_features()
    }

    #[getter]
    fn num_thresholds(&self) -> usize {
        self.inner.num_thresholds()
    }

    #[getter]
    fn num_leaf_values(&self) -> usize {
        self.inner.num_leaf_values()
    }
}
