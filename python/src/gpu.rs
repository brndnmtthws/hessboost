//! `GpuModel`: a model laid out for GPU batch prediction on Metal.
//!
//! Built with [`Booster::to_gpu`](crate::booster::Booster::to_gpu); the
//! forest, category pools, and per-tree weights are uploaded once, and each
//! prediction call uploads its rows. Predictions are bit-identical to the
//! CPU's. Where the backend is not compiled in (off macOS), only
//! [`GpuModel::available`] and [`GpuModel::device_name`] work, reporting no
//! device; building one is refused.
use crate::data::DMatrix;
use hessboost::backend::metal::GpuModel as RustGpuModel;
use hessboost::model::Iterations;
use numpy::PyArrayDyn;
use pyo3::prelude::*;

#[cfg(target_os = "macos")]
use crate::data::to_numpy;

#[cfg(target_os = "macos")]
use crate::errors::DetachExt;
#[cfg(not(target_os = "macos"))]
use crate::errors::refuse;
#[cfg(target_os = "macos")]
use pyo3::exceptions::PyValueError;

/// A model laid out for GPU batch prediction on Metal.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct GpuModel {
    #[cfg_attr(
        not(target_os = "macos"),
        allow(dead_code, reason = "only read where the Metal backend is compiled in")
    )]
    gpu: RustGpuModel,
}

impl GpuModel {
    #[cfg_attr(
        not(target_os = "macos"),
        allow(
            dead_code,
            reason = "only built where the Metal backend is compiled in"
        )
    )]
    pub(crate) fn new(gpu: RustGpuModel) -> Self {
        Self { gpu }
    }

    /// XGBoost's `iteration_range` as iterations: `(begin, 0)` runs through
    /// the last iteration, and `None` is the method's `default`.
    fn iterations(&self, range: Option<(usize, usize)>, default: Iterations) -> Iterations {
        #[cfg(target_os = "macos")]
        {
            range.map_or(default, |(begin, end)| {
                let end = if end == 0 {
                    self.gpu.model().num_boost_rounds()
                } else {
                    end
                };
                (begin..end).into()
            })
        }
        // Unreachable in practice: `to_gpu` never succeeds where the Metal
        // backend is not compiled in, so no `GpuModel` exists to call this.
        #[cfg(not(target_os = "macos"))]
        {
            let _ = range;
            default
        }
    }
}

#[pymethods]
impl GpuModel {
    /// Whether a Metal device with working compute pipelines is available
    /// (`false` where the backend is not compiled in, or the machine has no
    /// GPU).
    #[staticmethod]
    fn available() -> bool {
        #[cfg(target_os = "macos")]
        {
            hessboost::backend::metal::available()
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    /// The name of the Metal device predictions would run on, if any (for
    /// diagnostics and benchmarks).
    #[staticmethod]
    fn device_name() -> Option<String> {
        #[cfg(target_os = "macos")]
        {
            hessboost::backend::metal::device_name()
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    }

    /// Predictions of `kind` (`value`, `margin`) shaped as XGBoost's Python
    /// package shapes them; bit-identical to the model's.
    #[pyo3(signature = (data, kind, iteration_range=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        kind: &str,
        iteration_range: Option<(usize, usize)>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        #[cfg(target_os = "macos")]
        {
            let margin = match kind {
                "value" => false,
                "margin" => true,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown prediction kind {other:?}"
                    )));
                }
            };
            let gpu = &self.gpu;
            let matrix = &data.inner;
            // `None`: through `best_iteration` after early stopping.
            let iterations = self.iterations(iteration_range, Iterations::Best);
            let (values, shape) = py.detached(|| -> hessboost::error::Result<_> {
                let predictions = if margin {
                    gpu.predict_margin(matrix, iterations)?
                } else {
                    gpu.predict(matrix, iterations)?
                };
                let (rows, width) = (predictions.n_rows(), predictions.width());
                let shape = if width == 1 {
                    vec![rows]
                } else {
                    vec![rows, width]
                };
                Ok((predictions.into_vec(), shape))
            })?;
            to_numpy(py, values, &shape)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (self, py, data, kind, iteration_range);
            Err(refuse(
                "GPU prediction requires the `metal` feature on macOS",
            ))
        }
    }
}
