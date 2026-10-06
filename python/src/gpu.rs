//! `GpuModel`: a model laid out for GPU batch prediction, on Metal (macOS)
//! or through wgpu (Vulkan, Metal, DirectX 12).
//!
//! Built with [`Booster::to_gpu`](crate::booster::Booster::to_gpu); the
//! forest, category pools, and per-tree weights are uploaded once, and each
//! prediction call uploads its rows. Predictions are bit-identical to the
//! CPU's. wgpu is compiled into every build and Metal into macOS builds
//! only: elsewhere `"metal"` reports no device and building on it is
//! refused.

use crate::booster::{dense, iterations};
use crate::data::{DMatrix, to_numpy};
use crate::errors::{DetachExt, refuse};
use hessboost::backend::wgpu;
use hessboost::error::Result;
use hessboost::model::{BoostedModel, Iterations, Predictions};
use numpy::PyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A GPU backend, as the `device` argument names it.
#[derive(Clone, Copy)]
enum Backend {
    /// Native Metal (macOS).
    Metal,
    /// wgpu over Vulkan, Metal, or DirectX 12.
    Wgpu,
}

impl Backend {
    /// The backend `device` names; `None` is the platform's default, Metal
    /// on macOS and wgpu elsewhere.
    fn parse(device: Option<&str>) -> PyResult<Self> {
        match device {
            None if cfg!(target_os = "macos") => Ok(Self::Metal),
            None | Some("wgpu") => Ok(Self::Wgpu),
            Some("metal") => Ok(Self::Metal),
            Some(other) => Err(refuse(format!(
                "unknown GPU device {other:?}; expected \"metal\" or \"wgpu\""
            ))),
        }
    }

    /// Whether the backend predicts here ([`Predictor::build`] lays forest
    /// models out on it). The first call per backend initializes it
    /// (adapter selection, kernel compilation, wgpu's addition-order probe).
    fn available(self) -> bool {
        match self {
            #[cfg(target_os = "macos")]
            Self::Metal => hessboost::backend::metal::available(),
            #[cfg(not(target_os = "macos"))]
            Self::Metal => false,
            Self::Wgpu => wgpu::prediction_available(),
        }
    }

    /// The name of the GPU the backend picked, if any.
    fn device_name(self) -> Option<String> {
        match self {
            #[cfg(target_os = "macos")]
            Self::Metal => hessboost::backend::metal::device_name(),
            #[cfg(not(target_os = "macos"))]
            Self::Metal => None,
            Self::Wgpu => wgpu::device_name(),
        }
    }
}

/// A model uploaded to one backend's GPU.
enum Predictor {
    #[cfg(target_os = "macos")]
    Metal(hessboost::backend::metal::GpuModel),
    Wgpu(wgpu::GpuModel),
}

impl Predictor {
    /// `model` laid out on `backend`, without the GIL.
    fn build(py: Python<'_>, model: &BoostedModel, backend: Backend) -> PyResult<Self> {
        match backend {
            #[cfg(target_os = "macos")]
            Backend::Metal => py.detached(|| model.to_gpu()).map(Self::Metal),
            #[cfg(not(target_os = "macos"))]
            Backend::Metal => {
                let _ = (py, model);
                Err(refuse(
                    "Metal GPU prediction is only available on macOS; device=\"wgpu\" runs on \
                     Vulkan, Metal, or DirectX 12",
                ))
            }
            Backend::Wgpu => py.detached(|| model.to_wgpu()).map(Self::Wgpu),
        }
    }

    fn model(&self) -> &BoostedModel {
        match self {
            #[cfg(target_os = "macos")]
            Self::Metal(gpu) => gpu.model(),
            Self::Wgpu(gpu) => gpu.model(),
        }
    }

    /// The `device` name of the backend.
    fn device(&self) -> &'static str {
        match self {
            #[cfg(target_os = "macos")]
            Self::Metal(_) => "metal",
            Self::Wgpu(_) => "wgpu",
        }
    }

    /// Raw margins (`margin`) or values of `data` from `iterations`.
    fn predict(
        &self,
        data: &hessboost::data::DMatrix,
        iterations: Iterations,
        margin: bool,
    ) -> Result<Predictions> {
        match self {
            #[cfg(target_os = "macos")]
            Self::Metal(gpu) if margin => gpu.predict_margin(data, iterations),
            #[cfg(target_os = "macos")]
            Self::Metal(gpu) => gpu.predict(data, iterations),
            Self::Wgpu(gpu) if margin => gpu.predict_margin(data, iterations),
            Self::Wgpu(gpu) => gpu.predict(data, iterations),
        }
    }
}

/// A model laid out for GPU batch prediction, on Metal or through wgpu.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct GpuModel {
    gpu: Predictor,
}

impl GpuModel {
    /// `model` laid out on the GPU `device` names.
    pub(crate) fn build(
        py: Python<'_>,
        model: &BoostedModel,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let backend = Backend::parse(device)?;
        Predictor::build(py, model, backend).map(|gpu| Self { gpu })
    }
}

#[pymethods]
impl GpuModel {
    /// Whether `device` (`None`: the platform's default) predicts here: for
    /// Metal, a device with working compute pipelines (`false` off macOS);
    /// for wgpu, an adapter with 64-bit shader integers that passed the
    /// addition-order probe.
    #[staticmethod]
    fn available(py: Python<'_>, device: Option<&str>) -> PyResult<bool> {
        let backend = Backend::parse(device)?;
        Ok(py.detach(|| backend.available()))
    }

    /// The name of the GPU `device` picked, if any (for diagnostics and
    /// benchmarks).
    #[staticmethod]
    fn device_name(py: Python<'_>, device: Option<&str>) -> PyResult<Option<String>> {
        let backend = Backend::parse(device)?;
        Ok(py.detach(|| backend.device_name()))
    }

    /// The backend this model predicts on (`"metal"` or `"wgpu"`).
    #[getter]
    fn device(&self) -> &'static str {
        self.gpu.device()
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
        let iterations = iterations(gpu.model(), iteration_range, Iterations::Best);
        let predictions = py.detached(|| gpu.predict(matrix, iterations, margin))?;
        let (values, shape) = dense(predictions);
        to_numpy(py, values, &shape)
    }
}
