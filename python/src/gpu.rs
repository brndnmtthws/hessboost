//! `GpuModel`: Metal, wgpu or NVIDIA CUDA batch prediction.
//!
//! wgpu is compiled into every wheel, Metal into macOS wheels and CUDA
//! into Linux wheels. Drivers are loaded at run time; absence never
//! prevents importing the package. Forests upload once; prediction
//! preserves CPU bits, with CPU objective transforms and model shrinkage.

use crate::booster::{dense, iterations};
use crate::data::{DMatrix, to_numpy};
use crate::errors::{DetachExt, refuse};
use hessboost::backend::wgpu;
use hessboost::config::Device;
use hessboost::error::Result;
use hessboost::model::{BoostedModel, Iterations, Predictions};
use numpy::PyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;
use serde::de::value::StrDeserializer;

/// A GPU backend, as the `device` argument names it.
#[derive(Clone, Copy)]
enum Backend {
    /// Native Metal (macOS).
    Metal,
    /// wgpu over Vulkan, Metal, or DirectX 12.
    Wgpu,
    /// NVIDIA CUDA on Linux, on the device `ordinal` (`"cuda:<ordinal>"`).
    Cuda { ordinal: usize },
}

impl Backend {
    /// The backend `device` names, spelled as training's `device` spells it
    /// (`"metal"`, `"wgpu"`, `"cuda"`, `"cuda:<ordinal>"`); `None` is the
    /// platform's default, Metal on macOS and wgpu elsewhere.
    fn parse(device: Option<&str>) -> PyResult<Self> {
        let Some(device) = device else {
            return Ok(if cfg!(target_os = "macos") {
                Self::Metal
            } else {
                Self::Wgpu
            });
        };
        let parsed = Device::deserialize(StrDeserializer::<serde::de::value::Error>::new(device));
        match parsed {
            Ok(Device::Metal) => Ok(Self::Metal),
            Ok(Device::Wgpu) => Ok(Self::Wgpu),
            Ok(Device::Cuda { ordinal }) => Ok(Self::Cuda { ordinal }),
            _ => Err(refuse(format!(
                "unknown GPU device {device:?}; expected \"metal\", \"wgpu\", \"cuda\" or \
                 \"cuda:<ordinal>\""
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
            #[cfg(target_os = "linux")]
            Self::Cuda { ordinal } => hessboost::backend::cuda::prediction_available(ordinal),
            #[cfg(not(target_os = "linux"))]
            Self::Cuda { .. } => false,
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
            #[cfg(target_os = "linux")]
            Self::Cuda { ordinal } => hessboost::backend::cuda::prediction_device_name(ordinal),
            #[cfg(not(target_os = "linux"))]
            Self::Cuda { .. } => None,
        }
    }
}

/// A model uploaded to one backend's GPU.
enum Predictor {
    #[cfg(target_os = "macos")]
    Metal(hessboost::backend::metal::GpuModel),
    Wgpu(wgpu::GpuModel),
    #[cfg(target_os = "linux")]
    Cuda(hessboost::backend::cuda::GpuModel),
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
            #[cfg(target_os = "linux")]
            Backend::Cuda { ordinal } => py.detached(|| model.to_cuda(ordinal)).map(Self::Cuda),
            #[cfg(not(target_os = "linux"))]
            Backend::Cuda { ordinal } => Err(refuse(format!(
                "CUDA GPU prediction (device \"cuda:{ordinal}\") is only available on Linux"
            ))),
        }
    }

    fn model(&self) -> &BoostedModel {
        match self {
            #[cfg(target_os = "macos")]
            Self::Metal(gpu) => gpu.model(),
            Self::Wgpu(gpu) => gpu.model(),
            #[cfg(target_os = "linux")]
            Self::Cuda(gpu) => gpu.model(),
        }
    }

    /// The `device` name of the backend.
    fn device(&self) -> &'static str {
        match self {
            #[cfg(target_os = "macos")]
            Self::Metal(_) => "metal",
            Self::Wgpu(_) => "wgpu",
            #[cfg(target_os = "linux")]
            Self::Cuda(_) => "cuda",
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
            #[cfg(target_os = "linux")]
            Self::Cuda(gpu) if margin => gpu.predict_margin(data, iterations),
            #[cfg(target_os = "linux")]
            Self::Cuda(gpu) => gpu.predict(data, iterations),
        }
    }
}

/// A resident model for Metal, wgpu or CUDA batch prediction.
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
    /// addition-order probe; for CUDA, a driver and device that load the
    /// prediction kernels (`false` off Linux).
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

    /// The backend this model predicts on (`"metal"`, `"wgpu"` or `"cuda"`).
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
