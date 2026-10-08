//! The kernels' PTX: the `cuda-kernels/` crate's Rust, compiled by
//! cuda-oxide (`cuda-kernels/build.sh`, which also refuses PTX holding a
//! non-IEEE floating-point instruction) and committed as `training.ptx` and
//! `prediction.ptx`. CI rebuilds them and fails when a file differs from the
//! source's.

use cudarc::driver::{CudaContext, CudaModule};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// The oldest compute capability the PTX targets (`sm_75`, Turing); the
/// driver JIT-compiles a module for the device's own architecture (any
/// later one) when a context loads it, and caches the machine code.
const MIN_COMPUTE_CAPABILITY: (i32, i32) = (7, 5);

/// One of the embedded modules.
#[derive(Clone, Copy)]
pub(super) enum Module {
    /// Histograms, partitions, reductions, gradients and split search.
    Training,
    /// The compact-forest walks: a module of their own, so a prediction
    /// context JIT-compiles only them.
    Prediction,
}

impl Module {
    pub(super) fn ptx(self) -> &'static str {
        match self {
            Module::Training => include_str!("training.ptx"),
            Module::Prediction => include_str!("prediction.ptx"),
        }
    }
}

/// Load `module` into `ctx`, refusing a device older than the PTX's
/// target. Only the PTX container comes from cudarc's `nvrtc` module; NVRTC
/// itself is never loaded.
pub(super) fn load(
    ctx: &Arc<CudaContext>,
    module: Module,
) -> std::result::Result<Arc<CudaModule>, String> {
    let (major, minor) = ctx
        .compute_capability()
        .map_err(|e| format!("CUDA compute capability: {e}"))?;
    if (major, minor) < MIN_COMPUTE_CAPABILITY {
        let (min_major, min_minor) = MIN_COMPUTE_CAPABILITY;
        return Err(format!(
            "CUDA device {} has compute capability {major}.{minor}; the backend needs \
             {min_major}.{min_minor} or newer",
            ctx.ordinal()
        ));
    }
    ctx.load_module(Ptx::from_src(module.ptx()))
        .map_err(|e| format!("CUDA module load (sm_{major}{minor}): {e}"))
}
