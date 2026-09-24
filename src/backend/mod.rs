//! Optional compute backends.
//!
//! Training and prediction run on the CPU by default, exactly as they always
//! have. This module holds the opt-in accelerators:
//!
//! - [`metal`] (macOS only, `metal` feature): native Metal GPU acceleration
//!   for histogram construction during training (`device = metal`) and for
//!   batch prediction ([`GpuModel`](metal::GpuModel), from
//!   `BoostedModel::to_gpu`).
//!
//! The backends keep the crate's determinism contract: a GPU run reproduces
//! the single-threaded CPU result bit for bit (work the GPU cannot compute
//! exactly runs on the CPU; see [`metal`]), and repeats itself exactly
//! across runs and machines.
//!
//! A future `wgpu` backend will extend the same seam to Linux and Windows.

/// When the Metal backend's double-float sums reproduce the CPU's `f64`
/// chain (platform-independent, so its proof is tested everywhere).
#[cfg_attr(
    not(all(target_os = "macos", feature = "metal")),
    allow(
        dead_code,
        reason = "only the Metal backend calls it; its unit tests run on every platform"
    )
)]
mod exact_sum;

/// The native Metal backend (macOS, `metal` feature).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal;

/// The Metal backend's stand-in when it is not compiled in (any other
/// platform, or the feature off): the module exists so `backend::metal`
/// paths and doc links resolve on every platform, but holds only the
/// [`GpuModel`](self::metal::GpuModel) handle, which
/// [`BoostedModel::to_gpu`](crate::model::BoostedModel::to_gpu) then never
/// constructs — it always returns an error naming the missing feature.
#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub mod metal {
    /// The GPU predictor handle when the Metal backend is not compiled in.
    /// [`BoostedModel::to_gpu`](crate::model::BoostedModel::to_gpu) then
    /// always returns an error, so this is never constructed.
    #[derive(Debug)]
    pub struct GpuModel;
}
