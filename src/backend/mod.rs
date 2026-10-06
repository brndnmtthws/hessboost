//! Optional compute backends.
//!
//! Training and prediction run on the CPU by default, exactly as they always
//! have. This module holds the opt-in accelerators:
//!
//! - [`metal`] (macOS only, `metal` feature): native Metal GPU acceleration
//!   for histogram construction during training (`device = metal`) and for
//!   batch prediction ([`GpuModel`](metal::GpuModel), from
//!   `BoostedModel::to_gpu`).
//! - [`wgpu`] (`wgpu` feature; Linux, macOS, Windows): the same two paths
//!   through [wgpu](https://wgpu.rs) over Vulkan, Metal, or DirectX 12
//!   (`device = wgpu`; [`GpuModel`](wgpu::GpuModel), from
//!   `BoostedModel::to_wgpu`).
//!
//! The backends keep the crate's determinism contract: a GPU run reproduces
//! the CPU result bit for bit (work the GPU cannot compute exactly runs on
//! the CPU; see [`metal`] and [`wgpu`]), and repeats itself exactly across
//! runs and machines.

/// When a GPU backend's integer histogram sums reproduce the CPU's `f64`
/// sums (platform-independent, so its proof is tested everywhere).
#[cfg_attr(
    not(any(all(target_os = "macos", feature = "metal"), feature = "wgpu")),
    allow(
        dead_code,
        reason = "only the GPU backends call it; its unit tests run on every platform"
    )
)]
mod exact_sum;

/// The host-side plumbing both GPU backends share.
#[cfg(any(all(target_os = "macos", feature = "metal"), feature = "wgpu"))]
mod shared;

/// The native Metal backend (macOS, `metal` feature).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal;

/// The Metal backend's stand-in when it is not compiled in (any other
/// platform, or the feature off): the module exists so `backend::metal`
/// paths and doc links resolve on every platform, but holds only the
/// [`GpuModel`](self::metal::GpuModel) handle, which
/// [`BoostedModel::to_gpu`](crate::model::BoostedModel::to_gpu) then never
/// constructs — it always returns an error naming the missing feature.
///
/// These docs are the stand-in (docs.rs builds on Linux). The Metal API and
/// the backend's design, exactness bound, and limitations are documented
/// in the real module: run `cargo doc --features metal --open` on macOS.
#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub mod metal {
    /// The GPU predictor handle when the Metal backend is not compiled in.
    /// [`BoostedModel::to_gpu`](crate::model::BoostedModel::to_gpu) then
    /// always returns an error, so this is never constructed.
    #[derive(Debug)]
    #[non_exhaustive]
    pub struct GpuModel;
}

/// The portable wgpu backend (`wgpu` feature).
#[cfg(feature = "wgpu")]
pub mod wgpu;

/// The wgpu backend's stand-in when the `wgpu` feature is off: the module
/// exists so `backend::wgpu` paths and doc links resolve, but holds only
/// the [`GpuModel`](self::wgpu::GpuModel) handle, which
/// [`BoostedModel::to_wgpu`](crate::model::BoostedModel::to_wgpu) then never
/// constructs — it always returns an error naming the missing feature.
///
/// The wgpu API and the backend's design, exactness bound, and limitations
/// are documented in the real module: `cargo doc --features wgpu --open`.
#[cfg(not(feature = "wgpu"))]
pub mod wgpu {
    /// The GPU predictor handle when the wgpu backend is not compiled in.
    /// [`BoostedModel::to_wgpu`](crate::model::BoostedModel::to_wgpu) then
    /// always returns an error, so this is never constructed.
    #[derive(Debug)]
    #[non_exhaustive]
    pub struct GpuModel;
}
