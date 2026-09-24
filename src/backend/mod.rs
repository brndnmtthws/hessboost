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
//! the single-threaded CPU result bit for bit on every realistic dataset (see
//! [`metal`] for the exact guarantee and its edge cases), and repeats itself
//! exactly across runs and machines.
//!
//! A future `wgpu` backend will extend the same seam to Linux and Windows.

#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal;

/// The GPU predictor handle when the Metal backend is not compiled in.
/// [`BoostedModel::to_gpu`](crate::model::BoostedModel::to_gpu) then always
/// returns an error naming the missing feature, so this type is never
/// constructed.
#[cfg(not(all(target_os = "macos", feature = "metal")))]
#[derive(Debug)]
pub struct GpuModel;
