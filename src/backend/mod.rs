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

/// Rows one threadgroup may scan with a scatter kernel's shared 32-bit
/// accumulators, given the staged slices' magnitude statistics: a grain
/// count `k` is split as `k = hi * 2^16 + lo`, so one threadgroup's `hi` sum
/// must stay inside an `i32` (`lo` is 16-bit each and sums inside a `u32`).
/// Past it the node runs on the CPU backend (or, on Metal, the register
/// kernels), whose sums are exact by the same argument (see
/// [`exact_sum`]).
#[cfg(any(all(target_os = "macos", feature = "metal"), feature = "wgpu"))]
pub(crate) fn scatter_row_bound(grad: &exact_sum::SumDomain, hess: &exact_sum::SumDomain) -> usize {
    let bound = |domain: &exact_sum::SumDomain| -> u64 {
        let max = domain.max_units();
        if max == 0 {
            return u64::from(u32::MAX);
        }
        let hi = max.div_ceil(1 << 16);
        // Both accumulators stay exact: `hi` in an `i32`, `lo` in a `u32`
        // (the largest 16-bit sum, 65535 per row).
        (((1u64 << 31) - 1) / hi).min((u64::from(u32::MAX) - 1) / 65_535)
    };
    usize::try_from(bound(grad).min(bound(hess))).unwrap_or(usize::MAX)
}

/// Write `data`'s rows starting at row `begin` into `rows` as a dense
/// `NaN`-for-missing matrix, the same materialization the CPU's row blocks
/// use: dense NaN-sentinel matrices copy in place, a dense matrix with
/// another sentinel maps sentinel values to `NaN`, and CSR rows materialize
/// per entry. `rows` holds a whole number of rows; it is one prediction
/// block of the batch.
#[cfg(any(all(target_os = "macos", feature = "metal"), feature = "wgpu"))]
pub(crate) fn materialize_rows(data: &crate::data::DMatrix, begin: usize, rows: &mut [f32]) {
    use rayon::prelude::*;
    let n_cols = data.n_cols();
    if let Some(dense) = data.dense_values()
        && data.missing().is_nan()
        && dense.len() == data.n_rows() * n_cols
    {
        let start = begin * n_cols;
        rows.copy_from_slice(&dense[start..start + rows.len()]);
        return;
    }
    let missing = data.missing();
    rows.par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(i, row)| {
            let r = begin + i;
            for (f, slot) in row.iter_mut().enumerate() {
                *slot = match data.get(r, f) {
                    Some(v) if v != missing || missing.is_nan() => v,
                    _ => f32::NAN,
                };
            }
        });
}
