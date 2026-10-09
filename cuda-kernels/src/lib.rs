//! The CUDA kernels of hessboost's `cuda` backend, in Rust, compiled to PTX
//! by [cuda-oxide](https://nvidia.github.io/cuda-rust/cuda-oxide/).
//!
//! This crate is not part of hessboost's build. `build.sh` compiles it with
//! `cargo oxide build` (cuda-oxide's codegen backend on the nightly in
//! `rust-toolchain.toml`) twice: the `train` feature's kernels into
//! `../src/backend/cuda/training.ptx`, the `predict` feature's into
//! `prediction.ptx` next to it. The backend embeds both, and the driver
//! JIT-compiles a module for the device when a context loads it. CI rebuilds
//! the PTX and fails when a committed file differs, so the PTX is always
//! this source's. The build is byte-for-byte reproducible for a given host
//! architecture (rustc's crate hashes, which name shared-memory symbols,
//! depend on it): commit PTX built on `x86_64` Linux, as CI is, or take the
//! files CI uploads when the check fails.
//!
//! # Exactness
//!
//! The backend's histograms, split search, gradients and prediction
//! reproduce the CPU bit for bit (`src/backend/cuda/mod.rs` has the design),
//! so every floating-point operation here is the one IEEE operation the CPU
//! performs, in the CPU's order:
//!
//! - `build.sh` passes `--no-fmad`: no operation is contracted, so `a * b +
//!   c` is a `mul.rn` and an `add.rn`, which ptxas never fuses. The only
//!   fused operation is the explicit `fma.rn.f32` of the logistic gradient's
//!   exponential, exactly where the host's vector kernel fuses.
//! - Division and `as f32` round to nearest, with subnormals kept (no
//!   flush-to-zero). Nothing comes from libdevice, so the build needs no
//!   CUDA toolkit.
//! - No floating-point atomics: integer sums are order-free, and every `f64`
//!   sum is one thread's chain in the CPU's order, except the split scans'
//!   warp-parallel prefixes over histograms the host certified exact (every
//!   partial sum an exact multiple of the grain below 2^53 grains, so every
//!   association of the additions has the chain's bits).
//!
//! `build.sh` rejects PTX holding an approximate, flush-to-zero or
//! contractible floating-point instruction.
//!
//! # Safety model
//!
//! The kernels follow cuda-oxide's tiers. Those with one output element
//! per thread (`stage_units`, `iota_rows`, `squared_error`, `logistic`,
//! `reduce_chains`, `route_runs`) are safe: `&[T]` inputs (bounds-checked
//! reads), a [`DisjointSlice`] output written through
//! `thread::index_1d()`, launched 1-D with one thread per element.
//! `grad_domain` is safe too: it reads a `&[T]` and publishes through
//! atomics on a `&[DeviceAtomicU32]`. The slot-scatter reductions
//! (`finalize_exact`, `finalize_exact_sub`, `reduce_chunks`,
//! `subtract_hists`) read `&[T]`s, but write output slots drawn from data:
//! they are `unsafe` (the host guarantees the slots are distinct) and write
//! through [`scatter`], which traps on an index past the slice. The rest
//! (shared memory, warp collectives, atomics, scatter through data) are
//! `unsafe fn`s over raw device pointers: each `# Safety` section states
//! what the host's launch guarantees, and each `unsafe` block what it
//! relies on. An index past a `&[T]` traps (cuda-oxide's bounds check), so
//! a host bug fails the launch, which the backend treats as a context
//! error, rather than corrupting memory.
//!
//! # ABI
//!
//! A slice parameter (`&[T]`, [`DisjointSlice<T>`]) is two PTX parameters,
//! the address and the element count (cuda-oxide's slice ABI); scalars,
//! raw pointers and `#[repr(C)]` parameter structs (one PTX `.param .align
//! N .b8` array each, read with `ld.param` at the fields' offsets) are one
//! each. The host's launch builders push them in order
//! (`src/backend/cuda/{mod,categorical,predict}.rs`: `Launch::slice`,
//! `pairs` for slices of pair types over flat buffers, `raw_slice` for
//! staged descriptors, `arg` for the rest); entry names are the function
//! names. Record and parameter types are `#[repr(C)]` in the host's
//! layouts. Pointers a kernel's inner loop dereferences stay direct
//! parameters, which the backend lowers to global-space accesses; a
//! struct's pointers are generic.
//!
//! Layouts (row ids `u32`, element offsets 64-bit):
//! - bins: row-major ELLPACK, `n_cols` feature-local bins per row (`u8`,
//!   `u16` or `u32`); `sentinel` marks a missing value (a dense index passes
//!   a sentinel no stored value equals). CSR bins are global, with 64-bit
//!   row offsets.
//! - `feature_first`: each feature's first global bin.
//! - `gpair`: one [`F32x2`] (gradient, Hessian) per row, as `GradPair`.
//! - `units`: one [`I64x2`] per row, the pair in integer grains.
//! - integer histograms: `[slot][bin][2]` 64-bit words, two's complement.
//! - `f64` histograms: `[slot][bin]` [`F64x2`], as `GradStats`.

#![cfg_attr(
    not(feature = "train"),
    allow(
        dead_code,
        reason = "the prediction kernels use none of the training helpers"
    )
)]

#[cfg(feature = "train")]
mod categorical;
#[cfg(feature = "predict")]
mod predict;
#[cfg(feature = "train")]
mod train;

use cuda_device::{DisjointSlice, debug, thread};

/// Every lane of a warp, for the warp-synchronous intrinsics.
const FULL: u32 = u32::MAX;

/// Warps per split-scan block (`SCAN_WARPS` in `src/backend/cuda/mod.rs`);
/// the scan kernels' `launch_bounds` are `32 * SCAN_WARPS`. One: a scan is
/// latency-bound on its SM's `f64` units, so each feature's warp gets an SM
/// of its own while a batch has fewer features than the device has SMs.
const SCAN_WARPS: usize = 1;

/// CUDA's `float2`: one gradient pair.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct F32x2 {
    pub x: f32,
    pub y: f32,
}

/// CUDA's `double2`: one histogram bin's gradient and Hessian sums.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct F64x2 {
    pub x: f64,
    pub y: f64,
}

/// CUDA's `longlong2`: one row's gradient pair in integer grains.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct I64x2 {
    pub x: i64,
    pub y: i64,
}

/// CUDA's `uint2`.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct U32x2 {
    pub x: u32,
    pub y: u32,
}

/// CUDA's `uint4`.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct U32x4 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub w: u32,
}

/// A bin index stored in 1, 2 or 4 bytes.
pub trait Bin: Copy {
    /// The stored value.
    fn get(self) -> u32;
    /// `value` truncated to this width.
    fn narrow(value: u32) -> Self;
}

impl Bin for u8 {
    #[inline(always)]
    fn get(self) -> u32 {
        u32::from(self)
    }
    #[inline(always)]
    fn narrow(value: u32) -> Self {
        value as u8
    }
}

impl Bin for u16 {
    #[inline(always)]
    fn get(self) -> u32 {
        u32::from(self)
    }
    #[inline(always)]
    fn narrow(value: u32) -> Self {
        value as u16
    }
}

impl Bin for u32 {
    #[inline(always)]
    fn get(self) -> u32 {
        self
    }
    #[inline(always)]
    fn narrow(value: u32) -> Self {
        value
    }
}

/// `p[i]`.
///
/// # Safety
///
/// `p.add(i)` is in bounds of a live allocation holding a `T`.
#[inline(always)]
unsafe fn ld<T: Copy>(p: *const T, i: u64) -> T {
    // SAFETY: the caller's.
    unsafe { *p.add(i as usize) }
}

/// `p[i] = value`.
///
/// # Safety
///
/// `p.add(i)` is in bounds of a live allocation, and no other thread
/// accesses the element concurrently.
#[inline(always)]
unsafe fn st<T>(p: *mut T, i: u64, value: T) {
    // SAFETY: the caller's.
    unsafe { *p.add(i as usize) = value }
}

/// `out[i]` for a thread whose element comes from data (a histogram slot
/// the host assigned), not from its thread index: traps past `out`'s end,
/// so a bad slot fails the launch instead of writing outside the buffer.
///
/// # Safety
///
/// No other thread of the launch accesses element `i` while the returned
/// reference lives.
#[inline(always)]
unsafe fn scatter<'s, T>(out: &'s mut DisjointSlice<'_, T>, i: usize) -> &'s mut T {
    if i >= out.len() {
        debug::trap();
    }
    // SAFETY: `i < out.len()` (checked above); the exclusivity is the
    // caller's.
    unsafe { out.get_unchecked_mut(i) }
}

/// This thread's index in a one-dimensional grid.
#[inline(always)]
fn grid_index() -> u64 {
    u64::from(thread::blockIdx_x()) * u64::from(thread::blockDim_x())
        + u64::from(thread::threadIdx_x())
}

/// The thread count of a one-dimensional grid: a grid-stride loop's step.
#[inline(always)]
fn grid_threads() -> u64 {
    u64::from(thread::gridDim_x()) * u64::from(thread::blockDim_x())
}
