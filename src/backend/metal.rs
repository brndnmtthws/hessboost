//! Native Metal acceleration for macOS (opt-in `metal` feature).
//!
//! Two paths have GPU implementations; everything else stays on the CPU:
//!
//! - **Prediction** ([`GpuModel`](crate::backend::metal::GpuModel), from [`BoostedModel::to_gpu`](crate::model::BoostedModel::to_gpu)): every row
//!   walks the branch-free compact forest arena on the GPU, one thread per
//!   row, adding each tree's leaf value in tree order. This is the speed
//!   win: on an M4 Max, 500k rows through 200 depth-8 trees predict about
//!   2.5× faster than the CPU walk (100 depth-6 trees: ~1.1–1.8×,
//!   thermal-sensitive), and the gap widens with more trees and rows (each
//!   call's fixed row-upload cost amortizes).
//! - **Training** (`device = metal`, see
//!   [`TrainingParams::device`](crate::config::TrainingParams::device)):
//!   the histogram construction of `tree_method = hist` moves to the GPU,
//!   bit-identical to the CPU's (below). It is correct and deterministic
//!   everywhere, but on multicore Apple Silicon it is currently *slower*
//!   than the CPU histogram path (measured 1M rows × 30 features: ~11 ms
//!   GPU vs ~2 ms for a 14-core CPU; end-to-end 200k × 30 depth-8 training
//!   ~1.8× slower). The cause is structural: Apple GPUs have no `double`,
//!   so the exact accumulation needs ~six times the arithmetic of the
//!   CPU's native `f64` adds, and the determinism contract forbids the
//!   floating-point atomics other GPU histogram implementations use. Set
//!   `device = metal` to exercise the GPU path; for speed, keep training
//!   on the CPU and use [`BoostedModel::to_gpu`](crate::model::BoostedModel::to_gpu) for prediction.
//!
//! # Determinism
//!
//! Metal on Apple GPUs has no `double` type, while the CPU accumulates each
//! histogram bin as a chain of `f64` additions in row order. The GPU kernel
//! instead accumulates each (feature, bin) cell in an *exact* double-float
//! pair (Knuth's two-sum) over fixed 65,536-row chunks, merges the chunk
//! partials in chunk order, and rounds once to `f64`. For any bin holding
//! fewer than 2^24 rows the whole pipeline is exact — zero intermediate
//! rounding — so the GPU histogram equals the single-threaded CPU histogram
//! bit for bit, and a `device = metal` training run reproduces the
//! single-threaded CPU model exactly. Beyond that, or when one bin's
//! gradients span more than ~2^29 in magnitude (where the CPU's own `f64`
//! chain starts rounding), the GPU value is the correctly rounded sum and
//! may differ from the CPU by an `f64` ulp. In all cases the GPU result is
//! identical across runs, thread counts, and machines: there are no atomics,
//! and every constant (chunk size, thread layout, merge order) is fixed.
//!
//! Guard rails keep the two paths identical in the edge cases where they
//! could not be: nodes below 8,192 rows and gradients large enough to
//! overflow an `f32` accumulator run on the CPU's sequential path inside the
//! GPU backend, and the kernels are compiled with safe math mode (and FP
//! contraction off) so the compiler cannot distort the two-sums or the
//! prediction's rounding order.
//!
//! # Limitations
//!
//! - macOS with a Metal device (Apple Silicon or an Intel Mac with a
//!   supported GPU). Training on a machine without one fails with an error,
//!   as do the combinations listed under
//!   [`TrainingParams::validate`](crate::config::TrainingParams::validate).
//! - Sparse (CSR, missing-value) training data is supported through an
//!   interleaved column copy (up to 8 bytes per row per feature block);
//!   datasets whose copy would exceed 4 GiB are refused.
//! - [`GpuModel`](crate::backend::metal::GpuModel) refuses `gblinear` and `linear_tree` models (they do not
//!   predict through the compact forest), and prediction needs a dense row
//!   copy, so `rows × features × 4` bytes must stay under 4 GiB.
//! - The gradient slice is re-uploaded once per tree (8 bytes per row) and
//!   each prediction call uploads its rows; on unified memory (Apple
//!   Silicon) these are plain memory copies.

// `MTLCreateSystemDefaultDevice` links against CoreGraphics; the binding
// crate documents this as the required linkage.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

use crate::data::ghist::{Bins, GHistIndex};
use crate::error::{HessboostError, Result};
use crate::model::{BoostedModel, initial_margins, transform_model_margins};
use crate::objective::GradPair;
use crate::tree::gain::GradStats;
use crate::tree::hist::HistogramBackend;
use crate::tree::hist::accumulate as cpu_accumulate;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};
use rayon::prelude::*;
use std::ops::RangeBounds;
use std::ptr::NonNull;
use std::ptr::copy_nonoverlapping;
use std::slice;
use std::sync::{Arc, LazyLock, Mutex};

/// Rows per GPU histogram chunk. A chunk's shared data (row ids plus
/// gradient pairs, 12 bytes per row) stays resident in the GPU's level-2
/// cache while every (feature, window) threadgroup sweeps it, so the
/// re-reads by the ~hundreds of threadgroups do not reach DRAM. A fixed
/// constant: it is part of the summation order, so it must not vary by
/// machine.
const CHUNK_ROWS: usize = 65_536;

/// Row slices per chunk: each chunk dispatch runs one threadgroup per
/// (feature window, slice), the `.y` grid dimension, so the scan has enough
/// threadgroups in flight to hide memory latency. A fixed constant: it is
/// part of the summation order and must not vary by machine.
const ROW_SLICES: usize = 64;
/// Nodes below this many rows run on the CPU's sequential path: the kernel
/// dispatch and readback cost more than the scan. Below this size the CPU
/// path is sequential anyway (it matches `CpuBackend`'s own threshold), so
/// the results agree bit for bit.
const CPU_ROWS: usize = 8_192;
/// Threads of one histogram threadgroup per feature: 64 threads times 4
/// register bins cover a feature's whole 256-bin window, so a group covers
/// every bin of each of its features.
const THREADS_PER_FEATURE: usize = 64;
/// Features packed into one interleaved column record.
const RECORD_FEATURES: usize = 8;
/// Largest feature count one threadgroup covers (1024 threads, the Metal
/// per-threadgroup ceiling).
const MAX_GROUP_FEATURES: usize = 16;
/// Bins owned by one histogram thread, held in registers.
const BINS_PER_THREAD: usize = 4;
/// Bins of one feature covered by a threadgroup's window.
const WINDOW_BINS: usize = THREADS_PER_FEATURE * BINS_PER_THREAD;
/// Threads per merge threadgroup (one bin per thread).
const MERGE_THREADS: usize = 64;
/// Threads per prediction threadgroup (rows are independent).
const PREDICT_THREADS: usize = 256;
/// Upper bound on GPU buffer sizes (entries), keeping index math in `u32`.
const MAX_BUFFER_ENTRIES: usize = 1 << 30;

/// Whether a Metal device and working compute pipelines are available.
/// `false` on hardware without Metal support (for example a macOS VM).
#[must_use]
pub fn available() -> bool {
    MetalContext::shared().is_some()
}

/// Why the Metal backend is unavailable (no device, or a kernel compile
/// failure), for diagnostics; `None` when it is available.
#[must_use]
pub fn unavailable_reason() -> Option<String> {
    MetalContext::shared()
        .is_none()
        .then(|| CONTEXT.as_ref().err().cloned().unwrap_or_default())
}

/// The name of the Metal device this process would use, if any (for
/// diagnostics and benchmarks).
#[must_use]
pub fn device_name() -> Option<String> {
    MetalContext::shared().map(|ctx| ctx.device_name.clone())
}
// ---------------------------------------------------------------------------
// Metal Shading Language kernels

/// The compute kernels, compiled from source once per process.
///
/// All arithmetic is `float` (Apple GPUs have no `double`); the histogram
/// accumulations use exact double-float two-sums, so the source must be
/// compiled with safe math mode (see [`MetalContext::new`]).
const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
// The prediction kernel's `out += weight * leaf` must round twice (multiply,
// then add) like the CPU: fused multiply-add would change results by an ulp.
#pragma clang fp contract(off)

// Accumulate one row's gradient pair into this thread's bin registers:
// `wl` is the row's bin offset within this thread's 32-bin window, `gh` its
// gradient pair. The owner check and the four-way register select stay
// branch-predicated; `j`, `n_win`, and `win_base` are thread-uniform.
#define ACC_BIN(wl_, gh) { \
    if ((wl_) >= 0 && (uint)(wl_) < n_win && (uint)((wl_) >> 2) == j) { \
        if (((wl_) & 3) == 0) { \
            DF_ADD(ghi0, glo0, (gh).x) \
            DF_ADD(hhi0, hlo0, (gh).y) \
        } else if (((wl_) & 3) == 1) { \
            DF_ADD(ghi1, glo1, (gh).x) \
            DF_ADD(hhi1, hlo1, (gh).y) \
        } else if (((wl_) & 3) == 2) { \
            DF_ADD(ghi2, glo2, (gh).x) \
            DF_ADD(hhi2, hlo2, (gh).y) \
        } else { \
            DF_ADD(ghi3, glo3, (gh).x) \
            DF_ADD(hhi3, hlo3, (gh).y) \
        } \
    } \
}

// One (feature block, bin-window) work item of the scan kernel. A block
// covers `features_per_group` features (see `HistRun`); a window covers 256
// bins of each of them.
struct ScanGroup {
    uint block;       // feature block whose bins this group accumulates
    uint window_base; // first bin of this window, feature-relative
};

// Per-block bin ranges: feature `f` of the block owns global bins
// `[fs[f], fs[f] + nbins[f])`; a feature past the dataset's end (the last
// block's padding) has `nbins = 0`. Sized for the largest feature group.
struct BlockInfo {
    uint fs[16];
    uint nbins[16];
};

// The rows of one chunk.
struct ChunkArgs { uint chunk_rows; uint chunk_index; uint slices; };
// Partial count of the merge.
struct MergeArgs { uint n_partials; };
// Dataset-wide kernel constants.
struct HistRun {
    uint total_bins;         // bins in one partial (and the histogram)
    uint n_records;          // interleaved column records per row
    uint features_per_group; // features one threadgroup covers
};

// Exact double-float add of `x` to `(hi, lo)` (Knuth's two-sum, then the
// error folded into `lo`). Exact while the running pair needs at most 48
// bits of significand, which a chunk's per-bin sum always does.
#define DF_ADD(hi, lo, x) { \
    float s = hi + (x); \
    float bb = s - hi; \
    float e = (hi - (s - bb)) + ((x) - bb); \
    float s2 = lo + e; \
    lo = s2; \
    hi = s; \
}

// Exact double-float add of `(bhi, blo)` to `(ahi, alo)` (two-sum on the
// high parts, remainder folded once).
#define DF_ADD_DF(ahi, alo, bhi, blo) { \
    float s = ahi + bhi; \
    float bb = s - ahi; \
    float e = (ahi - (s - bb)) + (bhi - bb); \
    float t = (alo + blo) + e; \
    alo = t; \
    ahi = s; \
}

// One threadgroup scans one (feature block, 32-bin window) of one row
// chunk: thread `tid` covers feature `tid / 8` and its bin quarter
// `(tid % 8) * 4`, four bins held in registers. Every bin has a single
// writer that adds its rows in ascending order, so the result is exact and
// deterministic. The interleaved column store packs 8 features' bins into
// one 16-byte word per (row, block), so a row step costs one uniform load
// for the row id, one for the word, and one for the gradient pair —
// amortized across the block's 8 features.
kernel void hist_scan_u16(
    const device uint* rows [[buffer(0)]],
    const device uint4* columns [[buffer(1)]],
    const device float2* gpair [[buffer(2)]],
    const device BlockInfo* blocks [[buffer(3)]],
    const device ScanGroup* groups [[buffer(4)]],
    device float4* partials [[buffer(5)]],
    constant ChunkArgs& chunk [[buffer(6)]],
    constant HistRun& run [[buffer(7)]],
    uint2 gpos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    ScanGroup sg = groups[gpos.x];
    // Thread `tid` covers feature `tid / 64` of the block and bin quarter
    // `(tid % 64) * 4`: 64 threads times 4 register bins cover a whole
    // 256-bin window of one feature, so one group covers every bin of its
    // `features_per_group` features and the row ids and gradient pairs load
    // once per group instead of once per feature.
    uint f = tid / 64u;
    uint j = tid % 64u;
    uint my_feature = sg.block * run.features_per_group + f;
    uint record = my_feature >> 3;
    uint word = (my_feature & 7u) >> 1;
    uint shift = (my_feature & 1u) * 16u;
    uint fs = blocks[sg.block].fs[f];
    uint nbins = blocks[sg.block].nbins[f];
    uint win_base = fs + sg.window_base;
    uint n_win = (nbins > sg.window_base)
        ? min(256u, nbins - sg.window_base)
        : 0u;
    uint slice_len = (chunk.chunk_rows + chunk.slices - 1u) / chunk.slices;
    uint slice_begin = gpos.y * slice_len;
    // Slices beyond this chunk's rows (the grid always launches
    // `chunk.slices` of them, and a tail chunk holds fewer) contribute
    // nothing: their loops are empty and their partials still write as
    // zeros, which the merge adds harmlessly. The guarded subtraction
    // cannot underflow.
    uint slice_rows = (slice_begin < chunk.chunk_rows)
        ? min(slice_len, chunk.chunk_rows - slice_begin)
        : 0u;
    float ghi0 = 0.0f, ghi1 = 0.0f, ghi2 = 0.0f, ghi3 = 0.0f;
    float glo0 = 0.0f, glo1 = 0.0f, glo2 = 0.0f, glo3 = 0.0f;
    float hhi0 = 0.0f, hhi1 = 0.0f, hhi2 = 0.0f, hhi3 = 0.0f;
    float hlo0 = 0.0f, hlo1 = 0.0f, hlo2 = 0.0f, hlo3 = 0.0f;
    // Batches of 8: the row ids load together, then the column words and
    // gradient pairs, so the dependent uniform loads pipeline across the
    // batch. The accumulation order per bin is unchanged (ascending rows
    // within the slice).
    uint i = 0;
    for (; i + 8 <= slice_rows; i += 8) {
        uint r0 = rows[slice_begin + i + 0];
        uint r1 = rows[slice_begin + i + 1];
        uint r2 = rows[slice_begin + i + 2];
        uint r3 = rows[slice_begin + i + 3];
        uint r4 = rows[slice_begin + i + 4];
        uint r5 = rows[slice_begin + i + 5];
        uint r6 = rows[slice_begin + i + 6];
        uint r7 = rows[slice_begin + i + 7];
        uint4 c0 = columns[(size_t)r0 * run.n_records + record];
        uint4 c1 = columns[(size_t)r1 * run.n_records + record];
        uint4 c2 = columns[(size_t)r2 * run.n_records + record];
        uint4 c3 = columns[(size_t)r3 * run.n_records + record];
        uint4 c4 = columns[(size_t)r4 * run.n_records + record];
        uint4 c5 = columns[(size_t)r5 * run.n_records + record];
        uint4 c6 = columns[(size_t)r6 * run.n_records + record];
        uint4 c7 = columns[(size_t)r7 * run.n_records + record];
        float2 q0 = gpair[r0];
        float2 q1 = gpair[r1];
        float2 q2 = gpair[r2];
        float2 q3 = gpair[r3];
        float2 q4 = gpair[r4];
        float2 q5 = gpair[r5];
        float2 q6 = gpair[r6];
        float2 q7 = gpair[r7];
        ACC_BIN((int)((c0[word] >> shift) & 0xFFFFu) - (int)win_base, q0)
        ACC_BIN((int)((c1[word] >> shift) & 0xFFFFu) - (int)win_base, q1)
        ACC_BIN((int)((c2[word] >> shift) & 0xFFFFu) - (int)win_base, q2)
        ACC_BIN((int)((c3[word] >> shift) & 0xFFFFu) - (int)win_base, q3)
        ACC_BIN((int)((c4[word] >> shift) & 0xFFFFu) - (int)win_base, q4)
        ACC_BIN((int)((c5[word] >> shift) & 0xFFFFu) - (int)win_base, q5)
        ACC_BIN((int)((c6[word] >> shift) & 0xFFFFu) - (int)win_base, q6)
        ACC_BIN((int)((c7[word] >> shift) & 0xFFFFu) - (int)win_base, q7)
    }
    for (; i < slice_rows; i++) {
        uint r = rows[slice_begin + i];
        uint4 c = columns[(size_t)r * run.n_records + record];
        float2 gh = gpair[r];
        ACC_BIN((int)((c[word] >> shift) & 0xFFFFu) - (int)win_base, gh)
    }
    device float4* out = partials + (size_t)(chunk.chunk_index * chunk.slices + gpos.y) * run.total_bins;
    uint base = win_base + j * 4;
    if (j * 4 + 0 < n_win) { out[base + 0] = float4(ghi0, glo0, hhi0, hlo0); }
    if (j * 4 + 1 < n_win) { out[base + 1] = float4(ghi1, glo1, hhi1, hlo1); }
    if (j * 4 + 2 < n_win) { out[base + 2] = float4(ghi2, glo2, hhi2, hlo2); }
    if (j * 4 + 3 < n_win) { out[base + 3] = float4(ghi3, glo3, hhi3, hlo3); }
}

// The wide-bin variant (more than 65,536 total bins, or a sparse dataset
// that cannot spare a `u16` sentinel): the interleaved store holds eight
// `u32` bins per (row, block), read as one coalesced 32-byte run.
kernel void hist_scan_u32(
    const device uint* rows [[buffer(0)]],
    const device uint* columns [[buffer(1)]],
    const device float2* gpair [[buffer(2)]],
    const device BlockInfo* blocks [[buffer(3)]],
    const device ScanGroup* groups [[buffer(4)]],
    device float4* partials [[buffer(5)]],
    constant ChunkArgs& chunk [[buffer(6)]],
    constant HistRun& run [[buffer(7)]],
    uint2 gpos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    ScanGroup sg = groups[gpos.x];
    // See hist_scan_u16 for the thread mapping.
    uint f = tid / 64u;
    uint j = tid % 64u;
    uint my_feature = sg.block * run.features_per_group + f;
    uint record = my_feature >> 3;
    uint slot = my_feature & 7u;
    uint fs = blocks[sg.block].fs[f];
    uint nbins = blocks[sg.block].nbins[f];
    uint win_base = fs + sg.window_base;
    uint n_win = (nbins > sg.window_base)
        ? min(256u, nbins - sg.window_base)
        : 0u;
    uint slice_len = (chunk.chunk_rows + chunk.slices - 1u) / chunk.slices;
    uint slice_begin = gpos.y * slice_len;
    // See hist_scan_u16: tail-chunk slices contribute nothing.
    uint slice_rows = (slice_begin < chunk.chunk_rows)
        ? min(slice_len, chunk.chunk_rows - slice_begin)
        : 0u;
    float ghi0 = 0.0f, ghi1 = 0.0f, ghi2 = 0.0f, ghi3 = 0.0f;
    float glo0 = 0.0f, glo1 = 0.0f, glo2 = 0.0f, glo3 = 0.0f;
    float hhi0 = 0.0f, hhi1 = 0.0f, hhi2 = 0.0f, hhi3 = 0.0f;
    float hlo0 = 0.0f, hlo1 = 0.0f, hlo2 = 0.0f, hlo3 = 0.0f;
    uint stride = run.n_records * 8u;
    uint i = 0;
    for (; i + 8 <= slice_rows; i += 8) {
        uint r0 = rows[slice_begin + i + 0];
        uint r1 = rows[slice_begin + i + 1];
        uint r2 = rows[slice_begin + i + 2];
        uint r3 = rows[slice_begin + i + 3];
        uint r4 = rows[slice_begin + i + 4];
        uint r5 = rows[slice_begin + i + 5];
        uint r6 = rows[slice_begin + i + 6];
        uint r7 = rows[slice_begin + i + 7];
        uint b0 = columns[(size_t)r0 * stride + record * 8u + slot];
        uint b1 = columns[(size_t)r1 * stride + record * 8u + slot];
        uint b2 = columns[(size_t)r2 * stride + record * 8u + slot];
        uint b3 = columns[(size_t)r3 * stride + record * 8u + slot];
        uint b4 = columns[(size_t)r4 * stride + record * 8u + slot];
        uint b5 = columns[(size_t)r5 * stride + record * 8u + slot];
        uint b6 = columns[(size_t)r6 * stride + record * 8u + slot];
        uint b7 = columns[(size_t)r7 * stride + record * 8u + slot];
        float2 q0 = gpair[r0];
        float2 q1 = gpair[r1];
        float2 q2 = gpair[r2];
        float2 q3 = gpair[r3];
        float2 q4 = gpair[r4];
        float2 q5 = gpair[r5];
        float2 q6 = gpair[r6];
        float2 q7 = gpair[r7];
        ACC_BIN((int)b0 - (int)win_base, q0)
        ACC_BIN((int)b1 - (int)win_base, q1)
        ACC_BIN((int)b2 - (int)win_base, q2)
        ACC_BIN((int)b3 - (int)win_base, q3)
        ACC_BIN((int)b4 - (int)win_base, q4)
        ACC_BIN((int)b5 - (int)win_base, q5)
        ACC_BIN((int)b6 - (int)win_base, q6)
        ACC_BIN((int)b7 - (int)win_base, q7)
    }
    for (; i < slice_rows; i++) {
        uint r = rows[slice_begin + i];
        uint b = columns[(size_t)r * stride + record * 8u + slot];
        float2 gh = gpair[r];
        ACC_BIN((int)b - (int)win_base, gh)
    }
    device float4* out = partials + (size_t)(chunk.chunk_index * chunk.slices + gpos.y) * run.total_bins;
    uint base = win_base + j * 4;
    if (j * 4 + 0 < n_win) { out[base + 0] = float4(ghi0, glo0, hhi0, hlo0); }
    if (j * 4 + 1 < n_win) { out[base + 1] = float4(ghi1, glo1, hhi1, hlo1); }
    if (j * 4 + 2 < n_win) { out[base + 2] = float4(ghi2, glo2, hhi2, hlo2); }
    if (j * 4 + 3 < n_win) { out[base + 3] = float4(ghi3, glo3, hhi3, hlo3); }
}

// Merge the chunk partials of every bin in chunk order, rounding once.
kernel void hist_merge(
    const device float4* partials [[buffer(0)]],
    device float4* hist [[buffer(1)]],
    constant MergeArgs& merge [[buffer(2)]],
    constant HistRun& run [[buffer(3)]],
    uint b [[thread_position_in_grid]])
{
    if (b >= run.total_bins) { return; }
    float ghi = 0.0f, glo = 0.0f, hhi = 0.0f, hlo = 0.0f;
    for (uint c = 0; c < merge.n_partials; c++) {
        float4 p = partials[(size_t)c * run.total_bins + b];
        DF_ADD_DF(ghi, glo, p.x, p.y)
        DF_ADD_DF(hhi, hlo, p.z, p.w)
    }
    hist[b] = float4(ghi, glo, hhi, hlo);
}

// One compact-forest node: the layout of `CNode` in `tree/compact.rs`.
//   slot: feature * 32 (+16 when the split reads the negated value)
//   key:  threshold key ("go right when greater"); `cat_begin` if categorical
//   left: child taken when the compare is false; a leaf points at itself
//   aux:  0 numeric; leaf value bits or leaf-vector offset; categorical flags
struct PNode { uint slot; uint key; uint left; uint aux; };
// One tree's dispatch record.
struct PTree { uint root; float weight; uint output; uint vector; };
struct PredictArgs { uint n_rows; uint n_cols; uint k; uint tree_begin; uint tree_end; };

// Monotone unsigned key of a float's bits, matching `tree::compact::key`:
// NaN maps to 0, -0.0 is treated as +0.0, negatives are complemented.
inline uint key_of(uint vb) {
    if ((vb & 0x7F800000u) == 0x7F800000u && (vb & 0x007FFFFFu) != 0u) { return 0u; }
    if (vb == 0x80000000u) { vb = 0u; }
    return vb ^ (uint((int)vb >> 31) | 0x80000000u);
}

// Whether a non-missing value is in the categorical left set, matching
// `tree::in_category_set` (including Rust's saturating `as u32` semantics).
inline bool cat_in_set(
    const device uint* categories, uint begin, uint end, float v)
{
    uint cat;
    if (v >= 4294967296.0f) { cat = 0xFFFFFFFFu; }
    else if (v <= 0.0f) { cat = 0u; }
    else { cat = (uint)v; }
    for (uint i = begin; i < end; i++) {
        if (categories[i] == cat) { return true; }
    }
    return false;
}

// One row per thread: walk every tree of the range, adding each tree's
// weighted leaf value onto the (pre-initialized) margins in tree order, the
// same order the CPU accumulates, so the sums are bit-identical.
// Single-output models keep the running margin in a register (the adds are
// the same sequence, just without the per-tree global round trip);
// multi-output models read-modify-write their slot per tree.
kernel void forest_predict(
    const device uint4* nodes [[buffer(0)]],
    const device uint* categories [[buffer(1)]],
    const device float* leaf_vectors [[buffer(2)]],
    const device PTree* trees [[buffer(3)]],
    const device float* rows [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant PredictArgs& a [[buffer(6)]],
    uint r [[thread_position_in_grid]])
{
    if (r >= a.n_rows) { return; }
    const device float* row = rows + (size_t)r * a.n_cols;
    bool scalar_run = (a.k == 1u);
    float acc = scalar_run ? out[r] : 0.0f;
    for (uint t = a.tree_begin; t < a.tree_end; t++) {
        PTree tr = trees[t];
        uint nid = tr.root;
        uint4 n;
        while (true) {
            n = nodes[nid];
            if (n.z == nid) { break; }
            float v = row[n.x / 32u];
            if (n.w & 1u) {
                bool go_left = isnan(v)
                    ? (n.w & 2u) != 0u
                    : cat_in_set(categories, n.y, n.w >> 2u, v);
                nid = n.z + (go_left ? 0u : 1u);
            } else {
                uint vb = as_type<uint>(v);
                if (n.x & 16u) { vb ^= 0x80000000u; }
                nid = n.z + (key_of(vb) > n.y ? 1u : 0u);
            }
        }
        if (tr.vector != 0u) {
            device float* o = out + (size_t)r * a.k;
            const device float* lv = leaf_vectors + n.w;
            for (uint j = 0; j < a.k; j++) { o[j] += tr.weight * lv[j]; }
        } else if (scalar_run) {
            acc += tr.weight * as_type<float>(n.w);
        } else {
            out[(size_t)r * a.k + tr.output] += tr.weight * as_type<float>(n.w);
        }
    }
    if (scalar_run) { out[r] = acc; }
}
"#;

// ---------------------------------------------------------------------------
// Runtime context
// ---------------------------------------------------------------------------

/// An immutable Metal pipeline state. `objc2-metal` only derives
/// `Send`/`Sync` for protocols that declare it; `MTLComputePipelineState`
/// does not, although Apple documents pipeline states as thread-safe
/// immutable objects.
struct Pipeline(Retained<ProtocolObject<dyn MTLComputePipelineState>>);
// SAFETY: `MTLComputePipelineState` is immutable after creation and Apple
// documents it as safe to use from any thread.
unsafe impl Send for Pipeline {}
// SAFETY: see the `Send` impl.
unsafe impl Sync for Pipeline {}

/// A Metal buffer in shared (unified) memory, CPU-writable without blits.
struct GpuBuffer(Retained<ProtocolObject<dyn MTLBuffer>>);
// SAFETY: the `MTLBuffer` *object* is thread-safe; the memory it owns is
// governed by this backend's ownership rules — writes go through `unsafe`
// methods whose contracts require exclusive access, and the safe methods are
// read-only.
unsafe impl Send for GpuBuffer {}
// SAFETY: see the `Send` impl: concurrent access happens only through the
// read-only safe methods.
unsafe impl Sync for GpuBuffer {}

impl GpuBuffer {
    /// Allocate `len` bytes of shared storage.
    fn new(device: &ProtocolObject<dyn MTLDevice>, len: usize) -> Result<Self> {
        // Metal drivers may return `nil` for zero-length buffers; empty
        // logical buffers (never read by the kernels) get a small stand-in.
        let len = len.max(16);
        device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .map(GpuBuffer)
            .ok_or_else(|| HessboostError::gpu("allocating a Metal buffer failed"))
    }

    /// Length in bytes.
    fn len(&self) -> usize {
        self.0.length()
    }

    /// The buffer's contents as a mutable slice of `n` elements of `T`.
    ///
    /// # Safety
    ///
    /// The caller must exclusively own the buffer: no in-flight GPU work may
    /// read it, and no other CPU access may alias the returned slice. `T`
    /// must be a plain data type matching the kernel's layout.
    #[allow(
        clippy::mut_from_ref,
        reason = "the caller proves exclusive access; the buffer is plain shared memory"
    )]
    unsafe fn as_slice_mut<T>(&self, n: usize) -> &mut [T] {
        debug_assert!(n * std::mem::size_of::<T>() <= self.len());
        // SAFETY: the caller guarantees exclusive access (see above); Metal
        // shared buffers are plain CPU-accessible memory of `self.len()`
        // bytes, and `n * size_of::<T>()` is within it.
        unsafe { slice::from_raw_parts_mut(self.0.contents().as_ptr().cast(), n) }
    }

    /// Copy `bytes` into the buffer at `offset`. No-op for empty input.
    ///
    /// # Safety
    ///
    /// The caller must exclusively own the buffer (see [`Self::as_slice_mut`]).
    unsafe fn write(&self, offset: usize, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        debug_assert!(offset + bytes.len() <= self.len());
        // SAFETY: exclusive access per the caller; the destination lies
        // within the buffer and cannot overlap the source.
        unsafe {
            copy_nonoverlapping(
                bytes.as_ptr(),
                self.0.contents().as_ptr().cast::<u8>().add(offset),
                bytes.len(),
            );
        }
    }
}

/// The bytes of a slice of plain data.
///
/// # Safety
///
/// `T` must have no padding.
unsafe fn as_bytes<T>(v: &[T]) -> &[u8] {
    // SAFETY: the caller guarantees a plain-data element type; the byte
    // length follows from the element size.
    unsafe { slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

/// The process-wide Metal context: device, queue, and compiled kernels.
struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    hist_u16: Pipeline,
    hist_u32: Pipeline,
    hist_merge: Pipeline,
    forest_predict: Pipeline,
    device_name: String,
}

/// The lazily initialized process-wide Metal context, or the reason it is
/// unavailable (no device, or a kernel compile failure).
static CONTEXT: LazyLock<Result<MetalContext, String>> = LazyLock::new(MetalContext::new);

impl MetalContext {
    /// The shared context, or `None` when it failed to initialize
    /// (see [`unavailable_reason`]). Computed once per process.
    fn shared() -> Option<&'static Self> {
        CONTEXT.as_ref().ok()
    }

    fn new() -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| "no system default Metal device".to_string())?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| "creating the Metal command queue failed".to_string())?;
        // Safe math mode: the two-sums require exact IEEE 754 adds with no
        // FMA contraction, and the compile-time default (`fastMathEnabled`)
        // permits both.
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        let source = NSString::from_str(MSL);
        let library = device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|e| format!("compiling the Metal kernels failed: {e}"))?;
        let pipeline = |name: &str| -> Result<Pipeline, String> {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| format!("kernel `{name}` is missing from the library"))?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map(Pipeline)
                .map_err(|e| format!("building the `{name}` pipeline failed: {e}"))
        };
        Ok(MetalContext {
            hist_u16: pipeline("hist_scan_u16")?,
            hist_u32: pipeline("hist_scan_u32")?,
            hist_merge: pipeline("hist_merge")?,
            forest_predict: pipeline("forest_predict")?,
            device_name: device.name().to_string(),
            device,
            queue,
        })
    }

    /// A new command buffer. Per call: command buffers are not thread-safe
    /// objects, so they are never stored.
    fn command_buffer(&self) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        self.queue
            .commandBuffer()
            .ok_or_else(|| HessboostError::gpu("creating a Metal command buffer failed"))
    }
}

// ---------------------------------------------------------------------------
// Histogram backend
/// One (feature block, bin window) histogram work item, uploaded as the
/// kernel's group list.
#[repr(C)]
struct ScanGroup {
    block: u32,
    window_base: u32,
}

/// Per-block bin ranges for the kernel: feature `f` of the block owns
/// global bins `[fs[f], fs[f] + nbins[f])`; `nbins = 0` marks padding past
/// the dataset's feature count.
#[repr(C)]
struct BlockInfo {
    fs: [u32; MAX_GROUP_FEATURES],
    nbins: [u32; MAX_GROUP_FEATURES],
}

/// Kernel argument block for one chunk dispatch.
#[repr(C)]
#[derive(Clone, Copy)]
struct ChunkArgs {
    chunk_rows: u32,
    chunk_index: u32,
    slices: u32,
}

/// Kernel argument block of the merge dispatch.
#[repr(C)]
#[derive(Clone, Copy)]
struct MergeArgs {
    n_partials: u32,
}

/// Dataset-wide kernel constants.
#[repr(C)]
#[derive(Clone, Copy)]
struct HistRun {
    total_bins: u32,
    n_records: u32,
    features_per_group: u32,
}

/// The identity (address) of a gradient slice, for change detection. A
/// newtype because raw pointers are neither `Send` nor `Sync`; it is only
/// ever compared, never dereferenced.
struct SliceId(*const GradPair);
// SAFETY: the pointer is never dereferenced or otherwise used, only
// compared for identity.
unsafe impl Send for SliceId {}

/// The gradient slice staged by [`HistogramBackend::prepare`], uploaded to
/// the GPU together with the magnitude bound the overflow guard needs.
struct PreparedGradients {
    buffer: GpuBuffer,
    /// (pointer, length) identity of the uploaded slice.
    ptr: SliceId,
    len: usize,
    /// `max(|grad|)` and `max(|hess|)` over the slice (NaN inputs are
    /// ignored, matching `f32::max`; an infinity always trips the guard).
    max_g: f32,
    max_h: f32,
}

/// Per-call GPU buffers of the histogram backend, pooled across the parallel
/// node builds of a training run. Each concurrent `build` owns one set.
struct CallBuffers {
    rows: GpuBuffer,
    partials: GpuBuffer,
    hist: GpuBuffer,
}

/// The Metal histogram backend: implements [`HistogramBackend`] by scanning
/// the binned column store on the GPU. Constructed once per training run (the
/// column upload and group descriptors are per-dataset); the gradient slice
/// is re-uploaded by [`HistogramBackend::prepare`] once per tree.
///
/// Training selects it automatically through
/// [`device = metal`](crate::config::TrainingParams::device); constructing it
/// directly serves custom training loops against a [`GHistIndex`].
pub struct MetalHistBackend {
    ctx: &'static MetalContext,
    /// `true` when `columns` packs each record as eight `u16` bins (dense
    /// data, or sparse data with a spare `u16` sentinel); `false` for eight
    /// `u32` bins (wider bin counts).
    columns_u16: bool,
    /// The interleaved column store: one record per (row, feature block).
    columns: GpuBuffer,
    blocks_bytes: GpuBuffer,
    groups: Vec<ScanGroup>,
    groups_bytes: GpuBuffer,
    /// Threads per scan threadgroup (`features_per_group` × 64).
    threads_per_group: usize,
    total_bins: usize,
    n_rows: usize,
    run: HistRun,
    gradients: Mutex<PreparedGradients>,
    pool: Mutex<Vec<CallBuffers>>,
}

impl MetalHistBackend {
    /// Build the backend for `index`: upload (or pad) its feature-major
    /// column store and precompute the (feature, bin window) group list.
    pub fn new(index: &GHistIndex) -> Result<Self> {
        let ctx = MetalContext::shared().ok_or_else(|| {
            HessboostError::gpu(
                unavailable_reason()
                    .unwrap_or_else(|| "no Metal device is available on this machine".into()),
            )
        })?;
        let n_rows = index.n_rows();
        let n_cols = index.n_cols();
        let total_bins = index.total_bins();
        if total_bins == 0 || n_rows == 0 {
            return Err(HessboostError::invalid_param(
                "device",
                "the Metal backend needs a non-empty binned dataset",
            ));
        }
        if total_bins > MAX_BUFFER_ENTRIES || n_rows > MAX_BUFFER_ENTRIES {
            return Err(HessboostError::invalid_param(
                "device",
                format!(
                    "the dataset exceeds the Metal backend's index limits \
                     ({n_rows} rows, {total_bins} bins)"
                ),
            ));
        }
        // Interleaved column store, the scan kernels' layout: one record
        // per (row, `RECORD_FEATURES` features). `u16` records (one
        // 16-byte word) cover dense indexes and sparse ones with a spare
        // `u16` sentinel; wider bin counts (or a sparse index that cannot
        // spare one) get `u32` records.
        let n_records = n_cols.div_ceil(RECORD_FEATURES);
        let columns_u16 = match index.column_bins() {
            Some(Bins::U16(_)) => true,
            Some(Bins::U32(_)) => false,
            None => u16::try_from(total_bins).is_ok(),
        };
        let columns = interleaved_columns(&ctx.device, index, columns_u16, n_records)?;
        // Feature blocks per threadgroup. Measured on Apple Silicon: one
        // feature per 64-thread group (a whole 256-bin window) beats every
        // wider block — the interleaved record load costs more than the
        // row/gradient amortization saves, and 512+-thread groups lose
        // occupancy to register pressure. A fixed constant: it is part of
        // the summation order. Wider blocks are a tuning direction if the
        // scan kernels are reworked around the scalar-load path.
        let features_per_group = 1;
        let threads_per_group = features_per_group * THREADS_PER_FEATURE;
        // Per-block bin ranges, and one group per (block, 256-bin window);
        // the block's deepest feature decides the window count.
        let cuts = index.cuts();
        let n_group_blocks = n_cols.div_ceil(features_per_group);
        let mut blocks = Vec::with_capacity(n_group_blocks);
        let mut groups = Vec::new();
        for b in 0..n_group_blocks {
            let mut info = BlockInfo {
                fs: [0; MAX_GROUP_FEATURES],
                nbins: [0; MAX_GROUP_FEATURES],
            };
            for f in 0..features_per_group {
                let feature = b * features_per_group + f;
                if feature < n_cols {
                    let (fs, fe) = cuts.feature_bins(feature);
                    info.fs[f] = fs as u32;
                    info.nbins[f] = (fe - fs) as u32;
                }
            }
            let windows = info
                .nbins
                .iter()
                .map(|&n| n.div_ceil(WINDOW_BINS as u32))
                .max()
                .unwrap_or(0);
            for w in 0..windows {
                groups.push(ScanGroup {
                    block: b as u32,
                    window_base: w * WINDOW_BINS as u32,
                });
            }
            blocks.push(info);
        }
        let blocks_bytes = GpuBuffer::new(&ctx.device, blocks.len() * 128)?;
        // SAFETY: freshly allocated buffer, written once before any
        // dispatch; `BlockInfo` is `repr(C)` of two arrays of sixteen
        // `u32`s.
        unsafe { blocks_bytes.write(0, as_bytes(&blocks)) };
        let groups_bytes = GpuBuffer::new(&ctx.device, groups.len() * 8)?;
        // SAFETY: see above; `ScanGroup` is `repr(C)` of two `u32`s.
        unsafe { groups_bytes.write(0, as_bytes(&groups)) };
        let run = HistRun {
            total_bins: total_bins as u32,
            n_records: n_records as u32,
            features_per_group: features_per_group as u32,
        };
        let chunks = n_rows.div_ceil(CHUNK_ROWS).max(1);
        let call = CallBuffers {
            rows: GpuBuffer::new(&ctx.device, n_rows * 4)?,
            partials: GpuBuffer::new(&ctx.device, chunks * ROW_SLICES * total_bins * 16)?,
            hist: GpuBuffer::new(&ctx.device, total_bins * 16)?,
        };
        let gpair = GpuBuffer::new(&ctx.device, n_rows * 8)?;
        Ok(MetalHistBackend {
            ctx,
            columns_u16,
            columns,
            blocks_bytes,
            groups,
            groups_bytes,
            threads_per_group,
            total_bins,
            n_rows,
            run,
            gradients: Mutex::new(PreparedGradients {
                buffer: gpair,
                ptr: SliceId(std::ptr::null()),
                len: 0,
                max_g: 0.0,
                max_h: 0.0,
            }),
            pool: Mutex::new(vec![call]),
        })
    }

    /// Stage `gpair` on the GPU, returning the magnitude bound of the staged
    /// data.
    ///
    /// `force` stages unconditionally: the trainer refills its gradient
    /// buffer in place every round, so a new tree's `prepare` cannot rely on
    /// the slice's identity. Within one tree (a `build` whose `prepare` just
    /// ran) the identity check skips the re-upload: the slice is constant
    /// while a tree grows.
    fn ensure_gradients(&self, gpair: &[GradPair], force: bool) -> f32 {
        let mut prepared = self.gradients.lock().expect("gradient lock poisoned");
        let (ptr, len) = (gpair.as_ptr(), gpair.len());
        if force || prepared.ptr.0 != ptr || prepared.len != len {
            // SAFETY: the gradient buffer is only written here, under the
            // gradient lock, while no GPU work reads it: every `build` waits
            // for its own command buffer before returning, and trees are
            // built one at a time.
            unsafe { prepared.buffer.write(0, as_bytes(gpair)) };
            prepared.max_g = gpair.iter().map(|p| p.grad.abs()).fold(0.0f32, f32::max);
            prepared.max_h = gpair.iter().map(|p| p.hess.abs()).fold(0.0f32, f32::max);
            prepared.ptr = SliceId(ptr);
            prepared.len = len;
        }
        prepared.max_g.max(prepared.max_h)
    }

    /// Check out a per-call buffer set from the pool.
    fn checkout(&self) -> Result<CallBuffers> {
        if let Some(call) = self.pool.lock().expect("buffer pool lock poisoned").pop() {
            return Ok(call);
        }
        let chunks = self.n_rows.div_ceil(CHUNK_ROWS).max(1);
        Ok(CallBuffers {
            rows: GpuBuffer::new(&self.ctx.device, self.n_rows * 4)?,
            partials: GpuBuffer::new(&self.ctx.device, chunks * ROW_SLICES * self.total_bins * 16)?,
            hist: GpuBuffer::new(&self.ctx.device, self.total_bins * 16)?,
        })
    }

    fn checkin(&self, call: CallBuffers) {
        self.pool
            .lock()
            .expect("buffer pool lock poisoned")
            .push(call);
    }

    /// The sequential CPU path, used below the row threshold and for guard
    /// trips. It matches the GPU's semantics: row-order `f64` accumulation
    /// of the same exact sums.
    fn serial(ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]) {
        out.fill(GradStats::default());
        cpu_accumulate(ghist, rows, gpair, out);
    }

    /// Run the GPU accumulation of `rows` into `out`. The guards have
    /// passed and the gradients are staged.
    fn gpu_build(&self, rows: &[u32], out: &mut [GradStats]) -> Result<()> {
        let call = self.checkout()?;
        let result = self.dispatch(&call, rows, out);
        self.checkin(call);
        result
    }

    fn dispatch(&self, call: &CallBuffers, rows: &[u32], out: &mut [GradStats]) -> Result<()> {
        let cb = self.ctx.command_buffer()?;
        // SAFETY: this call exclusively owns `call`; nothing has been
        // dispatched on it yet, and `rows` are `u32`s within its capacity.
        unsafe { call.rows.write(0, as_bytes(rows)) };
        let chunks = rows.len().div_ceil(CHUNK_ROWS);
        for c in 0..chunks {
            let begin = c * CHUNK_ROWS;
            let chunk_rows = (rows.len() - begin).min(CHUNK_ROWS);
            let args = ChunkArgs {
                chunk_rows: chunk_rows as u32,
                chunk_index: c as u32,
                slices: ROW_SLICES as u32,
            };
            // SAFETY: the argument block is a live `repr(C)` plain-data
            // local outliving the encoder, and its pointer/length are valid.
            unsafe {
                let enc = cb
                    .computeCommandEncoder()
                    .ok_or_else(|| HessboostError::gpu("encoder creation failed"))?;
                let pipe = if self.columns_u16 {
                    &self.ctx.hist_u16
                } else {
                    &self.ctx.hist_u32
                };
                enc.setComputePipelineState(&pipe.0);
                enc.setBuffer_offset_atIndex(Some(&call.rows.0), begin * 4, 0);
                enc.setBuffer_offset_atIndex(Some(&self.columns.0), 0, 1);
                let gradients = self.gradients.lock().expect("gradient lock poisoned");
                enc.setBuffer_offset_atIndex(Some(&gradients.buffer.0), 0, 2);
                drop(gradients);
                enc.setBuffer_offset_atIndex(Some(&self.blocks_bytes.0), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&self.groups_bytes.0), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&call.partials.0), 0, 5);
                enc.setBytes_length_atIndex(
                    NonNull::from(&args).cast(),
                    std::mem::size_of::<ChunkArgs>(),
                    6,
                );
                enc.setBytes_length_atIndex(
                    NonNull::from(&self.run).cast(),
                    std::mem::size_of::<HistRun>(),
                    7,
                );
                let grid = MTLSize {
                    width: self.groups.len(),
                    height: ROW_SLICES,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: self.threads_per_group,
                    height: 1,
                    depth: 1,
                };
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                enc.endEncoding();
            }
        }
        // SAFETY: encoders of one command buffer run in the order they were
        // created, so the merge reads complete chunk partials; the argument
        // blocks are live plain-data locals.
        unsafe {
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| HessboostError::gpu("encoder creation failed"))?;
            enc.setComputePipelineState(&self.ctx.hist_merge.0);
            enc.setBuffer_offset_atIndex(Some(&call.partials.0), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&call.hist.0), 0, 1);
            let args = MergeArgs {
                n_partials: (chunks * ROW_SLICES) as u32,
            };
            enc.setBytes_length_atIndex(
                NonNull::from(&args).cast(),
                std::mem::size_of::<MergeArgs>(),
                2,
            );
            enc.setBytes_length_atIndex(
                NonNull::from(&self.run).cast(),
                std::mem::size_of::<HistRun>(),
                3,
            );
            let grid = MTLSize {
                width: self.total_bins,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: MERGE_THREADS,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
        // SAFETY: the command buffer has completed, so the GPU is done with
        // the buffer; this call owns it.
        let hist = unsafe { call.hist.as_slice_mut::<[f32; 4]>(self.total_bins) };
        for (o, h) in out.iter_mut().zip(hist) {
            // (hi, lo) as f64 sums exactly: the pair holds at most 48
            // significand bits and f64 carries 53.
            o.grad = f64::from(h[0]) + f64::from(h[1]);
            o.hess = f64::from(h[2]) + f64::from(h[3]);
        }
        Ok(())
    }
}

impl HistogramBackend for MetalHistBackend {
    fn build(&self, ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]) {
        debug_assert_eq!(out.len(), self.total_bins);
        // Small nodes and guard trips stay on the sequential CPU path, which
        // the GPU reproduces exactly: both are row-order accumulation of the
        // same exact sums.
        if rows.len() < CPU_ROWS {
            Self::serial(ghist, rows, gpair, out);
            return;
        }
        let max = self.ensure_gradients(gpair, false);
        if f64::from(max) * rows.len() as f64 > f64::from(f32::MAX) {
            Self::serial(ghist, rows, gpair, out);
            return;
        }
        match self.gpu_build(rows, out) {
            Ok(()) => {}
            // A dispatch failure is not recoverable on the GPU, but the
            // histogram is a pure function of the inputs: fall back to the
            // CPU rather than failing the training run.
            Err(_) => Self::serial(ghist, rows, gpair, out),
        }
    }

    fn prepare(&self, _ghist: &GHistIndex, gpair: &[GradPair]) {
        self.ensure_gradients(gpair, true);
    }
}

/// Build the interleaved column store the scan kernels read: one record per
/// (row, feature block of [`RECORD_FEATURES`]), each packing the block's
/// eight bins — `u16` records as one 16-byte word (`u16::MAX` marking a
/// missing entry), `u32` records as two (`u32::MAX` marking one).
///
/// Dense indexes read their existing feature-major store; sparse indexes
/// walk each row's CSR entries once, filling the rest with the sentinel.
fn interleaved_columns(
    device: &ProtocolObject<dyn MTLDevice>,
    index: &GHistIndex,
    u16_pack: bool,
    n_records: usize,
) -> Result<GpuBuffer> {
    let n = index.n_rows();
    let f_count = index.n_cols();
    let record = if u16_pack { 4 } else { 8 };
    let words = n
        .checked_mul(n_records)
        .filter(|&r| r * record <= MAX_BUFFER_ENTRIES)
        .ok_or_else(|| {
            HessboostError::invalid_param(
                "device",
                format!(
                    "the interleaved column copy the Metal backend needs \
                     ({n} rows x {f_count} features) exceeds 4 GiB"
                ),
            )
        })?
        * record;
    let buffer = GpuBuffer::new(device, words * 4)?;
    // SAFETY: the buffer was just allocated and is not yet shared; its
    // length covers `words` u32s.
    let store = unsafe { buffer.as_slice_mut::<u32>(words) };
    let sentinel = if u16_pack {
        [u32::from(u16::MAX); RECORD_FEATURES]
    } else {
        [u32::MAX; RECORD_FEATURES]
    };
    match index.column_bins() {
        Some(Bins::U16(cols)) => {
            let row_words = record * n_records;
            store
                .par_chunks_mut(row_words)
                .enumerate()
                .for_each(|(r, row)| {
                    for (b, out) in row.chunks_exact_mut(record).enumerate() {
                        let mut bins = sentinel;
                        for (f, bin) in bins.iter_mut().enumerate() {
                            let feature = b * RECORD_FEATURES + f;
                            if feature < f_count {
                                *bin = u32::from(cols[feature * n + r]);
                            }
                        }
                        write_record(out, &bins, u16_pack);
                    }
                });
        }
        Some(Bins::U32(cols)) => {
            let row_words = record * n_records;
            store
                .par_chunks_mut(row_words)
                .enumerate()
                .for_each(|(r, row)| {
                    for (b, out) in row.chunks_exact_mut(record).enumerate() {
                        let mut bins = sentinel;
                        for (f, bin) in bins.iter_mut().enumerate() {
                            let feature = b * RECORD_FEATURES + f;
                            if feature < f_count {
                                *bin = cols[feature * n + r];
                            }
                        }
                        write_record(out, &bins, u16_pack);
                    }
                });
        }
        None => {
            // Sparse: walk each row's CSR entries once, mapping bins to
            // their owning feature, and leave the rest at the sentinel.
            let cuts = index.cuts();
            let starts: Vec<u32> = (0..f_count)
                .map(|f| cuts.feature_bins(f).0 as u32)
                .collect();
            let row_ptr = index.row_ptr();
            let bins = index.bins();
            let row_words = record * n_records;
            store.par_chunks_mut(row_words).enumerate().for_each_init(
                || vec![sentinel; n_records],
                |scratch, (r, row)| {
                    scratch.fill(sentinel);
                    let (s, e) = (row_ptr[r], row_ptr[r + 1]);
                    let row_bins: Vec<u32> = match bins {
                        Bins::U16(b) => b[s..e].iter().copied().map(u32::from).collect(),
                        Bins::U32(b) => b[s..e].to_vec(),
                    };
                    for bin in row_bins {
                        let feature = starts.partition_point(|&start| start <= bin) - 1;
                        scratch[feature / RECORD_FEATURES][feature % RECORD_FEATURES] = bin;
                    }
                    for (b, out) in row.chunks_exact_mut(record).enumerate() {
                        write_record(out, &scratch[b], u16_pack);
                    }
                },
            );
        }
    }
    Ok(buffer)
}

/// Pack one interleaved column record from a feature block's eight bins.
fn write_record(out: &mut [u32], bins: &[u32; RECORD_FEATURES], u16_pack: bool) {
    if u16_pack {
        for (f, &bin) in bins.iter().enumerate() {
            if f % 2 == 0 {
                out[f / 2] = bin & 0xFFFF;
            } else {
                out[f / 2] |= (bin & 0xFFFF) << 16;
            }
        }
    } else {
        out[..RECORD_FEATURES].copy_from_slice(bins);
    }
}

// ---------------------------------------------------------------------------
// GPU prediction
// ---------------------------------------------------------------------------

/// One tree's dispatch record, mirroring the kernel's `PTree`.
#[repr(C)]
struct PTree {
    root: u32,
    weight: f32,
    output: u32,
    vector: u32,
}

/// Kernel argument block of `forest_predict`.
#[repr(C)]
struct PredictArgs {
    n_rows: u32,
    n_cols: u32,
    k: u32,
    tree_begin: u32,
    tree_end: u32,
}

/// Per-call prediction buffers, pooled across concurrent calls.
struct PredictBuffers {
    rows: GpuBuffer,
    out: GpuBuffer,
}

/// A [`BoostedModel`] laid out for GPU batch prediction on Metal.
///
/// Built with [`BoostedModel::to_gpu`] (requires the `metal` feature and a
/// Metal device). The compact forest, category pools, and per-tree weights
/// are uploaded once; each prediction call uploads its rows, runs one thread
/// per row over every tree, and applies the objective's transform on the
/// CPU. Predictions are bit-identical to the CPU's: the walk and the
/// per-tree accumulation order match the CPU kernels exactly.
///
/// Small batches (a few rows) are faster on the CPU; the GPU wins from
/// roughly a few thousand row-trees upward.
pub struct GpuModel {
    model: Arc<BoostedModel>,
    ctx: &'static MetalContext,
    nodes: GpuBuffer,
    categories: GpuBuffer,
    leaf_vectors: GpuBuffer,
    trees_bytes: GpuBuffer,
    pool: Mutex<Vec<PredictBuffers>>,
}

impl std::fmt::Debug for GpuModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The Metal handles are opaque; the source model says everything.
        f.debug_struct("GpuModel")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl GpuModel {
    /// The model this GPU predictor was built from.
    #[must_use]
    pub fn model(&self) -> &BoostedModel {
        &self.model
    }

    /// Raw margin predictions of `data` from the boosting `iterations`
    /// (the convention of
    /// [`BoostedModel::predict_margin_range`](crate::model::BoostedModel::predict_margin_range)),
    /// computed on the GPU. Bit-identical to the CPU margins.
    pub fn predict_margin_range(
        &self,
        data: &crate::data::DMatrix,
        iterations: impl RangeBounds<usize>,
    ) -> Result<Vec<f32>> {
        let model = &self.model;
        model.validate_prediction_data(data)?;
        let trees = model.iteration_trees(model.resolve_iterations(iterations, "iterations")?);
        let k = model.n_outputs();
        let n = data.n_rows();
        let mut margins = initial_margins(model.base_scores(), data);
        if trees.is_empty() || n == 0 {
            return Ok(margins);
        }
        if n > MAX_BUFFER_ENTRIES || n * data.n_cols() > MAX_BUFFER_ENTRIES {
            return Err(HessboostError::invalid_param(
                "data",
                format!(
                    "GPU prediction needs a dense row copy ({} rows x {} features) \
                     that exceeds 4 GiB",
                    n,
                    data.n_cols()
                ),
            ));
        }
        let call = self.checkout(n * data.n_cols() * 4, margins.len() * 4)?;
        let result: Result<()> = (|| {
            let cb = self.ctx.command_buffer()?;
            // SAFETY: this call owns `call`; nothing is dispatched on it yet.
            unsafe {
                let rows_slice = call.rows.as_slice_mut::<f32>(n * data.n_cols());
                materialize_rows(data, rows_slice);
                call.out.write(0, as_bytes(&margins));
            }
            // SAFETY: the encoder runs after the writes above (encoders of
            // one command buffer are ordered), the argument block is a live
            // plain-data local, and this call owns `call`.
            unsafe {
                let enc = cb
                    .computeCommandEncoder()
                    .ok_or_else(|| HessboostError::gpu("encoder creation failed"))?;
                enc.setComputePipelineState(&self.ctx.forest_predict.0);
                enc.setBuffer_offset_atIndex(Some(&self.nodes.0), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&self.categories.0), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&self.leaf_vectors.0), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&self.trees_bytes.0), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&call.rows.0), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&call.out.0), 0, 5);
                let args = PredictArgs {
                    n_rows: n as u32,
                    n_cols: data.n_cols() as u32,
                    k: k as u32,
                    tree_begin: trees.start as u32,
                    tree_end: trees.end as u32,
                };
                enc.setBytes_length_atIndex(
                    NonNull::from(&args).cast(),
                    std::mem::size_of::<PredictArgs>(),
                    6,
                );
                let grid = MTLSize {
                    width: n,
                    height: 1,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: PREDICT_THREADS,
                    height: 1,
                    depth: 1,
                };
                enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
                enc.endEncoding();
            }
            cb.commit();
            cb.waitUntilCompleted();
            // SAFETY: the command buffer completed; this call owns the buffer.
            let out = unsafe { call.out.as_slice_mut::<f32>(margins.len()) };
            margins.copy_from_slice(out);
            Ok(())
        })();
        self.checkin(call);
        result?;
        Ok(margins)
    }

    /// Raw margin predictions of `data` (the model's effective iterations),
    /// computed on the GPU. Bit-identical to
    /// [`BoostedModel::predict_margin`](crate::prelude::BoostedModel::predict_margin).
    pub fn predict_margin(&self, data: &crate::data::DMatrix) -> Result<Vec<f32>> {
        self.predict_margin_range(data, self.model.default_iteration_range())
    }

    /// Predictions in the objective's reported space (the model's effective
    /// iterations), computed on the GPU. Bit-identical to
    /// [`BoostedModel::predict`](crate::prelude::BoostedModel::predict).
    pub fn predict(&self, data: &crate::data::DMatrix) -> Result<Vec<f32>> {
        let margin = self.predict_margin(data)?;
        Ok(transform_model_margins(
            self.model.objective(),
            self.model.objective_params(),
            self.model.num_class(),
            self.model.n_targets(),
            self.model.n_outputs(),
            margin,
        ))
    }

    /// The predicted class per row, matching
    /// [`BoostedModel::predict_class`](crate::prelude::BoostedModel::predict_class)
    /// on top of the GPU probabilities.
    pub fn predict_class(&self, data: &crate::data::DMatrix) -> Result<Vec<u32>> {
        let probs = self.predict(data)?;
        let k = self.model.n_outputs();
        if k == 1 || self.model.n_targets() > 1 {
            return Ok(probs.iter().map(|&p| u32::from(p > 0.5)).collect());
        }
        if self.model.objective() == "multi:softmax" {
            return Ok(probs.iter().map(|&class| class as u32).collect());
        }
        Ok(probs
            .chunks_exact(k)
            .map(|row| crate::simd::argmax_scalar(row) as u32)
            .collect())
    }

    /// Check out buffers that fit the call, allocating when the pool has
    /// none large enough.
    fn checkout(&self, rows_bytes: usize, out_bytes: usize) -> Result<PredictBuffers> {
        let mut pool = self.pool.lock().expect("predict pool lock poisoned");
        if let Some(idx) = pool
            .iter()
            .position(|b| b.rows.len() >= rows_bytes && b.out.len() >= out_bytes)
        {
            return Ok(pool.swap_remove(idx));
        }
        Ok(PredictBuffers {
            rows: GpuBuffer::new(&self.ctx.device, rows_bytes)?,
            out: GpuBuffer::new(&self.ctx.device, out_bytes)?,
        })
    }

    fn checkin(&self, call: PredictBuffers) {
        let mut pool = self.pool.lock().expect("predict pool lock poisoned");
        if pool.len() < 8 {
            pool.push(call);
        }
    }
}

impl BoostedModel {
    /// Lay this model out for GPU batch prediction on Metal (the `metal`
    /// feature and a Metal device are required; `gblinear` and `linear_tree`
    /// models, which do not predict through the compact forest, are
    /// refused).
    ///
    /// The returned [`GpuModel`] shares this model's objective, transforms,
    /// and layout; its predictions are bit-identical to the CPU's.
    pub fn to_gpu(&self) -> Result<GpuModel> {
        let ctx = MetalContext::shared().ok_or_else(|| {
            HessboostError::gpu(
                unavailable_reason()
                    .unwrap_or_else(|| "no Metal device is available on this machine".into()),
            )
        })?;
        if self.is_gblinear() {
            return Err(HessboostError::invalid_param(
                "model",
                "gblinear models predict from their linear weights, not the tree forest",
            ));
        }
        if self.has_linear_leaves() {
            return Err(HessboostError::invalid_param(
                "model",
                "`linear_tree` models predict through per-leaf linear models, \
                 which the GPU forest does not hold",
            ));
        }
        let forest = self.compact_forest();
        let parts = forest.gpu_parts();
        let nodes = GpuBuffer::new(&ctx.device, parts.nodes.len())?;
        // SAFETY: fresh buffers, written once before any dispatch.
        unsafe { nodes.write(0, parts.nodes) };
        let categories = GpuBuffer::new(&ctx.device, parts.categories.len() * 4)?;
        // SAFETY: fresh buffer, written once before any dispatch.
        unsafe { categories.write(0, as_bytes(parts.categories)) };
        let leaf_vectors = GpuBuffer::new(&ctx.device, parts.leaf_vectors.len() * 4)?;
        // SAFETY: fresh buffer, written once before any dispatch (a
        // scalar-leaf model's is never read).
        unsafe { leaf_vectors.write(0, as_bytes(parts.leaf_vectors)) };
        let trees: Vec<PTree> = parts
            .roots
            .iter()
            .enumerate()
            .map(|(t, &root)| PTree {
                root,
                weight: self.tree_weight(t),
                output: self.tree_output(t) as u32,
                vector: u32::from(self.tree_is_vector_leaf(t)),
            })
            .collect();
        let trees_bytes = GpuBuffer::new(&ctx.device, trees.len() * 16)?;
        // SAFETY: see above; `PTree` is `repr(C)` of `u32, f32, u32, u32`.
        unsafe { trees_bytes.write(0, as_bytes(&trees)) };
        Ok(GpuModel {
            model: Arc::new(self.clone()),
            ctx,
            nodes,
            categories,
            leaf_vectors,
            trees_bytes,
            pool: Mutex::new(Vec::new()),
        })
    }
}

/// Write `data`'s rows into `rows` as a dense `NaN`-for-missing matrix, the
/// same materialization the CPU's row blocks use: dense NaN-sentinel
/// matrices copy in place, a dense matrix with another sentinel maps
/// sentinel values to `NaN`, and CSR rows materialize per entry.
fn materialize_rows(data: &crate::data::DMatrix, rows: &mut [f32]) {
    let n = data.n_rows();
    let n_cols = data.n_cols();
    if let Some(dense) = data.dense_values()
        && data.missing().is_nan()
        && dense.len() == n * n_cols
    {
        rows.copy_from_slice(dense);
        return;
    }
    let missing = data.missing();
    rows.par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(r, row)| {
            for (f, slot) in row.iter_mut().enumerate() {
                *slot = match data.get(r, f) {
                    Some(v) if v != missing || missing.is_nan() => v,
                    _ => f32::NAN,
                };
            }
        });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::quantile::HistCuts;
    use crate::tree::hist::CpuBackend;

    fn context() -> bool {
        if let Some(reason) = super::unavailable_reason() {
            eprintln!("skipping metal tests: {reason}");
        }
        MetalContext::shared().is_some()
    }

    fn one_thread_pool() -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
    }

    /// The double-float two-sum must be exact: a bin's rows of
    /// near-identical gradients must keep a nonzero low word, which FMA
    /// contraction (safe math mode off) or a plain `f32` sum would erase.
    #[test]
    fn two_sum_is_exact() {
        if !context() {
            return;
        }
        let n = 300;
        let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.25 - 0.75).collect();
        let gpair: Vec<GradPair> = x
            .iter()
            .map(|&v| GradPair {
                grad: v + v.abs() * 2f32.powi(-23),
                hess: 1.0,
            })
            .collect();
        let data = crate::data::DMatrix::from_dense(&x, n, 1).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 512);
        let index = GHistIndex::from_dmatrix(&data, cuts);
        let rows: Vec<u32> = (0..n as u32).collect();
        let backend = MetalHistBackend::new(&index).unwrap();
        let mut gpu = vec![GradStats::default(); index.total_bins()];
        HistogramBackend::build(&backend, &index, &rows, &gpair, &mut gpu);
        let mut cpu = vec![GradStats::default(); index.total_bins()];
        one_thread_pool().install(|| CpuBackend.build(&index, &rows, &gpair, &mut cpu));
        for (g, c) in gpu.iter().zip(&cpu) {
            assert_eq!(g.grad, c.grad, "grad mismatch (GPU {g:?} vs CPU {c:?})");
            assert_eq!(g.hess, c.hess, "hess mismatch");
        }
    }

    /// GPU histograms equal the single-threaded CPU histograms bit for bit
    /// on dense data, across chunk boundaries (`rows > CHUNK_ROWS`) and on
    /// sub-sampled row lists.
    #[test]
    fn hist_matches_cpu_dense() {
        if !context() {
            return;
        }
        let n = CHUNK_ROWS * 2 + 123;
        // 30 columns: more than one interleaved record (8 features each) per
        // row, so the split-record indexing is exercised too.
        let cols = 30;
        let x: Vec<f32> = (0..n * cols)
            .map(|i| ((i.wrapping_mul(2_654_435_761)) % 1000) as f32 * 0.001)
            .collect();
        let data = crate::data::DMatrix::from_dense(&x, n, cols).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 64);
        let index = GHistIndex::from_dmatrix(&data, cuts);
        let gpair: Vec<GradPair> = (0..n)
            .map(|i| GradPair {
                grad: ((i as i32 % 11) as f32 - 5.0).powi(3) * 0.01,
                hess: ((i % 3) as f32 + 1.0).powi(2),
            })
            .collect();
        let all: Vec<u32> = (0..n as u32).collect();
        let sampled: Vec<u32> = all.iter().copied().step_by(3).collect();
        let pool = one_thread_pool();
        let backend = MetalHistBackend::new(&index).unwrap();
        for rows in [&all, &sampled] {
            let mut gpu = vec![GradStats::default(); index.total_bins()];
            HistogramBackend::build(&backend, &index, rows, &gpair, &mut gpu);
            let mut cpu = vec![GradStats::default(); index.total_bins()];
            pool.install(|| CpuBackend.build(&index, rows, &gpair, &mut cpu));
            for (g, c) in gpu.iter().zip(&cpu) {
                assert_eq!(g.grad, c.grad);
                assert_eq!(g.hess, c.hess);
            }
        }
    }

    /// Sparse (missing-value) data goes through the padded column copy and
    /// still matches the CPU exactly.
    #[test]
    fn hist_matches_cpu_sparse() {
        if !context() {
            return;
        }
        let n = 20_000;
        let cols = 5;
        let x: Vec<f32> = (0..n * cols)
            .map(|i: usize| {
                if (i * 7).is_multiple_of(5) {
                    f32::NAN
                } else {
                    ((i.wrapping_mul(40_503)) % 97) as f32
                }
            })
            .collect();
        let data = crate::data::DMatrix::from_dense_with_missing(&x, n, cols, f32::NAN).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 32);
        let index = GHistIndex::from_dmatrix(&data, cuts);
        let gpair: Vec<GradPair> = (0..n)
            .map(|i| GradPair {
                grad: ((i as i32 % 13) as f32 - 6.0) * 0.05,
                hess: 1.0 + (i % 2) as f32,
            })
            .collect();
        let rows: Vec<u32> = (0..n as u32).collect();
        let backend = MetalHistBackend::new(&index).unwrap();
        let mut gpu = vec![GradStats::default(); index.total_bins()];
        HistogramBackend::build(&backend, &index, &rows, &gpair, &mut gpu);
        let mut cpu = vec![GradStats::default(); index.total_bins()];
        one_thread_pool().install(|| CpuBackend.build(&index, &rows, &gpair, &mut cpu));
        for (g, c) in gpu.iter().zip(&cpu) {
            assert_eq!(g.grad, c.grad);
            assert_eq!(g.hess, c.hess);
        }
    }

    /// Tail chunks — a node of `CHUNK_ROWS * k + tiny` rows leaves the last
    /// chunk mostly empty, and the slices the grid still launches must
    /// contribute nothing (their partials write as zeros, not as reads of
    /// stale buffer contents). Deep children of large datasets hit this all
    /// the time.
    #[test]
    fn hist_matches_cpu_tail_chunks() {
        if !context() {
            return;
        }
        let pool = one_thread_pool();
        for &n in &[CHUNK_ROWS + 4, 2 * CHUNK_ROWS + 123] {
            let cols = 5;
            let x: Vec<f32> = (0..n * cols)
                .map(|i| ((i.wrapping_mul(2_654_435_761)) % 997) as f32 * 0.001)
                .collect();
            let data = crate::data::DMatrix::from_dense(&x, n, cols).unwrap();
            let cuts = HistCuts::from_dmatrix(&data, 64);
            let index = GHistIndex::from_dmatrix(&data, cuts);
            let gpair: Vec<GradPair> = (0..n)
                .map(|i| GradPair::new((i % 5) as f32 - 2.0, 1.0 + (i % 3) as f32 * 0.5))
                .collect();
            let rows: Vec<u32> = (0..n as u32).collect();
            let backend = MetalHistBackend::new(&index).unwrap();
            let mut gpu = vec![GradStats::default(); index.total_bins()];
            HistogramBackend::build(&backend, &index, &rows, &gpair, &mut gpu);
            let mut cpu = vec![GradStats::default(); index.total_bins()];
            pool.install(|| CpuBackend.build(&index, &rows, &gpair, &mut cpu));
            for (g, c) in gpu.iter().zip(&cpu) {
                assert_eq!(g.grad, c.grad, "grad mismatch at {n} rows");
                assert_eq!(g.hess, c.hess, "hess mismatch at {n} rows");
            }
        }
    }

    /// A model trained with `device = metal` (DART weights, categorical
    /// splits, missing values) predicts identically through `to_gpu`.
    #[test]
    fn gpu_predicts_like_cpu() {
        use crate::config::{BoosterKind, Device, TreeMethod};
        use crate::data::FeatureType;
        use crate::prelude::*;
        if !context() {
            return;
        }
        let n = 3_000;
        let cols = 6;
        let mut x = vec![0.0f32; n * cols];
        for r in 0..n {
            for f in 0..cols {
                x[r * cols + f] = if f == 0 {
                    ((r * 31) % 5) as f32 // categorical codes
                } else if (r + f) % 11 == 0 {
                    f32::NAN
                } else {
                    (((r * 97 + f * 13) % 100) as f32) * 0.01
                };
            }
        }
        let y: Vec<f32> = (0..n)
            .map(|r| (((r * 31) % 5) as f32 * 0.2 + ((r * 17) % 100) as f32 * 0.001) % 2.0)
            .collect();
        let types = [
            FeatureType::Categorical,
            FeatureType::Numerical,
            FeatureType::Numerical,
            FeatureType::Numerical,
            FeatureType::Numerical,
            FeatureType::Numerical,
        ];
        let data = crate::data::DMatrix::from_dense_with_missing(&x, n, cols, f32::NAN)
            .unwrap()
            .with_feature_types(&types)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .tree_method(TreeMethod::Hist)
            .max_depth(5)
            .eta(0.3)
            .booster(BoosterKind::Dart)
            .device(Device::Metal)
            .build()
            .unwrap();
        let model = train(&params, &data, 12).unwrap();
        let gpu = model.to_gpu().unwrap();
        assert_eq!(model.predict(&data).unwrap(), gpu.predict(&data).unwrap());
        assert_eq!(
            model.predict_margin(&data).unwrap(),
            gpu.predict_margin(&data).unwrap()
        );
        assert_eq!(
            model.predict_class(&data).unwrap(),
            gpu.predict_class(&data).unwrap()
        );
    }
}
