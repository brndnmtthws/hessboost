//! Native Metal acceleration for macOS (opt-in `metal` feature).
//!
//! Two paths have GPU implementations; everything else stays on the CPU:
//!
//! - **Prediction** ([`GpuModel`](crate::backend::metal::GpuModel), from [`BoostedModel::to_gpu`](crate::model::BoostedModel::to_gpu)): every row
//!   walks the branch-free compact forest arena on the GPU, one thread per
//!   row, adding each tree's leaf value in tree order. This is the speed
//!   win: on an M4 Max, 500k rows through 200 depth-8 trees predict about
//!   3.2× faster than the CPU walk, and the gap widens with more trees and
//!   rows (each call's fixed row-upload cost amortizes). A call is pipelined
//!   in row blocks, so one block's rows upload while the GPU walks another. A
//!   model that fits it is uploaded in an 8-byte-per-node encoding (a
//!   threshold value plus a packed feature/child word) instead of the arena's
//!   16 bytes: prediction is bound by the cache lines a warp's scattered node
//!   loads touch, and twice as many nodes per line is what that buys. That
//!   arena also stores each threshold as a float, so the walk compares values
//!   instead of rebuilding the CPU's monotone key at every node. Categorical
//!   splits, vector leaves, and multi-output models keep the 16-byte arena.
//! - **Training** (`device = metal`, see
//!   [`TrainingParams::device`](crate::config::TrainingParams::device)):
//!   the histogram construction of `tree_method = hist` moves to the GPU
//!   for every node it can sum exactly (below), bit-identical to the
//!   CPU's. It is correct and deterministic everywhere, but a GPU build pays
//!   a fixed gather-and-merge cost per node, so on a multicore Apple Silicon
//!   CPU it wins only for large nodes: on an M4 Max, a 4,000,000-row,
//!   30-feature histogram takes 5.8 ms against 6.6 ms for the CPU, 1,000,000
//!   rows 2.1 ms against 1.8 ms, and 100,000 rows 0.54 ms against 0.30 ms;
//!   200k × 30 depth-8 training takes 285 ms against 177 ms, its nodes
//!   sitting below that crossover. A node of at least `CPU_ROWS` (8,192)
//!   rows is offered to the GPU; where that threshold belongs is machine- and
//!   workload-dependent (a CPU thread has slack while the GPU waits), so it
//!   is a conservative default rather than a portable optimum.
//!
//!   The scan is a scatter (the `hist_scatter_u16` kernel): one thread per row,
//!   summing into a threadgroup-shared histogram through 32-bit atomics
//!   (Metal has no 64-bit ones), each 64-bit grain count split into high and
//!   low pieces that the piece merge rejoins exactly. Integer addition
//!   is order-free, so inside the exactness domain this reproduces the CPU's
//!   `f64` sums bit for bit; a node whose grain counts would overflow the
//!   pieces (or an index without the feature-major store the scatter reads)
//!   runs the register-bin kernels instead. The determinism contract still
//!   forbids floating-point atomics. Set `device = metal` to exercise the
//!   GPU path; for speed, keep training on the CPU and use
//!   [`BoostedModel::to_gpu`](crate::model::BoostedModel::to_gpu) for prediction.
//!
//! # Determinism
//!
//! The CPU accumulates each histogram bin in `f64`: a chain of additions in
//! row order within fixed blocks of rows, and the block partials added in
//! block order (a node below 8,192 rows is one chain). Metal on Apple GPUs
//! has no `double`, so the GPU sums integers
//! instead. When the backend stages a tree's gradients it finds, per
//! component (gradients, Hessians), the grain `u`: the largest power of
//! two that divides every value. It uploads each value as the integer
//! `x / u`. The kernels add those integers in 64-bit arithmetic, per row
//! slice and then across slices, and the host scales each bin total back
//! by `u`. A node of `n` rows goes to the GPU only when `n * max <= 2^53 u`
//! for both components, checked exactly. Inside that bound every integer
//! partial is exact. So is every partial sum the CPU forms, a multiple of
//! `u` below `2^53 u`, which makes the two histograms equal bit for bit
//! (proof in the private `backend::exact_sum` module). No wider bound on
//! these statistics works: past it the CPU's sums themselves can round,
//! and no other grouping reproduces that rounding. Every other build runs
//! the CPU backend's build inside the backend, so a `device = metal`
//! training run reproduces the CPU model (the same at every thread count)
//! exactly. The result is also identical across runs, thread counts, and
//! machines: there are no atomics, and integer sums do not depend on the
//! order.
//!
//! The bound depends on the data: `u` is set by the finest-grained value,
//! so the GPU limit per node is `2^53 u / max` rows. On synthetic data
//! (100k rows, depth-6 trees), squared error with real-valued labels near
//! zero allowed 1.3M to 6M rows per node over 100 rounds, and its constant
//! Hessians allowed 9e15. `binary:logistic` allowed 1.8e7 to 2e9 rows in
//! the first 20 rounds, falling to about 2.6e5 by round 100 as confident
//! rows push `p (1 - p)` toward 0. Larger nodes run on the CPU.
//!
//! Guard rails keep the two paths identical in the remaining edge cases:
//! nodes below 8,192 rows, non-finite gradients, inputs that do not match
//! the index the backend was built from, and GPU command failures run on
//! the CPU backend. The kernels are compiled with safe math
//! (`mathMode = safe` from macOS 15 on, `fastMathEnabled = false` before)
//! and FP contraction off, so the compiler cannot change the prediction's
//! rounding order.
//!
//! # Limitations
//!
//! - macOS 10.15 or later (Metal 2.2, for 64-bit integers in kernels) with
//!   a Metal device (Apple Silicon or an Intel Mac with a supported GPU).
//!   Training on a machine without one fails with an error, as do the
//!   combinations listed under
//!   [`TrainingParams::validate`](crate::config::TrainingParams::validate).
//! - Sparse (CSR, missing-value) training data is supported through an
//!   interleaved column copy (up to 8 bytes per row per feature block);
//!   datasets whose copy would exceed 4 GiB are refused.
//! - [`GpuModel`](crate::backend::metal::GpuModel) refuses `gblinear` and `linear_tree` models (they do not
//!   predict through the compact forest), and prediction needs a dense row
//!   copy, so `rows × features × 4` bytes must stay under 4 GiB.
//! - The gradient slice is converted and re-uploaded once per tree (16
//!   bytes per row) and each prediction call uploads its rows; on unified
//!   memory (Apple Silicon) these are plain memory copies.

// `MTLCreateSystemDefaultDevice` links against CoreGraphics; the binding
// crate documents this as the required linkage.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

use crate::backend::exact_sum::SumDomain;
use crate::data::ghist::{Bins, GHistIndex};
use crate::error::{HessboostError, Result};
use crate::model::{BoostedModel, Iterations, initial_margins, transform_model_margins};
use crate::objective::GradPair;
use crate::tree::gain::GradStats;
use crate::tree::hist::{CpuBackend, HistogramBackend};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};
use rayon::prelude::*;
use std::ptr::NonNull;
use std::ptr::copy_nonoverlapping;
use std::slice;
use std::sync::{Arc, LazyLock, Mutex, RwLock, RwLockReadGuard};

/// Rows per GPU histogram chunk. A chunk's shared data (row ids plus
/// gradient pairs in grains, 20 bytes per row) is meant to stay resident in
/// the GPU's level-2 cache while every (feature, window) threadgroup sweeps
/// it, so the re-reads by the ~hundreds of threadgroups do not reach DRAM.
/// The sums are exact integers, so the chunking does not affect results.
const CHUNK_ROWS: usize = 65_536;

/// Row slices per chunk: each chunk dispatch runs one threadgroup per
/// (feature window, slice), the `.y` grid dimension, so the scan has enough
/// threadgroups in flight to hide memory latency. Like the chunking, it
/// does not affect results.
const ROW_SLICES: usize = 64;
/// Nodes below this many rows run on the CPU backend: the kernel dispatch
/// and readback cost more than the scan.
const CPU_ROWS: usize = 8_192;
/// Threads of one histogram threadgroup per feature: 32 threads times 8
/// register bins cover a feature's whole 256-bin window, so a group covers
/// every bin of each of its features. The lanes that walk every row of a
/// slice limit the scan, so fewer, wider per-thread register bins win
/// (measured on M4 Max: a 200k-row, 30-feature build takes 2.0 ms at 32
/// threads x 8 bins, 2.4 ms at 64 x 4, and more at 16 bins per thread).
const THREADS_PER_FEATURE: usize = 32;
/// Features packed into one interleaved column record.
const RECORD_FEATURES: usize = 8;
/// Largest feature count one threadgroup covers (1024 threads, the Metal
/// per-threadgroup ceiling).
const MAX_GROUP_FEATURES: usize = 16;
/// Bins owned by one histogram thread, held in registers.
const BINS_PER_THREAD: usize = 8;
/// Bins of one feature covered by a threadgroup's window.
const WINDOW_BINS: usize = THREADS_PER_FEATURE * BINS_PER_THREAD;
/// Threads per merge threadgroup (one bin per thread).
const MERGE_THREADS: usize = 64;
/// Threads per scatter threadgroup: one thread per row of the slice, summing
/// into the threadgroup's shared histogram (one 256-bin feature window).
const SCATTER_THREADS: usize = 256;
/// Threads per prediction threadgroup (rows are independent).
const PREDICT_THREADS: usize = 256;
/// Rows per prediction block: a call is split into blocks this size (at most
/// [`PREDICT_BLOCKS_MAX`] of them) so the row upload of one block overlaps
/// the GPU walking another. Blocks are large because a dispatch costs
/// ~90 us before its threads run (measured), so fewer of them keeps the GPU
/// busier: on an M4 Max, 500k rows measure 7.99 ms unsplit and 7.2 ms in 2
/// blocks.
const PREDICT_BLOCK_ROWS: usize = 262_144;
/// Largest number of prediction blocks. The pipeline needs two row buffers
/// (one per block in flight), not one per block.
const PREDICT_BLOCKS_MAX: usize = 4;
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
/// The histogram kernels only add 64-bit integers (MSL `long`, Metal 2.2
/// and later): the host stages each gradient pair as integer multiples of
/// its slice's grain (`backend/exact_sum.rs`). Prediction is `float`
/// arithmetic that must round like the CPU, so the source is compiled with
/// safe math mode (see [`MetalContext::new`]).
const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
// The prediction kernel's `out += weight * leaf` must round twice (multiply,
// then add) like the CPU: fused multiply-add would change results by an ulp.
#pragma clang fp contract(off)

// Accumulate one row's gradient pair (in grains) into this thread's bin
// registers: `wl` is the row's bin offset within this thread's window,
// `gh` its gradient pair. The owner check and the eight-way register select
// stay branch-predicated; `j`, `n_win`, and `win_base` are thread-uniform.
#define ACC_BIN(wl_, gh) { \
    if ((wl_) >= 0 && (uint)(wl_) < n_win && (uint)((wl_) >> 3) == j) { \
        if (((wl_) & 7) == 0) { \
            g0 += (gh).x; \
            h0 += (gh).y; \
        } else if (((wl_) & 7) == 1) { \
            g1 += (gh).x; \
            h1 += (gh).y; \
        } else if (((wl_) & 7) == 2) { \
            g2 += (gh).x; \
            h2 += (gh).y; \
        } else if (((wl_) & 7) == 3) { \
            g3 += (gh).x; \
            h3 += (gh).y; \
        } else if (((wl_) & 7) == 4) { \
            g4 += (gh).x; \
            h4 += (gh).y; \
        } else if (((wl_) & 7) == 5) { \
            g5 += (gh).x; \
            h5 += (gh).y; \
        } else if (((wl_) & 7) == 6) { \
            g6 += (gh).x; \
            h6 += (gh).y; \
        } else { \
            g7 += (gh).x; \
            h7 += (gh).y; \
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

// One threadgroup scans one (feature block, 256-bin window) of one row
// slice. Every bin has a single writer per slice and integer adds are
// exact, so the result is exact and deterministic (the host keeps every
// partial below 2^53 grains). The
// interleaved column store packs 8 features' bins into
// one 16-byte word per (row, block), so a row step costs one uniform load
// for the row id, one for the word, and one for the gradient pair —
// amortized across the block's 8 features.
// One thread per entry of the node's row listing: `out[i] = gpair[rows[i]]`.
// The scatter kernel then reads the pairs sequentially instead of by row
// id, which turns one scattered 16-byte read per (row, feature) into one
// scattered read per row.
struct GatherArgs { uint n; };

kernel void hist_gather(
    const device uint* rows [[buffer(0)]],
    const device long2* gpair [[buffer(1)]],
    device long2* out [[buffer(2)]],
    constant GatherArgs& a [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= a.n) { return; }
    out[i] = gpair[rows[i]];
}

// The scatter scan: one threadgroup per (feature, 256-bin window) like
// `hist_scan_u16`, but each thread takes whole rows instead of owning bins,
// and sums them into a threadgroup-shared histogram. A row's bin is read
// from the index's feature-major `u16` store, so a warp's reads are
// coalesced whenever the node's row listing is (it is a contiguous range at
// the top of a tree, and never worse than the register kernel's per-window
// scan otherwise); the gradient pairs come from `hist_gather`'s compacted
// array, read in listing order.
//
// Shared atomics are 32-bit, so each 64-bit grain count is split as
// `k = hi * 2^16 + lo` with `lo` in `[0, 2^16)` and the pieces accumulated
// in separate counters. Integer addition is order-free, so the histogram is
// the same one the CPU's `f64` chain and the register kernels produce, as
// long as each threadgroup's piece sums stay inside their counters — the
// host gates a node on that (`scatter_row_bound`) and runs the CPU backend
// otherwise.
struct ScatterArgs {
    uint chunk_rows;   // rows in this chunk
    uint chunk_index;
    uint slices;
    uint total_bins;
    uint n_rows;       // dataset rows: the feature-major store's stride
};

kernel void hist_scatter_u16(
    const device ushort* bins [[buffer(0)]],
    const device uint* rows [[buffer(1)]],
    const device long2* grad [[buffer(2)]],
    const device BlockInfo* blocks [[buffer(3)]],
    const device ScanGroup* groups [[buffer(4)]],
    device int4* partials [[buffer(5)]],
    constant ScatterArgs& chunk [[buffer(6)]],
    uint2 gpos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    // One feature per threadgroup (see `features_per_group`).
    ScanGroup sg = groups[gpos.x];
    uint feature = sg.block;
    uint fs = blocks[feature].fs[0];
    uint nbins = blocks[feature].nbins[0];
    uint win_base = fs + sg.window_base;
    uint n_win = (nbins > sg.window_base)
        ? min(256u, nbins - sg.window_base)
        : 0u;
    threadgroup atomic_uint hist[256u * 4u];
    for (uint i = tid; i < 256u * 4u; i += 256u) {
        atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device ushort* column = bins + (size_t)feature * chunk.n_rows;
    uint slice_len = (chunk.chunk_rows + chunk.slices - 1u) / chunk.slices;
    uint slice_begin = gpos.y * slice_len;
    uint slice_rows = (slice_begin < chunk.chunk_rows)
        ? min(slice_len, chunk.chunk_rows - slice_begin)
        : 0u;
    for (uint i = tid; i < slice_rows; i += 256u) {
        // A missing entry's sentinel and a bin outside this window both leave
        // the range below (the sentinel is `u16::MAX`, far past `win_base`).
        uint w = (uint)column[rows[slice_begin + i]] - win_base;
        if (w >= n_win) { continue; }
        long2 q = grad[slice_begin + i];
        atomic_fetch_add_explicit(&hist[w * 4u + 0u], (uint)((long)q.x >> 16), memory_order_relaxed);
        atomic_fetch_add_explicit(&hist[w * 4u + 1u], (uint)((ulong)q.x & 0xFFFFul), memory_order_relaxed);
        atomic_fetch_add_explicit(&hist[w * 4u + 2u], (uint)((long)q.y >> 16), memory_order_relaxed);
        atomic_fetch_add_explicit(&hist[w * 4u + 3u], (uint)((ulong)q.y & 0xFFFFul), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device int4* out = partials + (size_t)(chunk.chunk_index * chunk.slices + gpos.y) * chunk.total_bins;
    for (uint b = tid; b < n_win; b += 256u) {
        out[win_base + b] = int4(
            (int)atomic_load_explicit(&hist[b * 4u + 0u], memory_order_relaxed),
            (int)atomic_load_explicit(&hist[b * 4u + 1u], memory_order_relaxed),
            (int)atomic_load_explicit(&hist[b * 4u + 2u], memory_order_relaxed),
            (int)atomic_load_explicit(&hist[b * 4u + 3u], memory_order_relaxed));
    }
}

// Merge the scatter kernel's (high, low) piece partials of every bin: the
// pieces are exact integers, so any order gives the bin's total.
struct PiecesArgs { uint n_partials; uint total_bins; };

kernel void hist_merge_pieces(
    const device int4* partials [[buffer(0)]],
    device long2* hist [[buffer(1)]],
    constant PiecesArgs& merge [[buffer(2)]],
    uint b [[thread_position_in_grid]])
{
    if (b >= merge.total_bins) { return; }
    long g_hi = 0, g_lo = 0, h_hi = 0, h_lo = 0;
    for (uint c = 0; c < merge.n_partials; c++) {
        int4 p = partials[(size_t)c * merge.total_bins + b];
        // The high pieces are signed, the low ones unsigned (they add up
        // below `2^32` by the host's row bound).
        g_hi += (long)p.x;
        g_lo += (long)(uint)p.y;
        h_hi += (long)p.z;
        h_lo += (long)(uint)p.w;
    }
    hist[b] = long2(g_hi * 65536L + g_lo, h_hi * 65536L + h_lo);
}

kernel void hist_scan_u16(
    const device uint* rows [[buffer(0)]],
    const device uint4* columns [[buffer(1)]],
    const device long2* gpair [[buffer(2)]],
    const device BlockInfo* blocks [[buffer(3)]],
    const device ScanGroup* groups [[buffer(4)]],
    device long2* partials [[buffer(5)]],
    constant ChunkArgs& chunk [[buffer(6)]],
    constant HistRun& run [[buffer(7)]],
    uint2 gpos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    ScanGroup sg = groups[gpos.x];
    // Thread `tid` covers feature `tid / 32` of the block and bin eighth
    // `(tid % 32) * 8`: 32 threads times 8 register bins cover a whole
    // 256-bin window of one feature, so one group covers every bin of its
    // `features_per_group` features and the row ids and gradient pairs load
    // once per group instead of once per feature.
    uint f = tid / 32u;
    uint j = tid % 32u;
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
    long g0 = 0, g1 = 0, g2 = 0, g3 = 0, g4 = 0, g5 = 0, g6 = 0, g7 = 0;
    long h0 = 0, h1 = 0, h2 = 0, h3 = 0, h4 = 0, h5 = 0, h6 = 0, h7 = 0;
    // Batches of 4: the row ids load together, then the column words and
    // gradient pairs, so the dependent uniform loads pipeline across the
    // batch. (Four, not eight: with the 8-bin registers below, a batch of
    // eight needs more live values than the register file has room for, and
    // the spills measured slower than the pipelining saved.) The
    // accumulation order per bin is unchanged (ascending rows within the
    // slice).
    uint i = 0;
    for (; i + 4 <= slice_rows; i += 4) {
        uint r0 = rows[slice_begin + i + 0];
        uint r1 = rows[slice_begin + i + 1];
        uint r2 = rows[slice_begin + i + 2];
        uint r3 = rows[slice_begin + i + 3];
        uint4 c0 = columns[(size_t)r0 * run.n_records + record];
        uint4 c1 = columns[(size_t)r1 * run.n_records + record];
        uint4 c2 = columns[(size_t)r2 * run.n_records + record];
        uint4 c3 = columns[(size_t)r3 * run.n_records + record];
        long2 q0 = gpair[r0];
        long2 q1 = gpair[r1];
        long2 q2 = gpair[r2];
        long2 q3 = gpair[r3];
        ACC_BIN((int)((c0[word] >> shift) & 0xFFFFu) - (int)win_base, q0)
        ACC_BIN((int)((c1[word] >> shift) & 0xFFFFu) - (int)win_base, q1)
        ACC_BIN((int)((c2[word] >> shift) & 0xFFFFu) - (int)win_base, q2)
        ACC_BIN((int)((c3[word] >> shift) & 0xFFFFu) - (int)win_base, q3)
    }
    for (; i < slice_rows; i++) {
        uint r = rows[slice_begin + i];
        uint4 c = columns[(size_t)r * run.n_records + record];
        long2 gh = gpair[r];
        ACC_BIN((int)((c[word] >> shift) & 0xFFFFu) - (int)win_base, gh)
    }
    device long2* out = partials + (size_t)(chunk.chunk_index * chunk.slices + gpos.y) * run.total_bins;
    uint base = win_base + j * 8u;
    if (j * 8u + 0u < n_win) { out[base + 0u] = long2(g0, h0); }
    if (j * 8u + 1u < n_win) { out[base + 1u] = long2(g1, h1); }
    if (j * 8u + 2u < n_win) { out[base + 2u] = long2(g2, h2); }
    if (j * 8u + 3u < n_win) { out[base + 3u] = long2(g3, h3); }
    if (j * 8u + 4u < n_win) { out[base + 4u] = long2(g4, h4); }
    if (j * 8u + 5u < n_win) { out[base + 5u] = long2(g5, h5); }
    if (j * 8u + 6u < n_win) { out[base + 6u] = long2(g6, h6); }
    if (j * 8u + 7u < n_win) { out[base + 7u] = long2(g7, h7); }
}

// The wide-bin variant (more than 65,536 total bins, or a sparse dataset
// that cannot spare a `u16` sentinel): the interleaved store holds eight
// `u32` bins per (row, block), read as one coalesced 32-byte run.
kernel void hist_scan_u32(
    const device uint* rows [[buffer(0)]],
    const device uint* columns [[buffer(1)]],
    const device long2* gpair [[buffer(2)]],
    const device BlockInfo* blocks [[buffer(3)]],
    const device ScanGroup* groups [[buffer(4)]],
    device long2* partials [[buffer(5)]],
    constant ChunkArgs& chunk [[buffer(6)]],
    constant HistRun& run [[buffer(7)]],
    uint2 gpos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    ScanGroup sg = groups[gpos.x];
    // See hist_scan_u16 for the thread mapping.
    uint f = tid / 32u;
    uint j = tid % 32u;
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
    long g0 = 0, g1 = 0, g2 = 0, g3 = 0, g4 = 0, g5 = 0, g6 = 0, g7 = 0;
    long h0 = 0, h1 = 0, h2 = 0, h3 = 0, h4 = 0, h5 = 0, h6 = 0, h7 = 0;
    uint stride = run.n_records * 8u;
    // See hist_scan_u16: batches of four keep the register pressure within
    // the register file.
    uint i = 0;
    for (; i + 4 <= slice_rows; i += 4) {
        uint r0 = rows[slice_begin + i + 0];
        uint r1 = rows[slice_begin + i + 1];
        uint r2 = rows[slice_begin + i + 2];
        uint r3 = rows[slice_begin + i + 3];
        uint b0 = columns[(size_t)r0 * stride + record * 8u + slot];
        uint b1 = columns[(size_t)r1 * stride + record * 8u + slot];
        uint b2 = columns[(size_t)r2 * stride + record * 8u + slot];
        uint b3 = columns[(size_t)r3 * stride + record * 8u + slot];
        long2 q0 = gpair[r0];
        long2 q1 = gpair[r1];
        long2 q2 = gpair[r2];
        long2 q3 = gpair[r3];
        ACC_BIN((int)b0 - (int)win_base, q0)
        ACC_BIN((int)b1 - (int)win_base, q1)
        ACC_BIN((int)b2 - (int)win_base, q2)
        ACC_BIN((int)b3 - (int)win_base, q3)
    }
    for (; i < slice_rows; i++) {
        uint r = rows[slice_begin + i];
        uint b = columns[(size_t)r * stride + record * 8u + slot];
        long2 gh = gpair[r];
        ACC_BIN((int)b - (int)win_base, gh)
    }
    device long2* out = partials + (size_t)(chunk.chunk_index * chunk.slices + gpos.y) * run.total_bins;
    uint base = win_base + j * 8u;
    if (j * 8u + 0u < n_win) { out[base + 0u] = long2(g0, h0); }
    if (j * 8u + 1u < n_win) { out[base + 1u] = long2(g1, h1); }
    if (j * 8u + 2u < n_win) { out[base + 2u] = long2(g2, h2); }
    if (j * 8u + 3u < n_win) { out[base + 3u] = long2(g3, h3); }
    if (j * 8u + 4u < n_win) { out[base + 4u] = long2(g4, h4); }
    if (j * 8u + 5u < n_win) { out[base + 5u] = long2(g5, h5); }
    if (j * 8u + 6u < n_win) { out[base + 6u] = long2(g6, h6); }
    if (j * 8u + 7u < n_win) { out[base + 7u] = long2(g7, h7); }
}

// Merge the slice partials of every bin. Integer adds: exact, so the
// order does not matter (it is still fixed).
kernel void hist_merge(
    const device long2* partials [[buffer(0)]],
    device long2* hist [[buffer(1)]],
    constant MergeArgs& merge [[buffer(2)]],
    constant HistRun& run [[buffer(3)]],
    uint b [[thread_position_in_grid]])
{
    if (b >= run.total_bins) { return; }
    long2 sum = long2(0L, 0L);
    for (uint c = 0; c < merge.n_partials; c++) {
        sum += partials[(size_t)c * run.total_bins + b];
    }
    hist[b] = sum;
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
// One tree's dispatch record in the 8-byte arena (`tree::compact::GpuArena8`).
struct PTree8 { uint root; float weight; };
struct PredictArgs8 { uint n_rows; uint n_cols; uint tree_begin; uint tree_end; };

// The compact-forest walk over the 8-byte node arena: two `u32`s per node,
// `key` (the threshold key, or the leaf value's bits) and
// `packed = child | feature << 15 | MIRRORED | LEAF`. Half the bytes per
// node means 16 nodes share a cache line instead of 8, which is what the
// scattered node loads of a warp are bound by. The arithmetic per node is
// the same as `forest_predict`'s, in the same order, so the margins are the
// same bit for bit; the single-output, scalar-leaf models it covers are the
// ones whose walk needs nothing else.
kernel void forest_predict8(
    const device uint2* nodes [[buffer(0)]],
    const device PTree8* trees [[buffer(1)]],
    const device float* rows [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant PredictArgs8& a [[buffer(4)]],
    uint r [[thread_position_in_grid]])
{
    if (r >= a.n_rows) { return; }
    const device float* row = rows + (size_t)r * a.n_cols;
    float acc = out[r];
    for (uint t = a.tree_begin; t < a.tree_end; t++) {
        uint base = trees[t].root;
        uint nid = 0u;
        uint2 n = nodes[base];
        while ((n.y & 0x80000000u) == 0u) {
            float v = row[(n.y >> 15) & 0x7FFFu];
            // A mirrored node compares the negated value, and `n.x` holds the
            // threshold itself, so the compare is a float compare: the same
            // order the CPU's monotone keys give, including for `±0.0` (equal
            // either way) and a missing `NaN` (false, so the left child).
            if (n.y & 0x40000000u) { v = -v; }
            nid = (n.y & 0x7FFFu) + (v > as_type<float>(n.x) ? 1u : 0u);
            n = nodes[(size_t)base + nid];
        }
        acc += trees[t].weight * as_type<float>(n.x);
    }
    out[r] = acc;
}

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
// methods whose contracts require exclusive access (the gradient buffer's
// write lock, a checked-out per-call buffer set, or a buffer not yet shared),
// and the safe methods do not touch the contents.
unsafe impl Send for GpuBuffer {}
// SAFETY: see the `Send` impl: shared references only reach the contents
// through the `unsafe` methods and their exclusivity contracts.
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

    /// Check that `len` bytes at `offset` lie within the buffer.
    fn check_range(&self, offset: usize, len: usize) -> Result<()> {
        match offset.checked_add(len) {
            Some(end) if end <= self.len() => Ok(()),
            _ => Err(HessboostError::gpu(format!(
                "a {len}-byte access at offset {offset} overruns a {}-byte Metal buffer",
                self.len()
            ))),
        }
    }

    /// The buffer's contents as a mutable slice of `n` elements of `T`, or
    /// an error when `n` of them do not fit (or the contents are misaligned
    /// for `T`).
    ///
    /// # Safety
    ///
    /// The caller must exclusively own the buffer: no in-flight GPU work may
    /// read it, and no other CPU access may alias the returned slice. `T`
    /// must be a plain data type (every bit pattern valid) matching the
    /// kernel's layout.
    #[allow(
        clippy::mut_from_ref,
        reason = "the caller proves exclusive access; the buffer is plain shared memory"
    )]
    unsafe fn as_slice_mut<T>(&self, n: usize) -> Result<&mut [T]> {
        let bytes = n
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| HessboostError::gpu("a Metal buffer view overflows `usize`"))?;
        self.check_range(0, bytes)?;
        let data = self.0.contents().as_ptr().cast::<T>();
        if !data.is_aligned() {
            return Err(HessboostError::gpu("a Metal buffer is misaligned"));
        }
        // SAFETY: the caller guarantees exclusive access and a plain-data
        // `T` (see above); Metal shared buffers are plain CPU-accessible
        // memory of `self.len()` bytes, `n * size_of::<T>()` of which were
        // just checked to lie within it, at an address aligned for `T`.
        Ok(unsafe { slice::from_raw_parts_mut(data, n) })
    }

    /// Copy `bytes` into the buffer at `offset`, or return an error (and
    /// write nothing) when they do not fit. No-op for empty input.
    ///
    /// # Safety
    ///
    /// The caller must exclusively own the buffer (see [`Self::as_slice_mut`]).
    unsafe fn write(&self, offset: usize, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        self.check_range(offset, bytes.len())?;
        // SAFETY: exclusive access per the caller; the destination range was
        // just checked to lie within the buffer, and a caller's slice cannot
        // overlap memory this backend owns exclusively.
        unsafe {
            copy_nonoverlapping(
                bytes.as_ptr(),
                self.0.contents().as_ptr().cast::<u8>().add(offset),
                bytes.len(),
            );
        }
        Ok(())
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
    hist_gather: Pipeline,
    hist_scatter_u16: Pipeline,
    hist_merge_pieces: Pipeline,
    hist_u16: Pipeline,
    hist_u32: Pipeline,
    hist_merge: Pipeline,
    forest_predict: Pipeline,
    forest_predict8: Pipeline,
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
        // The histogram kernels use 64-bit integers (MSL `long`), which
        // need Metal 2.2, macOS 10.15. Older systems would fail the kernel
        // compile; `dispatchThreads:threadsPerThreadgroup:` (macOS 10.13)
        // would otherwise raise an unknown-selector exception.
        if !objc2::available!(macos = 10.15) {
            return Err("the Metal backend needs macOS 10.15 or later".to_string());
        }
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| "no system default Metal device".to_string())?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| "creating the Metal command queue failed".to_string())?;
        // Safe math mode: the prediction kernel must round like the CPU,
        // with no FMA contraction or reassociation, and the compile-time default
        // (fast math) permits both. `mathMode` exists from macOS 15 on (an
        // older system has no such selector, so sending it would raise an
        // exception); before that, `fastMathEnabled = false` is the same
        // setting.
        let options = MTLCompileOptions::new();
        if objc2::available!(macos = 15.0) {
            options.setMathMode(MTLMathMode::Safe);
        } else {
            #[allow(
                deprecated,
                reason = "the only safe-math setting before macOS 15, where `mathMode` replaced it"
            )]
            options.setFastMathEnabled(false);
        }
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
            hist_gather: pipeline("hist_gather")?,
            hist_scatter_u16: pipeline("hist_scatter_u16")?,
            hist_merge_pieces: pipeline("hist_merge_pieces")?,
            hist_u16: pipeline("hist_scan_u16")?,
            hist_u32: pipeline("hist_scan_u32")?,
            hist_merge: pipeline("hist_merge")?,
            forest_predict: pipeline("forest_predict")?,
            forest_predict8: pipeline("forest_predict8")?,
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

/// Commit `cb`, wait until the GPU is done with it, and check that it
/// completed: a command buffer that ends in the error state (a GPU fault, a
/// timeout, a lost device) has not produced its outputs. Either way the GPU
/// no longer touches the buffers it referenced when this returns.
fn submit(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<()> {
    cb.commit();
    completed(cb)
}

/// Wait for an already committed `cb` to finish and check that it did.
fn completed(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<()> {
    cb.waitUntilCompleted();
    let status = cb.status();
    if status == MTLCommandBufferStatus::Completed {
        return Ok(());
    }
    let detail = cb
        .error()
        .map_or_else(|| format!("status {}", status.0), |e| e.to_string());
    Err(HessboostError::gpu(format!(
        "a Metal command buffer failed: {detail}"
    )))
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

/// The gradient slice staged by [`HistogramBackend::prepare`]: uploaded to
/// the GPU, with the statistics that decide whether a node's sums are
/// exact on both paths.
///
/// Lives behind the backend's `RwLock`: staging (the only write to
/// `buffer`) takes the write lock, and every GPU build holds a read guard
/// from checking that the staged slice is its own until its command buffer
/// has completed, so the buffer never changes while the GPU reads it.
struct StagedGradients {
    buffer: GpuBuffer,
    /// (address, length) identity of the staged slice; length 0 when
    /// nothing is staged (a staged slice always has `n_rows > 0` entries).
    addr: usize,
    len: usize,
    grad: SumDomain,
    hess: SumDomain,
}

impl StagedGradients {
    /// Whether `gpair` is the staged slice.
    fn holds(&self, gpair: &[GradPair]) -> bool {
        self.len != 0 && self.len == gpair.len() && self.addr == gpair.as_ptr().addr()
    }

    /// Whether every sum of at most `n` staged gradient pairs is exact on
    /// both paths (see `backend::exact_sum`).
    fn sums_exact(&self, n: usize) -> bool {
        self.grad.sums_exact(n) && self.hess.sums_exact(n)
    }
}

/// Which scan kernels a node's histogram is built with.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scan {
    /// [`hist_scatter_u16`]: the index's feature-major `u16` store, one row
    /// per thread, summed into shared 32-bit accumulators.
    Scatter,
    /// [`hist_scan_u16`]: the interleaved `u16` record, one bin per register.
    Packed,
    /// [`hist_scan_u32`]: the interleaved `u32` record, one bin per register.
    Wide,
}

/// Rows one threadgroup scans: a node's chunk is split into this many slices,
/// and the bound below only has to admit that many.
const ROWS_PER_SLICE: usize = CHUNK_ROWS / ROW_SLICES;

/// Rows one threadgroup may scan with the scatter kernel's shared 32-bit
/// accumulators, given the staged slices' magnitude statistics: a grain count
/// `k` is split as `k = hi * 2^16 + lo`, so one threadgroup's `hi` sum must
/// stay inside an `i32` (`lo` is 16-bit each and sums inside a `u32`). The
/// bound is at least [`SCATTER_MIN_ROWS`] for any slice whose values allow
/// the GPU at all; past it the node runs on the CPU backend, whose sums are
/// exact by the same argument (see `backend::exact_sum`).
fn scatter_row_bound(grad: &SumDomain, hess: &SumDomain) -> usize {
    let bound = |domain: &SumDomain| -> u64 {
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

/// Per-call GPU buffers of the histogram backend, pooled across the parallel
/// node builds of a training run. Each concurrent `build` owns one set.
struct CallBuffers {
    rows: GpuBuffer,
    partials: GpuBuffer,
    hist: GpuBuffer,
}

impl CallBuffers {
    /// One set of buffers for an index of `n_rows` rows and `total_bins`
    /// bins: the node's row list, the per-chunk partial histograms, and the
    /// merged histogram.
    fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        n_rows: usize,
        total_bins: usize,
    ) -> Result<Self> {
        let chunks = n_rows.div_ceil(CHUNK_ROWS).max(1);
        Ok(CallBuffers {
            rows: GpuBuffer::new(device, n_rows * 4)?,
            partials: GpuBuffer::new(device, chunks * ROW_SLICES * total_bins * 16)?,
            hist: GpuBuffer::new(device, total_bins * 16)?,
        })
    }
}

/// The Metal histogram backend: implements [`HistogramBackend`] by scanning
/// the binned column store on the GPU. Constructed once per training run (the
/// column upload and group descriptors are per-dataset); the gradient slice
/// is re-uploaded by [`HistogramBackend::prepare`] once per tree.
///
/// Training selects it automatically through
/// [`device = metal`](crate::config::TrainingParams::device); constructing it
/// directly serves custom training loops against a [`GHistIndex`]. Its
/// histograms equal the CPU backend's bit for bit: nodes the GPU cannot sum
/// exactly (see the [module docs](crate::backend::metal)) run the CPU
/// backend's build instead. `build` must receive the index
/// the backend was built from, and the gradient slice must not change
/// between `prepare` and the tree's last `build`; inputs that do not fit
/// the backend's buffers (a different index shape, a gradient slice of
/// another length, row indices past the index) never reach the GPU and
/// take the CPU path, which checks them.
pub struct MetalHistBackend {
    ctx: &'static MetalContext,
    /// `true` when `columns` packs each record as eight `u16` bins (dense
    /// data, or sparse data with a spare `u16` sentinel); `false` for eight
    /// `u32` bins (wider bin counts).
    columns_u16: bool,
    /// The index's feature-major `u16` bin store (feature `f` of row `r` at
    /// `f * n_rows + r`), which the scatter kernel reads directly. `None`
    /// when the index has no such store (or its bins need `u32`), and the
    /// register kernels serve the node instead.
    feature_bins: Option<GpuBuffer>,
    /// The node's gradient pairs compacted into listing order by
    /// [`hist_gather`], sized for the largest node.
    gathered: GpuBuffer,
    /// The interleaved column store: one record per (row, feature block).
    columns: GpuBuffer,
    blocks_bytes: GpuBuffer,
    groups: Vec<ScanGroup>,
    groups_bytes: GpuBuffer,
    /// Threads per scan threadgroup (`features_per_group` × 64).
    threads_per_group: usize,
    total_bins: usize,
    n_rows: usize,
    n_cols: usize,
    run: HistRun,
    gradients: RwLock<StagedGradients>,
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
            return Err(HessboostError::invalid_data(
                "data",
                "the Metal backend needs a non-empty binned dataset",
            ));
        }
        if total_bins > MAX_BUFFER_ENTRIES || n_rows > MAX_BUFFER_ENTRIES {
            return Err(HessboostError::invalid_data(
                "data",
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
        // The scatter kernel needs no interleaved copy at all: it reads the
        // feature-major store the index already keeps, when that store is
        // `u16` (the dense one, or the one with the missing-value sentinel).
        let feature_bins = match index.column_bins().or_else(|| index.missing_columns()) {
            Some(Bins::U16(bins)) => {
                let buffer = GpuBuffer::new(&ctx.device, bins.len() * 2)?;
                // SAFETY: the buffer was just allocated and is not yet
                // shared; its length covers `bins`.
                unsafe { buffer.write(0, as_bytes(bins))? };
                Some(buffer)
            }
            _ => None,
        };
        let gathered = GpuBuffer::new(&ctx.device, n_rows * 16)?;
        // Feature blocks per threadgroup. Measured on Apple Silicon: one
        // feature per threadgroup (a whole 256-bin window) beats every wider
        // block — the interleaved record load costs more than the
        // row/gradient amortization saves, and 512+-thread groups lose
        // occupancy to register pressure. 2 and 4 features per group are
        // within noise of 1 (a 200k-row, 30-feature build: 1.98 ms, 1.98 ms,
        // 2.05 ms).
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
        unsafe { blocks_bytes.write(0, as_bytes(&blocks))? };
        let groups_bytes = GpuBuffer::new(&ctx.device, groups.len() * 8)?;
        // SAFETY: see above; `ScanGroup` is `repr(C)` of two `u32`s.
        unsafe { groups_bytes.write(0, as_bytes(&groups))? };
        let run = HistRun {
            total_bins: total_bins as u32,
            n_records: n_records as u32,
            features_per_group: features_per_group as u32,
        };
        let call = CallBuffers::new(&ctx.device, n_rows, total_bins)?;
        // One `[i64; 2]` gradient pair in grains per row.
        let gpair = GpuBuffer::new(&ctx.device, n_rows * 16)?;
        Ok(MetalHistBackend {
            ctx,
            feature_bins,
            gathered,
            columns_u16,
            columns,
            blocks_bytes,
            groups,
            groups_bytes,
            threads_per_group,
            total_bins,
            n_rows,
            n_cols,
            run,
            gradients: RwLock::new(StagedGradients {
                buffer: gpair,
                addr: 0,
                len: 0,
                grad: SumDomain::EMPTY,
                hess: SumDomain::EMPTY,
            }),
            pool: Mutex::new(vec![call]),
        })
    }

    /// Stage `gpair` on the GPU with its exactness statistics, unless it is
    /// the staged slice already and `force` is off. A slice of any length
    /// other than `n_rows` is not staged: it does not fit the buffer, and
    /// the builds it serves run on the CPU.
    ///
    /// `force` stages unconditionally: the trainer refills its gradient
    /// buffer in place every round, so a new tree's `prepare` cannot rely on
    /// the slice's identity. Within one tree (a `build` whose `prepare` just
    /// ran) the identity check skips the re-upload: the slice is constant
    /// while a tree grows.
    fn stage(&self, gpair: &[GradPair], force: bool) {
        let mut staged = self.gradients.write().expect("gradient lock poisoned");
        if !force && staged.holds(gpair) {
            return;
        }
        // Unstage first, so a slice that is not uploaded below is never
        // mistaken for the previous one.
        staged.len = 0;
        if gpair.len() != self.n_rows {
            return;
        }
        let grad = SumDomain::of_slice(gpair, |p| p.grad);
        let hess = SumDomain::of_slice(gpair, |p| p.hess);
        // SAFETY: the write lock excludes every GPU build (each holds a
        // read guard until its command buffer has completed), so no GPU work
        // reads the buffer and nothing else aliases it; `[i64; 2]` is plain
        // data laid out as the kernels' `long2`.
        let Ok(units) = (unsafe { staged.buffer.as_slice_mut::<[i64; 2]>(gpair.len()) }) else {
            return;
        };
        // Integer multiples of each component's grain, the values the
        // kernels sum: exact whenever a node's sums can be (`exact_sum`).
        units
            .par_iter_mut()
            .zip(gpair)
            .for_each(|(u, p)| *u = [grad.units(p.grad), hess.units(p.hess)]);
        staged.grad = grad;
        staged.hess = hess;
        staged.addr = gpair.as_ptr().addr();
        staged.len = gpair.len();
    }

    /// A read guard on the staged gradients when they hold `gpair`, staging
    /// it first if needed; `None` when `gpair` cannot be staged or another
    /// thread staged a different slice in between. While the guard lives,
    /// the gradient buffer does not change.
    fn staged_for(&self, gpair: &[GradPair]) -> Option<RwLockReadGuard<'_, StagedGradients>> {
        if gpair.len() != self.n_rows {
            return None;
        }
        {
            let staged = self.gradients.read().expect("gradient lock poisoned");
            if staged.holds(gpair) {
                return Some(staged);
            }
        }
        self.stage(gpair, false);
        let staged = self.gradients.read().expect("gradient lock poisoned");
        staged.holds(gpair).then_some(staged)
    }

    /// Check out a per-call buffer set from the pool.
    fn checkout(&self) -> Result<CallBuffers> {
        if let Some(call) = self.pool.lock().expect("buffer pool lock poisoned").pop() {
            return Ok(call);
        }
        CallBuffers::new(&self.ctx.device, self.n_rows, self.total_bins)
    }

    fn checkin(&self, call: CallBuffers) {
        self.pool
            .lock()
            .expect("buffer pool lock poisoned")
            .push(call);
    }

    /// The CPU backend's build, used below the row threshold and whenever
    /// the GPU cannot take a node. Inside the exactness domain the GPU
    /// reproduces it bit for bit; its own bounds checks reject inputs that
    /// do not match `ghist`, as the CPU backend's do.
    fn cpu(ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]) {
        CpuBackend.build(ghist, rows, gpair, out);
    }

    /// Build the histogram of `rows` into `out` on the GPU when the inputs
    /// allow it, returning whether it did (otherwise `out` is unspecified).
    fn try_gpu(
        &self,
        ghist: &GHistIndex,
        rows: &[u32],
        gpair: &[GradPair],
        out: &mut [GradStats],
    ) -> bool {
        // The kernels do not bounds-check: inputs the GPU buffers were not
        // sized for (another index shape, more rows than the index holds, a
        // row past its end) must never reach a dispatch.
        let fits = out.len() == self.total_bins
            && ghist.n_rows() == self.n_rows
            && ghist.n_cols() == self.n_cols
            && ghist.total_bins() == self.total_bins
            && rows.len() <= self.n_rows
            && rows.iter().all(|&r| (r as usize) < self.n_rows);
        if !fits {
            return false;
        }
        // The guard lives until this function returns, after the dispatch's
        // command buffer has completed: no `stage` can rewrite the gradient
        // buffer while the GPU reads it.
        let Some(staged) = self.staged_for(gpair) else {
            return false;
        };
        if !staged.sums_exact(rows.len()) {
            return false;
        }
        // The scatter kernel needs the index's feature-major `u16` store, and
        // its 32-bit shared accumulators only stay exact while a
        // threadgroup's rows fit `scatter_row_bound`; anything else takes the
        // register kernels.
        let scan = match &self.feature_bins {
            Some(_) if scatter_row_bound(&staged.grad, &staged.hess) >= ROWS_PER_SLICE => {
                Scan::Scatter
            }
            _ if self.columns_u16 => Scan::Packed,
            _ => Scan::Wide,
        };
        let Ok(call) = self.checkout() else {
            return false;
        };
        let result = self.dispatch(&call, &staged, rows, scan, out);
        self.checkin(call);
        result.is_ok()
    }

    /// Encode, run, and read back the scatter path of `rows` over the staged
    /// `gradients`: one gather pass into listing order, one scatter per chunk,
    /// and the piece merge. `try_gpu` has checked the inputs, the feature-major
    /// store's presence, and the accumulator bound.
    fn dispatch_scatter(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        call: &CallBuffers,
        gradients: &StagedGradients,
        rows: &[u32],
        out: &mut [GradStats],
    ) -> Result<()> {
        let bins = self.feature_bins.as_ref().ok_or_else(|| {
            HessboostError::gpu("the scatter kernel needs the feature-major store")
        })?;
        // SAFETY: this call exclusively owns `call` (checked out of the pool),
        // and nothing has been dispatched on it yet.
        unsafe { call.rows.write(0, as_bytes(rows))? };
        let chunks = rows.len().div_ceil(CHUNK_ROWS);
        // SAFETY: the encoders below run after the writes above (encoders of
        // one command buffer are ordered), the argument blocks are live
        // plain-data locals outliving them, and every access is in bounds:
        // `rows` holds `rows.len() <= n_rows` ids below `n_rows` (the
        // feature-major store's rows and the compacted pair array's length),
        // and the partials of `chunks <= ceil(n_rows / CHUNK_ROWS)` chunks
        // fit the pool's buffer.
        unsafe {
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| HessboostError::gpu("encoder creation failed"))?;
            enc.setComputePipelineState(&self.ctx.hist_gather.0);
            enc.setBuffer_offset_atIndex(Some(&call.rows.0), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&gradients.buffer.0), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.gathered.0), 0, 2);
            let gather = GatherArgs {
                n: rows.len() as u32,
            };
            enc.setBytes_length_atIndex(
                NonNull::from(&gather).cast(),
                std::mem::size_of::<GatherArgs>(),
                3,
            );
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: rows.len(),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: MERGE_THREADS * 4,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
        }
        for c in 0..chunks {
            let begin = c * CHUNK_ROWS;
            let chunk_rows = (rows.len() - begin).min(CHUNK_ROWS);
            let args = ScatterArgs {
                chunk_rows: chunk_rows as u32,
                chunk_index: c as u32,
                slices: ROW_SLICES as u32,
                total_bins: self.total_bins as u32,
                n_rows: self.n_rows as u32,
            };
            // SAFETY: see above; the row offset lies within the rows buffer,
            // and the gather wrote one pair per listing position.
            unsafe {
                let enc = cb
                    .computeCommandEncoder()
                    .ok_or_else(|| HessboostError::gpu("encoder creation failed"))?;
                enc.setComputePipelineState(&self.ctx.hist_scatter_u16.0);
                enc.setBuffer_offset_atIndex(Some(&bins.0), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&call.rows.0), begin * 4, 1);
                enc.setBuffer_offset_atIndex(Some(&self.gathered.0), begin * 16, 2);
                enc.setBuffer_offset_atIndex(Some(&self.blocks_bytes.0), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&self.groups_bytes.0), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&call.partials.0), 0, 5);
                enc.setBytes_length_atIndex(
                    NonNull::from(&args).cast(),
                    std::mem::size_of::<ScatterArgs>(),
                    6,
                );
                let grid = MTLSize {
                    width: self.groups.len(),
                    height: ROW_SLICES,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: SCATTER_THREADS,
                    height: 1,
                    depth: 1,
                };
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                enc.endEncoding();
            }
        }
        // SAFETY: encoders of one command buffer run in order, so the merge
        // reads complete partials; its `total_bins` threads index both
        // buffers within their sizes.
        unsafe {
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| HessboostError::gpu("encoder creation failed"))?;
            enc.setComputePipelineState(&self.ctx.hist_merge_pieces.0);
            enc.setBuffer_offset_atIndex(Some(&call.partials.0), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&call.hist.0), 0, 1);
            let merge = PiecesArgs {
                n_partials: (chunks * ROW_SLICES) as u32,
                total_bins: self.total_bins as u32,
            };
            enc.setBytes_length_atIndex(
                NonNull::from(&merge).cast(),
                std::mem::size_of::<PiecesArgs>(),
                2,
            );
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: self.total_bins,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: MERGE_THREADS,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
        }
        submit(cb)?;
        // SAFETY: the command buffer has completed, so the GPU is done with
        // the buffer; this call owns it.
        let hist = unsafe { call.hist.as_slice_mut::<[i64; 2]>(self.total_bins)? };
        for (o, &[g, h]) in out.iter_mut().zip(hist.iter()) {
            // Exact piece sums in grains (below 2^53), scaled back exactly.
            o.grad = gradients.grad.value(g);
            o.hess = gradients.hess.value(h);
        }
        Ok(())
    }

    /// Encode, run, and read back the scan and merge of `rows` over the
    /// staged `gradients`. The caller has checked that `rows` fit the
    /// backend (at most `n_rows` entries, each below `n_rows`) and that
    /// their sums are exact, and holds the gradients' read guard
    /// throughout.
    fn dispatch(
        &self,
        call: &CallBuffers,
        gradients: &StagedGradients,
        rows: &[u32],
        scan: Scan,
        out: &mut [GradStats],
    ) -> Result<()> {
        let cb = self.ctx.command_buffer()?;
        if scan == Scan::Scatter {
            return self.dispatch_scatter(&cb, call, gradients, rows, out);
        }
        // SAFETY: this call exclusively owns `call` (checked out of the
        // pool), and nothing has been dispatched on it yet.
        unsafe { call.rows.write(0, as_bytes(rows))? };
        let chunks = rows.len().div_ceil(CHUNK_ROWS);
        for c in 0..chunks {
            let begin = c * CHUNK_ROWS;
            let chunk_rows = (rows.len() - begin).min(CHUNK_ROWS);
            let args = ChunkArgs {
                chunk_rows: chunk_rows as u32,
                chunk_index: c as u32,
                slices: ROW_SLICES as u32,
            };
            // SAFETY: the argument blocks are live `repr(C)` plain-data
            // locals outliving the encoder. Every access the kernel makes is
            // in bounds: the row offset `begin * 4` lies within the rows
            // buffer (`begin < rows.len() <= n_rows`), each row id is below
            // `n_rows`, the size the column store and gradient buffer were
            // allocated for, and the partials of `chunks <= ceil(n_rows /
            // CHUNK_ROWS)` chunks fit the pool's partials buffer.
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
                enc.setBuffer_offset_atIndex(Some(&gradients.buffer.0), 0, 2);
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
        // created, so the merge reads complete chunk partials (the same
        // `chunks * ROW_SLICES` the scans wrote); the argument blocks are
        // live plain-data locals, and the merge's `total_bins` threads
        // index the partials and histogram buffers within their sizes.
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
        submit(&cb)?;
        // SAFETY: the command buffer has completed, so the GPU is done with
        // the buffer; this call owns it.
        let hist = unsafe { call.hist.as_slice_mut::<[i64; 2]>(self.total_bins)? };
        for (o, &[g, h]) in out.iter_mut().zip(hist.iter()) {
            // Exact bin sums in grains (below 2^53), scaled back exactly.
            o.grad = gradients.grad.value(g);
            o.hess = gradients.hess.value(h);
        }
        Ok(())
    }
}

impl HistogramBackend for MetalHistBackend {
    fn build(&self, ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]) {
        // Small nodes, and every node the GPU cannot take (inputs that do
        // not fit, sums outside the exactness domain, a failed dispatch),
        // run on the CPU backend. The histogram is a pure function of the
        // inputs, so a GPU failure costs time, never the training run.
        if rows.len() < CPU_ROWS || !self.try_gpu(ghist, rows, gpair, out) {
            Self::cpu(ghist, rows, gpair, out);
        }
    }

    fn prepare(&self, _ghist: &GHistIndex, gpair: &[GradPair]) {
        self.stage(gpair, true);
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
            HessboostError::invalid_data(
                "data",
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
    let store = unsafe { buffer.as_slice_mut::<u32>(words)? };
    let sentinel = if u16_pack {
        [u32::from(u16::MAX); RECORD_FEATURES]
    } else {
        [u32::MAX; RECORD_FEATURES]
    };
    if let Some(cols) = index.column_bins() {
        // The bin at feature-major position `i` (`feature * n + row`).
        let bin_at = |i: usize| match &cols {
            Bins::U16(c) => u32::from(c[i]),
            Bins::U32(c) => c[i],
        };
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
                            *bin = bin_at(feature * n + r);
                        }
                    }
                    write_record(out, &bins, u16_pack);
                }
            });
    } else {
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
                let mut place = |bin: u32| {
                    let feature = starts.partition_point(|&start| start <= bin) - 1;
                    scratch[feature / RECORD_FEATURES][feature % RECORD_FEATURES] = bin;
                };
                match &bins {
                    Bins::U16(b) => b[s..e].iter().for_each(|&bin| place(u32::from(bin))),
                    Bins::U32(b) => b[s..e].iter().for_each(|&bin| place(bin)),
                }
                for (b, out) in row.chunks_exact_mut(record).enumerate() {
                    write_record(out, &scratch[b], u16_pack);
                }
            },
        );
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

/// Kernel argument block of `forest_predict8`.
#[repr(C)]
struct PredictArgs8 {
    n_rows: u32,
    n_cols: u32,
    tree_begin: u32,
    tree_end: u32,
}

/// One tree's dispatch record of `forest_predict8`.
#[repr(C)]
struct PTree8 {
    root: u32,
    weight: f32,
}

/// The prediction arena in the 8-byte node encoding, when the model fits it
/// (see [`tree::compact::GpuArena8`](crate::tree::compact)). Prediction is
/// bound by the cache lines a warp's scattered node loads touch, and this
/// arena has twice as many nodes per line as the 16-byte one.
struct NarrowArena {
    nodes: GpuBuffer,
    trees: GpuBuffer,
}

/// Kernel argument block of `hist_gather`.
#[repr(C)]
struct GatherArgs {
    n: u32,
}

/// Kernel argument block of `hist_scatter_u16`.
#[repr(C)]
struct ScatterArgs {
    chunk_rows: u32,
    chunk_index: u32,
    slices: u32,
    total_bins: u32,
    n_rows: u32,
}

/// Kernel argument block of `hist_merge_pieces`.
#[repr(C)]
struct PiecesArgs {
    n_partials: u32,
    total_bins: u32,
}

/// Per-call prediction buffers, pooled across concurrent calls. The two row
/// buffers are the pipeline's stages: block `b` uploads into `rows[b % 2]`
/// after the command buffer that read that slot has completed.
struct PredictBuffers {
    rows: [GpuBuffer; 2],
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
    /// The 8-byte arena, when the model fits it; the 16-byte buffers above
    /// are then never dispatched (their contents stay empty).
    narrow: Option<NarrowArena>,
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
    /// ([`Iterations`], as for
    /// [`BoostedModel::predict_margin`](crate::model::BoostedModel::predict_margin)),
    /// computed on the GPU. Bit-identical to the CPU margins. A model
    /// trained with model shrinkage predicts on the CPU, whose per-iteration
    /// shrink-then-add arithmetic repeats training's.
    pub fn predict_margin(
        &self,
        data: &crate::data::DMatrix,
        iterations: impl Into<Iterations>,
    ) -> Result<crate::model::Predictions> {
        let model = &self.model;
        let iterations = iterations.into();
        if model.shrinkage().is_some() {
            return model.predict_margin(data, iterations);
        }
        model.validate_prediction_data(data)?;
        let trees = model.iteration_trees(model.resolve_iterations(iterations, "iterations")?);
        let k = model.n_outputs();
        let n = data.n_rows();
        let mut margins = initial_margins(model.base_scores(), data);
        if trees.is_empty() || n == 0 {
            return Ok(crate::model::Predictions::new(margins, n, k));
        }
        if n > MAX_BUFFER_ENTRIES || n * data.n_cols() > MAX_BUFFER_ENTRIES {
            return Err(HessboostError::invalid_data(
                "data",
                format!(
                    "GPU prediction needs a dense row copy ({} rows x {} features) \
                     that exceeds 4 GiB",
                    n,
                    data.n_cols()
                ),
            ));
        }
        // Split the batch into row blocks and pipeline them: each block's
        // rows upload into one of two slots while the GPU walks the other
        // block's. The margins are block-local inside the kernel (the row and
        // out buffers are bound at the block's base), so the walk's
        // arithmetic and its order are exactly the single-block call's.
        let blocks = n.div_ceil(PREDICT_BLOCK_ROWS).clamp(1, PREDICT_BLOCKS_MAX);
        let block_rows = n.div_ceil(blocks);
        let call = self.checkout(block_rows * data.n_cols() * 4, margins.len() * 4)?;
        let result: Result<()> = (|| {
            // The command buffers whose blocks the GPU may still be reading.
            let mut pending: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>> =
                Vec::with_capacity(blocks);
            for b in 0..blocks {
                let begin = b * block_rows;
                let rows_here = (n - begin).min(block_rows);
                if rows_here == 0 {
                    break;
                }
                // The slot this block uploads into was read by the block two
                // back, whose command buffer must therefore have completed.
                if b >= 2 {
                    let previous = pending.remove(0);
                    completed(&previous)?;
                }
                let rows_buffer = &call.rows[b % 2];
                // SAFETY: this call owns `call`, and the slot's previous
                // reader (if any) has completed above; nothing else writes
                // the slot, its block, or the margins' block range.
                unsafe {
                    let rows_slice = rows_buffer.as_slice_mut::<f32>(rows_here * data.n_cols())?;
                    materialize_rows(data, begin, rows_slice);
                    call.out.write(
                        begin * k * 4,
                        as_bytes(&margins[begin * k..(begin + rows_here) * k]),
                    )?;
                }
                let cb = self.ctx.command_buffer()?;
                // SAFETY: the writes above precede the encoder (encoders of
                // one command buffer are ordered, and this buffer is committed
                // after them), the argument block is a live plain-data local,
                // and every buffer index is within the ranges this call owns.
                unsafe {
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or_else(|| HessboostError::gpu("encoder creation failed"))?;
                    if let Some(narrow) = &self.narrow {
                        enc.setComputePipelineState(&self.ctx.forest_predict8.0);
                        enc.setBuffer_offset_atIndex(Some(&narrow.nodes.0), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(&narrow.trees.0), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(&rows_buffer.0), 0, 2);
                        enc.setBuffer_offset_atIndex(Some(&call.out.0), begin * k * 4, 3);
                        let args = PredictArgs8 {
                            n_rows: rows_here as u32,
                            n_cols: data.n_cols() as u32,
                            tree_begin: trees.start as u32,
                            tree_end: trees.end as u32,
                        };
                        enc.setBytes_length_atIndex(
                            NonNull::from(&args).cast(),
                            std::mem::size_of::<PredictArgs8>(),
                            4,
                        );
                    } else {
                        enc.setComputePipelineState(&self.ctx.forest_predict.0);
                        enc.setBuffer_offset_atIndex(Some(&self.nodes.0), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(&self.categories.0), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(&self.leaf_vectors.0), 0, 2);
                        enc.setBuffer_offset_atIndex(Some(&self.trees_bytes.0), 0, 3);
                        enc.setBuffer_offset_atIndex(Some(&rows_buffer.0), 0, 4);
                        enc.setBuffer_offset_atIndex(Some(&call.out.0), begin * k * 4, 5);
                        let args = PredictArgs {
                            n_rows: rows_here as u32,
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
                    }
                    let grid = MTLSize {
                        width: rows_here,
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
                pending.push(cb);
            }
            for cb in &pending {
                completed(cb)?;
            }
            // SAFETY: every command buffer completed; this call owns the
            // buffer, and nothing else writes it.
            let out = unsafe { call.out.as_slice_mut::<f32>(margins.len())? };
            margins.copy_from_slice(out);
            Ok(())
        })();
        self.checkin(call);
        result?;
        Ok(crate::model::Predictions::new(margins, n, k))
    }

    /// Predictions in the objective's reported space from the boosting
    /// `iterations`, computed on the GPU. Bit-identical to
    /// [`BoostedModel::predict`](crate::prelude::BoostedModel::predict).
    pub fn predict(
        &self,
        data: &crate::data::DMatrix,
        iterations: impl Into<Iterations>,
    ) -> Result<crate::model::Predictions> {
        let margin = self.predict_margin(data, iterations)?;
        Ok(transform_model_margins(
            self.model.objective(),
            self.model.max_delta_step(),
            self.model.n_targets(),
            margin,
        ))
    }

    /// The predicted class per row, matching
    /// [`BoostedModel::predict_class`](crate::prelude::BoostedModel::predict_class)
    /// on top of the GPU probabilities.
    pub fn predict_class(
        &self,
        data: &crate::data::DMatrix,
        iterations: impl Into<Iterations>,
    ) -> Result<crate::model::Predictions<u32>> {
        Ok(self.model.classes(&self.predict(data, iterations)?))
    }

    /// Check out buffers that fit the call, allocating when the pool has
    /// none large enough.
    fn checkout(&self, rows_bytes: usize, out_bytes: usize) -> Result<PredictBuffers> {
        let mut pool = self.pool.lock().expect("predict pool lock poisoned");
        if let Some(idx) = pool.iter().position(|b| {
            b.rows.iter().all(|row| row.len() >= rows_bytes) && b.out.len() >= out_bytes
        }) {
            return Ok(pool.swap_remove(idx));
        }
        Ok(PredictBuffers {
            rows: [
                GpuBuffer::new(&self.ctx.device, rows_bytes)?,
                GpuBuffer::new(&self.ctx.device, rows_bytes)?,
            ],
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
            return Err(HessboostError::incompatible_model(
                "model",
                "gblinear models predict from their linear weights, not the tree forest",
            ));
        }
        if self.has_linear_leaves() {
            return Err(HessboostError::incompatible_model(
                "model",
                "`linear_tree` models predict through per-leaf linear models, \
                 which the GPU forest does not hold",
            ));
        }
        let forest = self.compact_forest();
        // The 8-byte arena covers single-output, categorical-free,
        // scalar-leaf trees; everything else predicts through the 16-byte
        // one. Both walks are the same arithmetic, so the choice is speed.
        let narrow = if self.n_outputs() == 1 {
            forest
                .gpu_arena8(|t| self.tree_is_vector_leaf(t))
                .map(|arena| -> Result<NarrowArena> {
                    let nodes = GpuBuffer::new(&ctx.device, arena.words.len() * 4)?;
                    // SAFETY: fresh buffer, written once before any dispatch.
                    unsafe { nodes.write(0, as_bytes(&arena.words))? };
                    let trees: Vec<PTree8> = arena
                        .roots
                        .iter()
                        .enumerate()
                        .map(|(t, &root)| PTree8 {
                            root,
                            weight: self.tree_weight(t),
                        })
                        .collect();
                    let trees_bytes = GpuBuffer::new(&ctx.device, trees.len() * 8)?;
                    // SAFETY: see above; `PTree8` is `repr(C)` of `u32, f32`.
                    unsafe { trees_bytes.write(0, as_bytes(&trees))? };
                    Ok(NarrowArena {
                        nodes,
                        trees: trees_bytes,
                    })
                })
                .transpose()?
        } else {
            None
        };
        let parts = forest.gpu_parts();
        let nodes = GpuBuffer::new(&ctx.device, parts.nodes.len())?;
        // SAFETY: fresh buffers, written once before any dispatch.
        unsafe { nodes.write(0, parts.nodes)? };
        let categories = GpuBuffer::new(&ctx.device, parts.categories.len() * 4)?;
        // SAFETY: fresh buffer, written once before any dispatch.
        unsafe { categories.write(0, as_bytes(parts.categories))? };
        let leaf_vectors = GpuBuffer::new(&ctx.device, parts.leaf_vectors.len() * 4)?;
        // SAFETY: fresh buffer, written once before any dispatch (a
        // scalar-leaf model's is never read).
        unsafe { leaf_vectors.write(0, as_bytes(parts.leaf_vectors))? };
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
        unsafe { trees_bytes.write(0, as_bytes(&trees))? };
        Ok(GpuModel {
            model: Arc::new(self.clone()),
            ctx,
            nodes,
            categories,
            leaf_vectors,
            trees_bytes,
            narrow,
            pool: Mutex::new(Vec::new()),
        })
    }
}

/// Write `data`'s rows starting at row `begin` into `rows` as a dense
/// `NaN`-for-missing matrix, the same materialization the CPU's row blocks
/// use: dense NaN-sentinel matrices copy in place, a dense matrix with
/// another sentinel maps sentinel values to `NaN`, and CSR rows materialize
/// per entry. `rows` holds a whole number of rows; it is one prediction
/// block of the batch.
fn materialize_rows(data: &crate::data::DMatrix, begin: usize, rows: &mut [f32]) {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::quantile::HistCuts;

    fn context() -> bool {
        if let Some(reason) = super::unavailable_reason() {
            eprintln!("skipping metal tests: {reason}");
        }
        MetalContext::shared().is_some()
    }

    /// The single-threaded CPU histogram of `rows`.
    fn cpu_hist(index: &GHistIndex, rows: &[u32], gpair: &[GradPair]) -> Vec<GradStats> {
        let mut out = vec![GradStats::default(); index.total_bins()];
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| CpuBackend.build(index, rows, gpair, &mut out));
        out
    }

    /// The histogram of `rows` built on the GPU, asserting that the GPU
    /// path ran rather than the CPU fallback.
    fn gpu_hist(
        backend: &MetalHistBackend,
        index: &GHistIndex,
        rows: &[u32],
        gpair: &[GradPair],
    ) -> Vec<GradStats> {
        backend.prepare(index, gpair);
        let mut out = vec![GradStats::default(); index.total_bins()];
        assert!(
            backend.try_gpu(index, rows, gpair, &mut out),
            "the GPU path must take this node"
        );
        out
    }

    /// A one-feature dataset whose rows cycle through `values` distinct
    /// feature values (one bin each).
    fn one_feature(n: usize, values: usize) -> GHistIndex {
        let x: Vec<f32> = (0..n).map(|i| (i % values) as f32).collect();
        let data = crate::data::DMatrix::from_dense(&x, n, 1).unwrap();
        GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, 256))
    }

    /// Gradients one `f32` ulp away from coarse values keep that last bit:
    /// a bin of such rows needs more than 24 significant bits, which a
    /// `float` accumulator would drop.
    #[test]
    fn near_identical_gradients_keep_their_low_bits() {
        if !context() {
            return;
        }
        let n = CPU_ROWS + 808;
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
        assert_eq!(
            gpu_hist(&backend, &index, &rows, &gpair),
            cpu_hist(&index, &rows, &gpair)
        );
    }

    /// At the exactness bound (`n * max = 2^53` grains) the GPU still takes
    /// the node and keeps every low bit: each bin sums to about `2^52`, with
    /// odd small gradients mixed in, so it needs all 53 bits of an `f64`.
    /// One more row and the node goes to the CPU.
    #[test]
    fn hist_is_exact_at_the_domain_edge() {
        if !context() {
            return;
        }
        let grad = |i: usize| match i {
            1 => 2f32.powi(37),
            i if i % 11 == 0 => 1975.0,
            _ => 2f32.powi(37) - 2f32.powi(13),
        };
        let n = CHUNK_ROWS; // 2^16 rows of at most 2^37
        let gpair: Vec<GradPair> = (0..n).map(|i| GradPair::new(grad(i), 1.0)).collect();
        let index = one_feature(n, 2);
        let backend = MetalHistBackend::new(&index).unwrap();
        let rows: Vec<u32> = (0..n as u32).collect();
        assert_eq!(
            gpu_hist(&backend, &index, &rows, &gpair),
            cpu_hist(&index, &rows, &gpair)
        );
        let gpair: Vec<GradPair> = (0..=n).map(|i| GradPair::new(grad(i), 1.0)).collect();
        let index = one_feature(n + 1, 2);
        let backend = MetalHistBackend::new(&index).unwrap();
        let rows: Vec<u32> = (0..=n as u32).collect();
        backend.prepare(&index, &gpair);
        let mut out = vec![GradStats::default(); index.total_bins()];
        assert!(!backend.try_gpu(&index, &rows, &gpair, &mut out));
        HistogramBackend::build(&backend, &index, &rows, &gpair, &mut out);
        assert_eq!(out, cpu_hist(&index, &rows, &gpair));
    }

    /// Outside the exactness bound the backend runs the CPU path: the
    /// review's six gradients (up to `2^50`, grain 1) allow at most 8 rows
    /// per node. The CPU sums the bins to +64 and -64; the old double-float
    /// kernels returned 0 and 0.
    #[test]
    fn out_of_domain_nodes_run_on_the_cpu() {
        if !context() {
            return;
        }
        let six = [
            2f32.powi(50),
            2f32.powi(26),
            2f32.powi(23) + 1.0,
            -(2f32.powi(50)),
            -(2f32.powi(26)),
            -(2f32.powi(23)),
        ];
        let n = CPU_ROWS;
        let grad = |i: usize| match (i % 128, i % 128 >= 64) {
            (p, false) if p < 6 => six[p],
            (p, true) if p - 64 < 6 => -six[p - 64],
            _ => 0.0,
        };
        let gpair: Vec<GradPair> = (0..n).map(|i| GradPair::new(grad(i), 1.0)).collect();
        let x: Vec<f32> = (0..n).map(|i| f32::from(i % 128 >= 64)).collect();
        let data = crate::data::DMatrix::from_dense(&x, n, 1).unwrap();
        let index = GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, 256));
        let backend = MetalHistBackend::new(&index).unwrap();
        let rows: Vec<u32> = (0..n as u32).collect();
        backend.prepare(&index, &gpair);
        let mut out = vec![GradStats::default(); index.total_bins()];
        assert!(!backend.try_gpu(&index, &rows, &gpair, &mut out));
        HistogramBackend::build(&backend, &index, &rows, &gpair, &mut out);
        let cpu = cpu_hist(&index, &rows, &gpair);
        assert_eq!(out, cpu);
        assert_eq!([cpu[0].grad, cpu[1].grad], [64.0, -64.0]);
    }

    /// Inputs that do not fit the backend's GPU buffers (a longer gradient
    /// slice, more row ids than rows, a row past the index, another
    /// histogram length) never reach a dispatch, and those the CPU path
    /// accepts give the CPU's histogram.
    #[test]
    fn mismatched_inputs_never_reach_the_gpu() {
        if !context() {
            return;
        }
        let n = CPU_ROWS + 100;
        let index = one_feature(n, 5);
        let backend = MetalHistBackend::new(&index).unwrap();
        let bins = index.total_bins();
        let rows: Vec<u32> = (0..n as u32).collect();
        let long: Vec<GradPair> = (0..n + 1000)
            .map(|i| GradPair::new((i % 7) as f32 - 3.0, 1.0))
            .collect();
        let gpair = &long[..n];
        let twice: Vec<u32> = rows.iter().chain(&rows).copied().collect();
        let past_end: Vec<u32> = (1..=n as u32).collect();
        let mut out = vec![GradStats::default(); bins];
        backend.prepare(&index, &long);
        assert!(!backend.try_gpu(&index, &rows, &long, &mut out));
        HistogramBackend::build(&backend, &index, &rows, &long, &mut out);
        assert_eq!(out, cpu_hist(&index, &rows, &long));
        backend.prepare(&index, gpair);
        assert!(!backend.try_gpu(&index, &twice, gpair, &mut out));
        HistogramBackend::build(&backend, &index, &twice, gpair, &mut out);
        assert_eq!(out, cpu_hist(&index, &twice, gpair));
        assert!(!backend.try_gpu(&index, &past_end, gpair, &mut out));
        let mut short = vec![GradStats::default(); bins - 1];
        assert!(!backend.try_gpu(&index, &rows, gpair, &mut short));
        // The matching inputs still run on the GPU.
        assert_eq!(
            gpu_hist(&backend, &index, &rows, gpair),
            cpu_hist(&index, &rows, gpair)
        );
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
        let backend = MetalHistBackend::new(&index).unwrap();
        for rows in [&all, &sampled] {
            assert_eq!(
                gpu_hist(&backend, &index, rows, &gpair),
                cpu_hist(&index, rows, &gpair)
            );
        }
    }

    /// A slice whose grain counts are too coarse for the scatter kernel's
    /// 32-bit shared accumulators (a huge magnitude next to a value with a
    /// fine grain) still builds its histogram on the GPU, through the
    /// register kernel, and still matches the CPU.
    #[test]
    fn coarse_grains_fall_back_to_the_register_kernel() {
        if !context() {
            return;
        }
        // 8,192 rows of at most 2^40 grains: inside the CPU/GPU exactness
        // domain (`n * max <= 2^53`), past the scatter kernel's bound (the
        // high piece alone would need more than an `i32` per threadgroup).
        let n = CPU_ROWS;
        let gpair: Vec<GradPair> = (0..n)
            .map(|i| {
                let grad = if i % 32 == 0 {
                    2f32.powi(40)
                } else {
                    1.0 + (i % 3) as f32
                };
                GradPair::new(grad, 1.0)
            })
            .collect();
        let index = one_feature(n, 8);
        let backend = MetalHistBackend::new(&index).unwrap();
        let rows: Vec<u32> = (0..n as u32).collect();
        backend.prepare(&index, &gpair);
        let staged = backend.gradients.read().unwrap();
        assert!(
            staged.sums_exact(rows.len()),
            "the node must be inside the exactness domain"
        );
        assert!(
            scatter_row_bound(&staged.grad, &staged.hess) < ROWS_PER_SLICE,
            "the scatter kernel must not take this node"
        );
        drop(staged);
        assert_eq!(
            gpu_hist(&backend, &index, &rows, &gpair),
            cpu_hist(&index, &rows, &gpair)
        );
    }

    /// More than 65,536 bins in total: the binned store is 32-bit, so the
    /// scan runs [`hist_scan_u32`](MetalHistBackend)'s eight-`u32`-record
    /// kernel instead of the packed `u16` one, and its histograms must match
    /// the CPU's just the same.
    #[test]
    fn hist_matches_cpu_wide_bins() {
        if !context() {
            return;
        }
        // 300 features at up to 256 bins each is 76,800 bins, past the
        // `u16` sentinel the packed record needs.
        let (n, cols) = (12_000, 300);
        let x: Vec<f32> = (0..n * cols)
            .map(|i: usize| ((i.wrapping_mul(2_654_435_761)) % 100_003) as f32 * 0.001)
            .collect();
        let data = crate::data::DMatrix::from_dense(&x, n, cols).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let index = GHistIndex::from_dmatrix(&data, cuts);
        assert!(
            index.total_bins() > u16::MAX as usize + 1,
            "the test needs more bins than the packed record holds"
        );
        let gpair: Vec<GradPair> = (0..n)
            .map(|i| GradPair {
                grad: ((i as i32 % 11) as f32 - 5.0).powi(3) * 0.01,
                hess: ((i % 3) as f32 + 1.0).powi(2),
            })
            .collect();
        let all: Vec<u32> = (0..n as u32).collect();
        let sampled: Vec<u32> = all.iter().copied().step_by(7).collect();
        let backend = MetalHistBackend::new(&index).unwrap();
        assert!(
            !backend.columns_u16,
            "the wide-bin kernel must be the one run"
        );
        for rows in [&all, &sampled] {
            assert_eq!(
                gpu_hist(&backend, &index, rows, &gpair),
                cpu_hist(&index, rows, &gpair)
            );
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
        assert_eq!(
            gpu_hist(&backend, &index, &rows, &gpair),
            cpu_hist(&index, &rows, &gpair)
        );
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
            assert_eq!(
                gpu_hist(&backend, &index, &rows, &gpair),
                cpu_hist(&index, &rows, &gpair),
                "at {n} rows"
            );
        }
    }

    /// A model trained with `device = metal` (DART weights, categorical
    /// splits, missing values) predicts identically through `to_gpu`.
    #[test]
    fn gpu_predicts_like_cpu() {
        use crate::config::{BoosterKind, Dart, Device, TreeMethod};
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
            .objective(Objective::SquaredError(RegLoss::default()))
            .tree_method(TreeMethod::Hist)
            .max_depth(5)
            .eta(0.3)
            .booster(BoosterKind::Dart(Dart::default()))
            .device(Device::Metal)
            .build()
            .unwrap();
        let model = train(&params, &data, 12).unwrap();
        let gpu = model.to_gpu().unwrap();
        assert_eq!(
            model.predict(&data, Iterations::Best).unwrap(),
            gpu.predict(&data, Iterations::Best).unwrap()
        );
        assert_eq!(
            model.predict_margin(&data, Iterations::Best).unwrap(),
            gpu.predict_margin(&data, Iterations::Best).unwrap()
        );
        assert_eq!(
            model.predict_class(&data, Iterations::Best).unwrap(),
            gpu.predict_class(&data, Iterations::Best).unwrap()
        );
    }
}
