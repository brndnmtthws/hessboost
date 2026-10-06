//! Portable GPU acceleration through [wgpu](https://wgpu.rs) (opt-in `wgpu`
//! feature): Vulkan on Linux and Windows, Metal on macOS, DirectX 12 on
//! Windows.
//!
//! Two paths have GPU implementations; everything else stays on the CPU:
//!
//! - **Prediction** ([`GpuModel`](crate::backend::wgpu::GpuModel), from
//!   [`BoostedModel::to_wgpu`](crate::model::BoostedModel::to_wgpu)): every
//!   row walks the compact forest arena on the GPU, one thread per row,
//!   adding each tree's leaf value in tree order. Each call uploads its rows
//!   in blocks and reads the margins back once.
//! - **Training** (`device = wgpu`, see
//!   [`TrainingParams::device`](crate::config::TrainingParams::device)): the
//!   histogram construction of `tree_method = hist` moves to the GPU for
//!   every node it can sum exactly (below), bit-identical to the CPU's.
//!
//! The backend is the portable sibling of [`backend::metal`](crate::backend::metal)
//! and shares its design: the same exactness domain, the same scatter
//! kernel, the same CPU fallbacks. It is a correctness and portability path
//! so far, not a speedup: nothing has been measured on a real GPU, and on a
//! software adapter it is slower than the CPU. Use it to run and test the
//! GPU paths on machines without one, and measure before relying on it.
//!
//! # Adapter
//!
//! The process picks one adapter the first time the backend is used: the
//! one named by `WGPU_ADAPTER_NAME` (a case-insensitive substring of the
//! adapter's name) when that is set, else the highest-ranked of the
//! adapters wgpu finds — discrete GPUs first, then integrated, then
//! virtual, and a software adapter (Mesa's lavapipe, Microsoft's WARP,
//! SwiftShader) only when it is the only one. `WGPU_BACKEND` (`vulkan`,
//! `metal`, `dx12`) restricts the API, `WGPU_POWER_PREF=low` prefers an
//! integrated GPU, as in wgpu itself. The adapter must offer 64-bit shader
//! integers (`SHADER_INT64`: desktop Vulkan and DirectX 12 drivers, Apple
//! GPUs from Metal 2.3 on, lavapipe);
//! [`unavailable_reason`](crate::backend::wgpu::unavailable_reason) says why
//! none was usable.
//!
//! # Determinism
//!
//! The CPU accumulates each histogram bin in `f64`: a chain of additions in
//! row order within fixed blocks of rows, and the block partials added in
//! block order. The GPU sums integers instead. When the backend stages a
//! tree's gradients it finds, per component (gradients, Hessians), the grain
//! `u`: the largest power of two that divides every value. It uploads each
//! value as the integer `x / u`, the kernels add those integers, and the
//! host scales each bin total back by `u`. A node of `n` rows goes to the
//! GPU only when `n * max <= 2^53 u` for both components, checked exactly;
//! inside that bound every partial sum either path forms is exact, so the
//! two histograms are equal bit for bit (proof in the private
//! `backend::exact_sum` module). Every other node — below `CPU_ROWS`
//! (8,192) rows, with non-finite gradients, with grains too coarse for the
//! kernel's accumulators, with inputs that do not match the index the
//! backend was built from, or after a failed GPU command — runs the CPU
//! backend's build inside the backend, so a `device = wgpu` run reproduces
//! the CPU model (the same at every thread count) exactly.
//!
//! The scan is a scatter (the `hist_scatter` kernel): one workgroup per
//! (feature, 256-bin window, row slice), one thread per row, summing into
//! a workgroup-shared histogram through 32-bit atomics (the portable
//! choice: 64-bit atomics are not universal), each 64-bit grain count split
//! into a high and a low 16-bit piece that the merge kernel rejoins in
//! 64-bit integers. Integer addition is order-free, so the result does not
//! depend on the thread schedule. A workgroup's piece sums must stay inside
//! their 32-bit counters, which bounds the rows one workgroup may scan
//! (`scatter_row_bound` in `backend/mod.rs`); a node whose grain
//! counts are too coarse for that runs on the CPU.
//!
//! Prediction works on bit patterns: the walk compares integer keys of the
//! values' bits (as the CPU does), and every addition goes through the
//! kernel's `add_f32`. Multiplications are kept off the GPU: each tree's
//! weighted leaf values are formed on the host (`weight * leaf` in `f32`,
//! the product the CPU forms per row). `add_f32` uses the adapter's own
//! `f32` add only for finite normal operands whose sum is finite and normal
//! (correctly rounded on every Vulkan, Metal, and DirectX 12
//! implementation, and out of reach of subnormal flushing); zeros,
//! subnormal operands or results, infinities, NaN, and overflow take
//! `soft_add`, an IEEE 754 addition in integer arithmetic (round to nearest,
//! ties to even), so a model with subnormal leaves or margins, or two
//! normal leaves whose sum is subnormal, predicts the CPU's bits on an
//! adapter that flushes subnormals. The remaining assumption is the order
//! of the additions: Metal through wgpu compiles with fast math, which
//! permits reassociating them, so the backend checks once per process with
//! a chain of additions that any reassociation changes and refuses
//! [`to_wgpu`](crate::model::BoostedModel::to_wgpu) on an adapter that
//! fails it (training, which only adds integers, stays available). That
//! check is a probe, not a proof.
//!
//! # Limitations
//!
//! - The whole feature-major bin store (2 or 4 bytes per row per feature),
//!   the node's row listing and gradient pairs, and the histogram partials
//!   must each fit one storage buffer binding (`max_storage_buffer_binding_size`,
//!   128 MiB on lavapipe and on wgpu's default limits, usually gigabytes on a
//!   GPU); a dataset that does not is refused.
//! - [`GpuModel`](crate::backend::wgpu::GpuModel) refuses `gblinear` and `linear_tree` models (they do not
//!   predict through the compact forest), and a model trained with model
//!   shrinkage predicts on the CPU.
//! - The gradient slice is converted and re-uploaded once per tree (16
//!   bytes per row) and each prediction call uploads its rows.

use crate::backend::exact_sum::SumDomain;
use crate::backend::{materialize_rows, scatter_row_bound};
use crate::data::DMatrix;
use crate::data::ghist::{Bins, GHistIndex};
use crate::error::{HessboostError, Result};
use crate::model::{
    BoostedModel, Iterations, Predictions, initial_margins, transform_model_margins,
};
use crate::objective::GradPair;
use crate::tree::gain::GradStats;
use crate::tree::hist::{CpuBackend, HistogramBackend};
use bytemuck::{Pod, Zeroable};
use parking_lot::{Mutex, RwLock, RwLockReadGuard};
use rayon::prelude::*;
use std::pin::pin;
use std::sync::{Arc, LazyLock};
use std::task::{Context as TaskContext, Poll, Waker};
use wgpu::util::DeviceExt;

/// Nodes below this many rows run on the CPU backend: the dispatch and
/// readback cost more than the scan.
const CPU_ROWS: usize = 8_192;
/// Row slices a node is scanned in: one workgroup per (window, slice).
const ROW_SLICES: usize = 64;
/// Bins of one feature covered by a scatter workgroup (its 256 threads).
const WINDOW_BINS: usize = 256;
/// Threads per one-item-per-thread workgroup (gather, merge, prediction),
/// the kernels' `@workgroup_size(64)`.
const LINEAR_THREADS: u32 = 64;
/// Workgroups per grid dimension wgpu guarantees on every adapter.
const MAX_GROUPS_PER_DIM: u32 = 65_535;
/// Rows per prediction block: a call uploads and walks its rows this many
/// at a time, so a batch of any size needs bounded GPU buffers.
const PREDICT_BLOCK_ROWS: usize = 262_144;
/// Largest number of pooled prediction buffer sets kept per model.
const PREDICT_POOL: usize = 8;
/// Upper bound on GPU buffer sizes (entries), keeping index math in `u32`.
const MAX_BUFFER_ENTRIES: usize = 1 << 30;
/// Bytes the per-slice histogram partials of one build may take; fewer
/// slices are used when the dataset's bins would exceed it.
const PARTIALS_BUDGET: u64 = 64 << 20;
/// Smallest buffer the backend allocates: wgpu refuses zero-length storage
/// bindings, and an empty logical buffer (never read) gets this stand-in.
const MIN_BUFFER_BYTES: u64 = 16;

/// Whether a usable adapter and compiled kernels are available: `false`
/// without a wgpu adapter offering 64-bit shader integers.
#[must_use]
pub fn available() -> bool {
    Context::shared().is_some()
}

/// Why the wgpu backend is unavailable (no adapter, a missing feature, or
/// a kernel compile failure), for diagnostics; `None` when it is available.
#[must_use]
pub fn unavailable_reason() -> Option<String> {
    CONTEXT.as_ref().err().cloned()
}

/// The name of the adapter this process uses, if any (for diagnostics and
/// benchmarks), as the driver reports it.
#[must_use]
pub fn device_name() -> Option<String> {
    Context::shared().map(|ctx| ctx.device_name.clone())
}

/// Whether the adapter is a software renderer (Mesa's lavapipe, WARP,
/// SwiftShader), which runs the GPU paths on the CPU: correct, useful for
/// testing, and slower than the CPU backend. `None` without an adapter.
#[must_use]
pub fn is_software_adapter() -> Option<bool> {
    Context::shared().map(|ctx| ctx.software)
}

// ---------------------------------------------------------------------------
// WGSL kernels
// ---------------------------------------------------------------------------
// The kernels that run one item per thread (gather, merge, prediction)
// index `gid.y * GRID_X + gid.x` with `GRID_X = 65535 * 64`, the items one
// row of `grid_2d`'s grid covers (`MAX_GROUPS_PER_DIM` workgroups of
// `LINEAR_THREADS`).

/// `hist_gather`: one thread per entry of the node's row listing,
/// `gathered[i] = gpair[rows[i]]`, so the scatter reads the pairs
/// sequentially instead of by row id.
const GATHER_WGSL: &str = r"
struct GatherArgs { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> ga: GatherArgs;
@group(0) @binding(1) var<storage, read> rows: array<u32>;
@group(0) @binding(2) var<storage, read> gpair: array<vec2<i64>>;
@group(0) @binding(3) var<storage, read_write> gathered: array<vec2<i64>>;
const GRID_X: u32 = 65535u * 64u;

@compute @workgroup_size(64)
fn hist_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * GRID_X + gid.x;
    if i >= ga.n { return; }
    gathered[i] = gpair[rows[i]];
}
";

/// `hist_scatter`: one workgroup per (feature window, row slice); each
/// thread takes whole rows of the slice and adds their gradient pairs (in
/// grains) into the workgroup's shared histogram of the window's 256 bins.
///
/// Shared atomics are 32-bit, so each 64-bit grain count is split as
/// `k = hi * 2^16 + lo` with `lo` in `[0, 2^16)`, the pieces accumulated in
/// separate counters; the merge rejoins them. Integer addition is
/// order-free, so the histogram equals the CPU's `f64` chain as long as
/// each workgroup's piece sums stay inside their counters, which the host
/// checks (`scatter_row_bound`) before dispatching.
///
/// The bin store is feature-major (`feature * n_rows + row`), `u16` pairs
/// packed in `u32` words or plain `u32`s (`wide`); a missing entry holds the
/// width's maximum, which lands outside every window.
const SCATTER_WGSL: &str = r"
struct ScatterArgs {
    n_node_rows: u32,    // entries in the node's row listing
    rows_per_slice: u32, // rows one workgroup scans
    n_rows: u32,         // dataset rows: the feature-major store's stride
    total_bins: u32,     // bins in one partial histogram
    wide: u32,           // 1 for u32 bins, 0 for packed u16 bins
    pad0: u32, pad1: u32, pad2: u32,
}
// One (feature, bin window) work item: the feature owns global bins
// `[fs, fs + nbins)`; the window covers `[fs + base, +256)`.
struct Window { feature: u32, fs: u32, nbins: u32, base: u32 }

@group(0) @binding(0) var<uniform> sa: ScatterArgs;
@group(0) @binding(1) var<storage, read> bins: array<u32>;
@group(0) @binding(2) var<storage, read> rows: array<u32>;
@group(0) @binding(3) var<storage, read> grad: array<vec2<i64>>;
@group(0) @binding(4) var<storage, read> windows: array<Window>;
@group(0) @binding(5) var<storage, read_write> partials: array<vec4<i32>>;

const THREADS: u32 = 256u;
const WINDOW_BINS: u32 = 256u;
var<workgroup> hist: array<atomic<u32>, 1024>;

fn load_bin(idx: u32) -> u32 {
    if sa.wide != 0u { return bins[idx]; }
    return (bins[idx >> 1u] >> ((idx & 1u) * 16u)) & 0xFFFFu;
}

@compute @workgroup_size(256)
fn hist_scatter(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let w = windows[wg.x];
    let win_base = w.fs + w.base;
    var n_win = 0u;
    if w.nbins > w.base { n_win = min(WINDOW_BINS, w.nbins - w.base); }
    for (var i = tid; i < WINDOW_BINS * 4u; i += THREADS) { atomicStore(&hist[i], 0u); }
    workgroupBarrier();
    let slice_begin = wg.y * sa.rows_per_slice;
    var slice_rows = 0u;
    if slice_begin < sa.n_node_rows {
        slice_rows = min(sa.rows_per_slice, sa.n_node_rows - slice_begin);
    }
    let column = w.feature * sa.n_rows;
    for (var i = tid; i < slice_rows; i += THREADS) {
        // A sentinel and a bin outside this window both wrap past `n_win`.
        let b = load_bin(column + rows[slice_begin + i]) - win_base;
        if b < n_win {
            let q = grad[slice_begin + i];
            atomicAdd(&hist[b * 4u + 0u], u32(q.x >> 16u));
            atomicAdd(&hist[b * 4u + 1u], u32(q.x & 0xFFFFli));
            atomicAdd(&hist[b * 4u + 2u], u32(q.y >> 16u));
            atomicAdd(&hist[b * 4u + 3u], u32(q.y & 0xFFFFli));
        }
    }
    workgroupBarrier();
    let out = wg.y * sa.total_bins + win_base;
    for (var b = tid; b < n_win; b += THREADS) {
        partials[out + b] = vec4<i32>(
            i32(atomicLoad(&hist[b * 4u + 0u])),
            i32(atomicLoad(&hist[b * 4u + 1u])),
            i32(atomicLoad(&hist[b * 4u + 2u])),
            i32(atomicLoad(&hist[b * 4u + 3u])));
    }
}
";

/// `hist_merge`: one thread per bin, rejoining the slices' (high, low)
/// pieces in 64-bit integers. Exact, so the order does not matter (it is
/// still fixed).
const MERGE_WGSL: &str = r"
struct MergeArgs { slices: u32, total_bins: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> ma: MergeArgs;
@group(0) @binding(1) var<storage, read> partials: array<vec4<i32>>;
@group(0) @binding(2) var<storage, read_write> hist: array<vec2<i64>>;
const GRID_X: u32 = 65535u * 64u;

@compute @workgroup_size(64)
fn hist_merge(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.y * GRID_X + gid.x;
    if b >= ma.total_bins { return; }
    var g_hi = 0li; var g_lo = 0li; var h_hi = 0li; var h_lo = 0li;
    for (var s = 0u; s < ma.slices; s++) {
        let p = partials[s * ma.total_bins + b];
        // The high pieces are signed, the low ones unsigned (they add up
        // below 2^32 by the host's row bound): sign- and zero-extend.
        g_hi += i64(p.x);
        g_lo += i64(u64(u32(p.y)));
        h_hi += i64(p.z);
        h_lo += i64(u64(u32(p.w)));
    }
    hist[b] = vec2<i64>(g_hi * 65536li + g_lo, h_hi * 65536li + h_lo);
}
";

/// `forest_predict`: one row per thread, walking every tree of the range
/// and adding each tree's (host-weighted) leaf value onto the
/// pre-initialized margins in tree order, the order the CPU accumulates.
///
/// A node is the `CNode` of `tree/compact.rs`: `slot = feature * 32` (+16
/// when the split reads the negated value), `key` the threshold key ("go
/// right when greater") or the category pool start, `left` the child taken
/// when the compare is false (a leaf points at itself), `aux` the leaf
/// value's bits, the leaf-vector offset, or the categorical flags. Rows,
/// leaves, and margins are handled as `u32` bit patterns: the walk compares
/// integer keys (as the CPU does, `tree::compact::key`), so a missing
/// `NaN`, `-0.0`, and infinities route like the CPU's, and the additions go
/// through `add_f32`, which is exact for every finite input even where the
/// adapter flushes subnormals to zero (see the module docs).
const PREDICT_WGSL: &str = r"
struct PredictArgs {
    n_rows: u32, n_cols: u32, k: u32, tree_begin: u32,
    tree_end: u32, soft: u32, pad0: u32, pad1: u32,
}
struct PTree { root: u32, output: u32, vector: u32, pad: u32 }

@group(0) @binding(0) var<uniform> pa: PredictArgs;
@group(0) @binding(1) var<storage, read> nodes: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> categories: array<u32>;
@group(0) @binding(3) var<storage, read> leaf_vectors: array<u32>;
@group(0) @binding(4) var<storage, read> trees: array<PTree>;
@group(0) @binding(5) var<storage, read> rows: array<u32>;
@group(0) @binding(6) var<storage, read_write> out: array<u32>;

// Monotone unsigned key of a float's bits, matching `tree::compact::key`:
// NaN maps to 0, -0.0 is treated as +0.0, negatives are complemented.
fn key_of(bits: u32) -> u32 {
    var vb = bits;
    if (vb & 0x7F800000u) == 0x7F800000u && (vb & 0x007FFFFFu) != 0u { return 0u; }
    if vb == 0x80000000u { vb = 0u; }
    return vb ^ (u32(i32(vb) >> 31u) | 0x80000000u);
}

fn is_nan(bits: u32) -> bool {
    return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u;
}

// A non-missing value's category code: Rust's saturating `v as u32`,
// decided on the bits (negative values and -0.0 give 0, values of 2^32 or
// more and +inf give the maximum) so no float compare is involved; the
// remaining conversion truncates a value in [0, 2^32), where a flushed
// subnormal truncates to 0 either way.
fn category_of(bits: u32) -> u32 {
    if (bits & 0x80000000u) != 0u { return 0u; }
    if (bits >> 23u) >= 159u { return 0xFFFFFFFFu; }
    return u32(bitcast<f32>(bits));
}

fn cat_in_set(begin: u32, end: u32, cat: u32) -> bool {
    for (var i = begin; i < end; i++) {
        if categories[i] == cat { return true; }
    }
    return false;
}

// IEEE 754 binary32 addition on the bit patterns, in integer arithmetic:
// round to nearest, ties to even, exact for every finite input including
// subnormals (which the adapter may flush to zero in its own float adds)
// and for zero signs and overflow. NaNs propagate quieted.
fn soft_add(a: u32, b: u32) -> u32 {
    // `x` holds the larger magnitude.
    var x = a;
    var y = b;
    if (b & 0x7FFFFFFFu) > (a & 0x7FFFFFFFu) { x = b; y = a; }
    let ex = (x >> 23u) & 0xFFu;
    let ey = (y >> 23u) & 0xFFu;
    if ex == 0xFFu {
        if (x & 0x7FFFFFu) != 0u { return x | 0x400000u; }
        if ey == 0xFFu {
            if (y & 0x7FFFFFu) != 0u { return y | 0x400000u; }
            if ((x ^ y) & 0x80000000u) != 0u { return 0x7FC00000u; }
        }
        return x;
    }
    if (y & 0x7FFFFFFFu) == 0u {
        // Adding a zero: `x`, except that two zeros give +0 unless both are -0.
        if (x & 0x7FFFFFFFu) == 0u { return x & y; }
        return x;
    }
    // Significands with the hidden bit at bit 29 (a subnormal has none and
    // the exponent of the smallest normal), six extra bits for rounding.
    var mx = (x & 0x7FFFFFu) << 6u;
    var my = (y & 0x7FFFFFu) << 6u;
    var e = ex;
    var ey2 = ey;
    if ex == 0u { e = 1u; } else { mx |= 0x20000000u; }
    if ey == 0u { ey2 = 1u; } else { my |= 0x20000000u; }
    // Align `y` to `x`'s exponent, jamming the lost bits into the sticky bit.
    let shift = e - ey2;
    if shift >= 32u {
        my = select(0u, 1u, my != 0u);
    } else if shift != 0u {
        let lost = my & ((1u << shift) - 1u);
        my = (my >> shift) | select(0u, 1u, lost != 0u);
    }
    let sign = x & 0x80000000u;
    var m: u32;
    if ((x ^ y) & 0x80000000u) == 0u {
        m = mx + my;
        if (m & 0x40000000u) != 0u {
            m = (m >> 1u) | (m & 1u);
            e += 1u;
        }
    } else {
        m = mx - my;
        if m == 0u { return 0u; }
        // Renormalize (bit 29 is the hidden bit's place) as far as the
        // exponent allows; a result that cannot reach it is subnormal.
        let s = min(countLeadingZeros(m) - 2u, e - 1u);
        m = m << s;
        e -= s;
    }
    // Round the six extra bits to nearest, ties to even.
    let low = m & 0x3Fu;
    m = m >> 6u;
    if low > 0x20u || (low == 0x20u && (m & 1u) != 0u) { m += 1u; }
    if (m & 0x1000000u) != 0u {
        m = m >> 1u;
        e += 1u;
    }
    if e >= 0xFFu { return sign | 0x7F800000u; }
    if (m & 0x800000u) == 0u { return sign | m; }
    return sign | (e << 23u) | (m & 0x7FFFFFu);
}

// `a + b` on the bit patterns, bit for bit as the CPU computes it. Finite
// normal operands whose native sum is finite and normal take the adapter's
// add (correctly rounded, and flushing cannot touch it); everything else
// (zeros, subnormals, infinities, NaN, a sum that leaves the normal range)
// goes through `soft_add`. `pa.soft` forces the integer path (tests).
fn add_f32(a: u32, b: u32) -> u32 {
    if pa.soft == 0u
        && ((a >> 23u) & 0xFFu) - 1u < 254u
        && ((b >> 23u) & 0xFFu) - 1u < 254u
    {
        let s = bitcast<u32>(bitcast<f32>(a) + bitcast<f32>(b));
        if ((s >> 23u) & 0xFFu) - 1u < 254u { return s; }
    }
    return soft_add(a, b);
}

@compute @workgroup_size(64)
fn forest_predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= pa.n_rows { return; }
    let row = r * pa.n_cols;
    let scalar_run = pa.k == 1u;
    var acc = 0u;
    if scalar_run { acc = out[r]; }
    for (var t = pa.tree_begin; t < pa.tree_end; t++) {
        let tr = trees[t];
        var nid = tr.root;
        var n = nodes[nid];
        while n.z != nid {
            let bits = rows[row + n.x / 32u];
            if (n.w & 1u) != 0u {
                var go_left: bool;
                if is_nan(bits) {
                    go_left = (n.w & 2u) != 0u;
                } else {
                    go_left = cat_in_set(n.y, n.w >> 2u, category_of(bits));
                }
                nid = n.z + select(1u, 0u, go_left);
            } else {
                var vb = bits;
                if (n.x & 16u) != 0u { vb ^= 0x80000000u; }
                nid = n.z + select(0u, 1u, key_of(vb) > n.y);
            }
            n = nodes[nid];
        }
        if tr.vector != 0u {
            let o = r * pa.k;
            for (var j = 0u; j < pa.k; j++) {
                out[o + j] = add_f32(out[o + j], leaf_vectors[n.w + j]);
            }
        } else if scalar_run {
            acc = add_f32(acc, n.w);
        } else {
            let o = r * pa.k + tr.output;
            out[o] = add_f32(out[o], n.w);
        }
    }
    if scalar_run { out[r] = acc; }
}
";

// ---------------------------------------------------------------------------
// Runtime context
// ---------------------------------------------------------------------------

/// Kernel argument block of `hist_gather`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GatherArgs {
    n: u32,
    pad: [u32; 3],
}

/// Kernel argument block of `hist_scatter`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScatterArgs {
    n_node_rows: u32,
    rows_per_slice: u32,
    n_rows: u32,
    total_bins: u32,
    wide: u32,
    pad: [u32; 3],
}

/// One (feature, bin window) work item of the scatter kernel.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Window {
    feature: u32,
    fs: u32,
    nbins: u32,
    base: u32,
}

/// Kernel argument block of `hist_merge`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MergeArgs {
    slices: u32,
    total_bins: u32,
    pad: [u32; 2],
}

/// Kernel argument block of `forest_predict`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PredictArgs {
    n_rows: u32,
    n_cols: u32,
    k: u32,
    tree_begin: u32,
    tree_end: u32,
    /// Force `soft_add` for every addition (tests; `0` in production).
    soft: u32,
    pad: [u32; 2],
}

/// One tree's dispatch record of `forest_predict`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PTree {
    root: u32,
    output: u32,
    vector: u32,
    pad: u32,
}

/// The process-wide wgpu context: device, queue, limits, and compiled
/// kernels.
struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: wgpu::Limits,
    gather: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    merge: wgpu::ComputePipeline,
    predict: wgpu::ComputePipeline,
    device_name: String,
    software: bool,
    /// Whether the adapter's float addition matched the CPU's in the
    /// reassociation probe, or why it did not; prediction needs a pass.
    predict_check: std::result::Result<(), String>,
}

/// The lazily initialized process-wide context, or the reason it is
/// unavailable.
static CONTEXT: LazyLock<std::result::Result<Context, String>> = LazyLock::new(Context::new);

/// Drive a wgpu future to completion. wgpu's native futures (adapter and
/// device requests, error-scope pops) are ready at the first poll; the
/// loop only yields if one ever is not.
fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut cx = TaskContext::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
            return output;
        }
        std::thread::yield_now();
    }
}

/// Ranking of an adapter for automatic selection (lower is better): real
/// GPUs before virtual ones, a software renderer last. `WGPU_POWER_PREF=low`
/// puts integrated GPUs before discrete ones.
fn adapter_rank(info: &wgpu::AdapterInfo, prefer_low_power: bool) -> u8 {
    use wgpu::DeviceType;
    match info.device_type {
        DeviceType::DiscreteGpu => u8::from(prefer_low_power),
        DeviceType::IntegratedGpu => u8::from(!prefer_low_power),
        DeviceType::Other => 2,
        DeviceType::VirtualGpu => 3,
        DeviceType::Cpu => 4,
    }
}

/// The adapter to use, by name when `WGPU_ADAPTER_NAME` is set, else the
/// best-ranked adapter offering 64-bit shader integers.
fn select_adapter(instance: &wgpu::Instance) -> std::result::Result<wgpu::Adapter, String> {
    let adapters = block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    if adapters.is_empty() {
        return Err("no wgpu adapter found (no Vulkan, Metal, or DirectX 12 driver)".to_string());
    }
    if let Ok(wanted) = std::env::var("WGPU_ADAPTER_NAME") {
        let wanted = wanted.to_lowercase();
        return adapters
            .into_iter()
            .find(|a| a.get_info().name.to_lowercase().contains(&wanted))
            .ok_or_else(|| format!("no wgpu adapter matches WGPU_ADAPTER_NAME={wanted:?}"));
    }
    let prefer_low_power = matches!(
        wgpu::PowerPreference::from_env(),
        Some(wgpu::PowerPreference::LowPower)
    );
    let mut names = Vec::new();
    let mut best: Option<(u8, wgpu::Adapter)> = None;
    for adapter in adapters {
        let info = adapter.get_info();
        names.push(format!(
            "{} ({:?}, {:?})",
            info.name, info.device_type, info.backend
        ));
        if !adapter.features().contains(wgpu::Features::SHADER_INT64) {
            continue;
        }
        let rank = adapter_rank(&info, prefer_low_power);
        if best.as_ref().is_none_or(|(r, _)| rank < *r) {
            best = Some((rank, adapter));
        }
    }
    best.map(|(_, adapter)| adapter).ok_or_else(|| {
        format!(
            "no wgpu adapter offers 64-bit shader integers (found: {})",
            names.join(", ")
        )
    })
}

impl Context {
    /// The shared context, or `None` when it failed to initialize
    /// (see [`unavailable_reason`]). Computed once per process.
    fn shared() -> Option<&'static Self> {
        CONTEXT.as_ref().ok()
    }

    fn new() -> std::result::Result<Self, String> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc.with_env());
        let adapter = select_adapter(&instance)?;
        let info = adapter.get_info();
        let device_name = info.name.clone();
        let software = info.device_type == wgpu::DeviceType::Cpu;
        if !adapter.features().contains(wgpu::Features::SHADER_INT64) {
            return Err(format!(
                "wgpu adapter {device_name} has no 64-bit shader integers (SHADER_INT64)"
            ));
        }
        let limits = adapter.limits();
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("hessboost"),
            required_features: wgpu::Features::SHADER_INT64,
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|e| format!("requesting a wgpu device from {device_name} failed: {e}"))?;
        // Validation errors are reported to error scopes; what escapes them
        // must never panic inside a library, so the uncaptured handler only
        // records.
        device.on_uncaptured_error(Arc::new(|error: wgpu::Error| {
            UNCAPTURED.lock().get_or_insert_with(|| error.to_string());
        }));
        let pipeline = |name: &str, source: &str| -> wgpu::ComputePipeline {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: Some(name),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let [gather, scatter, merge, predict] =
            scoped(&device, "compiling the wgpu kernels", || {
                Ok([
                    pipeline("hist_gather", GATHER_WGSL),
                    pipeline("hist_scatter", SCATTER_WGSL),
                    pipeline("hist_merge", MERGE_WGSL),
                    pipeline("forest_predict", PREDICT_WGSL),
                ])
            })
            .map_err(|e| format!("{e} ({device_name})"))?;
        let mut ctx = Context {
            device,
            queue,
            limits,
            gather,
            scatter,
            merge,
            predict,
            device_name,
            software,
            predict_check: Ok(()),
        };
        ctx.predict_check = ctx.probe_addition_order();
        Ok(ctx)
    }
}

/// Run `f` with every kind of wgpu error (validation, out of memory,
/// internal) captured and turned into an error naming `what`: the scopes
/// are thread-local, so concurrent builds and predictions each see their
/// own failures, and a failed allocation or command never panics or poisons
/// the device.
fn scoped<T>(device: &wgpu::Device, what: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let value = f();
    let captured = [
        block_on(validation.pop()),
        block_on(memory.pop()),
        block_on(internal.pop()),
    ];
    match captured.into_iter().flatten().next() {
        Some(error) => Err(HessboostError::gpu(format!("{what} failed: {error}"))),
        None => value,
    }
}

impl Context {
    /// [`scoped`] on this context's device.
    fn scoped<T>(&self, what: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
        scoped(&self.device, what, f)
    }

    /// Submit what `encode` encodes and wait for the submission: a captured
    /// error, a lost device, or a failed poll is an error, after which the
    /// GPU no longer touches the buffers the commands referenced.
    fn run(&self, encode: impl FnOnce(&mut wgpu::CommandEncoder)) -> Result<()> {
        self.scoped("a wgpu command", || {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encode(&mut encoder);
            let submission = self.queue.submit([encoder.finish()]);
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: None,
                })
                .map_err(|e| HessboostError::gpu(format!("waiting for the GPU failed: {e}")))?;
            Ok(())
        })?;
        if let Some(error) = UNCAPTURED.lock().as_ref() {
            return Err(HessboostError::gpu(format!(
                "the wgpu device reported: {error}"
            )));
        }
        Ok(())
    }

    /// Map `staging` (just written by a completed submission) and decode
    /// its first `size_of_val(into)` bytes into `into`.
    fn read_back<T: Pod>(&self, staging: &wgpu::Buffer, into: &mut [T]) -> Result<()> {
        let into: &mut [u8] = bytemuck::cast_slice_mut(into);
        let bytes = into.len() as u64;
        let slice = staging.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let mapped = loop {
            match rx.try_recv() {
                Ok(result) => break result,
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    self.device
                        .poll(wgpu::PollType::wait_indefinitely())
                        .map_err(|e| {
                            HessboostError::gpu(format!("waiting for the GPU failed: {e}"))
                        })?;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err(HessboostError::gpu("mapping a GPU buffer was abandoned"));
                }
            }
        };
        mapped.map_err(|e| HessboostError::gpu(format!("mapping a GPU buffer failed: {e}")))?;
        {
            let view = slice
                .get_mapped_range()
                .map_err(|e| HessboostError::gpu(format!("reading a GPU buffer failed: {e}")))?;
            into.copy_from_slice(&view[..into.len()]);
        }
        staging.unmap();
        Ok(())
    }

    /// A storage buffer of `bytes` (at least [`MIN_BUFFER_BYTES`]) the CPU
    /// writes through the queue, or an error when it exceeds the adapter's
    /// binding limit.
    fn storage(&self, label: &str, bytes: u64, extra: wgpu::BufferUsages) -> Result<wgpu::Buffer> {
        let bytes = bytes
            .max(MIN_BUFFER_BYTES)
            .next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        if bytes > self.limits.max_storage_buffer_binding_size
            || bytes > self.limits.max_buffer_size
        {
            return Err(HessboostError::invalid_data(
                "data",
                format!(
                    "the {label} buffer ({bytes} bytes) exceeds the wgpu adapter's \
                     storage binding limit ({} bytes)",
                    self.limits.max_storage_buffer_binding_size
                ),
            ));
        }
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
            mapped_at_creation: false,
        }))
    }

    /// A storage buffer holding `contents` (checked against the binding
    /// limit like [`storage`](Self::storage)).
    fn storage_init(&self, label: &str, contents: &[u8]) -> Result<wgpu::Buffer> {
        if contents.len() < MIN_BUFFER_BYTES as usize {
            let buffer = self.storage(label, MIN_BUFFER_BYTES, wgpu::BufferUsages::empty())?;
            self.queue.write_buffer(&buffer, 0, contents);
            return Ok(buffer);
        }
        let bytes = contents.len() as u64;
        if bytes > self.limits.max_storage_buffer_binding_size
            || bytes > self.limits.max_buffer_size
        {
            return Err(HessboostError::invalid_data(
                "data",
                format!(
                    "the {label} buffer ({bytes} bytes) exceeds the wgpu adapter's \
                     storage binding limit ({} bytes)",
                    self.limits.max_storage_buffer_binding_size
                ),
            ));
        }
        Ok(self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE,
            }))
    }

    /// A uniform buffer for one argument block of `T`.
    fn uniform<T: Pod>(&self, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: std::mem::size_of::<T>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// A CPU-readable staging buffer of `bytes`.
    fn staging(&self, label: &str, bytes: u64) -> Result<wgpu::Buffer> {
        let bytes = bytes
            .max(MIN_BUFFER_BYTES)
            .next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        if bytes > self.limits.max_buffer_size {
            return Err(HessboostError::invalid_data(
                "data",
                format!(
                    "the {label} readback ({bytes} bytes) exceeds the wgpu adapter's \
                     buffer limit ({} bytes)",
                    self.limits.max_buffer_size
                ),
            ));
        }
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }

    /// Bind `entries` (in binding order) to `pipeline`'s only group.
    fn bind(&self, pipeline: &wgpu::ComputePipeline, entries: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry<'_>> = entries
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    /// Whether this adapter adds floats in the order the kernel writes
    /// them: a forest of single-leaf trees whose leaf values sum to `0` in
    /// sequence (`1`, then sixteen `2^-24`s that each round away against the
    /// running `1`, then `-1`) and to `16 * 2^-24` under any reassociation
    /// of the small terms. Fast-math compilers (Metal's default) may
    /// reassociate; the probe is a check, not a proof.
    fn probe_addition_order(&self) -> std::result::Result<(), String> {
        let small = 2f32.powi(-24);
        let mut leaves = vec![1.0f32];
        leaves.extend(std::iter::repeat_n(small, 16));
        leaves.push(-1.0);
        let chain = AdditionChain {
            k: 1,
            vector: false,
            leaves,
            margins: vec![0.0; 64],
            soft: false,
        };
        let expected = chain.cpu();
        if expected.iter().any(|&m| m != 0.0) {
            return Err("the CPU probe sum is not 0".to_string());
        }
        let got = self.run_chain(&chain).map_err(|e| e.to_string())?;
        match got
            .iter()
            .zip(&expected)
            .find(|(g, e)| g.to_bits() != e.to_bits())
        {
            None => Ok(()),
            Some((got, expected)) => Err(format!(
                "the adapter {} reassociates float additions (a probe summed to {got} instead \
                 of {expected}); its predictions could differ from the CPU's",
                self.device_name
            )),
        }
    }

    /// Run `chain` through the prediction kernel: one single-leaf tree per
    /// leaf (or per `k` leaf values with vector leaves), one row per margin
    /// row, no features. Returns the accumulated margins.
    fn run_chain(&self, chain: &AdditionChain) -> Result<Vec<f32>> {
        let k = chain.k;
        let n_trees = chain.leaves.len() / if chain.vector { k } else { 1 };
        let rows = chain.margins.len() / k;
        // Single-leaf trees: node `t` is its own left child; `aux` holds the
        // leaf's bits, or the tree's offset into the leaf-vector pool.
        let nodes: Vec<[u32; 4]> = (0..n_trees)
            .map(|t| {
                let aux = if chain.vector {
                    (t * k) as u32
                } else {
                    chain.leaves[t].to_bits()
                };
                [0, 0, t as u32, aux]
            })
            .collect();
        let trees: Vec<PTree> = (0..n_trees)
            .map(|t| PTree {
                root: t as u32,
                output: (t % k) as u32,
                vector: u32::from(chain.vector),
                pad: 0,
            })
            .collect();
        let (forest, call) = self.scoped("allocating the addition chain", || {
            let forest = ForestBuffers {
                nodes: self.storage_init("chain nodes", bytemuck::cast_slice(&nodes))?,
                categories: self.storage_init("chain categories", &[0; 16])?,
                leaf_vectors: self
                    .storage_init("chain leaf vectors", bytemuck::cast_slice(&chain.leaves))?,
                trees: self.storage_init("chain trees", bytemuck::cast_slice(&trees))?,
            };
            let call = PredictBuffers::new(self, &forest, rows, 1, k, rows)?;
            Ok((forest, call))
        })?;
        drop(forest);
        let features = vec![0.0f32; rows];
        let mut margins = chain.margins.clone();
        call.run_block(
            self,
            &PredictArgs {
                n_rows: rows as u32,
                n_cols: 1,
                k: k as u32,
                tree_begin: 0,
                tree_end: n_trees as u32,
                soft: u32::from(chain.soft),
                pad: [0; 2],
            },
            &features,
            &margins,
            0,
        )?;
        call.read_margins(self, &mut margins)?;
        Ok(margins)
    }
}

/// A sequence of leaf additions run through the prediction kernel without
/// any tree walk: the probe's and the tests' way to exercise `add_f32` on
/// chosen bit patterns, through each of the kernel's three accumulation
/// paths (one output, several scalar outputs, vector leaves).
struct AdditionChain {
    /// Outputs per row.
    k: usize,
    /// Vector leaves (`k` values per tree) instead of one scalar leaf per
    /// tree feeding output `t % k`.
    vector: bool,
    /// Leaf values, `k` per tree when `vector`.
    leaves: Vec<f32>,
    /// Initial margins, `k` per row.
    margins: Vec<f32>,
    /// Force the integer addition path for every add.
    soft: bool,
}

impl AdditionChain {
    /// What the CPU computes: each row's margins plus every tree's leaf in
    /// tree order, in `f32`.
    fn cpu(&self) -> Vec<f32> {
        let k = self.k;
        let mut out = self.margins.clone();
        for row in out.chunks_exact_mut(k) {
            if self.vector {
                for leaf in self.leaves.chunks_exact(k) {
                    for (o, &l) in row.iter_mut().zip(leaf) {
                        *o += l;
                    }
                }
            } else {
                for (t, &l) in self.leaves.iter().enumerate() {
                    row[t % k] += l;
                }
            }
        }
        out
    }
}

/// The first error wgpu reported outside an error scope (a device-wide
/// condition such as a lost device), if any.
static UNCAPTURED: Mutex<Option<String>> = Mutex::new(None);

/// A 2-D grid of one-item-per-thread workgroups covering `items` (the
/// kernels index `gid.y * GRID_X + gid.x`): wgpu guarantees 65,535
/// workgroups per dimension, which one dimension of 64-thread groups
/// exceeds past 4.2 million items.
fn grid_2d(items: usize) -> (u32, u32) {
    let groups = (items as u64).div_ceil(u64::from(LINEAR_THREADS)).max(1);
    let x = groups.min(u64::from(MAX_GROUPS_PER_DIM));
    let y = groups.div_ceil(u64::from(MAX_GROUPS_PER_DIM));
    (x as u32, y as u32)
}

// ---------------------------------------------------------------------------
// Histogram backend
// ---------------------------------------------------------------------------

/// The gradient slice staged by [`HistogramBackend::prepare`]: uploaded to
/// the GPU, with the statistics that decide whether a node's sums are
/// exact on both paths.
///
/// Lives behind the backend's `RwLock`: staging (the only write to
/// `buffer`) takes the write lock, and every GPU build holds a read guard
/// from checking that the staged slice is its own until its submission has
/// completed, so the slice the kernels read is the one whose statistics
/// gated the node.
struct StagedGradients {
    buffer: wgpu::Buffer,
    /// Conversion scratch: the slice in grains, uploaded through the queue.
    units: Vec<[i64; 2]>,
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

/// Per-call GPU buffers of the histogram backend, pooled across the parallel
/// node builds of a training run. Each concurrent `build` owns one set, with
/// its bind groups (every buffer they reference is fixed for the backend's
/// lifetime).
struct CallBuffers {
    rows: wgpu::Buffer,
    hist: wgpu::Buffer,
    readback: wgpu::Buffer,
    gather_args: wgpu::Buffer,
    scatter_args: wgpu::Buffer,
    merge_args: wgpu::Buffer,
    gather_bind: wgpu::BindGroup,
    scatter_bind: wgpu::BindGroup,
    merge_bind: wgpu::BindGroup,
}

impl CallBuffers {
    /// One set for `backend`: the node's row listing, its gathered
    /// gradient pairs, the per-slice partials, and the merged histogram
    /// with its readback copy.
    fn new(backend: &WgpuHistBackend, gradients: &wgpu::Buffer) -> Result<Self> {
        let ctx = backend.ctx;
        let n_rows = backend.n_rows as u64;
        let total_bins = backend.total_bins as u64;
        let rows = ctx.storage("row listing", n_rows * 4, wgpu::BufferUsages::empty())?;
        let gathered = ctx.storage(
            "gathered gradients",
            n_rows * 16,
            wgpu::BufferUsages::empty(),
        )?;
        let partials = ctx.storage(
            "histogram partials",
            backend.slices as u64 * total_bins * 16,
            wgpu::BufferUsages::empty(),
        )?;
        let hist = ctx.storage("histogram", total_bins * 16, wgpu::BufferUsages::COPY_SRC)?;
        let readback = ctx.staging("histogram", total_bins * 16)?;
        let gather_args = ctx.uniform::<GatherArgs>("gather args");
        let scatter_args = ctx.uniform::<ScatterArgs>("scatter args");
        let merge_args = ctx.uniform::<MergeArgs>("merge args");
        let gather_bind = ctx.bind(&ctx.gather, &[&gather_args, &rows, gradients, &gathered]);
        let scatter_bind = ctx.bind(
            &ctx.scatter,
            &[
                &scatter_args,
                &backend.bins,
                &rows,
                &gathered,
                &backend.windows,
                &partials,
            ],
        );
        let merge_bind = ctx.bind(&ctx.merge, &[&merge_args, &partials, &hist]);
        Ok(CallBuffers {
            rows,
            hist,
            readback,
            gather_args,
            scatter_args,
            merge_args,
            gather_bind,
            scatter_bind,
            merge_bind,
        })
    }
}

/// The wgpu histogram backend: implements [`HistogramBackend`] by scanning
/// the binned column store on the GPU. Constructed once per training run
/// (the column upload and window descriptors are per-dataset); the gradient
/// slice is re-uploaded by [`HistogramBackend::prepare`] once per tree.
///
/// Training selects it automatically through
/// [`device = wgpu`](crate::config::TrainingParams::device); constructing it
/// directly serves custom training loops against a [`GHistIndex`]. Its
/// histograms equal the CPU backend's bit for bit: nodes the GPU cannot sum
/// exactly (see the [module docs](crate::backend::wgpu)) run the CPU
/// backend's build instead. `build` must receive the index the backend was
/// built from, and the gradient slice must not change between `prepare` and
/// the tree's last `build`; inputs that do not fit the backend's buffers (a
/// different index shape, a gradient slice of another length, row indices
/// past the index) never reach the GPU and take the CPU path, which checks
/// them.
pub struct WgpuHistBackend {
    ctx: &'static Context,
    /// The feature-major bin store (feature `f` of row `r` at
    /// `f * n_rows + r`): `u16` pairs packed in `u32` words, or `u32`s when
    /// `wide`; a missing entry holds the width's maximum.
    bins: wgpu::Buffer,
    wide: bool,
    /// One (feature, 256-bin window) work item per scatter workgroup column.
    windows: wgpu::Buffer,
    n_windows: usize,
    /// Row slices a node is split into (one scatter workgroup row each):
    /// [`ROW_SLICES`], or fewer when the dataset's bins would overrun
    /// [`PARTIALS_BUDGET`].
    slices: usize,
    total_bins: usize,
    n_rows: usize,
    n_cols: usize,
    gradients: RwLock<StagedGradients>,
    pool: Mutex<Vec<CallBuffers>>,
}

impl std::fmt::Debug for WgpuHistBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuHistBackend")
            .field("device", &self.ctx.device_name)
            .field("n_rows", &self.n_rows)
            .field("n_cols", &self.n_cols)
            .field("total_bins", &self.total_bins)
            .finish_non_exhaustive()
    }
}

/// The feature-major bin store the scatter kernel reads, as the `u32` words
/// to upload, and whether they hold `u32` bins (`true`) or packed `u16`
/// pairs. Dense indexes and sparse ones at least half full already keep
/// such a store (with the width's maximum marking a missing entry); the
/// rest is built from the CSR rows here.
fn feature_major_bins(index: &GHistIndex) -> (Vec<u32>, bool) {
    let n = index.n_rows();
    let cols = index.n_cols();
    let pack_u16 = |bins: &[u16]| -> Vec<u32> {
        let mut words = vec![0u32; bins.len().div_ceil(2)];
        words
            .par_iter_mut()
            .zip(bins.par_chunks(2))
            .for_each(|(word, pair)| {
                *word = u32::from(pair[0]) | (pair.get(1).map_or(0, |&b| u32::from(b)) << 16);
            });
        words
    };
    match index.column_bins().or_else(|| index.missing_columns()) {
        Some(Bins::U16(bins)) => (pack_u16(bins), false),
        Some(Bins::U32(bins)) => (bins.to_vec(), true),
        None => {
            // Sparse and under half full: scatter each row's entries into a
            // sentinel-filled store. The sentinel must never be a bin, so
            // `u16` stores need `total_bins < u16::MAX`.
            let narrow = index.total_bins() < usize::from(u16::MAX);
            let cuts = index.cuts();
            let starts: Vec<u32> = (0..cols).map(|f| cuts.feature_bins(f).0 as u32).collect();
            let feature_of = |bin: u32| starts.partition_point(|&start| start <= bin) - 1;
            let row_ptr = index.row_ptr();
            let place = |r: usize, bin: u32, store: &mut dyn FnMut(usize, u32)| {
                store(feature_of(bin) * n + r, bin);
            };
            if narrow {
                let mut store = vec![u16::MAX; n * cols];
                let mut put = |i: usize, bin: u32| store[i] = bin as u16;
                for r in 0..n {
                    let (s, e) = (row_ptr[r], row_ptr[r + 1]);
                    match index.bins() {
                        Bins::U16(b) => b[s..e]
                            .iter()
                            .for_each(|&bin| place(r, u32::from(bin), &mut put)),
                        Bins::U32(b) => b[s..e].iter().for_each(|&bin| place(r, bin, &mut put)),
                    }
                }
                (pack_u16(&store), false)
            } else {
                let mut store = vec![u32::MAX; n * cols];
                let mut put = |i: usize, bin: u32| store[i] = bin;
                for r in 0..n {
                    let (s, e) = (row_ptr[r], row_ptr[r + 1]);
                    match index.bins() {
                        Bins::U16(b) => b[s..e]
                            .iter()
                            .for_each(|&bin| place(r, u32::from(bin), &mut put)),
                        Bins::U32(b) => b[s..e].iter().for_each(|&bin| place(r, bin, &mut put)),
                    }
                }
                (store, true)
            }
        }
    }
}

impl WgpuHistBackend {
    /// Build the backend for `index`: upload its feature-major bin store and
    /// the (feature, bin window) work items.
    pub fn new(index: &GHistIndex) -> Result<Self> {
        let ctx = Context::shared().ok_or_else(|| {
            HessboostError::gpu(
                unavailable_reason().unwrap_or_else(|| "no wgpu adapter is available".into()),
            )
        })?;
        let n_rows = index.n_rows();
        let n_cols = index.n_cols();
        let total_bins = index.total_bins();
        if total_bins == 0 || n_rows == 0 {
            return Err(HessboostError::invalid_data(
                "data",
                "the wgpu backend needs a non-empty binned dataset",
            ));
        }
        if total_bins > MAX_BUFFER_ENTRIES
            || n_rows > MAX_BUFFER_ENTRIES
            || n_rows
                .checked_mul(n_cols)
                .is_none_or(|e| e > MAX_BUFFER_ENTRIES)
        {
            return Err(HessboostError::invalid_data(
                "data",
                format!(
                    "the dataset exceeds the wgpu backend's index limits \
                     ({n_rows} rows x {n_cols} features, {total_bins} bins)"
                ),
            ));
        }
        let (words, wide) = feature_major_bins(index);
        let bins = ctx.scoped("uploading the bin store", || {
            ctx.storage_init("feature-major bins", bytemuck::cast_slice(&words))
        })?;
        drop(words);
        let cuts = index.cuts();
        let mut windows = Vec::new();
        for feature in 0..n_cols {
            let (fs, fe) = cuts.feature_bins(feature);
            for w in 0..(fe - fs).div_ceil(WINDOW_BINS) {
                windows.push(Window {
                    feature: feature as u32,
                    fs: fs as u32,
                    nbins: (fe - fs) as u32,
                    base: (w * WINDOW_BINS) as u32,
                });
            }
        }
        if windows.len() > MAX_GROUPS_PER_DIM as usize {
            return Err(HessboostError::invalid_data(
                "data",
                format!(
                    "the dataset's {} (feature, bin window) items exceed the wgpu \
                     workgroup limit ({MAX_GROUPS_PER_DIM})",
                    windows.len()
                ),
            ));
        }
        let n_windows = windows.len();
        let windows = ctx.scoped("uploading the bin windows", || {
            ctx.storage_init("bin windows", bytemuck::cast_slice(&windows))
        })?;
        let partial_bytes = total_bins as u64 * 16;
        if partial_bytes > PARTIALS_BUDGET {
            return Err(HessboostError::invalid_data(
                "data",
                format!("the dataset's {total_bins} bins exceed the wgpu backend's limit"),
            ));
        }
        let slices = (PARTIALS_BUDGET / partial_bytes).min(ROW_SLICES as u64) as usize;
        // One `[i64; 2]` gradient pair in grains per row.
        let gpair = ctx.scoped("allocating the gradient buffer", || {
            ctx.storage(
                "gradient pairs",
                n_rows as u64 * 16,
                wgpu::BufferUsages::empty(),
            )
        })?;
        let mut backend = WgpuHistBackend {
            ctx,
            bins,
            wide,
            windows,
            n_windows,
            slices,
            total_bins,
            n_rows,
            n_cols,
            gradients: RwLock::new(StagedGradients {
                buffer: gpair,
                units: Vec::new(),
                addr: 0,
                len: 0,
                grad: SumDomain::EMPTY,
                hess: SumDomain::EMPTY,
            }),
            pool: Mutex::new(Vec::new()),
        };
        let first = backend.checkout()?;
        backend.checkin(first);
        backend.gradients.get_mut().units = Vec::with_capacity(n_rows);
        Ok(backend)
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
        let mut staged = self.gradients.write();
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
        // Integer multiples of each component's grain, the values the
        // kernels sum: exact whenever a node's sums can be (`exact_sum`).
        staged.units.clear();
        staged.units.par_extend(
            gpair
                .par_iter()
                .map(|p| [grad.units(p.grad), hess.units(p.hess)]),
        );
        // The write is queue-ordered after every submitted build, and the
        // write lock excludes new ones until it is staged.
        self.ctx
            .queue
            .write_buffer(&staged.buffer, 0, bytemuck::cast_slice(&staged.units));
        staged.grad = grad;
        staged.hess = hess;
        staged.addr = gpair.as_ptr().addr();
        staged.len = gpair.len();
    }

    /// A read guard on the staged gradients when they hold `gpair`, staging
    /// it first if needed; `None` when `gpair` cannot be staged or another
    /// thread staged a different slice in between.
    fn staged_for(&self, gpair: &[GradPair]) -> Option<RwLockReadGuard<'_, StagedGradients>> {
        if gpair.len() != self.n_rows {
            return None;
        }
        {
            let staged = self.gradients.read();
            if staged.holds(gpair) {
                return Some(staged);
            }
        }
        self.stage(gpair, false);
        let staged = self.gradients.read();
        staged.holds(gpair).then_some(staged)
    }

    /// Check out a per-call buffer set from the pool.
    fn checkout(&self) -> Result<CallBuffers> {
        if let Some(call) = self.pool.lock().pop() {
            return Ok(call);
        }
        let gradients = self.gradients.read();
        self.ctx.scoped("allocating histogram buffers", || {
            CallBuffers::new(self, &gradients.buffer)
        })
    }

    fn checkin(&self, call: CallBuffers) {
        self.pool.lock().push(call);
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
        // The guard lives until this function returns, after the dispatch
        // has completed: no `stage` can rewrite the gradient buffer while
        // the GPU reads it.
        let Some(staged) = self.staged_for(gpair) else {
            return false;
        };
        if !staged.sums_exact(rows.len()) {
            return false;
        }
        // The scatter kernel's 32-bit shared accumulators only stay exact
        // while a workgroup's rows fit `scatter_row_bound`.
        let rows_per_slice = rows.len().div_ceil(self.slices);
        if scatter_row_bound(&staged.grad, &staged.hess) < rows_per_slice {
            return false;
        }
        let Ok(call) = self.checkout() else {
            return false;
        };
        let result = self.dispatch(&call, &staged, rows, rows_per_slice, out);
        self.checkin(call);
        result.is_ok()
    }

    /// Encode, run, and read back the gather, scatter, and merge of `rows`
    /// over the staged `gradients`. The caller has checked that `rows` fit
    /// the backend (at most `n_rows` entries, each below `n_rows`), that
    /// their sums are exact, and that `rows_per_slice` rows fit a
    /// workgroup's accumulators, and holds the gradients' read guard
    /// throughout.
    fn dispatch(
        &self,
        call: &CallBuffers,
        gradients: &StagedGradients,
        rows: &[u32],
        rows_per_slice: usize,
        out: &mut [GradStats],
    ) -> Result<()> {
        let ctx = self.ctx;
        let n = rows.len();
        ctx.queue
            .write_buffer(&call.rows, 0, bytemuck::cast_slice(rows));
        ctx.queue.write_buffer(
            &call.gather_args,
            0,
            bytemuck::bytes_of(&GatherArgs {
                n: n as u32,
                pad: [0; 3],
            }),
        );
        ctx.queue.write_buffer(
            &call.scatter_args,
            0,
            bytemuck::bytes_of(&ScatterArgs {
                n_node_rows: n as u32,
                rows_per_slice: rows_per_slice as u32,
                n_rows: self.n_rows as u32,
                total_bins: self.total_bins as u32,
                wide: u32::from(self.wide),
                pad: [0; 3],
            }),
        );
        ctx.queue.write_buffer(
            &call.merge_args,
            0,
            bytemuck::bytes_of(&MergeArgs {
                slices: self.slices as u32,
                total_bins: self.total_bins as u32,
                pad: [0; 2],
            }),
        );
        let hist_bytes = self.total_bins as u64 * 16;
        ctx.run(|encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&ctx.gather);
            pass.set_bind_group(0, &call.gather_bind, &[]);
            let (x, y) = grid_2d(n);
            pass.dispatch_workgroups(x, y, 1);
            pass.set_pipeline(&ctx.scatter);
            pass.set_bind_group(0, &call.scatter_bind, &[]);
            pass.dispatch_workgroups(self.n_windows as u32, self.slices as u32, 1);
            pass.set_pipeline(&ctx.merge);
            pass.set_bind_group(0, &call.merge_bind, &[]);
            let (x, y) = grid_2d(self.total_bins);
            pass.dispatch_workgroups(x, y, 1);
            drop(pass);
            encoder.copy_buffer_to_buffer(&call.hist, 0, &call.readback, 0, hist_bytes);
        })?;
        let mut hist = vec![[0i64; 2]; self.total_bins];
        ctx.read_back(&call.readback, &mut hist)?;
        for (o, &[g, h]) in out.iter_mut().zip(&hist) {
            // Exact piece sums in grains (below 2^53), scaled back exactly.
            o.grad = gradients.grad.value(g);
            o.hess = gradients.hess.value(h);
        }
        Ok(())
    }
}

impl HistogramBackend for WgpuHistBackend {
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

// ---------------------------------------------------------------------------
// GPU prediction
// ---------------------------------------------------------------------------

/// A forest's uploaded buffers: the node arena (leaf values pre-weighted on
/// the host), the category pool, the (pre-weighted) leaf-vector pool, and
/// the tree records.
struct ForestBuffers {
    nodes: wgpu::Buffer,
    categories: wgpu::Buffer,
    leaf_vectors: wgpu::Buffer,
    trees: wgpu::Buffer,
}

/// Per-call prediction buffers, pooled across concurrent calls: one block
/// of rows and margins, the whole call's readback, and the bind group over
/// them.
struct PredictBuffers {
    rows: wgpu::Buffer,
    out: wgpu::Buffer,
    readback: wgpu::Buffer,
    args: wgpu::Buffer,
    bind: wgpu::BindGroup,
    /// Dense row materialization scratch, one block.
    scratch: Vec<f32>,
    block_rows: usize,
    n_cols: usize,
    k: usize,
    readback_rows: usize,
}

impl PredictBuffers {
    /// Buffers for blocks of `block_rows` rows of `n_cols` features and
    /// `k` outputs, reading back `total_rows` rows.
    fn new(
        ctx: &Context,
        forest: &ForestBuffers,
        block_rows: usize,
        n_cols: usize,
        k: usize,
        total_rows: usize,
    ) -> Result<Self> {
        let rows = ctx.storage(
            "prediction rows",
            (block_rows * n_cols) as u64 * 4,
            wgpu::BufferUsages::empty(),
        )?;
        let out = ctx.storage(
            "prediction margins",
            (block_rows * k) as u64 * 4,
            wgpu::BufferUsages::COPY_SRC,
        )?;
        let readback = ctx.staging("prediction margins", (total_rows * k) as u64 * 4)?;
        let args = ctx.uniform::<PredictArgs>("predict args");
        let bind = ctx.bind(
            &ctx.predict,
            &[
                &args,
                &forest.nodes,
                &forest.categories,
                &forest.leaf_vectors,
                &forest.trees,
                &rows,
                &out,
            ],
        );
        Ok(PredictBuffers {
            rows,
            out,
            readback,
            args,
            bind,
            scratch: vec![0.0; block_rows * n_cols],
            block_rows,
            n_cols,
            k,
            readback_rows: total_rows,
        })
    }

    /// Whether this set serves a call of this shape.
    fn fits(&self, block_rows: usize, n_cols: usize, k: usize, total_rows: usize) -> bool {
        self.block_rows >= block_rows
            && self.n_cols == n_cols
            && self.k == k
            && self.readback_rows >= total_rows
    }

    /// Upload one block's dense `rows` and initial `margins`, walk the
    /// trees of `args` over them, and copy the block's margins into the
    /// readback at row `begin`.
    fn run_block(
        &self,
        ctx: &Context,
        args: &PredictArgs,
        rows: &[f32],
        margins: &[f32],
        begin: usize,
    ) -> Result<()> {
        ctx.queue
            .write_buffer(&self.rows, 0, bytemuck::cast_slice(rows));
        ctx.queue
            .write_buffer(&self.out, 0, bytemuck::cast_slice(margins));
        ctx.queue
            .write_buffer(&self.args, 0, bytemuck::bytes_of(args));
        let bytes = margins.len() as u64 * 4;
        ctx.run(|encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&ctx.predict);
            pass.set_bind_group(0, &self.bind, &[]);
            let (x, y) = grid_2d(args.n_rows as usize);
            pass.dispatch_workgroups(x, y, 1);
            drop(pass);
            encoder.copy_buffer_to_buffer(
                &self.out,
                0,
                &self.readback,
                (begin * self.k) as u64 * 4,
                bytes,
            );
        })
    }

    /// Read the call's margins back (after its last block completed).
    fn read_margins(&self, ctx: &Context, margins: &mut [f32]) -> Result<()> {
        ctx.read_back(&self.readback, margins)
    }
}

/// A [`BoostedModel`] laid out for GPU batch prediction through wgpu.
///
/// Built with [`BoostedModel::to_wgpu`] (requires the `wgpu` feature and an
/// adapter). The compact forest, category pools, and weighted leaf values
/// are uploaded once; each prediction call uploads its rows in blocks, runs
/// one thread per row over every tree, and applies the objective's
/// transform on the CPU. Predictions are bit-identical to the CPU's: the
/// walk and the per-tree accumulation order match the CPU kernels exactly
/// (see the [module docs](crate::backend::wgpu) for the one assumption
/// about the adapter and how it is checked).
pub struct GpuModel {
    model: Arc<BoostedModel>,
    ctx: &'static Context,
    forest: ForestBuffers,
    pool: Mutex<Vec<PredictBuffers>>,
}

impl std::fmt::Debug for GpuModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The GPU handles are opaque; the source model says everything.
        f.debug_struct("GpuModel")
            .field("model", &self.model)
            .field("device", &self.ctx.device_name)
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
        data: &DMatrix,
        iterations: impl Into<Iterations>,
    ) -> Result<Predictions> {
        let model = &self.model;
        let iterations = iterations.into();
        if model.shrinkage().is_some() {
            return model.predict_margin(data, iterations);
        }
        model.validate_prediction_data(data)?;
        let trees = model.iteration_trees(model.resolve_iterations(iterations, "iterations")?);
        let k = model.n_outputs();
        let n = data.n_rows();
        let n_cols = data.n_cols();
        let mut margins = initial_margins(model.base_scores(), data);
        if trees.is_empty() || n == 0 {
            return Ok(Predictions::new(margins, n, k));
        }
        if n > MAX_BUFFER_ENTRIES || n.checked_mul(n_cols).is_none_or(|e| e > MAX_BUFFER_ENTRIES) {
            return Err(HessboostError::invalid_data(
                "data",
                format!(
                    "GPU prediction needs a dense row copy ({n} rows x {n_cols} features) \
                     that exceeds 4 GiB"
                ),
            ));
        }
        // Blocks of rows the per-block buffers hold within the adapter's
        // binding limit; the margins are block-local inside the kernel, so
        // the walk's arithmetic and order are those of a single-block call.
        let binding = self.ctx.limits.max_storage_buffer_binding_size as usize / 4;
        let block_rows = n
            .min(PREDICT_BLOCK_ROWS)
            .min(binding / n_cols.max(1))
            .min(binding / k.max(1));
        if block_rows == 0 {
            return Err(HessboostError::invalid_data(
                "data",
                format!(
                    "a row of {n_cols} features (or {k} outputs) exceeds the wgpu adapter's \
                     storage binding limit"
                ),
            ));
        }
        let mut call = self.checkout(block_rows, n_cols, k, n)?;
        let result: Result<()> = (|| {
            let mut begin = 0;
            while begin < n {
                let rows_here = (n - begin).min(block_rows);
                let scratch = &mut call.scratch[..rows_here * n_cols];
                materialize_rows(data, begin, scratch);
                let args = PredictArgs {
                    n_rows: rows_here as u32,
                    n_cols: n_cols as u32,
                    k: k as u32,
                    tree_begin: trees.start as u32,
                    tree_end: trees.end as u32,
                    soft: 0,
                    pad: [0; 2],
                };
                call.run_block(
                    self.ctx,
                    &args,
                    &call.scratch[..rows_here * n_cols],
                    &margins[begin * k..(begin + rows_here) * k],
                    begin,
                )?;
                begin += rows_here;
            }
            call.read_margins(self.ctx, &mut margins)
        })();
        self.checkin(call);
        result?;
        Ok(Predictions::new(margins, n, k))
    }

    /// Predictions in the objective's reported space from the boosting
    /// `iterations`, computed on the GPU. Bit-identical to
    /// [`BoostedModel::predict`](crate::prelude::BoostedModel::predict).
    pub fn predict(
        &self,
        data: &DMatrix,
        iterations: impl Into<Iterations>,
    ) -> Result<Predictions> {
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
        data: &DMatrix,
        iterations: impl Into<Iterations>,
    ) -> Result<Predictions<u32>> {
        Ok(self.model.classes(&self.predict(data, iterations)?))
    }

    /// Check out buffers that fit the call, allocating when the pool has
    /// none that do.
    fn checkout(
        &self,
        block_rows: usize,
        n_cols: usize,
        k: usize,
        total_rows: usize,
    ) -> Result<PredictBuffers> {
        let mut pool = self.pool.lock();
        if let Some(idx) = pool
            .iter()
            .position(|b| b.fits(block_rows, n_cols, k, total_rows))
        {
            return Ok(pool.swap_remove(idx));
        }
        drop(pool);
        self.ctx.scoped("allocating prediction buffers", || {
            PredictBuffers::new(self.ctx, &self.forest, block_rows, n_cols, k, total_rows)
        })
    }

    fn checkin(&self, call: PredictBuffers) {
        let mut pool = self.pool.lock();
        if pool.len() < PREDICT_POOL {
            pool.push(call);
        }
    }
}

impl BoostedModel {
    /// Lay this model out for GPU batch prediction through wgpu (the `wgpu`
    /// feature and an adapter are required; `gblinear` and `linear_tree`
    /// models, which do not predict through the compact forest, are
    /// refused, as is an adapter whose float additions the backend found
    /// reassociated, see the [module docs](crate::backend::wgpu)).
    ///
    /// The returned [`GpuModel`] shares this model's objective, transforms,
    /// and layout; its predictions are bit-identical to the CPU's.
    pub fn to_wgpu(&self) -> Result<GpuModel> {
        let ctx = Context::shared().ok_or_else(|| {
            HessboostError::gpu(
                unavailable_reason().unwrap_or_else(|| "no wgpu adapter is available".into()),
            )
        })?;
        if let Err(reason) = &ctx.predict_check {
            return Err(HessboostError::gpu(reason.clone()));
        }
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
        let parts = forest.gpu_parts();
        let mut nodes: Vec<[u32; 4]> = bytemuck::pod_collect_to_vec(parts.nodes);
        let mut leaf_vectors = parts.leaf_vectors.to_vec();
        let k = self.n_outputs();
        // Weight each tree's leaves on the host: `weight * leaf` in `f32`
        // is the product the CPU forms per row, and the GPU then only adds.
        for (t, &root) in parts.roots.iter().enumerate() {
            let weight = self.tree_weight(t);
            if weight == 1.0 {
                continue;
            }
            let end = parts
                .roots
                .get(t + 1)
                .map_or(nodes.len(), |&next| next as usize);
            let vector = self.tree_is_vector_leaf(t);
            for (offset, node) in nodes[root as usize..end].iter_mut().enumerate() {
                if node[2] as usize != root as usize + offset {
                    continue;
                }
                if vector {
                    let offset = node[3] as usize;
                    for w in &mut leaf_vectors[offset..offset + k] {
                        *w *= weight;
                    }
                } else {
                    node[3] = (weight * f32::from_bits(node[3])).to_bits();
                }
            }
        }
        let trees: Vec<PTree> = parts
            .roots
            .iter()
            .enumerate()
            .map(|(t, &root)| PTree {
                root,
                output: self.tree_output(t) as u32,
                vector: u32::from(self.tree_is_vector_leaf(t)),
                pad: 0,
            })
            .collect();
        let forest = ctx.scoped("uploading the forest", || {
            Ok(ForestBuffers {
                nodes: ctx.storage_init("forest nodes", bytemuck::cast_slice(&nodes))?,
                categories: ctx
                    .storage_init("forest categories", bytemuck::cast_slice(parts.categories))?,
                leaf_vectors: ctx
                    .storage_init("forest leaf vectors", bytemuck::cast_slice(&leaf_vectors))?,
                trees: ctx.storage_init("forest trees", bytemuck::cast_slice(&trees))?,
            })
        })?;
        Ok(GpuModel {
            model: Arc::new(self.clone()),
            ctx,
            forest,
            pool: Mutex::new(Vec::new()),
        })
    }
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
            eprintln!("skipping wgpu tests: {reason}");
        }
        Context::shared().is_some()
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
        backend: &WgpuHistBackend,
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
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
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
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let index = GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, 512));
        let rows: Vec<u32> = (0..n as u32).collect();
        let backend = WgpuHistBackend::new(&index).unwrap();
        assert_eq!(
            gpu_hist(&backend, &index, &rows, &gpair),
            cpu_hist(&index, &rows, &gpair)
        );
    }

    /// Near the exactness bound the GPU still takes the node and keeps every
    /// low bit: each bin sums to about `2^52`, with odd small gradients mixed
    /// in, so it needs all 53 bits of an `f64`. The scatter kernel's 32-bit
    /// piece counters sit a hair inside the `2^53` domain (with 64 slices,
    /// `n * max <= (2^31 - 1) * 2^22`), so the test uses the largest
    /// magnitude 65,536 rows admit, `2^37 - 2^16`; one more row and the node
    /// goes to the CPU, which still gives the same histogram.
    #[test]
    fn hist_is_exact_near_the_domain_edge() {
        if !context() {
            return;
        }
        let grad = |i: usize| match i {
            i if i % 11 == 0 => 1975.0,
            _ => 2f32.powi(37) - 2f32.powi(16),
        };
        let n = 1 << 16;
        let gpair: Vec<GradPair> = (0..n).map(|i| GradPair::new(grad(i), 1.0)).collect();
        let index = one_feature(n, 2);
        let backend = WgpuHistBackend::new(&index).unwrap();
        let rows: Vec<u32> = (0..n as u32).collect();
        let gpu = gpu_hist(&backend, &index, &rows, &gpair);
        assert_eq!(gpu, cpu_hist(&index, &rows, &gpair));
        assert!(gpu.iter().all(|s| s.grad > 2f64.powi(51)));
        let gpair: Vec<GradPair> = (0..=n).map(|i| GradPair::new(grad(i), 1.0)).collect();
        let index = one_feature(n + 1, 2);
        let backend = WgpuHistBackend::new(&index).unwrap();
        let rows: Vec<u32> = (0..=n as u32).collect();
        backend.prepare(&index, &gpair);
        let mut out = vec![GradStats::default(); index.total_bins()];
        assert!(!backend.try_gpu(&index, &rows, &gpair, &mut out));
        HistogramBackend::build(&backend, &index, &rows, &gpair, &mut out);
        assert_eq!(out, cpu_hist(&index, &rows, &gpair));
    }

    /// Outside the exactness bound the backend runs the CPU path: six
    /// gradients up to `2^50` at grain 1 allow at most 8 rows per node. The
    /// CPU sums the bins to +64 and -64, which a float kernel would not.
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
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let index = GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, 256));
        let backend = WgpuHistBackend::new(&index).unwrap();
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
        let backend = WgpuHistBackend::new(&index).unwrap();
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
    /// on dense data, with every slice busy and on sub-sampled row lists.
    #[test]
    fn hist_matches_cpu_dense() {
        if !context() {
            return;
        }
        let n = 2 * 65_536 + 123;
        let cols = 30;
        let x: Vec<f32> = (0..n * cols)
            .map(|i: usize| ((i.wrapping_mul(2_654_435_761)) % 1000) as f32 * 0.001)
            .collect();
        let data = DMatrix::from_dense(&x, n, cols).unwrap();
        let index = GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, 64));
        let gpair: Vec<GradPair> = (0..n)
            .map(|i| GradPair {
                grad: ((i as i32 % 11) as f32 - 5.0).powi(3) * 0.01,
                hess: ((i % 3) as f32 + 1.0).powi(2),
            })
            .collect();
        let all: Vec<u32> = (0..n as u32).collect();
        let sampled: Vec<u32> = all.iter().copied().step_by(3).collect();
        let backend = WgpuHistBackend::new(&index).unwrap();
        for rows in [&all, &sampled] {
            assert_eq!(
                gpu_hist(&backend, &index, rows, &gpair),
                cpu_hist(&index, rows, &gpair)
            );
        }
    }

    /// A slice whose grain counts are too coarse for the scatter kernel's
    /// 32-bit shared accumulators (a huge magnitude next to a value with a
    /// fine grain) is inside the exactness domain but runs on the CPU, and
    /// still matches it.
    #[test]
    fn coarse_grains_run_on_the_cpu() {
        if !context() {
            return;
        }
        // 8,192 rows of at most 2^40 grains: inside `n * max <= 2^53`, past
        // the scatter bound (the high piece alone needs more than an `i32`
        // per workgroup).
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
        let backend = WgpuHistBackend::new(&index).unwrap();
        let rows: Vec<u32> = (0..n as u32).collect();
        backend.prepare(&index, &gpair);
        let staged = backend.gradients.read();
        assert!(staged.sums_exact(rows.len()));
        assert!(scatter_row_bound(&staged.grad, &staged.hess) < n.div_ceil(backend.slices));
        drop(staged);
        let mut out = vec![GradStats::default(); index.total_bins()];
        assert!(!backend.try_gpu(&index, &rows, &gpair, &mut out));
        HistogramBackend::build(&backend, &index, &rows, &gpair, &mut out);
        assert_eq!(out, cpu_hist(&index, &rows, &gpair));
    }

    /// More than 65,536 bins in total: the store is 32-bit and the kernel
    /// reads `u32` bins, matching the CPU just the same.
    #[test]
    fn hist_matches_cpu_wide_bins() {
        if !context() {
            return;
        }
        let (n, cols) = (12_000, 300);
        let x: Vec<f32> = (0..n * cols)
            .map(|i: usize| ((i.wrapping_mul(2_654_435_761)) % 100_003) as f32 * 0.001)
            .collect();
        let data = DMatrix::from_dense(&x, n, cols).unwrap();
        let index = GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, 256));
        assert!(index.total_bins() > u16::MAX as usize + 1);
        let gpair: Vec<GradPair> = (0..n)
            .map(|i| GradPair {
                grad: ((i as i32 % 11) as f32 - 5.0).powi(3) * 0.01,
                hess: ((i % 3) as f32 + 1.0).powi(2),
            })
            .collect();
        let all: Vec<u32> = (0..n as u32).collect();
        let sampled: Vec<u32> = all.iter().copied().step_by(7).collect();
        let backend = WgpuHistBackend::new(&index).unwrap();
        assert!(backend.wide, "the wide store must be the one read");
        for rows in [&all, &sampled] {
            assert_eq!(
                gpu_hist(&backend, &index, rows, &gpair),
                cpu_hist(&index, rows, &gpair)
            );
        }
    }

    /// Sparse (missing-value) data: an index at least half full reads its
    /// own sentinel store, a sparser one the store built from its rows;
    /// both match the CPU exactly.
    #[test]
    fn hist_matches_cpu_sparse() {
        if !context() {
            return;
        }
        for missing_every in [5usize, 10] {
            // `missing_every = 5` keeps 80% of the entries (the index keeps
            // a sentinel store); `10` here drops 90% (it does not).
            let n = 20_000;
            let cols = 5;
            let x: Vec<f32> = (0..n * cols)
                .map(|i: usize| {
                    let drop = if missing_every == 5 {
                        (i * 7).is_multiple_of(5)
                    } else {
                        !(i * 7).is_multiple_of(10)
                    };
                    if drop {
                        f32::NAN
                    } else {
                        ((i.wrapping_mul(40_503)) % 97) as f32
                    }
                })
                .collect();
            let data = DMatrix::from_dense_with_missing(&x, n, cols, f32::NAN).unwrap();
            let index = GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, 32));
            assert_eq!(
                index.missing_columns().is_some(),
                missing_every == 5,
                "the test must cover both stores"
            );
            let gpair: Vec<GradPair> = (0..n)
                .map(|i| GradPair {
                    grad: ((i as i32 % 13) as f32 - 6.0) * 0.05,
                    hess: 1.0 + (i % 2) as f32,
                })
                .collect();
            let rows: Vec<u32> = (0..n as u32).collect();
            let backend = WgpuHistBackend::new(&index).unwrap();
            assert_eq!(
                gpu_hist(&backend, &index, &rows, &gpair),
                cpu_hist(&index, &rows, &gpair)
            );
        }
    }

    /// Row counts that leave the last slice short or empty contribute
    /// nothing from the idle lanes.
    #[test]
    fn hist_matches_cpu_ragged_slices() {
        if !context() {
            return;
        }
        for &n in &[CPU_ROWS + 1, 65_536 + 4, 2 * 65_536 + 123] {
            let cols = 5;
            let x: Vec<f32> = (0..n * cols)
                .map(|i| ((i.wrapping_mul(2_654_435_761)) % 997) as f32 * 0.001)
                .collect();
            let data = DMatrix::from_dense(&x, n, cols).unwrap();
            let index = GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, 64));
            let gpair: Vec<GradPair> = (0..n)
                .map(|i| GradPair::new((i % 5) as f32 - 2.0, 1.0 + (i % 3) as f32 * 0.5))
                .collect();
            let rows: Vec<u32> = (0..n as u32).collect();
            let backend = WgpuHistBackend::new(&index).unwrap();
            assert_eq!(
                gpu_hist(&backend, &index, &rows, &gpair),
                cpu_hist(&index, &rows, &gpair),
                "at {n} rows"
            );
        }
    }

    /// A deterministic dataset with missing values and a categorical first
    /// column, `n` rows by `cols` features.
    fn dataset(n: usize, cols: usize) -> DMatrix {
        use crate::data::FeatureType;
        let mut x = vec![0.0f32; n * cols];
        let mut y = vec![0.0f32; n];
        for r in 0..n {
            let mut target = 0.0;
            for f in 0..cols {
                let v = if f == 0 {
                    ((r * 31 + f) % 5) as f32
                } else if (r + f) % 11 == 0 {
                    f32::NAN
                } else {
                    (((r * 97 + f * 13) % 100) as f32) * 0.01
                };
                x[r * cols + f] = v;
                if f > 0 && v.is_finite() {
                    target += v * f as f32;
                }
            }
            y[r] = target % 2.0;
        }
        let types: Vec<FeatureType> = (0..cols)
            .map(|f| {
                if f == 0 {
                    FeatureType::Categorical
                } else {
                    FeatureType::Numerical
                }
            })
            .collect();
        DMatrix::from_dense_with_missing(&x, n, cols, f32::NAN)
            .unwrap()
            .with_feature_types(&types)
            .unwrap()
            .with_labels(&y)
            .unwrap()
    }

    /// A model trained with `device = wgpu` (DART weights, categorical
    /// splits, missing values) predicts identically through `to_wgpu`.
    #[test]
    fn gpu_predicts_like_cpu() {
        use crate::config::{BoosterKind, Dart, Device, TreeMethod};
        use crate::prelude::*;
        if !context() {
            return;
        }
        let data = dataset(3_000, 6);
        let params = TrainingParams::builder()
            .objective(Objective::SquaredError(RegLoss::default()))
            .tree_method(TreeMethod::Hist)
            .max_depth(5)
            .eta(0.3)
            .booster(BoosterKind::Dart(Dart::default()))
            .device(Device::Wgpu)
            .build()
            .unwrap();
        let model = train(&params, &data, 12).unwrap();
        let gpu = model.to_wgpu().unwrap();
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

    /// Multi-output models through both leaf layouts: one scalar tree per
    /// output (`multi:softprob`) and vector leaves (`multi_output_tree`),
    /// with DART weights so the host-side leaf weighting is exercised.
    #[test]
    fn gpu_predicts_multi_output_like_cpu() {
        use crate::config::{BoosterKind, Dart, MultiStrategy, TreeMethod};
        use crate::objective::Multiclass;
        use crate::prelude::*;
        if !context() {
            return;
        }
        let data = dataset(2_500, 5);
        let labels: Vec<f32> = data
            .labels()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, &y)| ((y * 7.0) as usize % 3 + i % 2) as f32 % 3.0)
            .collect();
        let data = data.with_labels(&labels).unwrap();
        for strategy in [
            MultiStrategy::OneOutputPerTree,
            MultiStrategy::MultiOutputTree,
        ] {
            let params = TrainingParams::builder()
                .objective(Objective::Softprob(Multiclass::new(3).unwrap()))
                .tree_method(TreeMethod::Hist)
                .multi_strategy(strategy)
                .max_depth(4)
                .eta(0.4)
                .booster(BoosterKind::Dart(Dart::default()))
                .build()
                .unwrap();
            let model = train(&params, &data, 6).unwrap();
            let gpu = model.to_wgpu().unwrap();
            assert_eq!(
                model.predict_margin(&data, Iterations::Best).unwrap(),
                gpu.predict_margin(&data, Iterations::Best).unwrap(),
                "{strategy:?}"
            );
            assert_eq!(
                model.predict(&data, ..2).unwrap(),
                gpu.predict(&data, ..2).unwrap(),
                "{strategy:?}: iteration range"
            );
        }
    }

    /// A batch larger than one prediction block lands bit-identical block
    /// by block.
    #[test]
    fn gpu_predicts_across_blocks() {
        use crate::config::TreeMethod;
        use crate::prelude::*;
        if !context() {
            return;
        }
        let train_data = dataset(3_000, 4);
        let params = TrainingParams::builder()
            .objective(Objective::SquaredError(RegLoss::default()))
            .tree_method(TreeMethod::Hist)
            .max_depth(4)
            .build()
            .unwrap();
        let model = train(&params, &train_data, 5).unwrap();
        let gpu = model.to_wgpu().unwrap();
        let batch = dataset(PREDICT_BLOCK_ROWS + 1_001, 4);
        assert_eq!(
            model.predict(&batch, Iterations::Best).unwrap(),
            gpu.predict(&batch, Iterations::Best).unwrap()
        );
    }

    /// The 2-D grid covers every item count below the dimension limit and
    /// beyond it.
    #[test]
    fn grid_covers_the_items() {
        for items in [1usize, 63, 64, 65, 65_535 * 64, 65_535 * 64 + 1, 1 << 30] {
            let (x, y) = grid_2d(items);
            assert!(x <= MAX_GROUPS_PER_DIM && y <= MAX_GROUPS_PER_DIM);
            assert!(
                (x as usize) * LINEAR_THREADS as usize * (y as usize) >= items,
                "{items}"
            );
        }
    }

    /// Run `chain` on the GPU and require the CPU's bits, row by row (two
    /// NaNs count as equal: their payloads are platform-defined on the CPU
    /// too).
    fn assert_chain_matches(chain: &AdditionChain, what: &str) {
        let ctx = Context::shared().unwrap();
        let got = ctx.run_chain(chain).unwrap();
        let expected = chain.cpu();
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!(
                g.to_bits() == e.to_bits() || (g.is_nan() && e.is_nan()),
                "{what}: slot {i}: GPU {g:e} ({:#010x}) != CPU {e:e} ({:#010x}); margin \
                 {:#010x}",
                g.to_bits(),
                e.to_bits(),
                chain.margins[i].to_bits()
            );
        }
    }

    /// Operand pairs that a native float add gets wrong on a flush-to-zero
    /// adapter or that stress the integer path's rounding: subnormal
    /// operands and results, signed zeros, ties, sticky bits, carries,
    /// cancellation, and overflow.
    fn addition_edge_cases() -> Vec<(f32, f32)> {
        let b = f32::from_bits;
        let min_normal = b(0x0080_0000);
        let min_sub = b(0x0000_0001);
        vec![
            // Two normals whose exact sum is the smallest subnormal.
            (b(0x0100_0000), -b(0x00ff_ffff)),
            (-b(0x00ff_ffff), b(0x0100_0000)),
            (min_sub, min_sub),
            (min_sub, -min_sub),
            (b(0x007f_ffff), min_sub),
            (min_normal, -min_sub),
            (min_normal, min_sub),
            (b(0x0040_0000), b(0x0040_0000)),
            (1.0, min_sub),
            (-1.0, b(0x0012_3456)),
            (b(0x0012_3456), b(0x0065_4321)),
            (b(0x0012_3456), -b(0x0065_4321)),
            (0.0, -0.0),
            (-0.0, 0.0),
            (-0.0, -0.0),
            (0.0, 0.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (min_sub, 0.0),
            (-0.0, min_sub),
            (1.0, 2f32.powi(-24)),
            (b(0x3f80_0001), 2f32.powi(-24)),
            (1.0, 3.0 * 2f32.powi(-24)),
            (1.0, 2f32.powi(-25)),
            (1.0, -2f32.powi(-25)),
            (1.0, -(2f32.powi(-24) + 2f32.powi(-30))),
            (1.0, 1e-30),
            (1e30, 1.0),
            (1.0, -b(0x3f7f_ffff)),
            (b(0x3f7f_ffff), -1.0),
            (3.0, 5.0),
            (-2.5, 1.25),
            (f32::MAX, f32::MAX),
            (f32::MAX, 2f32.powi(103)),
            (f32::MAX, 2f32.powi(102)),
            (f32::MAX, -f32::MAX),
            (-f32::MAX, -f32::MAX),
            (f32::INFINITY, 1.0),
            (-f32::INFINITY, f32::MAX),
            (f32::INFINITY, f32::INFINITY),
            (f32::INFINITY, -f32::INFINITY),
            (f32::INFINITY, f32::NAN),
            (f32::NAN, 1.0),
            (b(0x3f80_0000), b(0x3f80_0000)),
            (b(0x7f7f_ffff), b(0x3380_0000)),
            (b(0x4000_0000), b(0x3fff_ffff)),
            (b(0x4000_0000), -b(0x3fff_ffff)),
        ]
    }

    /// A deterministic finite `f32` from a SplitMix64 step, over every
    /// exponent (including subnormals and zeros), both signs.
    fn random_finite(state: &mut u64) -> f32 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let bits = z as u32;
        // Keep the exponent out of the NaN/inf range; bias half the draws
        // toward the bottom of the range so subnormals and near-subnormal
        // sums are common.
        let exponent = if z >> 63 == 0 {
            (bits >> 23) & 0xFF
        } else {
            (bits >> 23) & 0x07
        };
        f32::from_bits((bits & 0x807F_FFFF) | (exponent.min(0xFE) << 23))
    }

    /// The kernel's additions equal the CPU's bit for bit on every edge case
    /// of `f32` addition and on random operands, through the integer path
    /// (`soft`) and the adapter's own add behind the normal-range check,
    /// for each accumulation path (one output, scalar outputs, vector
    /// leaves).
    #[test]
    fn additions_match_the_cpu_bit_for_bit() {
        if !context() {
            return;
        }
        // Every edge pair as (margin, leaf) and as (leaf, margin): one tree,
        // one row per pair.
        let pairs = addition_edge_cases();
        for soft in [true, false] {
            for &(m, l) in &pairs {
                for (margin, leaf) in [(m, l), (l, m)] {
                    assert_chain_matches(
                        &AdditionChain {
                            k: 1,
                            vector: false,
                            leaves: vec![leaf],
                            margins: vec![margin],
                            soft,
                        },
                        &format!("{margin:e} + {leaf:e} (soft {soft})"),
                    );
                }
            }
        }
        // Random chains: 32 leaves over 8,192 rows of random margins, so the
        // running sums meet every alignment and cancellation.
        let mut state = 0x5EED_u64;
        for (k, vector) in [(1, false), (3, false), (2, true)] {
            for soft in [true, false] {
                let leaves: Vec<f32> = (0..32 * if vector { k } else { 1 })
                    .map(|_| random_finite(&mut state))
                    .collect();
                let margins: Vec<f32> = (0..8_192 * k).map(|_| random_finite(&mut state)).collect();
                assert_chain_matches(
                    &AdditionChain {
                        k,
                        vector,
                        leaves,
                        margins,
                        soft,
                    },
                    &format!("random chain (k {k}, vector {vector}, soft {soft})"),
                );
            }
        }
    }

    /// A model whose leaves and base margins are subnormal predicts
    /// bit-identically through the real tree walk: every addition routes
    /// through the integer path.
    #[test]
    fn gpu_predicts_subnormal_margins_like_cpu() {
        use crate::config::TreeMethod;
        use crate::prelude::*;
        if !context() {
            return;
        }
        let n = 3_000;
        let cols = 4;
        let x: Vec<f32> = (0..n * cols)
            .map(|i: usize| ((i.wrapping_mul(2_654_435_761)) % 1000) as f32 * 0.001)
            .collect();
        // Labels around the smallest normal, so the leaves (label means)
        // and most running margins are subnormal or straddle the boundary.
        let min_normal = f32::from_bits(0x0080_0000);
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let scale = ((i * 7919) % 1000) as f32 * 0.003 - 1.0;
                min_normal * scale
            })
            .collect();
        let data = DMatrix::from_dense(&x, n, cols)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective(Objective::SquaredError(RegLoss::default()))
            .tree_method(TreeMethod::Hist)
            .base_score(0.0)
            .max_depth(4)
            .eta(1.0)
            .build()
            .unwrap();
        let model = train(&params, &data, 6).unwrap();
        let subnormal_leaves = model
            .trees()
            .iter()
            .flat_map(|t| {
                t.nodes()
                    .iter()
                    .filter(|n| n.is_leaf())
                    .map(|n| n.leaf_value)
            })
            .filter(|v| *v != 0.0 && v.abs() < min_normal)
            .count();
        assert!(subnormal_leaves > 0, "the test needs subnormal leaves");
        let gpu = model.to_wgpu().unwrap();
        let base_margin: Vec<f32> = (0..n)
            .map(|i| {
                f32::from_bits(((i * 48_271) % 0x00FF_FFFF) as u32)
                    * if i % 2 == 0 { 1.0 } else { -1.0 }
            })
            .collect();
        let with_margin = data.clone().with_base_margin(&base_margin).unwrap();
        for d in [&data, &with_margin] {
            assert_eq!(
                model.predict_margin(d, Iterations::Best).unwrap(),
                gpu.predict_margin(d, Iterations::Best).unwrap()
            );
        }
    }
}
