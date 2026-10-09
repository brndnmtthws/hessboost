//! Histogram tree growth: binning, gradients, histograms, partitions,
//! reductions and the numeric split search.

use crate::{
    Bin, F32x2, F64x2, FULL, I64x2, SCAN_WARPS, U32x2, U32x4, grid_index, grid_threads, ld,
    scatter, st,
};
use core::cmp::Ordering;
use cuda_device::atomic::{
    AtomicOrdering::Relaxed, BlockAtomicU32, DeviceAtomicU32, DeviceAtomicU64,
};
use cuda_device::{
    DisjointSlice, DynamicSharedArray, SharedArray, kernel, launch_bounds, ptx_asm, thread, warp,
};

// ---------------------------------------------------------------------------
// Binning

/// The global CPU bins are the cut authority: each dense cell's global bin,
/// made feature-local, written in both layouts (row-major `rows` with
/// `stride` entries per row, feature-major `cols`), with 64-bit cell
/// offsets and no host transpose. Blocks are 32 x 8 threads, one 32 x 32
/// (rows x features) tile per block per pass.
///
/// # Safety
///
/// Every thread of the block calls this (it holds barriers), and the block
/// is exactly 32 x 8 x 1 threads, so the tile cells `(y + j) * 33 + x` and
/// `x * 33 + y + j` stay below `32 * 33`. `global` holds `n_rows * n_cols`
/// bins and `first` `n_cols` entries; `rows` holds `n_rows * stride`
/// entries with `n_cols <= stride`, and `cols` `n_rows * n_cols`; neither
/// overlaps the inputs or the other.
#[inline(always)]
unsafe fn encode_dense<G: Bin, L: Bin>(
    global: *const G,
    first: *const u32,
    n_rows: u64,
    n_cols: u32,
    stride: u32,
    rows: *mut L,
    cols: *mut L,
) {
    // `u32` cells keep 32 banks distinct for all output widths; the extra
    // column removes the transpose's bank conflicts. Both writes coalesce.
    static mut TILE: SharedArray<u32, { 32 * 33 }> = SharedArray::UNINIT;
    // SAFETY: `TILE` is this kernel's `static mut SharedArray`; each access
    // below states why it is disjoint and ordered by the barriers.
    let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
    let (x, y) = (
        u64::from(thread::threadIdx_x()),
        u64::from(thread::threadIdx_y()),
    );
    let n_cols = u64::from(n_cols);
    let feature_tiles = n_cols.div_ceil(32);
    let row_tiles = n_rows.div_ceil(32);
    // The bound depends on the block only: every thread of the block takes
    // the same passes and reaches both barriers.
    let mut t = u64::from(thread::blockIdx_x());
    while t < feature_tiles * row_tiles {
        let row_base = (t / feature_tiles) * 32;
        let feature_base = (t % feature_tiles) * 32;
        let f = feature_base + x;
        let mut j = 0;
        while j < 32 {
            let r = row_base + y + j;
            if r < n_rows && f < n_cols {
                // SAFETY: `r < n_rows` and `f < n_cols` (checked above), so
                // the cell is in `global`, the entry in `first`, and the
                // cell `r * stride + f` in `rows` (`f < n_cols <= stride`),
                // the caller's sizes; blocks take distinct tiles, so this
                // thread alone writes that `rows` cell. Tile cell `(y + j) *
                // 33 + x` (`x < 32`, `y + j < 32`) is this thread's alone
                // until the barrier below.
                unsafe {
                    let local = ld(global, r * n_cols + f).get() - ld(first, f);
                    st(tile, (y + j) * 33 + x, local);
                    st(rows, r * u64::from(stride) + f, L::narrow(local));
                }
            }
            j += 8;
        }
        thread::sync_threads();
        let r = row_base + x;
        let mut j = 0;
        while j < 32 {
            let out_f = feature_base + y + j;
            if r < n_rows && out_f < n_cols {
                // SAFETY: `r < n_rows` and `out_f < n_cols` (checked above),
                // so the cell is in `cols` (the caller's size), written by
                // this thread alone (distinct tiles). Tile cell `x * 33 + y +
                // j` is (row `x`, feature `y + j`) of the tile, which its
                // thread wrote before the barrier above under this same
                // condition, and no thread rewrites it before the barrier
                // below.
                unsafe {
                    st(
                        cols,
                        out_f * n_rows + r,
                        L::narrow(ld(tile, x * 33 + y + j)),
                    );
                }
            }
            j += 8;
        }
        thread::sync_threads();
        t += u64::from(thread::gridDim_x());
    }
}

/// One encode entry: `$name` reads `$global` bins and writes `$local` ones.
macro_rules! encode {
    ($name:ident, $global:ty, $local:ty) => {
        /// [`encode_dense`] for these bin widths.
        ///
        /// # Safety
        ///
        /// Blocks of exactly 32 x 8 x 1 threads in a one-dimensional grid of
        /// any size. `global` holds `n_rows * n_cols` bins and `first`
        /// `n_cols` entries; `rows` holds `n_rows * stride` entries with
        /// `n_cols <= stride`, and `cols` `n_rows * n_cols`; neither overlaps
        /// the inputs or the other.
        #[kernel]
        pub unsafe fn $name(
            global: *const $global,
            first: *const u32,
            n_rows: u64,
            n_cols: u32,
            stride: u32,
            rows: *mut $local,
            cols: *mut $local,
        ) {
            // SAFETY: every thread of the block runs the kernel, and this
            // kernel's contract is `encode_dense`'s.
            unsafe { encode_dense(global, first, n_rows, n_cols, stride, rows, cols) }
        }
    };
}

encode!(encode_u8_u8, u8, u8);
encode!(encode_u8_u16, u16, u8);
encode!(encode_u8_u32, u32, u8);
encode!(encode_u16_u8, u8, u16);
encode!(encode_u16_u16, u16, u16);
encode!(encode_u16_u32, u32, u16);
encode!(encode_u32_u8, u8, u32);
encode!(encode_u32_u16, u16, u32);
encode!(encode_u32_u32, u32, u32);

// ---------------------------------------------------------------------------
// Gradients and rows

/// Each gradient pair as integer multiples of its component's grain:
/// `f64::from(x) * 2^-grain`, an exact power-of-two scaling, truncated to
/// `i64` (exact for every slice the exact paths read). One thread per pair.
#[kernel]
pub fn stage_units(
    gpair: &[F32x2],
    mut units: DisjointSlice<I64x2>,
    grad_scale: f64,
    hess_scale: f64,
) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(unit) = units.get_mut(idx) {
        let p = gpair[i];
        *unit = I64x2 {
            x: (f64::from(p.x) * grad_scale) as i64,
            y: (f64::from(p.y) * hess_scale) as i64,
        };
    }
}

/// `rows[i] = first + i`: an unsampled tree's rows without an upload. One
/// thread per row.
#[kernel]
pub fn iota_rows(mut rows: DisjointSlice<u32>, first: u32) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(row) = rows.get_mut(idx) {
        *row = first + i as u32;
    }
}

/// `reg:squarederror`'s gradients from the device margins, the CPU's
/// operations in `f32`: `w` (the row weight; 1 when `weights` is empty),
/// times `scale_pos_weight` for a label of exactly 1, then `((p - y) * w,
/// w)`. One thread per row of `gpair`.
#[kernel]
pub fn squared_error(
    margins: &[f32],
    labels: &[f32],
    weights: &[f32],
    scale_pos_weight: f32,
    mut gpair: DisjointSlice<F32x2>,
) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(pair) = gpair.get_mut(idx) {
        let (p, y) = (margins[i], labels[i]);
        let mut w = if weights.is_empty() { 1.0 } else { weights[i] };
        if y == 1.0 {
            w *= scale_pos_weight;
        }
        *pair = F32x2 {
            x: (p - y) * w,
            y: w,
        };
    }
}

/// `a * b + c` rounded once: `fma.rn.f32`, the host's vector FMA. Inline
/// PTX because cuda-oxide lowers `f32::mul_add` to libdevice's `__nv_fmaf`,
/// which would tie the build to a CUDA toolkit's libdevice.
#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    let out: f32;
    // SAFETY: one register-only arithmetic instruction.
    unsafe {
        ptx_asm!(
            "fma.rn.f32 %0, %1, %2, %3;",
            out("=f") out,
            in("f") a,
            in("f") b,
            in("f") c,
            options(register_only),
        );
    }
    out
}

/// The host vector kernels' exponential (`simd/x86_64.rs` `exp_f32`,
/// `simd/aarch64.rs` `expq_f32`, identical operations) for
/// `|v| <= 80`: range reduction by a split ln 2, Estrin's seventh-order
/// polynomial, the `2^e` scaling. [`fma`] where the host fuses; every other
/// operation is a separate IEEE one.
#[inline(always)]
fn exp_vector(v: f32) -> f32 {
    let scaled = v * f32::from_bits(0x3fb8_aa3b); // log2(e)
    let e = scaled.round_ties_even() as i32;
    let ef = e as f32;
    let r = fma(-ef, f32::from_bits(0x3f31_8000), v);
    let r = fma(ef, f32::from_bits(0x395e_8083), r);
    let sq = r * r;
    let fo = sq * sq;
    let p0 = 1.0 + r;
    let p1 = fma(f32::from_bits(0x3e2a_aaab), r, 0.5); // 1/6
    let p2 = fma(f32::from_bits(0x3c08_8889), r, f32::from_bits(0x3d2a_aaab)); // 1/120, 1/24
    let p3 = fma(f32::from_bits(0x3950_0d01), r, f32::from_bits(0x3ab6_0b61)); // 1/5040, 1/720
    let low = fma(p1, sq, p0);
    let high = fma(p3, sq, p2);
    let poly = fma(high, fo, low);
    poly * f32::from_bits(((e + 127) << 23) as u32)
}

/// [`logistic`]'s objective: the positive-label weight scale, the Hessian
/// floor, the largest margin magnitude the vector path takes, and the
/// host's vector width.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LogisticParams {
    pub scale_pos_weight: f32,
    pub min_hess: f32,
    pub max_input: f32,
    pub lanes: u32,
}

/// The logistic objectives' gradients of `gpair`'s rows (a whole number of
/// vectors), the host's vector kernel (`simd::logistic_gradient`): `lanes`
/// rows per vector; a vector holding a margin above `max_input` in
/// magnitude (or a NaN) takes the host's scalar path, which uses the C
/// library's `expf`, so its rows get NaN gradients here and the tree grows
/// on the host. `weights` is empty when rows are unweighted. One thread per
/// row, reading its vector's margins.
#[kernel]
pub fn logistic(
    margins: &[f32],
    labels: &[f32],
    weights: &[f32],
    params: LogisticParams,
    mut gpair: DisjointSlice<F32x2>,
) {
    let LogisticParams {
        scale_pos_weight,
        min_hess,
        max_input,
        lanes,
    } = params;
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(pair) = gpair.get_mut(idx) {
        let lanes = lanes as usize;
        let first = i - i % lanes;
        let mut regular = true;
        let mut k = 0;
        while k < lanes {
            regular = regular && margins[first + k].abs() <= max_input;
            k += 1;
        }
        *pair = if regular {
            let (x, y) = (margins[i], labels[i]);
            let ex = exp_vector(-x.abs());
            let den = 1.0 + ex;
            let p = if x >= 0.0 { 1.0 / den } else { ex / den };
            let w = if weights.is_empty() { 1.0 } else { weights[i] };
            let w = w * if y == 1.0 { scale_pos_weight } else { 1.0 };
            let g = (p - y) * w;
            let h = (p * (1.0 - p)).max(min_hess) * w;
            F32x2 { x: g, y: h }
        } else {
            let nan = f32::from_bits(0x7fc0_0000);
            F32x2 { x: nan, y: nan }
        };
    }
}

/// One component's exactness statistics (the host's `SumDomain::of`): the
/// largest magnitude's bits (non-negative floats order as their bits), the
/// smallest grain exponent plus 150 (so positive), and whether every value
/// is finite.
#[derive(Clone, Copy)]
struct Domain {
    max_bits: u32,
    grain: u32,
    finite: u32,
}

impl Domain {
    const EMPTY: Self = Self {
        max_bits: 0,
        grain: u32::MAX,
        finite: 1,
    };

    /// Fold in one value; zeros contribute nothing.
    #[inline(always)]
    fn fold(&mut self, v: f32) {
        let bits = v.to_bits();
        let e = (bits >> 23) & 0xff;
        let m = bits & 0x7f_ffff;
        if e == 0xff {
            self.finite = 0;
        } else if e != 0 || m != 0 {
            self.max_bits = self.max_bits.max(bits & 0x7fff_ffff);
            // The grain exponent plus 150 of a value whose lowest set
            // significand bit is bit `k`: `1 + k` for a subnormal, else
            // `e + k` (the implicit bit counts for a power of two).
            let code = if e == 0 {
                1 + m.trailing_zeros()
            } else {
                e + (m | 0x80_0000).trailing_zeros()
            };
            self.grain = self.grain.min(code);
        }
    }

    /// Combine with the statistics `delta` lanes down the warp.
    #[inline(always)]
    fn shuffle_down(&mut self, delta: u32) {
        self.max_bits = self
            .max_bits
            .max(warp::shuffle_down_sync(FULL, self.max_bits, delta));
        self.grain = self
            .grain
            .min(warp::shuffle_down_sync(FULL, self.grain, delta));
        self.finite &= warp::shuffle_down_sync(FULL, self.finite, delta);
    }

    /// Combine into `out`, the global `[max bits, grain + 150, finite]`.
    #[inline(always)]
    fn publish(self, out: &[DeviceAtomicU32]) {
        out[0].fetch_max(self.max_bits, Relaxed);
        out[1].fetch_min(self.grain, Relaxed);
        out[2].fetch_and(self.finite, Relaxed);
    }
}

/// Both components' statistics of the pairs of `gpair` into `domain`
/// (`[max bits, grain + 150, finite]` per component; initialized to `[0,
/// u32::MAX, 1]`). Each thread folds its grid-stride share, each warp
/// combines its lanes' by shuffles (blocks hold whole warps, and every lane
/// reaches them), and lane 0 publishes the warp's: one set of atomics per
/// warp of the capped grid, not per pair.
#[kernel]
pub fn grad_domain(gpair: &[F32x2], domain: &[DeviceAtomicU32]) {
    let (mut grad, mut hess) = (Domain::EMPTY, Domain::EMPTY);
    let mut i = grid_index();
    while i < gpair.len() as u64 {
        let p = gpair[i as usize];
        grad.fold(p.x);
        hess.fold(p.y);
        i += grid_threads();
    }
    let mut delta = 16;
    while delta > 0 {
        grad.shuffle_down(delta);
        hess.shuffle_down(delta);
        delta >>= 1;
    }
    if thread::threadIdx_x().is_multiple_of(32) {
        grad.publish(&domain[..3]);
        hess.publish(&domain[3..]);
    }
}

/// Rows per partition and leaf-update tile (`PART_TILE` in
/// `src/backend/cuda/mod.rs`).
const PART_TILE: u64 = 4096;

/// A tile of a segment list: `code = segment << 32 | tile`, rows `[tile *
/// PART_TILE, ...)` of the segment `(offset, len)` in `segs`. Returns the
/// segment, its offset, and the tile's row range `[begin, end)` within it
/// (`end <= len`; empty for a tile past the segment's end).
///
/// # Safety
///
/// `ptiles` holds an entry at `blockIdx.x`, and `segs` the two words `2 s`
/// and `2 s + 1` of the segment `s` that code names.
#[inline(always)]
unsafe fn part_tile(segs: *const u64, ptiles: *const u64) -> (u32, u64, u64, u64) {
    // SAFETY: `ptiles` holds the block's code (the caller's).
    let code = unsafe { ld(ptiles, u64::from(thread::blockIdx_x())) };
    let s = (code >> 32) as u32;
    let begin = u64::from(code as u32) * PART_TILE;
    // SAFETY: `segs` holds segment `s`'s two words (the caller's).
    let (off, len) = unsafe { (ld(segs, 2 * u64::from(s)), ld(segs, 2 * u64::from(s) + 1)) };
    (s, off, begin, (begin + PART_TILE).min(len))
}

/// `margins[r] += values[leaf]` for every row `r` of every leaf segment,
/// one block per tile: each row is in one leaf, so every margin receives
/// one `f32` add, the CPU's.
///
/// # Safety
///
/// A one-dimensional grid of one block per `ptiles` entry, blocks of any
/// size. Each code names a segment `s` whose `(offset, len)` is `segs[2
/// s..2 s + 2]` and whose value is `values[s]`; each segment lies within
/// `rows` (`offset + len` at most its length), and every row id in the
/// segments is below `margins`' length and appears once across them all
/// (distinct segments of distinct rows, no two codes for one tile).
#[kernel]
pub unsafe fn add_leaves(
    segs: *const u64,
    values: *const f32,
    ptiles: *const u64,
    rows: *const u32,
    margins: *mut f32,
) {
    // SAFETY: one block per `ptiles` entry, each code's segment in `segs`
    // (the contract).
    let (s, off, begin, end) = unsafe { part_tile(segs, ptiles) };
    // SAFETY: `values` holds segment `s`'s value (the contract).
    let v = unsafe { ld(values, u64::from(s)) };
    let mut i = begin + u64::from(thread::threadIdx_x());
    while i < end {
        // SAFETY: `i < end <= len` (loop condition, `part_tile`), so `off +
        // i` is in the segment, within `rows`; its row is within `margins`
        // and listed nowhere else, so this thread alone accesses the margin
        // (the contract).
        unsafe {
            let r = u64::from(ld(rows, off + i));
            st(margins, r, ld(margins, r) + v);
        }
        i += u64::from(thread::blockDim_x());
    }
}

/// `v` summed over the warp, into lane 0.
#[inline(always)]
fn warp_sum_i64(mut v: i64) -> i64 {
    let mut delta = 16;
    while delta > 0 {
        v = v.wrapping_add(warp::shuffle_down_u64_sync(FULL, v as u64, delta) as i64);
        delta >>= 1;
    }
    v
}

/// The integer totals of the listed rows' grains per `grain`-row chunk, one
/// block (of whole warps) per chunk; exact when the caller checked that the
/// chunks' sums are.
///
/// # Safety
///
/// Blocks of whole warps (`blockDim.x` a multiple of 32, at most 1024 as on
/// every device), one-dimensional. `rows` holds `n` entries, each a row
/// below `units`' length, and `totals` holds two words per block (`2 *
/// gridDim.x`).
#[kernel]
pub unsafe fn chunk_totals(
    rows: *const u32,
    n: u64,
    grain: u64,
    units: *const I64x2,
    totals: *mut i64,
) {
    static mut WARP_G: SharedArray<i64, 32> = SharedArray::UNINIT;
    static mut WARP_H: SharedArray<i64, 32> = SharedArray::UNINIT;
    // SAFETY: `WARP_G` is this kernel's `static mut SharedArray`; the
    // accesses below state their disjointness and barrier.
    let warp_g = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WARP_G) };
    // SAFETY: as for `WARP_G`.
    let warp_h = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WARP_H) };
    let tid = thread::threadIdx_x();
    let block = u64::from(thread::blockIdx_x());
    let begin = block * grain;
    let end = (begin + grain).min(n);
    let (mut g, mut h) = (0i64, 0i64);
    let mut i = begin + u64::from(tid);
    while i < end {
        // SAFETY: `i < end <= n` (loop condition) indexes `rows`, and its
        // row is within `units` (the contract).
        let u = unsafe { ld(units, u64::from(ld(rows, i))) };
        g = g.wrapping_add(u.x);
        h = h.wrapping_add(u.y);
        i += u64::from(thread::blockDim_x());
    }
    let (g, h) = (warp_sum_i64(g), warp_sum_i64(h));
    if tid.is_multiple_of(32) {
        // SAFETY: slot `tid >> 5 < 32` (at most 1024 threads), written by
        // this warp's lane 0 alone, before the barrier below.
        unsafe {
            st(warp_g, u64::from(tid >> 5), g);
            st(warp_h, u64::from(tid >> 5), h);
        }
    }
    thread::sync_threads();
    if tid == 0 {
        let (mut g, mut h) = (g, h);
        let mut w = 1;
        while w < thread::blockDim_x().div_ceil(32) {
            // SAFETY: `w` is below the block's warp count (loop condition),
            // at most 32: the slots warp `w`'s lane 0 wrote before the
            // barrier above (whole warps), not written again.
            unsafe {
                g = g.wrapping_add(ld(warp_g, u64::from(w)));
                h = h.wrapping_add(ld(warp_h, u64::from(w)));
            }
            w += 1;
        }
        // SAFETY: `totals` holds the block's two words (the contract),
        // written by its thread 0 alone.
        unsafe {
            st(totals, 2 * block, g);
            st(totals, 2 * block + 1, h);
        }
    }
}

// ---------------------------------------------------------------------------
// Integer histograms

/// A 64-bit add into shared memory as two 32-bit atomics with a carry (as
/// XGBoost's `AtomicAdd64As32`), on the word's low and high halves: each
/// add carries exactly when its own low add wrapped, so the halves sum to
/// the 64-bit total modulo 2^64.
///
/// # Safety
///
/// `lo` and `hi` are valid, 4-byte aligned words of this block's shared
/// memory, which no thread accesses non-atomically until a barrier after
/// the block's last add.
#[inline(always)]
unsafe fn add_shared(lo: *mut u32, hi: *mut u32, v: i64) {
    let low = v as u64 as u32;
    let high = ((v as u64) >> 32) as u32;
    // SAFETY: `lo` is a valid aligned shared word, only accessed atomically
    // meanwhile (the caller's).
    let old = unsafe { BlockAtomicU32::from_ptr(lo) }.fetch_add(low, Relaxed);
    let add_hi = high.wrapping_add(u32::from(old > u32::MAX - low));
    if add_hi != 0 {
        // SAFETY: as for `lo`, for `hi` (the caller's).
        unsafe { BlockAtomicU32::from_ptr(hi) }.fetch_add(add_hi, Relaxed);
    }
}

/// `n` added to the 64-bit word at `dst` in global memory.
///
/// # Safety
///
/// `dst` is a valid, 8-byte aligned word of global memory, which nothing
/// accesses non-atomically while any thread may update it.
#[inline(always)]
unsafe fn add_global(dst: *mut u64, n: u64) {
    // SAFETY: `dst` is a valid aligned global word, only accessed atomically
    // meanwhile (the caller's).
    unsafe { DeviceAtomicU64::from_ptr(dst) }.fetch_add(n, Relaxed);
}

/// A unit of histogram work: `count` rows from `rows[begin]`, summed into
/// integer slot `target & 0x7fff_ffff` of the exact accumulators, or (high
/// bit set) of the per-chunk partials, which it then owns.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct Tile {
    pub begin: u64,
    pub count: u32,
    pub target: u32,
}

impl Tile {
    /// The tile's integer histogram (`[bin][2]` words).
    #[inline(always)]
    fn histogram(self, acc: *mut u64, partials: *mut u64, total_bins: u64) -> *mut u64 {
        let base = if self.target >> 31 != 0 {
            partials
        } else {
            acc
        };
        base.wrapping_add((u64::from(self.target & 0x7fff_ffff) * total_bins * 2) as usize)
    }

    /// Whether the tile owns a partial (rather than adding to a shared
    /// accumulator).
    #[inline(always)]
    fn partial(self) -> bool {
        self.target >> 31 != 0
    }
}

/// A feature group: features `[f0, f1)`, whose global bins `[bin0, bin0 +
/// bins)` one block's shared histogram holds.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct Group {
    pub f0: u32,
    pub f1: u32,
    pub bin0: u32,
    pub bins: u32,
}

/// A group-relative bin no element names (they are below 2^31): past the
/// tile, or a missing value.
const NONE: u32 = u32::MAX;

/// One element's pair added to bin `bin` (unless [`NONE`]) of the block's
/// shared histogram: four planes of `bins` words (gradient low and high
/// halves, then the Hessian's), so a warp's bins fall in distinct banks.
///
/// # Safety
///
/// Unless `bin` is [`NONE`], `bin < bins`, and `planes` points to `4 *
/// bins` words of the block's shared memory, as [`add_shared`] requires of
/// each.
#[inline(always)]
unsafe fn add_unit_shared(planes: *mut u32, bins: u32, bin: u32, unit: I64x2) {
    if bin != NONE {
        let (bin, bins) = (bin as usize, bins as usize);
        // SAFETY: `bin < bins` (the caller's, as `bin != NONE`), so word
        // `k * bins + bin` of each plane `k < 4` is among the `4 * bins`
        // shared words, only updated atomically meanwhile (the caller's).
        unsafe {
            add_shared(planes.add(bin), planes.add(bins + bin), unit.x);
            add_shared(
                planes.add(2 * bins + bin),
                planes.add(3 * bins + bin),
                unit.y,
            );
        }
    }
}

/// One element's pair added to bin `bin` (unless [`NONE`]) of the target
/// histogram, from the group's first bin `bin0`.
///
/// # Safety
///
/// Unless `bin` is [`NONE`], words `2 * (bin0 + bin)` and the next of the
/// histogram at `target` are valid, as [`add_global`] requires of each.
#[inline(always)]
unsafe fn add_unit_global(target: *mut u64, bin0: u32, bin: u32, unit: I64x2) {
    if bin != NONE {
        // SAFETY: `bin != NONE`, so the bin's two words are valid global
        // words only updated atomically meanwhile (the caller's).
        unsafe {
            let t = target.add(2 * (bin0 as usize + bin as usize));
            add_global(t, unit.x as u64);
            add_global(t.add(1), unit.y as u64);
        }
    }
}

/// A dense histogram launch's work and layout: its tiles and feature
/// groups (block `tile * n_groups + group`), the bins' row stride and
/// missing sentinel, and the histograms' width.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TileWork {
    pub tiles: *const Tile,
    pub groups: *const Group,
    pub total_bins: u64,
    pub stride: u32,
    pub sentinel: u32,
    pub n_groups: u32,
}

/// One block per (tile, group): the tile's rows over the group's features,
/// element `idx` = (row `idx / nf`, feature `idx % nf`), so a warp reads one
/// row's bins contiguously and its gradient once. A tile's groups are
/// consecutive blocks (`blockIdx.x = tile * n_groups + group`), which run
/// together, so the tile's rows, bins and gradients come from L2 for all
/// but the first group. `SHARED`: privatized in dynamic shared memory (16
/// bytes per bin: [`add_unit_shared`]'s four word planes), then flushed
/// (64-bit atomics into an exact accumulator, or plain stores into the
/// tile's own partial); otherwise straight into the target with 64-bit
/// global atomics (a group too wide for shared memory). The pointers the
/// element loop reads are kernel parameters (global-space loads); `work`
/// carries what each block reads once.
///
/// # Safety
///
/// Every thread of the block calls this (it holds barriers), in a grid of
/// `n_tiles * n_groups` blocks with `n_groups` positive; any block size.
/// `tiles` holds entry `blockIdx.x / n_groups` and `groups` entry
/// `blockIdx.x % n_groups`. For each such tile and group:
/// - `rows` holds the tile's `count` rows from `begin`, each a row of
///   `bins` (`stride` entries per row) and of `units`;
/// - the group's features `[f0, f1)` are below `stride` and have entries in
///   `feature_first`, and every stored bin `b` of feature `f` other than
///   `sentinel` lies in the group: `bin0 <= feature_first[f] + b < bin0 +
///   bins`;
/// - the tile's target slot (`target & 0x7fff_ffff`) has `2 * total_bins`
///   words in `partials` (high bit set) or `acc` (clear), with `bin0 + bins
///   <= total_bins`. Accumulator words are only updated atomically during
///   the launch; a partial slot belongs to one tile, whose groups cover
///   disjoint bin ranges, so each of its bins is one block's.
///
/// With `SHARED`, the dynamic shared memory holds at least `16 * bins`
/// bytes for every group.
#[inline(always)]
unsafe fn hist_tile<B: Bin, const SHARED: bool>(
    bins: *const B,
    feature_first: *const u32,
    rows: *const u32,
    units: *const I64x2,
    acc: *mut u64,
    partials: *mut u64,
    work: TileWork,
) {
    let TileWork {
        tiles,
        groups,
        total_bins,
        stride,
        sentinel,
        n_groups,
    } = work;
    let words = DynamicSharedArray::<u64>::get();
    let smem = words.cast::<u32>();
    let block = thread::blockIdx_x();
    // SAFETY: `tiles` holds the block's tile (the caller's).
    let tile = unsafe { ld(tiles, u64::from(block / n_groups)) };
    // SAFETY: `groups` holds the block's group (the caller's).
    let g = unsafe { ld(groups, u64::from(block % n_groups)) };
    let partial = tile.partial();
    let target = tile.histogram(acc, partials, total_bins);
    let (tid, step) = (thread::threadIdx_x(), thread::blockDim_x());
    // The branches and loops below depend on the block's tile and group
    // only, so every thread of the block reaches each barrier.
    if SHARED {
        // Counted in `u64`: `2 * bins` and the stride wrap a `u32` near
        // 2^31 bins, which `with_global` accepts.
        let mut i = u64::from(tid);
        while i < 2 * u64::from(g.bins) {
            // SAFETY: `i < 2 * bins` (loop condition) 8-byte words lie in
            // the dynamic shared memory's `16 * bins` bytes (the caller's);
            // each thread zeroes its own words before the barrier below.
            unsafe { st(words, i, 0) };
            i += u64::from(step);
        }
        thread::sync_threads();
    } else if partial {
        let mut i = u64::from(tid);
        while i < 2 * u64::from(g.bins) {
            // SAFETY: `2 * bin0 + i < 2 * (bin0 + bins) <= 2 * total_bins`
            // (loop condition, the caller's): a word of the tile's own
            // partial in the group's bins, which only this block accesses;
            // each thread zeroes its own words before the barrier below.
            unsafe { st(target, 2 * u64::from(g.bin0) + i, 0) };
            i += u64::from(step);
        }
        thread::sync_threads();
    }
    let nf = g.f1 - g.f0;
    // SAFETY: `rows` holds the tile's rows from `begin` (the caller's).
    let tile_rows = unsafe { rows.add(tile.begin as usize) };
    // A thread's elements are `step` apart: track each one's (row,
    // feature) incrementally instead of dividing (`nf` is positive
    // whenever an element exists; past the tile the row reaches
    // `tile.count`, the same bound as `idx < tile.count * nf`).
    let nf_div = nf.max(1);
    let (row_step, feature_step) = (step / nf_div, step % nf_div);
    let next = |(i, f): (u32, u32)| {
        let f = f + feature_step;
        if f >= nf {
            (i + row_step + 1, f - nf)
        } else {
            (i + row_step, f)
        }
    };
    // Element (row `i`, feature `g.f0 + f`)'s row and group-relative
    // bin (`NONE` past the tile or for a missing value).
    let element = |(i, f): (u32, u32)| {
        if i < tile.count {
            let f = g.f0 + f;
            // SAFETY: `i < count` (checked above) is one of the tile's rows
            // (the caller's).
            let r = unsafe { ld(tile_rows, u64::from(i)) };
            // SAFETY: `r` is a row of `bins`, and the feature `f0 + f` (`f <
            // nf`: `next` keeps it so) is a group feature below `stride`
            // (the caller's).
            let b = unsafe { ld(bins, u64::from(r) * u64::from(stride) + u64::from(f)) }.get();
            let bin = if b == sentinel {
                NONE
            } else {
                // SAFETY: `feature_first` holds the group feature's entry
                // (the caller's).
                let first = unsafe { ld(feature_first, u64::from(f)) };
                first + b - g.bin0
            };
            (r, bin)
        } else {
            (0, NONE)
        }
    };
    let unit = |(r, bin): (u32, u32)| {
        if bin == NONE {
            I64x2 { x: 0, y: 0 }
        } else {
            // SAFETY: `bin` is not `NONE`, so `r` is a tile row, a row of
            // `units` (the caller's).
            unsafe { ld(units, u64::from(r)) }
        }
    };
    // Four elements per thread per pass, their loads issued before any
    // atomic, so each thread keeps several gathers in flight (in
    // registers: named, not an array the compiler may spill to local
    // memory).
    let mut c0 = (tid / nf_div, tid % nf_div);
    while nf != 0 && c0.0 < tile.count {
        let c1 = next(c0);
        let c2 = next(c1);
        let c3 = next(c2);
        let (e0, e1, e2, e3) = (element(c0), element(c1), element(c2), element(c3));
        let (u0, u1, u2, u3) = (unit(e0), unit(e1), unit(e2), unit(e3));
        if SHARED {
            // SAFETY: each bin is `NONE` or below `g.bins` (the caller's bin
            // ranges); the planes are the `4 * g.bins` shared words zeroed
            // before the barrier above (16 bytes per bin, the caller's),
            // only updated atomically until the barrier after this loop.
            unsafe {
                add_unit_shared(smem, g.bins, e0.1, u0);
                add_unit_shared(smem, g.bins, e1.1, u1);
                add_unit_shared(smem, g.bins, e2.1, u2);
                add_unit_shared(smem, g.bins, e3.1, u3);
            }
        } else {
            // SAFETY: each bin is `NONE` or below `g.bins`, so `bin0 + bin <
            // total_bins` is a bin of the target (the caller's), whose
            // words are only updated atomically meanwhile: an accumulator's
            // throughout, a partial's after the zeroing barrier above.
            unsafe {
                add_unit_global(target, g.bin0, e0.1, u0);
                add_unit_global(target, g.bin0, e1.1, u1);
                add_unit_global(target, g.bin0, e2.1, u2);
                add_unit_global(target, g.bin0, e3.1, u3);
            }
        }
        c0 = next(c3);
    }
    if SHARED {
        thread::sync_threads();
        let (b0, b1, b2, b3) = (0, g.bins, 2 * g.bins, 3 * g.bins);
        let mut b = tid;
        while b < g.bins {
            let word = |lo: u32, hi: u32| {
                // SAFETY: `b < g.bins` (loop condition), so `lo + b` and `hi
                // + b` are among the `4 * g.bins` shared words, complete
                // after the barrier above and no longer updated.
                unsafe {
                    u64::from(ld(smem, u64::from(lo + b)))
                        | u64::from(ld(smem, u64::from(hi + b))) << 32
                }
            };
            let (x, y) = (word(b0, b1), word(b2, b3));
            // SAFETY: `bin0 + b < bin0 + bins <= total_bins` (loop
            // condition, the caller's): a bin of the target histogram.
            let t = unsafe { target.add(2 * (g.bin0 as usize + b as usize)) };
            if partial {
                // SAFETY: the bin's two words in the tile's own partial,
                // which only this thread accesses (one block per group of
                // the tile, one thread per bin).
                unsafe {
                    *t = x;
                    *t.add(1) = y;
                }
            } else {
                if x != 0 {
                    // SAFETY: an accumulator word, only updated atomically
                    // during the launch (the caller's).
                    unsafe { add_global(t, x) };
                }
                if y != 0 {
                    // SAFETY: as for `t`, the bin's second word.
                    unsafe { add_global(t.add(1), y) };
                }
            }
            b += step;
        }
    }
}

/// A chain histogram launch's chunks: `segs` chunks of `seg_rows` of the
/// `n` listed rows each, one `total_bins`-bin partial per chunk.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Chunks {
    pub n: u64,
    pub seg_rows: u64,
    pub segs: u64,
    pub total_bins: u64,
}

/// A dense chain histogram launch: its [`Chunks`], and the bins' row
/// stride, feature count and missing sentinel.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ChainWork {
    pub chunks: Chunks,
    pub stride: u32,
    pub n_cols: u32,
    pub sentinel: u32,
}

/// Integer histograms of `segs` chunks of `seg_rows` rows each, as `f64`
/// chains: one thread per (chunk, feature) adds the chunk's rows in order
/// to its own feature's bins of the chunk's `f64` partial (zeroed), so every
/// bin is a chain from `+0.0` in row order, the CPU's.
///
/// # Safety
///
/// One thread per (chunk, feature): a one-dimensional grid of at least
/// `segs * n_cols` threads (the rest return). `rows` holds `n` entries,
/// each a row of `gpair` and of `bins` (`stride` entries per row, `n_cols
/// <= stride`); `feature_first` holds `n_cols` entries, and every stored
/// bin `b` of feature `f` other than `sentinel` is below that feature's
/// count, so `feature_first[f] + b` is below the next feature's first bin
/// (`total_bins` for the last); `partials` holds `segs * total_bins` bins.
#[inline(always)]
unsafe fn hist_chain<B: Bin>(
    bins: *const B,
    feature_first: *const u32,
    rows: *const u32,
    gpair: *const F32x2,
    partials: *mut F64x2,
    work: ChainWork,
) {
    let ChainWork {
        chunks:
            Chunks {
                n,
                seg_rows,
                segs,
                total_bins,
            },
        stride,
        n_cols,
        sentinel,
    } = work;
    let t = grid_index();
    let n_cols = u64::from(n_cols);
    if t >= segs * n_cols {
        return;
    }
    let seg = t / n_cols;
    let f = t - seg * n_cols;
    let begin = seg * seg_rows;
    let end = (begin + seg_rows).min(n);
    // SAFETY: `t < segs * n_cols` (checked above), so `f < n_cols` has an
    // entry in `feature_first`, and chunk `seg < segs`'s partial from the
    // feature's first bin is within `partials` (the caller's).
    let h = unsafe { partials.add((seg * total_bins + u64::from(ld(feature_first, f))) as usize) };
    let mut i = begin;
    while i < end {
        // SAFETY: `i < end <= n` (loop condition) indexes `rows` (the
        // caller's).
        let r = u64::from(unsafe { ld(rows, i) });
        // SAFETY: `r` is a row of `bins`, and `f < n_cols <= stride` (the
        // caller's).
        let b = unsafe { ld(bins, r * u64::from(stride) + f) }.get();
        if b != sentinel {
            // SAFETY: `r` is a row of `gpair` (the caller's).
            let p = unsafe { ld(gpair, r) };
            // SAFETY: `b` is below feature `f`'s count (the caller's): a bin
            // of feature `f` in chunk `seg`'s partial, which only this
            // thread (the one for (`seg`, `f`)) accesses.
            let mut a = unsafe { ld(h, u64::from(b)) };
            a.x += f64::from(p.x);
            a.y += f64::from(p.y);
            // SAFETY: as for the load of `a`.
            unsafe { st(h, u64::from(b), a) };
        }
        i += 1;
    }
}

/// A CSR histogram launch's tiles (one per block) and the histograms'
/// width.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SparseTiles {
    pub tiles: *const Tile,
    pub total_bins: u64,
}

/// Integer CSR scatter: one warp per listed row, each stored entry visited
/// once whatever the feature count. All sums were proven exact.
///
/// # Safety
///
/// One block per tile (`tiles` holds `gridDim.x` entries), blocks of whole
/// warps, one-dimensional. `rows` holds each tile's `count` rows from
/// `begin`, each below the CSR's row count; `row_ptr` holds that count plus
/// one ascending offsets into `bins`, and `units` one pair per row; every
/// stored bin is below `total_bins`; each tile's target slot (`target &
/// 0x7fff_ffff`) has `2 * total_bins` words in `partials` (high bit set)
/// or `acc` (clear), only updated atomically during the launch.
#[inline(always)]
unsafe fn hist_sparse<B: Bin>(
    bins: *const B,
    row_ptr: *const u64,
    rows: *const u32,
    units: *const I64x2,
    acc: *mut u64,
    partials: *mut u64,
    work: SparseTiles,
) {
    let SparseTiles { tiles, total_bins } = work;
    // SAFETY: `tiles` holds the block's tile (the caller's).
    let tile = unsafe { ld(tiles, u64::from(thread::blockIdx_x())) };
    let target = tile.histogram(acc, partials, total_bins);
    let tid = thread::threadIdx_x();
    let (lane, warp) = (u64::from(tid & 31), u64::from(tid >> 5));
    let warps = u64::from(thread::blockDim_x() / 32);
    let mut i = warp;
    while i < u64::from(tile.count) {
        // SAFETY: `i < count` (loop condition) is one of the tile's rows in
        // `rows` (the caller's).
        let r = u64::from(unsafe { ld(rows, tile.begin + i) });
        // SAFETY: `r` is below the row count: a pair of `units` (the
        // caller's).
        let p = unsafe { ld(units, r) };
        // SAFETY: as for `p`, entries `r` and `r + 1` of `row_ptr`.
        let mut at = unsafe { ld(row_ptr, r) } + lane;
        // SAFETY: as for `at`.
        let end = unsafe { ld(row_ptr, r + 1) };
        while at < end {
            // SAFETY: `at < end = row_ptr[r + 1]` (loop condition) is a
            // stored entry of `bins`; its bin is below `total_bins`, so both
            // words are in the target slot, only updated atomically during
            // the launch (the caller's).
            unsafe {
                let b = u64::from(ld(bins, at).get());
                add_global(target.add((b * 2) as usize), p.x as u64);
                add_global(target.add((b * 2 + 1) as usize), p.y as u64);
            }
            at += 32;
        }
        i += warps;
    }
}

/// One thread per CPU chunk chains each stored CSR bin in row order. Unlike
/// feature sweeps this visits the stored entries once even for many mostly
/// absent features.
///
/// # Safety
///
/// A one-dimensional grid of any size (a grid-stride loop over the chunks,
/// one thread each). `rows` holds `n` entries, each below the CSR's row
/// count; `row_ptr` holds that count plus one ascending offsets into
/// `bins`, and `gpair` one pair per row; every stored bin is below
/// `total_bins`; `partials` holds `segs * total_bins` bins (zeroed).
#[inline(always)]
unsafe fn hist_sparse_chain<B: Bin>(
    bins: *const B,
    row_ptr: *const u64,
    rows: *const u32,
    gpair: *const F32x2,
    partials: *mut F64x2,
    chunks: Chunks,
) {
    let Chunks {
        n,
        seg_rows,
        segs,
        total_bins,
    } = chunks;
    let mut seg = grid_index();
    while seg < segs {
        let begin = seg * seg_rows;
        let end = (begin + seg_rows).min(n);
        // SAFETY: `seg < segs` (loop condition): chunk `seg`'s partial is
        // within `partials` (the caller's).
        let h = unsafe { partials.add((seg * total_bins) as usize) };
        let mut i = begin;
        while i < end {
            // SAFETY: `i < end <= n` (loop condition) indexes `rows` (the
            // caller's).
            let r = u64::from(unsafe { ld(rows, i) });
            // SAFETY: `r` is below the row count: a pair of `gpair` (the
            // caller's).
            let p = unsafe { ld(gpair, r) };
            // SAFETY: as for `p`, entries `r` and `r + 1` of `row_ptr`.
            let mut at = unsafe { ld(row_ptr, r) };
            // SAFETY: as for `at`.
            let stop = unsafe { ld(row_ptr, r + 1) };
            while at < stop {
                // SAFETY: `at < stop = row_ptr[r + 1]` (loop condition) is a
                // stored entry of `bins` (the caller's).
                let b = u64::from(unsafe { ld(bins, at) }.get());
                // SAFETY: `b < total_bins` (the caller's): a bin of chunk
                // `seg`'s partial, which only this thread accesses (one
                // thread per chunk).
                let mut a = unsafe { ld(h, b) };
                a.x += f64::from(p.x);
                a.y += f64::from(p.y);
                // SAFETY: as for the load of `a`.
                unsafe { st(h, b, a) };
                at += 1;
            }
            i += 1;
        }
        seg += grid_threads();
    }
}

// ---------------------------------------------------------------------------
// Partition

/// How a split routes a row: feature, then present feature-local bins below
/// `limit` go left (or, with flag 2, `table[table_at + bin]`), a missing
/// value follows flag 1 (`default_left`).
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct Rule {
    pub feature: u32,
    pub limit: u32,
    pub table_at: u32,
    pub flags: u32,
}

impl Rule {
    /// Whether a row with feature-local bin `b` (`None` if missing) goes
    /// left.
    ///
    /// # Safety
    ///
    /// With flag 2 and `b` present, `table` holds entry `table_at + b`.
    #[inline(always)]
    unsafe fn left(self, table: *const u8, b: Option<u32>) -> bool {
        match b {
            None => self.flags & 1 != 0,
            // SAFETY: flag 2 is set and `b` present (the arm's pattern), so
            // `table` holds entry `table_at + b` (the caller's).
            Some(b) if self.flags & 2 != 0 => unsafe {
                ld(table, u64::from(self.table_at) + u64::from(b)) != 0
            },
            Some(b) => b < self.limit,
        }
    }
}

/// `count` summed over the block (of whole warps), into thread 0.
///
/// # Safety
///
/// Every thread of the block calls this, once per launch (no barrier
/// follows thread 0's reads of the warps' slots), in a block of whole warps
/// (`blockDim.x` a multiple of 32, at most 1024 as on every device).
#[inline(always)]
unsafe fn block_sum_u32(mut count: u32) -> u32 {
    static mut WARP_SUM: SharedArray<u32, 32> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x();
    let mut delta = 16;
    while delta > 0 {
        count += warp::shuffle_down_sync(FULL, count, delta);
        delta >>= 1;
    }
    // SAFETY: `WARP_SUM` is this kernel's `static mut SharedArray`; the
    // accesses below state their disjointness and barrier.
    let warp_sum = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WARP_SUM) };
    if tid.is_multiple_of(32) {
        // SAFETY: slot `tid >> 5 < 32` (at most 1024 threads), written by
        // this warp's lane 0 alone, once (the caller's), before the barrier
        // below.
        unsafe { st(warp_sum, u64::from(tid >> 5), count) };
    }
    thread::sync_threads();
    let mut total = 0;
    if tid == 0 {
        let mut w = 0;
        while w < thread::blockDim_x().div_ceil(32) {
            // SAFETY: `w` is below the block's warp count (loop condition),
            // at most 32: the slot warp `w`'s lane 0 wrote before the
            // barrier above (whole warps), not written again.
            total += unsafe { ld(warp_sum, u64::from(w)) };
            w += 1;
        }
    }
    total
}

/// A routing launch's tiles: the splits' segments and rules, the tile
/// codes (one per block), and each tile's left count (out).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PartTiles {
    pub segs: *const u64,
    pub ptiles: *const u64,
    pub rules: *const Rule,
    pub tile_left: *mut u32,
}

/// Each tile's rows' directions (`flags[off + i]`, 1 = left) from the
/// feature-major bins (`cols[f * n_rows + r]`: one byte stream per feature,
/// so a node's ascending rows read few sectors), and the tile's left count.
///
/// # Safety
///
/// Every thread of the block calls this, once per launch, in a
/// one-dimensional grid of one block per `ptiles` entry, blocks of whole
/// warps (as [`block_sum_u32`]). Each code names a split `s` whose segment
/// `(off, len)` is `segs[2 s..2 s + 2]` and whose rule is `rules[s]`; each
/// segment lies within `rows` and `flags`, the segments do not overlap and
/// no two codes name one tile, and every row id in them is below `n_rows`.
/// `cols` holds `n_rows` bins for each rule's feature; every stored bin
/// other than `sentinel` is below its feature's count, and a tabled rule's
/// (flag 2) `table` holds `table_at + b` for each such bin `b`. `tile_left`
/// holds one entry per block.
#[inline(always)]
unsafe fn route_count<B: Bin>(
    cols: *const B,
    n_rows: u64,
    sentinel: u32,
    table: *const u8,
    rows: *const u32,
    flags: *mut u8,
    tiles: PartTiles,
) {
    // SAFETY: one block per `ptiles` entry, each code's segment in `segs`
    // (the caller's).
    let (s, off, begin, end) = unsafe { part_tile(tiles.segs, tiles.ptiles) };
    // SAFETY: `rules` holds split `s`'s rule (the caller's).
    let rule = unsafe { ld(tiles.rules, u64::from(s)) };
    // SAFETY: `cols` holds the rule's feature's `n_rows` bins (the
    // caller's).
    let col = unsafe { cols.add((u64::from(rule.feature) * n_rows) as usize) };
    let mut count = 0;
    let mut i = begin + u64::from(thread::threadIdx_x());
    while i < end {
        // SAFETY: `i < end <= len` (loop condition, `part_tile`), so `off +
        // i` is in the segment, within `rows`; its row is below `n_rows`, a
        // bin of the feature's column (the caller's).
        let b = unsafe { ld(col, u64::from(ld(rows, off + i))) }.get();
        // SAFETY: a present `b` (not the sentinel) is below the feature's
        // count, which a tabled rule's table covers (the caller's).
        let left = unsafe { rule.left(table, (b != sentinel).then_some(b)) };
        // SAFETY: `off + i` is in the segment, within `flags`, and no other
        // thread writes it (disjoint segments and tiles, the caller's).
        unsafe { st(flags, off + i, u8::from(left)) };
        count += u32::from(left);
        i += u64::from(thread::blockDim_x());
    }
    // SAFETY: every thread of the block reaches this one call (no early
    // return), in a block of whole warps (the caller's).
    let total = unsafe { block_sum_u32(count) };
    if thread::threadIdx_x() == 0 {
        // SAFETY: `tile_left` holds one entry per block (the caller's),
        // written by its thread 0 alone.
        unsafe { st(tiles.tile_left, u64::from(thread::blockIdx_x()), total) };
    }
}

/// Sparse routing locates the row's first stored bin in the feature's global
/// range, as the CPU's `feature_bin` does; absence is missing, never bin 0.
///
/// # Safety
///
/// As [`route_count`], with the CSR in place of `cols`: every row id in the
/// segments is below the CSR's row count, `row_ptr` holds that count plus
/// one ascending offsets into `bins`, `first` holds `n_cols + 1` entries
/// and each rule's feature is below `n_cols`, and a tabled rule's `table`
/// holds `table_at + b` for every `b` below its feature's count
/// (`first[feature + 1] - first[feature]`).
#[inline(always)]
unsafe fn route_sparse<B: Bin>(
    bins: *const B,
    row_ptr: *const u64,
    first: *const u32,
    table: *const u8,
    rows: *const u32,
    flags: *mut u8,
    tiles: PartTiles,
) {
    // SAFETY: one block per `ptiles` entry, each code's segment in `segs`
    // (the caller's).
    let (s, off, begin, end) = unsafe { part_tile(tiles.segs, tiles.ptiles) };
    // SAFETY: `rules` holds split `s`'s rule (the caller's).
    let rule = unsafe { ld(tiles.rules, u64::from(s)) };
    // SAFETY: the rule's feature is below `n_cols`, so `first` holds its
    // entry and the next (the caller's).
    let (fs, fe) = unsafe {
        (
            ld(first, u64::from(rule.feature)),
            ld(first, u64::from(rule.feature) + 1),
        )
    };
    let mut count = 0;
    let mut i = begin + u64::from(thread::threadIdx_x());
    while i < end {
        // SAFETY: `i < end <= len` (loop condition, `part_tile`), so `off +
        // i` is in the segment, within `rows` (the caller's).
        let r = u64::from(unsafe { ld(rows, off + i) });
        let mut b = None;
        // SAFETY: `r` is below the CSR's row count, so `row_ptr` holds
        // entries `r` and `r + 1` (the caller's).
        let mut at = unsafe { ld(row_ptr, r) };
        // SAFETY: as for `at`.
        let stop = unsafe { ld(row_ptr, r + 1) };
        while at < stop {
            // SAFETY: `at < stop = row_ptr[r + 1]` (loop condition) is a
            // stored entry of `bins` (the caller's).
            let global = unsafe { ld(bins, at) }.get();
            if global >= fs && global < fe {
                b = Some(global - fs);
                break;
            }
            at += 1;
        }
        // SAFETY: a present `b` is below `fe - fs`, the feature's count
        // (checked in the loop), which a tabled rule's table covers (the
        // caller's).
        let left = unsafe { rule.left(table, b) };
        // SAFETY: `off + i` is in the segment, within `flags`, and no other
        // thread writes it (disjoint segments and tiles, the caller's).
        unsafe { st(flags, off + i, u8::from(left)) };
        count += u32::from(left);
        i += u64::from(thread::blockDim_x());
    }
    // SAFETY: every thread of the block reaches this one call (no early
    // return), in a block of whole warps (the caller's).
    let total = unsafe { block_sum_u32(count) };
    if thread::threadIdx_x() == 0 {
        // SAFETY: `tile_left` holds one entry per block (the caller's),
        // written by its thread 0 alone.
        unsafe { st(tiles.tile_left, u64::from(thread::blockIdx_x()), total) };
    }
}

/// Per split (one thread each): the exclusive prefix of its tiles' left
/// counts, in tile order, and its left total.
///
/// # Safety
///
/// A one-dimensional grid of at least `n_splits` threads (the rest do
/// nothing). `split_tiles` and `left_len` hold `n_splits` entries; each
/// split's tiles `[x, x + y)` lie within `tile_left`, and no two splits'
/// ranges overlap.
#[kernel]
pub unsafe fn route_scan(
    split_tiles: *const U32x2,
    n_splits: u32,
    tile_left: *mut u32,
    left_len: *mut u32,
) {
    let s = grid_index();
    if s < u64::from(n_splits) {
        // SAFETY: `s < n_splits` (checked above) indexes `split_tiles` (the
        // contract).
        let t = unsafe { ld(split_tiles, s) };
        let mut run = 0;
        let mut k = t.x;
        while k < t.x + t.y {
            // SAFETY: `k` is in split `s`'s tile range (loop condition),
            // within `tile_left`, and only this thread accesses it (the
            // ranges are disjoint, the contract).
            let c = unsafe { ld(tile_left, u64::from(k)) };
            // SAFETY: as for `c`.
            unsafe { st(tile_left, u64::from(k), run) };
            run += c;
            k += 1;
        }
        // SAFETY: `s < n_splits` indexes `left_len` (the contract), this
        // split's thread alone writing it.
        unsafe { st(left_len, s, run) };
    }
}

/// Stable scatter of each tile's rows into `scratch` by their flags: left
/// rows to the segment's front in row order, right rows after all left
/// ones. Rows are ranked a block-width round at a time with warp ballots.
///
/// # Safety
///
/// As [`route_count`] for the grid, blocks, codes and segments (within
/// `rows`, `flags` and `scratch`; `scratch` overlaps neither input), after
/// it and [`route_scan`]: `flags` holds every segment row's direction,
/// `tile_left` one entry per block, each the left count of its split's
/// earlier tiles, and `left_len` each split's left total. Every rank then
/// lands inside its segment, one row per entry.
#[kernel]
pub unsafe fn route_scatter(
    segs: *const u64,
    ptiles: *const u64,
    rows: *const u32,
    flags: *const u8,
    tile_left: *const u32,
    left_len: *const u32,
    scratch: *mut u32,
) {
    static mut WARP_LEFT: SharedArray<u32, 32> = SharedArray::UNINIT;
    static mut WARP_VALID: SharedArray<u32, 32> = SharedArray::UNINIT;
    // SAFETY: `WARP_LEFT` is this kernel's `static mut SharedArray`; the
    // accesses below state their disjointness and barriers.
    let warp_left = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WARP_LEFT) };
    // SAFETY: as for `WARP_LEFT`.
    let warp_valid = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WARP_VALID) };
    // SAFETY: one block per `ptiles` entry, each code's segment in `segs`
    // (the contract).
    let (s, off, begin, end) = unsafe { part_tile(segs, ptiles) };
    let block = u64::from(thread::blockIdx_x());
    let tid = thread::threadIdx_x();
    // SAFETY: `tile_left` holds one entry per block (the contract).
    let mut left_at = off + u64::from(unsafe { ld(tile_left, block) });
    // SAFETY: `left_len` holds split `s`'s total and `tile_left` the block's
    // entry (the contract).
    let mut right_at = unsafe {
        off + u64::from(ld(left_len, u64::from(s))) + (begin - u64::from(ld(tile_left, block)))
    };
    let (lane, warp) = (tid & 31, tid >> 5);
    let warps = thread::blockDim_x().div_ceil(32);
    let below = (1u32 << lane) - 1;
    // The rounds depend on the block only: every thread reaches both
    // barriers of each, and every lane of a warp the ballots and shuffles.
    let mut base = begin;
    while base < end {
        let i = base + u64::from(tid);
        let valid = i < end;
        let (r, left) = if valid {
            // SAFETY: `i < end <= len` (`valid`), so `off + i` is in the
            // segment, within `rows` and `flags` (the contract).
            unsafe { (ld(rows, off + i), ld(flags, off + i) != 0) }
        } else {
            (0, false)
        };
        let lmask = warp::ballot_sync(FULL, left);
        let vmask = warp::ballot_sync(FULL, valid);
        if lane == 0 {
            // SAFETY: slot `warp < 32` (at most 1024 threads), written by
            // this warp's lane 0 alone, between the previous round's closing
            // barrier (after every read of it) and the barrier below.
            unsafe {
                st(warp_left, u64::from(warp), lmask.count_ones());
                st(warp_valid, u64::from(warp), vmask.count_ones());
            }
        }
        thread::sync_threads();
        // Every warp scans the per-warp counts itself (lane `w` holds
        // warp `w`'s) with shuffles: the earlier warps' left and right
        // counts and the round's totals, without another barrier.
        let (l, v) = if lane < warps {
            // SAFETY: `lane < warps` (checked above): a slot warp `lane`'s
            // lane 0 wrote before the barrier above, not rewritten before
            // the round's closing barrier.
            unsafe {
                (
                    ld(warp_left, u64::from(lane)),
                    ld(warp_valid, u64::from(lane)),
                )
            }
        } else {
            (0, 0)
        };
        let (mut lsum, mut rsum) = (l, v - l);
        let mut delta = 1;
        while delta < 32 {
            let up_l = warp::shuffle_up_sync(FULL, lsum, delta);
            let up_r = warp::shuffle_up_sync(FULL, rsum, delta);
            if lane >= delta {
                lsum += up_l;
                rsum += up_r;
            }
            delta <<= 1;
        }
        let lbefore = warp::shuffle_sync(FULL, lsum - l, warp);
        let rbefore = warp::shuffle_sync(FULL, rsum - (v - l), warp);
        let ltotal = warp::shuffle_sync(FULL, lsum, 31);
        let rtotal = warp::shuffle_sync(FULL, rsum, 31);
        if left {
            let rank = lbefore + (lmask & below).count_ones();
            // SAFETY: the row's rank among its split's left rows (earlier
            // tiles' via `tile_left`, earlier rounds' via `left_at`, then
            // earlier warps' and lanes') puts it in the segment's left part
            // of `scratch`, an entry no other row takes (the contract).
            unsafe { st(scratch, left_at + u64::from(rank), r) };
        } else if valid {
            let rank = rbefore + (vmask & !lmask & below).count_ones();
            // SAFETY: likewise among the right rows, after the split's
            // `left_len[s]` left ones: in the segment's right part.
            unsafe { st(scratch, right_at + u64::from(rank), r) };
        }
        left_at += u64::from(ltotal);
        right_at += u64::from(rtotal);
        thread::sync_threads();
        base += u64::from(thread::blockDim_x());
    }
}

/// Copy each tile's span of `scratch` back to `rows`.
///
/// # Safety
///
/// A one-dimensional grid of one block per `ptiles` entry, blocks of any
/// size. Each code names a segment `s` (`segs[2 s..2 s + 2]`) lying within
/// `scratch` and `rows`, which do not overlap; the segments do not overlap
/// and no two codes name one tile.
#[kernel]
pub unsafe fn route_copy(
    segs: *const u64,
    ptiles: *const u64,
    scratch: *const u32,
    rows: *mut u32,
) {
    // SAFETY: one block per `ptiles` entry, each code's segment in `segs`
    // (the contract).
    let (_, off, begin, end) = unsafe { part_tile(segs, ptiles) };
    let mut i = begin + u64::from(thread::threadIdx_x());
    while i < end {
        // SAFETY: `i < end <= len` (loop condition, `part_tile`), so `off +
        // i` is in the segment, within `scratch` and `rows`; only this
        // thread writes it (disjoint segments and tiles, the contract).
        unsafe { st(rows, off + i, ld(scratch, off + i)) };
        i += u64::from(thread::blockDim_x());
    }
}

// ---------------------------------------------------------------------------
// Reductions into the `f64` histograms

/// The `f64` value of integer histogram word `word` in `grain` units: `K *
/// 2^grain` is exact for `|K| <= 2^53`.
#[inline(always)]
fn scaled(word: u64, grain: f64) -> f64 {
    word as i64 as f64 * grain
}

/// Exact accumulators to `f64`: accumulator slot `nodes[k].x` into output
/// slot `nodes[k].y`, node `k = blockIdx.y`, bins in a grid-stride loop.
/// Equals the CPU's sum, which is exact too. `acc` holds two words per bin
/// of each slot.
///
/// # Safety
///
/// The nodes' output slots are distinct, so each output bin is one
/// thread's.
#[kernel]
pub unsafe fn finalize_exact(
    acc: &[u64],
    nodes: &[U32x2],
    total_bins: u64,
    grad_value: f64,
    hess_value: f64,
    mut out: DisjointSlice<F64x2>,
) {
    let nd = nodes[thread::blockIdx_y() as usize];
    let mut b = grid_index();
    while b < total_bins {
        let a = ((u64::from(nd.x) * total_bins + b) * 2) as usize;
        let value = F64x2 {
            x: scaled(acc[a], grad_value),
            y: scaled(acc[a + 1], hess_value),
        };
        // SAFETY: the output slots are distinct (the caller's), so no other
        // thread accesses this bin.
        unsafe { *scatter(&mut out, (u64::from(nd.y) * total_bins + b) as usize) = value };
        b += grid_threads();
    }
}

/// [`finalize_exact`] fused with [`subtract_hists`]: node `k = blockIdx.y`
/// is `nodes[3 k..]` = (accumulator slot, output slot, parent slot or
/// `u32::MAX`). The accumulator becomes the output slot's `f64` bins, and
/// a parent slot (the built child's parent, holding the parent's
/// histogram) becomes its sibling's, `parent - child`: the same IEEE
/// subtraction `subtract_hists` performs after a separate finalization, on
/// the same child bits.
///
/// # Safety
///
/// No slot is the output or parent slot of two nodes, nor both, so each
/// output and parent bin is one thread's.
#[kernel]
pub unsafe fn finalize_exact_sub(
    acc: &[u64],
    nodes: &[u32],
    total_bins: u64,
    grad_value: f64,
    hess_value: f64,
    mut out: DisjointSlice<F64x2>,
) {
    let k = 3 * thread::blockIdx_y() as usize;
    let (from, to, parent) = (nodes[k], nodes[k + 1], nodes[k + 2]);
    let mut b = grid_index();
    while b < total_bins {
        let a = ((u64::from(from) * total_bins + b) * 2) as usize;
        let child = F64x2 {
            x: scaled(acc[a], grad_value),
            y: scaled(acc[a + 1], hess_value),
        };
        // SAFETY: the output and parent slots are distinct (the caller's),
        // so no other thread accesses this bin.
        unsafe { *scatter(&mut out, (u64::from(to) * total_bins + b) as usize) = child };
        if parent != u32::MAX {
            // SAFETY: as above, for the parent slot's bin.
            let p = unsafe { scatter(&mut out, (u64::from(parent) * total_bins + b) as usize) };
            *p = F64x2 {
                x: p.x - child.x,
                y: p.y - child.y,
            };
        }
        b += grid_threads();
    }
}

/// Integer chunk partials to `f64`, per bin in chunk order: node `k =
/// blockIdx.y` reads partial slots `[x, x + y)` into output slot `z`,
/// copying the first when `w` (its first chunk) and adding the rest, the
/// CPU's copy-then-add; bins in a grid-stride loop. `partials` holds two
/// words per bin of each slot.
///
/// # Safety
///
/// The nodes' output slots are distinct, so each output bin is one
/// thread's.
#[kernel]
pub unsafe fn reduce_chunks(
    partials: &[u64],
    nodes: &[U32x4],
    total_bins: u64,
    grad_value: f64,
    hess_value: f64,
    mut out: DisjointSlice<F64x2>,
) {
    let nd = nodes[thread::blockIdx_y() as usize];
    let mut b = grid_index();
    while b < total_bins {
        // SAFETY: the output slots are distinct (the caller's), so no other
        // thread accesses this bin.
        let o = unsafe { scatter(&mut out, (u64::from(nd.z) * total_bins + b) as usize) };
        let (mut g, mut h, mut s) = if nd.w != 0 {
            let p = ((u64::from(nd.x) * total_bins + b) * 2) as usize;
            (
                scaled(partials[p], grad_value),
                scaled(partials[p + 1], hess_value),
                1,
            )
        } else {
            (o.x, o.y, 0)
        };
        while s < nd.y {
            let p = ((u64::from(nd.x + s) * total_bins + b) * 2) as usize;
            g += scaled(partials[p], grad_value);
            h += scaled(partials[p + 1], hess_value);
            s += 1;
        }
        *o = F64x2 { x: g, y: h };
        b += grid_threads();
    }
}

/// `f64` chain partials of `segs` chunks (`partials`, one histogram of
/// `out`'s bins each) into `out`, per bin in chunk order (the first copied
/// when `init`): one thread per bin.
#[kernel]
pub fn reduce_chains(partials: &[F64x2], segs: u64, init: i32, mut out: DisjointSlice<F64x2>) {
    let bins = out.len();
    let idx = thread::index_1d();
    let b = idx.get();
    if let Some(o) = out.get_mut(idx) {
        let (mut a, mut s) = if init != 0 { (partials[b], 1) } else { (*o, 0) };
        while s < segs as usize {
            let p = partials[s * bins + b];
            a.x += p.x;
            a.y += p.y;
            s += 1;
        }
        *o = a;
    }
}

// ---------------------------------------------------------------------------
// Resident split search: histograms stay in device slots.

/// Each `(parent, built)` slot pair's parent becomes `parent - built`, the
/// host's `subtract_in_place`: one thread per parent bin, in a grid-stride
/// loop over the pairs' bins.
///
/// # Safety
///
/// The pairs' parent slots are distinct, so are their built slots, and no
/// slot is both: each bin a thread reads or writes is that thread's.
#[kernel]
pub unsafe fn subtract_hists(mut pool: DisjointSlice<F64x2>, pairs: &[U32x2], total_bins: u64) {
    let n = pairs.len() as u64 * total_bins;
    let mut i = grid_index();
    while i < n {
        let (p, b) = (i / total_bins, i % total_bins);
        let pair = pairs[p as usize];
        // SAFETY: the built slots are distinct and none is a parent (the
        // caller's), so no other thread accesses this bin.
        let c = unsafe { *scatter(&mut pool, (u64::from(pair.y) * total_bins + b) as usize) };
        // SAFETY: the parent slots are distinct (the caller's), so no other
        // thread accesses this bin.
        let a = unsafe { scatter(&mut pool, (u64::from(pair.x) * total_bins + b) as usize) };
        *a = F64x2 {
            x: a.x - c.x,
            y: a.y - c.y,
        };
        i += grid_threads();
    }
}

/// The regularization a split search reads (the host's `RegParams`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Regularization {
    pub lambda: f64,
    pub alpha: f64,
    pub max_delta_step: f64,
    pub min_child_weight: f64,
}

/// The regularization a candidate's score reads (`SplitScorer`'s).
#[derive(Clone, Copy)]
pub(crate) struct ScanReg {
    pub(crate) lambda: f64,
    pub(crate) alpha: f64,
    pub(crate) max_delta_step: f64,
    pub(crate) min_child_weight: f64,
    pub(crate) root_gain: f32,
    pub(crate) lower: f32,
    pub(crate) upper: f32,
    pub(crate) dir: i32,
}

impl ScanReg {
    /// Request `request`'s scorer, in monotone direction `dir`.
    ///
    /// # Safety
    ///
    /// `params` holds the request's (root gain, lower, upper) at `3 *
    /// request`.
    #[inline(always)]
    pub(crate) unsafe fn new(
        reg: Regularization,
        params: *const f32,
        request: u64,
        dir: i32,
    ) -> Self {
        Self {
            lambda: reg.lambda,
            alpha: reg.alpha,
            max_delta_step: reg.max_delta_step,
            min_child_weight: reg.min_child_weight,
            // SAFETY: `params` holds the request's three values from `3 *
            // request` (the caller's).
            root_gain: unsafe { ld(params, 3 * request) },
            // SAFETY: as for `root_gain`.
            lower: unsafe { ld(params, 3 * request + 1) },
            // SAFETY: as for `root_gain`.
            upper: unsafe { ld(params, 3 * request + 2) },
            dir,
        }
    }

    /// The scorer's child weight (`SplitScorer::score_run`'s `weight`): the
    /// soft-thresholded gradient over `H + lambda` in `f64`,
    /// `max_delta_step`, rounded to `f32`, then clamped to the node's bounds.
    #[inline(always)]
    fn weight(&self, g: f64, h: f64) -> f32 {
        let t = if g > self.alpha {
            g - self.alpha
        } else if g < -self.alpha {
            g + self.alpha
        } else {
            0.0
        };
        let mut w = -t / (h + self.lambda);
        if self.max_delta_step != 0.0 && w.abs() > self.max_delta_step {
            w = self.max_delta_step.copysign(w);
        }
        let w = w as f32;
        if w < self.lower {
            self.lower
        } else if w > self.upper {
            self.upper
        } else {
            w
        }
    }

    /// XGBoost's `CalcGainGivenWeight` at an `f32` weight (`w * w` in
    /// `f32`).
    #[inline(always)]
    fn gain(&self, g: f64, h: f64, w: f32) -> f64 {
        -(2.0 * g * f64::from(w)
            + (h + self.lambda) * f64::from(w * w)
            + 2.0 * self.alpha * f64::from(w.abs()))
    }

    /// One candidate's loss change, `-inf` when a child is invalid or the
    /// monotone direction is violated (`SplitScorer::score_run`).
    #[inline(always)]
    pub(crate) fn score(&self, lg: f64, lh: f64, rg: f64, rh: f64) -> f32 {
        let valid =
            lh > 0.0 && rh > 0.0 && lh >= self.min_child_weight && rh >= self.min_child_weight;
        let wl = self.weight(lg, lh);
        let wr = self.weight(rg, rh);
        let monotone = match self.dir.cmp(&0) {
            Ordering::Greater => wl <= wr,
            Ordering::Less => wl >= wr,
            Ordering::Equal => true,
        };
        let chg = (self.gain(lg, lh, wl) as f32 + self.gain(rg, rh, wr) as f32) - self.root_gain;
        if valid && monotone {
            chg
        } else {
            f32::NEG_INFINITY
        }
    }
}

/// `v` from lane `src` of the warp, both components.
#[inline(always)]
pub(crate) fn shuffle_pair(v: F64x2, src: u32) -> F64x2 {
    F64x2 {
        x: warp::shuffle_f64_sync(FULL, v.x, src),
        y: warp::shuffle_f64_sync(FULL, v.y, src),
    }
}

/// The inclusive prefix sums of the lanes' `v` (lane `i` gets lanes `0..=i`)
/// by shuffles up 1, 2, 4, 8 and 16; every lane of the warp calls it. Its
/// association is not a sequential chain's, so it serves only sums the host
/// certified exact (`exact` scans), where every association of the
/// additions gives the chain's bits.
#[inline(always)]
pub(crate) fn warp_prefix(lane: u32, mut v: F64x2) -> F64x2 {
    let mut delta = 1;
    while delta < 32 {
        let up_g = warp::shuffle_up_f64_sync(FULL, v.x, delta);
        let up_h = warp::shuffle_up_f64_sync(FULL, v.y, delta);
        if lane >= delta {
            v = F64x2 {
                x: up_g + v.x,
                y: up_h + v.y,
            };
        }
        delta <<= 1;
    }
    v
}

/// The lowest lane whose `mine` holds (lane 0 when none does): the lane
/// holding a reduction's winning candidate, whose statistics the warp then
/// fetches with one shuffle. Candidates are each one lane's, so only the
/// score and position travel through the reduction.
#[inline(always)]
pub(crate) fn winner_lane(mine: bool) -> u32 {
    warp::ballot_sync(FULL, mine).trailing_zeros() & 31
}

/// Warp `w`'s 32 staged statistics in the block's shared memory, for
/// [`scan_splits`]'s sequential chains.
///
/// # Safety
///
/// `w < SCAN_WARPS`.
#[inline(always)]
unsafe fn chain_row(w: u32) -> *mut F64x2 {
    // 16-byte aligned: the PTX accesses the chains as `v2.b64` pairs.
    static mut CHAIN: SharedArray<F64x2, { SCAN_WARPS * 32 }, 16> = SharedArray::UNINIT;
    // SAFETY: `CHAIN` is this kernel's `static mut SharedArray`, and `w <
    // SCAN_WARPS` (the caller's) keeps the row's address within it.
    unsafe { SharedArray::as_raw_mut_ptr(&raw mut CHAIN).add(w as usize * 32) }
}

/// A numeric scan's tasks: `tasks[4 t..]` for `t < n_tasks`, each request's
/// `totals` and `params` (root gain, lower, upper), whether every feature
/// is dense (no backward pass), and whether the tree's histograms are
/// certified exact.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NumericTasks {
    pub tasks: *const u32,
    pub n_tasks: u64,
    pub totals: *const F64x2,
    pub params: *const f32,
    pub dense: i32,
    pub exact: u32,
}

/// A numeric scan's results, per task: `meta[4 t..]` and `acc[t]`
/// ([`scan_splits`] writes them, `merge_scans` reads them).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NumericResults {
    pub meta: *mut u32,
    pub acc: *mut F64x2,
}

/// One warp per feature (`tasks[4 t..]`: request, feature, histogram slot,
/// monotone direction) forms the prefix (then suffix) statistics 32 bins at
/// a time, and all lanes score independent candidates. Without `exact`,
/// lane 0 chains each window in the CPU's order (an arbitrary parallel
/// floating-point scan would change the CPU's bits); with `exact` (every
/// histogram and total of the tree is certified exact in grains, so every
/// association of the additions gives the chain's bits), each lane loads
/// one bin and the warp scans it ([`warp_prefix`]) onto the previous
/// window's total. The reduction keeps the largest finite loss, then the
/// earliest position (forward before backward). `meta[4 t..]` gets (status:
/// 0 none, 1 found, 2 NaN; bin; loss bits; backward), `acc[t]` the winner's
/// accumulated statistics.
///
/// # Safety
///
/// Blocks of exactly `32 * SCAN_WARPS` threads (the `launch_bounds`), so
/// every warp is whole and indexes a chain row, in a one-dimensional grid
/// of any size. `tasks` holds `4 * n_tasks` words, `meta` `4 * n_tasks`
/// and `acc` `n_tasks` entries; `totals` and `params` (root gain, lower,
/// upper) hold every request the tasks name; `feature_first` holds entries
/// `f` and `f + 1` of each task's feature `f`, with `feature_first[f] <=
/// feature_first[f + 1] <= total_bins`; and `pool` holds `total_bins` bins
/// for each task's slot.
#[kernel]
#[launch_bounds(32)]
pub unsafe fn scan_splits(
    pool: *const F64x2,
    feature_first: *const u32,
    total_bins: u64,
    work: NumericTasks,
    regularization: Regularization,
    out: NumericResults,
) {
    let NumericTasks {
        tasks,
        n_tasks,
        totals,
        params,
        dense,
        exact,
    } = work;
    let NumericResults { meta, acc } = out;
    let lane = thread::threadIdx_x() & 31;
    let w = thread::threadIdx_x() >> 5;
    // SAFETY: `w < SCAN_WARPS` (the block size, the contract).
    let chain = unsafe { chain_row(w) };
    let warps = u64::from(thread::gridDim_x()) * SCAN_WARPS as u64;
    // The task, its length and each window's `n` are the warp's: its lanes
    // diverge only in the `lane < n` and `lane == 0` branches, which rejoin
    // before each warp barrier and shuffle.
    let mut t = u64::from(thread::blockIdx_x()) * SCAN_WARPS as u64 + u64::from(w);
    while t < n_tasks {
        // SAFETY: `t < n_tasks` (loop condition): the task's four words are
        // within `tasks` (the contract).
        let request = u64::from(unsafe { ld(tasks, 4 * t) });
        // SAFETY: as for `request`.
        let f = u64::from(unsafe { ld(tasks, 4 * t + 1) });
        // SAFETY: as for `request`.
        let slot = u64::from(unsafe { ld(tasks, 4 * t + 2) });
        // SAFETY: as for `request`, and `params` holds the request's three
        // values (the contract).
        let reg =
            unsafe { ScanReg::new(regularization, params, request, ld(tasks, 4 * t + 3) as i32) };
        // SAFETY: `totals` holds the request's total (the contract).
        let total = unsafe { ld(totals, request) };
        // SAFETY: `feature_first` holds the feature's entries `f` and `f +
        // 1` (the contract).
        let first = unsafe { ld(feature_first, f) };
        // SAFETY: as for `first`.
        let len = u64::from(unsafe { ld(feature_first, f + 1) } - first);
        // SAFETY: the slot's `total_bins` bins are within `pool`, and the
        // feature's first bin is below `total_bins` (the contract).
        let bins = unsafe { pool.add((slot * total_bins + u64::from(first)) as usize) };
        let mut best = f32::NEG_INFINITY;
        let mut pos = u64::MAX;
        let mut best_acc = F64x2 { x: 0.0, y: 0.0 };
        let mut nan = false;
        // The statistics accumulated before the window: lane 0's chain,
        // or (exact) every lane's copy of the scan's total.
        let (mut g, mut h) = (0.0f64, 0.0f64);
        let mut backward = false;
        loop {
            if backward {
                let fg = warp::shuffle_f64_sync(FULL, g, 0);
                let fh = warp::shuffle_f64_sync(FULL, h, 0);
                if dense != 0 || (fg == total.x && fh == total.y) {
                    break;
                }
                g = 0.0;
                h = 0.0;
            }
            let mut base = 0;
            while base < len {
                let n = (len - base).min(32) as u32;
                let at = base + u64::from(lane);
                // This lane's bin of the window (`lane < n`).
                let bin = || {
                    // SAFETY: called only when `lane < n`, so `at < len`:
                    // one of the feature's bins (`len - 1 - at` too), within
                    // its slot's histogram (the contract).
                    unsafe { ld(bins, if backward { len - 1 - at } else { at }) }
                };
                let a = if exact != 0 {
                    let b = if lane < n {
                        bin()
                    } else {
                        F64x2 { x: 0.0, y: 0.0 }
                    };
                    let s = warp_prefix(lane, b);
                    let a = F64x2 {
                        x: g + s.x,
                        y: h + s.y,
                    };
                    let carry = shuffle_pair(a, n - 1);
                    (g, h) = (carry.x, carry.y);
                    a
                } else {
                    // The window's bins, staged in shared memory by
                    // every lane (one coalesced load each), then chained
                    // in order in place by lane 0: its serial adds wait
                    // on no global load.
                    if lane < n {
                        // SAFETY: `lane < n <= 32`: this lane's entry of the
                        // warp's own chain row, which no other lane accesses
                        // until the warp barrier below (the previous window's
                        // reads ended at its closing barrier).
                        unsafe { st(chain, u64::from(lane), bin()) };
                    }
                    warp::sync_mask(FULL);
                    if lane == 0 {
                        let mut i = 0;
                        while i < n {
                            // SAFETY: `i < n <= 32` (loop condition): an
                            // entry of the warp's chain row, which only lane
                            // 0 accesses between the two barriers.
                            let b = unsafe { ld(chain, u64::from(i)) };
                            g += b.x;
                            h += b.y;
                            // SAFETY: as for the load of `b`.
                            unsafe { st(chain, u64::from(i), F64x2 { x: g, y: h }) };
                            i += 1;
                        }
                    }
                    warp::sync_mask(FULL);
                    let a = if lane < n {
                        // SAFETY: `lane < n`: the lane's entry of the warp's
                        // chain row, complete after the barrier above and not
                        // written again before the barrier below.
                        unsafe { ld(chain, u64::from(lane)) }
                    } else {
                        F64x2 { x: 0.0, y: 0.0 }
                    };
                    warp::sync_mask(FULL);
                    a
                };
                if lane < n {
                    let (rest_g, rest_h) = (total.x - a.x, total.y - a.y);
                    let l = if backward {
                        reg.score(rest_g, rest_h, a.x, a.y)
                    } else {
                        reg.score(a.x, a.y, rest_g, rest_h)
                    };
                    if l.is_nan() {
                        nan = true;
                    } else if l > best && l.is_finite() {
                        best = l;
                        pos = if backward { len } else { 0 } + at;
                        best_acc = a;
                    }
                }
                base += 32;
            }
            if backward {
                break;
            }
            backward = true;
        }
        // Positions are unique to their lanes: reduce the loss and
        // position, then fetch the winner's statistics from its lane.
        let (mut top, mut top_pos) = (best, pos);
        let mut delta = 16;
        while delta > 0 {
            let other_best = warp::shuffle_down_f32_sync(FULL, top, delta);
            let other_pos = warp::shuffle_down_u64_sync(FULL, top_pos, delta);
            if other_best > top || (other_best == top && other_pos < top_pos) {
                top = other_best;
                top_pos = other_pos;
            }
            delta >>= 1;
        }
        let top_pos = warp::shuffle_u64_sync(FULL, top_pos, 0);
        let found = top_pos != u64::MAX;
        let winner = shuffle_pair(best_acc, winner_lane(found && pos == top_pos));
        let nan = warp::any_sync(FULL, nan);
        if lane == 0 {
            let backward = found && top_pos >= len;
            let status = if nan { 2 } else { u32::from(found) };
            let bin = match (found, backward) {
                (false, _) => 0,
                (true, true) => (2 * len - 1 - top_pos) as u32,
                (true, false) => top_pos as u32,
            };
            // SAFETY: `t < n_tasks`: the task's four `meta` words and its
            // `acc` entry (the contract), written by this warp's lane 0
            // alone (one warp per task).
            unsafe {
                st(meta, 4 * t, status);
                st(meta, 4 * t + 1, bin);
                st(meta, 4 * t + 2, top.to_bits());
                st(meta, 4 * t + 3, u32::from(backward));
                st(acc, t, winner);
            }
        }
        t += warps;
    }
}

// ---------------------------------------------------------------------------
// Instantiations per bin width

/// The per-width entries (`*_u8`, `*_u16`, `*_u32`) of the histogram and
/// routing kernels.
macro_rules! per_width {
    (
        $bin:ty,
        $hist_shared:ident,
        $hist_global:ident,
        $hist_chain:ident,
        $route_count:ident,
        $hist_sparse:ident,
        $hist_sparse_chain:ident,
        $route_sparse:ident
    ) => {
        /// [`hist_tile`] privatized in shared memory.
        ///
        /// # Safety
        ///
        /// As [`hist_tile`] for the grid, tiles, groups, rows, bins and
        /// targets, in blocks of at most 512 threads (the `launch_bounds`)
        /// with at least 16 bytes of dynamic shared memory per bin of the
        /// widest group.
        #[kernel]
        #[launch_bounds(512)]
        pub unsafe fn $hist_shared(
            bins: *const $bin,
            feature_first: *const u32,
            rows: *const u32,
            units: *const I64x2,
            acc: *mut u64,
            partials: *mut u64,
            work: TileWork,
        ) {
            // SAFETY: every thread of the block runs the kernel, and its
            // contract is `hist_tile`'s with `SHARED`'s shared memory.
            unsafe {
                hist_tile::<$bin, true>(bins, feature_first, rows, units, acc, partials, work)
            }
        }

        /// [`hist_tile`] with global atomics.
        ///
        /// # Safety
        ///
        /// As [`hist_tile`] for the grid, tiles, groups, rows, bins and
        /// targets, in blocks of at most 512 threads (the `launch_bounds`).
        #[kernel]
        #[launch_bounds(512)]
        pub unsafe fn $hist_global(
            bins: *const $bin,
            feature_first: *const u32,
            rows: *const u32,
            units: *const I64x2,
            acc: *mut u64,
            partials: *mut u64,
            work: TileWork,
        ) {
            // SAFETY: every thread of the block runs the kernel, and its
            // contract is `hist_tile`'s without `SHARED`.
            unsafe {
                hist_tile::<$bin, false>(bins, feature_first, rows, units, acc, partials, work)
            }
        }

        /// [`hist_chain`] for this bin width.
        ///
        /// # Safety
        ///
        /// As [`hist_chain`].
        #[kernel]
        pub unsafe fn $hist_chain(
            bins: *const $bin,
            feature_first: *const u32,
            rows: *const u32,
            gpair: *const F32x2,
            partials: *mut F64x2,
            work: ChainWork,
        ) {
            // SAFETY: this kernel's contract is `hist_chain`'s.
            unsafe { hist_chain(bins, feature_first, rows, gpair, partials, work) }
        }

        /// [`route_count`] for this bin width.
        ///
        /// # Safety
        ///
        /// As [`route_count`] for the grid, blocks of whole warps, codes,
        /// segments, bins and table.
        #[kernel]
        pub unsafe fn $route_count(
            cols: *const $bin,
            n_rows: u64,
            sentinel: u32,
            table: *const u8,
            rows: *const u32,
            flags: *mut u8,
            tiles: PartTiles,
        ) {
            // SAFETY: every thread of the block runs the kernel and calls
            // this once, and its contract is `route_count`'s.
            unsafe { route_count(cols, n_rows, sentinel, table, rows, flags, tiles) }
        }

        /// [`hist_sparse`] for this bin width.
        ///
        /// # Safety
        ///
        /// As [`hist_sparse`], in blocks of at most 512 threads (the
        /// `launch_bounds`).
        #[kernel]
        #[launch_bounds(512)]
        pub unsafe fn $hist_sparse(
            bins: *const $bin,
            row_ptr: *const u64,
            rows: *const u32,
            units: *const I64x2,
            acc: *mut u64,
            partials: *mut u64,
            work: SparseTiles,
        ) {
            // SAFETY: this kernel's contract is `hist_sparse`'s.
            unsafe { hist_sparse(bins, row_ptr, rows, units, acc, partials, work) }
        }

        /// [`hist_sparse_chain`] for this bin width.
        ///
        /// # Safety
        ///
        /// As [`hist_sparse_chain`].
        #[kernel]
        pub unsafe fn $hist_sparse_chain(
            bins: *const $bin,
            row_ptr: *const u64,
            rows: *const u32,
            gpair: *const F32x2,
            partials: *mut F64x2,
            chunks: Chunks,
        ) {
            // SAFETY: this kernel's contract is `hist_sparse_chain`'s.
            unsafe { hist_sparse_chain(bins, row_ptr, rows, gpair, partials, chunks) }
        }

        /// [`route_sparse`] for this bin width.
        ///
        /// # Safety
        ///
        /// As [`route_sparse`] for the grid, blocks of whole warps, codes,
        /// segments, CSR and table.
        #[kernel]
        pub unsafe fn $route_sparse(
            bins: *const $bin,
            row_ptr: *const u64,
            first: *const u32,
            table: *const u8,
            rows: *const u32,
            flags: *mut u8,
            tiles: PartTiles,
        ) {
            // SAFETY: every thread of the block runs the kernel and calls
            // this once, and its contract is `route_sparse`'s.
            unsafe { route_sparse(bins, row_ptr, first, table, rows, flags, tiles) }
        }
    };
}

per_width!(
    u8,
    hist_shared_u8,
    hist_global_u8,
    hist_chain_u8,
    route_count_u8,
    hist_sparse_u8,
    hist_sparse_chain_u8,
    route_sparse_u8
);
per_width!(
    u16,
    hist_shared_u16,
    hist_global_u16,
    hist_chain_u16,
    route_count_u16,
    hist_sparse_u16,
    hist_sparse_chain_u16,
    route_sparse_u16
);
per_width!(
    u32,
    hist_shared_u32,
    hist_global_u32,
    hist_chain_u32,
    route_count_u32,
    hist_sparse_u32,
    hist_sparse_chain_u32,
    route_sparse_u32
);

/// After a partition (`route_copy`), whether each split's children are runs
/// of consecutive rows: `runs[s]` gets bit 0 for split `s`'s left child and
/// bit 1 for its right one, from the split's segment of `rows` (`segs[2
/// s..]` = offset, length) and its left child's length `left_len[s]`. A
/// child's rows ascend without repeats, so it is a run when its last row is
/// its first plus its length minus one; an empty child counts as a run.
/// One thread per split.
#[kernel]
pub fn route_runs(segs: &[u64], rows: &[u32], left_len: &[u32], mut runs: DisjointSlice<u32>) {
    let idx = thread::index_1d();
    let s = idx.get();
    if let Some(run) = runs.get_mut(idx) {
        let (off, len) = (segs[2 * s] as usize, segs[2 * s + 1] as usize);
        let left = left_len[s] as usize;
        *run = u32::from(is_run(rows, off, left))
            | u32::from(is_run(rows, off + left, len - left)) << 1;
    }
}

/// Whether the `n` ascending, distinct rows at `rows[begin..]` are
/// consecutive.
#[inline(always)]
fn is_run(rows: &[u32], begin: usize, n: usize) -> bool {
    n == 0 || u64::from(rows[begin + n - 1]) - u64::from(rows[begin]) == n as u64 - 1
}
