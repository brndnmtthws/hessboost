//! Histogram tree growth: binning, gradients, histograms, partitions,
//! reductions and the numeric split search.

use crate::{
    Bin, F32x2, F64x2, FULL, I64x2, SCAN_WARPS, U32x2, U32x4, grid_index, grid_threads, ld, st,
};
use core::cmp::Ordering;
use cuda_device::atomic::{
    AtomicOrdering::Relaxed, BlockAtomicU32, DeviceAtomicU32, DeviceAtomicU64,
};
use cuda_device::{DynamicSharedArray, SharedArray, kernel, launch_bounds, ptx_asm, thread, warp};

// ---------------------------------------------------------------------------
// Binning

/// The global CPU bins are the cut authority: each dense cell's global bin,
/// made feature-local, written in both layouts (row-major `rows` with
/// `stride` entries per row, feature-major `cols`), with 64-bit cell
/// offsets and no host transpose. Blocks are 32 x 8 threads, one 32 x 32
/// (rows x features) tile per block per pass.
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
    // SAFETY: the caller's; each thread writes its own tile cells between
    // barriers, which every thread of the block reaches.
    unsafe {
        let tile = SharedArray::as_raw_mut_ptr(&raw mut TILE);
        let (x, y) = (
            u64::from(thread::threadIdx_x()),
            u64::from(thread::threadIdx_y()),
        );
        let n_cols = u64::from(n_cols);
        let feature_tiles = n_cols.div_ceil(32);
        let row_tiles = n_rows.div_ceil(32);
        let mut t = u64::from(thread::blockIdx_x());
        while t < feature_tiles * row_tiles {
            let row_base = (t / feature_tiles) * 32;
            let feature_base = (t % feature_tiles) * 32;
            let f = feature_base + x;
            let mut j = 0;
            while j < 32 {
                let r = row_base + y + j;
                if r < n_rows && f < n_cols {
                    let local = ld(global, r * n_cols + f).get() - ld(first, f);
                    st(tile, (y + j) * 33 + x, local);
                    st(rows, r * u64::from(stride) + f, L::narrow(local));
                }
                j += 8;
            }
            thread::sync_threads();
            let r = row_base + x;
            let mut j = 0;
            while j < 32 {
                let out_f = feature_base + y + j;
                if r < n_rows && out_f < n_cols {
                    st(
                        cols,
                        out_f * n_rows + r,
                        L::narrow(ld(tile, x * 33 + y + j)),
                    );
                }
                j += 8;
            }
            thread::sync_threads();
            t += u64::from(thread::gridDim_x());
        }
    }
}

/// One encode entry: `$name` reads `$global` bins and writes `$local` ones.
macro_rules! encode {
    ($name:ident, $global:ty, $local:ty) => {
        /// [`encode_dense`] for these bin widths.
        ///
        /// # Safety
        ///
        /// `global` holds `n_rows * n_cols` cells, `first` `n_cols`
        /// entries, `rows` `n_rows * stride` and `cols` `n_rows * n_cols`.
        #[kernel]
        pub unsafe extern "C" fn $name(
            global: *const $global,
            first: *const u32,
            n_rows: u64,
            n_cols: u32,
            stride: u32,
            rows: *mut $local,
            cols: *mut $local,
        ) {
            // SAFETY: the caller's.
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
/// `i64` (exact for every slice the exact paths read).
///
/// # Safety
///
/// `gpair` and `units` hold `n` entries.
#[kernel]
pub unsafe extern "C" fn stage_units(
    gpair: *const F32x2,
    units: *mut I64x2,
    n: u64,
    grad_scale: f64,
    hess_scale: f64,
) {
    // SAFETY: the caller's; one thread per row.
    unsafe {
        let mut i = grid_index();
        while i < n {
            let p = ld(gpair, i);
            let unit = I64x2 {
                x: (f64::from(p.x) * grad_scale) as i64,
                y: (f64::from(p.y) * hess_scale) as i64,
            };
            st(units, i, unit);
            i += grid_threads();
        }
    }
}

/// `rows[i] = first + i`: an unsampled tree's rows without an upload.
///
/// # Safety
///
/// `rows` holds `n` entries.
#[kernel]
pub unsafe extern "C" fn iota_rows(rows: *mut u32, n: u64, first: u32) {
    // SAFETY: the caller's; one thread per row.
    unsafe {
        let mut i = grid_index();
        while i < n {
            st(rows, i, first + i as u32);
            i += grid_threads();
        }
    }
}

/// `reg:squarederror`'s gradients from the device margins, the CPU's
/// operations in `f32`: `w` (the row weight, or 1), times
/// `scale_pos_weight` for a label of exactly 1, then `((p - y) * w, w)`.
///
/// # Safety
///
/// `margins`, `labels` and `gpair` hold `n` entries, and `weights` too when
/// `weighted` is nonzero.
#[kernel]
pub unsafe extern "C" fn squared_error(
    margins: *const f32,
    labels: *const f32,
    weights: *const f32,
    weighted: i32,
    scale_pos_weight: f32,
    n: u64,
    gpair: *mut F32x2,
) {
    // SAFETY: the caller's; one thread per row.
    unsafe {
        let mut i = grid_index();
        while i < n {
            let (p, y) = (ld(margins, i), ld(labels, i));
            let mut w = if weighted != 0 { ld(weights, i) } else { 1.0 };
            if y == 1.0 {
                w *= scale_pos_weight;
            }
            st(
                gpair,
                i,
                F32x2 {
                    x: (p - y) * w,
                    y: w,
                },
            );
            i += grid_threads();
        }
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

/// [`logistic`]'s objective and batch: whether rows are weighted, the
/// positive-label weight scale, the Hessian floor, the largest margin
/// magnitude the vector path takes, the host's vector width, and the row
/// count.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LogisticParams {
    pub weighted: i32,
    pub scale_pos_weight: f32,
    pub min_hess: f32,
    pub max_input: f32,
    pub lanes: u32,
    pub n: u64,
}

/// The logistic objectives' gradients of the first `n` rows, the host's
/// vector kernel (`simd::logistic_gradient`): `lanes` rows per vector; a
/// vector holding a margin above `max_input` in magnitude (or a NaN) takes
/// the host's scalar path, which uses the C library's `expf`, so its rows
/// get NaN gradients here and the tree grows on the host.
///
/// # Safety
///
/// `margins`, `labels` and `gpair` hold `n` entries (a multiple of
/// `lanes`), and `weights` too when `weighted` is nonzero.
#[kernel]
pub unsafe extern "C" fn logistic(
    margins: *const f32,
    labels: *const f32,
    weights: *const f32,
    params: LogisticParams,
    gpair: *mut F32x2,
) {
    let LogisticParams {
        weighted,
        scale_pos_weight,
        min_hess,
        max_input,
        lanes,
        n,
    } = params;
    // SAFETY: the caller's; one thread per row, reading its vector's rows.
    unsafe {
        let mut i = grid_index();
        while i < n {
            let first = i - i % u64::from(lanes);
            let mut regular = true;
            let mut k = 0;
            while k < u64::from(lanes) {
                regular = regular && ld(margins, first + k).abs() <= max_input;
                k += 1;
            }
            if regular {
                let (x, y) = (ld(margins, i), ld(labels, i));
                let ex = exp_vector(-x.abs());
                let den = 1.0 + ex;
                let p = if x >= 0.0 { 1.0 / den } else { ex / den };
                let w = if weighted != 0 { ld(weights, i) } else { 1.0 };
                let w = w * if y == 1.0 { scale_pos_weight } else { 1.0 };
                let g = (p - y) * w;
                let h = (p * (1.0 - p)).max(min_hess) * w;
                st(gpair, i, F32x2 { x: g, y: h });
            } else {
                let nan = f32::from_bits(0x7fc0_0000);
                st(gpair, i, F32x2 { x: nan, y: nan });
            }
            i += grid_threads();
        }
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

    /// Combine into the global `[max bits, grain + 150, finite]`.
    ///
    /// # Safety
    ///
    /// `out` holds three `u32`.
    #[inline(always)]
    unsafe fn publish(self, out: *mut u32) {
        // SAFETY: the caller's; the updates are atomic.
        unsafe {
            DeviceAtomicU32::from_ptr(out).fetch_max(self.max_bits, Relaxed);
            DeviceAtomicU32::from_ptr(out.add(1)).fetch_min(self.grain, Relaxed);
            DeviceAtomicU32::from_ptr(out.add(2)).fetch_and(self.finite, Relaxed);
        }
    }
}

/// Both components' statistics of `n` pairs into `domain` (`[max bits,
/// grain + 150, finite]` per component; initialized to `[0, u32::MAX,
/// 1]`). Blocks hold whole warps.
///
/// # Safety
///
/// `gpair` holds `n` entries and `domain` six.
#[kernel]
pub unsafe extern "C" fn grad_domain(gpair: *const F32x2, n: u64, domain: *mut u32) {
    let (mut grad, mut hess) = (Domain::EMPTY, Domain::EMPTY);
    // SAFETY: the caller's; every lane reaches the shuffles.
    unsafe {
        let mut i = grid_index();
        while i < n {
            let p = ld(gpair, i);
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
            grad.publish(domain);
            hess.publish(domain.add(3));
        }
    }
}

/// Rows per partition and leaf-update tile (`PART_TILE` in
/// `src/backend/cuda/mod.rs`).
const PART_TILE: u64 = 4096;

/// A tile of a segment list: `code = segment << 32 | tile`, rows `[tile *
/// PART_TILE, ...)` of the segment `(offset, len)` in `segs`. Returns the
/// segment, its offset, and the tile's row range within it.
///
/// # Safety
///
/// `ptiles` holds block `blockIdx.x`'s code and `segs` its segment.
#[inline(always)]
unsafe fn part_tile(segs: *const u64, ptiles: *const u64) -> (u32, u64, u64, u64) {
    // SAFETY: the caller's.
    unsafe {
        let code = ld(ptiles, u64::from(thread::blockIdx_x()));
        let s = (code >> 32) as u32;
        let begin = u64::from(code as u32) * PART_TILE;
        let (off, len) = (ld(segs, 2 * u64::from(s)), ld(segs, 2 * u64::from(s) + 1));
        (s, off, begin, (begin + PART_TILE).min(len))
    }
}

/// `margins[r] += values[leaf]` for every row `r` of every leaf segment,
/// one block per tile: each row is in one leaf, so every margin receives
/// one `f32` add, the CPU's.
///
/// # Safety
///
/// `ptiles` holds one code per block, `segs` and `values` every segment
/// they name, `rows` every segment's rows, and `margins` every row.
#[kernel]
pub unsafe extern "C" fn add_leaves(
    segs: *const u64,
    values: *const f32,
    ptiles: *const u64,
    rows: *const u32,
    margins: *mut f32,
) {
    // SAFETY: the caller's; each row is in one segment, so one thread
    // updates its margin.
    unsafe {
        let (s, off, begin, end) = part_tile(segs, ptiles);
        let v = ld(values, u64::from(s));
        let mut i = begin + u64::from(thread::threadIdx_x());
        while i < end {
            let r = u64::from(ld(rows, off + i));
            st(margins, r, ld(margins, r) + v);
            i += u64::from(thread::blockDim_x());
        }
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
/// `rows` holds `n` entries, `units` every row they name, and `totals` two
/// words per block.
#[kernel]
pub unsafe extern "C" fn chunk_totals(
    rows: *const u32,
    n: u64,
    grain: u64,
    units: *const I64x2,
    totals: *mut i64,
) {
    static mut WARP_G: SharedArray<i64, 32> = SharedArray::UNINIT;
    static mut WARP_H: SharedArray<i64, 32> = SharedArray::UNINIT;
    // SAFETY: the caller's; lane 0 of each warp writes its own slot before
    // the barrier, thread 0 reads them after it.
    unsafe {
        let warp_g = SharedArray::as_raw_mut_ptr(&raw mut WARP_G);
        let warp_h = SharedArray::as_raw_mut_ptr(&raw mut WARP_H);
        let tid = thread::threadIdx_x();
        let block = u64::from(thread::blockIdx_x());
        let begin = block * grain;
        let end = (begin + grain).min(n);
        let (mut g, mut h) = (0i64, 0i64);
        let mut i = begin + u64::from(tid);
        while i < end {
            let u = ld(units, u64::from(ld(rows, i)));
            g = g.wrapping_add(u.x);
            h = h.wrapping_add(u.y);
            i += u64::from(thread::blockDim_x());
        }
        let (g, h) = (warp_sum_i64(g), warp_sum_i64(h));
        if tid.is_multiple_of(32) {
            st(warp_g, u64::from(tid >> 5), g);
            st(warp_h, u64::from(tid >> 5), h);
        }
        thread::sync_threads();
        if tid == 0 {
            let (mut g, mut h) = (g, h);
            let mut w = 1;
            while w < thread::blockDim_x().div_ceil(32) {
                g = g.wrapping_add(ld(warp_g, u64::from(w)));
                h = h.wrapping_add(ld(warp_h, u64::from(w)));
                w += 1;
            }
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
/// `lo` and `hi` are block-shared words only updated atomically meanwhile.
#[inline(always)]
unsafe fn add_shared(lo: *mut u32, hi: *mut u32, v: i64) {
    let low = v as u64 as u32;
    let high = ((v as u64) >> 32) as u32;
    // SAFETY: the caller's.
    unsafe {
        let old = BlockAtomicU32::from_ptr(lo).fetch_add(low, Relaxed);
        let add_hi = high.wrapping_add(u32::from(old > u32::MAX - low));
        if add_hi != 0 {
            BlockAtomicU32::from_ptr(hi).fetch_add(add_hi, Relaxed);
        }
    }
}

/// `n` added to the 64-bit word at `dst` in global memory.
///
/// # Safety
///
/// `dst` is an 8-byte aligned word only updated atomically meanwhile.
#[inline(always)]
unsafe fn add_global(dst: *mut u64, n: u64) {
    // SAFETY: the caller's.
    unsafe {
        DeviceAtomicU64::from_ptr(dst).fetch_add(n, Relaxed);
    }
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
/// As [`add_shared`] for the bin's four words.
#[inline(always)]
unsafe fn add_unit_shared(planes: *mut u32, bins: u32, bin: u32, unit: I64x2) {
    if bin != NONE {
        let (bin, bins) = (bin as usize, bins as usize);
        // SAFETY: the caller's.
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
/// As [`add_global`] for the bin's two words.
#[inline(always)]
unsafe fn add_unit_global(target: *mut u64, bin0: u32, bin: u32, unit: I64x2) {
    if bin != NONE {
        // SAFETY: the caller's.
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
/// `tiles` and `groups` hold the block's entries, `rows` each tile's rows,
/// `bins` `stride` entries per row, `feature_first` the group's features,
/// `units` every row, and `acc`/`partials` every slot the tiles name.
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
    // SAFETY: the caller's; shared words are zeroed and flushed between
    // barriers every thread reaches, and only updated atomically between.
    unsafe {
        let words = DynamicSharedArray::<u64>::get();
        let smem = words.cast::<u32>();
        let block = thread::blockIdx_x();
        let tile = ld(tiles, u64::from(block / n_groups));
        let g = ld(groups, u64::from(block % n_groups));
        let partial = tile.partial();
        let target = tile.histogram(acc, partials, total_bins);
        let (tid, step) = (thread::threadIdx_x(), thread::blockDim_x());
        if SHARED {
            // Counted in `u64`: `2 * bins` and the stride wrap a `u32` near
            // 2^31 bins, which `with_global` accepts.
            let mut i = u64::from(tid);
            while i < 2 * u64::from(g.bins) {
                st(words, i, 0);
                i += u64::from(step);
            }
            thread::sync_threads();
        } else if partial {
            let mut i = u64::from(tid);
            while i < 2 * u64::from(g.bins) {
                st(target, 2 * u64::from(g.bin0) + i, 0);
                i += u64::from(step);
            }
            thread::sync_threads();
        }
        let nf = g.f1 - g.f0;
        let tile_rows = rows.add(tile.begin as usize);
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
                let r = ld(tile_rows, u64::from(i));
                let b = ld(bins, u64::from(r) * u64::from(stride) + u64::from(f)).get();
                let bin = if b == sentinel {
                    NONE
                } else {
                    ld(feature_first, u64::from(f)) + b - g.bin0
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
                ld(units, u64::from(r))
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
                add_unit_shared(smem, g.bins, e0.1, u0);
                add_unit_shared(smem, g.bins, e1.1, u1);
                add_unit_shared(smem, g.bins, e2.1, u2);
                add_unit_shared(smem, g.bins, e3.1, u3);
            } else {
                add_unit_global(target, g.bin0, e0.1, u0);
                add_unit_global(target, g.bin0, e1.1, u1);
                add_unit_global(target, g.bin0, e2.1, u2);
                add_unit_global(target, g.bin0, e3.1, u3);
            }
            c0 = next(c3);
        }
        if SHARED {
            thread::sync_threads();
            let (b0, b1, b2, b3) = (0, g.bins, 2 * g.bins, 3 * g.bins);
            let mut b = tid;
            while b < g.bins {
                let word = |lo: u32, hi: u32| {
                    u64::from(ld(smem, u64::from(lo + b)))
                        | u64::from(ld(smem, u64::from(hi + b))) << 32
                };
                let (x, y) = (word(b0, b1), word(b2, b3));
                let t = target.add(2 * (g.bin0 as usize + b as usize));
                if partial {
                    *t = x;
                    *t.add(1) = y;
                } else {
                    if x != 0 {
                        add_global(t, x);
                    }
                    if y != 0 {
                        add_global(t.add(1), y);
                    }
                }
                b += step;
            }
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
/// `rows` holds `n` entries, `bins` `stride` entries per row,
/// `feature_first` `n_cols`, `gpair` every row, and `partials` `segs *
/// total_bins` bins.
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
    // SAFETY: the caller's; one thread per (chunk, feature) owns those bins.
    unsafe {
        let h = partials.add((seg * total_bins + u64::from(ld(feature_first, f))) as usize);
        let mut i = begin;
        while i < end {
            let r = u64::from(ld(rows, i));
            let b = ld(bins, r * u64::from(stride) + f).get();
            if b != sentinel {
                let p = ld(gpair, r);
                let mut a = ld(h, u64::from(b));
                a.x += f64::from(p.x);
                a.y += f64::from(p.y);
                st(h, u64::from(b), a);
            }
            i += 1;
        }
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
/// `tiles` holds one tile per block, `rows` each tile's rows, `row_ptr` and
/// `units` every row, `bins` every stored entry, and `acc`/`partials` every
/// slot the tiles name.
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
    // SAFETY: the caller's; the updates are atomic.
    unsafe {
        let tile = ld(tiles, u64::from(thread::blockIdx_x()));
        let target = tile.histogram(acc, partials, total_bins);
        let tid = thread::threadIdx_x();
        let (lane, warp) = (u64::from(tid & 31), u64::from(tid >> 5));
        let warps = u64::from(thread::blockDim_x() / 32);
        let mut i = warp;
        while i < u64::from(tile.count) {
            let r = u64::from(ld(rows, tile.begin + i));
            let p = ld(units, r);
            let mut at = ld(row_ptr, r) + lane;
            let end = ld(row_ptr, r + 1);
            while at < end {
                let b = u64::from(ld(bins, at).get());
                add_global(target.add((b * 2) as usize), p.x as u64);
                add_global(target.add((b * 2 + 1) as usize), p.y as u64);
                at += 32;
            }
            i += warps;
        }
    }
}

/// One thread per CPU chunk chains each stored CSR bin in row order. Unlike
/// feature sweeps this visits the stored entries once even for many mostly
/// absent features.
///
/// # Safety
///
/// `rows` holds `n` entries, `row_ptr` and `gpair` every row, `bins` every
/// stored entry, and `partials` `segs * total_bins` bins (zeroed).
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
    // SAFETY: the caller's; one thread per chunk owns its partial.
    unsafe {
        let mut seg = grid_index();
        while seg < segs {
            let begin = seg * seg_rows;
            let end = (begin + seg_rows).min(n);
            let h = partials.add((seg * total_bins) as usize);
            let mut i = begin;
            while i < end {
                let r = u64::from(ld(rows, i));
                let p = ld(gpair, r);
                let mut at = ld(row_ptr, r);
                let stop = ld(row_ptr, r + 1);
                while at < stop {
                    let b = u64::from(ld(bins, at).get());
                    let mut a = ld(h, b);
                    a.x += f64::from(p.x);
                    a.y += f64::from(p.y);
                    st(h, b, a);
                    at += 1;
                }
                i += 1;
            }
            seg += grid_threads();
        }
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
    /// With flag 2, `table` covers the feature's bins from `table_at`.
    #[inline(always)]
    unsafe fn left(self, table: *const u8, b: Option<u32>) -> bool {
        match b {
            None => self.flags & 1 != 0,
            // SAFETY: the caller's.
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
/// Every thread of the block calls this.
#[inline(always)]
unsafe fn block_sum_u32(mut count: u32) -> u32 {
    static mut WARP_SUM: SharedArray<u32, 32> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x();
    let mut delta = 16;
    while delta > 0 {
        count += warp::shuffle_down_sync(FULL, count, delta);
        delta >>= 1;
    }
    // SAFETY: lane 0 of each warp writes its own slot before the barrier,
    // thread 0 reads them after it.
    unsafe {
        let warp_sum = SharedArray::as_raw_mut_ptr(&raw mut WARP_SUM);
        if tid.is_multiple_of(32) {
            st(warp_sum, u64::from(tid >> 5), count);
        }
        thread::sync_threads();
        let mut total = 0;
        if tid == 0 {
            let mut w = 0;
            while w < thread::blockDim_x().div_ceil(32) {
                total += ld(warp_sum, u64::from(w));
                w += 1;
            }
        }
        total
    }
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
/// `ptiles` holds one code per block, `segs` and `rules` every split they
/// name, `rows` and `flags` every segment's rows, `cols` `n_rows` bins per
/// feature, `table` every tabled rule's bins, and `tile_left` one entry per
/// block.
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
    // SAFETY: the caller's; each thread routes its own rows.
    unsafe {
        let (s, off, begin, end) = part_tile(tiles.segs, tiles.ptiles);
        let rule = ld(tiles.rules, u64::from(s));
        let col = cols.add((u64::from(rule.feature) * n_rows) as usize);
        let mut count = 0;
        let mut i = begin + u64::from(thread::threadIdx_x());
        while i < end {
            let b = ld(col, u64::from(ld(rows, off + i))).get();
            let left = rule.left(table, (b != sentinel).then_some(b));
            st(flags, off + i, u8::from(left));
            count += u32::from(left);
            i += u64::from(thread::blockDim_x());
        }
        let total = block_sum_u32(count);
        if thread::threadIdx_x() == 0 {
            st(tiles.tile_left, u64::from(thread::blockIdx_x()), total);
        }
    }
}

/// Sparse routing locates the row's first stored bin in the feature's global
/// range, as the CPU's `feature_bin` does; absence is missing, never bin 0.
///
/// # Safety
///
/// As [`route_count`], with `row_ptr` covering every row, `bins` every
/// stored entry, and `first` `n_cols + 1` entries.
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
    // SAFETY: the caller's; each thread routes its own rows.
    unsafe {
        let (s, off, begin, end) = part_tile(tiles.segs, tiles.ptiles);
        let rule = ld(tiles.rules, u64::from(s));
        let (fs, fe) = (
            ld(first, u64::from(rule.feature)),
            ld(first, u64::from(rule.feature) + 1),
        );
        let mut count = 0;
        let mut i = begin + u64::from(thread::threadIdx_x());
        while i < end {
            let r = u64::from(ld(rows, off + i));
            let mut b = None;
            let mut at = ld(row_ptr, r);
            let stop = ld(row_ptr, r + 1);
            while at < stop {
                let global = ld(bins, at).get();
                if global >= fs && global < fe {
                    b = Some(global - fs);
                    break;
                }
                at += 1;
            }
            let left = rule.left(table, b);
            st(flags, off + i, u8::from(left));
            count += u32::from(left);
            i += u64::from(thread::blockDim_x());
        }
        let total = block_sum_u32(count);
        if thread::threadIdx_x() == 0 {
            st(tiles.tile_left, u64::from(thread::blockIdx_x()), total);
        }
    }
}

/// Per split (one thread each): the exclusive prefix of its tiles' left
/// counts, in tile order, and its left total.
///
/// # Safety
///
/// `split_tiles` and `left_len` hold `n_splits` entries, and `tile_left`
/// every tile they name.
#[kernel]
pub unsafe extern "C" fn route_scan(
    split_tiles: *const U32x2,
    n_splits: u32,
    tile_left: *mut u32,
    left_len: *mut u32,
) {
    let s = grid_index();
    if s < u64::from(n_splits) {
        // SAFETY: the caller's; one thread per split owns its tiles.
        unsafe {
            let t = ld(split_tiles, s);
            let mut run = 0;
            let mut k = t.x;
            while k < t.x + t.y {
                let c = ld(tile_left, u64::from(k));
                st(tile_left, u64::from(k), run);
                run += c;
                k += 1;
            }
            st(left_len, s, run);
        }
    }
}

/// Stable scatter of each tile's rows into `scratch` by their flags: left
/// rows to the segment's front in row order, right rows after all left
/// ones. Rows are ranked a block-width round at a time with warp ballots.
///
/// # Safety
///
/// As [`route_count`], with `tile_left` holding the exclusive prefixes and
/// `left_len` every split's left total; `scratch` covers every segment.
#[kernel]
pub unsafe extern "C" fn route_scatter(
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
    // SAFETY: the caller's; lane 0 of each warp writes its counts before
    // the barrier, every thread reads them after it and before the next
    // round's barrier; each row lands in its own scratch slot.
    unsafe {
        let warp_left = SharedArray::as_raw_mut_ptr(&raw mut WARP_LEFT);
        let warp_valid = SharedArray::as_raw_mut_ptr(&raw mut WARP_VALID);
        let (s, off, begin, end) = part_tile(segs, ptiles);
        let block = u64::from(thread::blockIdx_x());
        let tid = thread::threadIdx_x();
        let mut left_at = off + u64::from(ld(tile_left, block));
        let mut right_at =
            off + u64::from(ld(left_len, u64::from(s))) + (begin - u64::from(ld(tile_left, block)));
        let (lane, warp) = (tid & 31, tid >> 5);
        let warps = thread::blockDim_x().div_ceil(32);
        let below = (1u32 << lane) - 1;
        let mut base = begin;
        while base < end {
            let i = base + u64::from(tid);
            let valid = i < end;
            let (r, left) = if valid {
                (ld(rows, off + i), ld(flags, off + i) != 0)
            } else {
                (0, false)
            };
            let lmask = warp::ballot_sync(FULL, left);
            let vmask = warp::ballot_sync(FULL, valid);
            if lane == 0 {
                st(warp_left, u64::from(warp), lmask.count_ones());
                st(warp_valid, u64::from(warp), vmask.count_ones());
            }
            thread::sync_threads();
            // Every warp scans the per-warp counts itself (lane `w` holds
            // warp `w`'s) with shuffles: the earlier warps' left and right
            // counts and the round's totals, without another barrier.
            let (l, v) = if lane < warps {
                (
                    ld(warp_left, u64::from(lane)),
                    ld(warp_valid, u64::from(lane)),
                )
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
                st(scratch, left_at + u64::from(rank), r);
            } else if valid {
                let rank = rbefore + (vmask & !lmask & below).count_ones();
                st(scratch, right_at + u64::from(rank), r);
            }
            left_at += u64::from(ltotal);
            right_at += u64::from(rtotal);
            thread::sync_threads();
            base += u64::from(thread::blockDim_x());
        }
    }
}

/// Copy each tile's span of `scratch` back to `rows`.
///
/// # Safety
///
/// `ptiles` holds one code per block, `segs` every split they name, and
/// `scratch` and `rows` every segment.
#[kernel]
pub unsafe extern "C" fn route_copy(
    segs: *const u64,
    ptiles: *const u64,
    scratch: *const u32,
    rows: *mut u32,
) {
    // SAFETY: the caller's; each thread copies its own rows.
    unsafe {
        let (_, off, begin, end) = part_tile(segs, ptiles);
        let mut i = begin + u64::from(thread::threadIdx_x());
        while i < end {
            st(rows, off + i, ld(scratch, off + i));
            i += u64::from(thread::blockDim_x());
        }
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

/// Exact accumulators to `f64`: slot `nodes[k].x` into output slot
/// `nodes[k].y`, node `k = blockIdx.y`. Equals the CPU's sum, which is exact
/// too.
///
/// # Safety
///
/// `nodes` holds one entry per grid row, and `acc` and `out` the slots
/// they name.
#[kernel]
pub unsafe extern "C" fn finalize_exact(
    acc: *const u64,
    nodes: *const U32x2,
    total_bins: u64,
    grad_value: f64,
    hess_value: f64,
    out: *mut F64x2,
) {
    // SAFETY: the caller's; one thread per output bin.
    unsafe {
        let nd = ld(nodes, u64::from(thread::blockIdx_y()));
        let mut b = grid_index();
        while b < total_bins {
            let a = acc.add(((u64::from(nd.x) * total_bins + b) * 2) as usize);
            let value = F64x2 {
                x: scaled(*a, grad_value),
                y: scaled(*a.add(1), hess_value),
            };
            st(out, u64::from(nd.y) * total_bins + b, value);
            b += grid_threads();
        }
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
/// `nodes` holds one entry per grid row, and `acc` and `out` the slots
/// they name; no slot is the output or parent slot of two nodes, nor both.
#[kernel]
pub unsafe extern "C" fn finalize_exact_sub(
    acc: *const u64,
    nodes: *const u32,
    total_bins: u64,
    grad_value: f64,
    hess_value: f64,
    out: *mut F64x2,
) {
    // SAFETY: the caller's; one thread per output bin and its parent bin.
    unsafe {
        let k = 3 * u64::from(thread::blockIdx_y());
        let (from, to, parent) = (ld(nodes, k), ld(nodes, k + 1), ld(nodes, k + 2));
        let mut b = grid_index();
        while b < total_bins {
            let a = acc.add(((u64::from(from) * total_bins + b) * 2) as usize);
            let child = F64x2 {
                x: scaled(*a, grad_value),
                y: scaled(*a.add(1), hess_value),
            };
            st(out, u64::from(to) * total_bins + b, child);
            if parent != u32::MAX {
                let at = u64::from(parent) * total_bins + b;
                let p = ld(out, at);
                st(
                    out,
                    at,
                    F64x2 {
                        x: p.x - child.x,
                        y: p.y - child.y,
                    },
                );
            }
            b += grid_threads();
        }
    }
}

/// Integer chunk partials to `f64`, per bin in chunk order: node `k =
/// blockIdx.y` reads partial slots `[x, x + y)` into output slot `z`,
/// copying the first when `w` (its first chunk) and adding the rest, the
/// CPU's copy-then-add.
///
/// # Safety
///
/// `nodes` holds one entry per grid row, and `partials` and `out` the slots
/// they name.
#[kernel]
pub unsafe extern "C" fn reduce_chunks(
    partials: *const u64,
    nodes: *const U32x4,
    total_bins: u64,
    grad_value: f64,
    hess_value: f64,
    out: *mut F64x2,
) {
    // SAFETY: the caller's; one thread per output bin.
    unsafe {
        let nd = ld(nodes, u64::from(thread::blockIdx_y()));
        let mut b = grid_index();
        while b < total_bins {
            let o = out.add((u64::from(nd.z) * total_bins + b) as usize);
            let (mut g, mut h, mut s) = if nd.w != 0 {
                let p = partials.add(((u64::from(nd.x) * total_bins + b) * 2) as usize);
                (scaled(*p, grad_value), scaled(*p.add(1), hess_value), 1)
            } else {
                ((*o).x, (*o).y, 0)
            };
            while s < nd.y {
                let p = partials.add(((u64::from(nd.x + s) * total_bins + b) * 2) as usize);
                g += scaled(*p, grad_value);
                h += scaled(*p.add(1), hess_value);
                s += 1;
            }
            *o = F64x2 { x: g, y: h };
            b += grid_threads();
        }
    }
}

/// `f64` chain partials of `segs` chunks into `out`, per bin in chunk order
/// (the first copied when `init`).
///
/// # Safety
///
/// `partials` holds `segs * total_bins` bins and `out` `total_bins`.
#[kernel]
pub unsafe extern "C" fn reduce_chains(
    partials: *const F64x2,
    segs: u64,
    total_bins: u64,
    init: i32,
    out: *mut F64x2,
) {
    // SAFETY: the caller's; one thread per output bin.
    unsafe {
        let mut b = grid_index();
        while b < total_bins {
            let (mut a, mut s) = if init != 0 {
                (ld(partials, b), 1)
            } else {
                (ld(out, b), 0)
            };
            while s < segs {
                let p = ld(partials, s * total_bins + b);
                a.x += p.x;
                a.y += p.y;
                s += 1;
            }
            st(out, b, a);
            b += grid_threads();
        }
    }
}

// ---------------------------------------------------------------------------
// Resident split search: histograms stay in device slots.

/// Each `(parent, built)` slot pair's parent becomes `parent - built`, the
/// host's `subtract_in_place`.
///
/// # Safety
///
/// `pairs` holds `2 * n_pairs` slots, each in `pool` (`total_bins` bins
/// per slot), no parent twice.
#[kernel]
pub unsafe extern "C" fn subtract_hists(
    pool: *mut F64x2,
    pairs: *const u32,
    n_pairs: u64,
    total_bins: u64,
) {
    // SAFETY: the caller's; one thread per parent bin.
    unsafe {
        let mut i = grid_index();
        while i < n_pairs * total_bins {
            let (p, b) = (i / total_bins, i % total_bins);
            let parent = u64::from(ld(pairs, 2 * p)) * total_bins + b;
            let c = ld(pool, u64::from(ld(pairs, 2 * p + 1)) * total_bins + b);
            let a = ld(pool, parent);
            st(
                pool,
                parent,
                F64x2 {
                    x: a.x - c.x,
                    y: a.y - c.y,
                },
            );
            i += grid_threads();
        }
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
        // SAFETY: the caller's.
        unsafe {
            Self {
                lambda: reg.lambda,
                alpha: reg.alpha,
                max_delta_step: reg.max_delta_step,
                min_child_weight: reg.min_child_weight,
                root_gain: ld(params, 3 * request),
                lower: ld(params, 3 * request + 1),
                upper: ld(params, 3 * request + 2),
                dir,
            }
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
#[inline(always)]
fn chain_row(w: u32) -> *mut F64x2 {
    static mut CHAIN: SharedArray<F64x2, { SCAN_WARPS * 32 }> = SharedArray::UNINIT;
    // SAFETY: an address within the array (`w < SCAN_WARPS`).
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
/// `tasks`, `meta` and `acc` hold `n_tasks` tasks' entries, `totals` and
/// `params` (root gain, lower, upper) every request, and `pool` every slot
/// (`total_bins` bins each).
#[kernel]
#[launch_bounds(32)]
pub unsafe extern "C" fn scan_splits(
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
    // SAFETY: the caller's; each warp owns its chain row, staged by its
    // lanes, chained by lane 0 and read by the lanes, each step between
    // warp barriers every lane reaches; every lane reaches the shuffles.
    unsafe {
        let lane = thread::threadIdx_x() & 31;
        let w = thread::threadIdx_x() >> 5;
        let chain = chain_row(w);
        let warps = u64::from(thread::gridDim_x()) * SCAN_WARPS as u64;
        let mut t = u64::from(thread::blockIdx_x()) * SCAN_WARPS as u64 + u64::from(w);
        while t < n_tasks {
            let request = u64::from(ld(tasks, 4 * t));
            let f = u64::from(ld(tasks, 4 * t + 1));
            let slot = u64::from(ld(tasks, 4 * t + 2));
            let reg = ScanReg::new(regularization, params, request, ld(tasks, 4 * t + 3) as i32);
            let total = ld(totals, request);
            let first = ld(feature_first, f);
            let len = u64::from(ld(feature_first, f + 1) - first);
            let bins = pool.add((slot * total_bins + u64::from(first)) as usize);
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
                    let bin = || ld(bins, if backward { len - 1 - at } else { at });
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
                            st(chain, u64::from(lane), bin());
                        }
                        warp::sync_mask(FULL);
                        if lane == 0 {
                            let mut i = 0;
                            while i < n {
                                let b = ld(chain, u64::from(i));
                                g += b.x;
                                h += b.y;
                                st(chain, u64::from(i), F64x2 { x: g, y: h });
                                i += 1;
                            }
                        }
                        warp::sync_mask(FULL);
                        let a = if lane < n {
                            ld(chain, u64::from(lane))
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
                st(meta, 4 * t, status);
                st(meta, 4 * t + 1, bin);
                st(meta, 4 * t + 2, top.to_bits());
                st(meta, 4 * t + 3, u32::from(backward));
                st(acc, t, winner);
            }
            t += warps;
        }
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
        /// As [`hist_tile`], with 16 bytes of dynamic shared memory per bin
        /// of every group.
        #[kernel]
        #[launch_bounds(512)]
        pub unsafe extern "C" fn $hist_shared(
            bins: *const $bin,
            feature_first: *const u32,
            rows: *const u32,
            units: *const I64x2,
            acc: *mut u64,
            partials: *mut u64,
            work: TileWork,
        ) {
            // SAFETY: the caller's.
            unsafe {
                hist_tile::<$bin, true>(bins, feature_first, rows, units, acc, partials, work)
            }
        }

        /// [`hist_tile`] with global atomics.
        ///
        /// # Safety
        ///
        /// As [`hist_tile`].
        #[kernel]
        #[launch_bounds(512)]
        pub unsafe extern "C" fn $hist_global(
            bins: *const $bin,
            feature_first: *const u32,
            rows: *const u32,
            units: *const I64x2,
            acc: *mut u64,
            partials: *mut u64,
            work: TileWork,
        ) {
            // SAFETY: the caller's.
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
        pub unsafe extern "C" fn $hist_chain(
            bins: *const $bin,
            feature_first: *const u32,
            rows: *const u32,
            gpair: *const F32x2,
            partials: *mut F64x2,
            work: ChainWork,
        ) {
            // SAFETY: the caller's.
            unsafe { hist_chain(bins, feature_first, rows, gpair, partials, work) }
        }

        /// [`route_count`] for this bin width.
        ///
        /// # Safety
        ///
        /// As [`route_count`].
        #[kernel]
        pub unsafe extern "C" fn $route_count(
            cols: *const $bin,
            n_rows: u64,
            sentinel: u32,
            table: *const u8,
            rows: *const u32,
            flags: *mut u8,
            tiles: PartTiles,
        ) {
            // SAFETY: the caller's.
            unsafe { route_count(cols, n_rows, sentinel, table, rows, flags, tiles) }
        }

        /// [`hist_sparse`] for this bin width.
        ///
        /// # Safety
        ///
        /// As [`hist_sparse`].
        #[kernel]
        #[launch_bounds(512)]
        pub unsafe extern "C" fn $hist_sparse(
            bins: *const $bin,
            row_ptr: *const u64,
            rows: *const u32,
            units: *const I64x2,
            acc: *mut u64,
            partials: *mut u64,
            work: SparseTiles,
        ) {
            // SAFETY: the caller's.
            unsafe { hist_sparse(bins, row_ptr, rows, units, acc, partials, work) }
        }

        /// [`hist_sparse_chain`] for this bin width.
        ///
        /// # Safety
        ///
        /// As [`hist_sparse_chain`].
        #[kernel]
        pub unsafe extern "C" fn $hist_sparse_chain(
            bins: *const $bin,
            row_ptr: *const u64,
            rows: *const u32,
            gpair: *const F32x2,
            partials: *mut F64x2,
            chunks: Chunks,
        ) {
            // SAFETY: the caller's.
            unsafe { hist_sparse_chain(bins, row_ptr, rows, gpair, partials, chunks) }
        }

        /// [`route_sparse`] for this bin width.
        ///
        /// # Safety
        ///
        /// As [`route_sparse`].
        #[kernel]
        pub unsafe extern "C" fn $route_sparse(
            bins: *const $bin,
            row_ptr: *const u64,
            first: *const u32,
            table: *const u8,
            rows: *const u32,
            flags: *mut u8,
            tiles: PartTiles,
        ) {
            // SAFETY: the caller's.
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
/// of consecutive rows: `left_len[n_splits + s]` holds bit 0 for split
/// `s`'s left child and bit 1 for its right one. A child's rows ascend
/// without repeats, so it is a run when its last row is its first plus its
/// length minus one; an empty child counts as a run.
///
/// # Safety
///
/// `segs` holds `n_splits` (offset, length) pairs naming segments of
/// `rows`, and `left_len` `2 * n_splits` entries, the first `n_splits` the
/// left children's lengths.
#[kernel]
pub unsafe extern "C" fn route_runs(
    segs: *const u64,
    rows: *const u32,
    n_splits: u32,
    left_len: *mut u32,
) {
    let s = grid_index();
    if s < u64::from(n_splits) {
        // SAFETY: the caller's; one thread per split writes its own word.
        unsafe {
            let off = ld(segs, 2 * s);
            let len = ld(segs, 2 * s + 1);
            let left = u64::from(ld(left_len, s));
            let runs = u32::from(is_run(rows, off, left))
                | u32::from(is_run(rows, off + left, len - left)) << 1;
            st(left_len, u64::from(n_splits) + s, runs);
        }
    }
}

/// Whether the `n` ascending, distinct rows at `rows[begin..]` are
/// consecutive.
///
/// # Safety
///
/// `rows` holds `begin + n` entries.
#[inline(always)]
unsafe fn is_run(rows: *const u32, begin: u64, n: u64) -> bool {
    // SAFETY: the caller's.
    n == 0 || unsafe { u64::from(ld(rows, begin + n - 1)) - u64::from(ld(rows, begin)) == n - 1 }
}
