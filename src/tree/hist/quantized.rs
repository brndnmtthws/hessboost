//! Quantized-gradient histograms: the opt-in `use_quantized_grad` training
//! mode (LightGBM's quantized training; Shi et al., "Quantized Training of
//! Gradient Boosting Decision Trees", NeurIPS 2022, arXiv:2207.09682). Beyond
//! XGBoost; the default float path never reaches this module.
//!
//! For each tree the gradients `g` and Hessians `h` are mapped to small
//! integers with iteration-global scales (LightGBM `gradient_discretizer.cpp`):
//!
//! * `δg = max|g| / ⌊Q/2⌋`, `ĝ = round(g / δg)`, so `|ĝ| ≤ ⌊Q/2⌋`;
//! * `δh = max|h| / Q`, `ĥ = round(h / δh)`, so `|ĥ| ≤ Q`, or, when every
//!   Hessian is equal, `δh = h` and `ĥ = 1` exactly.
//!
//! Rounding is stochastic by default (`E[ĝ δg] = g`), drawn from a
//! counter-based stream keyed by the row index, so the result does not depend
//! on the thread count. Histogram bins accumulate `(Ĝ, Ĥ)` as one packed
//! integer, gradient in the high half and Hessian in the low half, whose width
//! is chosen per node from its row count: a bin of `n` rows is bounded by
//! `n · Q`, so small nodes add 32-bit words, larger ones 64-bit, and only
//! above `2³¹ / Q` rows 128-bit. Integer sums are exact, so the parallel and
//! serial builds agree bit for bit and the sibling subtraction is exact.
//! Split search reads the dequantized sums `(Ĝ δg, Ĥ δh)`; the gain formula is
//! unchanged.
//!
//! The scales are `f32` values, so every dequantized bin and every prefix sum
//! of bins is an exact multiple of the scale's last mantissa bit while the
//! integer sums stay below `2²⁹`: the split evaluator's exact
//! "forward sum equals node total" missing-value test then holds exactly as it
//! does for the integers.

use super::{
    BinIndex, PARALLEL_THRESHOLD, PREFETCH_ROWS, REDUCE_BINS, ROWS_PER_TASK, TILE_ROWS,
    contiguous_range, feature_blocks, feature_slices, prefetch_bins,
};
use crate::config::TrainingParams;
use crate::data::ghist::{Bins, GHistIndex};
use crate::objective::GradPair;
use crate::rng::{GOLDEN, mix64};
use crate::tree::gain::GradStats;
use rayon::prelude::*;
use std::sync::Arc;

/// Rows per parallel quantization task.
const QUANTIZE_CHUNK: usize = 8192;
/// Bytes of histogram a dense feature block may span, so the block stays in
/// L1 while a row tile is accumulated into it (the float path's 64 KiB).
const BLOCK_BYTES: usize = 64 * 1024;

/// Stream salts separating the gradient and Hessian rounding variates.
const GRAD_STREAM: u64 = 0x6772_6164_5F71_6E74;
const HESS_STREAM: u64 = 0x6865_7373_5F71_6E74;

/// Uniform variate in `[0, 1)` for `row` of the stream keyed by `key`
/// (SplitMix64 at position `row + 1`, top 53 bits).
#[inline]
fn uniform(key: u64, row: usize) -> f64 {
    let bits = mix64(key.wrapping_add((row as u64).wrapping_add(1).wrapping_mul(GOLDEN)));
    (bits >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Round `x` to an adjacent integer toward or away from zero: truncating
/// `x + u` (`x ≥ 0`) or `x − u` (`x < 0`) returns `⌈x⌉` with probability
/// `frac(x)` when `u ~ U[0, 1)`, and rounds half away from zero when
/// `u = 0.5`. Non-finite input maps to 0 (`as` saturates NaN to 0).
#[inline]
fn round_signed(x: f64, u: f64) -> i32 {
    if x >= 0.0 {
        (x + u) as i32
    } else {
        (x - u) as i32
    }
}

/// Split one row's packed value `ĝ·2¹⁶ + ĥ` into `(ĝ, ĥ)`.
#[inline(always)]
fn row_parts(p: i32) -> (i64, i64) {
    let h = i64::from(p as i16);
    (i64::from((p - h as i32) >> 16), h)
}

/// One tree's quantized gradients and their dequantization scales.
#[derive(Debug)]
pub(crate) struct QuantizedGradients {
    /// Per row, `ĝ·2¹⁶ + ĥ` (both signed, `|ĝ|, |ĥ| < 2¹⁵`).
    packed: Vec<i32>,
    /// `δg`, an `f32` value widened exactly.
    grad_scale: f64,
    /// `δh`, an `f32` value widened exactly.
    hess_scale: f64,
    /// Bound on `max(|ĝ|, |ĥ|)` of any row, which sizes histogram widths.
    row_bound: u64,
}

impl QuantizedGradients {
    /// Quantize `gpair` with `bins` levels, stochastically or to nearest,
    /// drawing rounding variates from the stream `seed`.
    fn quantize(gpair: &[GradPair], bins: usize, stochastic: bool, seed: u64) -> Self {
        debug_assert!((2..=127).contains(&bins), "validated by TrainingParams");
        // Max |g|, max |h|, and whether every Hessian equals the first. `max`
        // and `&&` are exact and order-independent, so the chunked reduction
        // is deterministic.
        let first_hess = gpair.first().map_or(0.0, |gp| gp.hess);
        let (max_grad, max_hess, constant) = gpair
            .par_chunks(QUANTIZE_CHUNK)
            .map(|chunk| {
                chunk.iter().fold((0f32, 0f32, true), |(g, h, c), gp| {
                    (
                        g.max(gp.grad.abs()),
                        h.max(gp.hess.abs()),
                        c && gp.hess == first_hess,
                    )
                })
            })
            .reduce(
                || (0.0, 0.0, true),
                |(g1, h1, c1), (g2, h2, c2)| (g1.max(g2), h1.max(h2), c1 && c2),
            );

        let half = (bins / 2) as i32;
        let levels = bins as i32;
        // The correctly rounded `f32` quotient, floored at the least positive
        // `f32`: a nonzero maximum of at most `divisor · 2⁻¹⁵⁰` would otherwise
        // underflow to a zero scale and erase every value. The floor only
        // raises the scale, so the quantized values stay within their bound.
        let scale_of = |max: f32, divisor: i32| {
            if max > 0.0 {
                (max / divisor as f32).max(f32::from_bits(1))
            } else {
                0.0
            }
        };
        let grad_scale = scale_of(max_grad, half);
        let hess_scale = if constant {
            first_hess
        } else {
            scale_of(max_hess, levels)
        };
        let inverse = |scale: f32| {
            if scale > 0.0 && scale.is_finite() {
                1.0 / f64::from(scale)
            } else {
                0.0
            }
        };
        let (inv_grad, inv_hess) = (inverse(grad_scale), inverse(hess_scale));
        let (grad_key, hess_key) = (mix64(seed ^ GRAD_STREAM), mix64(seed ^ HESS_STREAM));

        let mut packed = vec![0i32; gpair.len()];
        packed
            .par_chunks_mut(QUANTIZE_CHUNK)
            .zip(gpair.par_chunks(QUANTIZE_CHUNK))
            .enumerate()
            .for_each(|(chunk, (out, grads))| {
                let first = chunk * QUANTIZE_CHUNK;
                for (i, (o, gp)) in out.iter_mut().zip(grads).enumerate() {
                    let row = first + i;
                    let (ug, uh) = if stochastic {
                        (uniform(grad_key, row), uniform(hess_key, row))
                    } else {
                        (0.5, 0.5)
                    };
                    let g = round_signed(f64::from(gp.grad) * inv_grad, ug).clamp(-half, half);
                    let h = if constant {
                        1
                    } else {
                        round_signed(f64::from(gp.hess) * inv_hess, uh).clamp(-levels, levels)
                    };
                    *o = (g << 16) + h;
                }
            });

        QuantizedGradients {
            packed,
            grad_scale: f64::from(grad_scale),
            hess_scale: f64::from(hess_scale),
            row_bound: if constant { half.max(1) } else { levels } as u64,
        }
    }

    /// The narrowest packed width that holds any bin of `rows` rows: each
    /// half must stay below `2^(s−1)` for a half width `s`.
    fn width(&self, rows: usize) -> Width {
        let bound = rows as u64 * self.row_bound;
        if bound < 1 << 15 {
            Width::W32
        } else if bound < 1 << 31 {
            Width::W64
        } else {
            Width::W128
        }
    }

    /// The dequantized statistics of one row.
    #[cfg(test)]
    fn row_stats(&self, row: usize) -> GradStats {
        let (g, h) = row_parts(self.packed[row]);
        self.dequantize_parts(g, h)
    }

    #[inline]
    fn dequantize_parts(&self, g: i64, h: i64) -> GradStats {
        GradStats::new(g as f64 * self.grad_scale, h as f64 * self.hess_scale)
    }

    /// Dequantized statistics of `rows` (exact integer sums, then scaled).
    fn node_stats(&self, rows: &[u32]) -> GradStats {
        let (g, h) = rows.iter().fold((0i64, 0i64), |(g, h), &r| {
            let (rg, rh) = row_parts(self.packed[r as usize]);
            (g + rg, h + rh)
        });
        self.dequantize_parts(g, h)
    }

    /// Dequantize a node histogram for split evaluation.
    fn dequantize(&self, hist: &QuantHist) -> Vec<GradStats> {
        let mut out = Vec::new();
        self.dequantize_into(hist, &mut out);
        out
    }

    /// Dequantize into `out`, reusing its allocation.
    fn dequantize_into(&self, hist: &QuantHist, out: &mut Vec<GradStats>) {
        fn map<A: Packed>(q: &QuantizedGradients, bins: &[A], out: &mut Vec<GradStats>) {
            out.clear();
            out.extend(bins.iter().map(|b| {
                let (g, h) = b.parts();
                q.dequantize_parts(g, h)
            }));
        }
        match hist {
            QuantHist::W32(b) => map(self, b, out),
            QuantHist::W64(b) => map(self, b, out),
            QuantHist::W128(b) => map(self, b, out),
        }
    }

    /// Build the histogram of `rows` at the width their count requires.
    fn build(&self, ghist: &GHistIndex, rows: &[u32]) -> QuantHist {
        match self.width(rows.len()) {
            Width::W32 => QuantHist::W32(self.build_typed(ghist, rows)),
            Width::W64 => QuantHist::W64(self.build_typed(ghist, rows)),
            Width::W128 => QuantHist::W128(self.build_typed(ghist, rows)),
        }
    }

    /// The histogram of `rows` with `A` accumulators, parallelized like the
    /// float backend: per-feature columns for a contiguous row range, row
    /// chunks reduced in chunk order otherwise. Integer sums are exact, so
    /// every strategy gives the same bins.
    fn build_typed<A: Packed>(&self, ghist: &GHistIndex, rows: &[u32]) -> Vec<A> {
        let total = ghist.total_bins();
        let threads = rayon::current_num_threads();
        if threads <= 1 || rows.len() < PARALLEL_THRESHOLD {
            let mut out = vec![A::default(); total];
            self.accumulate_narrow(ghist, rows, &mut out);
            return out;
        }

        if let (Some(columns), Some(range)) = (ghist.column_bins(), contiguous_range(rows)) {
            let mut out = vec![A::default(); total];
            let values: Vec<A> = self.packed[range.clone()]
                .iter()
                .map(|&p| A::from_row(p))
                .collect();
            let n_rows = ghist.n_rows();
            feature_slices(ghist, &mut out, 1)
                .into_par_iter()
                .enumerate()
                .for_each(|(f, (fs, slice))| {
                    let span = range.clone();
                    match columns {
                        Bins::U16(c) => {
                            accumulate_column(&c[f * n_rows..][span], fs, &values, slice);
                        }
                        Bins::U32(c) => {
                            accumulate_column(&c[f * n_rows..][span], fs, &values, slice);
                        }
                    }
                });
            return out;
        }

        let tasks = threads.min(rows.len() / ROWS_PER_TASK);
        let grain = rows.len().div_ceil(tasks);
        // Each chunk's partial histogram only needs the width of its own row
        // count, which is narrower than the node's for large nodes.
        match self.width(grain) {
            Width::W32 => self.reduce_chunks::<A, i32>(ghist, rows, grain),
            Width::W64 => self.reduce_chunks::<A, i64>(ghist, rows, grain),
            Width::W128 => self.reduce_chunks::<A, i128>(ghist, rows, grain),
        }
    }

    fn reduce_chunks<A: Packed, P: Packed>(
        &self,
        ghist: &GHistIndex,
        rows: &[u32],
        grain: usize,
    ) -> Vec<A> {
        let total = ghist.total_bins();
        let partials: Vec<Vec<P>> = rows
            .par_chunks(grain)
            .map(|chunk| {
                let mut local = vec![P::default(); total];
                self.accumulate_narrow(ghist, chunk, &mut local);
                local
            })
            .collect();
        let mut out = vec![A::default(); total];
        out.par_chunks_mut(REDUCE_BINS)
            .enumerate()
            .for_each(|(i, out)| {
                let start = i * REDUCE_BINS;
                for partial in &partials {
                    for (o, &p) in out.iter_mut().zip(&partial[start..]) {
                        *o = o.add(A::repack(p));
                    }
                }
            });
        out
    }

    /// Accumulate `rows` into `out` (added, not reset) through 32-bit
    /// scratch histograms: runs of rows small enough for 32-bit bins are
    /// summed in a histogram half the size of a 64-bit one, then added into
    /// `out`. The narrower bins keep more of the histogram in L1.
    fn accumulate_narrow<A: Packed>(&self, ghist: &GHistIndex, rows: &[u32], out: &mut [A]) {
        let run = ((1 << 15) - 1) / self.row_bound as usize;
        if A::BITS == 32 || rows.len() <= run {
            accumulate(ghist, rows, &self.packed, out);
            return;
        }
        let mut scratch = vec![0i32; out.len()];
        for chunk in rows.chunks(run) {
            accumulate(ghist, chunk, &self.packed, &mut scratch);
            for (o, s) in out.iter_mut().zip(&mut scratch) {
                *o = o.add(A::repack(*s));
                *s = 0;
            }
        }
    }
}

/// Packed accumulator widths, narrowest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Width {
    W32,
    W64,
    W128,
}

/// A quantized node histogram: packed `(Ĝ, Ĥ)` per global bin.
#[derive(Debug, PartialEq)]
pub(crate) enum QuantHist {
    W32(Vec<i32>),
    W64(Vec<i64>),
    W128(Vec<i128>),
}

impl QuantHist {
    fn width(&self) -> Width {
        match self {
            QuantHist::W32(_) => Width::W32,
            QuantHist::W64(_) => Width::W64,
            QuantHist::W128(_) => Width::W128,
        }
    }

    /// Re-pack into at least `width`.
    fn widen_to(&mut self, width: Width) {
        fn convert<A: Packed, P: Packed>(bins: &[P]) -> Vec<A> {
            bins.iter().map(|&p| A::repack(p)).collect()
        }
        if self.width() >= width {
            return;
        }
        *self = match (&*self, width) {
            (QuantHist::W32(b), Width::W64) => QuantHist::W64(convert(b)),
            (QuantHist::W32(b), _) => QuantHist::W128(convert(b)),
            (QuantHist::W64(b), _) => QuantHist::W128(convert(b)),
            (QuantHist::W128(_), _) => unreachable!("W128 is the widest width"),
        };
    }

    /// `self − child`, in place, at the wider of the two widths. Exact: both
    /// halves of every bin subtract as integers.
    fn subtract(&mut self, child: &QuantHist) {
        fn sub<A: Packed>(parent: &mut [A], child: &QuantHist) {
            fn typed<A: Packed, P: Packed>(parent: &mut [A], child: &[P]) {
                debug_assert_eq!(parent.len(), child.len());
                for (p, &c) in parent.iter_mut().zip(child) {
                    *p = p.sub(A::repack(c));
                }
            }
            match child {
                QuantHist::W32(c) => typed(parent, c),
                QuantHist::W64(c) => typed(parent, c),
                QuantHist::W128(c) => typed(parent, c),
            }
        }
        self.widen_to(child.width());
        match self {
            QuantHist::W32(p) => sub(p, child),
            QuantHist::W64(p) => sub(p, child),
            QuantHist::W128(p) => sub(p, child),
        }
    }
}

/// A child node with its dequantized histogram for split evaluation.
pub(crate) type QuantChild = (QuantNode, Vec<GradStats>);

/// A node's quantized histogram together with the tree's quantized gradients,
/// which its children's histograms are built from.
#[derive(Debug)]
pub(crate) struct QuantNode {
    grads: Arc<QuantizedGradients>,
    hist: QuantHist,
}

impl QuantNode {
    /// Quantize the tree's gradients (`params.num_grad_quant_bins` levels,
    /// rounded per `params.stochastic_rounding` from the stream `seed`) and
    /// build the root over `rows`. Returns the node, its dequantized
    /// statistics, and its dequantized histogram.
    pub(crate) fn root(
        ghist: &GHistIndex,
        gpair: &[GradPair],
        rows: &[u32],
        params: &TrainingParams,
        seed: u64,
    ) -> (Self, GradStats, Vec<GradStats>) {
        let grads = Arc::new(QuantizedGradients::quantize(
            gpair,
            params.num_grad_quant_bins,
            params.stochastic_rounding,
            seed,
        ));
        let hist = grads.build(ghist, rows);
        let stats = grads.node_stats(rows);
        let float = grads.dequantize(&hist);
        (QuantNode { grads, hist }, stats, float)
    }

    /// Build both children of this node: the smaller directly, the sibling by
    /// exact subtraction from this node's histogram. Each child comes with its
    /// dequantized histogram for split evaluation; the sibling's reuses
    /// `spare`, the parent's dead float histogram.
    pub(crate) fn children(
        self,
        ghist: &GHistIndex,
        left_rows: &[u32],
        right_rows: &[u32],
        spare: Vec<GradStats>,
    ) -> (QuantChild, QuantChild) {
        let QuantNode {
            grads,
            hist: mut sibling,
        } = self;
        let left_smaller = left_rows.len() <= right_rows.len();
        let small = grads.build(ghist, if left_smaller { left_rows } else { right_rows });
        sibling.subtract(&small);
        let small_float = grads.dequantize(&small);
        let mut sibling_float = spare;
        grads.dequantize_into(&sibling, &mut sibling_float);
        let small = QuantNode {
            grads: Arc::clone(&grads),
            hist: small,
        };
        let sibling = QuantNode {
            grads,
            hist: sibling,
        };
        if left_smaller {
            ((small, small_float), (sibling, sibling_float))
        } else {
            ((sibling, sibling_float), (small, small_float))
        }
    }
}

/// A packed `(G, H)` accumulator: `G·2^s + H` with both halves signed and
/// below `2^(s−1)` in magnitude, `s` half the width. Sums of packed values are
/// the packed sums while both halves stay in range, which the per-node width
/// choice guarantees.
trait Packed: Copy + Default + Send + Sync {
    /// Width in bits.
    const BITS: u32;
    /// Pack `(g, h)`.
    fn from_parts(g: i64, h: i64) -> Self;
    /// Unpack into `(G, H)`.
    fn parts(self) -> (i64, i64);
    /// Re-pack one row's `ĝ·2¹⁶ + ĥ`.
    fn from_row(p: i32) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    /// Re-pack a value of another width.
    #[inline]
    fn repack<P: Packed>(p: P) -> Self {
        let (g, h) = p.parts();
        Self::from_parts(g, h)
    }
}

macro_rules! packed {
    ($ty:ty, $half:ty, $shift:expr) => {
        impl Packed for $ty {
            const BITS: u32 = <$ty>::BITS;
            #[inline(always)]
            fn from_parts(g: i64, h: i64) -> Self {
                ((g as $ty) << $shift) + h as $ty
            }
            #[inline(always)]
            #[allow(clippy::cast_lossless, reason = "the same cast narrows for i128")]
            fn parts(self) -> (i64, i64) {
                // Sign-extend the low half, then the high half is exact.
                let h = self as $half;
                (((self - <$ty>::from(h)) >> $shift) as i64, i64::from(h))
            }
            #[inline(always)]
            fn from_row(p: i32) -> Self {
                let (g, h) = row_parts(p);
                Self::from_parts(g, h)
            }
            #[inline(always)]
            fn add(self, other: Self) -> Self {
                self + other
            }
            #[inline(always)]
            fn sub(self, other: Self) -> Self {
                self - other
            }
        }
    };
}
packed!(i32, i16, 16);
packed!(i64, i32, 32);
packed!(i128, i64, 64);

/// Sequential accumulation of `rows` into `out` (added, not reset).
fn accumulate<A: Packed>(ghist: &GHistIndex, rows: &[u32], packed: &[i32], out: &mut [A]) {
    match ghist.bins() {
        Bins::U16(bins) => accumulate_bins(ghist, bins, rows, packed, out),
        Bins::U32(bins) => accumulate_bins(ghist, bins, rows, packed, out),
    }
}

#[inline(always)]
fn accumulate_bins<A: Packed, B: BinIndex>(
    ghist: &GHistIndex,
    bins: &[B],
    rows: &[u32],
    packed: &[i32],
    out: &mut [A],
) {
    // Establishes the bound used by `add_row`: with `out` covering every bin,
    // the `GHistIndex` invariant (all stored bins < total_bins) makes every
    // histogram index in range.
    assert_eq!(
        out.len(),
        ghist.total_bins(),
        "histogram length must equal the binned index's bin count"
    );
    let add_row = |row_bins: &[B], v: A, out: &mut [A]| {
        for &bin in row_bins {
            // SAFETY: `bin < ghist.total_bins() == out.len()` by the index
            // invariant and the assertion above.
            let slot = unsafe { out.get_unchecked_mut(bin.index()) };
            *slot = slot.add(v);
        }
    };

    if let Some(columns) = ghist.column_bins()
        && let Some(range) = contiguous_range(rows)
    {
        let n_rows = ghist.n_rows();
        let values: Vec<A> = packed[range.clone()]
            .iter()
            .map(|&p| A::from_row(p))
            .collect();
        match columns {
            Bins::U16(columns) => {
                for column in columns.chunks_exact(n_rows) {
                    accumulate_column(&column[range.clone()], 0, &values, out);
                }
            }
            Bins::U32(columns) => {
                for column in columns.chunks_exact(n_rows) {
                    accumulate_column(&column[range.clone()], 0, &values, out);
                }
            }
        }
        return;
    }

    if let Some(stride) = ghist.dense_stride() {
        accumulate_dense(ghist, bins, stride, rows, packed, out, add_row);
    } else {
        let rp = ghist.row_ptr();
        for (i, &r) in rows.iter().enumerate() {
            if let Some(&ahead) = rows.get(i + PREFETCH_ROWS) {
                let ahead = ahead as usize;
                if let (Some(&start), Some(&end)) = (rp.get(ahead), rp.get(ahead + 1)) {
                    prefetch_bins(bins, start, end - start);
                }
            }
            let ri = r as usize;
            add_row(&bins[rp[ri]..rp[ri + 1]], A::from_row(packed[ri]), out);
        }
    }
}

/// Accumulate one feature column (`column[i]` is the bin of `values[i]`'s
/// row) into `slice`, whose first entry is global bin `first_bin`.
#[inline(always)]
fn accumulate_column<A: Packed, B: BinIndex>(
    column: &[B],
    first_bin: usize,
    values: &[A],
    slice: &mut [A],
) {
    for (&bin, &v) in column.iter().zip(values) {
        let slot = &mut slice[bin.index() - first_bin];
        *slot = slot.add(v);
    }
}

/// Dense accumulation tiled by rows and feature blocks, as the float path
/// does, with blocks sized for the accumulator width.
#[inline(always)]
fn accumulate_dense<A: Packed, B: BinIndex>(
    ghist: &GHistIndex,
    bins: &[B],
    stride: usize,
    rows: &[u32],
    packed: &[i32],
    out: &mut [A],
    add_row: impl Fn(&[B], A, &mut [A]),
) {
    let blocks = feature_blocks(ghist, stride, BLOCK_BYTES / std::mem::size_of::<A>());

    let prefetch = |rows: &[u32], i: usize| {
        if let Some(&ahead) = rows.get(i + PREFETCH_ROWS) {
            prefetch_bins(bins, ahead as usize * stride, stride);
        }
    };
    if blocks.len() == 1 {
        for (i, &r) in rows.iter().enumerate() {
            prefetch(rows, i);
            let start = r as usize * stride;
            add_row(
                &bins[start..start + stride],
                A::from_row(packed[r as usize]),
                out,
            );
        }
        return;
    }
    let mut values: Vec<A> = Vec::with_capacity(TILE_ROWS.min(rows.len()));
    for tile in rows.chunks(TILE_ROWS) {
        values.clear();
        values.extend(tile.iter().map(|&r| A::from_row(packed[r as usize])));
        for (block, &(f0, f1)) in blocks.iter().enumerate() {
            for (i, (&r, &v)) in tile.iter().zip(&values).enumerate() {
                if block == 0 {
                    prefetch(tile, i);
                }
                let start = r as usize * stride;
                add_row(&bins[start + f0..start + f1], v, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DMatrix;
    use crate::data::quantile::HistCuts;

    fn gp(grad: f32, hess: f32) -> GradPair {
        GradPair::new(grad, hess)
    }

    /// Deterministic pseudo-random gradients with varying Hessians.
    fn gradients(n: usize, seed: u64) -> Vec<GradPair> {
        (0..n)
            .map(|i| {
                let a = uniform(seed, i) as f32;
                let b = uniform(seed ^ 1, i) as f32;
                gp(4.0 * a - 1.5, 0.05 + b)
            })
            .collect()
    }

    fn binned(n: usize, features: usize, missing: bool) -> GHistIndex {
        let x: Vec<f32> = (0..n * features)
            .map(|i| {
                if missing && i % 7 == 3 {
                    f32::NAN
                } else {
                    uniform(99, i) as f32
                }
            })
            .collect();
        let data = DMatrix::from_dense(&x, n, features).unwrap();
        GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, 64))
    }

    #[test]
    fn packed_widths_round_trip_signed_halves() {
        for (g, h) in [(0, 0), (-3, 5), (7, -2), (-1, -1), (12_000, 16_000)] {
            assert_eq!(i32::from_parts(g, h).parts(), (g, h));
            assert_eq!(
                i64::from_parts(g << 14, h << 14).parts(),
                (g << 14, h << 14)
            );
            assert_eq!(
                i128::from_parts(g << 40, h << 40).parts(),
                (g << 40, h << 40)
            );
        }
        // Sums of packed values are packed sums, carries between halves
        // included.
        let a = i32::from_parts(-5, 3);
        let b = i32::from_parts(2, -7);
        assert_eq!(a.add(b).parts(), (-3, -4));
        assert_eq!(a.sub(b).parts(), (-7, 10));
        assert_eq!(i64::repack(a).parts(), (-5, 3));
    }

    #[test]
    fn deterministic_rounding_error_is_at_most_half_a_step() {
        let gpair = gradients(10_000, 3);
        for bins in [2, 4, 16, 127] {
            let q = QuantizedGradients::quantize(&gpair, bins, false, 0);
            let half = (bins / 2) as f64;
            for (row, g) in gpair.iter().enumerate() {
                let s = q.row_stats(row);
                assert!((s.grad - f64::from(g.grad)).abs() <= 0.5 * q.grad_scale * (1.0 + 1e-6));
                assert!((s.hess - f64::from(g.hess)).abs() <= 0.5 * q.hess_scale * (1.0 + 1e-6));
                let (qg, qh) = row_parts(q.packed[row]);
                assert!(qg.abs() as f64 <= half && qh >= 0 && qh <= bins as i64);
            }
        }
    }

    #[test]
    fn stochastic_rounding_is_unbiased_and_within_one_step() {
        // One gradient value, quantized under many independent streams: the
        // mean of the dequantized values converges to the input.
        let gpair = vec![gp(1.0, 1.0), gp(0.37, 0.61), gp(-0.83, 0.2)];
        let trials = 20_000;
        let mut sums = [(0f64, 0f64); 3];
        for seed in 0..trials {
            let q = QuantizedGradients::quantize(&gpair, 4, true, seed);
            for (row, sum) in sums.iter_mut().enumerate() {
                let s = q.row_stats(row);
                assert!((s.grad - f64::from(gpair[row].grad)).abs() < q.grad_scale);
                assert!((s.hess - f64::from(gpair[row].hess)).abs() < q.hess_scale);
                sum.0 += s.grad;
                sum.1 += s.hess;
            }
        }
        let n = trials as f64;
        for (row, (g, h)) in sums.iter().enumerate() {
            // Standard error of a two-point variable with a step of 0.5
            // (scale max|g| / 2) is at most 0.25 / sqrt(n) ≈ 0.0018.
            assert!(
                (g / n - f64::from(gpair[row].grad)).abs() < 0.01,
                "row {row}: {}",
                g / n
            );
            assert!(
                (h / n - f64::from(gpair[row].hess)).abs() < 0.01,
                "row {row}: {}",
                h / n
            );
        }
        // A value on the grid is never perturbed.
        let q = QuantizedGradients::quantize(&[gp(1.0, 1.0), gp(-0.5, 0.25)], 4, true, 7);
        assert_eq!(q.row_stats(0), GradStats::new(1.0, 1.0));
        assert_eq!(q.row_stats(1), GradStats::new(-0.5, 0.25));
    }

    #[test]
    fn constant_hessians_are_exact() {
        let gpair: Vec<GradPair> = (0..100).map(|i| gp(i as f32 - 50.0, 0.75)).collect();
        let q = QuantizedGradients::quantize(&gpair, 4, true, 1);
        for row in 0..gpair.len() {
            assert_eq!(q.row_stats(row).hess, 0.75);
        }
        // All-zero statistics quantize to zero without dividing by zero.
        let zeros = QuantizedGradients::quantize(&[gp(0.0, 0.0); 5], 4, true, 1);
        assert_eq!(zeros.node_stats(&[0, 1, 2, 3, 4]), GradStats::default());
    }

    #[test]
    fn negative_hessians_keep_their_sign() {
        let gpair = vec![gp(0.5, -1.0), gp(-0.5, 2.0), gp(0.25, -0.5)];
        let q = QuantizedGradients::quantize(&gpair, 8, false, 0);
        assert_eq!(q.node_stats(&[0, 1, 2]), GradStats::new(0.25, 0.5));
    }

    #[test]
    fn widths_follow_row_count_and_levels() {
        let gpair = gradients(8, 5);
        let q = QuantizedGradients::quantize(&gpair, 4, true, 0);
        assert_eq!(q.width(8191), Width::W32);
        assert_eq!(q.width(8192), Width::W64);
        assert_eq!(q.width((1 << 29) - 1), Width::W64);
        assert_eq!(q.width(1 << 29), Width::W128);
        // Constant Hessians bound a row by ⌊Q/2⌋ instead of Q.
        let constant = QuantizedGradients::quantize(&[gp(1.0, 1.0); 4], 4, true, 0);
        assert_eq!(constant.width(16_383), Width::W32);
        assert_eq!(constant.width(16_384), Width::W64);
    }

    /// Integer histograms equal a direct per-row, per-feature sum of the
    /// dequantized rows: both add exact multiples of the scales.
    fn assert_matches_reference(ghist: &GHistIndex, q: &QuantizedGradients, rows: &[u32]) {
        let cuts = ghist.cuts();
        let mut expected = vec![GradStats::default(); ghist.total_bins()];
        for &r in rows {
            for f in 0..ghist.n_cols() {
                let (fs, fe) = cuts.feature_bins(f);
                if let Some(bin) = ghist.feature_bin_at(r as usize, f, fs, fe) {
                    expected[bin as usize].add(q.row_stats(r as usize));
                }
            }
        }
        assert_eq!(q.dequantize(&q.build(ghist, rows)), expected);
    }

    #[test]
    fn histograms_are_exact_for_every_layout_and_thread_count() {
        let n = 40_000;
        let gpair = gradients(n, 11);
        let q = QuantizedGradients::quantize(&gpair, 4, true, 3);
        let dense = binned(n, 6, false);
        let sparse = binned(n, 6, true);
        let all: Vec<u32> = (0..n as u32).collect();
        let every_third: Vec<u32> = (0..n as u32).step_by(3).collect();
        let small: Vec<u32> = (0..n as u32).step_by(9).take(3000).collect();
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(6)
            .build()
            .unwrap();
        for ghist in [&dense, &sparse] {
            for rows in [&all, &every_third, &small] {
                let a = serial.install(|| q.build(ghist, rows));
                let b = parallel.install(|| q.build(ghist, rows));
                assert_eq!(a, b);
                serial.install(|| assert_matches_reference(ghist, &q, rows));
            }
        }
        // Each width gives the same bins.
        let hist = q.build(&dense, &all);
        assert_eq!(hist.width(), Width::W64);
        let wide = QuantHist::W128(q.build_typed(&dense, &all));
        assert_eq!(q.dequantize(&wide), q.dequantize(&hist));
        let narrow = q.build(&dense, &small);
        assert_eq!(narrow.width(), Width::W32);
        let widened = QuantHist::W64(q.build_typed(&dense, &small));
        assert_eq!(q.dequantize(&widened), q.dequantize(&narrow));
    }

    #[test]
    fn subtraction_yields_the_sibling_exactly() {
        let n = 20_000;
        let gpair = gradients(n, 17);
        let q = Arc::new(QuantizedGradients::quantize(&gpair, 4, true, 9));
        let ghist = binned(n, 5, true);
        let all: Vec<u32> = (0..n as u32).collect();
        let (left, right): (Vec<u32>, Vec<u32>) = all.iter().partition(|&&r| r % 5 == 0);
        let parent = QuantNode {
            grads: Arc::clone(&q),
            hist: q.build(&ghist, &all),
        };
        let ((l, lf), (r, rf)) = parent.children(&ghist, &left, &right, Vec::new());
        assert_eq!(q.dequantize(&l.hist), q.dequantize(&q.build(&ghist, &left)));
        assert_eq!(
            q.dequantize(&r.hist),
            q.dequantize(&q.build(&ghist, &right))
        );
        assert_eq!(lf, q.dequantize(&q.build(&ghist, &left)));
        assert_eq!(rf, q.dequantize(&q.build(&ghist, &right)));
        // The smaller child keeps its own narrower width.
        assert_eq!(l.hist.width(), Width::W32);
        assert_eq!(r.hist.width(), Width::W64);
    }
}
