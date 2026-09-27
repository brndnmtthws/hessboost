//! Gradient-histogram construction backends.
//!
//! The [`HistogramBackend`] trait is the single seam a future GPU implementation
//! plugs into: everything above it (the histogram tree builder) is
//! backend-agnostic. The CPU backend uses `rayon` to split the work by
//! feature or into fixed blocks of rows whose partial histograms it reduces
//! in block order, so every histogram is independent of the thread count,
//! and provides the *subtraction trick* (`sibling = parent − child`) that
//! halves histogram construction cost.

pub(crate) mod quantized;
mod walk;

use crate::data::ghist::GHistIndex;
use crate::objective::GradPair;
use crate::tree::gain::GradStats;
use rayon::prelude::*;
use walk::{Bucket, RowValue, SweepRows, accumulate, by_features, contiguous_range};

/// A gradient histogram: one [`GradStats`] bucket per global bin.
pub type Histogram = Vec<GradStats>;

/// Construct a fresh zeroed histogram of the given length.
pub fn zeroed(total_bins: usize) -> Histogram {
    vec![GradStats::default(); total_bins]
}

/// Turn `parent` into the sibling histogram `parent − child` in place. Reusing
/// the parent's buffer avoids allocating and writing a third histogram.
pub fn subtract_in_place(parent: &mut [GradStats], child: &[GradStats]) {
    debug_assert_eq!(parent.len(), child.len());
    for (p, c) in parent.iter_mut().zip(child) {
        *p = p.sub(*c);
    }
}

/// Split a flat histogram of `stride` entries per global bin into each
/// feature's disjoint bin range, in feature order: `(first_bin, slice)`.
pub(crate) fn feature_slices<'a, T>(
    ghist: &GHistIndex,
    hist: &'a mut [T],
    stride: usize,
) -> Vec<(usize, &'a mut [T])> {
    let cuts = ghist.cuts();
    let mut slices = Vec::with_capacity(ghist.n_cols());
    let mut rest = hist;
    let mut next = 0;
    for f in 0..ghist.n_cols() {
        let (fs, fe) = cuts.feature_bins(f);
        assert_eq!(
            fs, next,
            "feature bin ranges must be contiguous and ordered"
        );
        let (head, tail) = rest.split_at_mut((fe - fs) * stride);
        slices.push((fs, head));
        rest = tail;
        next = fe;
    }
    assert!(
        rest.is_empty(),
        "feature bin ranges must cover the histogram"
    );
    slices
}

/// Backend that builds and combines gradient histograms.
///
/// Sibling histograms reuse the parent buffer via the free `subtract_in_place`;
/// there is intentionally no `subtract` hook (a three-slice method would only
/// add an allocation).
pub trait HistogramBackend: Send + Sync {
    /// Accumulate the gradients of `rows` into `out` (length = total bins).
    /// `out` is overwritten (not added to).
    fn build(&self, ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]);

    /// Announce the gradient slice the following [`build`](Self::build) calls
    /// of this tree will read. Called once per tree, before any `build`.
    /// Backends that stage the gradients (a GPU) upload them here; the
    /// default does nothing.
    fn prepare(&self, _ghist: &GHistIndex, _gpair: &[GradPair]) {}
}

/// Multi-core CPU histogram backend.
///
/// Each bin's `f64` sum is a function of the rows alone, never of the
/// thread count, so the serial and parallel builds agree bit for bit. A node
/// below 8,192 rows, and a column-major index swept by feature (a
/// contiguous row range, or any subset of an index of at most 2^18 rows),
/// add each bin's rows in ascending order: the plain chain XGBoost's
/// single-threaded build forms. Every other node (a sparse index, or a row
/// subset of a larger dense one) sums fixed blocks of about 4,096 rows,
/// each in row order from zero, and adds
/// the block partials to the first block's in block order. Outside the
/// range where `f64` sums are exact (`backend/exact_sum.rs`) that can round
/// differently from the chain, which XGBoost's threaded build does too
/// (its per-thread buffers are reduced in thread order); inside it every
/// grouping gives the chain's sum.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

/// Rows per block of the blocked build, and per task of the quantized one:
/// enough to amortize a partial histogram's zeroing and reduction.
const ROWS_PER_TASK: usize = 4096;
/// Nodes below this many rows are one block (built as a single chain).
const PARALLEL_THRESHOLD: usize = 2 * ROWS_PER_TASK;
/// Bins per task when the partial histograms are summed.
const REDUCE_BINS: usize = 2048;
/// Datasets up to this many rows gather row subsets feature by feature: the
/// gradients (8 bytes a row) then fit a core's 2 MiB L2.
const GATHER_MAX_ROWS: usize = 1 << 18;

impl HistogramBackend for CpuBackend {
    fn build(&self, ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]) {
        if rows.len() < PARALLEL_THRESHOLD {
            out.fill(GradStats::default());
            accumulate(ghist, rows, gpair, out);
            return;
        }
        let threads = rayon::current_num_threads();
        let Some(columns) = ghist.column_bins() else {
            accumulate_blocks(ghist, rows, gpair, out, threads);
            return;
        };
        let range = contiguous_range(rows);
        if range.is_none() && ghist.n_rows() > GATHER_MAX_ROWS {
            accumulate_blocks(ghist, rows, gpair, out, threads);
            return;
        }
        // The feature sweeps below are chains in row order, as is
        // `accumulate`, which runs them serially.
        if threads <= 1 {
            out.fill(GradStats::default());
            accumulate(ghist, rows, gpair, out);
            return;
        }

        // A contiguous row range with a column-major copy is split by
        // feature. Any other row subset of a small enough column-major index
        // is gathered the same way: every feature group re-reads the
        // gradients, so this pays only while they stay in a core's cache.
        let rows = match range {
            Some(range) => SweepRows::Range(range),
            None => SweepRows::Subset(rows),
        };
        by_features(ghist, &columns, &rows, gpair, out, threads);
    }
}

/// The blocked build of [`CpuBackend`]: `rows` (at least
/// [`PARALLEL_THRESHOLD`]) split into `rows.len() / ROWS_PER_TASK` blocks
/// of equal size (the last shorter), each accumulated from zero into a
/// partial histogram, and `out` = the first partial plus every later one in
/// block order. The blocks depend only on the row count; `threads` only
/// sets how many are built at once (in waves, each reduced into `out`
/// before the next), which bounds the partials held to one per thread.
fn accumulate_blocks(
    ghist: &GHistIndex,
    rows: &[u32],
    gpair: &[GradPair],
    out: &mut [GradStats],
    threads: usize,
) {
    let total = out.len();
    let blocks = rows.len() / ROWS_PER_TASK;
    let grain = rows.len().div_ceil(blocks);
    let wave = threads.clamp(1, blocks);
    let mut partials: Vec<Histogram> = Vec::with_capacity(wave);
    for (w, wave_rows) in rows.chunks(grain * wave).enumerate() {
        let built = wave_rows.len().div_ceil(grain);
        if w == 0 {
            // Each task allocates (and zeroes) its own partial.
            wave_rows
                .par_chunks(grain)
                .map(|block| {
                    let mut partial = zeroed(total);
                    accumulate(ghist, block, gpair, &mut partial);
                    partial
                })
                .collect_into_vec(&mut partials);
        } else {
            partials
                .par_iter_mut()
                .zip(wave_rows.par_chunks(grain))
                .for_each(|(partial, block)| {
                    partial.fill(GradStats::default());
                    accumulate(ghist, block, gpair, partial);
                });
        }
        // Split by bin range, which keeps each bin's block order while
        // using every worker.
        let (head, rest) = partials[..built].split_at(usize::from(w == 0));
        out.par_chunks_mut(REDUCE_BINS)
            .enumerate()
            .for_each(|(i, out)| {
                let start = i * REDUCE_BINS;
                let end = start + out.len();
                if let [first] = head {
                    out.copy_from_slice(&first[start..end]);
                }
                for partial in rest {
                    for (o, p) in out.iter_mut().zip(&partial[start..end]) {
                        o.add(*p);
                    }
                }
            });
    }
}

/// Bin-index storage widths the accumulation and partition loops specialize on.
pub(crate) trait BinIndex: Copy + Send + Sync {
    fn index(self) -> usize;
}

impl BinIndex for u16 {
    #[inline(always)]
    fn index(self) -> usize {
        self as usize
    }
}

impl BinIndex for u32 {
    #[inline(always)]
    fn index(self) -> usize {
        self as usize
    }
}

impl Bucket for GradStats {
    #[inline(always)]
    fn push(&mut self, value: GradStats) {
        self.add(value);
    }
}

impl RowValue<GradStats> for GradPair {
    #[inline(always)]
    fn value(self) -> GradStats {
        GradStats::from_pair(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DMatrix;
    use crate::data::ghist::Bins;
    use crate::data::quantile::HistCuts;

    #[test]
    fn subtraction_identity() {
        // parent = left + right, so parent - left = right.
        let total = 8;
        let mut parent = zeroed(total);
        let mut left = zeroed(total);
        let mut right = zeroed(total);
        for i in 0..total {
            left[i] = GradStats::new(i as f64, 1.0);
            right[i] = GradStats::new(-(i as f64) * 0.5, 2.0);
            parent[i] = GradStats::new(left[i].grad + right[i].grad, left[i].hess + right[i].hess);
        }
        let mut out = parent.clone();
        subtract_in_place(&mut out, &left);
        for i in 0..total {
            assert!((out[i].grad - right[i].grad).abs() < 1e-12);
            assert!((out[i].hess - right[i].hess).abs() < 1e-12);
        }
    }

    /// Row-major reference independent of `accumulate`: every bin receives
    /// its rows in ascending order, which is the order both the row sweep and
    /// the column sweep must reproduce bit for bit.
    fn row_order_reference(ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair]) -> Histogram {
        let stride = ghist.dense_stride().expect("dense index");
        let mut h = zeroed(ghist.total_bins());
        for &r in rows {
            let r = r as usize;
            let g = GradStats::from_pair(gpair[r]);
            for f in 0..stride {
                let bin = match ghist.bins() {
                    Bins::U16(b) => b[r * stride + f] as usize,
                    Bins::U32(b) => b[r * stride + f] as usize,
                };
                h[bin].add(g);
            }
        }
        h
    }

    #[test]
    fn column_and_row_sweeps_match_reference_bit_for_bit() {
        let (n, f) = (3 * PARALLEL_THRESHOLD + 129, 7);
        let x: Vec<f32> = (0..n * f)
            .map(|i| ((i * 2_654_435_761_usize) % 1009) as f32 / 7.0)
            .collect();
        let data = DMatrix::from_dense(&x, n, f).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 64);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        assert!(
            ghist.column_bins().is_some(),
            "dense index keeps a column copy"
        );
        let gpair: Vec<GradPair> = (0..n)
            .map(|i| {
                GradPair::new(
                    ((i * 7919) % 1237) as f32 / 331.0 - 1.9,
                    0.25 + (i % 5) as f32,
                )
            })
            .collect();
        let bits = |h: &Histogram| -> Vec<(u64, u64)> {
            h.iter()
                .map(|s| (s.grad.to_bits(), s.hess.to_bits()))
                .collect()
        };
        let all: Vec<u32> = (0..n as u32).collect();
        let subset: Vec<u32> = (0..n as u32).filter(|r| r % 3 != 1).collect();
        let offset: Vec<u32> = (1000..(1000 + PARALLEL_THRESHOLD) as u32).collect();
        assert!(contiguous_range(&all).is_some() && contiguous_range(&offset).is_some());
        assert!(contiguous_range(&subset).is_none());
        assert!(contiguous_range(&[2, 0, 1]).is_none() && contiguous_range(&[5, 5]).is_none());
        for rows in [&all, &subset, &offset] {
            let expect = bits(&row_order_reference(&ghist, rows, &gpair));
            for threads in [1, 4] {
                let mut out = zeroed(ghist.total_bins());
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| CpuBackend.build(&ghist, rows, &gpair, &mut out));
                assert_eq!(bits(&out), expect, "rows={} threads={threads}", rows.len());
            }
        }
    }

    /// A sparse index of `n` rows holding only feature 0 (of 3), binned from
    /// `x`.
    fn sparse_index(x: &[f32]) -> GHistIndex {
        let n = x.len();
        let data = DMatrix::from_csr((0..=n).collect(), vec![0; n], x.to_vec(), 3).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        assert!(
            ghist.column_bins().is_none(),
            "sparse index has no column copy"
        );
        ghist
    }

    fn build_on(
        threads: usize,
        ghist: &GHistIndex,
        rows: &[u32],
        gpair: &[GradPair],
    ) -> Vec<(f64, f64)> {
        let mut out = zeroed(ghist.total_bins());
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| CpuBackend.build(ghist, rows, gpair, &mut out));
        out.iter().map(|s| (s.grad, s.hess)).collect()
    }

    /// The reported case: 8,192 sparse rows in one bin, gradients `2^54` at
    /// row 0, `-2^54` at row 4096, `1` at row 4097. The row-order chain sums
    /// to `1`; the two blocks of 4,096 rows sum to `2^54` and
    /// `-2^54 + 1 = -2^54` (rounded), so `0`. Every thread count, one
    /// included, builds the blocks.
    #[test]
    fn histograms_do_not_depend_on_the_thread_count() {
        let n = 2 * ROWS_PER_TASK;
        let ghist = sparse_index(&vec![1.0; n]);
        assert_eq!(
            ghist.cuts().feature_bins(0),
            (0, 1),
            "feature 0 has one bin"
        );
        let mut gpair = vec![GradPair::new(0.0, 1.0); n];
        gpair[0].grad = 2f32.powi(54);
        gpair[4096].grad = -(2f32.powi(54));
        gpair[4097].grad = 1.0;
        let rows: Vec<u32> = (0..n as u32).collect();
        for threads in [1, 2, 4, 8] {
            let hist = build_on(threads, &ghist, &rows, &gpair);
            assert_eq!(hist[0], (0.0, n as f64), "{threads} threads");
            assert!(
                hist[1..].iter().all(|&b| b == (0.0, 0.0)),
                "{threads} threads"
            );
        }
    }

    /// The blocked build over several waves (five blocks, built two or
    /// three at a time) equals an independent block-order reference for
    /// every thread count, on gradients spanning enough exponents that the
    /// grouping shows in the low bits.
    #[test]
    fn blocked_build_matches_the_block_order_reference() {
        let n = 6 * ROWS_PER_TASK + 123;
        let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let ghist = sparse_index(&x);
        let gpair: Vec<GradPair> = (0..n)
            .map(|i| {
                let scale = 2f32.powi((i * 37 % 61) as i32 - 30);
                GradPair::new(((i * 7919) % 1237) as f32 / 331.0 * scale - 1.0, scale)
            })
            .collect();
        let rows: Vec<u32> = (0..n as u32).filter(|r| r % 11 != 3).collect();
        let grain = rows.len().div_ceil(rows.len() / ROWS_PER_TASK);
        let mut expect = zeroed(ghist.total_bins());
        for block in rows.chunks(grain) {
            let mut partial = zeroed(ghist.total_bins());
            for &r in block {
                let bin = match ghist.bins() {
                    Bins::U16(b) => usize::from(b[ghist.row_ptr()[r as usize]]),
                    Bins::U32(b) => b[ghist.row_ptr()[r as usize]] as usize,
                };
                partial[bin].add(GradStats::from_pair(gpair[r as usize]));
            }
            for (e, p) in expect.iter_mut().zip(&partial) {
                e.add(*p);
            }
        }
        let expect: Vec<(f64, f64)> = expect.iter().map(|s| (s.grad, s.hess)).collect();
        let chain: Vec<(f64, f64)> = {
            let mut h = zeroed(ghist.total_bins());
            accumulate(&ghist, &rows, &gpair, &mut h);
            h.iter().map(|s| (s.grad, s.hess)).collect()
        };
        assert_ne!(expect, chain, "the case must separate the groupings");
        for threads in [1, 2, 3, 8] {
            assert_eq!(
                build_on(threads, &ghist, &rows, &gpair),
                expect,
                "{threads} threads"
            );
        }
    }
}
