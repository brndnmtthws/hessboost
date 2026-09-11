//! Gradient-histogram construction backends.
//!
//! The [`HistogramBackend`] trait is the single seam a future GPU implementation
//! plugs into: everything above it (the histogram tree builder) is
//! backend-agnostic. The CPU backend uses `rayon` to accumulate per-thread
//! partial histograms and reduce them, and provides the *subtraction trick*
//! (`sibling = parent − child`) that halves histogram construction cost.

use crate::data::ghist::{Bins, GHistIndex};
use crate::objective::GradPair;
use crate::tree::gain::GradStats;
use rayon::prelude::*;

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

/// Backend that builds and combines gradient histograms.
pub trait HistogramBackend: Send + Sync {
    /// Accumulate the gradients of `rows` into `out` (length = total bins).
    /// `out` is overwritten (not added to).
    fn build(&self, ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]);

    /// Compute `out[i] = parent[i] − child[i]` for every bin (the sibling
    /// histogram via subtraction).
    fn subtract(&self, parent: &[GradStats], child: &[GradStats], out: &mut [GradStats]) {
        debug_assert_eq!(parent.len(), child.len());
        debug_assert_eq!(parent.len(), out.len());
        for i in 0..out.len() {
            out[i] = parent[i].sub(child[i]);
        }
    }
}

/// Multi-core CPU histogram backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

/// Each task needs enough rows to amortize its histogram allocation and merge.
/// Capping the task count avoids creating a full histogram for every worker
/// when a shallow node has only a few thousand rows.
const ROWS_PER_TASK: usize = 4096;
const PARALLEL_THRESHOLD: usize = 2 * ROWS_PER_TASK;

impl HistogramBackend for CpuBackend {
    fn build(&self, ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]) {
        let total = out.len();
        let threads = rayon::current_num_threads();
        if threads <= 1 || rows.len() < PARALLEL_THRESHOLD {
            out.iter_mut().for_each(|s| *s = GradStats::default());
            accumulate(ghist, rows, gpair, out);
            return;
        }

        // Each task builds a private histogram over a contiguous run of rows;
        // the partials are then summed into `out` in task order, so the result
        // is deterministic for a given worker count.
        let tasks = threads.min(rows.len() / ROWS_PER_TASK);
        let grain = rows.len().div_ceil(tasks);
        let partials: Vec<Histogram> = rows
            .par_chunks(grain)
            .map(|chunk| {
                let mut local = zeroed(total);
                accumulate(ghist, chunk, gpair, &mut local);
                local
            })
            .collect();
        let mut partials = partials.into_iter();
        out.copy_from_slice(&partials.next().expect("at least one row chunk"));
        for partial in partials {
            for (o, p) in out.iter_mut().zip(&partial) {
                o.add(*p);
            }
        }
    }
}

/// Rows to run ahead of the accumulation loop when prefetching. Each row's bins
/// and gradient are fetched into L1 before the loop needs them; subsets deep in
/// the tree are too sparse for hardware stride prediction.
const PREFETCH_ROWS: usize = 8;
const CACHE_LINE: usize = 64;

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

/// Sequential accumulation of `rows` into `out` (added, not reset). Specialized
/// on the bin-index width so the inner loop reads the narrowest integers.
#[inline]
fn accumulate(ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]) {
    match ghist.bins() {
        Bins::U16(bins) => accumulate_bins(ghist, bins, rows, gpair, out),
        Bins::U32(bins) => accumulate_bins(ghist, bins, rows, gpair, out),
    }
}

#[inline(always)]
fn accumulate_bins<B: BinIndex>(
    ghist: &GHistIndex,
    bins: &[B],
    rows: &[u32],
    gpair: &[GradPair],
    out: &mut [GradStats],
) {
    // Establishes the bound used by `add_row`: with `out` covering every bin,
    // the `GHistIndex` invariant (all stored bins < total_bins) makes every
    // histogram index in range.
    assert_eq!(
        out.len(),
        ghist.total_bins(),
        "histogram length must equal the binned index's bin count"
    );
    let add_row = |row_bins: &[B], gp: GradPair, out: &mut [GradStats]| {
        let g = GradStats::new(gp.grad as f64, gp.hess as f64);
        for &bin in row_bins {
            // SAFETY: `bin < ghist.total_bins() == out.len()` by the index
            // invariant and the assertion above.
            unsafe { out.get_unchecked_mut(bin.index()) }.add(g);
        }
    };
    let prefetch_row = |start: usize, len: usize| {
        for offset in (0..len).step_by(CACHE_LINE / std::mem::size_of::<B>()) {
            if let Some(bin) = bins.get(start + offset) {
                crate::simd::prefetch_read(bin);
            }
        }
    };

    if let Some(stride) = ghist.dense_stride() {
        accumulate_dense(ghist, bins, stride, rows, gpair, out, add_row, prefetch_row);
    } else {
        let rp = ghist.row_ptr();
        for (i, &r) in rows.iter().enumerate() {
            if let Some(&ahead) = rows.get(i + PREFETCH_ROWS) {
                let ahead = ahead as usize;
                if let (Some(&start), Some(&end)) = (rp.get(ahead), rp.get(ahead + 1)) {
                    prefetch_row(start, end - start);
                }
                if let Some(gp) = gpair.get(ahead) {
                    crate::simd::prefetch_read(gp);
                }
            }
            let ri = r as usize;
            add_row(&bins[rp[ri]..rp[ri + 1]], gpair[ri], out);
        }
    }
}

/// Rows per tile of the dense accumulation. A tile's row lines stay in L2 while
/// its feature blocks are swept, so re-reading them per block is cheap.
const TILE_ROWS: usize = 4096;
/// Histogram bins a feature block may span, sized so the block's histogram
/// slice (16 bytes per bin) stays resident in a 64 KiB L1 while a tile is
/// accumulated into it.
const BLOCK_BINS: usize = 4096;

/// Dense accumulation tiled by rows and feature blocks. Every bin still
/// receives its rows in ascending order, so the result is identical to a
/// straight row sweep; the tiling only changes which histogram bins are hot.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn accumulate_dense<B: BinIndex>(
    ghist: &GHistIndex,
    bins: &[B],
    stride: usize,
    rows: &[u32],
    gpair: &[GradPair],
    out: &mut [GradStats],
    add_row: impl Fn(&[B], GradPair, &mut [GradStats]),
    prefetch_row: impl Fn(usize, usize),
) {
    // Feature blocks `[f0, f1)` whose bin ranges each span at most BLOCK_BINS.
    let cuts = ghist.cuts();
    let mut blocks: Vec<(usize, usize)> = Vec::new();
    let mut block_start = 0;
    for f in 1..=stride {
        let span = cuts.feature_bins(f - 1).1 - cuts.feature_bins(block_start).0;
        if span > BLOCK_BINS && f - 1 > block_start {
            blocks.push((block_start, f - 1));
            block_start = f - 1;
        }
    }
    blocks.push((block_start, stride));

    let prefetch = |rows: &[u32], i: usize| {
        if let Some(&ahead) = rows.get(i + PREFETCH_ROWS) {
            let ahead = ahead as usize;
            prefetch_row(ahead * stride, stride);
            if let Some(gp) = gpair.get(ahead) {
                crate::simd::prefetch_read(gp);
            }
        }
    };
    if blocks.len() == 1 {
        // Dense rows sit at `r * stride`: no row-pointer load, and the address
        // of a future row is known without touching memory.
        for (i, &r) in rows.iter().enumerate() {
            prefetch(rows, i);
            let start = r as usize * stride;
            add_row(&bins[start..start + stride], gpair[r as usize], out);
        }
        return;
    }
    for tile in rows.chunks(TILE_ROWS) {
        for (block, &(f0, f1)) in blocks.iter().enumerate() {
            for (i, &r) in tile.iter().enumerate() {
                if block == 0 {
                    prefetch(tile, i);
                }
                let start = r as usize * stride;
                add_row(&bins[start + f0..start + f1], gpair[r as usize], out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::quantile::HistCuts;
    use crate::data::DMatrix;

    fn brute_force(
        ghist: &GHistIndex,
        rows: &[u32],
        gpair: &[GradPair],
        total: usize,
    ) -> Histogram {
        let mut h = zeroed(total);
        accumulate(ghist, rows, gpair, &mut h);
        h
    }

    #[test]
    fn build_matches_brute_force() {
        let n = 200;
        let x: Vec<f32> = (0..n).map(|i| (i % 17) as f32).collect();
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 32);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        let gpair: Vec<GradPair> = (0..n)
            .map(|i| GradPair::new((i as f32) * 0.1 - 5.0, 1.0))
            .collect();
        let rows: Vec<u32> = (0..n as u32).collect();

        let mut out = zeroed(ghist.total_bins());
        CpuBackend.build(&ghist, &rows, &gpair, &mut out);
        let expect = brute_force(&ghist, &rows, &gpair, ghist.total_bins());
        for (a, b) in out.iter().zip(&expect) {
            assert!((a.grad - b.grad).abs() < 1e-4);
            assert!((a.hess - b.hess).abs() < 1e-4);
        }
    }

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
        let mut out = zeroed(total);
        CpuBackend.subtract(&parent, &left, &mut out);
        for i in 0..total {
            assert!((out[i].grad - right[i].grad).abs() < 1e-12);
            assert!((out[i].hess - right[i].hess).abs() < 1e-12);
        }
    }

    #[test]
    fn parallel_matches_sequential_large() {
        let n = PARALLEL_THRESHOLD + 37;
        let x: Vec<f32> = (0..n).map(|i| (i % 251) as f32).collect();
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 64);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        let gpair: Vec<GradPair> = (0..n)
            .map(|i| GradPair::new(((i * 7) % 13) as f32 - 6.0, 1.0))
            .collect();
        let rows: Vec<u32> = (0..n as u32).collect();

        let mut out = zeroed(ghist.total_bins());
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| CpuBackend.build(&ghist, &rows, &gpair, &mut out));
        let expect = brute_force(&ghist, &rows, &gpair, ghist.total_bins());
        for (a, b) in out.iter().zip(&expect) {
            assert!(
                (a.grad - b.grad).abs() < 1e-2,
                "grad {} vs {}",
                a.grad,
                b.grad
            );
            assert!((a.hess - b.hess).abs() < 1e-2);
        }
    }
}
