//! The traversal the `f64` and quantized histograms share: which layout a
//! build sweeps (a dense index's feature columns, dense row tiles, or CSR
//! rows), in which order its rows reach each bin, and what it prefetches.
//! Both histograms instantiate the same sequential loops ([`accumulate`]),
//! monomorphized over a [`Bucket`] (the histogram entry: `GradStats`, or a
//! packed integer) and a [`RowValue`] (a row's stored statistic, turned
//! into the value it adds to each of its bins). The feature-parallel sweep
//! ([`by_features`]) serves the `f64` build; the quantized one widens a
//! contiguous range's values once for all its feature tasks instead, which
//! the per-row widening here would repeat per feature.
//!
//! Every traversal adds each bin's rows in the order of `rows`, ascending,
//! so the `f64` sums are the row-order chain; tiling and grouping change
//! only which bins are hot.

use super::feature_slices;
use crate::data::ghist::{BinIndex, Bins, ColumnSet, Columns, GHistIndex};
use rayon::prelude::*;
use std::ops::Range;

/// A histogram bin's running sum.
pub(super) trait Bucket: Copy + Default + Send + Sync {
    /// Add one row's value.
    fn push(&mut self, value: Self);
}

/// A row's stored statistic (the slices the traversal reads are indexed by
/// row), turned into the value it adds to each of its bins.
pub(super) trait RowValue<E: Bucket>: Copy + Sync {
    fn value(self) -> E;
}

/// Rows to run ahead of the accumulation loop when prefetching. Each row's bins
/// and value are fetched into L1 before the loop needs them. Subsets deep in
/// the tree are too sparse for hardware stride prediction.
const PREFETCH_ROWS: usize = 8;
const CACHE_LINE: usize = 64;
/// Rows per tile of the dense accumulation. A tile's row lines stay in cache while
/// its feature blocks are swept, so re-reading them per block is cheap.
const TILE_ROWS: usize = 1024;
/// Histogram bytes a feature block may span, so the block's histogram slice
/// stays resident in a 64 KiB L1 while a tile is accumulated into it (4,096
/// `f64` bins).
const BLOCK_BYTES: usize = 64 * 1024;

/// Sequential accumulation of `rows` into `out` (added, not reset), `values`
/// indexed by row. Specialized on the bin-index width so the inner loop reads
/// the narrowest integers.
#[inline]
pub(super) fn accumulate<E: Bucket, V: RowValue<E>>(
    ghist: &GHistIndex,
    rows: &[u32],
    values: &[V],
    out: &mut [E],
) {
    match ghist.bins() {
        Bins::U16(bins) => accumulate_bins(ghist, bins, rows, values, out),
        Bins::U32(bins) => accumulate_bins(ghist, bins, rows, values, out),
    }
}

/// Prefetch the `len` bins of one row starting at `start`, a cache line at a
/// time.
#[inline(always)]
fn prefetch_bins<B>(bins: &[B], start: usize, len: usize) {
    for offset in (0..len).step_by(CACHE_LINE / std::mem::size_of::<B>()) {
        if let Some(bin) = bins.get(start + offset) {
            crate::simd::prefetch_read(bin);
        }
    }
}

/// Feature blocks `[f0, f1)` of a dense index with `stride` features whose
/// bin ranges each span at most `block_bins` (a single feature may exceed it).
fn feature_blocks(ghist: &GHistIndex, stride: usize, block_bins: usize) -> Vec<(usize, usize)> {
    let cuts = ghist.cuts();
    let mut blocks = Vec::new();
    let mut block_start = 0;
    for f in 1..=stride {
        let span = cuts.feature_bins(f - 1).1 - cuts.feature_bins(block_start).0;
        if span > block_bins && f - 1 > block_start {
            blocks.push((block_start, f - 1));
            block_start = f - 1;
        }
    }
    blocks.push((block_start, stride));
    blocks
}

#[inline(always)]
fn accumulate_bins<E: Bucket, V: RowValue<E>, B: BinIndex>(
    ghist: &GHistIndex,
    bins: &[B],
    rows: &[u32],
    values: &[V],
    out: &mut [E],
) {
    // Establishes the bound used by `add_row`: with `out` covering every bin,
    // the `GHistIndex` invariant (all stored bins < total_bins) makes every
    // histogram index in range.
    assert_eq!(
        out.len(),
        ghist.total_bins(),
        "histogram length must equal the binned index's bin count"
    );
    // A row holds at most one bin per feature (dense rows one each, and
    // `DMatrix` refuses duplicate CSR columns), from disjoint ranges, so its
    // bins are distinct: four histogram entries are loaded before any is
    // stored, which lets the loads overlap. Each bin still receives its rows
    // in order.
    let add_row = |row_bins: &[B], v: V, out: &mut [E]| {
        let v = v.value();
        let base = out.as_mut_ptr();
        let (quads, rest) = row_bins.as_chunks::<4>();
        for &quad in quads {
            let [a, b, c, d] = quad.map(BinIndex::index);
            // SAFETY: every bin is `< ghist.total_bins() == out.len()` by the
            // index invariant and the assertion above, and the four bins are
            // distinct (different features of one row), so the reads and
            // writes are in bounds and never overlap.
            unsafe {
                let (ha, hb, hc, hd) = (*base.add(a), *base.add(b), *base.add(c), *base.add(d));
                let add = |mut h: E| {
                    h.push(v);
                    h
                };
                *base.add(a) = add(ha);
                *base.add(b) = add(hb);
                *base.add(c) = add(hc);
                *base.add(d) = add(hd);
            }
        }
        for &bin in rest {
            // SAFETY: as above.
            unsafe { &mut *base.add(bin.index()) }.push(v);
        }
    };

    // A dense index's contiguous row range (the root, or a root chunk,
    // without row sampling) sweeps the feature columns a few features at a
    // time: the bins stream sequentially and each feature's histogram slice
    // stays in L1. Every bin still receives its rows in ascending order, so
    // the sums are identical to the row sweep.
    if ghist.is_dense()
        && let Some(columns) = ghist.columns()
        && let Some(range) = contiguous_range(rows)
    {
        let rows = SweepRows::Range(range);
        let mut slices = feature_slices(ghist, out, 1);
        for (task, features) in slices.chunks_mut(FEATURES_PER_PASS).enumerate() {
            feature_group(columns, task * FEATURES_PER_PASS, features, &rows, values);
        }
        return;
    }
    if ghist.is_dense() {
        accumulate_dense(ghist, bins, ghist.n_cols(), rows, values, out, add_row);
    } else {
        let rp = ghist.row_ptr();
        for (i, &r) in rows.iter().enumerate() {
            if let Some(&ahead) = rows.get(i + PREFETCH_ROWS) {
                let ahead = ahead as usize;
                if let (Some(&start), Some(&end)) = (rp.get(ahead), rp.get(ahead + 1)) {
                    prefetch_bins(bins, start, end - start);
                }
                if let Some(v) = values.get(ahead) {
                    crate::simd::prefetch_read(v);
                }
            }
            let ri = r as usize;
            add_row(&bins[rp[ri]..rp[ri + 1]], values[ri], out);
        }
    }
}

/// `Some(first..end)` when `rows` is exactly the ascending run
/// `first, first + 1, ..., end - 1` (checked element by element, so unsorted
/// or repeated indices never take the column path).
#[inline]
pub(super) fn contiguous_range(rows: &[u32]) -> Option<Range<usize>> {
    let first = *rows.first()? as usize;
    let end = first.checked_add(rows.len())?;
    let contiguous = rows
        .iter()
        .enumerate()
        .all(|(i, &row)| row as usize == first + i);
    contiguous.then_some(first..end)
}

/// Features one pass of a column sweep visits: each pass reads every row's
/// value once for all of them.
const FEATURES_PER_PASS: usize = 4;

/// Dense accumulation tiled by rows and feature blocks. Every bin still
/// receives its rows in ascending order, so the result is identical to a
/// straight row sweep. The tiling only changes which histogram bins are hot.
#[inline(always)]
fn accumulate_dense<E: Bucket, V: RowValue<E>, B: BinIndex>(
    ghist: &GHistIndex,
    bins: &[B],
    stride: usize,
    rows: &[u32],
    values: &[V],
    out: &mut [E],
    add_row: impl Fn(&[B], V, &mut [E]),
) {
    let blocks = feature_blocks(ghist, stride, BLOCK_BYTES / std::mem::size_of::<E>());
    // A row of at most a cache line spans one or two lines: its first and
    // last bins cover both, without a loop. Longer rows take every line.
    let short_rows = stride * std::mem::size_of::<B>() <= CACHE_LINE;
    let prefetch = |rows: &[u32], i: usize| {
        if let Some(&ahead) = rows.get(i + PREFETCH_ROWS) {
            let ahead = ahead as usize;
            if short_rows {
                if let Some(row) = bins.get(ahead * stride..(ahead + 1) * stride)
                    && let (Some(first), Some(last)) = (row.first(), row.last())
                {
                    crate::simd::prefetch_read(first);
                    crate::simd::prefetch_read(last);
                }
            } else {
                prefetch_bins(bins, ahead * stride, stride);
            }
            if let Some(v) = values.get(ahead) {
                crate::simd::prefetch_read(v);
            }
        }
    };
    if blocks.len() == 1 {
        // Dense rows sit at `r * stride`: no row-pointer load, and the address
        // of a future row is known without touching memory.
        for (i, &r) in rows.iter().enumerate() {
            prefetch(rows, i);
            let start = r as usize * stride;
            add_row(&bins[start..start + stride], values[r as usize], out);
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
                add_row(&bins[start + f0..start + f1], values[r as usize], out);
            }
        }
    }
}

/// The rows a feature-parallel build visits: a contiguous range, or an
/// ascending subset.
pub(super) enum SweepRows<'a> {
    Range(Range<usize>),
    Subset(&'a [u32]),
}

/// Build `out` (overwritten) over `rows` from a dense index's feature
/// `columns`, split by feature instead of by rows: each task streams its
/// features' columns straight into their slices of `out`. Every bin has one
/// writer that adds its rows in ascending order, so there are no partial
/// histograms to allocate or reduce and the result is the sequential
/// sweep's. Tasks take as few features as keep every one of `threads`
/// workers busy (at most [`FEATURES_PER_PASS`]), swept together so each
/// pass reads and widens the row values once for all of them.
pub(super) fn by_features<E: Bucket, V: RowValue<E>>(
    ghist: &GHistIndex,
    columns: Columns<'_>,
    rows: &SweepRows<'_>,
    values: &[V],
    out: &mut [E],
    threads: usize,
) {
    out.fill(E::default());
    let per_task = features_per_task(ghist, threads);
    let mut slices = feature_slices(ghist, out, 1);
    slices
        .par_chunks_mut(per_task)
        .enumerate()
        .for_each(|(task, features)| {
            feature_group(columns, task * per_task, features, rows, values);
        });
}

/// [`by_features`] on `threads` scoped threads instead of Rayon's pool,
/// with the same bits (each bin still has one writer adding its rows in
/// ascending order), for a caller holding a lock that work stealing could
/// re-enter: the CUDA backend's state. A group whose thread the OS cannot
/// create (a process or container thread limit) is built on the calling
/// thread instead, with the same bits, since each group owns its features.
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub(super) fn by_features_scoped<E: Bucket, V: RowValue<E>>(
    ghist: &GHistIndex,
    columns: Columns<'_>,
    rows: &SweepRows<'_>,
    values: &[V],
    out: &mut [E],
    threads: usize,
) {
    /// One thread's features: `(task, features)` pairs as in [`by_features`].
    type Group<'a, 'b, E> = Vec<(usize, &'a mut [(usize, &'b mut [E])])>;
    out.fill(E::default());
    let per_task = features_per_task(ghist, threads);
    let mut slices = feature_slices(ghist, out, 1);
    let tasks: Vec<_> = slices.chunks_mut(per_task).enumerate().collect();
    let per_thread = tasks.len().div_ceil(threads.max(1)).max(1);
    let build = |group: Group<'_, '_, E>| {
        for (task, features) in group {
            feature_group(columns, task * per_task, features, rows, values);
        }
    };
    // Each group sits behind its own lock, so a failed spawn (which drops
    // the closure unrun) leaves the group for the calling thread to build.
    let mut groups = Vec::new();
    let mut tasks = tasks.into_iter().peekable();
    while tasks.peek().is_some() {
        groups.push(parking_lot::Mutex::new(
            tasks.by_ref().take(per_thread).collect(),
        ));
    }
    let take = |group: &parking_lot::Mutex<Vec<_>>| std::mem::take(&mut *group.lock());
    std::thread::scope(|scope| {
        for group in &groups {
            let spawned = std::thread::Builder::new().spawn_scoped(scope, || build(take(group)));
            if spawned.is_err() {
                build(take(group));
            }
        }
    });
}

/// Features per task of a sweep split across `threads` workers: as few as
/// keep every worker busy, at most [`FEATURES_PER_PASS`].
fn features_per_task(ghist: &GHistIndex, threads: usize) -> usize {
    ghist
        .n_cols()
        .div_ceil(threads.max(1))
        .clamp(1, FEATURES_PER_PASS)
}

/// Add `rows` into the histogram slices `features` (one to
/// [`FEATURES_PER_PASS`] consecutive features `f..`: `features[k]` is
/// feature `f + k`'s first global bin and its slice) from a dense index's
/// `columns`, in one pass that reads each row's value once. Every bin
/// receives its rows in ascending order, as in a one-feature sweep.
#[inline(always)]
fn feature_group<E: Bucket, V: RowValue<E>>(
    columns: Columns<'_>,
    f: usize,
    features: &mut [(usize, &mut [E])],
    rows: &SweepRows<'_>,
    values: &[V],
) {
    match columns {
        Columns::U8(set) => feature_group_in(set, f, features, rows, values),
        Columns::U16(set) => feature_group_in(set, f, features, rows, values),
        Columns::U32(set) => feature_group_in(set, f, features, rows, values),
    }
}

/// [`feature_group`] in the column width `B`.
#[inline(always)]
fn feature_group_in<E: Bucket, V: RowValue<E>, B: BinIndex>(
    set: ColumnSet<'_, B>,
    f: usize,
    features: &mut [(usize, &mut [E])],
    rows: &SweepRows<'_>,
    values: &[V],
) {
    let column = |f: usize| {
        let column = set.column(f);
        debug_assert!(column.missing.is_none(), "a dense index's columns");
        column.bins
    };
    match features {
        [a] => sweep_slices([column(f)], [a], rows, values),
        [a, b] => sweep_slices([column(f), column(f + 1)], [a, b], rows, values),
        [a, b, c] => sweep_slices(
            [column(f), column(f + 1), column(f + 2)],
            [a, b, c],
            rows,
            values,
        ),
        [a, b, c, d] => sweep_slices(
            [column(f), column(f + 1), column(f + 2), column(f + 3)],
            [a, b, c, d],
            rows,
            values,
        ),
        _ => unreachable!("callers pass one to four features"),
    }
}

/// Add each row's value to its local bin in each of the `K` full-length
/// feature columns, into that feature's slice. The binned-index invariant
/// keeps every bin of a dense index's column below its feature's bin
/// count; the slice index bounds check catches any violation.
#[inline(always)]
fn sweep_slices<const K: usize, E: Bucket, V: RowValue<E>, B: BinIndex>(
    columns: [&[B]; K],
    slices: [&mut (usize, &mut [E]); K],
    rows: &SweepRows<'_>,
    values: &[V],
) {
    let mut slices = slices.map(|(_, slice)| &mut **slice);
    match rows {
        SweepRows::Range(range) => {
            let values = &values[range.clone()];
            let n = values.len();
            let columns = columns.map(|c| &c[range.clone()][..n]);
            for (r, v) in values.iter().enumerate() {
                let v = v.value();
                for (column, slice) in columns.iter().zip(&mut slices) {
                    slice[column[r].index()].push(v);
                }
            }
        }
        SweepRows::Subset(rows) => {
            for &r in *rows {
                let r = r as usize;
                let v = values[r].value();
                for (column, slice) in columns.iter().zip(&mut slices) {
                    slice[column[r].index()].push(v);
                }
            }
        }
    }
}
