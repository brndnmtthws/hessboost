//! Pre-binned feature storage (`GHistIndexMatrix` in XGBoost).
//!
//! Every non-missing entry is replaced by its bin, held in two layouts that
//! one binning pass writes:
//!
//! - **Rows** ([`GHistIndex::bins`], sliced by [`GHistIndex::row_ptr`]):
//!   each row's present entries as *global* histogram bins (a feature's
//!   first bin, [`HistCuts::feature_bins`], plus its local bin), CSR-shaped,
//!   in the narrowest width that fits the total bin count (`u16` up to
//!   65,536 bins, else `u32`). A node's rows are scattered through the
//!   dataset, and a row's entries stream from a few cache lines, so row-wise
//!   histogram builds gather these, each entry a single indexed add.
//! - **Columns** ([`GHistIndex::columns`]), kept by dense indexes and sparse
//!   ones at least half full: every feature's bins *local to the feature*
//!   (`0` up to its bin count), feature-major (feature `f` of row `r` at
//!   `f * n_rows + r`), in the narrowest width that holds every feature's
//!   bins (`u8` up to 256 bins, the default `max_bin`, else `u16`, else
//!   `u32`). An index with missing entries adds one bit per entry marking
//!   them ([`Column::missing`]). Routing a node's rows on one feature
//!   streams its column, and feature-parallel sweeps read a few columns per
//!   task.
//!
//! Neither layout serves the other's access pattern: sweeping columns over a
//! node's scattered rows touches a cache line per row and feature, and
//! routing through rows a line per row.

use crate::data::DMatrix;
use crate::data::dmatrix::is_missing;
use crate::data::quantile::{BinSearch, HistCuts};
use rayon::prelude::*;
use std::ops::Range;

#[cfg(all(target_os = "linux", feature = "cuda"))]
static NEXT_IDENTITY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Rows per task of the binning pass: a block of the panel's 173 features
/// (692 KiB of row-major values) stays in a core's L2 while every feature
/// reads its stretch out of it. A multiple of 64, so every block owns whole
/// words of each feature's missing bitmap.
const BIN_BLOCK_ROWS: usize = 1024;

/// Matrices of at least this many cells (rows times features) cut and bin
/// in parallel; smaller ones run serially on the calling thread (which the
/// bin cache relies on, see `bin_cache`).
pub(crate) const PARALLEL_BINNING_CELLS: usize = 1 << 16;

/// A bin storage width (`u8`, `u16`, `u32`), which the histogram,
/// partition, and sweep loops specialize on.
pub(crate) trait BinIndex: Copy + PartialEq + Send + Sync + 'static {
    /// How many bins a value of this width can name (`0..MAX_BINS`).
    const MAX_BINS: usize;
    /// The bin `bin` (below [`Self::MAX_BINS`]) in this width.
    fn from_bin(bin: u32) -> Self;
    /// The bin as an index.
    fn index(self) -> usize;
}

macro_rules! bin_index {
    ($ty:ty) => {
        impl BinIndex for $ty {
            const MAX_BINS: usize = <$ty>::MAX as usize + 1;

            #[inline(always)]
            fn from_bin(bin: u32) -> Self {
                debug_assert!(<$ty>::try_from(bin).is_ok());
                bin as $ty
            }

            #[inline(always)]
            fn index(self) -> usize {
                self as usize
            }
        }
    };
}
bin_index!(u8);
bin_index!(u16);
bin_index!(u32);

/// The rows' global bins, in the narrowest width that fits.
#[derive(Debug, Clone)]
enum RowStore {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

/// A view of every row's global bins (see [`GHistIndex::bins`]), tagged by
/// width so hot loops specialize with a single outer branch.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Bins<'a> {
    U16(&'a [u16]),
    U32(&'a [u32]),
}

/// The owned storage behind [`Columns`]: the local bins in their width, and
/// the missing-entry bitmap of an index that has missing entries (each
/// feature's bits padded to whole words, so feature `f` of row `r` is bit
/// `r` of words `f * n_rows.div_ceil(64)..`).
#[derive(Debug, Clone)]
struct ColumnStore {
    bins: ColumnBins,
    missing: Option<Vec<u64>>,
}

#[derive(Debug, Clone)]
enum ColumnBins {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
}

/// The feature-major local bins of a [`GHistIndex`] (see
/// [`GHistIndex::columns`]), tagged by width.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Columns<'a> {
    U8(ColumnSet<'a, u8>),
    U16(ColumnSet<'a, u16>),
    U32(ColumnSet<'a, u32>),
}

impl Columns<'_> {
    /// Whether the index has missing entries, which the columns' bitmap
    /// marks.
    #[inline]
    pub(crate) fn has_missing(&self) -> bool {
        match self {
            Columns::U8(set) => set.missing.is_some(),
            Columns::U16(set) => set.missing.is_some(),
            Columns::U32(set) => set.missing.is_some(),
        }
    }
}

/// Every feature's column in one width (see [`Columns`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ColumnSet<'a, B> {
    bins: &'a [B],
    missing: Option<&'a [u64]>,
    n_rows: usize,
}

impl<'a, B: BinIndex> ColumnSet<'a, B> {
    /// Feature `f`'s column.
    #[inline]
    pub(crate) fn column(&self, f: usize) -> Column<'a, B> {
        let words = self.n_rows.div_ceil(64);
        Column {
            bins: &self.bins[f * self.n_rows..][..self.n_rows],
            missing: self.missing.map(|bits| &bits[f * words..][..words]),
        }
    }
}

/// One feature's column: `bins[r]` is row `r`'s local bin, unless the
/// column's `missing` bitmap (absent when no entry is missing) marks the row
/// as lacking the feature; such a row's bin is `0` and means nothing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Column<'a, B> {
    pub(crate) bins: &'a [B],
    pub(crate) missing: Option<&'a [u64]>,
}

impl<B: BinIndex> Column<'_, B> {
    /// Row `r`'s local bin, or `None` when it lacks the feature.
    #[inline(always)]
    pub(crate) fn get(&self, r: usize) -> Option<u32> {
        let missing = self.missing.is_some_and(|bits| missing_bit(bits, r));
        (!missing).then(|| self.bins[r].index() as u32)
    }
}

/// Whether row `r`'s bit is set in a column's missing bitmap.
#[inline(always)]
pub(crate) fn missing_bit(bits: &[u64], r: usize) -> bool {
    (bits[r >> 6] >> (r & 63)) & 1 != 0
}

/// Global bins as the Metal and wgpu backends upload them: `u16` when every
/// bin (and, with missing entries, the sentinel) fits, else `u32`.
#[cfg(any(all(target_os = "macos", feature = "metal"), feature = "wgpu"))]
#[derive(Debug)]
pub(crate) enum GlobalBins {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

/// Binned dataset (see the module docs).
///
/// Invariant: every stored row bin is below [`GHistIndex::total_bins`] and
/// every stored column bin (missing entries' `0` included) below its
/// feature's bin count, so a histogram of that length (or a feature's slice
/// of one) can be indexed by any stored bin without bounds checks.
/// `from_dmatrix` is the only constructor and maintains it.
#[derive(Debug, Clone)]
pub struct GHistIndex {
    /// Immutable content identity, retained by clones and across moves.
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    identity: u64,
    n_rows: usize,
    n_cols: usize,
    row_ptr: Vec<usize>,
    rows: RowStore,
    columns: Option<ColumnStore>,
    cuts: HistCuts,
    /// Every row holds every feature, stored in feature order: a dense
    /// matrix without missing values, or CSR rows that are complete and
    /// sorted. Feature `f` of row `r` is then row entry `r * n_cols + f`.
    dense: bool,
}

impl GHistIndex {
    /// Bin a dataset against precomputed cuts.
    pub fn from_dmatrix(data: &DMatrix, cuts: HistCuts) -> Self {
        let n_rows = data.n_rows();
        let n_cols = cuts.n_features();
        // Binning reads `data`'s features and stores the cuts' width, which
        // the unsafe histogram loops (and the device backends) index rows
        // by: they must agree, as must every feature's bin count with the
        // search's clamp to its last bin (the type's invariant). `HistCuts`
        // from the data always satisfies both; these guard against any
        // other source.
        assert_eq!(data.n_cols(), n_cols, "a cut table for every feature");
        assert!(
            (0..n_cols).all(|f| cuts.num_bins(f) > 0),
            "every feature of a cut table owns at least one bin"
        );
        let parallel = rayon::current_num_threads() > 1
            && n_rows.saturating_mul(n_cols) >= PARALLEL_BINNING_CELLS;
        let row_ptr = row_offsets(data, parallel);
        let cells = n_rows.saturating_mul(n_cols);
        let entries = row_ptr[n_rows];
        let layout = Layout {
            row_ptr: &row_ptr,
            parallel,
            columns: (cells <= entries.saturating_mul(2)).then_some(entries < cells),
        };
        let widest = (0..n_cols).map(|f| cuts.num_bins(f)).max().unwrap_or(0);
        let (rows, columns, dense) = if cuts.total_bins() <= u16::MAX_BINS {
            let (rows, columns, dense) = bin_rows::<u16>(data, &cuts, layout, widest);
            (RowStore::U16(rows), columns, dense)
        } else {
            let (rows, columns, dense) = bin_rows::<u32>(data, &cuts, layout, widest);
            (RowStore::U32(rows), columns, dense)
        };
        GHistIndex {
            // 2^64 indexes: never exhausted.
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            identity: NEXT_IDENTITY.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            n_rows,
            n_cols,
            row_ptr,
            rows,
            columns,
            cuts,
            dense,
        }
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub(crate) fn identity(&self) -> u64 {
        self.identity
    }

    /// Number of rows.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of feature columns.
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// The cut table this index was built against.
    #[inline]
    pub fn cuts(&self) -> &HistCuts {
        &self.cuts
    }

    /// Total number of histogram bins.
    #[inline]
    pub fn total_bins(&self) -> usize {
        self.cuts.total_bins()
    }

    /// CSR row-offset table of [`Self::bins`] (length `n_rows + 1`).
    #[inline]
    pub(crate) fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Every row's present entries as global bins, in feature order for a
    /// dense index (storage order for CSR input), sliced by
    /// [`Self::row_ptr`].
    #[inline]
    pub(crate) fn bins(&self) -> Bins<'_> {
        match &self.rows {
            RowStore::U16(bins) => Bins::U16(bins),
            RowStore::U32(bins) => Bins::U32(bins),
        }
    }

    /// Whether every row holds every feature in feature order (see the
    /// field): row `r`'s entries are then `r * n_cols..(r + 1) * n_cols`.
    #[inline]
    pub(crate) fn is_dense(&self) -> bool {
        self.dense
    }

    /// Every feature's column of local bins (see the module docs), when the
    /// index keeps them: dense indexes and sparse ones at least half full.
    #[inline]
    pub(crate) fn columns(&self) -> Option<Columns<'_>> {
        let store = self.columns.as_ref()?;
        let missing = store.missing.as_deref();
        let n_rows = self.n_rows;
        Some(match &store.bins {
            ColumnBins::U8(bins) => Columns::U8(ColumnSet {
                bins,
                missing,
                n_rows,
            }),
            ColumnBins::U16(bins) => Columns::U16(ColumnSet {
                bins,
                missing,
                n_rows,
            }),
            ColumnBins::U32(bins) => Columns::U32(ColumnSet {
                bins,
                missing,
                n_rows,
            }),
        })
    }

    /// The global bin of feature `f` in row `r`, or `None` when the row
    /// lacks the feature: one column read when the index keeps columns,
    /// else a scan of the row's entries. A reference lookup for tests.
    pub fn feature_bin(&self, r: usize, f: usize) -> Option<u32> {
        let (fs, fe) = self.cuts.feature_bins(f);
        if let Some(columns) = self.columns() {
            let local = match columns {
                Columns::U8(set) => set.column(f).get(r),
                Columns::U16(set) => set.column(f).get(r),
                Columns::U32(set) => set.column(f).get(r),
            }?;
            return Some(fs as u32 + local);
        }
        let entries = self.row_ptr[r]..self.row_ptr[r + 1];
        let in_feature = |b: u32| (fs..fe).contains(&(b as usize));
        match self.bins() {
            Bins::U16(b) => b[entries]
                .iter()
                .map(|&b| u32::from(b))
                .find(|&b| in_feature(b)),
            Bins::U32(b) => b[entries].iter().copied().find(|&b| in_feature(b)),
        }
    }

    /// The index of only `rows` (in order, repeats allowed) under the same
    /// cuts: their bins gathered, never binned again. It is the index
    /// [`from_dmatrix`](Self::from_dmatrix) builds from a copy of those rows
    /// with these cuts, layout rules included (columns kept from half full,
    /// the missing bitmap only with missing entries, `is_dense` for
    /// complete rows in feature order).
    pub(crate) fn select_rows(&self, rows: &[usize]) -> GHistIndex {
        let n = rows.len();
        let parallel = rayon::current_num_threads() > 1
            && n.saturating_mul(self.n_cols) >= PARALLEL_BINNING_CELLS;
        let mut row_ptr = Vec::with_capacity(n + 1);
        row_ptr.push(0);
        for &r in rows {
            row_ptr.push(row_ptr[row_ptr.len() - 1] + self.row_ptr[r + 1] - self.row_ptr[r]);
        }
        let gathered = Gathered {
            rows,
            source_ptr: &self.row_ptr,
            row_ptr: &row_ptr,
            n_cols: self.n_cols,
            parallel,
        };
        let (selected, dense) = match &self.rows {
            RowStore::U16(bins) => {
                let (selected, dense) = gathered.rows(bins);
                (RowStore::U16(selected), dense)
            }
            RowStore::U32(bins) => {
                let (selected, dense) = gathered.rows(bins);
                (RowStore::U32(selected), dense)
            }
        };
        let (cells, entries) = (n.saturating_mul(self.n_cols), row_ptr[n]);
        let keep_columns = cells <= entries.saturating_mul(2);
        let with_missing = entries < cells;
        let columns = match self.columns() {
            Some(source) if keep_columns => Some(gathered.columns(source, with_missing)),
            _ => None,
        };
        let mut index = GHistIndex {
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            identity: NEXT_IDENTITY.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            n_rows: n,
            n_cols: self.n_cols,
            row_ptr,
            rows: selected,
            columns,
            cuts: self.cuts.clone(),
            dense,
        };
        if keep_columns && index.columns.is_none() {
            index.columns = Some(index.columns_from_rows(with_missing));
        }
        index
    }

    /// The feature-major columns of this index's rows (with the missing
    /// bitmap when `with_missing`), scattered from them.
    fn columns_from_rows(&self, with_missing: bool) -> ColumnStore {
        fn scatter<R: BinIndex, C: BinIndex + Default>(
            index: &GHistIndex,
            bins: &[R],
            with_missing: bool,
        ) -> (Vec<C>, Option<Vec<u64>>) {
            let (n, n_cols) = (index.n_rows, index.n_cols);
            let words = n.div_ceil(64);
            let owner = bin_features(&index.cuts);
            let mut columns = vec![C::default(); n * n_cols];
            // Every bit starts missing; each stored entry clears its own.
            let mut missing = with_missing.then(|| {
                let mut bits = vec![0u64; words * n_cols];
                for column in bits.chunks_mut(words.max(1)) {
                    for (w, word) in column.iter_mut().enumerate() {
                        let valid = n.saturating_sub(w * 64).min(64);
                        *word = if valid == 64 {
                            u64::MAX
                        } else {
                            (1 << valid) - 1
                        };
                    }
                }
                bits
            });
            for r in 0..n {
                for &bin in &bins[index.row_ptr[r]..index.row_ptr[r + 1]] {
                    let f = owner[bin.index()] as usize;
                    let local = bin.index() - index.cuts.feature_bins(f).0;
                    columns[f * n + r] = C::from_bin(local as u32);
                    if let Some(bits) = missing.as_mut() {
                        bits[f * words + (r >> 6)] &= !(1 << (r & 63));
                    }
                }
            }
            (columns, missing)
        }
        let widest = (0..self.n_cols)
            .map(|f| self.cuts.num_bins(f))
            .max()
            .unwrap_or(0);
        let (bins, missing) = match (&self.rows, widest) {
            (RowStore::U16(rows), w) if w <= u8::MAX_BINS => {
                let (bins, missing) = scatter::<u16, u8>(self, rows, with_missing);
                (ColumnBins::U8(bins), missing)
            }
            (RowStore::U32(rows), w) if w <= u8::MAX_BINS => {
                let (bins, missing) = scatter::<u32, u8>(self, rows, with_missing);
                (ColumnBins::U8(bins), missing)
            }
            (RowStore::U16(rows), w) if w <= u16::MAX_BINS => {
                let (bins, missing) = scatter::<u16, u16>(self, rows, with_missing);
                (ColumnBins::U16(bins), missing)
            }
            (RowStore::U32(rows), w) if w <= u16::MAX_BINS => {
                let (bins, missing) = scatter::<u32, u16>(self, rows, with_missing);
                (ColumnBins::U16(bins), missing)
            }
            (RowStore::U16(rows), _) => {
                let (bins, missing) = scatter::<u16, u32>(self, rows, with_missing);
                (ColumnBins::U32(bins), missing)
            }
            (RowStore::U32(rows), _) => {
                let (bins, missing) = scatter::<u32, u32>(self, rows, with_missing);
                (ColumnBins::U32(bins), missing)
            }
        };
        ColumnStore { bins, missing }
    }

    /// Every feature's column as local `u32` bins, `u32::MAX` where the row
    /// lacks the feature (EBM pair ranking): widened from the columns when
    /// the index keeps them, else scattered from one pass over the rows.
    pub(crate) fn feature_columns(&self) -> Vec<Vec<u32>> {
        fn widen<B: BinIndex>(set: ColumnSet<'_, B>, f: usize) -> Vec<u32> {
            let column = set.column(f);
            (0..set.n_rows)
                .map(|r| column.get(r).unwrap_or(u32::MAX))
                .collect()
        }
        fn scatter<B: BinIndex>(index: &GHistIndex, bins: &[B]) -> Vec<Vec<u32>> {
            let owner = bin_features(&index.cuts);
            let mut columns = vec![vec![u32::MAX; index.n_rows]; index.n_cols];
            for r in 0..index.n_rows {
                for &bin in &bins[index.row_ptr[r]..index.row_ptr[r + 1]] {
                    let f = owner[bin.index()] as usize;
                    columns[f][r] = (bin.index() - index.cuts.feature_bins(f).0) as u32;
                }
            }
            columns
        }
        let Some(columns) = self.columns() else {
            return match self.bins() {
                Bins::U16(bins) => scatter(self, bins),
                Bins::U32(bins) => scatter(self, bins),
            };
        };
        (0..self.n_cols)
            .into_par_iter()
            .map(|f| match columns {
                Columns::U8(set) => widen(set, f),
                Columns::U16(set) => widen(set, f),
                Columns::U32(set) => widen(set, f),
            })
            .collect()
    }

    /// Every feature's column of *global* bins, feature-major, the width's
    /// maximum where the row lacks the feature: the store the Metal and
    /// wgpu backends upload, in `u16` when
    /// [`global_columns_fit_u16`](Self::global_columns_fit_u16), else `u32`.
    #[cfg(any(all(target_os = "macos", feature = "metal"), feature = "wgpu"))]
    pub(crate) fn global_columns(&self) -> GlobalBins {
        if self.global_columns_fit_u16() {
            GlobalBins::U16(self.globalized(u16::MAX, |bin| bin as u16))
        } else {
            GlobalBins::U32(self.globalized(u32::MAX, |bin| bin))
        }
    }

    /// Whether [`Self::global_columns`] are `u16`: every global bin fits it
    /// and, with missing entries, stays below its maximum, their sentinel.
    #[cfg(any(all(target_os = "macos", feature = "metal"), feature = "wgpu"))]
    pub(crate) fn global_columns_fit_u16(&self) -> bool {
        let total = self.total_bins();
        if self.dense {
            total <= u16::MAX_BINS
        } else {
            total < u16::MAX_BINS
        }
    }

    /// [`Self::global_columns`] in `T`: each present entry's global bin,
    /// stored as `stored(bin)`, `missing` elsewhere.
    #[cfg(any(all(target_os = "macos", feature = "metal"), feature = "wgpu"))]
    fn globalized<T: Copy + Send + Sync>(
        &self,
        missing: T,
        stored: impl Fn(u32) -> T + Sync,
    ) -> Vec<T> {
        fn fill<B: BinIndex, T>(
            column: Column<'_, B>,
            first: u32,
            out: &mut [T],
            stored: impl Fn(u32) -> T,
        ) {
            for (r, slot) in out.iter_mut().enumerate() {
                if let Some(local) = column.get(r) {
                    *slot = stored(first + local);
                }
            }
        }
        fn scatter<B: BinIndex, T>(
            index: &GHistIndex,
            bins: &[B],
            out: &mut [T],
            stored: impl Fn(u32) -> T,
        ) {
            let owner = bin_features(&index.cuts);
            let n = index.n_rows;
            for r in 0..n {
                for &bin in &bins[index.row_ptr[r]..index.row_ptr[r + 1]] {
                    let f = owner[bin.index()] as usize;
                    out[f * n + r] = stored(bin.index() as u32);
                }
            }
        }
        let n = self.n_rows;
        let mut out = vec![missing; n * self.n_cols];
        if n == 0 {
            return out;
        }
        let Some(columns) = self.columns() else {
            match self.bins() {
                Bins::U16(bins) => scatter(self, bins, &mut out, stored),
                Bins::U32(bins) => scatter(self, bins, &mut out, stored),
            }
            return out;
        };
        out.par_chunks_mut(n).enumerate().for_each(|(f, out)| {
            let first = self.cuts.feature_bins(f).0 as u32;
            match columns {
                Columns::U8(set) => fill(set.column(f), first, out, &stored),
                Columns::U16(set) => fill(set.column(f), first, out, &stored),
                Columns::U32(set) => fill(set.column(f), first, out, &stored),
            }
        });
        out
    }
}

/// A row gather ([`GHistIndex::select_rows`]): the selected rows, the
/// source's row offsets, the gathered rows' offsets, and whether to gather
/// in parallel.
struct Gathered<'a> {
    rows: &'a [usize],
    source_ptr: &'a [usize],
    row_ptr: &'a [usize],
    n_cols: usize,
    parallel: bool,
}

impl Gathered<'_> {
    /// The selected rows' global bins, a block of rows per task, and whether
    /// every one holds every feature in feature order.
    fn rows<B: BinIndex + Default>(&self, bins: &[B]) -> (Vec<B>, bool) {
        let n = self.rows.len();
        let mut out = vec![B::default(); self.row_ptr[n]];
        let mut parts = Vec::with_capacity(n.div_ceil(BIN_BLOCK_ROWS));
        let mut rest = out.as_mut_slice();
        for start in (0..n).step_by(BIN_BLOCK_ROWS) {
            let end = (start + BIN_BLOCK_ROWS).min(n);
            let (part, tail) = rest.split_at_mut(self.row_ptr[end] - self.row_ptr[start]);
            parts.push((start..end, part));
            rest = tail;
        }
        let copy = |(block, part): (Range<usize>, &mut [B])| {
            let (mut at, mut dense) = (0, true);
            for &r in &self.rows[block] {
                let row = &bins[self.source_ptr[r]..self.source_ptr[r + 1]];
                part[at..at + row.len()].copy_from_slice(row);
                at += row.len();
                // Global bins rise with the feature, so a complete row in
                // feature order is one of strictly rising bins.
                dense &=
                    row.len() == self.n_cols && row.windows(2).all(|w| w[0].index() < w[1].index());
            }
            dense
        };
        let dense = if self.parallel {
            parts
                .into_par_iter()
                .map(copy)
                .reduce(|| true, |a, b| a && b)
        } else {
            // Every block must be copied, so no short-circuiting `all`.
            let mut dense = true;
            for part in parts {
                dense &= copy(part);
            }
            dense
        };
        (out, dense)
    }

    /// The selected rows' columns gathered from the source's, with a
    /// missing bitmap when `with_missing`.
    fn columns(&self, source: Columns<'_>, with_missing: bool) -> ColumnStore {
        match source {
            Columns::U8(set) => {
                let (bins, missing) = self.gather_columns(set, with_missing);
                ColumnStore {
                    bins: ColumnBins::U8(bins),
                    missing,
                }
            }
            Columns::U16(set) => {
                let (bins, missing) = self.gather_columns(set, with_missing);
                ColumnStore {
                    bins: ColumnBins::U16(bins),
                    missing,
                }
            }
            Columns::U32(set) => {
                let (bins, missing) = self.gather_columns(set, with_missing);
                ColumnStore {
                    bins: ColumnBins::U32(bins),
                    missing,
                }
            }
        }
    }

    /// [`Self::columns`] in the width `B`, a feature per task.
    fn gather_columns<B: BinIndex + Default>(
        &self,
        set: ColumnSet<'_, B>,
        with_missing: bool,
    ) -> (Vec<B>, Option<Vec<u64>>) {
        let n = self.rows.len();
        let words = n.div_ceil(64);
        let mut bins = vec![B::default(); n * self.n_cols];
        let mut missing = with_missing.then(|| vec![0u64; words * self.n_cols]);
        if n == 0 {
            return (bins, missing);
        }
        let bit_parts: Vec<Option<&mut [u64]>> = match missing.as_mut() {
            Some(bits) => bits.chunks_mut(words).map(Some).collect(),
            None => (0..self.n_cols).map(|_| None).collect(),
        };
        let parts: Vec<(&mut [B], Option<&mut [u64]>)> =
            bins.chunks_mut(n).zip(bit_parts).collect();
        let fill = |(f, (column, bits)): (usize, (&mut [B], Option<&mut [u64]>))| {
            let source = set.column(f);
            for (slot, &r) in column.iter_mut().zip(self.rows) {
                *slot = source.bins[r];
            }
            if let (Some(bits), Some(source_bits)) = (bits, source.missing) {
                for (i, &r) in self.rows.iter().enumerate() {
                    if missing_bit(source_bits, r) {
                        bits[i >> 6] |= 1 << (i & 63);
                    }
                }
            }
        };
        if self.parallel {
            parts.into_par_iter().enumerate().for_each(fill);
        } else {
            parts.into_iter().enumerate().for_each(fill);
        }
        (bins, missing)
    }
}

/// The feature owning each global bin.
fn bin_features(cuts: &HistCuts) -> Vec<u32> {
    let mut owner = vec![0u32; cuts.total_bins()];
    for f in 0..cuts.n_features() {
        let (fs, fe) = cuts.feature_bins(f);
        owner[fs..fe].fill(f as u32);
    }
    owner
}

/// The offsets of every row's present entries (length `n_rows + 1`).
fn row_offsets(data: &DMatrix, parallel: bool) -> Vec<usize> {
    let n = data.n_rows();
    let mut row_ptr = vec![0usize; n + 1];
    let count = |(b, counts): (usize, &mut [usize])| {
        let first = b * BIN_BLOCK_ROWS;
        if let Some(rows) = data.dense_rows() {
            let missing = data.missing();
            for (i, count) in counts.iter_mut().enumerate() {
                let row = rows.row(first + i);
                *count = row.iter().filter(|&&v| !is_missing(v, missing)).count();
            }
        } else {
            for (i, count) in counts.iter_mut().enumerate() {
                data.for_row_entry(first + i, |_, _| *count += 1);
            }
        }
    };
    if parallel {
        row_ptr[1..]
            .par_chunks_mut(BIN_BLOCK_ROWS)
            .enumerate()
            .for_each(count);
    } else {
        row_ptr[1..]
            .chunks_mut(BIN_BLOCK_ROWS)
            .enumerate()
            .for_each(count);
    }
    for r in 0..n {
        row_ptr[r + 1] += row_ptr[r];
    }
    row_ptr
}

/// What the binning pass writes: the row offsets, whether to bin in
/// parallel, and whether to keep columns (`Some`) with a missing bitmap
/// (`Some(true)`: some entry is missing).
#[derive(Clone, Copy)]
struct Layout<'a> {
    row_ptr: &'a [usize],
    parallel: bool,
    columns: Option<bool>,
}

/// What one binning pass produced in widths `R` (rows) and `C` (columns).
struct Binned<R, C> {
    rows: Vec<R>,
    /// Empty unless the layout keeps columns.
    columns: Vec<C>,
    missing: Option<Vec<u64>>,
    /// Every row holds every feature in feature order.
    dense: bool,
}

/// Bin `data` into global row bins of width `R` and, when the layout keeps
/// columns, local columns of the narrowest width that holds `widest` bins;
/// with whether every row holds every feature in feature order.
fn bin_rows<R: BinIndex + Default>(
    data: &DMatrix,
    cuts: &HistCuts,
    layout: Layout<'_>,
    widest: usize,
) -> (Vec<R>, Option<ColumnStore>, bool) {
    fn store<R, C>(
        binned: Binned<R, C>,
        wrap: fn(Vec<C>) -> ColumnBins,
    ) -> (Vec<R>, Option<ColumnStore>, bool) {
        let columns = ColumnStore {
            bins: wrap(binned.columns),
            missing: binned.missing,
        };
        (binned.rows, Some(columns), binned.dense)
    }
    if layout.columns.is_none() {
        let binned = bin_layouts::<R, u8>(data, cuts, layout);
        return (binned.rows, None, binned.dense);
    }
    if widest <= u8::MAX_BINS {
        store(bin_layouts::<R, u8>(data, cuts, layout), ColumnBins::U8)
    } else if widest <= u16::MAX_BINS {
        store(bin_layouts::<R, u16>(data, cuts, layout), ColumnBins::U16)
    } else {
        store(bin_layouts::<R, u32>(data, cuts, layout), ColumnBins::U32)
    }
}

/// One binning task: a block of rows, its stretch of the row bins, and (when
/// the index keeps columns) its part of every feature's column and missing
/// bitmap.
struct BinBlock<'a, R, C> {
    rows: Range<usize>,
    row_bins: &'a mut [R],
    columns: Option<BlockColumns<'a, C>>,
}

/// A block's part of every feature's column (`bins[f][i]` is feature `f` of
/// the block's row `i`) and of its missing bitmap (`missing[f]`: the words
/// covering the block's rows).
struct BlockColumns<'a, C> {
    bins: Vec<&'a mut [C]>,
    missing: Option<Vec<&'a mut [u64]>>,
}

/// [`bin_rows`] in widths `R` (rows) and `C` (columns, empty unless the
/// layout keeps them), a block of rows per task: each block writes its own
/// stretch of the row bins and its own part of every column and bitmap.
fn bin_layouts<R: BinIndex + Default, C: BinIndex + Default>(
    data: &DMatrix,
    cuts: &HistCuts,
    Layout {
        row_ptr,
        parallel,
        columns,
    }: Layout<'_>,
) -> Binned<R, C> {
    let (n, n_cols) = (data.n_rows(), cuts.n_features());
    let words = n.div_ceil(64);
    // Zeroed allocations (the OS's zero pages), first touched by the
    // parallel binning tasks.
    let mut row_bins = vec![R::default(); row_ptr[n]];
    let mut column_bins = vec![C::default(); if columns.is_some() { n * n_cols } else { 0 }];
    let mut missing = (columns == Some(true)).then(|| vec![0u64; words * n_cols]);
    let n_blocks = n.div_ceil(BIN_BLOCK_ROWS);
    let mut blocks: Vec<BinBlock<'_, R, C>> = Vec::with_capacity(n_blocks);
    let mut rest = row_bins.as_mut_slice();
    for b in 0..n_blocks {
        let rows = b * BIN_BLOCK_ROWS..((b + 1) * BIN_BLOCK_ROWS).min(n);
        let (part, tail) = rest.split_at_mut(row_ptr[rows.end] - row_ptr[rows.start]);
        rest = tail;
        blocks.push(BinBlock {
            rows,
            row_bins: part,
            columns: columns.map(|with_missing| BlockColumns {
                bins: Vec::with_capacity(n_cols),
                missing: with_missing.then(|| Vec::with_capacity(n_cols)),
            }),
        });
    }
    if n > 0 {
        if columns.is_some() {
            for column in column_bins.chunks_mut(n) {
                for (block, part) in blocks.iter_mut().zip(column.chunks_mut(BIN_BLOCK_ROWS)) {
                    if let Some(parts) = block.columns.as_mut() {
                        parts.bins.push(part);
                    }
                }
            }
        }
        if let Some(missing) = missing.as_mut() {
            for bits in missing.chunks_mut(words) {
                for (block, part) in blocks.iter_mut().zip(bits.chunks_mut(BIN_BLOCK_ROWS / 64)) {
                    if let Some(parts) = block.columns.as_mut().and_then(|c| c.missing.as_mut()) {
                        parts.push(part);
                    }
                }
            }
        }
    }
    let search = BinSearch::new(cuts);
    let bin = |block: BinBlock<'_, R, C>| bin_block(data, &search, row_ptr, block);
    let dense = if parallel {
        blocks
            .into_par_iter()
            .map(bin)
            .reduce(|| true, |a, b| a && b)
    } else {
        // Every block must be binned, so no short-circuiting `all`.
        let mut dense = true;
        for block in blocks {
            dense &= bin(block);
        }
        dense
    };
    Binned {
        rows: row_bins,
        columns: column_bins,
        missing,
        dense,
    }
}

/// Bin one block's rows into its stretch of the row bins and its part of
/// the columns, setting the missing bit (and leaving bin `0`) where a row
/// lacks a feature. Returns whether every row holds every feature in
/// feature order.
fn bin_block<R: BinIndex, C: BinIndex>(
    data: &DMatrix,
    search: &BinSearch<'_>,
    row_ptr: &[usize],
    block: BinBlock<'_, R, C>,
) -> bool {
    let BinBlock {
        rows,
        row_bins,
        mut columns,
    } = block;
    let n_cols = data.n_cols();
    let base = row_ptr[rows.start];
    let mut dense = true;
    if let Some(dense_rows) = data.dense_rows() {
        // Feature by feature, each feature's search tables stay hot across
        // the block's rows, which stay in cache across its features; each
        // row's entries land in feature order at its own cursor. A whole
        // matrix's block is one contiguous stretch; a view's rows are read
        // through their slices.
        let missing = data.missing();
        let contiguous = dense_rows
            .contiguous()
            .map(|values| &values[rows.start * n_cols..rows.end * n_cols]);
        let block_rows: Vec<&[f32]> = match contiguous {
            Some(_) => Vec::new(),
            None => rows.clone().map(|r| dense_rows.row(r)).collect(),
        };
        let mut cursor: Vec<usize> = row_ptr[rows.clone()].iter().map(|&p| p - base).collect();
        for f in 0..n_cols {
            let feature = search.feature(f);
            let first = feature.first_bin();
            let mut column = columns.as_mut().map(|c| {
                let bits = c.missing.as_mut().map(|m| &mut *m[f]);
                (&mut *c.bins[f], bits)
            });
            let mut bin = |i: usize, v: f32| {
                if is_missing(v, missing) {
                    dense = false;
                    if let Some((_, Some(bits))) = column.as_mut() {
                        bits[i >> 6] |= 1 << (i & 63);
                    }
                    return;
                }
                let local = feature.local_bin_of(v);
                if let Some((bins, _)) = column.as_mut() {
                    bins[i] = C::from_bin(local);
                }
                row_bins[cursor[i]] = R::from_bin(first + local);
                cursor[i] += 1;
            };
            match contiguous {
                Some(block) => {
                    for (i, &v) in block[f..].iter().step_by(n_cols).enumerate() {
                        bin(i, v);
                    }
                }
                None => {
                    for (i, row) in block_rows.iter().enumerate() {
                        bin(i, row[f]);
                    }
                }
            }
        }
    } else {
        // CSR rows: every bit starts missing and each stored entry clears
        // its own.
        let len = rows.len();
        if let Some(bits) = columns.as_mut().and_then(|c| c.missing.as_mut()) {
            for part in bits.iter_mut() {
                for (w, word) in part.iter_mut().enumerate() {
                    let valid = (len - w * 64).min(64);
                    *word = if valid == 64 {
                        u64::MAX
                    } else {
                        (1 << valid) - 1
                    };
                }
            }
        }
        let mut at = 0;
        for (i, row) in rows.enumerate() {
            let (mut stored, mut in_order) = (0usize, true);
            data.for_row_entry(row, |f, v| {
                let f = f as usize;
                in_order &= f == stored;
                stored += 1;
                let feature = search.feature(f);
                let local = feature.local_bin_of(v);
                if let Some(columns) = columns.as_mut() {
                    columns.bins[f][i] = C::from_bin(local);
                    if let Some(bits) = columns.missing.as_mut() {
                        bits[f][i >> 6] &= !(1 << (i & 63));
                    }
                }
                row_bins[at] = R::from_bin(feature.first_bin() + local);
                at += 1;
            });
            dense &= in_order && stored == n_cols;
        }
    }
    dense
}

/// Feature-major copy of `n_rows` dense rows (row `r` is `row(r)`, of
/// `n_cols` values) into `columns`. Four-feature groups expose eight tasks
/// for a 30-feature matrix while each pass reads a short contiguous run per
/// row and writes sequential column streams.
pub(crate) fn transpose_rows_into<'a, B: Copy + Default + Send + Sync + 'a>(
    row: impl Fn(usize) -> &'a [B] + Sync,
    n_rows: usize,
    n_cols: usize,
    columns: &mut Vec<B>,
) {
    const GROUP: usize = 4;
    // Every slot is written below, so only slots the buffer gains are
    // zeroed.
    columns.resize(n_rows * n_cols, B::default());
    columns.truncate(n_rows * n_cols);
    if n_rows == 0 || n_cols == 0 {
        return;
    }
    let fill = |(group, chunk): (usize, &mut [B])| {
        let first = group * GROUP;
        let width = chunk.len() / n_rows;
        for r in 0..n_rows {
            let values = &row(r)[first..first + width];
            for (j, &value) in values.iter().enumerate() {
                chunk[j * n_rows + r] = value;
            }
        }
    };
    if rayon::current_num_threads() > 1 && n_rows.saturating_mul(n_cols) >= PARALLEL_BINNING_CELLS {
        columns
            .par_chunks_mut(GROUP * n_rows)
            .enumerate()
            .for_each(fill);
    } else {
        columns
            .chunks_mut(GROUP * n_rows)
            .enumerate()
            .for_each(fill);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::FeatureType;

    /// Rows of every width and layout: dense, missing values (a custom
    /// sentinel), mostly missing (no columns), a categorical feature, wide
    /// features (`u16` columns), many bins (`u32` rows), a row count off the
    /// bitmap's word and block boundaries, and CSR rows complete or not, in
    /// feature order or not.
    fn tables() -> Vec<(&'static str, DMatrix, usize)> {
        let dense = |rows: usize, cols: usize, missing_percent: usize, categorical: bool| {
            let values: Vec<f32> = (0..rows * cols)
                .map(|i| {
                    if (i * 7919 + i / cols * 104_729) % 100 < missing_percent {
                        -1.0
                    } else if categorical && i % cols == 1 {
                        (i / cols % 5) as f32
                    } else {
                        ((i / cols * 17 + i % cols * 31) % 1009) as f32
                    }
                })
                .collect();
            let data = DMatrix::from_dense_with_missing(&values, rows, cols, -1.0).unwrap();
            if categorical {
                let mut types = vec![FeatureType::Numerical; cols];
                types[1] = FeatureType::Categorical;
                data.with_feature_types(&types).unwrap()
            } else {
                data
            }
        };
        let csr = |sorted: bool, complete: bool| {
            let (rows, cols) = (1500, 3);
            let mut indptr = vec![0];
            let (mut indices, mut values) = (Vec::new(), Vec::new());
            for r in 0..rows {
                let mut row: Vec<u32> = (0..cols as u32)
                    .filter(|&c| complete || !(r + c as usize).is_multiple_of(3))
                    .collect();
                if !sorted {
                    row.reverse();
                }
                for c in row {
                    indices.push(c);
                    values.push(((r * 13 + c as usize * 7) % 101) as f32);
                }
                indptr.push(indices.len());
            }
            DMatrix::from_csr(indptr, indices, values, cols).unwrap()
        };
        vec![
            ("dense", dense(2500, 6, 0, false), 256),
            ("missing", dense(2500, 6, 30, false), 256),
            ("missing, ragged rows", dense(2049 + 37, 5, 30, false), 256),
            ("mostly missing", dense(2500, 6, 80, false), 256),
            ("categorical", dense(2500, 6, 30, true), 256),
            ("wide features", dense(2500, 4, 10, false), 600),
            ("many bins", dense(2500, 300, 10, false), 256),
            ("csr complete sorted", csr(true, true), 256),
            ("csr complete unsorted", csr(false, true), 256),
            ("csr sparse", csr(true, false), 256),
        ]
    }

    /// Both layouts hold each value's bin under the cuts (global in the
    /// rows, local in the columns), the missing bitmap exactly the missing
    /// values; the widths are the narrowest that fit; columns are kept from
    /// half full; `is_dense` marks complete rows in feature order; and
    /// serial and parallel binning agree.
    #[test]
    fn layouts_hold_each_values_bin() {
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for (name, data, max_bin) in tables() {
            let cuts = HistCuts::from_dmatrix(&data, max_bin);
            let bin = || GHistIndex::from_dmatrix(&data, cuts.clone());
            let index = parallel.install(bin);
            let (n, cols) = (data.n_rows(), data.n_cols());
            let mut complete = true;
            let mut entries = 0;
            for r in 0..n {
                let mut stored = Vec::new();
                data.for_row_entry(r, |f, v| stored.push((f, cuts.bin_of(f as usize, v))));
                complete &= stored.iter().map(|&(f, _)| f).eq(0..cols as u32);
                entries += stored.len();
                let row: Vec<u32> = match index.bins() {
                    Bins::U16(b) => b[index.row_ptr()[r]..index.row_ptr()[r + 1]]
                        .iter()
                        .map(|&b| u32::from(b))
                        .collect(),
                    Bins::U32(b) => b[index.row_ptr()[r]..index.row_ptr()[r + 1]].to_vec(),
                };
                assert_eq!(
                    row,
                    stored.iter().map(|&(_, b)| b).collect::<Vec<_>>(),
                    "{name}, row {r}"
                );
                for f in 0..cols {
                    let want = data.get(r, f).map(|v| cuts.bin_of(f, v));
                    assert_eq!(
                        index.feature_bin(r, f),
                        want,
                        "{name}, row {r}, feature {f}"
                    );
                }
            }
            assert_eq!(index.is_dense(), complete, "{name}");
            assert_eq!(index.columns().is_some(), 2 * entries >= n * cols, "{name}");
            assert_eq!(
                matches!(index.bins(), Bins::U16(_)),
                cuts.total_bins() <= 1 << 16,
                "{name}"
            );
            let widest = (0..cols).map(|f| cuts.num_bins(f)).max().unwrap();
            let missing = match index.columns() {
                Some(Columns::U8(set)) => {
                    assert!(widest <= 256, "{name}");
                    Some(set.missing)
                }
                Some(Columns::U16(set)) => {
                    assert!((257..=65_536).contains(&widest), "{name}");
                    Some(set.missing)
                }
                Some(Columns::U32(set)) => {
                    assert!(widest > 65_536, "{name}");
                    Some(set.missing)
                }
                None => None,
            };
            if let Some(missing) = missing {
                assert_eq!(
                    missing.is_some(),
                    entries < n * cols,
                    "{name}: bitmap only with missing entries"
                );
            }
            let same = serial.install(bin);
            assert_eq!(same.row_ptr, index.row_ptr, "{name}");
            assert_eq!(
                format!("{:?}", same.rows),
                format!("{:?}", index.rows),
                "{name}"
            );
            assert_eq!(
                format!("{:?}", same.columns),
                format!("{:?}", index.columns),
                "{name}"
            );
        }
    }

    /// The feature-major GPU store holds every present entry's global bin,
    /// and the width's sentinel exactly where a row lacks the feature, from
    /// the columns or (mostly missing data) the rows.
    #[cfg(any(all(target_os = "macos", feature = "metal"), feature = "wgpu"))]
    #[test]
    fn global_columns_hold_the_global_bins() {
        for (name, data, max_bin) in tables() {
            let index = GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, max_bin));
            let (n, cols) = (index.n_rows(), index.n_cols());
            let (columns, missing) = match index.global_columns() {
                GlobalBins::U16(b) => (
                    b.iter().map(|&b| u32::from(b)).collect::<Vec<_>>(),
                    u32::from(u16::MAX),
                ),
                GlobalBins::U32(b) => (b, u32::MAX),
            };
            for r in 0..n {
                for f in 0..cols {
                    let stored = columns[f * n + r];
                    let got = (index.is_dense() || stored != missing).then_some(stored);
                    assert_eq!(got, index.feature_bin(r, f), "{name}, row {r}, feature {f}");
                }
            }
        }
    }

    /// `feature_columns` holds each local bin, `u32::MAX` where missing,
    /// from the columns or (mostly missing data) the rows.
    #[test]
    fn feature_columns_match_row_lookups() {
        for (name, data, max_bin) in tables() {
            let index = GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, max_bin));
            for (f, column) in index.feature_columns().iter().enumerate() {
                let first = index.cuts().feature_bins(f).0 as u32;
                let want: Vec<u32> = (0..index.n_rows())
                    .map(|r| index.feature_bin(r, f).map_or(u32::MAX, |b| b - first))
                    .collect();
                assert_eq!(column, &want, "{name}, feature {f}");
            }
        }
    }

    /// Gathering rows equals binning a copy of them with the same cuts:
    /// rows, columns, missing bitmap and dense flag alike (the layout rules
    /// re-applied to the selection), for subsets that repeat, reorder, keep
    /// only complete rows of a sparse table, or drop below half full,
    /// serially and in parallel.
    #[test]
    fn gathered_rows_equal_binning_their_copy() {
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for (name, data, max_bin) in tables() {
            let n = data.n_rows();
            let complete: Vec<usize> = (0..n)
                .filter(|&r| (0..data.n_cols()).all(|f| data.get(r, f).is_some()))
                .collect();
            let sparse_rows: Vec<usize> = (0..n)
                .filter(|&r| {
                    (0..data.n_cols())
                        .filter(|&f| data.get(r, f).is_some())
                        .count()
                        <= 1
                })
                .collect();
            let selections: Vec<Vec<usize>> = vec![
                (0..n).rev().step_by(3).collect(),
                (0..n).flat_map(|r| [r, r]).take(n + 7).collect(),
                complete,
                sparse_rows,
            ];
            let index = GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, max_bin));
            for rows in selections.into_iter().filter(|rows| !rows.is_empty()) {
                let copy = copied_rows(&data, &rows);
                let want = GHistIndex::from_dmatrix(&copy, index.cuts().clone());
                for got in [
                    index.select_rows(&rows),
                    parallel.install(|| index.select_rows(&rows)),
                ] {
                    assert_eq!(got.row_ptr, want.row_ptr, "{name}, {} rows", rows.len());
                    assert_eq!(
                        format!("{:?}", got.rows),
                        format!("{:?}", want.rows),
                        "{name}"
                    );
                    assert_eq!(
                        format!("{:?}", got.columns),
                        format!("{:?}", want.columns),
                        "{name}"
                    );
                    assert_eq!(got.dense, want.dense, "{name}, {} rows", rows.len());
                }
            }
        }
    }

    /// A fresh matrix of `data`'s `rows` (their values read one by one, a
    /// CSR copy keeping each row's storage order), with its feature types.
    fn copied_rows(data: &DMatrix, rows: &[usize]) -> DMatrix {
        let n_cols = data.n_cols();
        let copy = if data.dense_rows().is_some() {
            let values: Vec<f32> = rows
                .iter()
                .flat_map(|&r| (0..n_cols).map(move |f| data.get(r, f).unwrap_or(f32::NAN)))
                .collect();
            DMatrix::from_dense_vec(values, rows.len(), n_cols).unwrap()
        } else {
            let (mut indptr, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
            for &r in rows {
                data.for_row_entry(r, |f, v| {
                    indices.push(f);
                    values.push(v);
                });
                indptr.push(values.len());
            }
            DMatrix::from_csr(indptr, indices, values, n_cols).unwrap()
        };
        copy.with_feature_types(data.feature_types()).unwrap()
    }
}
