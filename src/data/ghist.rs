//! Pre-binned feature storage (`GHistIndexMatrix` in XGBoost).
//!
//! Every non-missing entry is replaced by its global bin index (see
//! [`HistCuts`]). This compact, CSR-shaped layout is what the histogram builder
//! scans: accumulating gradients into per-bin buckets is a single indexed add.
//!
//! Bin indices are stored in the **narrowest** integer type that fits the total
//! bin count. It uses `u16` when there are at most 65,536 bins (the common case, e.g. 256
//! features × 256 bins), else `u32`. The build loop is memory-bandwidth bound,
//! so halving the index width is a direct throughput win.

use crate::data::DMatrix;
use crate::data::quantile::{BinSearch, HistCuts};
use rayon::prelude::*;
use std::ops::Range;

/// Backing storage for bin indices, in the narrowest width that fits.
#[derive(Debug, Clone)]
enum BinStore {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl BinStore {
    /// All bin indices as a width-tagged view.
    fn as_bins(&self) -> Bins<'_> {
        match self {
            BinStore::U16(v) => Bins::U16(v),
            BinStore::U32(v) => Bins::U32(v),
        }
    }

    /// The global bin index at storage position `idx`.
    #[inline]
    fn get(&self, idx: usize) -> u32 {
        match self {
            BinStore::U16(v) => u32::from(v[idx]),
            BinStore::U32(v) => v[idx],
        }
    }

    /// Number of stored bin indices.
    fn len(&self) -> usize {
        match self {
            BinStore::U16(v) => v.len(),
            BinStore::U32(v) => v.len(),
        }
    }

    /// Append all indices of `other`, which must have the same width.
    fn extend(&mut self, other: BinStore) {
        match (self, other) {
            (BinStore::U16(dst), BinStore::U16(src)) => dst.extend_from_slice(&src),
            (BinStore::U32(dst), BinStore::U32(src)) => dst.extend_from_slice(&src),
            _ => unreachable!("concatenated chunks share the index width"),
        }
    }

    /// The bin of the feature owning global range `[fs, fe)` within the stored
    /// slice `[s, e)`, or `None` when that feature is absent there.
    fn find(&self, s: usize, e: usize, fs: usize, fe: usize) -> Option<u32> {
        match self {
            BinStore::U16(v) => find_bin(&v[s..e], fs, fe),
            BinStore::U32(v) => find_bin(&v[s..e], fs, fe),
        }
    }
}

/// The bin of the feature owning global range `[fs, fe)` within one row's
/// slice, or `None` when that feature is missing for the row.
#[inline]
fn find_bin<T: Copy + Into<u32>>(row: &[T], fs: usize, fe: usize) -> Option<u32> {
    for &b in row {
        let b: u32 = b.into();
        if (b as usize) >= fs && (b as usize) < fe {
            return Some(b);
        }
    }
    None
}

/// A view over one row's (or all rows') bin indices, tagged by width so hot
/// loops can specialize with a single outer branch.
pub enum Bins<'a> {
    /// 16-bit bin indices.
    U16(&'a [u16]),
    /// 32-bit bin indices.
    U32(&'a [u32]),
}

/// Binned dataset: for each row, the global bin indices of its non-missing
/// features, stored CSR-style.
///
/// Invariant: every stored bin index is below [`GHistIndex::total_bins`], so a
/// histogram of that length can be indexed by any stored bin without bounds
/// checks. `from_dmatrix` is the only constructor and verifies this.
#[derive(Debug, Clone)]
pub struct GHistIndex {
    n_rows: usize,
    n_cols: usize,
    row_ptr: Vec<usize>,
    store: BinStore,
    /// Feature-major copy of `store` (feature `f` of row `r` at `f * n_rows +
    /// r`), trading up to double the bin storage for streaming partitions
    /// and column-wise histograms.
    columns: Columns,
    cuts: HistCuts,
    /// True when every row is complete and in ascending feature order (a dense
    /// matrix with no missing values). Then feature `f` of row `r` is at offset
    /// `row_ptr[r] + f`, so routing needs no per-row scan.
    dense: bool,
}

/// The feature-major copy a [`GHistIndex`] keeps, if any.
#[derive(Debug, Clone)]
enum Columns {
    None,
    /// Every row holds every feature ([`GHistIndex::column_bins`]).
    Dense(BinStore),
    /// Missing entries hold the width's maximum
    /// ([`GHistIndex::missing_columns`]).
    WithMissing(BinStore),
}

impl GHistIndex {
    /// Bin a dataset against precomputed cuts.
    pub fn from_dmatrix(data: &DMatrix, cuts: HistCuts) -> Self {
        let n_rows = data.n_rows();
        let n_cols = cuts.n_features();
        let total_bins = cuts.total_bins();
        let narrow = total_bins <= u16::MAX as usize + 1;
        let threads = rayon::current_num_threads();
        let search = BinSearch::new(&cuts);
        let chunks: Vec<_> = if threads > 1 && n_rows.saturating_mul(n_cols) >= 65_536 {
            let grain = n_rows.div_ceil(threads).max(1024);
            (0..n_rows.div_ceil(grain))
                .into_par_iter()
                .map(|chunk| {
                    bin_rows(
                        data,
                        &search,
                        chunk * grain..((chunk + 1) * grain).min(n_rows),
                        narrow,
                    )
                })
                .collect()
        } else {
            vec![bin_rows(data, &search, 0..n_rows, narrow)]
        };
        drop(search);
        let total = chunks.iter().map(|chunk| chunk.bins.len()).sum();
        let dense = chunks.iter().all(|chunk| chunk.dense);
        // The chunk maxima establish the bin-range invariant documented on the
        // type.
        let max_bin = chunks.iter().map(|chunk| chunk.max_bin).max().unwrap_or(0);
        assert!(
            total == 0 || (max_bin as usize) < total_bins,
            "binned index {max_bin} is outside the {total_bins} histogram bins"
        );
        let mut row_ptr = Vec::with_capacity(n_rows + 1);
        row_ptr.push(0);
        let mut offset = 0;
        for chunk in &chunks {
            row_ptr.extend(chunk.row_ends.iter().map(|end| offset + end));
            offset += chunk.bins.len();
        }
        // Chunks were binned in the final width; concatenation preserves the
        // input rows and feature order.
        let mut store = if narrow {
            BinStore::U16(Vec::with_capacity(total))
        } else {
            BinStore::U32(Vec::with_capacity(total))
        };
        for chunk in chunks {
            store.extend(chunk.bins);
        }

        // A dense index keeps a feature-major copy: routing rows on one split
        // feature then streams a single column instead of touching one cache
        // line per row. A sparse index at least half full keeps one too, with
        // a sentinel for missing entries.
        let columns = if dense {
            Columns::Dense(match &store {
                BinStore::U16(bins) => BinStore::U16(transpose_dense(bins, n_rows, n_cols)),
                BinStore::U32(bins) => BinStore::U32(transpose_dense(bins, n_rows, n_cols)),
            })
        } else if n_rows.saturating_mul(n_cols) <= total.saturating_mul(2) {
            let bin_feature = bin_features(&cuts);
            match &store {
                BinStore::U16(bins) if total_bins < u16::MAX as usize => {
                    Columns::WithMissing(BinStore::U16(transpose_sparse(
                        bins,
                        &row_ptr,
                        n_cols,
                        &bin_feature,
                        u16::MAX,
                    )))
                }
                BinStore::U16(_) => Columns::None,
                BinStore::U32(bins) => Columns::WithMissing(BinStore::U32(transpose_sparse(
                    bins,
                    &row_ptr,
                    n_cols,
                    &bin_feature,
                    u32::MAX,
                ))),
            }
        } else {
            Columns::None
        };

        GHistIndex {
            n_rows,
            n_cols,
            row_ptr,
            store,
            columns,
            cuts,
            dense,
        }
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

    /// CSR row-offset table (length `n_rows + 1`).
    #[inline]
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// All bin indices, tagged by width. Combine with [`GHistIndex::row_ptr`] to
    /// slice a row's entries in a width-specialized loop.
    #[inline]
    pub fn bins(&self) -> Bins<'_> {
        self.store.as_bins()
    }

    /// Row stride of a dense index (every row holds every feature in order),
    /// or `None` when rows must be located through [`GHistIndex::row_ptr`].
    #[inline]
    pub fn dense_stride(&self) -> Option<usize> {
        self.dense.then_some(self.n_cols)
    }

    /// Feature-major bin indices of a dense index (`f * n_rows + r`), tagged by
    /// width. `None` for sparse indexes.
    #[inline]
    pub fn column_bins(&self) -> Option<Bins<'_>> {
        match &self.columns {
            Columns::Dense(store) => Some(store.as_bins()),
            _ => None,
        }
    }

    /// Feature-major bin indices (`f * n_rows + r`) of a sparse index that
    /// is at least half full, with the width's maximum (`u16::MAX` or
    /// `u32::MAX`, never a bin) where row `r` lacks feature `f`. `None` for
    /// dense indexes (see [`GHistIndex::column_bins`]) and sparser ones.
    #[inline]
    pub fn missing_columns(&self) -> Option<Bins<'_>> {
        match &self.columns {
            Columns::WithMissing(store) => Some(store.as_bins()),
            _ => None,
        }
    }

    /// Number of present (non-missing) entries in row `r`.
    #[inline]
    pub fn row_len(&self, r: usize) -> usize {
        self.row_ptr[r + 1] - self.row_ptr[r]
    }

    /// The global bin of `feature` (with global bin range `[fs, fe)`) in row `r`,
    /// or `None` if missing. Uses an O(1) direct index for dense datasets and
    /// falls back to a per-row scan otherwise.
    #[inline]
    pub fn feature_bin_at(&self, r: usize, feature: usize, fs: usize, fe: usize) -> Option<u32> {
        if self.dense {
            // Dense entry at offset `feature` is that feature's bin.
            return Some(self.store.get(self.row_ptr[r] + feature));
        }
        self.feature_bin(r, fs, fe)
    }

    /// The global bin of `feature` (whose global bin range is `[fs, fe)`) in row
    /// `r`, or `None` when that feature is missing for the row.
    #[inline]
    pub fn feature_bin(&self, r: usize, fs: usize, fe: usize) -> Option<u32> {
        let (s, e) = (self.row_ptr[r], self.row_ptr[r + 1]);
        self.store.find(s, e, fs, fe)
    }
}

/// Feature-major copy of a dense row-major matrix (`n_rows * n_cols` entries,
/// row `r` at `r * n_cols`). Features are handled in groups so each pass over
/// the rows reads a short contiguous run per row and writes a few sequential
/// column streams.
pub(crate) fn transpose_dense<B: Copy + Default + Send + Sync>(
    bins: &[B],
    n_rows: usize,
    n_cols: usize,
) -> Vec<B> {
    const GROUP: usize = 8;
    let mut columns = vec![B::default(); n_rows * n_cols];
    if n_rows == 0 || n_cols == 0 {
        return columns;
    }
    let fill = |(group, chunk): (usize, &mut [B])| {
        let first = group * GROUP;
        let width = chunk.len() / n_rows;
        for r in 0..n_rows {
            let row = &bins[r * n_cols + first..r * n_cols + first + width];
            for (j, &bin) in row.iter().enumerate() {
                chunk[j * n_rows + r] = bin;
            }
        }
    };
    if rayon::current_num_threads() > 1 && n_rows.saturating_mul(n_cols) >= 65_536 {
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
    columns
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

/// Feature-major copy of a sparse index (`bins` sliced by `row_ptr`): feature
/// `f` of row `r` at `f * n_rows + r`, `missing` where the row has no entry
/// for `f`. A row holding several entries of one feature keeps the first, as
/// [`GHistIndex::feature_bin`] finds it. Features are filled in groups, each
/// group scanning every row.
fn transpose_sparse<B: Copy + Into<u32> + PartialEq + Send + Sync>(
    bins: &[B],
    row_ptr: &[usize],
    n_cols: usize,
    bin_feature: &[u32],
    missing: B,
) -> Vec<B> {
    const GROUP: usize = 8;
    let n_rows = row_ptr.len() - 1;
    let mut columns = vec![missing; n_rows * n_cols];
    if n_rows == 0 || n_cols == 0 {
        return columns;
    }
    let fill = |(group, chunk): (usize, &mut [B])| {
        let first = group * GROUP;
        let width = chunk.len() / n_rows;
        for r in 0..n_rows {
            for &bin in &bins[row_ptr[r]..row_ptr[r + 1]] {
                let j = (bin_feature[bin.into() as usize] as usize).wrapping_sub(first);
                if j < width {
                    let slot = &mut chunk[j * n_rows + r];
                    if *slot == missing {
                        *slot = bin;
                    }
                }
            }
        }
    };
    if rayon::current_num_threads() > 1 && n_rows.saturating_mul(n_cols) >= 65_536 {
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
    columns
}

struct BinnedRows {
    row_ends: Vec<usize>,
    bins: BinStore,
    dense: bool,
    /// Largest bin index in the chunk (0 when empty).
    max_bin: u32,
}

/// Storage widths a bin index can be narrowed to.
trait FromBin: Copy {
    fn from_bin(bin: u32) -> Self;
}

impl FromBin for u16 {
    #[inline(always)]
    fn from_bin(bin: u32) -> Self {
        bin as u16
    }
}

impl FromBin for u32 {
    #[inline(always)]
    fn from_bin(bin: u32) -> Self {
        bin
    }
}

fn bin_rows(data: &DMatrix, cuts: &BinSearch<'_>, rows: Range<usize>, narrow: bool) -> BinnedRows {
    if narrow {
        bin_rows_into::<u16>(data, cuts, rows, BinStore::U16)
    } else {
        bin_rows_into::<u32>(data, cuts, rows, BinStore::U32)
    }
}

fn bin_rows_into<B: FromBin>(
    data: &DMatrix,
    cuts: &BinSearch<'_>,
    rows: Range<usize>,
    wrap: fn(Vec<B>) -> BinStore,
) -> BinnedRows {
    let n_features = cuts.n_features();
    let mut row_ends = Vec::with_capacity(rows.len());
    // Sparse storage pushes only stored entries, so reserve by entry count.
    // A dense-sized reservation would request rows x features capacity even
    // when the row range holds a fraction of that in stored entries.
    let nnz = match data.csr_parts() {
        Some((indptr, _, _)) => indptr[rows.end] - indptr[rows.start],
        None => rows.len() * n_features,
    };
    let mut bins: Vec<B> = Vec::with_capacity(nnz);
    let mut dense = true;
    let mut max_bin = 0u32;
    let mut push = |bins: &mut Vec<B>, bin: u32| {
        max_bin = max_bin.max(bin);
        bins.push(B::from_bin(bin));
    };
    if let Some(values) = data.dense_values() {
        // Dense storage: read the row in place; features are already ascending.
        let missing = data.missing();
        for r in rows {
            let row = &values[r * n_features..(r + 1) * n_features];
            let start = bins.len();
            for (c, &v) in row.iter().enumerate() {
                if !crate::data::dmatrix::is_missing(v, missing) {
                    push(&mut bins, cuts.bin_of(c, v));
                }
            }
            dense &= bins.len() - start == n_features;
            row_ends.push(bins.len());
        }
    } else {
        for r in rows {
            // A row is dense when it stores every feature in order.
            let mut stored = 0usize;
            let mut in_order = true;
            data.for_row_entry(r, |index, value| {
                in_order &= index as usize == stored;
                stored += 1;
                push(&mut bins, cuts.bin_of(index as usize, value));
            });
            dense &= in_order && stored == n_features;
            row_ends.push(bins.len());
        }
    }
    BinnedRows {
        row_ends,
        bins: wrap(bins),
        dense,
        max_bin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_binning_preserves_cuts_rows_and_width() {
        use crate::data::FeatureType;
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for (rows, cols, missing, categorical) in [
            (1025, 80, false, false),
            (257, 300, false, false),
            (1025, 80, true, false),
            (1025, 80, false, true),
        ] {
            let values: Vec<_> = (0..rows * cols)
                .map(|i| {
                    if missing && i % 11 < 2 {
                        -1.0
                    } else if categorical && i % cols == 0 {
                        (i / cols % 4) as f32
                    } else {
                        ((i / cols * 17 + i % cols * 31) % 509) as f32
                    }
                })
                .collect();
            let mut data = DMatrix::from_dense_with_missing(&values, rows, cols, -1.0).unwrap();
            if categorical {
                let mut types = vec![FeatureType::Numerical; cols];
                types[0] = FeatureType::Categorical;
                data = data.with_feature_types(&types).unwrap();
            }
            let bin = || GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, 256));
            let expected = serial.install(bin);
            let actual = parallel.install(bin);
            assert_eq!(
                serde_json::to_value(actual.cuts()).unwrap(),
                serde_json::to_value(expected.cuts()).unwrap()
            );
            assert_eq!(actual.row_ptr, expected.row_ptr);
            assert_eq!(actual.dense, expected.dense);
            match (&actual.store, &expected.store) {
                (BinStore::U16(a), BinStore::U16(b)) => {
                    assert_eq!(a, b);
                    assert_eq!(cols, 80);
                }
                (BinStore::U32(a), BinStore::U32(b)) => {
                    assert_eq!(a, b);
                    assert_eq!(cols, 300);
                }
                _ => panic!("bin width changed"),
            }
        }
    }

    #[test]
    fn bins_roundtrip_dense() {
        let data = DMatrix::from_dense(&[0.0, 10.0, 1.0, 20.0], 2, 2).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        assert_eq!(ghist.row_len(0), 2);
        assert_eq!(ghist.row_len(1), 2);
        // Row 1's feature-0 value (1.0) bins higher than row 0's (0.0).
        let b0 = ghist.cuts().bin_of(0, 0.0);
        let b1 = ghist.cuts().bin_of(0, 1.0);
        assert!(b1 > b0);
        assert!(matches!(ghist.bins(), Bins::U16(_)));
    }

    #[test]
    fn missing_entries_absent() {
        let data = DMatrix::from_dense(&[0.0, f32::NAN, 1.0, 2.0], 2, 2).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        assert_eq!(ghist.row_len(0), 1);
        assert_eq!(ghist.row_len(1), 2);
    }

    #[test]
    fn feature_bin_lookup() {
        let data = DMatrix::from_dense(&[0.0, 10.0, 1.0, 20.0], 2, 2).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let (f0s, f0e) = cuts.feature_bins(0);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        let b = ghist.feature_bin(0, f0s, f0e).unwrap();
        assert!((b as usize) >= f0s && (b as usize) < f0e);
        // A feature range with no entry for a fully-present row still resolves.
        assert!(ghist.feature_bin(1, f0s, f0e).is_some());
    }
}
