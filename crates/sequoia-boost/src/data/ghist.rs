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

use crate::data::quantile::{BinSearch, HistCuts};
use crate::data::{DMatrix, Entry};
use rayon::prelude::*;
use std::ops::Range;

/// Backing storage for bin indices, in the narrowest width that fits.
#[derive(Debug, Clone)]
enum BinStore {
    U16(Vec<u16>),
    U32(Vec<u32>),
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
    /// Feature-major copy of `store` for dense indexes: feature `f` of row `r`
    /// is at `f * n_rows + r`. Doubles bin storage in exchange for streaming
    /// row partitions.
    columns: Option<BinStore>,
    cuts: HistCuts,
    /// True when every row is complete and in ascending feature order (a dense
    /// matrix with no missing values). Then feature `f` of row `r` is at offset
    /// `row_ptr[r] + f`, so routing needs no per-row scan.
    dense: bool,
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
        let store = if narrow {
            let mut bins = Vec::with_capacity(total);
            for chunk in chunks {
                if let BinChunk::U16(chunk) = chunk.bins {
                    bins.extend_from_slice(&chunk);
                }
            }
            BinStore::U16(bins)
        } else {
            let mut bins = Vec::with_capacity(total);
            for chunk in chunks {
                if let BinChunk::U32(chunk) = chunk.bins {
                    bins.extend_from_slice(&chunk);
                }
            }
            BinStore::U32(bins)
        };

        // A dense index also keeps a feature-major copy: routing rows on one
        // split feature then streams a single column instead of touching one
        // cache line per row.
        let columns = dense.then(|| match &store {
            BinStore::U16(bins) => BinStore::U16(transpose_dense(bins, n_rows, n_cols)),
            BinStore::U32(bins) => BinStore::U32(transpose_dense(bins, n_rows, n_cols)),
        });

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
        match &self.store {
            BinStore::U16(v) => Bins::U16(v),
            BinStore::U32(v) => Bins::U32(v),
        }
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
        self.columns.as_ref().map(|store| match store {
            BinStore::U16(v) => Bins::U16(v),
            BinStore::U32(v) => Bins::U32(v),
        })
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
            let idx = self.row_ptr[r] + feature;
            let b = match &self.store {
                BinStore::U16(v) => v[idx] as u32,
                BinStore::U32(v) => v[idx],
            };
            return Some(b); // dense entry at offset `feature` is that feature's bin
        }
        self.feature_bin(r, fs, fe)
    }

    /// The global bin of `feature` (whose global bin range is `[fs, fe)`) in row
    /// `r`, or `None` when that feature is missing for the row.
    #[inline]
    pub fn feature_bin(&self, r: usize, fs: usize, fe: usize) -> Option<u32> {
        let (s, e) = (self.row_ptr[r], self.row_ptr[r + 1]);
        match &self.store {
            BinStore::U16(v) => {
                for &b in &v[s..e] {
                    let b = b as usize;
                    if b >= fs && b < fe {
                        return Some(b as u32);
                    }
                }
            }
            BinStore::U32(v) => {
                for &b in &v[s..e] {
                    let b = b as usize;
                    if b >= fs && b < fe {
                        return Some(b as u32);
                    }
                }
            }
        }
        None
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

/// Bin indices of a row chunk, already in the index's storage width.
enum BinChunk {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl BinChunk {
    fn len(&self) -> usize {
        match self {
            BinChunk::U16(bins) => bins.len(),
            BinChunk::U32(bins) => bins.len(),
        }
    }
}

struct BinnedRows {
    row_ends: Vec<usize>,
    bins: BinChunk,
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
        bin_rows_into::<u16>(data, cuts, rows, BinChunk::U16)
    } else {
        bin_rows_into::<u32>(data, cuts, rows, BinChunk::U32)
    }
}

fn bin_rows_into<B: FromBin>(
    data: &DMatrix,
    cuts: &BinSearch<'_>,
    rows: Range<usize>,
    wrap: fn(Vec<B>) -> BinChunk,
) -> BinnedRows {
    let n_features = cuts.n_features();
    let mut row_ends = Vec::with_capacity(rows.len());
    let mut bins: Vec<B> = Vec::with_capacity(rows.len() * n_features);
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
        let mut row: Vec<Entry> = Vec::new();
        for r in rows {
            data.row_into(r, &mut row);
            dense &= row.len() == n_features
                && row.iter().enumerate().all(|(c, e)| e.index as usize == c);
            for e in &row {
                push(&mut bins, cuts.bin_of(e.index as usize, e.value));
            }
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
        // Two present features per row.
        assert_eq!(ghist.row_len(0), 2);
        assert_eq!(ghist.row_len(1), 2);
        // Row 1's feature-0 value (1.0) bins higher than row 0's (0.0).
        let b0 = ghist.cuts().bin_of(0, 0.0);
        let b1 = ghist.cuts().bin_of(0, 1.0);
        assert!(b1 > b0);
        // Small bin count -> u16 storage.
        assert!(matches!(ghist.bins(), Bins::U16(_)));
    }

    #[test]
    fn missing_entries_absent() {
        let data = DMatrix::from_dense(&[0.0, f32::NAN, 1.0, 2.0], 2, 2).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        // Row 0 has a missing feature 1 -> only one present entry.
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
