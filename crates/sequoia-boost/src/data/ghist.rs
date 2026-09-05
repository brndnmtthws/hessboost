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

use crate::data::quantile::HistCuts;
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
#[derive(Debug, Clone)]
pub struct GHistIndex {
    n_rows: usize,
    n_cols: usize,
    row_ptr: Vec<usize>,
    store: BinStore,
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
        let threads = rayon::current_num_threads();
        let chunks: Vec<_> = if threads > 1 && n_rows.saturating_mul(n_cols) >= 65_536 {
            let grain = n_rows.div_ceil(threads).max(1024);
            (0..n_rows.div_ceil(grain))
                .into_par_iter()
                .map(|chunk| {
                    bin_rows(
                        data,
                        &cuts,
                        chunk * grain..((chunk + 1) * grain).min(n_rows),
                    )
                })
                .collect()
        } else {
            vec![bin_rows(data, &cuts, 0..n_rows)]
        };
        let total = chunks.iter().map(|chunk| chunk.bins.len()).sum();
        let dense = chunks.iter().all(|chunk| chunk.dense);
        let mut row_ptr = Vec::with_capacity(n_rows + 1);
        row_ptr.push(0);
        let mut offset = 0;
        for chunk in &chunks {
            row_ptr.extend(chunk.row_ends.iter().map(|end| offset + end));
            offset += chunk.bins.len();
        }
        // Convert directly into the final width without an intermediate merged
        // u32 buffer. Chunk order preserves the input rows and feature order.
        let store = if cuts.total_bins() <= u16::MAX as usize + 1 {
            let mut bins = Vec::with_capacity(total);
            for chunk in chunks {
                bins.extend(chunk.bins.into_iter().map(|bin| bin as u16));
            }
            BinStore::U16(bins)
        } else {
            let mut bins = Vec::with_capacity(total);
            for chunk in chunks {
                bins.extend(chunk.bins);
            }
            BinStore::U32(bins)
        };

        GHistIndex {
            n_rows,
            n_cols,
            row_ptr,
            store,
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

struct BinnedRows {
    row_ends: Vec<usize>,
    bins: Vec<u32>,
    dense: bool,
}

fn bin_rows(data: &DMatrix, cuts: &HistCuts, rows: Range<usize>) -> BinnedRows {
    let mut chunk = BinnedRows {
        row_ends: Vec::with_capacity(rows.len()),
        bins: Vec::new(),
        dense: true,
    };
    let mut row: Vec<Entry> = Vec::new();
    for r in rows {
        data.row_into(r, &mut row);
        chunk.dense &= row.len() == cuts.n_features()
            && row.iter().enumerate().all(|(c, e)| e.index as usize == c);
        chunk
            .bins
            .extend(row.iter().map(|e| cuts.bin_of(e.index as usize, e.value)));
        chunk.row_ends.push(chunk.bins.len());
    }
    chunk
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
