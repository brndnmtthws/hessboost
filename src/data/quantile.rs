//! Quantile cut computation for histogram-based training.
//!
//! For each numeric feature we compute up to `max_bin` cut points with
//! XGBoost's weighted quantile sketch (the private `sketch` module), then map any
//! value to a bin with an `upper_bound` search (`bin = #{cuts ≤ value}`,
//! clamped). The cuts are exactly XGBoost's for the same data, weights, and
//! `max_bin`: the minimum value is never a cut (bin 0 is `(-inf, cut0]`) and a
//! trailing sentinel above the maximum closes the last bin. With
//! `tree_method=hist` the cuts are computed **once** from the data and its
//! sample weights ([`HistCuts::from_dmatrix`]); with `tree_method=approx`
//! they are recomputed each boosting round from the current Hessians
//! ([`HistCuts::from_dmatrix_weighted`]).

use crate::data::DMatrix;
use crate::data::meta::FeatureType;
use crate::data::sketch::WQSketch;
use crate::data::sort::{RadixScratch, sort_values};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Per-feature histogram cut points, laid out contiguously.
///
/// Feature `f` owns cut values `cut_values[feature_offset[f]..feature_offset[f+1]]`
/// and its bins occupy the same global index range, so `feature_offset` doubles
/// as both the cut-pointer and the global-bin offset table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistCuts {
    n_features: usize,
    /// Global bin/cut offsets, length `n_features + 1`.
    feature_offset: Vec<u32>,
    /// Concatenated ascending cut values. For a numeric feature these are split
    /// thresholds. For a categorical feature they are the distinct category
    /// values, one per bin (see `is_categorical`).
    cut_values: Vec<f32>,
    /// Per-feature flag: `true` when the feature is categorical and its bins map
    /// one category value each (no threshold semantics). Length `n_features`.
    is_categorical: Vec<bool>,
}

/// Global bin index for an `upper_bound` count `local` within a feature owning
/// `n_cuts` cuts starting at `start`: clamp into the last real bin.
#[inline]
fn global_bin(start: usize, local: usize, n_cuts: usize) -> u32 {
    start as u32 + local.min(n_cuts.saturating_sub(1)) as u32
}

/// How [`HistCuts::build`] feeds a numeric feature's sketch.
#[derive(Clone, Copy)]
struct Ingest {
    /// Sort the whole column first (upstream `PushColPage`) instead of
    /// streaming it in row order (`PushRowPage`).
    sorted: bool,
    /// Every weight is `1` (see [`WQSketch::with_unit_weights`]).
    unit_weights: bool,
}

impl HistCuts {
    /// Compute cuts from a dataset with at most `max_bin` bins per feature.
    ///
    /// Categorical features (per [`DMatrix::feature_types`]) are binned with one
    /// bin per distinct category value. Numeric features use XGBoost's
    /// streaming quantile sketch over the values in row order, weighted by the
    /// sample weights when present (upstream `PushRowPage`).
    pub fn from_dmatrix(data: &DMatrix, max_bin: usize) -> Self {
        let weights = data.weights();
        let unit_weights = weights.is_none();
        Self::build(
            data,
            max_bin,
            |row| weights.map_or(1.0, |w| w[row]),
            Ingest {
                sorted: false,
                unit_weights,
            },
        )
    }

    /// Compute **Hessian-weighted** cuts, as XGBoost's `tree_method=approx`
    /// does.
    ///
    /// Each value is weighted by `hessians[row] * sample_weight[row]`, exactly
    /// as upstream multiplies the (already weighted) Hessian by the sample
    /// weight again. `sorted` selects upstream's ingestion path: `false` is the
    /// streaming sketch XGBoost keeps for constant-Hessian objectives
    /// (`PushRowPage`), `true` the sorted-column summary it rebuilds every
    /// round otherwise (`PushColPage`). Categorical features are binned as in
    /// [`HistCuts::from_dmatrix`]. `hessians` is indexed by original row and
    /// must cover every row of `data`.
    pub fn from_dmatrix_weighted(
        data: &DMatrix,
        max_bin: usize,
        hessians: &[f32],
        sorted: bool,
    ) -> Self {
        Self::from_dmatrix_hessians(data, max_bin, |row| hessians[row], sorted)
    }

    /// [`Self::from_dmatrix_weighted`] with row `row`'s Hessian read through
    /// `hessian`, so callers need not collect them.
    pub(crate) fn from_dmatrix_hessians(
        data: &DMatrix,
        max_bin: usize,
        hessian: impl Fn(usize) -> f32 + Sync,
        sorted: bool,
    ) -> Self {
        let weights = data.weights();
        Self::build(
            data,
            max_bin,
            |row| weights.map_or(hessian(row), |w| hessian(row) * w[row]),
            Ingest {
                sorted,
                unit_weights: false,
            },
        )
    }

    fn build(
        data: &DMatrix,
        max_bin: usize,
        weight_of: impl Fn(usize) -> f32 + Sync,
        ingest: Ingest,
    ) -> Self {
        let n_rows = data.n_rows();
        let n_features = data.n_cols();
        let is_categorical: Vec<bool> = data
            .feature_types()
            .iter()
            .map(|&t| t == FeatureType::Categorical)
            .collect();
        let parallel = n_features > 1
            && n_rows.saturating_mul(n_features) >= 65_536
            && rayon::current_num_threads() > 1;
        let per_feature: Vec<Vec<f32>> = if let Some(values) = data.dense_values() {
            let missing = data.missing();
            let present = present_counts(values, n_features, missing, parallel);
            let mut columns: Vec<ColumnCuts> = (0..n_features)
                .map(|f| ColumnCuts::new(is_categorical[f], present[f], max_bin, ingest))
                .collect();
            // Rows stream through in blocks, each transposed so a
            // feature's values are contiguous: every feature's sketch
            // takes its values in row order, as a whole-column pass
            // would, while only one block is ever transposed.
            let mut block = Vec::new();
            for start in (0..n_rows).step_by(CUT_BLOCK_ROWS) {
                let end = (start + CUT_BLOCK_ROWS).min(n_rows);
                let len = end - start;
                let rows = &values[start * n_features..end * n_features];
                crate::data::ghist::transpose_dense_into(rows, len, n_features, &mut block);
                let feed = |(f, column): (usize, &mut ColumnCuts)| {
                    column.push_column(&block[f * len..][..len], start, missing, &weight_of);
                };
                if parallel {
                    columns.par_iter_mut().enumerate().for_each(feed);
                } else {
                    columns.iter_mut().enumerate().for_each(feed);
                }
            }
            drop(block);
            if parallel {
                columns
                    .into_par_iter()
                    .map(ColumnCuts::cut_values)
                    .collect()
            } else {
                columns.into_iter().map(ColumnCuts::cut_values).collect()
            }
        } else {
            // Sparse storage goes through its CSC view: each feature's
            // present entries, in row order, summarized one feature at
            // a time.
            let csc = data.to_csc();
            let column_cuts = |f: usize| {
                let (rows, values) = csc.column(f);
                let mut column = ColumnCuts::new(is_categorical[f], rows.len(), max_bin, ingest);
                column.push_entries(rows, values, &weight_of);
                column.cut_values()
            };
            if parallel {
                (0..n_features).into_par_iter().map(column_cuts).collect()
            } else {
                (0..n_features).map(column_cuts).collect()
            }
        };
        let mut feature_offset = Vec::with_capacity(n_features + 1);
        feature_offset.push(0u32);
        let mut cut_values = Vec::with_capacity(per_feature.iter().map(Vec::len).sum());
        for cuts in per_feature {
            cut_values.extend(cuts);
            feature_offset.push(cut_values.len() as u32);
        }
        HistCuts {
            n_features,
            feature_offset,
            cut_values,
            is_categorical,
        }
    }

    /// Whether feature `f` is categorical (bins map one category value each).
    #[inline]
    pub fn is_categorical(&self, f: usize) -> bool {
        self.is_categorical[f]
    }

    /// Number of features.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Total number of bins across all features (the histogram length).
    #[inline]
    pub fn total_bins(&self) -> usize {
        self.cut_values.len()
    }

    /// Global bin range `[start, end)` owned by feature `f`.
    #[inline]
    pub fn feature_bins(&self, f: usize) -> (usize, usize) {
        (
            self.feature_offset[f] as usize,
            self.feature_offset[f + 1] as usize,
        )
    }

    /// Number of bins for feature `f`.
    #[inline]
    pub fn num_bins(&self, f: usize) -> usize {
        (self.feature_offset[f + 1] - self.feature_offset[f]) as usize
    }

    /// The cut value at a global bin index (its exclusive upper threshold: an
    /// instance goes left of a split here when `value < cut_value(bin)`).
    #[inline]
    pub fn cut_value(&self, global_bin: usize) -> f32 {
        self.cut_values[global_bin]
    }

    /// Map a feature value to its **global** bin index.
    #[inline]
    pub fn bin_of(&self, f: usize, value: f32) -> u32 {
        let (start, end) = self.feature_bins(f);
        let slice = &self.cut_values[start..end];
        if self.is_categorical[f] {
            // Categorical: each bin holds one category value; find the exact
            // bin. Unseen categories (absent at fit time) clamp to bin 0.
            let local = slice
                .binary_search_by(|c| c.partial_cmp(&value).unwrap())
                .unwrap_or(0);
            return start as u32 + local as u32;
        }
        // upper_bound: first cut strictly greater than value.
        global_bin(start, slice.partition_point(|&c| c <= value), slice.len())
    }
}

/// Cuts per block of the two-level bin search.
const SEARCH_BLOCK: usize = 16;

/// Two-level search index over a cut table for binning many values quickly.
///
/// Each numeric feature's cuts are padded with `+inf` to whole blocks of
/// `SEARCH_BLOCK`, and a first-level table holds every block's last cut. A lookup
/// counts the first-level entries `<= value` (whole blocks below the value),
/// then the cuts `<= value` inside the next block. Both counts are branch-free
/// vector compares, and the result equals `partition_point(|c| c <= value)`.
pub struct BinSearch<'a> {
    cuts: &'a HistCuts,
    /// Where each feature's tables are, one load per lookup.
    features: Vec<FeatureSearch>,
    /// Padded cuts of every numeric feature.
    padded: Vec<f32>,
    /// Last cut of each block, padded with `+inf` to whole blocks.
    level1: Vec<f32>,
}

/// One feature's slice of the [`BinSearch`] tables.
#[derive(Debug, Clone, Copy)]
struct FeatureSearch {
    /// First global bin.
    start: usize,
    /// Number of cuts (bins).
    n_cuts: usize,
    /// Start of the padded cuts; `padded_len` of them.
    padded: usize,
    padded_len: usize,
    /// Start of the first-level table; `level1_len` entries, a multiple of
    /// `SEARCH_BLOCK`.
    level1: usize,
    level1_len: usize,
    categorical: bool,
}

impl<'a> BinSearch<'a> {
    /// Build the index for every numeric feature of `cuts`.
    pub fn new(cuts: &'a HistCuts) -> Self {
        let mut padded = Vec::new();
        let mut level1 = Vec::new();
        let mut features = Vec::with_capacity(cuts.n_features());
        for f in 0..cuts.n_features() {
            let (start, end) = cuts.feature_bins(f);
            let (padded_start, level1_start) = (padded.len(), level1.len());
            let categorical = cuts.is_categorical(f);
            if !categorical {
                let feature = &cuts.cut_values[start..end];
                let blocks = feature.len().div_ceil(SEARCH_BLOCK);
                padded.extend_from_slice(feature);
                padded.resize(padded_start + blocks * SEARCH_BLOCK, f32::INFINITY);
                level1.extend(
                    feature
                        .chunks(SEARCH_BLOCK)
                        .map(|block| *block.last().expect("blocks are non-empty")),
                );
                level1.resize(
                    level1_start + blocks.div_ceil(SEARCH_BLOCK) * SEARCH_BLOCK,
                    f32::INFINITY,
                );
            }
            features.push(FeatureSearch {
                start,
                n_cuts: end - start,
                padded: padded_start,
                padded_len: padded.len() - padded_start,
                level1: level1_start,
                level1_len: level1.len() - level1_start,
                categorical,
            });
        }
        BinSearch {
            cuts,
            features,
            padded,
            level1,
        }
    }

    /// Feature `f`'s tables, for binning many of its values in a row.
    #[inline(always)]
    pub(crate) fn feature(&self, f: usize) -> FeatureBins<'_> {
        let feature = self.features[f];
        FeatureBins {
            cuts: self.cuts,
            f,
            categorical: feature.categorical,
            start: feature.start,
            n_cuts: feature.n_cuts,
            level1: &self.level1[feature.level1..][..feature.level1_len],
            padded: &self.padded[feature.padded..][..feature.padded_len],
        }
    }
}

/// One feature's view of a [`BinSearch`] ([`BinSearch::feature`]).
#[derive(Clone, Copy)]
pub(crate) struct FeatureBins<'a> {
    cuts: &'a HistCuts,
    f: usize,
    categorical: bool,
    start: usize,
    n_cuts: usize,
    level1: &'a [f32],
    padded: &'a [f32],
}

impl FeatureBins<'_> {
    /// [`HistCuts::bin_of`] for this feature, through the two-level tables.
    #[inline(always)]
    pub(crate) fn bin_of(&self, value: f32) -> u32 {
        if self.categorical {
            return self.cuts.bin_of(self.f, value);
        }
        let mut block = 0;
        for chunk in self.level1.as_chunks::<SEARCH_BLOCK>().0 {
            block += crate::simd::count_le(chunk, value);
        }
        let local = if block * SEARCH_BLOCK < self.padded.len() {
            block * SEARCH_BLOCK
                + crate::simd::count_le(
                    &self.padded[block * SEARCH_BLOCK..(block + 1) * SEARCH_BLOCK],
                    value,
                )
        } else {
            self.n_cuts
        };
        global_bin(self.start, local, self.n_cuts)
    }

    /// [`Self::bin_of`] local to the feature: `0` up to its bin count.
    #[inline(always)]
    pub(crate) fn local_bin_of(&self, value: f32) -> u32 {
        self.bin_of(value) - self.first_bin()
    }

    /// The feature's first global bin.
    #[inline(always)]
    pub(crate) fn first_bin(&self) -> u32 {
        self.start as u32
    }
}

/// Rows per block of the dense cut pass: a block of the panel's 173
/// features is 44 MiB transposed.
const CUT_BLOCK_ROWS: usize = 1 << 16;

/// Non-missing values per feature of the dense row-major `values` (rows of
/// `n_features`).
fn present_counts(values: &[f32], n_features: usize, missing: f32, parallel: bool) -> Vec<usize> {
    let count = |block: &[f32]| {
        let mut counts = vec![0usize; n_features];
        for row in block.chunks_exact(n_features) {
            for (count, &v) in counts.iter_mut().zip(row) {
                *count += usize::from(!crate::data::dmatrix::is_missing(v, missing));
            }
        }
        counts
    };
    let sum = |mut a: Vec<usize>, b: Vec<usize>| {
        for (a, b) in a.iter_mut().zip(b) {
            *a += b;
        }
        a
    };
    if n_features == 0 {
        return Vec::new();
    }
    let blocks = values.chunks(CUT_BLOCK_ROWS * n_features);
    if parallel {
        blocks
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(count)
            .reduce(|| vec![0; n_features], sum)
    } else {
        blocks.map(count).fold(vec![0; n_features], sum)
    }
}

/// One feature's cut construction state while the rows stream through
/// [`HistCuts::build`].
enum ColumnCuts {
    /// A categorical feature's distinct values so far, ascending by
    /// [`f32::total_cmp`] and deduplicated by bit pattern: what the sorted
    /// column [`build_categorical_cuts`] reads reduces to. `block` and
    /// `scratch` are the buffers each block of rows is sorted in.
    Categorical {
        distinct: Vec<f32>,
        block: Vec<f32>,
        scratch: RadixScratch<f32>,
    },
    /// A numeric feature's streaming sketch (upstream `PushRowPage`), fed
    /// whole blocks without per-value weights when every weight is `1`.
    Streamed { sketch: WQSketch, unit: bool },
    /// A numeric feature's `(value, weight)` pairs in row order, sorted and
    /// summarized once every row is in (upstream `PushColPage`).
    Sorted {
        sketch: WQSketch,
        pairs: Vec<(f32, f32)>,
    },
}

impl ColumnCuts {
    fn new(categorical: bool, present: usize, max_bin: usize, ingest: Ingest) -> Self {
        if categorical {
            return ColumnCuts::Categorical {
                distinct: Vec::new(),
                block: Vec::new(),
                scratch: RadixScratch::default(),
            };
        }
        let sketch = WQSketch::new(present, max_bin);
        if ingest.sorted {
            ColumnCuts::Sorted {
                sketch,
                pairs: Vec::with_capacity(present),
            }
        } else if ingest.unit_weights {
            ColumnCuts::Streamed {
                sketch: sketch.with_unit_weights(),
                unit: true,
            }
        } else {
            ColumnCuts::Streamed {
                sketch,
                unit: false,
            }
        }
    }

    /// Take the feature's values of the dense rows from `first` on
    /// (`missing` entries skipped), weighted by `weight_of(row)`.
    fn push_column(
        &mut self,
        values: &[f32],
        first: usize,
        missing: f32,
        weight_of: impl Fn(usize) -> f32,
    ) {
        if let ColumnCuts::Streamed { sketch, unit: true } = self {
            sketch.push_unit_column(values, missing);
            return;
        }
        let present = values
            .iter()
            .enumerate()
            .filter(|&(_, &v)| !crate::data::dmatrix::is_missing(v, missing))
            .map(|(i, &v)| (first + i, v));
        self.push(present, weight_of);
    }

    /// Take a sparse feature's stored entries (every one present), `rows`
    /// ascending, weighted by `weight_of(row)`.
    fn push_entries(&mut self, rows: &[u32], values: &[f32], weight_of: impl Fn(usize) -> f32) {
        let entries = rows.iter().zip(values).map(|(&row, &v)| (row as usize, v));
        self.push(entries, weight_of);
    }

    /// Take present `(row, value)` entries in row order.
    fn push(
        &mut self,
        entries: impl Iterator<Item = (usize, f32)>,
        weight_of: impl Fn(usize) -> f32,
    ) {
        match self {
            ColumnCuts::Categorical {
                distinct,
                block,
                scratch,
            } => {
                block.clear();
                block.extend(entries.map(|(_, v)| v));
                sort_values(block, scratch);
                block.dedup_by(|a, b| a.to_bits() == b.to_bits());
                merge_distinct(distinct, block);
            }
            ColumnCuts::Streamed { sketch, .. } => {
                for (row, v) in entries {
                    sketch.push(v, weight_of(row));
                }
            }
            ColumnCuts::Sorted { pairs, .. } => {
                pairs.extend(entries.map(|(row, v)| (v, weight_of(row))));
            }
        }
    }

    /// The feature's cut values.
    fn cut_values(self) -> Vec<f32> {
        let mut out = Vec::new();
        match self {
            ColumnCuts::Categorical { distinct, .. } => build_categorical_cuts(&distinct, &mut out),
            ColumnCuts::Streamed { mut sketch, .. } => sketch.cut_values(&mut out),
            ColumnCuts::Sorted {
                mut sketch,
                mut pairs,
            } => {
                pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
                sketch.push_sorted(&pairs);
                sketch.cut_values(&mut out);
            }
        }
        out
    }
}

/// Merge the ascending, bit-distinct `block` into the ascending,
/// bit-distinct `distinct` (both by [`f32::total_cmp`]).
fn merge_distinct(distinct: &mut Vec<f32>, block: &[f32]) {
    if block.is_empty() {
        return;
    }
    let mut merged = Vec::with_capacity(distinct.len() + block.len());
    let (mut a, mut b) = (distinct.iter().peekable(), block.iter().peekable());
    while let (Some(&&x), Some(&&y)) = (a.peek(), b.peek()) {
        match x.total_cmp(&y) {
            std::cmp::Ordering::Less => merged.push(*a.next().expect("peeked")),
            std::cmp::Ordering::Greater => merged.push(*b.next().expect("peeked")),
            std::cmp::Ordering::Equal => {
                merged.push(x);
                a.next();
                b.next();
            }
        }
    }
    merged.extend(a);
    merged.extend(b);
    *distinct = merged;
}

/// Append one bin per distinct category value (ascending) for a categorical
/// feature. Unlike numeric cuts, no sentinel is added: the bin *is* the
/// category.
fn build_categorical_cuts(sorted_vals: &[f32], out: &mut Vec<f32>) {
    if sorted_vals.is_empty() {
        // No observed categories: a single degenerate bin keeps the layout
        // well-formed; the feature can never split.
        out.push(0.0);
        return;
    }
    out.push(sorted_vals[0]);
    for w in sorted_vals.windows(2) {
        if w[0] != w[1] {
            out.push(w[1]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each feature's cuts by one whole-column pass: every present value of
    /// the column, in row order, into one sketch (sorted first for
    /// `sorted`), or the column's sorted values for a categorical feature.
    fn whole_column_cuts(
        data: &DMatrix,
        max_bin: usize,
        weight_of: impl Fn(usize) -> f32,
        ingest: Ingest,
    ) -> Vec<Vec<f32>> {
        (0..data.n_cols())
            .map(|f| {
                let entries: Vec<(usize, f32)> = (0..data.n_rows())
                    .filter_map(|r| data.get(r, f).map(|v| (r, v)))
                    .collect();
                let mut out = Vec::new();
                if data.feature_types()[f] == FeatureType::Categorical {
                    let mut values: Vec<f32> = entries.iter().map(|&(_, v)| v).collect();
                    values.sort_by(f32::total_cmp);
                    build_categorical_cuts(&values, &mut out);
                    return out;
                }
                let mut sketch = WQSketch::new(entries.len(), max_bin);
                if ingest.unit_weights {
                    sketch = sketch.with_unit_weights();
                }
                if ingest.sorted {
                    let mut pairs: Vec<(f32, f32)> =
                        entries.iter().map(|&(r, v)| (v, weight_of(r))).collect();
                    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
                    sketch.push_sorted(&pairs);
                } else {
                    for &(r, v) in &entries {
                        sketch.push(v, weight_of(r));
                    }
                }
                sketch.cut_values(&mut out);
                out
            })
            .collect()
    }

    /// Cuts streamed through row blocks (dense storage) or the CSC view
    /// (CSR storage) equal one whole-column pass per feature, unweighted,
    /// weighted, and sorted, across several blocks with missing values and
    /// a categorical feature, serially and in parallel.
    #[test]
    fn streamed_cuts_equal_whole_column_cuts() {
        let (n, cols) = (2 * CUT_BLOCK_ROWS + 1_234, 4);
        let x: Vec<f32> = (0..n * cols)
            .map(|i| {
                let (r, f) = (i / cols, i % cols);
                if (r * 31 + f * 7) % 9 == 0 {
                    f32::NAN
                } else if f == 1 {
                    ((r * 13) % 11) as f32
                } else {
                    ((r * 2_654_435_761 + f * 97) % 100_003) as f32 / 7.0
                }
            })
            .collect();
        let mut types = vec![FeatureType::Numerical; cols];
        types[1] = FeatureType::Categorical;
        let dense = DMatrix::from_dense(&x, n, cols)
            .unwrap()
            .with_feature_types(&types)
            .unwrap();
        let (mut indptr, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
        for r in 0..n {
            for f in 0..cols {
                if let Some(v) = dense.get(r, f) {
                    indices.push(f as u32);
                    values.push(v);
                }
            }
            indptr.push(indices.len());
        }
        let csr = DMatrix::from_csr(indptr, indices, values, cols)
            .unwrap()
            .with_feature_types(&types)
            .unwrap();
        let weights: Vec<f32> = (0..n).map(|r| 0.25 + (r % 5) as f32).collect();
        let hessians: Vec<f32> = (0..n).map(|r| 1.0 + (r % 3) as f32 * 0.5).collect();
        let per_feature = |cuts: &HistCuts| -> Vec<Vec<f32>> {
            (0..cuts.n_features())
                .map(|f| {
                    let (s, e) = cuts.feature_bins(f);
                    cuts.cut_values[s..e].to_vec()
                })
                .collect()
        };
        let unit = Ingest {
            sorted: false,
            unit_weights: true,
        };
        let weighted = Ingest {
            sorted: false,
            unit_weights: false,
        };
        let sorted = Ingest {
            sorted: true,
            unit_weights: false,
        };
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for data in [&dense, &csr] {
                let weighted_data = data.clone().with_weights(&weights).unwrap();
                let hess = |r: usize| hessians[r] * weights[r];
                let cases = [
                    (
                        pool.install(|| HistCuts::from_dmatrix(data, 64)),
                        whole_column_cuts(data, 64, |_| 1.0, unit),
                    ),
                    (
                        pool.install(|| HistCuts::from_dmatrix(&weighted_data, 64)),
                        whole_column_cuts(data, 64, |r| weights[r], weighted),
                    ),
                    (
                        pool.install(|| {
                            HistCuts::from_dmatrix_weighted(&weighted_data, 64, &hessians, true)
                        }),
                        whole_column_cuts(data, 64, hess, sorted),
                    ),
                ];
                for (case, (streamed, whole)) in cases.into_iter().enumerate() {
                    assert_eq!(
                        per_feature(&streamed),
                        whole,
                        "case {case}, {threads} threads"
                    );
                }
            }
        }
    }

    #[test]
    fn two_level_search_matches_bin_of() {
        // Features with 257 cuts (full), 3 cuts (short), and a constant column.
        let n = 5000;
        let mut x = vec![0f32; n * 3];
        for r in 0..n {
            x[r * 3] = ((r * 7919) % n) as f32 / n as f32;
            x[r * 3 + 1] = (r % 3) as f32;
            x[r * 3 + 2] = 2.5;
        }
        let data = DMatrix::from_dense(&x, n, 3).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let search = BinSearch::new(&cuts);
        for f in 0..3 {
            let (start, end) = cuts.feature_bins(f);
            let mut probes: Vec<f32> = cuts.cut_values[start..end].to_vec();
            probes.extend(
                cuts.cut_values[start..end]
                    .windows(2)
                    .map(|w| f32::midpoint(w[0], w[1])),
            );
            probes.extend([-1e9, 1e9, -0.0, 0.0, 0.5, 2.5, 3.0]);
            for value in probes {
                assert_eq!(
                    search.feature(f).bin_of(value),
                    cuts.bin_of(f, value),
                    "feature {f} value {value}"
                );
            }
        }
    }

    #[test]
    fn few_distinct_values_one_bin_each() {
        // Three distinct values 0,1,2 -> cuts = [1, 2, sentinel]: the minimum is
        // never a cut, so bin 0 is (-inf, 1] and each value has its own bin.
        let data = DMatrix::from_dense(&[0.0, 1.0, 2.0, 1.0], 4, 1).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        assert_eq!(cuts.n_features(), 1);
        assert_eq!(cuts.num_bins(0), 3);
        assert_eq!(
            (
                cuts.bin_of(0, 0.0),
                cuts.bin_of(0, 1.0),
                cuts.bin_of(0, 2.0)
            ),
            (0, 1, 2)
        );
    }

    #[test]
    fn monotone_binning() {
        // 1000 distinct-ish values, capped at 16 bins.
        let n = 1000;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 16);
        assert!(
            cuts.num_bins(0) <= 16,
            "at most max_bin - 1 cuts plus sentinel"
        );
        // Binning is monotone non-decreasing in the value.
        let mut prev = 0u32;
        for i in 0..n {
            let b = cuts.bin_of(0, i as f32);
            assert!(b >= prev);
            prev = b;
        }
        // Distinct low and high values fall in different bins.
        assert!(cuts.bin_of(0, 0.0) < cuts.bin_of(0, 999.0));
    }

    #[test]
    fn constant_feature_has_one_bin() {
        // The minimum is never a cut: only the sentinel remains, so every
        // value of a constant feature lands in its single bin.
        let data = DMatrix::from_dense(&[5.0, 5.0, 5.0], 3, 1).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        assert_eq!(cuts.num_bins(0), 1);
        assert_eq!(cuts.bin_of(0, 5.0), 0);
    }

    #[test]
    fn split_threshold_consistency() {
        // A value maps left of cut c (value < c) iff its bin <= bin_of(c-) ...
        // Concretely: for values 0..10 with cuts, `value < cut_value(bin)` must
        // agree with `bin_of(value) <= target_bin`.
        let x: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let data = DMatrix::from_dense(&x, 10, 1).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let (start, end) = cuts.feature_bins(0);
        for target in start..end - 1 {
            let thr = cuts.cut_value(target);
            for &v in &x {
                let goes_left_by_value = v < thr;
                let goes_left_by_bin = cuts.bin_of(0, v) as usize <= target;
                assert_eq!(
                    goes_left_by_value, goes_left_by_bin,
                    "value {v}, target bin {target}, thr {thr}"
                );
            }
        }
    }
}
