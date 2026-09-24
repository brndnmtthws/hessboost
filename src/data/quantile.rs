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

impl HistCuts {
    /// Compute cuts from a dataset with at most `max_bin` bins per feature.
    ///
    /// Categorical features (per [`DMatrix::feature_types`]) are binned with one
    /// bin per distinct category value. Numeric features use XGBoost's
    /// streaming quantile sketch over the values in row order, weighted by the
    /// sample weights when present (upstream `PushRowPage`).
    pub fn from_dmatrix(data: &DMatrix, max_bin: usize) -> Self {
        let weights = data.weights();
        Self::build(data, max_bin, |row| weights.map_or(1.0, |w| w[row]), false)
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
        let weights = data.weights();
        Self::build(
            data,
            max_bin,
            |row| weights.map_or(hessians[row], |w| hessians[row] * w[row]),
            sorted,
        )
    }

    fn build(
        data: &DMatrix,
        max_bin: usize,
        weight_of: impl Fn(usize) -> f32 + Sync,
        sorted: bool,
    ) -> Self {
        let n_rows = data.n_rows();
        let n_features = data.n_cols();
        // Dense storage is transposed in blocks so each feature's values are
        // contiguous; sparse storage goes through the CSC view.
        let (columns, csc) = match data.dense_values() {
            Some(values) => (
                Some(crate::data::ghist::transpose_dense(
                    values, n_rows, n_features,
                )),
                None,
            ),
            None => (None, Some(data.to_csc())),
        };
        let missing = data.missing();
        let ftypes = data.feature_types();
        let is_categorical: Vec<bool> = ftypes
            .iter()
            .map(|&t| t == FeatureType::Categorical)
            .collect();
        let mut feature_offset = Vec::with_capacity(n_features + 1);
        feature_offset.push(0u32);
        let mut cut_values = Vec::new();
        // Per-feature `(row, value)` pairs in row order, for either storage.
        let for_each_value = |f: usize, visit: &mut dyn FnMut(usize, f32)| match (&columns, &csc) {
            (Some(columns), _) => {
                for (row, &v) in columns[f * n_rows..(f + 1) * n_rows].iter().enumerate() {
                    if !crate::data::dmatrix::is_missing(v, missing) {
                        visit(row, v);
                    }
                }
            }
            (None, Some(csc)) => {
                let (rows, values) = csc.column(f);
                for (&row, &v) in rows.iter().zip(values) {
                    visit(row as usize, v);
                }
            }
            (None, None) => unreachable!("one column source is always built"),
        };
        let build =
            |f, scratch: &mut (Vec<f32>, Vec<f32>, Vec<(f32, f32)>), output: &mut Vec<f32>| {
                let (values, spare, pairs) = scratch;
                if is_categorical[f] {
                    values.clear();
                    for_each_value(f, &mut |_, v| values.push(v));
                    sort_values(values, spare);
                    build_categorical_cuts(values, output);
                    return;
                }
                let mut n_values = 0usize;
                for_each_value(f, &mut |_, _| n_values += 1);
                let mut sketch = WQSketch::new(n_values, max_bin);
                if sorted {
                    pairs.clear();
                    for_each_value(f, &mut |row, v| pairs.push((v, weight_of(row))));
                    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
                    sketch.push_sorted(pairs);
                } else {
                    for_each_value(f, &mut |row, v| sketch.push(v, weight_of(row)));
                }
                sketch.cut_values(output);
            };
        if n_features > 1
            && n_rows.saturating_mul(n_features) >= 65_536
            && rayon::current_num_threads() > 1
        {
            let columns: Vec<_> = (0..n_features)
                .into_par_iter()
                .map_init(Default::default, |scratch, f| {
                    let mut output = Vec::new();
                    build(f, scratch, &mut output);
                    output
                })
                .collect();
            for column in columns {
                cut_values.extend(column);
                feature_offset.push(cut_values.len() as u32);
            }
        } else {
            let mut scratch = Default::default();
            for f in 0..n_features {
                build(f, &mut scratch, &mut cut_values);
                feature_offset.push(cut_values.len() as u32);
            }
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
    /// Padded cuts, `padded_offset[f]..padded_offset[f + 1]` per feature.
    padded: Vec<f32>,
    padded_offset: Vec<usize>,
    /// Last cut of each block, padded with `+inf` to whole blocks,
    /// `level1_offset[f]..level1_offset[f + 1]` per feature.
    level1: Vec<f32>,
    level1_offset: Vec<usize>,
}

impl<'a> BinSearch<'a> {
    /// Build the index for every numeric feature of `cuts`.
    pub fn new(cuts: &'a HistCuts) -> Self {
        let mut padded = Vec::new();
        let mut padded_offset = vec![0];
        let mut level1 = Vec::new();
        let mut level1_offset = vec![0];
        for f in 0..cuts.n_features() {
            if !cuts.is_categorical(f) {
                let (start, end) = cuts.feature_bins(f);
                let feature = &cuts.cut_values[start..end];
                let blocks = feature.len().div_ceil(SEARCH_BLOCK);
                padded.extend_from_slice(feature);
                padded.resize(padded_offset[f] + blocks * SEARCH_BLOCK, f32::INFINITY);
                level1.extend(
                    feature
                        .chunks(SEARCH_BLOCK)
                        .map(|block| *block.last().expect("blocks are non-empty")),
                );
                level1.resize(
                    level1_offset[f] + blocks.div_ceil(SEARCH_BLOCK) * SEARCH_BLOCK,
                    f32::INFINITY,
                );
            }
            padded_offset.push(padded.len());
            level1_offset.push(level1.len());
        }
        BinSearch {
            cuts,
            padded,
            padded_offset,
            level1,
            level1_offset,
        }
    }

    /// Number of features of the underlying cut table.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.cuts.n_features()
    }

    /// Map a feature value to its global bin index, with the same result as
    /// [`HistCuts::bin_of`].
    #[inline]
    pub fn bin_of(&self, f: usize, value: f32) -> u32 {
        if self.cuts.is_categorical(f) {
            return self.cuts.bin_of(f, value);
        }
        let (start, end) = self.cuts.feature_bins(f);
        let level1 = &self.level1[self.level1_offset[f]..self.level1_offset[f + 1]];
        let mut block = 0;
        for chunk in level1.as_chunks::<SEARCH_BLOCK>().0 {
            block += crate::simd::count_le(chunk, value);
        }
        let padded = &self.padded[self.padded_offset[f]..self.padded_offset[f + 1]];
        let local = if block * SEARCH_BLOCK < padded.len() {
            block * SEARCH_BLOCK
                + crate::simd::count_le(
                    &padded[block * SEARCH_BLOCK..(block + 1) * SEARCH_BLOCK],
                    value,
                )
        } else {
            end - start
        };
        global_bin(start, local, end - start)
    }
}

/// Below this length the comparison sort beats the radix passes' fixed cost.
const RADIX_MIN_LEN: usize = 2048;
const RADIX_BITS: u32 = 11;
const RADIX_BUCKETS: usize = 1 << RADIX_BITS;

/// Monotone map from `f32` to `u32` under [`f32::total_cmp`] order.
#[inline]
fn sort_key(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// Sort `values` ascending by total order. Column sorts dominate cut
/// construction, so long inputs use a three-pass LSD radix sort on
/// [`sort_key`] (identical order to `sort_unstable_by(f32::total_cmp)`).
/// `spare` is scratch reused across columns.
fn sort_values(values: &mut Vec<f32>, spare: &mut Vec<f32>) {
    let n = values.len();
    if n < RADIX_MIN_LEN {
        values.sort_unstable_by(f32::total_cmp);
        return;
    }
    // Bucket counts for all passes in one sweep.
    let mut counts = vec![[0u32; RADIX_BUCKETS]; 3];
    for &v in values.iter() {
        let key = sort_key(v);
        for (pass, count) in counts.iter_mut().enumerate() {
            count[((key >> (RADIX_BITS * pass as u32)) & (RADIX_BUCKETS as u32 - 1)) as usize] += 1;
        }
    }
    spare.clear();
    spare.resize(n, 0.0);
    // Passes ping-pong between the two buffers; track which one holds the data.
    let mut in_spare = false;
    for (pass, count) in counts.iter_mut().enumerate() {
        // A pass whose digit is constant across the input is a no-op.
        if count.iter().any(|&c| c as usize == n) {
            continue;
        }
        let mut offset = 0u32;
        for c in count.iter_mut() {
            let start = offset;
            offset += *c;
            *c = start;
        }
        let shift = RADIX_BITS * pass as u32;
        let (src, dst) = if in_spare {
            (&*spare, &mut *values)
        } else {
            (&*values, &mut *spare)
        };
        for &v in src {
            let bucket = ((sort_key(v) >> shift) & (RADIX_BUCKETS as u32 - 1)) as usize;
            dst[count[bucket] as usize] = v;
            count[bucket] += 1;
        }
        in_spare = !in_spare;
    }
    if in_spare {
        std::mem::swap(values, spare);
    }
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
                    search.bin_of(f, value),
                    cuts.bin_of(f, value),
                    "feature {f} value {value}"
                );
            }
        }
    }

    #[test]
    fn radix_sort_matches_total_order() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for n in [RADIX_MIN_LEN, RADIX_MIN_LEN + 1, 10_007, 65_536] {
            let mut values: Vec<f32> = (0..n)
                .map(|i| match i % 11 {
                    0 => -0.0,
                    1 => 0.0,
                    2 => f32::MAX,
                    3 => f32::MIN,
                    4 => f32::MIN_POSITIVE,
                    5 => -f32::MIN_POSITIVE,
                    _ => (next() as f32 / u64::MAX as f32 - 0.5) * 1e6,
                })
                .collect();
            let mut expected = values.clone();
            expected.sort_unstable_by(f32::total_cmp);
            let mut spare = Vec::new();
            sort_values(&mut values, &mut spare);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&values), bits(&expected));
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
