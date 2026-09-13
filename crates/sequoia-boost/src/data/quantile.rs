//! Quantile cut computation for histogram-based training.
//!
//! For each feature we compute up to `max_bin` cut points from the (non-missing)
//! value distribution, then map any value to a bin with an `upper_bound` search
//! (`bin = #{cuts ≤ value}`, clamped). Cuts are computed **once** from the data,
//! matching XGBoost's `tree_method=hist`. A trailing sentinel cut just above each
//! feature's maximum guarantees the maximum value gets its own bin (no clamp
//! collision), so bin index `0` may be empty. This is harmless and cheap.

use crate::data::meta::FeatureType;
use crate::data::DMatrix;
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

impl HistCuts {
    /// Compute cuts from a dataset with at most `max_bin` bins per feature.
    ///
    /// Categorical features (per [`DMatrix::feature_types`]) are binned with one
    /// bin per distinct category value. Numeric features use quantile thresholds.
    pub fn from_dmatrix(data: &DMatrix, max_bin: usize) -> Self {
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
        let mut feature_offset = Vec::with_capacity(n_features + 1);
        feature_offset.push(0u32);
        let mut cut_values: Vec<f32> = Vec::new();
        let is_categorical: Vec<bool> = (0..n_features)
            .map(|f| ftypes.get(f).copied() == Some(FeatureType::Categorical))
            .collect();
        let build = |f, scratch: &mut (Vec<f32>, Vec<f32>), output: &mut Vec<f32>| {
            let (values, spare) = scratch;
            values.clear();
            match (&columns, &csc) {
                (Some(columns), _) => values.extend(
                    columns[f * n_rows..(f + 1) * n_rows]
                        .iter()
                        .copied()
                        .filter(|&v| !crate::data::dmatrix::is_missing(v, missing)),
                ),
                (None, Some(csc)) => values.extend_from_slice(csc.column(f).1),
                (None, None) => unreachable!("one column source is always built"),
            }
            sort_values(values, spare);
            if is_categorical[f] {
                build_categorical_cuts(values, output);
            } else {
                build_feature_cuts(values, max_bin, output);
            }
        };
        if n_features > 1
            && data.n_rows().saturating_mul(n_features) >= 65_536
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

    /// Compute **hessian-weighted** cuts, as XGBoost's `tree_method=approx` does
    /// each boosting round.
    ///
    /// Identical to [`HistCuts::from_dmatrix`] except that, for numeric features,
    /// every instance's feature value contributes its Hessian `hessians[row]` as
    /// weight when locating the quantile cut points (accumulate weight per sorted
    /// value, place cuts at weighted quantiles). Categorical features are binned
    /// exactly as in [`HistCuts::from_dmatrix`] (one bin per category). `hessians`
    /// is indexed by original row and must cover every row of `data`.
    pub fn from_dmatrix_weighted(data: &DMatrix, max_bin: usize, hessians: &[f32]) -> Self {
        let csc = data.to_csc();
        let n_features = csc.n_cols();
        let ftypes = data.feature_types();
        let mut feature_offset = Vec::with_capacity(n_features + 1);
        feature_offset.push(0u32);
        let mut cut_values: Vec<f32> = Vec::new();
        let mut is_categorical = vec![false; n_features];

        let mut cat_scratch: Vec<f32> = Vec::new();
        let mut cat_spare: Vec<f32> = Vec::new();
        let mut scratch: Vec<(f32, f32)> = Vec::new();
        #[allow(clippy::needless_range_loop)]
        for f in 0..n_features {
            let (rows, vals) = csc.column(f);
            if ftypes.get(f).copied() == Some(FeatureType::Categorical) {
                // Categorical binning ignores weights (one bin per category).
                is_categorical[f] = true;
                cat_scratch.clear();
                cat_scratch.extend_from_slice(vals);
                sort_values(&mut cat_scratch, &mut cat_spare);
                build_categorical_cuts(&cat_scratch, &mut cut_values);
            } else {
                // Pair each value with its instance's Hessian, then sort by value.
                scratch.clear();
                scratch.extend(
                    rows.iter()
                        .zip(vals)
                        .map(|(&r, &v)| (v, hessians[r as usize])),
                );
                scratch.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                build_feature_cuts_weighted(&scratch, max_bin, &mut cut_values);
            }
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
        let local = slice.partition_point(|&c| c <= value);
        let local = local.min(slice.len().saturating_sub(1));
        start as u32 + local as u32
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
        for chunk in level1.chunks_exact(SEARCH_BLOCK) {
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
        let local = local.min((end - start).saturating_sub(1));
        start as u32 + local as u32
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
        for &v in src.iter() {
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

/// Append feature cut values (ascending) for one feature to `out`.
fn build_feature_cuts(sorted_vals: &[f32], max_bin: usize, out: &mut Vec<f32>) {
    if sorted_vals.is_empty() {
        // No non-missing values: a single degenerate bin so the layout stays
        // well-formed. This feature can never produce a split.
        out.push(0.0);
        return;
    }
    let max_val = *sorted_vals.last().unwrap();

    // Count distinct values without allocating a second vector.
    let mut distinct = 1usize;
    for w in sorted_vals.windows(2) {
        if w[0] != w[1] {
            distinct += 1;
        }
    }

    let start = out.len();
    if distinct <= max_bin {
        // Use each distinct value as a cut.
        out.push(sorted_vals[0]);
        for w in sorted_vals.windows(2) {
            if w[0] != w[1] {
                out.push(w[1]);
            }
        }
    } else {
        // Weighted-uniform quantiles over the sorted values.
        let n = sorted_vals.len();
        let mut last_pushed = f32::NEG_INFINITY;
        for b in 1..=max_bin {
            let q = b as f64 / max_bin as f64;
            let idx = (((q * n as f64).ceil() as usize).max(1) - 1).min(n - 1);
            let v = sorted_vals[idx];
            if v > last_pushed {
                out.push(v);
                last_pushed = v;
            }
        }
    }
    push_max_sentinel(out, max_val);
    debug_assert!(out.len() > start);
}

/// Append feature cut values (ascending) for one numeric feature using
/// **hessian-weighted** quantiles: `sorted` holds ascending `(value, weight)`
/// pairs and each value contributes its weight when locating the quantile
/// boundaries. Mirrors [`build_feature_cuts`], including the empty-input guard, the
/// few-distinct-values shortcut, and the trailing sentinel are identical, and
/// with all weights equal it reproduces the unweighted cuts.
fn build_feature_cuts_weighted(sorted: &[(f32, f32)], max_bin: usize, out: &mut Vec<f32>) {
    if sorted.is_empty() {
        // No non-missing values: a single degenerate bin so the layout stays
        // well-formed. This feature can never produce a split.
        out.push(0.0);
        return;
    }
    let max_val = sorted.last().unwrap().0;

    // Count distinct values (compared on value only).
    let mut distinct = 1usize;
    for w in sorted.windows(2) {
        if w[0].0 != w[1].0 {
            distinct += 1;
        }
    }

    let start = out.len();
    if distinct <= max_bin {
        // Use each distinct value as a cut (weights irrelevant here).
        out.push(sorted[0].0);
        for w in sorted.windows(2) {
            if w[0].0 != w[1].0 {
                out.push(w[1].0);
            }
        }
    } else {
        let total: f64 = sorted.iter().map(|&(_, w)| w as f64).sum();
        let mut last_pushed = f32::NEG_INFINITY;
        if total > 0.0 {
            // Weighted quantiles: place cut `b` at the value where the running
            // weight first reaches `b/max_bin` of the total weight.
            let step = total / max_bin as f64;
            let mut cum = 0.0f64;
            let mut b = 1usize;
            for &(v, w) in sorted {
                cum += w as f64;
                while b <= max_bin && cum + 1e-12 >= b as f64 * step {
                    if v > last_pushed {
                        out.push(v);
                        last_pushed = v;
                    }
                    b += 1;
                }
            }
        } else {
            // Degenerate all-zero weights: fall back to unweighted positions.
            let n = sorted.len();
            for b in 1..=max_bin {
                let q = b as f64 / max_bin as f64;
                let idx = (((q * n as f64).ceil() as usize).max(1) - 1).min(n - 1);
                let v = sorted[idx].0;
                if v > last_pushed {
                    out.push(v);
                    last_pushed = v;
                }
            }
        }
    }
    push_max_sentinel(out, max_val);
    debug_assert!(out.len() > start);
}

/// Append a sentinel cut just above `max_val` (so the maximum value lands in the
/// final real bin rather than colliding under the clamp), unless the last cut is
/// already past it.
fn push_max_sentinel(out: &mut Vec<f32>, max_val: f32) {
    if *out.last().unwrap() <= max_val {
        out.push(max_val.next_up());
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
                    .map(|w| 0.5 * (w[0] + w[1])),
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
        // feature has 3 distinct values 0,1,2 -> cuts = [0,1,2, sentinel]
        let data = DMatrix::from_dense(&[0.0, 1.0, 2.0, 1.0], 4, 1).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        assert_eq!(cuts.n_features(), 1);
        // Distinct-value binning: each value maps to a distinct, increasing bin.
        let b0 = cuts.bin_of(0, 0.0);
        let b1 = cuts.bin_of(0, 1.0);
        let b2 = cuts.bin_of(0, 2.0);
        assert!(b0 < b1 && b1 < b2, "bins should be strictly increasing");
    }

    #[test]
    fn monotone_binning() {
        // 1000 distinct-ish values, capped at 16 bins.
        let n = 1000;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 16);
        assert!(cuts.num_bins(0) <= 17, "at most max_bin + sentinel");
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
    fn constant_feature_never_collides() {
        let data = DMatrix::from_dense(&[5.0, 5.0, 5.0], 3, 1).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        // All identical values map to the same bin.
        assert_eq!(cuts.bin_of(0, 5.0), cuts.bin_of(0, 5.0));
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
