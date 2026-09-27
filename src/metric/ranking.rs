//! Learning-to-rank metrics over query groups: NDCG (`ndcg`), mean
//! average precision (`map`), and precision at `k` (`pre`), each with
//! XGBoost's `@k` cutoff.

use super::factory::cutoff_name;
use super::{Metric, argsort_desc, group_ranges, weighted_mean};
use crate::K_RT_EPS_F32;
use rayon::prelude::*;

/// Weighted mean of a per-group `score` over the non-empty query-group
/// ranges, weighted by each group's first document weight (`1.0` when
/// unweighted); zero-weight groups are skipped, and without any rows or
/// weight the result is `0`, like the elementwise metrics. Shared by the
/// ranking metrics' `eval_grouped`.
fn grouped_average(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    group: Option<&crate::data::GroupInfo>,
    score: impl Fn(&[f32], &[f32]) -> f64 + Sync,
) -> f64 {
    let ranges = group_ranges(preds.len(), group);
    let weight = |start: usize| weights.map_or(1.0, |values| f64::from(values[start]));
    let totals = fold_groups(
        &ranges,
        |start, end| (weight(start) != 0.0).then(|| score(&preds[start..end], &labels[start..end])),
        (0.0, 0.0),
        |(sum, weight_sum), (start, _), score| match score {
            Some(score) => {
                let weight = weight(start);
                (sum + weight * score, weight_sum + weight)
            }
            None => (sum, weight_sum),
        },
    );
    weighted_mean(totals)
}

/// Query groups covering at least this many rows are scored in parallel.
const PARALLEL_GROUP_ROWS: usize = 4096;

/// `fold` over `f(start, end)` of every `(start, end)` range, in range
/// order. When the ranges cover many rows and the pool has several threads,
/// the `f` values are computed in parallel first; the fold always runs in
/// range order, so its result does not depend on the thread count.
pub(super) fn fold_groups<T: Send, A>(
    ranges: &[(usize, usize)],
    f: impl Fn(usize, usize) -> T + Sync,
    init: A,
    mut fold: impl FnMut(A, (usize, usize), T) -> A,
) -> A {
    let rows: usize = ranges.iter().map(|(start, end)| end - start).sum();
    if ranges.len() > 1 && rows >= PARALLEL_GROUP_ROWS && rayon::current_num_threads() > 1 {
        let values: Vec<T> = ranges
            .par_iter()
            .map(|&(start, end)| f(start, end))
            .collect();
        ranges
            .iter()
            .zip(values)
            .fold(init, |acc, (&range, value)| fold(acc, range, value))
    } else {
        ranges.iter().fold(init, |acc, &(start, end)| {
            fold(acc, (start, end), f(start, end))
        })
    }
}

/// Normalized Discounted Cumulative Gain (`ndcg`), averaged over query groups.
///
/// Gains are `2^rel - 1` with the standard `1 / log2(rank + 2)` discount.
/// Supports XGBoost's `@k` truncation (e.g. `ndcg@5`). Higher is better.
/// A group whose ideal DCG is zero contributes `0`. Named `ndcg@k` with a
/// cutoff, else `ndcg`, as XGBoost reports it.
#[derive(Debug, Clone)]
pub(crate) struct Ndcg {
    /// Optional rank cutoff `k`. `None` uses the full list.
    k: Option<usize>,
    name: String,
}

impl Default for Ndcg {
    /// `ndcg` over the full list.
    fn default() -> Self {
        Self::new(None)
    }
}

impl Ndcg {
    /// Create an NDCG metric with an optional `@k` truncation.
    pub(crate) fn new(k: Option<usize>) -> Self {
        Ndcg {
            k,
            name: cutoff_name("ndcg", k),
        }
    }

    /// NDCG of a single group given its predictions and labels.
    fn group_ndcg(&self, preds: &[f32], labels: &[f32]) -> f64 {
        let m = preds.len();
        let cut = self.k.map_or(m, |k| k.min(m));

        // DCG in prediction order.
        let order = argsort_desc(preds);
        let dcg: f64 = order[..cut]
            .iter()
            .enumerate()
            .map(|(p, &i)| ndcg_gain(f64::from(labels[i])) * ndcg_discount(p))
            .sum();

        let idcg = ideal_dcg(labels, cut);

        if idcg <= 0.0 { 0.0 } else { dcg / idcg }
    }
}

impl Metric for Ndcg {
    fn name(&self) -> &str {
        &self.name
    }

    fn maximize(&self) -> bool {
        true
    }

    fn supports_label_matrix(&self) -> bool {
        false
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        // No group info: treat everything as a single query.
        self.eval_grouped(preds, labels, weights, None)
    }

    fn eval_grouped(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        group: Option<&crate::data::GroupInfo>,
    ) -> f64 {
        nan_unless_consistent!(preds, labels, weights, 1);
        grouped_average(preds, labels, weights, group, |p, l| self.group_ndcg(p, l))
    }
}

/// NDCG gain of a relevance label: `2^rel - 1`.
#[inline]
fn ndcg_gain(rel: f64) -> f64 {
    (2.0f64).powf(rel) - 1.0
}

/// NDCG position discount for 0-based rank `p`: `1 / log2(p + 2)`.
#[inline]
fn ndcg_discount(p: usize) -> f64 {
    1.0 / ((p + 2) as f64).log2()
}

/// Ideal DCG of a group: labels sorted by descending relevance, gains
/// accumulated with the standard discount, truncated at `cut` ranks.
fn ideal_dcg(labels: &[f32], cut: usize) -> f64 {
    let mut ideal: Vec<f64> = labels.iter().map(|&l| f64::from(l)).collect();
    ideal.sort_by(|a, b| b.total_cmp(a));
    ideal[..cut]
        .iter()
        .enumerate()
        .map(|(p, &l)| ndcg_gain(l) * ndcg_discount(p))
        .sum()
}

/// Mean Average Precision (`map`), averaged over query groups.
///
/// Relevance is binarized as `label > 0`. Supports `@k` truncation (e.g.
/// `map@10`), which restricts the precision sum to the top-`k` ranks. Higher is
/// better. A group with no relevant documents contributes `0`. Named `map@k`
/// with a cutoff, else `map`, as XGBoost reports it.
#[derive(Debug, Clone)]
pub(crate) struct MeanAveragePrecision {
    /// Optional rank cutoff `k`. `None` uses the full list.
    k: Option<usize>,
    name: String,
}

impl Default for MeanAveragePrecision {
    /// `map` over the full list.
    fn default() -> Self {
        Self::new(None)
    }
}

impl MeanAveragePrecision {
    /// Create a MAP metric with an optional `@k` truncation.
    pub(crate) fn new(k: Option<usize>) -> Self {
        MeanAveragePrecision {
            k,
            name: cutoff_name("map", k),
        }
    }

    /// Average precision of a single group.
    fn group_ap(&self, preds: &[f32], labels: &[f32]) -> f64 {
        let m = preds.len();
        let cut = self.k.map_or(m, |k| k.min(m));

        let order = argsort_desc(preds);

        let num_rel = labels.iter().filter(|&&l| l > 0.0).count();
        if num_rel == 0 {
            return 0.0;
        }

        let mut hits = 0usize;
        let mut ap = 0.0f64;
        for (p, &i) in order[..cut].iter().enumerate() {
            if labels[i] > 0.0 {
                hits += 1;
                ap += hits as f64 / (p + 1) as f64;
            }
        }
        ap / num_rel as f64
    }
}

impl Metric for MeanAveragePrecision {
    fn name(&self) -> &str {
        &self.name
    }

    fn maximize(&self) -> bool {
        true
    }

    fn supports_label_matrix(&self) -> bool {
        false
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        self.eval_grouped(preds, labels, weights, None)
    }

    fn eval_grouped(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        group: Option<&crate::data::GroupInfo>,
    ) -> f64 {
        nan_unless_consistent!(preds, labels, weights, 1);
        grouped_average(preds, labels, weights, group, |p, l| self.group_ap(p, l))
    }
}

/// XGBoost's default ranking cutoff (`LambdaRankParam::DefaultK`), used by
/// `pre` without an `@k` suffix.
const DEFAULT_TOP_K: usize = 32;

/// Precision at `k` over query groups (`pre`, `pre@k`), as XGBoost's
/// `EvalPrecision`. Each group ranks its documents by descending prediction
/// (stable for ties) and scores `Σ_{rank < n} label / n` with
/// `n = min(k, group size)`; groups are averaged with their weight (the
/// first document's weight; `1` when unweighted) and the result is capped
/// at `1`. Without group information the whole dataset is one query. Empty
/// and zero-weight groups are skipped; without any rows or weight the
/// result is `0`, like the other ranking metrics.
/// Plain `pre` uses `k = 32`, XGBoost's default cutoff. Labels must be
/// binary (within `1e-6` of `0` or `1`); XGBoost aborts otherwise, and this
/// metric returns NaN. Higher is better.
#[derive(Debug, Clone)]
pub(crate) struct Precision {
    k: usize,
    name: String,
}

impl Precision {
    /// `pre@k`, or `pre` (cutoff 32) when `k` is `None`. The caller
    /// guarantees `k >= 1`.
    pub(super) fn new(k: Option<usize>) -> Self {
        Precision {
            k: k.unwrap_or(DEFAULT_TOP_K),
            name: cutoff_name("pre", k),
        }
    }
}

impl Metric for Precision {
    fn name(&self) -> &str {
        &self.name
    }

    fn maximize(&self) -> bool {
        true
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        self.eval_grouped(preds, labels, weights, None)
    }

    fn eval_grouped(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        group: Option<&crate::data::GroupInfo>,
    ) -> f64 {
        nan_unless_consistent!(preds, labels, weights, 1);
        // XGBoost `IsBinaryRel`.
        let binary = |y: f32| (y - 1.0).abs() < K_RT_EPS_F32 || y.abs() < K_RT_EPS_F32;
        if !labels.iter().all(|&y| binary(y)) {
            return f64::NAN;
        }
        let ranges = group_ranges(preds.len(), group);
        let weight = |start: usize| weights.map_or(1.0f32, |w| w[start]);
        let precision = |start: usize, end: usize| {
            let weight = weight(start);
            (weight != 0.0).then(|| {
                let order = argsort_desc(&preds[start..end]);
                let n = self.k.min(end - start);
                let hits: f64 = order[..n]
                    .iter()
                    .map(|&i| f64::from(labels[start + i] * weight))
                    .sum();
                hits / n as f64
            })
        };
        let totals = fold_groups(
            &ranges,
            precision,
            (0.0f64, 0.0f64),
            |(score, weight_sum), (start, _), precision| match precision {
                Some(precision) => (score + precision, weight_sum + f64::from(weight(start))),
                None => (score, weight_sum),
            },
        );
        weighted_mean(totals).min(1.0)
    }

    /// Precision ranks one label per row within each query group.
    fn supports_label_matrix(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::GroupInfo;

    /// Two groups of four: the top-2 of the first holds one relevant
    /// document, the second two; `k` is clipped to a group's size.
    #[test]
    fn precision_at_k_per_group_and_clipped() {
        let preds = [0.9, 0.8, 0.1, 0.7, 0.2, 0.6, 0.5, 0.1];
        let labels = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let group = GroupInfo::from_sizes(&[4, 4]);
        let at2 = Precision::new(Some(2));
        // Group 1 top-2 = rows 0, 1 -> 1/2; group 2 top-2 = rows 5, 6 -> 1.
        let v = at2.eval_grouped(&preds, &labels, None, Some(&group));
        assert!((v - 0.75).abs() < 1e-12, "{v}");
        // Default k = 32 covers each whole group: 2/4 in both.
        let v = Precision::new(None).eval_grouped(&preds, &labels, None, Some(&group));
        assert!((v - 0.5).abs() < 1e-12, "{v}");
    }

    /// Group weights come from each group's first document.
    #[test]
    fn precision_weights_groups() {
        let preds = [0.9, 0.1, 0.9, 0.1];
        let labels = [1.0, 0.0, 0.0, 1.0];
        let weights = [3.0, 3.0, 1.0, 1.0];
        let group = GroupInfo::from_sizes(&[2, 2]);
        let v = Precision::new(Some(1)).eval_grouped(&preds, &labels, Some(&weights), Some(&group));
        assert!((v - 0.75).abs() < 1e-12, "{v}");
    }

    #[test]
    fn precision_rejects_graded_labels() {
        let v = Precision::new(None).eval(&[0.5, 0.2], &[2.0, 0.0], None);
        assert!(v.is_nan());
    }

    /// Empty input (weighted or not) and empty groups score like the other
    /// ranking metrics instead of indexing a missing first weight or
    /// dividing `0 / 0`.
    #[test]
    fn precision_of_empty_input_and_groups() {
        use crate::metric::{DEFAULT_SOURCE, named};
        for name in ["pre@5", "ndcg", "map"] {
            let m = named(name, 1, &DEFAULT_SOURCE).unwrap();
            assert_eq!(m.eval(&[], &[], Some(&[])), 0.0, "{name}");
            assert_eq!(m.eval(&[], &[], None), 0.0, "{name}");
        }
        // A trailing empty group starts past the last row.
        let pre = Precision::new(Some(1));
        let group = GroupInfo::from_sizes(&[2, 0]);
        let v = pre.eval_grouped(&[0.9, 0.1], &[1.0, 0.0], Some(&[1.0, 1.0]), Some(&group));
        assert_eq!(v, 1.0);
    }

    /// `pre` reports XGBoost's names and maximizes; a zero `mphe` slope is a
    /// parameter error rather than a NaN score.
    #[test]
    fn factory_names_and_rejections() {
        use crate::metric::{DEFAULT_SOURCE, XgboostMetricSource, named};
        for name in ["pre", "pre@5"] {
            let m = named(name, 1, &DEFAULT_SOURCE).unwrap();
            assert_eq!(m.name(), name);
            assert!(m.maximize());
        }
        let flat = XgboostMetricSource {
            huber_slope: 0.0,
            ..DEFAULT_SOURCE
        };
        assert!(named("mphe", 1, &flat).is_err());
    }
}
