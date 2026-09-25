//! Evaluation metrics used for reporting and early stopping.
//!
//! Metrics receive predictions that have already passed through the objective's
//! [`crate::objective::Objective::eval_transform`] (so classification metrics
//! see probabilities), matching XGBoost's evaluation pipeline.

/// Short-circuit a [`Metric::eval`] to NaN when its inputs are not
/// [`consistent`] with `width` predictions per label. Defined before the
/// submodules so their metrics can use it too.
macro_rules! nan_unless_consistent {
    ($preds:expr, $labels:expr, $weights:expr, $width:expr) => {
        if !$crate::metric::consistent($preds, $labels, $weights, $width) {
            return f64::NAN;
        }
    };
}

mod distributional;
mod elementwise;
mod quantile;
mod ranking;
mod survival;

pub use distributional::{DistCrps, DistNll};
pub use elementwise::{Mape, PseudoHuberError, Rmsle};
pub use quantile::{ExpectileError, QuantileError};
pub use ranking::Precision;
pub use survival::{AftNLogLik, CoxNLogLik, IntervalRegressionAccuracy};

use crate::config::{ObjectiveParams, TrainingParams};
use crate::data::MetaInfo;
use crate::error::{HessboostError, Result};
use rayon::prelude::*;

/// An evaluation metric over predictions and labels.
pub trait Metric: Send + Sync {
    /// The XGBoost-compatible metric name (e.g. `"rmse"`, `"logloss"`).
    fn name(&self) -> &str;

    /// Whether a *larger* value is better (e.g. AUC). Drives early stopping.
    fn maximize(&self) -> bool {
        false
    }

    /// Evaluate the metric. `preds` are post-transform predictions,
    /// [`Metric::prediction_width`] per label; inconsistent lengths
    /// (`preds`, or `weights` other than one per label) evaluate to NaN.
    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64;

    /// Evaluate the metric with optional query-group structure.
    ///
    /// Ranking metrics (`ndcg`, `map`) override this to compute the metric per
    /// query group and average across groups. The default ignores the grouping
    /// and forwards to [`Metric::eval`]. This is correct for all pointwise metrics.
    fn eval_grouped(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        _group: Option<&crate::data::GroupInfo>,
    ) -> f64 {
        self.eval(preds, labels, weights)
    }

    /// Evaluate the metric from a dataset's full metadata view; the entry
    /// point training uses. The default forwards the labels, weights, and
    /// groups to [`Metric::eval_grouped`]; metrics that read other metadata
    /// (label bounds) override it.
    ///
    /// For a label matrix (`info.n_targets > 1`) the default is XGBoost's
    /// elementwise reduction: `preds` and `labels` are both
    /// `[row][target]`, every cell counts as one instance, and each row's
    /// weight is repeated for its cells, so the metric averages over all
    /// rows and targets. Metrics that are not elementwise override it (or
    /// report [`Metric::supports_label_matrix`] `false`). Metadata whose
    /// lengths disagree with `n_rows` and `n_targets` evaluates to NaN.
    fn eval_info(&self, preds: &[f32], info: &MetaInfo) -> f64 {
        if info.check_layout().is_err() {
            return f64::NAN;
        }
        if info.n_targets > 1 {
            let Ok(cell_weights) = info.cell_weights() else {
                return f64::NAN;
            };
            return self.eval_grouped(preds, info.labels, cell_weights.as_deref(), None);
        }
        self.eval_grouped(preds, info.labels, info.weights, info.group)
    }

    /// Whether [`Metric::eval_info`] is defined on a label matrix
    /// (`n_targets > 1`). `true` by default (the elementwise reduction);
    /// ranking, multiclass, and per-row survival metrics return `false`, and
    /// training then rejects them for multi-target data.
    fn supports_label_matrix(&self) -> bool {
        true
    }

    /// Check that a dataset carries the metadata [`Metric::eval_info`]
    /// reads, before training evaluates it. The default requires ordinary
    /// labels; metrics that can read other metadata (the survival metrics'
    /// label bounds) override it. Errors are
    /// [`HessboostError::InvalidParameter`] naming `eval_metric`, with a
    /// reason mentioning "dataset" (training names the dataset there).
    fn validate_info(&self, info: &MetaInfo) -> Result<()> {
        if info.n_rows > 0 && info.labels.is_empty() {
            return Err(HessboostError::invalid_param(
                "eval_metric",
                format!(
                    "metric `{}` needs labels, but dataset has none",
                    self.name()
                ),
            ));
        }
        Ok(())
    }

    /// Predictions per row (`[row][output]`) that [`Metric::eval_info`]
    /// reads on `info`: one per label column by default (elementwise,
    /// ranking, and survival metrics), the class count for `mlogloss` /
    /// `merror`, one per alpha and label column for `quantile` /
    /// `expectile`, and the distribution's parameter count for `nll` /
    /// `crps`. `None` accepts any whole number of predictions per label (the
    /// [`CustomMetric`] hook). Training refuses an evaluation set on which a
    /// metric's width differs from the model's output count.
    fn prediction_width(&self, info: &MetaInfo) -> Option<usize> {
        Some(info.n_targets)
    }
}

/// Whether `preds` holds `width` values per label and `weights`, when
/// given, one per label: the lengths [`Metric::eval`] reads. Metrics
/// evaluate inconsistent inputs to NaN.
fn consistent(preds: &[f32], labels: &[f32], weights: Option<&[f32]>, width: usize) -> bool {
    labels.len().checked_mul(width) == Some(preds.len())
        && weights.is_none_or(|w| w.len() == labels.len())
}

/// Normalize a metric total, returning zero for an empty or nonpositive weight sum.
#[inline]
fn weighted_mean((total, weight): (f64, f64)) -> f64 {
    if weight > 0.0 { total / weight } else { 0.0 }
}

/// Define a purely-pointwise metric from its SIMD weighted-sum kernel.
/// Generates the metric struct plus its [`Metric`] impl from the metric name
/// and the `crate::simd` kernel path; `eval` is
/// `weighted_mean(kernel(preds, labels, weights))`, NaN for inconsistent
/// lengths. Metrics with metric-level state take a `field: Type` arm and
/// pass `self.field` as the kernel's final argument; `rmse` takes `=> sqrt`
/// for its root. A trailing `per_label` on the `field` arm marks a metric
/// reading `self.field` predictions per label (the multiclass metrics: one
/// probability per class, one class id per row, so no label matrices). All
/// generated metrics minimize (`maximize` keeps its default `false`);
/// metrics with non-trivial logic (`auc`, `aucpr`, ranking) stay
/// handwritten below.
macro_rules! simple_metric {
    ($(#[$m:meta])* $ty:ident, $name:literal, $simd:path $(=> $root:ident)?) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, Default)]
        #[non_exhaustive]
        pub struct $ty;
        impl Metric for $ty {
            fn name(&self) -> &str {
                $name
            }
            fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
                nan_unless_consistent!(preds, labels, weights, 1);
                weighted_mean($simd(preds, labels, weights))$(.$root())?
            }
        }
    };
    ($(#[$m:meta])* $ty:ident, $name:literal, $field:ident: $field_ty:ty, $simd:path) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy)]
        pub struct $ty {
            $field: $field_ty,
        }
        impl Metric for $ty {
            fn name(&self) -> &str {
                $name
            }
            fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
                nan_unless_consistent!(preds, labels, weights, 1);
                weighted_mean($simd(preds, labels, weights, self.$field))
            }
        }
    };
    ($(#[$m:meta])* $ty:ident, $name:literal, $field:ident: $field_ty:ty, $simd:path,
        per_label) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy)]
        pub struct $ty {
            $field: $field_ty,
        }
        impl Metric for $ty {
            fn name(&self) -> &str {
                $name
            }
            fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
                nan_unless_consistent!(preds, labels, weights, self.$field);
                weighted_mean($simd(preds, labels, weights, self.$field))
            }
            fn supports_label_matrix(&self) -> bool {
                false
            }
            fn prediction_width(&self, _info: &MetaInfo) -> Option<usize> {
                Some(self.$field)
            }
        }
    };
}

simple_metric!(
    /// Root-mean-square error (`rmse`).
    Rmse, "rmse", crate::simd::squared_error_sum => sqrt
);

simple_metric!(
    /// Mean absolute error (`mae`).
    Mae, "mae", crate::simd::absolute_error_sum
);

simple_metric!(
    /// Binary logistic loss (`logloss`), XGBoost's
    /// `-y·ln(max(p, ε)) − (1 − y)·ln(max(1 − p, ε))` with `ε = 1e-16` and a
    /// zero-coefficient term dropped. Predictions are probabilities, or raw
    /// margins for `binary:logitraw`, which are not clamped into `[0, 1]`.
    LogLoss, "logloss", crate::simd::log_loss_sum
);

simple_metric!(
    /// Binary classification error rate at threshold 0.5 (`error`).
    ErrorRate, "error", crate::simd::classification_error_sum
);

/// Ranges of consecutive positions in `order` whose `preds` values are equal
/// (tie runs). Shared by the tie handling of the AUC metrics.
fn tie_runs<'a>(
    order: &'a [usize],
    preds: &'a [f32],
) -> impl Iterator<Item = std::ops::Range<usize>> + 'a {
    let mut start = 0;
    std::iter::from_fn(move || {
        if start >= order.len() {
            return None;
        }
        let mut end = start + 1;
        while end < order.len() && preds[order[end]] == preds[order[start]] {
            end += 1;
        }
        let run = start..end;
        start = end;
        Some(run)
    })
}

/// [`Metric::eval_info`] of the curve metrics: for a label matrix, XGBoost's
/// multi-label macro average (`MultiAUC` with `MultiAUCType::kMultiLabel`):
/// evaluate `metric` on each target column of the `[row][target]` labels
/// with the row weights, then take the plain mean over targets. A single
/// label column evaluates through [`Metric::eval_grouped`].
fn macro_average_targets(metric: &dyn Metric, preds: &[f32], info: &MetaInfo) -> f64 {
    let k = info.n_targets;
    if k <= 1 {
        return metric.eval_grouped(preds, info.labels, info.weights, info.group);
    }
    let mut col_preds = Vec::with_capacity(info.n_rows);
    let mut col_labels = Vec::with_capacity(info.n_rows);
    let mut total = 0.0;
    for target in 0..k {
        col_preds.clear();
        col_preds.extend(preds.iter().skip(target).step_by(k));
        col_labels.clear();
        col_labels.extend(info.labels.iter().skip(target).step_by(k));
        total += metric.eval(&col_preds, &col_labels, info.weights);
    }
    total / k as f64
}

/// Binary ROC AUC (`auc`), computed with the Mann-Whitney rank-sum and average
/// ranks for ties. Higher is better. Weights are ignored (unweighted AUC).
/// For a label matrix it is the mean of the per-target AUCs (XGBoost's
/// multi-label macro average).
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct Auc;

impl Metric for Auc {
    fn name(&self) -> &'static str {
        "auc"
    }

    fn maximize(&self) -> bool {
        true
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        nan_unless_consistent!(preds, labels, weights, 1);
        let n = preds.len();
        let order = stable_argsort(n, |&a, &b| preds[a].total_cmp(&preds[b]));

        // Assign average ranks (1-based), resolving ties.
        let mut ranks = vec![0.0f64; n];
        for run in tie_runs(&order, preds) {
            let avg = ((run.start + 1 + run.end) as f64) / 2.0; // average of ranks start+1..=end
            for &idx in &order[run] {
                ranks[idx] = avg;
            }
        }

        let mut sum_pos_rank = 0.0f64;
        let mut n_pos = 0.0f64;
        let mut n_neg = 0.0f64;
        for k in 0..n {
            if labels[k] > 0.5 {
                sum_pos_rank += ranks[k];
                n_pos += 1.0;
            } else {
                n_neg += 1.0;
            }
        }
        if n_pos == 0.0 || n_neg == 0.0 {
            return 0.5; // undefined; XGBoost-like neutral value
        }
        (sum_pos_rank - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    }

    fn eval_info(&self, preds: &[f32], info: &MetaInfo) -> f64 {
        macro_average_targets(self, preds, info)
    }
}

simple_metric!(
    /// Multiclass log loss (`mlogloss`). Predictions are `n × num_class`
    /// probabilities. Labels are class indices.
    MLogLoss, "mlogloss", num_class: usize, crate::simd::multiclass_log_loss_sum,
    per_label
);

simple_metric!(
    /// Multiclass error rate (`merror`): fraction whose argmax ≠ label.
    MError, "merror", num_class: usize, crate::simd::multiclass_error_sum,
    per_label
);

simple_metric!(
    /// Poisson negative log-likelihood (`poisson-nloglik`). Predictions are rates.
    PoissonNLogLik, "poisson-nloglik", crate::simd::positive_nloglik_sum::<false>
);

simple_metric!(
    /// Gamma negative log-likelihood (`gamma-nloglik`). Predictions are means.
    GammaNLogLik, "gamma-nloglik", crate::simd::positive_nloglik_sum::<true>
);

simple_metric!(
    /// Tweedie negative log-likelihood (`tweedie-nloglik`) with variance power `rho`.
    TweedieNLogLik, "tweedie-nloglik", rho: f64, crate::simd::tweedie_nloglik_sum
);

/// The non-empty `(start, end)` row ranges of `group` when it partitions the
/// `n` rows (`GroupInfo::partitions`), otherwise the whole batch as one
/// range (none when `n == 0`). Every range indexes an `n`-row buffer and
/// holds at least one row. Shared by the ranking metrics and the LambdaMART
/// objective's usable-group fallback.
pub(crate) fn group_ranges(
    n: usize,
    group: Option<&crate::data::GroupInfo>,
) -> Vec<(usize, usize)> {
    match group {
        Some(g) if g.partitions(n) => g.iter_ranges().filter(|(s, e)| s < e).collect(),
        _ if n == 0 => Vec::new(),
        _ => vec![(0, n)],
    }
}

/// Indices of `values` sorted by descending value, stable under numeric
/// equality. Mirrors XGBoost `common::ArgSort(..., std::greater<>{})`
/// (`std::stable_sort`): `-0.0` and `+0.0` compare equal and keep input order,
/// which decides which documents fall inside a top-k truncation. NaN (absent in
/// valid predictions) falls back to `total_cmp` so the comparator stays a
/// consistent total preorder.
/// Shared by the ranking metrics and the LambdaMART objective.
pub(crate) fn argsort_desc(values: &[f32]) -> Vec<usize> {
    stable_argsort(values.len(), |&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or_else(|| values[b].total_cmp(&values[a]))
    })
}

/// Inputs at least this long are sorted in parallel.
const PARALLEL_SORT_LEN: usize = 1 << 15;

/// The indices `0..n` stably sorted by `cmp`. Long inputs use rayon's
/// parallel merge sort, which is stable too, so the order is the same.
/// Shared by the curve and ranking metrics and the objectives that sort rows.
pub(crate) fn stable_argsort(
    n: usize,
    cmp: impl Fn(&usize, &usize) -> std::cmp::Ordering + Sync,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    if n >= PARALLEL_SORT_LEN && rayon::current_num_threads() > 1 {
        order.par_sort_by(cmp);
    } else {
        order.sort_by(cmp);
    }
    order
}

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
/// A group whose ideal DCG is zero contributes `0`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ndcg {
    /// Optional rank cutoff `k`. `None` uses the full list.
    k: Option<usize>,
}

impl Ndcg {
    /// Create an NDCG metric with an optional `@k` truncation.
    pub fn new(k: Option<usize>) -> Self {
        Ndcg { k }
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
    fn name(&self) -> &'static str {
        "ndcg"
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
/// better. A group with no relevant documents contributes `0`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeanAveragePrecision {
    /// Optional rank cutoff `k`. `None` uses the full list.
    k: Option<usize>,
}

impl MeanAveragePrecision {
    /// Create a MAP metric with an optional `@k` truncation.
    pub fn new(k: Option<usize>) -> Self {
        MeanAveragePrecision { k }
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
    fn name(&self) -> &'static str {
        "map"
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

/// Area under the precision-recall curve for binary classification (`aucpr`).
///
/// Labels are `{0, 1}` and predictions are probabilities. The curve is traced by
/// sorting instances by descending prediction and sweeping the decision
/// threshold. The area is integrated over recall with the trapezoidal rule
/// (tied scores form a single operating point). Higher is better. A degenerate
/// problem (no positives or no negatives) yields `0`. For a label matrix it
/// is the mean of the per-target areas (XGBoost's multi-label macro average).
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct AucPr;

impl Metric for AucPr {
    fn name(&self) -> &'static str {
        "aucpr"
    }

    fn maximize(&self) -> bool {
        true
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        nan_unless_consistent!(preds, labels, weights, 1);
        let w_of = |i: usize| weights.map_or(1.0, |ws| f64::from(ws[i]));

        // Sort instance indices by descending predicted score.
        let order = argsort_desc(preds);

        let mut total_pos = 0.0f64;
        let mut total_neg = 0.0f64;
        for (i, &label) in labels.iter().enumerate() {
            if label > 0.5 {
                total_pos += w_of(i);
            } else {
                total_neg += w_of(i);
            }
        }
        if total_pos <= 0.0 || total_neg <= 0.0 {
            return 0.0;
        }

        // Sweep thresholds, accumulating (weighted) true/false positives and
        // integrating precision over recall with the trapezoidal rule. Runs of
        // tied scores collapse into a single operating point.
        let mut area = 0.0f64;
        let (mut tp, mut fp) = (0.0f64, 0.0f64);
        let (mut tp_prev, mut fp_prev) = (0.0f64, 0.0f64);
        for run in tie_runs(&order, preds) {
            for &idx in &order[run] {
                if labels[idx] > 0.5 {
                    tp += w_of(idx);
                } else {
                    fp += w_of(idx);
                }
            }
            if tp + fp > 0.0 {
                let recall = tp / total_pos;
                let recall_prev = tp_prev / total_pos;
                let prec = tp / (tp + fp);
                // At the first operating point precision is undefined; reuse the
                // current precision so the leading segment integrates cleanly.
                let prec_prev = if tp_prev + fp_prev > 0.0 {
                    tp_prev / (tp_prev + fp_prev)
                } else {
                    prec
                };
                area += (recall - recall_prev) * (prec + prec_prev) / 2.0;
            }
            tp_prev = tp;
            fp_prev = fp;
        }
        area
    }

    fn eval_info(&self, preds: &[f32], info: &MetaInfo) -> f64 {
        macro_average_targets(self, preds, info)
    }
}

/// Closure type backing a [`CustomMetric`].
type MetricFn = dyn Fn(&[f32], &[f32], Option<&[f32]>) -> f64 + Send + Sync;

/// A [`Metric`] backed by a user-supplied closure (the custom-metric hook).
///
/// The closure receives post-transform predictions (`[row][output]`, the
/// model's `n_outputs` per row), labels (`[row][target]`), and optional
/// weights (one per label), and returns the scalar metric value.
/// `maximize` declares the optimization direction used for early stopping.
/// Inputs whose prediction count is not a positive multiple of the label
/// count, or whose weights are not one per label, evaluate to NaN without
/// calling the closure; training refuses a model whose outputs are not a
/// whole number per label column.
///
/// For a label matrix the closure sees `[row][target]` labels with each
/// row's weight repeated for its cells (the default [`Metric::eval_info`]
/// reduction).
pub struct CustomMetric {
    name: String,
    maximize: bool,
    f: Box<MetricFn>,
}

impl CustomMetric {
    /// Build a custom metric from `name`, its `maximize` direction, and a
    /// `(preds, labels, weights) -> value` closure.
    pub fn new(
        name: impl Into<String>,
        maximize: bool,
        f: impl Fn(&[f32], &[f32], Option<&[f32]>) -> f64 + Send + Sync + 'static,
    ) -> Self {
        CustomMetric {
            name: name.into(),
            maximize,
            f: Box::new(f),
        }
    }
}

impl Metric for CustomMetric {
    fn name(&self) -> &str {
        &self.name
    }

    fn maximize(&self) -> bool {
        self.maximize
    }

    /// NaN without calling the closure unless `preds` holds a positive
    /// whole number of values per label and `weights`, when given, one per
    /// label.
    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        let width = preds.len().checked_div(labels.len()).unwrap_or(0);
        if width == 0 || !consistent(preds, labels, weights, width) {
            return f64::NAN;
        }
        (self.f)(preds, labels, weights)
    }

    /// Any whole number of predictions per label: the closure interprets
    /// them.
    fn prediction_width(&self, _info: &MetaInfo) -> Option<usize> {
        None
    }
}

/// The metric XGBoost calls `name` (e.g. `"auc"`, `"ndcg@5"`,
/// `"tweedie-nloglik@1.5"`), configured from `params`: multiclass metrics
/// read `num_class`, and objective-dependent metrics the loss parameters:
/// `mphe` takes its slope from `huber_slope`, `aft-nloglik` the AFT
/// distribution and scale, and `quantile` / `expectile` the configured
/// `quantile_alpha` / `expectile_alpha` (whatever the objective, like
/// XGBoost), failing when that list is empty or invalid. The distributional
/// metrics `nll` and `crps` (beyond XGBoost) take the family of a `dist:*`
/// objective and fail without one.
///
/// Only `ndcg`, `map`, and `pre` (a positive integer rank cutoff `@k`, as
/// the lower bound 1 XGBoost puts on the top-k it sets from the suffix) and
/// `tweedie-nloglik` (a variance power `@rho` in `[1, 2)`) take an `@`
/// suffix; any other suffix, including XGBoost's `error@t` threshold and the
/// `-` variants (`ndcg@3-`), is a parameter error.
pub fn create_metric(name: &str, params: &TrainingParams) -> Result<Box<dyn Metric>> {
    build(
        name,
        params.num_class,
        &ObjectiveParams::from_params(params),
    )
}

/// [`create_metric`] from a model's retained objective parameters.
pub(crate) fn build(
    name: &str,
    num_class: usize,
    objective: &ObjectiveParams,
) -> Result<Box<dyn Metric>> {
    let (base, suffix) = match name.split_once('@') {
        Some((b, s)) => (b, Some(s)),
        None => (name, None),
    };
    let invalid = |reason: &str| invalid_metric(name, reason);
    let cutoff = || rank_cutoff(name, suffix);
    let metric: Result<Box<dyn Metric>> = match base {
        "rmse" => Ok(Box::new(Rmse)),
        "mae" => Ok(Box::new(Mae)),
        "logloss" => Ok(Box::new(LogLoss)),
        "error" => Ok(Box::new(ErrorRate)),
        "auc" => Ok(Box::new(Auc)),
        "aucpr" => Ok(Box::new(AucPr)),
        "mlogloss" => Ok(Box::new(MLogLoss {
            num_class: num_class.max(2),
        })),
        "merror" => Ok(Box::new(MError {
            num_class: num_class.max(2),
        })),
        "poisson-nloglik" => Ok(Box::new(PoissonNLogLik)),
        "gamma-nloglik" => Ok(Box::new(GammaNLogLik)),
        "tweedie-nloglik" => Ok(Box::new(TweedieNLogLik {
            rho: tweedie_power(name, suffix)?,
        })),
        "ndcg" => Ok(Box::new(Ndcg::new(cutoff()?))),
        "map" => Ok(Box::new(MeanAveragePrecision::new(cutoff()?))),
        "rmsle" => Ok(Box::new(Rmsle)),
        "mape" => Ok(Box::new(Mape)),
        "mphe" => {
            let slope = objective.huber_slope as f32;
            if slope == 0.0 {
                return Err(HessboostError::invalid_param(
                    "huber_slope",
                    "the slope of `mphe` cannot be 0",
                ));
            }
            Ok(Box::new(PseudoHuberError::new(slope)))
        }
        "pre" => Ok(Box::new(Precision::new(name, cutoff()?))),
        "quantile" => Ok(Box::new(QuantileError::new(&objective.quantile_alpha)?)),
        "expectile" => Ok(Box::new(ExpectileError::new(&objective.expectile_alpha)?)),
        "cox-nloglik" => Ok(Box::new(CoxNLogLik)),
        "aft-nloglik" => Ok(Box::new(AftNLogLik::new(
            objective.aft_loss_distribution,
            objective.aft_loss_distribution_scale as f32,
        ))),
        "interval-regression-accuracy" => Ok(Box::new(IntervalRegressionAccuracy)),
        "nll" | "crps" => {
            let family = objective.distribution.ok_or_else(|| {
                HessboostError::invalid_param(
                    "eval_metric",
                    format!(
                        "`{name}` scores predicted distributions and needs a `dist:*` objective"
                    ),
                )
            })?;
            Ok(if base == "nll" {
                Box::new(DistNll::new(family))
            } else {
                Box::new(DistCrps::new(family))
            })
        }
        other => Err(HessboostError::unknown("metric", other)),
    };
    let metric = metric?;
    if suffix.is_some() && !matches!(base, "tweedie-nloglik" | "ndcg" | "map" | "pre") {
        return Err(invalid(&format!("`{base}` takes no `@` suffix")));
    }
    Ok(metric)
}

/// The parameter error of metric `name`.
fn invalid_metric(name: &str, reason: &str) -> HessboostError {
    HessboostError::invalid_param("eval_metric", format!("`{name}`: {reason}"))
}

/// The rank cutoff `@k` of ranking metric `name`: decimal digits only
/// (`usize::from_str` also takes a leading `+`), so `2.9`, `abc`, `1@2`, or
/// an empty suffix are refused rather than truncated or dropped.
fn rank_cutoff(name: &str, suffix: Option<&str>) -> Result<Option<usize>> {
    match suffix {
        None => Ok(None),
        Some(s) if s.ends_with('-') => Err(invalid_metric(
            name,
            "the `-` variants of the ranking metrics are not implemented",
        )),
        Some(s) => match s.parse::<usize>() {
            Ok(k) if k >= 1 && s.bytes().all(|b| b.is_ascii_digit()) => Ok(Some(k)),
            _ => Err(invalid_metric(
                name,
                "the `@k` cutoff must be a positive integer",
            )),
        },
    }
}

/// The variance power `@rho` of `tweedie-nloglik` (`1.5` without a
/// suffix), in the range `TrainingParams::validate` gives the objective's
/// `tweedie_variance_power`, whose default metric this is.
fn tweedie_power(name: &str, suffix: Option<&str>) -> Result<f64> {
    match suffix {
        None => Ok(1.5),
        Some(s) => s
            .parse::<f64>()
            .ok()
            .filter(|r| r.is_finite() && (1.0f32..2.0).contains(&(*r as f32)))
            .ok_or_else(|| invalid_metric(name, "the variance power `@rho` must be in [1, 2)")),
    }
}

/// Build the list of metrics to evaluate: the user's `eval_metric` list if any,
/// otherwise the single `default_name` supplied by the objective. `num_class`
/// and `objective` are forwarded to [`build`].
///
/// XGBoost configures the default metric from the objective's
/// `DefaultMetricConfig` but without the user's parameters (the learner has
/// cleared them by the time it evaluates). For `aft-nloglik` that keeps the
/// objective's distribution while the scale falls back to its default 1, so
/// the default metric here is built the same way; list `aft-nloglik` in
/// `eval_metric` to evaluate the likelihood at the configured scale.
pub(crate) fn create_metrics(
    eval_metric: &[String],
    default_name: &str,
    num_class: usize,
    objective: &ObjectiveParams,
) -> Result<Vec<Box<dyn Metric>>> {
    if eval_metric.is_empty() {
        let default_config = ObjectiveParams {
            aft_loss_distribution_scale: ObjectiveParams::default().aft_loss_distribution_scale,
            ..objective.clone()
        };
        Ok(vec![build(default_name, num_class, &default_config)?])
    } else {
        eval_metric
            .iter()
            .map(|n| build(n, num_class, objective))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// The parallel sorts, per-group scores, and per-row interval values give
    /// the serial metric values bit for bit: large inputs with many ties, many
    /// query groups (one empty), weights, and label matrices.
    #[test]
    fn parallel_evaluation_matches_serial() {
        use crate::config::AftDistribution;
        use crate::data::GroupInfo;
        let n = 3 * PARALLEL_SORT_LEN + 17;
        let preds: Vec<f32> = (0..3 * n)
            .map(|i| ((i * 7919) % 1013) as f32 / 1013.0)
            .collect();
        let labels: Vec<f32> = (0..3 * n).map(|i| ((i * 31) % 7 % 2) as f32).collect();
        let relevance: Vec<f32> = (0..n).map(|i| ((i * 13) % 5) as f32).collect();
        let weights: Vec<f32> = (0..n).map(|i| 0.5 + (i % 3) as f32 * 0.25).collect();
        let mut sizes: Vec<usize> = (0..n / 40).map(|g| 1 + (g * 17) % 79).collect();
        sizes[2] = 0;
        let covered: usize = sizes.iter().sum();
        sizes.push(n - covered);
        let group = GroupInfo::from_sizes(&sizes);
        let times: Vec<f32> = (0..n)
            .map(|i| if i % 4 == 0 { -1.0 } else { 1.0 } * (1.0 + (i % 97) as f32))
            .collect();
        let lower: Vec<f32> = (0..n).map(|i| 0.5 + (i % 101) as f32 * 0.03).collect();
        let upper: Vec<f32> = lower
            .iter()
            .enumerate()
            .map(|(i, &l)| {
                if i % 3 == 2 {
                    f32::INFINITY
                } else {
                    l * (1.0 + (i % 3) as f32)
                }
            })
            .collect();
        let margins: Vec<f32> = (0..n)
            .map(|i| ((i * 37) % 211) as f32 / 50.0 - 2.0)
            .collect();
        // Cox reads hazards: strictly positive, so its value is finite.
        let hazards: Vec<f32> = margins.iter().map(|m| m.exp()).collect();
        let evaluate = || {
            let p = &preds[..n];
            let w = Some(weights.as_slice());
            let matrix = MetaInfo {
                n_rows: n,
                n_targets: 3,
                ..MetaInfo::new(&labels, Some(&weights), None)
            };
            let bounds = MetaInfo {
                n_rows: n,
                label_lower_bound: Some(&lower),
                label_upper_bound: Some(&upper),
                ..MetaInfo::new(&[], Some(&weights), None)
            };
            let g = Some(&group);
            [
                Auc.eval(p, &labels[..n], None),
                AucPr.eval(p, &labels[..n], w),
                Auc.eval_info(&preds, &matrix),
                Ndcg::new(None).eval_grouped(p, &relevance, w, g),
                Ndcg::new(Some(5)).eval_grouped(p, &relevance, w, g),
                MeanAveragePrecision::new(Some(10)).eval_grouped(p, &relevance, w, g),
                Precision::new("pre@3", Some(3)).eval_grouped(p, &labels[..n], w, g),
                Ndcg::new(Some(20)).eval(p, &relevance, None),
                CoxNLogLik.eval(&hazards, &times, None),
                AftNLogLik::new(AftDistribution::Logistic, 1.2).eval_info(&margins, &bounds),
                IntervalRegressionAccuracy.eval_info(&margins, &bounds),
            ]
        };
        let pool = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        };
        let serial = pool(1).install(evaluate);
        // A NaN or infinite value would compare equal without exercising
        // the accumulation.
        assert!(serial.iter().all(|v| v.is_finite()), "{serial:?}");
        let parallel = pool(4).install(evaluate);
        assert_eq!(serial.map(f64::to_bits), parallel.map(f64::to_bits));
    }

    #[test]
    fn rmse_basic() {
        let m = Rmse;
        // errors: 1, -1 -> mean sq 1 -> rmse 1
        assert_relative_eq!(m.eval(&[2.0, 0.0], &[1.0, 1.0], None), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn logloss_perfect_and_wrong() {
        let m = LogLoss;
        // near-perfect predictions -> ~0 loss
        let loss = m.eval(&[0.999_999, 0.000_001], &[1.0, 0.0], None);
        assert!(loss < 1e-4);
        // p=0.5 everywhere -> ln 2
        let loss = m.eval(&[0.5, 0.5], &[1.0, 0.0], None);
        assert_relative_eq!(loss, 2.0f64.ln(), epsilon = 1e-6);
    }

    #[test]
    fn error_rate_counts_misclassified() {
        let m = ErrorRate;
        // preds: 0.9->pos ok, 0.4->neg but label pos -> wrong, 0.2->neg ok
        assert_relative_eq!(
            m.eval(&[0.9, 0.4, 0.2], &[1.0, 1.0, 0.0], None),
            1.0 / 3.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn factory_defaults_to_objective_metric() {
        let obj = ObjectiveParams::default();
        let ms = create_metrics(&[], "rmse", 0, &obj).unwrap();
        assert_eq!(ms.len(), 1);
        assert_eq!(ms[0].name(), "rmse");
        assert!(create_metrics(&["nope".to_string()], "rmse", 0, &obj).is_err());
    }

    /// XGBoost's elementwise reduction over a label matrix: every
    /// `(row, target)` cell is one instance carrying its row's weight, so
    /// weighted RMSE is `sqrt(Σ w_i (y_ij − p_ij)² / (K Σ w_i))`.
    #[test]
    fn elementwise_metrics_average_every_cell_with_row_weights() {
        let labels = [1.0f32, 0.0, 3.0, 2.0, 0.0, 1.0];
        let preds = [2.0f32, 0.0, 1.0, 2.0, 1.0, 1.0];
        let weights = [1.0f32, 3.0];
        let info = MetaInfo {
            n_rows: 2,
            n_targets: 3,
            ..MetaInfo::new(&labels, Some(&weights), None)
        };
        // Row 0 squared errors 1, 0, 4 (weight 1); row 1: 0, 1, 0 (weight 3).
        let expected = ((1.0 + 4.0 + 3.0) / 12.0f64).sqrt();
        assert_relative_eq!(Rmse.eval_info(&preds, &info), expected, epsilon = 1e-12);
        // Row 0 absolute errors 1, 0, 2; row 1: 0, 1, 0.
        assert_relative_eq!(Mae.eval_info(&preds, &info), 6.0 / 12.0, epsilon = 1e-12);
        let unweighted = MetaInfo {
            weights: None,
            ..info
        };
        assert_relative_eq!(
            Rmse.eval_info(&preds, &unweighted),
            (6.0f64 / 6.0).sqrt(),
            epsilon = 1e-12
        );
    }

    /// Multi-label AUC / AUCPR is the plain mean of the per-target values.
    #[test]
    fn ranking_curve_metrics_macro_average_label_columns() {
        // Target 0 is ranked perfectly (AUC 1), target 1 exactly backwards
        // (0); pooling all cells instead would give 9/16.
        let labels = [1.0f32, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let preds = [0.9f32, 0.1, 0.8, 0.3, 0.2, 0.7, 0.1, 0.9];
        let info = MetaInfo {
            n_rows: 4,
            n_targets: 2,
            ..MetaInfo::new(&labels, None, None)
        };
        assert_relative_eq!(Auc.eval_info(&preds, &info), 0.5, epsilon = 1e-12);
        let per_target: f64 = (0..2)
            .map(|t| {
                let col = |v: &[f32]| v.iter().skip(t).step_by(2).copied().collect::<Vec<_>>();
                AucPr.eval(&col(&preds), &col(&labels), None)
            })
            .sum();
        assert_relative_eq!(
            AucPr.eval_info(&preds, &info),
            per_target / 2.0,
            epsilon = 1e-12
        );
    }

    /// Ranking, multiclass, and per-row survival metrics read one label (or
    /// interval) per row and refuse label matrices; elementwise and curve
    /// metrics accept them.
    #[test]
    fn label_matrix_support_is_declared_per_metric() {
        let obj = ObjectiveParams::default();
        for (name, supported) in [
            ("rmse", true),
            ("logloss", true),
            ("error", true),
            ("auc", true),
            ("aucpr", true),
            ("tweedie-nloglik@1.5", true),
            ("mlogloss", false),
            ("merror", false),
            ("ndcg", false),
            ("map@5", false),
            ("pre@3", false),
            ("cox-nloglik", false),
            ("aft-nloglik", false),
            ("interval-regression-accuracy", false),
        ] {
            let metric = build(name, 3, &obj).unwrap();
            assert_eq!(metric.supports_label_matrix(), supported, "{name}");
        }
    }

    /// A rank cutoff `@k` must be a positive integer (XGBoost bounds the
    /// top-k it sets from the suffix below by 1): anything else is refused
    /// instead of being truncated (`pre@2.9`), dropped (`pre@abc`), or cast
    /// to a zero or saturated `usize`. Other metrics refuse any suffix but
    /// `tweedie-nloglik`'s variance power in `[1, 2)`.
    #[test]
    fn metrics_reject_invalid_suffixes() {
        let obj = ObjectiveParams::default();
        let mut names: Vec<String> = [
            "pre@0.5",
            "pre@2.9",
            "pre@1@2",
            "tweedie-nloglik@",
            "tweedie-nloglik@abc",
            "tweedie-nloglik@2",
            "tweedie-nloglik@0.5",
            "tweedie-nloglik@nan",
            "error@0.7",
            "rmse@3",
            "auc@",
        ]
        .map(String::from)
        .into();
        for base in ["pre", "ndcg", "map"] {
            for k in [
                "", "0", "abc", "+3", " 3", "3-", "-", "NaN", "inf", "-1", "1e3",
            ] {
                names.push(format!("{base}@{k}"));
            }
        }
        for name in &names {
            let err = build(name, 0, &obj).err();
            assert!(
                matches!(err, Some(HessboostError::InvalidParameter { name: param, .. }) if param == "eval_metric"),
                "{name}: {err:?}"
            );
        }
        for name in [
            "pre@1",
            "pre@32",
            "ndcg@3",
            "map@5",
            "tweedie-nloglik",
            "tweedie-nloglik@1",
            "tweedie-nloglik@1.25",
        ] {
            assert!(build(name, 0, &obj).is_ok(), "{name}");
        }
        let m = build("pre@2", 0, &obj).unwrap();
        // All labels zero: no hits at any cutoff.
        assert_eq!(m.eval(&[0.9, 0.5, 0.1], &[0.0, 0.0, 0.0], None), 0.0);
    }

    /// Every metric declares the predictions per row it reads, and direct
    /// calls with inconsistent lengths evaluate to NaN instead of indexing
    /// out of bounds.
    #[test]
    fn mismatched_lengths_evaluate_to_nan() {
        let obj = ObjectiveParams {
            quantile_alpha: vec![0.2, 0.8],
            expectile_alpha: vec![0.5],
            distribution: Some(crate::objective::distributional::DistFamily::Normal),
            ..ObjectiveParams::default()
        };
        let widths = [
            ("rmse", 1),
            ("mae", 1),
            ("logloss", 1),
            ("error", 1),
            ("auc", 1),
            ("aucpr", 1),
            ("mlogloss", 3),
            ("merror", 3),
            ("poisson-nloglik", 1),
            ("gamma-nloglik", 1),
            ("tweedie-nloglik", 1),
            ("ndcg", 1),
            ("map", 1),
            ("rmsle", 1),
            ("mape", 1),
            ("mphe", 1),
            ("pre@2", 1),
            ("quantile", 2),
            ("expectile", 1),
            ("cox-nloglik", 1),
            ("aft-nloglik", 1),
            ("interval-regression-accuracy", 1),
            ("nll", 2),
            ("crps", 2),
        ];
        let labels = [1.0, 0.0, 1.0];
        let info = MetaInfo::new(&labels, None, None);
        for (name, width) in widths {
            let metric = build(name, 3, &obj).unwrap();
            assert_eq!(metric.prediction_width(&info), Some(width), "{name}");
            let preds = vec![0.5f32; labels.len() * width + 1];
            let valid = &preds[..labels.len() * width];
            // Consistent lengths evaluate (the value itself may be NaN for
            // labels outside the metric's domain).
            let _ = metric.eval(valid, &labels, Some(&[1.0; 3]));
            for (preds, weights) in [
                (&preds[..], None),
                (&preds[..labels.len() * width - 1], None),
                (valid, Some(&[1.0f32; 2][..])),
            ] {
                assert!(metric.eval(preds, &labels, weights).is_nan(), "{name}");
                let info = MetaInfo::new(&labels, weights, None);
                assert!(metric.eval_info(preds, &info).is_nan(), "{name}");
            }
        }
        let custom = CustomMetric::new("first", false, |p, _, _| f64::from(p[0]));
        assert_eq!(custom.prediction_width(&info), None);
        assert_eq!(custom.eval(&[2.0, 3.0, 4.0], &labels, None), 2.0);
        assert!(custom.eval(&[], &labels, None).is_nan());
        assert!(custom.eval(&[2.0; 4], &labels, None).is_nan());
        assert!(custom.eval(&[2.0; 3], &labels, Some(&[1.0])).is_nan());
    }

    /// Metadata whose `n_targets` disagrees with its lengths evaluates to
    /// NaN through the default `eval_info` (a `usize::MAX` once overflowed
    /// the per-cell weight allocation, and with no labels to back the cells
    /// once allocated `n_rows * n_targets` weights).
    #[test]
    fn inconsistent_label_matrix_metadata_evaluates_to_nan() {
        let labels = [1.0f32, 2.0, 3.0, 4.0];
        let weights = [1.0f32, 1.0];
        let info = MetaInfo {
            n_rows: 2,
            n_targets: 2,
            ..MetaInfo::new(&labels, Some(&weights), None)
        };
        assert_eq!(Rmse.eval_info(&labels, &info), 0.0);
        for n_targets in [usize::MAX, 0, 3] {
            let bad = MetaInfo { n_targets, ..info };
            assert!(Rmse.eval_info(&labels, &bad).is_nan(), "{n_targets}");
        }
        let mut unlabeled = MetaInfo::new(&[], Some(&[1.0]), None);
        unlabeled.n_rows = 1;
        for n_targets in [2, usize::MAX] {
            unlabeled.n_targets = n_targets;
            assert!(Rmse.eval_info(&[], &unlabeled).is_nan(), "{n_targets}");
        }
        // Bounds-only metadata (no labels) stays valid where it is read.
        let bounds = [1.0f32];
        let aft = MetaInfo {
            label_lower_bound: Some(&bounds),
            label_upper_bound: Some(&bounds),
            ..MetaInfo::new(&[], Some(&[1.0]), None)
        };
        let aft = MetaInfo { n_rows: 1, ..aft };
        let nloglik = AftNLogLik::new(crate::config::AftDistribution::Normal, 1.0);
        assert!(nloglik.validate_info(&aft).is_ok());
        assert!(nloglik.eval_info(&[0.0], &aft).is_finite());
    }

    #[test]
    fn auc_ranks_perfectly_separable() {
        let m = Auc;
        // preds perfectly separate: positives above negatives -> AUC 1.
        let auc = m.eval(&[0.1, 0.2, 0.8, 0.9], &[0.0, 0.0, 1.0, 1.0], None);
        assert!((auc - 1.0).abs() < 1e-9);
        assert!(m.maximize());
    }

    #[test]
    fn ndcg_perfect_and_reversed() {
        use crate::data::GroupInfo;
        let m = Ndcg::new(None);
        let labels = [3.0f32, 2.0, 0.0];
        // Predictions rank docs in ideal order -> NDCG 1.
        let perfect = [0.9f32, 0.5, 0.1];
        let g = GroupInfo::from_sizes(&[3]);
        assert_relative_eq!(
            m.eval_grouped(&perfect, &labels, None, Some(&g)),
            1.0,
            epsilon = 1e-6
        );
        // Reversed order: gains 0,3,7 at discounts 1, 1/log2(3), 1/2.
        // DCG = 3/log2(3) + 7/2 = 5.39278; IDCG = 7 + 3/log2(3) = 8.89278.
        let reversed = [0.1f32, 0.5, 0.9];
        let got = m.eval_grouped(&reversed, &labels, None, Some(&g));
        assert_relative_eq!(got, 5.392_789 / 8.892_789, epsilon = 1e-5);
        assert!(m.maximize());
    }

    #[test]
    fn ndcg_truncation_at_k() {
        // With @1 only the top-ranked doc counts.
        let m = Ndcg::new(Some(1));
        let labels = [3.0f32, 2.0, 0.0];
        // Best doc on top -> DCG=IDCG -> 1.
        assert_relative_eq!(m.eval(&[0.9, 0.5, 0.1], &labels, None), 1.0, epsilon = 1e-6);
        // Worst doc on top -> DCG 0 -> NDCG 0.
        assert_relative_eq!(m.eval(&[0.1, 0.5, 0.9], &labels, None), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn map_hand_computed() {
        let m = MeanAveragePrecision::new(None);
        let labels = [1.0f32, 0.0, 1.0, 0.0]; // 2 relevant docs
        // Order both relevant docs first -> AP = (1/1 + 2/2)/2 = 1.
        assert_relative_eq!(
            m.eval(&[0.9, 0.1, 0.8, 0.2], &labels, None),
            1.0,
            epsilon = 1e-9
        );
        // Relevant docs at ranks 2 and 4 -> AP = (1/2 + 2/4)/2 = 0.5.
        assert_relative_eq!(
            m.eval(&[0.8, 0.9, 0.1, 0.7], &labels, None),
            0.5,
            epsilon = 1e-9
        );
        assert!(m.maximize());
    }

    #[test]
    fn ranking_metrics_average_over_groups() {
        use crate::data::GroupInfo;
        // Two groups: one perfectly ranked (MAP 1), one poorly (MAP 0.5).
        let labels = [1.0f32, 0.0, 1.0, 0.0];
        let preds = [0.9f32, 0.1, 0.2, 0.8]; // g0 perfect, g1 relevant last
        let g = GroupInfo::from_sizes(&[2, 2]);
        let m = MeanAveragePrecision::new(None);
        // g0: relevant on top -> AP 1. g1: relevant doc (idx2) ranked below -> AP 0.5.
        assert_relative_eq!(
            m.eval_grouped(&preds, &labels, None, Some(&g)),
            0.75,
            epsilon = 1e-9
        );
    }

    #[test]
    fn factory_parses_ranking_metrics_with_k() {
        // An `@k` suffix parses and is dropped from the name.
        for (name, base) in [
            ("ndcg", "ndcg"),
            ("map", "map"),
            ("ndcg@5", "ndcg"),
            ("map@10", "map"),
        ] {
            let metric = build(name, 0, &ObjectiveParams::default()).unwrap();
            assert_eq!(metric.name(), base, "{name}");
        }
    }

    #[test]
    fn aucpr_perfect_and_ranks_better_than_random() {
        let m = AucPr;
        assert!(m.maximize());
        assert_eq!(
            build("aucpr", 0, &ObjectiveParams::default())
                .unwrap()
                .name(),
            "aucpr"
        );

        // Perfectly separable: all positives scored above all negatives -> ~1.
        let perfect = m.eval(&[0.1, 0.2, 0.8, 0.9], &[0.0, 0.0, 1.0, 1.0], None);
        assert_relative_eq!(perfect, 1.0, epsilon = 1e-9);

        // A ranking that puts positives up front beats one that scatters them.
        let labels = [1.0f32, 1.0, 1.0, 0.0, 0.0, 0.0];
        let good = m.eval(&[0.9, 0.8, 0.7, 0.3, 0.2, 0.1], &labels, None);
        let poor = m.eval(&[0.9, 0.2, 0.7, 0.8, 0.1, 0.3], &labels, None);
        assert_relative_eq!(good, 1.0, epsilon = 1e-9);
        assert!(
            good > poor,
            "better ranking should score higher: {good} vs {poor}"
        );
        // The prevalence baseline (3/6) is the expected value of a random ranker;
        // a good ranking clears it comfortably.
        assert!(poor > 0.5, "poor ranking still beats nothing: {poor}");
    }

    #[test]
    fn aucpr_degenerate_returns_zero() {
        let m = AucPr;
        // No negatives (or no positives) -> undefined PR curve, reported as 0.
        assert_eq!(m.eval(&[0.3, 0.6, 0.9], &[1.0, 1.0, 1.0], None), 0.0);
        assert_eq!(m.eval(&[0.3, 0.6, 0.9], &[0.0, 0.0, 0.0], None), 0.0);
    }

    #[test]
    fn mlogloss_and_merror() {
        // 2 rows, 3 classes. Confident correct predictions.
        let ml = MLogLoss { num_class: 3 };
        let me = MError { num_class: 3 };
        let preds = [0.8, 0.1, 0.1, 0.05, 0.9, 0.05];
        let labels = [0.0, 1.0];
        assert!(ml.eval(&preds, &labels, None) < 0.3);
        assert_eq!(me.eval(&preds, &labels, None), 0.0);
        // A wrong argmax raises merror.
        let labels_wrong = [1.0, 1.0];
        assert_eq!(me.eval(&preds, &labels_wrong, None), 0.5);
    }
}
