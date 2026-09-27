//! Evaluation metrics used for reporting and early stopping.
//!
//! Metrics receive predictions that have already passed through the objective's
//! [`crate::objective::Loss::eval_transform`] (so classification metrics
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

/// The [`Metric::eval`] and [`Metric::eval_info`] of a `CellMetric`. Defined before the
/// submodules so their metrics can use it too.
macro_rules! cell_metric_eval {
    () => {
        fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
            $crate::metric::CellMetric::eval_cells(
                self,
                preds,
                labels,
                weights.map($crate::simd::RowWeights::from),
            )
        }
        fn eval_info(&self, preds: &[f32], info: &$crate::data::MetaInfo) -> f64 {
            $crate::metric::eval_cells_info(self, preds, info)
        }
    };
}

mod distributional;
mod elementwise;
mod quantile;
mod ranking;
mod survival;

use distributional::{DistCrps, DistNll};
use elementwise::{Mape, PseudoHuberError, Rmsle};
use quantile::{ExpectileError, QuantileError};
use ranking::Precision;
use survival::{AftNLogLik, CoxNLogLik, IntervalRegressionAccuracy};

use crate::data::MetaInfo;
use crate::error::{HessboostError, Result};
use crate::objective::distributional::DistFamily;
use crate::objective::{Aft, AftDistribution, Expectiles, PseudoHuber, Quantiles, Tweedie};
use crate::simd::RowWeights;
use rayon::prelude::*;
use std::borrow::Cow;
use std::num::NonZeroUsize;

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
        require_labels(self.name(), info)
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

/// A built-in elementwise metric: a weighted reduction over `[row][target]`
/// cells, one prediction per cell, whose weights are read through
/// [`RowWeights`] so a label matrix never materializes its repeated row
/// weights. [`CustomMetric`] and other external metrics keep the
/// materializing default [`Metric::eval_info`].
trait CellMetric: Metric {
    /// The metric over cells: NaN unless `preds` holds one value per label
    /// and `weights`, when given, cover exactly the labels.
    fn eval_cells(&self, preds: &[f32], labels: &[f32], weights: Option<RowWeights<'_>>) -> f64;
}

/// Whether `preds` holds one value per label and `weights`, when given,
/// cover exactly the labels: [`consistent`] for [`CellMetric::eval_cells`].
fn cells_consistent(preds: &[f32], labels: &[f32], weights: Option<RowWeights<'_>>) -> bool {
    preds.len() == labels.len() && weights.is_none_or(|w| w.cells() == Some(labels.len()))
}

/// [`Metric::eval_info`] of a [`CellMetric`]: the default elementwise
/// reduction, with each row's weight read for its `n_targets` cells in
/// place instead of repeated into a cell-weight buffer.
fn eval_cells_info(metric: &impl CellMetric, preds: &[f32], info: &MetaInfo) -> f64 {
    let stride = info.n_targets;
    if info.check_layout().is_err()
        || (stride > 1 && info.n_rows.checked_mul(stride) != Some(info.labels.len()))
    {
        return f64::NAN;
    }
    let weights = info.weights.map(|w| RowWeights::new(w, stride));
    metric.eval_cells(preds, info.labels, weights)
}

/// Normalize a metric total, returning zero for an empty or nonpositive weight sum.
#[inline]
fn weighted_mean((total, weight): (f64, f64)) -> f64 {
    if weight > 0.0 { total / weight } else { 0.0 }
}

/// [`Metric::validate_info`]'s default: the metric named `name` reads
/// ordinary labels, which a dataset with rows must carry.
fn require_labels(name: &str, info: &MetaInfo) -> Result<()> {
    if info.n_rows > 0 && info.labels.is_empty() {
        return Err(HessboostError::invalid_param(
            "eval_metric",
            format!("metric `{name}` needs labels, but dataset has none"),
        ));
    }
    Ok(())
}

/// The first label that is not a class index in `0..num_class` (XGBoost's
/// `MultiClassEvaluation` label check), if any.
fn first_non_class(labels: &[f32], num_class: usize) -> Option<f32> {
    labels
        .iter()
        .copied()
        .find(|&label| !(label >= 0.0 && label.fract() == 0.0 && (label as usize) < num_class))
}

/// Define a purely-pointwise metric from its SIMD weighted-sum kernel.
/// Generates the metric struct plus its [`Metric`] impl from the metric name
/// and the `crate::simd` kernel path. The plain arm is a [`CellMetric`]
/// whose `eval_cells` is `weighted_mean(kernel(preds, labels, weights))`,
/// NaN for inconsistent lengths; `rmse` takes `=> sqrt` for its root. The
/// `field: Type, …, per_label` arm is a metric reading `self.field`
/// predictions per label and passing `self.field` as the kernel's final
/// argument (the multiclass metrics: one probability per class, one class
/// id per row, so no label matrices). All generated metrics minimize
/// (`maximize` keeps its default `false`); metrics with non-trivial logic
/// (`auc`, `aucpr`, ranking) stay handwritten below.
macro_rules! simple_metric {
    ($(#[$m:meta])* $ty:ident, $name:literal, $simd:path $(=> $root:ident)?) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, Default)]
        #[non_exhaustive]
        pub(crate) struct $ty;
        impl Metric for $ty {
            fn name(&self) -> &str {
                $name
            }
            cell_metric_eval!();
        }
        impl CellMetric for $ty {
            fn eval_cells(
                &self,
                preds: &[f32],
                labels: &[f32],
                weights: Option<RowWeights<'_>>,
            ) -> f64 {
                if !cells_consistent(preds, labels, weights) {
                    return f64::NAN;
                }
                weighted_mean($simd(preds, labels, weights))$(.$root())?
            }
        }
    };
    ($(#[$m:meta])* $ty:ident, $name:literal, $field:ident: $field_ty:ty, $simd:path,
        per_label) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy)]
        pub(crate) struct $ty {
            $field: $field_ty,
        }
        impl Metric for $ty {
            fn name(&self) -> &str {
                $name
            }
            fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
                nan_unless_consistent!(preds, labels, weights, self.$field);
                if first_non_class(labels, self.$field).is_some() {
                    return f64::NAN;
                }
                weighted_mean($simd(preds, labels, weights, self.$field))
            }
            fn validate_info(&self, info: &MetaInfo) -> Result<()> {
                require_labels(self.name(), info)?;
                match first_non_class(info.labels, self.$field) {
                    None => Ok(()),
                    Some(label) => Err(HessboostError::invalid_param(
                        "eval_metric",
                        format!(
                            "metric `{}` reads labels as class indices in 0..{}, but dataset \
                             has label {label}",
                            self.name(),
                            self.$field
                        ),
                    )),
                }
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
pub(crate) struct Auc;

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

/// Tweedie negative log-likelihood (`tweedie-nloglik`) with variance power
/// `rho`, named `tweedie-nloglik@rho` as XGBoost reports it (`rho` to six
/// significant digits: `tweedie-nloglik@1.5`, `tweedie-nloglik@1`).
#[derive(Debug, Clone)]
pub(crate) struct TweedieNLogLik {
    rho: f64,
    name: String,
}

impl TweedieNLogLik {
    fn new(rho: f64) -> Self {
        TweedieNLogLik {
            rho,
            name: tweedie_name(rho),
        }
    }
}

/// `tweedie-nloglik@rho` as XGBoost names it: C++'s default stream
/// precision, six significant digits, then the shortest form (`1.0` ->
/// `1`), which Rust's `Display` gives.
fn tweedie_name(rho: f64) -> String {
    let rounded: f64 = format!("{rho:.5e}").parse().unwrap_or(rho);
    format!("tweedie-nloglik@{rounded}")
}

impl Metric for TweedieNLogLik {
    fn name(&self) -> &str {
        &self.name
    }
    cell_metric_eval!();
}

impl CellMetric for TweedieNLogLik {
    fn eval_cells(&self, preds: &[f32], labels: &[f32], weights: Option<RowWeights<'_>>) -> f64 {
        if !cells_consistent(preds, labels, weights) {
            return f64::NAN;
        }
        weighted_mean(crate::simd::tweedie_nloglik_sum(
            preds, labels, weights, self.rho,
        ))
    }
}

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
pub(crate) struct AucPr;

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

/// A rank cutoff of the ranking metrics (`ndcg`, `map`, `pre`): the top
/// `k` documents of every query group, or the whole list. XGBoost's `@k`
/// metric suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cutoff {
    top_k: Option<NonZeroUsize>,
}

impl Cutoff {
    /// No cutoff: `ndcg` and `map` score the whole list; `pre` without a
    /// cutoff scores the top 32, as XGBoost does, and is still named `pre`.
    pub fn all() -> Self {
        Cutoff { top_k: None }
    }

    /// The top `k` documents (`@k`).
    ///
    /// # Errors
    ///
    /// `k` is 0 (XGBoost's lower bound on the top-k it reads from the
    /// suffix is 1).
    pub fn top(k: usize) -> Result<Self> {
        NonZeroUsize::new(k).map(Cutoff::from).ok_or_else(|| {
            HessboostError::invalid_param("eval_metric", "the `@k` cutoff must be at least 1")
        })
    }

    /// The cutoff `k`, `None` for the whole list.
    pub fn top_k(&self) -> Option<NonZeroUsize> {
        self.top_k
    }

    /// The cutoff as the ranking metrics compute with it.
    fn k(self) -> Option<usize> {
        self.top_k.map(NonZeroUsize::get)
    }
}

impl From<NonZeroUsize> for Cutoff {
    fn from(k: NonZeroUsize) -> Self {
        Cutoff { top_k: Some(k) }
    }
}

/// A built-in evaluation metric with its parameters: the typed form of
/// XGBoost's `eval_metric` names, which training builds with
/// [`EvalMetric::build`]. Every metric carries the parameters it evaluates
/// with (unlike XGBoost, where `mphe`, `quantile`, `expectile`, and
/// `aft-nloglik` read the objective's parameters);
/// [`TrainingParams::from_xgboost`](crate::config::TrainingParams::from_xgboost)
/// reads XGBoost's names and fills those parameters the way XGBoost does.
///
/// [`name`](EvalMetric::name) is XGBoost's `evals_result` key, suffix
/// included (`ndcg@5`, `tweedie-nloglik@1.5`).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EvalMetric {
    /// Root-mean-square error (`rmse`).
    Rmse,
    /// Root-mean-square log error (`rmsle`).
    Rmsle,
    /// Mean absolute error (`mae`).
    Mae,
    /// Mean absolute percentage error (`mape`).
    Mape,
    /// Mean pseudo-Huber error (`mphe`) with this slope.
    Mphe(PseudoHuber),
    /// Binary log loss (`logloss`); raw margins for `binary:logitraw`.
    LogLoss,
    /// Binary error rate at threshold 0.5 (`error`).
    Error,
    /// ROC AUC (`auc`); the mean per-target AUC for a label matrix.
    Auc,
    /// Area under the precision-recall curve (`aucpr`).
    AucPr,
    /// Multiclass log loss (`mlogloss`), one probability per class.
    MLogLoss,
    /// Multiclass error rate (`merror`).
    MError,
    /// Poisson negative log-likelihood (`poisson-nloglik`).
    PoissonNLogLik,
    /// Gamma negative log-likelihood (`gamma-nloglik`).
    GammaNLogLik,
    /// Tweedie negative log-likelihood (`tweedie-nloglik@rho`) at this
    /// variance power.
    TweedieNLogLik(Tweedie),
    /// Normalized discounted cumulative gain (`ndcg`, `ndcg@k`).
    Ndcg(Cutoff),
    /// Mean average precision (`map`, `map@k`).
    Map(Cutoff),
    /// Precision (`pre`, `pre@k`).
    Precision(Cutoff),
    /// Pinball loss at these quantiles (`quantile`), one prediction per
    /// level.
    Quantile(Quantiles),
    /// Expectile loss at these levels (`expectile`).
    Expectile(Expectiles),
    /// Cox proportional-hazards negative partial log-likelihood
    /// (`cox-nloglik`).
    CoxNLogLik,
    /// Accelerated-failure-time negative log-likelihood (`aft-nloglik`)
    /// under this noise model.
    AftNLogLik(Aft),
    /// Fraction of predictions inside their label interval
    /// (`interval-regression-accuracy`).
    IntervalRegressionAccuracy,
    /// Negative log-likelihood of predicted distributions of this family
    /// (`nll`, beyond XGBoost; see [`crate::objective::distributional`]).
    Nll(DistFamily),
    /// Continuous ranked probability score of predicted distributions of
    /// this family (`crps`, beyond XGBoost).
    Crps(DistFamily),
}

/// XGBoost's name of a ranking metric: `base@k` with a cutoff, else `base`.
fn cutoff_name(base: &str, k: Option<usize>) -> String {
    k.map_or_else(|| base.to_string(), |k| format!("{base}@{k}"))
}

impl EvalMetric {
    /// The metric's spelling in XGBoost's flat `eval_metric`, which
    /// [`TrainingParams::from_xgboost`](crate::config::TrainingParams::from_xgboost)
    /// reads back to the same metric: [`name`](Self::name), except that a
    /// Tweedie power is written in full rather than rounded to the six
    /// digits of its `evals_result` key.
    pub(crate) fn flat_name(&self) -> Cow<'static, str> {
        match self {
            EvalMetric::TweedieNLogLik(tweedie) => {
                Cow::Owned(format!("tweedie-nloglik@{}", tweedie.variance_power()))
            }
            _ => self.name(),
        }
    }

    /// XGBoost's `evals_result` key: the metric's name with its suffix
    /// (`ndcg@5`, `tweedie-nloglik@1.5`), as
    /// [`Metric::name`] of the built metric reports it.
    pub fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed(match self {
            EvalMetric::Rmse => "rmse",
            EvalMetric::Rmsle => "rmsle",
            EvalMetric::Mae => "mae",
            EvalMetric::Mape => "mape",
            EvalMetric::Mphe(_) => "mphe",
            EvalMetric::LogLoss => "logloss",
            EvalMetric::Error => "error",
            EvalMetric::Auc => "auc",
            EvalMetric::AucPr => "aucpr",
            EvalMetric::MLogLoss => "mlogloss",
            EvalMetric::MError => "merror",
            EvalMetric::PoissonNLogLik => "poisson-nloglik",
            EvalMetric::GammaNLogLik => "gamma-nloglik",
            EvalMetric::TweedieNLogLik(tweedie) => {
                return Cow::Owned(tweedie_name(tweedie.variance_power()));
            }
            EvalMetric::Ndcg(cutoff) => return Cow::Owned(cutoff_name("ndcg", cutoff.k())),
            EvalMetric::Map(cutoff) => return Cow::Owned(cutoff_name("map", cutoff.k())),
            EvalMetric::Precision(cutoff) => return Cow::Owned(cutoff_name("pre", cutoff.k())),
            EvalMetric::Quantile(_) => "quantile",
            EvalMetric::Expectile(_) => "expectile",
            EvalMetric::CoxNLogLik => "cox-nloglik",
            EvalMetric::AftNLogLik(_) => "aft-nloglik",
            EvalMetric::IntervalRegressionAccuracy => "interval-regression-accuracy",
            EvalMetric::Nll(_) => "nll",
            EvalMetric::Crps(_) => "crps",
        })
    }

    /// The metric, ready to evaluate the predictions of a model with
    /// `n_outputs` outputs per row (`mlogloss` and `merror` read one
    /// probability per class).
    ///
    /// # Errors
    ///
    /// `mlogloss` or `merror` for fewer than two outputs.
    ///
    /// ```
    /// use hessboost::metric::{EvalMetric, Metric};
    ///
    /// # fn main() -> hessboost::error::Result<()> {
    /// let rmse = EvalMetric::Rmse.build(1)?;
    /// assert_eq!(rmse.eval(&[1.0, 3.0], &[1.0, 1.0], None), 2f64.sqrt());
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(&self, n_outputs: usize) -> Result<Box<dyn Metric>> {
        let classes = || {
            if n_outputs >= 2 {
                Ok(n_outputs)
            } else {
                Err(HessboostError::invalid_param(
                    "eval_metric",
                    format!(
                        "`{}` scores one probability per class and needs a multiclass model, \
                         got {n_outputs} output(s)",
                        self.name()
                    ),
                ))
            }
        };
        Ok(match self {
            EvalMetric::Rmse => Box::new(Rmse),
            EvalMetric::Rmsle => Box::new(Rmsle),
            EvalMetric::Mae => Box::new(Mae),
            EvalMetric::Mape => Box::new(Mape),
            EvalMetric::Mphe(huber) => Box::new(PseudoHuberError::new(huber.slope() as f32)),
            EvalMetric::LogLoss => Box::new(LogLoss),
            EvalMetric::Error => Box::new(ErrorRate),
            EvalMetric::Auc => Box::new(Auc),
            EvalMetric::AucPr => Box::new(AucPr),
            EvalMetric::MLogLoss => Box::new(MLogLoss {
                num_class: classes()?,
            }),
            EvalMetric::MError => Box::new(MError {
                num_class: classes()?,
            }),
            EvalMetric::PoissonNLogLik => Box::new(PoissonNLogLik),
            EvalMetric::GammaNLogLik => Box::new(GammaNLogLik),
            EvalMetric::TweedieNLogLik(tweedie) => {
                Box::new(TweedieNLogLik::new(tweedie.variance_power()))
            }
            EvalMetric::Ndcg(cutoff) => Box::new(Ndcg::new(cutoff.k())),
            EvalMetric::Map(cutoff) => Box::new(MeanAveragePrecision::new(cutoff.k())),
            EvalMetric::Precision(cutoff) => Box::new(Precision::new(cutoff.k())),
            EvalMetric::Quantile(quantiles) => Box::new(QuantileError::new(quantiles.alpha_f32())),
            EvalMetric::Expectile(expectiles) => {
                Box::new(ExpectileError::new(expectiles.alpha_f32()))
            }
            EvalMetric::CoxNLogLik => Box::new(CoxNLogLik),
            EvalMetric::AftNLogLik(aft) => {
                Box::new(AftNLogLik::new(aft.distribution(), aft.scale() as f32))
            }
            EvalMetric::IntervalRegressionAccuracy => Box::new(IntervalRegressionAccuracy),
            EvalMetric::Nll(family) => Box::new(DistNll::new(*family)),
            EvalMetric::Crps(family) => Box::new(DistCrps::new(*family)),
        })
    }

    /// The metric XGBoost names `name`, with the parameters XGBoost would
    /// give it: the `@k` cutoff of `ndcg`/`map`/`pre` and the `@rho` power of
    /// `tweedie-nloglik` (1.5 without one) from the suffix, the rest from
    /// `source` (the flat parameters `mphe`, `quantile`, `expectile`, and
    /// `aft-nloglik` read, and the `dist:*` family `nll` and `crps` score).
    ///
    /// Only those four names take an `@` suffix; any other suffix, including
    /// XGBoost's `error@t` threshold and the `-` variants (`ndcg@3-`), is
    /// refused, as are unknown names.
    pub(crate) fn from_xgboost(name: &str, source: &XgboostMetricSource<'_>) -> Result<Self> {
        let (base, suffix) = match name.split_once('@') {
            Some((b, s)) => (b, Some(s)),
            None => (name, None),
        };
        if suffix.is_some() && !matches!(base, "tweedie-nloglik" | "ndcg" | "map" | "pre") {
            return Err(invalid_metric(
                name,
                &format!("`{base}` takes no `@` suffix"),
            ));
        }
        let cutoff = || rank_cutoff(name, suffix);
        Ok(match base {
            "rmse" => EvalMetric::Rmse,
            "rmsle" => EvalMetric::Rmsle,
            "mae" => EvalMetric::Mae,
            "mape" => EvalMetric::Mape,
            "mphe" => EvalMetric::Mphe(source.huber()?),
            "logloss" => EvalMetric::LogLoss,
            "error" => EvalMetric::Error,
            "auc" => EvalMetric::Auc,
            "aucpr" => EvalMetric::AucPr,
            "mlogloss" => EvalMetric::MLogLoss,
            "merror" => EvalMetric::MError,
            "poisson-nloglik" => EvalMetric::PoissonNLogLik,
            "gamma-nloglik" => EvalMetric::GammaNLogLik,
            "tweedie-nloglik" => EvalMetric::TweedieNLogLik(tweedie_power(name, suffix)?),
            "ndcg" => EvalMetric::Ndcg(cutoff()?),
            "map" => EvalMetric::Map(cutoff()?),
            "pre" => EvalMetric::Precision(cutoff()?),
            "quantile" => EvalMetric::Quantile(source.quantiles()?),
            "expectile" => EvalMetric::Expectile(source.expectiles()?),
            "cox-nloglik" => EvalMetric::CoxNLogLik,
            "aft-nloglik" => EvalMetric::AftNLogLik(source.aft()?),
            "interval-regression-accuracy" => EvalMetric::IntervalRegressionAccuracy,
            "nll" | "crps" => {
                let family = source.distribution.ok_or_else(|| {
                    HessboostError::invalid_param(
                        "eval_metric",
                        format!(
                            "`{name}` scores predicted distributions and needs a `dist:*` objective"
                        ),
                    )
                })?;
                if base == "nll" {
                    EvalMetric::Nll(family)
                } else {
                    EvalMetric::Crps(family)
                }
            }
            other => return Err(HessboostError::unknown("metric", other)),
        })
    }
}

/// The flat XGBoost parameters the metrics named in `eval_metric` read
/// ([`EvalMetric::from_xgboost`]), as XGBoost gives them whatever the
/// objective.
pub(crate) struct XgboostMetricSource<'a> {
    /// `huber_slope` (`mphe`).
    pub(crate) huber_slope: f64,
    /// `quantile_alpha` (`quantile`).
    pub(crate) quantile_alpha: &'a [f64],
    /// `expectile_alpha` (`expectile`).
    pub(crate) expectile_alpha: &'a [f64],
    /// `aft_loss_distribution` (`aft-nloglik`).
    pub(crate) aft_loss_distribution: AftDistribution,
    /// `aft_loss_distribution_scale` (`aft-nloglik`).
    pub(crate) aft_loss_distribution_scale: f64,
    /// The family of a `dist:*` objective (`nll`, `crps`).
    pub(crate) distribution: Option<DistFamily>,
}

impl XgboostMetricSource<'_> {
    fn huber(&self) -> Result<PseudoHuber> {
        PseudoHuber::new(self.huber_slope)
    }

    fn quantiles(&self) -> Result<Quantiles> {
        Quantiles::new(self.quantile_alpha.iter().copied())
    }

    fn expectiles(&self) -> Result<Expectiles> {
        Expectiles::new(self.expectile_alpha.iter().copied())
    }

    fn aft(&self) -> Result<Aft> {
        Aft::new(self.aft_loss_distribution, self.aft_loss_distribution_scale)
    }
}

/// The parameter error of metric `name`.
fn invalid_metric(name: &str, reason: &str) -> HessboostError {
    HessboostError::invalid_param("eval_metric", format!("`{name}`: {reason}"))
}

/// The rank cutoff `@k` of ranking metric `name`: decimal digits only
/// (`usize::from_str` also takes a leading `+`), so `2.9`, `abc`, `1@2`, or
/// an empty suffix are refused rather than truncated or dropped.
fn rank_cutoff(name: &str, suffix: Option<&str>) -> Result<Cutoff> {
    match suffix {
        None => Ok(Cutoff::all()),
        Some(s) if s.ends_with('-') => Err(invalid_metric(
            name,
            "the `-` variants of the ranking metrics are not implemented",
        )),
        Some(s) => match s.parse::<usize>().ok().and_then(NonZeroUsize::new) {
            Some(k) if s.bytes().all(|b| b.is_ascii_digit()) => Ok(Cutoff::from(k)),
            _ => Err(invalid_metric(
                name,
                "the `@k` cutoff must be a positive integer",
            )),
        },
    }
}

/// The variance power `@rho` of `tweedie-nloglik` (`1.5` without a
/// suffix), in the range of the objective's `tweedie_variance_power`,
/// whose default metric this is.
fn tweedie_power(name: &str, suffix: Option<&str>) -> Result<Tweedie> {
    match suffix {
        None => Ok(Tweedie::default()),
        Some(s) => s
            .parse::<f64>()
            .ok()
            .and_then(|rho| Tweedie::new(rho).ok())
            .ok_or_else(|| invalid_metric(name, "the variance power `@rho` must be in [1, 2)")),
    }
}

/// The metric XGBoost names `name`, parameterized from `source` and built
/// for `n_outputs` outputs (tests).
#[cfg(test)]
pub(crate) fn named(
    name: &str,
    n_outputs: usize,
    source: &XgboostMetricSource<'_>,
) -> Result<Box<dyn Metric>> {
    EvalMetric::from_xgboost(name, source)?.build(n_outputs)
}

/// XGBoost's default flat parameters of the metrics (tests): slope 1, no
/// alphas, normal AFT noise at scale 1, no `dist:*` family.
#[cfg(test)]
pub(crate) const DEFAULT_SOURCE: XgboostMetricSource<'static> = XgboostMetricSource {
    huber_slope: 1.0,
    quantile_alpha: &[],
    expectile_alpha: &[],
    aft_loss_distribution: AftDistribution::Normal,
    aft_loss_distribution_scale: 1.0,
    distribution: None,
};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// The parallel sorts, per-group scores, and per-row interval values give
    /// the serial metric values bit for bit: large inputs with many ties, many
    /// query groups (one empty), weights, and label matrices.
    #[test]
    fn parallel_evaluation_matches_serial() {
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
                Precision::new(Some(3)).eval_grouped(p, &labels[..n], w, g),
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

    /// Every XGBoost metric name reads back as the metric whose name it is,
    /// and that name is the built metric's `evals_result` key; unknown
    /// names are refused.
    #[test]
    fn xgboost_names_parse_to_metrics_of_that_name() {
        let source = XgboostMetricSource {
            quantile_alpha: &[0.2, 0.8],
            expectile_alpha: &[0.5],
            distribution: Some(DistFamily::Normal),
            ..DEFAULT_SOURCE
        };
        for name in [
            "rmse",
            "rmsle",
            "mae",
            "mape",
            "mphe",
            "logloss",
            "error",
            "auc",
            "aucpr",
            "mlogloss",
            "merror",
            "poisson-nloglik",
            "gamma-nloglik",
            "tweedie-nloglik@1.5",
            "tweedie-nloglik@1",
            "ndcg",
            "ndcg@5",
            "map",
            "map@3",
            "pre",
            "pre@2",
            "quantile",
            "expectile",
            "cox-nloglik",
            "aft-nloglik",
            "interval-regression-accuracy",
            "nll",
            "crps",
        ] {
            let metric = EvalMetric::from_xgboost(name, &source).unwrap();
            assert_eq!(metric.name(), name);
            assert_eq!(metric.build(3).unwrap().name(), name);
        }
        // No suffix is the default power; the name always carries it.
        assert_eq!(
            EvalMetric::from_xgboost("tweedie-nloglik", &source).unwrap(),
            EvalMetric::TweedieNLogLik(Tweedie::default())
        );
        assert_eq!(
            EvalMetric::from_xgboost("ndcg@05", &source).unwrap().name(),
            "ndcg@5"
        );
        assert!(matches!(
            EvalMetric::from_xgboost("nope", &source),
            Err(HessboostError::Unknown { .. })
        ));
    }

    /// The multiclass metrics read one probability per model output, so
    /// they refuse a single-output model.
    #[test]
    fn multiclass_metrics_need_several_outputs() {
        for metric in [EvalMetric::MLogLoss, EvalMetric::MError] {
            assert!(metric.build(1).is_err());
            let built = metric.build(4).unwrap();
            let info = MetaInfo::new(&[0.0], None, None);
            assert_eq!(built.prediction_width(&info), Some(4));
        }
    }

    /// The class-index metrics read `labels` as classes of the model's
    /// outputs: anything else (out of range, negative, fractional, NaN) is
    /// refused by `validate_info` and evaluates to NaN, never indexing past
    /// the row.
    #[test]
    fn class_index_metrics_refuse_labels_outside_the_classes() {
        for metric in [EvalMetric::MLogLoss, EvalMetric::MError] {
            let built = metric.build(3).unwrap();
            for label in [3.0, 10.0, -1.0, 0.5, f32::NAN] {
                let labels = [0.0, label];
                let info = MetaInfo::new(&labels, None, None);
                assert!(built.validate_info(&info).is_err(), "{metric:?} {label}");
                let preds = [0.2, 0.3, 0.5, 0.2, 0.3, 0.5];
                assert!(
                    built.eval(&preds, &labels, None).is_nan(),
                    "{metric:?} {label}"
                );
            }
            let info = MetaInfo::new(&[0.0, 2.0], None, None);
            assert!(built.validate_info(&info).is_ok(), "{metric:?}");
        }
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

    /// The strided row weights of a label matrix give exactly the value of
    /// the metric over the materialized cell weights.
    #[test]
    fn label_matrix_row_weights_match_repeated_cell_weights_bit_for_bit() {
        let (n_rows, n_targets) = (1_367, 3);
        let cells = n_rows * n_targets;
        let preds: Vec<f32> = (0..cells).map(|i| 0.05 + (i % 97) as f32 * 0.009).collect();
        let labels: Vec<f32> = (0..cells)
            .map(|i| ((i * 7) % 5) as f32 * 0.25 + 0.125)
            .collect();
        let weights: Vec<f32> = (0..n_rows).map(|i| 0.5 + (i % 13) as f32 * 0.125).collect();
        let info = MetaInfo {
            n_rows,
            n_targets,
            ..MetaInfo::new(&labels, Some(&weights), None)
        };
        let cell_weights = info.cell_weights().unwrap().unwrap();
        let metrics: [Box<dyn Metric>; 10] = [
            Box::new(Rmse),
            Box::new(Mae),
            Box::new(LogLoss),
            Box::new(ErrorRate),
            Box::new(PoissonNLogLik),
            Box::new(GammaNLogLik),
            Box::new(TweedieNLogLik::new(1.5)),
            Box::new(Rmsle),
            Box::new(Mape),
            Box::new(PseudoHuberError::new(1.0)),
        ];
        for metric in metrics {
            let strided = metric.eval_info(&preds, &info);
            let repeated = metric.eval(&preds, &labels, Some(&cell_weights));
            assert!(strided.is_finite(), "{}", metric.name());
            assert_eq!(strided.to_bits(), repeated.to_bits(), "{}", metric.name());
        }
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
            let metric = named(name, 3, &DEFAULT_SOURCE).unwrap();
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
            let err = named(name, 1, &DEFAULT_SOURCE).err();
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
            assert!(named(name, 1, &DEFAULT_SOURCE).is_ok(), "{name}");
        }
        let m = named("pre@2", 1, &DEFAULT_SOURCE).unwrap();
        // All labels zero: no hits at any cutoff.
        assert_eq!(m.eval(&[0.9, 0.5, 0.1], &[0.0, 0.0, 0.0], None), 0.0);
    }

    /// Every metric declares the predictions per row it reads, and direct
    /// calls with inconsistent lengths evaluate to NaN instead of indexing
    /// out of bounds.
    #[test]
    fn mismatched_lengths_evaluate_to_nan() {
        let source = XgboostMetricSource {
            quantile_alpha: &[0.2, 0.8],
            expectile_alpha: &[0.5],
            distribution: Some(DistFamily::Normal),
            ..DEFAULT_SOURCE
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
            let metric = named(name, 3, &source).unwrap();
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
        let nloglik = AftNLogLik::new(crate::objective::AftDistribution::Normal, 1.0);
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

    /// Metrics report the names XGBoost 3.4.2 gives them in
    /// `evals_result`: the parsed `@k` cutoff or `@rho` power kept (so
    /// `ndcg@5` and `ndcg@10` stay apart), the power to six significant
    /// digits.
    #[test]
    fn metric_names_keep_their_suffix_as_xgboost_reports_it() {
        for (name, reported) in [
            ("ndcg", "ndcg"),
            ("ndcg@5", "ndcg@5"),
            ("ndcg@05", "ndcg@5"),
            ("map", "map"),
            ("map@10", "map@10"),
            ("pre", "pre"),
            ("pre@2", "pre@2"),
            ("tweedie-nloglik", "tweedie-nloglik@1.5"),
            ("tweedie-nloglik@1.30", "tweedie-nloglik@1.3"),
            ("tweedie-nloglik@1.0", "tweedie-nloglik@1"),
            ("tweedie-nloglik@1.23456789", "tweedie-nloglik@1.23457"),
        ] {
            let metric = named(name, 1, &DEFAULT_SOURCE).unwrap();
            assert_eq!(metric.name(), reported, "{name}");
        }
        assert_eq!(Ndcg::default().name(), "ndcg");
        assert_eq!(MeanAveragePrecision::new(Some(3)).name(), "map@3");
    }

    #[test]
    fn aucpr_perfect_and_ranks_better_than_random() {
        let m = AucPr;
        assert!(m.maximize());
        assert_eq!(named("aucpr", 1, &DEFAULT_SOURCE).unwrap().name(), "aucpr");

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
