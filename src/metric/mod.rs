//! Evaluation metrics used for reporting and early stopping.
//!
//! Metrics receive predictions that have already passed through the objective's
//! [`crate::objective::Objective::pred_transform`] (so classification metrics
//! see probabilities), matching XGBoost's evaluation pipeline.

use crate::error::{HessboostError, Result};

/// An evaluation metric over predictions and labels.
pub trait Metric: Send + Sync {
    /// The XGBoost-compatible metric name (e.g. `"rmse"`, `"logloss"`).
    fn name(&self) -> &str;

    /// Whether a *larger* value is better (e.g. AUC). Drives early stopping.
    fn maximize(&self) -> bool {
        false
    }

    /// Evaluate the metric. `preds` are post-transform predictions.
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
}

/// Normalize a metric total, returning zero for an empty or nonpositive weight sum.
#[inline]
fn weighted_mean((total, weight): (f64, f64)) -> f64 {
    if weight > 0.0 { total / weight } else { 0.0 }
}

/// Define a purely-pointwise metric from its SIMD weighted-sum kernel.
/// Generates the metric struct plus its [`Metric`] impl from the metric name
/// and the `crate::simd` kernel path; `eval` is
/// `weighted_mean(kernel(preds, labels, weights))`. Metrics with metric-level
/// state take a `field: Type` arm and pass `self.field` as the kernel's final
/// argument; `rmse` takes `=> sqrt` for its root. All generated metrics
/// minimize (`maximize` keeps its default `false`); metrics with non-trivial
/// logic (`auc`, `aucpr`, ranking) stay handwritten below.
macro_rules! simple_metric {
    ($(#[$m:meta])* $ty:ident, $name:literal, $simd:path) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $ty;
        impl Metric for $ty {
            fn name(&self) -> &str {
                $name
            }
            fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
                weighted_mean($simd(preds, labels, weights))
            }
        }
    };
    ($(#[$m:meta])* $ty:ident, $name:literal, $simd:path => sqrt) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $ty;
        impl Metric for $ty {
            fn name(&self) -> &str {
                $name
            }
            fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
                weighted_mean($simd(preds, labels, weights)).sqrt()
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
                weighted_mean($simd(preds, labels, weights, self.$field))
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
    /// Binary logistic loss (`logloss`). Predictions are probabilities.
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

/// Binary ROC AUC (`auc`), computed with the Mann-Whitney rank-sum and average
/// ranks for ties. Higher is better. Weights are ignored (unweighted AUC).
#[derive(Debug, Clone, Copy, Default)]
pub struct Auc;

impl Metric for Auc {
    fn name(&self) -> &'static str {
        "auc"
    }

    fn maximize(&self) -> bool {
        true
    }

    fn eval(&self, preds: &[f32], labels: &[f32], _weights: Option<&[f32]>) -> f64 {
        let n = preds.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| preds[a].total_cmp(&preds[b]));

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
}

simple_metric!(
    /// Multiclass log loss (`mlogloss`). Predictions are `n × num_class`
    /// probabilities. Labels are class indices.
    MLogLoss, "mlogloss", num_class: usize, crate::simd::multiclass_log_loss_sum
);

simple_metric!(
    /// Multiclass error rate (`merror`): fraction whose argmax ≠ label.
    MError, "merror", num_class: usize, crate::simd::multiclass_error_sum
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

/// Iterate `(start, end)` row ranges for a group, or a single whole-batch
/// range when no group info is present. Shared by the ranking metrics and the
/// LambdaMART objective's usable-group fallback.
pub(crate) fn group_ranges(
    n: usize,
    group: Option<&crate::data::GroupInfo>,
) -> Vec<(usize, usize)> {
    match group {
        Some(g) if g.num_rows() == n => g.iter_ranges().collect(),
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
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or_else(|| values[b].total_cmp(&values[a]))
    });
    order
}

/// Weighted mean of a per-group `score` over query-group ranges, weighted by
/// each group's first document weight (`1.0` when unweighted). Shared by the
/// ranking metrics' `eval_grouped`.
fn grouped_average(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    group: Option<&crate::data::GroupInfo>,
    mut score: impl FnMut(&[f32], &[f32]) -> f64,
) -> f64 {
    let ranges = group_ranges(preds.len(), group);
    if ranges.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut weight_sum = 0.0;
    for &(start, end) in &ranges {
        let weight = weights.map_or(1.0, |values| f64::from(values[start]));
        sum += weight * score(&preds[start..end], &labels[start..end]);
        weight_sum += weight;
    }
    weighted_mean((sum, weight_sum))
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

        let labels_f64: Vec<f64> = labels.iter().map(|&l| f64::from(l)).collect();
        let idcg = ideal_dcg(&labels_f64, cut);

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
        grouped_average(preds, labels, weights, group, |p, l| self.group_ndcg(p, l))
    }
}

/// NDCG gain of a relevance label: `2^rel - 1`. Shared with the LambdaMART
/// objective's `|ΔNDCG|` weighting.
#[inline]
pub(crate) fn ndcg_gain(rel: f64) -> f64 {
    (2.0f64).powf(rel) - 1.0
}

/// NDCG position discount for 0-based rank `p`: `1 / log2(p + 2)`. Shared with
/// the LambdaMART objective's `|ΔNDCG|` weighting.
#[inline]
pub(crate) fn ndcg_discount(p: usize) -> f64 {
    1.0 / ((p + 2) as f64).log2()
}

/// Ideal DCG of a group: labels sorted by descending relevance, gains
/// accumulated with the standard discount, truncated at `cut` ranks. Shared by
/// the `ndcg` metric and the LambdaMART objective's `|ΔNDCG|` weighting.
pub(crate) fn ideal_dcg(labels: &[f64], cut: usize) -> f64 {
    let mut ideal: Vec<f64> = labels.to_vec();
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
        grouped_average(preds, labels, weights, group, |p, l| self.group_ap(p, l))
    }
}

/// Area under the precision-recall curve for binary classification (`aucpr`).
///
/// Labels are `{0, 1}` and predictions are probabilities. The curve is traced by
/// sorting instances by descending prediction and sweeping the decision
/// threshold. The area is integrated over recall with the trapezoidal rule
/// (tied scores form a single operating point). Higher is better. A degenerate
/// problem (no positives or no negatives) yields `0`.
#[derive(Debug, Clone, Copy, Default)]
pub struct AucPr;

impl Metric for AucPr {
    fn name(&self) -> &'static str {
        "aucpr"
    }

    fn maximize(&self) -> bool {
        true
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
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
}

/// Closure type backing a [`CustomMetric`].
type MetricFn = dyn Fn(&[f32], &[f32], Option<&[f32]>) -> f64 + Send + Sync;

/// A [`Metric`] backed by a user-supplied closure (the custom-metric hook).
///
/// The closure receives post-transform predictions, labels, and optional
/// weights, and returns the scalar metric value. `maximize` declares the
/// optimization direction used for early stopping.
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

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        (self.f)(preds, labels, weights)
    }
}

/// Resolve a metric by name. `num_class` is used by multiclass metrics.
pub fn create_metric(name: &str, num_class: usize) -> Result<Box<dyn Metric>> {
    // Accept the XGBoost `tweedie-nloglik@1.5` suffix form.
    let (base, rho) = match name.split_once('@') {
        Some((b, r)) => (b, r.parse::<f64>().ok()),
        None => (name, None),
    };
    match base {
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
            rho: rho.unwrap_or(1.5),
        })),
        "ndcg" => Ok(Box::new(Ndcg::new(rho.map(|r| r as usize)))),
        "map" => Ok(Box::new(MeanAveragePrecision::new(rho.map(|r| r as usize)))),
        other => Err(HessboostError::unknown("metric", other)),
    }
}

/// Build the list of metrics to evaluate: the user's `eval_metric` list if any,
/// otherwise the single `default_name` supplied by the objective.
pub fn create_metrics(
    eval_metric: &[String],
    default_name: &str,
    num_class: usize,
) -> Result<Vec<Box<dyn Metric>>> {
    if eval_metric.is_empty() {
        Ok(vec![create_metric(default_name, num_class)?])
    } else {
        eval_metric
            .iter()
            .map(|n| create_metric(n, num_class))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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
        let ms = create_metrics(&[], "rmse", 0).unwrap();
        assert_eq!(ms.len(), 1);
        assert_eq!(ms[0].name(), "rmse");
        assert!(create_metrics(&["nope".to_string()], "rmse", 0).is_err());
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
        assert_eq!(create_metric("ndcg", 0).unwrap().name(), "ndcg");
        assert_eq!(create_metric("map", 0).unwrap().name(), "map");
        // `@k` suffix parses without error.
        assert_eq!(create_metric("ndcg@5", 0).unwrap().name(), "ndcg");
        assert_eq!(create_metric("map@10", 0).unwrap().name(), "map");
    }

    #[test]
    fn aucpr_perfect_and_ranks_better_than_random() {
        let m = AucPr;
        assert!(m.maximize());
        assert_eq!(create_metric("aucpr", 0).unwrap().name(), "aucpr");

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
