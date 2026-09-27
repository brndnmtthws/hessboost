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

mod curve;
mod distributional;
mod elementwise;
mod factory;
mod quantile;
mod ranking;
mod survival;

pub(crate) use factory::XgboostMetricSource;
pub use factory::{Cutoff, EvalMetric};
#[cfg(test)]
pub(crate) use factory::{DEFAULT_SOURCE, named};
#[cfg(test)]
pub(crate) use ranking::Ndcg;

use crate::data::MetaInfo;
use crate::error::{HessboostError, Result};
use crate::simd::RowWeights;
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
    /// For a label matrix (`info.n_targets() > 1`) the default is XGBoost's
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
        if info.n_targets() > 1 {
            let Ok(cell_weights) = info.cell_weights() else {
                return f64::NAN;
            };
            return self.eval_grouped(preds, info.label_values(), cell_weights.as_deref(), None);
        }
        self.eval_grouped(preds, info.label_values(), info.weights, info.group)
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
        Some(info.n_targets())
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
    let stride = info.n_targets();
    if info.check_layout().is_err()
        || (stride > 1 && info.n_rows.checked_mul(stride) != Some(info.label_values().len()))
    {
        return f64::NAN;
    }
    let weights = info.weights.map(|w| RowWeights::new(w, stride));
    metric.eval_cells(preds, info.label_values(), weights)
}

/// Normalize a metric total, returning zero for an empty or nonpositive weight sum.
#[inline]
fn weighted_mean((total, weight): (f64, f64)) -> f64 {
    if weight > 0.0 { total / weight } else { 0.0 }
}

/// [`Metric::validate_info`]'s default: the metric named `name` reads
/// ordinary labels, which a dataset with rows must carry.
fn require_labels(name: &str, info: &MetaInfo) -> Result<()> {
    if info.n_rows > 0 && info.label_values().is_empty() {
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
                match first_non_class(info.label_values(), self.$field) {
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
pub(super) fn tweedie_name(rho: f64) -> String {
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
    argsort_desc_by(values.len(), |i| values[i])
}

/// [`argsort_desc`] of the `n` values `value(0..n)`, read in place (the
/// curve metrics' strided label-matrix columns).
pub(crate) fn argsort_desc_by(n: usize, value: impl Fn(usize) -> f32 + Sync) -> Vec<usize> {
    stable_argsort(n, |&a, &b| {
        let (a, b) = (value(a), value(b));
        b.partial_cmp(&a).unwrap_or_else(|| b.total_cmp(&a))
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

#[cfg(test)]
mod tests {
    use super::curve::{Auc, AucPr};
    use super::elementwise::{Mape, PseudoHuberError, Rmsle};
    use super::ranking::{MeanAveragePrecision, Precision};
    use super::survival::{AftNLogLik, CoxNLogLik, IntervalRegressionAccuracy};
    use super::*;
    use crate::objective::distributional::DistFamily;
    use crate::objective::{AftDistribution, Tweedie};
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
                labels: Some(crate::data::Labels::new(
                    &labels,
                    std::num::NonZeroUsize::new(3).unwrap(),
                )),
                ..MetaInfo::new(&labels, Some(&weights), None)
            };
            let bounds = MetaInfo {
                n_rows: n,
                bounds: Some(crate::data::LabelBounds::new(&lower, &upper)),
                weights: Some(&weights),
                ..MetaInfo::unlabeled(0)
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
            labels: Some(crate::data::Labels::new(
                &labels,
                std::num::NonZeroUsize::new(3).unwrap(),
            )),
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
            labels: Some(crate::data::Labels::new(
                &labels,
                std::num::NonZeroUsize::new(n_targets).unwrap(),
            )),
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
            labels: Some(crate::data::Labels::new(
                &labels,
                std::num::NonZeroUsize::new(2).unwrap(),
            )),
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

    /// The strided per-target columns of a tied, weighted label matrix
    /// score exactly like the copied-out columns.
    #[test]
    fn curve_metric_label_columns_match_copied_columns_bit_for_bit() {
        let (n_rows, k) = (403, 3);
        // Coarse predictions tie within and across columns.
        let preds: Vec<f32> = (0..n_rows * k)
            .map(|i| ((i * 37) % 11) as f32 * 0.1)
            .collect();
        let labels: Vec<f32> = (0..n_rows * k)
            .map(|i| f32::from(u8::from((i + i / 5) % 3 == 0)))
            .collect();
        let weights: Vec<f32> = (0..n_rows).map(|i| 0.25 + (i % 7) as f32 * 0.5).collect();
        let info = MetaInfo {
            n_rows,
            labels: Some(crate::data::Labels::new(
                &labels,
                std::num::NonZeroUsize::new(k).unwrap(),
            )),
            ..MetaInfo::new(&labels, Some(&weights), None)
        };
        let col = |v: &[f32], t: usize| v.iter().skip(t).step_by(k).copied().collect::<Vec<_>>();
        for metric in [&Auc as &dyn Metric, &AucPr] {
            let copied: f64 = (0..k)
                .map(|t| metric.eval(&col(&preds, t), &col(&labels, t), Some(&weights)))
                .sum::<f64>()
                / k as f64;
            let strided = metric.eval_info(&preds, &info);
            assert!(strided > 0.0 && strided < 1.0, "{}", metric.name());
            assert_eq!(strided.to_bits(), copied.to_bits(), "{}", metric.name());
        }
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
            labels: Some(crate::data::Labels::new(
                &labels,
                std::num::NonZeroUsize::new(2).unwrap(),
            )),
            ..MetaInfo::new(&labels, Some(&weights), None)
        };
        assert_eq!(Rmse.eval_info(&labels, &info), 0.0);
        for n_targets in [usize::MAX, 1, 3] {
            let k = std::num::NonZeroUsize::new(n_targets).unwrap();
            let bad = MetaInfo {
                labels: Some(crate::data::Labels::new(&labels, k)),
                ..info
            };
            assert!(Rmse.eval_info(&labels, &bad).is_nan(), "{n_targets}");
        }
        let unlabeled = MetaInfo {
            weights: Some(&[1.0]),
            ..MetaInfo::unlabeled(1)
        };
        assert!(Rmse.eval_info(&[], &unlabeled).is_nan());
        // Bounds-only metadata (no labels) stays valid where it is read.
        let bounds = [1.0f32];
        let aft = MetaInfo {
            bounds: Some(crate::data::LabelBounds::new(&bounds, &bounds)),
            weights: Some(&[1.0]),
            ..MetaInfo::unlabeled(0)
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
