//! Curve metrics for binary labels: ROC AUC (`auc`) and the area under
//! the precision-recall curve (`aucpr`), macro-averaged over the columns
//! of a label matrix.

use super::{Metric, argsort_desc, stable_argsort};
use crate::data::MetaInfo;

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
