//! Survival metrics: `cox-nloglik`, `aft-nloglik`, and
//! `interval-regression-accuracy` (XGBoost `metric/rank_metric.cc`,
//! `metric/survival_metric.cu`).

use super::Metric;
use crate::config::AftDistribution;
use crate::data::MetaInfo;
use crate::objective::{abs_label_order, aft_nloglik};

/// Negative log partial likelihood of the Cox model (`cox-nloglik`), per
/// observed event, with Breslow ties.
///
/// Receives hazard ratios `exp(margin)` (the `survival:cox` evaluation
/// transform) and the signed survival-time labels (negative = censored).
/// Censored rows enter the risk sets only. Weights are ignored, as in
/// XGBoost; a dataset without events yields a non-finite value.
#[derive(Debug, Clone, Copy, Default)]
pub struct CoxNLogLik;

impl Metric for CoxNLogLik {
    fn name(&self) -> &'static str {
        "cox-nloglik"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], _weights: Option<&[f32]>) -> f64 {
        let n = labels.len();
        let order = abs_label_order(labels);
        let mut exp_p_sum: f64 = preds[..n].iter().map(|&p| f64::from(p)).sum();
        let mut out = 0.0f64;
        let mut accumulated_sum = 0.0f64;
        let mut num_events = 0u64;
        for (i, &ind) in order.iter().enumerate() {
            let label = labels[ind];
            if label > 0.0 {
                // XGBoost takes the log of the `f32` prediction in `f32`.
                out -= f64::from(preds[ind].ln()) - exp_p_sum.ln();
                num_events += 1;
            }
            // The risk set drops a time's rows only after the last of them.
            accumulated_sum += f64::from(preds[ind]);
            if i == n - 1 || label.abs() < labels[order[i + 1]].abs() {
                exp_p_sum -= accumulated_sum;
                accumulated_sum = 0.0;
            }
        }
        out / num_events as f64
    }
}

/// Weighted mean of `row(lower, upper, margin)` over the label intervals,
/// XGBoost's survival-metric reduction (`esum / wsum`, or `esum` when the
/// total weight is zero). Without label bounds, the labels are used as
/// observed times (`lower == upper == label`).
fn interval_mean(preds: &[f32], info: &MetaInfo, row: impl Fn(f64, f64, f64) -> f64) -> f64 {
    let (lower, upper) = match (info.label_lower_bound, info.label_upper_bound) {
        (Some(lower), Some(upper)) => (lower, upper),
        _ => (info.labels, info.labels),
    };
    let mut residue_sum = 0.0f64;
    let mut weights_sum = 0.0f64;
    for (i, ((&lo, &hi), &pred)) in lower.iter().zip(upper).zip(preds).enumerate() {
        let w = info.weights.map_or(1.0, |w| f64::from(w[i]));
        residue_sum += row(f64::from(lo), f64::from(hi), f64::from(pred)) * w;
        weights_sum += w;
    }
    if weights_sum == 0.0 {
        residue_sum
    } else {
        residue_sum / weights_sum
    }
}

/// Negative log-likelihood of the accelerated failure time model
/// (`aft-nloglik`), weighted mean over rows.
///
/// Receives raw log-time margins (the `survival:aft` evaluation transform
/// is the identity) and the dataset's label bounds; the noise distribution
/// and scale come from the objective's parameters
/// (`aft_loss_distribution`, `aft_loss_distribution_scale`).
#[derive(Debug, Clone, Copy)]
pub struct AftNLogLik {
    distribution: AftDistribution,
    sigma: f32,
}

impl AftNLogLik {
    /// Create for the noise `distribution` with scale `sigma`.
    pub fn new(distribution: AftDistribution, sigma: f32) -> Self {
        AftNLogLik {
            distribution,
            sigma,
        }
    }
}

impl Metric for AftNLogLik {
    fn name(&self) -> &'static str {
        "aft-nloglik"
    }

    /// Treats each label as an observed time.
    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        self.eval_info(preds, &MetaInfo::new(labels, weights, None))
    }

    fn eval_info(&self, preds: &[f32], info: &MetaInfo) -> f64 {
        let sigma = f64::from(self.sigma);
        interval_mean(preds, info, |lo, hi, pred| {
            aft_nloglik(self.distribution, lo, hi, pred, sigma)
        })
    }
}

/// Fraction of rows whose predicted time `exp(margin)` lies inside the
/// label interval, endpoints included (`interval-regression-accuracy`),
/// weighted. Receives raw log-time margins. Higher is better.
#[derive(Debug, Clone, Copy, Default)]
pub struct IntervalRegressionAccuracy;

impl Metric for IntervalRegressionAccuracy {
    fn name(&self) -> &'static str {
        "interval-regression-accuracy"
    }

    fn maximize(&self) -> bool {
        true
    }

    /// Treats each label as an observed time.
    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        self.eval_info(preds, &MetaInfo::new(labels, weights, None))
    }

    fn eval_info(&self, preds: &[f32], info: &MetaInfo) -> f64 {
        interval_mean(preds, info, |lo, hi, log_pred| {
            let pred = log_pred.exp();
            if pred >= lo && pred <= hi { 1.0 } else { 0.0 }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Breslow partial likelihood by hand (the metric takes each log in
    /// `f32`, hence the tolerance): times 1 (event), 2 (event),
    /// 2 (censored), 3 (event) with hazards h. The tied time-2 event's risk
    /// set holds both time-2 rows and the time-3 row.
    #[test]
    fn cox_nloglik_matches_breslow_likelihood() {
        let labels = [2.0f32, 1.0, -2.0, 3.0];
        let h = [2.0f32, 1.0, 0.5, 4.0];
        let total = 7.5f64;
        let want = (-(1.0f64.ln() - total.ln())
            - (2.0f64.ln() - (total - 1.0).ln())
            - (4.0f64.ln() - 4.0f64.ln()))
            / 3.0;
        assert_relative_eq!(
            CoxNLogLik.eval(&h, &labels, None),
            want,
            max_relative = 1e-7
        );
    }

    #[test]
    fn interval_accuracy_counts_inclusive_hits_weighted() {
        let lower = [1.0f32, 0.0, 1.5, 5.0];
        let upper = [1.0f32, 3.0, f32::INFINITY, 6.0];
        // exp(margin): 1 (exactly the observed time), 2 (inside), 2 (inside
        // the right-censored interval), 1 (below).
        let margins = [0.0f32, 2.0f32.ln(), 2.0f32.ln(), 0.0];
        let weights = [1.0f32, 1.0, 1.0, 3.0];
        let info = MetaInfo {
            n_rows: 4,
            labels: &[],
            n_targets: 1,
            weights: Some(&weights),
            group: None,
            label_lower_bound: Some(&lower),
            label_upper_bound: Some(&upper),
        };
        let m = IntervalRegressionAccuracy;
        assert!(m.maximize());
        assert_relative_eq!(m.eval_info(&margins, &info), 3.0 / 6.0);
    }

    /// With all rows uncensored and normal noise, `aft-nloglik` is the
    /// log-normal negative log-density.
    #[test]
    fn aft_nloglik_uncensored_normal_is_lognormal_density() {
        let lower = [1.5f32, 0.7];
        let margins = [0.2f32, -0.1];
        let sigma = 0.5f64;
        let info = MetaInfo {
            n_rows: 2,
            labels: &[],
            n_targets: 1,
            weights: None,
            group: None,
            label_lower_bound: Some(&lower),
            label_upper_bound: Some(&lower),
        };
        let want: f64 = lower
            .iter()
            .zip(&margins)
            .map(|(&t, &m)| {
                let t = f64::from(t);
                let z = (t.ln() - f64::from(m)) / sigma;
                0.5 * z * z + (sigma * t * (2.0 * std::f64::consts::PI).sqrt()).ln()
            })
            .sum::<f64>()
            / 2.0;
        let m = AftNLogLik::new(AftDistribution::Normal, 0.5);
        assert_relative_eq!(m.eval_info(&margins, &info), want, max_relative = 1e-12);
        // The plain-label entry point treats labels as observed times.
        assert_relative_eq!(m.eval(&margins, &lower, None), want, max_relative = 1e-12);
    }

    /// XGBoost's default `aft-nloglik` keeps the objective's distribution but
    /// evaluates at scale 1; an explicit `eval_metric` uses the configured
    /// scale.
    #[test]
    fn default_aft_metric_uses_unit_scale() {
        use crate::config::ObjectiveParams;
        use crate::metric::create_metrics;

        let lower = [1.5f32, 0.0, 2.0];
        let upper = [1.5f32, 3.0, f32::INFINITY];
        let margins = [0.2f32, 0.5, 0.1];
        let info = MetaInfo {
            n_rows: 3,
            labels: &[],
            n_targets: 1,
            weights: None,
            group: None,
            label_lower_bound: Some(&lower),
            label_upper_bound: Some(&upper),
        };
        let params = ObjectiveParams {
            aft_loss_distribution: AftDistribution::Logistic,
            aft_loss_distribution_scale: 0.8,
            ..ObjectiveParams::defaults_for("survival:aft")
        };
        let default = &create_metrics(&[], "aft-nloglik", 0, &params).unwrap()[0];
        let explicit =
            &create_metrics(&["aft-nloglik".to_string()], "aft-nloglik", 0, &params).unwrap()[0];
        let unit = AftNLogLik::new(AftDistribution::Logistic, 1.0).eval_info(&margins, &info);
        let scaled = AftNLogLik::new(AftDistribution::Logistic, 0.8).eval_info(&margins, &info);
        assert_ne!(unit, scaled);
        assert_eq!(default.eval_info(&margins, &info), unit);
        assert_eq!(explicit.eval_info(&margins, &info), scaled);
    }
}
