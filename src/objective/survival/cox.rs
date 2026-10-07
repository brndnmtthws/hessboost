//! Cox proportional hazards (`survival:cox`, XGBoost
//! `objective/regression_obj.cu`).

use super::abs_label_order;
use crate::error::Result;
use crate::objective::{
    GradPair, Loss, OutputDomain, check_base_score_domain, inverse_log_link, log_link,
};

/// Cox proportional-hazards regression (`survival:cox`) on right-censored
/// survival times.
///
/// A positive label is an observed event time; a negative label (or zero) is
/// a right-censoring time `|y|`. The margin is the log hazard ratio and
/// predictions are hazard ratios `exp(margin)`. Gradients are the Breslow
/// partial-likelihood derivatives over risk sets ordered by `|y|`: rows with
/// tied times share one risk-set denominator. The intercept is XGBoost's
/// one-Newton-step fit from zero margins.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct Cox;

impl Loss for Cox {
    fn name(&self) -> &'static str {
        "survival:cox"
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        crate::objective::check_gradient_inputs(labels.len(), 1, preds, labels, weights, out);
        let order = abs_label_order(labels);
        // The risk-set total uses `f32` exponentials accumulated in `f64`;
        // the per-row terms below exponentiate in `f64`, as upstream does.
        let mut exp_p_sum: f64 = order.iter().map(|&i| f64::from(preds[i].exp())).sum();
        let mut r_k = 0.0f64;
        let mut s_k = 0.0f64;
        let mut last_exp_p = 0.0f64;
        let mut last_abs_y = 0.0f64;
        let mut accumulated_sum = 0.0f64;
        for &ind in &order {
            let exp_p = f64::from(preds[ind]).exp();
            let w = weights.map_or(1.0, |w| f64::from(w[ind]));
            let y = f64::from(labels[ind]);
            let abs_y = y.abs();

            // Breslow ties: the denominator drops the previous time's rows
            // only once time moves forward.
            accumulated_sum += last_exp_p;
            if last_abs_y < abs_y {
                exp_p_sum -= accumulated_sum;
                accumulated_sum = 0.0;
            }
            let event = y > 0.0;
            if event {
                r_k += 1.0 / exp_p_sum;
                s_k += 1.0 / (exp_p_sum * exp_p_sum);
            }
            let grad = exp_p * r_k - if event { 1.0 } else { 0.0 };
            let hess = exp_p * r_k - exp_p * exp_p * s_k;
            out[ind] = GradPair::new((grad * w) as f32, (hess * w) as f32);

            last_abs_y = abs_y;
            last_exp_p = exp_p;
        }
    }

    fn pred_transform(&self, preds: &mut [f32]) {
        inverse_log_link(preds);
    }

    fn probs_to_margins(&self, scores: &mut [f32]) {
        log_link(scores);
    }

    fn validate_base_score(&self, base_score: f64) -> Result<()> {
        check_base_score_domain(base_score, OutputDomain::Positive)
    }

    fn default_metric(&self) -> crate::metric::EvalMetric {
        crate::metric::EvalMetric::CoxNLogLik
    }
}
