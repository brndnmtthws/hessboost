//! Classification objectives.

use super::{newton_intercepts, weighted_label_mean, GradPair, Objective, MIN_HESS};

/// Logistic regression: `binary:logistic` (classification, reported with
/// `logloss`) or `reg:logistic` (probability regression, reported with `rmse`
/// like XGBoost's `LogisticRegression`); the loss is identical.
///
/// With `p = σ(margin)` the gradient is `p − label` and the Hessian is
/// `max(p (1 − p), ε)`. `scale_pos_weight` rescales the loss of positive
/// instances to combat class imbalance, exactly as in XGBoost.
#[derive(Debug, Clone, Copy)]
pub struct LogisticObjective {
    scale_pos_weight: f32,
    /// `reg:logistic` rather than `binary:logistic`.
    regression: bool,
}

impl LogisticObjective {
    /// `binary:logistic` with the given positive-class weight.
    pub fn new(scale_pos_weight: f32) -> Self {
        LogisticObjective {
            scale_pos_weight,
            regression: false,
        }
    }

    /// `reg:logistic` with the given positive-class weight: the same loss,
    /// named and evaluated (`rmse`) as XGBoost's probability regression.
    pub fn regression(scale_pos_weight: f32) -> Self {
        LogisticObjective {
            scale_pos_weight,
            regression: true,
        }
    }
}

impl Default for LogisticObjective {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Objective for LogisticObjective {
    fn name(&self) -> &str {
        if self.regression {
            "reg:logistic"
        } else {
            "binary:logistic"
        }
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        super::check_gradient_inputs(labels.len(), 1, preds, labels, weights, out);
        let (scale_pos_weight, min_hess) = (self.scale_pos_weight, MIN_HESS);
        super::rowwise_gradient(
            labels.len(),
            1,
            preds,
            labels,
            weights,
            out,
            |preds, labels, weights, out| {
                crate::simd::logistic_gradient(
                    preds,
                    labels,
                    weights,
                    scale_pos_weight,
                    min_hess,
                    out,
                )
            },
        );
    }

    fn pred_transform(&self, preds: &mut [f32]) {
        crate::simd::sigmoid_inplace(preds);
    }

    fn base_margins(
        &self,
        labels: &[f32],
        weights: Option<&[f32]>,
        group: Option<&crate::data::GroupInfo>,
    ) -> Vec<f32> {
        // XGBoost `RegLossObj::InitEstimation`: the (weighted) positive rate
        // through the logit, unless `scale_pos_weight` is in play, in which
        // case the reweighted loss needs the Newton step.
        if (self.scale_pos_weight - 1.0).abs() > 1e-6 {
            return newton_intercepts(self, labels, weights, group);
        }
        vec![self.prob_to_margin(weighted_label_mean(labels, weights))]
    }

    fn prob_to_margin(&self, base_score: f32) -> f32 {
        // XGBoost `LogisticRegression::ProbToMargin`: bound the probability
        // away from the asymptotes, then `Logit(p) = -ln(1/p - 1)` in f32.
        let p = base_score.clamp(1e-6, 1.0 - 1e-6);
        -(1.0 / p - 1.0).ln()
    }

    fn default_metric(&self) -> String {
        if self.regression { "rmse" } else { "logloss" }.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sigmoid_symmetry() {
        let mut values = [0.0, 2.0, -2.0, 80.0, -80.0];
        LogisticObjective::default().pred_transform(&mut values);
        assert_relative_eq!(values[0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(values[1] + values[2], 1.0, epsilon = 1e-6);
        // Extreme values do not overflow.
        assert!(values[3] <= 1.0 && values[3] > 0.999);
        assert!(values[4] >= 0.0 && values[4] < 0.001);
    }

    #[test]
    fn gradient_matches_closed_form() {
        let obj = LogisticObjective::default();
        // margin 0 -> p = 0.5
        let preds = [0.0f32];
        let labels = [1.0f32];
        let mut out = vec![GradPair::default(); 1];
        obj.gradient(&preds, &labels, None, &mut out);
        assert_relative_eq!(out[0].grad, -0.5, epsilon = 1e-6); // 0.5 - 1
        assert_relative_eq!(out[0].hess, 0.25, epsilon = 1e-6); // 0.5 * 0.5
    }

    #[test]
    fn scale_pos_weight_scales_positive() {
        let obj = LogisticObjective::new(3.0);
        let preds = [0.0f32, 0.0];
        let labels = [1.0f32, 0.0];
        let mut out = vec![GradPair::default(); 2];
        obj.gradient(&preds, &labels, None, &mut out);
        // positive instance gradient/hess scaled by 3
        assert_relative_eq!(out[0].grad, -1.5, epsilon = 1e-6);
        assert_relative_eq!(out[0].hess, 0.75, epsilon = 1e-6);
        // negative instance unaffected
        assert_relative_eq!(out[1].grad, 0.5, epsilon = 1e-6);
        assert_relative_eq!(out[1].hess, 0.25, epsilon = 1e-6);
    }

    #[test]
    fn base_margins_is_logit_of_rate() {
        let obj = LogisticObjective::default();
        // 50% positive -> logit(0.5) = 0; 25% -> -ln(3) with XGBoost's f32 logit.
        assert_eq!(obj.base_margins(&[1.0, 0.0], None, None), vec![0.0]);
        let quarter = obj.base_margins(&[1.0, 0.0, 0.0, 0.0], None, None);
        assert_eq!(quarter, vec![-(1.0f32 / 0.25 - 1.0).ln()]);
    }

    #[test]
    fn prob_to_margin_clamps_to_xgboost_bounds() {
        let obj = LogisticObjective::default();
        assert_eq!(obj.prob_to_margin(0.0), obj.prob_to_margin(1e-6));
        assert_eq!(obj.prob_to_margin(1.0), obj.prob_to_margin(1.0 - 1e-6));
        assert!(obj.prob_to_margin(0.0).is_finite());
    }
}
