//! Classification objectives.

use super::{weighted_label_mean, GradPair, Objective};

/// Lower bound on the logistic Hessian, matching XGBoost's `kRtEps` guard so
/// that confidently-classified instances still contribute a positive Hessian.
const MIN_HESS: f32 = 1e-16;

/// Binary logistic regression (`binary:logistic`).
///
/// With `p = σ(margin)` the gradient is `p − label` and the Hessian is
/// `max(p (1 − p), ε)`. `scale_pos_weight` rescales the loss of positive
/// instances to combat class imbalance, exactly as in XGBoost.
#[derive(Debug, Clone, Copy)]
pub struct LogisticObjective {
    scale_pos_weight: f32,
}

impl LogisticObjective {
    /// Create a logistic objective with the given positive-class weight.
    pub fn new(scale_pos_weight: f32) -> Self {
        LogisticObjective { scale_pos_weight }
    }
}

impl Default for LogisticObjective {
    fn default() -> Self {
        LogisticObjective {
            scale_pos_weight: 1.0,
        }
    }
}

impl Objective for LogisticObjective {
    fn name(&self) -> &str {
        "binary:logistic"
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

    fn base_margin(&self, labels: &[f32], weights: Option<&[f32]>) -> f32 {
        // Optimal constant probability is the (weighted) positive rate; the
        // margin is its logit, clamped away from the asymptotes.
        let mut p = weighted_label_mean(labels, weights);
        p = p.clamp(1e-6, 1.0 - 1e-6);
        (p / (1.0 - p)).ln() as f32
    }

    fn prob_to_margin(&self, base_score: f32) -> f32 {
        let p = base_score.clamp(1e-6, 1.0 - 1e-6);
        (p / (1.0 - p)).ln()
    }

    fn default_metric(&self) -> &str {
        "logloss"
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
    fn base_margin_is_logit_of_rate() {
        let obj = LogisticObjective::default();
        // 50% positive -> logit(0.5) = 0
        assert_relative_eq!(obj.base_margin(&[1.0, 0.0], None), 0.0, epsilon = 1e-6);
    }
}
