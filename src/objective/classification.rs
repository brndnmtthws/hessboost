//! Classification objectives.

use super::{
    GradPair, MIN_HESS, Objective, check_label_domain, newton_intercepts, weighted_label_mean,
};
use crate::data::MetaInfo;
use crate::error::Result;

/// Logistic loss: `binary:logistic` (classification, reported with
/// `logloss`), `reg:logistic` (probability regression, reported with `rmse`
/// like XGBoost's `LogisticRegression`), or `binary:logitraw` (reports the
/// raw margin, evaluated with `logloss` on it); the loss is identical.
///
/// With `p = σ(margin)` the gradient is `p − label` and the Hessian is
/// `max(p (1 − p), ε)`. `scale_pos_weight` rescales the loss of positive
/// instances to combat class imbalance, exactly as in XGBoost.
#[derive(Debug, Clone, Copy)]
pub struct LogisticObjective {
    scale_pos_weight: f32,
    variant: LogisticVariant,
}

/// Which XGBoost objective a [`LogisticObjective`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LogisticVariant {
    /// `binary:logistic`.
    Binary,
    /// `reg:logistic`.
    Regression,
    /// `binary:logitraw`.
    Raw,
}

impl LogisticObjective {
    /// `binary:logistic` with the given positive-class weight.
    pub fn new(scale_pos_weight: f32) -> Self {
        LogisticObjective {
            scale_pos_weight,
            variant: LogisticVariant::Binary,
        }
    }

    /// `reg:logistic` with the given positive-class weight: the same loss,
    /// named and evaluated (`rmse`) as XGBoost's probability regression.
    pub fn regression(scale_pos_weight: f32) -> Self {
        LogisticObjective {
            scale_pos_weight,
            variant: LogisticVariant::Regression,
        }
    }

    /// `binary:logitraw` with the given positive-class weight: the same loss,
    /// but predictions (and the stored `base_score`) stay raw margins, as in
    /// XGBoost's `LogisticRaw`. Its unweighted-positive intercept is the
    /// plain label mean, taken as a margin.
    pub fn raw(scale_pos_weight: f32) -> Self {
        LogisticObjective {
            scale_pos_weight,
            variant: LogisticVariant::Raw,
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
        match self.variant {
            LogisticVariant::Binary => "binary:logistic",
            LogisticVariant::Regression => "reg:logistic",
            LogisticVariant::Raw => "binary:logitraw",
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
                );
            },
        );
    }

    fn pred_transform(&self, preds: &mut [f32]) {
        if self.variant != LogisticVariant::Raw {
            crate::simd::sigmoid_inplace(preds);
        }
    }

    fn base_margins(
        &self,
        labels: &[f32],
        weights: Option<&[f32]>,
        group: Option<&crate::data::GroupInfo>,
    ) -> Vec<f32> {
        // XGBoost `RegLossObj::InitEstimation`: the (weighted) positive rate
        // through the link (the logit; the identity for `binary:logitraw`),
        // unless `scale_pos_weight` is in play, in which case the reweighted
        // loss needs the Newton step.
        if (self.scale_pos_weight - 1.0).abs() > 1e-6 {
            return newton_intercepts(self, &MetaInfo::new(labels, weights, group));
        }
        vec![self.prob_to_margin(weighted_label_mean(labels, weights))]
    }

    fn prob_to_margin(&self, base_score: f32) -> f32 {
        if self.variant == LogisticVariant::Raw {
            // `LogisticRaw::ProbToMargin` is the identity.
            return base_score;
        }
        // XGBoost `LogisticRegression::ProbToMargin`: bound the probability
        // away from the asymptotes, then `Logit(p) = -ln(1/p - 1)` in f32.
        let p = base_score.clamp(1e-6, 1.0 - 1e-6);
        -(1.0 / p - 1.0).ln()
    }

    fn validate_info(&self, info: &MetaInfo) -> Result<()> {
        // XGBoost `LogisticRegression::CheckLabel` (shared by all three
        // variants): probabilities in [0, 1], not only {0, 1}.
        check_label_domain(info, |y| !(0.0..=1.0).contains(&y))
    }

    fn default_metric(&self) -> String {
        match self.variant {
            LogisticVariant::Regression => "rmse",
            LogisticVariant::Binary | LogisticVariant::Raw => "logloss",
        }
        .to_string()
    }
}

/// Hinge loss for binary classification (`binary:hinge`), as XGBoost's
/// `HingeObj`. With `z = 2y − 1` (computed in `f64`), a margin `m` with
/// `m·z < 1` gets gradient `−z·w` and Hessian `w`; otherwise the gradient is
/// `0` and the Hessian the smallest positive normal `f32` (unweighted).
/// Predictions are `1` when the margin is positive and `0` otherwise; the
/// intercept is the trait's default Newton step passed through that
/// threshold (XGBoost `FitIntercept`), so it is `0` or `1`. Labels are not
/// validated (upstream expects `{0, 1}` but does not check).
#[derive(Debug, Clone, Copy, Default)]
pub struct HingeObjective;

impl Objective for HingeObjective {
    fn name(&self) -> &'static str {
        "binary:hinge"
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        super::check_gradient_inputs(labels.len(), 1, preds, labels, weights, out);
        super::rowwise_gradient(
            labels.len(),
            1,
            preds,
            labels,
            weights,
            out,
            |preds, labels, weights, out| {
                for i in 0..preds.len() {
                    let w = weights.map_or(1.0, |ws| ws[i]);
                    let z = f64::from(labels[i]) * 2.0 - 1.0;
                    out[i] = if f64::from(preds[i]) * z < 1.0 {
                        GradPair::new((-z * f64::from(w)) as f32, w)
                    } else {
                        GradPair::new(0.0, f32::MIN_POSITIVE)
                    };
                }
            },
        );
    }

    fn pred_transform(&self, preds: &mut [f32]) {
        for p in preds {
            *p = if *p > 0.0 { 1.0 } else { 0.0 };
        }
    }

    fn margins_to_probs(&self, _margins: &mut [f32]) {
        // XGBoost's hinge `ProbToMargin` is the identity: its stored
        // `base_score` is the margin itself, not the thresholded prediction.
    }

    fn default_metric(&self) -> String {
        "error".to_string()
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

    /// `binary:logitraw` shares the logistic gradient but keeps margins raw:
    /// the identity transform, and an intercept that is the label mean itself
    /// (XGBoost stores the mean as the margin, not its logit).
    #[test]
    fn logitraw_keeps_margins_and_uses_mean_intercept() {
        let raw = LogisticObjective::raw(1.0);
        let mut values = [-3.0f32, 0.5];
        raw.pred_transform(&mut values);
        assert_eq!(values, [-3.0, 0.5]);
        assert_eq!(
            raw.base_margins(&[1.0, 0.0, 0.0, 0.0], None, None),
            vec![0.25]
        );
        let (mut a, mut b) = (vec![GradPair::default(); 2], vec![GradPair::default(); 2]);
        raw.gradient(&[0.3, -1.2], &[1.0, 0.0], None, &mut a);
        LogisticObjective::new(1.0).gradient(&[0.3, -1.2], &[1.0, 0.0], None, &mut b);
        assert_eq!(a, b);
    }

    /// Hinge: margins on the wrong side of the unit margin get `∓w`, the
    /// rest a zero gradient with the minimal positive Hessian.
    #[test]
    fn hinge_gradient_and_threshold() {
        let obj = HingeObjective;
        let mut out = vec![GradPair::default(); 4];
        obj.gradient(
            &[0.5, 1.0, -0.5, -2.0],
            &[1.0, 1.0, 0.0, 0.0],
            Some(&[2.0, 2.0, 3.0, 3.0]),
            &mut out,
        );
        assert_eq!(out[0], GradPair::new(-2.0, 2.0));
        assert_eq!(out[1], GradPair::new(0.0, f32::MIN_POSITIVE));
        assert_eq!(out[2], GradPair::new(3.0, 3.0));
        assert_eq!(out[3], GradPair::new(0.0, f32::MIN_POSITIVE));
        let mut p = [0.0f32, 1e-7, -1.0];
        obj.pred_transform(&mut p);
        assert_eq!(p, [0.0, 1.0, 0.0]);
    }

    /// The hinge intercept is the Newton step thresholded to a class
    /// (XGBoost `FitIntercept` applies `PredTransform`), and exporting it
    /// keeps the margin because hinge's `ProbToMargin` is the identity.
    #[test]
    fn hinge_intercept_is_thresholded_newton_step() {
        let obj = HingeObjective;
        // Step = -Σg/Σh = (3 - 1)/4 = 0.5 > 0 -> 1.
        assert_eq!(
            obj.base_margins(&[1.0, 1.0, 1.0, 0.0], None, None),
            vec![1.0]
        );
        assert_eq!(obj.base_margins(&[0.0, 0.0, 1.0], None, None), vec![0.0]);
        let mut stored = [0.5f32];
        obj.margins_to_probs(&mut stored);
        assert_eq!(stored, [0.5]);
    }
}
