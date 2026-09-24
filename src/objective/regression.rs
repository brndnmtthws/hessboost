//! Regression objectives.

use super::{GradPair, Objective, check_label_domain, weighted_label_mean};
use crate::data::MetaInfo;
use crate::error::Result;

/// Squared-error regression (`reg:squarederror`).
///
/// Loss `½ (pred − label)²` gives gradient `pred − label` and constant Hessian
/// `1`. The prediction transform is the identity and the optimal base margin is
/// the (weighted) label mean.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct SquaredError;

impl Objective for SquaredError {
    fn name(&self) -> &'static str {
        "reg:squarederror"
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        super::elementwise_gradient(preds, labels, weights, out, |p, y, w| {
            GradPair::new((p - y) * w, w)
        });
    }

    fn const_hess(&self) -> bool {
        true
    }

    fn base_margins_info(&self, info: &MetaInfo) -> Vec<f32> {
        // XGBoost `FitInterceptGlmLike`: the (weighted) label mean.
        vec![weighted_label_mean(info.labels, info.weights)]
    }

    fn pointwise_loss(&self) -> Option<super::PointwiseLoss<'_>> {
        // `½ (margin − y)²`.
        Some(Box::new(|margin, label| {
            0.5 * (f64::from(margin) - f64::from(label)).powi(2)
        }))
    }

    fn default_metric(&self) -> String {
        "rmse".to_string()
    }
}

/// Pseudo-Huber regression (`reg:pseudohubererror`): a smooth approximation of
/// the absolute error, robust to outliers. With residual `z = margin − y` and
/// slope `δ` (XGBoost `huber_slope`), the gradient is `z / √(1 + z²/δ²)` and
/// the Hessian `δ² / ((δ² + z²) · √(1 + z²/δ²))`, evaluated in `f32` exactly
/// like XGBoost's `PseudoHuberRegression`. The intercept is the trait's
/// default Newton step (XGBoost `FitIntercept`).
#[derive(Debug, Clone, Copy)]
pub struct PseudoHuber {
    slope: f32,
}

impl PseudoHuber {
    /// Create with the given Huber slope `δ`.
    pub fn new(slope: f32) -> Self {
        PseudoHuber { slope }
    }
}

impl Default for PseudoHuber {
    fn default() -> Self {
        PseudoHuber { slope: 1.0 }
    }
}

impl Objective for PseudoHuber {
    fn name(&self) -> &'static str {
        "reg:pseudohubererror"
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        let slope_sq = self.slope * self.slope;
        super::elementwise_gradient(preds, labels, weights, out, |p, y, w| {
            let z = p - y;
            let scale_sqrt = (1.0 + z * z / slope_sq).sqrt();
            let scale = slope_sq + z * z;
            GradPair::new((z / scale_sqrt) * w, (slope_sq / (scale * scale_sqrt)) * w)
        });
    }

    fn pointwise_loss(&self) -> Option<super::PointwiseLoss<'_>> {
        // `δ² (√(1 + z²/δ²) − 1)`, whose derivatives are the gradient above,
        // rationalized to `z² / (√(1 + z²/δ²) + 1)`: the subtraction would
        // cancel to `0` once `z²/δ²` drops below the `f64` epsilon (e.g. a
        // unit residual under a large slope), losing the quadratic regime.
        let slope_sq = f64::from(self.slope).powi(2);
        Some(Box::new(move |margin, label| {
            let z = f64::from(margin) - f64::from(label);
            z * z / ((1.0 + z * z / slope_sq).sqrt() + 1.0)
        }))
    }

    fn default_metric(&self) -> String {
        "mphe".to_string()
    }
}

/// Squared log error regression (`reg:squaredlogerror`): the loss
/// `½ [ln(1 + pred) − ln(1 + y)]²` behind the `rmsle` metric. Labels must
/// exceed `-1`. With the margin clamped to `p = max(margin, −1 + 10⁻⁶)` for
/// the gradient only, `g = [ln1p(p) − ln1p(y)] / (p + 1)` and
/// `h = max([−ln1p(p) + ln1p(y) + 1] / (p + 1)², 10⁻⁶)`, evaluated exactly
/// like XGBoost's `SquaredLogError` (the Hessian's square and division in
/// `f64`, as `std::pow(float, int)` promotes). The prediction transform is
/// the identity (predictions are not clamped) and the intercept is the
/// trait's default Newton step (XGBoost `FitIntercept`).
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct SquaredLogError;

/// XGBoost's `fmaxf(predt, -1 + 1e-6)` bound: the `f64` constant rounded to
/// `f32`.
const SQUARED_LOG_MIN_PRED: f32 = (-1.0f64 + 1e-6) as f32;

impl Objective for SquaredLogError {
    fn name(&self) -> &'static str {
        "reg:squaredlogerror"
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        super::elementwise_gradient(preds, labels, weights, out, |p, y, w| {
            let p = p.max(SQUARED_LOG_MIN_PRED);
            let (log_p, log_y) = (p.ln_1p(), y.ln_1p());
            let grad = (log_p - log_y) / (p + 1.0);
            let shifted = f64::from(p + 1.0);
            let hess = ((f64::from(-log_p + log_y + 1.0) / (shifted * shifted)) as f32).max(1e-6);
            GradPair::new(grad * w, hess * w)
        });
    }

    fn validate_info(&self, info: &MetaInfo) -> Result<()> {
        // XGBoost `SquaredLogError::CheckLabel`: `log1p(label)` must be defined.
        check_label_domain(info, |y| y <= -1.0)
    }

    fn default_metric(&self) -> String {
        "rmsle".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objective::{base_margins, gradient_pairs};

    #[test]
    fn gradient_matches_closed_form() {
        let obj = SquaredError;
        let preds = [2.0f32, 0.0, -1.0];
        let labels = [1.0f32, 0.5, -3.0];
        let out = gradient_pairs(&obj, &preds, &labels, None);
        assert_eq!(out[0], GradPair::new(1.0, 1.0)); // 2 - 1
        assert_eq!(out[1], GradPair::new(-0.5, 1.0)); // 0 - 0.5
        assert_eq!(out[2], GradPair::new(2.0, 1.0)); // -1 - (-3)
    }

    #[test]
    fn weighted_gradient_scales() {
        let obj = SquaredError;
        let preds = [2.0f32];
        let labels = [1.0f32];
        let w = [4.0f32];
        let out = gradient_pairs(&obj, &preds, &labels, Some(&w));
        assert_eq!(out[0], GradPair::new(4.0, 4.0));
    }

    #[test]
    fn base_margins_is_label_mean() {
        let obj = SquaredError;
        assert_eq!(base_margins(&obj, &[1.0, 2.0, 3.0], None), vec![2.0]);
    }

    /// Pseudo-Huber with slope δ: at `z = δ` the gradient is `δ/√2` and the
    /// Hessian `1/(2√2)`, so a wrong slope scaling would be visible.
    #[test]
    fn pseudo_huber_slope_scales_gradient() {
        let obj = PseudoHuber::new(2.0);
        let out = gradient_pairs(&obj, &[2.0], &[0.0], None);
        let root2 = 2f32.sqrt();
        assert!(
            (out[0].grad - 2.0 / root2).abs() < 1e-6,
            "grad {}",
            out[0].grad
        );
        assert!(
            (out[0].hess - 1.0 / (2.0 * root2)).abs() < 1e-6,
            "hess {}",
            out[0].hess
        );
        // Unit slope reproduces the classic form d/√(1+d²), 1/(1+d²)^{3/2}.
        let out = gradient_pairs(&PseudoHuber::default(), &[2.0], &[0.0], None);
        let s = 5f32;
        assert_eq!(out[0], GradPair::new(2.0 / s.sqrt(), 1.0 / (s * s.sqrt())));
    }

    /// The pseudo-Huber intercept is the Newton step, not the label mean
    /// (XGBoost `FitIntercept`): for labels {0, 4} the mean is 2, but the
    /// bounded gradient `z/√(1+z²)` makes the step from zero, `-Σg/Σh` with
    /// `h = 1/(1+z²)^{3/2}`, fall well short of it.
    #[test]
    fn pseudo_huber_intercept_is_newton_step() {
        let obj = PseudoHuber::default();
        let labels = [0.0f32, 4.0];
        let margins = base_margins(&obj, &labels, None);
        let s = 17f32; // 1 + 4²
        let g1 = -4.0f32 / s.sqrt();
        let h1 = 1.0f32 / (s * s.sqrt());
        let expected = (-f64::from(g1) / (1.0 + f64::from(h1))) as f32;
        assert_eq!(margins, vec![expected]);
        assert!(
            margins[0] < 1.0,
            "Newton step {} should undershoot the mean",
            margins[0]
        );
    }

    /// The loss keeps its quadratic regime `z²/2` under a slope so large that
    /// `1 + z²/δ²` rounds to `1`, and still matches `δ² (√(1 + z²/δ²) − 1)`
    /// where that form is accurate.
    #[test]
    fn pseudo_huber_loss_survives_large_slopes() {
        let obj = PseudoHuber::new(1e9);
        let loss = obj.pointwise_loss().unwrap();
        assert!((loss(0.0, 1.0) - 0.5).abs() < 1e-12, "{}", loss(0.0, 1.0));

        let obj = PseudoHuber::new(2.0);
        let loss = obj.pointwise_loss().unwrap();
        let naive = 4.0 * ((1.0f64 + 9.0 / 4.0).sqrt() - 1.0);
        assert!((loss(3.0, 0.0) - naive).abs() < 1e-12);
    }

    /// The squared-log gradient vanishes where `pred == label` and the
    /// margin clamp keeps it finite for margins at or below `-1`, where
    /// `ln1p` itself is undefined.
    #[test]
    fn squared_log_gradient_zero_at_label_and_finite_below_minus_one() {
        let obj = SquaredLogError;
        let out = gradient_pairs(&obj, &[3.0, -1.0, -7.0], &[3.0, 0.5, 0.5], None);
        assert_eq!(out[0].grad, 0.0);
        // Hessian at the optimum is 1 / (p + 1)².
        assert_eq!(out[0].hess, 1.0 / 16.0);
        assert!(out[1].grad.is_finite() && out[1].grad < 0.0);
        assert_eq!(out[1], out[2], "margins below the clamp share its gradient");
    }

    /// Far above the label the curvature term goes negative; XGBoost floors
    /// it at `1e-6` (times the weight).
    #[test]
    fn squared_log_hessian_floor_is_weighted() {
        let obj = SquaredLogError;
        let out = gradient_pairs(&obj, &[1e4], &[0.0], Some(&[2.0]));
        assert_eq!(out[0].hess, 2e-6);
        assert!(out[0].grad > 0.0);
    }

    #[test]
    fn squared_log_rejects_labels_at_minus_one() {
        let obj = SquaredLogError;
        assert!(
            obj.validate_info(&MetaInfo::new(&[-0.5, 2.0], None, None))
                .is_ok()
        );
        assert!(
            obj.validate_info(&MetaInfo::new(&[-1.0], None, None))
                .is_err()
        );
    }
}
