//! Count and positive-continuous regression objectives with a log link:
//! Poisson, Gamma, and Tweedie. All predict `exp(margin)`.

use super::{GradPair, Objective, check_label_domain, weighted_label_mean};
use crate::data::MetaInfo;
use crate::error::Result;

/// Half the Poisson unit deviance at log-mean `margin`: `y ln(y/μ) − (y − μ)`,
/// whose margin derivative is the Poisson gradient `μ − y`.
fn poisson_deviance(margin: f32, label: f32) -> f64 {
    let (m, y) = (f64::from(margin), f64::from(label));
    let mu = m.exp();
    if y > 0.0 {
        y * (y.ln() - m) - (y - mu)
    } else {
        mu
    }
}

/// Emit the `pred_transform`/`prob_to_margin`/`base_margins` trio shared by
/// the log-link objectives (all predict `exp(margin)`). The link is XGBoost's
/// `ProbToMargin`, `ln(v)` in `f32`; the intercept is XGBoost's
/// `FitInterceptGlmLike`, the (weighted) label mean through that link.
macro_rules! log_link_objective {
    () => {
        fn pred_transform(&self, preds: &mut [f32]) {
            crate::simd::exp_inplace(preds);
        }

        fn prob_to_margin(&self, base_score: f32) -> f32 {
            base_score.ln()
        }

        fn base_margins(
            &self,
            labels: &[f32],
            weights: Option<&[f32]>,
            _group: Option<&crate::data::GroupInfo>,
        ) -> Vec<f32> {
            vec![self.prob_to_margin(weighted_label_mean(labels, weights))]
        }
    };
}

/// Poisson regression (`count:poisson`). Gradient is `exp(m) − y`. The Hessian is
/// stabilized by `max_delta_step` (default 0.7 in XGBoost) via
/// `exp(m + max_delta_step)`.
#[derive(Debug, Clone, Copy)]
pub struct PoissonObjective {
    max_delta_step: f32,
}

impl PoissonObjective {
    /// Create with the given Hessian-stabilizing max delta step.
    pub fn new(max_delta_step: f32) -> Self {
        PoissonObjective { max_delta_step }
    }
}

impl Default for PoissonObjective {
    fn default() -> Self {
        PoissonObjective {
            max_delta_step: 0.7,
        }
    }
}

impl Objective for PoissonObjective {
    fn name(&self) -> &'static str {
        "count:poisson"
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        super::check_gradient_inputs(labels.len(), 1, preds, labels, weights, out);
        crate::simd::poisson_gradient(preds, labels, weights, self.max_delta_step, out);
    }

    log_link_objective!();

    fn pointwise_loss(&self) -> Option<super::PointwiseLoss<'_>> {
        Some(Box::new(poisson_deviance))
    }

    fn validate_info(&self, info: &MetaInfo) -> Result<()> {
        check_label_domain(info, |y| y < 0.0)
    }

    fn default_metric(&self) -> String {
        "poisson-nloglik".to_string()
    }
}

/// Gamma regression (`reg:gamma`), a log-link objective for positive targets.
/// Gradient `1 − y·exp(−m)`, Hessian `y·exp(−m)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct GammaObjective;

impl Objective for GammaObjective {
    fn name(&self) -> &'static str {
        "reg:gamma"
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        super::check_gradient_inputs(labels.len(), 1, preds, labels, weights, out);
        crate::simd::gamma_gradient(preds, labels, weights, out);
    }

    log_link_objective!();

    fn pointwise_loss(&self) -> Option<super::PointwiseLoss<'_>> {
        // Half the Gamma unit deviance: `y/μ − ln(y/μ) − 1`.
        Some(Box::new(|margin, label| {
            let (m, y) = (f64::from(margin), f64::from(label));
            y * (-m).exp() + m - y.ln() - 1.0
        }))
    }

    fn validate_info(&self, info: &MetaInfo) -> Result<()> {
        check_label_domain(info, |y| y <= 0.0)
    }

    fn default_metric(&self) -> String {
        "gamma-nloglik".to_string()
    }
}

/// Tweedie regression (`reg:tweedie`) with variance power `rho ∈ [1, 2)`
/// (XGBoost `tweedie_variance_power`; 1 is Poisson, 2 would be Gamma).
#[derive(Debug, Clone, Copy)]
pub struct TweedieObjective {
    rho: f32,
}

impl TweedieObjective {
    /// Create with the given Tweedie variance power.
    pub fn new(rho: f32) -> Self {
        TweedieObjective { rho }
    }
}

impl Default for TweedieObjective {
    fn default() -> Self {
        TweedieObjective { rho: 1.5 }
    }
}

impl Objective for TweedieObjective {
    fn name(&self) -> &'static str {
        "reg:tweedie"
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        super::check_gradient_inputs(labels.len(), 1, preds, labels, weights, out);
        crate::simd::tweedie_gradient(preds, labels, weights, self.rho, out);
    }

    log_link_objective!();

    fn pointwise_loss(&self) -> Option<super::PointwiseLoss<'_>> {
        // Half the Tweedie unit deviance for `1 < ρ < 2` (Poisson at `ρ = 1`):
        // `y^(2−ρ)/((1−ρ)(2−ρ)) − y μ^(1−ρ)/(1−ρ) + μ^(2−ρ)/(2−ρ)`.
        let rho = f64::from(self.rho);
        if (rho - 1.0).abs() < 1e-9 {
            return Some(Box::new(poisson_deviance));
        }
        let (a, b) = (1.0 - rho, 2.0 - rho);
        Some(Box::new(move |margin, label| {
            let (m, y) = (f64::from(margin), f64::from(label));
            y.powf(b) / (a * b) - y * (a * m).exp() / a + (b * m).exp() / b
        }))
    }

    fn validate_info(&self, info: &MetaInfo) -> Result<()> {
        check_label_domain(info, |y| y < 0.0)
    }

    fn default_metric(&self) -> String {
        // XGBoost `TweedieRegression::Configure` names the metric with the
        // configured power so evaluation uses the same distribution.
        format!("tweedie-nloglik@{}", self.rho)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objective::gradient_pairs;
    use approx::assert_relative_eq;

    #[test]
    fn poisson_gradient_at_log_mean_is_zero_sum() {
        // At margin = log(y) the gradient exp(m)-y = 0.
        let obj = PoissonObjective::default();
        let labels = [2.0f32, 5.0];
        let preds = [2.0f32.ln(), 5.0f32.ln()];
        let out = gradient_pairs(&obj, &preds, &labels, None);
        assert_relative_eq!(out[0].grad, 0.0, epsilon = 1e-5);
        assert_relative_eq!(out[1].grad, 0.0, epsilon = 1e-5);
        assert!(out[0].hess > 0.0);
    }

    #[test]
    fn gamma_gradient_zero_at_log_y() {
        let obj = GammaObjective;
        let labels = [3.0f32];
        let preds = [3.0f32.ln()];
        let out = gradient_pairs(&obj, &preds, &labels, None);
        // 1 - y*exp(-log y) = 1 - 1 = 0
        assert_relative_eq!(out[0].grad, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn tweedie_transform_is_exp() {
        let obj = TweedieObjective::default();
        let mut p = [0.0f32, 1.0];
        obj.pred_transform(&mut p);
        assert_relative_eq!(p[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(p[1], 1.0f32.exp(), epsilon = 1e-6);
    }

    /// The log-link intercept is `ln(mean)` in f32 (XGBoost
    /// `FitInterceptGlmLike` + `ProbToMargin`); with weights it is the
    /// weighted mean, and a zero mean maps to `-inf` like XGBoost (which the
    /// trainer rejects rather than floors).
    #[test]
    fn log_link_intercept_is_ln_of_mean() {
        let obj = PoissonObjective::default();
        assert_eq!(obj.base_margins(&[2.0, 6.0], None, None), vec![4f32.ln()]);
        let w = [3.0f32, 1.0];
        assert_eq!(
            obj.base_margins(&[2.0, 6.0], Some(&w), None),
            vec![3f32.ln()]
        );
        assert_eq!(
            GammaObjective.base_margins(&[0.0, 0.0], None, None),
            vec![f32::NEG_INFINITY]
        );
    }
}
