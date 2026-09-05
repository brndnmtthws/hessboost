//! Count and positive-continuous regression objectives with a log link:
//! Poisson, Gamma, and Tweedie. All predict `exp(margin)`.

use super::{weighted_label_mean, GradPair, Objective};

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
    fn name(&self) -> &str {
        "count:poisson"
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        crate::simd::poisson_gradient(preds, labels, weights, self.max_delta_step, out);
    }

    fn pred_transform(&self, preds: &mut [f32]) {
        crate::simd::exp_inplace(preds);
    }

    fn prob_to_margin(&self, base_score: f32) -> f32 {
        base_score.max(1e-6).ln()
    }

    fn base_margin(&self, labels: &[f32], weights: Option<&[f32]>) -> f32 {
        (weighted_label_mean(labels, weights).max(1e-6)).ln() as f32
    }

    fn default_metric(&self) -> &str {
        "poisson-nloglik"
    }
}

/// Gamma regression (`reg:gamma`), a log-link objective for positive targets.
/// Gradient `1 − y·exp(−m)`, Hessian `y·exp(−m)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct GammaObjective;

impl Objective for GammaObjective {
    fn name(&self) -> &str {
        "reg:gamma"
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        crate::simd::gamma_gradient(preds, labels, weights, out);
    }

    fn pred_transform(&self, preds: &mut [f32]) {
        crate::simd::exp_inplace(preds);
    }

    fn prob_to_margin(&self, base_score: f32) -> f32 {
        base_score.max(1e-6).ln()
    }

    fn base_margin(&self, labels: &[f32], weights: Option<&[f32]>) -> f32 {
        (weighted_label_mean(labels, weights).max(1e-6)).ln() as f32
    }

    fn default_metric(&self) -> &str {
        "gamma-nloglik"
    }
}

/// Tweedie regression (`reg:tweedie`) with variance power `rho ∈ (1, 2)`.
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
    fn name(&self) -> &str {
        "reg:tweedie"
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        crate::simd::tweedie_gradient(preds, labels, weights, self.rho, out);
    }

    fn pred_transform(&self, preds: &mut [f32]) {
        crate::simd::exp_inplace(preds);
    }

    fn prob_to_margin(&self, base_score: f32) -> f32 {
        base_score.max(1e-6).ln()
    }

    fn base_margin(&self, labels: &[f32], weights: Option<&[f32]>) -> f32 {
        (weighted_label_mean(labels, weights).max(1e-6)).ln() as f32
    }

    fn default_metric(&self) -> &str {
        "tweedie-nloglik"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn poisson_gradient_at_log_mean_is_zero_sum() {
        // At margin = log(y) the gradient exp(m)-y = 0.
        let obj = PoissonObjective::default();
        let labels = [2.0f32, 5.0];
        let preds = [2.0f32.ln(), 5.0f32.ln()];
        let mut out = vec![GradPair::default(); 2];
        obj.gradient(&preds, &labels, None, &mut out);
        assert_relative_eq!(out[0].grad, 0.0, epsilon = 1e-5);
        assert_relative_eq!(out[1].grad, 0.0, epsilon = 1e-5);
        assert!(out[0].hess > 0.0);
    }

    #[test]
    fn gamma_gradient_zero_at_log_y() {
        let obj = GammaObjective;
        let labels = [3.0f32];
        let preds = [3.0f32.ln()];
        let mut out = vec![GradPair::default(); 1];
        obj.gradient(&preds, &labels, None, &mut out);
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
}
