//! Accelerated failure time (`survival:aft`, XGBoost `objective/aft_obj.cu`,
//! `common/survival_util.h`, `common/probability_distribution.h`).

use rayon::prelude::*;

use crate::data::MetaInfo;
use crate::error::{HessboostError, Result};
use crate::objective::AftDistribution;
use crate::objective::distributional::special::erf_glibc as erf;
use crate::objective::{
    GradPair, Loss, MIN_HESS_F64, OutputDomain, check_base_score_domain, inverse_log_link,
    log_link,
};

/// Accelerated failure time model (`survival:aft`) on interval-censored
/// survival times.
///
/// Each row carries a label interval `[lower, upper]` (see
/// [`DMatrix::with_label_bounds`](crate::data::DMatrix::with_label_bounds)):
/// `lower == upper` is an observed time, `upper = +inf` is right-censored,
/// `lower <= 0` is left-censored, and any other pair is interval-censored.
/// The model is `ln T = margin + sigma * Z` with `Z` drawn from
/// `distribution`; the loss is the negative log-likelihood of the interval.
/// Gradients and Hessians are clipped to XGBoost's ranges (`[-15, 15]` and
/// `[1e-16, 15]`), with its limits substituted where the ratio degenerates.
///
/// The margin is the log survival time and predictions are `exp(margin)`;
/// evaluation metrics receive the raw margins. The intercept is XGBoost's
/// default `base_score` of 0.5 (margin `ln 0.5`), not estimated from data.
/// The default metric is `aft-nloglik` with this distribution at scale 1, as
/// in XGBoost.
///
/// Training reads only the label bounds. Called without bounds
/// ([`Loss::gradient`]), the ordinary labels are treated as observed
/// times.
#[derive(Debug, Clone, Copy)]
pub struct AftLoss {
    distribution: AftDistribution,
    sigma: f32,
}

impl AftLoss {
    /// Create with the noise `distribution` and its scale `sigma`
    /// (XGBoost `aft_loss_distribution`, `aft_loss_distribution_scale`).
    pub fn new(distribution: AftDistribution, sigma: f32) -> Self {
        AftLoss {
            distribution,
            sigma,
        }
    }

    /// Per-row gradients from interval bounds; `f64` arithmetic rounded to
    /// `f32` before weighting, like XGBoost.
    fn bound_gradient<D: Distribution>(
        self,
        preds: &[f32],
        lower: &[f32],
        upper: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        let sigma = f64::from(self.sigma);
        let row = |(i, gp): (usize, &mut GradPair)| {
            let (lo, hi, pred) = (
                f64::from(lower[i]),
                f64::from(upper[i]),
                f64::from(preds[i]),
            );
            let (grad, hess) = aft_grad_hess::<D>(lo, hi, pred, sigma);
            let (grad, hess) = (grad as f32, hess as f32);
            let w = weights.map_or(1.0, |w| w[i]);
            *gp = GradPair::new(grad * w, hess * w);
        };
        if out.len() >= 8192 && rayon::current_num_threads() > 1 {
            out.par_iter_mut()
                .enumerate()
                .with_min_len(4096)
                .for_each(row);
        } else {
            out.iter_mut().enumerate().for_each(row);
        }
    }

    fn dispatch(
        self,
        preds: &[f32],
        lower: &[f32],
        upper: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        crate::objective::check_gradient_inputs(lower.len(), 1, preds, upper, weights, out);
        match self.distribution {
            AftDistribution::Normal => {
                self.bound_gradient::<Normal>(preds, lower, upper, weights, out);
            }
            AftDistribution::Logistic => {
                self.bound_gradient::<Logistic>(preds, lower, upper, weights, out);
            }
            AftDistribution::Extreme => {
                self.bound_gradient::<Extreme>(preds, lower, upper, weights, out);
            }
        }
    }
}

impl Default for AftLoss {
    /// XGBoost's defaults: normal noise with scale 1.
    fn default() -> Self {
        AftLoss::new(AftDistribution::Normal, 1.0)
    }
}

impl Loss for AftLoss {
    fn name(&self) -> &'static str {
        "survival:aft"
    }

    /// Gradients treating each label as an observed (uncensored) time.
    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        self.dispatch(preds, labels, labels, weights, out);
    }

    fn gradient_info(&self, preds: &[f32], info: &MetaInfo, out: &mut [GradPair]) {
        match info.bounds {
            Some(bounds) => self.dispatch(preds, bounds.lower(), bounds.upper(), info.weights, out),
            None => self.gradient(preds, info.label_values(), info.weights, out),
        }
    }

    fn pred_transform(&self, preds: &mut [f32]) {
        inverse_log_link(preds);
    }

    /// Identity: the AFT metrics consume the raw log-time margins.
    fn eval_transform(&self, _preds: &mut [f32]) {}

    fn probs_to_margins(&self, scores: &mut [f32]) {
        log_link(scores);
    }

    fn validate_base_score(&self, base_score: f64) -> Result<()> {
        check_base_score_domain(base_score, OutputDomain::Positive)
    }

    /// XGBoost does not estimate an AFT intercept: its default `base_score`
    /// 0.5 maps to the margin `ln 0.5`.
    fn base_margins_info(&self, _info: &MetaInfo) -> Vec<f32> {
        vec![0.5f32.ln()]
    }

    fn validate_info(&self, info: &MetaInfo) -> Result<()> {
        if info.bounds.is_none() {
            return Err(HessboostError::invalid_data(
                "label_bounds",
                "missing; survival:aft needs \
                 `label_lower_bound` and `label_upper_bound` (DMatrix::with_label_bounds)",
            ));
        }
        Ok(())
    }

    fn requires_labels(&self) -> bool {
        false
    }

    /// XGBoost configures the default metric from the objective's
    /// `DefaultMetricConfig` but without the user's parameters (the learner
    /// has cleared them by the time it evaluates): the distribution carries
    /// over while the scale falls back to its default 1. List `aft-nloglik`
    /// in `eval_metric` to evaluate the likelihood at the configured scale.
    fn default_metric(&self) -> crate::metric::EvalMetric {
        crate::metric::EvalMetric::AftNLogLik(crate::objective::Aft::with_distribution(
            self.distribution,
        ))
    }
}

// ---------------------------------------------------------------------------
// AFT likelihood (common/survival_util.h, common/probability_distribution.h)
// ---------------------------------------------------------------------------

/// Gradient clip range.
pub(super) const MIN_GRADIENT: f64 = -15.0;
const MAX_GRADIENT: f64 = 15.0;
/// Hessian clip upper end (the lower end is [`MIN_HESS_F64`]).
const MAX_HESSIAN: f64 = 15.0;
/// Floor of the likelihood and threshold of a degenerate denominator.
const EPS: f64 = 1e-12;

/// How a row's label interval is censored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Censoring {
    Uncensored,
    Right,
    Left,
    Interval,
}

/// The noise distribution of `ln T`: density, CDF, and the density's first
/// two derivatives at a z-score, plus the gradient/Hessian limits used when
/// a prediction is so far off that the likelihood ratio degenerates.
pub(super) trait Distribution {
    fn pdf(z: f64) -> f64;
    fn cdf(z: f64) -> f64;
    fn grad_pdf(z: f64) -> f64;
    fn hess_pdf(z: f64) -> f64;
    /// The `(low, high)` gradient limits [`limit_grad`] substitutes.
    fn grad_limits(sigma: f64) -> (f64, f64);
    fn limit_hess(censoring: Censoring, sign: bool, sigma: f64) -> f64;
}

/// The gradient substituted for a degenerate ratio: `D`'s low limit for a
/// positive z-score sign, its high limit otherwise, and `0` on the side a
/// right (left) censored interval leaves flat.
fn limit_grad<D: Distribution>(censoring: Censoring, sign: bool, sigma: f64) -> f64 {
    let (low, high) = D::grad_limits(sigma);
    match (censoring, sign) {
        (Censoring::Uncensored | Censoring::Interval | Censoring::Right, true) => low,
        (Censoring::Uncensored | Censoring::Interval | Censoring::Left, false) => high,
        (Censoring::Right, false) | (Censoring::Left, true) => 0.0,
    }
}

pub(super) struct Normal;
pub(super) struct Logistic;
pub(super) struct Extreme;

impl Distribution for Normal {
    fn pdf(z: f64) -> f64 {
        (-z * z / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }

    #[allow(clippy::manual_midpoint, reason = "XGBoost's operation order")]
    fn cdf(z: f64) -> f64 {
        0.5 * (1.0 + erf(z / 2.0f64.sqrt()))
    }

    fn grad_pdf(z: f64) -> f64 {
        -z * Self::pdf(z)
    }

    fn hess_pdf(z: f64) -> f64 {
        (z * z - 1.0) * Self::pdf(z)
    }

    fn grad_limits(_sigma: f64) -> (f64, f64) {
        (MIN_GRADIENT, MAX_GRADIENT)
    }

    fn limit_hess(censoring: Censoring, sign: bool, sigma: f64) -> f64 {
        let flat = 1.0 / (sigma * sigma);
        match censoring {
            Censoring::Uncensored | Censoring::Interval => flat,
            Censoring::Right => {
                if sign {
                    flat
                } else {
                    MIN_HESS_F64
                }
            }
            Censoring::Left => {
                if sign {
                    MIN_HESS_F64
                } else {
                    flat
                }
            }
        }
    }
}

impl Distribution for Logistic {
    fn pdf(z: f64) -> f64 {
        let w = z.exp();
        let sqrt_denominator = 1.0 + w;
        if w.is_infinite() || (w * w).is_infinite() {
            0.0
        } else {
            w / (sqrt_denominator * sqrt_denominator)
        }
    }

    fn cdf(z: f64) -> f64 {
        let w = z.exp();
        if w.is_infinite() { 1.0 } else { w / (1.0 + w) }
    }

    fn grad_pdf(z: f64) -> f64 {
        let w = z.exp();
        if w.is_infinite() {
            0.0
        } else {
            Self::pdf(z) * (1.0 - w) / (1.0 + w)
        }
    }

    fn hess_pdf(z: f64) -> f64 {
        let w = z.exp();
        if w.is_infinite() || (w * w).is_infinite() {
            0.0
        } else {
            Self::pdf(z) * (w * w - 4.0 * w + 1.0) / ((1.0 + w) * (1.0 + w))
        }
    }

    fn grad_limits(sigma: f64) -> (f64, f64) {
        (-1.0 / sigma, 1.0 / sigma)
    }

    fn limit_hess(_censoring: Censoring, _sign: bool, _sigma: f64) -> f64 {
        MIN_HESS_F64
    }
}

impl Distribution for Extreme {
    fn pdf(z: f64) -> f64 {
        let w = z.exp();
        if w.is_infinite() { 0.0 } else { w * (-w).exp() }
    }

    fn cdf(z: f64) -> f64 {
        let w = z.exp();
        1.0 - (-w).exp()
    }

    fn grad_pdf(z: f64) -> f64 {
        let w = z.exp();
        if w.is_infinite() {
            0.0
        } else {
            (1.0 - w) * Self::pdf(z)
        }
    }

    fn hess_pdf(z: f64) -> f64 {
        let w = z.exp();
        if w.is_infinite() || (w * w).is_infinite() {
            0.0
        } else {
            (w * w - 3.0 * w + 1.0) * Self::pdf(z)
        }
    }

    fn grad_limits(sigma: f64) -> (f64, f64) {
        (MIN_GRADIENT, 1.0 / sigma)
    }

    fn limit_hess(censoring: Censoring, sign: bool, _sigma: f64) -> f64 {
        match censoring {
            Censoring::Uncensored | Censoring::Right | Censoring::Interval => {
                if sign {
                    MAX_HESSIAN
                } else {
                    MIN_HESS_F64
                }
            }
            Censoring::Left => MIN_HESS_F64,
        }
    }
}

/// Distribution values at one censored endpoint: `(pdf, cdf, grad_pdf, z)`,
/// with the sentinel `(0, at_limit_cdf, 0, 0)` for an absent endpoint
/// (`+inf` upper or `<= 0` lower bound).
struct Endpoint {
    pdf: f64,
    cdf: f64,
    grad_pdf: f64,
    z: f64,
}

impl Endpoint {
    fn at<D: Distribution>(log_y: f64, pred: f64, sigma: f64) -> Self {
        let z = (log_y - pred) / sigma;
        Endpoint {
            pdf: D::pdf(z),
            cdf: D::cdf(z),
            grad_pdf: D::grad_pdf(z),
            z,
        }
    }

    fn absent(cdf: f64) -> Self {
        Endpoint {
            pdf: 0.0,
            cdf,
            grad_pdf: 0.0,
            z: 0.0,
        }
    }
}

/// The two ends of a censored interval and its censoring type.
fn censored_ends<D: Distribution>(
    y_lower: f64,
    y_upper: f64,
    pred: f64,
    sigma: f64,
) -> (Endpoint, Endpoint, Censoring) {
    let mut censoring = Censoring::Interval;
    let upper = if y_upper.is_infinite() {
        censoring = Censoring::Right;
        Endpoint::absent(1.0)
    } else {
        Endpoint::at::<D>(y_upper.ln(), pred, sigma)
    };
    let lower = if y_lower <= 0.0 {
        censoring = Censoring::Left;
        Endpoint::absent(0.0)
    } else {
        Endpoint::at::<D>(y_lower.ln(), pred, sigma)
    };
    (lower, upper, censoring)
}

/// Negative log-likelihood of one row (XGBoost `AFTLoss::Loss`).
fn aft_loss<D: Distribution>(y_lower: f64, y_upper: f64, pred: f64, sigma: f64) -> f64 {
    if y_lower == y_upper {
        let z = (y_lower.ln() - pred) / sigma;
        let pdf = D::pdf(z);
        -(pdf / (sigma * y_lower)).max(EPS).ln()
    } else {
        let cdf_u = if y_upper.is_infinite() {
            1.0
        } else {
            D::cdf((y_upper.ln() - pred) / sigma)
        };
        let cdf_l = if y_lower <= 0.0 {
            0.0
        } else {
            D::cdf((y_lower.ln() - pred) / sigma)
        };
        -(cdf_u - cdf_l).max(EPS).ln()
    }
}

/// d loss / d margin and d² loss / d margin² (XGBoost `AFTLoss::Gradient`
/// and `AFTLoss::Hessian`), each clipped. Both are formed from the same
/// endpoint values, computed once.
pub(super) fn aft_grad_hess<D: Distribution>(
    y_lower: f64,
    y_upper: f64,
    pred: f64,
    sigma: f64,
) -> (f64, f64) {
    // `(numerator, denominator)` of the gradient and of the Hessian.
    let (grad, hess, censoring, z_sign) = if y_lower == y_upper {
        let z = (y_lower.ln() - pred) / sigma;
        let pdf = D::pdf(z);
        let grad_pdf = D::grad_pdf(z);
        let hess_pdf = D::hess_pdf(z);
        (
            (grad_pdf, sigma * pdf),
            (
                -(pdf * hess_pdf - grad_pdf * grad_pdf),
                sigma * sigma * pdf * pdf,
            ),
            Censoring::Uncensored,
            z > 0.0,
        )
    } else {
        let (lo, hi, censoring) = censored_ends::<D>(y_lower, y_upper, pred, sigma);
        let cdf_diff = hi.cdf - lo.cdf;
        let pdf_diff = hi.pdf - lo.pdf;
        let grad_diff = hi.grad_pdf - lo.grad_pdf;
        let sqrt_denominator = sigma * cdf_diff;
        (
            (pdf_diff, sigma * cdf_diff),
            (
                -(cdf_diff * grad_diff - pdf_diff * pdf_diff),
                sqrt_denominator * sqrt_denominator,
            ),
            censoring,
            hi.z > 0.0 || lo.z > 0.0,
        )
    };
    let mut gradient = grad.0 / grad.1;
    if grad.1 < EPS && !gradient.is_finite() {
        gradient = limit_grad::<D>(censoring, z_sign, sigma);
    }
    let mut hessian = hess.0 / hess.1;
    if hess.1 < EPS && !hessian.is_finite() {
        hessian = D::limit_hess(censoring, z_sign, sigma);
    }
    (
        clip(gradient, MIN_GRADIENT, MAX_GRADIENT),
        clip(hessian, MIN_HESS_F64, MAX_HESSIAN),
    )
}

/// XGBoost `aft::Clip` (NaN passes through unchanged).
fn clip(x: f64, min: f64, max: f64) -> f64 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

/// AFT negative log-likelihood of one row for `distribution` with scale
/// `sigma`, given the raw log-time margin `pred`. Backs the `aft-nloglik`
/// metric.
pub(crate) fn aft_nloglik(
    distribution: AftDistribution,
    y_lower: f64,
    y_upper: f64,
    pred: f64,
    sigma: f64,
) -> f64 {
    match distribution {
        AftDistribution::Normal => aft_loss::<Normal>(y_lower, y_upper, pred, sigma),
        AftDistribution::Logistic => aft_loss::<Logistic>(y_lower, y_upper, pred, sigma),
        AftDistribution::Extreme => aft_loss::<Extreme>(y_lower, y_upper, pred, sigma),
    }
}
