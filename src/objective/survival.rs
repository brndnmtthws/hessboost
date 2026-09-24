//! Survival objectives: Cox proportional hazards (`survival:cox`) and the
//! accelerated failure time model (`survival:aft`).
//!
//! Both model log-scale margins and report `exp(margin)`: a hazard ratio for
//! Cox, a survival time for AFT. The arithmetic follows XGBoost 3.4
//! (`objective/regression_obj.cu`, `objective/aft_obj.cu`,
//! `common/survival_util.h`, `common/probability_distribution.h`) operation
//! for operation, including its `f32`/`f64` boundaries.

use rayon::prelude::*;

use super::{GradPair, MIN_HESS_F64, Objective, log_link};
use crate::config::AftDistribution;
use crate::data::MetaInfo;
use crate::error::{HessboostError, Result};

/// `exp` of every element in `f32` (XGBoost's `std::exp` on a float), used
/// by both survival objectives' prediction transform.
fn exp_transform(preds: &mut [f32]) {
    for p in preds {
        *p = p.exp();
    }
}

/// Row indices sorted by increasing `|label|`, ties in row order (XGBoost
/// `MetaInfo::LabelAbsSort`, a stable sort). Shared with the `cox-nloglik`
/// metric.
pub(crate) fn abs_label_order(labels: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..labels.len()).collect();
    order.sort_by(|&a, &b| labels[a].abs().total_cmp(&labels[b].abs()));
    order
}

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

impl Objective for Cox {
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
        super::check_gradient_inputs(labels.len(), 1, preds, labels, weights, out);
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
        exp_transform(preds);
    }

    fn probs_to_margins(&self, scores: &mut [f32]) {
        log_link(scores);
    }

    fn default_metric(&self) -> String {
        "cox-nloglik".to_string()
    }
}

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
/// ([`Objective::gradient`]), the ordinary labels are treated as observed
/// times.
#[derive(Debug, Clone, Copy)]
pub struct Aft {
    distribution: AftDistribution,
    sigma: f32,
}

impl Aft {
    /// Create with the noise `distribution` and its scale `sigma`
    /// (XGBoost `aft_loss_distribution`, `aft_loss_distribution_scale`).
    pub fn new(distribution: AftDistribution, sigma: f32) -> Self {
        Aft {
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
            let grad = aft_gradient::<D>(lo, hi, pred, sigma) as f32;
            let hess = aft_hessian::<D>(lo, hi, pred, sigma) as f32;
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
        super::check_gradient_inputs(lower.len(), 1, preds, upper, weights, out);
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

impl Default for Aft {
    /// XGBoost's defaults: normal noise with scale 1.
    fn default() -> Self {
        Aft::new(AftDistribution::Normal, 1.0)
    }
}

impl Objective for Aft {
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
        match (info.label_lower_bound, info.label_upper_bound) {
            (Some(lower), Some(upper)) => self.dispatch(preds, lower, upper, info.weights, out),
            _ => self.gradient(preds, info.labels, info.weights, out),
        }
    }

    fn pred_transform(&self, preds: &mut [f32]) {
        exp_transform(preds);
    }

    /// Identity: the AFT metrics consume the raw log-time margins.
    fn eval_transform(&self, _preds: &mut [f32]) {}

    fn probs_to_margins(&self, scores: &mut [f32]) {
        log_link(scores);
    }

    /// XGBoost does not estimate an AFT intercept: its default `base_score`
    /// 0.5 maps to the margin `ln 0.5`.
    fn base_margins_info(&self, _info: &MetaInfo) -> Vec<f32> {
        vec![0.5f32.ln()]
    }

    fn validate_info(&self, info: &MetaInfo) -> Result<()> {
        if info.label_lower_bound.is_none() || info.label_upper_bound.is_none() {
            return Err(HessboostError::invalid_param(
                "label_bounds",
                "dataset has no label bounds; survival:aft needs \
                 `label_lower_bound` and `label_upper_bound` (DMatrix::with_label_bounds)",
            ));
        }
        Ok(())
    }

    fn requires_labels(&self) -> bool {
        false
    }

    fn default_metric(&self) -> String {
        "aft-nloglik".to_string()
    }
}

// ---------------------------------------------------------------------------
// AFT likelihood (common/survival_util.h, common/probability_distribution.h)
// ---------------------------------------------------------------------------

/// Gradient clip range.
const MIN_GRADIENT: f64 = -15.0;
const MAX_GRADIENT: f64 = 15.0;
/// Hessian clip upper end (the lower end is [`MIN_HESS_F64`]).
const MAX_HESSIAN: f64 = 15.0;
/// Floor of the likelihood and threshold of a degenerate denominator.
const EPS: f64 = 1e-12;

/// How a row's label interval is censored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Censoring {
    Uncensored,
    Right,
    Left,
    Interval,
}

/// The noise distribution of `ln T`: density, CDF, and the density's first
/// two derivatives at a z-score, plus the gradient/Hessian limits used when
/// a prediction is so far off that the likelihood ratio degenerates.
trait Distribution {
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

struct Normal;
struct Logistic;
struct Extreme;

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

/// d loss / d margin (XGBoost `AFTLoss::Gradient`), clipped.
fn aft_gradient<D: Distribution>(y_lower: f64, y_upper: f64, pred: f64, sigma: f64) -> f64 {
    let (numerator, denominator, censoring, z_sign) = if y_lower == y_upper {
        let z = (y_lower.ln() - pred) / sigma;
        (
            D::grad_pdf(z),
            sigma * D::pdf(z),
            Censoring::Uncensored,
            z > 0.0,
        )
    } else {
        let (lo, hi, censoring) = censored_ends::<D>(y_lower, y_upper, pred, sigma);
        (
            hi.pdf - lo.pdf,
            sigma * (hi.cdf - lo.cdf),
            censoring,
            hi.z > 0.0 || lo.z > 0.0,
        )
    };
    let mut gradient = numerator / denominator;
    if denominator < EPS && !gradient.is_finite() {
        gradient = limit_grad::<D>(censoring, z_sign, sigma);
    }
    clip(gradient, MIN_GRADIENT, MAX_GRADIENT)
}

/// d² loss / d margin² (XGBoost `AFTLoss::Hessian`), clipped.
fn aft_hessian<D: Distribution>(y_lower: f64, y_upper: f64, pred: f64, sigma: f64) -> f64 {
    let (numerator, denominator, censoring, z_sign) = if y_lower == y_upper {
        let z = (y_lower.ln() - pred) / sigma;
        let pdf = D::pdf(z);
        let grad_pdf = D::grad_pdf(z);
        let hess_pdf = D::hess_pdf(z);
        (
            -(pdf * hess_pdf - grad_pdf * grad_pdf),
            sigma * sigma * pdf * pdf,
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
            -(cdf_diff * grad_diff - pdf_diff * pdf_diff),
            sqrt_denominator * sqrt_denominator,
            censoring,
            hi.z > 0.0 || lo.z > 0.0,
        )
    };
    let mut hessian = numerator / denominator;
    if denominator < EPS && !hessian.is_finite() {
        hessian = D::limit_hess(censoring, z_sign, sigma);
    }
    clip(hessian, MIN_HESS_F64, MAX_HESSIAN)
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

// ---------------------------------------------------------------------------
// erf
// ---------------------------------------------------------------------------

// `erf` below is ported from glibc 2.41's sysdeps/ieee754/dbl-64/s_erf.c,
// whose notice follows verbatim:
//
// /* @(#)s_erf.c 5.1 93/09/24 */
// /*
//  * ====================================================
//  * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//  *
//  * Developed at SunPro, a Sun Microsystems, Inc. business.
//  * Permission to use, copy, modify, and distribute this
//  * software is freely granted, provided that this notice
//  * is preserved.
//  * ====================================================
//  */
// /* Modified by Naohiko Shimizu/Tokai University, Japan 1997/08/25,
//    for performance improvement on pipelined processors.
// */
/// The error function, ported from glibc's `s_erf.c` (Sun fdlibm with
/// glibc's polynomial evaluation order), the `erf` XGBoost's normal CDF
/// calls on Linux. Each `a + b * c` is a fused multiply-add, as GCC
/// contracts it: bit-identical to glibc 2.41 on aarch64, within 1 ulp of
/// the unfused evaluation elsewhere.
#[allow(
    clippy::excessive_precision,
    reason = "fdlibm's published coefficients"
)]
fn erf(x: f64) -> f64 {
    const ERX: f64 = 8.450_629_115_104_675_292_97e-01;
    const EFX: f64 = 1.283_791_670_955_125_852_83e-01;
    const PP0: f64 = 1.283_791_670_955_125_585_61e-01;
    const PP1: f64 = -3.250_421_072_470_014_993_70e-01;
    const PP2: f64 = -2.848_174_957_559_851_047_66e-02;
    const PP3: f64 = -5.770_270_296_489_441_591_57e-03;
    const PP4: f64 = -2.376_301_665_665_016_260_84e-05;
    const QQ1: f64 = 3.979_172_239_591_553_528_19e-01;
    const QQ2: f64 = 6.502_224_998_876_729_444_85e-02;
    const QQ3: f64 = 5.081_306_281_875_765_627_76e-03;
    const QQ4: f64 = 1.324_947_380_043_216_445_26e-04;
    const QQ5: f64 = -3.960_228_278_775_368_123_20e-06;
    const PA0: f64 = -2.362_118_560_752_659_440_77e-03;
    const PA1: f64 = 4.148_561_186_837_483_316_66e-01;
    const PA2: f64 = -3.722_078_760_357_013_238_47e-01;
    const PA3: f64 = 3.183_466_199_011_617_536_74e-01;
    const PA4: f64 = -1.108_946_942_823_966_774_76e-01;
    const PA5: f64 = 3.547_830_432_561_823_593_71e-02;
    const PA6: f64 = -2.166_375_594_868_790_843_00e-03;
    const QA1: f64 = 1.064_208_804_008_442_282_86e-01;
    const QA2: f64 = 5.403_979_177_021_710_489_37e-01;
    const QA3: f64 = 7.182_865_441_419_626_628_68e-02;
    const QA4: f64 = 1.261_712_198_087_616_421_12e-01;
    const QA5: f64 = 1.363_708_391_202_905_073_62e-02;
    const QA6: f64 = 1.198_449_984_679_910_741_70e-02;
    const RA0: f64 = -9.864_944_034_847_148_227_05e-03;
    const RA1: f64 = -6.938_585_727_071_817_643_72e-01;
    const RA2: f64 = -1.055_862_622_532_329_098_14e+01;
    const RA3: f64 = -6.237_533_245_032_600_603_96e+01;
    const RA4: f64 = -1.623_966_694_625_734_703_55e+02;
    const RA5: f64 = -1.846_050_929_067_110_359_94e+02;
    const RA6: f64 = -8.128_743_550_630_659_342_46e+01;
    const RA7: f64 = -9.814_329_344_169_145_485_92e+00;
    const SA1: f64 = 1.965_127_166_743_925_712_92e+01;
    const SA2: f64 = 1.376_577_541_435_190_426_00e+02;
    const SA3: f64 = 4.345_658_774_752_292_288_21e+02;
    const SA4: f64 = 6.453_872_717_332_678_803_36e+02;
    const SA5: f64 = 4.290_081_400_275_678_333_86e+02;
    const SA6: f64 = 1.086_350_055_417_794_351_34e+02;
    const SA7: f64 = 6.570_249_770_319_281_701_35e+00;
    const SA8: f64 = -6.042_441_521_485_809_874_38e-02;
    const RB0: f64 = -9.864_942_924_700_099_285_97e-03;
    const RB1: f64 = -7.992_832_376_805_230_065_74e-01;
    const RB2: f64 = -1.775_795_491_775_475_198_89e+01;
    const RB3: f64 = -1.606_363_848_558_219_160_62e+02;
    const RB4: f64 = -6.375_664_433_683_896_277_22e+02;
    const RB5: f64 = -1.025_095_131_611_077_249_54e+03;
    const RB6: f64 = -4.835_191_916_086_513_970_19e+02;
    const SB1: f64 = 3.033_806_074_348_245_829_24e+01;
    const SB2: f64 = 3.257_925_129_965_739_188_26e+02;
    const SB3: f64 = 1.536_729_586_084_436_959_94e+03;
    const SB4: f64 = 3.199_858_219_508_595_539_08e+03;
    const SB5: f64 = 2.553_050_406_433_164_425_83e+03;
    const SB6: f64 = 4.745_285_412_069_553_672_15e+02;
    const SB7: f64 = -2.244_095_244_658_581_833_62e+01;
    const TINY: f64 = 1e-300;

    let hx = (x.to_bits() >> 32) as u32;
    let negative = hx >> 31 != 0;
    let ix = hx & 0x7fff_ffff;
    if ix >= 0x7ff0_0000 {
        // erf(nan) = nan, erf(+-inf) = +-1.
        return if negative { -1.0 } else { 1.0 } + 1.0 / x;
    }
    if ix < 0x3feb_0000 {
        // |x| < 0.84375
        if ix < 0x3e30_0000 {
            // |x| < 2^-28
            if ix < 0x0080_0000 {
                return 0.0625 * (16.0 * x + (16.0 * EFX) * x);
            }
            return EFX.mul_add(x, x);
        }
        let z = x * x;
        let r1 = z.mul_add(PP1, PP0);
        let z2 = z * z;
        let r2 = z.mul_add(PP3, PP2);
        let z4 = z2 * z2;
        let s1 = z.mul_add(QQ1, 1.0);
        let s2 = z.mul_add(QQ3, QQ2);
        let s3 = z.mul_add(QQ5, QQ4);
        let r = z4.mul_add(PP4, z2.mul_add(r2, r1));
        let s = z4.mul_add(s3, z2.mul_add(s2, s1));
        return x.mul_add(r / s, x);
    }
    if ix < 0x3ff4_0000 {
        // 0.84375 <= |x| < 1.25
        let s = x.abs() - 1.0;
        let p1 = s.mul_add(PA1, PA0);
        let s2 = s * s;
        let q1 = s.mul_add(QA1, 1.0);
        let s4 = s2 * s2;
        let p2 = s.mul_add(PA3, PA2);
        let s6 = s4 * s2;
        let q2 = s.mul_add(QA3, QA2);
        let p3 = s.mul_add(PA5, PA4);
        let q3 = s.mul_add(QA5, QA4);
        let p = s6.mul_add(PA6, s4.mul_add(p3, s2.mul_add(p2, p1)));
        let q = s6.mul_add(QA6, s4.mul_add(q3, s2.mul_add(q2, q1)));
        return if negative { -ERX - p / q } else { ERX + p / q };
    }
    if ix >= 0x4018_0000 {
        // |x| >= 6
        return if negative { TINY - 1.0 } else { 1.0 - TINY };
    }
    let ax = x.abs();
    let s = 1.0 / (ax * ax);
    let (r, big_s) = if ix < 0x4006_db6e {
        // |x| < 1/0.35
        let r1 = s.mul_add(RA1, RA0);
        let s2 = s * s;
        let t1 = s.mul_add(SA1, 1.0);
        let s4 = s2 * s2;
        let r2 = s.mul_add(RA3, RA2);
        let s6 = s4 * s2;
        let t2 = s.mul_add(SA3, SA2);
        let s8 = s4 * s4;
        let r3 = s.mul_add(RA5, RA4);
        let t3 = s.mul_add(SA5, SA4);
        let r4 = s.mul_add(RA7, RA6);
        let t4 = s.mul_add(SA7, SA6);
        (
            s6.mul_add(r4, s4.mul_add(r3, s2.mul_add(r2, r1))),
            s8.mul_add(SA8, s6.mul_add(t4, s4.mul_add(t3, s2.mul_add(t2, t1)))),
        )
    } else {
        // |x| >= 1/0.35
        let r1 = s.mul_add(RB1, RB0);
        let s2 = s * s;
        let t1 = s.mul_add(SB1, 1.0);
        let s4 = s2 * s2;
        let r2 = s.mul_add(RB3, RB2);
        let s6 = s4 * s2;
        let t2 = s.mul_add(SB3, SB2);
        let r3 = s.mul_add(RB5, RB4);
        let t3 = s.mul_add(SB5, SB4);
        let t4 = s.mul_add(SB7, SB6);
        (
            s6.mul_add(RB6, s4.mul_add(r3, s2.mul_add(r2, r1))),
            s6.mul_add(t4, s4.mul_add(t3, s2.mul_add(t2, t1))),
        )
    };
    let z = f64::from_bits(ax.to_bits() & 0xffff_ffff_0000_0000);
    let r = (-z).mul_add(z, -0.5625).exp() * (z - ax).mul_add(z + ax, r / big_s).exp();
    if negative { r / ax - 1.0 } else { 1.0 - r / ax }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objective::gradient_pairs;
    use approx::assert_relative_eq;

    #[test]
    fn erf_matches_reference_values() {
        // Values of erf from the defining integral (Abramowitz & Stegun 7.1),
        // one per branch of the rational approximation.
        for (x, want) in [
            (1e-10, 1.128_379_167_095_512_6e-10),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (3.5, 0.999_999_256_901_627_7),
            (7.0, 1.0),
        ] {
            assert_relative_eq!(erf(x), want, max_relative = 2e-16);
            assert_relative_eq!(erf(-x), -want, max_relative = 2e-16);
        }
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert!(erf(f64::NAN).is_nan());
    }

    /// Cox gradients against the Breslow formulas evaluated by hand: rows
    /// (sorted by |y|) with times 1 (event), 2 (censored), 2 (event),
    /// 3 (event), all margins 0 so every risk is 1. Tied time 2 shares the
    /// risk set {2, 2, 3}.
    #[test]
    fn cox_breslow_gradient_with_tie_and_censoring() {
        let labels = [2.0, 1.0, 3.0, -2.0];
        let out = gradient_pairs(&Cox, &[0.0; 4], &labels, None);
        // Risk-set sizes seen by events: time 1 -> 4, time 2 -> 3, time 3 -> 1.
        let r = [
            1.0 / 4.0,
            1.0 / 4.0 + 1.0 / 3.0,
            1.0 / 4.0 + 1.0 / 3.0 + 1.0,
        ];
        let s = [
            1.0 / 16.0,
            1.0 / 16.0 + 1.0 / 9.0,
            1.0 / 16.0 + 1.0 / 9.0 + 1.0,
        ];
        let expect = |r: f64, s: f64, event: bool| {
            GradPair::new((r - if event { 1.0 } else { 0.0 }) as f32, (r - s) as f32)
        };
        assert_eq!(out[1], expect(r[0], s[0], true)); // t=1 event
        assert_eq!(out[0], expect(r[1], s[1], true)); // t=2 event
        // The censored t=2 row follows the t=2 event in the stable |y| order
        // (it comes later in the input), so its terms include that event.
        assert_eq!(out[3], expect(r[1], s[1], false));
        assert_eq!(out[2], expect(r[2], s[2], true)); // t=3 event
    }

    #[test]
    fn cox_weights_scale_gradients() {
        let labels = [1.0, -2.0, 3.0];
        let preds = [0.3, -0.2, 0.1];
        let plain = gradient_pairs(&Cox, &preds, &labels, None);
        let weighted = gradient_pairs(&Cox, &preds, &labels, Some(&[2.0, 0.5, 1.0]));
        for ((p, w), s) in plain.iter().zip(&weighted).zip([2.0f32, 0.5, 1.0]) {
            assert_relative_eq!(w.grad, p.grad * s, max_relative = 1e-6);
            assert_relative_eq!(w.hess, p.hess * s, max_relative = 1e-6);
        }
    }

    /// The analytic AFT gradient and Hessian are the derivatives of the loss
    /// for every censoring type and distribution (central differences).
    #[test]
    fn aft_derivatives_match_loss() {
        let rows = [
            (2.0, 2.0),           // uncensored
            (1.5, f64::INFINITY), // right
            (0.0, 3.0),           // left
            (1.0, 4.0),           // interval
        ];
        for dist in [
            AftDistribution::Normal,
            AftDistribution::Logistic,
            AftDistribution::Extreme,
        ] {
            for &(lo, hi) in &rows {
                for pred in [-0.5, 0.4, 1.2] {
                    let sigma = 0.8;
                    let h = 1e-5;
                    let loss = |m: f64| aft_nloglik(dist, lo, hi, m, sigma);
                    let (grad, hess) = match dist {
                        AftDistribution::Normal => (
                            aft_gradient::<Normal>(lo, hi, pred, sigma),
                            aft_hessian::<Normal>(lo, hi, pred, sigma),
                        ),
                        AftDistribution::Logistic => (
                            aft_gradient::<Logistic>(lo, hi, pred, sigma),
                            aft_hessian::<Logistic>(lo, hi, pred, sigma),
                        ),
                        AftDistribution::Extreme => (
                            aft_gradient::<Extreme>(lo, hi, pred, sigma),
                            aft_hessian::<Extreme>(lo, hi, pred, sigma),
                        ),
                    };
                    let fd_grad = (loss(pred + h) - loss(pred - h)) / (2.0 * h);
                    let fd_hess = (loss(pred + h) - 2.0 * loss(pred) + loss(pred - h)) / (h * h);
                    assert_relative_eq!(grad, fd_grad, epsilon = 1e-6, max_relative = 1e-5);
                    assert_relative_eq!(
                        hess.max(MIN_HESS_F64),
                        fd_hess.max(MIN_HESS_F64),
                        epsilon = 1e-4,
                        max_relative = 1e-3
                    );
                }
            }
        }
    }

    /// Far-off predictions take XGBoost's limits instead of NaN.
    #[test]
    fn aft_extreme_predictions_use_limits() {
        // Uncensored, prediction far below the observed log-time: z >> 0.
        let g = aft_gradient::<Normal>(1.0, 1.0, -100.0, 1.0);
        assert_eq!(g, MIN_GRADIENT);
        assert_eq!(aft_hessian::<Normal>(1.0, 1.0, -100.0, 1.0), 1.0);
        // Right-censored with the prediction far above the bound: the loss
        // vanishes and so does the gradient.
        let g = aft_gradient::<Logistic>(1.0, f64::INFINITY, 1e3, 1.0);
        assert_eq!(g, 0.0);
        assert_eq!(
            aft_hessian::<Logistic>(1.0, f64::INFINITY, 1e3, 1.0),
            MIN_HESS_F64
        );
        // Interval far below the prediction.
        let g = aft_gradient::<Extreme>(1.0, 2.0, 50.0, 1.0);
        assert!(g.is_finite());
    }

    #[test]
    fn aft_reads_bounds_and_weights() {
        let obj = Aft::new(AftDistribution::Normal, 1.0);
        let lower = [1.0, 2.0, 0.0];
        let upper = [1.0, f32::INFINITY, 3.0];
        let weights = [1.0, 2.0, 0.5];
        let preds = [0.1, 0.2, 0.3];
        let info = MetaInfo {
            n_rows: 3,
            label_lower_bound: Some(&lower),
            label_upper_bound: Some(&upper),
            ..MetaInfo::new(&[], Some(&weights), None)
        };
        obj.validate_info(&info).unwrap();
        let mut out = [GradPair::default(); 3];
        obj.gradient_info(&preds, &info, &mut out);
        for i in 0..3 {
            let (lo, hi, p) = (
                f64::from(lower[i]),
                f64::from(upper[i]),
                f64::from(preds[i]),
            );
            let g = aft_gradient::<Normal>(lo, hi, p, 1.0) as f32 * weights[i];
            let h = aft_hessian::<Normal>(lo, hi, p, 1.0) as f32 * weights[i];
            assert_eq!(out[i], GradPair::new(g, h));
        }
        let unbounded = MetaInfo::new(&[1.0], None, None);
        assert!(obj.validate_info(&unbounded).is_err());
        assert_eq!(obj.base_margins_info(&unbounded), vec![0.5f32.ln()]);
    }

    /// `survival:aft` trains from label bounds alone, starts from margin
    /// `ln 0.5`, reports survival times, and refuses an evaluation set
    /// without bounds, naming it.
    #[test]
    fn aft_trains_from_bounds_without_labels() {
        use crate::config::TrainingParams;
        use crate::data::DMatrix;
        use crate::training::{Trainer, train};

        let n = 60;
        let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let t: Vec<f32> = x.iter().map(|&v| (1.0 + 2.0 * v).exp()).collect();
        let upper: Vec<f32> = t
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 4 == 0 { f32::INFINITY } else { v })
            .collect();
        let d = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_label_bounds(&t, &upper)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("survival:aft")
            .max_depth(2)
            .eta(0.5)
            .build()
            .unwrap();
        let model = train(&params, &d, 20).unwrap();
        assert_eq!(model.base_scores(), &[0.5f32.ln()]);
        let pred = model.predict(&d).unwrap();
        let margin = model.predict_margin(&d).unwrap();
        for (p, m) in pred.iter().zip(&margin) {
            assert_eq!(*p, m.exp());
        }
        // The fit follows the (uncensored) times upward.
        assert!(pred[n - 1] > 3.0 * pred[1]);

        let unbounded = crate::test_support::labeled_dense(&x, n, 1, &t);
        let err = Trainer::new(&params, &d, 1)
            .eval(&unbounded, "valid")
            .train()
            .unwrap_err();
        assert!(err.to_string().contains("`valid`"), "{err}");
    }
}
