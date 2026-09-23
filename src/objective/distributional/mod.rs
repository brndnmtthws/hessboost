//! Distributional (probabilistic) boosting: objectives that predict a full
//! conditional distribution `p(y | x)` instead of a point, in the style of
//! NGBoost (Duan et al., 2020, arXiv:1910.03225) and XGBoostLSS (März, 2019,
//! arXiv:1907.03178). Opt-in and beyond XGBoost: the objective names live
//! outside XGBoost's namespace, and XGBoost-format export refuses them.
//!
//! # Families and parameterizations
//!
//! Objective `dist:<family>` gives the model one output per distribution
//! parameter (`n_outputs = n_params`, one tree per parameter and round on the
//! ordinary `one_output_per_tree` path). Each output is an *unconstrained*
//! margin `η`; positive parameters use a log link. Log-link margins are
//! clamped to `[-30, 30]` ([`LOG_LINK_BOUND`]) before the link, in the
//! gradients as in prediction, so no parameter over- or underflows.
//!
//! | objective | [`DistFamily`] | margins | natural parameters |
//! |---|---|---|---|
//! | `dist:normal` | `Normal` | `(μ, ln σ)` | mean `μ`, standard deviation `σ` |
//! | `dist:lognormal` | `LogNormal` | `(μ, ln σ)` | `ln y ~ N(μ, σ²)` |
//! | `dist:gamma` | `Gamma` | `(ln m, ln a)` | mean `m`, shape `a` (rate `a / m`) |
//! | `dist:poisson` | `Poisson` | `ln λ` | rate `λ` |
//! | `dist:negbinomial` | `NegativeBinomial` | `(ln m, ln r)` | mean `m`, size `r` (variance `m + m²/r`) |
//!
//! The parameterizations are chosen *orthogonal*: the Fisher information of
//! each family is diagonal in its margins, so the diagonal Fisher that fits
//! XGBoost's per-output second-order trees is the full Fisher matrix, and
//! the per-row natural gradient `I(η)⁻¹ ∇η` is elementwise.
//!
//! # Gradients and the "Hessian" (`dist_gradient`)
//!
//! The loss is the negative log-likelihood `-ln p(y | η)` (the log scoring
//! rule). Every mode uses its exact gradient `g = ∇η NLL`; the
//! [`DistGradient`] parameter selects what the trees see as the second-order
//! statistic:
//!
//! - [`DistGradient::Fisher`] (default): `(g, diag I(η))`, the expected
//!   Hessian (Fisher scoring). Always positive, and independent of the label.
//!   Because the Fisher matrix is diagonal here, a leaf's Newton step
//!   `-Σg / (ΣI + λ)` is a natural-gradient step.
//! - [`DistGradient::Hessian`]: `(g, diag ∇²η NLL)`, the diagonal of the
//!   exact (observed) Hessian as XGBoostLSS uses it. The negative-binomial
//!   size entry turns negative for counts far above the mean and the
//!   Normal / LogNormal `ln σ` entry `2z²` vanishes at `y = μ`; values are
//!   floored at `1e-16`, like XGBoost's own objectives.
//! - [`DistGradient::Natural`]: NGBoost's natural gradient, `(I(η)⁻¹ g, 1)`:
//!   trees regress the per-row natural gradient by (weighted) least
//!   squares, and `eta` is the step size (no line search).
//!
//! Row weights multiply both statistics. The per-family formulas (`t = y/m`,
//! `z = (y - μ)/σ`, `ψ` digamma, `ψ'` trigamma):
//!
//! - Normal: `g = (-z/σ, 1 - z²)`, `I = (1/σ², 2)`, exact diagonal
//!   `(1/σ², 2z²)`. LogNormal is the same on `ln y`.
//! - Gamma: `g = (a(1 - t), a(ψ(a) - ln a + t - 1 - ln t))`,
//!   `I = (a, a²(ψ'(a) - 1/a))`, exact diagonal `(a t, g₂ + a²(ψ'(a) - 1/a))`.
//! - Poisson: `g = λ - y`, `I = λ` (exactly `count:poisson` without its
//!   `max_delta_step` Hessian inflation).
//! - Negative binomial: `g₁ = r(m - y)/(r + m)`, `g₂ = -r D` with
//!   `D = ψ(y + r) - ψ(r) + ln(r/(r + m)) + (m - y)/(r + m)`;
//!   `I₁ = m r/(m + r)` and `I₂ = r² (E[ψ'(r) - ψ'(Y + r)] - m/(r(r + m)))`.
//!   The expectation has no closed form: it is the series
//!   `Σ_k P(Y > k)/(r + k)²`, summed from the probability mass function
//!   until the upper tail falls below `1e-12` (at most 100 000 terms, then
//!   closed with a geometric-tail estimate; the terms below
//!   `m - 12·sd` use `P(Y > k) = 1` and the trigamma difference).
//!
//! # Intercepts
//!
//! The intercept is the maximum-likelihood fit of the *marginal* (weighted)
//! label distribution: sample mean and (biased) standard deviation for
//! Normal / LogNormal (on `ln y`), the mean and the shape solving
//! `ln a - ψ(a) = ln ȳ - mean(ln y)` for Gamma, the mean for Poisson, and
//! the mean and the size solving the profile score equation (by bisection
//! on `ln r`) for the negative binomial. A scalar `base_score` is only
//! accepted for the one-parameter `dist:poisson` (as its rate).
//!
//! # Predictions
//!
//! [`BoostedModel::predict`](crate::learner::BoostedModel::predict) returns
//! the natural parameters `[row][parameter]` (the table's last column);
//! [`BoostedModel::predict_distribution`](crate::learner::BoostedModel::predict_distribution)
//! returns one [`Dist`] per row with its mean, variance, CDF, quantiles,
//! log density, CRPS, intervals, and inverse-CDF sampling. Metrics `nll`
//! (the default) and `crps` score them.

mod special;

use rand::Rng;
use serde::{Deserialize, Serialize};

use super::{GradPair, MIN_HESS, Objective, check_label_domain};
use crate::config::DistGradient;
use crate::data::MetaInfo;
use crate::error::{HessboostError, Result};
use special::{
    beta_inc, digamma_minus_log, gamma_p, gamma_q, ln_gamma, norm_cdf, norm_pdf, norm_ppf,
    trigamma_minus_inv,
};

/// Bound on log-link margins: `ln` of a positive parameter is clamped to
/// `[-LOG_LINK_BOUND, LOG_LINK_BOUND]` before the link is applied.
pub const LOG_LINK_BOUND: f64 = 30.0;

/// `ln(2π) / 2`.
const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;
/// `1 / √π`.
const FRAC_1_SQRT_PI: f64 = 0.564_189_583_547_756_3;
/// Upper-tail probability below which the count sums stop.
const COUNT_TAIL: f64 = 1e-12;
/// Most probability-mass terms any count sum takes before it closes the
/// remainder with a geometric-tail estimate.
const MAX_COUNT_TERMS: usize = 100_000;
/// Standard deviations below the mean from which count sums start (the mass
/// below is below `1e-30`).
const COUNT_HEAD_SDS: f64 = 12.0;
/// Floor of the second-order statistic (XGBoost's `kRtEps`-style guard).
const MIN_CURVATURE: f64 = MIN_HESS as f64;

/// A parametric distribution family for the `dist:*` objectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistFamily {
    /// Normal `N(μ, σ²)`, margins `(μ, ln σ)` (`dist:normal`).
    Normal,
    /// Log-normal, `ln y ~ N(μ, σ²)`, margins `(μ, ln σ)` (`dist:lognormal`).
    LogNormal,
    /// Gamma with mean `m` and shape `a`, margins `(ln m, ln a)`
    /// (`dist:gamma`).
    Gamma,
    /// Poisson with rate `λ`, margin `ln λ` (`dist:poisson`).
    Poisson,
    /// Negative binomial (NB2) with mean `m` and size `r`, variance
    /// `m + m²/r`, margins `(ln m, ln r)` (`dist:negbinomial`).
    NegativeBinomial,
}

impl DistFamily {
    /// Every family.
    pub const ALL: [DistFamily; 5] = [
        DistFamily::Normal,
        DistFamily::LogNormal,
        DistFamily::Gamma,
        DistFamily::Poisson,
        DistFamily::NegativeBinomial,
    ];

    /// The family of objective `name` (`"dist:normal"`, ...), if any.
    pub fn from_objective(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.objective_name() == name)
    }

    /// The objective name, e.g. `"dist:normal"`.
    pub fn objective_name(self) -> &'static str {
        match self {
            DistFamily::Normal => "dist:normal",
            DistFamily::LogNormal => "dist:lognormal",
            DistFamily::Gamma => "dist:gamma",
            DistFamily::Poisson => "dist:poisson",
            DistFamily::NegativeBinomial => "dist:negbinomial",
        }
    }

    /// Number of distribution parameters (the model's outputs).
    pub fn n_params(self) -> usize {
        self.param_names().len()
    }

    /// The natural parameters, in output order.
    pub fn param_names(self) -> &'static [&'static str] {
        match self {
            DistFamily::Normal | DistFamily::LogNormal => &["mu", "sigma"],
            DistFamily::Gamma => &["mean", "shape"],
            DistFamily::Poisson => &["rate"],
            DistFamily::NegativeBinomial => &["mean", "size"],
        }
    }

    /// Whether parameter `j` uses a log link (the others are the identity).
    fn log_link(self, j: usize) -> bool {
        !(matches!(self, DistFamily::Normal | DistFamily::LogNormal) && j == 0)
    }

    /// Whether the family is supported on the non-negative integers.
    pub fn is_discrete(self) -> bool {
        matches!(self, DistFamily::Poisson | DistFamily::NegativeBinomial)
    }

    /// Natural parameters from margins (log links clamped to
    /// [`LOG_LINK_BOUND`]).
    fn link(self, eta: &[f64]) -> [f64; 2] {
        let mut out = [0.0; 2];
        for (j, (o, &e)) in out.iter_mut().zip(eta).enumerate() {
            *o = if self.log_link(j) {
                e.clamp(-LOG_LINK_BOUND, LOG_LINK_BOUND).exp()
            } else {
                e
            };
        }
        out
    }

    /// The distribution with margins `eta` (length [`Self::n_params`]).
    pub(crate) fn dist_from_margins(self, eta: &[f64]) -> Dist {
        let p = self.link(eta);
        Dist::from_natural(self, p[0], p[1])
    }

    /// The label must lie in the family's support.
    fn invalid_label(self, y: f32) -> bool {
        match self {
            DistFamily::Normal => false,
            DistFamily::LogNormal | DistFamily::Gamma => y <= 0.0,
            DistFamily::Poisson | DistFamily::NegativeBinomial => y < 0.0,
        }
    }

    /// Negative log-likelihood of `y` at margins `eta`.
    #[cfg(test)]
    pub(crate) fn nll(self, eta: &[f64], y: f64) -> f64 {
        -self.dist_from_margins(eta).log_prob(y)
    }

    /// Gradient of [`Self::nll`] with respect to the margins.
    pub(crate) fn gradient(self, eta: &[f64], y: f64) -> [f64; 2] {
        let p = self.link(eta);
        match self {
            DistFamily::Normal => normal_gradient(p[0], p[1], y),
            DistFamily::LogNormal => normal_gradient(p[0], p[1], y.ln()),
            DistFamily::Gamma => {
                let (m, a) = (p[0], p[1]);
                let t = y / m;
                [
                    a * (1.0 - t),
                    a * (digamma_minus_log(a) + shifted_log_gap(t)),
                ]
            }
            DistFamily::Poisson => [p[0] - y, 0.0],
            DistFamily::NegativeBinomial => {
                let (m, r) = (p[0], p[1]);
                [r * (m - y) / (r + m), -r * nb_size_score(m, r, y)]
            }
        }
    }

    /// Diagonal of the Fisher information in the margins (the full matrix:
    /// every parameterization is orthogonal).
    pub(crate) fn fisher(self, eta: &[f64]) -> [f64; 2] {
        let p = self.link(eta);
        match self {
            DistFamily::Normal | DistFamily::LogNormal => [1.0 / (p[1] * p[1]), 2.0],
            DistFamily::Gamma => {
                let a = p[1];
                [a, a * a * trigamma_minus_inv(a)]
            }
            DistFamily::Poisson => [p[0], 0.0],
            DistFamily::NegativeBinomial => {
                let (m, r) = (p[0], p[1]);
                [m * r / (m + r), r * r * nb_size_fisher(m, r)]
            }
        }
    }

    /// The exact Hessian of [`Self::nll`] in the margins (row-major 2×2; the
    /// unused entries of a one-parameter family are zero).
    pub(crate) fn hessian(self, eta: &[f64], y: f64) -> [[f64; 2]; 2] {
        let p = self.link(eta);
        match self {
            DistFamily::Normal => normal_hessian(p[1], (y - p[0]) / p[1]),
            DistFamily::LogNormal => normal_hessian(p[1], (y.ln() - p[0]) / p[1]),
            DistFamily::Gamma => {
                let (m, a) = (p[0], p[1]);
                let t = y / m;
                let g2 = a * (digamma_minus_log(a) + shifted_log_gap(t));
                let h12 = a * (1.0 - t);
                [[a * t, h12], [h12, g2 + a * a * trigamma_minus_inv(a)]]
            }
            DistFamily::Poisson => [[p[0], 0.0], [0.0, 0.0]],
            DistFamily::NegativeBinomial => {
                let (m, r) = (p[0], p[1]);
                let s = r + m;
                let h11 = m * r * (r + y) / (s * s);
                let h12 = r * m * (m - y) / (s * s);
                let d = nb_size_score(m, r, y);
                let d_prime = trigamma_minus_inv(y + r) - trigamma_minus_inv(r)
                    + (m - y) * (m - y) / (s * s * (r + y));
                [[h11, h12], [h12, -r * d - r * r * d_prime]]
            }
        }
    }

    /// Maximum-likelihood margins of the marginal (weighted) distribution of
    /// `labels`, clamped like the link.
    pub(crate) fn mle_margins(self, labels: &[f32], weights: Option<&[f32]>) -> Vec<f64> {
        let k = self.n_params();
        let w = |i: usize| weights.map_or(1.0, |ws| f64::from(ws[i]));
        let sum_w: f64 = (0..labels.len()).map(w).sum();
        if labels.is_empty() || sum_w.is_nan() || sum_w <= 0.0 {
            return vec![0.0; k];
        }
        let mean_of = |f: &dyn Fn(f64) -> f64| -> f64 {
            labels
                .iter()
                .enumerate()
                .map(|(i, &y)| w(i) * f(f64::from(y)))
                .sum::<f64>()
                / sum_w
        };
        let log = |x: f64| x.ln().clamp(-LOG_LINK_BOUND, LOG_LINK_BOUND);
        match self {
            DistFamily::Normal | DistFamily::LogNormal => {
                let tr = |y: f64| {
                    if self == DistFamily::LogNormal {
                        y.ln()
                    } else {
                        y
                    }
                };
                let mu = mean_of(&|y| tr(y));
                let var = mean_of(&|y| (tr(y) - mu).powi(2));
                vec![mu, log(var.sqrt())]
            }
            DistFamily::Gamma => {
                let m = mean_of(&|y| y);
                let s = m.ln() - mean_of(&f64::ln);
                vec![log(m), log(gamma_shape_mle(s))]
            }
            DistFamily::Poisson => vec![log(mean_of(&|y| y))],
            DistFamily::NegativeBinomial => {
                let m = mean_of(&|y| y).max((-LOG_LINK_BOUND).exp());
                let score = |rho: f64| -> f64 {
                    let r = rho.exp();
                    labels
                        .iter()
                        .enumerate()
                        .map(|(i, &y)| w(i) * nb_size_score(m, r, f64::from(y)))
                        .sum()
                };
                vec![log(m), nb_size_mle(score)]
            }
        }
    }
}

/// `t - 1 - ln t` without cancellation near `t = 1`.
fn shifted_log_gap(t: f64) -> f64 {
    let u = t - 1.0;
    u - u.ln_1p()
}

fn normal_gradient(mu: f64, sigma: f64, y: f64) -> [f64; 2] {
    let z = (y - mu) / sigma;
    [-z / sigma, 1.0 - z * z]
}

fn normal_hessian(sigma: f64, z: f64) -> [[f64; 2]; 2] {
    let h12 = 2.0 * z / sigma;
    [[1.0 / (sigma * sigma), h12], [h12, 2.0 * z * z]]
}

/// `D = ψ(y + r) - ψ(r) + ln(r/(r + m)) + (m - y)/(r + m)`, the negated
/// derivative of the negative-binomial NLL in `r`, written as
/// `[dml(y + r) - dml(r)] + [ln(1 + u) - u]` with `u = (y - m)/(r + m)` and
/// `dml(x) = ψ(x) - ln x` so it keeps its precision as `r` grows.
fn nb_size_score(m: f64, r: f64, y: f64) -> f64 {
    let u = (y - m) / (r + m);
    (digamma_minus_log(y + r) - digamma_minus_log(r)) + (u.ln_1p() - u)
}

/// Fisher information of the negative-binomial size `r` (natural scale):
/// `E[ψ'(r) - ψ'(Y + r)] - m/(r(r + m))`, the expectation summed as
/// `Σ_k P(Y > k)/(r + k)²` (see the module docs for the truncation).
fn nb_size_fisher(m: f64, r: f64) -> f64 {
    let dist = Dist::NegativeBinomial { mean: m, size: r };
    let mut walk = CountWalk::new(dist);
    // Terms below the walk's start have P(Y > k) = 1: their sum is
    // ψ'(r) - ψ'(r + k0).
    let k0 = walk.k;
    let mut sum = if k0 > 0.0 {
        (trigamma_minus_inv(r) + 1.0 / r) - (trigamma_minus_inv(r + k0) + 1.0 / (r + k0))
    } else {
        0.0
    };
    let mut steps = 0;
    loop {
        let (k, _, cdf) = walk.step();
        let survival = (1.0 - cdf).max(0.0);
        sum += survival / ((r + k) * (r + k));
        if k >= m && survival < COUNT_TAIL {
            break;
        }
        steps += 1;
        if steps >= MAX_COUNT_TERMS {
            // Remaining terms decay like the pmf ratio `ρ`.
            let rho = walk.ratio();
            if rho < 1.0 {
                sum += survival * rho / (1.0 - rho) / ((r + k) * (r + k));
            }
            break;
        }
    }
    sum - m / (r * (r + m))
}

/// Gamma shape MLE: the root of `ln a - ψ(a) = s` (`s = ln ȳ - mean(ln y)`
/// `>= 0`), by Newton's method on `ln a` from Minka's closed-form start.
fn gamma_shape_mle(s: f64) -> f64 {
    let upper = LOG_LINK_BOUND.exp();
    if s.is_nan() || s <= 0.0 {
        return upper;
    }
    let mut a = (3.0 - s + ((s - 3.0) * (s - 3.0) + 24.0 * s).sqrt()) / (12.0 * s);
    for _ in 0..100 {
        // f(ln a) = -dml(a) - s, df/d ln a = -a (ψ'(a) - 1/a).
        let f = -digamma_minus_log(a) - s;
        let df = -a * trigamma_minus_inv(a);
        let step = f / df;
        let next = (a.ln() - step).clamp(-LOG_LINK_BOUND, LOG_LINK_BOUND).exp();
        let done = (next / a - 1.0).abs() < 1e-14;
        a = next;
        if done {
            break;
        }
    }
    a
}

/// Negative-binomial size MLE in log space: the root of the summed size
/// score `Σ w D(ln r)` by bisection on `[-LOG_LINK_BOUND, LOG_LINK_BOUND]`;
/// a bound when the score keeps its sign (e.g. `+` bound for
/// under-dispersed labels, where the likelihood increases towards the
/// Poisson limit).
fn nb_size_mle(score: impl Fn(f64) -> f64) -> f64 {
    let (mut lo, mut hi) = (-LOG_LINK_BOUND, LOG_LINK_BOUND);
    // The NLL decreases in r while Σ D > 0.
    if score(hi) >= 0.0 {
        return hi;
    }
    if score(lo) <= 0.0 {
        return lo;
    }
    for _ in 0..200 {
        let mid = f64::midpoint(lo, hi);
        if score(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-12 {
            break;
        }
    }
    f64::midpoint(lo, hi)
}

/// A distribution predicted for one row, in its natural parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dist {
    /// Normal `N(mu, sigma²)`.
    Normal {
        /// Mean.
        mu: f64,
        /// Standard deviation.
        sigma: f64,
    },
    /// Log-normal: `ln y ~ N(mu, sigma²)`.
    LogNormal {
        /// Mean of `ln y`.
        mu: f64,
        /// Standard deviation of `ln y`.
        sigma: f64,
    },
    /// Gamma with the given mean and shape (rate `shape / mean`).
    Gamma {
        /// Mean.
        mean: f64,
        /// Shape `a`.
        shape: f64,
    },
    /// Poisson with the given rate (mean).
    Poisson {
        /// Rate `λ`.
        rate: f64,
    },
    /// Negative binomial (NB2) with the given mean and size `r`; variance
    /// `mean + mean² / r`.
    NegativeBinomial {
        /// Mean.
        mean: f64,
        /// Size (dispersion) `r > 0`.
        size: f64,
    },
}

impl Dist {
    /// Build a distribution of `family` from its natural parameters (in
    /// [`DistFamily::param_names`] order).
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] if `params` has the wrong length,
    /// a parameter is not finite, or a scale/shape/mean parameter is not
    /// positive.
    pub fn new(family: DistFamily, params: &[f64]) -> Result<Self> {
        if params.len() != family.n_params() {
            return Err(HessboostError::invalid_param(
                "params",
                format!(
                    "{} takes {} parameters, got {}",
                    family.objective_name(),
                    family.n_params(),
                    params.len()
                ),
            ));
        }
        for (j, &p) in params.iter().enumerate() {
            if !p.is_finite() || (family.log_link(j) && p <= 0.0) {
                return Err(HessboostError::invalid_param(
                    "params",
                    format!(
                        "`{}` of {} must be finite{}, got {p}",
                        family.param_names()[j],
                        family.objective_name(),
                        if family.log_link(j) {
                            " and positive"
                        } else {
                            ""
                        }
                    ),
                ));
            }
        }
        Ok(Self::from_natural(
            family,
            params[0],
            params.get(1).copied().unwrap_or(0.0),
        ))
    }

    fn from_natural(family: DistFamily, p0: f64, p1: f64) -> Self {
        match family {
            DistFamily::Normal => Dist::Normal { mu: p0, sigma: p1 },
            DistFamily::LogNormal => Dist::LogNormal { mu: p0, sigma: p1 },
            DistFamily::Gamma => Dist::Gamma {
                mean: p0,
                shape: p1,
            },
            DistFamily::Poisson => Dist::Poisson { rate: p0 },
            DistFamily::NegativeBinomial => Dist::NegativeBinomial { mean: p0, size: p1 },
        }
    }

    /// The distribution from one row of natural parameters as
    /// [`BoostedModel::predict`](crate::learner::BoostedModel::predict)
    /// reports them (no validation).
    pub(crate) fn from_row(family: DistFamily, row: &[f32]) -> Self {
        Self::from_natural(
            family,
            f64::from(row[0]),
            row.get(1).map_or(0.0, |&v| f64::from(v)),
        )
    }

    /// The family.
    pub fn family(&self) -> DistFamily {
        match self {
            Dist::Normal { .. } => DistFamily::Normal,
            Dist::LogNormal { .. } => DistFamily::LogNormal,
            Dist::Gamma { .. } => DistFamily::Gamma,
            Dist::Poisson { .. } => DistFamily::Poisson,
            Dist::NegativeBinomial { .. } => DistFamily::NegativeBinomial,
        }
    }

    /// The natural parameters in [`DistFamily::param_names`] order.
    pub fn params(&self) -> Vec<f64> {
        match *self {
            Dist::Normal { mu, sigma } | Dist::LogNormal { mu, sigma } => vec![mu, sigma],
            Dist::Gamma { mean, shape } => vec![mean, shape],
            Dist::Poisson { rate } => vec![rate],
            Dist::NegativeBinomial { mean, size } => vec![mean, size],
        }
    }

    /// The mean `E[Y]`.
    pub fn mean(&self) -> f64 {
        match *self {
            Dist::Normal { mu, .. } => mu,
            Dist::LogNormal { mu, sigma } => (mu + 0.5 * sigma * sigma).exp(),
            Dist::Gamma { mean, .. } | Dist::NegativeBinomial { mean, .. } => mean,
            Dist::Poisson { rate } => rate,
        }
    }

    /// The variance `Var[Y]`.
    pub fn variance(&self) -> f64 {
        match *self {
            Dist::Normal { sigma, .. } => sigma * sigma,
            Dist::LogNormal { mu, sigma } => {
                let s2 = sigma * sigma;
                s2.exp_m1() * (2.0 * mu + s2).exp()
            }
            Dist::Gamma { mean, shape } => mean * mean / shape,
            Dist::Poisson { rate } => rate,
            Dist::NegativeBinomial { mean, size } => mean + mean * mean / size,
        }
    }

    /// The standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Log density (continuous families) or log probability mass (counts) at
    /// `y`; `-∞` outside the support. The count families evaluate
    /// `ln Γ(y + 1)`, so a non-integer `y` gets the continuous extension the
    /// training loss uses.
    pub fn log_prob(&self, y: f64) -> f64 {
        match *self {
            Dist::Normal { mu, sigma } => {
                let z = (y - mu) / sigma;
                -sigma.ln() - HALF_LN_2PI - 0.5 * z * z
            }
            Dist::LogNormal { mu, sigma } => {
                if y <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                let ly = y.ln();
                let z = (ly - mu) / sigma;
                -ly - sigma.ln() - HALF_LN_2PI - 0.5 * z * z
            }
            Dist::Gamma { mean, shape } => {
                if y <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                let rate = shape / mean;
                shape * rate.ln() - ln_gamma(shape) + (shape - 1.0) * y.ln() - rate * y
            }
            Dist::Poisson { rate } => {
                if y < 0.0 {
                    return f64::NEG_INFINITY;
                }
                let term = if y == 0.0 { 0.0 } else { y * rate.ln() };
                term - rate - ln_gamma(y + 1.0)
            }
            Dist::NegativeBinomial { mean, size } => {
                if y < 0.0 {
                    return f64::NEG_INFINITY;
                }
                let ln_p = -(mean / size).ln_1p();
                let ln_q = mean.ln() - (size + mean).ln();
                let term = if y == 0.0 { 0.0 } else { y * ln_q };
                ln_gamma(y + size) - ln_gamma(size) - ln_gamma(y + 1.0) + size * ln_p + term
            }
        }
    }

    /// The CDF `P(Y <= y)` (for counts, of `floor(y)`).
    pub fn cdf(&self, y: f64) -> f64 {
        match *self {
            Dist::Normal { mu, sigma } => norm_cdf((y - mu) / sigma),
            Dist::LogNormal { mu, sigma } => {
                if y <= 0.0 {
                    0.0
                } else {
                    norm_cdf((y.ln() - mu) / sigma)
                }
            }
            Dist::Gamma { mean, shape } => {
                if y <= 0.0 {
                    0.0
                } else {
                    gamma_p(shape, y * shape / mean)
                }
            }
            Dist::Poisson { rate } => {
                if y < 0.0 {
                    0.0
                } else {
                    gamma_q(y.floor() + 1.0, rate)
                }
            }
            Dist::NegativeBinomial { mean, size } => {
                if y < 0.0 {
                    0.0
                } else {
                    let s = size + mean;
                    beta_inc(size, y.floor() + 1.0, size / s, mean / s)
                }
            }
        }
    }

    /// The quantile function: the smallest `y` with `cdf(y) >= p` (an
    /// integer for the count families). `p = 0` gives the lower end of the
    /// support (`-∞` for Normal), `p = 1` gives `+∞`; `NaN` outside `[0, 1]`.
    pub fn quantile(&self, p: f64) -> f64 {
        if p.is_nan() || !(0.0..=1.0).contains(&p) {
            return f64::NAN;
        }
        match *self {
            Dist::Normal { mu, sigma } => mu + sigma * norm_ppf(p),
            Dist::LogNormal { mu, sigma } => (mu + sigma * norm_ppf(p)).exp(),
            Dist::Gamma { mean, shape } => gamma_unit_quantile(shape, p) * mean / shape,
            Dist::Poisson { .. } | Dist::NegativeBinomial { .. } => self.count_quantile(p),
        }
    }

    /// The central interval holding probability `coverage`:
    /// `(quantile((1 - coverage)/2), quantile((1 + coverage)/2))`.
    pub fn interval(&self, coverage: f64) -> (f64, f64) {
        (
            self.quantile(0.5 * (1.0 - coverage)),
            self.quantile(f64::midpoint(1.0, coverage)),
        )
    }

    /// Continuous ranked probability score `∫ (F(s) - 1{s >= y})² ds`.
    ///
    /// Closed forms for Normal, LogNormal (Baran & Lerch, 2015) and Gamma
    /// (Scheuerer & Möller, 2015). The count families sum the integral
    /// exactly over the unit steps of their CDF (on each `[k, k + 1)` the
    /// integrand is constant, split at a non-integer `y`), starting 12
    /// standard deviations below the mean and stopping once the upper tail
    /// is below `1e-12` (or after 100 000 steps, closed with a geometric-tail
    /// estimate).
    pub fn crps(&self, y: f64) -> f64 {
        match *self {
            Dist::Normal { mu, sigma } => {
                let z = (y - mu) / sigma;
                sigma * (z * (2.0 * norm_cdf(z) - 1.0) + 2.0 * norm_pdf(z) - FRAC_1_SQRT_PI)
            }
            Dist::LogNormal { mu, sigma } => {
                let scale = (mu + 0.5 * sigma * sigma).exp();
                let half = 2.0 * norm_cdf(sigma * std::f64::consts::FRAC_1_SQRT_2) - 1.0;
                if y <= 0.0 {
                    // E|X - y| - E|X - X'|/2 with X > 0 > y.
                    return scale - y - scale * half;
                }
                let w = (y.ln() - mu) / sigma;
                y * (2.0 * norm_cdf(w) - 1.0)
                    - 2.0
                        * scale
                        * (norm_cdf(w - sigma) + norm_cdf(sigma * std::f64::consts::FRAC_1_SQRT_2)
                            - 1.0)
            }
            Dist::Gamma { mean, shape } => {
                let rate = shape / mean;
                // 1 / B(1/2, a) = Γ(a + 1/2) / (Γ(1/2) Γ(a)).
                let inv_beta =
                    (ln_gamma(shape + 0.5) - ln_gamma(shape) - 0.5 * std::f64::consts::PI.ln())
                        .exp();
                let (f_a, f_a1) = if y <= 0.0 {
                    (0.0, 0.0)
                } else {
                    (gamma_p(shape, rate * y), gamma_p(shape + 1.0, rate * y))
                };
                y * (2.0 * f_a - 1.0) - mean * (2.0 * f_a1 - 1.0) - inv_beta / rate
            }
            Dist::Poisson { .. } | Dist::NegativeBinomial { .. } => self.count_crps(y),
        }
    }

    /// Draw one value by inverse-CDF sampling, `quantile(u)` with `u`
    /// uniform on `(0, 1)` from 53 random bits of `rng`, so a seeded RNG
    /// gives a reproducible stream.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u = ((rng.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64);
        self.quantile(u)
    }

    /// Mean and standard deviation of a count distribution.
    fn count_moments(&self) -> (f64, f64) {
        (self.mean(), self.std_dev())
    }

    /// `ln P(Y = k)` for the count families.
    fn count_ln_pmf(&self, k: f64) -> f64 {
        self.log_prob(k)
    }

    /// `P(Y = k + 1) / P(Y = k)` for the count families.
    fn count_ratio(&self, k: f64) -> f64 {
        match *self {
            Dist::Poisson { rate } => rate / (k + 1.0),
            Dist::NegativeBinomial { mean, size } => (k + size) / (k + 1.0) * mean / (size + mean),
            _ => unreachable!("count_ratio on a continuous family"),
        }
    }

    /// Smallest integer `k >= 0` with `cdf(k) >= p`: bracket from the normal
    /// approximation by doubling steps, then bisect.
    fn count_quantile(&self, p: f64) -> f64 {
        if p >= 1.0 {
            return f64::INFINITY;
        }
        let (mean, sd) = self.count_moments();
        let guess = (mean + sd * norm_ppf(p)).round().max(0.0);
        let (mut lo, mut hi);
        if self.cdf(guess) >= p {
            // Find lo with cdf(lo) < p (or lo = -1).
            hi = guess;
            let mut step = 1.0;
            lo = hi - step;
            while lo >= 0.0 && self.cdf(lo) >= p {
                hi = lo;
                step *= 2.0;
                lo = (hi - step).max(-1.0);
            }
            lo = lo.max(-1.0);
        } else {
            lo = guess;
            let mut step = 1.0;
            hi = lo + step;
            while self.cdf(hi) < p {
                lo = hi;
                step *= 2.0;
                hi = lo + step;
                if !hi.is_finite() || hi > 1e300 {
                    return f64::INFINITY;
                }
            }
        }
        // Invariant: cdf(lo) < p <= cdf(hi) (cdf(-1) = 0).
        while hi - lo > 1.0 {
            let mid = f64::midpoint(lo, hi).floor();
            if self.cdf(mid) >= p {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        hi
    }

    /// Exact step-sum CRPS of a count distribution (see [`Self::crps`]).
    fn count_crps(&self, y: f64) -> f64 {
        let mut walk = CountWalk::new(*self);
        let k0 = walk.k;
        // Below 0 the CDF is 0: ∫_y^0 1 ds. Below k0 it is (numerically) 0,
        // so every unit step above y contributes 1.
        let mut total = (-y).max(0.0) + (k0 - y.max(0.0)).max(0.0);
        let mut steps = 0;
        loop {
            let (k, _, cdf) = walk.step();
            let upper = 1.0 - cdf;
            total += if y <= k {
                upper * upper
            } else if y >= k + 1.0 {
                cdf * cdf
            } else {
                (y - k) * cdf * cdf + (k + 1.0 - y) * upper * upper
            };
            if k + 1.0 >= y && upper < COUNT_TAIL {
                break;
            }
            steps += 1;
            if steps >= MAX_COUNT_TERMS {
                let rho = walk.ratio();
                if k + 1.0 >= y && rho < 1.0 {
                    total += upper * upper * rho * rho / (1.0 - rho * rho);
                }
                break;
            }
        }
        total
    }
}

/// Walks the probability mass function of a count distribution upward from
/// `max(0, floor(mean - 12 sd))`, tracking the CDF (mass below the start is
/// treated as zero).
struct CountWalk {
    dist: Dist,
    /// The next value to visit.
    k: f64,
    /// `ln P(Y = k)`.
    ln_pmf: f64,
    /// `P(Y < k)` accumulated since the start.
    below: f64,
}

impl CountWalk {
    fn new(dist: Dist) -> Self {
        let (mean, sd) = dist.count_moments();
        let k = (mean - COUNT_HEAD_SDS * sd).floor().max(0.0);
        CountWalk {
            dist,
            k,
            ln_pmf: dist.count_ln_pmf(k),
            below: 0.0,
        }
    }

    /// Visit the next value: `(k, P(Y = k), P(Y <= k))`.
    fn step(&mut self) -> (f64, f64, f64) {
        let k = self.k;
        let pmf = self.ln_pmf.exp();
        self.below = (self.below + pmf).min(1.0);
        self.ln_pmf += self.dist.count_ratio(k).ln();
        self.k += 1.0;
        (k, pmf, self.below)
    }

    /// The pmf ratio at the current position.
    fn ratio(&self) -> f64 {
        self.dist.count_ratio(self.k)
    }
}

/// Quantile of the unit-scale Gamma(`a`, 1): Halley iterations on
/// `P(a, x) = p` from the *Numerical Recipes* (`invgammp`) start.
fn gamma_unit_quantile(a: f64, p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let gln = ln_gamma(a);
    let a1 = a - 1.0;
    let (lna1, afac) = if a > 1.0 {
        let lna1 = a1.ln();
        (lna1, (a1 * (lna1 - 1.0) - gln).exp())
    } else {
        (0.0, 0.0)
    };
    let mut x = if a > 1.0 {
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * pp.ln()).sqrt();
        let mut z = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t;
        if p < 0.5 {
            z = -z;
        }
        (a * (1.0 - 1.0 / (9.0 * a) - z / (3.0 * a.sqrt())).powi(3)).max(1e-3)
    } else {
        let t = 1.0 - a * (0.253 + a * 0.12);
        if p < t {
            (p / t).powf(1.0 / a)
        } else {
            1.0 - (1.0 - (p - t) / (1.0 - t)).ln()
        }
    };
    for _ in 0..100 {
        if x <= 0.0 {
            return 0.0;
        }
        // Residual on the smaller tail for precision.
        let err = if p < 0.5 {
            gamma_p(a, x) - p
        } else {
            (1.0 - p) - gamma_q(a, x)
        };
        let density = if a > 1.0 {
            afac * (-(x - a1) + a1 * (x.ln() - lna1)).exp()
        } else {
            (-x + a1 * x.ln() - gln).exp()
        };
        if density == 0.0 {
            break;
        }
        let u = err / density;
        let t = u / (1.0 - 0.5 * (u * (a1 / x - 1.0)).min(1.0));
        x -= t;
        if x <= 0.0 {
            x = f64::midpoint(x, t);
        }
        if t.abs() < 1e-15 * x {
            break;
        }
    }
    x
}

/// A `dist:*` objective: the negative log-likelihood of a [`DistFamily`],
/// with the second-order statistic chosen by [`DistGradient`] (see the
/// [module docs](self)).
#[derive(Debug, Clone, Copy)]
pub struct DistObjective {
    family: DistFamily,
    gradient: DistGradient,
}

impl DistObjective {
    /// The objective for `family` with gradient mode `gradient`.
    pub fn new(family: DistFamily, gradient: DistGradient) -> Self {
        DistObjective { family, gradient }
    }

    /// The distribution family.
    pub fn family(&self) -> DistFamily {
        self.family
    }

    /// One row's `(gradient, curvature)` per parameter, before row weights.
    fn row_pairs(self, eta: &[f64], y: f64) -> ([f64; 2], [f64; 2]) {
        let g = self.family.gradient(eta, y);
        match self.gradient {
            DistGradient::Fisher => {
                let i = self.family.fisher(eta);
                (g, i.map(|v| v.max(MIN_CURVATURE)))
            }
            DistGradient::Hessian => {
                let h = self.family.hessian(eta, y);
                (g, [h[0][0], h[1][1]].map(|v| v.max(MIN_CURVATURE)))
            }
            DistGradient::Natural => {
                let i = self.family.fisher(eta);
                (
                    [
                        g[0] / i[0].max(MIN_CURVATURE),
                        g[1] / i[1].max(MIN_CURVATURE),
                    ],
                    [1.0; 2],
                )
            }
        }
    }
}

impl Objective for DistObjective {
    fn name(&self) -> &str {
        self.family.objective_name()
    }

    fn n_outputs(&self) -> usize {
        self.family.n_params()
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        let k = self.family.n_params();
        let n = labels.len();
        super::check_gradient_inputs(n, k, preds, labels, weights, out);
        super::rowwise_gradient(
            n,
            k,
            preds,
            labels,
            weights,
            out,
            |preds, labels, weights, out| {
                for (i, (row, out_row)) in preds
                    .chunks_exact(k)
                    .zip(out.chunks_exact_mut(k))
                    .enumerate()
                {
                    let w = weights.map_or(1.0, |ws| f64::from(ws[i]));
                    if w == 0.0 {
                        out_row.fill(GradPair::default());
                        continue;
                    }
                    let mut eta = [0.0; 2];
                    for (e, &m) in eta.iter_mut().zip(row) {
                        *e = f64::from(m);
                    }
                    let (g, h) = self.row_pairs(&eta[..k], f64::from(labels[i]));
                    for (j, o) in out_row.iter_mut().enumerate() {
                        *o = GradPair::new((w * g[j]) as f32, (w * h[j]) as f32);
                    }
                }
            },
        );
    }

    /// Margins to natural parameters (the log links, clamped).
    fn pred_transform(&self, preds: &mut [f32]) {
        let k = self.family.n_params();
        for row in preds.chunks_exact_mut(k) {
            for (j, v) in row.iter_mut().enumerate() {
                if self.family.log_link(j) {
                    *v = f64::from(*v).clamp(-LOG_LINK_BOUND, LOG_LINK_BOUND).exp() as f32;
                }
            }
        }
    }

    /// Natural parameters to margins (inverse links; a non-positive
    /// parameter maps to `NaN`, which training rejects).
    fn probs_to_margins(&self, scores: &mut [f32]) {
        let k = self.family.n_params();
        for row in scores.chunks_exact_mut(k) {
            for (j, v) in row.iter_mut().enumerate() {
                if self.family.log_link(j) {
                    *v = if *v > 0.0 { v.ln() } else { f32::NAN };
                }
            }
        }
    }

    /// Maximum-likelihood fit of the marginal label distribution.
    fn base_margins(
        &self,
        labels: &[f32],
        weights: Option<&[f32]>,
        _group: Option<&crate::data::GroupInfo>,
    ) -> Vec<f32> {
        self.family
            .mle_margins(labels, weights)
            .into_iter()
            .map(|m| m as f32)
            .collect()
    }

    fn validate_info(&self, info: &MetaInfo) -> Result<()> {
        check_label_domain(info, |y| self.family.invalid_label(y))
    }

    fn default_metric(&self) -> String {
        "nll".to_string()
    }
}

#[cfg(test)]
mod tests;
