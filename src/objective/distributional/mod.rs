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
//! ordinary `one_output_per_tree` path, or one shared tree per round with
//! `multi_output_tree`, see below). Each output is an *unconstrained*
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
//!   `m - 12·sd` use `P(Y > k) = 1` and the trigamma difference). When 24
//!   standard deviations span more than 50 000 values, the equivalent
//!   `ψ'(r) - 1/r - E[ψ'(r + Y) - 1/(r + Y) + (Y - m)²/((r + m)²(r + Y))]`
//!   is summed instead over blocks of values, the same blocks as the count
//!   CRPS.
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
//! # Shared trees: parallel gradient boosting
//!
//! With `multi_strategy = multi_output_tree` every round grows *one*
//! vector-leaf tree for all parameters instead of one tree per parameter.
//! [`DistSplitDirection`] (`dist_split_direction`) selects its structure:
//!
//! - [`DistSplitDirection::Random`] (default) and
//!   [`DistSplitDirection::Cyclic`] implement parallel gradient boosting
//!   (Chapelle, Vayatis, Falissard & Sedki, 2026, arXiv:2607.13550,
//!   Algorithm 1). The common descent direction of a round is a canonical
//!   basis vector `e_m`: parameter `m` is drawn uniformly at random from
//!   `seed` and the iteration (the paper's choice), or swept as
//!   `iteration mod n_params`; either visits every parameter infinitely
//!   often, the paper's convergence condition. The tree structure is grown
//!   from that parameter's gradient pairs alone
//!   ([`Objective::split_gradient`], the projected pseudo-residuals
//!   `⟨∇L_i, e_m⟩`), and every leaf then takes the per-parameter Newton step
//!   `-G_k / (H_k + λ)` over its rows. That is the second-order form of the
//!   paper's leaf-wise multidimensional line search `argmin_γ Σ L(g + h γ)`:
//!   with the diagonal curvature of the `dist_gradient` mode the line search
//!   separates across parameters, which the paper's convergence argument
//!   also relies on. With [`DistGradient::Natural`] (unit Hessians) the
//!   structure fit is the paper's least-squares fit of the projected
//!   pseudo-residuals, on the natural gradient.
//! - [`DistSplitDirection::All`]: plain vector-leaf trees, whose split gain
//!   sums over every parameter's gradients.
//!
//! One-parameter families (`dist:poisson`) keep ordinary trees. Shared trees
//! need one structure search per round instead of one per parameter, and
//! all parameters move together each round.
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

use super::{GradPair, MIN_HESS, Objective, SplitGradient, check_label_domain};
use crate::config::{DistGradient, DistSplitDirection};
use crate::data::MetaInfo;
use crate::error::{HessboostError, Result};
use crate::rng::splitmix64;
use special::{
    HALF_LN_2PI, beta_inc, digamma_minus_log, gamma_p, gamma_prefactor, gamma_q, ln_gamma,
    ln_gamma_prefactor, ln_gamma_ratio, ln_norm_cdf, log_gap, norm_cdf, norm_pdf, norm_ppf,
    trigamma_minus_inv,
};

/// Bound on log-link margins: `ln` of a positive parameter is clamped to
/// `[-LOG_LINK_BOUND, LOG_LINK_BOUND]` before the link is applied.
pub const LOG_LINK_BOUND: f64 = 30.0;

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
/// Floor of the second-order statistic: the objectives' [`MIN_HESS`], widened.
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

    /// Whether `y` lies below the family's support (`NaN` does not).
    fn below_support(self, y: f64) -> bool {
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
                [a * (1.0 - t), a * (digamma_minus_log(a) + log_gap(t))]
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
                let g2 = a * (digamma_minus_log(a) + log_gap(t));
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
/// `dml(x) = ψ(x) - ln x` so it keeps its precision as `r` grows. Where
/// `u` approaches `-1` (a count far below a large mean), `ln(1 + u)` is
/// taken as `ln((r + y)/(r + m))`: `1 + u` rounds to zero there.
fn nb_size_score(m: f64, r: f64, y: f64) -> f64 {
    let u = (y - m) / (r + m);
    let ln_ratio = if u < -0.5 {
        ((r + y) / (r + m)).ln()
    } else {
        u.ln_1p()
    };
    (digamma_minus_log(y + r) - digamma_minus_log(r)) + (ln_ratio - u)
}

/// Fisher information of the negative-binomial size `r` (natural scale):
/// `E[ψ'(r) - ψ'(Y + r)] - m/(r(r + m))`, the expectation summed as
/// `Σ_k P(Y > k)/(r + k)²` (see the module docs for the truncation).
/// Supports wider than the unit-step budget take
/// [`nb_size_fisher_blocked`] instead, so the sum always spans the
/// probability-bearing region.
fn nb_size_fisher(m: f64, r: f64) -> f64 {
    let mut blocks = CountBlocks::new(Dist::NegativeBinomial { mean: m, size: r });
    if blocks.h_max > 1.0 {
        return nb_size_fisher_blocked(m, r, blocks);
    }
    // Terms below the start have P(Y > k) = 1: their sum is
    // ψ'(r) - ψ'(r + k0).
    let k0 = blocks.k;
    let mut sum = if k0 > 0.0 {
        (trigamma_minus_inv(r) + 1.0 / r) - (trigamma_minus_inv(r + k0) + 1.0 / (r + k0))
    } else {
        0.0
    };
    let (mut cdf, mut steps) = (0.0f64, 0);
    loop {
        let k = blocks.k;
        cdf = (cdf + blocks.step().mass).min(1.0);
        let survival = (1.0 - cdf).max(0.0);
        sum += survival / ((r + k) * (r + k));
        if k >= m && survival < COUNT_TAIL {
            break;
        }
        steps += 1;
        if steps >= MAX_COUNT_TERMS {
            // Remaining terms decay like the pmf ratio `ρ`.
            let rho = blocks.dist.count_ratio(blocks.k);
            if rho < 1.0 {
                sum += survival * rho / (1.0 - rho) / ((r + k) * (r + k));
            }
            break;
        }
    }
    sum - m / (r * (r + m))
}

/// [`nb_size_fisher`] over the [`CountBlocks`] of a wide support, as
/// `[ψ'(r) - 1/r] - E[ψ'(r + Y) - 1/(r + Y) + (Y - m)²/((r + m)²(r + Y))]`:
/// the same quantity with `1/(r + m) - 1/(r + Y)` split into its
/// zero-mean first-order part (dropped, `E[Y] = m`) and a positive
/// remainder, so the block approximation of the masses only perturbs
/// second-order terms. Each block contributes its mass times the summand
/// at its centre, with `(Y - m)²` averaged over the block's values.
fn nb_size_fisher_blocked(m: f64, r: f64, mut blocks: CountBlocks) -> f64 {
    let s2 = (r + m) * (r + m);
    let (mut mass, mut expectation) = (0.0f64, 0.0f64);
    for _ in 0..MAX_COUNT_TERMS {
        let b = blocks.step();
        let c = b.k + 0.5 * (b.h - 1.0);
        let spread = (c - m) * (c - m) + (b.h * b.h - 1.0) / 12.0;
        expectation += b.mass * (trigamma_minus_inv(r + c) + spread / (s2 * (r + c)));
        mass += b.mass;
        if b.k + b.h > m && b.tail() < COUNT_TAIL * mass {
            break;
        }
    }
    trigamma_minus_inv(r) - expectation / mass
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
        if self.family().below_support(y) {
            return f64::NEG_INFINITY;
        }
        match *self {
            Dist::Normal { mu, sigma } => {
                let z = (y - mu) / sigma;
                -sigma.ln() - HALF_LN_2PI - 0.5 * z * z
            }
            Dist::LogNormal { mu, sigma } => {
                let ly = y.ln();
                let z = (ly - mu) / sigma;
                -ly - sigma.ln() - HALF_LN_2PI - 0.5 * z * z
            }
            // The density is the incomplete-gamma prefactor at `rate·y` over
            // `y`, stable for large shapes.
            Dist::Gamma { mean, shape } => ln_gamma_prefactor(shape, shape * y / mean) - y.ln(),
            Dist::Poisson { rate } => {
                let term = if y == 0.0 { 0.0 } else { y * rate.ln() };
                term - rate - ln_gamma(y + 1.0)
            }
            Dist::NegativeBinomial { mean, size } => {
                let ln_p = -(mean / size).ln_1p();
                let ln_q = mean.ln() - (size + mean).ln();
                let term = if y == 0.0 { 0.0 } else { y * ln_q };
                // `ln Γ(y + r) - ln Γ(r)` without cancellation at large `r`.
                ln_gamma_ratio(size, y) - ln_gamma(y + 1.0) + size * ln_p + term
            }
        }
    }

    /// The CDF `P(Y <= y)` (for counts, of `floor(y)`).
    pub fn cdf(&self, y: f64) -> f64 {
        if self.family().below_support(y) {
            return 0.0;
        }
        match *self {
            Dist::Normal { mu, sigma } => norm_cdf((y - mu) / sigma),
            Dist::LogNormal { mu, sigma } => norm_cdf((y.ln() - mu) / sigma),
            Dist::Gamma { mean, shape } => gamma_p(shape, y * shape / mean),
            Dist::Poisson { .. } | Dist::NegativeBinomial { .. } if y == f64::INFINITY => 1.0,
            Dist::Poisson { rate } => gamma_q(y.floor() + 1.0, rate),
            Dist::NegativeBinomial { mean, size } => {
                let s = size + mean;
                beta_inc(size, y.floor() + 1.0, size / s, mean / s)
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
    /// (Scheuerer & Möller, 2015). The count families sum the integral over
    /// the unit steps of their CDF (on each `[k, k + 1)` the integrand is
    /// constant, split at a non-integer `y`), from 12 standard deviations
    /// below the mean until the upper tail is below `1e-12`, in blocks of
    /// values once the support is wider than 50 000
    /// steps. The remaining geometric tail, and every step between the
    /// support and a `y` far outside it, are added in closed form.
    pub fn crps(&self, y: f64) -> f64 {
        match *self {
            Dist::Normal { mu, sigma } => {
                let z = (y - mu) / sigma;
                sigma * (z * (2.0 * norm_cdf(z) - 1.0) + 2.0 * norm_pdf(z) - FRAC_1_SQRT_PI)
            }
            Dist::LogNormal { mu, sigma } => {
                // `e^{mu + sigma²/2} Φ(t)` in log space, so a large scale
                // meets its small tail probability (below the smallest
                // double, too) without over- or underflow, and no near-one
                // probability is subtracted.
                let ln_scale = mu + 0.5 * sigma * sigma;
                let scaled = |t: f64| (ln_scale + ln_norm_cdf(t)).exp();
                // E[X] - E|X - X'|/2 = 2 e^{mu + sigma²/2} Φ(-sigma/√2), the
                // complement of `2Φ(sigma/√2) - 1` taken directly.
                let spread_tail = scaled(-sigma * std::f64::consts::FRAC_1_SQRT_2);
                if y <= 0.0 {
                    // E|X - y| - E|X - X'|/2 with X > 0 > y.
                    return 2.0 * spread_tail - y;
                }
                let w = (y.ln() - mu) / sigma;
                y * (2.0 * norm_cdf(w) - 1.0) + 2.0 * (spread_tail - scaled(w - sigma))
            }
            Dist::Gamma { mean, shape } => {
                let rate = shape / mean;
                // 1 / B(1/2, a) = Γ(a + 1/2) / (Γ(1/2) Γ(a)).
                let inv_beta = (ln_gamma_ratio(shape, 0.5) - 0.5 * std::f64::consts::PI.ln()).exp();
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
    /// uniform on `(0, 1)` from 52 random bits of `rng` (the midpoints
    /// `(j + 1/2)/2^52`, all exactly representable, so `u` never rounds to
    /// `1`), so a seeded RNG gives a reproducible stream.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u = ((rng.next_u64() >> 12) as f64 + 0.5) * (1.0 / (1u64 << 52) as f64);
        self.quantile(u)
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
        let (mean, sd) = (self.mean(), self.std_dev());
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

    /// Step-sum CRPS of a count distribution (see [`Self::crps`]).
    ///
    /// `F` is constant on each `[k, k + 1)`, so the integral is a sum over
    /// unit steps. It covers `[max(0, mean - 12 sd), end)` in
    /// [`CountBlocks`] (single steps unless 24 standard deviations exceed
    /// half of [`MAX_COUNT_TERMS`]; wider blocks take `F` linear across the
    /// block), normalized by the total mass.
    /// Below the start `F = 0`; from `end` on, `1 - F` decays geometrically
    /// at the pmf ratio, summed in closed form up to and beyond `y` however
    /// far away `y` lies.
    fn count_crps(&self, y: f64) -> f64 {
        let mean = self.mean();
        let first = CountBlocks::new(*self);
        let start = first.k;
        // Pass 1: the number of blocks and the total mass (with the
        // geometric estimate of the mass beyond the last block).
        let mut blocks = first.clone();
        let (mut n_blocks, mut mass, mut rest) = (0usize, 0.0f64, 0.0f64);
        loop {
            let b = blocks.step();
            mass += b.mass;
            n_blocks += 1;
            let tail = b.tail();
            let past_mean = b.k + b.h > mean;
            if (past_mean && tail < COUNT_TAIL * mass) || n_blocks >= MAX_COUNT_TERMS {
                if tail.is_finite() {
                    rest = tail;
                }
                break;
            }
        }
        let total_mass = mass + rest;
        // Below `start` F = 0: the integrand is 1 on `[y, start)`.
        let mut total = (start - y).max(0.0);
        // Pass 2: the blocks again, with the normalized CDF.
        let mut blocks = first;
        let mut cum = 0.0f64;
        for _ in 0..n_blocks {
            let b = blocks.step();
            let before = cum / total_mass;
            cum += b.mass;
            let after = (cum / total_mass).min(1.0);
            total += block_crps(b.k, b.h, before, after, y);
        }
        let end = blocks.k;
        let upper = (rest / total_mass).clamp(0.0, 1.0);
        let rho = self.count_ratio(end);
        total + geometric_tail_crps(upper, if rho < 1.0 { rho } else { 0.0 }, y - end)
    }
}

/// Relative width of the [`CountBlocks`] below their widest: a block at
/// `k` spans at most `(k + 1) / 1000` values.
const BLOCK_GROWTH: f64 = 1e-3;

/// One block of [`CountBlocks`].
struct CountBlock {
    /// First value.
    k: f64,
    /// Number of values.
    h: f64,
    /// Probability mass.
    mass: f64,
    /// Width of the next block.
    next_h: f64,
    /// `ln P(c') - ln P(c)` from this block's centre `c` to the next one's.
    ln_step: f64,
}

impl CountBlock {
    /// Geometric estimate of the mass beyond this block, at the ratio of
    /// the next block's mass to this one's (`∞` unless it is below one).
    fn tail(&self) -> f64 {
        let next_ratio = self.next_h / self.h * self.ln_step.exp();
        if next_ratio < 1.0 {
            self.mass * next_ratio / (1.0 - next_ratio)
        } else {
            f64::INFINITY
        }
    }
}

/// Blocks of consecutive values of a count distribution from
/// `max(0, ⌊mean - 12 sd⌋)` upward, with their probability masses. A block
/// at `k` spans `min(h_max, max(1, ⌊(k + 1)/1000⌋))` values: single values
/// in the head, where a skewed distribution (size below one) is sharp and
/// its mass concentrates, then widths growing in proportion to `k`, so the
/// log pmf (whose curvature there is of order `1/k²`) stays nearly linear
/// across every block. Single values are exact (the pmf recursion); a
/// wider block's mass is its width times the pmf at its centre, the centres
/// stepped by the midpoint rule on the log pmf ratio.
#[derive(Clone)]
struct CountBlocks {
    dist: Dist,
    /// Widest block: `1` (single values) unless 24 standard deviations
    /// exceed half of [`MAX_COUNT_TERMS`], else the width that fits them.
    h_max: f64,
    /// First value of the next block.
    k: f64,
    /// Width of the next block.
    h: f64,
    /// `ln P(Y = c)` at the next block's centre `c = k + (h - 1)/2`.
    ln_center: f64,
}

impl CountBlocks {
    fn new(dist: Dist) -> Self {
        let sd = dist.std_dev();
        let start = (dist.mean() - COUNT_HEAD_SDS * sd).floor().max(0.0);
        let h_max = (2.0 * COUNT_HEAD_SDS * sd / (MAX_COUNT_TERMS / 2) as f64)
            .ceil()
            .max(1.0);
        let h = Self::width(h_max, start);
        CountBlocks {
            dist,
            h_max,
            k: start,
            h,
            ln_center: dist.log_prob(start + 0.5 * (h - 1.0)),
        }
    }

    /// Width of the block starting at `k`.
    fn width(h_max: f64, k: f64) -> f64 {
        (BLOCK_GROWTH * (k + 1.0)).floor().clamp(1.0, h_max)
    }

    /// Visit the next block.
    fn step(&mut self) -> CountBlock {
        let (k, h) = (self.k, self.h);
        let mass = h * self.ln_center.exp();
        let next_h = Self::width(self.h_max, k + h);
        // The next centre lies `d` values on:
        // ln P(c + d) - ln P(c) = Σ_{t < d} ln ratio(c + t), by the midpoint.
        let d = f64::midpoint(h, next_h);
        let c = k + 0.5 * (h - 1.0);
        let ln_step = d * self.dist.count_ratio(c + 0.5 * (d - 1.0)).ln();
        self.ln_center += ln_step;
        self.k += h;
        self.h = next_h;
        CountBlock {
            k,
            h,
            mass,
            next_h,
            ln_step,
        }
    }
}

/// `Σ_{i0 <= i < i1} (a + d(i + 1))²`.
fn square_sum(a: f64, d: f64, i0: f64, i1: f64) -> f64 {
    if i1 <= i0 {
        return 0.0;
    }
    let squares = |x: f64| x * (x + 1.0) * (2.0 * x + 1.0) / 6.0;
    let (s0, s1) = (i0 + 1.0, i1);
    let n = s1 - s0 + 1.0;
    n * a * a + a * d * (s0 + s1) * n + d * d * (squares(s1) - squares(s0 - 1.0))
}

/// CRPS integrand over the block `[k, k + h)` of unit steps whose CDF rises
/// linearly from `before` (below `k`) to `after` (at its last step): unit
/// `i` has `F = before + (after - before)(i + 1)/h`, contributing `F²` below
/// `y` and `(1 - F)²` above it (split at a fractional `y`).
fn block_crps(k: f64, h: f64, before: f64, after: f64, y: f64) -> f64 {
    let d = (after - before) / h;
    let below = |i0, i1| square_sum(before, d, i0, i1);
    let above = |i0, i1| square_sum(1.0 - before, -d, i0, i1);
    if y <= k {
        return above(0.0, h);
    }
    if y >= k + h {
        return below(0.0, h);
    }
    let n = (y - k).floor();
    let f = y - k - n;
    let cdf = before + d * (n + 1.0);
    let upper = 1.0 - cdf;
    below(0.0, n) + f * cdf * cdf + (1.0 - f) * upper * upper + above(n + 1.0, h)
}

/// CRPS integrand beyond the summed support, `[end, ∞)`, with `y = end +
/// m`: unit `t >= 1` (`[end + t - 1, end + t)`) has `1 - F = u ρ^t`, the
/// geometric tail of the remaining mass `u` at pmf ratio `ρ < 1`. Units
/// below `y` contribute `(1 - u ρ^t)²`, the rest `(u ρ^t)²`, in closed form
/// however large `m` is.
fn geometric_tail_crps(u: f64, rho: f64, m: f64) -> f64 {
    let r2 = rho * rho;
    // Σ_{s >= t} (u ρ^s)².
    let beyond = |t: f64| u * u * rho.powf(2.0 * t) / (1.0 - r2);
    if m <= 0.0 {
        return beyond(1.0);
    }
    let n = m.floor();
    let f = m - n;
    let full = n - 2.0 * u * rho * (1.0 - rho.powf(n)) / (1.0 - rho)
        + u * u * r2 * (1.0 - rho.powf(2.0 * n)) / (1.0 - r2);
    let q = u * rho.powf(n + 1.0);
    full + f * (1.0 - q) * (1.0 - q) + (1.0 - f) * q * q + beyond(n + 2.0)
}

/// Quantile of the unit-scale Gamma(`a`, 1): Halley iterations on
/// `P(a, x) = p` from the *Numerical Recipes* (`invgammp`) start, which in
/// the deep lower tail (where its Wilson–Hilferty start fails) is the
/// small-`x` asymptote `P(a, x) ≈ x^a / Γ(a + 1)`, a lower bound of the
/// quantile. Should Halley not converge, a log-space bisection finishes.
fn gamma_unit_quantile(a: f64, p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let a1 = a - 1.0;
    let mut x = if a > 1.0 {
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * pp.ln()).sqrt();
        let mut z = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t;
        if p < 0.5 {
            z = -z;
        }
        let wilson_hilferty = a * (1.0 - 1.0 / (9.0 * a) - z / (3.0 * a.sqrt())).powi(3);
        if p < 0.5 {
            let asymptote = ((p.ln() + ln_gamma(a + 1.0)) / a).exp();
            if wilson_hilferty > 1e-3 {
                wilson_hilferty.max(asymptote)
            } else {
                asymptote
            }
        } else {
            wilson_hilferty.max(1e-3)
        }
    } else {
        let t = 1.0 - a * (0.253 + a * 0.12);
        if p < t {
            (p / t).powf(1.0 / a)
        } else {
            1.0 - (1.0 - (p - t) / (1.0 - t)).ln()
        }
    };
    // Residual on the smaller tail for precision; increasing in `x`.
    let residual = |x: f64| {
        if p < 0.5 {
            gamma_p(a, x) - p
        } else {
            (1.0 - p) - gamma_q(a, x)
        }
    };
    for _ in 0..100 {
        if x <= 0.0 {
            return 0.0;
        }
        let density = gamma_prefactor(a, x) / x;
        if density == 0.0 || !density.is_finite() {
            break;
        }
        let u = residual(x) / density;
        let t = u / (1.0 - 0.5 * (u * (a1 / x - 1.0)).min(1.0));
        x -= t;
        if x <= 0.0 {
            x = f64::midpoint(x, t);
        }
        if t.abs() < 1e-15 * x {
            return x;
        }
    }
    bisect_quantile(x, residual)
}

/// The root of the increasing `residual`: a bracket grown by halving and
/// doubling from `x`, then bisection of `ln x` to adjacent floats.
fn bisect_quantile(x: f64, residual: impl Fn(f64) -> f64) -> f64 {
    let (mut lo, mut hi) = (x.max(f64::MIN_POSITIVE), x.max(f64::MIN_POSITIVE));
    while residual(lo) > 0.0 {
        lo *= 0.5;
        if lo == 0.0 {
            return 0.0;
        }
    }
    while residual(hi) < 0.0 {
        hi *= 2.0;
        if hi == f64::INFINITY {
            return hi;
        }
    }
    for _ in 0..200 {
        let mid = f64::midpoint(lo.ln(), hi.ln()).exp();
        if !(lo < mid && mid < hi) {
            break;
        }
        if residual(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

/// A `dist:*` objective: the negative log-likelihood of a [`DistFamily`],
/// with the second-order statistic chosen by [`DistGradient`] (see the
/// [module docs](self)).
#[derive(Debug, Clone, Copy)]
pub struct DistObjective {
    family: DistFamily,
    gradient: DistGradient,
    /// Parallel-gradient-boosting direction and seed for shared trees, set
    /// only for `multi_strategy = multi_output_tree`.
    shared: Option<(DistSplitDirection, u64)>,
}

impl DistObjective {
    /// The objective for `family` with gradient mode `gradient`, growing one
    /// tree per parameter (no reduced split gradients).
    pub fn new(family: DistFamily, gradient: DistGradient) -> Self {
        DistObjective {
            family,
            gradient,
            shared: None,
        }
    }

    /// Grow shared vector-leaf trees (`multi_strategy = multi_output_tree`)
    /// with the given split direction: [`Objective::split_gradient`] then
    /// returns the gradients of the parameter the direction selects for the
    /// round (`seed` drives [`DistSplitDirection::Random`]), or `None` for
    /// [`DistSplitDirection::All`] and one-parameter families.
    #[must_use]
    pub fn with_split_direction(mut self, direction: DistSplitDirection, seed: u64) -> Self {
        self.shared = Some((direction, seed));
        self
    }

    /// The parameter whose gradients drive the structure of round
    /// `iteration`'s shared tree, if any.
    pub fn split_parameter(&self, iteration: usize) -> Option<usize> {
        let k = self.family.n_params();
        match self.shared? {
            _ if k < 2 => None,
            (DistSplitDirection::All, _) => None,
            (DistSplitDirection::Cyclic, _) => Some(iteration % k),
            (DistSplitDirection::Random, seed) => {
                let draw = splitmix64(seed ^ splitmix64(iteration as u64));
                Some((draw % k as u64) as usize)
            }
        }
    }

    /// The distribution family.
    pub fn family(&self) -> DistFamily {
        self.family
    }

    /// One row's `(gradient, curvature)` per parameter, before row weights.
    fn row_pairs(self, eta: &[f64], y: f64) -> ([f64; 2], [f64; 2]) {
        let g = self.family.gradient(eta, y);
        match self.gradient {
            DistGradient::Fisher => (g, self.family.fisher(eta).map(|v| v.max(MIN_CURVATURE))),
            DistGradient::Hessian => {
                let h = self.family.hessian(eta, y);
                (g, [h[0][0], h[1][1]].map(|v| v.max(MIN_CURVATURE)))
            }
            DistGradient::Natural => {
                let i = self.family.fisher(eta).map(|v| v.max(MIN_CURVATURE));
                ([g[0] / i[0], g[1] / i[1]], [1.0; 2])
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
        check_label_domain(info, |y| self.family.below_support(f64::from(y)))
    }

    fn default_metric(&self) -> String {
        "nll".to_string()
    }

    /// Parallel gradient boosting (Chapelle et al., 2026): the chosen
    /// parameter's gradient column, `⟨∇L_i, e_m⟩` per row.
    fn split_gradient(&self, iteration: usize, gpair: &[GradPair]) -> Option<SplitGradient> {
        let m = self.split_parameter(iteration)?;
        let k = self.family.n_params();
        Some(SplitGradient {
            gpair: gpair.iter().skip(m).step_by(k).copied().collect(),
            n_targets: 1,
        })
    }
}

#[cfg(test)]
mod tests;
