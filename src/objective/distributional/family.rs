//! The distribution families: their metadata, links, likelihood
//! derivatives in the margins, and marginal maximum-likelihood fits.

use serde::{Deserialize, Serialize};

use std::f64::consts::PI;

use super::LOG_LINK_BOUND;
use super::count::{COUNT_TAIL, CountBlocks, MAX_COUNT_TERMS};
use super::loss::MIN_CURVATURE;
use super::special::{
    digamma_minus_log, half_digamma_gap, half_trigamma_gap, ln_gamma_ratio, ln1p_minus_ratio,
    log_gap, trigamma_minus_inv,
};

/// The most parameters any family has (the Student-t's three): the length
/// of the fixed-size gradient, Fisher, and margin buffers. Entries beyond a
/// family's [`DistFamily::n_params`] are zero.
pub(crate) const MAX_PARAMS: usize = 3;

/// Declares the distribution families once: the [`DistFamily`] variant, its
/// objective name, and the [`Dist`] variant whose fields are the natural
/// parameters in output order ([`DistFamily::param_names`]). Generates
/// [`DistFamily::ALL`], [`DistFamily::objective_name`],
/// [`DistFamily::param_names`], `Dist::from_natural`, [`Dist::family`], and
/// [`Dist::params`]; each family's formulas stay explicit match arms.
macro_rules! dist_families {
    ($(
        $(#[$family_doc:meta])*
        $family:ident = $objective:literal,
        $(#[$dist_doc:meta])*
        { $($(#[$param_doc:meta])* $param:ident),+ $(,)? }
    )+) => {
        /// A parametric distribution family for the `dist:*` objectives. More
        /// families may be added; [`DistFamily::ALL`] lists the current ones.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        #[non_exhaustive]
        pub enum DistFamily {
            $($(#[$family_doc])* $family,)+
        }

        /// A distribution predicted for one row, in its natural parameters.
        ///
        /// One variant per [`DistFamily`], so it gains a variant with every new
        /// family: match the families you train (with a `_` arm), or read any
        /// distribution through [`Dist::family`] and [`Dist::params`].
        #[derive(Debug, Clone, Copy, PartialEq)]
        #[non_exhaustive]
        pub enum Dist {
            $(
                $(#[$dist_doc])*
                $family {
                    $($(#[$param_doc])* $param: f64,)+
                },
            )+
        }

        impl DistFamily {
            /// Every family (a slice, so adding a family keeps its type).
            pub const ALL: &'static [DistFamily] = &[$(DistFamily::$family),+];

            /// The objective name, e.g. `"dist:normal"`.
            pub fn objective_name(self) -> &'static str {
                match self {
                    $(DistFamily::$family => $objective,)+
                }
            }

            /// The natural parameters, in output order.
            pub fn param_names(self) -> &'static [&'static str] {
                match self {
                    $(DistFamily::$family => &[$(stringify!($param)),+],)+
                }
            }
        }

        impl Dist {
            /// The distribution of `family` with natural parameters
            /// `natural` in output order (missing trailing ones read as
            /// zero, extra ones are ignored), unvalidated.
            pub(super) fn from_natural(family: DistFamily, natural: &[f64]) -> Self {
                match family {
                    $(DistFamily::$family => {
                        let mut next = natural.iter().copied();
                        Dist::$family {
                            $($param: next.next().unwrap_or(0.0),)+
                        }
                    })+
                }
            }

            /// The family.
            pub fn family(&self) -> DistFamily {
                match self {
                    $(Dist::$family { .. } => DistFamily::$family,)+
                }
            }

            /// The natural parameters in [`DistFamily::param_names`] order.
            pub fn params(&self) -> Vec<f64> {
                match *self {
                    $(Dist::$family { $($param),+ } => vec![$($param),+],)+
                }
            }
        }
    };
}

dist_families! {
    /// Normal `N(μ, σ²)`, margins `(μ, ln σ)` (`dist:normal`).
    Normal = "dist:normal",
    /// Normal `N(mu, sigma²)`.
    {
        /// Mean.
        mu,
        /// Standard deviation.
        sigma,
    }
    /// Log-normal, `ln y ~ N(μ, σ²)`, margins `(μ, ln σ)` (`dist:lognormal`).
    LogNormal = "dist:lognormal",
    /// Log-normal: `ln y ~ N(mu, sigma²)`.
    {
        /// Mean of `ln y`.
        mu,
        /// Standard deviation of `ln y`.
        sigma,
    }
    /// Gamma with mean `m` and shape `a`, margins `(ln m, ln a)`
    /// (`dist:gamma`).
    Gamma = "dist:gamma",
    /// Gamma with the given mean and shape (rate `shape / mean`).
    {
        /// Mean.
        mean,
        /// Shape `a`.
        shape,
    }
    /// Poisson with rate `λ`, margin `ln λ` (`dist:poisson`).
    Poisson = "dist:poisson",
    /// Poisson with the given rate (mean).
    {
        /// Rate `λ`.
        rate,
    }
    /// Negative binomial (NB2) with mean `m` and size `r`, variance
    /// `m + m²/r`, margins `(ln m, ln r)` (`dist:negbinomial`).
    NegativeBinomial = "dist:negbinomial",
    /// Negative binomial (NB2) with the given mean and size `r`; variance
    /// `mean + mean² / r`.
    {
        /// Mean.
        mean,
        /// Size (dispersion) `r > 0`.
        size,
    }
    /// Student-t with location `μ`, scale `σ`, and `ν` degrees of
    /// freedom, margins `(μ, ln s, κ = ln(ν - 1))` with the orthogonal scale
    /// `s = σ (ν + 1)/ν` (`dist:student_t`). The link keeps every fitted
    /// `ν > 1`: as `ν → 0` the likelihood degenerates and the location and
    /// scale steps grow without bound, and the mean needs `ν > 1`. Under
    /// [`DistGradient::Fisher`](super::DistGradient::Fisher) and
    /// [`DistGradient::Natural`](super::DistGradient::Natural) the `κ`
    /// curvature is the Fisher information floored at the magnitude of the
    /// row's exact observed curvature: the information vanishes at both
    /// ends of the link (like `(ν - 1)²` and `1/ν²`) faster than the score.
    StudentT = "dist:student_t",
    /// Student-t: `(y - loc)/scale` follows Student's t distribution with
    /// `df` degrees of freedom (heavier tails for smaller `df`, the normal
    /// as `df → ∞`).
    {
        /// Location `μ` (the median, and the mean when `df > 1`).
        loc,
        /// Scale `σ > 0`.
        scale,
        /// Degrees of freedom `ν > 0`.
        df,
    }
}

impl DistFamily {
    /// The family of objective `name` (`"dist:normal"`, ...), if any.
    pub fn from_objective(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|f| f.objective_name() == name)
    }

    /// Number of distribution parameters (the model's outputs).
    pub fn n_params(self) -> usize {
        self.param_names().len()
    }

    /// Whether parameter `j` uses a log link (the others are the identity).
    pub(super) fn log_link(self, j: usize) -> bool {
        !(matches!(
            self,
            DistFamily::Normal | DistFamily::LogNormal | DistFamily::StudentT
        ) && j == 0)
    }

    /// Whether the family is supported on the non-negative integers.
    pub fn is_discrete(self) -> bool {
        matches!(self, DistFamily::Poisson | DistFamily::NegativeBinomial)
    }

    /// Natural parameters from margins (log links clamped to
    /// [`LOG_LINK_BOUND`]; the Student-t `ν = 1 + e^κ` and its scale from
    /// the orthogonal margin `ln s`, `σ = s ν/(ν + 1)`).
    pub(super) fn natural_params(self, eta: &[f64]) -> [f64; MAX_PARAMS] {
        let mut out = [0.0; MAX_PARAMS];
        for (j, (o, &e)) in out.iter_mut().zip(eta).enumerate() {
            *o = if self.log_link(j) {
                e.clamp(-LOG_LINK_BOUND, LOG_LINK_BOUND).exp()
            } else {
                e
            };
        }
        if self == DistFamily::StudentT {
            out[2] += 1.0;
            out[1] = student_t_scale(out[1], out[2]);
        }
        out
    }

    /// The distribution with margins `eta` (length [`Self::n_params`]).
    pub(crate) fn dist_from_margins(self, eta: &[f64]) -> Dist {
        Dist::from_natural(self, &self.natural_params(eta))
    }

    /// Whether `y` lies below the family's support (`NaN` does not).
    pub(super) fn below_support(self, y: f64) -> bool {
        match self {
            DistFamily::Normal | DistFamily::StudentT => false,
            DistFamily::LogNormal | DistFamily::Gamma => y <= 0.0,
            DistFamily::Poisson | DistFamily::NegativeBinomial => y < 0.0,
        }
    }

    /// Negative log-likelihood of `y` at margins `eta`.
    #[cfg(test)]
    pub(crate) fn nll(self, eta: &[f64], y: f64) -> f64 {
        -self.dist_from_margins(eta).log_prob(y)
    }

    /// Gradient of [`Self::nll`] with respect to the margins (entries past
    /// [`Self::n_params`] are zero).
    pub(crate) fn gradient(self, eta: &[f64], y: f64) -> [f64; MAX_PARAMS] {
        let p = self.natural_params(eta);
        match self {
            DistFamily::Normal => normal_gradient(p[0], p[1], y),
            DistFamily::LogNormal => normal_gradient(p[0], p[1], y.ln()),
            DistFamily::Gamma => {
                let (m, a) = (p[0], p[1]);
                let t = y / m;
                [a * (1.0 - t), a * (digamma_minus_log(a) + log_gap(t)), 0.0]
            }
            DistFamily::Poisson => [p[0] - y, 0.0, 0.0],
            DistFamily::NegativeBinomial => {
                let (m, r) = (p[0], p[1]);
                [r * (m - y) / (r + m), -r * nb_size_score(m, r, y), 0.0]
            }
            DistFamily::StudentT => StudentLikelihood::at_margins(eta).gradient(y),
        }
    }

    /// Diagonal of the Fisher information in the margins (the full matrix:
    /// every parameterization is orthogonal; entries past
    /// [`Self::n_params`] are zero).
    pub(crate) fn fisher(self, eta: &[f64]) -> [f64; MAX_PARAMS] {
        let p = self.natural_params(eta);
        match self {
            DistFamily::Normal | DistFamily::LogNormal => [1.0 / (p[1] * p[1]), 2.0, 0.0],
            DistFamily::Gamma => {
                let a = p[1];
                [a, a * a * trigamma_minus_inv(a), 0.0]
            }
            DistFamily::Poisson => [p[0], 0.0, 0.0],
            DistFamily::NegativeBinomial => {
                let (m, r) = (p[0], p[1]);
                [m * r / (m + r), r * r * nb_size_fisher(m, r), 0.0]
            }
            DistFamily::StudentT => StudentLikelihood::at_margins(eta).fisher(),
        }
    }

    /// The curvature of the [`DistGradient::Fisher`](super::DistGradient::Fisher)
    /// and [`DistGradient::Natural`](super::DistGradient::Natural) modes: the
    /// Fisher diagonal, with the Student-t `κ` entry floored at the
    /// magnitude of the row's exact observed curvature (see the module docs).
    pub(crate) fn fisher_curvature(self, eta: &[f64], y: f64) -> [f64; MAX_PARAMS] {
        if self == DistFamily::StudentT {
            StudentLikelihood::at_margins(eta).fisher_curvature(y)
        } else {
            self.fisher(eta)
        }
    }

    /// The exact Hessian of [`Self::nll`] in the margins (row-major
    /// `MAX_PARAMS × MAX_PARAMS`; the entries past [`Self::n_params`] are
    /// zero).
    pub(crate) fn hessian(self, eta: &[f64], y: f64) -> [[f64; MAX_PARAMS]; MAX_PARAMS] {
        let p = self.natural_params(eta);
        match self {
            DistFamily::Normal => normal_hessian(p[1], (y - p[0]) / p[1]),
            DistFamily::LogNormal => normal_hessian(p[1], (y.ln() - p[0]) / p[1]),
            DistFamily::Gamma => {
                let (m, a) = (p[0], p[1]);
                let t = y / m;
                let g2 = a * (digamma_minus_log(a) + log_gap(t));
                let h12 = a * (1.0 - t);
                [
                    [a * t, h12, 0.0],
                    [h12, g2 + a * a * trigamma_minus_inv(a), 0.0],
                    [0.0; MAX_PARAMS],
                ]
            }
            DistFamily::Poisson => [[p[0], 0.0, 0.0], [0.0; MAX_PARAMS], [0.0; MAX_PARAMS]],
            DistFamily::NegativeBinomial => {
                let (m, r) = (p[0], p[1]);
                let s = r + m;
                let h11 = m * r * (r + y) / (s * s);
                let h12 = r * m * (m - y) / (s * s);
                let d = nb_size_score(m, r, y);
                let d_prime = trigamma_minus_inv(y + r) - trigamma_minus_inv(r)
                    + (m - y) * (m - y) / (s * s * (r + y));
                [
                    [h11, h12, 0.0],
                    [h12, -r * d - r * r * d_prime, 0.0],
                    [0.0; MAX_PARAMS],
                ]
            }
            DistFamily::StudentT => StudentLikelihood::at_margins(eta).hessian(y),
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
            DistFamily::StudentT => student_t_mle(labels, &w, sum_w),
        }
    }
}

pub(super) fn normal_gradient(mu: f64, sigma: f64, y: f64) -> [f64; MAX_PARAMS] {
    let z = (y - mu) / sigma;
    [-z / sigma, 1.0 - z * z, 0.0]
}

pub(super) fn normal_hessian(sigma: f64, z: f64) -> [[f64; MAX_PARAMS]; MAX_PARAMS] {
    let h12 = 2.0 * z / sigma;
    [
        [1.0 / (sigma * sigma), h12, 0.0],
        [h12, 2.0 * z * z, 0.0],
        [0.0; MAX_PARAMS],
    ]
}

/// `σ = s ν/(ν + 1)`: the Student-t scale from the orthogonal scale `s` of
/// its margin `ln s` (Cox & Reid, 1987), which makes the Fisher
/// information diagonal in `(μ, ln s, κ)` for any smooth link `ν(κ)`.
fn student_t_scale(s: f64, nu: f64) -> f64 {
    s * (nu / (nu + 1.0))
}

/// The Student-t likelihood at natural parameters `(μ, σ, ν)`, with its
/// label-independent functions of `ν` evaluated once, differentiated in
/// the margins `(μ, ln s, κ = ln(ν - 1))` (see the module docs for the
/// formulas).
#[derive(Debug, Clone, Copy)]
pub(super) struct StudentLikelihood {
    mu: f64,
    sigma: f64,
    nu: f64,
    /// `ν - 1 = e^κ = dν/dκ`, from the margin (not from `ν`, where it would
    /// lose its precision as `ν → 1`).
    excess: f64,
    /// [`half_digamma_gap`]`(ν) = ψ(ν/2) - ψ((ν + 1)/2) + 1/ν`.
    digamma_gap: f64,
    /// [`half_trigamma_gap`]`(ν)`: `(ν - 1)²` times it is the `κ` Fisher
    /// information.
    trigamma_gap: f64,
    /// `ln Γ((ν + 1)/2) - ln Γ(ν/2) - ln(νπ)/2 - ln σ`, the log density at
    /// `y = μ`.
    ln_peak: f64,
}

impl StudentLikelihood {
    /// The likelihood at margins `eta = (μ, ln s, κ)`.
    pub(super) fn at_margins(eta: &[f64]) -> Self {
        let [mu, sigma, nu] = DistFamily::StudentT.natural_params(eta);
        StudentLikelihood {
            mu,
            sigma,
            nu,
            excess: eta[2].clamp(-LOG_LINK_BOUND, LOG_LINK_BOUND).exp(),
            digamma_gap: half_digamma_gap(nu),
            trigamma_gap: half_trigamma_gap(nu),
            ln_peak: ln_gamma_ratio(0.5 * nu, 0.5) - 0.5 * (nu * PI).ln() - sigma.ln(),
        }
    }

    /// `-ln p(y)`.
    fn nll(&self, y: f64) -> f64 {
        let z = (y - self.mu) / self.sigma;
        f64::midpoint(self.nu, 1.0) * (z * z / self.nu).ln_1p() - self.ln_peak
    }

    /// `2 ∂NLL/∂ν` at fixed `σ`, `E(ν) + [ln(1 + q) - q/(1 + q)] - q/(ν(1 + q))`
    /// with `q = z²/ν` (each bracket without cancellation at large `ν`).
    fn df_score(&self, z: f64) -> f64 {
        let q = z * z / self.nu;
        self.digamma_gap + ln1p_minus_ratio(q) - q / (1.0 + q) / self.nu
    }

    /// The gradient in the margins.
    pub(super) fn gradient(&self, y: f64) -> [f64; MAX_PARAMS] {
        let (sigma, nu) = (self.sigma, self.nu);
        let z = (y - self.mu) / sigma;
        let w = (nu + 1.0) / (nu + z * z);
        let g1 = 1.0 - w * z * z;
        [
            -w * z / sigma,
            g1,
            self.excess * (0.5 * self.df_score(z) + g1 / (nu * (nu + 1.0))),
        ]
    }

    /// The (diagonal) Fisher information in the margins.
    pub(super) fn fisher(&self) -> [f64; MAX_PARAMS] {
        let (sigma, nu) = (self.sigma, self.nu);
        [
            (nu + 1.0) / ((nu + 3.0) * sigma * sigma),
            2.0 * nu / (nu + 3.0),
            self.excess * self.excess * self.trigamma_gap,
        ]
    }

    /// [`Self::fisher`] with the `κ` entry floored at the magnitude of the
    /// exact observed curvature, `max(I₂, |H₂₂|)`. `I₂ = (ν - 1)² R(ν)`
    /// vanishes at both ends of the link, like `(ν - 1)²` as `ν → 1` and
    /// like `3/(2ν²)` as `ν → ∞`, faster than the score `g₂` (like `ν - 1`,
    /// and like `ν` for a label far out): the step `g₂/I₂` reached `1e14`.
    /// There `|H₂₂|` follows `|g₂|` (it is the derivative of a score that
    /// scales with `ν - 1` or `1/ν`), so `|g₂/I₂*|` stays of order one;
    /// the observed curvature is often negative there (as `ν → 1` for
    /// `z² ≈ 1`, and for labels near the centre at large `ν`), hence its
    /// magnitude. The location and scale entries are left as they are:
    /// their steps `g/I` are at most the normal family's for `z² >= 3`
    /// (outliers are down-weighted by `w = (ν + 1)/(ν + z²)`), and with
    /// `ν > 1` they stay bounded near the centre too.
    pub(super) fn fisher_curvature(&self, y: f64) -> [f64; MAX_PARAMS] {
        let mut curvature = self.fisher();
        curvature[2] = curvature[2].max(self.hessian(y)[2][2].abs());
        curvature
    }

    /// The exact Hessian in the margins: the derivatives in
    /// `(μ, L = ln σ, ν)` composed with `L = ln s + ln ν - ln(ν + 1)` and
    /// `ν = 1 + e^κ`.
    pub(super) fn hessian(&self, y: f64) -> [[f64; MAX_PARAMS]; MAX_PARAMS] {
        let (sigma, nu, d) = (self.sigma, self.nu, self.excess);
        let z = (y - self.mu) / sigma;
        let z2 = z * z;
        let a = nu + z2;
        let w = (nu + 1.0) / a;
        let n_l = 1.0 - w * z2;
        let n_nu = 0.5 * self.df_score(z);
        let n_mm = w * (nu - z2) / (a * sigma * sigma);
        let n_ml = 2.0 * nu * w * z / (a * sigma);
        let n_ll = 2.0 * nu * w * z2 / a;
        let n_mn = -z * (z2 - 1.0) / (a * a * sigma);
        let n_ln = -z2 * (z2 - 1.0) / (a * a);
        // ∂²/∂ν² of the normalization, R(ν) + (ν - 1)/(2ν²(ν + 1)²), plus
        // that of ((ν + 1)/2) ln(1 + z²/ν).
        let n_nn = self.trigamma_gap
            + d / (2.0 * nu * nu * (nu + 1.0).powi(2))
            + z2 * (2.0 * nu + z2 * (1.0 - nu)) / (2.0 * nu * nu * a * a);
        // dL/dν, d²L/dν², and with dν/dκ = d²ν/dκ² = ν - 1: dL/dκ and
        // d²L/dκ².
        let l1 = 1.0 / (nu * (nu + 1.0));
        let l2 = -(2.0 * nu + 1.0) / (nu * nu * (nu + 1.0).powi(2));
        let dl = d * l1;
        let ddl = l2 * d * d + l1 * d;
        let h02 = dl * n_ml + d * n_mn;
        let h12 = dl * n_ll + d * n_ln;
        let h22 = dl * dl * n_ll + 2.0 * dl * d * n_ln + d * d * n_nn + ddl * n_l + d * n_nu;
        [[n_mm, n_ml, h02], [n_ml, n_ll, h12], [h02, h12, h22]]
    }
}

/// `1/Φ⁻¹(3/4)`: the median absolute deviation of a normal over its
/// standard deviation.
const MAD_TO_SD: f64 = 1.482_602_218_505_602;
/// Newton iterations of the Student-t intercept (it converges in a handful
/// unless `ν` runs to its bound).
const STUDENT_MLE_ITERATIONS: usize = 200;
/// Step halvings of one Newton iteration's line search.
const LINE_SEARCH_HALVINGS: usize = 50;

/// The smallest of `values` whose cumulative weight (in ascending order)
/// reaches half of `sum_w`.
fn weighted_median(values: &[f64], w: &dyn Fn(usize) -> f64, sum_w: f64) -> f64 {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_unstable_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut cumulative = 0.0;
    for &i in &order {
        cumulative += w(i);
        if cumulative >= 0.5 * sum_w {
            return values[i];
        }
    }
    order.last().map_or(f64::NAN, |&i| values[i])
}

/// `Some(√v)` for `v > 0`, `None` otherwise (`NaN` included).
fn positive_sqrt(v: f64) -> Option<f64> {
    if v > 0.0 { Some(v.sqrt()) } else { None }
}

/// The Newton step `x` solving `H x = -g` by Cholesky, or `None` if `H` is
/// not positive definite.
fn newton_step(
    h: &[[f64; MAX_PARAMS]; MAX_PARAMS],
    g: &[f64; MAX_PARAMS],
) -> Option<[f64; MAX_PARAMS]> {
    let l00 = positive_sqrt(h[0][0])?;
    let (l10, l20) = (h[1][0] / l00, h[2][0] / l00);
    let l11 = positive_sqrt(h[1][1] - l10 * l10)?;
    let l21 = (h[2][1] - l20 * l10) / l11;
    let l22 = positive_sqrt(h[2][2] - l20 * l20 - l21 * l21)?;
    // L u = -g, then Lᵀ x = u.
    let u0 = -g[0] / l00;
    let u1 = (-g[1] - l10 * u0) / l11;
    let u2 = (-g[2] - l20 * u0 - l21 * u1) / l22;
    let x2 = u2 / l22;
    let x1 = (u1 - l21 * x2) / l11;
    let x0 = (u0 - l10 * x1 - l20 * x2) / l00;
    Some([x0, x1, x2])
}

/// Student-t maximum-likelihood margins `(μ, ln s, κ = ln(ν - 1))` of the
/// (weighted) labels, from the weighted median, `σ` = 1.4826 weighted
/// median absolute deviations (the standard deviation if that is zero),
/// and `ν = 5`: Newton's method on the summed exact Hessian (Fisher scoring
/// where it is not positive definite) with a backtracking line search on
/// the summed NLL, until the margins move less than `1e-13` relative. A log
/// margin at its bound whose score points past it stays there: labels with
/// lighter tails than the normal's send `κ` to `+LOG_LINK_BOUND`, labels
/// with tails heavier than the Cauchy's send it to
/// `-LOG_LINK_BOUND` (`ν → 1`). The likelihood is unbounded as `σ → 0` at
/// a label holding enough of the weight, so this is the local maximum
/// reached from the robust start; constant labels give
/// `(y, -LOG_LINK_BOUND, LOG_LINK_BOUND)`.
fn student_t_mle(labels: &[f32], w: &dyn Fn(usize) -> f64, sum_w: f64) -> Vec<f64> {
    let values: Vec<f64> = labels.iter().map(|&y| f64::from(y)).collect();
    let mu = weighted_median(&values, w, sum_w);
    let deviations: Vec<f64> = values.iter().map(|&y| (y - mu).abs()).collect();
    let mut spread = MAD_TO_SD * weighted_median(&deviations, w, sum_w);
    if spread.is_nan() || spread <= 0.0 {
        let weighted_sum = |f: &dyn Fn(f64) -> f64| -> f64 {
            values
                .iter()
                .enumerate()
                .map(|(i, &y)| w(i) * f(y))
                .sum::<f64>()
        };
        let mean = weighted_sum(&|y| y) / sum_w;
        spread = (weighted_sum(&|y| (y - mean).powi(2)) / sum_w).sqrt();
    }
    if spread.is_nan() || spread <= 0.0 {
        return vec![mu, -LOG_LINK_BOUND, LOG_LINK_BOUND];
    }
    let clamp = |e: f64| e.clamp(-LOG_LINK_BOUND, LOG_LINK_BOUND);
    let start_df = 5.0f64;
    let mut eta = [
        mu,
        clamp((spread * (start_df + 1.0) / start_df).ln()),
        (start_df - 1.0).ln(),
    ];
    let total_nll = |eta: &[f64; MAX_PARAMS]| -> f64 {
        let t = StudentLikelihood::at_margins(eta);
        values
            .iter()
            .enumerate()
            .map(|(i, &y)| w(i) * t.nll(y))
            .sum()
    };
    let mut current = total_nll(&eta);
    for _ in 0..STUDENT_MLE_ITERATIONS {
        let t = StudentLikelihood::at_margins(&eta);
        let mut grad = [0.0; MAX_PARAMS];
        let mut hess = [[0.0; MAX_PARAMS]; MAX_PARAMS];
        for (i, &y) in values.iter().enumerate() {
            let wi = w(i);
            if wi == 0.0 {
                continue;
            }
            let (g, h) = (t.gradient(y), t.hessian(y));
            for ((sum_g, sum_h), (&gi, hi)) in grad.iter_mut().zip(&mut hess).zip(g.iter().zip(&h))
            {
                *sum_g += wi * gi;
                for (s, &v) in sum_h.iter_mut().zip(hi) {
                    *s += wi * v;
                }
            }
        }
        // A log margin at its bound whose score points past it stays put.
        for j in 1..MAX_PARAMS {
            let outward = (eta[j] >= LOG_LINK_BOUND && grad[j] < 0.0)
                || (eta[j] <= -LOG_LINK_BOUND && grad[j] > 0.0);
            if outward {
                grad[j] = 0.0;
                hess[j] = [0.0; MAX_PARAMS];
                for row in &mut hess {
                    row[j] = 0.0;
                }
                hess[j][j] = 1.0;
            }
        }
        let step = newton_step(&hess, &grad).unwrap_or_else(|| {
            let fisher = t.fisher();
            std::array::from_fn(|j| -grad[j] / (sum_w * fisher[j]).max(MIN_CURVATURE))
        });
        let decrease = -grad.iter().zip(&step).map(|(g, s)| g * s).sum::<f64>();
        let tolerance = 1e-12 * (current.abs() + sum_w);
        let mut scale = 1.0;
        let mut accepted = None;
        for _ in 0..LINE_SEARCH_HALVINGS {
            let next = [
                eta[0] + scale * step[0],
                clamp(eta[1] + scale * step[1]),
                clamp(eta[2] + scale * step[2]),
            ];
            let value = total_nll(&next);
            if value.is_finite() && (value <= current || scale * decrease <= tolerance) {
                accepted = Some((next, value));
                break;
            }
            scale *= 0.5;
        }
        let Some((next, value)) = accepted else {
            break;
        };
        let moved = next
            .iter()
            .zip(&eta)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let size = 1.0 + eta.iter().map(|&e| e.abs()).fold(0.0, f64::max);
        eta = next;
        current = value;
        if moved <= 1e-13 * size {
            break;
        }
    }
    eta.to_vec()
}

/// `D = ψ(y + r) - ψ(r) + ln(r/(r + m)) + (m - y)/(r + m)`, the negated
/// derivative of the negative-binomial NLL in `r`, written as
/// `[dml(y + r) - dml(r)] + [ln(1 + u) - u]` with `u = (y - m)/(r + m)` and
/// `dml(x) = ψ(x) - ln x` so it keeps its precision as `r` grows. Where
/// `u` approaches `-1` (a count far below a large mean), `ln(1 + u)` is
/// taken as `ln((r + y)/(r + m))`: `1 + u` rounds to zero there.
pub(super) fn nb_size_score(m: f64, r: f64, y: f64) -> f64 {
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
pub(super) fn nb_size_fisher(m: f64, r: f64) -> f64 {
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
pub(super) fn nb_size_fisher_blocked(m: f64, r: f64, mut blocks: CountBlocks) -> f64 {
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
pub(super) fn gamma_shape_mle(s: f64) -> f64 {
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
pub(super) fn nb_size_mle(score: impl Fn(f64) -> f64) -> f64 {
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
