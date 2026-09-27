//! The distribution families: their metadata, links, likelihood
//! derivatives in the margins, and marginal maximum-likelihood fits.

use serde::{Deserialize, Serialize};

use super::LOG_LINK_BOUND;
use super::count::{COUNT_TAIL, CountBlocks, MAX_COUNT_TERMS};
use super::special::{digamma_minus_log, log_gap, trigamma_minus_inv};

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
            /// The distribution of `family` with natural parameters `p0` and
            /// (for two-parameter families) `p1`, unvalidated.
            pub(super) fn from_natural(family: DistFamily, p0: f64, p1: f64) -> Self {
                let natural = [p0, p1];
                match family {
                    $(DistFamily::$family => {
                        let mut next = natural.into_iter();
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
    pub(super) fn below_support(self, y: f64) -> bool {
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

pub(super) fn normal_gradient(mu: f64, sigma: f64, y: f64) -> [f64; 2] {
    let z = (y - mu) / sigma;
    [-z / sigma, 1.0 - z * z]
}

pub(super) fn normal_hessian(sigma: f64, z: f64) -> [[f64; 2]; 2] {
    let h12 = 2.0 * z / sigma;
    [[1.0 / (sigma * sigma), h12], [h12, 2.0 * z * z]]
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
