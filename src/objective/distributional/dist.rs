//! [`Dist`]: one row's predicted distribution and its moments, CDF,
//! quantiles, density, CRPS, intervals, and sampling.

use super::count::{
    COUNT_TAIL, CountBlocks, MAX_COUNT_TERMS, block_crps, gamma_unit_quantile, geometric_tail_crps,
};
use super::special::{
    HALF_LN_2PI, beta_inc, gamma_p, gamma_q, ln_gamma, ln_gamma_prefactor, ln_gamma_ratio,
    ln_norm_cdf, norm_cdf, norm_pdf, norm_ppf,
};
use super::{Dist, DistFamily};
use crate::error::{HessboostError, Result};

/// `1 / √π`.
pub(super) const FRAC_1_SQRT_PI: f64 = 0.564_189_583_547_756_3;

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

    /// The distribution from one row of natural parameters as
    /// [`BoostedModel::predict`](crate::model::BoostedModel::predict)
    /// reports them (no validation).
    pub(crate) fn from_row(family: DistFamily, row: &[f32]) -> Self {
        Self::from_natural(
            family,
            f64::from(row[0]),
            row.get(1).map_or(0.0, |&v| f64::from(v)),
        )
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
    /// uniform on `(0, 1)` from 52 of the random bits `next_u64` returns
    /// (the midpoints `(j + 1/2)/2^52`, all exactly representable, so `u`
    /// never rounds to `1`), so a seeded generator gives a reproducible
    /// stream. Any source of uniform `u64` words works, e.g.
    /// `dist.sample(|| rng.next_u64())` with a `rand` generator.
    pub fn sample(&self, mut next_u64: impl FnMut() -> u64) -> f64 {
        let u = ((next_u64() >> 12) as f64 + 0.5) * (1.0 / (1u64 << 52) as f64);
        self.quantile(u)
    }

    /// `P(Y = k + 1) / P(Y = k)` for the count families.
    pub(super) fn count_ratio(&self, k: f64) -> f64 {
        match *self {
            Dist::Poisson { rate } => rate / (k + 1.0),
            Dist::NegativeBinomial { mean, size } => (k + size) / (k + 1.0) * mean / (size + mean),
            _ => unreachable!("count_ratio on a continuous family"),
        }
    }

    /// Smallest integer `k >= 0` with `cdf(k) >= p`: bracket from the normal
    /// approximation by doubling steps, then bisect (to adjacent integers,
    /// or adjacent floats where those are more than 1 apart).
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
        // Invariant: cdf(lo) < p <= cdf(hi) (cdf(-1) = 0). Past 2^53 adjacent
        // floats are more than 1 apart, so stop once no integer lies strictly
        // between them.
        while hi - lo > 1.0 {
            let mid = f64::midpoint(lo, hi).floor();
            if !(lo < mid && mid < hi) {
                break;
            }
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
