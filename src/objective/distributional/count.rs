//! Count-distribution helpers: blocked probability-mass sums, their
//! CRPS terms, and the quantile root finders.

use super::Dist;
use super::special::{gamma_p, gamma_prefactor, gamma_q, ln_gamma};

/// Upper-tail probability below which the count sums stop.
pub(super) const COUNT_TAIL: f64 = 1e-12;
/// Most probability-mass terms any count sum takes before it closes the
/// remainder with a geometric-tail estimate.
pub(super) const MAX_COUNT_TERMS: usize = 100_000;
/// Standard deviations below the mean from which count sums start (the mass
/// below is below `1e-30`).
pub(super) const COUNT_HEAD_SDS: f64 = 12.0;

/// Relative width of the [`CountBlocks`] below their widest: a block at
/// `k` spans at most `(k + 1) / 1000` values.
pub(super) const BLOCK_GROWTH: f64 = 1e-3;

/// One block of [`CountBlocks`].
pub(super) struct CountBlock {
    /// First value.
    pub(super) k: f64,
    /// Number of values.
    pub(super) h: f64,
    /// Probability mass.
    pub(super) mass: f64,
    /// Width of the next block.
    pub(super) next_h: f64,
    /// `ln P(c') - ln P(c)` from this block's centre `c` to the next one's.
    pub(super) ln_step: f64,
}

impl CountBlock {
    /// Geometric estimate of the mass beyond this block, at the ratio of
    /// the next block's mass to this one's (`∞` unless it is below one).
    pub(super) fn tail(&self) -> f64 {
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
pub(super) struct CountBlocks {
    pub(super) dist: Dist,
    /// Widest block: `1` (single values) unless 24 standard deviations
    /// exceed half of [`MAX_COUNT_TERMS`], else the width that fits them.
    pub(super) h_max: f64,
    /// First value of the next block.
    pub(super) k: f64,
    /// Width of the next block.
    pub(super) h: f64,
    /// `ln P(Y = c)` at the next block's centre `c = k + (h - 1)/2`.
    pub(super) ln_center: f64,
}

impl CountBlocks {
    pub(super) fn new(dist: Dist) -> Self {
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
    pub(super) fn width(h_max: f64, k: f64) -> f64 {
        (BLOCK_GROWTH * (k + 1.0)).floor().clamp(1.0, h_max)
    }

    /// Visit the next block.
    pub(super) fn step(&mut self) -> CountBlock {
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
pub(super) fn square_sum(a: f64, d: f64, i0: f64, i1: f64) -> f64 {
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
pub(super) fn block_crps(k: f64, h: f64, before: f64, after: f64, y: f64) -> f64 {
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
pub(super) fn geometric_tail_crps(u: f64, rho: f64, m: f64) -> f64 {
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
pub(super) fn gamma_unit_quantile(a: f64, p: f64) -> f64 {
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
pub(super) fn bisect_quantile(x: f64, residual: impl Fn(f64) -> f64) -> f64 {
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
