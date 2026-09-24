//! Special functions for the distributional objectives, in `f64`: log-gamma,
//! digamma/trigamma (with cancellation-free large-argument forms),
//! regularized incomplete gamma and beta functions, and the standard normal
//! CDF and quantile.
//!
//! Accuracy targets are close to double precision for moderate arguments:
//! the Stirling/asymptotic series are used from `x >= 10` (truncation error
//! below `1e-14` relative), the incomplete functions use the series and
//! Lentz continued fractions of *Numerical Recipes* to a relative tolerance
//! of `1e-15` (with Temme's uniform asymptotic expansion for shapes from
//! `1e4`, where the series would need `O(sqrt(a))` terms), log-gamma
//! differences of large arguments are formed from the Stirling series
//! without subtracting the large values, and the normal quantile refines
//! Acklam's rational approximation with one Halley step.

use std::f64::consts::PI;

/// `ln(2π) / 2`.
pub(super) const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;
/// Convergence tolerance of the series and continued fractions.
const EPS: f64 = 1e-15;
/// Lentz's guard against a zero denominator.
const FPMIN: f64 = 1e-300;
/// Iteration cap of the series / continued fractions. They converge in
/// `O(sqrt(a))` steps, so this only bounds pathological arguments.
const MAX_ITER: usize = 1_000_000;
/// Arguments from which the asymptotic series replace the recurrences.
const ASYMPTOTIC_FROM: f64 = 10.0;
/// Shapes from which the incomplete gamma functions use Temme's expansion:
/// its first omitted term, `c₃(η) a⁻³`, is below `1e-14` relative to the
/// retained ones there.
const TEMME_FROM: f64 = 1e4;

/// Taylor coefficients in `η` of Temme's `c₀(η)`, `c₁(η)`, `c₂(η)` (DLMF
/// §8.12; convergent for `|η| < 2√π`), evaluated for `|η| < 1`.
#[allow(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "computed series coefficients"
)]
const TEMME_C0: [f64; 27] = [
    -0.33333333333333333,
    0.083333333333333333,
    -0.014814814814814815,
    0.0011574074074074074,
    0.0003527336860670194,
    -0.00017875514403292181,
    3.9192631785224378e-5,
    -2.1854485106799922e-6,
    -1.85406221071516e-6,
    8.296711340953086e-7,
    -1.7665952736826079e-7,
    6.7078535434014986e-9,
    1.0261809784240308e-8,
    -4.3820360184533532e-9,
    9.1476995822367902e-10,
    -2.551419399494625e-11,
    -5.8307721325504251e-11,
    2.4361948020667416e-11,
    -5.0276692801141756e-12,
    1.1004392031956135e-13,
    3.3717632624009854e-13,
    -1.3923887224181621e-13,
    2.8534893807047443e-14,
    -5.1391118342425726e-16,
    -1.9752288294349443e-15,
    8.0995211567045613e-16,
    -1.6522531216398162e-16,
];
#[allow(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "computed series coefficients"
)]
const TEMME_C1: [f64; 23] = [
    -0.0018518518518518519,
    -0.0034722222222222222,
    0.0026455026455026455,
    -0.00099022633744855967,
    0.00020576131687242798,
    -4.0187757201646091e-7,
    -1.8098550334489978e-5,
    7.6491609160811101e-6,
    -1.6120900894563446e-6,
    4.6471278028074343e-9,
    1.378633446915721e-7,
    -5.752545603517705e-8,
    1.1951628599778147e-8,
    -1.7543241719747648e-11,
    -1.0091543710600413e-9,
    4.1627929918425826e-10,
    -8.5639070264929806e-11,
    6.0672151016047586e-14,
    7.1624989648114854e-12,
    -2.9331866437714371e-12,
    5.9966963656836887e-13,
    -2.1671786527323314e-16,
    -4.9783399723692616e-14,
];
#[allow(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "computed series coefficients"
)]
const TEMME_C2: [f64; 19] = [
    0.0041335978835978836,
    -0.0026813271604938272,
    0.00077160493827160494,
    2.0093878600823045e-6,
    -0.00010736653226365161,
    5.2923448829120125e-5,
    -1.2760635188618728e-5,
    3.4235787340961381e-8,
    1.3721957309062933e-6,
    -6.298992138380055e-7,
    1.4280614206064242e-7,
    -2.0477098421990866e-10,
    -1.4092529910867521e-8,
    6.228974084922022e-9,
    -1.3670488396617113e-9,
    9.4283561590146782e-13,
    1.2872252400089318e-10,
    -5.5645956134363321e-11,
    1.1975935546366981e-11,
];

/// Lanczos coefficients (`g = 7`, `n = 9`).
#[allow(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "published Lanczos coefficients"
)]
const LANCZOS: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_13,
    -176.615_029_162_140_59,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_571_6e-6,
    1.505_632_735_149_311_6e-7,
];

/// `ln Γ(x)` for `x > 0` (`+∞` at `0`, `NaN` below).
pub(crate) fn ln_gamma(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x < 0.5 {
        // Reflection: Γ(x)Γ(1 - x) = π / sin(πx).
        return (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    if x >= ASYMPTOTIC_FROM {
        return (x - 0.5) * x.ln() - x + HALF_LN_2PI + stirling_series(x);
    }
    let x = x - 1.0;
    let mut a = LANCZOS[0];
    for (i, &c) in LANCZOS.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    let t = x + 7.5;
    HALF_LN_2PI + (x + 0.5) * t.ln() - t + a.ln()
}

/// `ln Γ(x) - [(x - 1/2) ln x - x + ln(2π)/2]`, the Stirling series
/// (`x >= 10`).
fn stirling_series(x: f64) -> f64 {
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    inv * (1.0 / 12.0
        - inv2
            * (1.0 / 360.0
                - inv2
                    * (1.0 / 1260.0
                        - inv2
                            * (1.0 / 1680.0 - inv2 * (1.0 / 1188.0 - inv2 * (691.0 / 360_360.0))))))
}

/// `ln Γ(x + d) - ln Γ(x)` for `x > 0`, `d >= 0`. From `x >= 10` it is
/// formed from the Stirling series as
/// `(x - 1/2) ln(1 + d/x) + d ln(x + d) - d` plus the series difference,
/// so it keeps its precision when both log-gammas are large and nearly
/// equal (large `x`, small `d`).
pub(crate) fn ln_gamma_ratio(x: f64, d: f64) -> f64 {
    if x >= ASYMPTOTIC_FROM {
        let s = x + d;
        (x - 0.5) * (d / x).ln_1p() + d * s.ln() - d + (stirling_series(s) - stirling_series(x))
    } else {
        ln_gamma(x + d) - ln_gamma(x)
    }
}

/// `t - 1 - ln t` for `t > 0`, accurate in relative terms near `t = 1`
/// (where it is `(t - 1)²/2 + ...`) and without losing `ln t` far from it.
pub(crate) fn log_gap(t: f64) -> f64 {
    if (0.5..=2.0).contains(&t) {
        // `t - 1` is exact here.
        log1p_gap(t - 1.0)
    } else {
        t - 1.0 - t.ln()
    }
}

/// `u - ln(1 + u)` for `u` in `[-1/2, 1]`, accurate in relative terms near
/// `u = 0`, where `u - ln_1p(u)` would lose `log10(1/|u|)` digits: with
/// `s = u/(2 + u)` (`|s| <= 1/3`), `ln(1 + u) = 2 atanh s` and
/// `u - ln(1 + u) = 2s²/(1 - s) - 2s Σ_{k>=1} s^{2k}/(2k + 1)`.
fn log1p_gap(u: f64) -> f64 {
    let s = u / (2.0 + u);
    let s2 = s * s;
    let (mut term, mut sum) = (1.0, 0.0);
    for k in 1..=20 {
        term *= s2;
        sum += term / f64::from(2 * k + 1);
        if term <= EPS * s2 {
            break;
        }
    }
    2.0 * s2 / (1.0 - s) - 2.0 * s * sum
}

/// `ψ(x) - ln x` from the asymptotic series (`x >= 10`).
fn digamma_minus_log_asymptotic(x: f64) -> f64 {
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    -0.5 * inv
        - inv2
            * (1.0 / 12.0
                - inv2
                    * (1.0 / 120.0
                        - inv2
                            * (1.0 / 252.0
                                - inv2
                                    * (1.0 / 240.0
                                        - inv2 * (1.0 / 132.0 - inv2 * (691.0 / 32760.0))))))
}

/// The digamma function `ψ(x) = d/dx ln Γ(x)` for `x > 0`.
fn digamma(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    let (mut x, mut acc) = (x, 0.0);
    while x < ASYMPTOTIC_FROM {
        acc -= 1.0 / x;
        x += 1.0;
    }
    acc + x.ln() + digamma_minus_log_asymptotic(x)
}

/// `ψ(x) - ln x` for `x > 0`, without the cancellation of the difference
/// at large `x` (where both terms grow like `ln x` but the difference is
/// `-1/(2x) + O(x⁻²)`).
pub(crate) fn digamma_minus_log(x: f64) -> f64 {
    if x >= ASYMPTOTIC_FROM {
        digamma_minus_log_asymptotic(x)
    } else {
        digamma(x) - x.ln()
    }
}

/// `ψ'(x) - 1/x` from the asymptotic series (`x >= 10`).
fn trigamma_minus_inv_asymptotic(x: f64) -> f64 {
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    inv2 * (0.5
        + inv
            * (1.0 / 6.0
                - inv2
                    * (1.0 / 30.0
                        - inv2
                            * (1.0 / 42.0
                                - inv2
                                    * (1.0 / 30.0
                                        - inv2
                                            * (5.0 / 66.0
                                                - inv2 * (691.0 / 2730.0 - inv2 * (7.0 / 6.0))))))))
}

/// The trigamma function `ψ'(x)` for `x > 0`.
fn trigamma(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    let (mut x, mut acc) = (x, 0.0);
    while x < ASYMPTOTIC_FROM {
        acc += 1.0 / (x * x);
        x += 1.0;
    }
    acc + 1.0 / x + trigamma_minus_inv_asymptotic(x)
}

/// `ψ'(x) - 1/x` for `x > 0`, without cancellation at large `x` (where the
/// difference is `1/(2x²) + O(x⁻³)`).
pub(crate) fn trigamma_minus_inv(x: f64) -> f64 {
    if x >= ASYMPTOTIC_FROM {
        trigamma_minus_inv_asymptotic(x)
    } else {
        trigamma(x) - 1.0 / x
    }
}

/// `μ = x/a - 1` and `x/a - 1 - ln(x/a)` from `x - a`, which is exact near
/// `x = a` (so the gap keeps its precision where it is `μ²/2`).
fn shape_gap(a: f64, x: f64) -> (f64, f64) {
    let mu = (x - a) / a;
    let gap = if (-0.5..=1.0).contains(&mu) {
        log1p_gap(mu)
    } else {
        log_gap(x / a)
    };
    (mu, gap)
}

/// `ln(x^a e^-x / Γ(a))`, the log of the incomplete gamma prefactor (and of
/// `x` times the unit-rate Gamma(`a`) density at `x`). From `a >= 10` it is
/// `-a (x/a - 1 - ln(x/a)) + ln(a/(2π))/2 - σ(a)` with the Stirling series
/// `σ`, instead of the difference of `a ln x` and `ln Γ(a)`, which lose
/// all precision for large shapes.
pub(crate) fn ln_gamma_prefactor(a: f64, x: f64) -> f64 {
    if a >= ASYMPTOTIC_FROM {
        -a * shape_gap(a, x).1 + 0.5 * a.ln() - HALF_LN_2PI - stirling_series(a)
    } else {
        a * x.ln() - x - ln_gamma(a)
    }
}

/// `x^a e^-x / Γ(a)`, the common prefactor of the incomplete gamma forms.
pub(crate) fn gamma_prefactor(a: f64, x: f64) -> f64 {
    ln_gamma_prefactor(a, x).exp()
}

/// `(P(a, x), Q(a, x))` from Temme's uniform asymptotic expansion (DLMF
/// §8.12): `Q = erfc(η √(a/2))/2 + R`, `P = erfc(-η √(a/2))/2 - R` with
/// `η = sign(x - a) √(2(λ - 1 - ln λ))`, `λ = x/a`, and
/// `R = e^{-aη²/2} / √(2πa) Σ_{k<3} c_k(η) a^{-k}`. Uniform in `x`, so it
/// needs no iteration however large `a` is (`a >= 1e4`). From `|η| = 1`
/// on, `e^{-aη²/2} <= e^{-5000}` underflows: `R` vanishes and the `erfc`
/// terms saturate, so the `c_k` are only needed (as Taylor series) inside.
fn gamma_temme(a: f64, x: f64) -> (f64, f64) {
    let (mu, gap) = shape_gap(a, x);
    let eta = (2.0 * gap).sqrt().copysign(mu);
    let r = if eta.abs() < 1.0 {
        let poly = |c: &[f64]| c.iter().rev().fold(0.0, |acc, &v| acc * eta + v);
        let series = poly(&TEMME_C0) + (poly(&TEMME_C1) + poly(&TEMME_C2) / a) / a;
        (-a * gap).exp() / (2.0 * PI * a).sqrt() * series
    } else {
        0.0
    };
    let t = eta * (0.5 * a).sqrt();
    (
        (0.5 * erfc(-t) - r).clamp(0.0, 1.0),
        (0.5 * erfc(t) + r).clamp(0.0, 1.0),
    )
}

/// Series for `P(a, x)`, valid (fast) for `x < a + 1`.
fn gamma_p_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut del = 1.0 / a;
    let mut sum = del;
    for _ in 0..MAX_ITER {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPS {
            break;
        }
    }
    sum * gamma_prefactor(a, x)
}

/// Lentz's guard: `v`, or [`FPMIN`] where `v` is (nearly) zero.
fn nonzero(v: f64) -> f64 {
    if v.abs() < FPMIN { FPMIN } else { v }
}

/// Continued fraction for `Q(a, x)` over its prefactor `x^a e^-x / Γ(a)`,
/// valid (fast) for `x >= a + 1`.
fn gamma_q_fraction(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..MAX_ITER {
        let i = i as f64;
        let an = -i * (i - a);
        b += 2.0;
        d = 1.0 / nonzero(an * d + b);
        c = nonzero(b + an / c);
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularized lower incomplete gamma `P(a, x) = γ(a, x) / Γ(a)`, `a > 0`.
pub(crate) fn gamma_p(a: f64, x: f64) -> f64 {
    if x.is_nan() || a.is_nan() || a <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        0.0
    } else if x == f64::INFINITY {
        1.0
    } else if a >= TEMME_FROM {
        gamma_temme(a, x).0
    } else if x < a + 1.0 {
        gamma_p_series(a, x)
    } else {
        1.0 - gamma_prefactor(a, x) * gamma_q_fraction(a, x)
    }
}

/// Regularized upper incomplete gamma `Q(a, x) = 1 - P(a, x)`, computed
/// directly in the upper tail so it keeps its relative precision there.
pub(crate) fn gamma_q(a: f64, x: f64) -> f64 {
    if x.is_nan() || a.is_nan() || a <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        1.0
    } else if x == f64::INFINITY {
        0.0
    } else if a >= TEMME_FROM {
        gamma_temme(a, x).1
    } else if x < a + 1.0 {
        1.0 - gamma_p_series(a, x)
    } else {
        gamma_prefactor(a, x) * gamma_q_fraction(a, x)
    }
}

/// Lentz continued fraction of the incomplete beta function.
fn beta_fraction(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 / nonzero(1.0 - qab * x / qap);
    let mut h = d;
    // One Lentz step with coefficient `aa`, returning its factor of `h`.
    let mut step = |aa: f64| {
        d = 1.0 / nonzero(1.0 + aa * d);
        c = nonzero(1.0 + aa / c);
        d * c
    };
    for m in 1..MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;
        h *= step(m * (b - m) * x / ((qam + m2) * (a + m2)));
        let del = step(-(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)));
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta `I_x(a, b)` for `a, b > 0`, `x` in `[0, 1]`,
/// with `y = 1 - x` supplied by the caller (so `ln x` and `ln(1 - x)` keep
/// their precision when either is near 1). The normalization
/// `ln Γ(a + b) - ln Γ(a) - ln Γ(b)` is formed as a log-gamma ratio of the
/// larger parameter, which stays accurate when one parameter is huge and
/// the other small (negative binomials near their Poisson limit).
pub(crate) fn beta_inc(a: f64, b: f64, x: f64, y: f64) -> f64 {
    if !(a > 0.0 && b > 0.0) || x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if y <= 0.0 {
        return 1.0;
    }
    let (big, small) = if a >= b { (a, b) } else { (b, a) };
    let ln_x = if x > 0.5 { (-y).ln_1p() } else { x.ln() };
    let ln_y = if y > 0.5 { (-x).ln_1p() } else { y.ln() };
    let front = (ln_gamma_ratio(big, small) - ln_gamma(small) + a * ln_x + b * ln_y).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_fraction(a, b, x) / a
    } else {
        1.0 - front * beta_fraction(b, a, y) / b
    }
}

/// Complementary error function `erfc(x) = Q(1/2, x²)` for `x >= 0`, keeping
/// relative precision far into the upper tail.
fn erfc(x: f64) -> f64 {
    if x >= 0.0 {
        gamma_q(0.5, x * x)
    } else {
        1.0 + gamma_p(0.5, x * x)
    }
}

/// Standard normal density.
pub(crate) fn norm_pdf(z: f64) -> f64 {
    (-0.5 * z * z - HALF_LN_2PI).exp()
}

/// Standard normal CDF `Φ(z)`, accurate in relative terms in the lower tail.
pub(crate) fn norm_cdf(z: f64) -> f64 {
    0.5 * erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
}

/// `ln Φ(z)`, keeping its relative precision where `Φ(z)` underflows: in
/// the lower tail, where `Φ(z) = Q(1/2, z²/2)/2` takes the continued
/// fraction (`z²/2 >= 3/2`), as the log prefactor plus the log of the
/// fraction; above zero as `ln(1 - Φ(-z))`.
pub(crate) fn ln_norm_cdf(z: f64) -> f64 {
    let s = -z * std::f64::consts::FRAC_1_SQRT_2;
    let x = s * s;
    if s > 0.0 && x >= 1.5 {
        if x == f64::INFINITY {
            return f64::NEG_INFINITY;
        }
        ln_gamma_prefactor(0.5, x) + gamma_q_fraction(0.5, x).ln() - std::f64::consts::LN_2
    } else if z > 0.0 {
        (-norm_cdf(-z)).ln_1p()
    } else {
        norm_cdf(z).ln()
    }
}

/// Standard normal quantile `Φ⁻¹(p)`: Acklam's rational approximation
/// (relative error below `1.2e-9`) refined by one Halley step on `Φ`.
/// `-∞` at `p = 0`, `+∞` at `p = 1`, `NaN` outside `[0, 1]`.
#[allow(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "published coefficients"
)]
pub(crate) fn norm_ppf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let tail = |q: f64| {
        let q = (-2.0 * q.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    let x = if p < P_LOW {
        tail(p)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -tail(1.0 - p)
    };
    // One Halley step on `e = Φ(x) - p`, evaluated in the upper half as
    // `(1 - p) - Φ(-x)`, which keeps its precision there.
    let e = if p > 0.5 {
        (1.0 - p) - norm_cdf(-x)
    } else {
        norm_cdf(x) - p
    };
    let u = e / norm_pdf(x);
    x - u / (1.0 + 0.5 * x * u)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-300)
    }

    #[test]
    fn ln_gamma_matches_factorials_and_half_integers() {
        let mut fact = 1.0f64;
        for n in 1..30 {
            // Γ(n + 1) = n!
            fact *= f64::from(n);
            assert!(close(ln_gamma(f64::from(n) + 1.0), fact.ln(), 1e-14), "{n}");
        }
        // Γ(1/2) = √π, Γ(3/2) = √π / 2, Γ(10.5) = 1133278.3889487855.
        assert!(close(ln_gamma(0.5), PI.sqrt().ln(), 1e-14));
        assert!(close(ln_gamma(1.5), (PI.sqrt() / 2.0).ln(), 1e-13));
        assert!(close(
            ln_gamma(10.5),
            1_133_278.388_948_785_5_f64.ln(),
            1e-14
        ));
        assert!(ln_gamma(1.0).abs() < 1e-15 && ln_gamma(2.0).abs() < 1e-15);
        assert!(close(ln_gamma(1e-3), 6.907_178_885_383_853, 1e-13));
    }

    #[test]
    fn digamma_and_trigamma_match_reference_values() {
        // ψ(1) = -γ, ψ(1/2) = -γ - 2 ln 2, ψ'(1) = π²/6, ψ'(1/2) = π²/2.
        let euler = 0.577_215_664_901_532_9;
        assert!(close(digamma(1.0), -euler, 1e-14));
        assert!(close(digamma(0.5), -euler - 2.0 * 2f64.ln(), 1e-14));
        assert!(close(trigamma(1.0), PI * PI / 6.0, 1e-14));
        assert!(close(trigamma(0.5), PI * PI / 2.0, 1e-14));
        // Both sides of the switch to the asymptotic series agree with the
        // recurrences ψ(x+1) = ψ(x) + 1/x, ψ'(x+1) = ψ'(x) - 1/x².
        for x in [9.3, 9.9, 9.99, 25.0] {
            assert!(close(digamma(x + 1.0), digamma(x) + 1.0 / x, 1e-14), "{x}");
            assert!(
                close(trigamma(x + 1.0), trigamma(x) - 1.0 / (x * x), 1e-13),
                "{x}"
            );
        }
        // The cancellation-free forms at large arguments.
        let x = 1e12;
        assert!(close(
            digamma_minus_log(x),
            -0.5 / x - 1.0 / (12.0 * x * x),
            1e-12
        ));
        assert!(close(trigamma_minus_inv(x), 0.5 / (x * x), 1e-11));
        assert!(close(
            digamma_minus_log(3.0),
            digamma(3.0) - 3f64.ln(),
            1e-15
        ));
    }

    #[test]
    fn incomplete_gamma_matches_closed_forms() {
        // P(1, x) = 1 - e^-x; P(1/2, x²) = erf(x); Q(n, x) = e^-x Σ_{k<n} x^k/k!.
        for x in [1e-3, 0.3, 1.0, 2.5, 10.0, 40.0] {
            assert!(close(gamma_p(1.0, x), -(-x).exp_m1(), 1e-14), "{x}");
            assert!(close(gamma_q(1.0, x), (-x).exp(), 1e-13), "{x}");
            let mut term = 1.0;
            let mut sum = 1.0;
            for k in 1..4 {
                term *= x / f64::from(k);
                sum += term;
            }
            assert!(close(gamma_q(4.0, x), (-x).exp() * sum, 1e-13), "{x}");
        }
        assert!(close(gamma_p(0.5, 1.0), 0.842_700_792_949_714_9, 1e-14));
        assert_eq!(gamma_p(2.0, 0.0), 0.0);
        assert_eq!(gamma_q(2.0, f64::INFINITY), 0.0);
    }

    /// Shapes up to the log-link bound `e^30`: the series must converge
    /// within its iteration cap (not stop short at `P(a, a) ≈ 0.118`), and
    /// the prefactor must keep its precision despite the cancellation in
    /// `a ln x - ln Γ(a)`. References by mpmath (quadrature of the density
    /// at 40 digits for `a = e^30`, `gammainc` otherwise).
    #[test]
    fn incomplete_gamma_converges_for_huge_shapes() {
        let a = 30f64.exp();
        assert!(close(gamma_p(a, a), 0.500_000_040_679_123_1, 1e-12));
        assert!(close(gamma_q(a, a), 0.499_999_959_320_876_9, 1e-12));
        // 5 standard deviations below, 7 above (x rounded as here).
        let (lo, hi) = (a - 5.0 * a.sqrt(), a + 7.0 * a.sqrt());
        assert!(close(gamma_p(a, lo), 2.866_479_331_517_756_6e-7, 1e-8));
        assert!(close(gamma_q(a, hi), 1.279_857_253_582_386_7e-12, 1e-8));
        // The Poisson CDF at a large rate: P(Y <= 1e12 | λ = 1e12).
        assert!(close(
            gamma_q(1e12 + 1.0, 1e12),
            0.500_000_265_961_520_3,
            1e-12
        ));
        // Around the switch to the expansion, into the far tails.
        assert!(close(gamma_p(1e4, 9e3), 2.073_299_202_433_928e-25, 1e-11));
        assert!(close(
            gamma_q(1e4, 1.2e4),
            3.327_202_492_345_161_5e-79,
            1e-11
        ));
        assert!(close(gamma_p(2e4, 19_900.0), 0.240_115_607_185_431, 1e-12));
        assert!(close(
            gamma_q(5e4, 51_000.0),
            4.411_939_255_120_306e-6,
            1e-11
        ));
        assert_eq!(gamma_p(1e5, 5e4), 0.0);
        assert_eq!(gamma_q(1e5, 5e4), 1.0);
    }

    #[test]
    fn log_gamma_ratio_and_log_gap_keep_their_precision() {
        // Γ(x + 1)/Γ(x) = x, Γ(x + 3)/Γ(x) = x(x + 1)(x + 2), also where
        // ln Γ(x) ≈ 3e14 would swamp a subtraction.
        for x in [12.5, 1e6, 1e13] {
            assert!(close(ln_gamma_ratio(x, 1.0), x.ln(), 1e-14), "{x}");
            let rising = (x * (x + 1.0) * (x + 2.0)).ln();
            assert!(close(ln_gamma_ratio(x, 3.0), rising, 1e-14), "{x}");
        }
        assert_eq!(ln_gamma_ratio(1e13, 0.0), 0.0);
        // t - 1 - ln t: near 1 (≈ (t - 1)²/2) and far below, where t - 1
        // rounds to -1 (mpmath references).
        let t = 1.0 + 2f64.powi(-26);
        assert!(close(log_gap(t), 1.110_223_013_596_081_7e-16, 1e-14));
        assert!(close(log_gap(1e-20), 45.051_701_859_880_914, 1e-15));
        assert!(close(log_gap(1e20), 1e20, 1e-15));
    }

    #[test]
    fn incomplete_beta_matches_closed_forms() {
        // I_x(a, 1) = x^a; I_x(1, b) = 1 - (1 - x)^b; I_x(2, 2) = 3x² - 2x³.
        for x in [1e-4, 0.1, 0.5, 0.9, 0.999] {
            let y = 1.0 - x;
            assert!(close(beta_inc(2.5, 1.0, x, y), x.powf(2.5), 1e-13), "{x}");
            assert!(
                close(beta_inc(1.0, 3.5, x, y), 1.0 - y.powf(3.5), 1e-13),
                "{x}"
            );
            assert!(
                close(
                    beta_inc(2.0, 2.0, x, y),
                    3.0 * x * x - 2.0 * x * x * x,
                    1e-13
                ),
                "{x}"
            );
        }
    }

    #[test]
    fn normal_cdf_and_quantile_round_trip() {
        assert!(close(norm_cdf(0.0), 0.5, 1e-15));
        assert!(close(norm_cdf(1.959_963_984_540_054), 0.975, 1e-14));
        // Φ(-10) = 7.619853024160527e-24: relative precision in the tail.
        assert!(close(norm_cdf(-10.0), 7.619_853_024_160_527e-24, 1e-12));
        for p in [1e-300, 1e-12, 1e-4, 0.02, 0.3, 0.5, 0.77, 0.975, 1.0 - 1e-9] {
            let z = norm_ppf(p);
            let back = if p > 0.5 {
                1.0 - norm_cdf(-z)
            } else {
                norm_cdf(z)
            };
            assert!(close(back, p, 1e-12), "{p}: {z} -> {back}");
        }
        assert_eq!(norm_ppf(0.0), f64::NEG_INFINITY);
        assert_eq!(norm_ppf(1.0), f64::INFINITY);
        assert!(norm_ppf(1.5).is_nan());
    }

    /// `ln Φ(z)` from far below the smallest double (`Φ(-1000) ≈ e^-500008`)
    /// to near one (mpmath `log(ncdf(z))`, 60 digits).
    #[test]
    fn log_normal_cdf_keeps_its_precision_in_both_tails() {
        for (z, reference) in [
            (-1000.0, -500_007.826_694_812_16),
            (-300.0, -45_006.622_732_118_66),
            (-38.890_872_965_260_115, -760.830_358_195_908_2),
            (-10.0, -53.231_285_150_512_47),
            (-5.0, -15.064_998_393_988_725),
            (-1.7, -3.110_796_097_552_481_3),
            (-1.0, -1.841_021_645_009_263_6),
            (0.0, -std::f64::consts::LN_2),
            (0.5, -0.368_946_415_288_656_4),
            (3.0, -0.001_350_809_964_748_193_8),
            (10.0, -7.619_853_024_160_525e-24),
        ] {
            let got = ln_norm_cdf(z);
            assert!(close(got, reference, 1e-12), "{z}: {got} vs {reference}");
        }
        assert_eq!(ln_norm_cdf(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(ln_norm_cdf(f64::INFINITY), 0.0);
    }
}
