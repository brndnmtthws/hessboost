//! Special functions for the distributional objectives, in `f64`: log-gamma,
//! digamma/trigamma (with cancellation-free large-argument forms),
//! regularized incomplete gamma and beta functions, and the standard normal
//! CDF and quantile.
//!
//! Accuracy targets are close to double precision for moderate arguments:
//! the Stirling/asymptotic series are used from `x >= 10` (truncation error
//! below `1e-14` relative), the incomplete functions use the series and
//! Lentz continued fractions of *Numerical Recipes* to a relative tolerance
//! of `1e-15`, and the normal quantile refines Acklam's rational
//! approximation with one Halley step.

use std::f64::consts::PI;

/// `ln(2π) / 2`.
const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;
/// Convergence tolerance of the series and continued fractions.
const EPS: f64 = 1e-15;
/// Lentz's guard against a zero denominator.
const FPMIN: f64 = 1e-300;
/// Iteration cap of the series / continued fractions. They converge in
/// `O(sqrt(a))` steps, so this only bounds pathological arguments.
const MAX_ITER: usize = 1_000_000;
/// Arguments from which the asymptotic series replace the recurrences.
const ASYMPTOTIC_FROM: f64 = 10.0;

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
        // Stirling series.
        let inv = 1.0 / x;
        let inv2 = inv * inv;
        let series = inv
            * (1.0 / 12.0
                - inv2
                    * (1.0 / 360.0
                        - inv2
                            * (1.0 / 1260.0
                                - inv2
                                    * (1.0 / 1680.0
                                        - inv2 * (1.0 / 1188.0 - inv2 * (691.0 / 360_360.0))))));
        return (x - 0.5) * x.ln() - x + HALF_LN_2PI + series;
    }
    let x = x - 1.0;
    let mut a = LANCZOS[0];
    for (i, &c) in LANCZOS.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    let t = x + 7.5;
    HALF_LN_2PI + (x + 0.5) * t.ln() - t + a.ln()
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
pub(crate) fn digamma(x: f64) -> f64 {
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
pub(crate) fn trigamma(x: f64) -> f64 {
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

/// `x^a e^-x / Γ(a)`, the common prefactor of the incomplete gamma forms.
fn gamma_prefactor(a: f64, x: f64) -> f64 {
    (a * x.ln() - x - ln_gamma(a)).exp()
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

/// Continued fraction for `Q(a, x)`, valid (fast) for `x >= a + 1`.
fn gamma_q_fraction(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..MAX_ITER {
        let i = i as f64;
        let an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    gamma_prefactor(a, x) * h
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
    } else if x < a + 1.0 {
        gamma_p_series(a, x)
    } else {
        1.0 - gamma_q_fraction(a, x)
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
    } else if x < a + 1.0 {
        1.0 - gamma_p_series(a, x)
    } else {
        gamma_q_fraction(a, x)
    }
}

/// Lentz continued fraction of the incomplete beta function.
fn beta_fraction(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta `I_x(a, b)` for `a, b > 0`, `x` in `[0, 1]`,
/// with `y = 1 - x` supplied by the caller (so `ln(1 - x)` keeps its
/// precision when `x` is near 1).
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
    let front = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * y.ln()).exp();
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
}
