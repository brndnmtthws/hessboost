//! The noising processes: the diffusion SDEs' Gaussian perturbation kernels
//! and reverse-time coefficients, the flow-matching paths, the training-time
//! distributions of `t`, and the standard-normal draws.

use super::{FlowPath, Sde, TimeSampling};
use crate::rng::{GOLDEN, Rng, mix64};

/// Smallest time drawn for training and the end point of reverse-time
/// integration (Treeffuser's and DiffGBM's `EPS`).
pub(super) const T_EPS: f64 = 1e-5;

/// Grid size of the `t ↦ ln(noise scale)` lookup that inverts
/// [`TimeSampling::LogNoiseNormal`] draws (DiffGBM's `table_size`).
const TABLE_SIZE: usize = 1024;

/// Fraction of flow-matching training rows whose uniform `t` is set to
/// exactly `1` (DiffGBM's `uniform_endpoint_fraction`), so the regressor
/// sees the prior the sampler starts from.
const ENDPOINT_FRACTION: f64 = 0.05;

impl Sde {
    /// `(α(t), σ(t))` of the perturbation kernel
    /// `p_t(y | y₀) = N(α(t)·y₀, σ(t)²)`.
    pub(super) fn marginal(&self, t: f64) -> (f64, f64) {
        match *self {
            Sde::VarianceExploding {
                sigma_min,
                sigma_max,
            } => {
                let sigma = sigma_min * (sigma_max / sigma_min).powf(t);
                (1.0, (sigma * sigma - sigma_min * sigma_min).sqrt())
            }
            Sde::VariancePreserving { beta_min, beta_max } => {
                let integral = beta_integral(beta_min, beta_max, t);
                ((-0.5 * integral).exp(), (-(-integral).exp_m1()).sqrt())
            }
            Sde::SubVariancePreserving { beta_min, beta_max } => {
                let integral = beta_integral(beta_min, beta_max, t);
                ((-0.5 * integral).exp(), -(-integral).exp_m1())
            }
        }
    }

    /// `(c(t), g(t)²)` of the forward SDE `dY = c(t)·Y dt + g(t) dW`.
    pub(super) fn drift_diffusion(&self, t: f64) -> (f64, f64) {
        match *self {
            Sde::VarianceExploding {
                sigma_min,
                sigma_max,
            } => {
                // `σ(t)² = σ_min² (σ_max/σ_min)^{2t}`, so `2σσ' = 2σ² ln(σ_max/σ_min)`.
                let ratio = sigma_max / sigma_min;
                let sigma = sigma_min * ratio.powf(t);
                (0.0, 2.0 * sigma * sigma * ratio.ln())
            }
            Sde::VariancePreserving { beta_min, beta_max } => {
                let beta = beta_min + (beta_max - beta_min) * t;
                (-0.5 * beta, beta)
            }
            Sde::SubVariancePreserving { beta_min, beta_max } => {
                let beta = beta_min + (beta_max - beta_min) * t;
                let discount = -(-2.0 * beta_integral(beta_min, beta_max, t)).exp_m1();
                (-0.5 * beta, beta * discount)
            }
        }
    }

    /// Standard deviation of the prior sampling starts from (the
    /// distribution `p_T` approaches).
    pub(super) fn prior_std(&self) -> f64 {
        match *self {
            Sde::VarianceExploding { sigma_max, .. } => sigma_max,
            Sde::VariancePreserving { .. } | Sde::SubVariancePreserving { .. } => 1.0,
        }
    }
}

/// `∫₀ᵗ β(s) ds` for the linear schedule `β(s) = β_min + (β_max - β_min) s`.
fn beta_integral(beta_min: f64, beta_max: f64, t: f64) -> f64 {
    beta_min * t + 0.5 * (beta_max - beta_min) * t * t
}

/// The coefficients of a flow-matching path at `t`.
#[derive(Debug, Clone, Copy)]
pub(super) struct PathCoefficients {
    /// Signal scale `a(t)` in `y_t = a(t)·y₀ + b(t)·z`.
    pub(super) a: f64,
    /// Noise scale `b(t)`.
    pub(super) b: f64,
    /// `a'(t)`.
    pub(super) da: f64,
    /// `b'(t)`.
    pub(super) db: f64,
}

impl FlowPath {
    /// `a(t)`, `b(t)` and their derivatives: the path interpolates
    /// `y_t = a·y₀ + b·z` and its target velocity is `a'·y₀ + b'·z`.
    pub(super) fn coefficients(&self, t: f64) -> PathCoefficients {
        match *self {
            FlowPath::Linear => PathCoefficients {
                a: 1.0 - t,
                b: t,
                da: -1.0,
                db: 1.0,
            },
            FlowPath::Trigonometric => {
                let half_pi = std::f64::consts::FRAC_PI_2;
                let (b, a) = (half_pi * t).sin_cos();
                PathCoefficients {
                    a,
                    b,
                    da: -half_pi * b,
                    db: half_pi * a,
                }
            }
            FlowPath::VariancePreserving { beta_min, beta_max } => {
                let integral = 0.5 * beta_min * t + 0.25 * (beta_max - beta_min) * t * t;
                let derivative = f64::midpoint(beta_min, (beta_max - beta_min) * t);
                let alpha_bar = (-integral).exp();
                let a = alpha_bar.sqrt();
                let b = (-(-integral).exp_m1()).clamp(0.0, 1.0).sqrt();
                // `2bb' = T'·ᾱ`; the floor only matters at `t = 0`, which
                // neither training nor sampling evaluates.
                PathCoefficients {
                    a,
                    b,
                    da: -0.5 * derivative * a,
                    db: derivative * alpha_bar / (2.0 * b.max(1e-12)),
                }
            }
        }
    }
}

/// Training times for `n` rows drawn from `sampling` on `[T_EPS, 1]`.
/// `noise_scale` is the process's noise scale (`σ(t)` of an SDE, `b(t)` of a
/// flow path), increasing in `t`; `anchor_endpoint` sets a
/// [`ENDPOINT_FRACTION`] of uniform draws to `t = 1` (flow matching).
pub(super) fn draw_times(
    n: usize,
    sampling: TimeSampling,
    noise_scale: impl Fn(f64) -> f64,
    anchor_endpoint: bool,
    rng: &mut Rng,
    normal: &mut Normal,
) -> Vec<f64> {
    match sampling {
        TimeSampling::Uniform => {
            let mut t: Vec<f64> = (0..n).map(|_| rng.f64() * (1.0 - T_EPS) + T_EPS).collect();
            if anchor_endpoint && n > 0 {
                let count = ((n as f64 * ENDPOINT_FRACTION).round_ties_even() as usize).clamp(1, n);
                // A uniform `count`-subset: the head of a partial Fisher–Yates shuffle.
                let mut rows: Vec<usize> = (0..n).collect();
                for i in 0..count {
                    let j = rng.range(i..n);
                    rows.swap(i, j);
                    t[rows[i]] = 1.0;
                }
            }
            t
        }
        TimeSampling::LogNoiseNormal { mean, std } => {
            let (log_scale, times) = log_noise_table(noise_scale);
            let (lo, hi) = (log_scale[0], log_scale[TABLE_SIZE - 1]);
            (0..n)
                .map(|_| {
                    let draw = (mean + std * normal.draw(rng)).clamp(lo, hi);
                    interpolate(&log_scale, &times, draw)
                })
                .collect()
        }
    }
}

/// `(ln noise_scale(tᵢ), tᵢ)` on an even grid of [`TABLE_SIZE`] times over
/// `[T_EPS, 1]`.
fn log_noise_table(noise_scale: impl Fn(f64) -> f64) -> (Vec<f64>, Vec<f64>) {
    let step = (1.0 - T_EPS) / (TABLE_SIZE - 1) as f64;
    let times: Vec<f64> = (0..TABLE_SIZE)
        .map(|i| {
            if i == TABLE_SIZE - 1 {
                1.0
            } else {
                T_EPS + i as f64 * step
            }
        })
        .collect();
    let log_scale = times.iter().map(|&t| noise_scale(t).ln()).collect();
    (log_scale, times)
}

/// Piecewise-linear interpolation of `(xs, ys)` at `x` (NumPy's `interp`),
/// `xs` non-decreasing and `x` within its range.
fn interpolate(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let hi = xs.partition_point(|&v| v < x).clamp(1, xs.len() - 1);
    let (x0, x1) = (xs[hi - 1], xs[hi]);
    if x1 <= x0 {
        return ys[hi];
    }
    let w = (x - x0) / (x1 - x0);
    ys[hi - 1] + w * (ys[hi] - ys[hi - 1])
}

/// Standard-normal draws from a sequential [`Rng`] by the Box–Muller
/// transform, which yields them in pairs.
#[derive(Debug, Default)]
pub(super) struct Normal {
    spare: Option<f64>,
}

impl Normal {
    /// The next standard-normal draw.
    pub(super) fn draw(&mut self, rng: &mut Rng) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let (z0, z1) = box_muller(rng.f64(), rng.f64());
        self.spare = Some(z1);
        z0
    }
}

/// Two independent standard normals from two uniforms on `[0, 1)`.
fn box_muller(u1: f64, u2: f64) -> (f64, f64) {
    // `1 - u1` is in `(0, 1]`, so the logarithm is finite.
    let radius = (-2.0 * (1.0 - u1).ln()).sqrt();
    let (sin, cos) = (std::f64::consts::TAU * u2).sin_cos();
    (radius * cos, radius * sin)
}

/// The standard-normal draw `counter` of the counter-based stream `key`,
/// independent of every other `(key, counter)`: the sampler's noise, so a
/// draw depends on its row, sample, step, and output only (never on
/// batching or the thread count).
pub(super) fn keyed_normal(key: u64, counter: u64) -> f64 {
    let uniform = |i: u64| {
        let bits = mix64(key.wrapping_add(i.wrapping_add(1).wrapping_mul(GOLDEN)));
        (bits >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    };
    let pair = counter.wrapping_mul(2);
    box_muller(uniform(pair), uniform(pair.wrapping_add(1))).0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Finite-difference check of the paths' derivatives and of the SDEs'
    /// kernel moments: for the linear SDE `dY = cY dt + g dW`, `α' = cα` and
    /// `(σ²)' = 2cσ² + g²`, the identities tying the reverse drift to the
    /// perturbation kernel the regressor is trained on.
    #[test]
    fn coefficients_match_their_kernels() {
        let h = 1e-6;
        for path in [
            FlowPath::Linear,
            FlowPath::Trigonometric,
            FlowPath::VariancePreserving {
                beta_min: 0.1,
                beta_max: 20.0,
            },
        ] {
            for t in [0.1, 0.5, 0.9] {
                let (lo, hi, at) = (
                    path.coefficients(t - h),
                    path.coefficients(t + h),
                    path.coefficients(t),
                );
                assert!(((hi.a - lo.a) / (2.0 * h) - at.da).abs() < 1e-5, "{path:?}");
                assert!(((hi.b - lo.b) / (2.0 * h) - at.db).abs() < 1e-5, "{path:?}");
            }
        }
        for sde in [
            Sde::VarianceExploding {
                sigma_min: 0.01,
                sigma_max: 20.0,
            },
            Sde::VariancePreserving {
                beta_min: 0.1,
                beta_max: 20.0,
            },
            Sde::SubVariancePreserving {
                beta_min: 0.1,
                beta_max: 20.0,
            },
        ] {
            for t in [0.1, 0.5, 0.9] {
                let alpha = |t: f64| sde.marginal(t).0;
                let var = |t: f64| sde.marginal(t).1.powi(2);
                let (c, g2) = sde.drift_diffusion(t);
                let dalpha = (alpha(t + h) - alpha(t - h)) / (2.0 * h);
                assert!((dalpha - c * alpha(t)).abs() < 1e-4, "{sde:?}");
                let dvar = (var(t + h) - var(t - h)) / (2.0 * h);
                let expected = 2.0 * c * var(t) + g2;
                assert!(
                    (dvar - expected).abs() < 1e-4 * expected.abs().max(1.0),
                    "{sde:?}"
                );
            }
        }
    }

    #[test]
    fn log_noise_draws_invert_the_noise_scale() {
        let sde = Sde::VarianceExploding {
            sigma_min: 0.01,
            sigma_max: 20.0,
        };
        let mut rng = Rng::new(3);
        let times = draw_times(
            2000,
            TimeSampling::LogNoiseNormal {
                mean: -1.2,
                std: 1.2,
            },
            |t| sde.marginal(t).1,
            false,
            &mut rng,
            &mut Normal::default(),
        );
        let logs: Vec<f64> = times.iter().map(|&t| sde.marginal(t).1.ln()).collect();
        let mean = logs.iter().sum::<f64>() / logs.len() as f64;
        assert!((mean - -1.2).abs() < 0.1, "mean log σ {mean}");
        assert!(times.iter().all(|&t| (T_EPS..=1.0).contains(&t)));
    }
}
