//! Pointwise regression metrics without a vector kernel: `rmsle`, `mape`,
//! and `mphe`. Each row's loss is computed in `f32` and multiplied by the
//! row weight in `f32` before the `f64` sums, exactly as XGBoost's
//! `elementwise_metric.cu` reduction does.

use super::{Metric, consistent, weighted_mean};

/// `(Σ wᵢ·loss(yᵢ, pᵢ), Σ wᵢ)` with the per-row product in `f32`; absent
/// weights count as `1`. Inconsistent lengths give `(NaN, 1)`, so the
/// metric is NaN.
fn weighted_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    loss: impl Fn(f32, f32) -> f32,
) -> (f64, f64) {
    if !consistent(preds, labels, weights, 1) {
        return (f64::NAN, 1.0);
    }
    let mut total = 0.0f64;
    let mut weight = 0.0f64;
    for (i, (&p, &y)) in preds.iter().zip(labels).enumerate() {
        let w = weights.map_or(1.0, |ws| ws[i]);
        total += f64::from(loss(y, p) * w);
        weight += f64::from(w);
    }
    (total, weight)
}

/// Root mean squared log error (`rmsle`):
/// `√(Σ w [ln1p(y) − ln1p(p)]² / Σ w)`. Predictions or labels at or below
/// `-1` give NaN, as upstream (no clamp).
#[derive(Debug, Clone, Copy, Default)]
pub struct Rmsle;

impl Metric for Rmsle {
    fn name(&self) -> &'static str {
        "rmsle"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        weighted_mean(weighted_sum(preds, labels, weights, |y, p| {
            let diff = y.ln_1p() - p.ln_1p();
            diff * diff
        }))
        .sqrt()
    }
}

/// Mean absolute percentage error (`mape`): `Σ w |(y − p) / y| / Σ w`. A
/// zero label divides by zero (infinite or NaN), as upstream.
#[derive(Debug, Clone, Copy, Default)]
pub struct Mape;

impl Metric for Mape {
    fn name(&self) -> &'static str {
        "mape"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        weighted_mean(weighted_sum(preds, labels, weights, |y, p| {
            ((y - p) / y).abs()
        }))
    }
}

/// Mean pseudo-Huber error (`mphe`) with slope `δ` (XGBoost `huber_slope`):
/// `Σ w δ² (√(1 + ((y − p)/δ)²) − 1) / Σ w`. This is the plain pseudo-Huber
/// loss, without the objective's factor conventions.
#[derive(Debug, Clone, Copy)]
pub struct PseudoHuberError {
    slope: f32,
}

impl PseudoHuberError {
    /// `mphe` with slope `δ`; the caller guarantees `δ != 0`.
    pub(super) fn new(slope: f32) -> Self {
        PseudoHuberError { slope }
    }
}

impl Metric for PseudoHuberError {
    fn name(&self) -> &'static str {
        "mphe"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        let slope = self.slope;
        weighted_mean(weighted_sum(preds, labels, weights, |y, p| {
            let scaled = (y - p) / slope;
            slope * slope * ((1.0 + scaled * scaled).sqrt() - 1.0)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsle_is_rmse_in_log1p_space() {
        // ln1p(e - 1) = 1 and ln1p(0) = 0: one unit error on each row.
        let e1 = std::f32::consts::E - 1.0;
        let v = Rmsle.eval(&[e1, 0.0], &[0.0, e1], None);
        assert!((v - 1.0).abs() < 1e-6, "{v}");
        assert_eq!(Rmsle.eval(&[3.0, 5.0], &[3.0, 5.0], None), 0.0);
    }

    #[test]
    fn mape_is_relative_to_label_and_weighted() {
        // |(2-1)/2| = 0.5 with weight 3, |(4-5)/4| = 0.25 with weight 1.
        let v = Mape.eval(&[1.0, 5.0], &[2.0, 4.0], Some(&[3.0, 1.0]));
        assert!((v - (0.5 * 3.0 + 0.25) / 4.0).abs() < 1e-7, "{v}");
    }

    /// At `r = 3δ/4` the exact value is `δ²(√(1 + 9/16) − 1) = δ²/4`; a
    /// factor-two convention or an unscaled residual would miss it.
    #[test]
    fn mphe_uses_slope_without_factor_two() {
        let v = PseudoHuberError::new(2.0).eval(&[0.0], &[1.5], None);
        assert!((v - 1.0).abs() < 1e-6, "{v}");
        let unit = PseudoHuberError::new(1.0).eval(&[0.0], &[0.75], None);
        assert!((unit - 0.25).abs() < 1e-6, "{unit}");
    }
}
