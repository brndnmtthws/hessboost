//! Alpha-list regression metrics: `quantile` (pinball loss) and `expectile`
//! (asymmetric squared loss), averaged over every alpha.

use super::{Metric, weighted_mean};
use crate::data::MetaInfo;
use crate::error::Result;
use crate::objective::validate_alphas;

/// Sum `loss(alpha, pred, label) · w` and the matching weights over every
/// (row, alpha, target) cell of `preds` laid out `[row][alpha][target]`
/// against `labels` laid out `[row][target]` (XGBoost's elementwise
/// `Reduce`: row weights are repeated for every alpha and target). `NaN`
/// when `preds` does not hold one value per label and alpha.
fn alpha_average(
    alpha: &[f32],
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    n_rows: usize,
    loss: impl Fn(f32, f32, f32) -> f32,
) -> f64 {
    if n_rows == 0 || preds.len() != labels.len() * alpha.len() {
        return f64::NAN;
    }
    let n_targets = labels.len() / n_rows;
    let (mut total, mut weight) = (0.0f64, 0.0f64);
    for (i, row) in preds.chunks_exact(alpha.len() * n_targets).enumerate() {
        let w = weights.map_or(1.0, |ws| ws[i]);
        let y_row = &labels[i * n_targets..(i + 1) * n_targets];
        for (&a, cells) in alpha.iter().zip(row.chunks_exact(n_targets)) {
            for (&p, &y) in cells.iter().zip(y_row) {
                total += f64::from(loss(a, p, y) * w);
                weight += f64::from(w);
            }
        }
    }
    weighted_mean((total, weight))
}

/// Pinball loss (`quantile`), XGBoost's `QuantileError`: for residual `d = y
/// − p`, `α·d` when `d ≥ 0` and `(α − 1)·d` otherwise, averaged over rows
/// and every `quantile_alpha` (predictions `[row][alpha]`, one label per row;
/// the weighted denominator is `len(alpha) · Σw`). It reads the configured
/// `quantile_alpha` whatever the objective, as XGBoost does, so with any
/// objective other than `reg:quantileerror` only a one-element list matches
/// the one prediction per row. A prediction count that is not labels ×
/// alphas evaluates to `NaN` (XGBoost raises an error).
#[derive(Debug, Clone)]
pub struct QuantileError {
    alpha: Vec<f32>,
}

impl QuantileError {
    /// Create for the quantile levels `alpha` (XGBoost `quantile_alpha`).
    ///
    /// # Errors
    ///
    /// `alpha` is empty, has an entry outside `[0, 1]`, or is not ascending.
    pub fn new(alpha: &[f64]) -> Result<Self> {
        Ok(QuantileError {
            alpha: validate_alphas("quantile_alpha", alpha)?,
        })
    }
}

impl Metric for QuantileError {
    fn name(&self) -> &'static str {
        "quantile"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        self.eval_info(preds, &MetaInfo::new(labels, weights, None))
    }

    fn eval_info(&self, preds: &[f32], info: &MetaInfo) -> f64 {
        alpha_average(
            &self.alpha,
            preds,
            info.labels,
            info.weights,
            info.n_rows,
            |a, p, y| {
                let d = y - p;
                let sign = f32::from(u8::from(d >= 0.0));
                (a * sign * d) - (1.0 - a) * (1.0 - sign) * d
            },
        )
    }
}

/// Expectile loss (`expectile`), XGBoost's `ExpectileError`: `a·(p − y)²`
/// with `a = 1 − α` for `p ≥ y` and `α` otherwise, averaged over rows and
/// every `expectile_alpha` exactly like [`QuantileError`] (including its
/// `NaN` on a prediction count that is not labels × alphas).
#[derive(Debug, Clone)]
pub struct ExpectileError {
    alpha: Vec<f32>,
}

impl ExpectileError {
    /// Create for the expectile levels `alpha` (XGBoost `expectile_alpha`).
    ///
    /// # Errors
    ///
    /// `alpha` is empty, has an entry outside `[0, 1]`, or is not ascending.
    pub fn new(alpha: &[f64]) -> Result<Self> {
        Ok(ExpectileError {
            alpha: validate_alphas("expectile_alpha", alpha)?,
        })
    }
}

impl Metric for ExpectileError {
    fn name(&self) -> &'static str {
        "expectile"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        self.eval_info(preds, &MetaInfo::new(labels, weights, None))
    }

    fn eval_info(&self, preds: &[f32], info: &MetaInfo) -> f64 {
        alpha_average(
            &self.alpha,
            preds,
            info.labels,
            info.weights,
            info.n_rows,
            |a, p, y| {
                let diff = p - y;
                let scale = if diff >= 0.0 { 1.0 - a } else { a };
                scale * diff * diff
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two alphas, two rows: the denominator counts every (row, alpha) cell
    /// with its row weight.
    #[test]
    fn quantile_averages_pinball_over_alphas() {
        let m = QuantileError::new(&[0.25, 0.75]).unwrap();
        // Row 0 (y = 1): preds 0 (d = 1 → 0.25), 2 (d = −1 → 0.25).
        // Row 1 (y = 0): preds 0, 0 (d = 0 → 0).
        let preds = [0.0, 2.0, 0.0, 0.0];
        assert!((m.eval(&preds, &[1.0, 0.0], None) - 0.5 / 4.0).abs() < 1e-12);
        // Weight 3 on row 0, 1 on row 1: (3·0.5) / (2·4).
        let v = m.eval(&preds, &[1.0, 0.0], Some(&[3.0, 1.0]));
        assert!((v - 1.5 / 8.0).abs() < 1e-12, "{v}");
        assert!(m.eval(&preds[..2], &[1.0, 0.0], None).is_nan());
        assert!(QuantileError::new(&[]).is_err());
    }

    #[test]
    fn expectile_weights_residual_sides() {
        let m = ExpectileError::new(&[0.2]).unwrap();
        // Over-prediction by 2 → 0.8·4; under-prediction by 1 → 0.2·1.
        let v = m.eval(&[2.0, -1.0], &[0.0, 0.0], None);
        assert!((v - 1.7).abs() < 1e-6, "{v}");
    }
}
