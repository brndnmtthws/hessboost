//! Alpha-list regression metrics: `quantile` (pinball loss) and `expectile`
//! (asymmetric squared loss), averaged over every alpha; and their
//! threshold-ladder siblings `exceedance-logloss` and `exceedance-brier`.

use super::{Metric, weighted_mean};
use crate::data::MetaInfo;
use crate::error::{HessboostError, Result};
use crate::objective::Thresholds;

/// Sum `loss(alpha, pred, label) · w` and the matching weights over every
/// (row, alpha, target) cell of `preds` laid out `[row][alpha][target]`
/// against `labels` laid out `[row][target]` (XGBoost's elementwise
/// `Reduce`: row weights are repeated for every alpha and target). `NaN`
/// when there are no labels, `preds` does not hold one value per label and
/// alpha, the labels are not a whole number per row, or `weights` is not
/// one per row.
fn alpha_average(
    alpha: &[f32],
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    n_rows: usize,
    loss: impl Fn(f32, f32, f32) -> f32,
) -> f64 {
    if n_rows == 0
        || labels.is_empty()
        || !labels.len().is_multiple_of(n_rows)
        || labels.len().checked_mul(alpha.len()) != Some(preds.len())
        || weights.is_some_and(|w| w.len() != n_rows)
    {
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
pub(crate) struct QuantileError {
    alpha: Vec<f32>,
}

impl QuantileError {
    /// The pinball loss at the (validated) quantile levels `alpha`.
    pub(crate) fn new(alpha: Vec<f32>) -> Self {
        QuantileError { alpha }
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
            info.label_values(),
            info.weights,
            info.n_rows,
            |a, p, y| {
                let d = y - p;
                let sign = f32::from(u8::from(d >= 0.0));
                (a * sign * d) - (1.0 - a) * (1.0 - sign) * d
            },
        )
    }

    /// One prediction per alpha and label column.
    fn prediction_width(&self, info: &MetaInfo) -> Option<usize> {
        Some(self.alpha.len() * info.n_targets())
    }
}

/// Expectile loss (`expectile`), XGBoost's `ExpectileError`: `a·(p − y)²`
/// with `a = 1 − α` for `p ≥ y` and `α` otherwise, averaged over rows and
/// every `expectile_alpha` exactly like [`QuantileError`] (including its
/// `NaN` on a prediction count that is not labels × alphas).
#[derive(Debug, Clone)]
pub(crate) struct ExpectileError {
    alpha: Vec<f32>,
}

impl ExpectileError {
    /// The expectile loss at the (validated) levels `alpha`.
    pub(crate) fn new(alpha: Vec<f32>) -> Self {
        ExpectileError { alpha }
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
            info.label_values(),
            info.weights,
            info.n_rows,
            |a, p, y| {
                let diff = p - y;
                let scale = if diff >= 0.0 { 1.0 - a } else { a };
                scale * diff * diff
            },
        )
    }

    /// One prediction per alpha and label column.
    fn prediction_width(&self, info: &MetaInfo) -> Option<usize> {
        Some(self.alpha.len() * info.n_targets())
    }
}

/// Binary log loss over a threshold ladder (`exceedance-logloss`, the
/// default metric of `reg:exceedance`): for each row and threshold `k_j`,
/// `logloss`'s term (with its `ε = 1e-16` floor) of the indicator `y ≥ k_j`
/// (compared in `f64`) against the predicted `P(y ≥ k_j)`, weighted by the
/// row weight and averaged over rows and thresholds (the weighted
/// denominator is `K · Σw`). Predictions are `[row][threshold]`, one label
/// per row; any other prediction count, or a `NaN` label, evaluates to
/// `NaN`.
#[derive(Debug, Clone)]
pub(crate) struct ExceedanceLogLoss {
    thresholds: Thresholds,
}

impl ExceedanceLogLoss {
    /// The loss at the thresholds `thresholds`.
    pub(crate) fn new(thresholds: Thresholds) -> Self {
        ExceedanceLogLoss { thresholds }
    }
}

impl Metric for ExceedanceLogLoss {
    fn name(&self) -> &'static str {
        "exceedance-logloss"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        let levels = self.thresholds.values();
        nan_unless_consistent!(preds, labels, weights, levels.len());
        let (mut total, mut weight) = (0.0f64, 0.0f64);
        for (i, (row, &y)) in preds.chunks_exact(levels.len()).zip(labels).enumerate() {
            if y.is_nan() {
                return f64::NAN;
            }
            let w = weights.map_or(1.0, |ws| f64::from(ws[i]));
            for (&p, &level) in row.iter().zip(levels) {
                let t = if f64::from(y) >= level { 1.0 } else { 0.0 };
                total += crate::simd::log_loss_term(t, f64::from(p)) * w;
                weight += w;
            }
        }
        weighted_mean((total, weight))
    }

    /// One label column only: every threshold reads the same label.
    fn supports_label_matrix(&self) -> bool {
        false
    }

    /// One prediction per threshold.
    fn prediction_width(&self, _info: &MetaInfo) -> Option<usize> {
        Some(self.thresholds.values().len())
    }
}

/// Trapezoid-weighted Brier score over a finite threshold grid. Unlike
/// log loss, only row weights enter the denominator: the score retains
/// the label's units. Both endpoints contribute, without extrapolated
/// tails; this is neither exact finite-interval integration nor full CRPS.
#[derive(Debug, Clone)]
pub(crate) struct ExceedanceBrier {
    thresholds: Thresholds,
    quadrature: Vec<f64>,
}

impl ExceedanceBrier {
    pub(crate) fn new(thresholds: Thresholds) -> Result<Self> {
        let levels = thresholds.values();
        if levels.len() < 2 {
            return Err(HessboostError::invalid_param(
                "eval_metric",
                "`exceedance-brier` requires at least two thresholds",
            ));
        }
        if !(levels[levels.len() - 1] - levels[0]).is_finite() {
            return Err(HessboostError::invalid_param(
                "eval_metric",
                "`exceedance-brier` requires a finite representable grid width",
            ));
        }
        let mut quadrature = Vec::with_capacity(levels.len());
        for j in 0..levels.len() {
            let lo = levels[j.saturating_sub(1)];
            let hi = levels[(j + 1).min(levels.len() - 1)];
            // Subtract before halving: avoids overflowing a sum of large
            // same-sign endpoints and preserves small endpoint gaps.
            let weight = (hi - lo) * 0.5;
            if !weight.is_finite() || weight <= 0.0 {
                return Err(HessboostError::invalid_param(
                    "eval_metric",
                    "`exceedance-brier` requires positive finite representable quadrature weights",
                ));
            }
            quadrature.push(weight);
        }
        Ok(Self {
            thresholds,
            quadrature,
        })
    }
}

impl Metric for ExceedanceBrier {
    fn name(&self) -> &'static str {
        "exceedance-brier"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        let levels = self.thresholds.values();
        nan_unless_consistent!(preds, labels, weights, levels.len());
        // Even a finite row score can overflow when multiplied by a weight
        // or summed over rows. Normalize first; f32 row weights sum safely
        // in f64 for every addressable input length.
        let weight = weights.map_or(labels.len() as f64, |ws| {
            ws.iter().map(|&w| f64::from(w)).sum::<f64>()
        });
        let mut total = 0.0;
        for (i, (row, &y)) in preds.chunks_exact(levels.len()).zip(labels).enumerate() {
            if !y.is_finite() {
                return f64::NAN;
            }
            let mut score = 0.0;
            for ((&p, &level), &q) in row.iter().zip(levels).zip(&self.quadrature) {
                if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                    return f64::NAN;
                }
                let target = if f64::from(y) >= level { 1.0 } else { 0.0 };
                let error = f64::from(p) - target;
                score += q * error * error;
            }
            if weight > 0.0 {
                let fraction = weights.map_or(1.0, |ws| f64::from(ws[i])) / weight;
                total += score * fraction;
            }
        }
        total
    }

    fn supports_label_matrix(&self) -> bool {
        false
    }

    fn prediction_width(&self, _info: &MetaInfo) -> Option<usize> {
        Some(self.thresholds.values().len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exceedance_brier_matches_interval_quadrature_and_row_weights() {
        let grid = [-2.0, -0.5, 3.0, 8.0];
        let m = ExceedanceBrier::new(Thresholds::new(grid).unwrap()).unwrap();
        let predictions = [0.875, 0.75, 0.5, 0.125, 1.0, 0.875, 0.75, 0.5];
        let labels = [-0.5, 8.0];
        let weights = [3.0, 1.0];
        // Independent interval-by-interval trapezoids, rather than the
        // implementation's precomputed per-threshold weights.
        let rows: Vec<f64> = predictions
            .chunks_exact(grid.len())
            .zip(labels)
            .map(|(row, y)| {
                grid.windows(2)
                    .zip(row.windows(2))
                    .map(|(ks, ps)| {
                        let square = |p: f32, k: f64| {
                            let target = if f64::from(y) >= k { 1.0 } else { 0.0 };
                            (f64::from(p) - target).powi(2)
                        };
                        (ks[1] - ks[0]) * (square(ps[0], ks[0]) + square(ps[1], ks[1])) / 2.0
                    })
                    .sum()
            })
            .collect();
        assert_eq!(
            m.eval(&predictions, &labels, None),
            f64::midpoint(rows[0], rows[1])
        );
        assert_eq!(
            m.eval(&predictions, &labels, Some(&weights)),
            (3.0 * rows[0] + rows[1]) / 4.0
        );
        assert_eq!(
            m.eval(&predictions, &labels, Some(&[6.0, 2.0])),
            m.eval(&predictions, &labels, Some(&weights))
        );
        // Changing label units scales the score rather than canceling
        // out through normalization by total threshold weight.
        let scaled = ExceedanceBrier::new(Thresholds::new(grid.map(|k| 2.0 * k)).unwrap()).unwrap();
        assert_eq!(
            scaled.eval(&predictions, &labels.map(|y| 2.0 * y), Some(&weights)),
            2.0 * m.eval(&predictions, &labels, Some(&weights))
        );
    }

    #[test]
    fn exceedance_brier_includes_both_endpoints_and_threshold_equality() {
        let m = ExceedanceBrier::new(Thresholds::new([0.0, 1.0, 4.0]).unwrap()).unwrap();
        assert_eq!(m.eval(&[0.0, 0.0, 0.0], &[0.0], None), 0.5);
        assert_eq!(m.eval(&[1.0, 0.0, 1.0], &[0.0], None), 1.5);
        assert_eq!(m.eval(&[1.0, 1.0, 0.0], &[4.0], None), 1.5);
        assert_eq!(m.eval(&[1.0, 1.0, 1.0], &[4.0], None), 0.0);
    }

    #[test]
    fn finite_grid_brier_is_not_full_or_exact_interval_crps() {
        let m = ExceedanceBrier::new(Thresholds::new([0.0, 2.0]).unwrap()).unwrap();
        // Point forecast at 3 and observation at 4 agree everywhere on
        // [0, 2], but their full CRPS is |3 - 4| = 1 from the omitted tail.
        assert_eq!(m.eval(&[1.0, 1.0], &[4.0], None), 0.0);
        // Point forecast at 0.25 and observation at 1.75 also agree at
        // both grid points, yet exact integration inside [0, 2] is 1.5.
        assert_eq!(m.eval(&[1.0, 0.0], &[1.75], None), 0.0);
    }

    #[test]
    fn exceedance_brier_is_proper_at_each_threshold() {
        let m = ExceedanceBrier::new(Thresholds::new([0.0, 1.0, 4.0]).unwrap()).unwrap();
        let labels = [-1.0, 0.0, 2.0, 4.0];
        let true_probabilities = [0.75, 0.5, 0.25];
        let optimum = m.eval(&true_probabilities.repeat(4), &labels, None);
        for j in 0..3 {
            for delta in [-0.125, 0.125] {
                let mut perturbed = true_probabilities;
                perturbed[j] += delta;
                assert!(m.eval(&perturbed.repeat(4), &labels, None) > optimum);
            }
        }
    }

    #[test]
    fn exceedance_brier_averages_large_scores_without_overflow() {
        let m = ExceedanceBrier::new(Thresholds::new([0.0, 1e308]).unwrap()).unwrap();
        let expected = 2.5e307;
        assert_eq!(m.eval(&[0.5, 0.5], &[0.0], Some(&[8.0])), expected);
        assert_eq!(m.eval(&[0.5; 16], &[0.0; 8], None), expected);
        assert_eq!(m.eval(&[0.5; 4], &[0.0; 2], Some(&[0.0, 8.0])), expected);
    }

    #[test]
    fn exceedance_brier_preserves_f64_threshold_comparisons() {
        let m = ExceedanceBrier::new(Thresholds::new([1.0, 1.0 + f64::EPSILON]).unwrap()).unwrap();
        assert_eq!(m.eval(&[1.0, 0.0], &[1.0], None), 0.0);
        assert_eq!(m.eval(&[1.0, 1.0], &[1.0], None), f64::EPSILON / 2.0);
    }

    #[test]
    fn exceedance_brier_rejects_invalid_values_and_shapes() {
        let m = ExceedanceBrier::new(Thresholds::new([0.0, 1.0]).unwrap()).unwrap();
        for p in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 1.01] {
            assert!(m.eval(&[0.5, p], &[0.0], None).is_nan());
        }
        for y in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(m.eval(&[0.5, 0.5], &[y], None).is_nan());
        }
        assert!(m.eval(&[0.5], &[0.0], None).is_nan());
        assert!(m.eval(&[0.5, 0.5], &[0.0], Some(&[])).is_nan());
        assert!(!m.supports_label_matrix());
        assert_eq!(m.prediction_width(&MetaInfo::unlabeled(1)), Some(2));
    }

    #[test]
    fn exceedance_brier_validates_only_its_own_grid_requirements() {
        let single = Thresholds::new([0.0]).unwrap();
        assert!(ExceedanceBrier::new(single.clone()).is_err());
        assert_eq!(
            ExceedanceLogLoss::new(single).eval(&[1.0], &[0.0], None),
            0.0
        );
        // Finite endpoints do not imply representable integration width.
        let wide = Thresholds::new([-f64::MAX, f64::MAX]).unwrap();
        assert!(ExceedanceBrier::new(wide).is_err());
        let tiny = Thresholds::new([0.0, f64::from_bits(1)]).unwrap();
        assert!(ExceedanceBrier::new(tiny).is_err());
        // Large same-sign endpoints must not overflow through addition.
        let scale = 2.0f64.powi(1022);
        let grid = [2.0 * scale, 3.0 * scale, 3.5 * scale];
        let m = ExceedanceBrier::new(Thresholds::new(grid).unwrap()).unwrap();
        assert_eq!(m.eval(&[1.0, 1.0, 1.0], &[0.0], None), grid[2] - grid[0]);
    }

    /// Two alphas, two rows: the denominator counts every (row, alpha) cell
    /// with its row weight.
    #[test]
    fn quantile_averages_pinball_over_alphas() {
        let m = QuantileError::new(vec![0.25, 0.75]);
        // Row 0 (y = 1): preds 0 (d = 1 → 0.25), 2 (d = −1 → 0.25).
        // Row 1 (y = 0): preds 0, 0 (d = 0 → 0).
        let preds = [0.0, 2.0, 0.0, 0.0];
        assert!((m.eval(&preds, &[1.0, 0.0], None) - 0.5 / 4.0).abs() < 1e-12);
        // Weight 3 on row 0, 1 on row 1: (3·0.5) / (2·4).
        let v = m.eval(&preds, &[1.0, 0.0], Some(&[3.0, 1.0]));
        assert!((v - 1.5 / 8.0).abs() < 1e-12, "{v}");
        assert!(m.eval(&preds[..2], &[1.0, 0.0], None).is_nan());
    }

    #[test]
    fn expectile_weights_residual_sides() {
        let m = ExpectileError::new(vec![0.2]);
        // Over-prediction by 2 → 0.8·4; under-prediction by 1 → 0.2·1.
        let v = m.eval(&[2.0, -1.0], &[0.0, 0.0], None);
        assert!((v - 1.7).abs() < 1e-6, "{v}");
    }

    /// An unlabeled dataset with rows (zero inferred targets) evaluates to
    /// NaN instead of panicking on a zero chunk size.
    #[test]
    fn empty_labels_evaluate_to_nan() {
        let info = MetaInfo {
            n_rows: 3,
            weights: None,
            ..MetaInfo::unlabeled(0)
        };
        let q = QuantileError::new(vec![0.5]);
        let e = ExpectileError::new(vec![0.2, 0.8]);
        assert!(q.eval_info(&[], &info).is_nan());
        assert!(e.eval_info(&[], &info).is_nan());
    }

    /// Two thresholds, two rows: each cell is `logloss`'s term of its
    /// indicator, and the weighted denominator counts every cell.
    #[test]
    fn exceedance_logloss_averages_indicator_logloss() {
        let m = ExceedanceLogLoss::new(Thresholds::new([1.0, 3.0]).unwrap());
        // Row 0 (y = 2): indicators 1, 0. Row 1 (y = 3): indicators 1, 1.
        let preds = [0.8, 0.3, 0.9, 0.6];
        let labels = [2.0, 3.0];
        let cells = [-0.8f64.ln(), -0.7f64.ln(), -0.9f64.ln(), -0.6f64.ln()];
        let close = |a: f64, b: f64| (a - b).abs() < 1e-6;
        let v = m.eval(&preds, &labels, None);
        assert!(close(v, cells.iter().sum::<f64>() / 4.0), "{v}");
        let v = m.eval(&preds, &labels, Some(&[3.0, 1.0]));
        let want = (3.0 * (cells[0] + cells[1]) + cells[2] + cells[3]) / 8.0;
        assert!(close(v, want), "{v}");
        // A probability of exactly 0 for a reached threshold is floored.
        assert!(m.eval(&[0.0, 0.0], &[5.0], None).is_finite());
        assert!(m.eval(&preds[..3], &labels, None).is_nan());
        assert_eq!(m.name(), "exceedance-logloss");
        assert!(!m.supports_label_matrix());
    }
}
