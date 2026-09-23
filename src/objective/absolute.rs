//! Mean absolute error regression (`reg:absoluteerror`).

use super::{GradPair, Objective, fit_stump, weighted_label_mean};

/// XGBoost's per-output automatic residual scale, shared by the quantile and
/// absolute-error objectives: `S_j = (Σᵢ wᵢ √|pᵢⱼ − yᵢⱼ| / Σᵢ wᵢ)²` for each of
/// the `k` outputs of `preds` (`[row][output]`), where `label(i, j)` is the
/// label output `j` of row `i` is fitted to. Each term `wᵢ √|r|` is formed in
/// `f32` and summed in `f64` (XGBoost `common::Reduce`); a total weight
/// within `1e-6` of zero (`common::CloseTo`) gives `S_j = 0`.
pub(super) fn residual_scales(
    preds: &[f32],
    weights: Option<&[f32]>,
    k: usize,
    label: impl Fn(usize, usize) -> f32,
) -> Vec<f32> {
    let n = preds.len() / k;
    let sum_weight = weights.map_or(n as f64, |w| w.iter().map(|&wi| f64::from(wi)).sum());
    (0..k)
        .map(|j| {
            if sum_weight.abs() < 1e-6 {
                return 0.0;
            }
            let root_sum: f64 = (0..n)
                .map(|i| {
                    let w = weights.map_or(1.0, |ws| ws[i]);
                    f64::from(w * (preds[i * k + j] - label(i, j)).abs().sqrt())
                })
                .sum();
            let root_mean = root_sum / sum_weight;
            (root_mean * root_mean) as f32
        })
        .collect()
}

/// Mean absolute error (`reg:absoluteerror`) with XGBoost 3.4's automatic
/// smooth majorization of the L1 loss.
///
/// Supports label matrices: output `j` fits label column `j`
/// ([`DMatrix::with_label_matrix`](crate::data::DMatrix::with_label_matrix)),
/// so the objective has one output per target. Each gradient call computes,
/// per output, the residual scale `δ_j = (Σ wᵢ √|rᵢⱼ| / Σ wᵢ)²` of `r =
/// margin − label` and emits `g = w·r·c`, `h = w·c` with `c = δ /
/// hypot(δ, r)` (`c = 1` when both are zero): the pseudo-Huber score with
/// its majorizing curvature `1/√(1 + (r/δ)²)`, all in `f32`.
///
/// The intercept of each target is one Newton step of this surrogate from
/// the target's (weighted) label mean, added to that mean — not a median.
/// A total weight within `1e-6` of zero gives zero intercepts. Predictions
/// are margins (no link); the default metric is `mae`.
#[derive(Debug, Clone, Copy)]
pub struct AbsoluteErrorObjective {
    n_targets: usize,
}

impl AbsoluteErrorObjective {
    /// Create for `n_targets` label columns (at least one).
    pub fn new(n_targets: usize) -> Self {
        AbsoluteErrorObjective {
            n_targets: n_targets.max(1),
        }
    }
}

impl Default for AbsoluteErrorObjective {
    fn default() -> Self {
        Self::new(1)
    }
}

impl Objective for AbsoluteErrorObjective {
    fn name(&self) -> &'static str {
        "reg:absoluteerror"
    }

    fn n_outputs(&self) -> usize {
        self.n_targets
    }

    /// `labels` holds `n_rows * n_targets` values laid out `[row][target]`,
    /// like `preds` and `out`.
    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        let k = self.n_targets;
        let n = preds.len() / k;
        debug_assert_eq!(labels.len(), preds.len());
        debug_assert_eq!(out.len(), preds.len());
        let scales = residual_scales(preds, weights, k, |i, j| labels[i * k + j]);
        let kernel =
            |preds: &[f32], labels: &[f32], weights: Option<&[f32]>, out: &mut [GradPair]| {
                for (idx, ((&p, &y), o)) in preds.iter().zip(labels).zip(out.iter_mut()).enumerate()
                {
                    let (i, j) = (idx / k, idx % k);
                    let residual = p - y;
                    let delta = scales[j];
                    let norm = delta.hypot(residual);
                    let curvature = if norm > 0.0 { delta / norm } else { 1.0 };
                    let w = weights.map_or(1.0, |ws| ws[i]);
                    *o = GradPair::new(w * residual * curvature, w * curvature);
                }
            };
        if k == 1 {
            super::rowwise_gradient(n, 1, preds, labels, weights, out, kernel);
        } else {
            kernel(preds, labels, weights, out);
        }
    }

    fn base_margins(
        &self,
        labels: &[f32],
        weights: Option<&[f32]>,
        _group: Option<&crate::data::GroupInfo>,
    ) -> Vec<f32> {
        let k = self.n_targets;
        let n = labels.len() / k;
        let sum_weight = weights.map_or(n as f64, |w| w.iter().map(|&wi| f64::from(wi)).sum());
        if sum_weight.abs() < 1e-6 {
            return vec![0.0; k];
        }
        let mean: Vec<f32> = (0..k)
            .map(|j| {
                let column: Vec<f32> = labels.iter().skip(j).step_by(k).copied().collect();
                weighted_label_mean(&column, weights)
            })
            .collect();
        let preds: Vec<f32> = (0..n).flat_map(|_| mean.iter().copied()).collect();
        let mut gpair = vec![GradPair::default(); n * k];
        self.gradient(&preds, labels, weights, &mut gpair);
        let mut out = fit_stump(&gpair, k);
        for (v, m) in out.iter_mut().zip(&mean) {
            *v += m;
        }
        out
    }

    fn default_metric(&self) -> String {
        "mae".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `δ = (Σ√|r| / n)²`; `g = r·δ/√(δ² + r²)`, `h = δ/√(δ² + r²)`, which
    /// tends to `sign(r)` and `δ/|r|` for large residuals.
    #[test]
    fn gradient_is_scaled_pseudo_huber_score() {
        let obj = AbsoluteErrorObjective::default();
        let mut out = vec![GradPair::default(); 2];
        obj.gradient(&[4.0, 0.0], &[0.0, 1.0], None, &mut out);
        let delta = 1.5f64.powi(2) as f32; // ((√4 + √1) / 2)²
        let norm0 = delta.hypot(4.0);
        assert_eq!(out[0], GradPair::new(4.0 * (delta / norm0), delta / norm0));
        let norm1 = delta.hypot(-1.0);
        assert_eq!(out[1], GradPair::new(-(delta / norm1), delta / norm1));
    }

    /// Zero residuals everywhere: `δ = 0` and `hypot = 0`, so the curvature
    /// defaults to 1 with a zero gradient; zero total weight zeroes all.
    #[test]
    fn gradient_edge_cases() {
        let obj = AbsoluteErrorObjective::default();
        let mut out = vec![GradPair::default(); 2];
        obj.gradient(&[1.0, 2.0], &[1.0, 2.0], None, &mut out);
        assert_eq!(out, vec![GradPair::new(0.0, 1.0); 2]);
        obj.gradient(&[3.0, 2.0], &[1.0, 2.0], Some(&[0.0, 0.0]), &mut out);
        assert!(
            out.iter().all(|p| p.grad == 0.0 && p.hess == 0.0),
            "{out:?}"
        );
    }

    /// Output `j` fits label column `j` with its own scale.
    #[test]
    fn multi_target_uses_each_label_column() {
        let two = AbsoluteErrorObjective::new(2);
        assert_eq!(two.n_outputs(), 2);
        let preds = [0.0f32, 0.0, 0.0, 0.0];
        let labels = [1.0f32, -9.0, 1.0, -9.0];
        let mut out = vec![GradPair::default(); 4];
        two.gradient(&preds, &labels, None, &mut out);
        let mut single = vec![GradPair::default(); 2];
        AbsoluteErrorObjective::default().gradient(&[0.0, 0.0], &[-9.0, -9.0], None, &mut single);
        assert_eq!([out[1], out[3]], [single[0], single[1]]);
        assert!(out[0].grad < 0.0 && out[1].grad > 0.0);

        let margins = two.base_margins(&labels, None, None);
        // Constant columns: the Newton step from the mean is zero.
        assert_eq!(margins, vec![1.0, -9.0]);
    }

    /// The intercept is a Newton step from the mean, not the median: labels
    /// {0, 0, 10} have median 0 and mean 10/3, and the step moves from the
    /// mean towards (not onto) the median.
    #[test]
    fn intercept_is_newton_step_from_mean() {
        let obj = AbsoluteErrorObjective::default();
        let labels = [0.0f32, 0.0, 10.0];
        let mean = weighted_label_mean(&labels, None);
        let preds = [mean; 3];
        let mut gpair = vec![GradPair::default(); 3];
        obj.gradient(&preds, &labels, None, &mut gpair);
        let expected = fit_stump(&gpair, 1)[0] + mean;
        let got = obj.base_margins(&labels, None, None);
        assert_eq!(got, vec![expected]);
        assert!(got[0] > 0.0 && got[0] < mean, "{got:?}");
        assert_eq!(
            obj.base_margins(&labels, Some(&[0.0, 0.0, 0.0]), None),
            vec![0.0]
        );
    }

    /// A two-column label matrix trains each output exactly like a
    /// single-target model on that column: the per-output scales, intercepts,
    /// and trees never mix the columns.
    #[test]
    fn multi_target_training_matches_per_column_models() {
        use crate::config::TrainingParams;
        use crate::data::DMatrix;
        let n = 64;
        let x: Vec<f32> = (0..n * 2)
            .map(|i| ((i * 37) % 101) as f32 / 101.0)
            .collect();
        let y0: Vec<f32> = (0..n).map(|i| x[2 * i] * 4.0 - x[2 * i + 1]).collect();
        let y1: Vec<f32> = (0..n).map(|i| (x[2 * i + 1] * 9.0).sin() * 3.0).collect();
        let matrix: Vec<f32> = y0.iter().zip(&y1).flat_map(|(&a, &b)| [a, b]).collect();
        let params = TrainingParams::builder()
            .objective("reg:absoluteerror")
            .max_depth(3)
            .build()
            .unwrap();
        let base = DMatrix::from_dense(&x, n, 2).unwrap();
        let both = base.clone().with_label_matrix(&matrix, 2).unwrap();
        let joint = crate::learner::train(&params, &both, 5).unwrap();
        assert_eq!(joint.n_outputs(), 2);
        let joint_pred = joint.predict(&base).unwrap();
        for (j, y) in [&y0, &y1].into_iter().enumerate() {
            let single = base.clone().with_labels(y).unwrap();
            let alone = crate::learner::train(&params, &single, 5).unwrap();
            let pred = alone.predict(&base).unwrap();
            let column: Vec<f32> = joint_pred.iter().skip(j).step_by(2).copied().collect();
            assert_eq!(column, pred, "target {j}");
        }
    }
}
