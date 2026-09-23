//! Metrics of the distributional `dist:*` objectives (beyond XGBoost): the
//! mean negative log-likelihood (`nll`) and the mean continuous ranked
//! probability score (`crps`) of the predicted distributions.
//!
//! Both read predictions as the objective reports them, one row of natural
//! parameters per instance (`[row][parameter]`), and take the family from
//! the objective (`ObjectiveParams::distribution`).

use super::{Metric, consistent};
use crate::data::MetaInfo;
use crate::objective::{Dist, DistFamily};
use rayon::prelude::*;

/// Weighted mean of `score(dist_i, y_i)` over the rows. Per-row scores are
/// computed in parallel and summed sequentially in row order, so the value
/// does not depend on the thread count. NaN unless `preds` holds one row of
/// parameters per label and `weights` one weight per label.
fn mean_score(
    family: DistFamily,
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    score: impl Fn(&Dist, f64) -> f64 + Sync,
) -> f64 {
    let k = family.n_params();
    if !consistent(preds, labels, weights, k) {
        return f64::NAN;
    }
    let scores: Vec<f64> = preds
        .par_chunks_exact(k)
        .zip(labels.par_iter())
        .map(|(row, &y)| score(&Dist::from_row(family, row), f64::from(y)))
        .collect();
    let (mut total, mut weight) = (0.0f64, 0.0f64);
    for (i, s) in scores.into_iter().enumerate() {
        let w = weights.map_or(1.0, |ws| f64::from(ws[i]));
        total += w * s;
        weight += w;
    }
    super::weighted_mean((total, weight))
}

/// Mean negative log-likelihood `-ln p(y)` of the predicted distributions
/// (`nll`, the `dist:*` objectives' default metric): the log density for the
/// continuous families, the log probability mass for the count families.
#[derive(Debug, Clone, Copy)]
pub struct DistNll {
    family: DistFamily,
}

impl DistNll {
    /// The metric for `family`.
    pub fn new(family: DistFamily) -> Self {
        DistNll { family }
    }
}

impl Metric for DistNll {
    fn name(&self) -> &'static str {
        "nll"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        mean_score(self.family, preds, labels, weights, |d, y| -d.log_prob(y))
    }

    /// One row of the family's natural parameters per label.
    fn prediction_width(&self, _info: &MetaInfo) -> Option<usize> {
        Some(self.family.n_params())
    }

    fn supports_label_matrix(&self) -> bool {
        false
    }
}

/// Mean continuous ranked probability score of the predicted distributions
/// (`crps`), in the label's units; see [`Dist::crps`] for the closed forms
/// and the exact step sums of the count families.
#[derive(Debug, Clone, Copy)]
pub struct DistCrps {
    family: DistFamily,
}

impl DistCrps {
    /// The metric for `family`.
    pub fn new(family: DistFamily) -> Self {
        DistCrps { family }
    }
}

impl Metric for DistCrps {
    fn name(&self) -> &'static str {
        "crps"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        mean_score(self.family, preds, labels, weights, Dist::crps)
    }

    /// One row of the family's natural parameters per label.
    fn prediction_width(&self, _info: &MetaInfo) -> Option<usize> {
        Some(self.family.n_params())
    }

    fn supports_label_matrix(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ObjectiveParams;
    use crate::metric::create_metric;

    #[test]
    fn metrics_average_the_per_row_scores_with_weights() {
        // Two Normal rows: N(0, 1) at y = 0 and N(1, 2) at y = 3.
        let preds = [0.0f32, 1.0, 1.0, 2.0];
        let labels = [0.0f32, 3.0];
        let d0 = Dist::Normal {
            mu: 0.0,
            sigma: 1.0,
        };
        let d1 = Dist::Normal {
            mu: 1.0,
            sigma: 2.0,
        };
        let nll = DistNll::new(DistFamily::Normal);
        let crps = DistCrps::new(DistFamily::Normal);
        let expect = (-d0.log_prob(0.0) - 3.0 * d1.log_prob(3.0)) / 4.0;
        assert!((nll.eval(&preds, &labels, Some(&[1.0, 3.0])) - expect).abs() < 1e-12);
        let expect = f64::midpoint(d0.crps(0.0), d1.crps(3.0));
        assert!((crps.eval(&preds, &labels, None) - expect).abs() < 1e-12);
    }

    #[test]
    fn factory_takes_the_family_from_the_objective() {
        let dist = ObjectiveParams::defaults_for("dist:gamma");
        assert_eq!(dist.distribution, Some(DistFamily::Gamma));
        for name in ["nll", "crps"] {
            let metric = create_metric(name, 0, &dist).unwrap();
            assert_eq!(metric.name(), name);
            assert!(!metric.maximize());
            // Not a distributional objective: nothing to score.
            assert!(create_metric(name, 0, &ObjectiveParams::default()).is_err());
        }
    }
}
