//! Precision at `k` (`pre`, `pre@k`) for learning to rank.

use super::{Metric, argsort_desc, group_ranges};

/// XGBoost's default ranking cutoff (`LambdaRankParam::DefaultK`), used by
/// `pre` without an `@k` suffix.
const DEFAULT_TOP_K: usize = 32;

/// Precision at `k` over query groups (`pre`, `pre@k`), as XGBoost's
/// `EvalPrecision`. Each group ranks its documents by descending prediction
/// (stable for ties) and scores `Σ_{rank < n} label / n` with
/// `n = min(k, group size)`; groups are averaged with their weight (the
/// first document's weight; `1` when unweighted) and the result is capped
/// at `1`. Without group information the whole dataset is one query.
/// Plain `pre` uses `k = 32`, XGBoost's default cutoff. Labels must be
/// binary (within `1e-6` of `0` or `1`); XGBoost aborts otherwise, and this
/// metric returns NaN. Higher is better.
#[derive(Debug, Clone)]
pub struct Precision {
    k: usize,
    name: String,
}

impl Precision {
    /// `pre@k`, or `pre` (cutoff 32) when `k` is `None`. The caller
    /// guarantees `k >= 1`.
    pub(super) fn new(name: &str, k: Option<usize>) -> Self {
        Precision {
            k: k.unwrap_or(DEFAULT_TOP_K),
            name: name.to_string(),
        }
    }
}

impl Metric for Precision {
    fn name(&self) -> &str {
        &self.name
    }

    fn maximize(&self) -> bool {
        true
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        self.eval_grouped(preds, labels, weights, None)
    }

    fn eval_grouped(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        group: Option<&crate::data::GroupInfo>,
    ) -> f64 {
        // XGBoost `IsBinaryRel` with `kRtEps`.
        let binary = |y: f32| (y - 1.0).abs() < 1e-6 || y.abs() < 1e-6;
        if !labels.iter().all(|&y| binary(y)) {
            return f64::NAN;
        }
        let mut score = 0.0f64;
        let mut weight_sum = 0.0f64;
        for (start, end) in group_ranges(preds.len(), group) {
            let weight = weights.map_or(1.0f32, |w| w[start]);
            let order = argsort_desc(&preds[start..end]);
            let n = self.k.min(end - start);
            let hits: f64 = order[..n]
                .iter()
                .map(|&i| f64::from(labels[start + i] * weight))
                .sum();
            score += hits / n as f64;
            weight_sum += f64::from(weight);
        }
        if weight_sum > 0.0 {
            score /= weight_sum;
        }
        score.min(1.0)
    }

    /// Precision ranks one label per row within each query group.
    fn supports_label_matrix(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::GroupInfo;

    /// Two groups of four: the top-2 of the first holds one relevant
    /// document, the second two; `k` is clipped to a group's size.
    #[test]
    fn precision_at_k_per_group_and_clipped() {
        let preds = [0.9, 0.8, 0.1, 0.7, 0.2, 0.6, 0.5, 0.1];
        let labels = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let group = GroupInfo::from_sizes(&[4, 4]);
        let at2 = Precision::new("pre@2", Some(2));
        // Group 1 top-2 = rows 0, 1 -> 1/2; group 2 top-2 = rows 5, 6 -> 1.
        let v = at2.eval_grouped(&preds, &labels, None, Some(&group));
        assert!((v - 0.75).abs() < 1e-12, "{v}");
        // Default k = 32 covers each whole group: 2/4 in both.
        let v = Precision::new("pre", None).eval_grouped(&preds, &labels, None, Some(&group));
        assert!((v - 0.5).abs() < 1e-12, "{v}");
    }

    /// Group weights come from each group's first document.
    #[test]
    fn precision_weights_groups() {
        let preds = [0.9, 0.1, 0.9, 0.1];
        let labels = [1.0, 0.0, 0.0, 1.0];
        let weights = [3.0, 3.0, 1.0, 1.0];
        let group = GroupInfo::from_sizes(&[2, 2]);
        let v = Precision::new("pre@1", Some(1)).eval_grouped(
            &preds,
            &labels,
            Some(&weights),
            Some(&group),
        );
        assert!((v - 0.75).abs() < 1e-12, "{v}");
    }

    #[test]
    fn precision_rejects_graded_labels() {
        let v = Precision::new("pre", None).eval(&[0.5, 0.2], &[2.0, 0.0], None);
        assert!(v.is_nan());
    }

    /// `pre` reports XGBoost's names and maximizes; a zero cutoff and a zero
    /// `mphe` slope are parameter errors rather than NaN scores.
    #[test]
    fn factory_names_and_rejections() {
        use crate::config::{ObjectiveParams, TrainingParams};
        use crate::metric::create_metric;
        let defaults = ObjectiveParams::from_params(&TrainingParams::default());
        for name in ["pre", "pre@5"] {
            let m = create_metric(name, 0, &defaults).unwrap();
            assert_eq!(m.name(), name);
            assert!(m.maximize());
        }
        assert!(create_metric("pre@0", 0, &defaults).is_err());
        let flat = ObjectiveParams {
            huber_slope: 0.0,
            ..defaults
        };
        assert!(create_metric("mphe", 0, &flat).is_err());
    }
}
