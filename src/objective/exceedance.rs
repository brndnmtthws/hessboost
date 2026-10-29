//! Distribution regression over a threshold ladder (`reg:exceedance`), a
//! hessboost extension: one model whose output `j` predicts `P(y ≥ k_j)`
//! for the strictly increasing thresholds `k_1 < … < k_K` of
//! [`Thresholds`].
//!
//! **Outputs and labels.** The model has `K` outputs against one scalar
//! label column. Output `j` is trained exactly as `binary:logistic` on the
//! indicator `t_ij = 1` when `f64::from(y_i) ≥ k_j`, else `0`: the same
//! sigmoid, Hessian floor, and sample weights, so its gradient pairs are
//! bit for bit those of `binary:logistic` fitted to that indicator column.
//! The indicators are formed from the labels one threshold at a time; no
//! `K`-column label matrix is built. Labels must be finite and one column
//! wide.
//!
//! **Intercepts.** Output `j` starts from `binary:logistic`'s own estimate
//! for its indicator column: the logit of the (weighted) share of rows
//! with `y ≥ k_j`. A threshold no label reaches (or every label reaches)
//! gives a share of `0` (or `1`), which `binary:logistic` clamps to
//! `1e-6` (or `1 − 1e-6`) before the logit: a finite intercept of about
//! `∓13.8`, not an infinite one.
//!
//! **Rearrangement.** Predictions take the same scalar sigmoid per output,
//! then sort each row descending (a stable insertion sort, the monotone
//! rearrangement), so `P(y ≥ k_j)` never increases with `j` even where the
//! per-threshold fits cross. Prediction is that transform of
//! [`predict_margin`](crate::model::BoostedModel::predict_margin), bit for
//! bit. The default metric is `exceedance-logloss`.
//!
//! **When to prefer it.** Over `K` separate `binary:logistic` classifiers
//! on the same indicator labels, one model shares every row pass, saves and
//! loads as one file, and reports non-crossing probabilities; with
//! `multi_strategy = multi_output_tree` its trees also share splits across
//! thresholds. Separate classifiers remain the choice when each threshold
//! needs its own parameters or feature set.

use super::{GradPair, LogisticLoss, Loss, Thresholds, check_label_domain, check_label_width};
use crate::data::MetaInfo;
use crate::error::Result;
use crate::metric::EvalMetric;

/// The `reg:exceedance` loss: `binary:logistic` per threshold on the
/// indicator `y ≥ k_j` (see the [module docs](self)).
#[derive(Debug, Clone)]
pub(crate) struct Exceedance {
    thresholds: Thresholds,
    /// `binary:logistic` at `scale_pos_weight = 1`, every output's loss.
    logistic: LogisticLoss,
}

impl Exceedance {
    /// The loss at the thresholds `thresholds`.
    pub(crate) fn new(thresholds: Thresholds) -> Self {
        Exceedance {
            thresholds,
            logistic: LogisticLoss::default(),
        }
    }
}

/// The indicator label of `y` at threshold `k`: `1` when `y ≥ k`, compared
/// in `f64`.
#[inline]
fn exceeds(y: f32, k: f64) -> f32 {
    if f64::from(y) >= k { 1.0 } else { 0.0 }
}

/// Write the indicators of `labels` at threshold `k` into `out`.
fn fill_indicators(out: &mut [f32], labels: &[f32], k: f64) {
    for (t, &y) in out.iter_mut().zip(labels) {
        *t = exceeds(y, k);
    }
}

impl Loss for Exceedance {
    fn name(&self) -> &'static str {
        "reg:exceedance"
    }

    fn n_outputs(&self) -> usize {
        self.thresholds.values().len()
    }

    /// Per threshold, `binary:logistic`'s gradient on the contiguous column
    /// of that output's margins against its indicator labels (the same
    /// call, so the same vector and scalar paths as a separate model on
    /// that column), scattered back into the `[row][output]` layout.
    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        let k = self.n_outputs();
        let n = labels.len();
        super::check_gradient_inputs(n, k, preds, labels, weights, out);
        let mut margins = vec![0.0f32; n];
        let mut indicators = vec![0.0f32; n];
        let mut pairs = vec![GradPair::default(); n];
        for (j, &threshold) in self.thresholds.values().iter().enumerate() {
            for (m, row) in margins.iter_mut().zip(preds.chunks_exact(k)) {
                *m = row[j];
            }
            fill_indicators(&mut indicators, labels, threshold);
            self.logistic
                .gradient(&margins, &indicators, weights, &mut pairs);
            for (row, &pair) in out.chunks_exact_mut(k).zip(&pairs) {
                row[j] = pair;
            }
        }
    }

    /// `binary:logistic`'s sigmoid of every margin, then a stable
    /// descending insertion sort of each row (the monotone rearrangement,
    /// the mirror of `reg:quantileerror`'s ascending sort).
    fn pred_transform(&self, preds: &mut [f32]) {
        for row in preds.chunks_exact_mut(self.n_outputs()) {
            self.logistic.pred_transform(row);
            for i in 1..row.len() {
                let value = row[i];
                let mut pos = i;
                while pos > 0 && row[pos - 1] < value {
                    row[pos] = row[pos - 1];
                    pos -= 1;
                }
                row[pos] = value;
            }
        }
    }

    /// The sigmoid of every output, without the rearrangement: the
    /// inverse of [`Loss::probs_to_margins`] per output.
    fn margins_to_probs(&self, margins: &mut [f32]) {
        self.logistic.pred_transform(margins);
    }

    /// `binary:logistic`'s link (its clamped logit) of every output.
    fn probs_to_margins(&self, scores: &mut [f32]) {
        self.logistic.probs_to_margins(scores);
    }

    /// A probability in `(0, 1)`, as for `binary:logistic`.
    fn validate_base_score(&self, base_score: f64) -> Result<()> {
        self.logistic.validate_base_score(base_score)
    }

    /// One finite label column: every output is fitted to it.
    fn validate_info(&self, info: &MetaInfo) -> Result<()> {
        check_label_width(info, 1)?;
        check_label_domain(info, |y| !y.is_finite())
    }

    /// Per threshold, `binary:logistic`'s intercept for that indicator
    /// column, weights included (see the [module docs](self) for a
    /// constant column).
    fn base_margins_info(&self, info: &MetaInfo) -> Vec<f32> {
        let labels = info.label_values();
        let mut indicators = vec![0.0f32; labels.len()];
        self.thresholds
            .values()
            .iter()
            .map(|&threshold| {
                fill_indicators(&mut indicators, labels, threshold);
                let column = MetaInfo::new(&indicators, info.weights, None);
                self.logistic
                    .base_margins_info(&column)
                    .first()
                    .copied()
                    .unwrap_or(f32::NAN)
            })
            .collect()
    }

    fn default_metric(&self) -> EvalMetric {
        EvalMetric::ExceedanceLogLoss(self.thresholds.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Labels;
    use crate::error::HessboostError;
    use crate::objective::spec::LabelMatrix;
    use crate::objective::{LossContext, Objective, base_margins, gradient_pairs};
    use std::num::NonZeroUsize;

    fn thresholds(values: &[f64]) -> Thresholds {
        Thresholds::new(values.iter().copied()).unwrap()
    }

    /// A right-skewed label: mostly small, a long upper tail.
    fn skewed(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let u = ((i * 7919) % 1009) as f32 / 1009.0;
                (u * u * u) * 20.0
            })
            .collect()
    }

    /// Every output's pair is `binary:logistic`'s on that output's margin
    /// column against the indicator labels, bit for bit, with and without
    /// weights, over a length that is not a multiple of any vector block.
    #[test]
    fn gradient_pairs_equal_binary_logistic_on_indicators() {
        let levels = [0.5, 2.0, 6.0, 15.0];
        let loss = Exceedance::new(thresholds(&levels));
        let k = levels.len();
        let n = 1_003;
        let labels = skewed(n);
        let preds: Vec<f32> = (0..n * k)
            .map(|i| ((i * 7919) % 2003) as f32 / 97.0 - 10.0)
            .collect();
        let weights: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.25).collect();
        let logistic = LogisticLoss::default();
        for weights in [None, Some(weights.as_slice())] {
            let got = gradient_pairs(&loss, &preds, &labels, weights);
            for (j, &level) in levels.iter().enumerate() {
                let column: Vec<f32> = preds.iter().skip(j).step_by(k).copied().collect();
                let indicators: Vec<f32> = labels
                    .iter()
                    .map(|&y| if f64::from(y) >= level { 1.0 } else { 0.0 })
                    .collect();
                let want = gradient_pairs(&logistic, &column, &indicators, weights);
                for (i, w) in want.iter().enumerate() {
                    let g = got[i * k + j];
                    assert_eq!(g.grad.to_bits(), w.grad.to_bits(), "row {i} output {j}");
                    assert_eq!(g.hess.to_bits(), w.hess.to_bits(), "row {i} output {j}");
                }
            }
        }
    }

    /// The transform is the scalar sigmoid then a descending sort per row,
    /// so every row is non-increasing.
    #[test]
    fn transform_is_sigmoid_then_descending_sort() {
        let k = 5;
        let loss = Exceedance::new(thresholds(&[0.0, 1.0, 2.0, 3.0, 4.0]));
        let margins: Vec<f32> = (0..40 * k)
            .map(|i| ((i * 4111) % 997) as f32 / 31.0 - 16.0)
            .collect();
        let mut got = margins.clone();
        loss.pred_transform(&mut got);
        for (row, margin_row) in got.chunks_exact(k).zip(margins.chunks_exact(k)) {
            let mut want: Vec<f32> = margin_row
                .iter()
                .map(|&m| crate::simd::sigmoid_scalar(m))
                .collect();
            want.sort_by(|a, b| b.total_cmp(a));
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(row), bits(&want));
            assert!(row.windows(2).all(|pair| pair[0] >= pair[1]), "{row:?}");
        }
        // `margins_to_probs` is the sigmoid alone, inverted by the link.
        let mut probs = [-2.0f32, 3.0];
        loss.margins_to_probs(&mut probs);
        assert!(probs[0] < probs[1]);
        assert_eq!(probs[0], crate::simd::sigmoid_scalar(-2.0));
    }

    /// Each intercept is `binary:logistic`'s on its indicator column,
    /// weights included; constant columns give its clamped, finite logit.
    #[test]
    fn intercepts_are_binary_logistic_per_indicator() {
        let levels = [-1.0, 0.5, 6.0, 100.0];
        let loss = Exceedance::new(thresholds(&levels));
        let labels = skewed(301);
        let weights: Vec<f32> = (0..labels.len()).map(|i| 1.0 + (i % 3) as f32).collect();
        let logistic = LogisticLoss::default();
        for weights in [None, Some(weights.as_slice())] {
            let got = base_margins(&loss, &labels, weights);
            assert_eq!(got.len(), levels.len());
            for (j, &level) in levels.iter().enumerate() {
                let indicators: Vec<f32> = labels.iter().map(|&y| exceeds(y, level)).collect();
                let want = base_margins(&logistic, &indicators, weights);
                assert_eq!(got[j].to_bits(), want[0].to_bits(), "threshold {level}");
                assert!(got[j].is_finite());
            }
            // Every label reaches -1, none reaches 100.
            assert!(got[0] > 13.0 && got[3] < -13.0, "{got:?}");
        }
    }

    /// Empty, unsorted, duplicate, and non-finite thresholds are refused by
    /// name; so are a label matrix and non-finite labels.
    #[test]
    fn refusals() {
        for values in [
            vec![],
            vec![2.0, 1.0],
            vec![1.0, 1.0],
            vec![f64::NAN],
            vec![0.0, f64::INFINITY],
        ] {
            assert!(
                matches!(
                    Thresholds::new(values.clone()),
                    Err(HessboostError::InvalidParameter {
                        name: "exceedance_threshold",
                        ..
                    })
                ),
                "{values:?}"
            );
        }
        let objective = Objective::Exceedance(thresholds(&[1.0, 2.0]));
        assert_eq!(objective.label_matrix(), LabelMatrix::Refused);
        let context = |n_targets| LossContext {
            n_targets,
            max_delta_step: 0.0,
            shared_tree_seed: None,
            seed: 0,
        };
        assert_eq!(objective.build_loss(&context(1)).unwrap().n_outputs(), 2);
        assert!(matches!(
            objective.build_loss(&context(2)),
            Err(HessboostError::InvalidData {
                input: "labels",
                ..
            })
        ));
        let loss = Exceedance::new(thresholds(&[1.0, 2.0]));
        let matrix = [0.0f32, 1.0, 2.0, 3.0];
        let wide = MetaInfo {
            n_rows: 2,
            labels: Some(Labels::new(&matrix, NonZeroUsize::new(2).unwrap())),
            weights: None,
            ..MetaInfo::unlabeled(2)
        };
        assert!(matches!(
            loss.validate_info(&wide),
            Err(HessboostError::InvalidData { .. })
        ));
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let labels = [0.0f32, bad];
            assert!(matches!(
                loss.validate_info(&MetaInfo::new(&labels, None, None)),
                Err(HessboostError::InvalidData { .. })
            ));
        }
        assert!(
            loss.validate_info(&MetaInfo::new(&[0.0, 3.0], None, None))
                .is_ok()
        );
    }
}
