//! Multi-target labels for elementwise objectives.
//!
//! XGBoost's elementwise objectives (`RegLossObj` for `reg:squarederror`,
//! `reg:logistic` and `binary:logistic`, and `PseudoHuberRegression`) take a
//! label matrix `[n_rows, K]` and model one output per label column
//! (`Targets(info) = labels.Shape(1)`). Every cell's gradient is the
//! single-target formula on `(margin(i, j), label(i, j))`, scaled by the
//! row's weight (`weight[idx / n_targets]`), and the intercept is estimated
//! independently per column (`FitInterceptGlmLike`'s per-column weighted mean
//! or `FitIntercept`'s per-target Newton stump). [`MultiTarget`] reproduces
//! that on top of the single-target objective: the loss runs over all
//! `n_rows * K` cells as one flat batch with the row weights broadcast per
//! cell, and intercepts come from the objective's own estimator on each label
//! column.

use super::{GradPair, Objective};
use crate::data::{GroupInfo, MetaInfo};
use crate::error::Result;

/// An elementwise single-target objective applied to each of `n_targets`
/// label columns (output `j` fits `labels[row * n_targets + j]`).
pub(crate) struct MultiTarget {
    inner: Box<dyn Objective>,
    n_targets: usize,
}

impl MultiTarget {
    /// Wrap the elementwise objective `inner` for `n_targets` label columns.
    pub(crate) fn new(inner: Box<dyn Objective>, n_targets: usize) -> Self {
        debug_assert_eq!(inner.n_outputs(), 1);
        MultiTarget { inner, n_targets }
    }

    /// The same metadata viewed as `n_rows * n_targets` single-target rows,
    /// one per cell, carrying `cell_weights` (the row weights broadcast per
    /// cell).
    fn cells<'a>(info: &MetaInfo<'a>, cell_weights: Option<&'a [f32]>) -> MetaInfo<'a> {
        MetaInfo::new(info.labels, cell_weights, None)
    }
}

impl Objective for MultiTarget {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn n_outputs(&self) -> usize {
        self.n_targets
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        let info = MetaInfo {
            n_rows: labels.len() / self.n_targets,
            n_targets: self.n_targets,
            ..MetaInfo::new(labels, weights, None)
        };
        self.gradient_info(preds, &info, out);
    }

    fn gradient_info(&self, preds: &[f32], info: &MetaInfo, out: &mut [GradPair]) {
        let cell_weights = info.cell_weights();
        self.inner
            .gradient_info(preds, &Self::cells(info, cell_weights.as_deref()), out);
    }

    fn const_hess(&self) -> bool {
        self.inner.const_hess()
    }

    fn pred_transform(&self, preds: &mut [f32]) {
        self.inner.pred_transform(preds);
    }

    fn base_margins(
        &self,
        labels: &[f32],
        weights: Option<&[f32]>,
        group: Option<&GroupInfo>,
    ) -> Vec<f32> {
        let info = MetaInfo {
            n_rows: labels.len() / self.n_targets,
            n_targets: self.n_targets,
            ..MetaInfo::new(labels, weights, group)
        };
        self.base_margins_info(&info)
    }

    /// Per-column intercepts: the wrapped objective's estimator on label
    /// column `j` with the row weights.
    fn base_margins_info(&self, info: &MetaInfo) -> Vec<f32> {
        let mut column = Vec::with_capacity(info.n_rows);
        (0..self.n_targets)
            .map(|j| {
                column.clear();
                column.extend(info.labels.iter().skip(j).step_by(self.n_targets));
                self.inner
                    .base_margins_info(&MetaInfo::new(&column, info.weights, None))[0]
            })
            .collect()
    }

    fn eval_transform(&self, preds: &mut [f32]) {
        self.inner.eval_transform(preds);
    }

    fn prob_to_margin(&self, base_score: f32) -> f32 {
        self.inner.prob_to_margin(base_score)
    }

    fn probs_to_margins(&self, scores: &mut [f32]) {
        self.inner.probs_to_margins(scores);
    }

    /// The dataset must carry this objective's label width: every cell of
    /// the flattened metadata is paired with one output margin.
    fn validate_info(&self, info: &MetaInfo) -> Result<()> {
        super::check_label_width(info, self.n_targets)?;
        let cell_weights = info.cell_weights();
        self.inner
            .validate_info(&Self::cells(info, cell_weights.as_deref()))
    }

    fn requires_labels(&self) -> bool {
        self.inner.requires_labels()
    }

    fn default_metric(&self) -> String {
        self.inner.default_metric()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainingParams;
    use crate::error::HessboostError;
    use crate::objective::create_objective;

    fn objective(name: &str, n_targets: usize) -> Box<dyn Objective> {
        let params = TrainingParams::builder()
            .objective(name)
            .scale_pos_weight(if name == "binary:logistic" { 2.0 } else { 1.0 })
            .build()
            .unwrap();
        create_objective(&params, n_targets).unwrap()
    }

    /// Two label columns as a `[row][target]` matrix plus each column alone.
    fn columns() -> (Vec<f32>, [Vec<f32>; 2], Vec<f32>) {
        let a: Vec<f32> = (0..7).map(|i| (i % 2) as f32).collect();
        let b: Vec<f32> = (0..7).map(|i| f32::from(u8::from(i % 3 == 0))).collect();
        let matrix = a.iter().zip(&b).flat_map(|(&x, &y)| [x, y]).collect();
        let weights = (0..7).map(|i| 0.5 + i as f32 * 0.25).collect();
        (matrix, [a, b], weights)
    }

    /// Output `j` of the multi-target objective is exactly the single-target
    /// objective on label column `j` with the row weights: gradients bit for
    /// bit, and the intercept of each column (weighted mean, or the Newton
    /// step `binary:logistic` takes under `scale_pos_weight`).
    #[test]
    fn each_output_is_the_single_target_objective_on_its_column() {
        let (matrix, cols, weights) = columns();
        for name in [
            "reg:squarederror",
            "reg:pseudohubererror",
            "reg:logistic",
            "binary:logistic",
        ] {
            let multi = objective(name, 2);
            let single = objective(name, 1);
            assert_eq!(multi.n_outputs(), 2);
            assert_eq!(multi.name(), single.name());
            let preds: Vec<f32> = (0..14).map(|i| i as f32 * 0.3 - 2.0).collect();
            for w in [None, Some(weights.as_slice())] {
                let info = MetaInfo {
                    n_rows: 7,
                    n_targets: 2,
                    ..MetaInfo::new(&matrix, w, None)
                };
                let mut out = vec![GradPair::default(); 14];
                multi.gradient_info(&preds, &info, &mut out);
                let margins = multi.base_margins_info(&info);
                for (j, col) in cols.iter().enumerate() {
                    let col_preds: Vec<f32> = preds.iter().skip(j).step_by(2).copied().collect();
                    let mut expected = vec![GradPair::default(); 7];
                    single.gradient(&col_preds, col, w, &mut expected);
                    for (row, e) in expected.iter().enumerate() {
                        let got = out[row * 2 + j];
                        assert_eq!(got.grad.to_bits(), e.grad.to_bits(), "{name} ({row},{j})");
                        assert_eq!(got.hess.to_bits(), e.hess.to_bits(), "{name} ({row},{j})");
                    }
                    let intercept = single.base_margins(col, w, None);
                    assert_eq!(margins[j].to_bits(), intercept[0].to_bits(), "{name} {j}");
                }
            }
        }
    }

    /// Label-domain checks cover every cell: one out-of-range probability in
    /// the second column rejects the dataset.
    #[test]
    fn label_domain_checks_every_cell() {
        let multi = objective("binary:logistic", 2);
        let labels = [0.0, 1.0, 1.0, 1.5];
        let info = MetaInfo {
            n_rows: 2,
            n_targets: 2,
            ..MetaInfo::new(&labels, None, None)
        };
        assert!(multi.validate_info(&info).is_err());
        let info = MetaInfo {
            labels: &[0.0, 1.0, 1.0, 0.5],
            ..info
        };
        assert!(multi.validate_info(&info).is_ok());
    }

    /// A dataset whose label width differs from the wrapped target count is
    /// refused before any gradient pairs predictions with labels.
    #[test]
    fn rejects_a_different_label_width() {
        let multi = objective("reg:squarederror", 2);
        let labels = [0.0, 1.0];
        let single = MetaInfo::new(&labels, None, None);
        assert!(matches!(
            multi.validate_info(&single),
            Err(HessboostError::InvalidParameter { name, .. }) if name == "labels"
        ));
    }
}
