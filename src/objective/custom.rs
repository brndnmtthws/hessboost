//! User-defined objective hook.
//!
//! Wraps caller-supplied closures so any custom loss can drive training via
//! [`crate::training::Trainer`]. The gradient closure receives raw
//! margins and writes gradient/Hessian pairs, matching the built-in objectives.

use super::{GradPair, Objective, SplitGradient};

type GradFn = dyn Fn(&[f32], &[f32], Option<&[f32]>, &mut [GradPair]) + Send + Sync;
type TransformFn = dyn Fn(&mut [f32]) + Send + Sync;
type SplitGradFn = dyn Fn(usize, &[GradPair]) -> Option<SplitGradient> + Send + Sync;

/// An [`Objective`] backed by user closures.
pub struct CustomObjective {
    name: String,
    n_outputs: usize,
    base: f32,
    default_metric: String,
    grad_fn: Box<GradFn>,
    transform_fn: Option<Box<TransformFn>>,
    split_grad_fn: Option<Box<SplitGradFn>>,
}

impl CustomObjective {
    /// Build a custom objective.
    ///
    /// * `grad_fn`: `(margins, labels, weights, out)` fills `out` with gradients
    ///   (margins and `out` are `[row][output]`; labels hold one value per row,
    ///   or `[row][target]` for a label matrix, which then needs one column
    ///   per output).
    /// * `base`: the initial margin (base score), used for every output.
    /// * `transform_fn`: optional prediction transform (identity if `None`).
    pub fn new(
        name: impl Into<String>,
        n_outputs: usize,
        base: f32,
        default_metric: impl Into<String>,
        grad_fn: impl Fn(&[f32], &[f32], Option<&[f32]>, &mut [GradPair]) + Send + Sync + 'static,
    ) -> Self {
        CustomObjective {
            name: name.into(),
            n_outputs,
            base,
            default_metric: default_metric.into(),
            grad_fn: Box::new(grad_fn),
            transform_fn: None,
            split_grad_fn: None,
        }
    }

    /// Attach a prediction transform.
    #[must_use]
    pub fn with_transform(
        mut self,
        transform: impl Fn(&mut [f32]) + Send + Sync + 'static,
    ) -> Self {
        self.transform_fn = Some(Box::new(transform));
        self
    }

    /// Attach a reduced split-gradient hook for vector-leaf trees
    /// (`multi_strategy = multi_output_tree`): `split_grad(iteration, gpair)`
    /// receives the round's full `[row][output]` gradients and returns the
    /// narrower `[row][target]` gradients the tree structure is grown from,
    /// or `None` to use the full gradients that round. See
    /// [`Objective::split_gradient`].
    #[must_use]
    pub fn with_split_gradient(
        mut self,
        split_grad: impl Fn(usize, &[GradPair]) -> Option<SplitGradient> + Send + Sync + 'static,
    ) -> Self {
        self.split_grad_fn = Some(Box::new(split_grad));
        self
    }
}

impl Objective for CustomObjective {
    fn name(&self) -> &str {
        &self.name
    }

    fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        // Labels hold one value per row, or one per row and output for a
        // label matrix (`DMatrix::with_label_matrix`, row-major).
        let n_rows = preds.len() / self.n_outputs.max(1);
        debug_assert!(labels.len() == n_rows || labels.len() == preds.len());
        debug_assert_eq!(out.len(), preds.len());
        debug_assert!(weights.is_none_or(|w| w.len() == n_rows));
        (self.grad_fn)(preds, labels, weights, out);
    }

    /// Labels hold one value per row, or one per row and output: a label
    /// matrix of another width is refused before the gradient closure sees
    /// it.
    fn validate_info(&self, info: &crate::data::MetaInfo) -> crate::error::Result<()> {
        if info.n_targets != 1 && info.n_targets != self.n_outputs {
            return Err(crate::error::HessboostError::invalid_param(
                "objective",
                format!(
                    "custom objective `{}` has {} outputs; dataset has a {}-column label matrix \
                     (one label per row or per output expected)",
                    self.name, self.n_outputs, info.n_targets
                ),
            ));
        }
        Ok(())
    }

    fn pred_transform(&self, preds: &mut [f32]) {
        if let Some(t) = &self.transform_fn {
            t(preds);
        }
    }

    fn base_margins_info(&self, _info: &crate::data::MetaInfo) -> Vec<f32> {
        vec![self.base; self.n_outputs]
    }

    fn default_metric(&self) -> String {
        self.default_metric.clone()
    }

    fn split_gradient(&self, iteration: usize, gpair: &[GradPair]) -> Option<SplitGradient> {
        self.split_grad_fn
            .as_ref()
            .and_then(|f| f(iteration, gpair))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_squared_error_behaves() {
        // Reimplement squared error as a custom objective.
        let obj = CustomObjective::new("custom:sqerr", 1, 0.0, "rmse", |preds, labels, w, out| {
            for i in 0..preds.len() {
                let wi = w.map_or(1.0, |ws| ws[i]);
                out[i] = GradPair::new((preds[i] - labels[i]) * wi, wi);
            }
        });
        let mut out = vec![GradPair::default(); 2];
        obj.gradient(&[2.0, 0.0], &[1.0, 0.5], None, &mut out);
        assert_eq!(out[0], GradPair::new(1.0, 1.0));
        assert_eq!(out[1], GradPair::new(-0.5, 1.0));
    }
}
