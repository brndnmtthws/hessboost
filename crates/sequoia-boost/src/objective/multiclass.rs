//! Multiclass classification objectives (`multi:softmax`, `multi:softprob`).
//!
//! These are the first *multi-output* objectives: with `K` classes each instance
//! carries `K` raw margins, laid out `[instance][class]` (row-major). Each round
//! the trainer grows one tree per class from that class's gradient slice.

use super::{GradPair, Objective, MIN_HESS};

/// Multiclass softmax objective. `output_prob` distinguishes `multi:softprob`
/// (report per-class probabilities) from `multi:softmax` (report the argmax
/// class), but both share identical gradients.
#[derive(Debug, Clone, Copy)]
pub struct SoftmaxObjective {
    num_class: usize,
    output_prob: bool,
}

impl SoftmaxObjective {
    /// Create a softmax objective over `num_class` classes.
    pub fn new(num_class: usize, output_prob: bool) -> Self {
        SoftmaxObjective {
            num_class,
            output_prob,
        }
    }
}

impl Objective for SoftmaxObjective {
    fn name(&self) -> &str {
        if self.output_prob {
            "multi:softprob"
        } else {
            "multi:softmax"
        }
    }

    fn n_outputs(&self) -> usize {
        self.num_class
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        let k = self.num_class;
        let n = labels.len();
        super::check_gradient_inputs(n, k, preds, labels, weights, out);

        super::rowwise_gradient(n, k, preds, labels, weights, out, |p, l, w, o| {
            crate::simd::softmax_gradient(p, l, w, k, MIN_HESS, o)
        });
    }

    fn pred_transform(&self, preds: &mut [f32]) {
        // Convert every instance's margins to a probability distribution.
        let k = self.num_class;
        crate::simd::softmax_rows_inplace(preds, k);
    }

    fn base_margin(&self, _labels: &[f32], _weights: Option<&[f32]>) -> f32 {
        // Multiclass initializes every class margin to zero.
        0.0
    }

    fn default_metric(&self) -> &str {
        "mlogloss"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn softmax_normalizes() {
        let mut r = [1.0f32, 2.0, 3.0];
        let num_class = r.len();
        crate::simd::softmax_rows_inplace(&mut r, num_class);
        assert_relative_eq!(r.iter().sum::<f32>(), 1.0, epsilon = 1e-6);
        assert!(r[2] > r[1] && r[1] > r[0]);
    }

    #[test]
    fn gradient_layout_and_values() {
        // 2 instances, 3 classes, all margins 0 -> uniform p = 1/3.
        let obj = SoftmaxObjective::new(3, true);
        let preds = [0.0f32; 6];
        let labels = [0.0f32, 2.0];
        let mut out = vec![GradPair::default(); 6];
        obj.gradient(&preds, &labels, None, &mut out);
        // instance 0, correct class 0: grad = 1/3 - 1
        assert_relative_eq!(out[0].grad, 1.0 / 3.0 - 1.0, epsilon = 1e-6);
        // instance 0, class 1: grad = 1/3
        assert_relative_eq!(out[1].grad, 1.0 / 3.0, epsilon = 1e-6);
        // instance 1, correct class 2: grad = 1/3 - 1
        assert_relative_eq!(out[5].grad, 1.0 / 3.0 - 1.0, epsilon = 1e-6);
        // hess = 2 * p * (1-p) = 2 * 1/3 * 2/3
        assert_relative_eq!(out[0].hess, 2.0 * (1.0 / 3.0) * (2.0 / 3.0), epsilon = 1e-6);
    }
}
