//! Multiclass classification objectives (`multi:softmax`, `multi:softprob`).
//!
//! These are the first *multi-output* objectives: with `K` classes each instance
//! carries `K` raw margins, laid out `[instance][class]` (row-major). Each round
//! the trainer grows one tree per class from that class's gradient slice.

use super::{GradPair, MIN_HESS, Objective, check_label_domain};
use crate::data::MetaInfo;
use crate::error::Result;

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
        super::rowwise_gradient(n, k, preds, labels, weights, out, |p, l, w, o| {
            crate::simd::softmax_gradient(p, l, w, k, MIN_HESS, o);
        });
    }

    fn pred_transform(&self, preds: &mut [f32]) {
        // Convert every instance's margins to a probability distribution.
        crate::simd::softmax_rows_inplace(preds, self.num_class);
    }

    fn base_margins(
        &self,
        labels: &[f32],
        weights: Option<&[f32]>,
        _group: Option<&crate::data::GroupInfo>,
    ) -> Vec<f32> {
        // XGBoost `SoftmaxMultiClassObj::InitEstimation`, step for step in its
        // precision: class weight totals accumulated in f32 (`SmallHistogram`),
        // divided by the f64 weight sum (`VecScaDiv` multiplies by `1/Σw`),
        // `ln(p + 1e-6)` (`LogE` with `kRtEps`), then centered by the f32 mean.
        let k = self.num_class;
        let mut margins = vec![0.0f32; k];
        for (i, &y) in labels.iter().enumerate() {
            if let Some(slot) = margins.get_mut(y as usize) {
                *slot += weights.map_or(1.0, |ws| ws[i]);
            }
        }
        let sum_w = match weights {
            Some(ws) => ws.iter().map(|&w| f64::from(w)).sum::<f64>(),
            None => f64::from(labels.len() as f32),
        };
        let inv_sum_w = 1.0 / sum_w;
        for m in &mut margins {
            *m = ((f64::from(*m) * inv_sum_w) as f32 + crate::K_RT_EPS_F32).ln();
        }
        let n = k as f32;
        let mean = margins.iter().map(|m| m / n).sum::<f32>();
        for m in &mut margins {
            *m -= mean;
        }
        margins
    }

    fn validate_info(&self, info: &MetaInfo) -> Result<()> {
        // Class indices: non-negative integers below `num_class`.
        let k = self.num_class as f32;
        check_label_domain(info, |y| y.fract() != 0.0 || y < 0.0 || y >= k)
    }

    fn default_metric(&self) -> String {
        "mlogloss".to_string()
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

    /// Class intercepts are centered log frequencies: `ln(p_c + 1e-6)` minus
    /// their mean, so they sum to ~0 and differ by the log-odds between
    /// classes. Weights shift the frequencies; an unweighted uniform split
    /// gives all-zero margins.
    #[test]
    fn base_margins_are_centered_log_frequencies() {
        let obj = SoftmaxObjective::new(3, true);
        let labels = [0.0f32, 0.0, 1.0, 2.0];
        let m = obj.base_margins(&labels, None, None);
        assert_eq!(m.len(), 3);
        assert!(m.iter().sum::<f32>().abs() < 1e-6);
        let expected_gap = (0.5f32 + 1e-6).ln() - (0.25f32 + 1e-6).ln();
        assert!((m[0] - m[1] - expected_gap).abs() < 1e-6, "{m:?}");
        assert_eq!(m[1], m[2]);
        // Weight 2 on the class-1 row makes every class equally frequent.
        let w = [1.0f32, 1.0, 2.0, 2.0];
        let uniform = obj.base_margins(&labels, Some(&w), None);
        assert!(uniform.iter().all(|v| v.abs() < 1e-6), "{uniform:?}");
    }
}
