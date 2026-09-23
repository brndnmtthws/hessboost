//! Scalar formulas shared by dispatch fallbacks and NEON tails.

use super::{LOG_LOSS_EPSILON, MIN_POSITIVE_PREDICTION, sigmoid_scalar};
use crate::objective::GradPair;

pub(super) fn logistic_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    parameters: (f32, f32),
    out: &mut [GradPair],
    range: std::ops::Range<usize>,
) {
    let (scale_pos_weight, min_hess) = parameters;
    for index in range {
        let label = labels[index];
        let probability = sigmoid_scalar(preds[index]);
        let mut weight = weights.map_or(1.0, |values| values[index]);
        if label == 1.0 {
            weight *= scale_pos_weight;
        }
        out[index] = GradPair::new(
            (probability - label) * weight,
            (probability * (1.0 - probability)).max(min_hess) * weight,
        );
    }
}

pub(super) fn poisson_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    max_delta_step: f32,
    out: &mut [GradPair],
    range: std::ops::Range<usize>,
) {
    for index in range {
        let weight = weights.map_or(1.0, |values| values[index]);
        out[index] = GradPair::new(
            (preds[index].exp() - labels[index]) * weight,
            (preds[index] + max_delta_step).exp() * weight,
        );
    }
}

pub(super) fn gamma_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    out: &mut [GradPair],
    range: std::ops::Range<usize>,
) {
    // XGBoost `GammaDeviance`: `p = expf(x)`, `g = 1 - y / p`, `h = y / p`.
    for index in range {
        let weight = weights.map_or(1.0, |values| values[index]);
        let p = preds[index].exp();
        let scaled = labels[index] / p;
        out[index] = GradPair::new((1.0 - scaled) * weight, scaled * weight);
    }
}

pub(super) fn tweedie_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    rho: f32,
    out: &mut [GradPair],
    range: std::ops::Range<usize>,
) {
    for index in range {
        let weight = weights.map_or(1.0, |values| values[index]);
        let margin = preds[index];
        let label = labels[index];
        let exp_1 = ((1.0 - rho) * margin).exp();
        let exp_2 = ((2.0 - rho) * margin).exp();
        out[index] = GradPair::new(
            (-label * exp_1 + exp_2) * weight,
            (-label * (1.0 - rho) * exp_1 + (2.0 - rho) * exp_2) * weight,
        );
    }
}

pub(super) fn distance_sum<const SQUARED: bool>(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    range: std::ops::Range<usize>,
) -> (f64, f64) {
    let mut sum = 0.0;
    let mut weight_sum = 0.0;
    for index in range {
        let weight = weights.map_or(1.0, |values| f64::from(values[index]));
        let difference = f64::from(preds[index]) - f64::from(labels[index]);
        let distance = if SQUARED {
            difference * difference
        } else {
            difference.abs()
        };
        sum += weight * distance;
        weight_sum += weight;
    }
    (sum, weight_sum)
}

pub(super) fn classification_error_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    range: std::ops::Range<usize>,
) -> (f64, f64) {
    let mut wrong = 0.0;
    let mut weight_sum = 0.0;
    for index in range {
        let weight = weights.map_or(1.0, |values| f64::from(values[index]));
        if (preds[index] > 0.5) != (labels[index] > 0.5) {
            wrong += weight;
        }
        weight_sum += weight;
    }
    (wrong, weight_sum)
}

pub(super) fn log_loss(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    range: std::ops::Range<usize>,
) -> (f64, f64) {
    let mut loss = 0.0;
    let mut weight_sum = 0.0;
    for index in range {
        let weight = weights.map_or(1.0, |values| f64::from(values[index]));
        let probability = f64::from(preds[index]).clamp(LOG_LOSS_EPSILON, 1.0 - LOG_LOSS_EPSILON);
        let label = f64::from(labels[index]);
        loss += weight * -(label * probability.ln() + (1.0 - label) * (1.0 - probability).ln());
        weight_sum += weight;
    }
    (loss, weight_sum)
}

pub(super) fn positive_nloglik<const GAMMA: bool>(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    range: std::ops::Range<usize>,
) -> (f64, f64) {
    let mut loss = 0.0;
    let mut weight_sum = 0.0;
    for index in range {
        let weight = weights.map_or(1.0, |values| f64::from(values[index]));
        let prediction = f64::from(preds[index]).max(MIN_POSITIVE_PREDICTION);
        let label = f64::from(labels[index]);
        loss += weight
            * if GAMMA {
                label / prediction + prediction.ln()
            } else {
                prediction - label * prediction.ln()
            };
        weight_sum += weight;
    }
    (loss, weight_sum)
}
