//! Scalar formulas shared by dispatch fallbacks and NEON tails.

use super::{BINARY_LOG_LOSS_EPSILON, LOG_LOSS_EPSILON, MIN_POSITIVE_PREDICTION, sigmoid_scalar};
use crate::objective::GradPair;

pub(super) fn logistic_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    scale_pos_weight: f32,
    min_hess: f32,
    out: &mut [GradPair],
    range: std::ops::Range<usize>,
) {
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

/// `(Σ wᵢ·term(i), Σ wᵢ)` over `range`, with unit weights when `weights` is
/// `None`: the accumulation shared by the weighted metric sums.
#[inline]
fn weighted_sum(
    weights: Option<&[f32]>,
    range: std::ops::Range<usize>,
    term: impl Fn(usize) -> f64,
) -> (f64, f64) {
    let mut sum = 0.0;
    let mut weight_sum = 0.0;
    for index in range {
        let weight = weights.map_or(1.0, |values| f64::from(values[index]));
        sum += weight * term(index);
        weight_sum += weight;
    }
    (sum, weight_sum)
}

pub(super) fn distance_sum<const SQUARED: bool>(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    range: std::ops::Range<usize>,
) -> (f64, f64) {
    weighted_sum(weights, range, |index| {
        let difference = f64::from(preds[index]) - f64::from(labels[index]);
        if SQUARED {
            difference * difference
        } else {
            difference.abs()
        }
    })
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

/// XGBoost's `logloss` term `x·ln(max(y, ε))`, exactly `0` when `x == 0`;
/// `max` keeps a NaN `y` like `std::max(y, eps)`. Out-of-range predictions
/// (e.g. `binary:logitraw` margins) are not clamped to `[0, 1]`.
fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        let floored = if y < BINARY_LOG_LOSS_EPSILON {
            BINARY_LOG_LOSS_EPSILON
        } else {
            y
        };
        x * floored.ln()
    }
}

pub(super) fn log_loss(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    range: std::ops::Range<usize>,
) -> (f64, f64) {
    weighted_sum(weights, range, |index| {
        let probability = f64::from(preds[index]);
        let label = f64::from(labels[index]);
        xlogy(-label, probability) + xlogy(-(1.0 - label), 1.0 - probability)
    })
}

pub(super) fn positive_nloglik<const GAMMA: bool>(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    range: std::ops::Range<usize>,
) -> (f64, f64) {
    weighted_sum(weights, range, |index| {
        let prediction = f64::from(preds[index]).max(MIN_POSITIVE_PREDICTION);
        let label = f64::from(labels[index]);
        if GAMMA {
            label / prediction + prediction.ln()
        } else {
            prediction - label * prediction.ln()
        }
    })
}

pub(super) fn tweedie_nloglik(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    rho: f64,
    range: std::ops::Range<usize>,
) -> (f64, f64) {
    weighted_sum(weights, range, |index| {
        let prediction = f64::from(preds[index]).max(MIN_POSITIVE_PREDICTION);
        let label = f64::from(labels[index]);
        let first = label * prediction.powf(1.0 - rho) / (1.0 - rho);
        let second = prediction.powf(2.0 - rho) / (2.0 - rho);
        -first + second
    })
}

/// Multiclass log loss over the label `rows` of a row-major `num_class`
/// prediction matrix.
pub(super) fn multiclass_log_loss(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
    rows: std::ops::Range<usize>,
) -> (f64, f64) {
    weighted_sum(weights, rows, |row| {
        -f64::from(preds[row * num_class + labels[row] as usize])
            .clamp(LOG_LOSS_EPSILON, 1.0)
            .ln()
    })
}
