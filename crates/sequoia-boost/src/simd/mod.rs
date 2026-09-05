//! Runtime-dispatched numeric kernels for operations the compiler cannot
//! auto-vectorize, chiefly transcendental objective functions.

mod scalar;

const LOG_LOSS_EPSILON: f64 = 1e-15;
const MIN_POSITIVE_PREDICTION: f64 = 1e-8;

use crate::objective::GradPair;
use crate::tree::gain::GradStats;

#[cfg(target_arch = "aarch64")]
use crate::tree::gain::RegParams;

#[cfg(target_arch = "aarch64")]
use std::sync::OnceLock;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "aarch64")]
const MIN_SIMD_LEN: usize = 16;

#[cfg(target_arch = "aarch64")]
static NEON_AVAILABLE: OnceLock<bool> = OnceLock::new();

#[cfg(target_arch = "aarch64")]
pub(crate) struct SplitCandidate {
    pub(crate) loss_change: f64,
    pub(crate) split_offset: usize,
    pub(crate) left: GradStats,
    pub(crate) right: GradStats,
}

#[cfg(target_arch = "aarch64")]
pub(crate) enum DenseSplitScan {
    ScalarFallback,
    Scanned(Option<SplitCandidate>),
}

/// Resolve the process-wide AArch64 backend lazily on the first numeric-kernel
/// call. `OnceLock` makes feature detection a one-time initialization cost;
/// subsequent calls are a cached load and comparison.
#[cfg(target_arch = "aarch64")]
#[inline]
fn neon_available() -> bool {
    *NEON_AVAILABLE.get_or_init(|| std::arch::is_aarch64_feature_detected!("neon"))
}

#[inline]
#[cfg(target_arch = "aarch64")]
fn gradient_slices_cover(
    len: usize,
    labels: &[f32],
    weights: Option<&[f32]>,
    out: &[GradPair],
) -> bool {
    metric_slices_cover(len, labels, weights) && out.len() >= len
}

#[inline]
#[cfg(target_arch = "aarch64")]
fn metric_slices_cover(len: usize, labels: &[f32], weights: Option<&[f32]>) -> bool {
    labels.len() >= len && weights.is_none_or(|values| values.len() >= len)
}

#[inline]
fn sigmoid_scalar(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp = x.exp();
        exp / (1.0 + exp)
    }
}

#[inline]
pub(crate) fn sum_grad_stats(values: &[GradStats]) -> GradStats {
    #[cfg(target_arch = "aarch64")]
    if values.len() >= MIN_SIMD_LEN && neon_available() {
        // SAFETY: NEON is present; GradStats is repr(C) with two adjacent f64
        // fields, and the kernel bounds all loads by the slice length.
        return unsafe { aarch64::sum_grad_stats(values) };
    }

    let mut sum = GradStats::default();
    for &value in values {
        sum.add(value);
    }
    sum
}

/// Try the NEON split-gain scan used by the common dense, unconstrained
/// histogram path. `ScalarFallback` asks the caller to use its scalar scan.
#[cfg(target_arch = "aarch64")]
pub(crate) fn dense_unconstrained_best_split(
    histogram: &[GradStats],
    total: GradStats,
    reg: &RegParams,
    parent_gain: f64,
    comparison_epsilon: f64,
) -> DenseSplitScan {
    if histogram.len() >= MIN_SIMD_LEN && reg.max_delta_step == 0.0 && neon_available() {
        // SAFETY: NEON is present. The kernel only reads `histogram` and keeps
        // all vector loads within the complete candidate range.
        return DenseSplitScan::Scanned(unsafe {
            aarch64::dense_unconstrained_best_split(
                histogram,
                total,
                reg,
                parent_gain,
                comparison_epsilon,
            )
        });
    }

    DenseSplitScan::ScalarFallback
}

#[inline]
pub(crate) fn exp_inplace(values: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    if values.len() >= MIN_SIMD_LEN && neon_available() {
        // SAFETY: runtime feature detection proves NEON is available, and the
        // kernel bounds vector accesses by the slice length.
        unsafe { aarch64::exp_inplace(values) };
        return;
    }

    values.iter_mut().for_each(|value| *value = value.exp());
}

#[inline]
pub(crate) fn sigmoid_inplace(values: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    if values.len() >= MIN_SIMD_LEN && neon_available() {
        // SAFETY: runtime feature detection proves NEON is available, and the
        // kernel bounds vector accesses by the slice length.
        unsafe { aarch64::sigmoid_inplace(values) };
        return;
    }

    values
        .iter_mut()
        .for_each(|value| *value = sigmoid_scalar(*value));
}

pub(crate) fn logistic_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    scale_pos_weight: f32,
    min_hess: f32,
    out: &mut [GradPair],
) {
    #[cfg(target_arch = "aarch64")]
    if preds.len() >= MIN_SIMD_LEN
        && gradient_slices_cover(preds.len(), labels, weights, out)
        && neon_available()
    {
        // SAFETY: runtime feature detection proves NEON is available. The
        // objective validates equal slice lengths before entering this kernel.
        unsafe {
            aarch64::logistic_gradient(preds, labels, weights, scale_pos_weight, min_hess, out)
        };
        return;
    }

    logistic_gradient_scalar(preds, labels, weights, scale_pos_weight, min_hess, out);
}

fn logistic_gradient_scalar(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    scale_pos_weight: f32,
    min_hess: f32,
    out: &mut [GradPair],
) {
    scalar::logistic_gradient(
        preds,
        labels,
        weights,
        (scale_pos_weight, min_hess),
        out,
        0..preds.len(),
    );
}

pub(crate) fn poisson_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    max_delta_step: f32,
    out: &mut [GradPair],
) {
    #[cfg(target_arch = "aarch64")]
    if preds.len() >= MIN_SIMD_LEN
        && gradient_slices_cover(preds.len(), labels, weights, out)
        && neon_available()
    {
        // SAFETY: NEON is present and objective inputs share a common length.
        unsafe { aarch64::poisson_gradient(preds, labels, weights, max_delta_step, out) };
        return;
    }
    scalar::poisson_gradient(preds, labels, weights, max_delta_step, out, 0..preds.len());
}

pub(crate) fn gamma_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    out: &mut [GradPair],
) {
    #[cfg(target_arch = "aarch64")]
    if preds.len() >= MIN_SIMD_LEN
        && gradient_slices_cover(preds.len(), labels, weights, out)
        && neon_available()
    {
        // SAFETY: NEON is present and objective inputs share a common length.
        unsafe { aarch64::gamma_gradient(preds, labels, weights, out) };
        return;
    }
    scalar::gamma_gradient(preds, labels, weights, out, 0..preds.len());
}

pub(crate) fn tweedie_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    rho: f32,
    out: &mut [GradPair],
) {
    #[cfg(target_arch = "aarch64")]
    if preds.len() >= MIN_SIMD_LEN
        && gradient_slices_cover(preds.len(), labels, weights, out)
        && neon_available()
    {
        // SAFETY: NEON is present and objective inputs share a common length.
        unsafe { aarch64::tweedie_gradient(preds, labels, weights, rho, out) };
        return;
    }
    scalar::tweedie_gradient(preds, labels, weights, rho, out, 0..preds.len());
}

/// Apply softmax to every contiguous `num_class` row while resolving the SIMD
/// backend only once for the whole matrix.
pub(crate) fn softmax_rows_inplace(values: &mut [f32], num_class: usize) {
    #[cfg(target_arch = "aarch64")]
    if num_class >= 8 && values.len() >= MIN_SIMD_LEN && neon_available() {
        // SAFETY: NEON is present; the objective guarantees complete rows and
        // the kernel bounds each row by `num_class`.
        unsafe { aarch64::softmax_rows_inplace(values, num_class) };
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if (2..=4).contains(&num_class) && values.len() >= MIN_SIMD_LEN && neon_available() {
        // SAFETY: NEON is present; each specialization processes bounded
        // batches of four rows and handles the remaining values scalarly.
        unsafe {
            match num_class {
                2 => aarch64::short_softmax_rows::<2>(values),
                3 => aarch64::short_softmax_rows::<3>(values),
                _ => aarch64::short_softmax_rows::<4>(values),
            }
        }
        return;
    }

    for row in values.chunks_mut(num_class) {
        softmax_scalar(row);
    }
}

pub(crate) fn softmax_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
    min_hess: f32,
    out: &mut [GradPair],
) {
    let complete = labels
        .len()
        .checked_mul(num_class)
        .is_some_and(|len| len == preds.len() && out.len() >= len)
        && weights.is_none_or(|values| values.len() >= labels.len());
    #[cfg(target_arch = "aarch64")]
    if num_class >= 8 && preds.len() >= MIN_SIMD_LEN && complete && neon_available() {
        // SAFETY: NEON is present and all input/output slices cover the complete
        // row-major prediction matrix.
        unsafe { aarch64::softmax_gradient(preds, labels, weights, num_class, min_hess, out) };
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if (2..=4).contains(&num_class) && preds.len() >= MIN_SIMD_LEN && complete && neon_available() {
        // SAFETY: the complete-matrix check covers every prediction, label,
        // weight and output row. NEON is available and K is 2, 3, or 4.
        unsafe {
            match num_class {
                2 => aarch64::short_softmax_gradient::<2>(preds, labels, weights, min_hess, out),
                3 => aarch64::short_softmax_gradient::<3>(preds, labels, weights, min_hess, out),
                _ => aarch64::short_softmax_gradient::<4>(preds, labels, weights, min_hess, out),
            }
        }
        return;
    }

    debug_assert!(complete);
    for row in 0..labels.len() {
        let base = row * num_class;
        softmax_gradient_row_scalar(
            &preds[base..base + num_class],
            labels[row] as usize,
            weights.map_or(1.0, |values| values[row]),
            min_hess,
            &mut out[base..base + num_class],
        );
    }
}

pub(super) fn softmax_scalar(values: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &value in values.iter() {
        if value > max {
            max = value;
        }
    }
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    let inverse = 1.0 / sum;
    values.iter_mut().for_each(|value| *value *= inverse);
}

pub(super) fn softmax_gradient_row_scalar(
    preds: &[f32],
    label: usize,
    weight: f32,
    min_hess: f32,
    out: &mut [GradPair],
) {
    let mut max = f32::NEG_INFINITY;
    for &value in preds {
        if value > max {
            max = value;
        }
    }
    let mut sum = 0.0;
    for (output, &prediction) in out.iter_mut().zip(preds) {
        let exp = (prediction - max).exp();
        output.grad = exp;
        sum += exp;
    }
    let inverse = 1.0 / sum;
    for (class, output) in out.iter_mut().enumerate() {
        let probability = output.grad * inverse;
        let target = if class == label { 1.0 } else { 0.0 };
        *output = GradPair::new(
            (probability - target) * weight,
            (2.0 * probability * (1.0 - probability) * weight).max(min_hess),
        );
    }
}

pub(crate) fn squared_error_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
) -> (f64, f64) {
    distance_sum::<true>(preds, labels, weights)
}

pub(crate) fn absolute_error_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
) -> (f64, f64) {
    distance_sum::<false>(preds, labels, weights)
}

fn distance_sum<const SQUARED: bool>(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
) -> (f64, f64) {
    #[cfg(target_arch = "aarch64")]
    if preds.len() >= MIN_SIMD_LEN
        && metric_slices_cover(preds.len(), labels, weights)
        && neon_available()
    {
        // SAFETY: NEON is present and every input slice covers `preds`.
        return unsafe { aarch64::distance_sum::<SQUARED>(preds, labels, weights) };
    }

    let mut sum = 0.0;
    let mut weight_sum = 0.0;
    for i in 0..preds.len() {
        let weight = weights.map_or(1.0, |values| values[i] as f64);
        let difference = preds[i] as f64 - labels[i] as f64;
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

pub(crate) fn classification_error_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
) -> (f64, f64) {
    #[cfg(target_arch = "aarch64")]
    if preds.len() >= MIN_SIMD_LEN
        && metric_slices_cover(preds.len(), labels, weights)
        && neon_available()
    {
        // SAFETY: NEON is present and every input slice covers `preds`.
        return unsafe { aarch64::classification_error_sum(preds, labels, weights) };
    }

    let mut wrong = 0.0;
    let mut weight_sum = 0.0;
    for i in 0..preds.len() {
        let weight = weights.map_or(1.0, |values| values[i] as f64);
        if (preds[i] > 0.5) != (labels[i] > 0.5) {
            wrong += weight;
        }
        weight_sum += weight;
    }
    (wrong, weight_sum)
}

pub(crate) fn log_loss_sum(preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> (f64, f64) {
    #[cfg(target_arch = "aarch64")]
    if preds.len() >= MIN_SIMD_LEN
        && metric_slices_cover(preds.len(), labels, weights)
        && neon_available()
    {
        // SAFETY: NEON is present and every input slice covers `preds`.
        return unsafe { aarch64::log_loss_sum(preds, labels, weights) };
    }

    scalar::log_loss(preds, labels, weights, 0..preds.len())
}

pub(crate) fn positive_nloglik_sum<const GAMMA: bool>(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
) -> (f64, f64) {
    #[cfg(target_arch = "aarch64")]
    if preds.len() >= MIN_SIMD_LEN
        && metric_slices_cover(preds.len(), labels, weights)
        && neon_available()
    {
        // SAFETY: NEON is present and every input slice covers `preds`.
        return unsafe { aarch64::positive_nloglik_sum::<GAMMA>(preds, labels, weights) };
    }

    scalar::positive_nloglik::<GAMMA>(preds, labels, weights, 0..preds.len())
}

pub(crate) fn tweedie_nloglik_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    rho: f64,
) -> (f64, f64) {
    #[cfg(target_arch = "aarch64")]
    if rho.is_finite()
        && rho > 1.0
        && rho < 2.0
        && preds.len() >= MIN_SIMD_LEN
        && metric_slices_cover(preds.len(), labels, weights)
        && neon_available()
    {
        // SAFETY: NEON is present and every input slice covers `preds`.
        return unsafe { aarch64::tweedie_nloglik_sum(preds, labels, weights, rho) };
    }

    tweedie_nloglik_sum_scalar(preds, labels, weights, rho)
}

pub(super) fn tweedie_nloglik_sum_scalar(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    rho: f64,
) -> (f64, f64) {
    let mut loss = 0.0;
    let mut weight_sum = 0.0;
    for index in 0..preds.len() {
        let weight = weights.map_or(1.0, |values| values[index] as f64);
        let prediction = (preds[index] as f64).max(MIN_POSITIVE_PREDICTION);
        let label = labels[index] as f64;
        let first = label * prediction.powf(1.0 - rho) / (1.0 - rho);
        let second = prediction.powf(2.0 - rho) / (2.0 - rho);
        loss += weight * (-first + second);
        weight_sum += weight;
    }
    (loss, weight_sum)
}

pub(crate) fn multiclass_log_loss_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
) -> (f64, f64) {
    let complete = labels
        .len()
        .checked_mul(num_class)
        .is_some_and(|len| preds.len() >= len)
        && weights.is_none_or(|values| values.len() >= labels.len());
    #[cfg(target_arch = "aarch64")]
    if labels.len() >= MIN_SIMD_LEN && complete && neon_available() {
        // SAFETY: NEON is present and the complete matrix/weight checks cover
        // every selected class probability.
        return unsafe { aarch64::multiclass_log_loss_sum(preds, labels, weights, num_class) };
    }

    debug_assert!(complete);
    multiclass_log_loss_sum_scalar(preds, labels, weights, num_class)
}

pub(super) fn multiclass_log_loss_sum_scalar(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
) -> (f64, f64) {
    let mut loss = 0.0;
    let mut weight_sum = 0.0;
    for (row, &label) in labels.iter().enumerate() {
        let weight = weights.map_or(1.0, |values| values[row] as f64);
        let probability =
            (preds[row * num_class + label as usize] as f64).clamp(LOG_LOSS_EPSILON, 1.0);
        loss += -weight * probability.ln();
        weight_sum += weight;
    }
    (loss, weight_sum)
}

pub(crate) fn multiclass_error_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
) -> (f64, f64) {
    let complete = labels
        .len()
        .checked_mul(num_class)
        .is_some_and(|len| preds.len() >= len)
        && weights.is_none_or(|values| values.len() >= labels.len());
    #[cfg(target_arch = "aarch64")]
    if num_class >= 8
        && num_class <= u32::MAX as usize
        && labels.len() >= MIN_SIMD_LEN
        && complete
        && neon_available()
    {
        // SAFETY: NEON is present and each complete probability row is covered.
        return unsafe { aarch64::multiclass_error_sum(preds, labels, weights, num_class) };
    }

    debug_assert!(complete);
    multiclass_error_sum_scalar(preds, labels, weights, num_class)
}

fn multiclass_error_sum_scalar(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
) -> (f64, f64) {
    let mut wrong = 0.0;
    let mut weight_sum = 0.0;
    for (row_index, &label) in labels.iter().enumerate() {
        let weight = weights.map_or(1.0, |values| values[row_index] as f64);
        let row = &preds[row_index * num_class..(row_index + 1) * num_class];
        let best = argmax_scalar(row);
        if best != label as usize {
            wrong += weight;
        }
        weight_sum += weight;
    }
    (wrong, weight_sum)
}

pub(super) fn argmax_scalar(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

#[cfg(test)]
mod tests;
