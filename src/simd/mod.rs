//! Runtime-dispatched numeric kernels for operations the compiler cannot
//! auto-vectorize, chiefly transcendental objective functions.

mod scalar;

const LOG_LOSS_EPSILON: f64 = 1e-15;
/// XGBoost's binary `logloss` floor (`float eps = 1e-16`), widened to `f64`.
const BINARY_LOG_LOSS_EPSILON: f64 = 1e-16f32 as f64;
const MIN_POSITIVE_PREDICTION: f64 = 1e-8;

use crate::objective::GradPair;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use std::sync::LazyLock;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const MIN_SIMD_LEN: usize = 16;

// Layout contract the deinterleaving vector loads and stores rely on.
const _: () = assert!(std::mem::size_of::<GradPair>() == 2 * std::mem::size_of::<f32>());

/// Inputs with a larger magnitude take the scalar path in the fast
/// exponential, sigmoid, and softmax kernels.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const MAX_FAST_EXP_INPUT: f32 = 80.0;

#[cfg(target_arch = "aarch64")]
static NEON_AVAILABLE: LazyLock<bool> =
    LazyLock::new(|| std::arch::is_aarch64_feature_detected!("neon"));

#[cfg(target_arch = "x86_64")]
static AVX2_FMA_AVAILABLE: LazyLock<bool> = LazyLock::new(|| {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
});

/// Run the per-arch kernel for a `&mut [f32]` unary inplace op when the slice
/// is long enough, falling through to the caller's scalar tail otherwise.
macro_rules! dispatch_unary_inplace {
    ($values:expr, $kernel:ident) => {
        #[cfg(target_arch = "aarch64")]
        if $values.len() >= MIN_SIMD_LEN && neon_available() {
            // SAFETY: runtime detection proves NEON is present and the kernel
            // bounds vector accesses by the slice length.
            unsafe { aarch64::$kernel($values) };
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if $values.len() >= MIN_SIMD_LEN && avx2_fma_available() {
            // SAFETY: AVX2/FMA are present and the kernel bounds vector
            // accesses by the slice length.
            unsafe { x86_64::$kernel($values) };
            return;
        }
    };
}

/// Run the per-arch gradient kernel when `$gate` (typically `gradient_gate`
/// or `metric_gate`) holds, falling through to the caller's scalar
/// tail otherwise. The three-arm form also dispatches the `x86_64` kernel; the
/// two-arm form is NEON-only for kernels with no `x86_64` counterpart. Both
/// forms return the kernel's value, so `()`-valued gradient kernels and
/// `(f64, f64)`-valued metric-sum kernels share the same expansion.
macro_rules! dispatch_gradient {
    ($gate:expr, $neon_call:expr) => {
        #[cfg(target_arch = "aarch64")]
        if $gate && neon_available() {
            // SAFETY: NEON is present and the gate's cover check proves every
            // input and output slice spans the dispatched length.
            return unsafe { $neon_call };
        }
    };
    ($gate:expr, $neon_call:expr, $avx_call:expr) => {
        dispatch_gradient!($gate, $neon_call);
        #[cfg(target_arch = "x86_64")]
        if $gate && avx2_fma_available() {
            // SAFETY: AVX2/FMA are present and the gate's cover check proves
            // every input and output slice spans the dispatched length.
            unsafe { $avx_call };
            return;
        }
    };
}

/// Resolve the process-wide AArch64 backend lazily on the first numeric-kernel
/// call. `LazyLock` makes feature detection a one-time initialization cost, and
/// subsequent calls are a cached load and comparison.
#[cfg(target_arch = "aarch64")]
#[inline]
fn neon_available() -> bool {
    *NEON_AVAILABLE
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn avx2_fma_available() -> bool {
    *AVX2_FMA_AVAILABLE
}

/// Hint the cache hierarchy that `value` will be read soon. A pure performance
/// hint: it never faults and has no observable effect on program state.
#[inline(always)]
pub(crate) fn prefetch_read<T>(value: &T) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: PRFM only touches the cache and cannot fault or write memory;
    // the operand is a valid reference.
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{ptr}]",
            ptr = in(reg) std::ptr::from_ref::<T>(value),
            options(nostack, readonly, preserves_flags)
        );
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: PREFETCHT0 is available on every x86_64 CPU and cannot fault.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(
            std::ptr::from_ref::<T>(value).cast::<i8>(),
        );
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = value;
}

/// Number of entries of `cuts` that are `<= value` (ordered comparison).
#[inline]
pub(crate) fn count_le(cuts: &[f32], value: f32) -> usize {
    #[cfg(target_arch = "aarch64")]
    if cuts.len() == 16 && neon_available() {
        // SAFETY: NEON is present and the slice holds exactly four vectors.
        return unsafe { aarch64::count_le_16(cuts, value) };
    }
    #[cfg(target_arch = "x86_64")]
    if cuts.len() == 16 {
        // SAFETY: the slice holds exactly four vectors; SSE2 is baseline.
        return unsafe { x86_64::count_le_16(cuts, value) };
    }
    cuts.iter().filter(|&&cut| cut <= value).count()
}

/// Whether a gradient kernel may take the vector path: at least
/// `MIN_SIMD_LEN` predictions, with labels, weights, and `out` covering them.
#[inline]
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn gradient_gate(preds: &[f32], labels: &[f32], weights: Option<&[f32]>, out: &[GradPair]) -> bool {
    metric_gate(preds, labels, weights) && out.len() >= preds.len()
}

/// Whether a metric-sum kernel may take the vector path: at least
/// `MIN_SIMD_LEN` predictions, with labels and weights covering them.
#[inline]
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn metric_gate(preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> bool {
    let len = preds.len();
    len >= MIN_SIMD_LEN && labels.len() >= len && weights.is_none_or(|values| values.len() >= len)
}

/// Whether `labels` (and `weights`, if any) index complete `num_class` rows
/// of `preds`.
#[inline]
fn class_rows_cover(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
) -> bool {
    labels
        .len()
        .checked_mul(num_class)
        .is_some_and(|len| preds.len() >= len)
        && weights.is_none_or(|values| values.len() >= labels.len())
}

/// XGBoost's `common::Sigmoid`: `1 / (expf(min(-x, 88.7)) + 1)` (the
/// `1e-16f` upstream adds to the denominator vanishes in `f32`).
#[inline]
pub(crate) fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / ((-x).min(88.7).exp() + 1.0)
}

#[inline]
pub(crate) fn exp_inplace(values: &mut [f32]) {
    dispatch_unary_inplace!(values, exp_inplace);
    for value in values.iter_mut() {
        *value = value.exp();
    }
}

#[inline]
pub(crate) fn sigmoid_inplace(values: &mut [f32]) {
    dispatch_unary_inplace!(values, sigmoid_inplace);
    for value in values.iter_mut() {
        *value = sigmoid_scalar(*value);
    }
}

pub(crate) fn logistic_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    scale_pos_weight: f32,
    min_hess: f32,
    out: &mut [GradPair],
) {
    dispatch_gradient!(
        gradient_gate(preds, labels, weights, out),
        aarch64::logistic_gradient(preds, labels, weights, scale_pos_weight, min_hess, out),
        x86_64::logistic_gradient(preds, labels, weights, scale_pos_weight, min_hess, out)
    );
    scalar::logistic_gradient(
        preds,
        labels,
        weights,
        scale_pos_weight,
        min_hess,
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
    dispatch_gradient!(
        gradient_gate(preds, labels, weights, out),
        aarch64::poisson_gradient(preds, labels, weights, max_delta_step, out)
    );
    scalar::poisson_gradient(preds, labels, weights, max_delta_step, out, 0..preds.len());
}

pub(crate) fn gamma_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    out: &mut [GradPair],
) {
    dispatch_gradient!(
        gradient_gate(preds, labels, weights, out),
        aarch64::gamma_gradient(preds, labels, weights, out)
    );
    scalar::gamma_gradient(preds, labels, weights, out, 0..preds.len());
}

pub(crate) fn tweedie_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    rho: f32,
    out: &mut [GradPair],
) {
    dispatch_gradient!(
        gradient_gate(preds, labels, weights, out),
        aarch64::tweedie_gradient(preds, labels, weights, rho, out)
    );
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
    #[cfg(target_arch = "x86_64")]
    if (num_class == 2 || num_class == 4) && values.len() >= MIN_SIMD_LEN && avx2_fma_available() {
        // SAFETY: AVX2/FMA are present; each specialization processes whole
        // rows per vector and handles the remaining rows scalarly.
        unsafe {
            if num_class == 2 {
                x86_64::short_softmax_rows::<2>(values);
            } else {
                x86_64::short_softmax_rows::<4>(values);
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
    #[cfg(target_arch = "x86_64")]
    if (num_class == 2 || num_class == 4)
        && preds.len() >= MIN_SIMD_LEN
        && complete
        && avx2_fma_available()
    {
        // SAFETY: the complete-matrix check covers every prediction, label,
        // weight and output row; AVX2/FMA are present and K is 2 or 4.
        unsafe {
            if num_class == 2 {
                x86_64::short_softmax_gradient::<2>(preds, labels, weights, min_hess, out);
            } else {
                x86_64::short_softmax_gradient::<4>(preds, labels, weights, min_hess, out);
            }
        }
        return;
    }

    debug_assert!(complete);
    softmax_gradient_rows_scalar(
        preds,
        labels,
        weights,
        min_hess,
        out,
        0..labels.len(),
        num_class,
    );
}

/// Scalar softmax gradient over the given rows of a complete row-major
/// prediction matrix; the remainder path of the short-row vector kernels.
pub(super) fn softmax_gradient_rows_scalar(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    min_hess: f32,
    out: &mut [GradPair],
    rows: std::ops::Range<usize>,
    k: usize,
) {
    for current in rows {
        let base = current * k;
        softmax_gradient_row_scalar(
            &preds[base..base + k],
            labels[current] as usize,
            weights.map_or(1.0, |values| values[current]),
            min_hess,
            &mut out[base..base + k],
        );
    }
}

/// XGBoost's `common::Softmax`: shift by the row maximum, sum the
/// exponentials in `f64`, and divide each entry by that sum rounded to `f32`.
pub(super) fn softmax_scalar(values: &mut [f32]) {
    let Some(&first) = values.first() else {
        return;
    };
    let wmax = values[1..].iter().fold(first, |m, &v| v.max(m));
    let mut wsum = 0f64;
    for value in values.iter_mut() {
        *value = (*value - wmax).exp();
        wsum += f64::from(*value);
    }
    let wsum = wsum as f32;
    for value in values.iter_mut() {
        *value /= wsum;
    }
}

/// XGBoost's `SoftmaxMultiClassObj::GetGradient` for one row: the shift is
/// `max(f32::MIN_POSITIVE, preds...)` (upstream seeds `wmax` with
/// `numeric_limits<float>::min()`), the exponentials are summed in `f64`, and
/// `p = expf(x - wmax) / (float)wsum`.
pub(super) fn softmax_gradient_row_scalar(
    preds: &[f32],
    label: usize,
    weight: f32,
    min_hess: f32,
    out: &mut [GradPair],
) {
    let mut wmax = f32::MIN_POSITIVE;
    for &value in preds {
        wmax = value.max(wmax);
    }
    let mut wsum = 0f64;
    for &prediction in preds {
        wsum += f64::from((prediction - wmax).exp());
    }
    let wsum = wsum as f32;
    for (class, (output, &prediction)) in out.iter_mut().zip(preds).enumerate() {
        let p = (prediction - wmax).exp() / wsum;
        let h = (2.0 * p * (1.0 - p) * weight).max(min_hess);
        let g = if class == label { p - 1.0 } else { p };
        *output = GradPair::new(g * weight, h);
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
    dispatch_gradient!(
        metric_gate(preds, labels, weights),
        aarch64::distance_sum::<SQUARED>(preds, labels, weights)
    );

    scalar::distance_sum::<SQUARED>(preds, labels, weights, 0..preds.len())
}

pub(crate) fn classification_error_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
) -> (f64, f64) {
    dispatch_gradient!(
        metric_gate(preds, labels, weights),
        aarch64::classification_error_sum(preds, labels, weights)
    );

    scalar::classification_error_sum(preds, labels, weights, 0..preds.len())
}

pub(crate) fn log_loss_sum(preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> (f64, f64) {
    dispatch_gradient!(
        metric_gate(preds, labels, weights),
        aarch64::log_loss_sum(preds, labels, weights)
    );

    scalar::log_loss(preds, labels, weights, 0..preds.len())
}

pub(crate) fn positive_nloglik_sum<const GAMMA: bool>(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
) -> (f64, f64) {
    dispatch_gradient!(
        metric_gate(preds, labels, weights),
        aarch64::positive_nloglik_sum::<GAMMA>(preds, labels, weights)
    );

    scalar::positive_nloglik::<GAMMA>(preds, labels, weights, 0..preds.len())
}

pub(crate) fn tweedie_nloglik_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    rho: f64,
) -> (f64, f64) {
    dispatch_gradient!(
        rho.is_finite() && rho > 1.0 && rho < 2.0 && metric_gate(preds, labels, weights),
        aarch64::tweedie_nloglik_sum(preds, labels, weights, rho)
    );

    scalar::tweedie_nloglik(preds, labels, weights, rho, 0..preds.len())
}

pub(crate) fn multiclass_log_loss_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
) -> (f64, f64) {
    let complete = class_rows_cover(preds, labels, weights, num_class);
    dispatch_gradient!(
        labels.len() >= MIN_SIMD_LEN && complete,
        aarch64::multiclass_log_loss_sum(preds, labels, weights, num_class)
    );

    debug_assert!(complete);
    scalar::multiclass_log_loss(preds, labels, weights, num_class, 0..labels.len())
}

pub(crate) fn multiclass_error_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
) -> (f64, f64) {
    let complete = class_rows_cover(preds, labels, weights, num_class);
    dispatch_gradient!(
        num_class >= 8
            && u32::try_from(num_class).is_ok()
            && labels.len() >= MIN_SIMD_LEN
            && complete,
        aarch64::multiclass_error_sum(preds, labels, weights, num_class)
    );

    debug_assert!(complete);
    multiclass_error_sum_rows(preds, labels, weights, num_class, argmax_scalar)
}

/// Row loop of the multiclass error sum, parameterized on the argmax so the
/// NEON kernel can reuse it with its vectorized `argmax`.
pub(super) fn multiclass_error_sum_rows(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
    argmax: impl Fn(&[f32]) -> usize,
) -> (f64, f64) {
    let mut wrong = 0.0;
    let mut weight_sum = 0.0;
    for (row_index, &label) in labels.iter().enumerate() {
        let weight = weights.map_or(1.0, |values| f64::from(values[row_index]));
        let row = &preds[row_index * num_class..(row_index + 1) * num_class];
        let best = argmax(row);
        if best != label as usize {
            wrong += weight;
        }
        weight_sum += weight;
    }
    (wrong, weight_sum)
}

pub(crate) fn argmax_scalar(values: &[f32]) -> usize {
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
