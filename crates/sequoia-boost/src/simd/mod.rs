//! Runtime-dispatched numeric kernels for operations the compiler cannot
//! auto-vectorize, chiefly transcendental objective functions.

mod scalar;

const LOG_LOSS_EPSILON: f64 = 1e-15;
const MIN_POSITIVE_PREDICTION: f64 = 1e-8;

use crate::objective::GradPair;
use crate::tree::gain::GradStats;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use crate::tree::gain::RegParams;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use std::sync::LazyLock;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const MIN_SIMD_LEN: usize = 16;

// Layout contracts the deinterleaving vector loads and stores rely on.
const _: () = assert!(std::mem::size_of::<GradPair>() == 2 * std::mem::size_of::<f32>());
const _: () = assert!(std::mem::size_of::<GradStats>() == 2 * std::mem::size_of::<f64>());

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

/// Run the per-arch gradient kernel when `$gate` (length plus
/// `gradient_slices_cover`) holds, falling through to the caller's scalar
/// tail otherwise. The three-arm form also dispatches the x86_64 kernel; the
/// two-arm form is NEON-only for kernels with no x86_64 counterpart. Both
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

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(crate) struct SplitCandidate {
    pub(crate) loss_change: f64,
    pub(crate) split_offset: usize,
    pub(crate) left: GradStats,
    pub(crate) right: GradStats,
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(crate) enum DenseSplitScan {
    ScalarFallback,
    Scanned(Option<SplitCandidate>),
}

/// Relative slack applied to the division-free prefilter threshold. Both the
/// cross-multiplied test and the exact quotient test round to within a few
/// ULPs, so this margin guarantees the prefilter never rejects a candidate the
/// exact comparison would accept. False positives merely pay for a division.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(super) const PREFILTER_SLACK: f64 = 1e-9;

/// Scaled `gain(L) + gain(R)` a candidate must exceed to beat `best_loss`.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline]
pub(super) fn prefilter_target(best_loss: f64, comparison_epsilon: f64, parent_gain: f64) -> f64 {
    (best_loss + comparison_epsilon + parent_gain) * (1.0 - PREFILTER_SLACK)
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
            ptr = in(reg) value as *const T,
            options(nostack, readonly, preserves_flags)
        );
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: PREFETCHT0 is available on every x86_64 CPU and cannot fault.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(
            (value as *const T).cast::<i8>(),
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

#[inline]
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn gradient_slices_cover(
    len: usize,
    labels: &[f32],
    weights: Option<&[f32]>,
    out: &[GradPair],
) -> bool {
    metric_slices_cover(len, labels, weights) && out.len() >= len
}

#[inline]
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
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
    dispatch_gradient!(
        values.len() >= MIN_SIMD_LEN,
        aarch64::sum_grad_stats(values)
    );

    let mut sum = GradStats::default();
    for &value in values {
        sum.add(value);
    }
    sum
}

/// Try the vector split-gain scan used by the common dense, unconstrained
/// histogram path. `ScalarFallback` asks the caller to use its scalar scan.
///
/// `incumbent_loss` is the best loss change already found for the node, from
/// any earlier feature. The scan seeds its sequential epsilon comparison with
/// that value, so its acceptance decisions replay the caller's scalar scan
/// for this feature and the returned candidate (if any) is the one the scalar
/// path would have accepted in the same position.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(crate) fn dense_unconstrained_best_split(
    histogram: &[GradStats],
    total: GradStats,
    reg: &RegParams,
    parent_gain: f64,
    incumbent_loss: f64,
    comparison_epsilon: f64,
) -> DenseSplitScan {
    if histogram.len() < MIN_SIMD_LEN || reg.max_delta_step != 0.0 {
        return DenseSplitScan::ScalarFallback;
    }
    #[cfg(target_arch = "aarch64")]
    if neon_available() {
        // SAFETY: NEON is present. The kernel only reads `histogram` and keeps
        // all vector loads within the complete candidate range.
        return DenseSplitScan::Scanned(unsafe {
            aarch64::dense_unconstrained_best_split(
                histogram,
                total,
                reg,
                parent_gain,
                incumbent_loss,
                comparison_epsilon,
            )
        });
    }
    #[cfg(target_arch = "x86_64")]
    if avx2_fma_available() {
        // SAFETY: AVX2 and FMA are present. The kernel only reads `histogram`
        // and keeps all vector loads within the complete candidate range.
        return DenseSplitScan::Scanned(unsafe {
            x86_64::dense_unconstrained_best_split(
                histogram,
                total,
                reg,
                parent_gain,
                incumbent_loss,
                comparison_epsilon,
            )
        });
    }
    DenseSplitScan::ScalarFallback
}

#[inline]
pub(crate) fn exp_inplace(values: &mut [f32]) {
    dispatch_unary_inplace!(values, exp_inplace);
    values.iter_mut().for_each(|value| *value = value.exp());
}

#[inline]
pub(crate) fn sigmoid_inplace(values: &mut [f32]) {
    dispatch_unary_inplace!(values, sigmoid_inplace);
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
    dispatch_gradient!(
        preds.len() >= MIN_SIMD_LEN && gradient_slices_cover(preds.len(), labels, weights, out),
        aarch64::logistic_gradient(preds, labels, weights, scale_pos_weight, min_hess, out),
        x86_64::logistic_gradient(preds, labels, weights, scale_pos_weight, min_hess, out)
    );
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
    dispatch_gradient!(
        preds.len() >= MIN_SIMD_LEN && gradient_slices_cover(preds.len(), labels, weights, out),
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
        preds.len() >= MIN_SIMD_LEN && gradient_slices_cover(preds.len(), labels, weights, out),
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
        preds.len() >= MIN_SIMD_LEN && gradient_slices_cover(preds.len(), labels, weights, out),
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
                x86_64::short_softmax_rows::<2>(values)
            } else {
                x86_64::short_softmax_rows::<4>(values)
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
                x86_64::short_softmax_gradient::<2>(preds, labels, weights, min_hess, out)
            } else {
                x86_64::short_softmax_gradient::<4>(preds, labels, weights, min_hess, out)
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
    dispatch_gradient!(
        preds.len() >= MIN_SIMD_LEN && metric_slices_cover(preds.len(), labels, weights),
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
        preds.len() >= MIN_SIMD_LEN && metric_slices_cover(preds.len(), labels, weights),
        aarch64::classification_error_sum(preds, labels, weights)
    );

    scalar::classification_error_sum(preds, labels, weights, 0..preds.len())
}

pub(crate) fn log_loss_sum(preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> (f64, f64) {
    dispatch_gradient!(
        preds.len() >= MIN_SIMD_LEN && metric_slices_cover(preds.len(), labels, weights),
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
        preds.len() >= MIN_SIMD_LEN && metric_slices_cover(preds.len(), labels, weights),
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
        rho.is_finite()
            && rho > 1.0
            && rho < 2.0
            && preds.len() >= MIN_SIMD_LEN
            && metric_slices_cover(preds.len(), labels, weights),
        aarch64::tweedie_nloglik_sum(preds, labels, weights, rho)
    );

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
    dispatch_gradient!(
        labels.len() >= MIN_SIMD_LEN && complete,
        aarch64::multiclass_log_loss_sum(preds, labels, weights, num_class)
    );

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
    dispatch_gradient!(
        num_class >= 8
            && num_class <= u32::MAX as usize
            && labels.len() >= MIN_SIMD_LEN
            && complete,
        aarch64::multiclass_error_sum(preds, labels, weights, num_class)
    );

    debug_assert!(complete);
    multiclass_error_sum_scalar(preds, labels, weights, num_class)
}

fn multiclass_error_sum_scalar(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
) -> (f64, f64) {
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
        let weight = weights.map_or(1.0, |values| values[row_index] as f64);
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
