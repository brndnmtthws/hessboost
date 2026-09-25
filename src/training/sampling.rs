//! Gradient-based row sampling (`sampling_method=gradient_based`).
//!
//! This is XGBoost 3.4.2's CPU minimal-variance sampling
//! (`src/tree/hist/sampler.cc`), applied to each tree's gradients before the
//! tree is grown:
//!
//! 1. every row gets the regularized absolute gradient
//!    `r_i = sqrt(sum_t(g_it^2 + 0.1 * h_it^2))` (summed over targets);
//! 2. with `m = trunc(n * subsample)` rows to keep in expectation, a threshold
//!    `u` is binary-searched over the ascending `r` values and their running
//!    sums so that `sum_i min(1, r_i / u) = m`;
//! 3. row `i` is kept with probability `p_i = min(1, r_i / u)`; a kept row with
//!    `p_i < 1` has its gradient and Hessian scaled by `1 / p_i`, so every
//!    node's gradient sums are unbiased estimates of the full-data sums.
//!    Dropped rows contribute nothing.
//!
//! Arithmetic mirrors upstream in `f32` (including the sequential running
//! sum). Where that overflows (finite gradients whose squares or sums exceed
//! `f32`, e.g. from labels near `1e20`), `r` and `u` are recomputed from
//! gradients scaled by a power of two (with upstream's `kRtEps` floor on `u`
//! applied to the unscaled value), which leaves every ratio `r_i / u`
//! exact; non-finite gradients, or rescaled ones that overflow, are an error.
//! The random stream is hessboost's own: one `u64` seed per call from the
//! caller's RNG, then one uniform draw per row from a stream fixed per block
//! of [`BLOCK_ROWS`] rows, so the sample does not depend on the thread count.

use crate::K_RT_EPS_F32;
use crate::error::{HessboostError, Result};
use crate::objective::GradPair;
use crate::rng::{GOLDEN, Rng};
use rayon::prelude::*;

/// XGBoost `kDefaultMvsLambda`: the Hessian weight inside `r_i`. A fixed
/// sampling regularizer, unrelated to the tree `lambda`.
const MVS_LAMBDA: f32 = 0.1;

/// Rows per independently seeded random stream (and per parallel task).
const BLOCK_ROWS: usize = 4096;

/// The outcome of one gradient-based sample.
#[derive(Debug)]
pub(crate) struct GradientSample {
    /// Kept rows, ascending.
    pub rows: Vec<u32>,
    /// Row-major `[row][target]` gradients: rescaled for kept rows, zero for
    /// dropped ones.
    pub gpair: Vec<GradPair>,
    /// Inclusion probability of each kept row (parallel to `rows`).
    pub probability: Vec<f32>,
}

impl GradientSample {
    /// Replay this sample on other gradients of the same rows (`n_targets`
    /// pairs per row): kept rows rescaled by their inclusion probability,
    /// dropped rows zeroed (XGBoost's `ApplySampling`, used for the value
    /// gradients of reduced-gradient training).
    pub(crate) fn apply(&self, gpair: &[GradPair], n_targets: usize) -> Vec<GradPair> {
        let mut out = vec![GradPair::default(); gpair.len()];
        for (&row, &p) in self.rows.iter().zip(&self.probability) {
            let range = row as usize * n_targets..(row as usize + 1) * n_targets;
            for (d, s) in out[range.clone()].iter_mut().zip(&gpair[range]) {
                *d = rescale(p, *s);
            }
        }
        out
    }
}

/// Sample the rows of `gpair` (row-major, `n_targets` pairs per row) for one
/// tree. Returns `None` when `trunc(n * subsample) >= n`, i.e. XGBoost would
/// not sample at all; the caller then uses every row unchanged. Fails when a
/// gradient or Hessian is not finite, or when rescaling a kept row by its
/// inclusion probability overflows `f32`.
pub(crate) fn gradient_based_sample(
    gpair: &[GradPair],
    n_targets: usize,
    subsample: f64,
    rng: &mut Rng,
) -> Result<Option<GradientSample>> {
    debug_assert!(n_targets >= 1 && gpair.len().is_multiple_of(n_targets));
    let n = gpair.len() / n_targets;
    let budget = (n as f32 * subsample as f32) as usize;
    if n == 0 || budget >= n {
        return Ok(None);
    }
    let seed = rng.next_u64();
    if budget == 0 {
        // An empty budget keeps nothing (upstream zeroes every pair).
        return Ok(Some(GradientSample {
            rows: Vec::new(),
            gpair: vec![GradPair::default(); gpair.len()],
            probability: Vec::new(),
        }));
    }

    // XGBoost's `f32` arithmetic, unless it overflows.
    let (reg_abs_grad, threshold, scale) = if let Some((r, u)) =
        sampling_statistics(gpair, n_targets, budget, 1.0)
    {
        (r, u, 1.0)
    } else {
        let scale = overflow_scale(gpair)?;
        let (r, u) = sampling_statistics(gpair, n_targets, budget, scale).ok_or_else(non_finite)?;
        (r, u, scale)
    };

    let blocks: Vec<(Vec<u32>, Vec<f32>, Vec<GradPair>)> = gpair
        .par_chunks(BLOCK_ROWS * n_targets)
        .zip(reg_abs_grad.par_chunks(BLOCK_ROWS))
        .enumerate()
        .map(|(block, (pairs, rag))| {
            let mut stream = Rng::new(block_seed(seed, block));
            let first = block * BLOCK_ROWS;
            let mut rows = Vec::new();
            let mut kept_p = Vec::new();
            let mut out = vec![GradPair::default(); pairs.len()];
            for (i, &r) in rag.iter().enumerate() {
                let p = probability(threshold, r, scale);
                // Exactly one draw per row, kept or not.
                let draw = stream.f32();
                if p >= 1.0 || (p > 0.0 && draw <= p) {
                    let src = &pairs[i * n_targets..(i + 1) * n_targets];
                    let dst = &mut out[i * n_targets..(i + 1) * n_targets];
                    for (d, s) in dst.iter_mut().zip(src) {
                        *d = rescale(p, *s);
                    }
                    rows.push((first + i) as u32);
                    kept_p.push(p);
                }
            }
            (rows, kept_p, out)
        })
        .collect();

    let mut rows = Vec::new();
    let mut probability = Vec::new();
    let mut out = Vec::with_capacity(gpair.len());
    for (block_rows, block_p, block_pairs) in blocks {
        rows.extend(block_rows);
        probability.extend(block_p);
        out.extend(block_pairs);
    }
    if !out.iter().all(|g| g.grad.is_finite() && g.hess.is_finite()) {
        return Err(non_finite());
    }
    Ok(Some(GradientSample {
        rows,
        gpair: out,
        probability,
    }))
}

/// The regularized absolute gradients of `gpair` scaled by `scale`, and
/// their threshold for `budget`, or `None` when either is not finite.
fn sampling_statistics(
    gpair: &[GradPair],
    n_targets: usize,
    budget: usize,
    scale: f32,
) -> Option<(Vec<f32>, f32)> {
    let reg_abs_grad: Vec<f32> = gpair
        .par_chunks(n_targets)
        .with_min_len(BLOCK_ROWS)
        .map(|row| regularized_abs_grad(row, scale))
        .collect();
    if !reg_abs_grad.iter().all(|r| r.is_finite()) {
        return None;
    }
    let threshold = threshold(&reg_abs_grad, budget);
    threshold.is_finite().then_some((reg_abs_grad, threshold))
}

/// The power of two that brings the largest gradient or Hessian magnitude
/// of `gpair` into `(1/2, 1]` (magnitudes at most `1` are left unscaled),
/// so neither the squares nor the sums over rows of the scaled regularized
/// gradients overflow. Scaling by a power of two is exact, so every ratio
/// `r_i / u` is the one unbounded `f32` would give (up to underflow of
/// values `2^-126` below the largest). Fails for a non-finite value.
fn overflow_scale(gpair: &[GradPair]) -> Result<f32> {
    let mut largest = 0.0f32;
    for g in gpair {
        if !(g.grad.is_finite() && g.hess.is_finite()) {
            return Err(non_finite());
        }
        largest = largest.max(g.grad.abs()).max(g.hess.abs());
    }
    if largest <= 1.0 {
        return Ok(1.0);
    }
    // The smallest `k` with `largest <= 2^k`, from the bits (`largest` is
    // normal and below `2^128`, so `1 <= k <= 128`).
    let bits = largest.to_bits();
    let k = (bits >> 23) as i32 - 127 + i32::from(bits & 0x007f_ffff != 0);
    // `2^-k` is exact in `f32` (subnormal for `k > 126`). Built as a normal
    // `f64` and narrowed exactly: `powi` would compute `1 / 2^k`, which is
    // `1 / inf = 0` at `k = 128`.
    Ok(f64::from_bits(((1023 - k) as u64) << 52) as f32)
}

/// The error for gradients the sample cannot be computed from.
fn non_finite() -> HessboostError {
    HessboostError::invalid_param(
        "sampling_method",
        "`gradient_based` sampling needs finite gradients and Hessians whose rescaled \
         values stay finite in f32; the objective produced values that do not",
    )
}

/// The seed of one row block's random stream.
fn block_seed(seed: u64, block: usize) -> u64 {
    seed ^ (block as u64 + 1).wrapping_mul(GOLDEN)
}

/// A row's regularized absolute gradient `sqrt(sum_t(g_t^2 + 0.1 * h_t^2))`,
/// of the pairs scaled by `scale` (`1` reproduces XGBoost exactly).
fn regularized_abs_grad(row: &[GradPair], scale: f32) -> f32 {
    let sum_sq = row.iter().fold(0.0f32, |acc, g| {
        let (grad, hess) = (g.grad * scale, g.hess * scale);
        acc + (grad * grad + MVS_LAMBDA * (hess * hess))
    });
    sum_sq.sqrt()
}

/// XGBoost `CalculateThreshold`: the `u` with `sum_i min(1, r_i / u) = budget`,
/// found by binary search over the ascending `r` values (`budget < n`).
fn threshold(reg_abs_grad: &[f32], budget: usize) -> f32 {
    let n = reg_abs_grad.len();
    let mut sorted = reg_abs_grad.to_vec();
    sorted.par_sort_unstable_by(f32::total_cmp);
    sorted.push(f32::MAX); // upper bound of the last interval
    let mut csum = Vec::with_capacity(n);
    let mut acc = 0.0f32;
    for &r in &sorted[..n] {
        acc += r;
        csum.push(acc);
    }

    let (mut low, mut high) = (0i64, n as i64 - 1);
    while low <= high {
        let i = low + (high - low) / 2;
        let iu = i as usize;
        let (lower, upper) = (sorted[iu], sorted[iu + 1]);
        let n_above = n - iu - 1;
        let denom = budget as f32 - n_above as f32;
        if denom <= 0.0 {
            low = i + 1;
            continue;
        }
        let u = csum[iu] / denom;
        if u > lower && u <= upper {
            return u;
        }
        if u <= lower {
            high = i - 1;
        } else {
            low = i + 1;
        }
    }
    // Every gradient equal: no interval brackets `u`; spread the budget evenly.
    csum[n - 1] / budget as f32
}

/// XGBoost `SamplingProbability`: `r / u`, with `|u|` floored at `kRtEps`.
/// `threshold` and `reg_abs_grad` are scaled by the power of two `scale`
/// (`1` reproduces XGBoost exactly); the floor applies to the unscaled `u`,
/// and a floored ratio divides the unscaled `r` (infinite only where the
/// ratio would be anyway, which keeps the row unscaled), so the ratio is
/// the unscaled one. Upstream's `0` for an infinite `u` is unreachable
/// here: [`sampling_statistics`] only yields finite thresholds.
fn probability(threshold: f32, reg_abs_grad: f32, scale: f32) -> f32 {
    // Exact in `f64`: `scale` is a power of two.
    if f64::from(threshold.abs()) < f64::from(K_RT_EPS_F32) * f64::from(scale) {
        (reg_abs_grad / scale) / K_RT_EPS_F32.copysign(threshold)
    } else {
        reg_abs_grad / threshold
    }
}

/// XGBoost `RescaleGrad`: divide a kept pair by its inclusion probability.
fn rescale(p: f32, g: GradPair) -> GradPair {
    if p >= 1.0 {
        return g;
    }
    let inv = 1.0 / p;
    GradPair::new(g.grad * inv, g.hess * inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradients(n: usize, n_targets: usize, seed: u64) -> Vec<GradPair> {
        let mut rng = Rng::new(seed);
        (0..n * n_targets)
            .map(|i| {
                // Heavy-tailed magnitudes so inclusion probabilities vary widely.
                let u: f32 = rng.f32();
                let g = (u * u * u * 8.0 - 1.0) * if i % 3 == 0 { -1.0 } else { 1.0 };
                GradPair::new(g, 0.5 + rng.f32())
            })
            .collect()
    }

    fn rag(g: &[GradPair], n_targets: usize) -> Vec<f32> {
        g.chunks(n_targets)
            .map(|row| regularized_abs_grad(row, 1.0))
            .collect()
    }

    #[test]
    fn threshold_solves_the_budget_equation() {
        let g = gradients(5000, 1, 3);
        let r = rag(&g, 1);
        for budget in [1usize, 50, 1500, 2500, 4999] {
            let u = threshold(&r, budget);
            let expected: f64 = r.iter().map(|&x| f64::from((x / u).min(1.0))).sum();
            assert!(
                (expected - budget as f64).abs() < 1e-3 * budget as f64 + 1e-2,
                "budget {budget}: sum p = {expected}"
            );
        }
        // Equal gradients: no bracketing interval, fall back to total / budget.
        let flat = vec![2.0f32; 10];
        assert_eq!(threshold(&flat, 4), 20.0 / 4.0);
    }

    #[test]
    fn full_budget_does_not_sample() {
        let g = gradients(100, 1, 1);
        let mut rng = Rng::new(0);
        assert!(
            gradient_based_sample(&g, 1, 1.0, &mut rng)
                .unwrap()
                .is_none()
        );
        // 3 rows * 0.999 = 2.997 -> 2 rows: sampling happens.
        assert!(
            gradient_based_sample(&g[..3], 1, 0.999, &mut rng)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn empty_budget_keeps_nothing() {
        let g = gradients(3, 1, 1);
        let mut rng = Rng::new(0);
        let s = gradient_based_sample(&g, 1, 0.3, &mut rng)
            .unwrap()
            .unwrap();
        assert!(s.rows.is_empty());
        assert!(s.gpair.iter().all(|p| *p == GradPair::default()));
    }

    /// Each row's empirical inclusion frequency matches `min(1, r_i / u)`, the
    /// expected kept count is the budget, and the rescaled gradient/Hessian
    /// totals are unbiased for the full-data totals.
    #[test]
    fn inclusion_probabilities_and_rescaled_sums_are_unbiased() {
        let n = 400;
        let g = gradients(n, 1, 11);
        let r = rag(&g, 1);
        let budget = (n as f32 * 0.3f32) as usize;
        let u = threshold(&r, budget);
        let p: Vec<f64> = r.iter().map(|&x| f64::from((x / u).min(1.0))).collect();

        let trials = 4000usize;
        let mut hits = vec![0usize; n];
        let (mut kept, mut grad_sum, mut hess_sum) = (0usize, 0.0f64, 0.0f64);
        let mut rng = Rng::new(5);
        for _ in 0..trials {
            let s = gradient_based_sample(&g, 1, 0.3, &mut rng)
                .unwrap()
                .unwrap();
            kept += s.rows.len();
            for &row in &s.rows {
                hits[row as usize] += 1;
            }
            // Dropped rows are zero, so summing everything sums the kept rows.
            let nonzero = s.gpair.iter().filter(|pair| pair.hess != 0.0).count();
            assert_eq!(nonzero, s.rows.len());
            for pair in &s.gpair {
                grad_sum += f64::from(pair.grad);
                hess_sum += f64::from(pair.hess);
            }
        }
        for (i, (&h, &pi)) in hits.iter().zip(&p).enumerate() {
            let freq = h as f64 / trials as f64;
            let se = (pi * (1.0 - pi) / trials as f64).sqrt();
            assert!(
                (freq - pi).abs() <= 5.0 * se + 1e-9,
                "row {i}: {freq} vs {pi}"
            );
        }
        let mean_kept = kept as f64 / trials as f64;
        assert!((mean_kept - budget as f64).abs() < 1.0, "kept {mean_kept}");

        let true_grad: f64 = g.iter().map(|x| f64::from(x.grad)).sum();
        let true_hess: f64 = g.iter().map(|x| f64::from(x.hess)).sum();
        let mean_grad = grad_sum / trials as f64;
        let mean_hess = hess_sum / trials as f64;
        assert!(
            (mean_hess - true_hess).abs() < 0.01 * true_hess,
            "hess {mean_hess} vs {true_hess}"
        );
        let grad_scale: f64 = g.iter().map(|x| f64::from(x.grad.abs())).sum();
        assert!(
            (mean_grad - true_grad).abs() < 0.01 * grad_scale,
            "grad {mean_grad} vs {true_grad}"
        );
    }

    /// Rows with the largest regularized gradients (p >= 1) are always kept
    /// and never rescaled.
    #[test]
    fn certain_rows_are_kept_unscaled() {
        let mut g = gradients(2000, 1, 2);
        g[7] = GradPair::new(1e4, 1.0);
        g[1500] = GradPair::new(-3e4, 2.0);
        let mut rng = Rng::new(9);
        for _ in 0..20 {
            let s = gradient_based_sample(&g, 1, 0.2, &mut rng)
                .unwrap()
                .unwrap();
            for row in [7usize, 1500] {
                assert!(s.rows.contains(&(row as u32)));
                assert_eq!(s.gpair[row], g[row]);
            }
        }
    }

    /// Multi-target rows are kept or dropped as a whole, with one probability
    /// from the norm over targets.
    #[test]
    fn multi_target_rows_share_one_decision() {
        let g = gradients(3000, 3, 4);
        let mut rng = Rng::new(1);
        let s = gradient_based_sample(&g, 3, 0.4, &mut rng)
            .unwrap()
            .unwrap();
        let r = rag(&g, 3);
        let u = threshold(&r, (3000f32 * 0.4f32) as usize);
        let mut kept = vec![false; 3000];
        for &row in &s.rows {
            kept[row as usize] = true;
        }
        for row in 0..3000 {
            let p = r[row] / u;
            for t in 0..3 {
                let (src, dst) = (g[row * 3 + t], s.gpair[row * 3 + t]);
                if !kept[row] {
                    assert_eq!(dst, GradPair::default());
                } else if p >= 1.0 {
                    assert_eq!(dst, src);
                } else {
                    assert_eq!(dst, rescale(p, src));
                }
            }
        }
    }

    #[test]
    fn sample_is_seeded_and_thread_count_independent() {
        let g = gradients(3 * BLOCK_ROWS + 17, 1, 8);
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut rng = Rng::new(42);
                let s = gradient_based_sample(&g, 1, 0.35, &mut rng)
                    .unwrap()
                    .unwrap();
                (s.rows, s.gpair)
            })
        };
        let serial = run(1);
        assert_eq!(serial, run(4));
        let mut other = Rng::new(43);
        let different = gradient_based_sample(&g, 1, 0.35, &mut other)
            .unwrap()
            .unwrap();
        assert_ne!(serial.0, different.rows, "the seed drives the sample");
    }

    /// Finite gradients whose squares overflow `f32` sample exactly like the
    /// same gradients at an ordinary scale: the same rows with the same
    /// probabilities, and the rescaled pairs scaled back up. The second case
    /// has rows at `f32::MAX` (scale `2^-128`, subnormal); they are certain
    /// (`p = 1`), so the rows rescaled by `1 / p` stay finite.
    #[test]
    fn overflowing_gradients_sample_like_scaled_down_ones() {
        let mut near_max = gradients(3000, 1, 6);
        for (i, row) in [5usize, 1234, 2999].into_iter().enumerate() {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            near_max[row] = GradPair::new(sign * f32::MAX * 2f32.powi(-85), 1.0);
        }
        for (g, n_targets, exponent) in [(gradients(3000, 2, 6), 2, 70), (near_max, 1, 85)] {
            let factor = 2f32.powi(exponent);
            let big: Vec<GradPair> = g
                .iter()
                .map(|p| GradPair::new(p.grad * factor, p.hess * factor))
                .collect();
            assert!(rag(&big, n_targets).iter().any(|r| r.is_infinite()));
            let s = gradient_based_sample(&g, n_targets, 0.3, &mut Rng::new(3))
                .unwrap()
                .unwrap();
            let b = gradient_based_sample(&big, n_targets, 0.3, &mut Rng::new(3))
                .unwrap()
                .unwrap();
            assert!(!s.rows.is_empty());
            assert_eq!(b.rows, s.rows, "2^{exponent}");
            assert_eq!(b.probability, s.probability, "2^{exponent}");
            if exponent == 85 {
                assert_eq!(big[1234].grad, -f32::MAX);
                assert!([5, 1234, 2999].iter().all(|r| b.rows.contains(r)));
            }
            for (bp, sp) in b.gpair.iter().zip(&s.gpair) {
                assert_eq!(bp.grad, sp.grad * factor);
                assert_eq!(bp.hess, sp.hess * factor);
            }
        }
    }

    /// The scale is the exact power of two bringing the largest magnitude
    /// into `(1/2, 1]`, also where it is subnormal (`2^-127`, `2^-128`).
    #[test]
    fn overflow_scale_is_an_exact_power_of_two_up_to_f32_max() {
        let pow2 = |e: i32| 2f32.powi(e);
        for (largest, exponent) in [
            (1.5, 1),
            (pow2(126), 126),
            (pow2(126) * 1.5, 127),
            (pow2(127), 127),
            (pow2(127) * 1.5, 128),
            (f32::MAX, 128),
        ] {
            // A runtime value, as in training.
            let largest: f32 = std::hint::black_box(largest);
            let scale = overflow_scale(&[GradPair::new(1.0, -largest)]).unwrap();
            assert!(scale > 0.0 && scale.is_finite(), "{largest}: {scale}");
            assert_eq!(f64::from(scale), (-f64::from(exponent)).exp2(), "{largest}");
            let scaled = largest * scale;
            assert!(scaled > 0.5 && scaled <= 1.0, "{largest}: {scaled}");
        }
        // Magnitudes at most 1 are not scaled.
        assert_eq!(overflow_scale(&[GradPair::new(0.25, 1.0)]).unwrap(), 1.0);
        assert_eq!(overflow_scale(&[GradPair::new(0.0, 0.0)]).unwrap(), 1.0);
    }

    #[test]
    fn non_finite_gradients_are_an_error() {
        let mut g = gradients(100, 1, 1);
        for bad in [f32::INFINITY, f32::NAN] {
            g[17] = GradPair::new(bad, 1.0);
            assert!(gradient_based_sample(&g, 1, 0.5, &mut Rng::new(0)).is_err());
        }
    }
}
