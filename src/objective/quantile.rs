//! Alpha-list regression objectives: quantile (`reg:quantileerror`) and
//! expectile (`reg:expectileerror`) regression, one model output per alpha
//! against the same scalar label.

use super::absolute::residual_scales;
use super::{GradPair, Objective, fit_stump, weighted_label_mean};
use crate::K_RT_EPS_F32;
use crate::error::{HessboostError, Result};

/// Bandwidth factor `c` of XGBoost's smoothed quantile score
/// (`kSmoothingScale`).
const SMOOTHING_SCALE: f32 = 0.04;
/// Relative floor of the quantile surrogate curvature `tanh(x)/x`
/// (`kMinSurrogateRatio`).
const MIN_SURROGATE_RATIO: f32 = 3.0e-4;

/// Check an alpha list the way XGBoost's `QuantileLossParam::Validate` /
/// `ExpectileLossParam::Validate` do (after rounding to `f32`, as XGBoost
/// stores them): non-empty, every entry in `[0, 1]`, ascending (equal
/// neighbours allowed).
pub(crate) fn validate_alphas(param: &'static str, alphas: &[f64]) -> Result<Vec<f32>> {
    let alpha: Vec<f32> = alphas.iter().map(|&a| a as f32).collect();
    if alpha.is_empty() {
        return Err(HessboostError::invalid_param(
            param,
            "is required and must list at least one value",
        ));
    }
    if !alpha.iter().all(|a| (0.0..=1.0).contains(a)) {
        return Err(HessboostError::invalid_param(
            param,
            "every value must be in the range [0, 1]",
        ));
    }
    if !alpha.is_sorted() {
        return Err(HessboostError::invalid_param(
            param,
            "values must be sorted in ascending order",
        ));
    }
    Ok(alpha)
}

/// Quantile regression (`reg:quantileerror`) with XGBoost 3.4's
/// automatically scaled, logistic-smoothed pinball score.
///
/// Output `j` estimates the `alpha[j]` quantile of the (single) label. Each
/// gradient call computes, per output, the scale `S_j = (Σ wᵢ √|rᵢⱼ| / Σ
/// wᵢ)²` of the residuals `r = margin − label` and, with `x = r / (0.04 S_j)`,
/// emits `g = w·S_j/2·(tanh x + 1 − 2α_j)` and the majorizing curvature `h =
/// w/(2·0.04)·max(tanh(x)/x, 3e-4)` (`tanh(x)/x = 1` at `x = 0`), all in
/// `f32`. A non-positive scale (e.g. zero total weight) or a zero row weight
/// gives an exactly zero pair.
///
/// The intercept of output `j` is the label's `alpha[j]` quantile: linear
/// interpolation on the `(n + 1)α` grid without weights, a step quantile of
/// the weighted CDF with weights (XGBoost `common::Quantile` /
/// `WeightedQuantile`). Predictions sort each row's outputs ascending, so
/// reported quantiles never cross; there is no link function.
#[derive(Debug, Clone)]
pub struct Quantile {
    alpha: Vec<f32>,
}

impl Quantile {
    /// Create for the quantile levels `alpha` (XGBoost `quantile_alpha`).
    ///
    /// # Errors
    ///
    /// `alpha` is empty, has an entry outside `[0, 1]`, or is not ascending.
    pub fn new(alpha: &[f64]) -> Result<Self> {
        Ok(Quantile {
            alpha: validate_alphas("quantile_alpha", alpha)?,
        })
    }
}

/// One output's smoothed pinball pair for residual `r`, scale `s`, level
/// `alpha`, and row weight `w` (XGBoost `QuantileRegression::GetGradient`).
#[inline]
fn quantile_pair(r: f32, s: f32, alpha: f32, w: f32) -> GradPair {
    if s.is_nan() || s <= 0.0 || w == 0.0 {
        return GradPair::new(0.0, 0.0);
    }
    let x = r / (SMOOTHING_SCALE * s);
    let tanh_x = x.tanh();
    let ratio = if x == 0.0 { 1.0 } else { tanh_x / x };
    let ratio = ratio.max(MIN_SURROGATE_RATIO);
    let grad = 0.5 * s * (tanh_x + 1.0 - 2.0 * alpha);
    let hess = 0.5 / SMOOTHING_SCALE * ratio;
    GradPair::new(w * grad, w * hess)
}

/// Sort order of `labels` under `<`, stable for equal values (XGBoost
/// `StableSort` with `operator<`).
fn stable_order(labels: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..labels.len()).collect();
    order.sort_by(|&l, &r| {
        labels[l]
            .partial_cmp(&labels[r])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
}

/// XGBoost `common::Quantile`: linear interpolation on the `(n + 1)α` grid,
/// clamped to the extremes, over the values already sorted ascending. `NaN`
/// for no values.
fn interpolated_quantile(alpha: f32, sorted: &[f32]) -> f32 {
    let Some((&first, &last)) = sorted.first().zip(sorted.last()) else {
        return f32::NAN;
    };
    let alpha = f64::from(alpha);
    let n = sorted.len() as f64;
    if alpha <= 1.0 / (n + 1.0) {
        return first;
    }
    if alpha >= n / (n + 1.0) {
        return last;
    }
    let x = alpha * (n + 1.0);
    let k = x.floor() - 1.0;
    let d = (x - 1.0) - k;
    let v0 = sorted[k as usize];
    let v1 = sorted[k as usize + 1];
    (f64::from(v0) + d * f64::from(v1 - v0)) as f32
}

/// XGBoost `common::WeightedQuantile`: the first sorted value whose `f32`
/// cumulative weight reaches `α · total` (no interpolation), capped at the
/// largest value. `NaN` for no values.
fn weighted_quantile(alpha: f32, labels: &[f32], weights: &[f32], order: &[usize]) -> f32 {
    if order.is_empty() {
        return f32::NAN;
    }
    let mut cdf = Vec::with_capacity(order.len());
    let mut acc = 0.0f32;
    for (i, &row) in order.iter().enumerate() {
        acc = if i == 0 {
            weights[row]
        } else {
            acc + weights[row]
        };
        cdf.push(acc);
    }
    let thresh = (f64::from(acc) * f64::from(alpha)) as f32;
    let idx = cdf.partition_point(|&c| c < thresh).min(order.len() - 1);
    labels[order[idx]]
}

impl Objective for Quantile {
    fn name(&self) -> &'static str {
        "reg:quantileerror"
    }

    fn n_outputs(&self) -> usize {
        self.alpha.len()
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        let k = self.alpha.len();
        let n = labels.len();
        let scales = residual_scales(preds, weights, k, |i, _| labels[i]);
        let alpha = &self.alpha;
        super::rowwise_gradient(
            n,
            k,
            preds,
            labels,
            weights,
            out,
            |preds, labels, weights, out| {
                for (i, (row, out_row)) in preds
                    .chunks_exact(k)
                    .zip(out.chunks_exact_mut(k))
                    .enumerate()
                {
                    let y = labels[i];
                    let w = weights.map_or(1.0, |ws| ws[i]);
                    for j in 0..k {
                        out_row[j] = quantile_pair(row[j] - y, scales[j], alpha[j], w);
                    }
                }
            },
        );
    }

    /// Insertion-sort each row's outputs ascending (XGBoost's non-crossing
    /// `PredTransform`).
    fn pred_transform(&self, preds: &mut [f32]) {
        for row in preds.chunks_exact_mut(self.alpha.len()) {
            for i in 1..row.len() {
                let value = row[i];
                let mut pos = i;
                while pos > 0 && row[pos - 1] > value {
                    row[pos] = row[pos - 1];
                    pos -= 1;
                }
                row[pos] = value;
            }
        }
    }

    /// The identity: XGBoost's quantile `ProbToMargin` is the identity, so
    /// the stored `base_score` is the margin row itself, in output order,
    /// not the sorted prediction.
    fn margins_to_probs(&self, _margins: &mut [f32]) {}

    /// Every output is fitted to the same single label column.
    fn validate_info(&self, info: &crate::data::MetaInfo) -> Result<()> {
        super::check_label_width(info, 1)
    }

    fn base_margins_info(&self, info: &crate::data::MetaInfo) -> Vec<f32> {
        let (labels, weights) = (info.labels, info.weights);
        let order = stable_order(labels);
        match weights {
            None => {
                let sorted: Vec<f32> = order.iter().map(|&i| labels[i]).collect();
                self.alpha
                    .iter()
                    .map(|&a| interpolated_quantile(a, &sorted))
                    .collect()
            }
            Some(w) => self
                .alpha
                .iter()
                .map(|&a| weighted_quantile(a, labels, w, &order))
                .collect(),
        }
    }

    fn default_metric(&self) -> String {
        "quantile".to_string()
    }
}

/// Expectile regression (`reg:expectileerror`), XGBoost's non-crossing
/// multi-expectile parameterization.
///
/// Margins `u` map to predictions `q₀ = u₀`, `q_k = q_{k−1} + 1e-6 +
/// softplus(u_k)`, so expectiles are strictly increasing. With `d_k = q_k −
/// y` and asymmetric weight `a_k = 1 − α_k` for `d_k ≥ 0` (else `α_k`),
/// output `j` gets the diagonal Gauss–Newton pair `g_j = s_j Σ_{k≥j} w a_k
/// d_k`, `h_j = s_j² Σ_{k≥j} w a_k` with chain factor `s₀ = 1`, `s_j =
/// sigmoid(u_j)`, in `f32`.
///
/// The intercept is one Newton step of each expectile loss from the
/// (weighted) label mean, plus the mean, made non-decreasing by a running
/// maximum, then mapped to margins with the inverse softplus gaps.
#[derive(Debug, Clone)]
pub struct Expectile {
    alpha: Vec<f32>,
}

impl Expectile {
    /// Create for the expectile levels `alpha` (XGBoost `expectile_alpha`).
    ///
    /// # Errors
    ///
    /// `alpha` is empty, has an entry outside `[0, 1]`, or is not ascending.
    pub fn new(alpha: &[f64]) -> Result<Self> {
        Ok(Expectile {
            alpha: validate_alphas("expectile_alpha", alpha)?,
        })
    }
}

/// XGBoost `common::SoftPlus` in `f32`.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// XGBoost `common::SoftPlusInv` in `f32`, clamping its argument to at least
/// `1e-6`.
#[inline]
fn softplus_inv(x: f32) -> f32 {
    let x = x.max(K_RT_EPS_F32);
    x + (-(-x).exp_m1()).ln()
}

/// Asymmetric squared-loss weight of an expectile residual `diff = q − y`.
#[inline]
fn expectile_scale(diff: f32, alpha: f32) -> f32 {
    if diff >= 0.0 { 1.0 - alpha } else { alpha }
}

impl Objective for Expectile {
    fn name(&self) -> &'static str {
        "reg:expectileerror"
    }

    fn n_outputs(&self) -> usize {
        self.alpha.len()
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        let k = self.alpha.len();
        let n = labels.len();
        let alpha = &self.alpha;
        super::rowwise_gradient(
            n,
            k,
            preds,
            labels,
            weights,
            out,
            |preds, labels, weights, out| {
                for (i, (row, out_row)) in preds
                    .chunks_exact(k)
                    .zip(out.chunks_exact_mut(k))
                    .enumerate()
                {
                    let label = labels[i];
                    let w = weights.map_or(1.0, |ws| ws[i]);
                    for j in 0..k {
                        let mut pred = row[0];
                        let mut grad_sum = 0.0f32;
                        let mut hess_sum = 0.0f32;
                        for (kk, &a) in alpha.iter().enumerate() {
                            if kk > 0 {
                                pred += K_RT_EPS_F32 + softplus(row[kk]);
                            }
                            if kk >= j {
                                let diff = pred - label;
                                let scale = expectile_scale(diff, a);
                                grad_sum += scale * diff * w;
                                hess_sum += scale * w;
                            }
                        }
                        let chain = if j == 0 {
                            1.0
                        } else {
                            crate::simd::sigmoid_scalar(row[j])
                        };
                        out_row[j] = GradPair::new(chain * grad_sum, chain * chain * hess_sum);
                    }
                }
            },
        );
    }

    /// Rebuild each row's expectiles from the first margin and the softplus
    /// gaps.
    fn pred_transform(&self, preds: &mut [f32]) {
        for row in preds.chunks_exact_mut(self.alpha.len()) {
            let mut pred = row[0];
            for value in &mut row[1..] {
                pred += K_RT_EPS_F32 + softplus(*value);
                *value = pred;
            }
        }
    }

    /// Inverse of [`Expectile::pred_transform`] on one row: each
    /// later entry becomes the inverse softplus of its gap to the previous
    /// prediction, less `1e-6` (XGBoost `ProbToMargin`).
    fn probs_to_margins(&self, scores: &mut [f32]) {
        for j in (1..scores.len()).rev() {
            let gap = scores[j] - scores[j - 1];
            scores[j] = softplus_inv(gap - K_RT_EPS_F32);
        }
    }

    /// Every output is fitted to the same single label column.
    fn validate_info(&self, info: &crate::data::MetaInfo) -> Result<()> {
        super::check_label_width(info, 1)
    }

    fn base_margins_info(&self, info: &crate::data::MetaInfo) -> Vec<f32> {
        let (labels, weights) = (info.labels, info.weights);
        let k = self.alpha.len();
        let mean = weighted_label_mean(labels, weights);
        let mut gpair = Vec::with_capacity(labels.len() * k);
        for (i, &y) in labels.iter().enumerate() {
            let diff = mean - y;
            let w = weights.map_or(1.0, |ws| ws[i]);
            for &a in &self.alpha {
                let scale = expectile_scale(diff, a);
                gpair.push(GradPair::new(scale * diff * w, scale * w));
            }
        }
        let mut out = fit_stump(&gpair, k);
        for v in &mut out {
            *v += mean;
        }
        for j in 1..k {
            out[j] = out[j].max(out[j - 1]);
        }
        self.probs_to_margins(&mut out);
        out
    }

    fn default_metric(&self) -> String {
        "expectile".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objective::{base_margins, gradient_pairs};
    use crate::training::Trainer;

    #[test]
    fn alpha_lists_are_validated() {
        for bad in [&[][..], &[0.5, 0.2], &[-0.1], &[1.5], &[f64::NAN]] {
            assert!(Quantile::new(bad).is_err(), "{bad:?}");
            assert!(Expectile::new(bad).is_err(), "{bad:?}");
        }
        // Equal neighbours and both endpoints are allowed.
        assert!(Quantile::new(&[0.0, 0.5, 0.5, 1.0]).is_ok());
        assert!(Expectile::new(&[0.0, 1.0]).is_ok());
    }

    /// With 19 zero residuals and one of −4, the scale is `S = (2/20)² =
    /// 0.01`; the zero residual has `x = 0` (curvature ratio 1) and the
    /// outlier `x = −4/(0.04·0.01) = −10⁴`, whose ratio `10⁻⁴` is floored at
    /// `3e-4`.
    #[test]
    fn quantile_gradient_at_zero_and_saturated_residuals() {
        let obj = Quantile::new(&[0.25]).unwrap();
        let mut labels = vec![0.0f32; 20];
        labels[19] = 4.0;
        let out = gradient_pairs(&obj, &[0.0; 20], &labels, None);
        let s = 0.01f32;
        assert_eq!(out[0], GradPair::new(0.5 * s * (1.0 - 0.5), 12.5));
        let tanh = (-4.0f32 / (SMOOTHING_SCALE * s)).tanh();
        assert_eq!(out[19].grad, 0.5 * s * (tanh + 1.0 - 0.5));
        assert_eq!(out[19].hess, 0.5 / SMOOTHING_SCALE * MIN_SURROGATE_RATIO);
        // The saturated gradient approaches the pinball slope −α (times S).
        assert!((out[19].grad + 0.25 * s).abs() < 1e-8);
    }

    /// All residuals zero → `S = 0` → every pair is exactly zero; a zero row
    /// weight zeroes its pair while the others keep the weighted scale.
    #[test]
    fn quantile_gradient_zero_scale_and_zero_weight() {
        let obj = Quantile::new(&[0.5]).unwrap();
        let out = gradient_pairs(&obj, &[1.0, 2.0], &[1.0, 2.0], None);
        assert!(out.iter().all(|p| *p == GradPair::new(0.0, 0.0)));

        let out = gradient_pairs(&obj, &[1.0, 0.0], &[0.0, 0.0], Some(&[0.0, 2.0]));
        assert_eq!(out[0], GradPair::new(0.0, 0.0));
        // Only the zero-weight row has a residual, so S = 0 for everyone.
        assert_eq!(out[1], GradPair::new(0.0, 0.0));

        let out = gradient_pairs(&obj, &[1.0, 1.0], &[0.0, 0.0], Some(&[0.0, 2.0]));
        assert_eq!(out[0], GradPair::new(0.0, 0.0));
        // S = (2·1 / 2)² = 1, x = 25: grad = 2·½·tanh(25), hess = 2·12.5·tanh(25)/25.
        let t = 25.0f32.tanh();
        assert_eq!(
            out[1],
            GradPair::new(2.0 * (0.5 * (t + 1.0 - 1.0)), 2.0 * (12.5 * (t / 25.0)))
        );
    }

    /// Every output uses its own alpha and its own scale.
    #[test]
    fn quantile_outputs_use_their_own_alpha() {
        let obj = Quantile::new(&[0.1, 0.9]).unwrap();
        let out = gradient_pairs(&obj, &[0.0, 0.0], &[0.0], None);
        // r = 0 on both outputs but S = 0 there too: zero pairs.
        assert_eq!(out, vec![GradPair::default(); 2]);
        let out = gradient_pairs(&obj, &[-10.0, 10.0, 10.0, 10.0], &[0.0, 0.0], None);
        // Output 0 residuals {-10, 10}: gradients of opposite sign around the
        // alpha tilt; output 1 residuals {10, 10}: both at the +x saturation.
        let tilt0 = 1.0 - 2.0 * 0.1f32;
        let tilt1 = 1.0 - 2.0 * 0.9f32;
        assert!(out[0].grad < 0.0 && out[2].grad > 0.0);
        assert_eq!(out[1].grad, out[3].grad);
        assert!((out[1].grad - 0.5 * 10.0 * (1.0 + tilt1)).abs() < 1e-4);
        assert!((out[2].grad - 0.5 * 10.0 * (1.0 + tilt0)).abs() < 1e-4);
    }

    #[test]
    fn quantile_transform_sorts_each_row() {
        let obj = Quantile::new(&[0.1, 0.5, 0.9]).unwrap();
        let mut p = [3.0, 1.0, 2.0, 0.0, 5.0, -1.0];
        obj.pred_transform(&mut p);
        assert_eq!(p, [1.0, 2.0, 3.0, -1.0, 0.0, 5.0]);
    }

    /// Unweighted intercepts interpolate on the (n+1)α grid and clamp at the
    /// ends; weighted ones step through the cumulative weights.
    #[test]
    fn quantile_intercepts() {
        let obj = Quantile::new(&[0.1, 0.25, 0.5, 0.9]).unwrap();
        let labels = [4.0f32, 1.0, 3.0, 2.0];
        // n = 4: α ≤ 0.2 → min; 0.25·5 = 1.25 → v0 + 0.25(v1−v0) = 1.25;
        // 0.5·5 = 2.5 → 2.5; α ≥ 0.8 → max.
        assert_eq!(base_margins(&obj, &labels, None), vec![1.0, 1.25, 2.5, 4.0]);
        // Sorted weights 1,1,1,5 (labels 1,2,3,4): cdf 1,2,3,8.
        let w = [5.0f32, 1.0, 1.0, 1.0];
        // thresholds 0.8, 2, 4, 7.2 → first cdf ≥ threshold: 1, 2, 4, 4.
        assert_eq!(
            base_margins(&obj, &labels, Some(&w)),
            vec![1.0, 2.0, 4.0, 4.0]
        );
        assert!(base_margins(&obj, &[], None)[0].is_nan());
    }

    #[test]
    fn expectile_transform_is_monotone_and_inverts() {
        let obj = Expectile::new(&[0.1, 0.5, 0.9]).unwrap();
        let mut p = [2.0, -30.0, 3.0];
        obj.pred_transform(&mut p);
        assert_eq!(p[0], 2.0);
        assert!(p[0] < p[1] && p[1] < p[2]);
        let mut scores = [0.5f32, 1.0, 3.0];
        obj.probs_to_margins(&mut scores);
        obj.pred_transform(&mut scores);
        for (a, b) in scores.iter().zip([0.5f32, 1.0, 3.0]) {
            assert!((a - b).abs() < 1e-5, "{scores:?}");
        }
        // Equal prediction-space scores clamp to the minimal gap.
        let mut tied = [1.0f32, 1.0];
        obj.probs_to_margins(&mut tied);
        assert_eq!(tied[1], softplus_inv(K_RT_EPS_F32));
    }

    /// For one output the pair is the asymmetric squared loss; the first
    /// output of a pair also collects the later output's term, the later one
    /// is scaled by `sigmoid(u)`.
    #[test]
    fn expectile_gradient_formulas() {
        let single = Expectile::new(&[0.2]).unwrap();
        let out = gradient_pairs(&single, &[1.0, -1.0], &[0.0, 0.0], Some(&[2.0, 0.0]));
        assert_eq!(out[0], GradPair::new(0.8 * 1.0 * 2.0, 0.8 * 2.0));
        assert_eq!(out[1], GradPair::new(0.0, 0.0));

        let obj = Expectile::new(&[0.2, 0.8]).unwrap();
        let out = gradient_pairs(&obj, &[0.0, 0.0], &[1.0], None);
        let q1 = K_RT_EPS_F32 + softplus(0.0);
        let (d0, d1) = (-1.0f32, q1 - 1.0);
        let (a0, a1) = (0.2f32, if d1 >= 0.0 { 0.2 } else { 0.8 });
        assert_eq!(out[0], GradPair::new(a0 * d0 + a1 * d1, a0 + a1));
        let s = 0.5f32;
        assert_eq!(out[1], GradPair::new(s * (a1 * d1), s * s * a1));
    }

    #[test]
    fn expectile_intercept_is_newton_step_from_mean_then_running_max() {
        let obj = Expectile::new(&[0.5]).unwrap();
        // α = 0.5 is the mean.
        assert_eq!(base_margins(&obj, &[1.0, 2.0, 6.0], None), vec![3.0]);
        let obj = Expectile::new(&[0.1, 0.9]).unwrap();
        let labels = [0.0f32, 0.0, 0.0, 10.0];
        let mut q = base_margins(&obj, &labels, None);
        obj.pred_transform(&mut q);
        // mean 2.5: α=0.1 → residuals {2.5×3 at weight 0.9, −7.5 at 0.1}:
        // step −(6.75 − 0.75)/(2.7 + 0.1) → 2.5 − 2.142857.
        let low = 2.5 - (6.0f64 / 2.8) as f32;
        let high = 2.5 - ((0.75f64 - 6.75) / (0.3 + 0.9)) as f32;
        assert!(
            (q[0] - low).abs() < 1e-6 && (q[1] - high).abs() < 1e-5,
            "{q:?}"
        );
        // A running max keeps equal alphas' intercepts ordered.
        let tied = Expectile::new(&[0.5, 0.5]).unwrap();
        let mut q = base_margins(&tied, &labels, None);
        tied.pred_transform(&mut q);
        assert!(q[1] >= q[0]);
    }

    /// End to end: three quantile outputs train against one label column,
    /// report the averaged `quantile` metric by default, never cross, and
    /// cover increasing label fractions.
    #[test]
    fn multi_quantile_training_is_calibrated_and_ordered() {
        use crate::config::TrainingParams;
        let n = 400;
        let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        // Deterministic noise with a spread that grows with x.
        let y: Vec<f32> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| v + (1.0 + v) * (((i * 7919) % 1000) as f32 / 1000.0 - 0.5))
            .collect();
        let d = crate::test_support::labeled_dense(&x, n, 1, &y);
        let params = TrainingParams::builder()
            .objective("reg:quantileerror")
            .quantile_alpha(vec![0.1, 0.5, 0.9])
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        let result = Trainer::new(&params, &d, 30)
            .eval(&d, "train")
            .train()
            .unwrap();
        let history = &result.history;
        assert_eq!(history[0].scores[0].1, "quantile");
        assert!(history.last().unwrap().scores[0].2 < history[0].scores[0].2);
        let pred = result.model.predict(&d).unwrap();
        assert_eq!(pred.len(), 3 * n);
        let mut below = [0usize; 3];
        for (row, &yi) in pred.as_chunks::<3>().0.iter().zip(&y) {
            assert!(row[0] <= row[1] && row[1] <= row[2], "{row:?}");
            for (count, &q) in below.iter_mut().zip(row) {
                *count += usize::from(yi <= q);
            }
        }
        let frac = below.map(|c| c as f64 / n as f64);
        for (f, alpha) in frac.iter().zip([0.1, 0.5, 0.9]) {
            assert!((f - alpha).abs() < 0.1, "coverage {frac:?}");
        }
    }

    /// XGBoost stores quantile intercepts as margins in output order: the
    /// export-side mapping is the identity, not the sorting transform.
    #[test]
    fn quantile_intercepts_export_unsorted() {
        let obj = Quantile::new(&[0.1, 0.9]).unwrap();
        let mut stored = [10.0f32, 0.0];
        obj.margins_to_probs(&mut stored);
        assert_eq!(stored, [10.0, 0.0]);
    }

    /// An XGBoost-JSON round trip keeps unsorted per-output intercepts where
    /// they are, so margins and sorted predictions are unchanged.
    #[test]
    fn quantile_xgboost_round_trip_keeps_intercept_order() {
        use crate::config::TrainingParams;
        use crate::model::BoostedModel;
        let n = 32;
        let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let d = crate::test_support::labeled_dense(&x, n, 1, &x);
        let params = TrainingParams::builder()
            .objective("reg:quantileerror")
            .quantile_alpha(vec![0.1, 0.9])
            .max_depth(2)
            .build()
            .unwrap();
        let mut model = crate::training::train(&params, &d, 3).unwrap();
        model.set_base_scores(vec![10.0, 0.0]);
        let restored = BoostedModel::from_xgboost_json(&model.to_xgboost_json().unwrap()).unwrap();
        assert_eq!(restored.base_scores(), [10.0, 0.0]);
        let close = |a: Vec<f32>, b: Vec<f32>| {
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(&b) {
                assert!((x - y).abs() <= 1e-5 * x.abs().max(1.0), "{a:?} vs {b:?}");
            }
        };
        close(
            model.predict_margin(&d).unwrap(),
            restored.predict_margin(&d).unwrap(),
        );
        close(model.predict(&d).unwrap(), restored.predict(&d).unwrap());
    }

    /// Every alpha output fits the one label column: a label matrix handed
    /// to training with a directly constructed objective is refused.
    #[test]
    fn alpha_objectives_require_one_label_column() {
        use crate::config::TrainingParams;
        use crate::data::DMatrix;
        let d = DMatrix::from_dense(&[0.0, 1.0], 2, 1)
            .unwrap()
            .with_label_matrix(&[0.0, 1.0, 1.0, 2.0], 2)
            .unwrap();
        let params = TrainingParams::builder().build().unwrap();
        let objectives: [Box<dyn Objective>; 2] = [
            Box::new(Quantile::new(&[0.1, 0.9]).unwrap()),
            Box::new(Expectile::new(&[0.1, 0.9]).unwrap()),
        ];
        for obj in &objectives {
            assert!(matches!(
                Trainer::new(&params, &d, 1).objective(obj.as_ref()).train(),
                Err(HessboostError::InvalidParameter { name, .. }) if name == "labels"
            ));
        }
    }

    /// A directly constructed built-in objective must match the training
    /// parameters it is saved with, or the trained model could not be loaded:
    /// training refuses the mismatch and accepts the matching alphas.
    #[test]
    fn training_refuses_alphas_the_saved_model_cannot_rebuild() {
        use crate::config::TrainingParams;
        use crate::model::BoostedModel;
        let d =
            crate::test_support::labeled_dense(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let obj = Quantile::new(&[0.1, 0.9]).unwrap();
        for alphas in [vec![], vec![0.5]] {
            let params = TrainingParams::builder()
                .quantile_alpha(alphas)
                .build()
                .unwrap();
            assert!(matches!(
                Trainer::new(&params, &d, 1).objective(&obj).train(),
                Err(HessboostError::InvalidParameter { name, .. }) if name == "objective"
            ));
        }
        let params = TrainingParams::builder()
            .quantile_alpha(vec![0.1, 0.9])
            .build()
            .unwrap();
        let model = Trainer::new(&params, &d, 1)
            .objective(&obj)
            .train()
            .unwrap()
            .model;
        BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
    }
}
