//! Distribution-free prediction intervals by split conformal prediction.
//!
//! Two calibrators wrap already-fitted models:
//!
//! - [`SplitConformal`] turns a point regressor `f` into symmetric intervals
//!   `[f(x) - Q, f(x) + Q]` from absolute-residual scores `|y - f(x)|`.
//! - [`ConformalizedQuantile`] implements conformalized quantile regression
//!   (CQR, Romano, Patterson & Candès, 2019): a lower/upper quantile band
//!   `[q_lo(x), q_hi(x)]` is widened (or shrunk) to `[q_lo(x) - Q, q_hi(x) + Q]`
//!   using the scores `max(q_lo(x) - y, y - q_hi(x))`. The band keeps the
//!   input-dependent width of the quantile models, so intervals adapt to
//!   heteroscedastic noise.
//!
//! Both need a **calibration set**: labelled rows that were *not* used to train
//! the model(s). With `n` calibration scores and miscoverage level `alpha`,
//! `Q` is the `k`-th smallest score, where `k = ceil((n + 1)(1 - alpha))`.
//!
//! # Coverage guarantee
//!
//! If the `n` calibration rows and a test row `(X, Y)` are exchangeable (for
//! example, drawn i.i.d. from the same distribution) and independent of the
//! data the model(s) were trained on, the returned interval `C(X)` satisfies
//!
//! ```text
//! P(Y ∈ C(X)) ≥ 1 - alpha
//! ```
//!
//! and, when the scores are almost surely distinct (no ties),
//! `P(Y ∈ C(X)) ≤ 1 - alpha + 1/(n + 1)`.
//!
//! The probability is **marginal**: it is taken jointly over the calibration
//! sample and the test row. It is *not* conditional coverage: a given input
//! `x` (or a region of inputs) may be covered less often than `1 - alpha`,
//! and for one fixed calibration set the realized coverage fluctuates around
//! the nominal level (it follows a Beta distribution with mean `k / (n + 1)`).
//! No assumption on the model's quality is needed; a poor model yields wide
//! intervals, not invalid ones.
//!
//! When `alpha < 1 / (n + 1)`, `k > n`: the calibration set is too small to
//! certify the requested level, `Q = +∞`, and every interval is
//! `(-∞, +∞)`.
//!
//! # Weights
//!
//! Calibration sets with non-uniform instance weights are rejected. Weighted
//! conformal prediction (Tibshirani et al., 2019) also requires the weight of
//! each test point, which this API does not model; uniform weights are
//! accepted because they do not change the quantile.
//!
//! # Example
//!
//! ```
//! use hessboost::conformal::SplitConformal;
//! use hessboost::prelude::*;
//!
//! # fn main() -> Result<()> {
//! let n = 200;
//! let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
//! let y: Vec<f32> = x.iter().map(|v| 2.0 * v + 0.1 * (37.0 * v).sin()).collect();
//! let (train_rows, cal_rows): (Vec<usize>, Vec<usize>) = (0..n).partition(|i| i % 2 == 0);
//! let all = DMatrix::from_dense(&x, n, 1)?.with_labels(&y)?;
//! let (dtrain, dcal) = (all.select_rows(&train_rows)?, all.select_rows(&cal_rows)?);
//!
//! let params = TrainingParams::builder().objective("reg:squarederror").build()?;
//! let model = train(&params, &dtrain, 20)?;
//!
//! let conformal = SplitConformal::calibrate(&model, &dcal, 0.1)?;
//! let intervals = conformal.predict_interval(&dcal)?;
//! assert!(intervals.iter().all(|(lo, hi)| lo <= hi));
//! # Ok(())
//! # }
//! ```

use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;

/// Split-conformal intervals `[f(x) - Q, f(x) + Q]` around a single-output
/// point model `f`.
///
/// `Q` is the finite-sample-corrected quantile of the absolute residuals
/// `|y_i - f(x_i)|` on the calibration set; see the
/// [module docs](crate::conformal) for the exact coverage guarantee.
#[derive(Debug, Clone)]
pub struct SplitConformal<'a> {
    model: &'a BoostedModel,
    alpha: f64,
    n_calibration: usize,
    half_width: f64,
}

impl<'a> SplitConformal<'a> {
    /// Calibrate `model` on a held-out, labelled `calibration` set for
    /// miscoverage level `alpha` (target coverage `1 - alpha`).
    ///
    /// The model must have a single output (`model.n_outputs() == 1`); scores
    /// are computed on [`BoostedModel::predict`] (the objective's reported
    /// space), honouring the calibration set's `base_margin` if present.
    ///
    /// # Errors
    ///
    /// - [`HessboostError::InvalidParameter`] if `alpha` is not in `(0, 1)`,
    ///   the model has more than one output, the calibration set carries a
    ///   label matrix or non-uniform weights, or a prediction is not finite.
    /// - [`HessboostError::EmptyDataset`] if the calibration set has no rows or
    ///   no labels.
    /// - [`HessboostError::DimensionMismatch`] if the calibration set's feature
    ///   count differs from the model's.
    pub fn calibrate(model: &'a BoostedModel, calibration: &DMatrix, alpha: f64) -> Result<Self> {
        validate_alpha(alpha)?;
        let labels = calibration_labels(calibration)?;
        let preds = single_output_predictions(model, calibration)?;
        let half_width = score_quantile(labels, alpha, |i, y| {
            let p = f64::from(preds[i]);
            sub_round_up(y, p).max(sub_round_up(p, y))
        });
        Ok(SplitConformal {
            model,
            alpha,
            n_calibration: labels.len(),
            half_width,
        })
    }

    /// Prediction intervals `(lower, upper)` for every row of `data`.
    ///
    /// Bounds are computed in `f64` and rounded outward to `f32`, so the
    /// `f32` interval always contains the exact one. When the half-width is
    /// infinite (see [`Self::half_width`]) every interval is `(-∞, +∞)`.
    ///
    /// # Errors
    ///
    /// [`HessboostError::DimensionMismatch`] on a feature-count mismatch and
    /// [`HessboostError::InvalidParameter`] if a prediction is not finite.
    pub fn predict_interval(&self, data: &DMatrix) -> Result<Vec<(f32, f32)>> {
        let preds = single_output_predictions(self.model, data)?;
        Ok(preds
            .iter()
            .map(|&p| widen(p, p, self.half_width))
            .collect())
    }

    /// The calibrated half-width `Q`: the `k`-th smallest absolute residual
    /// with `k = ceil((n + 1)(1 - alpha))`, or `+∞` when `k > n`. Residuals
    /// are computed in `f64` and rounded up, never below the exact value.
    pub fn half_width(&self) -> f64 {
        self.half_width
    }

    /// The miscoverage level `alpha` the calibrator was built for.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Number of calibration rows `n`.
    pub fn n_calibration(&self) -> usize {
        self.n_calibration
    }
}

/// Conformalized quantile regression (CQR): a lower/upper quantile band
/// `[q_lo(x), q_hi(x)]` adjusted to `[q_lo(x) - Q, q_hi(x) + Q]`.
///
/// `Q` is the finite-sample-corrected quantile of the calibration scores
/// `E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))`; see the
/// [module docs](crate::conformal) for the exact coverage guarantee. `Q` is positive when
/// the band under-covers and negative when it over-covers, so CQR both widens
/// a too-narrow band and tightens a too-wide one.
///
/// The quantile models are typically trained with a pinball loss at levels
/// `alpha / 2` and `1 - alpha / 2`, but the guarantee holds for any pair of
/// models. If the adjusted bounds cross (`lower > upper`, possible when
/// `Q < 0` or the models' quantiles cross), the interval is returned as-is and
/// represents the empty set; the coverage guarantee already accounts for it.
#[derive(Debug, Clone)]
pub struct ConformalizedQuantile<'a> {
    band: QuantileBand<'a>,
    alpha: f64,
    n_calibration: usize,
    correction: f64,
}

/// Where the raw quantile band comes from.
#[derive(Debug, Clone, Copy)]
enum QuantileBand<'a> {
    /// Two single-output models.
    Pair {
        lower: &'a BoostedModel,
        upper: &'a BoostedModel,
    },
    /// Two outputs of one multi-output model.
    Outputs {
        model: &'a BoostedModel,
        lower: usize,
        upper: usize,
    },
    /// Two quantiles of the distributions a `dist:*` model predicts.
    Distribution {
        model: &'a BoostedModel,
        lower: f64,
        upper: f64,
    },
}

impl QuantileBand<'_> {
    /// Check that the band can be evaluated on `data` (model kind, feature
    /// count, prediction layout) without requiring finite bounds from a
    /// `dist:*` model. Used when `Q = +∞`, where the band does not affect the
    /// intervals and a tiny `alpha` puts the distribution quantiles at levels
    /// that round to 0 or 1 (infinite bounds).
    fn check(self, data: &DMatrix) -> Result<()> {
        match self {
            QuantileBand::Distribution { model, .. } => model.predict_distribution(data).map(drop),
            QuantileBand::Pair { lower, upper } => {
                single_output_predictions(lower, data)?;
                single_output_predictions(upper, data).map(drop)
            }
            QuantileBand::Outputs { model, .. } => {
                checked_predictions(model, data, model.n_outputs()).map(drop)
            }
        }
    }

    /// The raw `(q_lo, q_hi)` predictions for every row, validated finite.
    fn predict(self, data: &DMatrix) -> Result<Vec<(f32, f32)>> {
        match self {
            QuantileBand::Pair { lower, upper } => {
                let lo = single_output_predictions(lower, data)?;
                let hi = single_output_predictions(upper, data)?;
                Ok(lo.into_iter().zip(hi).collect())
            }
            QuantileBand::Outputs {
                model,
                lower,
                upper,
            } => {
                let k = model.n_outputs();
                let preds = checked_predictions(model, data, k)?;
                Ok(preds
                    .chunks_exact(k)
                    .map(|row| (row[lower], row[upper]))
                    .collect())
            }
            QuantileBand::Distribution {
                model,
                lower,
                upper,
            } => {
                let band: Vec<(f32, f32)> = model
                    .predict_distribution(data)?
                    .iter()
                    .map(|d| (round_down(d.quantile(lower)), round_up(d.quantile(upper))))
                    .collect();
                check_finite(band.iter().flat_map(|&(lo, hi)| [lo, hi]))?;
                Ok(band)
            }
        }
    }
}

impl<'a> ConformalizedQuantile<'a> {
    /// Calibrate a band given by two single-output quantile models, `lower`
    /// (e.g. trained at quantile `alpha / 2`) and `upper` (at `1 - alpha / 2`).
    ///
    /// # Errors
    ///
    /// - [`HessboostError::InvalidParameter`] if `alpha` is not in `(0, 1)`,
    ///   either model has more than one output, the calibration set carries
    ///   a label matrix or non-uniform weights, or a prediction is not finite.
    /// - [`HessboostError::EmptyDataset`] if the calibration set has no rows or
    ///   no labels.
    /// - [`HessboostError::DimensionMismatch`] if the two models expect
    ///   different feature counts or the calibration set does not match them.
    pub fn calibrate(
        lower: &'a BoostedModel,
        upper: &'a BoostedModel,
        calibration: &DMatrix,
        alpha: f64,
    ) -> Result<Self> {
        if lower.n_features() != upper.n_features() {
            return Err(HessboostError::dimension_mismatch(
                "upper quantile model feature count",
                lower.n_features(),
                upper.n_features(),
            ));
        }
        Self::calibrate_band(QuantileBand::Pair { lower, upper }, calibration, alpha)
    }

    /// Calibrate a band given by two outputs of one multi-output model, e.g.
    /// a quantile model trained at levels `[alpha / 2, 1 - alpha / 2]`.
    /// `lower_output` and `upper_output` index the model's outputs.
    ///
    /// # Errors
    ///
    /// As [`Self::calibrate`], plus [`HessboostError::InvalidParameter`] if an
    /// output index is out of range, the two indices are equal, or the model's
    /// predictions are not laid out `[row][output]` (e.g. `multi:softmax`).
    pub fn calibrate_outputs(
        model: &'a BoostedModel,
        lower_output: usize,
        upper_output: usize,
        calibration: &DMatrix,
        alpha: f64,
    ) -> Result<Self> {
        let k = model.n_outputs();
        for (name, index) in [
            ("lower_output", lower_output),
            ("upper_output", upper_output),
        ] {
            if index >= k {
                return Err(HessboostError::invalid_param(
                    name,
                    format!("output index {index} is out of range for a model with {k} outputs"),
                ));
            }
        }
        if lower_output == upper_output {
            return Err(HessboostError::invalid_param(
                "upper_output",
                "must differ from lower_output",
            ));
        }
        let band = QuantileBand::Outputs {
            model,
            lower: lower_output,
            upper: upper_output,
        };
        Self::calibrate_band(band, calibration, alpha)
    }

    /// Calibrate the central band of the distributions a `dist:*` model
    /// predicts ([`BoostedModel::predict_distribution`]): its
    /// `alpha / 2` and `1 - alpha / 2` quantiles, rounded outward to `f32`.
    ///
    /// The predicted distribution's own interval covers `1 - alpha` only if
    /// the model is well specified and fitted; CQR restores the finite-sample
    /// marginal guarantee regardless, widening the band when it under-covers
    /// and tightening it when it over-covers, while keeping the per-row
    /// widths the distribution learned.
    ///
    /// # Errors
    ///
    /// As [`Self::calibrate`], plus [`HessboostError::InvalidParameter`] if
    /// the model's objective is not a `dist:*` objective or a predicted
    /// quantile is not finite. Quantiles are not evaluated when `k > n`
    /// (`Q = +∞`), so a tiny `alpha` yields `(-∞, +∞)` intervals rather than
    /// an error.
    pub fn calibrate_distribution(
        model: &'a BoostedModel,
        calibration: &DMatrix,
        alpha: f64,
    ) -> Result<Self> {
        let band = QuantileBand::Distribution {
            model,
            lower: 0.5 * alpha,
            upper: 1.0 - 0.5 * alpha,
        };
        Self::calibrate_band(band, calibration, alpha)
    }

    fn calibrate_band(band: QuantileBand<'a>, calibration: &DMatrix, alpha: f64) -> Result<Self> {
        validate_alpha(alpha)?;
        let labels = calibration_labels(calibration)?;
        let correction = if conformal_rank(labels.len(), alpha).is_none() {
            band.check(calibration)?;
            f64::INFINITY
        } else {
            let raw = band.predict(calibration)?;
            score_quantile(labels, alpha, |i, y| {
                let (lo, hi) = raw[i];
                sub_round_up(f64::from(lo), y).max(sub_round_up(y, f64::from(hi)))
            })
        };
        Ok(ConformalizedQuantile {
            band,
            alpha,
            n_calibration: labels.len(),
            correction,
        })
    }

    /// Prediction intervals `(q_lo(x) - Q, q_hi(x) + Q)` for every row of
    /// `data`.
    ///
    /// Bounds are computed in `f64` and rounded outward to `f32`. When `Q` is
    /// infinite (see [`Self::correction`]) every interval is `(-∞, +∞)`.
    ///
    /// # Errors
    ///
    /// [`HessboostError::DimensionMismatch`] on a feature-count mismatch and
    /// [`HessboostError::InvalidParameter`] if a prediction is not finite.
    pub fn predict_interval(&self, data: &DMatrix) -> Result<Vec<(f32, f32)>> {
        if self.correction == f64::INFINITY {
            self.band.check(data)?;
            return Ok(vec![(f32::NEG_INFINITY, f32::INFINITY); data.n_rows()]);
        }
        Ok(self
            .band
            .predict(data)?
            .into_iter()
            .map(|(lo, hi)| widen(lo, hi, self.correction))
            .collect())
    }

    /// The calibrated correction `Q`: the `k`-th smallest CQR score with
    /// `k = ceil((n + 1)(1 - alpha))`, or `+∞` when `k > n`. Negative when
    /// the raw band over-covers the calibration set. Score differences are
    /// computed in `f64` and rounded up, never below the exact value.
    pub fn correction(&self) -> f64 {
        self.correction
    }

    /// The miscoverage level `alpha` the calibrator was built for.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Number of calibration rows `n`.
    pub fn n_calibration(&self) -> usize {
        self.n_calibration
    }
}

fn validate_alpha(alpha: f64) -> Result<()> {
    if alpha > 0.0 && alpha < 1.0 {
        Ok(())
    } else {
        Err(HessboostError::invalid_param(
            "alpha",
            format!("miscoverage level must be in (0, 1), got {alpha}"),
        ))
    }
}

/// The calibration labels, after checking the set is non-empty, labelled
/// with one label column, and unweighted (or uniformly weighted).
fn calibration_labels(calibration: &DMatrix) -> Result<&[f32]> {
    if calibration.n_rows() == 0 {
        return Err(HessboostError::EmptyDataset(
            "conformal calibration set has no rows",
        ));
    }
    let labels = calibration.labels().ok_or(HessboostError::EmptyDataset(
        "conformal calibration set has no labels",
    ))?;
    if calibration.n_targets() != 1 {
        return Err(HessboostError::invalid_param(
            "labels",
            format!(
                "conformal calibration needs one label per row, got a {}-column label matrix",
                calibration.n_targets()
            ),
        ));
    }
    if let Some(weights) = calibration.weights()
        && weights.iter().any(|&w| w != weights[0])
    {
        return Err(HessboostError::invalid_param(
            "weights",
            "conformal calibration does not support non-uniform instance weights",
        ));
    }
    Ok(labels)
}

/// `model.predict(data)` for a single-output model, validated finite.
fn single_output_predictions(model: &BoostedModel, data: &DMatrix) -> Result<Vec<f32>> {
    if model.n_outputs() != 1 {
        return Err(HessboostError::invalid_param(
            "model",
            format!(
                "expected a single-output model, got {} outputs",
                model.n_outputs()
            ),
        ));
    }
    checked_predictions(model, data, 1)
}

/// `model.predict(data)`, validated as `width` finite values per row.
fn checked_predictions(model: &BoostedModel, data: &DMatrix, width: usize) -> Result<Vec<f32>> {
    let preds = model.predict(data)?;
    let expected = data.n_rows() * width;
    if preds.len() != expected {
        return Err(HessboostError::invalid_param(
            "model",
            format!(
                "predictions must be laid out [row][output] ({expected} values), got {} values",
                preds.len()
            ),
        ));
    }
    check_finite(preds.iter().copied())?;
    Ok(preds)
}

/// Rejects the first non-finite value, by its index in `preds`.
fn check_finite(preds: impl IntoIterator<Item = f32>) -> Result<()> {
    match preds.into_iter().enumerate().find(|(_, p)| !p.is_finite()) {
        None => Ok(()),
        Some((i, p)) => Err(HessboostError::invalid_param(
            "model",
            format!("prediction {i} is not finite ({p})"),
        )),
    }
}

/// The rank `k = ceil((n + 1)(1 - alpha))` of the conformal quantile among
/// `n` scores, or `None` when `k > n` (the quantile is `+∞`).
///
/// `k` is evaluated as `(n + 1) - floor((n + 1) * alpha)`, which equals the
/// ceiling form exactly and avoids the rounding of `1 - alpha`.
fn conformal_rank(n: usize, alpha: f64) -> Option<usize> {
    let n1 = n + 1;
    // `0 < alpha < 1` bounds the floor to `[0, n]`, so `1 <= k <= n + 1`.
    let k = n1 - ((n1 as f64) * alpha).floor() as usize;
    (k <= n).then_some(k)
}

/// [`conformal_quantile`] of the calibration scores `score(i, y_i)` over
/// the `labels` `y_i`, in row order.
fn score_quantile(labels: &[f32], alpha: f64, score: impl Fn(usize, f64) -> f64) -> f64 {
    let mut scores: Vec<f64> = labels
        .iter()
        .enumerate()
        .map(|(i, &y)| score(i, f64::from(y)))
        .collect();
    conformal_quantile(&mut scores, alpha)
}

/// The `k`-th smallest score with `k` from [`conformal_rank`], or `+∞` when
/// `k > n`. Reorders `scores`; `scores` must be non-empty and finite.
fn conformal_quantile(scores: &mut [f64], alpha: f64) -> f64 {
    let Some(k) = conformal_rank(scores.len(), alpha) else {
        return f64::INFINITY;
    };
    let (_, kth, _) = scores.select_nth_unstable_by(k - 1, f64::total_cmp);
    *kth
}

/// An upper bound on the exact difference `a - b`: the rounded difference,
/// raised by one ulp when rounding went down. Scores built from it are never
/// below the exact ones, so an interval widened by their quantile contains
/// every label whose rounded score is within it (`f64` subtraction of two
/// `f32` values is not always exact, e.g. `1 - (-2^-80)`).
///
/// `a` and `b` must be finite and small enough for `a - b` not to overflow
/// (true for values converted from `f32`), which makes the 2Sum error term
/// exact.
fn sub_round_up(a: f64, b: f64) -> f64 {
    let s = a - b;
    let bb = s - a;
    // Exact `a - b - s` (Knuth's 2Sum on `a + (-b)`).
    let err = (a - (s - bb)) + (-b - bb);
    if err > 0.0 { s.next_up() } else { s }
}

/// `(lo - q, hi + q)` in `f64`, rounded outward to `f32`.
fn widen(lo: f32, hi: f32, q: f64) -> (f32, f32) {
    (round_down(f64::from(lo) - q), round_up(f64::from(hi) + q))
}

/// The largest `f32` not above `x`.
fn round_down(x: f64) -> f32 {
    let r = x as f32;
    if f64::from(r) > x { r.next_down() } else { r }
}

/// The smallest `f32` not below `x`.
fn round_up(x: f64) -> f32 {
    let r = x as f32;
    if f64::from(r) < x { r.next_up() } else { r }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainingParams;
    use crate::objective::{CustomObjective, GradPair};
    use crate::rng::Rng;
    use crate::test_support::labeled_dense;
    use crate::training::{Trainer, train};

    const N_FEATURES: usize = 2;

    /// Heteroscedastic regression: `y = sin(2π x0) + (0.1 + x1) · ε`,
    /// `x ~ U(0, 1)²`, `ε ~ N(0, 1)`.
    fn hetero(n: usize, rng: &mut Rng) -> DMatrix {
        let mut x = Vec::with_capacity(n * N_FEATURES);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let (x0, x1) = (rng.f32(), rng.f32());
            // Box–Muller.
            let u1 = 1.0 - rng.f64();
            let u2 = rng.f64();
            let eps = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            x.extend_from_slice(&[x0, x1]);
            y.push((std::f32::consts::TAU * x0).sin() + (0.1 + x1) * eps as f32);
        }
        labeled_dense(&x, n, N_FEATURES, &y)
    }

    fn with_shifted_labels(d: &DMatrix, shift: f32) -> DMatrix {
        let y: Vec<f32> = d.labels().unwrap().iter().map(|v| v + shift).collect();
        d.clone().with_labels(&y).unwrap()
    }

    fn point_model(d: &DMatrix) -> BoostedModel {
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        train(&params, d, 20).unwrap()
    }

    /// Two-output pinball-loss model at quantile levels `taus`.
    fn quantile_model(d: &DMatrix, taus: [f32; 2]) -> BoostedModel {
        let obj = CustomObjective::new("test:quantile", 2, 0.0, "mae", move |p, y, _w, out| {
            for (i, &yi) in y.iter().enumerate() {
                for (j, tau) in taus.iter().enumerate() {
                    let g = if p[2 * i + j] > yi { 1.0 - tau } else { -tau };
                    out[2 * i + j] = GradPair::new(g, 1.0);
                }
            }
        });
        let params = TrainingParams::builder()
            .max_depth(4)
            .eta(0.3)
            .build()
            .unwrap();
        Trainer::new(&params, d, 150)
            .objective(&obj)
            .train()
            .unwrap()
            .model
    }

    fn coverage(intervals: &[(f32, f32)], d: &DMatrix) -> f64 {
        let covered = intervals
            .iter()
            .zip(d.labels().unwrap())
            .filter(|&(&(lo, hi), &y)| lo <= y && y <= hi)
            .count();
        covered as f64 / intervals.len() as f64
    }

    const ALPHA: f64 = 0.1;
    // n + 1 = 100 so that k/(n+1) = 1 - alpha exactly and the no-ties upper
    // bound is 1 - alpha + 0.01.
    const N_CAL: usize = 99;
    const N_TEST: usize = 200;
    const TRIALS: u64 = 400;
    // Per-trial coverage sd ≈ sqrt(0.09/101 + 0.09/200) ≈ 0.037, so the mean
    // over 400 trials has sd ≈ 0.0018; 0.0065 is ~3.5 sd.
    const SLACK: f64 = 0.0065;

    /// Mean test coverage over `TRIALS` independent calibration/test draws,
    /// using `calibrate` to build the interval predictor.
    fn mean_coverage(
        seed: u64,
        mut intervals: impl FnMut(&DMatrix, &DMatrix) -> Vec<(f32, f32)>,
    ) -> f64 {
        let mut rng = Rng::new(seed);
        let total: f64 = (0..TRIALS)
            .map(|_| {
                let cal = hetero(N_CAL, &mut rng);
                let test = hetero(N_TEST, &mut rng);
                coverage(&intervals(&cal, &test), &test)
            })
            .sum();
        total / TRIALS as f64
    }

    fn assert_nominal(mean: f64, what: &str) {
        let lo = 1.0 - ALPHA - SLACK;
        let hi = 1.0 - ALPHA + 1.0 / (N_CAL + 1) as f64 + SLACK;
        assert!(
            (lo..=hi).contains(&mean),
            "{what}: mean coverage {mean:.4} outside [{lo:.4}, {hi:.4}]"
        );
    }

    #[test]
    fn split_conformal_attains_marginal_coverage() {
        let train_set = hetero(2000, &mut Rng::new(1));
        let model = point_model(&train_set);
        let mean = mean_coverage(2, |cal, test| {
            SplitConformal::calibrate(&model, cal, ALPHA)
                .unwrap()
                .predict_interval(test)
                .unwrap()
        });
        assert_nominal(mean, "split conformal");
    }

    #[test]
    fn cqr_widens_a_too_narrow_band_to_nominal_coverage() {
        // Point fits on y ∓ 0.05: a band of width ≈ 0.1 against noise with
        // standard deviation 0.1–1.1.
        let train_set = hetero(2000, &mut Rng::new(3));
        let lower = point_model(&with_shifted_labels(&train_set, -0.05));
        let upper = point_model(&with_shifted_labels(&train_set, 0.05));

        let mut rng = Rng::new(4);
        let (cal, test) = (hetero(N_CAL, &mut rng), hetero(1000, &mut rng));
        let raw = ConformalizedQuantile::calibrate(&lower, &upper, &cal, ALPHA)
            .unwrap()
            .band
            .predict(&test)
            .unwrap();
        assert!(coverage(&raw, &test) < 0.5, "raw band should under-cover");

        let mean = mean_coverage(5, |cal, test| {
            let cqr = ConformalizedQuantile::calibrate(&lower, &upper, cal, ALPHA).unwrap();
            assert!(cqr.correction() > 0.0);
            cqr.predict_interval(test).unwrap()
        });
        assert_nominal(mean, "CQR on a too-narrow band");
    }

    #[test]
    fn cqr_tightens_a_too_wide_band_without_over_covering() {
        let train_set = hetero(2000, &mut Rng::new(6));
        let lower = point_model(&with_shifted_labels(&train_set, -5.0));
        let upper = point_model(&with_shifted_labels(&train_set, 5.0));
        let mean = mean_coverage(7, |cal, test| {
            let cqr = ConformalizedQuantile::calibrate(&lower, &upper, cal, ALPHA).unwrap();
            assert!(cqr.correction() < 0.0);
            cqr.predict_interval(test).unwrap()
        });
        assert_nominal(mean, "CQR on a too-wide band");
    }

    #[test]
    fn cqr_on_multi_output_quantile_model_is_adaptive() {
        let train_set = hetero(2000, &mut Rng::new(8));
        let model = quantile_model(&train_set, [0.05, 0.95]);
        let mean = mean_coverage(9, |cal, test| {
            ConformalizedQuantile::calibrate_outputs(&model, 0, 1, cal, ALPHA)
                .unwrap()
                .predict_interval(test)
                .unwrap()
        });
        assert_nominal(mean, "CQR on a pinball-loss model");

        // The band follows the noise scale (0.1 + x1): rows with x1 = 0.9 get
        // much wider intervals than rows with x1 = 0.1, unlike split conformal.
        let cal = hetero(500, &mut Rng::new(10));
        let cqr = ConformalizedQuantile::calibrate_outputs(&model, 0, 1, &cal, ALPHA).unwrap();
        let probe = DMatrix::from_dense(&[0.25, 0.1, 0.25, 0.9], 2, N_FEATURES).unwrap();
        let widths: Vec<f32> = cqr
            .predict_interval(&probe)
            .unwrap()
            .iter()
            .map(|(lo, hi)| hi - lo)
            .collect();
        assert!(
            widths[1] > 2.0 * widths[0],
            "widths {widths:?} should grow with the noise scale"
        );
    }

    #[test]
    fn conformal_quantile_uses_the_finite_sample_rank() {
        // n = 9 distinct scores 1..=9 (shuffled): k = ceil(10 (1 - alpha)).
        let base = [5.0, 2.0, 9.0, 1.0, 7.0, 3.0, 8.0, 4.0, 6.0];
        let q = |alpha| conformal_quantile(&mut base.clone(), alpha);
        assert_eq!(q(0.5), 5.0); // k = 5
        assert_eq!(q(0.2), 8.0); // k = 8
        assert_eq!(q(0.15), 9.0); // k = ceil(8.5) = 9
        assert_eq!(q(0.1), 9.0); // alpha = 1/(n+1): k = 9 = n, still finite
        assert_eq!(q(0.099), f64::INFINITY); // k = 10 > n
        assert_eq!(q(0.95), 1.0); // k = 1
        assert_eq!(q(0.999), 1.0); // floor(9.99) = 9: k = 1
    }

    #[test]
    fn too_small_calibration_set_gives_infinite_intervals() {
        let train_set = hetero(300, &mut Rng::new(11));
        let model = point_model(&train_set);
        let cal = hetero(9, &mut Rng::new(12));
        let test = hetero(5, &mut Rng::new(13));

        let finite = SplitConformal::calibrate(&model, &cal, 0.1).unwrap();
        assert!(finite.half_width().is_finite());

        let sc = SplitConformal::calibrate(&model, &cal, 0.05).unwrap();
        assert_eq!(sc.half_width(), f64::INFINITY);
        assert_eq!(sc.n_calibration(), 9);
        let intervals = sc.predict_interval(&test).unwrap();
        assert!(
            intervals
                .iter()
                .all(|&iv| iv == (f32::NEG_INFINITY, f32::INFINITY))
        );

        let cqr = ConformalizedQuantile::calibrate(&model, &model, &cal, 0.05).unwrap();
        assert_eq!(cqr.correction(), f64::INFINITY);
        assert!(
            cqr.predict_interval(&test)
                .unwrap()
                .iter()
                .all(|&iv| iv == (f32::NEG_INFINITY, f32::INFINITY))
        );
    }

    #[test]
    fn calibration_rows_within_the_quantile_are_covered_exactly() {
        // Outward rounding: every calibration row whose score is <= Q lies in
        // its own f32 interval, including the row that attains Q.
        let train_set = hetero(300, &mut Rng::new(14));
        let model = point_model(&train_set);
        let cal = hetero(N_CAL, &mut Rng::new(15));
        let sc = SplitConformal::calibrate(&model, &cal, ALPHA).unwrap();
        let intervals = sc.predict_interval(&cal).unwrap();
        let k = N_CAL + 1 - ((N_CAL + 1) as f64 * ALPHA).floor() as usize;
        assert_eq!(
            (coverage(&intervals, &cal) * N_CAL as f64).round() as usize,
            k
        );
    }

    #[test]
    fn rounding_is_outward() {
        let x = 0.1f64; // not representable in f32
        assert!(f64::from(round_down(x)) <= x && f64::from(round_up(x)) >= x);
        assert!(round_down(x) < round_up(x));
        assert_eq!(round_down(0.5), 0.5);
        assert_eq!(round_up(0.5), 0.5);
        assert_eq!(round_up(1e300), f32::INFINITY);
        assert_eq!(round_down(1e300), f32::MAX);
        assert_eq!(
            widen(1.0, 2.0, f64::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY)
        );
    }

    #[test]
    fn scores_are_rounded_up_so_the_quantile_row_stays_covered() {
        // `1 - (-2^-80)` rounds to 1 in f64: with Q = 1 the interval [0, 2]
        // excluded the only calibration label, whose score attains Q.
        let y = -(2f32.powi(-80));
        let cal = labeled_dense(&[0.0], 1, 1, &[y]);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .base_score(1.0)
            .build()
            .unwrap();
        let model = train(&params, &cal, 0).unwrap();
        assert_eq!(model.predict(&cal).unwrap(), [1.0]);
        let covers = |(lo, hi): (f32, f32)| lo <= y && y <= hi;

        let sc = SplitConformal::calibrate(&model, &cal, 0.5).unwrap();
        assert!(sc.half_width() > 1.0);
        assert!(covers(sc.predict_interval(&cal).unwrap()[0]));
        let cqr = ConformalizedQuantile::calibrate(&model, &model, &cal, 0.5).unwrap();
        assert!(cqr.correction() > 1.0);
        assert!(covers(cqr.predict_interval(&cal).unwrap()[0]));

        // Only a difference that rounded down is raised; exact ones are kept.
        let tiny = f64::from(y);
        assert_eq!(sub_round_up(1.0, tiny), 1.0f64.next_up());
        assert_eq!(sub_round_up(tiny, 1.0), -1.0);
        assert_eq!(sub_round_up(3.0, 0.5), 2.5);
    }

    #[test]
    fn tiny_alpha_on_a_distribution_band_gives_infinite_intervals() {
        // `1 - alpha / 2` rounds to 1 for alpha = 1e-20, where the normal
        // quantile is +∞; with 9 rows k = 10 > n, so the documented result is
        // an unbounded interval, not a non-finite-prediction error.
        let train_set = hetero(300, &mut Rng::new(17));
        let params = TrainingParams::builder()
            .objective("dist:normal")
            .max_depth(2)
            .build()
            .unwrap();
        let model = train(&params, &train_set, 3).unwrap();
        let cal = hetero(9, &mut Rng::new(18));
        let test = hetero(4, &mut Rng::new(19));

        let cqr = ConformalizedQuantile::calibrate_distribution(&model, &cal, 1e-20).unwrap();
        assert_eq!(cqr.correction(), f64::INFINITY);
        assert_eq!(
            cqr.predict_interval(&test).unwrap(),
            vec![(f32::NEG_INFINITY, f32::INFINITY); 4]
        );
        // The model and data are still validated without evaluating the band.
        let point = point_model(&train_set);
        assert!(ConformalizedQuantile::calibrate_distribution(&point, &cal, 1e-20).is_err());
        let wide = DMatrix::from_dense(&[0.0; 3], 1, 3).unwrap();
        assert!(matches!(
            cqr.predict_interval(&wide),
            Err(HessboostError::DimensionMismatch { .. })
        ));
    }

    fn assert_invalid<T: std::fmt::Debug>(r: Result<T>, param: &str) {
        match r {
            Err(HessboostError::InvalidParameter { name, .. }) => assert_eq!(name, param),
            other => panic!("expected InvalidParameter `{param}`, got {other:?}"),
        }
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let mut rng = Rng::new(16);
        let train_set = hetero(200, &mut rng);
        let model = point_model(&train_set);
        let cal = hetero(50, &mut rng);

        for alpha in [0.0, 1.0, -0.1, 1.5, f64::NAN, f64::INFINITY] {
            assert_invalid(SplitConformal::calibrate(&model, &cal, alpha), "alpha");
            assert_invalid(
                ConformalizedQuantile::calibrate(&model, &model, &cal, alpha),
                "alpha",
            );
        }

        // No labels.
        let unlabeled = DMatrix::from_dense(&[0.1, 0.2], 1, N_FEATURES).unwrap();
        assert!(matches!(
            SplitConformal::calibrate(&model, &unlabeled, ALPHA),
            Err(HessboostError::EmptyDataset(_))
        ));

        // A label matrix: scores would pair each row's prediction with the
        // flattened cells, i.e. with other rows' and targets' labels.
        let y = cal.labels().unwrap();
        let two_targets = y.iter().flat_map(|&v| [v, v + 1.0]).collect::<Vec<_>>();
        let matrix = DMatrix::from_dense(
            &vec![0.5; cal.n_rows() * N_FEATURES],
            cal.n_rows(),
            N_FEATURES,
        )
        .unwrap()
        .with_label_matrix(&two_targets, 2)
        .unwrap();
        assert_invalid(SplitConformal::calibrate(&model, &matrix, ALPHA), "labels");
        assert_invalid(
            ConformalizedQuantile::calibrate(&model, &model, &matrix, ALPHA),
            "labels",
        );
        let multi = quantile_model(&train_set, [0.1, 0.9]);
        assert_invalid(
            ConformalizedQuantile::calibrate_outputs(&multi, 0, 1, &matrix, ALPHA),
            "labels",
        );

        // Non-uniform weights are rejected; uniform weights are equivalent to none.
        let ones = vec![2.0; cal.n_rows()];
        let mut uneven = ones.clone();
        uneven[3] = 1.0;
        assert_invalid(
            SplitConformal::calibrate(&model, &cal.clone().with_weights(&uneven).unwrap(), ALPHA),
            "weights",
        );
        let uniform =
            SplitConformal::calibrate(&model, &cal.clone().with_weights(&ones).unwrap(), ALPHA)
                .unwrap();
        let plain = SplitConformal::calibrate(&model, &cal, ALPHA).unwrap();
        assert_eq!(uniform.half_width(), plain.half_width());

        // Feature-count mismatch at calibration and prediction time.
        let wide = labeled_dense(&[0.0; 3], 1, 3, &[0.0]);
        assert!(matches!(
            SplitConformal::calibrate(&model, &wide, ALPHA),
            Err(HessboostError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            plain.predict_interval(&wide),
            Err(HessboostError::DimensionMismatch { .. })
        ));

        // Quantile models with different feature counts.
        let one_feature = labeled_dense(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 2.0]);
        let narrow_model = point_model(&one_feature);
        assert!(matches!(
            ConformalizedQuantile::calibrate(&model, &narrow_model, &cal, ALPHA),
            Err(HessboostError::DimensionMismatch { .. })
        ));

        // Output selection on multi-output models.
        assert_invalid(SplitConformal::calibrate(&multi, &cal, ALPHA), "model");
        assert_invalid(
            ConformalizedQuantile::calibrate(&multi, &model, &cal, ALPHA),
            "model",
        );
        assert_invalid(
            ConformalizedQuantile::calibrate_outputs(&multi, 0, 2, &cal, ALPHA),
            "upper_output",
        );
        assert_invalid(
            ConformalizedQuantile::calibrate_outputs(&multi, 5, 1, &cal, ALPHA),
            "lower_output",
        );
        assert_invalid(
            ConformalizedQuantile::calibrate_outputs(&multi, 1, 1, &cal, ALPHA),
            "upper_output",
        );
        assert_invalid(
            ConformalizedQuantile::calibrate_outputs(&model, 0, 0, &cal, ALPHA),
            "upper_output",
        );

        // Non-finite predictions: a base margin at f32::MAX plus large positive
        // leaves overflows to +inf.
        let exploding = CustomObjective::new("test:explode", 1, 0.0, "rmse", |p, _y, _w, out| {
            for g in out.iter_mut().take(p.len()) {
                *g = GradPair::new(-1e36, 1.0);
            }
        });
        let params = TrainingParams::builder().max_depth(1).build().unwrap();
        let exploding = Trainer::new(&params, &train_set, 1)
            .objective(&exploding)
            .train()
            .unwrap()
            .model;
        let at_max = |d: DMatrix| {
            let n = d.n_rows();
            d.with_base_margin(&vec![f32::MAX; n]).unwrap()
        };
        assert_invalid(
            SplitConformal::calibrate(&exploding, &at_max(cal.clone()), ALPHA),
            "model",
        );
        let finite = SplitConformal::calibrate(&exploding, &cal, ALPHA).unwrap();
        assert_invalid(
            finite.predict_interval(&at_max(hetero(10, &mut rng))),
            "model",
        );
    }

    #[test]
    fn multiclass_softmax_outputs_are_rejected() {
        let n = 60;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
        let d = labeled_dense(&x, n, 1, &y);
        let params = TrainingParams::builder()
            .objective("multi:softmax")
            .num_class(3)
            .build()
            .unwrap();
        let model = train(&params, &d, 2).unwrap();
        // One class index per row, not `[row][output]`.
        assert_invalid(
            ConformalizedQuantile::calibrate_outputs(&model, 0, 2, &d, ALPHA),
            "model",
        );
    }
}
