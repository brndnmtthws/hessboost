//! Online calibration of nonconformity scores, independent of the forecaster.
//!
//! [`OnlineCalibrator`] tracks a score's `1 - alpha` quantile by fixed-step
//! online gradient descent on pinball loss. With no features this is scalar
//! quantile tracking. With lagged scores or caller covariates it is linear
//! quantile tracking, following [Areces et al., ICML 2025](https://proceedings.mlr.press/v267/areces25a.html).
//! Non-bias coefficients are projected onto an L2 ball after each update.
//!
//! A prediction returns a [`Forecast`] receipt containing its threshold and
//! the features available when it was issued. Pass that same receipt to
//! [`OnlineCalibrator::update`] when its score becomes observable. Receipts
//! must be updated **once, in issue order, on their creating calibrator**.
//! Delayed feedback uses the issued threshold and features, not a prediction
//! recomputed with newer state. The caller schedules labels and retains
//! outstanding receipts; there is no internal queue, clock, or model owner.
//! If a receipt is lost, the remaining sequence cannot be updated: start a
//! new calibrator rather than silently omit feedback.
//!
//! # Scores and intervals
//!
//! A threshold `q` defines the set `{y: score(x, y) <= q}`. For a point
//! forecast `f(x)`, absolute-residual scores give `[f(x) - q, f(x) + q]`.
//! For a quantile band `[lo, hi]`, scores `max(lo - y, y - hi)` give
//! `[lo - q, hi + q]`. The same construction works with quantiles from a
//! distributional model. Preserve the original point forecast or band until
//! the label arrives, even if the model has since been retrained.
//!
//! Thresholds may be negative: a negative absolute-residual threshold means
//! the empty set; a negative CQR correction shrinks the band. Do not clamp
//! thresholds without changing the calibration procedure. Standardize
//! covariates using past data only. Lag features are the most recently
//! **observed** scores, newest first, initially zero, followed by the caller's
//! covariates. All scores and covariates must be finite.
//!
//! # Guarantees and limits
//!
//! This is **not** split conformal prediction: the exchangeability-based
//! finite-sample guarantee of [`super::SplitConformal`] does not apply.
//! With immediate feedback, bounded scores and covariates, and a fixed
//! positive step, the projected tracker has the paper's mathematical
//! long-run average coverage property (ignoring floating-point rounding).
//! It does not guarantee conditional coverage, coverage in every window,
//! or coverage on a selected subset. A fixed step does not provide the
//! paper's decaying-step conditional convergence result. The delayed-feedback
//! API prevents using future labels, but makes **no general delayed-feedback
//! coverage claim**. Validate calibration at the actual deployment delays.
//!
//! State and receipts are process-local, not serialized. Calls never enter
//! Rayon. Updates allocate nothing; issuing a linear forecast owns one
//! feature snapshot so several forecasts can remain outstanding.
//!
//! ```
//! use hessboost::conformal::online::{OnlineCalibrator, OnlineParams};
//!
//! # fn main() -> hessboost::error::Result<()> {
//! let params = OnlineParams::builder(0.1, 0.05)
//!     .initial_threshold(1.0)
//!     .lags(1)
//!     .build()?;
//! let mut calibration = OnlineCalibrator::new(params)?;
//! let point_prediction = 2.0;
//! let issued = calibration.predict(&[])?;
//! let interval = (point_prediction - issued.threshold(), point_prediction + issued.threshold());
//! // Only after this forecast's label arrives:
//! let label: f64 = 3.5;
//! calibration.update(&issued, (label - point_prediction).abs())?;
//! assert_eq!(calibration.n_updates(), 1);
//! # let _ = interval;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use crate::check::{ensure, positive};
use crate::error::{HessboostError, Result};

/// Validated configuration for scalar or linear online quantile tracking.
#[derive(Debug, Clone)]
pub struct OnlineParams {
    alpha: f64,
    learning_rate: f64,
    initial_threshold: f64,
    lags: usize,
    covariates: usize,
    coefficient_bound: f64,
}

impl OnlineParams {
    /// Start with no features, threshold zero, and coefficient bound one.
    /// The step size is explicit because it depends on the score's scale.
    pub fn builder(alpha: f64, learning_rate: f64) -> OnlineParamsBuilder {
        OnlineParamsBuilder {
            params: Self {
                alpha,
                learning_rate,
                initial_threshold: 0.0,
                lags: 0,
                covariates: 0,
                coefficient_bound: 1.0,
            },
        }
    }

    /// Target miscoverage, strictly between zero and one.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Fixed positive step size in score units.
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// The intercept before any feedback.
    pub fn initial_threshold(&self) -> f64 {
        self.initial_threshold
    }

    /// Number of most recently observed scores included as features.
    pub fn lags(&self) -> usize {
        self.lags
    }

    /// Caller covariates required by every prediction.
    pub fn covariates(&self) -> usize {
        self.covariates
    }

    /// L2 bound on the non-bias coefficients, not on the intercept.
    pub fn coefficient_bound(&self) -> f64 {
        self.coefficient_bound
    }
}

/// Builder for [`OnlineParams`]; every value is checked by [`Self::build`].
#[derive(Debug, Clone)]
pub struct OnlineParamsBuilder {
    params: OnlineParams,
}

impl OnlineParamsBuilder {
    /// Set the initial threshold, a finite value in the score's units.
    #[must_use]
    pub fn initial_threshold(mut self, value: f64) -> Self {
        self.params.initial_threshold = value;
        self
    }

    /// Include this many most recently observed scores, initially zero.
    #[must_use]
    pub fn lags(mut self, count: usize) -> Self {
        self.params.lags = count;
        self
    }

    /// Require this many caller-provided covariates per prediction.
    #[must_use]
    pub fn covariates(mut self, count: usize) -> Self {
        self.params.covariates = count;
        self
    }

    /// Set the positive finite L2 bound for all non-bias coefficients.
    #[must_use]
    pub fn coefficient_bound(mut self, value: f64) -> Self {
        self.params.coefficient_bound = value;
        self
    }

    /// Validate the probability, step, initial threshold, and dimensions.
    pub fn build(self) -> Result<OnlineParams> {
        let p = self.params;
        super::validate_alpha(p.alpha)?;
        positive("learning_rate", p.learning_rate)?;
        positive("coefficient_bound", p.coefficient_bound)?;
        ensure(
            "initial_threshold",
            p.initial_threshold.is_finite(),
            "must be finite",
        )?;
        let size = p.lags.checked_add(p.covariates);
        ensure(
            "covariates",
            size.and_then(|n| n.checked_mul(size_of::<f64>()))
                .is_some_and(|bytes| isize::try_from(bytes).is_ok()),
            "lags plus covariates must fit a feature vector",
        )?;
        Ok(p)
    }
}

/// An issued score threshold and its immutable forecast-time feature snapshot.
/// Only the creating calibrator can accept it, once and in issue order.
#[derive(Debug)]
pub struct Forecast {
    owner: Arc<()>,
    index: u64,
    threshold: f64,
    features: Vec<f64>,
}

impl Forecast {
    /// Issued threshold `q`, defining the set `score(x, y) <= q`.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Zero-based issue index within this calibrator.
    pub fn index(&self) -> u64 {
        self.index
    }
}

/// Mutable scalar or linear online score calibrator. See the [module docs](self).
///
/// Invalid feedback leaves the intercept, coefficients, lags and counters
/// unchanged; a refused receipt can be retried with valid feedback. This
/// type is deliberately not `Clone`: receipts identify one evolving state.
#[derive(Debug)]
pub struct OnlineCalibrator {
    params: OnlineParams,
    owner: Arc<()>,
    intercept: f64,
    coefficients: Vec<f64>,
    scratch: Vec<f64>,
    history: Vec<f64>,
    cursor: usize,
    issued: u64,
    updated: u64,
}

fn buffer(size: usize) -> Result<Vec<f64>> {
    let mut values = Vec::new();
    values.try_reserve_exact(size).map_err(|_| {
        HessboostError::invalid_param("covariates", "cannot allocate the feature vector")
    })?;
    values.resize(size, 0.0);
    Ok(values)
}

impl OnlineCalibrator {
    /// Construct zero coefficients and a zero-filled score history.
    /// Refuses dimensions for which storage cannot be allocated.
    pub fn new(params: OnlineParams) -> Result<Self> {
        let width = params.lags + params.covariates;
        Ok(Self {
            coefficients: buffer(width)?,
            scratch: buffer(width)?,
            history: buffer(params.lags)?,
            intercept: params.initial_threshold,
            params,
            owner: Arc::new(()),
            cursor: 0,
            issued: 0,
            updated: 0,
        })
    }

    /// Issue a threshold using only the state and covariates available now.
    ///
    /// `covariates` must have the configured length and finite values. Each
    /// successful call creates a receipt that must eventually be updated.
    /// An overflowed threshold is refused without consuming an issue index.
    pub fn predict(&mut self, covariates: &[f64]) -> Result<Forecast> {
        if covariates.len() != self.params.covariates {
            return Err(HessboostError::dimension_mismatch(
                "covariates",
                self.params.covariates,
                covariates.len(),
            ));
        }
        if covariates.iter().any(|x| !x.is_finite()) {
            return Err(HessboostError::invalid_data("covariates", "must be finite"));
        }
        let next = self.issued.checked_add(1).ok_or_else(|| {
            HessboostError::invalid_param("forecast", "issue counter is exhausted")
        })?;
        let mut features = Vec::new();
        features
            .try_reserve_exact(self.coefficients.len())
            .map_err(|_| {
                HessboostError::invalid_param("covariates", "cannot allocate the feature snapshot")
            })?;
        let mut cursor = self.cursor;
        for _ in 0..self.params.lags {
            cursor = if cursor == 0 {
                self.history.len() - 1
            } else {
                cursor - 1
            };
            features.push(self.history[cursor]);
        }
        features.extend_from_slice(covariates);
        let threshold = self
            .coefficients
            .iter()
            .zip(&features)
            .fold(self.intercept, |sum, (&coefficient, &feature)| {
                sum + coefficient * feature
            });
        ensure(
            "covariates",
            threshold.is_finite(),
            "issued threshold would overflow",
        )?;
        let forecast = Forecast {
            owner: Arc::clone(&self.owner),
            index: self.issued,
            threshold,
            features,
        };
        self.issued = next;
        Ok(forecast)
    }

    /// Apply one score to its original receipt, strictly in issue order.
    ///
    /// With `miss = score > forecast.threshold()`, the intercept moves by
    /// `learning_rate * (miss - alpha)`, and each coefficient by that change
    /// times its **issued** feature, followed by L2 projection. Ties count
    /// as covered. The score becomes the newest lag only after this update.
    ///
    /// Foreign, duplicate, out-of-order receipts, nonfinite scores and
    /// numerically unrepresentable updates are refused without changing
    /// observable state. No allocations or Rayon calls occur here.
    pub fn update(&mut self, forecast: &Forecast, score: f64) -> Result<()> {
        ensure(
            "forecast",
            Arc::ptr_eq(&self.owner, &forecast.owner),
            "belongs to another calibrator",
        )?;
        ensure(
            "forecast",
            forecast.index == self.updated,
            "must be updated once, in issue order",
        )?;
        if !score.is_finite() {
            return Err(HessboostError::invalid_data("score", "must be finite"));
        }
        let miss = f64::from(u8::from(score > forecast.threshold));
        let delta = self.params.learning_rate * (miss - self.params.alpha);
        let intercept = self.intercept + delta;
        ensure(
            "learning_rate",
            intercept.is_finite(),
            "intercept update would overflow",
        )?;
        let mut largest = 0.0f64;
        for ((candidate, &coefficient), &feature) in self
            .scratch
            .iter_mut()
            .zip(&self.coefficients)
            .zip(&forecast.features)
        {
            *candidate = coefficient + delta * feature;
            ensure(
                "learning_rate",
                candidate.is_finite(),
                "coefficient update would overflow",
            )?;
            largest = largest.max(candidate.abs());
        }
        if largest > 0.0 {
            // Scaled norm avoids overflow even when the coefficients' norm
            // itself is not representable. Divide before multiplying to
            // avoid underflow of a tiny projection factor.
            let norm = self
                .scratch
                .iter()
                .map(|&x| (x / largest).powi(2))
                .sum::<f64>()
                .sqrt();
            let radius = self.params.coefficient_bound / norm;
            if largest > radius {
                for value in &mut self.scratch {
                    *value = (*value / largest) * radius;
                }
            }
        }
        self.intercept = intercept;
        std::mem::swap(&mut self.coefficients, &mut self.scratch);
        if !self.history.is_empty() {
            self.history[self.cursor] = score;
            self.cursor += 1;
            if self.cursor == self.history.len() {
                self.cursor = 0;
            }
        }
        // The receipt was issued successfully, so its successor fits u64.
        self.updated += 1;
        Ok(())
    }

    /// The immutable configuration.
    pub fn params(&self) -> &OnlineParams {
        &self.params
    }

    /// Successfully applied feedback count.
    pub fn n_updates(&self) -> u64 {
        self.updated
    }

    /// Issued receipts still awaiting feedback.
    pub fn pending(&self) -> u64 {
        self.issued - self.updated
    }

    /// Current bias term, before adding lag and covariate contributions.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Current non-bias coefficients: lagged scores, then caller covariates.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar() -> OnlineCalibrator {
        OnlineCalibrator::new(OnlineParams::builder(0.25, 0.5).build().unwrap()).unwrap()
    }

    #[test]
    fn scalar_updates_include_ties_and_negative_thresholds() {
        let mut cal = scalar();
        let first = cal.predict(&[]).unwrap();
        cal.update(&first, 0.0).unwrap();
        assert_eq!(cal.intercept(), -0.125);
        let second = cal.predict(&[]).unwrap();
        assert_eq!(second.threshold(), -0.125);
        cal.update(&second, 0.0).unwrap();
        assert_eq!(cal.intercept(), 0.25);
        assert_eq!((cal.n_updates(), cal.pending()), (2, 0));
    }

    #[test]
    fn delayed_feedback_uses_issued_threshold_and_features() {
        let mut cal = OnlineCalibrator::new(
            OnlineParams::builder(0.5, 1.0)
                .lags(1)
                .covariates(1)
                .coefficient_bound(10.0)
                .build()
                .unwrap(),
        )
        .unwrap();
        let first = cal.predict(&[2.0]).unwrap();
        let second = cal.predict(&[3.0]).unwrap();
        cal.update(&first, 1.0).unwrap();
        assert_eq!(cal.coefficients(), [0.0, 1.0]);
        // Current prediction for these covariates would be 3.5, but the
        // issued threshold was zero, and its lag was zero, not one.
        cal.update(&second, 1.0).unwrap();
        assert_eq!(cal.intercept(), 1.0);
        assert_eq!(cal.coefficients(), [0.0, 2.5]);
        let third = cal.predict(&[0.0]).unwrap();
        cal.update(&third, 2.0).unwrap();
        assert_eq!(cal.coefficients(), [0.5, 2.5]);
        assert_eq!(cal.predict(&[0.0]).unwrap().threshold(), 2.5);
    }

    #[test]
    fn receipts_refuse_reordering_replay_and_foreign_state_without_mutation() {
        let mut cal = scalar();
        let first = cal.predict(&[]).unwrap();
        let second = cal.predict(&[]).unwrap();
        let mut other = scalar();
        assert!(matches!(
            other.update(&first, 1.0),
            Err(HessboostError::InvalidParameter {
                name: "forecast",
                ..
            })
        ));
        assert!(matches!(
            cal.update(&second, 1.0),
            Err(HessboostError::InvalidParameter {
                name: "forecast",
                ..
            })
        ));
        assert!(matches!(
            cal.update(&first, f64::NAN),
            Err(HessboostError::InvalidData { input: "score", .. })
        ));
        assert_eq!(
            (cal.intercept(), cal.n_updates(), cal.pending()),
            (0.0, 0, 2)
        );
        cal.update(&first, 1.0).unwrap();
        assert!(matches!(
            cal.update(&first, 1.0),
            Err(HessboostError::InvalidParameter {
                name: "forecast",
                ..
            })
        ));
        cal.update(&second, 1.0).unwrap();
        assert_eq!(
            (cal.intercept(), cal.n_updates(), cal.pending()),
            (0.75, 2, 0)
        );
    }

    #[test]
    fn coefficients_project_together_without_overflowing_norm() {
        let mut cal = OnlineCalibrator::new(
            OnlineParams::builder(0.5, 2.0)
                .covariates(2)
                .coefficient_bound(5.0)
                .build()
                .unwrap(),
        )
        .unwrap();
        let forecast = cal.predict(&[3.0, 4.0]).unwrap();
        cal.update(&forecast, 1.0).unwrap();
        assert_eq!(cal.coefficients(), [3.0, 4.0]);
        let forecast = cal.predict(&[3.0, 4.0]).unwrap();
        cal.update(&forecast, 100.0).unwrap();
        assert_eq!(cal.coefficients(), [3.0, 4.0]);
        let mut huge = OnlineCalibrator::new(
            OnlineParams::builder(0.5, 2.0)
                .covariates(2)
                .coefficient_bound(1.0)
                .build()
                .unwrap(),
        )
        .unwrap();
        let forecast = huge.predict(&[f64::MAX, f64::MAX]).unwrap();
        huge.update(&forecast, 1.0).unwrap();
        assert!(
            huge.coefficients()
                .iter()
                .all(|&v| (v - 0.5f64.sqrt()).abs() < 1e-15)
        );
    }

    #[test]
    fn invalid_prediction_and_overflowed_update_preserve_state() {
        let mut cal = OnlineCalibrator::new(
            OnlineParams::builder(0.25, f64::MAX)
                .covariates(1)
                .build()
                .unwrap(),
        )
        .unwrap();
        assert!(matches!(
            cal.predict(&[]),
            Err(HessboostError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            cal.predict(&[f64::INFINITY]),
            Err(HessboostError::InvalidData {
                input: "covariates",
                ..
            })
        ));
        assert_eq!(cal.pending(), 0);
        let forecast = cal.predict(&[f64::MAX]).unwrap();
        assert!(matches!(
            cal.update(&forecast, 1.0),
            Err(HessboostError::InvalidParameter {
                name: "learning_rate",
                ..
            })
        ));
        assert_eq!(cal.coefficients(), [0.0]);
        assert_eq!(
            (cal.intercept(), cal.n_updates(), cal.pending()),
            (0.0, 0, 1)
        );
    }

    #[test]
    fn lag_ring_keeps_newest_scores_first() {
        let mut cal =
            OnlineCalibrator::new(OnlineParams::builder(0.5, 0.01).lags(3).build().unwrap())
                .unwrap();
        for score in 1..=7 {
            let forecast = cal.predict(&[]).unwrap();
            cal.update(&forecast, f64::from(score)).unwrap();
        }
        let forecast = cal.predict(&[]).unwrap();
        assert_eq!(forecast.features, [7.0, 6.0, 5.0]);
    }

    #[test]
    fn refused_numeric_update_can_be_retried() {
        let mut cal = OnlineCalibrator::new(
            OnlineParams::builder(0.75, f64::MAX)
                .covariates(1)
                .build()
                .unwrap(),
        )
        .unwrap();
        let forecast = cal.predict(&[2.0]).unwrap();
        assert!(cal.update(&forecast, -1.0).is_err());
        cal.update(&forecast, 1.0).unwrap();
        assert_eq!(cal.coefficients(), [1.0]);
        assert_eq!(cal.n_updates(), 1);
        assert_eq!(cal.pending(), 0);
    }

    #[test]
    fn parameter_boundaries_are_checked() {
        for alpha in [0.0, 1.0, f64::NAN] {
            assert!(matches!(
                OnlineParams::builder(alpha, 1.0).build(),
                Err(HessboostError::InvalidParameter { name: "alpha", .. })
            ));
        }
        for rate in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            assert!(matches!(
                OnlineParams::builder(0.1, rate).build(),
                Err(HessboostError::InvalidParameter {
                    name: "learning_rate",
                    ..
                })
            ));
        }
        assert!(matches!(
            OnlineParams::builder(0.1, 1.0)
                .initial_threshold(f64::INFINITY)
                .build(),
            Err(HessboostError::InvalidParameter {
                name: "initial_threshold",
                ..
            })
        ));
        assert!(matches!(
            OnlineParams::builder(0.1, 1.0)
                .coefficient_bound(0.0)
                .build(),
            Err(HessboostError::InvalidParameter {
                name: "coefficient_bound",
                ..
            })
        ));
        assert!(matches!(
            OnlineParams::builder(0.1, 1.0)
                .lags(usize::MAX)
                .covariates(1)
                .build(),
            Err(HessboostError::InvalidParameter {
                name: "covariates",
                ..
            })
        ));
    }
}
