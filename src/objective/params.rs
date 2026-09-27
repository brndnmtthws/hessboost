//! The parameters of the built-in objectives, shared by the evaluation
//! metrics that read them ([`EvalMetric`](crate::metric::EvalMetric)). Each
//! is validated when it is built, with the XGBoost parameter name in the
//! error, so a value that exists is valid.

use crate::error::{HessboostError, Result};
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;

/// Noise distribution of the accelerated-failure-time survival loss.
///
/// Mirrors XGBoost's `aft_loss_distribution`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum AftDistribution {
    /// Normal (Gaussian) noise. XGBoost default.
    #[default]
    Normal,
    /// Logistic noise.
    Logistic,
    /// Type-1 extreme-value (Gumbel minimum) noise.
    Extreme,
}

stored_names! {
    AftDistribution { Normal => "normal", Logistic => "logistic", Extreme => "extreme" }
}

/// The slope `δ` of the pseudo-Huber loss (`reg:pseudohubererror` and the
/// `mphe` metric). XGBoost `huber_slope`, default `1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PseudoHuber {
    slope: f64,
}

impl PseudoHuber {
    /// The pseudo-Huber loss with slope `slope`.
    ///
    /// # Errors
    ///
    /// `huber_slope` is not finite and positive, or its square is not
    /// positive and finite in the `f32` the loss computes in.
    pub fn new(slope: f64) -> Result<Self> {
        if !(slope.is_finite() && slope > 0.0) {
            return Err(HessboostError::invalid_param(
                "huber_slope",
                format!("must be > 0, got {slope}"),
            ));
        }
        let square = (slope as f32) * (slope as f32);
        if !(square.is_finite() && square > 0.0) {
            return Err(HessboostError::invalid_param(
                "huber_slope",
                format!("squared slope must stay positive and finite in f32, got {slope}"),
            ));
        }
        Ok(PseudoHuber { slope })
    }

    /// The slope `δ`.
    pub fn slope(&self) -> f64 {
        self.slope
    }
}

impl Default for PseudoHuber {
    /// XGBoost's default slope `1`.
    fn default() -> Self {
        PseudoHuber { slope: 1.0 }
    }
}

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

/// Declares an alpha-list parameter: [`Quantiles`] and [`Expectiles`].
macro_rules! alpha_list {
    ($(#[$m:meta])* $ty:ident, $param:literal) => {
        $(#[$m])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $ty {
            alpha: Vec<f64>,
        }

        impl $ty {
            #[doc = concat!("The levels `alpha` (XGBoost `", $param, "`).")]
            ///
            /// # Errors
            ///
            /// `alpha` is empty, has an entry outside `[0, 1]`, or is not
            /// ascending (checked in the `f32` the loss computes in; equal
            /// neighbours are allowed).
            pub fn new(alpha: impl IntoIterator<Item = f64>) -> Result<Self> {
                let alpha: Vec<f64> = alpha.into_iter().collect();
                validate_alphas($param, &alpha)?;
                Ok($ty { alpha })
            }

            /// The levels, ascending.
            pub fn alpha(&self) -> &[f64] {
                &self.alpha
            }

            /// The levels as the loss and metric compute with them.
            pub(crate) fn alpha_f32(&self) -> Vec<f32> {
                self.alpha.iter().map(|&a| a as f32).collect()
            }
        }
    };
}

alpha_list!(
    /// The target quantiles of `reg:quantileerror` and the `quantile`
    /// metric: one output per level.
    Quantiles,
    "quantile_alpha"
);

alpha_list!(
    /// The target expectiles of `reg:expectileerror` and the `expectile`
    /// metric: one output per level.
    Expectiles,
    "expectile_alpha"
);

/// The variance power `ρ` of the Tweedie distribution (`reg:tweedie` and
/// the `tweedie-nloglik` metric): `1` is Poisson, `2` (excluded) Gamma.
/// XGBoost `tweedie_variance_power`, default `1.5`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tweedie {
    variance_power: f64,
}

impl Tweedie {
    /// The Tweedie distribution with variance power `variance_power`.
    ///
    /// # Errors
    ///
    /// `tweedie_variance_power` is outside `[1, 2)` once rounded to the
    /// `f32` the loss computes in.
    pub fn new(variance_power: f64) -> Result<Self> {
        let rho = variance_power as f32;
        if !(variance_power.is_finite() && (1.0f32..2.0).contains(&rho)) {
            return Err(HessboostError::invalid_param(
                "tweedie_variance_power",
                format!("must be in [1, 2) (as f32), got {variance_power}"),
            ));
        }
        Ok(Tweedie { variance_power })
    }

    /// The variance power `ρ`.
    pub fn variance_power(&self) -> f64 {
        self.variance_power
    }
}

impl Default for Tweedie {
    /// XGBoost's default variance power `1.5`.
    fn default() -> Self {
        Tweedie {
            variance_power: 1.5,
        }
    }
}

/// The noise model of the accelerated-failure-time survival loss
/// (`survival:aft` and the `aft-nloglik` metric): a distribution and its
/// scale. XGBoost `aft_loss_distribution` (default normal) and
/// `aft_loss_distribution_scale` (default `1`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aft {
    distribution: AftDistribution,
    scale: f64,
}

impl Aft {
    /// AFT noise from `distribution` with scale `scale`.
    ///
    /// # Errors
    ///
    /// `aft_loss_distribution_scale` is not finite and positive, also once
    /// rounded to the `f32` the loss computes in.
    pub fn new(distribution: AftDistribution, scale: f64) -> Result<Self> {
        let narrowed = scale as f32;
        if !(scale.is_finite() && scale > 0.0) {
            return Err(HessboostError::invalid_param(
                "aft_loss_distribution_scale",
                format!("must be > 0, got {scale}"),
            ));
        }
        if !(narrowed.is_finite() && narrowed > 0.0) {
            return Err(HessboostError::invalid_param(
                "aft_loss_distribution_scale",
                format!("must stay positive and finite in f32, got {scale}"),
            ));
        }
        Ok(Aft {
            distribution,
            scale,
        })
    }

    /// `distribution` at XGBoost's default scale `1`.
    pub fn with_distribution(distribution: AftDistribution) -> Self {
        Aft {
            distribution,
            scale: 1.0,
        }
    }

    /// The noise distribution.
    pub fn distribution(&self) -> AftDistribution {
        self.distribution
    }

    /// The noise scale.
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Default for Aft {
    /// XGBoost's defaults: normal noise with scale `1`.
    fn default() -> Self {
        Aft::with_distribution(AftDistribution::Normal)
    }
}

/// The parameter of XGBoost's `RegLossObj` objectives
/// (`reg:squarederror`, `reg:gamma`, `reg:logistic`, `binary:logistic`,
/// `binary:logitraw`): the positive-label weight `scale_pos_weight`
/// (XGBoost's `reg_loss_param`), default `1`.
///
/// The gradient and Hessian of every row labeled exactly `1` (every cell of
/// a label matrix) are multiplied by it on top of the sample weight, in
/// `f32` as XGBoost's `GetGradient` does; to balance imbalanced classes, or
/// to reweight the rows at `1` of a regression. Away from `1` an estimated
/// intercept is one Newton step on these reweighted gradients (XGBoost's
/// `FitIntercept`) instead of the (weighted) label mean.
///
/// ```
/// use hessboost::objective::{Objective, RegLoss};
///
/// # fn main() -> hessboost::error::Result<()> {
/// let balanced = Objective::BinaryLogistic(RegLoss::new(3.0)?);
/// assert_eq!(Objective::default(), Objective::SquaredError(RegLoss::default()));
/// # let _ = balanced;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegLoss {
    scale_pos_weight: f64,
}

impl RegLoss {
    /// The loss with positive-label weight `scale_pos_weight`.
    ///
    /// # Errors
    ///
    /// `scale_pos_weight` is not finite and positive, also once rounded to
    /// the `f32` the loss computes in.
    pub fn new(scale_pos_weight: f64) -> Result<Self> {
        let narrowed = scale_pos_weight as f32;
        if !(scale_pos_weight.is_finite() && scale_pos_weight > 0.0) {
            return Err(HessboostError::invalid_param(
                "scale_pos_weight",
                format!("must be > 0, got {scale_pos_weight}"),
            ));
        }
        if !(narrowed.is_finite() && narrowed > 0.0) {
            return Err(HessboostError::invalid_param(
                "scale_pos_weight",
                format!("must stay positive and finite in f32, got {scale_pos_weight}"),
            ));
        }
        Ok(RegLoss { scale_pos_weight })
    }

    /// The positive-label weight.
    pub fn scale_pos_weight(&self) -> f64 {
        self.scale_pos_weight
    }
}

impl Default for RegLoss {
    /// XGBoost's default weight `1` (rows labeled `1` weigh as the others).
    fn default() -> Self {
        RegLoss {
            scale_pos_weight: 1.0,
        }
    }
}

/// The class count of the multiclass objectives (`multi:softmax`,
/// `multi:softprob`): one model output per class. XGBoost `num_class`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Multiclass {
    num_class: usize,
}

impl Multiclass {
    /// `num_class` classes, labeled `0 .. num_class`.
    ///
    /// # Errors
    ///
    /// `num_class` is below 2.
    pub fn new(num_class: usize) -> Result<Self> {
        if num_class < 2 {
            return Err(HessboostError::invalid_param(
                "num_class",
                "multiclass objectives require num_class >= 2",
            ));
        }
        Ok(Multiclass { num_class })
    }

    /// The class count.
    pub fn num_class(&self) -> usize {
        self.num_class
    }
}

/// The LambdaMART ranking objectives' pairing (`rank:pairwise`,
/// `rank:ndcg`, `rank:map`): XGBoost's `topk` pair method pairs each of the
/// top `num_pair_per_sample` documents of a query with every lower-ranked
/// one. XGBoost `lambdarank_num_pair_per_sample`, default `32`; the default
/// evaluation metric is `ndcg@k` / `map@k` with this `k`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LambdaRank {
    num_pair_per_sample: NonZeroUsize,
}

impl LambdaRank {
    /// Pair the top `num_pair_per_sample` documents of every query.
    ///
    /// # Errors
    ///
    /// `num_pair_per_sample` is 0.
    pub fn new(num_pair_per_sample: usize) -> Result<Self> {
        NonZeroUsize::new(num_pair_per_sample)
            .map(|num_pair_per_sample| LambdaRank {
                num_pair_per_sample,
            })
            .ok_or_else(|| {
                HessboostError::invalid_param("lambdarank_num_pair_per_sample", "must be >= 1")
            })
    }

    /// The top-k pair count.
    pub fn num_pair_per_sample(&self) -> usize {
        self.num_pair_per_sample.get()
    }
}

impl Default for LambdaRank {
    /// XGBoost's default pair count 32.
    fn default() -> Self {
        LambdaRank {
            num_pair_per_sample: NonZeroUsize::new(32).unwrap_or(NonZeroUsize::MIN),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parameter a constructor refuses, if any.
    fn refused<T>(built: &Result<T>) -> Option<&'static str> {
        match built {
            Err(HessboostError::InvalidParameter { name, .. }) => Some(name),
            _ => None,
        }
    }

    /// Each parameter refuses the values XGBoost refuses, and those that
    /// leave their range once narrowed to the `f32` the losses compute in.
    #[test]
    fn constructors_refuse_out_of_range_values_by_xgboost_name() {
        for slope in [0.0, -1.0, f64::NAN, 2e19, 1e-30] {
            assert_eq!(
                refused(&PseudoHuber::new(slope)),
                Some("huber_slope"),
                "{slope}"
            );
        }
        for rho in [2.0, 2.0 - f64::EPSILON, 0.5, f64::INFINITY] {
            assert_eq!(
                refused(&Tweedie::new(rho)),
                Some("tweedie_variance_power"),
                "{rho}"
            );
        }
        assert!(Tweedie::new(1.0).is_ok());
        for scale in [0.0, -1.0, f64::INFINITY, f64::NAN, 1e100, 1e-50] {
            assert_eq!(
                refused(&Aft::new(AftDistribution::Normal, scale)),
                Some("aft_loss_distribution_scale"),
                "{scale}"
            );
        }
        for alpha in [vec![], vec![0.5, f64::NAN], vec![1.5], vec![0.9, 0.1]] {
            assert_eq!(
                refused(&Quantiles::new(alpha.clone())),
                Some("quantile_alpha"),
                "{alpha:?}"
            );
            assert_eq!(
                refused(&Expectiles::new(alpha.clone())),
                Some("expectile_alpha"),
                "{alpha:?}"
            );
        }
        assert_eq!(
            Quantiles::new([0.1, 0.1, 0.9]).unwrap().alpha(),
            [0.1, 0.1, 0.9]
        );
    }

    /// The names model files store are the serde (and XGBoost) spellings,
    /// and read back to the same variant.
    #[test]
    fn stored_enum_names_match_serde_and_read_back() {
        fn check<T: serde::Serialize + Copy + PartialEq + std::fmt::Debug>(
            variants: &[T],
            name: fn(T) -> &'static str,
            from_name: fn(&str) -> Option<T>,
        ) {
            for &v in variants {
                assert_eq!(serde_json::to_value(v).unwrap(), name(v), "{v:?}");
                assert_eq!(from_name(name(v)), Some(v));
            }
            assert_eq!(from_name("no such variant"), None);
        }
        check(
            &[
                AftDistribution::Normal,
                AftDistribution::Logistic,
                AftDistribution::Extreme,
            ],
            AftDistribution::name,
            AftDistribution::from_name,
        );
        check(
            &[
                crate::objective::distributional::DistGradient::Fisher,
                crate::objective::distributional::DistGradient::Hessian,
                crate::objective::distributional::DistGradient::Natural,
            ],
            crate::objective::distributional::DistGradient::name,
            crate::objective::distributional::DistGradient::from_name,
        );
        check(
            &[
                crate::objective::distributional::DistSplitDirection::Random,
                crate::objective::distributional::DistSplitDirection::Cyclic,
                crate::objective::distributional::DistSplitDirection::All,
            ],
            crate::objective::distributional::DistSplitDirection::name,
            crate::objective::distributional::DistSplitDirection::from_name,
        );
    }
}
