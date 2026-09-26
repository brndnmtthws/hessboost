//! The option groups of [`TrainingParams`](super::TrainingParams): settings
//! that only mean something when a switch is on live inside that switch
//! (`BoosterKind::Dart(Dart)`, `ProcessType::Update(Refresh)`,
//! `Option<QuantizedGrad>`, `Option<ExtraTrees>`, `Option<LinearTree>`,
//! `Option<BalancedBagging>`), so
//! they cannot be set while the switch is off. Each validates its values
//! when built.

use crate::error::{HessboostError, Result};

/// DART's dropout (XGBoost `booster = dart`): the fraction of trees
/// dropped each round (`rate_drop`) and the probability of skipping the
/// dropout in a round (`skip_drop`), both in `[0, 1]` and `0` by default.
///
/// ```
/// use hessboost::config::Dart;
///
/// # fn main() -> hessboost::error::Result<()> {
/// let dart = Dart::builder().rate_drop(0.1).skip_drop(0.5).build()?;
/// assert_eq!((dart.rate_drop(), dart.skip_drop()), (0.1, 0.5));
/// assert!(Dart::builder().rate_drop(1.5).build().is_err());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Dart {
    rate_drop: f64,
    skip_drop: f64,
}

impl Dart {
    /// Start a builder at XGBoost's defaults (no dropout).
    pub fn builder() -> DartBuilder {
        DartBuilder {
            dart: Dart::default(),
        }
    }

    /// Fraction of trees dropped each round. XGBoost `rate_drop`.
    pub fn rate_drop(&self) -> f64 {
        self.rate_drop
    }

    /// Probability of skipping the dropout in a round. XGBoost `skip_drop`.
    pub fn skip_drop(&self) -> f64 {
        self.skip_drop
    }
}

/// Builder of [`Dart`].
#[derive(Debug, Clone, Copy)]
pub struct DartBuilder {
    dart: Dart,
}

impl DartBuilder {
    /// Set the fraction of trees dropped each round (`rate_drop`).
    #[must_use]
    pub fn rate_drop(mut self, rate_drop: f64) -> Self {
        self.dart.rate_drop = rate_drop;
        self
    }

    /// Set the probability of skipping the dropout (`skip_drop`).
    #[must_use]
    pub fn skip_drop(mut self, skip_drop: f64) -> Self {
        self.dart.skip_drop = skip_drop;
        self
    }

    /// The validated dropout.
    ///
    /// # Errors
    ///
    /// `rate_drop` or `skip_drop` outside `[0, 1]`.
    pub fn build(self) -> Result<Dart> {
        unit("rate_drop", self.dart.rate_drop)?;
        unit("skip_drop", self.dart.skip_drop)?;
        Ok(self.dart)
    }
}

/// The refresh updater of `process_type = update`: whether it also rewrites
/// leaf values, not only node statistics (XGBoost `refresh_leaf`, default
/// on).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Refresh {
    refresh_leaf: bool,
}

impl Refresh {
    /// Refresh node statistics only, keeping the leaf values
    /// (`refresh_leaf = false`).
    pub fn stats_only() -> Self {
        Refresh {
            refresh_leaf: false,
        }
    }

    /// Whether leaf values are rewritten too. XGBoost `refresh_leaf`.
    pub fn refresh_leaf(&self) -> bool {
        self.refresh_leaf
    }
}

impl Default for Refresh {
    /// XGBoost's default: refresh statistics and leaf values.
    fn default() -> Self {
        Refresh { refresh_leaf: true }
    }
}

/// Training on quantized gradients (LightGBM `use_quantized_grad`; Shi et
/// al., NeurIPS 2022): gradients and Hessians become small integers
/// summed in integer histograms. Opt-in and beyond XGBoost: trees differ
/// from full-precision training.
///
/// ```
/// use hessboost::config::QuantizedGrad;
///
/// # fn main() -> hessboost::error::Result<()> {
/// let quantized = QuantizedGrad::builder().bins(8).renew_leaf(true).build()?;
/// assert_eq!(quantized.bins(), 8);
/// assert!(quantized.stochastic_rounding());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizedGrad {
    bins: usize,
    stochastic_rounding: bool,
    renew_leaf: bool,
}

impl QuantizedGrad {
    /// Start a builder at LightGBM's defaults (4 levels, stochastic
    /// rounding, no leaf renewal).
    pub fn builder() -> QuantizedGradBuilder {
        QuantizedGradBuilder {
            quantized: QuantizedGrad::default(),
        }
    }

    /// Quantization levels `Q`: gradients map to integers in
    /// `[-⌊Q/2⌋, ⌊Q/2⌋]`, non-negative Hessians to `[0, Q]`. LightGBM
    /// `num_grad_quant_bins`.
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// Whether gradients are rounded stochastically (unbiased) rather than
    /// to the nearest level. LightGBM `stochastic_rounding`.
    pub fn stochastic_rounding(&self) -> bool {
        self.stochastic_rounding
    }

    /// Whether each leaf value is recomputed from the full-precision
    /// gradients of its rows once a tree is grown. LightGBM
    /// `quant_train_renew_leaf`.
    pub fn renew_leaf(&self) -> bool {
        self.renew_leaf
    }
}

impl Default for QuantizedGrad {
    fn default() -> Self {
        QuantizedGrad {
            bins: 4,
            stochastic_rounding: true,
            renew_leaf: false,
        }
    }
}

/// Builder of [`QuantizedGrad`].
#[derive(Debug, Clone, Copy)]
pub struct QuantizedGradBuilder {
    quantized: QuantizedGrad,
}

impl QuantizedGradBuilder {
    /// Set the quantization levels (`num_grad_quant_bins`).
    #[must_use]
    pub fn bins(mut self, bins: usize) -> Self {
        self.quantized.bins = bins;
        self
    }

    /// Set stochastic rounding (`stochastic_rounding`).
    #[must_use]
    pub fn stochastic_rounding(mut self, stochastic: bool) -> Self {
        self.quantized.stochastic_rounding = stochastic;
        self
    }

    /// Set full-precision leaf renewal (`quant_train_renew_leaf`).
    #[must_use]
    pub fn renew_leaf(mut self, renew: bool) -> Self {
        self.quantized.renew_leaf = renew;
        self
    }

    /// The validated quantization.
    ///
    /// # Errors
    ///
    /// `bins` outside `[2, 127]` (LightGBM stores each value in 8 bits).
    pub fn build(self) -> Result<QuantizedGrad> {
        let bins = self.quantized.bins;
        if !(2..=127).contains(&bins) {
            return Err(HessboostError::invalid_param(
                "num_grad_quant_bins",
                format!("must be in [2, 127], got {bins}"),
            ));
        }
        Ok(self.quantized)
    }
}

/// Extremely randomized split search (LightGBM `extra_trees`): every
/// numerical feature is scored at one random bin boundary per node, drawn
/// uniformly between the node's lowest and highest occupied bin, and every
/// categorical feature at one random prefix of its gradient-ordered
/// categories. The draws are keyed by `seed` (LightGBM `extra_seed`,
/// default 6) and the per-tree seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtraTrees {
    seed: u64,
}

impl ExtraTrees {
    /// Randomized splits drawn from `seed` (LightGBM `extra_seed`).
    pub fn with_seed(seed: u64) -> Self {
        ExtraTrees { seed }
    }

    /// The seed of the threshold draws.
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl Default for ExtraTrees {
    /// LightGBM's default seed 6.
    fn default() -> Self {
        ExtraTrees { seed: 6 }
    }
}

/// Ridge-regularized linear models in the leaves (LightGBM `linear_tree`):
/// every leaf fits the numerical features split on along its path, with an
/// L2 penalty `lambda >= 0` on the slopes, not the intercepts (LightGBM
/// `linear_lambda`, default 0).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LinearTree {
    lambda: f64,
}

impl LinearTree {
    /// Linear leaves with slope penalty `lambda`.
    ///
    /// # Errors
    ///
    /// `linear_lambda` is negative or not finite.
    pub fn new(lambda: f64) -> Result<Self> {
        if !(lambda.is_finite() && lambda >= 0.0) {
            return Err(HessboostError::invalid_param(
                "linear_lambda",
                format!("must be >= 0, got {lambda}"),
            ));
        }
        Ok(LinearTree { lambda })
    }

    /// The penalty on the leaf models' slopes.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

/// LightGBM's class-balanced bagging for binary classification
/// (`pos_bagging_fraction`, `neg_bagging_fraction`): every round keeps each
/// positive row (label `1`) with probability `pos` and each negative row
/// (label `0`) with probability `neg`, in place of `subsample`. Both
/// fractions lie in `(0, 1]` and at least one is below `1` (both at `1`
/// is no bagging: leave the option unset).
///
/// ```
/// use hessboost::config::BalancedBagging;
///
/// # fn main() -> hessboost::error::Result<()> {
/// let bagging = BalancedBagging::new(1.0, 0.2)?;
/// assert_eq!((bagging.pos_fraction(), bagging.neg_fraction()), (1.0, 0.2));
/// assert!(BalancedBagging::new(1.0, 1.0).is_err());
/// assert!(BalancedBagging::new(0.0, 0.5).is_err());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BalancedBagging {
    pos: f64,
    neg: f64,
}

impl BalancedBagging {
    /// Keep positives with probability `pos` and negatives with `neg`.
    ///
    /// # Errors
    ///
    /// A fraction outside `(0, 1]` (named `pos_bagging_fraction` or
    /// `neg_bagging_fraction`), or both at `1`.
    pub fn new(pos: f64, neg: f64) -> Result<Self> {
        fraction("pos_bagging_fraction", pos)?;
        fraction("neg_bagging_fraction", neg)?;
        if pos == 1.0 && neg == 1.0 {
            return Err(HessboostError::invalid_param(
                "pos_bagging_fraction",
                "with `neg_bagging_fraction` also 1 nothing is bagged; \
                 leave balanced bagging unset",
            ));
        }
        Ok(BalancedBagging { pos, neg })
    }

    /// The probability of keeping a positive row (LightGBM
    /// `pos_bagging_fraction`).
    pub fn pos_fraction(&self) -> f64 {
        self.pos
    }

    /// The probability of keeping a negative row (LightGBM
    /// `neg_bagging_fraction`).
    pub fn neg_fraction(&self) -> f64 {
        self.neg
    }
}

/// Fail unless `v` is in `(0, 1]`.
fn fraction(name: &'static str, v: f64) -> Result<()> {
    if v > 0.0 && v <= 1.0 {
        Ok(())
    } else {
        Err(HessboostError::invalid_param(
            name,
            format!("must be in (0, 1], got {v}"),
        ))
    }
}

/// Fail unless `v` is finite and in `[0, 1]`.
fn unit(name: &'static str, v: f64) -> Result<()> {
    if v.is_finite() && (0.0..=1.0).contains(&v) {
        Ok(())
    } else {
        Err(HessboostError::invalid_param(
            name,
            format!("must be in [0, 1], got {v}"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parameter a build refuses, if any.
    fn refused<T>(built: &Result<T>) -> Option<&'static str> {
        match built {
            Err(HessboostError::InvalidParameter { name, .. }) => Some(name),
            _ => None,
        }
    }

    /// Each group refuses the values of its switch's range by the
    /// parameter's XGBoost/LightGBM name.
    #[test]
    fn groups_refuse_out_of_range_values_by_name() {
        for v in [-0.1, 1.5, f64::NAN] {
            assert_eq!(
                refused(&Dart::builder().rate_drop(v).build()),
                Some("rate_drop")
            );
            assert_eq!(
                refused(&Dart::builder().skip_drop(v).build()),
                Some("skip_drop")
            );
        }
        for bins in [0, 1, 128] {
            assert_eq!(
                refused(&QuantizedGrad::builder().bins(bins).build()),
                Some("num_grad_quant_bins")
            );
        }
        assert!(QuantizedGrad::builder().bins(127).build().is_ok());
        for lambda in [-1.0, f64::INFINITY] {
            assert_eq!(refused(&LinearTree::new(lambda)), Some("linear_lambda"));
        }
        assert!(Refresh::default().refresh_leaf());
        assert!(!Refresh::stats_only().refresh_leaf());
        assert_eq!(ExtraTrees::default().seed(), 6);
        for v in [0.0, -0.5, 1.1, f64::NAN, f64::INFINITY] {
            assert_eq!(
                refused(&BalancedBagging::new(v, 0.5)),
                Some("pos_bagging_fraction")
            );
            assert_eq!(
                refused(&BalancedBagging::new(0.5, v)),
                Some("neg_bagging_fraction")
            );
        }
        assert_eq!(
            refused(&BalancedBagging::new(1.0, 1.0)),
            Some("pos_bagging_fraction")
        );
    }
}
