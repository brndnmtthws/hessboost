//! The option groups of [`TrainingParams`](super::TrainingParams): settings
//! that only mean something when a switch is on live inside that switch
//! (`BoosterKind::Dart(Dart)`, `ProcessType::Update(Refresh)`,
//! `Option<QuantizedGrad>`, `Option<ExtraTrees>`, `Option<LinearTree>`,
//! `Option<BalancedBagging>`, `Option<QueryBagging>`, `Option<Langevin>`,
//! `Option<ModelShrink>`), so
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

/// Stochastic Gradient Langevin Boosting (CatBoost `langevin`; Ustimenko
/// and Prokhorenkova, ICML 2021). Every round adds Gaussian noise of
/// standard deviation `sqrt(2 / (eta * T))` (`T` the diffusion temperature)
/// to every row's gradient the tree structure is searched on (CatBoost's
/// per-row noise), then re-estimates every leaf from the noise-free
/// gradients of its rows plus independent noise
/// `sqrt(2 / (eta * T)) * sqrt(|H| + lambda)` on the leaf's gradient sum
/// (CatBoost's Newton leaves). The draws are keyed by `seed`, iteration,
/// tree, and row or leaf, so they do not depend on the thread count.
///
/// ```
/// use hessboost::config::Langevin;
///
/// # fn main() -> hessboost::error::Result<()> {
/// let langevin = Langevin::builder().diffusion_temperature(500.0).build()?;
/// assert_eq!(langevin.diffusion_temperature(), Some(500.0));
/// assert_eq!(Langevin::default().diffusion_temperature(), None); // 10000
/// assert!(Langevin::builder().diffusion_temperature(0.0).build().is_err());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Langevin {
    diffusion_temperature: Option<f64>,
}

impl Langevin {
    /// Start a builder at CatBoost's defaults (temperature unset).
    pub fn builder() -> LangevinBuilder {
        LangevinBuilder {
            langevin: Langevin::default(),
        }
    }

    /// The inverse diffusion temperature `T > 0` (CatBoost
    /// `diffusion_temperature`; larger is quieter). `None` takes CatBoost's
    /// `10000`, or the training row count under posterior sampling.
    pub fn diffusion_temperature(&self) -> Option<f64> {
        self.diffusion_temperature
    }
}

/// Builder of [`Langevin`].
#[derive(Debug, Clone, Copy)]
pub struct LangevinBuilder {
    langevin: Langevin,
}

impl LangevinBuilder {
    /// Set the inverse diffusion temperature (`diffusion_temperature`).
    #[must_use]
    pub fn diffusion_temperature(mut self, temperature: f64) -> Self {
        self.langevin.diffusion_temperature = Some(temperature);
        self
    }

    /// The validated settings.
    ///
    /// # Errors
    ///
    /// A diffusion temperature that is not finite and positive (`0` would
    /// switch off the noise Langevin asks for).
    pub fn build(self) -> Result<Langevin> {
        if let Some(t) = self.langevin.diffusion_temperature
            && !(t.is_finite() && t > 0.0)
        {
            return Err(HessboostError::invalid_param(
                "diffusion_temperature",
                format!("must be > 0, got {t}"),
            ));
        }
        Ok(self.langevin)
    }
}

/// How the model shrinkage coefficient of each boosting iteration is
/// computed (CatBoost `model_shrink_mode`). At the start of iteration
/// `i >= 1` the whole current model, intercept included, is multiplied by
/// the coefficient `s_i`; iteration `0` does not shrink.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ModelShrinkMode {
    /// `s_i = 1 - rate * eta` (CatBoost's default; the mode posterior
    /// sampling uses).
    #[default]
    Constant,
    /// `s_i = 1 - rate / i`.
    Decreasing,
}

/// Per-iteration model shrinkage (CatBoost `model_shrink_rate` and
/// `model_shrink_mode`): at the start of every iteration `i >= 1` the
/// current model (trees and intercept) is multiplied by `1 - rate * eta`
/// (constant) or `1 - rate / i` (decreasing). A rate of `0` is no
/// shrinkage (which turns off Langevin's default rate).
///
/// ```
/// use hessboost::config::{ModelShrink, ModelShrinkMode};
///
/// # fn main() -> hessboost::error::Result<()> {
/// let shrink = ModelShrink::builder()
///     .rate(0.1)
///     .mode(ModelShrinkMode::Decreasing)
///     .build()?;
/// assert_eq!((shrink.rate(), shrink.mode()), (0.1, ModelShrinkMode::Decreasing));
/// assert!(ModelShrink::builder().rate(-1.0).build().is_err());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ModelShrink {
    rate: f64,
    mode: ModelShrinkMode,
}

impl ModelShrink {
    /// Start a builder at rate `0` in the constant mode.
    pub fn builder() -> ModelShrinkBuilder {
        ModelShrinkBuilder {
            shrink: ModelShrink::default(),
        }
    }

    /// The shrinkage rate `>= 0` (CatBoost `model_shrink_rate`).
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// How the coefficient follows from the rate (CatBoost
    /// `model_shrink_mode`).
    pub fn mode(&self) -> ModelShrinkMode {
        self.mode
    }
}

/// Builder of [`ModelShrink`].
#[derive(Debug, Clone, Copy)]
pub struct ModelShrinkBuilder {
    shrink: ModelShrink,
}

impl ModelShrinkBuilder {
    /// Set the shrinkage rate (`model_shrink_rate`).
    #[must_use]
    pub fn rate(mut self, rate: f64) -> Self {
        self.shrink.rate = rate;
        self
    }

    /// Set how the coefficient is computed (`model_shrink_mode`).
    #[must_use]
    pub fn mode(mut self, mode: ModelShrinkMode) -> Self {
        self.shrink.mode = mode;
        self
    }

    /// The validated shrinkage. The constant mode's `rate * eta < 1` needs
    /// the learning rate and is checked by
    /// [`TrainingParams::validate`](super::TrainingParams::validate).
    ///
    /// # Errors
    ///
    /// A rate that is negative or not finite; in the decreasing mode, a
    /// rate outside `(0, 1)` (CatBoost's range, keeping every `1 - rate / i`
    /// positive).
    pub fn build(self) -> Result<ModelShrink> {
        let ModelShrink { rate, mode } = self.shrink;
        if !(rate.is_finite() && rate >= 0.0) {
            return Err(HessboostError::invalid_param(
                "model_shrink_rate",
                format!("must be >= 0, got {rate}"),
            ));
        }
        if mode == ModelShrinkMode::Decreasing && !(rate > 0.0 && rate < 1.0) {
            return Err(HessboostError::invalid_param(
                "model_shrink_rate",
                format!("must be in (0, 1) in the decreasing mode, got {rate}"),
            ));
        }
        Ok(self.shrink)
    }
}

/// LightGBM's query-level bagging for ranking (`bagging_by_query`): every
/// round keeps each query group whole with probability `fraction` (its
/// `bagging_fraction`, in `(0, 1)`), in place of `subsample`'s per-row draw.
///
/// ```
/// use hessboost::config::QueryBagging;
///
/// # fn main() -> hessboost::error::Result<()> {
/// assert_eq!(QueryBagging::new(0.8)?.fraction(), 0.8);
/// assert!(QueryBagging::new(1.0).is_err());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryBagging {
    fraction: f64,
}

impl QueryBagging {
    /// Keep each query with probability `fraction`.
    ///
    /// # Errors
    ///
    /// `fraction` outside `(0, 1)` (at `1` nothing is bagged: leave the
    /// option unset), named `bagging_by_query`.
    pub fn new(fraction: f64) -> Result<Self> {
        if !(fraction > 0.0 && fraction < 1.0) {
            return Err(HessboostError::invalid_param(
                "bagging_by_query",
                format!("the fraction of queries kept must be in (0, 1), got {fraction}"),
            ));
        }
        Ok(QueryBagging { fraction })
    }

    /// The probability of keeping a query.
    pub fn fraction(&self) -> f64 {
        self.fraction
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
        for t in [0.0, -1.0, f64::NAN] {
            assert_eq!(
                refused(&Langevin::builder().diffusion_temperature(t).build()),
                Some("diffusion_temperature")
            );
        }
        assert_eq!(
            refused(&ModelShrink::builder().rate(-0.1).build()),
            Some("model_shrink_rate")
        );
        for rate in [0.0, 1.0] {
            let decreasing = ModelShrink::builder()
                .rate(rate)
                .mode(ModelShrinkMode::Decreasing);
            assert_eq!(refused(&decreasing.build()), Some("model_shrink_rate"));
        }
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
        for fraction in [0.0, 1.0, 1.5, f64::NAN] {
            assert_eq!(
                refused(&QueryBagging::new(fraction)),
                Some("bagging_by_query")
            );
        }
    }
}
