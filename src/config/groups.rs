//! The option groups of [`TrainingParams`](super::TrainingParams): settings
//! that only mean something when a switch is on live inside that switch
//! (`BoosterKind::Dart(Dart)`, `BoosterKind::Boulevard(Boulevard)`,
//! `BoosterKind::Ebm(Ebm)` (with `Option<EbmEarlyStopping>`), `ProcessType::Update(Refresh)`,
//! `Option<QuantizedGrad>`, `Option<ExtraTrees>`, `Option<LinearTree>`,
//! `Option<BalancedBagging>`, `Option<QueryBagging>`, `Option<Langevin>`,
//! `Option<ModelShrink>`), so
//! they cannot be set while the switch is off. Each validates its values
//! when built.

use std::num::NonZeroUsize;

use crate::check::{ensure, fraction, non_negative, positive, unit};
use crate::error::Result;

/// DART's dropout (XGBoost `booster = dart`): each round drops every
/// existing tree with probability `rate_drop`, unless the whole dropout is
/// skipped (probability `skip_drop`); both are in `[0, 1]` and `0` by
/// default. When no tree is drawn, `one_drop` (default `false`) drops one
/// at random; otherwise nothing is dropped and the round's trees are not
/// normalized. At the defaults DART never drops a tree and trains exactly
/// like `gbtree`, as in XGBoost (`GBTree::DropTrees`).
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
    /// `rate_drop`.
    rate: f64,
    /// `skip_drop`.
    skip: f64,
    /// `one_drop`.
    force_one: bool,
}

impl Dart {
    /// Start a builder at XGBoost's defaults (`rate_drop = skip_drop = 0`,
    /// `one_drop = false`: no dropout).
    pub fn builder() -> DartBuilder {
        DartBuilder {
            dart: Dart::default(),
        }
    }

    /// Fraction of trees dropped each round. XGBoost `rate_drop`.
    pub fn rate_drop(&self) -> f64 {
        self.rate
    }

    /// Probability of skipping the dropout in a round. XGBoost `skip_drop`.
    pub fn skip_drop(&self) -> f64 {
        self.skip
    }

    /// Whether a round that draws no tree drops one at random. XGBoost
    /// `one_drop`.
    pub fn one_drop(&self) -> bool {
        self.force_one
    }

    /// Whether any round can drop a tree (XGBoost `HasDropout`); without it
    /// DART trains as `gbtree`.
    pub(crate) fn has_dropout(&self) -> bool {
        self.rate != 0.0 || self.force_one || self.skip != 0.0
    }
}

/// Builder of [`Dart`].
#[derive(Debug, Clone, Copy)]
pub struct DartBuilder {
    dart: Dart,
}

impl DartBuilder {
    setter!(
        /// Set the fraction of trees dropped each round (`rate_drop`).
        rate_drop: f64 => dart.rate
    );

    setter!(
        /// Set the probability of skipping the dropout (`skip_drop`).
        skip_drop: f64 => dart.skip
    );

    setter!(
        /// Set whether a round that draws no tree drops one at random
        /// (`one_drop`).
        one_drop: bool => dart.force_one
    );

    /// The validated dropout.
    ///
    /// # Errors
    ///
    /// `rate_drop` or `skip_drop` outside `[0, 1]`.
    pub fn build(self) -> Result<Dart> {
        unit("rate_drop", self.dart.rate)?;
        unit("skip_drop", self.dart.skip)?;
        Ok(self.dart)
    }
}

/// Boulevard boosting's settings (`booster = boulevard`, beyond XGBoost;
/// see [`crate::inference`]): BRAT-D's dropout probability `p` in `[0, 1)`
/// (each earlier tree left out of a round's residuals independently with
/// probability `p`, the kept ones still divided by the full tree count;
/// `0`, the default, is Zhou & Hooker's Boulevard, and BRAT-P with
/// `num_parallel_tree > 1` needs `0`), and the optional truncation level
/// `M > 0` of the ensemble part subtracted in a round's residuals (clipped
/// to `[-M, M]`, the `Γ_M` of Fang, Tan & Hooker's proofs; unset, the
/// default, is none).
///
/// ```
/// use hessboost::config::Boulevard;
///
/// # fn main() -> hessboost::error::Result<()> {
/// let boulevard = Boulevard::builder().dropout(0.5).build()?;
/// assert_eq!((boulevard.dropout(), boulevard.truncation()), (0.5, None));
/// assert!(Boulevard::builder().dropout(1.0).build().is_err());
/// // No truncation is no level, not a level of 0.
/// assert!(Boulevard::builder().truncation(0.0).build().is_err());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Boulevard {
    dropout: f64,
    truncation: Option<f64>,
}

impl Boulevard {
    /// Start a builder at the defaults (no dropout, no truncation).
    pub fn builder() -> BoulevardBuilder {
        BoulevardBuilder {
            boulevard: Boulevard::default(),
        }
    }

    /// BRAT-D's dropout probability `p` (`boulevard_dropout`).
    pub fn dropout(&self) -> f64 {
        self.dropout
    }

    /// The residual truncation level `M`, `None` for none
    /// (`boulevard_truncation`, whose `0` is none).
    pub fn truncation(&self) -> Option<f64> {
        self.truncation
    }
}

/// Builder of [`Boulevard`].
#[derive(Debug, Clone, Copy)]
pub struct BoulevardBuilder {
    boulevard: Boulevard,
}

impl BoulevardBuilder {
    setter!(
        /// Set BRAT-D's dropout probability (`boulevard_dropout`).
        dropout: f64 => boulevard.dropout
    );

    setter!(
        /// Set the residual truncation level `M > 0` (`boulevard_truncation`;
        /// leave it unset for none).
        truncation: f64 => Some(boulevard.truncation)
    );

    /// The validated settings.
    ///
    /// # Errors
    ///
    /// `dropout` outside `[0, 1)` or a `truncation` that is not finite and
    /// positive.
    pub fn build(self) -> Result<Boulevard> {
        let Boulevard {
            dropout,
            truncation,
        } = self.boulevard;
        ensure(
            "boulevard_dropout",
            dropout.is_finite() && (0.0..1.0).contains(&dropout),
            format!("must be in [0, 1), got {dropout}"),
        )?;
        if let Some(truncation) = truncation {
            ensure(
                "boulevard_truncation",
                truncation.is_finite() && truncation > 0.0,
                format!("must be finite and > 0 (leave it unset for none), got {truncation}"),
            )?;
        }
        Ok(self.boulevard)
    }
}

/// Largest [`Ebm::outer_bags`]: every bag keeps its margins over the
/// training rows while the bags train.
const MAX_EBM_OUTER_BAGS: usize = 1024;

/// Per-bag early stopping of a classic EBM ([`Ebm::early_stopping`]) on
/// the bag's held-out rows (the `1 − bag_fraction` it does not train on),
/// after InterpretML: after every tree the bag scores its held-out rows
/// with [`Trainer::custom_metric`](crate::training::Trainer::custom_metric)'s
/// metric when given, else the last eval metric, stops once no tree of the
/// last `rounds × terms` improved on the best score before them by the
/// relative `tolerance`, and keeps its trees up to its best score. Each
/// stage (main effects, pairs) stops separately.
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use hessboost::config::EbmEarlyStopping;
///
/// # fn main() -> hessboost::error::Result<()> {
/// let rounds = NonZeroUsize::new(50).unwrap();
/// let stopping = EbmEarlyStopping::new(rounds, EbmEarlyStopping::DEFAULT_TOLERANCE)?;
/// assert_eq!((stopping.rounds(), stopping.tolerance()), (rounds, 1e-5));
/// assert!(EbmEarlyStopping::new(rounds, f64::NAN).is_err());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EbmEarlyStopping {
    rounds: NonZeroUsize,
    tolerance: f64,
}

impl EbmEarlyStopping {
    /// InterpretML's `early_stopping_tolerance`.
    pub const DEFAULT_TOLERANCE: f64 = 1e-5;

    /// Stop after `rounds` rounds (`ebm_early_stopping_rounds`) without a
    /// relative improvement of `tolerance` (`ebm_early_stopping_tolerance`;
    /// negative values keep boosting through small losses).
    ///
    /// # Errors
    ///
    /// A tolerance that is not finite, named `ebm_early_stopping_tolerance`.
    pub fn new(rounds: NonZeroUsize, tolerance: f64) -> Result<Self> {
        ensure(
            "ebm_early_stopping_tolerance",
            tolerance.is_finite(),
            format!("must be finite, got {tolerance}"),
        )?;
        Ok(EbmEarlyStopping { rounds, tolerance })
    }

    /// The patience in rounds (`ebm_early_stopping_rounds`).
    pub fn rounds(&self) -> NonZeroUsize {
        self.rounds
    }

    /// The relative improvement required (`ebm_early_stopping_tolerance`).
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }
}

/// The settings of an explainable boosting machine (`booster = ebm`,
/// beyond XGBoost; see [`crate::ebm`]). Build with [`Ebm::builder`]; the
/// defaults are one bag of every row, main effects only, no early stopping,
/// and the classic cyclic EBM.
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use hessboost::config::{Ebm, EbmEarlyStopping};
///
/// # fn main() -> hessboost::error::Result<()> {
/// let stopping = EbmEarlyStopping::new(
///     NonZeroUsize::new(50).unwrap(),
///     EbmEarlyStopping::DEFAULT_TOLERANCE,
/// )?;
/// let ebm = Ebm::builder()
///     .interactions(2)
///     .outer_bags(8)
///     .bag_fraction(0.85)
///     .early_stopping(stopping)
///     .build()?;
/// assert_eq!((ebm.interactions(), ebm.outer_bags()), (2, 8));
/// assert_eq!(ebm.early_stopping(), Some(stopping));
/// // Early stopping scores each bag on the rows it does not train on.
/// assert!(Ebm::builder().early_stopping(stopping).build().is_err());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ebm {
    interactions: usize,
    outer_bags: usize,
    bag_fraction: f64,
    boulevard: bool,
    early_stopping: Option<EbmEarlyStopping>,
}

impl Default for Ebm {
    fn default() -> Self {
        Ebm {
            interactions: 0,
            outer_bags: 1,
            bag_fraction: 1.0,
            boulevard: false,
            early_stopping: None,
        }
    }
}

impl Ebm {
    /// Start a builder at the defaults.
    pub fn builder() -> EbmBuilder {
        EbmBuilder {
            ebm: Ebm::default(),
        }
    }

    /// Number of pairwise interaction terms added after the main effects:
    /// the top pairs by FAST (Lou et al., KDD 2013) on the main-effect
    /// model's gradients, each boosted like a main effect on its two
    /// features. At most `n_features (n_features − 1) / 2` (checked at
    /// training). `ebm_interactions`.
    pub fn interactions(&self) -> usize {
        self.interactions
    }

    /// Outer bags (InterpretML's `outer_bags`): each bag boosts its own copy
    /// of every term on its own row sample, and the model averages them.
    /// `ebm_outer_bags`.
    pub fn outer_bags(&self) -> usize {
        self.outer_bags
    }

    /// Fraction of the rows each outer bag trains on, drawn without
    /// replacement per bag (InterpretML trains each bag on `1 −
    /// validation_size = 0.85`). `ebm_bag_fraction`.
    pub fn bag_fraction(&self) -> f64 {
        self.bag_fraction
    }

    /// Boulevard-regularized EBM (Fang, Tan, Pipping & Hooker, AISTATS
    /// 2026, Algorithm 1): every round fits one tree per term to the same
    /// residuals, centers it, and averages it into its term with learning
    /// rate `eta ∈ (0, 1]`, so the terms converge to a feature-wise kernel
    /// ridge regression with confidence bands
    /// ([`EbmInference`](crate::inference::EbmInference)). Squared error
    /// only, with Boulevard's refusals, one bag of every row, and no early
    /// stopping. `ebm_boulevard`.
    pub fn boulevard(&self) -> bool {
        self.boulevard
    }

    /// Per-bag early stopping of a classic EBM ([`EbmEarlyStopping`]),
    /// `None` for none.
    pub fn early_stopping(&self) -> Option<EbmEarlyStopping> {
        self.early_stopping
    }
}

/// Builder of [`Ebm`].
#[derive(Debug, Clone, Copy)]
pub struct EbmBuilder {
    ebm: Ebm,
}

impl EbmBuilder {
    setter!(
        /// Set the number of pairwise interaction terms (`ebm_interactions`).
        interactions: usize => ebm.interactions
    );

    setter!(
        /// Set the number of outer bags (`ebm_outer_bags`).
        outer_bags: usize => ebm.outer_bags
    );

    setter!(
        /// Set the row fraction of each outer bag (`ebm_bag_fraction`).
        bag_fraction: f64 => ebm.bag_fraction
    );

    setter!(
        /// Boulevard-average the terms for inference (`ebm_boulevard`).
        boulevard: bool => ebm.boulevard
    );

    setter!(
        /// Stop each bag early on its held-out rows (`ebm_early_stopping_rounds`
        /// and `ebm_early_stopping_tolerance`).
        early_stopping: EbmEarlyStopping => Some(ebm.early_stopping)
    );

    /// The validated settings.
    ///
    /// # Errors
    ///
    /// `outer_bags` outside `[1, 1024]`, `bag_fraction` outside `(0, 1]`,
    /// early stopping without held-out rows (`bag_fraction = 1`) or with
    /// `boulevard`, and `boulevard` with several bags or a bag fraction
    /// below 1.
    pub fn build(self) -> Result<Ebm> {
        let e = self.ebm;
        ensure(
            "ebm_outer_bags",
            (1..=MAX_EBM_OUTER_BAGS).contains(&e.outer_bags),
            format!("must be in [1, {MAX_EBM_OUTER_BAGS}], got {}", e.outer_bags),
        )?;
        fraction("ebm_bag_fraction", e.bag_fraction)?;
        if e.early_stopping.is_some() {
            ensure(
                "ebm_early_stopping_rounds",
                !e.boulevard,
                "a Boulevard EBM averages every round, so it cannot stop at a best round",
            )?;
            ensure(
                "ebm_early_stopping_rounds",
                e.bag_fraction < 1.0,
                "each bag stops on the rows it does not train on; set `ebm_bag_fraction < 1`",
            )?;
        }
        ensure(
            "ebm_outer_bags",
            !e.boulevard || (e.outer_bags == 1 && e.bag_fraction == 1.0),
            "a bagged Boulevard EBM has no kernel ridge limit its inference covers; use one \
             bag of every row",
        )?;
        Ok(e)
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
    setter!(
        /// Set the quantization levels (`num_grad_quant_bins`).
        bins: usize => quantized.bins
    );

    setter!(
        /// Set stochastic rounding (`stochastic_rounding`).
        stochastic_rounding: bool => quantized.stochastic_rounding
    );

    setter!(
        /// Set full-precision leaf renewal (`quant_train_renew_leaf`).
        renew_leaf: bool => quantized.renew_leaf
    );

    /// The validated quantization.
    ///
    /// # Errors
    ///
    /// `bins` outside `[2, 127]` (LightGBM stores each value in 8 bits).
    pub fn build(self) -> Result<QuantizedGrad> {
        let bins = self.quantized.bins;
        ensure(
            "num_grad_quant_bins",
            (2..=127).contains(&bins),
            format!("must be in [2, 127], got {bins}"),
        )?;
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
        non_negative("linear_lambda", lambda)?;
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
    setter!(
        /// Set the inverse diffusion temperature (`diffusion_temperature`).
        diffusion_temperature: f64 => Some(langevin.diffusion_temperature)
    );

    /// The validated settings. The noise scale `sqrt(2 / (eta * T))` needs
    /// the learning rate: [`TrainingParams::validate`](super::TrainingParams::validate)
    /// refuses a temperature that makes it 0 or infinite in `f32`.
    ///
    /// # Errors
    ///
    /// A diffusion temperature that is not finite and positive (`0` would
    /// switch off the noise Langevin asks for).
    pub fn build(self) -> Result<Langevin> {
        if let Some(t) = self.langevin.diffusion_temperature {
            positive("diffusion_temperature", t)?;
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
/// (constant) or `1 - rate / i` (decreasing). No shrinkage is
/// [`TrainingParams::model_shrink`](super::TrainingParams::model_shrink)
/// `None` (CatBoost's rate `0`).
///
/// ```
/// use hessboost::config::{ModelShrink, ModelShrinkMode};
///
/// # fn main() -> hessboost::error::Result<()> {
/// let shrink = ModelShrink::new(0.1, ModelShrinkMode::Decreasing)?;
/// assert_eq!((shrink.rate(), shrink.mode()), (0.1, ModelShrinkMode::Decreasing));
/// // No shrinkage is no `ModelShrink`, not a rate of 0.
/// assert!(ModelShrink::new(0.0, ModelShrinkMode::Constant).is_err());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelShrink {
    rate: f64,
    mode: ModelShrinkMode,
}

impl ModelShrink {
    /// Shrink at `rate` in `mode`. The constant mode's `rate * eta < 1`
    /// needs the learning rate and is checked by
    /// [`TrainingParams::validate`](super::TrainingParams::validate).
    ///
    /// # Errors
    ///
    /// A rate that is not finite and positive (leave the option unset for
    /// no shrinkage); in the decreasing mode, a rate outside `(0, 1)`
    /// (CatBoost's range, keeping every `1 - rate / i` positive). Named
    /// `model_shrink_rate`.
    pub fn new(rate: f64, mode: ModelShrinkMode) -> Result<Self> {
        ensure(
            "model_shrink_rate",
            rate.is_finite() && rate > 0.0,
            format!("must be > 0 (leave model shrinkage unset for none), got {rate}"),
        )?;
        ensure(
            "model_shrink_rate",
            mode != ModelShrinkMode::Decreasing || rate < 1.0,
            format!("must be in (0, 1) in the decreasing mode, got {rate}"),
        )?;
        Ok(ModelShrink { rate, mode })
    }

    /// The shrinkage rate `> 0` (CatBoost `model_shrink_rate`).
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// How the coefficient follows from the rate (CatBoost
    /// `model_shrink_mode`).
    pub fn mode(&self) -> ModelShrinkMode {
        self.mode
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
        ensure(
            "bagging_by_query",
            fraction > 0.0 && fraction < 1.0,
            format!("the fraction of queries kept must be in (0, 1), got {fraction}"),
        )?;
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
        ensure(
            "pos_bagging_fraction",
            pos != 1.0 || neg != 1.0,
            "with `neg_bagging_fraction` also 1 nothing is bagged; \
             leave balanced bagging unset",
        )?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::HessboostError;

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
        for rate in [-0.1, 0.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                refused(&ModelShrink::new(rate, ModelShrinkMode::Constant)),
                Some("model_shrink_rate")
            );
        }
        assert_eq!(
            refused(&ModelShrink::new(1.0, ModelShrinkMode::Decreasing)),
            Some("model_shrink_rate")
        );
        assert!(ModelShrink::new(1.0, ModelShrinkMode::Constant).is_ok());
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
