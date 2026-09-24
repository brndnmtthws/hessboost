//! Training configuration.
//!
//! Parameter names and default values deliberately mirror XGBoost so that
//! existing knowledge and configurations transfer directly. Where XGBoost
//! exposes aliases (e.g. `eta`/`learning_rate`), we pick the canonical field
//! name and document the alias.

use crate::error::{HessboostError, Result};
use serde::{Deserialize, Serialize};

/// Which booster to use in the ensemble.
///
/// Mirrors XGBoost's `booster` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum BoosterKind {
    /// Gradient boosted trees (XGBoost `gbtree`).
    #[default]
    GbTree,
    /// Dropout Additive Regression Trees (XGBoost `dart`).
    Dart,
    /// Linear booster with coordinate descent (XGBoost `gblinear`).
    GbLinear,
}

/// Tree construction algorithm.
///
/// Mirrors XGBoost's `tree_method`. `Auto` resolves to [`TreeMethod::Hist`] for
/// all but the smallest datasets, matching modern XGBoost behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TreeMethod {
    /// Pick automatically based on dataset size.
    #[default]
    Auto,
    /// Exact greedy algorithm (enumerate every split candidate).
    Exact,
    /// Approximate algorithm using weighted quantile sketch per split.
    Approx,
    /// Fast histogram algorithm with pre-binned features.
    Hist,
}

/// Order in which the tree is grown.
///
/// Mirrors XGBoost's `grow_policy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum GrowPolicy {
    /// Split nodes closest to the root first (level-wise). XGBoost default.
    #[default]
    DepthWise,
    /// Split nodes with the highest loss reduction first (leaf-wise).
    LossGuide,
    /// Symmetric (oblivious) trees, CatBoost-style: every level applies one
    /// shared split (feature, threshold, missing direction) chosen to maximize
    /// the summed gain over the level's nodes. Beyond XGBoost (opt-in). Needs
    /// a tree booster (`gbtree` or `dart`), `tree_method = hist` or `approx`,
    /// numerical features only, `max_depth`
    /// in `1..=`[`MAX_SYMMETRIC_DEPTH`], and `max_leaves = 0`. A node whose
    /// level split would violate `min_child_weight`, `gamma`, or a monotone
    /// constraint stays a leaf. The trees are ordinary [`RegTree`]s, so they
    /// export to XGBoost unchanged; prediction routes rows through them by
    /// bit pattern.
    ///
    /// [`RegTree`]: crate::tree::RegTree
    Symmetric,
}

/// Deepest tree `grow_policy = symmetric` grows (`2^16` leaves), CatBoost's
/// depth limit.
pub const MAX_SYMMETRIC_DEPTH: usize = 16;

/// Per-feature monotonicity direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Monotone {
    /// No constraint on this feature.
    #[default]
    None,
    /// Prediction must be non-decreasing in this feature.
    Increasing,
    /// Prediction must be non-increasing in this feature.
    Decreasing,
}

/// Noise distribution of the accelerated-failure-time survival loss.
///
/// Mirrors XGBoost's `aft_loss_distribution`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AftDistribution {
    /// Normal (Gaussian) noise. XGBoost default.
    #[default]
    Normal,
    /// Logistic noise.
    Logistic,
    /// Type-1 extreme-value (Gumbel minimum) noise.
    Extreme,
}

/// How rows are subsampled each round.
///
/// Mirrors XGBoost's `sampling_method`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SamplingMethod {
    /// Every row is kept with probability `subsample`. XGBoost default.
    #[default]
    Uniform,
    /// Minimal-variance sampling (XGBoost `gradient_based`): each tree keeps
    /// row `i` with probability `min(1, sqrt(g_i^2 + 0.1 h_i^2) / u)`, where
    /// `u` makes the expected kept count `trunc(n * subsample)`, and scales a
    /// kept row's gradient and Hessian by the inverse of that probability.
    /// Supported by `tree_method = hist | approx | auto`; `exact` rejects it
    /// when `subsample < 1`.
    GradientBased,
}

/// How multi-target and multiclass models allocate outputs to trees.
///
/// Mirrors XGBoost's `multi_strategy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MultiStrategy {
    /// One tree per output each round. XGBoost default.
    #[default]
    OneOutputPerTree,
    /// One tree per round whose leaves hold a vector of all outputs
    /// (vector-leaf trees; `tree_method = hist` only). With a single output
    /// it trains scalar trees, like XGBoost.
    MultiOutputTree,
}

/// Whether a round grows new trees or updates existing ones.
///
/// Mirrors XGBoost's `process_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ProcessType {
    /// Grow new trees. XGBoost default.
    #[default]
    Default,
    /// Revisit the trees of an existing model instead of growing new ones.
    Update,
}

/// The second-order statistic the `dist:*` distributional objectives give
/// the trees (beyond XGBoost; see [`crate::objective::distributional`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DistGradient {
    /// Gradient of the negative log-likelihood with the diagonal Fisher
    /// information as Hessian (Fisher scoring; a natural-gradient Newton
    /// step for the orthogonal parameterizations used).
    #[default]
    Fisher,
    /// Gradient with the diagonal of the exact (observed) Hessian, floored
    /// at `1e-16` (XGBoostLSS-style).
    Hessian,
    /// NGBoost's natural gradient `I⁻¹ ∇` with unit Hessian: trees regress
    /// the natural gradient by least squares.
    Natural,
}

/// How the shared tree of a `dist:*` objective chooses its structure under
/// `multi_strategy = multi_output_tree` (beyond XGBoost; see
/// [`crate::objective::distributional`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DistSplitDirection {
    /// Parallel gradient boosting (Chapelle et al., 2026, Algorithm 1): each
    /// round grows the structure from the gradients of one distribution
    /// parameter drawn uniformly at random (seeded by `seed` and the
    /// iteration), a canonical descent direction `e_m`.
    #[default]
    Random,
    /// Parallel gradient boosting with a deterministic sweep: parameter
    /// `iteration mod n_params` drives round `iteration`.
    Cyclic,
    /// Plain vector-leaf trees: the split gain sums over every parameter.
    All,
}

/// Stable names of the enums a model file stores, matching their serde
/// (and XGBoost) spellings.
macro_rules! stored_names {
    ($($ty:ident { $($variant:ident => $name:literal),+ $(,)? })+) => {$(
        impl $ty {
            /// The variant's stored name.
            pub(crate) fn name(self) -> &'static str {
                match self {
                    $($ty::$variant => $name,)+
                }
            }

            /// The variant stored as `name`, if any.
            pub(crate) fn from_name(name: &str) -> Option<Self> {
                match name {
                    $($name => Some($ty::$variant),)+
                    _ => None,
                }
            }
        }
    )+};
}

stored_names! {
    AftDistribution { Normal => "normal", Logistic => "logistic", Extreme => "extreme" }
    DistGradient { Fisher => "fisher", Hessian => "hessian", Natural => "natural" }
    DistSplitDirection { Random => "random", Cyclic => "cyclic", All => "all" }
}

/// The complete training configuration.
///
/// Construct with [`TrainingParams::builder`] or start from
/// [`TrainingParams::default`] and mutate fields directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent XGBoost/LightGBM switches, not a state machine"
)]
pub struct TrainingParams {
    // ---- General ----
    /// Which booster to train. XGBoost `booster`.
    pub booster: BoosterKind,
    /// Number of worker threads. `0` uses the global Rayon pool. XGBoost `nthread`.
    pub nthread: usize,
    /// RNG seed for subsampling and column sampling. XGBoost `seed`.
    pub seed: u64,

    // ---- Learning task ----
    /// Objective function name, e.g. `"reg:squarederror"`, `"binary:logistic"`.
    /// XGBoost `objective`.
    pub objective: String,
    /// Number of classes for multiclass objectives. XGBoost `num_class`.
    pub num_class: usize,
    /// Global bias / initial prediction (in probability space where applicable).
    /// `None` means "estimate from the labels", matching modern XGBoost.
    /// XGBoost `base_score`.
    pub base_score: Option<f64>,
    /// Evaluation metric names. Empty means "use the objective's default".
    /// XGBoost `eval_metric`.
    pub eval_metric: Vec<String>,

    // ---- Objective-specific ----
    /// Variance power of the Tweedie distribution for `reg:tweedie`, in
    /// `[1, 2)` (1 = Poisson, 2 = Gamma). XGBoost `tweedie_variance_power`.
    pub tweedie_variance_power: f64,
    /// Slope `δ` of the pseudo-Huber loss for `reg:pseudohubererror` and the
    /// `mphe` metric (which rejects `0`). XGBoost `huber_slope`.
    pub huber_slope: f64,
    /// Number of top-ranked documents paired with all lower-ranked documents
    /// by LambdaRank's `topk` pair method. XGBoost
    /// `lambdarank_num_pair_per_sample` (default 32 for `topk`).
    pub lambdarank_num_pair_per_sample: usize,
    /// Target quantiles for `reg:quantileerror`: non-empty, ascending, each in
    /// `[0, 1]`. XGBoost `quantile_alpha`.
    pub quantile_alpha: Vec<f64>,
    /// Target expectiles for `reg:expectileerror`: non-empty, ascending, each
    /// in `[0, 1]`. XGBoost `expectile_alpha`.
    pub expectile_alpha: Vec<f64>,
    /// Noise distribution for `survival:aft`. XGBoost `aft_loss_distribution`.
    pub aft_loss_distribution: AftDistribution,
    /// Scale of the `survival:aft` noise distribution (`> 0` and finite, also
    /// once rounded to the `f32` the objective computes in).
    /// XGBoost `aft_loss_distribution_scale`.
    pub aft_loss_distribution_scale: f64,
    /// Second-order statistic of the `dist:*` objectives (beyond XGBoost):
    /// Fisher information (default), exact Hessian, or NGBoost natural
    /// gradient. Ignored by every other objective.
    pub dist_gradient: DistGradient,
    /// Split direction of the shared (vector-leaf) trees of the `dist:*`
    /// objectives with `multi_strategy = multi_output_tree` (beyond XGBoost):
    /// parallel gradient boosting on a random (default) or cyclic parameter,
    /// or the full vector-leaf gain. Ignored otherwise.
    pub dist_split_direction: DistSplitDirection,

    // ---- Tree booster ----
    /// Learning rate / step-size shrinkage. XGBoost `eta` / `learning_rate`.
    pub eta: f64,
    /// Minimum loss reduction to make a split. XGBoost `gamma` / `min_split_loss`.
    pub gamma: f64,
    /// Maximum tree depth (`0` = no limit). XGBoost `max_depth`.
    pub max_depth: usize,
    /// Maximum number of leaves for `LossGuide` growth (`0` = no limit).
    /// XGBoost `max_leaves`.
    pub max_leaves: usize,
    /// Minimum sum of instance hessian needed in a child. XGBoost `min_child_weight`.
    pub min_child_weight: f64,
    /// Maximum delta step allowed for each leaf weight; `Some(0.0)` means no
    /// constraint. `None` leaves XGBoost's objective-dependent default: `0.7`
    /// for `count:poisson` (where the same value also stabilizes the Poisson
    /// Hessian), otherwise unconstrained. XGBoost `max_delta_step`.
    pub max_delta_step: Option<f64>,
    /// Row subsample ratio per boosting round. XGBoost `subsample`.
    pub subsample: f64,
    /// Column subsample ratio per tree. XGBoost `colsample_bytree`.
    pub colsample_bytree: f64,
    /// Column subsample ratio per level. XGBoost `colsample_bylevel`.
    pub colsample_bylevel: f64,
    /// Column subsample ratio per node. XGBoost `colsample_bynode`.
    pub colsample_bynode: f64,
    /// L2 regularization on leaf weights. XGBoost `lambda` / `reg_lambda`.
    pub lambda: f64,
    /// L1 regularization on leaf weights. XGBoost `alpha` / `reg_alpha`.
    pub alpha: f64,
    /// Balancing of positive/negative weights for imbalanced binary problems.
    /// XGBoost `scale_pos_weight`.
    pub scale_pos_weight: f64,
    /// Tree construction algorithm. XGBoost `tree_method`.
    pub tree_method: TreeMethod,
    /// Tree growth order. XGBoost `grow_policy`.
    pub grow_policy: GrowPolicy,
    /// Maximum number of histogram bins per feature. XGBoost `max_bin`.
    pub max_bin: usize,
    /// Per-feature monotone constraints (empty = none). XGBoost `monotone_constraints`.
    pub monotone_constraints: Vec<Monotone>,
    /// Allowed feature-interaction groups (empty = none). Each inner vector lists
    /// feature indices permitted to appear together on a single root-to-leaf path.
    /// XGBoost `interaction_constraints`.
    pub interaction_constraints: Vec<Vec<u32>>,
    /// Trees grown per output per round (boosted random forests; `>= 1`).
    /// XGBoost `num_parallel_tree`.
    pub num_parallel_tree: usize,
    /// Row subsampling method. XGBoost `sampling_method`.
    pub sampling_method: SamplingMethod,
    /// Output-to-tree allocation for multi-output models. XGBoost
    /// `multi_strategy`.
    pub multi_strategy: MultiStrategy,
    /// Grow new trees or update existing ones. XGBoost `process_type`.
    pub process_type: ProcessType,
    /// With `process_type = update`, whether the refresh updater also
    /// rewrites leaf values (not only node statistics). XGBoost
    /// `refresh_leaf`.
    pub refresh_leaf: bool,

    // ---- LightGBM tree options (opt-in, beyond XGBoost) ----
    /// Extremely randomized split search (LightGBM `extra_trees`): every
    /// numerical feature is scored at one random bin boundary per node, drawn
    /// uniformly between the node's lowest and highest occupied bin, and every
    /// categorical feature at one random prefix of its gradient-ordered
    /// categories. Requires the histogram builder (`hist`/`approx`).
    pub extra_trees: bool,
    /// Seed of the [`extra_trees`](Self::extra_trees) threshold draws,
    /// combined with the per-tree seed derived from [`seed`](Self::seed).
    /// LightGBM `extra_seed` (default `6`).
    pub extra_seed: u64,
    /// Path smoothing strength `s >= 0` (LightGBM `path_smooth`, `0` = off).
    /// Each child's output is pulled toward its parent's:
    /// `w = w_raw·(n/s)/(n/s + 1) + w_parent/(n/s + 1)` with `n` the child's
    /// row count, and splits are scored at the smoothed outputs. Requires the
    /// histogram builder (`hist`/`approx`).
    pub path_smooth: f64,
    /// Fit a ridge-regularized linear model in every leaf (LightGBM
    /// `linear_tree`) on the numerical features split on along the leaf's
    /// path; rows with a missing value in any of them predict the constant
    /// leaf value. The first boosting round keeps constant leaves. Requires
    /// the histogram builder (`hist`/`approx`).
    pub linear_tree: bool,
    /// L2 penalty on the leaf linear models' slopes (not their intercepts),
    /// `>= 0`. LightGBM `linear_lambda`.
    pub linear_lambda: f64,
    // ---- Quantized training (LightGBM; beyond XGBoost) ----
    /// Train on gradients and Hessians quantized to small integers with
    /// integer histograms (LightGBM `use_quantized_grad`; Shi et al., NeurIPS
    /// 2022). Opt-in and not part of XGBoost: trees differ from
    /// full-precision training. Needs `tree_method` `hist`/`approx` (or
    /// `auto`) and a tree booster.
    pub use_quantized_grad: bool,
    /// Quantization levels `Q` for [`use_quantized_grad`](Self::use_quantized_grad):
    /// gradients map to integers in `[-⌊Q/2⌋, ⌊Q/2⌋]`, non-negative Hessians
    /// to `[0, Q]`. In `[2, 127]` (LightGBM stores each value in 8 bits).
    /// LightGBM `num_grad_quant_bins`.
    pub num_grad_quant_bins: usize,
    /// Round quantized gradients stochastically (unbiased) rather than to the
    /// nearest level. Only used with `use_quantized_grad`. LightGBM
    /// `stochastic_rounding`.
    pub stochastic_rounding: bool,
    /// Recompute each leaf value from the full-precision gradients of its rows
    /// once a quantized tree is grown. Only used with `use_quantized_grad`;
    /// refused with [`path_smooth`](Self::path_smooth), whose leaves keep the
    /// outputs their splits recorded. LightGBM `quant_train_renew_leaf`.
    pub quant_train_renew_leaf: bool,

    // ---- DART-specific ----
    /// Fraction of trees to drop each round (DART). XGBoost `rate_drop`.
    pub rate_drop: f64,
    /// Probability of skipping dropout in a round (DART). XGBoost `skip_drop`.
    pub skip_drop: f64,

    // ---- Compact training (Trees on a Diet; beyond XGBoost, opt-in) ----
    /// Penalty `ι` subtracted from the loss change of a split on a feature the
    /// ensemble does not use yet (Herrmann et al., *Boosted Trees on a Diet*,
    /// ICLR 2026, eq. 3). Same units as [`gamma`](Self::gamma); `0` (the
    /// default) disables it. Pair with
    /// [`BoostedModel::to_compact_bytes`](crate::model::BoostedModel::to_compact_bytes),
    /// whose dictionaries shrink as features and thresholds are reused. The
    /// paper's `toad_penalty_feature`.
    pub toad_penalty_feature: f64,
    /// Penalty `ξ` subtracted from the loss change of a split at a threshold
    /// (or categorical left set) not yet used for its feature anywhere in the
    /// ensemble; a new feature pays both penalties. Same units as
    /// [`gamma`](Self::gamma); `0` (the default) disables it. The paper's
    /// `toad_penalty_threshold`.
    pub toad_penalty_threshold: f64,

    // ---- Missing value ----
    /// Value treated as "missing" in dense inputs. Defaults to NaN, like XGBoost.
    pub missing: f64,
}

impl Default for TrainingParams {
    fn default() -> Self {
        TrainingParams {
            booster: BoosterKind::GbTree,
            nthread: 0,
            seed: 0,
            objective: "reg:squarederror".to_string(),
            num_class: 0,
            base_score: None,
            eval_metric: Vec::new(),
            tweedie_variance_power: 1.5,
            huber_slope: 1.0,
            lambdarank_num_pair_per_sample: 32,
            quantile_alpha: Vec::new(),
            expectile_alpha: Vec::new(),
            aft_loss_distribution: AftDistribution::Normal,
            aft_loss_distribution_scale: 1.0,
            dist_gradient: DistGradient::Fisher,
            dist_split_direction: DistSplitDirection::Random,
            eta: 0.3,
            gamma: 0.0,
            max_depth: 6,
            max_leaves: 0,
            min_child_weight: 1.0,
            max_delta_step: None,
            subsample: 1.0,
            colsample_bytree: 1.0,
            colsample_bylevel: 1.0,
            colsample_bynode: 1.0,
            lambda: 1.0,
            alpha: 0.0,
            scale_pos_weight: 1.0,
            tree_method: TreeMethod::Auto,
            grow_policy: GrowPolicy::DepthWise,
            max_bin: 256,
            monotone_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            num_parallel_tree: 1,
            sampling_method: SamplingMethod::Uniform,
            multi_strategy: MultiStrategy::OneOutputPerTree,
            process_type: ProcessType::Default,
            refresh_leaf: true,
            extra_trees: false,
            extra_seed: 6,
            path_smooth: 0.0,
            linear_tree: false,
            linear_lambda: 0.0,
            use_quantized_grad: false,
            num_grad_quant_bins: 4,
            stochastic_rounding: true,
            quant_train_renew_leaf: false,
            rate_drop: 0.0,
            skip_drop: 0.0,
            toad_penalty_feature: 0.0,
            toad_penalty_threshold: 0.0,
            missing: f64::NAN,
        }
    }
}

/// Fail with [`HessboostError::invalid_param`] unless `ok`. Shared by the range
/// checks in [`TrainingParams::validate`].
fn ensure(name: &'static str, ok: bool, reason: impl Into<String>) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(HessboostError::invalid_param(name, reason))
    }
}

/// [`ensure`] that `v` is finite and in `[0, 1]`.
fn unit(name: &'static str, v: f64) -> Result<()> {
    ensure(
        name,
        v.is_finite() && (0.0..=1.0).contains(&v),
        format!("must be in [0, 1], got {v}"),
    )
}

/// [`ensure`] that `v` is finite and `> 0`.
fn positive(name: &'static str, v: f64) -> Result<()> {
    ensure(
        name,
        v.is_finite() && v > 0.0,
        format!("must be > 0, got {v}"),
    )
}

/// [`ensure`] that `v` is finite and `>= 0`.
fn non_negative(name: &'static str, v: f64) -> Result<()> {
    ensure(
        name,
        v.is_finite() && v >= 0.0,
        format!("must be >= 0, got {v}"),
    )
}

impl TrainingParams {
    /// Start a builder for ergonomic, chained configuration.
    pub fn builder() -> TrainingParamsBuilder {
        TrainingParamsBuilder {
            params: TrainingParams::default(),
        }
    }

    /// Validate mutually-consistent ranges. Called automatically before training.
    pub fn validate(&self) -> Result<()> {
        positive("eta", self.eta)?;
        non_negative("gamma", self.gamma)?;
        non_negative("min_child_weight", self.min_child_weight)?;
        if let Some(max_delta_step) = self.max_delta_step {
            non_negative("max_delta_step", max_delta_step)?;
        }
        non_negative("lambda", self.lambda)?;
        non_negative("alpha", self.alpha)?;
        positive("scale_pos_weight", self.scale_pos_weight)?;
        unit("subsample", self.subsample)?;
        // subsample of exactly 0 is meaningless.
        ensure("subsample", self.subsample != 0.0, "must be > 0")?;
        unit("colsample_bytree", self.colsample_bytree)?;
        unit("colsample_bylevel", self.colsample_bylevel)?;
        unit("colsample_bynode", self.colsample_bynode)?;
        unit("rate_drop", self.rate_drop)?;
        unit("skip_drop", self.skip_drop)?;
        non_negative("toad_penalty_feature", self.toad_penalty_feature)?;
        non_negative("toad_penalty_threshold", self.toad_penalty_threshold)?;
        ensure(
            "toad_penalty_feature",
            self.booster != BoosterKind::GbLinear
                || (self.toad_penalty_feature == 0.0 && self.toad_penalty_threshold == 0.0),
            "reuse penalties need a tree booster (`gbtree` or `dart`)",
        )?;
        // The penalties act in the XGBoost histogram/exact split searches;
        // the LightGBM split search and symmetric level-wise growth do not
        // apply them, so refuse the combination instead of ignoring it.
        let reuse_on = self.toad_penalty_feature > 0.0 || self.toad_penalty_threshold > 0.0;
        ensure(
            "toad_penalty_feature",
            !(reuse_on
                && (self.extra_trees
                    || self.path_smooth > 0.0
                    || self.grow_policy == GrowPolicy::Symmetric)),
            "reuse penalties are not supported with `extra_trees`, `path_smooth`, or \
             `grow_policy=symmetric`",
        )?;

        if let Some(base_score) = self.base_score {
            ensure("base_score", base_score.is_finite(), "must be finite")?;
        }
        // The objectives run in `f32`: validate the narrowed values so a
        // configuration cannot pass here and leave the documented range once
        // it reaches the objective.
        let rho = self.tweedie_variance_power as f32;
        ensure(
            "tweedie_variance_power",
            self.tweedie_variance_power.is_finite() && (1.0f32..2.0).contains(&rho),
            format!(
                "must be in [1, 2) (as f32), got {}",
                self.tweedie_variance_power
            ),
        )?;
        positive("huber_slope", self.huber_slope)?;
        let slope_sq = (self.huber_slope as f32) * (self.huber_slope as f32);
        ensure(
            "huber_slope",
            slope_sq.is_finite() && slope_sq > 0.0,
            format!(
                "squared slope must stay positive and finite in f32, got {}",
                self.huber_slope
            ),
        )?;
        ensure(
            "lambdarank_num_pair_per_sample",
            self.lambdarank_num_pair_per_sample >= 1,
            "must be >= 1",
        )?;
        positive(
            "aft_loss_distribution_scale",
            self.aft_loss_distribution_scale,
        )?;
        let aft_scale = self.aft_loss_distribution_scale as f32;
        ensure(
            "aft_loss_distribution_scale",
            aft_scale.is_finite() && aft_scale > 0.0,
            format!(
                "must stay positive and finite in f32, got {}",
                self.aft_loss_distribution_scale
            ),
        )?;
        ensure(
            "num_parallel_tree",
            self.num_parallel_tree >= 1,
            "must be >= 1",
        )?;

        ensure(
            "max_bin",
            self.max_bin >= 2,
            format!("must be >= 2, got {}", self.max_bin),
        )?;
        ensure(
            "max_leaves",
            !(self.grow_policy == GrowPolicy::LossGuide
                && self.max_leaves == 0
                && self.max_depth == 0),
            "lossguide growth needs a bound: set max_leaves or max_depth > 0",
        )?;
        if self.grow_policy == GrowPolicy::Symmetric {
            ensure(
                "grow_policy",
                self.booster != BoosterKind::GbLinear,
                "`symmetric` growth needs a tree booster (`gbtree` or `dart`)",
            )?;
            ensure(
                "max_depth",
                (1..=MAX_SYMMETRIC_DEPTH).contains(&self.max_depth),
                format!(
                    "symmetric growth needs 1 <= max_depth <= {MAX_SYMMETRIC_DEPTH}, got {}",
                    self.max_depth
                ),
            )?;
            ensure(
                "max_leaves",
                self.max_leaves == 0,
                "symmetric growth sizes trees by max_depth; max_leaves must be 0",
            )?;
        }
        ensure(
            "num_grad_quant_bins",
            (2..=127).contains(&self.num_grad_quant_bins),
            format!("must be in [2, 127], got {}", self.num_grad_quant_bins),
        )?;
        if self.multi_strategy == MultiStrategy::MultiOutputTree {
            // The vector-leaf builder has its own (XGBoost) split search:
            // symmetric level-wise growth and the reuse penalties do not
            // reach it.
            ensure(
                "grow_policy",
                self.grow_policy != GrowPolicy::Symmetric,
                "`symmetric` growth is not supported with `multi_strategy=multi_output_tree`",
            )?;
            ensure(
                "toad_penalty_feature",
                !reuse_on,
                "reuse penalties are not supported with `multi_strategy=multi_output_tree`",
            )?;
        }
        if self.use_quantized_grad {
            ensure(
                "use_quantized_grad",
                self.tree_method != TreeMethod::Exact && self.booster != BoosterKind::GbLinear,
                "quantized training needs a tree booster with `tree_method` hist, approx or auto",
            )?;
            ensure(
                "use_quantized_grad",
                self.multi_strategy == MultiStrategy::OneOutputPerTree,
                "quantized training grows one-output trees only",
            )?;
            // Symmetric growth builds its level histograms outside the
            // quantized node path, so the setting would be silently ignored.
            ensure(
                "use_quantized_grad",
                self.grow_policy != GrowPolicy::Symmetric,
                "quantized training is not supported with `grow_policy=symmetric`",
            )?;
            // Path-smoothed leaves keep the outputs their (quantized) splits
            // recorded, so renewed leaf statistics would be discarded.
            ensure(
                "quant_train_renew_leaf",
                !(self.quant_train_renew_leaf && self.path_smooth > 0.0),
                "leaf renewal is not supported with `path_smooth`",
            )?;
        }
        self.validate_tree_options()
    }

    /// Range and compatibility checks of the opt-in LightGBM tree options
    /// ([`extra_trees`](Self::extra_trees), [`path_smooth`](Self::path_smooth),
    /// [`linear_tree`](Self::linear_tree)). They act inside the histogram tree
    /// builder only, so every other booster, builder, or tree layout is
    /// refused instead of silently ignoring them. The split-search options
    /// live in the per-node histogram split search, which symmetric growth
    /// replaces with its level-wise search, so they are refused there too;
    /// linear leaves are fitted after growth and apply to symmetric trees.
    fn validate_tree_options(&self) -> Result<()> {
        non_negative("path_smooth", self.path_smooth)?;
        non_negative("linear_lambda", self.linear_lambda)?;
        // The compatibility checks do not depend on the option, so the first
        // enabled one names the error.
        let enabled = [
            ("extra_trees", self.extra_trees),
            ("path_smooth", self.path_smooth > 0.0),
            ("linear_tree", self.linear_tree),
        ];
        if let Some(&(name, _)) = enabled.iter().find(|&&(_, on)| on) {
            ensure(
                name,
                self.booster != BoosterKind::GbLinear,
                "requires a tree booster (`gbtree` or `dart`)",
            )?;
            ensure(
                name,
                self.tree_method != TreeMethod::Exact,
                "requires the histogram tree builder (`tree_method` `hist`, `approx` or `auto`)",
            )?;
            ensure(
                name,
                self.multi_strategy == MultiStrategy::OneOutputPerTree,
                "is not supported with `multi_strategy=multi_output_tree`",
            )?;
        }
        if let Some(&(name, _)) = enabled[..2].iter().find(|&&(_, on)| on) {
            ensure(
                name,
                self.grow_policy != GrowPolicy::Symmetric,
                "is not supported with `grow_policy=symmetric` (level-wise split search)",
            )?;
        }
        // LightGBM refuses `regression_l1` with linear trees: objectives whose
        // leaves are re-estimated after growth (XGBoost's adaptive leaves)
        // would overwrite the constant that linear leaves fall back to.
        ensure(
            "linear_tree",
            !(self.linear_tree
                && matches!(
                    self.objective.as_str(),
                    "reg:absoluteerror" | "reg:quantileerror"
                )),
            format!(
                "is not supported with the adaptive-leaf objective `{}`",
                self.objective
            ),
        )
    }

    /// The `max_delta_step` in effect: the configured value, or XGBoost's
    /// default when unset (`0.7` for `count:poisson`, which XGBoost's learner
    /// injects before configuring the objective and tree updater; `0`,
    /// unconstrained, otherwise).
    pub(crate) fn effective_max_delta_step(&self) -> f64 {
        self.max_delta_step
            .unwrap_or(if self.objective == "count:poisson" {
                0.7
            } else {
                0.0
            })
    }
}

/// The objective hyper-parameters a trained [`BoostedModel`](crate::model::BoostedModel)
/// retains. XGBoost saves them in the model's `objective` block, so they are
/// needed to write an XGBoost-format model faithfully and to rebuild the
/// objective when predicting. Tree-construction parameters are not retained.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveParams {
    /// XGBoost `scale_pos_weight` (`reg_loss_param`).
    pub scale_pos_weight: f64,
    /// The effective `max_delta_step` (`poisson_regression_param`): the
    /// configured [`TrainingParams::max_delta_step`], or `0.7` for
    /// `count:poisson` and `0` otherwise when unset.
    pub max_delta_step: f64,
    /// XGBoost `tweedie_variance_power` (`tweedie_regression_param`).
    pub tweedie_variance_power: f64,
    /// XGBoost `huber_slope` (`pseudo_huber_param`).
    pub huber_slope: f64,
    /// XGBoost `lambdarank_num_pair_per_sample` (`lambdarank_param`).
    pub lambdarank_num_pair_per_sample: usize,
    /// XGBoost `quantile_alpha` (`reg:quantileerror`).
    pub quantile_alpha: Vec<f64>,
    /// XGBoost `expectile_alpha` (`reg:expectileerror`).
    pub expectile_alpha: Vec<f64>,
    /// XGBoost `aft_loss_distribution` (`survival:aft`).
    pub aft_loss_distribution: AftDistribution,
    /// XGBoost `aft_loss_distribution_scale` (`survival:aft`).
    pub aft_loss_distribution_scale: f64,
    /// Second-order statistic of the `dist:*` objectives (beyond XGBoost).
    pub dist_gradient: DistGradient,
    /// Shared-tree split direction of the `dist:*` objectives (beyond
    /// XGBoost).
    pub dist_split_direction: DistSplitDirection,
    /// The distribution family of a `dist:*` objective, derived from the
    /// objective name (not a parameter): the `nll` / `crps` metrics read it.
    pub distribution: Option<crate::objective::distributional::DistFamily>,
}

impl ObjectiveParams {
    /// Snapshot the objective parameters of a training configuration.
    pub fn from_params(p: &TrainingParams) -> Self {
        ObjectiveParams {
            scale_pos_weight: p.scale_pos_weight,
            max_delta_step: p.effective_max_delta_step(),
            tweedie_variance_power: p.tweedie_variance_power,
            huber_slope: p.huber_slope,
            lambdarank_num_pair_per_sample: p.lambdarank_num_pair_per_sample,
            quantile_alpha: p.quantile_alpha.clone(),
            expectile_alpha: p.expectile_alpha.clone(),
            aft_loss_distribution: p.aft_loss_distribution,
            aft_loss_distribution_scale: p.aft_loss_distribution_scale,
            dist_gradient: p.dist_gradient,
            dist_split_direction: p.dist_split_direction,
            distribution: crate::objective::distributional::DistFamily::from_objective(
                &p.objective,
            ),
        }
    }

    /// XGBoost's defaults for `objective` (e.g. `max_delta_step = 0.7` for
    /// `count:poisson`).
    pub fn defaults_for(objective: &str) -> Self {
        Self::from_params(
            &TrainingParams::builder()
                .objective(objective)
                .build_unchecked(),
        )
    }

    /// A training configuration for `objective` (with `num_class`) carrying
    /// these parameters: the objective it rebuilds is the one the model was
    /// trained with. Callers `build()` to validate or `build_unchecked()`.
    pub fn training_params(&self, objective: &str, num_class: usize) -> TrainingParamsBuilder {
        TrainingParams::builder()
            .objective(objective)
            .num_class(num_class)
            .scale_pos_weight(self.scale_pos_weight)
            .max_delta_step(self.max_delta_step)
            .tweedie_variance_power(self.tweedie_variance_power)
            .huber_slope(self.huber_slope)
            .lambdarank_num_pair_per_sample(self.lambdarank_num_pair_per_sample)
            .quantile_alpha(self.quantile_alpha.clone())
            .expectile_alpha(self.expectile_alpha.clone())
            .aft_loss_distribution(self.aft_loss_distribution)
            .aft_loss_distribution_scale(self.aft_loss_distribution_scale)
            .dist_gradient(self.dist_gradient)
            .dist_split_direction(self.dist_split_direction)
    }
}

impl Default for ObjectiveParams {
    /// XGBoost's defaults (those of [`TrainingParams::default`]).
    fn default() -> Self {
        Self::from_params(&TrainingParams::default())
    }
}

/// Builder for [`TrainingParams`].
///
/// Every setter returns `self` for chaining. Terminal method is
/// [`TrainingParamsBuilder::build`], which validates the configuration.
#[derive(Debug, Clone)]
pub struct TrainingParamsBuilder {
    params: TrainingParams,
}

macro_rules! setter {
    ($(#[$m:meta])* $name:ident, $ty:ty) => {
        $(#[$m])*
        #[must_use]
        pub fn $name(mut self, v: $ty) -> Self {
            self.params.$name = v;
            self
        }
    };
}

impl TrainingParamsBuilder {
    setter!(/// Set the booster kind.
        booster, BoosterKind);
    setter!(/// Set the number of worker threads (`0` = global pool).
        nthread, usize);
    setter!(/// Set the RNG seed.
        seed, u64);
    setter!(/// Set the number of classes (multiclass objectives).
        num_class, usize);
    setter!(/// Set the learning rate (`eta`).
        eta, f64);
    setter!(/// Set the minimum split loss (`gamma`).
        gamma, f64);
    setter!(/// Set the maximum tree depth.
        max_depth, usize);
    setter!(/// Set the maximum number of leaves (lossguide).
        max_leaves, usize);
    setter!(/// Set the minimum child hessian weight.
        min_child_weight, f64);
    /// Set the maximum delta step (`0` = no constraint). Unset, `count:poisson`
    /// defaults to `0.7` like XGBoost.
    #[must_use]
    pub fn max_delta_step(mut self, v: f64) -> Self {
        self.params.max_delta_step = Some(v);
        self
    }
    setter!(/// Set the row subsample ratio.
        subsample, f64);
    setter!(/// Set the per-tree column subsample ratio.
        colsample_bytree, f64);
    setter!(/// Set the per-level column subsample ratio.
        colsample_bylevel, f64);
    setter!(/// Set the per-node column subsample ratio.
        colsample_bynode, f64);
    setter!(/// Set the L2 regularization (`lambda`).
        lambda, f64);
    setter!(/// Set the L1 regularization (`alpha`).
        alpha, f64);
    setter!(/// Set the positive-class weight scaling.
        scale_pos_weight, f64);
    setter!(/// Set the tree construction method.
        tree_method, TreeMethod);
    setter!(/// Set the tree growth policy.
        grow_policy, GrowPolicy);
    setter!(/// Set the maximum histogram bins per feature.
        max_bin, usize);
    setter!(/// Set the DART per-round drop rate (`rate_drop`).
        rate_drop, f64);
    setter!(/// Set the DART dropout-skip probability (`skip_drop`).
        skip_drop, f64);
    setter!(/// Set the Tweedie variance power (`tweedie_variance_power`).
        tweedie_variance_power, f64);
    setter!(/// Set the pseudo-Huber slope (`huber_slope`).
        huber_slope, f64);
    setter!(/// Set LambdaRank's top-k pair count (`lambdarank_num_pair_per_sample`).
        lambdarank_num_pair_per_sample, usize);
    setter!(/// Set the target quantiles of `reg:quantileerror` (`quantile_alpha`).
        quantile_alpha, Vec<f64>);
    setter!(/// Set the target expectiles of `reg:expectileerror` (`expectile_alpha`).
        expectile_alpha, Vec<f64>);
    setter!(/// Set the `survival:aft` noise distribution (`aft_loss_distribution`).
        aft_loss_distribution, AftDistribution);
    setter!(/// Set the `survival:aft` noise scale (`aft_loss_distribution_scale`).
        aft_loss_distribution_scale, f64);
    setter!(/// Set the second-order statistic of the `dist:*` objectives (`dist_gradient`).
        dist_gradient, DistGradient);
    setter!(/// Set the shared-tree split direction of the `dist:*` objectives (`dist_split_direction`).
        dist_split_direction, DistSplitDirection);
    setter!(/// Set the number of trees grown per output per round (`num_parallel_tree`).
        num_parallel_tree, usize);
    setter!(/// Set the row subsampling method (`sampling_method`).
        sampling_method, SamplingMethod);
    setter!(/// Set the multi-output tree strategy (`multi_strategy`).
        multi_strategy, MultiStrategy);
    setter!(/// Set whether rounds grow or update trees (`process_type`).
        process_type, ProcessType);
    setter!(/// Set whether `process_type = update` refreshes leaf values (`refresh_leaf`).
        refresh_leaf, bool);
    setter!(/// Enable LightGBM's randomized split search (`extra_trees`).
        extra_trees, bool);
    setter!(/// Set the seed of the `extra_trees` threshold draws (`extra_seed`).
        extra_seed, u64);
    setter!(/// Set LightGBM's path smoothing strength (`path_smooth`, `0` = off).
        path_smooth, f64);
    setter!(/// Enable LightGBM's per-leaf linear models (`linear_tree`).
        linear_tree, bool);
    setter!(/// Set the L2 penalty on leaf linear-model slopes (`linear_lambda`).
        linear_lambda, f64);
    setter!(/// Set the new-feature reuse penalty `ι` (`toad_penalty_feature`).
        toad_penalty_feature, f64);
    setter!(/// Set the new-threshold reuse penalty `ξ` (`toad_penalty_threshold`).
        toad_penalty_threshold, f64);
    setter!(/// Enable quantized-gradient training (`use_quantized_grad`, LightGBM).
        use_quantized_grad, bool);
    setter!(/// Set the gradient quantization levels (`num_grad_quant_bins`, LightGBM).
        num_grad_quant_bins, usize);
    setter!(/// Set stochastic rounding of quantized gradients (`stochastic_rounding`, LightGBM).
        stochastic_rounding, bool);
    setter!(/// Set full-precision leaf renewal after quantized growth (`quant_train_renew_leaf`, LightGBM).
        quant_train_renew_leaf, bool);

    /// Set the objective by name (e.g. `"binary:logistic"`).
    #[must_use]
    pub fn objective(mut self, name: impl Into<String>) -> Self {
        self.params.objective = name.into();
        self
    }

    /// Set the base score / global bias.
    #[must_use]
    pub fn base_score(mut self, v: f64) -> Self {
        self.params.base_score = Some(v);
        self
    }

    /// Add an evaluation metric by name.
    #[must_use]
    pub fn eval_metric(mut self, name: impl Into<String>) -> Self {
        self.params.eval_metric.push(name.into());
        self
    }

    setter!(/// Set the per-feature monotone constraints.
        monotone_constraints, Vec<Monotone>);
    setter!(
        /// Set the allowed feature-interaction groups.
        ///
        /// Each inner vector lists feature indices that are permitted to appear
        /// together on a single root-to-leaf path. An empty list disables the
        /// constraint. Mirrors XGBoost `interaction_constraints`.
        interaction_constraints,
        Vec<Vec<u32>>
    );

    /// Validate and produce the [`TrainingParams`].
    pub fn build(self) -> Result<TrainingParams> {
        self.params.validate()?;
        Ok(self.params)
    }

    /// Produce the [`TrainingParams`] without validation (useful in tests).
    pub fn build_unchecked(self) -> TrainingParams {
        self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parameter `builder.build()` rejects, if any.
    fn rejected(builder: TrainingParamsBuilder) -> Option<&'static str> {
        match builder.build() {
            Err(HessboostError::InvalidParameter { name, .. }) => Some(name),
            _ => None,
        }
    }

    #[test]
    fn defaults_match_xgboost() {
        let p = TrainingParams::default();
        assert_eq!(p.eta, 0.3);
        assert_eq!(p.max_depth, 6);
        assert_eq!(p.min_child_weight, 1.0);
        assert_eq!(p.lambda, 1.0);
        assert_eq!(p.alpha, 0.0);
        assert_eq!(p.max_bin, 256);
        assert_eq!(p.booster, BoosterKind::GbTree);
        assert_eq!(p.grow_policy, GrowPolicy::DepthWise);
        assert!(p.base_score.is_none());
        assert_eq!(p.tweedie_variance_power, 1.5);
        assert_eq!(p.huber_slope, 1.0);
        p.validate().unwrap();
    }

    #[test]
    fn builder_chains_and_validates() {
        let p = TrainingParams::builder()
            .objective("binary:logistic")
            .eta(0.1)
            .max_depth(4)
            .subsample(0.8)
            .lambda(2.0)
            .build()
            .unwrap();
        assert_eq!(p.objective, "binary:logistic");
        assert_eq!(p.eta, 0.1);
        assert_eq!(p.max_depth, 4);
        assert_eq!(p.subsample, 0.8);
    }

    #[test]
    fn rejects_bad_params() {
        let b = TrainingParams::builder;
        for (name, builder) in [
            ("eta", b().eta(0.0)),
            ("subsample", b().subsample(1.5)),
            ("lambda", b().lambda(-1.0)),
            ("max_bin", b().max_bin(1)),
            ("tweedie_variance_power", b().tweedie_variance_power(2.0)),
            // Passes the f64 range but rounds to 2.0 in f32, where the objective runs.
            (
                "tweedie_variance_power",
                b().tweedie_variance_power(2.0 - f64::EPSILON),
            ),
            ("huber_slope", b().huber_slope(0.0)),
            // Finite and positive in f64, but the f32 square overflows / vanishes.
            ("huber_slope", b().huber_slope(2e19)),
            ("huber_slope", b().huber_slope(1e-30)),
            (
                "lambdarank_num_pair_per_sample",
                b().lambdarank_num_pair_per_sample(0),
            ),
            ("max_delta_step", b().max_delta_step(-1.0)),
            ("num_parallel_tree", b().num_parallel_tree(0)),
            // Lossguide growth needs a leaf or depth bound.
            (
                "max_leaves",
                b().grow_policy(GrowPolicy::LossGuide)
                    .max_depth(0)
                    .max_leaves(0),
            ),
        ] {
            assert_eq!(rejected(builder), Some(name));
        }
        assert!(b().tweedie_variance_power(1.0).build().is_ok());
        assert!(
            b().grow_policy(GrowPolicy::LossGuide)
                .max_leaves(31)
                .build()
                .is_ok()
        );
        // Positive finite `f64` scales that become infinite or zero once
        // narrowed to the `f32` the objective and metric compute in.
        for scale in [0.0, -1.0, f64::INFINITY, f64::NAN, 1e100, 1e-50] {
            assert_eq!(
                rejected(b().aft_loss_distribution_scale(scale)),
                Some("aft_loss_distribution_scale"),
                "scale {scale}"
            );
        }
    }

    /// XGBoost injects `max_delta_step = 0.7` for `count:poisson` only when
    /// the user did not set it; an explicit `0` disables the constraint.
    #[test]
    fn poisson_delta_step_default_respects_explicit_zero() {
        let unset = TrainingParams::builder()
            .objective("count:poisson")
            .build()
            .unwrap();
        assert_eq!(unset.effective_max_delta_step(), 0.7);
        let zero = TrainingParams::builder()
            .objective("count:poisson")
            .max_delta_step(0.0)
            .build()
            .unwrap();
        assert_eq!(zero.effective_max_delta_step(), 0.0);
        assert_eq!(TrainingParams::default().effective_max_delta_step(), 0.0);
    }

    /// Parameters take XGBoost's names on the wire, so JSON
    /// configurations written for XGBoost deserialize unchanged, and a
    /// configuration that omits them gets XGBoost's defaults.
    #[test]
    fn params_use_xgboost_spellings_and_defaults() {
        let p: TrainingParams = serde_json::from_str(
            r#"{"aft_loss_distribution": "extreme", "sampling_method": "gradient_based",
                "multi_strategy": "multi_output_tree", "process_type": "update",
                "refresh_leaf": false, "num_parallel_tree": 4,
                "quantile_alpha": [0.1, 0.9]}"#,
        )
        .unwrap();
        assert_eq!(p.aft_loss_distribution, AftDistribution::Extreme);
        assert_eq!(p.sampling_method, SamplingMethod::GradientBased);
        assert_eq!(p.multi_strategy, MultiStrategy::MultiOutputTree);
        assert_eq!(p.process_type, ProcessType::Update);
        assert!(!p.refresh_leaf);
        assert_eq!(p.num_parallel_tree, 4);
        assert_eq!(p.quantile_alpha, vec![0.1, 0.9]);

        let d: TrainingParams = serde_json::from_str("{}").unwrap();
        assert_eq!(d.aft_loss_distribution, AftDistribution::Normal);
        assert_eq!(d.aft_loss_distribution_scale, 1.0);
        assert_eq!(d.num_parallel_tree, 1);
        assert_eq!(d.sampling_method, SamplingMethod::Uniform);
        assert_eq!(d.multi_strategy, MultiStrategy::OneOutputPerTree);
        assert_eq!(d.process_type, ProcessType::Default);
        assert!(d.refresh_leaf);
        assert!(d.quantile_alpha.is_empty() && d.expectile_alpha.is_empty());
        d.validate().unwrap();
    }

    /// A trained model rebuilds its objective from `ObjectiveParams`, so the
    /// objective-specific parameters must survive the snapshot/restore trip.
    #[test]
    fn objective_params_round_trip_objective_fields() {
        let p = TrainingParams::builder()
            .objective("survival:aft")
            .quantile_alpha(vec![0.25, 0.75])
            .expectile_alpha(vec![0.5])
            .aft_loss_distribution(AftDistribution::Logistic)
            .aft_loss_distribution_scale(1.7)
            .build_unchecked();
        let snapshot = ObjectiveParams::from_params(&p);
        let restored = snapshot
            .training_params("survival:aft", 0)
            .build_unchecked();
        assert_eq!(restored.quantile_alpha, vec![0.25, 0.75]);
        assert_eq!(restored.expectile_alpha, vec![0.5]);
        assert_eq!(restored.aft_loss_distribution, AftDistribution::Logistic);
        assert_eq!(restored.aft_loss_distribution_scale, 1.7);
        assert_eq!(ObjectiveParams::from_params(&restored), snapshot);
    }

    /// Reuse penalties apply in the XGBoost split searches only; the LightGBM
    /// split search and symmetric growth would silently ignore them.
    #[test]
    fn reuse_penalties_refuse_searches_that_ignore_them() {
        let toad = || TrainingParams::builder().toad_penalty_feature(1.0);
        for params in [
            toad().extra_trees(true),
            toad().path_smooth(1.0),
            toad().grow_policy(GrowPolicy::Symmetric).max_depth(3),
        ] {
            assert_eq!(rejected(params), Some("toad_penalty_feature"));
        }
        assert!(toad().linear_tree(true).build().is_ok());
    }

    /// Symmetric growth builds histograms outside the quantized path, and
    /// path-smoothed leaves would discard renewed leaf statistics.
    #[test]
    fn quantized_training_refuses_options_it_would_ignore() {
        let q = || {
            TrainingParams::builder()
                .use_quantized_grad(true)
                .max_depth(3)
        };
        assert_eq!(
            rejected(q().grow_policy(GrowPolicy::Symmetric)),
            Some("use_quantized_grad")
        );
        assert!(q().grow_policy(GrowPolicy::LossGuide).build().is_ok());
        // Leaf renewal would be discarded: path-smoothed leaves keep the
        // outputs their quantized splits recorded.
        assert_eq!(
            rejected(q().quant_train_renew_leaf(true).path_smooth(1.0)),
            Some("quant_train_renew_leaf")
        );
        assert!(q().quant_train_renew_leaf(true).build().is_ok());
        assert!(q().path_smooth(1.0).build().is_ok());
    }

    /// The linear booster never grows trees, so symmetric growth would be
    /// silently discarded.
    #[test]
    fn symmetric_growth_refuses_the_linear_booster() {
        let sym = || {
            TrainingParams::builder()
                .grow_policy(GrowPolicy::Symmetric)
                .max_depth(3)
        };
        assert_eq!(
            rejected(sym().booster(BoosterKind::GbLinear)),
            Some("grow_policy")
        );
        assert!(sym().booster(BoosterKind::Dart).build().is_ok());
    }

    /// The names model files store are the serde (and XGBoost) spellings,
    /// and read back to the same variant.
    #[test]
    fn stored_enum_names_match_serde_and_read_back() {
        fn check<T: Serialize + Copy + PartialEq + std::fmt::Debug>(
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
                DistGradient::Fisher,
                DistGradient::Hessian,
                DistGradient::Natural,
            ],
            DistGradient::name,
            DistGradient::from_name,
        );
        check(
            &[
                DistSplitDirection::Random,
                DistSplitDirection::Cyclic,
                DistSplitDirection::All,
            ],
            DistSplitDirection::name,
            DistSplitDirection::from_name,
        );
    }
}
