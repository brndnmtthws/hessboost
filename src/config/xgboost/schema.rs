//! The flat form's keys: every accepted key with its value type
//! ([`Flat`]), XGBoost's aliases, the one-setting options, and the lookup
//! of a given key (with a suggestion for a typo).

use super::super::groups::ModelShrinkMode;
use super::super::params::{
    Device, GrowPolicy, Monotone, MultiStrategy, SamplingMethod, TreeMethod,
};
use crate::error::{HessboostError, Result};
use crate::objective::AftDistribution;
use crate::objective::distributional::{DistGradient, DistSplitDirection};
use serde::{Deserialize, Deserializer};
use std::num::NonZeroUsize;

/// XGBoost's aliases and the key each one sets.
const ALIASES: &[(&str, &str)] = &[
    ("learning_rate", "eta"),
    ("min_split_loss", "gamma"),
    ("reg_lambda", "lambda"),
    ("reg_alpha", "alpha"),
    ("random_state", "seed"),
    ("n_jobs", "nthread"),
];

/// XGBoost options hessboost implements at one setting only (the crate
/// docs' "Not implemented"), with that setting as JSON: accepted at exactly
/// this value, refused otherwise.
pub(super) const FIXED: &[(&str, &str)] = &[
    ("updater", "\"coord_descent\""),
    ("feature_selector", "\"cyclic\""),
    ("lambdarank_pair_method", "\"topk\""),
    ("max_cat_to_onehot", "4"),
    ("max_cat_threshold", "64"),
];

/// XGBoost's `booster` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum FlatBooster {
    GbTree,
    Dart,
    GbLinear,
    Boulevard,
    Ebm,
}

/// XGBoost's `process_type` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum FlatProcess {
    Default,
    Update,
}

/// A key that is present, with a value that may itself be `null` only
/// where the type is an `Option` (plain `Option` fields would read `null`
/// as absent).
fn present<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error> {
    T::deserialize(deserializer).map(Some)
}
/// A flat count whose `0` means no limit or off (XGBoost's `max_depth`,
/// `max_leaves`, and `nthread`, `ebm_early_stopping_rounds`): `None` for `0`.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(from = "usize")]
pub(super) struct FlatLimit(pub(super) Option<NonZeroUsize>);

impl From<usize> for FlatLimit {
    fn from(n: usize) -> Self {
        FlatLimit(NonZeroUsize::new(n))
    }
}

/// A flat rate whose `0` means off (`boulevard_truncation`, CatBoost's
/// `model_shrink_rate`): `None` for `0`.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(from = "f64")]
pub(super) struct FlatRate(pub(super) Option<f64>);

impl From<f64> for FlatRate {
    fn from(rate: f64) -> Self {
        FlatRate((rate != 0.0).then_some(rate))
    }
}

/// Declares [`Flat`], every accepted key with its value type, and
/// [`KEYS`], the same keys as a list (for unknown-key suggestions).
macro_rules! flat_params {
    ($($key:ident: $ty:ty,)*) => {
        /// The settings of one flat parameter map, each `None` when absent.
        #[derive(Default, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub(super) struct Flat {
            $(
                #[serde(default, deserialize_with = "present")]
                pub(super) $key: Option<$ty>,
            )*
        }

        /// Every key [`Flat`] accepts.
        pub(super) const KEYS: &[&str] = &[$(stringify!($key)),*];

        impl Flat {
            /// Whether `key` is set.
            pub(super) fn is_set(&self, key: &str) -> bool {
                match key {
                    $(stringify!($key) => self.$key.is_some(),)*
                    _ => false,
                }
            }
        }
    };
}

flat_params! {
    booster: FlatBooster,
    nthread: FlatLimit,
    seed: u64,
    device: Device,
    objective: String,
    num_class: usize,
    base_score: Option<f64>,
    eval_metric: Vec<String>,
    tweedie_variance_power: f64,
    huber_slope: f64,
    lambdarank_num_pair_per_sample: usize,
    quantile_alpha: Vec<f64>,
    expectile_alpha: Vec<f64>,
    aft_loss_distribution: AftDistribution,
    aft_loss_distribution_scale: f64,
    dist_gradient: DistGradient,
    dist_split_direction: DistSplitDirection,
    eta: f64,
    gamma: f64,
    max_depth: FlatLimit,
    max_leaves: FlatLimit,
    min_child_weight: f64,
    max_delta_step: Option<f64>,
    subsample: f64,
    colsample_bytree: f64,
    colsample_bylevel: f64,
    colsample_bynode: f64,
    lambda: f64,
    alpha: f64,
    scale_pos_weight: f64,
    tree_method: TreeMethod,
    grow_policy: GrowPolicy,
    max_bin: usize,
    monotone_constraints: Vec<Monotone>,
    interaction_constraints: Vec<Vec<u32>>,
    num_parallel_tree: usize,
    sampling_method: SamplingMethod,
    pos_bagging_fraction: f64,
    neg_bagging_fraction: f64,
    bagging_by_query: bool,
    multi_strategy: MultiStrategy,
    process_type: FlatProcess,
    refresh_leaf: bool,
    extra_trees: bool,
    extra_seed: u64,
    path_smooth: f64,
    linear_tree: bool,
    linear_lambda: f64,
    use_quantized_grad: bool,
    num_grad_quant_bins: usize,
    stochastic_rounding: bool,
    quant_train_renew_leaf: bool,
    rate_drop: f64,
    skip_drop: f64,
    one_drop: bool,
    toad_penalty_feature: f64,
    toad_penalty_threshold: f64,
    langevin: bool,
    diffusion_temperature: f64,
    model_shrink_rate: FlatRate,
    model_shrink_mode: ModelShrinkMode,
    posterior_sampling: bool,
    boulevard_dropout: f64,
    boulevard_truncation: FlatRate,
    ebm_interactions: usize,
    ebm_outer_bags: usize,
    ebm_bag_fraction: f64,
    ebm_boulevard: bool,
    ebm_early_stopping_rounds: FlatLimit,
    ebm_early_stopping_tolerance: f64,
}

/// Levenshtein distance, for suggesting a key.
fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut previous = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let substituted = previous + usize::from(ca != *cb);
            previous = row[j + 1];
            row[j + 1] = substituted.min(row[j] + 1).min(previous + 1);
        }
    }
    row[b.len()]
}

/// The error for an unknown `key`, suggesting the closest known key (or
/// alias, or one-setting option) when it is close enough to be a typo.
fn unknown_key(key: &str) -> HessboostError {
    let candidates = KEYS
        .iter()
        .copied()
        .chain(ALIASES.iter().map(|&(alias, _)| alias))
        .chain(FIXED.iter().map(|&(name, _)| name));
    let suggestion = candidates
        .map(|name| (distance(key, name), name))
        .min_by_key(|&(d, name)| (d, name))
        .filter(|&(d, _)| d <= (key.len() / 3).max(1))
        .map(|(_, name)| name);
    HessboostError::Unknown {
        kind: "parameter",
        name: key.to_owned(),
        suggestion,
    }
}

/// The canonical key `key` sets: itself, or the key its alias stands for.
pub(super) fn canonical_key(key: &str) -> Result<&'static str> {
    if key == "missing" {
        return Err(HessboostError::invalid_param(
            "missing",
            "belongs to the data, not the training parameters: set it when constructing the DMatrix (`DMatrix::from_dense_with_missing`)",
        ));
    }
    let key = ALIASES
        .iter()
        .find(|&&(alias, _)| alias == key)
        .map_or(key, |&(_, canonical)| canonical);
    KEYS.iter()
        .copied()
        .find(|&known| known == key)
        .ok_or_else(|| unknown_key(key))
}
