//! XGBoost's flat parameter form, the one boundary between XGBoost-named
//! key/value settings and [`TrainingParams`]: [`TrainingParams::from_xgboost`]
//! and [`TrainingParams::to_xgboost`]. The Python bindings, the parity tests,
//! and the training fuzz target all go through it.

use super::params::{
    AftDistribution, BoosterKind, Device, DistGradient, DistSplitDirection, GrowPolicy, Monotone,
    MultiStrategy, ProcessType, SamplingMethod, TrainingParams, TreeMethod,
};
use crate::error::{HessboostError, Result};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

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
const FIXED: &[(&str, &str)] = &[
    ("updater", "\"coord_descent\""),
    ("feature_selector", "\"cyclic\""),
    ("lambdarank_pair_method", "\"topk\""),
    ("max_cat_to_onehot", "4"),
    ("max_cat_threshold", "64"),
];

/// A key that is present, with a value that may itself be `null` only
/// where the type is an `Option` (plain `Option` fields would read `null`
/// as absent).
fn present<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error> {
    T::deserialize(deserializer).map(Some)
}

/// Declares [`Flat`], every accepted key with its value type, and
/// [`KEYS`], the same keys as a list (for unknown-key suggestions).
macro_rules! flat_params {
    ($($key:ident: $ty:ty,)*) => {
        /// The settings of one flat parameter map, each `None` when absent.
        #[derive(Default, Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Flat {
            $(
                #[serde(default, deserialize_with = "present")]
                $key: Option<$ty>,
            )*
        }

        /// Every key [`Flat`] accepts.
        const KEYS: &[&str] = &[$(stringify!($key)),*];
    };
}

flat_params! {
    booster: BoosterKind,
    nthread: usize,
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
    max_depth: usize,
    max_leaves: usize,
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
    multi_strategy: MultiStrategy,
    process_type: ProcessType,
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
    toad_penalty_feature: f64,
    toad_penalty_threshold: f64,
}

impl Flat {
    /// The configuration these settings describe, every absent key at its
    /// default.
    fn into_params(self) -> TrainingParams {
        let Flat {
            booster,
            nthread,
            seed,
            device,
            objective,
            num_class,
            base_score,
            eval_metric,
            tweedie_variance_power,
            huber_slope,
            lambdarank_num_pair_per_sample,
            quantile_alpha,
            expectile_alpha,
            aft_loss_distribution,
            aft_loss_distribution_scale,
            dist_gradient,
            dist_split_direction,
            eta,
            gamma,
            max_depth,
            max_leaves,
            min_child_weight,
            max_delta_step,
            subsample,
            colsample_bytree,
            colsample_bylevel,
            colsample_bynode,
            lambda,
            alpha,
            scale_pos_weight,
            tree_method,
            grow_policy,
            max_bin,
            monotone_constraints,
            interaction_constraints,
            num_parallel_tree,
            sampling_method,
            multi_strategy,
            process_type,
            refresh_leaf,
            extra_trees,
            extra_seed,
            path_smooth,
            linear_tree,
            linear_lambda,
            use_quantized_grad,
            num_grad_quant_bins,
            stochastic_rounding,
            quant_train_renew_leaf,
            rate_drop,
            skip_drop,
            toad_penalty_feature,
            toad_penalty_threshold,
        } = self;
        let d = TrainingParams::default();
        TrainingParams {
            booster: booster.unwrap_or(d.booster),
            nthread: nthread.unwrap_or(d.nthread),
            seed: seed.unwrap_or(d.seed),
            device: device.unwrap_or(d.device),
            objective: objective.unwrap_or(d.objective),
            num_class: num_class.unwrap_or(d.num_class),
            base_score: base_score.unwrap_or(d.base_score),
            eval_metric: eval_metric.unwrap_or(d.eval_metric),
            tweedie_variance_power: tweedie_variance_power.unwrap_or(d.tweedie_variance_power),
            huber_slope: huber_slope.unwrap_or(d.huber_slope),
            lambdarank_num_pair_per_sample: lambdarank_num_pair_per_sample
                .unwrap_or(d.lambdarank_num_pair_per_sample),
            quantile_alpha: quantile_alpha.unwrap_or(d.quantile_alpha),
            expectile_alpha: expectile_alpha.unwrap_or(d.expectile_alpha),
            aft_loss_distribution: aft_loss_distribution.unwrap_or(d.aft_loss_distribution),
            aft_loss_distribution_scale: aft_loss_distribution_scale
                .unwrap_or(d.aft_loss_distribution_scale),
            dist_gradient: dist_gradient.unwrap_or(d.dist_gradient),
            dist_split_direction: dist_split_direction.unwrap_or(d.dist_split_direction),
            eta: eta.unwrap_or(d.eta),
            gamma: gamma.unwrap_or(d.gamma),
            max_depth: max_depth.unwrap_or(d.max_depth),
            max_leaves: max_leaves.unwrap_or(d.max_leaves),
            min_child_weight: min_child_weight.unwrap_or(d.min_child_weight),
            max_delta_step: max_delta_step.unwrap_or(d.max_delta_step),
            subsample: subsample.unwrap_or(d.subsample),
            colsample_bytree: colsample_bytree.unwrap_or(d.colsample_bytree),
            colsample_bylevel: colsample_bylevel.unwrap_or(d.colsample_bylevel),
            colsample_bynode: colsample_bynode.unwrap_or(d.colsample_bynode),
            lambda: lambda.unwrap_or(d.lambda),
            alpha: alpha.unwrap_or(d.alpha),
            scale_pos_weight: scale_pos_weight.unwrap_or(d.scale_pos_weight),
            tree_method: tree_method.unwrap_or(d.tree_method),
            grow_policy: grow_policy.unwrap_or(d.grow_policy),
            max_bin: max_bin.unwrap_or(d.max_bin),
            monotone_constraints: monotone_constraints.unwrap_or(d.monotone_constraints),
            interaction_constraints: interaction_constraints.unwrap_or(d.interaction_constraints),
            num_parallel_tree: num_parallel_tree.unwrap_or(d.num_parallel_tree),
            sampling_method: sampling_method.unwrap_or(d.sampling_method),
            multi_strategy: multi_strategy.unwrap_or(d.multi_strategy),
            process_type: process_type.unwrap_or(d.process_type),
            refresh_leaf: refresh_leaf.unwrap_or(d.refresh_leaf),
            extra_trees: extra_trees.unwrap_or(d.extra_trees),
            extra_seed: extra_seed.unwrap_or(d.extra_seed),
            path_smooth: path_smooth.unwrap_or(d.path_smooth),
            linear_tree: linear_tree.unwrap_or(d.linear_tree),
            linear_lambda: linear_lambda.unwrap_or(d.linear_lambda),
            use_quantized_grad: use_quantized_grad.unwrap_or(d.use_quantized_grad),
            num_grad_quant_bins: num_grad_quant_bins.unwrap_or(d.num_grad_quant_bins),
            stochastic_rounding: stochastic_rounding.unwrap_or(d.stochastic_rounding),
            quant_train_renew_leaf: quant_train_renew_leaf.unwrap_or(d.quant_train_renew_leaf),
            rate_drop: rate_drop.unwrap_or(d.rate_drop),
            skip_drop: skip_drop.unwrap_or(d.skip_drop),
            toad_penalty_feature: toad_penalty_feature.unwrap_or(d.toad_penalty_feature),
            toad_penalty_threshold: toad_penalty_threshold.unwrap_or(d.toad_penalty_threshold),
        }
    }
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
fn canonical_key(key: &str) -> Result<&'static str> {
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

/// XGBoost's other spellings of the values whose serde form differs: a
/// single metric name, a single alpha, monotone constraints as a
/// `"(1,-1,0)"` string or `-1`/`0`/`1` entries, and interaction constraints
/// as a JSON string.
fn normalize(key: &'static str, value: Value) -> Result<Value> {
    Ok(match (key, value) {
        ("eval_metric", Value::String(name)) => Value::Array(vec![Value::String(name)]),
        ("quantile_alpha" | "expectile_alpha", Value::Number(alpha)) => {
            Value::Array(vec![Value::Number(alpha)])
        }
        ("monotone_constraints", Value::String(text)) => {
            let inner = text.trim().trim_start_matches('(').trim_end_matches(')');
            let entries = inner
                .split(',')
                .map(str::trim)
                .filter(|entry| !entry.is_empty())
                .map(|entry| {
                    entry.parse::<i64>().map(Value::from).map_err(|_| {
                        HessboostError::invalid_param(
                            key,
                            format!("bad entry `{entry}` in {text:?}"),
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            normalize(key, Value::Array(entries))?
        }
        ("monotone_constraints", Value::Array(entries)) => entries
            .into_iter()
            .map(|entry| match entry.as_i64() {
                Some(1) => Ok(Value::from("increasing")),
                Some(-1) => Ok(Value::from("decreasing")),
                Some(0) => Ok(Value::from("none")),
                _ if entry.is_string() => Ok(entry),
                _ => Err(HessboostError::invalid_param(
                    key,
                    format!("entries must be -1, 0 or 1, got {entry}"),
                )),
            })
            .collect::<Result<Value>>()?,
        ("interaction_constraints", Value::String(text)) => {
            serde_json::from_str(&text).map_err(|error| {
                HessboostError::invalid_param(
                    key,
                    format!("{text:?} is not a list of index lists: {error}"),
                )
            })?
        }
        (_, value) => value,
    })
}

/// Deserialize `settings` (canonical keys only) into [`Flat`], naming
/// `key` in the error.
fn flat_from(key: &'static str, settings: Map<String, Value>) -> Result<Flat> {
    serde_json::from_value(Value::Object(settings))
        .map_err(|error| HessboostError::invalid_param(key, error.to_string()))
}

/// `value` as JSON; every setting serializes (non-finite numbers as `null`).
fn json(value: impl Serialize) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

impl TrainingParams {
    /// Parse XGBoost's flat parameter form: key/value pairs with XGBoost's
    /// names (`eta`, `max_depth`, `objective`, ...) and JSON values, as in
    /// an XGBoost `params` dict. Absent keys keep their defaults.
    ///
    /// Besides the canonical keys, XGBoost's aliases (`learning_rate`,
    /// `min_split_loss`, `reg_lambda`, `reg_alpha`, `random_state`,
    /// `n_jobs`) are accepted, as are XGBoost's other value spellings: a
    /// single string for `eval_metric`, a number for `quantile_alpha` /
    /// `expectile_alpha`, `monotone_constraints` as `"(1,-1,0)"` or
    /// `-1`/`0`/`1` entries, and `interaction_constraints` as a JSON string.
    /// XGBoost options hessboost implements at one setting only
    /// (`updater = "coord_descent"` with `booster = gblinear`,
    /// `feature_selector = "cyclic"`, `lambdarank_pair_method = "topk"`,
    /// `max_cat_to_onehot = 4`, `max_cat_threshold = 64`) are accepted at
    /// that setting.
    ///
    /// # Errors
    ///
    /// Refuses, never ignores: an unknown key (suggesting the closest
    /// known one), a key set twice (directly and through an alias), a value
    /// of the wrong type, `missing` (a property of the data), a one-setting
    /// option at another setting, and every configuration
    /// [`validate`](Self::validate) refuses.
    ///
    /// ```
    /// use hessboost::prelude::*;
    /// use serde_json::json;
    ///
    /// # fn main() -> Result<()> {
    /// let params = TrainingParams::from_xgboost([
    ///     ("objective", json!("binary:logistic")),
    ///     ("learning_rate", json!(0.1)),
    ///     ("max_depth", json!(4)),
    /// ])?;
    /// assert_eq!(params.eta, 0.1);
    /// assert!(TrainingParams::from_xgboost([("max_dept", json!(4))]).is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_xgboost<K: AsRef<str>>(
        params: impl IntoIterator<Item = (K, Value)>,
    ) -> Result<Self> {
        let mut settings = Map::new();
        let mut updater = false;
        for (key, value) in params {
            let key = key.as_ref();
            if let Some(&(name, fixed)) = FIXED.iter().find(|&&(name, _)| name == key) {
                let expected: Value = serde_json::from_str(fixed)?;
                if value != expected {
                    return Err(HessboostError::invalid_param(
                        name,
                        format!("is only implemented as {fixed}, got {value}"),
                    ));
                }
                updater |= name == "updater";
                continue;
            }
            let canonical = canonical_key(key)?;
            let value = normalize(canonical, value)?;
            // Deserialized alone first, so a type error names its key.
            flat_from(
                canonical,
                Map::from_iter([(canonical.to_owned(), value.clone())]),
            )?;
            if settings.insert(canonical.to_owned(), value).is_some() {
                return Err(HessboostError::invalid_param(
                    canonical,
                    "is set twice (through an alias)",
                ));
            }
        }
        let params = flat_from("params", settings)?.into_params();
        if updater && params.booster != BoosterKind::GbLinear {
            return Err(HessboostError::invalid_param(
                "updater",
                "`coord_descent` is gblinear's updater; tree boosters take no `updater`",
            ));
        }
        params.validate()?;
        Ok(params)
    }

    /// This configuration in XGBoost's flat parameter form, under the
    /// canonical keys [`from_xgboost`](Self::from_xgboost) reads back to the
    /// same configuration (monotone constraints as `-1`/`0`/`1`; an unset
    /// `base_score` or `max_delta_step` is left out).
    ///
    /// # Errors
    ///
    /// None yet: every configuration has a flat form.
    pub fn to_xgboost(&self) -> Result<Map<String, Value>> {
        let TrainingParams {
            booster,
            nthread,
            seed,
            device,
            objective,
            num_class,
            base_score,
            eval_metric,
            tweedie_variance_power,
            huber_slope,
            lambdarank_num_pair_per_sample,
            quantile_alpha,
            expectile_alpha,
            aft_loss_distribution,
            aft_loss_distribution_scale,
            dist_gradient,
            dist_split_direction,
            eta,
            gamma,
            max_depth,
            max_leaves,
            min_child_weight,
            max_delta_step,
            subsample,
            colsample_bytree,
            colsample_bylevel,
            colsample_bynode,
            lambda,
            alpha,
            scale_pos_weight,
            tree_method,
            grow_policy,
            max_bin,
            monotone_constraints,
            interaction_constraints,
            num_parallel_tree,
            sampling_method,
            multi_strategy,
            process_type,
            refresh_leaf,
            extra_trees,
            extra_seed,
            path_smooth,
            linear_tree,
            linear_lambda,
            use_quantized_grad,
            num_grad_quant_bins,
            stochastic_rounding,
            quant_train_renew_leaf,
            rate_drop,
            skip_drop,
            toad_penalty_feature,
            toad_penalty_threshold,
        } = self;
        let monotone: Vec<i8> = monotone_constraints
            .iter()
            .map(|m| match m {
                Monotone::None => 0,
                Monotone::Increasing => 1,
                Monotone::Decreasing => -1,
            })
            .collect();
        let mut flat = Map::new();
        let mut set = |key: &str, value: Value| {
            flat.insert(key.to_owned(), value);
        };
        set("booster", json(booster));
        set("nthread", json(nthread));
        set("seed", json(seed));
        set("device", json(device));
        set("objective", json(objective));
        set("num_class", json(num_class));
        if let Some(base_score) = base_score {
            set("base_score", json(base_score));
        }
        set("eval_metric", json(eval_metric));
        set("tweedie_variance_power", json(tweedie_variance_power));
        set("huber_slope", json(huber_slope));
        set(
            "lambdarank_num_pair_per_sample",
            json(lambdarank_num_pair_per_sample),
        );
        set("quantile_alpha", json(quantile_alpha));
        set("expectile_alpha", json(expectile_alpha));
        set("aft_loss_distribution", json(aft_loss_distribution));
        set(
            "aft_loss_distribution_scale",
            json(aft_loss_distribution_scale),
        );
        set("dist_gradient", json(dist_gradient));
        set("dist_split_direction", json(dist_split_direction));
        set("eta", json(eta));
        set("gamma", json(gamma));
        set("max_depth", json(max_depth));
        set("max_leaves", json(max_leaves));
        set("min_child_weight", json(min_child_weight));
        if let Some(max_delta_step) = max_delta_step {
            set("max_delta_step", json(max_delta_step));
        }
        set("subsample", json(subsample));
        set("colsample_bytree", json(colsample_bytree));
        set("colsample_bylevel", json(colsample_bylevel));
        set("colsample_bynode", json(colsample_bynode));
        set("lambda", json(lambda));
        set("alpha", json(alpha));
        set("scale_pos_weight", json(scale_pos_weight));
        set("tree_method", json(tree_method));
        set("grow_policy", json(grow_policy));
        set("max_bin", json(max_bin));
        set("monotone_constraints", json(monotone));
        set("interaction_constraints", json(interaction_constraints));
        set("num_parallel_tree", json(num_parallel_tree));
        set("sampling_method", json(sampling_method));
        set("multi_strategy", json(multi_strategy));
        set("process_type", json(process_type));
        set("refresh_leaf", json(refresh_leaf));
        set("extra_trees", json(extra_trees));
        set("extra_seed", json(extra_seed));
        set("path_smooth", json(path_smooth));
        set("linear_tree", json(linear_tree));
        set("linear_lambda", json(linear_lambda));
        set("use_quantized_grad", json(use_quantized_grad));
        set("num_grad_quant_bins", json(num_grad_quant_bins));
        set("stochastic_rounding", json(stochastic_rounding));
        set("quant_train_renew_leaf", json(quant_train_renew_leaf));
        set("rate_drop", json(rate_drop));
        set("skip_drop", json(skip_drop));
        set("toad_penalty_feature", json(toad_penalty_feature));
        set("toad_penalty_threshold", json(toad_penalty_threshold));
        Ok(flat)
    }

    /// The XGBoost keys whose settings differ between `self` and `other`,
    /// in key order. Destructures `self` so that every field is compared.
    pub(crate) fn changed_keys(&self, other: &TrainingParams) -> Vec<&'static str> {
        let TrainingParams {
            booster,
            nthread,
            seed,
            device,
            objective,
            num_class,
            base_score,
            eval_metric,
            tweedie_variance_power,
            huber_slope,
            lambdarank_num_pair_per_sample,
            quantile_alpha,
            expectile_alpha,
            aft_loss_distribution,
            aft_loss_distribution_scale,
            dist_gradient,
            dist_split_direction,
            eta,
            gamma,
            max_depth,
            max_leaves,
            min_child_weight,
            max_delta_step,
            subsample,
            colsample_bytree,
            colsample_bylevel,
            colsample_bynode,
            lambda,
            alpha,
            scale_pos_weight,
            tree_method,
            grow_policy,
            max_bin,
            monotone_constraints,
            interaction_constraints,
            num_parallel_tree,
            sampling_method,
            multi_strategy,
            process_type,
            refresh_leaf,
            extra_trees,
            extra_seed,
            path_smooth,
            linear_tree,
            linear_lambda,
            use_quantized_grad,
            num_grad_quant_bins,
            stochastic_rounding,
            quant_train_renew_leaf,
            rate_drop,
            skip_drop,
            toad_penalty_feature,
            toad_penalty_threshold,
        } = self;
        let mut changed = Vec::new();
        let mut differs = |key: &'static str, same: bool| {
            if !same {
                changed.push(key);
            }
        };
        differs("booster", *booster == other.booster);
        differs("nthread", *nthread == other.nthread);
        differs("seed", *seed == other.seed);
        differs("device", *device == other.device);
        differs("objective", *objective == other.objective);
        differs("num_class", *num_class == other.num_class);
        differs("base_score", *base_score == other.base_score);
        differs("eval_metric", *eval_metric == other.eval_metric);
        differs(
            "tweedie_variance_power",
            *tweedie_variance_power == other.tweedie_variance_power,
        );
        differs("huber_slope", *huber_slope == other.huber_slope);
        differs(
            "lambdarank_num_pair_per_sample",
            *lambdarank_num_pair_per_sample == other.lambdarank_num_pair_per_sample,
        );
        differs("quantile_alpha", *quantile_alpha == other.quantile_alpha);
        differs("expectile_alpha", *expectile_alpha == other.expectile_alpha);
        differs(
            "aft_loss_distribution",
            *aft_loss_distribution == other.aft_loss_distribution,
        );
        differs(
            "aft_loss_distribution_scale",
            *aft_loss_distribution_scale == other.aft_loss_distribution_scale,
        );
        differs("dist_gradient", *dist_gradient == other.dist_gradient);
        differs(
            "dist_split_direction",
            *dist_split_direction == other.dist_split_direction,
        );
        differs("eta", *eta == other.eta);
        differs("gamma", *gamma == other.gamma);
        differs("max_depth", *max_depth == other.max_depth);
        differs("max_leaves", *max_leaves == other.max_leaves);
        differs(
            "min_child_weight",
            *min_child_weight == other.min_child_weight,
        );
        differs("max_delta_step", *max_delta_step == other.max_delta_step);
        differs("subsample", *subsample == other.subsample);
        differs(
            "colsample_bytree",
            *colsample_bytree == other.colsample_bytree,
        );
        differs(
            "colsample_bylevel",
            *colsample_bylevel == other.colsample_bylevel,
        );
        differs(
            "colsample_bynode",
            *colsample_bynode == other.colsample_bynode,
        );
        differs("lambda", *lambda == other.lambda);
        differs("alpha", *alpha == other.alpha);
        differs(
            "scale_pos_weight",
            *scale_pos_weight == other.scale_pos_weight,
        );
        differs("tree_method", *tree_method == other.tree_method);
        differs("grow_policy", *grow_policy == other.grow_policy);
        differs("max_bin", *max_bin == other.max_bin);
        differs(
            "monotone_constraints",
            *monotone_constraints == other.monotone_constraints,
        );
        differs(
            "interaction_constraints",
            *interaction_constraints == other.interaction_constraints,
        );
        differs(
            "num_parallel_tree",
            *num_parallel_tree == other.num_parallel_tree,
        );
        differs("sampling_method", *sampling_method == other.sampling_method);
        differs("multi_strategy", *multi_strategy == other.multi_strategy);
        differs("process_type", *process_type == other.process_type);
        differs("refresh_leaf", *refresh_leaf == other.refresh_leaf);
        differs("extra_trees", *extra_trees == other.extra_trees);
        differs("extra_seed", *extra_seed == other.extra_seed);
        differs("path_smooth", *path_smooth == other.path_smooth);
        differs("linear_tree", *linear_tree == other.linear_tree);
        differs("linear_lambda", *linear_lambda == other.linear_lambda);
        differs(
            "use_quantized_grad",
            *use_quantized_grad == other.use_quantized_grad,
        );
        differs(
            "num_grad_quant_bins",
            *num_grad_quant_bins == other.num_grad_quant_bins,
        );
        differs(
            "stochastic_rounding",
            *stochastic_rounding == other.stochastic_rounding,
        );
        differs(
            "quant_train_renew_leaf",
            *quant_train_renew_leaf == other.quant_train_renew_leaf,
        );
        differs("rate_drop", *rate_drop == other.rate_drop);
        differs("skip_drop", *skip_drop == other.skip_drop);
        differs(
            "toad_penalty_feature",
            *toad_penalty_feature == other.toad_penalty_feature,
        );
        differs(
            "toad_penalty_threshold",
            *toad_penalty_threshold == other.toad_penalty_threshold,
        );
        changed.sort_unstable();
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The parameter [`TrainingParams::from_xgboost`] refuses, if any.
    fn refused(pairs: Value) -> Option<String> {
        let Value::Object(map) = pairs else {
            unreachable!("test input is an object")
        };
        TrainingParams::from_xgboost(map)
            .err()
            .map(|e| e.to_string())
    }

    /// XGBoost's key and value spellings configure the same settings, and a
    /// configuration's flat form reads back to itself.
    #[test]
    fn xgboost_spellings_parse_and_round_trip() {
        let p = TrainingParams::from_xgboost([
            ("aft_loss_distribution", json!("extreme")),
            ("sampling_method", json!("gradient_based")),
            ("multi_strategy", json!("multi_output_tree")),
            ("process_type", json!("update")),
            ("refresh_leaf", json!(false)),
            ("num_parallel_tree", json!(4)),
            ("quantile_alpha", json!(0.25)),
            ("eval_metric", json!("mae")),
            ("monotone_constraints", json!("(1,-1,0)")),
            ("interaction_constraints", json!("[[0, 1], [2]]")),
            ("learning_rate", json!(0.1)),
            ("reg_lambda", json!(2.0)),
            ("max_delta_step", json!(0.0)),
            ("lambdarank_pair_method", json!("topk")),
        ])
        .unwrap();
        assert_eq!(p.aft_loss_distribution, AftDistribution::Extreme);
        assert_eq!(p.sampling_method, SamplingMethod::GradientBased);
        assert_eq!(p.multi_strategy, MultiStrategy::MultiOutputTree);
        assert_eq!(p.process_type, ProcessType::Update);
        assert!(!p.refresh_leaf);
        assert_eq!(p.num_parallel_tree, 4);
        assert_eq!(p.quantile_alpha, [0.25]);
        assert_eq!(p.eval_metric, ["mae"]);
        assert_eq!(
            p.monotone_constraints,
            [Monotone::Increasing, Monotone::Decreasing, Monotone::None]
        );
        assert_eq!(p.interaction_constraints, [vec![0, 1], vec![2]]);
        assert_eq!((p.eta, p.lambda), (0.1, 2.0));
        assert_eq!(p.max_delta_step, Some(0.0));

        let flat = p.to_xgboost().unwrap();
        let back = TrainingParams::from_xgboost(flat.clone()).unwrap();
        assert_eq!(back.to_xgboost().unwrap(), flat);
        let defaults = TrainingParams::from_xgboost(Map::new()).unwrap();
        assert_eq!(
            defaults.to_xgboost().unwrap(),
            TrainingParams::default().to_xgboost().unwrap()
        );
        assert!(
            !defaults
                .to_xgboost()
                .unwrap()
                .contains_key("max_delta_step")
        );
    }

    /// Nothing is ignored: unknown keys (with a suggestion for typos), a key
    /// set twice through an alias, `null` for a plain value, `missing`,
    /// one-setting options at another setting, and invalid configurations.
    #[test]
    fn unsupported_settings_are_refused_by_name() {
        for (pairs, message) in [
            (
                json!({"max_dept": 3}),
                "unknown parameter `max_dept` (did you mean `max_depth`?)",
            ),
            (json!({"zzz": 1}), "unknown parameter `zzz`"),
            (
                json!({"eta": 0.1, "learning_rate": 0.2}),
                "invalid parameter `eta`: is set twice",
            ),
            (
                json!({"eta": null}),
                "invalid parameter `eta`: invalid type: null",
            ),
            (
                json!({"eta": "0.1"}),
                "invalid parameter `eta`: invalid type: string",
            ),
            (json!({"missing": 0.0}), "invalid parameter `missing`"),
            (
                json!({"lambdarank_pair_method": "mean"}),
                r#"invalid parameter `lambdarank_pair_method`: is only implemented as "topk""#,
            ),
            (json!({"updater": "coord_descent"}), "gblinear's updater"),
            (json!({"monotone_constraints": "(1,x)"}), "bad entry `x`"),
            (json!({"monotone_constraints": [2]}), "-1, 0 or 1"),
            (json!({"subsample": 1.5}), "invalid parameter `subsample`"),
        ] {
            let refusal = refused(pairs.clone()).unwrap_or_else(|| panic!("{pairs} accepted"));
            assert!(refusal.contains(message), "{pairs}: {refusal}");
        }
        assert_eq!(
            refused(json!({"zzz": 1})).unwrap(),
            "unknown parameter `zzz`"
        );
        assert!(refused(json!({"booster": "gblinear", "updater": "coord_descent"})).is_none());
        assert!(refused(json!({"base_score": null, "max_delta_step": null})).is_none());
    }
}
