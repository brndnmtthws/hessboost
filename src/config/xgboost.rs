//! XGBoost's flat parameter form, the one boundary between XGBoost-named
//! key/value settings and [`TrainingParams`]: [`TrainingParams::from_xgboost`]
//! and [`TrainingParams::to_xgboost`]. The Python bindings, the parity tests,
//! and the training fuzz target all go through it.

use super::groups::{
    BalancedBagging, Boulevard, Dart, ExtraTrees, Langevin, LinearTree, ModelShrink,
    ModelShrinkMode, QuantizedGrad, QueryBagging, Refresh,
};
use super::params::{
    BoosterKind, Device, GrowPolicy, MaxDeltaStep, Monotone, MultiStrategy, ProcessType,
    SamplingMethod, TrainingParams, TreeMethod,
};
use crate::error::{HessboostError, Result};
use crate::metric::{EvalMetric, XgboostMetricSource};
use crate::objective::distributional::{DistGradient, DistSplitDirection};
use crate::objective::{AftDistribution, Objective, ObjectiveParts};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
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
const FIXED: &[(&str, &str)] = &[
    ("updater", "\"coord_descent\""),
    ("feature_selector", "\"cyclic\""),
    ("lambdarank_pair_method", "\"topk\""),
    ("max_cat_to_onehot", "4"),
    ("max_cat_threshold", "64"),
];

/// XGBoost's `booster` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum FlatBooster {
    GbTree,
    Dart,
    GbLinear,
    Boulevard,
}

/// XGBoost's `process_type` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum FlatProcess {
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
    booster: FlatBooster,
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
    model_shrink_rate: f64,
    model_shrink_mode: ModelShrinkMode,
    posterior_sampling: bool,
    boulevard_dropout: f64,
    boulevard_truncation: f64,
}

/// The flat keys of the objective parameters, with the objectives and
/// metrics that read each (for the refusal of a key nothing reads).
const OBJECTIVE_KEYS: &[(&str, &str)] = &[
    ("num_class", "`multi:softmax` and `multi:softprob`"),
    (
        "scale_pos_weight",
        "`binary:logistic`, `binary:logitraw`, and `reg:logistic` (hessboost does not apply \
         it to `reg:squarederror` or `reg:gamma`)",
    ),
    ("tweedie_variance_power", "`reg:tweedie`"),
    (
        "huber_slope",
        "`reg:pseudohubererror` and the `mphe` metric",
    ),
    ("lambdarank_num_pair_per_sample", "the `rank:*` objectives"),
    (
        "quantile_alpha",
        "`reg:quantileerror` and the `quantile` metric",
    ),
    (
        "expectile_alpha",
        "`reg:expectileerror` and the `expectile` metric",
    ),
    (
        "aft_loss_distribution",
        "`survival:aft` and the `aft-nloglik` metric",
    ),
    (
        "aft_loss_distribution_scale",
        "`survival:aft` and the `aft-nloglik` metric",
    ),
    ("dist_gradient", "the `dist:*` objectives"),
    ("dist_split_direction", "the `dist:*` objectives"),
];

/// The flat keys a metric named `name` borrows (XGBoost's metrics read
/// them from the objective's parameters, whatever the objective).
fn borrowed_keys(name: &str) -> &'static [&'static str] {
    match name {
        "mphe" => &["huber_slope"],
        "quantile" => &["quantile_alpha"],
        "expectile" => &["expectile_alpha"],
        "aft-nloglik" => &["aft_loss_distribution", "aft_loss_distribution_scale"],
        _ => &[],
    }
}

impl Flat {
    /// The configuration these settings describe, every absent key at its
    /// default. The objective reads its parameters from their keys, metric
    /// names are read as XGBoost reads them (with the flat parameters the
    /// metrics borrow), and an objective-parameter key neither reads is
    /// refused.
    fn into_params(self) -> Result<TrainingParams> {
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
            pos_bagging_fraction,
            neg_bagging_fraction,
            bagging_by_query,
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
            one_drop,
            toad_penalty_feature,
            toad_penalty_threshold,
            langevin,
            diffusion_temperature,
            model_shrink_rate,
            model_shrink_mode,
            posterior_sampling,
            boulevard_dropout,
            boulevard_truncation,
        } = self;
        // Aligned with `OBJECTIVE_KEYS`.
        let present = [
            num_class.is_some(),
            scale_pos_weight.is_some(),
            tweedie_variance_power.is_some(),
            huber_slope.is_some(),
            lambdarank_num_pair_per_sample.is_some(),
            quantile_alpha.is_some(),
            expectile_alpha.is_some(),
            aft_loss_distribution.is_some(),
            aft_loss_distribution_scale.is_some(),
            dist_gradient.is_some(),
            dist_split_direction.is_some(),
        ];
        let p = ObjectiveParts::default();
        let parts = ObjectiveParts {
            num_class: num_class.unwrap_or(p.num_class),
            scale_pos_weight: scale_pos_weight.unwrap_or(p.scale_pos_weight),
            tweedie_variance_power: tweedie_variance_power.unwrap_or(p.tweedie_variance_power),
            huber_slope: huber_slope.unwrap_or(p.huber_slope),
            lambdarank_num_pair_per_sample: lambdarank_num_pair_per_sample
                .unwrap_or(p.lambdarank_num_pair_per_sample),
            quantile_alpha: quantile_alpha.unwrap_or(p.quantile_alpha),
            expectile_alpha: expectile_alpha.unwrap_or(p.expectile_alpha),
            aft_loss_distribution: aft_loss_distribution.unwrap_or(p.aft_loss_distribution),
            aft_loss_distribution_scale: aft_loss_distribution_scale
                .unwrap_or(p.aft_loss_distribution_scale),
            dist_gradient: dist_gradient.unwrap_or(p.dist_gradient),
            dist_split_direction: dist_split_direction.or(p.dist_split_direction),
        };
        let name = objective.as_deref().unwrap_or("reg:squarederror");
        let objective = Objective::from_parts(name, &parts)
            .ok_or_else(|| HessboostError::unknown("objective", name))??;
        let metric_names = eval_metric.unwrap_or_default();
        for (&(key, users), set) in OBJECTIVE_KEYS.iter().zip(present) {
            let read = objective.parameter_keys().contains(&key)
                || metric_names
                    .iter()
                    .any(|metric| borrowed_keys(metric).contains(&key));
            if set && !read {
                return Err(HessboostError::invalid_param(
                    key,
                    format!(
                        "applies only to {users}; nothing reads it with objective `{}`",
                        objective.name()
                    ),
                ));
            }
        }
        let source = XgboostMetricSource {
            huber_slope: parts.huber_slope,
            quantile_alpha: &parts.quantile_alpha,
            expectile_alpha: &parts.expectile_alpha,
            aft_loss_distribution: parts.aft_loss_distribution,
            aft_loss_distribution_scale: parts.aft_loss_distribution_scale,
            distribution: objective.dist_family(),
        };
        let eval_metric = metric_names
            .iter()
            .map(|name| EvalMetric::from_xgboost(name, &source))
            .collect::<Result<Vec<_>>>()?;
        // Group keys only mean something under their switch.
        let switches = [
            (
                "rate_drop",
                rate_drop.is_some(),
                booster == Some(FlatBooster::Dart),
                "`booster=dart`",
            ),
            (
                "skip_drop",
                skip_drop.is_some(),
                booster == Some(FlatBooster::Dart),
                "`booster=dart`",
            ),
            (
                "one_drop",
                one_drop.is_some(),
                booster == Some(FlatBooster::Dart),
                "`booster=dart`",
            ),
            (
                "boulevard_dropout",
                boulevard_dropout.is_some(),
                booster == Some(FlatBooster::Boulevard),
                "`booster=boulevard`",
            ),
            (
                "boulevard_truncation",
                boulevard_truncation.is_some(),
                booster == Some(FlatBooster::Boulevard),
                "`booster=boulevard`",
            ),
            (
                "refresh_leaf",
                refresh_leaf.is_some(),
                process_type == Some(FlatProcess::Update),
                "`process_type=update`",
            ),
            (
                "extra_seed",
                extra_seed.is_some(),
                extra_trees == Some(true),
                "`extra_trees=true`",
            ),
            (
                "linear_lambda",
                linear_lambda.is_some(),
                linear_tree == Some(true),
                "`linear_tree=true`",
            ),
            (
                "num_grad_quant_bins",
                num_grad_quant_bins.is_some(),
                use_quantized_grad == Some(true),
                "`use_quantized_grad=true`",
            ),
            (
                "stochastic_rounding",
                stochastic_rounding.is_some(),
                use_quantized_grad == Some(true),
                "`use_quantized_grad=true`",
            ),
            (
                "quant_train_renew_leaf",
                quant_train_renew_leaf.is_some(),
                use_quantized_grad == Some(true),
                "`use_quantized_grad=true`",
            ),
            (
                "diffusion_temperature",
                diffusion_temperature.is_some(),
                langevin == Some(true),
                "`langevin=true`",
            ),
            (
                "model_shrink_mode",
                model_shrink_mode.is_some(),
                model_shrink_rate.is_some(),
                "`model_shrink_rate`",
            ),
        ];
        for (key, set, on, needs) in switches {
            if set && !on {
                return Err(HessboostError::invalid_param(
                    key,
                    format!("applies only with {needs}"),
                ));
            }
        }
        let booster = match booster.unwrap_or(FlatBooster::GbTree) {
            FlatBooster::GbTree => BoosterKind::GbTree,
            FlatBooster::GbLinear => BoosterKind::GbLinear,
            FlatBooster::Dart => {
                let mut dart = Dart::builder();
                if let Some(rate_drop) = rate_drop {
                    dart = dart.rate_drop(rate_drop);
                }
                if let Some(skip_drop) = skip_drop {
                    dart = dart.skip_drop(skip_drop);
                }
                if let Some(one_drop) = one_drop {
                    dart = dart.one_drop(one_drop);
                }
                BoosterKind::Dart(dart.build()?)
            }
            FlatBooster::Boulevard => {
                let mut boulevard = Boulevard::builder();
                if let Some(dropout) = boulevard_dropout {
                    boulevard = boulevard.dropout(dropout);
                }
                if let Some(truncation) = boulevard_truncation {
                    boulevard = boulevard.truncation(truncation);
                }
                BoosterKind::Boulevard(boulevard.build()?)
            }
        };
        let process_type = match process_type.unwrap_or(FlatProcess::Default) {
            FlatProcess::Default => ProcessType::Default,
            FlatProcess::Update => ProcessType::Update(match refresh_leaf {
                Some(false) => Refresh::stats_only(),
                Some(true) | None => Refresh::default(),
            }),
        };
        let extra_trees = (extra_trees == Some(true))
            .then(|| extra_seed.map_or_else(ExtraTrees::default, ExtraTrees::with_seed));
        let linear_tree = if linear_tree == Some(true) {
            Some(LinearTree::new(linear_lambda.unwrap_or_default())?)
        } else {
            None
        };
        let quantized = if use_quantized_grad == Some(true) {
            let mut q = QuantizedGrad::builder();
            if let Some(bins) = num_grad_quant_bins {
                q = q.bins(bins);
            }
            if let Some(stochastic) = stochastic_rounding {
                q = q.stochastic_rounding(stochastic);
            }
            if let Some(renew) = quant_train_renew_leaf {
                q = q.renew_leaf(renew);
            }
            Some(q.build()?)
        } else {
            None
        };
        // LightGBM's default of 1 for both fractions is no balanced bagging.
        let balanced_bagging = match (pos_bagging_fraction, neg_bagging_fraction) {
            (None, None) => None,
            (pos, neg) => {
                let (pos, neg) = (pos.unwrap_or(1.0), neg.unwrap_or(1.0));
                (pos != 1.0 || neg != 1.0)
                    .then(|| BalancedBagging::new(pos, neg))
                    .transpose()?
            }
        };
        // LightGBM's `bagging_by_query` turns `bagging_fraction`
        // (`subsample`) into the fraction of queries kept.
        let (subsample, bagging_by_query) = if bagging_by_query == Some(true) {
            let fraction = subsample.unwrap_or(1.0);
            if fraction >= 1.0 {
                return Err(HessboostError::invalid_param(
                    "bagging_by_query",
                    format!(
                        "needs `subsample` < 1, the fraction of queries kept each round, \
                         got {fraction}"
                    ),
                ));
            }
            (None, Some(QueryBagging::new(fraction)?))
        } else {
            (subsample, None)
        };
        // CatBoost: posterior sampling needs Langevin "not set or true".
        if posterior_sampling == Some(true) && langevin == Some(false) {
            return Err(HessboostError::invalid_param(
                "langevin",
                "`posterior_sampling` requires Langevin boosting; leave `langevin` unset or true",
            ));
        }
        let langevin = if langevin == Some(true) {
            let mut l = Langevin::builder();
            if let Some(temperature) = diffusion_temperature {
                l = l.diffusion_temperature(temperature);
            }
            Some(l.build()?)
        } else {
            None
        };
        let model_shrink = match model_shrink_rate {
            Some(rate) => {
                let mut shrink = ModelShrink::builder().rate(rate);
                if let Some(mode) = model_shrink_mode {
                    shrink = shrink.mode(mode);
                }
                Some(shrink.build()?)
            }
            None => None,
        };
        let d = TrainingParams::default();
        Ok(TrainingParams {
            booster,
            nthread: nthread.map_or(d.nthread, NonZeroUsize::new),
            seed: seed.unwrap_or(d.seed),
            device: device.unwrap_or(d.device),
            objective,
            base_score: base_score.unwrap_or(d.base_score),
            eval_metric,
            eta: eta.unwrap_or(d.eta),
            gamma: gamma.unwrap_or(d.gamma),
            max_depth: max_depth.map_or(d.max_depth, NonZeroUsize::new),
            max_leaves: max_leaves.map_or(d.max_leaves, NonZeroUsize::new),
            min_child_weight: min_child_weight.unwrap_or(d.min_child_weight),
            max_delta_step: match max_delta_step.flatten() {
                None => MaxDeltaStep::ObjectiveDefault,
                Some(0.0) => MaxDeltaStep::Unbounded,
                Some(bound) => MaxDeltaStep::Bounded(bound),
            },
            subsample: subsample.unwrap_or(d.subsample),
            colsample_bytree: colsample_bytree.unwrap_or(d.colsample_bytree),
            colsample_bylevel: colsample_bylevel.unwrap_or(d.colsample_bylevel),
            colsample_bynode: colsample_bynode.unwrap_or(d.colsample_bynode),
            lambda: lambda.unwrap_or(d.lambda),
            alpha: alpha.unwrap_or(d.alpha),
            tree_method: tree_method.unwrap_or(d.tree_method),
            grow_policy: grow_policy.unwrap_or(d.grow_policy),
            max_bin: max_bin.unwrap_or(d.max_bin),
            monotone_constraints: monotone_constraints.unwrap_or(d.monotone_constraints),
            interaction_constraints: interaction_constraints.unwrap_or(d.interaction_constraints),
            num_parallel_tree: num_parallel_tree.unwrap_or(d.num_parallel_tree),
            sampling_method: sampling_method.unwrap_or(d.sampling_method),
            balanced_bagging,
            bagging_by_query,
            multi_strategy: multi_strategy.unwrap_or(d.multi_strategy),
            process_type,
            extra_trees,
            path_smooth: path_smooth.unwrap_or(d.path_smooth),
            linear_tree,
            quantized,
            toad_penalty_feature: toad_penalty_feature.unwrap_or(d.toad_penalty_feature),
            toad_penalty_threshold: toad_penalty_threshold.unwrap_or(d.toad_penalty_threshold),
            langevin,
            model_shrink,
            posterior_sampling: posterior_sampling.unwrap_or(d.posterior_sampling),
        })
    }
}

/// The flat keys and values of `objective`'s own parameters (a default
/// split direction is left out).
fn objective_keys(objective: &Objective) -> Vec<(&'static str, Value)> {
    let parts = objective.parts();
    objective
        .parameter_keys()
        .iter()
        .filter_map(|&key| {
            let value = match key {
                "num_class" => json(parts.num_class),
                "scale_pos_weight" => json(parts.scale_pos_weight),
                "tweedie_variance_power" => json(parts.tweedie_variance_power),
                "huber_slope" => json(parts.huber_slope),
                "lambdarank_num_pair_per_sample" => json(parts.lambdarank_num_pair_per_sample),
                "quantile_alpha" => json(&parts.quantile_alpha),
                "expectile_alpha" => json(&parts.expectile_alpha),
                "aft_loss_distribution" => json(parts.aft_loss_distribution),
                "aft_loss_distribution_scale" => json(parts.aft_loss_distribution_scale),
                "dist_gradient" => json(parts.dist_gradient),
                "dist_split_direction" => json(parts.dist_split_direction?),
                _ => return None,
            };
            Some((key, value))
        })
        .collect()
}

/// The flat keys and values `metric`'s parameters take in XGBoost's form,
/// where they are the objective's parameters (`mphe`'s `huber_slope`, ...);
/// `nll` and `crps` take the `dist:*` objective's family, so another family
/// has no flat form.
fn metric_keys(metric: &EvalMetric, objective: &Objective) -> Result<Vec<(&'static str, Value)>> {
    Ok(match metric {
        EvalMetric::Mphe(huber) => vec![("huber_slope", json(huber.slope()))],
        EvalMetric::Quantile(q) => vec![("quantile_alpha", json(q.alpha()))],
        EvalMetric::Expectile(e) => vec![("expectile_alpha", json(e.alpha()))],
        EvalMetric::AftNLogLik(aft) => vec![
            ("aft_loss_distribution", json(aft.distribution())),
            ("aft_loss_distribution_scale", json(aft.scale())),
        ],
        EvalMetric::Nll(family) | EvalMetric::Crps(family)
            if objective.dist_family() != Some(*family) =>
        {
            return Err(HessboostError::invalid_param(
                "eval_metric",
                format!(
                    "`{}` of the `{}` family has no flat form with objective `{}` (XGBoost's \
                     form takes the family from the `dist:*` objective)",
                    metric.name(),
                    family.objective_name(),
                    objective.name()
                ),
            ));
        }
        _ => Vec::new(),
    })
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
    /// that setting. XGBoost's `0` for `max_depth`, `max_leaves`, and
    /// `nthread` reads as `None` (no limit, the global pool), and
    /// `max_delta_step` as a [`MaxDeltaStep`]: absent or `null` is
    /// `ObjectiveDefault`, `0` is `Unbounded`, anything else `Bounded`.
    ///
    /// # Errors
    ///
    /// Refuses, never ignores: an unknown key (suggesting the closest
    /// known one), a key set twice (directly and through an alias), a value
    /// of the wrong type, `missing` (a property of the data), a one-setting
    /// option at another setting, an objective parameter that neither the
    /// objective nor a listed metric reads (e.g. `num_class` with
    /// `binary:logistic`; the error names the objectives and metrics that
    /// read it), `lambdarank_pair_method` without a `rank:*` objective,
    /// `updater`/`feature_selector` without `booster = gblinear`, and every
    /// configuration [`validate`](Self::validate) refuses.
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
        let mut fixed_set = Vec::new();
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
                fixed_set.push(name);
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
        let params = flat_from("params", settings)?.into_params()?;
        for name in fixed_set {
            let gblinear = params.booster == BoosterKind::GbLinear;
            let refusal = match name {
                "updater" if !gblinear => {
                    "`coord_descent` is gblinear's updater; tree boosters take no `updater`"
                }
                "feature_selector" if !gblinear => {
                    "is gblinear's coordinate selection; tree boosters take no `feature_selector`"
                }
                "lambdarank_pair_method"
                    if !matches!(
                        params.objective,
                        Objective::RankPairwise(_) | Objective::RankNdcg(_) | Objective::RankMap(_)
                    ) =>
                {
                    "applies only to the LambdaMART `rank:*` objectives"
                }
                _ => continue,
            };
            return Err(HessboostError::invalid_param(name, refusal));
        }
        params.validate()?;
        Ok(params)
    }

    /// This configuration in XGBoost's flat parameter form, under the
    /// canonical keys [`from_xgboost`](Self::from_xgboost) reads back to the
    /// same configuration: the objective's name and the keys of its own
    /// parameters, the metrics' names and the objective keys they borrow,
    /// monotone constraints as `-1`/`0`/`1`, `None` limits as `0`; an unset
    /// `base_score` or [`MaxDeltaStep::ObjectiveDefault`] is left out.
    ///
    /// # Errors
    ///
    /// A configuration [`validate`](Self::validate) refuses, or one
    /// XGBoost's form cannot state: a custom loss, or a
    /// metric whose parameters differ from the objective's (XGBoost's
    /// `mphe`, `quantile`, `expectile`, and `aft-nloglik` read the same
    /// keys as the objective, and `nll` / `crps` its `dist:*` family).
    pub fn to_xgboost(&self) -> Result<Map<String, Value>> {
        let TrainingParams {
            booster,
            nthread,
            seed,
            device,
            objective,
            base_score,
            eval_metric,
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
            tree_method,
            grow_policy,
            max_bin,
            monotone_constraints,
            interaction_constraints,
            num_parallel_tree,
            sampling_method,
            balanced_bagging,
            bagging_by_query,
            multi_strategy,
            process_type,
            extra_trees,
            path_smooth,
            linear_tree,
            quantized,
            toad_penalty_feature,
            toad_penalty_threshold,
            langevin,
            model_shrink,
            posterior_sampling,
        } = self;
        if let Objective::Custom(loss) = objective {
            return Err(HessboostError::invalid_param(
                "objective",
                format!("the custom loss `{}` has no XGBoost flat form", loss.name()),
            ));
        }
        // The fields are public: an invalid value (a NaN bound) would be
        // written as one that reads back as another (`null`, unset).
        self.validate()?;
        let mut objective_keys = objective_keys(objective);
        for metric in eval_metric {
            for (key, value) in metric_keys(metric, objective)? {
                match objective_keys.iter().find(|(k, _)| *k == key) {
                    Some((_, set)) if *set != value => {
                        return Err(HessboostError::invalid_param(
                            "eval_metric",
                            format!(
                                "`{}` reads `{key}` = {value} in XGBoost's flat form, which the \
                                 configuration sets to {set}",
                                metric.name()
                            ),
                        ));
                    }
                    Some(_) => {}
                    None => objective_keys.push((key, value)),
                }
            }
        }
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
        match booster {
            BoosterKind::GbTree => set("booster", json("gbtree")),
            BoosterKind::GbLinear => set("booster", json("gblinear")),
            BoosterKind::Dart(dart) => {
                set("booster", json("dart"));
                set("rate_drop", json(dart.rate_drop()));
                set("skip_drop", json(dart.skip_drop()));
                set("one_drop", json(dart.one_drop()));
            }
            BoosterKind::Boulevard(boulevard) => {
                set("booster", json("boulevard"));
                set("boulevard_dropout", json(boulevard.dropout()));
                set("boulevard_truncation", json(boulevard.truncation()));
            }
        }
        set("nthread", json(nthread.map_or(0, NonZeroUsize::get)));
        set("seed", json(seed));
        set("device", json(device));
        set("objective", json(objective.name()));
        for (key, value) in objective_keys {
            set(key, value);
        }
        if let Some(base_score) = base_score {
            set("base_score", json(base_score));
        }
        let names: Vec<_> = eval_metric.iter().map(EvalMetric::flat_name).collect();
        set("eval_metric", json(names));
        set("eta", json(eta));
        set("gamma", json(gamma));
        set("max_depth", json(max_depth.map_or(0, NonZeroUsize::get)));
        set("max_leaves", json(max_leaves.map_or(0, NonZeroUsize::get)));
        set("min_child_weight", json(min_child_weight));
        match max_delta_step {
            MaxDeltaStep::ObjectiveDefault => {}
            MaxDeltaStep::Unbounded => set("max_delta_step", json(0.0)),
            MaxDeltaStep::Bounded(bound) => set("max_delta_step", json(bound)),
        }
        match bagging_by_query {
            Some(bagging) => {
                set("bagging_by_query", json(true));
                set("subsample", json(bagging.fraction()));
            }
            None => set("subsample", json(subsample)),
        }
        set("colsample_bytree", json(colsample_bytree));
        set("colsample_bylevel", json(colsample_bylevel));
        set("colsample_bynode", json(colsample_bynode));
        set("lambda", json(lambda));
        set("alpha", json(alpha));
        set("tree_method", json(tree_method));
        set("grow_policy", json(grow_policy));
        set("max_bin", json(max_bin));
        set("monotone_constraints", json(monotone));
        set("interaction_constraints", json(interaction_constraints));
        set("num_parallel_tree", json(num_parallel_tree));
        set("sampling_method", json(sampling_method));
        if let Some(bagging) = balanced_bagging {
            set("pos_bagging_fraction", json(bagging.pos_fraction()));
            set("neg_bagging_fraction", json(bagging.neg_fraction()));
        }
        set("multi_strategy", json(multi_strategy));
        match process_type {
            ProcessType::Default => set("process_type", json("default")),
            ProcessType::Update(refresh) => {
                set("process_type", json("update"));
                set("refresh_leaf", json(refresh.refresh_leaf()));
            }
        }
        set("extra_trees", json(extra_trees.is_some()));
        if let Some(extra_trees) = extra_trees {
            set("extra_seed", json(extra_trees.seed()));
        }
        set("path_smooth", json(path_smooth));
        set("linear_tree", json(linear_tree.is_some()));
        if let Some(linear_tree) = linear_tree {
            set("linear_lambda", json(linear_tree.lambda()));
        }
        set("use_quantized_grad", json(quantized.is_some()));
        if let Some(quantized) = quantized {
            set("num_grad_quant_bins", json(quantized.bins()));
            set("stochastic_rounding", json(quantized.stochastic_rounding()));
            set("quant_train_renew_leaf", json(quantized.renew_leaf()));
        }
        set("toad_penalty_feature", json(toad_penalty_feature));
        set("toad_penalty_threshold", json(toad_penalty_threshold));
        if let Some(langevin) = langevin {
            set("langevin", json(true));
            if let Some(temperature) = langevin.diffusion_temperature() {
                set("diffusion_temperature", json(temperature));
            }
        }
        if let Some(shrink) = model_shrink {
            set("model_shrink_rate", json(shrink.rate()));
            set("model_shrink_mode", json(shrink.mode()));
        }
        set("posterior_sampling", json(posterior_sampling));
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
            base_score,
            eval_metric,
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
            tree_method,
            grow_policy,
            max_bin,
            monotone_constraints,
            interaction_constraints,
            num_parallel_tree,
            sampling_method,
            balanced_bagging,
            bagging_by_query,
            multi_strategy,
            process_type,
            extra_trees,
            path_smooth,
            linear_tree,
            quantized,
            toad_penalty_feature,
            toad_penalty_threshold,
            langevin,
            model_shrink,
            posterior_sampling,
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
        differs("base_score", *base_score == other.base_score);
        differs("eval_metric", *eval_metric == other.eval_metric);
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
        // Each fraction as LightGBM states it, 1 when balanced bagging is off.
        let fractions = |bagging: &Option<BalancedBagging>| {
            bagging.map_or((1.0, 1.0), |b| (b.pos_fraction(), b.neg_fraction()))
        };
        let (ours, theirs) = (
            fractions(balanced_bagging),
            fractions(&other.balanced_bagging),
        );
        differs("pos_bagging_fraction", ours.0 == theirs.0);
        differs("neg_bagging_fraction", ours.1 == theirs.1);
        differs(
            "bagging_by_query",
            *bagging_by_query == other.bagging_by_query,
        );
        differs("multi_strategy", *multi_strategy == other.multi_strategy);
        differs("process_type", *process_type == other.process_type);
        differs("extra_trees", *extra_trees == other.extra_trees);
        differs("path_smooth", *path_smooth == other.path_smooth);
        differs("linear_tree", *linear_tree == other.linear_tree);
        differs("use_quantized_grad", *quantized == other.quantized);
        differs(
            "toad_penalty_feature",
            *toad_penalty_feature == other.toad_penalty_feature,
        );
        differs(
            "toad_penalty_threshold",
            *toad_penalty_threshold == other.toad_penalty_threshold,
        );
        differs("langevin", *langevin == other.langevin);
        differs("model_shrink_rate", *model_shrink == other.model_shrink);
        differs(
            "posterior_sampling",
            *posterior_sampling == other.posterior_sampling,
        );
        changed.sort_unstable();
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objective::distributional::{DistFamily, Distributional};
    use crate::objective::{
        Aft, Expectiles, LambdaRank, Logistic, Multiclass, PseudoHuber, Quantiles, Tweedie,
    };
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
            ("objective", json!("reg:quantileerror")),
            ("aft_loss_distribution", json!("extreme")),
            ("sampling_method", json!("gradient_based")),
            ("multi_strategy", json!("multi_output_tree")),
            ("process_type", json!("update")),
            ("refresh_leaf", json!(false)),
            ("num_parallel_tree", json!(4)),
            ("quantile_alpha", json!(0.25)),
            ("eval_metric", json!("aft-nloglik")),
            ("monotone_constraints", json!("(1,-1,0)")),
            ("interaction_constraints", json!("[[0, 1], [2]]")),
            ("learning_rate", json!(0.1)),
            ("reg_lambda", json!(2.0)),
            ("max_delta_step", json!(0.0)),
        ])
        .unwrap();
        assert_eq!(
            p.objective,
            Objective::Quantile(Quantiles::new([0.25]).unwrap())
        );
        assert_eq!(
            p.eval_metric,
            [EvalMetric::AftNLogLik(Aft::with_distribution(
                AftDistribution::Extreme
            ))]
        );
        assert_eq!(p.sampling_method, SamplingMethod::GradientBased);
        assert_eq!(p.multi_strategy, MultiStrategy::MultiOutputTree);
        assert_eq!(p.process_type, ProcessType::Update(Refresh::stats_only()));
        assert_eq!(p.num_parallel_tree, 4);
        assert_eq!(
            p.monotone_constraints,
            [Monotone::Increasing, Monotone::Decreasing, Monotone::None]
        );
        assert_eq!(p.interaction_constraints, [vec![0, 1], vec![2]]);
        assert_eq!((p.eta, p.lambda), (0.1, 2.0));
        assert_eq!(p.max_delta_step, MaxDeltaStep::Unbounded);

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
        let rank = TrainingParams::from_xgboost([
            ("objective", json!("rank:ndcg")),
            ("lambdarank_pair_method", json!("topk")),
        ])
        .unwrap();
        assert_eq!(rank.objective, Objective::RankNdcg(LambdaRank::default()));
    }

    /// A key of an option group means nothing while the group's switch is
    /// off, so it is refused by name rather than ignored.
    #[test]
    fn dependent_keys_without_their_switch_are_refused_by_name() {
        for (pairs, key) in [
            (json!({"booster": "gbtree", "rate_drop": 0.1}), "rate_drop"),
            (json!({"one_drop": true}), "one_drop"),
            (json!({"refresh_leaf": false}), "refresh_leaf"),
            (json!({"extra_seed": 3}), "extra_seed"),
            (json!({"extra_trees": false, "extra_seed": 3}), "extra_seed"),
            (json!({"linear_lambda": 0.5}), "linear_lambda"),
            (json!({"num_grad_quant_bins": 8}), "num_grad_quant_bins"),
        ] {
            let refusal = refused(pairs.clone()).unwrap_or_else(|| panic!("{pairs} accepted"));
            assert!(
                refusal.starts_with(&format!("invalid parameter `{key}`")),
                "{pairs}: {refusal}"
            );
        }
    }

    /// LightGBM's class fractions become one [`BalancedBagging`] (a missing
    /// one at LightGBM's default 1, both at 1 meaning off) and read back
    /// from the flat form; the fractions are refused by name where nothing
    /// would bag by class.
    #[test]
    fn balanced_bagging_reads_lightgbm_fractions() {
        let binary = json!("binary:logistic");
        let p = TrainingParams::from_xgboost([
            ("objective", binary.clone()),
            ("neg_bagging_fraction", json!(0.2)),
        ])
        .unwrap();
        assert_eq!(
            p.balanced_bagging,
            Some(BalancedBagging::new(1.0, 0.2).unwrap())
        );
        assert_eq!(
            TrainingParams::from_xgboost(p.to_xgboost().unwrap()).unwrap(),
            p
        );
        let off = TrainingParams::from_xgboost([
            ("objective", binary.clone()),
            ("pos_bagging_fraction", json!(1.0)),
            ("neg_bagging_fraction", json!(1.0)),
        ])
        .unwrap();
        assert_eq!(off.balanced_bagging, None);
        for (pairs, key) in [
            (json!({"pos_bagging_fraction": 0.5}), "pos_bagging_fraction"),
            (
                json!({"objective": binary, "neg_bagging_fraction": 0.5, "subsample": 0.8}),
                "subsample",
            ),
            (
                json!({"objective": binary, "neg_bagging_fraction": 0.0}),
                "neg_bagging_fraction",
            ),
        ] {
            let refusal = refused(pairs.clone()).unwrap_or_else(|| panic!("{pairs} accepted"));
            assert!(
                refusal.starts_with(&format!("invalid parameter `{key}`")),
                "{pairs}: {refusal}"
            );
        }
    }

    /// LightGBM's `bagging_by_query` reads `subsample` as the fraction of
    /// queries kept, and writes it back there; without a fraction below 1,
    /// or with a non-ranking objective, it is refused by name.
    #[test]
    fn bagging_by_query_reads_subsample_as_the_query_fraction() {
        let p = TrainingParams::from_xgboost([
            ("objective", json!("rank:xendcg")),
            ("bagging_by_query", json!(true)),
            ("subsample", json!(0.7)),
        ])
        .unwrap();
        assert_eq!(p.bagging_by_query, Some(QueryBagging::new(0.7).unwrap()));
        assert_eq!(p.subsample, 1.0);
        let flat = p.to_xgboost().unwrap();
        assert_eq!(
            (&flat["bagging_by_query"], &flat["subsample"]),
            (&json!(true), &json!(0.7))
        );
        assert_eq!(TrainingParams::from_xgboost(flat).unwrap(), p);
        let off = TrainingParams::from_xgboost([
            ("bagging_by_query", json!(false)),
            ("subsample", json!(0.7)),
        ])
        .unwrap();
        assert_eq!((off.bagging_by_query, off.subsample), (None, 0.7));
        for pairs in [
            json!({"objective": "rank:ndcg", "bagging_by_query": true}),
            json!({"bagging_by_query": true, "subsample": 0.5}),
        ] {
            let refusal = refused(pairs.clone()).unwrap_or_else(|| panic!("{pairs} accepted"));
            assert!(
                refusal.starts_with("invalid parameter `bagging_by_query`"),
                "{pairs}: {refusal}"
            );
        }
    }

    /// Every option group, switched on with non-default values, reads back
    /// from its flat form unchanged.
    #[test]
    fn option_groups_round_trip_through_the_flat_form() {
        let p = TrainingParams {
            booster: BoosterKind::Dart(
                Dart::builder()
                    .rate_drop(0.2)
                    .skip_drop(0.3)
                    .one_drop(true)
                    .build()
                    .unwrap(),
            ),
            process_type: ProcessType::Update(Refresh::stats_only()),
            extra_trees: Some(ExtraTrees::with_seed(11)),
            linear_tree: Some(LinearTree::new(0.5).unwrap()),
            quantized: Some(
                QuantizedGrad::builder()
                    .bins(8)
                    .stochastic_rounding(false)
                    .renew_leaf(true)
                    .build()
                    .unwrap(),
            ),
            ..TrainingParams::default()
        };
        let flat = p.to_xgboost().unwrap();
        assert_eq!(TrainingParams::from_xgboost(flat).unwrap(), p);
    }

    /// XGBoost's `0` limits are `None`, and `max_delta_step` keeps its three
    /// states: absent or `null` is the objective's default, `0` no bound,
    /// anything else a bound. Each reads back from its flat form.
    #[test]
    fn sentinels_map_to_typed_states_and_back() {
        let parse = |pairs: Value| TrainingParams::from_xgboost(pairs.as_object().unwrap().clone());
        let p = parse(
            json!({"max_depth": 0, "max_leaves": 0, "nthread": 0, "grow_policy": "depthwise"}),
        )
        .unwrap();
        assert_eq!((p.max_depth, p.max_leaves, p.nthread), (None, None, None));
        let p = parse(json!({"max_depth": 3, "max_leaves": 7, "nthread": 2})).unwrap();
        assert_eq!(
            (p.max_depth, p.max_leaves, p.nthread),
            (
                NonZeroUsize::new(3),
                NonZeroUsize::new(7),
                NonZeroUsize::new(2)
            )
        );
        for (value, step) in [
            (None, MaxDeltaStep::ObjectiveDefault),
            (Some(json!(null)), MaxDeltaStep::ObjectiveDefault),
            (Some(json!(0.0)), MaxDeltaStep::Unbounded),
            (Some(json!(0.5)), MaxDeltaStep::Bounded(0.5)),
        ] {
            let mut pairs = json!({"objective": "count:poisson"});
            if let Some(value) = value.clone() {
                pairs["max_delta_step"] = value;
            }
            let p = parse(pairs).unwrap();
            assert_eq!(p.max_delta_step, step, "{value:?}");
            let back = TrainingParams::from_xgboost(p.to_xgboost().unwrap()).unwrap();
            assert_eq!(back, p, "{value:?}");
        }
        let unlimited = TrainingParams {
            max_depth: None,
            grow_policy: GrowPolicy::LossGuide,
            max_leaves: NonZeroUsize::new(5),
            ..TrainingParams::default()
        };
        let flat = unlimited.to_xgboost().unwrap();
        assert_eq!(
            (flat["max_depth"].clone(), flat["nthread"].clone()),
            (json!(0), json!(0))
        );
        assert_eq!(TrainingParams::from_xgboost(flat).unwrap(), unlimited);
        let negative = parse(json!({"max_delta_step": -1.0})).unwrap_err();
        assert!(
            negative
                .to_string()
                .starts_with("invalid parameter `max_delta_step`")
        );
    }

    /// Every built-in objective's flat form (its name and the keys of its
    /// own parameters) reads back to the same objective.
    #[test]
    fn typed_objectives_round_trip_through_the_flat_form() {
        let dist = Distributional::new(DistFamily::Gamma).with_gradient(DistGradient::Natural);
        let objectives = [
            Objective::SquaredError,
            Objective::SquaredLogError,
            Objective::PseudoHuber(PseudoHuber::new(0.4).unwrap()),
            Objective::AbsoluteError,
            Objective::Quantile(Quantiles::new([0.1, 0.5, 0.9]).unwrap()),
            Objective::Expectile(Expectiles::new([0.2, 0.8]).unwrap()),
            Objective::RegLogistic(Logistic::new(3.0).unwrap()),
            Objective::BinaryLogistic(Logistic::new(2.5).unwrap()),
            Objective::BinaryLogitRaw(Logistic::new(0.5).unwrap()),
            Objective::BinaryHinge,
            Objective::Softmax(Multiclass::new(4).unwrap()),
            Objective::Softprob(Multiclass::new(3).unwrap()),
            Objective::Poisson,
            Objective::Gamma,
            Objective::Tweedie(Tweedie::new(1.3).unwrap()),
            Objective::RankPairwise(LambdaRank::new(4).unwrap()),
            Objective::RankNdcg(LambdaRank::new(8).unwrap()),
            Objective::RankMap(LambdaRank::default()),
            Objective::RankXendcg,
            Objective::Cox,
            Objective::Aft(Aft::new(AftDistribution::Logistic, 1.7).unwrap()),
            Objective::Dist(dist),
        ];
        for objective in objectives {
            let p = TrainingParams {
                objective: objective.clone(),
                ..TrainingParams::default()
            };
            let flat = p.to_xgboost().unwrap();
            assert_eq!(flat["objective"], json!(objective.name()));
            let back = TrainingParams::from_xgboost(flat).unwrap();
            assert_eq!(back.objective, objective, "{}", objective.name());
            assert_eq!(back, p, "{}", objective.name());
        }
        // A split direction needs shared vector-leaf trees.
        let p = TrainingParams {
            objective: Objective::Dist(dist.with_split_direction(DistSplitDirection::Cyclic)),
            multi_strategy: MultiStrategy::MultiOutputTree,
            ..TrainingParams::default()
        };
        let flat = p.to_xgboost().unwrap();
        assert_eq!(flat["dist_split_direction"], json!("cyclic"));
        assert_eq!(TrainingParams::from_xgboost(flat).unwrap(), p);
    }

    /// An objective-parameter key the objective does not read is refused by
    /// name, unless a configured metric borrows it (as XGBoost's metrics
    /// read the objective's parameters).
    #[test]
    fn objective_keys_nothing_reads_are_refused_by_name() {
        for (pairs, key) in [
            (
                json!({"objective": "binary:logistic", "num_class": 3}),
                "num_class",
            ),
            (
                json!({"objective": "reg:squarederror", "scale_pos_weight": 2.0}),
                "scale_pos_weight",
            ),
            (
                json!({"objective": "reg:gamma", "dist_gradient": "hessian"}),
                "dist_gradient",
            ),
            (
                json!({"objective": "reg:squarederror", "huber_slope": 0.5}),
                "huber_slope",
            ),
            (
                json!({"objective": "multi:softprob", "num_class": 3, "tweedie_variance_power": 1.2}),
                "tweedie_variance_power",
            ),
        ] {
            let refusal = refused(pairs.clone()).unwrap_or_else(|| panic!("{pairs} accepted"));
            assert!(
                refusal.starts_with(&format!("invalid parameter `{key}`")),
                "{pairs}: {refusal}"
            );
        }
        let p = TrainingParams::from_xgboost([
            ("objective", json!("reg:squarederror")),
            ("eval_metric", json!("mphe")),
            ("huber_slope", json!(0.5)),
        ])
        .unwrap();
        assert_eq!(p.objective, Objective::SquaredError);
        assert_eq!(
            p.eval_metric,
            [EvalMetric::Mphe(PseudoHuber::new(0.5).unwrap())]
        );
    }

    /// XGBoost's metrics read the flat parameters whatever the objective:
    /// `mphe` the `huber_slope`, `quantile` / `expectile` the alpha lists,
    /// `aft-nloglik` the AFT noise, and `nll` / `crps` the `dist:*` family.
    /// A metric with other parameters has no flat form.
    /// The fields are public, so a configuration can be invalid: it is
    /// refused by name rather than written as a flat form that reads back
    /// as a different one (a NaN bound or base score as `null`, i.e. unset).
    #[test]
    fn invalid_configurations_are_refused_not_serialized() {
        for (p, key) in [
            (
                TrainingParams {
                    max_delta_step: MaxDeltaStep::Bounded(f64::NAN),
                    ..TrainingParams::default()
                },
                "max_delta_step",
            ),
            (
                TrainingParams {
                    base_score: Some(f64::NAN),
                    ..TrainingParams::default()
                },
                "base_score",
            ),
        ] {
            match p.to_xgboost() {
                Err(HessboostError::InvalidParameter { name, .. }) => assert_eq!(name, key),
                other => panic!("{key}: expected a refusal, got {other:?}"),
            }
        }
    }

    /// A Tweedie metric's variance power survives the flat form in full:
    /// its `evals_result` key rounds to six digits, its flat spelling not.
    #[test]
    fn tweedie_metric_powers_round_trip_exactly() {
        for power in ["1.999999", "1.23456789", "1.5"] {
            let p = TrainingParams::from_xgboost([(
                "eval_metric",
                json!(format!("tweedie-nloglik@{power}")),
            )])
            .unwrap();
            let back = TrainingParams::from_xgboost(p.to_xgboost().unwrap())
                .unwrap_or_else(|e| panic!("{power}: {e}"));
            assert_eq!(back.eval_metric, p.eval_metric, "{power}");
        }
    }

    #[test]
    fn metrics_take_the_flat_parameters_xgboost_gives_them() {
        use crate::objective::distributional::DistFamily;
        let p = TrainingParams::from_xgboost([
            ("huber_slope", json!(0.7)),
            ("quantile_alpha", json!([0.2, 0.8])),
            ("aft_loss_distribution", json!("logistic")),
            (
                "eval_metric",
                json!(["mphe", "quantile", "aft-nloglik", "ndcg@3"]),
            ),
        ])
        .unwrap();
        assert_eq!(
            p.eval_metric,
            [
                EvalMetric::Mphe(PseudoHuber::new(0.7).unwrap()),
                EvalMetric::Quantile(Quantiles::new([0.2, 0.8]).unwrap()),
                EvalMetric::AftNLogLik(Aft::with_distribution(AftDistribution::Logistic)),
                EvalMetric::Ndcg(crate::metric::Cutoff::top(3).unwrap()),
            ]
        );
        let flat = p.to_xgboost().unwrap();
        assert_eq!(
            flat["eval_metric"],
            json!(["mphe", "quantile", "aft-nloglik", "ndcg@3"])
        );
        assert_eq!(TrainingParams::from_xgboost(flat).unwrap(), p);

        // One flat `huber_slope` serves the objective and every metric, so
        // metrics or an objective with another slope have no flat form.
        let slope = |s| PseudoHuber::new(s).unwrap();
        let mut other_slope = p.clone();
        other_slope.eval_metric = vec![EvalMetric::Mphe(slope(2.0))];
        let flat = other_slope.to_xgboost().unwrap();
        assert_eq!(flat["huber_slope"], json!(2.0));
        assert_eq!(TrainingParams::from_xgboost(flat).unwrap(), other_slope);
        other_slope.eval_metric.push(EvalMetric::Mphe(slope(0.7)));
        assert!(other_slope.to_xgboost().is_err());
        let huber = TrainingParams {
            objective: Objective::PseudoHuber(slope(0.7)),
            eval_metric: vec![EvalMetric::Mphe(slope(2.0))],
            ..TrainingParams::default()
        };
        assert!(huber.to_xgboost().is_err());
        let no_dist = TrainingParams {
            eval_metric: vec![EvalMetric::Nll(DistFamily::Normal)],
            ..TrainingParams::default()
        };
        assert!(no_dist.to_xgboost().is_err());

        assert!(refused(json!({"eval_metric": "nll"})).is_some());
        assert!(refused(json!({"eval_metric": "quantile"})).is_some());
        let dist = TrainingParams::from_xgboost([
            ("objective", json!("dist:normal")),
            ("eval_metric", json!(["nll", "crps"])),
        ])
        .unwrap();
        assert_eq!(
            dist.eval_metric,
            [
                EvalMetric::Nll(DistFamily::Normal),
                EvalMetric::Crps(DistFamily::Normal)
            ]
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
