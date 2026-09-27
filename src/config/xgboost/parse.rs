//! Reading the flat form: [`TrainingParams::from_xgboost`] resolves keys and
//! value spellings into a [`Flat`], and [`Flat::into_params`] turns it into
//! the typed configuration.

use super::super::groups::{
    BalancedBagging, Boulevard, Dart, Ebm, EbmEarlyStopping, ExtraTrees, Langevin, LinearTree,
    ModelShrink, ModelShrinkMode, QuantizedGrad, QueryBagging, Refresh,
};
use super::super::params::{BoosterKind, MaxDeltaStep, ProcessType, TrainingParams};
use super::schema::{FIXED, Flat, FlatBooster, FlatProcess, FlatRate, canonical_key};
use crate::error::{HessboostError, Result};
use crate::metric::{EvalMetric, XgboostMetricSource};
use crate::objective::{OBJECTIVE_PARAMS, Objective, ObjectiveParts};
use serde_json::{Map, Value};

/// CatBoost's `model_shrink_rate` for `langevin=true` in the constant mode
/// when the rate is not given (`SetNotSpecifiedOptionsToDefaults`).
const LANGEVIN_SHRINK_RATE: f64 = 0.001;

/// The settings the mode switches own, built from their keys.
struct ModeOptions {
    process_type: ProcessType,
    extra_trees: Option<ExtraTrees>,
    linear_tree: Option<LinearTree>,
    quantized: Option<QuantizedGrad>,
    balanced_bagging: Option<BalancedBagging>,
    /// `subsample` unless query bagging reads it as its fraction.
    subsample: Option<f64>,
    bagging_by_query: Option<QueryBagging>,
    langevin: Option<Langevin>,
    model_shrink: Option<ModelShrink>,
}

impl Flat {
    /// The configuration these settings describe, every absent key at its
    /// default: the objective and metrics, the switches' dependent keys,
    /// the booster's and the modes' option groups, then the plain keys.
    fn into_params(self) -> Result<TrainingParams> {
        let (objective, eval_metric) = self.objective_and_metrics()?;
        self.check_switch_dependencies()?;
        let booster = self.booster_kind()?;
        let modes = self.mode_options()?;
        // Every key is listed, so a new one does not compile until handled;
        // those read above are `_`.
        let Flat {
            booster: _,
            nthread,
            seed,
            device,
            objective: _,
            num_class: _,
            base_score,
            eval_metric: _,
            tweedie_variance_power: _,
            huber_slope: _,
            lambdarank_num_pair_per_sample: _,
            quantile_alpha: _,
            expectile_alpha: _,
            aft_loss_distribution: _,
            aft_loss_distribution_scale: _,
            dist_gradient: _,
            dist_split_direction: _,
            eta,
            gamma,
            max_depth,
            max_leaves,
            min_child_weight,
            max_delta_step,
            subsample: _,
            colsample_bytree,
            colsample_bylevel,
            colsample_bynode,
            lambda,
            alpha,
            scale_pos_weight: _,
            tree_method,
            grow_policy,
            max_bin,
            monotone_constraints,
            interaction_constraints,
            num_parallel_tree,
            sampling_method,
            pos_bagging_fraction: _,
            neg_bagging_fraction: _,
            bagging_by_query: _,
            multi_strategy,
            process_type: _,
            refresh_leaf: _,
            extra_trees: _,
            extra_seed: _,
            path_smooth,
            linear_tree: _,
            linear_lambda: _,
            use_quantized_grad: _,
            num_grad_quant_bins: _,
            stochastic_rounding: _,
            quant_train_renew_leaf: _,
            rate_drop: _,
            skip_drop: _,
            one_drop: _,
            toad_penalty_feature,
            toad_penalty_threshold,
            langevin: _,
            diffusion_temperature: _,
            model_shrink_rate: _,
            model_shrink_mode: _,
            posterior_sampling,
            boulevard_dropout: _,
            boulevard_truncation: _,
            ebm_interactions: _,
            ebm_outer_bags: _,
            ebm_bag_fraction: _,
            ebm_boulevard: _,
            ebm_early_stopping_rounds: _,
            ebm_early_stopping_tolerance: _,
        } = self;
        let d = TrainingParams::default();
        Ok(TrainingParams {
            booster,
            nthread: nthread.map_or(d.nthread, |limit| limit.0),
            seed: seed.unwrap_or(d.seed),
            device: device.unwrap_or(d.device),
            objective,
            base_score: base_score.unwrap_or(d.base_score),
            eval_metric,
            eta: eta.unwrap_or(d.eta),
            gamma: gamma.unwrap_or(d.gamma),
            max_depth: max_depth.map_or(d.max_depth, |limit| limit.0),
            max_leaves: max_leaves.map_or(d.max_leaves, |limit| limit.0),
            min_child_weight: min_child_weight.unwrap_or(d.min_child_weight),
            max_delta_step: match max_delta_step.flatten() {
                None => MaxDeltaStep::ObjectiveDefault,
                Some(0.0) => MaxDeltaStep::Unbounded,
                Some(bound) => MaxDeltaStep::Bounded(bound),
            },
            subsample: modes.subsample.unwrap_or(d.subsample),
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
            balanced_bagging: modes.balanced_bagging,
            bagging_by_query: modes.bagging_by_query,
            multi_strategy: multi_strategy.unwrap_or(d.multi_strategy),
            process_type: modes.process_type,
            extra_trees: modes.extra_trees,
            path_smooth: path_smooth.unwrap_or(d.path_smooth),
            linear_tree: modes.linear_tree,
            quantized: modes.quantized,
            toad_penalty_feature: toad_penalty_feature.unwrap_or(d.toad_penalty_feature),
            toad_penalty_threshold: toad_penalty_threshold.unwrap_or(d.toad_penalty_threshold),
            langevin: modes.langevin,
            model_shrink: modes.model_shrink,
            posterior_sampling: posterior_sampling.unwrap_or(d.posterior_sampling),
        })
    }

    /// The objective, reading its parameters from their keys, and the
    /// metrics, read as XGBoost reads them (with the flat parameters they
    /// borrow). An objective-parameter key neither reads is refused.
    fn objective_and_metrics(&self) -> Result<(Objective, Vec<EvalMetric>)> {
        let p = ObjectiveParts::default();
        let parts = ObjectiveParts {
            num_class: self.num_class.unwrap_or(p.num_class),
            scale_pos_weight: self.scale_pos_weight.unwrap_or(p.scale_pos_weight),
            tweedie_variance_power: self
                .tweedie_variance_power
                .unwrap_or(p.tweedie_variance_power),
            huber_slope: self.huber_slope.unwrap_or(p.huber_slope),
            lambdarank_num_pair_per_sample: self
                .lambdarank_num_pair_per_sample
                .unwrap_or(p.lambdarank_num_pair_per_sample),
            quantile_alpha: self.quantile_alpha.clone().unwrap_or(p.quantile_alpha),
            expectile_alpha: self.expectile_alpha.clone().unwrap_or(p.expectile_alpha),
            aft_loss_distribution: self
                .aft_loss_distribution
                .unwrap_or(p.aft_loss_distribution),
            aft_loss_distribution_scale: self
                .aft_loss_distribution_scale
                .unwrap_or(p.aft_loss_distribution_scale),
            dist_gradient: self.dist_gradient.unwrap_or(p.dist_gradient),
            dist_split_direction: self.dist_split_direction.or(p.dist_split_direction),
        };
        let name = self.objective.as_deref().unwrap_or("reg:squarederror");
        let objective = Objective::from_parts(name, &parts)
            .ok_or_else(|| HessboostError::unknown("objective", name))??;
        let metric_names = self.eval_metric.as_deref().unwrap_or_default();
        for param in OBJECTIVE_PARAMS {
            let read = (param.read_by)(&objective)
                || metric_names
                    .iter()
                    .any(|metric| EvalMetric::borrowed_keys(metric).contains(&param.key));
            if self.is_set(param.key) && !read {
                return Err(HessboostError::invalid_param(
                    param.key,
                    format!(
                        "applies only to {}; nothing reads it with objective `{}`",
                        param.users,
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
        Ok((objective, eval_metric))
    }

    /// Refuse a group key set without its switch: they only mean something
    /// under it.
    fn check_switch_dependencies(&self) -> Result<()> {
        let dart = self.booster == Some(FlatBooster::Dart);
        let boulevard = self.booster == Some(FlatBooster::Boulevard);
        let ebm = self.booster == Some(FlatBooster::Ebm);
        let update = self.process_type == Some(FlatProcess::Update);
        let quantized = self.use_quantized_grad == Some(true);
        let switches = [
            (
                "rate_drop",
                self.rate_drop.is_some(),
                dart,
                "`booster=dart`",
            ),
            (
                "skip_drop",
                self.skip_drop.is_some(),
                dart,
                "`booster=dart`",
            ),
            ("one_drop", self.one_drop.is_some(), dart, "`booster=dart`"),
            (
                "boulevard_dropout",
                self.boulevard_dropout.is_some(),
                boulevard,
                "`booster=boulevard`",
            ),
            (
                "boulevard_truncation",
                self.boulevard_truncation.is_some(),
                boulevard,
                "`booster=boulevard`",
            ),
            (
                "ebm_interactions",
                self.ebm_interactions.is_some(),
                ebm,
                "`booster=ebm`",
            ),
            (
                "ebm_outer_bags",
                self.ebm_outer_bags.is_some(),
                ebm,
                "`booster=ebm`",
            ),
            (
                "ebm_bag_fraction",
                self.ebm_bag_fraction.is_some(),
                ebm,
                "`booster=ebm`",
            ),
            (
                "ebm_boulevard",
                self.ebm_boulevard.is_some(),
                ebm,
                "`booster=ebm`",
            ),
            (
                "ebm_early_stopping_rounds",
                self.ebm_early_stopping_rounds.is_some(),
                ebm,
                "`booster=ebm`",
            ),
            (
                "ebm_early_stopping_tolerance",
                self.ebm_early_stopping_tolerance.is_some(),
                self.ebm_early_stopping_rounds
                    .is_some_and(|rounds| rounds.0.is_some()),
                "`ebm_early_stopping_rounds > 0`",
            ),
            (
                "refresh_leaf",
                self.refresh_leaf.is_some(),
                update,
                "`process_type=update`",
            ),
            (
                "extra_seed",
                self.extra_seed.is_some(),
                self.extra_trees == Some(true),
                "`extra_trees=true`",
            ),
            (
                "linear_lambda",
                self.linear_lambda.is_some(),
                self.linear_tree == Some(true),
                "`linear_tree=true`",
            ),
            (
                "num_grad_quant_bins",
                self.num_grad_quant_bins.is_some(),
                quantized,
                "`use_quantized_grad=true`",
            ),
            (
                "stochastic_rounding",
                self.stochastic_rounding.is_some(),
                quantized,
                "`use_quantized_grad=true`",
            ),
            (
                "quant_train_renew_leaf",
                self.quant_train_renew_leaf.is_some(),
                quantized,
                "`use_quantized_grad=true`",
            ),
            (
                "diffusion_temperature",
                self.diffusion_temperature.is_some(),
                self.langevin == Some(true),
                "`langevin=true`",
            ),
            (
                "model_shrink_mode",
                self.model_shrink_mode.is_some(),
                self.model_shrink_rate.is_some_and(|rate| rate.0.is_some()),
                "a `model_shrink_rate` other than 0",
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
        Ok(())
    }

    /// The booster with its option group.
    fn booster_kind(&self) -> Result<BoosterKind> {
        Ok(match self.booster.unwrap_or(FlatBooster::GbTree) {
            FlatBooster::GbTree => BoosterKind::GbTree,
            FlatBooster::GbLinear => BoosterKind::GbLinear,
            FlatBooster::Dart => {
                let mut dart = Dart::builder();
                if let Some(rate_drop) = self.rate_drop {
                    dart = dart.rate_drop(rate_drop);
                }
                if let Some(skip_drop) = self.skip_drop {
                    dart = dart.skip_drop(skip_drop);
                }
                if let Some(one_drop) = self.one_drop {
                    dart = dart.one_drop(one_drop);
                }
                BoosterKind::Dart(dart.build()?)
            }
            FlatBooster::Boulevard => {
                let mut boulevard = Boulevard::builder();
                if let Some(dropout) = self.boulevard_dropout {
                    boulevard = boulevard.dropout(dropout);
                }
                if let Some(truncation) = self.boulevard_truncation.and_then(|t| t.0) {
                    boulevard = boulevard.truncation(truncation);
                }
                BoosterKind::Boulevard(boulevard.build()?)
            }
            FlatBooster::Ebm => {
                let mut ebm = Ebm::builder();
                if let Some(v) = self.ebm_interactions {
                    ebm = ebm.interactions(v);
                }
                if let Some(v) = self.ebm_outer_bags {
                    ebm = ebm.outer_bags(v);
                }
                if let Some(v) = self.ebm_bag_fraction {
                    ebm = ebm.bag_fraction(v);
                }
                if let Some(v) = self.ebm_boulevard {
                    ebm = ebm.boulevard(v);
                }
                if let Some(rounds) = self.ebm_early_stopping_rounds.and_then(|r| r.0) {
                    let tolerance = self
                        .ebm_early_stopping_tolerance
                        .unwrap_or(EbmEarlyStopping::DEFAULT_TOLERANCE);
                    ebm = ebm.early_stopping(EbmEarlyStopping::new(rounds, tolerance)?);
                }
                BoosterKind::Ebm(ebm.build()?)
            }
        })
    }

    /// The option groups of the mode switches (refresh, extra trees,
    /// linear leaves, quantized gradients, balanced and query bagging,
    /// Langevin, model shrinkage).
    fn mode_options(&self) -> Result<ModeOptions> {
        let process_type = match self.process_type.unwrap_or(FlatProcess::Default) {
            FlatProcess::Default => ProcessType::Default,
            FlatProcess::Update => ProcessType::Update(match self.refresh_leaf {
                Some(false) => Refresh::stats_only(),
                Some(true) | None => Refresh::default(),
            }),
        };
        let extra_trees = (self.extra_trees == Some(true)).then(|| {
            self.extra_seed
                .map_or_else(ExtraTrees::default, ExtraTrees::with_seed)
        });
        let linear_tree = if self.linear_tree == Some(true) {
            Some(LinearTree::new(self.linear_lambda.unwrap_or_default())?)
        } else {
            None
        };
        let quantized = if self.use_quantized_grad == Some(true) {
            let mut q = QuantizedGrad::builder();
            if let Some(bins) = self.num_grad_quant_bins {
                q = q.bins(bins);
            }
            if let Some(stochastic) = self.stochastic_rounding {
                q = q.stochastic_rounding(stochastic);
            }
            if let Some(renew) = self.quant_train_renew_leaf {
                q = q.renew_leaf(renew);
            }
            Some(q.build()?)
        } else {
            None
        };
        // LightGBM's default of 1 for both fractions is no balanced bagging.
        let balanced_bagging = match (self.pos_bagging_fraction, self.neg_bagging_fraction) {
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
        let (subsample, bagging_by_query) = if self.bagging_by_query == Some(true) {
            let fraction = self.subsample.unwrap_or(1.0);
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
            (self.subsample, None)
        };
        // CatBoost: posterior sampling needs Langevin "not set or true", and
        // derives the shrink rate (an explicit one, `0` included, is refused
        // rather than overridden).
        let posterior_sampling = self.posterior_sampling == Some(true);
        if posterior_sampling && self.langevin == Some(false) {
            return Err(HessboostError::invalid_param(
                "langevin",
                "`posterior_sampling` requires Langevin boosting; leave `langevin` unset or true",
            ));
        }
        if posterior_sampling && self.model_shrink_rate.is_some() {
            return Err(HessboostError::invalid_param(
                "model_shrink_rate",
                "is derived by `posterior_sampling` (constant, 1 / (2 * rows)); leave it unset",
            ));
        }
        let langevin = if self.langevin == Some(true) {
            let mut l = Langevin::builder();
            if let Some(temperature) = self.diffusion_temperature {
                l = l.diffusion_temperature(temperature);
            }
            Some(l.build()?)
        } else {
            None
        };
        let model_shrink = match self.model_shrink_rate {
            Some(FlatRate(Some(rate))) => Some(ModelShrink::new(
                rate,
                self.model_shrink_mode.unwrap_or_default(),
            )?),
            // CatBoost's default under Langevin (`SetNotSpecifiedOptionsToDefaults`
            // in `catboost_options.cpp`); posterior sampling derives its own.
            None if langevin.is_some() && !posterior_sampling => Some(ModelShrink::new(
                LANGEVIN_SHRINK_RATE,
                ModelShrinkMode::Constant,
            )?),
            // CatBoost's rate `0` is no shrinkage.
            Some(FlatRate(None)) | None => None,
        };
        Ok(ModeOptions {
            process_type,
            extra_trees,
            linear_tree,
            quantized,
            balanced_bagging,
            subsample,
            bagging_by_query,
            langevin,
            model_shrink,
        })
    }
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
    /// CatBoost's `model_shrink_rate = 0` reads as `model_shrink = None`,
    /// and `langevin = true` without a rate (and without posterior
    /// sampling) as CatBoost's default, a constant rate `0.001`;
    /// [`to_xgboost`](Self::to_xgboost) writes a `0` rate for Langevin
    /// without shrinkage.
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
}
