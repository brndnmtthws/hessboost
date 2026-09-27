//! Writing the flat form: [`TrainingParams::to_xgboost`] and the key
//! comparison [`TrainingParams::changed_keys`].

use super::super::groups::BalancedBagging;
use super::super::params::{BoosterKind, MaxDeltaStep, Monotone, ProcessType, TrainingParams};
use crate::error::{HessboostError, Result};
use crate::metric::EvalMetric;
use crate::objective::{OBJECTIVE_PARAMS, Objective};
use serde::Serialize;
use serde_json::{Map, Value};
use std::num::NonZeroUsize;

/// The flat keys and values of `objective`'s own parameters (a default
/// split direction is left out).
fn objective_keys(objective: &Objective) -> Vec<(&'static str, Value)> {
    let parts = objective.parts();
    OBJECTIVE_PARAMS
        .iter()
        .filter(|param| (param.read_by)(objective))
        .filter_map(|param| Some((param.key, (param.value)(&parts)?)))
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

/// `value` as JSON; every setting serializes (non-finite numbers as `null`).
fn json(value: impl Serialize) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

impl TrainingParams {
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
                set(
                    "boulevard_truncation",
                    json(boulevard.truncation().unwrap_or(0.0)),
                );
            }
            BoosterKind::Ebm(ebm) => {
                set("booster", json("ebm"));
                set("ebm_interactions", json(ebm.interactions()));
                set("ebm_outer_bags", json(ebm.outer_bags()));
                set("ebm_bag_fraction", json(ebm.bag_fraction()));
                set("ebm_boulevard", json(ebm.boulevard()));
                // `0` rounds is no early stopping; the tolerance needs it.
                let stopping = ebm.early_stopping();
                set(
                    "ebm_early_stopping_rounds",
                    json(stopping.map_or(0, |s| s.rounds().get())),
                );
                if let Some(stopping) = stopping {
                    set("ebm_early_stopping_tolerance", json(stopping.tolerance()));
                }
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
        match model_shrink {
            Some(shrink) => {
                set("model_shrink_rate", json(shrink.rate()));
                set("model_shrink_mode", json(shrink.mode()));
            }
            // The flat `langevin=true` alone shrinks at CatBoost's default
            // rate; `0` keeps it off.
            None if langevin.is_some() && !posterior_sampling => {
                set("model_shrink_rate", json(0.0));
            }
            None => {}
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
