//! Continued training from an existing model (XGBoost's `xgb_model=`) and the
//! checks for `process_type=update`.
//!
//! A continued model keeps its trees, intercepts, objective and output
//! layout; training starts from its full current margins and appends new
//! iterations (or, with `process_type=update`, refreshes the existing ones).
//! As in XGBoost, an explicit `base_score` in the new parameters replaces the
//! stored intercept, while an absent one keeps it: the intercept is never
//! re-estimated from the new labels.

use crate::config::{
    BoosterKind, GrowPolicy, Monotone, ObjectiveParams, ProcessType, TrainingParams,
};
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::objective::Objective;
use crate::training::multi_output;

/// Check that `init` can be trained further with `params` on `dtrain` and
/// return the model training continues from: a copy of `init` without its
/// early-stopping selection, carrying the new objective parameters, an
/// explicit `base_score` from `params` when set (via `intercepts`), and one
/// explicit weight per tree so appended trees line up with DART weights.
/// `init` must be structurally valid (as every loaded model is), whichever
/// way it was built, before its trees and intercepts are read.
///
/// `best_iteration` is cleared because hessboost predicts with it by
/// default: a stale selection from the earlier run would hide the new
/// iterations. Early stopping during the continued run sets it again, as an
/// absolute iteration index.
pub(super) fn resume_model(
    init: &BoostedModel,
    params: &TrainingParams,
    objective: &dyn Objective,
    dtrain: &DMatrix,
    num_boost_round: usize,
    intercepts: impl FnOnce() -> Result<Vec<f32>>,
) -> Result<BoostedModel> {
    init.validate_structure().map_err(|e| {
        let reason = match e {
            HessboostError::ModelFormat(reason) => reason,
            other => other.to_string(),
        };
        HessboostError::model_format(format!("invalid init model: {reason}"))
    })?;
    // A Boulevard model averages all of its rounds (its leaves carry the
    // `1/B` of the run), so appending or refreshing rounds, with any
    // booster, would not give a Boulevard average; nor can Boulevard
    // continue another model's sum.
    if params.booster == BoosterKind::Boulevard || init.boulevard().is_some() {
        return Err(HessboostError::invalid_param(
            "init_model",
            "Boulevard models average every round of one run and cannot be trained further, \
             and `booster = boulevard` cannot continue another model",
        ));
    }
    let is_linear = init.linear().is_some();
    if is_linear != (params.booster == BoosterKind::GbLinear) {
        return Err(HessboostError::invalid_param(
            "booster",
            if is_linear {
                "a gblinear model can only be trained further with booster=gblinear"
            } else {
                "a tree model cannot be trained further with booster=gblinear"
            },
        ));
    }
    if objective.name() != init.objective() {
        return Err(HessboostError::invalid_param(
            "objective",
            format!(
                "`{}` does not match the model's objective `{}`",
                objective.name(),
                init.objective()
            ),
        ));
    }
    if params.num_class != init.num_class() || objective.n_outputs() != init.n_outputs() {
        return Err(HessboostError::invalid_param(
            "num_class",
            format!(
                "{} (with {} outputs) does not match the model's num_class {} ({} outputs)",
                params.num_class,
                objective.n_outputs(),
                init.num_class(),
                init.n_outputs()
            ),
        ));
    }
    if dtrain.n_cols() != init.n_features() {
        return Err(HessboostError::dimension_mismatch(
            "continued-training feature count",
            init.n_features(),
            dtrain.n_cols(),
        ));
    }
    if dtrain.n_targets() != init.n_targets() {
        return Err(HessboostError::dimension_mismatch(
            "continued-training label targets",
            init.n_targets(),
            dtrain.n_targets(),
        ));
    }
    if !is_linear && init.num_trees() > 0 {
        // Iterations must stay uniform: every layer holds the same forest size.
        if params.num_parallel_tree != init.num_parallel_tree() {
            return Err(HessboostError::invalid_param(
                "num_parallel_tree",
                format!(
                    "{} does not match the model's num_parallel_tree {}",
                    params.num_parallel_tree,
                    init.num_parallel_tree()
                ),
            ));
        }
        // A model's trees are either all vector-leaf or all scalar-leaf.
        if init.has_vector_leaves() != multi_output::vector_leaf(params, objective.n_outputs()) {
            return Err(HessboostError::invalid_param(
                "multi_strategy",
                if init.has_vector_leaves() {
                    "a vector-leaf model can only be trained further with \
                     `multi_strategy=multi_output_tree`"
                } else {
                    "a one-output-per-tree model cannot be trained further with \
                     `multi_strategy=multi_output_tree`"
                },
            ));
        }
    }
    if params.process_type == ProcessType::Update {
        check_update(init, params, num_boost_round)?;
    }

    let mut model = init.clone();
    model.set_best_iteration(None);
    model.set_objective_params(ObjectiveParams::for_objective(params, init.objective()));
    model.set_num_parallel_tree(params.num_parallel_tree);
    model.materialize_tree_weights();
    if params.base_score.is_some() {
        model.set_base_scores(intercepts()?);
    }
    Ok(model)
}

/// `process_type=update` refreshes existing gbtree trees one iteration per
/// round (XGBoost `GBTree::InitNewTrees` in update mode). Beyond the
/// specific refusals, every setting refresh does not read must keep its
/// default ([`reject_unused_by_refresh`]).
fn check_update(
    init: &BoostedModel,
    params: &TrainingParams,
    num_boost_round: usize,
) -> Result<()> {
    if params.booster != BoosterKind::GbTree {
        return Err(HessboostError::invalid_param(
            "process_type",
            "`update` refreshes gbtree models only (booster=gbtree)",
        ));
    }
    // XGBoost's refresh updater handles single-target trees only.
    if init.has_vector_leaves() {
        return Err(HessboostError::invalid_param(
            "process_type",
            "`update` cannot refresh vector-leaf trees (`multi_output_tree`)",
        ));
    }
    if init.has_non_unit_tree_weights() {
        return Err(HessboostError::invalid_param(
            "process_type",
            "`update` cannot refresh a model with DART tree weights",
        ));
    }
    if params
        .monotone_constraints
        .iter()
        .any(|&m| m != Monotone::None)
    {
        return Err(HessboostError::invalid_param(
            "monotone_constraints",
            "are not supported by the refresh updater (`process_type=update`)",
        ));
    }
    // The refresh updater recomputes constant leaf weights from the gradient
    // sums: a linear-leaf model would silently go stale.
    if init
        .trees()
        .iter()
        .any(|tree| tree.linear_leaves().is_some())
    {
        return Err(HessboostError::invalid_param(
            "process_type",
            "`update` cannot refresh linear-leaf trees (`linear_tree`)",
        ));
    }
    reject_unused_by_refresh(params)?;
    if num_boost_round > init.num_boost_rounds() {
        return Err(HessboostError::invalid_param(
            "num_boost_round",
            format!(
                "{num_boost_round} exceeds the {} iterations `process_type=update` can refresh",
                init.num_boost_rounds()
            ),
        ));
    }
    Ok(())
}

/// Refuse every [`TrainingParams`] field `process_type=update` does not read,
/// comparing the serialized configuration against the defaults plus an
/// allow-list (as budget mode does), so newly added fields are covered too.
///
/// Refresh keeps every split, sums the gradients of every row over the
/// existing trees, and draws nothing at random. Allowed to differ from the
/// default are:
///
/// * what refresh or the objective's gradients read: the booster and
///   device (checked elsewhere), `nthread`, `seed`, the objective and its
///   parameters, `eval_metric`, `eta`, `lambda`, `alpha`, `max_delta_step`,
///   `num_parallel_tree`, `multi_strategy`, `monotone_constraints` (refused
///   with their own message), `process_type`, `refresh_leaf`, `missing`;
/// * XGBoost's tree-shape settings, which describe how the refreshed trees
///   were grown and which XGBoost 3.4.2's refresh updater accepts with a
///   training run's parameters: `tree_method`, `max_depth`, `max_leaves`,
///   `min_child_weight`, `gamma`, `max_bin`, `interaction_constraints`, and
///   a `depthwise` or `lossguide` `grow_policy`.
///
/// Refused are row and column sampling (`subsample`, `sampling_method`,
/// `colsample_*`), DART's `rate_drop`/`skip_drop`, symmetric growth, and the
/// beyond-XGBoost split-search and leaf options (`extra_trees`,
/// `path_smooth`, `linear_tree`, quantized gradients, reuse penalties): they
/// would silently have no effect.
fn reject_unused_by_refresh(params: &TrainingParams) -> Result<()> {
    let p = params.clone();
    let reference = TrainingParams {
        booster: p.booster,
        nthread: p.nthread,
        seed: p.seed,
        device: p.device,
        objective: p.objective,
        num_class: p.num_class,
        base_score: p.base_score,
        eval_metric: p.eval_metric,
        tweedie_variance_power: p.tweedie_variance_power,
        huber_slope: p.huber_slope,
        lambdarank_num_pair_per_sample: p.lambdarank_num_pair_per_sample,
        quantile_alpha: p.quantile_alpha,
        expectile_alpha: p.expectile_alpha,
        aft_loss_distribution: p.aft_loss_distribution,
        aft_loss_distribution_scale: p.aft_loss_distribution_scale,
        dist_gradient: p.dist_gradient,
        dist_split_direction: p.dist_split_direction,
        scale_pos_weight: p.scale_pos_weight,
        eta: p.eta,
        lambda: p.lambda,
        alpha: p.alpha,
        max_delta_step: p.max_delta_step,
        num_parallel_tree: p.num_parallel_tree,
        multi_strategy: p.multi_strategy,
        monotone_constraints: p.monotone_constraints,
        process_type: p.process_type,
        refresh_leaf: p.refresh_leaf,
        missing: p.missing,
        tree_method: p.tree_method,
        max_depth: p.max_depth,
        max_leaves: p.max_leaves,
        min_child_weight: p.min_child_weight,
        gamma: p.gamma,
        max_bin: p.max_bin,
        interaction_constraints: p.interaction_constraints,
        grow_policy: match p.grow_policy {
            GrowPolicy::Symmetric => GrowPolicy::default(),
            policy => policy,
        },
        ..TrainingParams::default()
    };
    params.refuse_changes_from(
        &reference,
        "process_type",
        "`update` keeps the existing splits and refreshes them from every row, so it applies \
         no sampling, growth, or split-search options",
    )
}

/// Reject `process_type=update` without a model to update.
pub(super) fn require_model_for_update(params: &TrainingParams) -> Result<()> {
    if params.process_type == ProcessType::Update {
        return Err(HessboostError::invalid_param(
            "process_type",
            "`update` refreshes an existing model; use Trainer::init_model",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::config::TrainingParams;
    use crate::error::HessboostError;
    use crate::test_support::labeled_dense;
    use crate::training::{Trainer, train};

    /// A model built in memory is checked like a loaded one before training
    /// reads its intercepts and trees: one without an intercept per output
    /// is refused instead of indexing past its intercepts.
    #[test]
    fn invalid_init_models_are_refused() {
        let x: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let d = labeled_dense(&x, 20, 1, &x);
        let params = TrainingParams::default();
        let mut model = train(&params, &d, 2).unwrap();
        model.set_base_scores(Vec::new());
        assert!(matches!(
            Trainer::new(&params, &d, 1).init_model(&model).train(),
            Err(HessboostError::ModelFormat(_))
        ));
    }
}
