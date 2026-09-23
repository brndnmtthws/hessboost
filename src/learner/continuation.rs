//! Continued training from an existing model (XGBoost's `xgb_model=`) and the
//! checks for `process_type=update`.
//!
//! A continued model keeps its trees, intercepts, objective and output
//! layout; training starts from its full current margins and appends new
//! iterations (or, with `process_type=update`, refreshes the existing ones).
//! As in XGBoost, an explicit `base_score` in the new parameters replaces the
//! stored intercept, while an absent one keeps it: the intercept is never
//! re-estimated from the new labels.

use crate::config::{BoosterKind, Monotone, ObjectiveParams, ProcessType, TrainingParams};
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::learner::model::BoostedModel;
use crate::objective::Objective;

/// Check that `init` can be trained further with `params` on `dtrain` and
/// return the model training continues from: a copy of `init` without its
/// early-stopping selection, carrying the new objective parameters, an
/// explicit `base_score` from `params` when set (via `intercepts`), and one
/// explicit weight per tree so appended trees line up with DART weights.
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
        return Err(HessboostError::DimensionMismatch {
            what: "continued-training feature count",
            expected: init.n_features(),
            got: dtrain.n_cols(),
        });
    }
    if dtrain.n_targets() != init.n_targets() {
        return Err(HessboostError::DimensionMismatch {
            what: "continued-training label targets",
            expected: init.n_targets(),
            got: dtrain.n_targets(),
        });
    }
    // Iterations must stay uniform: every layer holds the same forest size.
    if !is_linear && init.num_trees() > 0 && params.num_parallel_tree != init.num_parallel_tree() {
        return Err(HessboostError::invalid_param(
            "num_parallel_tree",
            format!(
                "{} does not match the model's num_parallel_tree {}",
                params.num_parallel_tree,
                init.num_parallel_tree()
            ),
        ));
    }
    if params.process_type == ProcessType::Update {
        check_update(init, params, num_boost_round)?;
    }

    let mut model = init.clone();
    model.set_best_iteration(None);
    model.set_objective_params(ObjectiveParams::from_params(params));
    model.set_num_parallel_tree(params.num_parallel_tree);
    model.materialize_tree_weights();
    if params.base_score.is_some() {
        model.set_base_scores(intercepts()?);
    }
    Ok(model)
}

/// `process_type=update` refreshes existing gbtree trees one iteration per
/// round (XGBoost `GBTree::InitNewTrees` in update mode).
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
    // sums; linear leaf models and path-smoothed outputs would silently go
    // stale or be dropped.
    if params.linear_tree
        || params.path_smooth > 0.0
        || init
            .trees()
            .iter()
            .any(|tree| tree.linear_leaves().is_some())
    {
        return Err(HessboostError::invalid_param(
            "process_type",
            "`update` cannot refresh linear-leaf trees or path-smoothed leaves \
             (`linear_tree` / `path_smooth`)",
        ));
    }
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

/// Reject `process_type=update` without a model to update.
pub(super) fn require_model_for_update(params: &TrainingParams) -> Result<()> {
    if params.process_type == ProcessType::Update {
        return Err(HessboostError::invalid_param(
            "process_type",
            "`update` refreshes an existing model; use train_continue",
        ));
    }
    Ok(())
}
