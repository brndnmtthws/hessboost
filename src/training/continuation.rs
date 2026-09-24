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
use crate::model::BoostedModel;
use crate::objective::Objective;
use crate::training::multi_output;

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
    // The refresh updater sums the full-precision gradients directly; it
    // never builds the quantized histograms `use_quantized_grad` asks for.
    if params.use_quantized_grad {
        return Err(HessboostError::invalid_param(
            "process_type",
            "`update` refreshes from full-precision gradients and does not support \
             `use_quantized_grad`",
        ));
    }
    // Refresh keeps every split and never searches for one, so options that
    // only change the split search would silently have no effect.
    if params.extra_trees
        || params.toad_penalty_feature > 0.0
        || params.toad_penalty_threshold > 0.0
    {
        return Err(HessboostError::invalid_param(
            "process_type",
            "`update` keeps the existing splits and does not support the split-search \
             options `extra_trees`, `toad_penalty_feature`, or `toad_penalty_threshold`",
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
            "`update` refreshes an existing model; use Trainer::init_model",
        ));
    }
    Ok(())
}
