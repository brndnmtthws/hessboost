//! Request validation: what training refuses before anything is built.

use super::eval::{EvalSet, name_dataset};
use crate::config::{BoosterKind, ProcessType, TrainingParams};
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::objective::Loss;
use crate::training::multi_output;
use std::num::NonZeroUsize;

/// Check that a freshly trained model would load again. Arithmetic that
/// overflows `f32` (from extreme labels, weights, or margins) leaves
/// non-finite values the model formats refuse; report it when training
/// returns, not at load time.
pub(super) fn validate_trained_model(model: &BoostedModel) -> Result<()> {
    model.validate_structure().map_err(|e| {
        let reason = match e {
            HessboostError::ModelFormat(reason) => reason,
            other => other.to_string(),
        };
        HessboostError::model_format(format!("training produced an invalid model: {reason}"))
    })
}

/// Refuse `dtrain`'s feature weights on a training path that samples no
/// columns (`reason` names it): they only steer the tree builders' column
/// sampling.
pub(super) fn reject_feature_weights(dtrain: &DMatrix, reason: &'static str) -> Result<()> {
    if dtrain.feature_weights().is_some() {
        return Err(HessboostError::invalid_param("feature_weights", reason));
    }
    Ok(())
}

/// The checks [`Trainer::train`](super::Trainer::train) runs on `params` and `dtrain` before
/// anything is built (no eval sets, no early stopping), with the loss
/// `params` train: for callers that train on `dtrain` without going through
/// [`Trainer`](super::Trainer) and must refuse exactly what training refuses.
pub(super) fn validate_training_data(params: &TrainingParams, dtrain: &DMatrix) -> Result<()> {
    let objective = params.loss(dtrain.n_targets())?;
    validate_request(
        &TrainRequest {
            params,
            dtrain,
            evals: &[],
            early_stopping_rounds: None,
        },
        objective.as_ref(),
    )
}

/// What [`validate_request`] checks: the training call's configuration and
/// data, before anything is built.
pub(super) struct TrainRequest<'a> {
    pub(super) params: &'a TrainingParams,
    pub(super) dtrain: &'a DMatrix,
    pub(super) evals: &'a [EvalSet<'a>],
    pub(super) early_stopping_rounds: Option<NonZeroUsize>,
}
/// Refuse a training call that cannot run: invalid parameters, early
/// stopping without eval sets, unlabeled or mismatched data, constraints
/// naming missing features, feature weights on a path that samples no
/// columns, and objective settings a saved model could not rebuild. The
/// checks run, and report their first refusal, in this order.
pub(super) fn validate_request(request: &TrainRequest, objective: &dyn Loss) -> Result<()> {
    let &TrainRequest {
        params,
        dtrain,
        evals,
        ..
    } = request;
    validate_setup(request, objective)?;
    validate_query_bagging(params, dtrain)?;
    if objective.requires_labels() && dtrain.labels().is_none() {
        return Err(HessboostError::EmptyDataset("train: dtrain has no labels"));
    }
    validate_balanced_bagging(params, dtrain)?;
    validate_datasets(objective, dtrain, evals)?;
    validate_constraints(params, dtrain.n_cols())?;
    validate_booster(request, objective)?;
    validate_shrinkage_margins(request)?;
    BoostedModel::check_iteration_size(objective.n_outputs(), params.num_parallel_tree)
}

/// The parameters themselves, the multi-output strategy for `objective`'s
/// outputs, and the early-stopping patience and its eval sets.
fn validate_setup(request: &TrainRequest, objective: &dyn Loss) -> Result<()> {
    let &TrainRequest {
        params,
        evals,
        early_stopping_rounds,
        ..
    } = request;
    params.validate()?;
    multi_output::validate(params, objective.n_outputs())?;
    if early_stopping_rounds.is_some() && evals.is_empty() {
        return Err(HessboostError::invalid_param(
            "early_stopping_rounds",
            "requires at least one evaluation dataset",
        ));
    }
    Ok(())
}

/// Query bagging needs non-empty query groups covering `dtrain`.
fn validate_query_bagging(params: &TrainingParams, dtrain: &DMatrix) -> Result<()> {
    if params.bagging_by_query.is_none() {
        return Ok(());
    }
    let Some(group) = dtrain.group() else {
        return Err(HessboostError::invalid_param(
            "bagging_by_query",
            "requires query group sizes on the training dataset",
        ));
    };
    if !group.partitions(dtrain.n_rows()) || group.iter_ranges().any(|(start, end)| start == end) {
        return Err(HessboostError::invalid_param(
            "bagging_by_query",
            "requires non-empty query groups covering all training rows",
        ));
    }
    Ok(())
}

/// Class-balanced bagging needs one column of labels exactly 0 or 1.
fn validate_balanced_bagging(params: &TrainingParams, dtrain: &DMatrix) -> Result<()> {
    if params.balanced_bagging.is_none() {
        return Ok(());
    }
    if dtrain.n_targets() != 1 {
        return Err(HessboostError::invalid_param(
            "labels",
            "balanced bagging requires exactly one label column",
        ));
    }
    let labels = dtrain.labels().ok_or(HessboostError::EmptyDataset(
        "train: balanced bagging requires binary labels",
    ))?;
    if labels.iter().any(|&label| label != 0.0 && label != 1.0) {
        return Err(HessboostError::invalid_param(
            "labels",
            "balanced bagging requires labels exactly 0 or 1",
        ));
    }
    Ok(())
}

/// The monotone and interaction constraints name only features of the
/// training matrix's `n_features`.
fn validate_constraints(params: &TrainingParams, n_features: usize) -> Result<()> {
    if params.monotone_constraints.len() > n_features {
        return Err(HessboostError::invalid_param(
            "monotone_constraints",
            "contains more entries than the training matrix has features",
        ));
    }
    for group in &params.interaction_constraints {
        if group.is_empty() {
            return Err(HessboostError::invalid_param(
                "interaction_constraints",
                "constraint groups cannot be empty",
            ));
        }
        if let Some(&feature) = group.iter().find(|&&f| f as usize >= n_features) {
            return Err(HessboostError::FeatureOutOfBounds {
                index: feature as usize,
                num_features: n_features,
            });
        }
    }
    Ok(())
}

/// The booster- and process-specific refusals: feature weights where no
/// columns are sampled, then Boulevard's and the EBM's data requirements.
fn validate_booster(request: &TrainRequest, objective: &dyn Loss) -> Result<()> {
    let &TrainRequest { params, dtrain, .. } = request;
    if params.booster == BoosterKind::GbLinear {
        reject_feature_weights(dtrain, "gblinear does not sample columns")?;
    }
    if matches!(params.process_type, ProcessType::Update(_)) {
        reject_feature_weights(dtrain, "`process_type=update` does not sample columns")?;
    }
    if matches!(params.booster, BoosterKind::Boulevard(_)) {
        validate_boulevard_request(request, objective, "booster = boulevard")?;
    }
    if matches!(params.booster, BoosterKind::Ebm(_)) {
        validate_ebm_request(request, objective)?;
    }
    Ok(())
}

/// Model shrinkage multiplies the intercept-and-trees margin every
/// iteration; a per-row `base_margin` replaces the intercept and would be
/// shrunk with it in the caches but not in prediction (CatBoost refuses
/// baselines with shrinkage too, `options_helper.cpp`).
fn validate_shrinkage_margins(request: &TrainRequest) -> Result<()> {
    let &TrainRequest {
        params,
        dtrain,
        evals,
        ..
    } = request;
    if params.model_shrinkage_on()
        && let Some(name) = std::iter::once(EvalSet {
            data: dtrain,
            name: "dtrain",
        })
        .chain(evals.iter().copied())
        .find_map(|set| set.data.base_margin().map(|_| set.name))
    {
        return Err(HessboostError::invalid_param(
            "model_shrink_rate",
            format!("model shrinkage is not supported with a `base_margin` (dataset `{name}`)"),
        ));
    }
    Ok(())
}

/// The data-dependent refusals of `booster = boulevard` (and, as `who`
/// names, of the Boulevard EBM): its inference ([`crate::inference`])
/// models one squared-error label column with equal noise per row, around
/// the intercept alone. Early stopping is refused too: the prediction
/// averages every round, so a `best_iteration` prefix of the trees is not a
/// Boulevard estimate.
fn validate_boulevard_request(
    request: &TrainRequest,
    objective: &dyn Loss,
    who: &str,
) -> Result<()> {
    let refuse = |name: &'static str, reason: &str| {
        Err(HessboostError::invalid_param(
            name,
            format!("`{who}`: {reason}"),
        ))
    };
    if request.early_stopping_rounds.is_some() {
        return refuse(
            "early_stopping_rounds",
            "the model averages every round, so it cannot stop at a best iteration",
        );
    }
    if objective.name() != "reg:squarederror" {
        return refuse(
            "objective",
            &format!("supports reg:squarederror only, got `{}`", objective.name()),
        );
    }
    let dtrain = request.dtrain;
    if dtrain.n_targets() != 1 || objective.n_outputs() != 1 {
        return refuse(
            "labels",
            &format!("needs one label column, got {}", dtrain.n_targets()),
        );
    }
    if dtrain
        .weights()
        .is_some_and(|w| w.iter().any(|&v| v != 1.0))
    {
        return refuse("weights", "row weights other than 1 are not supported");
    }
    for data in std::iter::once(dtrain).chain(request.evals.iter().map(|set| set.data)) {
        if data.base_margin().is_some() {
            return refuse("base_margin", "base margins are not supported");
        }
    }
    Ok(())
}

/// The data-dependent refusals of `booster = ebm`: one output, numerical
/// features, no feature weights or base margins (the terms and their
/// centering assume the intercept alone), and no eval sets or early stopping (the
/// terms of one run are boosted round by round, so no prefix of the trees
/// is a model of every term); with `ebm_boulevard` also Boulevard's
/// refusals (squared error, unit row weights, no base margins).
fn validate_ebm_request(request: &TrainRequest, objective: &dyn Loss) -> Result<()> {
    let refuse = |name: &'static str, reason: &str| {
        Err(HessboostError::invalid_param(
            name,
            format!("`booster = ebm`: {reason}"),
        ))
    };
    if request.early_stopping_rounds.is_some() || !request.evals.is_empty() {
        return refuse(
            "early_stopping_rounds",
            "eval sets and early stopping are not supported; evaluate the trained model",
        );
    }
    if request.dtrain.n_targets() != 1 {
        return refuse(
            "labels",
            &format!("needs one label column, got {}", request.dtrain.n_targets()),
        );
    }
    if objective.n_outputs() != 1 {
        return refuse(
            "objective",
            &format!(
                "needs a single-output objective, got `{}`",
                objective.name()
            ),
        );
    }
    if request.dtrain.base_margin().is_some() {
        return refuse(
            "base_margin",
            "base margins are not supported: the terms and their centering assume the \
             intercept alone",
        );
    }
    if request.params.ebm_settings().early_stopping().is_some() && request.dtrain.group().is_some()
    {
        return refuse(
            "ebm_early_stopping_rounds",
            "early stopping scores each bag's held-out rows, which split the query groups; not \
             supported with query groups",
        );
    }
    super::ebm::validate_data(request.dtrain)?;
    if request.params.ebm_settings().boulevard() {
        validate_boulevard_request(request, objective, "ebm_boulevard")?;
    }
    Ok(())
}
/// What every dataset of a training run must match: the training matrix's
/// label columns and feature count, and the objective's output count (a
/// `base_margin`'s width).
#[derive(Clone, Copy)]
struct DatasetContract {
    targets: usize,
    features: usize,
    outputs: usize,
}

impl DatasetContract {
    /// The contract `dtrain` sets for training with `objective`.
    fn of(dtrain: &DMatrix, objective: &dyn Loss) -> Self {
        DatasetContract {
            targets: dtrain.n_targets(),
            features: dtrain.n_cols(),
            outputs: objective.n_outputs(),
        }
    }
}

/// Check `dtrain` and then every eval set, in order, against the contract
/// `dtrain` sets ([`validate_dataset`]).
pub(super) fn validate_datasets(
    objective: &dyn Loss,
    dtrain: &DMatrix,
    evals: &[EvalSet],
) -> Result<()> {
    let contract = DatasetContract::of(dtrain, objective);
    validate_dataset(objective, dtrain, contract, "dtrain")?;
    for set in evals {
        validate_dataset(objective, set.data, contract, set.name)?;
    }
    Ok(())
}

/// Shape and metadata checks for one dataset (named `name` in errors)
/// against `contract`, followed by the objective's own [`validate_info`]
/// label-domain checks.
///
/// [`validate_info`]: crate::objective::Loss::validate_info
fn validate_dataset(
    objective: &dyn Loss,
    data: &DMatrix,
    contract: DatasetContract,
    name: &str,
) -> Result<()> {
    let DatasetContract {
        targets: n_targets,
        features: n_features,
        outputs: n_out,
    } = contract;
    match data.labels() {
        None if objective.requires_labels() => {
            return Err(HessboostError::invalid_param(
                "evals",
                format!("dataset `{name}` has no labels"),
            ));
        }
        None => {}
        Some(labels) => {
            let expected = data.n_rows().checked_mul(n_targets).ok_or_else(|| {
                HessboostError::invalid_param("labels", "expected length overflows usize")
            })?;
            if labels.len() != expected {
                return Err(HessboostError::dimension_mismatch(
                    "labels length (n_rows * training n_targets)",
                    expected,
                    labels.len(),
                ));
            }
        }
    }
    if data.n_cols() != n_features {
        return Err(HessboostError::dimension_mismatch(
            "dataset feature count",
            n_features,
            data.n_cols(),
        ));
    }
    if let Some(margin) = data.base_margin() {
        let expected = data.n_rows().checked_mul(n_out).ok_or_else(|| {
            HessboostError::invalid_param("base_margin", "expected length overflows usize")
        })?;
        if margin.len() != data.n_rows() && margin.len() != expected {
            return Err(HessboostError::dimension_mismatch(
                "base_margin length",
                expected,
                margin.len(),
            ));
        }
    }
    objective
        .validate_info(&data.info())
        .map_err(|error| name_dataset(error, name))
}
