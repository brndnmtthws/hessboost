use super::{BoostedModel, ModelObjective, TreeWeights, rebuild_objective};
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::objective::Objective;

impl BoostedModel {
    /// Check everything prediction and the formats rely on: the output
    /// layout, the objective and its parameters, and the stored values
    /// (in that order; the first failure is reported).
    pub(crate) fn validate_structure(&self) -> Result<()> {
        self.validate_layout()?;
        self.validate_objective()?;
        self.validate_values()
    }

    /// Feature count, outputs, targets, forest size, and tree kinds.
    fn validate_layout(&self) -> Result<()> {
        if self.n_features == 0 {
            return Err(HessboostError::model_format(
                "model has an invalid feature count",
            ));
        }
        let vector = self.has_vector_leaves();
        // Both factors come from the file: check the product before any
        // divisibility or round-count arithmetic relies on it.
        let per_iteration = if vector {
            Some(self.num_parallel_tree)
        } else {
            self.n_outputs.checked_mul(self.num_parallel_tree)
        };
        if self.n_outputs == 0
            || self.n_targets == 0
            || self.num_parallel_tree == 0
            || (self.num_class >= 2 && self.n_outputs != self.num_class)
            || per_iteration.is_none_or(|per| !self.trees.len().is_multiple_of(per))
            || self.trees.iter().any(|tree| {
                tree.is_vector_leaf() != vector
                    || (vector && tree.size_leaf_vector() != self.n_outputs)
            })
        {
            return Err(HessboostError::ModelFormat(format!(
                "invalid output layout: {} outputs, num_class {}, num_parallel_tree {}, {} trees",
                self.n_outputs,
                self.num_class,
                self.num_parallel_tree,
                self.trees.len()
            )));
        }
        Ok(())
    }

    /// The objective's output width and the stored `max_delta_step`
    /// (the objective's own parameters are valid by construction).
    fn validate_objective(&self) -> Result<()> {
        if !(self.max_delta_step.is_finite() && self.max_delta_step >= 0.0) {
            return Err(HessboostError::model_format(format!(
                "invalid objective parameters: max_delta_step {} is not finite and >= 0",
                self.max_delta_step
            )));
        }
        check_objective_width(
            &self.objective,
            self.max_delta_step,
            self.num_class,
            self.n_targets,
            self.n_outputs,
        )
    }

    /// `best_iteration`, intercepts, tree weights, trees, and the linear
    /// booster's parameters.
    fn validate_values(&self) -> Result<()> {
        // Early stopping selects iterations of a tree ensemble; gblinear has
        // none (training refuses early stopping for it), and a stored value
        // would make plain prediction ask it for an iteration range.
        if let Some(best) = self.best_iteration {
            if self.linear.is_some() {
                return Err(HessboostError::ModelFormat(format!(
                    "gblinear models have no boosting iterations, but best_iteration is {best}"
                )));
            }
            if best >= self.num_boost_rounds() {
                return Err(HessboostError::ModelFormat(format!(
                    "best_iteration {best} is out of range for {} iterations",
                    self.num_boost_rounds()
                )));
            }
        }
        if self.base_score.len() != self.n_outputs()
            || self.base_score.iter().any(|v| !v.is_finite())
        {
            return Err(HessboostError::ModelFormat(format!(
                "base_score must hold one finite value per output ({} outputs, got {:?})",
                self.n_outputs(),
                self.base_score
            )));
        }
        if matches!(&self.tree_weights, TreeWeights::Explicit(weights) if weights.len() != self.trees.len())
        {
            return Err(HessboostError::model_format(
                "tree_weights length does not match trees",
            ));
        }
        if self.tree_weights.iter().any(|weight| !weight.is_finite()) {
            return Err(HessboostError::model_format("tree weights must be finite"));
        }
        for (tree_id, tree) in self.trees.iter().enumerate() {
            if !tree.is_valid_for_features(self.n_features) {
                return Err(HessboostError::ModelFormat(format!(
                    "tree {tree_id} contains invalid nodes"
                )));
            }
        }
        if let Some(linear) = &self.linear {
            let outputs = self.n_outputs();
            if linear.bias.len() != outputs
                || Some(linear.weights.len()) != self.n_features.checked_mul(outputs)
            {
                return Err(HessboostError::model_format(
                    "linear model dimensions are invalid",
                ));
            }
            if !linear
                .weights
                .iter()
                .chain(&linear.bias)
                .all(|v| v.is_finite())
            {
                return Err(HessboostError::model_format(
                    "linear model parameters must be finite",
                ));
            }
        }
        if let Some(info) = &self.boulevard {
            info.validate(self)?;
        }
        if let Some(info) = &self.ebm {
            info.validate(self)?;
        }
        self.validate_shrinkage()
    }

    /// A shrinkage record must describe a tree model's iterations and
    /// agree bit for bit with the tree weights and intercepts it derives,
    /// and early stopping of a shrunk model keeps only the best iterations,
    /// so a `best_iteration` can only name the last one.
    fn validate_shrinkage(&self) -> Result<()> {
        let Some(shrinkage) = &self.shrinkage else {
            return Ok(());
        };
        if self.linear.is_some() {
            return Err(HessboostError::model_format(
                "gblinear models cannot carry a shrinkage record",
            ));
        }
        let rounds = self.num_boost_rounds();
        shrinkage.validate(rounds, self.n_outputs)?;
        if self.best_iteration.is_some_and(|best| best + 1 != rounds) {
            return Err(HessboostError::model_format(
                "a shrunk model's best_iteration must be its last iteration",
            ));
        }
        if !shrinkage.matches(
            self.trees_per_iteration(),
            |t| self.tree_weight(t),
            &self.base_score,
        ) {
            return Err(HessboostError::model_format(
                "the tree weights and intercepts do not match the shrinkage record",
            ));
        }
        Ok(())
    }
}

/// Check that `data` fits a model with `n_features` inputs and `n_outputs`
/// outputs: matching column count and a per-row or per-row-and-output
/// `base_margin`.
pub(crate) fn validate_prediction_data(
    n_features: usize,
    n_outputs: usize,
    data: &DMatrix,
) -> Result<()> {
    if data.n_cols() != n_features {
        return Err(HessboostError::dimension_mismatch(
            "prediction feature count",
            n_features,
            data.n_cols(),
        ));
    }
    let n = data.n_rows();
    if let Some(margin) = data.base_margin()
        && margin.len() != n
        && margin.len() != n * n_outputs
    {
        return Err(HessboostError::dimension_mismatch(
            "prediction base_margin length",
            n * n_outputs,
            margin.len(),
        ));
    }
    Ok(())
}

/// Check that a built-in objective produces the model's `n_outputs`
/// outputs. Loaders call this before returning a model: the prediction
/// transform of a multi-output objective works on `[row][output]` blocks of
/// its own width, so a mismatched width would transform values of
/// neighboring rows together. Other objectives (custom losses) are skipped:
/// they predict margins. A built-in objective that cannot be rebuilt for the
/// stored layout (e.g. a distribution with a label matrix) is a format
/// error, since predicting without its transform would misreport every
/// output. So is a `num_class >= 2` on a built-in objective other than
/// `multi:softmax`/`multi:softprob`: XGBoost reads `num_class` as the
/// multiclass class count and refuses it together with several targets.
pub(crate) fn check_objective_width(
    objective: &ModelObjective,
    max_delta_step: f64,
    num_class: usize,
    n_targets: usize,
    n_outputs: usize,
) -> Result<()> {
    let name = objective.name();
    let multiclass = objective
        .built_in()
        .and_then(Objective::num_class)
        .is_some();
    match rebuild_objective(objective, max_delta_step, n_targets) {
        None => Ok(()),
        Some(Ok(rebuilt)) if rebuilt.n_outputs() == n_outputs => {
            if num_class >= 2 && !multiclass {
                Err(HessboostError::ModelFormat(format!(
                    "num_class {num_class} applies only to multiclass objectives, not `{name}`"
                )))
            } else {
                Ok(())
            }
        }
        Some(Ok(rebuilt)) => Err(HessboostError::ModelFormat(format!(
            "objective `{name}` has {} outputs but the model stores {n_outputs}",
            rebuilt.n_outputs()
        ))),
        Some(Err(e)) => Err(HessboostError::ModelFormat(format!(
            "objective `{name}` cannot be rebuilt from the stored configuration: {e}"
        ))),
    }
}
