//! [`BoostedModel::average`]: one model predicting its members' mean margin.

use super::{BoostedModel, TreeLayouts, TreeWeights};
use crate::error::{HessboostError, Result};

/// Members [`BoostedModel::average`] takes at most: every count below `2^24`
/// is a 24-bit float, which makes each leaf quotient correctly rounded.
const MAX_MEMBERS: usize = 1 << 24;

impl BoostedModel {
    /// One model whose margin is the mean of `models`' margins (a bagged
    /// ensemble of fits that differ only in their seed or rows, as one
    /// model): every member's trees, in member order, with every leaf
    /// divided by the member count `k`, and every output's intercept the
    /// members' mean intercept. Predictions apply the shared objective's
    /// transform to that mean margin: for an identity link that is the mean
    /// prediction, for a log link (`count:poisson`, `reg:gamma`,
    /// `reg:tweedie`, `survival:cox`) the geometric mean of the members'
    /// predictions, for `binary:logistic` the sigmoid of the mean logit, for
    /// `multi:softprob` the softmax of the mean logits, and for
    /// `reg:quantileerror` the sorted mean margins.
    ///
    /// Arithmetic, which fixes the result bit for bit: each scalar or vector
    /// leaf `v` becomes the `f32` nearest `v / k` (computed as `f64` division
    /// rounded to `f32`, which is correctly rounded for `k < 2^24`); each
    /// `linear_tree` leaf model's `f64` intercept and slopes are divided by
    /// `k` in `f64`; each output's intercept is the members' intercepts
    /// summed in `f64` in member order, divided by `k` in `f64`, and rounded
    /// to `f32` (so members with equal intercepts keep it exactly). The
    /// average's margin is the members' mean margin up to that rounding and
    /// to its `f32` sum over the concatenated trees. Every member iteration
    /// stays an iteration (the average has the members' iteration counts
    /// summed), covers (`sum_hess`) and split gains are kept, and the result
    /// has no `best_iteration`.
    ///
    /// Members must agree on everything their margins depend on besides
    /// the trees and intercepts: the objective with its parameters (and the
    /// stored `max_delta_step`), the feature count, the output, target, and
    /// class counts, `num_parallel_tree`, and scalar or vector leaves (a
    /// member without trees fits either).
    /// Feature names and categories are not part of a [`BoostedModel`]; the
    /// caller checks them (Python's `Booster.average` does).
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] (`models`) for no models or
    /// `2^24` or more; [`HessboostError::IncompatibleModel`] (`average`) for
    /// a member that differs from the first in any of the above, or that
    /// is not a plain tree ensemble with unit tree weights: a `gblinear`
    /// model, DART (tree weights other than `1`), model shrinkage, a
    /// Boulevard fit or EBM (whose statistical records would not describe
    /// the average), or an early-stopped model (slice it to the iterations
    /// to average first, e.g. [`slice(..=best, 1)`](Self::slice)).
    ///
    /// ```
    /// use hessboost::prelude::*;
    ///
    /// # fn main() -> Result<()> {
    /// let x: Vec<f32> = (0..200).map(|i| (i % 50) as f32).collect();
    /// let y: Vec<f32> = x.iter().map(|v| (v / 7.0).sin()).collect();
    /// let data = DMatrix::from_dense(&x, 200, 1)?.with_labels(&y)?;
    /// let bag: Vec<BoostedModel> = (0..3)
    ///     .map(|seed| {
    ///         let params = TrainingParams::builder().subsample(0.7).seed(seed).build()?;
    ///         train(&params, &data, 10)
    ///     })
    ///     .collect::<Result<_>>()?;
    /// let average = BoostedModel::average(&bag)?;
    /// assert_eq!(average.num_trees(), 30);
    ///
    /// let mean = average.predict_margin(&data, Iterations::Best)?;
    /// let members: Vec<_> = bag
    ///     .iter()
    ///     .map(|model| model.predict_margin(&data, Iterations::Best))
    ///     .collect::<Result<_>>()?;
    /// for (row, &margin) in mean.as_slice().iter().enumerate() {
    ///     let expected = members.iter().map(|m| m.as_slice()[row]).sum::<f32>() / 3.0;
    ///     assert!((margin - expected).abs() < 1e-5);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn average<'a>(models: impl IntoIterator<Item = &'a BoostedModel>) -> Result<BoostedModel> {
        let members: Vec<&BoostedModel> = models.into_iter().collect();
        let Some(&first) = members.first() else {
            return Err(HessboostError::invalid_param(
                "models",
                "no models to average",
            ));
        };
        if members.len() >= MAX_MEMBERS {
            return Err(HessboostError::invalid_param(
                "models",
                format!(
                    "{} models to average; at most {} are supported",
                    members.len(),
                    MAX_MEMBERS - 1
                ),
            ));
        }
        // Tree-less members fit either leaf kind; the rest share the kind
        // of the first one with trees.
        let leaf_kind = members
            .iter()
            .enumerate()
            .find(|(_, member)| !member.trees.is_empty())
            .map(|(index, member)| (member.has_vector_leaves(), index));
        for (index, member) in members.iter().enumerate() {
            let conflict = member
                .averageable()
                .map(String::from)
                .or_else(|| first.layout_conflict(member))
                .or_else(|| match leaf_kind {
                    Some((vector, reference))
                        if !member.trees.is_empty() && member.has_vector_leaves() != vector =>
                    {
                        Some(format!(
                            "has vector leaves {}, model {reference} {vector}",
                            member.has_vector_leaves()
                        ))
                    }
                    _ => None,
                });
            if let Some(reason) = conflict {
                return Err(HessboostError::incompatible_model(
                    "average",
                    format!("model {index} {reason}"),
                ));
            }
        }
        // Exact: `MAX_MEMBERS` is far below `2^53`.
        let count = members.len() as f64;
        let trees = members
            .iter()
            .flat_map(|member| member.trees.iter())
            .map(|tree| {
                let mut tree = tree.clone();
                tree.divide_leaves(count);
                tree
            })
            .collect();
        let base_score = (0..first.n_outputs)
            .map(|output| {
                let sum: f64 = members
                    .iter()
                    .map(|member| f64::from(member.base_score[output]))
                    .sum();
                (sum / count) as f32
            })
            .collect();
        let average = BoostedModel {
            trees,
            base_score,
            objective: first.objective.clone(),
            max_delta_step: first.max_delta_step,
            num_class: first.num_class,
            n_outputs: first.n_outputs,
            n_targets: first.n_targets,
            n_features: first.n_features,
            best_iteration: None,
            tree_weights: TreeWeights::Unit,
            num_parallel_tree: first.num_parallel_tree,
            linear: None,
            shrinkage: None,
            boulevard: None,
            ebm: None,
            layouts: TreeLayouts::default(),
            transform: first.transform.clone(),
        };
        // The checks above make the result valid; keep a loader's check as
        // the last guard, so a missed combination is an error, not a model
        // that cannot predict.
        average.validate_structure().map_err(|error| {
            HessboostError::incompatible_model(
                "average",
                format!("the members do not combine into a valid model: {error}"),
            )
        })?;
        Ok(average)
    }

    /// Why this model cannot be an average's member on its own, if it
    /// cannot: `None` for a plain tree ensemble with unit tree weights.
    fn averageable(&self) -> Option<&'static str> {
        if self.linear.is_some() {
            Some("is a gblinear model")
        } else if self.shrinkage.is_some() {
            Some("was trained with model shrinkage")
        } else if self.tree_weights().any(|weight| weight != 1.0) {
            Some("has tree weights other than 1 (DART)")
        } else if self.boulevard.is_some() {
            Some("is a Boulevard fit")
        } else if self.ebm.is_some() {
            Some("is an EBM")
        } else if self.best_iteration.is_some() {
            Some("was early-stopped: slice it to the iterations to average first")
        } else {
            None
        }
    }

    /// How `other` differs from this model in what their margins depend
    /// on besides trees and intercepts, if it does.
    fn layout_conflict(&self, other: &BoostedModel) -> Option<String> {
        let differs = |what: &str, ours: &dyn std::fmt::Debug, theirs: &dyn std::fmt::Debug| {
            Some(format!("has {what} {theirs:?}, model 0 {ours:?}"))
        };
        if self.objective != other.objective {
            differs("objective", &self.objective, &other.objective)
        } else if self.max_delta_step.to_bits() != other.max_delta_step.to_bits() {
            differs(
                "max_delta_step",
                &self.max_delta_step,
                &other.max_delta_step,
            )
        } else if self.n_features != other.n_features {
            differs("feature count", &self.n_features, &other.n_features)
        } else if self.n_outputs != other.n_outputs {
            differs("output count", &self.n_outputs, &other.n_outputs)
        } else if self.n_targets != other.n_targets {
            differs("target count", &self.n_targets, &other.n_targets)
        } else if self.num_class != other.num_class {
            differs("num_class", &self.num_class, &other.num_class)
        } else if self.num_parallel_tree != other.num_parallel_tree {
            differs(
                "num_parallel_tree",
                &self.num_parallel_tree,
                &other.num_parallel_tree,
            )
        } else {
            None
        }
    }
}
