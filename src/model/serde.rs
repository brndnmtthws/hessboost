use super::objective::{PartialStoredObjectiveParams, StoredObjectiveParams};
use super::{BoostedModel, LinearModel, ModelObjective, Shrinkage, TreeWeights};
use crate::ebm::EbmInfo;
use crate::error::{HessboostError, Result};
use crate::inference::BoulevardInfo;
use crate::tree::{RegTree, UncheckedRegTree};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// A [`BoostedModel`] as native JSON stores it (same names and order as
/// [`UncheckedBoostedModel`]), borrowed from the model.
#[derive(Serialize)]
struct SerializedBoostedModel<'a> {
    trees: &'a [RegTree],
    base_score: &'a [f32],
    objective: &'a str,
    objective_params: StoredObjectiveParams,
    num_class: usize,
    n_outputs: usize,
    n_targets: usize,
    n_features: usize,
    best_iteration: Option<usize>,
    tree_weights: &'a [f32],
    num_parallel_tree: usize,
    linear: &'a Option<LinearModel>,
    shrinkage: &'a Option<Shrinkage>,
    boulevard: &'a Option<BoulevardInfo>,
    ebm: &'a Option<EbmInfo>,
}

impl Serialize for BoostedModel {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        SerializedBoostedModel {
            trees: &self.trees,
            base_score: &self.base_score,
            objective: self.objective.name(),
            objective_params: StoredObjectiveParams::of(&self.objective, self.max_delta_step),
            num_class: self.num_class,
            n_outputs: self.n_outputs,
            n_targets: self.n_targets,
            n_features: self.n_features,
            best_iteration: self.best_iteration,
            tree_weights: self.tree_weights.as_slice(),
            num_parallel_tree: self.num_parallel_tree,
            linear: &self.linear,
            shrinkage: &self.shrinkage,
            boulevard: &self.boulevard,
            ebm: &self.ebm,
        }
        .serialize(serializer)
    }
}

/// The serialized fields of a [`BoostedModel`] (the native JSON format:
/// same names and layout), before validation. `BoostedModel`'s
/// `Deserialize` converts it with [`BoostedModel::try_from`], which runs
/// [`BoostedModel::validate_structure`].
///
/// Everything predictions depend on is required, including the nullable
/// `linear` (the gblinear weights), which a plain `Option` field would
/// default to `null` when absent; an absent `best_iteration` means none was
/// selected (every iteration predicts). Only the objective
/// parameters may be omitted (all of them, or any subset): each missing
/// one takes the recorded objective's default
/// ([`StoredObjectiveParams::defaults_for`]). A tree may omit `size_leaf_vector`
/// (scalar) and `leaf_vectors` (none), except that a multi-output model's
/// trees must state `size_leaf_vector`, since it decides whether they are
/// vector-leaf trees; each tree's `linear` is required
/// ([`UncheckedRegTree`]). An absent `boulevard` or `ebm` (files written
/// before those boosters existed) means the model is not such a fit.
#[derive(Deserialize)]
pub(super) struct UncheckedBoostedModel {
    trees: Vec<UncheckedRegTree>,
    base_score: Vec<f32>,
    objective: String,
    #[serde(default)]
    objective_params: PartialStoredObjectiveParams,
    num_class: usize,
    n_outputs: usize,
    n_targets: usize,
    n_features: usize,
    best_iteration: Option<usize>,
    tree_weights: Vec<f32>,
    num_parallel_tree: usize,
    #[serde(deserialize_with = "Option::deserialize")]
    linear: Option<LinearModel>,
    /// Absent in files written before model shrinkage existed: none.
    #[serde(default)]
    shrinkage: Option<Shrinkage>,
    #[serde(default)]
    boulevard: Option<BoulevardInfo>,
    #[serde(default)]
    ebm: Option<EbmInfo>,
}

impl TryFrom<UncheckedBoostedModel> for BoostedModel {
    type Error = HessboostError;

    fn try_from(m: UncheckedBoostedModel) -> Result<Self> {
        if m.n_outputs > 1
            && let Some(tree) = m.trees.iter().position(|t| !t.states_leaf_width())
        {
            return Err(HessboostError::ModelFormat(format!(
                "tree {tree} of a {}-output model does not state size_leaf_vector",
                m.n_outputs
            )));
        }
        let stored = m.objective_params.fill(&m.objective);
        let objective = ModelObjective::from_stored(&m.objective, &stored, m.num_class)?;
        let model = BoostedModel {
            trees: m
                .trees
                .into_iter()
                .map(UncheckedRegTree::into_unchecked)
                .collect(),
            base_score: m.base_score,
            max_delta_step: stored.max_delta_step,
            objective,
            num_class: m.num_class,
            n_outputs: m.n_outputs,
            n_targets: m.n_targets,
            n_features: m.n_features,
            best_iteration: m.best_iteration,
            tree_weights: TreeWeights::from_vec(m.tree_weights),
            num_parallel_tree: m.num_parallel_tree,
            linear: m.linear,
            shrinkage: m.shrinkage,
            boulevard: m.boulevard,
            ebm: m.ebm,
            compact: OnceLock::new(),
            transform: OnceLock::new(),
        };
        model.validate_structure()?;
        Ok(model)
    }
}
