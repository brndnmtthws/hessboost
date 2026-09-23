//! TreeSHAP for vector-leaf models (`multi_strategy = multi_output_tree`).
//!
//! XGBoost 3.3+ computes exact contributions and interactions for vector
//! leaves by walking each tree once per output with that output's leaf
//! values, sharing the tree's covers (the Hessians summed over targets) as
//! branch probabilities and taking each output's root expectation
//! separately. That is the scalar algorithm applied to `K` scalar views of
//! every tree, so this module expands the model into those views — tree `t`,
//! output `j` becomes scalar tree `t * K + j`, which the scalar layout routes
//! to output `j`, with tree `t`'s weight — and runs the scalar TreeSHAP over
//! them. Per output the trees keep their order, so the accumulation order is
//! XGBoost's.

use super::model::{BoostedModel, ModelSpec};

impl BoostedModel {
    /// The scalar model equivalent, for attribution, to this vector-leaf
    /// model's effective trees (see the module docs).
    pub(super) fn vector_leaf_shap_model(&self) -> BoostedModel {
        let k = self.n_outputs();
        let trees = &self.trees()[..self.effective_ntrees()];
        let mut expanded = Vec::with_capacity(trees.len() * k);
        let mut weights = Vec::with_capacity(trees.len() * k);
        for (t, tree) in trees.iter().enumerate() {
            for j in 0..k {
                expanded.push(tree.output_tree(j));
                weights.push(self.tree_weight(t));
            }
        }
        BoostedModel::from_parts(
            expanded,
            weights,
            self.base_scores().to_vec(),
            ModelSpec {
                objective: self.objective().to_string(),
                objective_params: self.objective_params().clone(),
                num_class: self.num_class(),
                n_outputs: k,
                n_targets: self.n_targets(),
                n_features: self.n_features(),
            },
        )
    }
}
