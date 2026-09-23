//! Frozen decoder for version 1 of the native binary format (`SQB\0`, version
//! byte `1`), written by hessboost 0.1.1 and earlier.
//!
//! Postcard is not self-describing: a payload decodes only against the exact
//! field order and types it was written with, and `#[serde(default)]` does
//! not apply. The structs below are private copies of the serialized types as
//! they were in 0.1.1 and must never change; the current types have since
//! gained fields (`BoostedModel::n_targets` and `num_parallel_tree`,
//! `RegTree` vector leaves and linear leaves, `ObjectiveParams` quantile /
//! expectile / AFT and `dist:*` parameters).
//!
//! Version 1 laid trees out round-robin over outputs (tree `t` feeds output
//! `t % n_outputs`, `best_iteration` counts rounds of `n_outputs` trees).
//! That is the current layout with `num_parallel_tree = 1`, where tree `t`
//! feeds output `(t / 1) % n_outputs`, so trees keep their order. Every other
//! new field takes the value the native JSON format's serde defaults give a
//! 0.1.1 JSON file: one label column, one tree per output per iteration,
//! scalar constant leaves, empty quantile / expectile lists with a normal
//! AFT distribution of scale `1`, and no `dist:*` family (Fisher scoring,
//! random split direction).

use super::model::{BoostedModel, LinearModel, ModelSpec};
use crate::config::{AftDistribution, DistGradient, DistSplitDirection, ObjectiveParams};
use crate::error::{HessboostError, Result};
use crate::tree::{Node, RegTree};
use serde::Deserialize;

/// `BoostedModel` as of 0.1.1 (its `compact` field was `#[serde(skip)]`).
#[derive(Deserialize)]
struct ModelV1 {
    trees: Vec<TreeV1>,
    base_score: Vec<f32>,
    objective: String,
    objective_params: ObjectiveParamsV1,
    num_class: usize,
    n_outputs: usize,
    n_features: usize,
    best_iteration: Option<usize>,
    tree_weights: Vec<f32>,
    linear: Option<LinearModelV1>,
}

/// `ObjectiveParams` as of 0.1.1.
#[derive(Deserialize)]
struct ObjectiveParamsV1 {
    scale_pos_weight: f64,
    max_delta_step: f64,
    tweedie_variance_power: f64,
    huber_slope: f64,
    lambdarank_num_pair_per_sample: usize,
}

/// `RegTree` as of 0.1.1.
#[derive(Deserialize)]
struct TreeV1 {
    nodes: Vec<NodeV1>,
    categories: Vec<u32>,
}

/// `Node` as of 0.1.1.
#[derive(Deserialize)]
struct NodeV1 {
    split_feature: u32,
    split_cond: f32,
    default_left: bool,
    left: i32,
    right: i32,
    leaf_value: f32,
    sum_hess: f32,
    split_gain: f32,
    is_categorical: bool,
    cat_begin: u32,
    cat_end: u32,
}

/// `LinearModel` as of 0.1.1.
#[derive(Deserialize)]
struct LinearModelV1 {
    weights: Vec<f32>,
    bias: Vec<f32>,
}

/// Decode a version-1 payload (the bytes after the magic and version byte)
/// into the current model. The caller validates the result.
pub(super) fn decode(payload: &[u8]) -> Result<BoostedModel> {
    let v1: ModelV1 =
        postcard::from_bytes(payload).map_err(|e| HessboostError::ModelFormat(e.to_string()))?;
    let op = v1.objective_params;
    let spec = ModelSpec {
        objective: v1.objective,
        objective_params: ObjectiveParams {
            scale_pos_weight: op.scale_pos_weight,
            max_delta_step: op.max_delta_step,
            tweedie_variance_power: op.tweedie_variance_power,
            huber_slope: op.huber_slope,
            lambdarank_num_pair_per_sample: op.lambdarank_num_pair_per_sample,
            quantile_alpha: Vec::new(),
            expectile_alpha: Vec::new(),
            aft_loss_distribution: AftDistribution::Normal,
            aft_loss_distribution_scale: 1.0,
            dist_gradient: DistGradient::Fisher,
            dist_split_direction: DistSplitDirection::Random,
            distribution: None,
        },
        num_class: v1.num_class,
        n_outputs: v1.n_outputs,
        n_targets: 1,
        n_features: v1.n_features,
    };
    let trees = v1.trees.into_iter().map(TreeV1::into_current).collect();
    let mut model = BoostedModel::from_parts(trees, v1.tree_weights, v1.base_score, spec);
    model.set_best_iteration(v1.best_iteration);
    if let Some(linear) = v1.linear {
        model.set_linear(LinearModel::new(linear.weights, linear.bias));
    }
    Ok(model)
}

impl TreeV1 {
    fn into_current(self) -> RegTree {
        let nodes = self
            .nodes
            .into_iter()
            .map(|n| Node {
                split_feature: n.split_feature,
                split_cond: n.split_cond,
                default_left: n.default_left,
                left: n.left,
                right: n.right,
                leaf_value: n.leaf_value,
                sum_hess: n.sum_hess,
                split_gain: n.split_gain,
                is_categorical: n.is_categorical,
                cat_begin: n.cat_begin,
                cat_end: n.cat_end,
            })
            .collect();
        RegTree::from_scalar_parts(nodes, self.categories)
    }
}
