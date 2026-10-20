//! Exact leaf values ([`LeafEstimation::Exact`]): once a tree's structure is
//! grown on the smooth surrogate's gradients, output `j`'s value in each
//! leaf becomes the quantile at that output's level of the residuals
//! `label - margin_j` of the training rows the tree routes there: the exact
//! minimizer of the pinball loss (`reg:quantileerror`) or of the absolute
//! loss (`reg:absoluteerror`, the median) over those rows.
//!
//! Every training row takes part, sampled or not. [`route_rows`] sends each
//! one through the finished tree as prediction does (by its bins when they
//! route it as its values do), and the same partition then adds the tree to
//! the training margins. Leaves (and a vector leaf's outputs) are fit
//! independently and in parallel, each from its residuals in a stable sort,
//! so the values do not depend on the thread count.

use super::margins::{RowRouting, TreeOutput};
use crate::config::{LeafEstimation, TrainingParams};
use crate::data::sort::{RadixScratch, sort_by_value, sort_values};
use crate::data::{DMatrix, MetaInfo};
use crate::objective::{LeafQuantiles, interpolated_quantile, weighted_step_quantile};
use crate::tree::RegTree;
use crate::tree::builder::LeafRows;
use rayon::prelude::*;

/// Rows per task when routing every training row through a tree.
const ROUTE_CHUNK_ROWS: usize = 16_384;

/// The exact leaves of one training run: every output's quantile level with
/// the label column its residuals read, the training labels and sample
/// weights, and the bound a positive `max_delta_step` puts on an unscaled
/// leaf value.
pub(super) struct ExactLeaves<'a> {
    /// `(alpha, label column)` of each of the loss's outputs.
    levels: Vec<(f32, usize)>,
    /// `[row][label column]`.
    labels: &'a [f32],
    label_cols: usize,
    weights: Option<&'a [f32]>,
    bound: Option<f32>,
}

impl<'a> ExactLeaves<'a> {
    /// The exact leaves `params` asks for, over the training metadata `info`
    /// and the loss's `n_out` outputs; `None` for Newton leaves. Validation
    /// has checked that the objective has exact leaves.
    pub(super) fn new(params: &TrainingParams, info: &MetaInfo<'a>, n_out: usize) -> Option<Self> {
        if params.leaf_estimation != LeafEstimation::Exact {
            return None;
        }
        let levels = match params.objective.leaf_quantiles()? {
            LeafQuantiles::Levels(levels) => {
                levels.alpha_f32().into_iter().map(|a| (a, 0)).collect()
            }
            LeafQuantiles::Medians => (0..n_out).map(|j| (0.5, j)).collect(),
        };
        let max_delta_step = params.effective_max_delta_step();
        Some(ExactLeaves {
            levels,
            labels: info.label_values(),
            label_cols: info.n_targets(),
            weights: info.weights,
            bound: (max_delta_step > 0.0).then_some(max_delta_step as f32),
        })
    }

    /// Set every leaf of `tree` (feeding `output`) to the residual quantile
    /// of the rows `leaves` assigns it at the training `margins` (`[row]`
    /// then one per output) the tree's gradients came from, before the
    /// learning rate scales it.
    pub(super) fn fit(
        &self,
        tree: &mut RegTree,
        output: TreeOutput,
        leaves: &[LeafRows],
        margins: &[f32],
    ) {
        let outputs = match output {
            TreeOutput::Scalar(k) => k..k + 1,
            TreeOutput::Vector => 0..self.levels.len(),
        };
        let width = outputs.len();
        // One task per (leaf, output): a vector leaf's outputs each sort
        // their own residuals.
        let values: Vec<f32> = (0..leaves.len() * width)
            .into_par_iter()
            .map(|task| {
                let (leaf, j) = (&leaves[task / width], outputs.start + task % width);
                self.leaf_value(&leaf.rows, j, margins)
            })
            .collect();
        for (leaf, value) in leaves.iter().zip(values.chunks_exact(width)) {
            match output {
                TreeOutput::Scalar(_) => tree.set_leaf_value(leaf.node, value[0]),
                TreeOutput::Vector => tree.set_leaf_vector(leaf.node, value),
            }
        }
    }

    /// Output `j`'s exact value for a leaf holding `rows`: the quantile of
    /// their residuals (weighted by the sample weights when the data has
    /// them), `0` without rows or weight, within `±max_delta_step` when that
    /// is positive.
    fn leaf_value(&self, rows: &[u32], j: usize, margins: &[f32]) -> f32 {
        let (alpha, label) = self.levels[j];
        let n_out = self.levels.len();
        let residual = |row: u32| {
            let row = row as usize;
            self.labels[row * self.label_cols + label] - margins[row * n_out + j]
        };
        let value = match self.weights {
            None if rows.is_empty() => 0.0,
            None => {
                let mut residuals: Vec<f32> = rows.iter().map(|&row| residual(row)).collect();
                sort_values(&mut residuals, &mut RadixScratch::default());
                interpolated_quantile(alpha, &residuals)
            }
            Some(weights) => {
                let mut pairs: Vec<(f32, f32)> = rows
                    .iter()
                    .map(|&row| (residual(row), weights[row as usize]))
                    .collect();
                if pairs.iter().all(|&(_, weight)| weight == 0.0) {
                    return 0.0;
                }
                sort_by_value(&mut pairs, &mut RadixScratch::default(), |&(r, _)| r);
                weighted_step_quantile(alpha, pairs.iter().copied())
            }
        };
        match self.bound {
            Some(bound) => value.clamp(-bound, bound),
            None => value,
        }
    }
}

/// Every row of `data` routed through `tree` as prediction routes it by raw
/// value, through `routing`: one entry per leaf (an empty one for a leaf no
/// row reaches), leaves in node order, each leaf's rows ascending.
pub(super) fn route_rows(tree: &RegTree, data: &DMatrix, routing: RowRouting) -> Vec<LeafRows> {
    let n = data.n_rows();
    let mut leaf_of = vec![0u32; n];
    match routing.binned(tree) {
        Some(binned) => leaf_of
            .par_chunks_mut(ROUTE_CHUNK_ROWS)
            .enumerate()
            .for_each(|(chunk, leaves)| {
                let first = chunk * ROUTE_CHUNK_ROWS;
                binned.for_each_leaf(first..first + leaves.len(), |row, leaf| {
                    leaves[row - first] = leaf as u32;
                });
            }),
        None => leaf_of
            .par_iter_mut()
            .with_min_len(ROUTE_CHUNK_ROWS)
            .enumerate()
            .for_each(|(row, leaf)| {
                *leaf = tree.leaf_id_with(|f| data.get(row, f as usize)) as u32;
            }),
    }
    let mut counts = vec![0usize; tree.num_nodes()];
    for &leaf in &leaf_of {
        counts[leaf as usize] += 1;
    }
    let mut leaves: Vec<LeafRows> = (0..tree.num_nodes())
        .filter(|&node| tree.node(node).is_leaf())
        .map(|node| LeafRows {
            node,
            rows: Vec::with_capacity(counts[node]),
        })
        .collect();
    // Node id to its entry of `leaves`.
    let mut entry = vec![usize::MAX; tree.num_nodes()];
    for (i, leaf) in leaves.iter().enumerate() {
        entry[leaf.node] = i;
    }
    for (row, &leaf) in leaf_of.iter().enumerate() {
        leaves[entry[leaf as usize]].rows.push(row as u32);
    }
    leaves
}
