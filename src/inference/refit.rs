//! The honest ("integrity") refit of a Boulevard model's leaves on an
//! independent sample.

use super::check_data;
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::rng::Rng;
use crate::training::boulevard::{Recursion, RoundRequest, Schedule};
use crate::tree::RegTree;

/// The refit's RNG salt, separating its draws from the training run's.
const REFIT_SALT: u64 = 0x1_0E57;

/// Every node's count of the rows whose leaves are `leaf_counts` (per node
/// id; internal nodes `0`), summed up the tree.
fn node_counts(tree: &RegTree, leaf_counts: &[usize]) -> Vec<usize> {
    let nodes = tree.nodes();
    let mut counts = leaf_counts.to_vec();
    // Post-order without assuming children follow their parents.
    let mut stack = vec![(0usize, false)];
    while let Some((id, expanded)) = stack.pop() {
        let node = &nodes[id];
        if node.is_leaf() {
            continue;
        }
        let (l, r) = (node.left as usize, node.right as usize);
        if expanded {
            counts[id] = counts[l] + counts[r];
        } else {
            stack.push((id, true));
            stack.push((l, false));
            stack.push((r, false));
        }
    }
    counts
}

/// Refit every leaf of the Boulevard model `model` on `values`, labelled
/// rows independent of its training data, keeping every tree's structure:
/// the Boulevard recursion the model was trained with (BRAT-D with its
/// dropout, or BRAT-P, with the same learning rate, row subsample ratio, L2
/// penalty, and truncation) is rerun on `values`, each tree's leaves set to
/// `Σ z / (m + lambda)` over the `m` rows of a fresh row sample reaching
/// them (`0` for a leaf none reaches), and the result scaled as training
/// scales it. A label-mean intercept is re-estimated on `values`.
///
/// The structures then depend on the training labels only, and the leaf
/// values on `values`' labels only: Fang, Tan & Hooker's *integrity*
/// (Zhou & Hooker's structure–value isolation), under which
/// [`BoulevardInference`](super::BoulevardInference) is fitted on `values`
/// (not the training rows). Node covers become `values`' row counts, so
/// SHAP values of the refitted model are relative to `values`.
///
/// # Errors
///
/// [`HessboostError::InvalidParameter`] when `model` is not a Boulevard
/// fit, or `values` lacks labels or has row weights or base margins;
/// [`HessboostError::DimensionMismatch`] for a different feature count.
pub fn honest_refit(model: &BoostedModel, values: &DMatrix) -> Result<BoostedModel> {
    let info = *model.boulevard().ok_or_else(|| {
        HessboostError::invalid_param(
            "model",
            "not a Boulevard fit: train it with `booster = boulevard`",
        )
    })?;
    check_data(model, values, "values", true)?;
    let n = values.n_rows();
    let labels = values.labels().unwrap_or_default();
    if labels.iter().any(|y| !y.is_finite()) {
        return Err(HessboostError::invalid_param(
            "values",
            "labels must be finite",
        ));
    }
    let mu = if info.intercept_from_labels {
        (labels.iter().map(|&y| f64::from(y)).sum::<f64>() / n as f64) as f32
    } else {
        model.base_score()
    };
    let t_count = model.num_trees();
    let node_ids = model.predict_leaf_range(values, ..)?;
    let parallel = model.num_parallel_tree();
    let schedule = Schedule::from_info(&info, parallel, REFIT_SALT);
    let mut refit = model.clone();
    let mut recursion = Recursion::new(schedule, n);
    let trees = refit.trees_mut();
    for _ in 0..model.num_boost_rounds() {
        recursion.step(|request| {
            let RoundRequest {
                index,
                first_slot,
                offsets,
                rng,
            } = request;
            let mut preds = Vec::with_capacity(offsets.len());
            for (j, offset) in offsets.iter().enumerate() {
                let t = index * parallel + first_slot + j;
                let in_bag = row_sample(n, info.subsample, rng);
                let tree = &mut trees[t];
                let mut sums = vec![0.0f64; tree.num_nodes()];
                let mut counts = vec![0usize; tree.num_nodes()];
                for (row, &keep) in in_bag.iter().enumerate() {
                    if keep {
                        let leaf = node_ids[row * t_count + t] as usize;
                        sums[leaf] += f64::from(labels[row]) - f64::from(mu) - offset[row];
                        counts[leaf] += 1;
                    }
                }
                let mut values = vec![0.0f32; tree.num_nodes()];
                for (id, node) in tree.nodes().iter().enumerate() {
                    if node.is_leaf() {
                        let denom = counts[id] as f64 + info.reg_lambda;
                        if denom > 0.0 {
                            values[id] = (sums[id] / denom) as f32;
                        }
                    }
                }
                for (id, &v) in values.iter().enumerate() {
                    if tree.nodes()[id].is_leaf() {
                        tree.set_leaf_value(id, v);
                    }
                }
                preds.push(
                    (0..n)
                        .map(|row| values[node_ids[row * t_count + t] as usize])
                        .collect(),
                );
            }
            Ok(preds)
        })?;
    }
    let scale = recursion.scale() as f32;
    for (t, tree) in trees.iter_mut().enumerate() {
        tree.scale_leaves(scale);
        let mut leaf_counts = vec![0usize; tree.num_nodes()];
        for row in 0..n {
            leaf_counts[node_ids[row * t_count + t] as usize] += 1;
        }
        for (id, &c) in node_counts(tree, &leaf_counts).iter().enumerate() {
            tree.set_sum_hess(id, c as f32);
        }
    }
    refit.set_base_scores(vec![mu]);
    refit.set_best_iteration(None);
    if refit
        .trees()
        .iter()
        .any(|t| t.nodes().iter().any(|n| !n.leaf_value.is_finite()))
    {
        return Err(HessboostError::invalid_param(
            "values",
            "the refitted leaves overflow f32",
        ));
    }
    Ok(refit)
}

/// A Bernoulli row sample: row `i` is kept with probability `ratio` (every
/// row when `ratio >= 1`), at least one row, as training samples them.
fn row_sample(n: usize, ratio: f64, rng: &mut Rng) -> Vec<bool> {
    if ratio >= 1.0 {
        return vec![true; n];
    }
    let mut keep: Vec<bool> = (0..n).map(|_| rng.f64() < ratio).collect();
    if !keep.iter().any(|&k| k) {
        keep[rng.range(0..n)] = true;
    }
    keep
}
