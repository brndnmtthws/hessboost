//! Boosting rounds for `multi_strategy = multi_output_tree`: one vector-leaf
//! tree per round fits every output at once (XGBoost's `IsVectorLeaf` path),
//! for `gbtree` and DART, optionally growing its structure from reduced split
//! gradients supplied by the objective ([`Objective::split_gradient`]).

use super::train::{
    EvalSet, TrainContext, dart_new_tree_weight, finish_dart, for_each_row_margins,
    gradient_sampling, make_column_sampler, round_gradients, sample_rows, tree_eta,
};
use crate::config::{BoosterKind, Device, MultiStrategy, TrainingParams, TreeMethod};
use crate::data::DMatrix;
use crate::data::ghist::GHistIndex;
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::objective::{GradPair, Objective, SplitGradient};
use crate::rng::Rng;
use crate::training::sampling::gradient_based_sample;
use crate::tree::RegTree;
use crate::tree::builder::{LeafRows, MultiTreeBuilder, VectorGradients};
use crate::tree::constraints::MonotoneConstraints;

/// Whether training grows vector-leaf trees: `multi_output_tree` with more
/// than one output. A single output always gets scalar trees, as in XGBoost.
pub(super) fn vector_leaf(params: &TrainingParams, n_outputs: usize) -> bool {
    params.multi_strategy == MultiStrategy::MultiOutputTree
        && params.booster != BoosterKind::GbLinear
        && n_outputs > 1
}

/// Configuration checks for `multi_output_tree`: like XGBoost, vector-leaf
/// trees are built by the histogram method only, and their builder keeps
/// its own histogram loop, which no GPU backend accelerates.
pub(super) fn validate(params: &TrainingParams, n_outputs: usize) -> Result<()> {
    if params.multi_strategy == MultiStrategy::MultiOutputTree
        && params.booster != BoosterKind::GbLinear
        && !matches!(params.tree_method, TreeMethod::Hist | TreeMethod::Auto)
    {
        return Err(HessboostError::invalid_param(
            "multi_strategy",
            "`multi_output_tree` requires `tree_method=hist` (or `auto`)",
        ));
    }
    if vector_leaf(params, n_outputs) && params.device != Device::Cpu {
        return Err(HessboostError::invalid_param(
            "device",
            "`metal` does not support `multi_strategy = multi_output_tree` \
             (the vector-leaf builder has its own histogram loop)",
        ));
    }
    Ok(())
}

/// Reduced split gradients are defined for vector-leaf trees only: refuse an
/// objective that supplies them to any other booster or strategy.
pub(crate) fn reject_split_gradient(
    objective: &dyn Objective,
    round: usize,
    gpair: &[GradPair],
) -> Result<()> {
    if objective.split_gradient(round, gpair).is_some() {
        return Err(HessboostError::invalid_param(
            "objective",
            "reduced split gradients require `multi_strategy=multi_output_tree` \
             with more than one output",
        ));
    }
    Ok(())
}

/// The objective's split gradients for this round, shape-checked.
fn split_gradient(
    objective: &dyn Objective,
    params: &TrainingParams,
    round: usize,
    gpair: &[GradPair],
    n_rows: usize,
) -> Result<Option<SplitGradient>> {
    let Some(split) = objective.split_gradient(round, gpair) else {
        return Ok(None);
    };
    if split.n_targets == 0 || Some(split.gpair.len()) != n_rows.checked_mul(split.n_targets) {
        return Err(HessboostError::dimension_mismatch(
            "split gradient length (n_rows * split n_targets)",
            n_rows.saturating_mul(split.n_targets.max(1)),
            split.gpair.len(),
        ));
    }
    if MonotoneConstraints::from_params(&params.monotone_constraints).is_active() {
        return Err(HessboostError::invalid_param(
            "monotone_constraints",
            "monotone constraints are not supported with reduced split gradients",
        ));
    }
    Ok(Some(split))
}

/// What every vector-leaf round reads: the run's shared inputs, the training
/// matrix's gradient index, and the eval sets whose margins it keeps current.
pub(super) struct VectorRound<'a> {
    pub(super) run: TrainContext<'a>,
    pub(super) ghist: &'a GHistIndex,
    pub(super) evals: &'a [EvalSet<'a>],
}

/// One boosting iteration: grow `num_parallel_tree` vector-leaf trees from
/// the gradients at the current margins (DART: the ensemble minus this
/// iteration's dropout set), add them to the model, and bring the train and
/// eval margin caches up to date. `iteration` is the model's absolute
/// iteration index (continued training counts on), which seeds the RNG.
pub(super) fn boost_round(
    ctx: &VectorRound,
    model: &mut BoostedModel,
    iteration: usize,
    train_margin: &mut [f32],
    eval_margins: &mut [Vec<f32>],
    gpair: &mut [GradPair],
) -> Result<()> {
    let params = ctx.run.params;
    let n = ctx.run.dtrain.n_rows();
    let n_out = model.n_outputs();
    let (mut rng, dropped) = round_gradients(&ctx.run, model, iteration, train_margin, gpair);
    let split = split_gradient(ctx.run.objective, params, iteration, gpair, n)?;
    let weight = dart_new_tree_weight(dropped.as_deref().unwrap_or_default(), params);
    // One row sample per parallel tree, all drawn before the trees.
    let row_subsets: Vec<Vec<u32>> = (0..params.num_parallel_tree)
        .map(|_| sample_rows(n, params, &mut rng))
        .collect();
    for rows in &row_subsets {
        let (tree, leaf_rows) = fit_tree(ctx, gpair, split.as_ref(), &mut rng, rows, n_out)?;
        // DART's gradients come from the ensemble, not the margin caches
        // (`finish_dart` recomputes the eval ones).
        if dropped.is_none() {
            // Leaf row lists identify every training row's leaf when all
            // rows took part in growing the tree.
            if rows.len() == n && !gradient_sampling(params) {
                add_leaf_rows(&tree, &leaf_rows, train_margin, n_out);
            } else {
                add_tree(&tree, ctx.run.dtrain, train_margin, n_out);
            }
            for (margins, (d, _)) in eval_margins.iter_mut().zip(ctx.evals) {
                add_tree(&tree, d, margins, n_out);
            }
        }
        model.push_tree_weighted(tree, weight);
    }
    if let Some(dropped) = &dropped {
        finish_dart(model, params, dropped, ctx.evals, eval_margins);
    }
    Ok(())
}

/// Grow one vector-leaf tree: gradient-based row sampling (on the split
/// gradients, replayed on the value gradients), the tree's column sampler,
/// the build, and `eta / num_parallel_tree` shrinkage of every leaf vector.
fn fit_tree(
    ctx: &VectorRound,
    gpair: &[GradPair],
    split: Option<&SplitGradient>,
    rng: &mut Rng,
    rows: &[u32],
    n_out: usize,
) -> Result<(RegTree, Vec<LeafRows>)> {
    let params = ctx.run.params;
    let (split_gpair, n_split) = split.map_or((gpair, n_out), |s| (&s.gpair[..], s.n_targets));
    let sampled = if gradient_sampling(params) {
        gradient_based_sample(split_gpair, n_split, params.subsample, rng)?
    } else {
        None
    };
    let sampled_value = match (&sampled, split) {
        (Some(sample), Some(_)) => Some(sample.apply(gpair, n_out)),
        _ => None,
    };
    let (split_gpair, rows) = match &sampled {
        Some(sample) => (sample.gpair.as_slice(), sample.rows.as_slice()),
        None => (split_gpair, rows),
    };
    let value = split.map(|_| sampled_value.as_deref().unwrap_or(gpair));
    let mut sampler = make_column_sampler(ctx.run.dtrain, params, rng);
    let grad = VectorGradients {
        split: split_gpair,
        n_split,
        value,
        n_outputs: n_out,
    };
    let (mut tree, leaf_rows) =
        MultiTreeBuilder::new(params).build(ctx.ghist, &grad, rows, &mut sampler);
    tree.scale_leaves(tree_eta(params));
    Ok((tree, leaf_rows))
}

/// Add a vector-leaf tree's leaf vector to every row's margins (`[row][k]`).
fn add_tree(tree: &RegTree, data: &DMatrix, margins: &mut [f32], k: usize) {
    for_each_row_margins(margins, k, |(row, margin)| {
        let leaf = tree.leaf_id_with(|f| data.get(row, f as usize));
        for (m, &v) in margin.iter_mut().zip(tree.leaf_vector(leaf)) {
            *m += v;
        }
    });
}

/// Add each leaf's vector to the margins of the training rows that reached
/// it.
fn add_leaf_rows(tree: &RegTree, leaf_rows: &[LeafRows], margins: &mut [f32], k: usize) {
    for leaf in leaf_rows {
        let value = tree.leaf_vector(leaf.node);
        for &row in &leaf.rows {
            let margin = &mut margins[row as usize * k..(row as usize + 1) * k];
            for (m, &v) in margin.iter_mut().zip(value) {
                *m += v;
            }
        }
    }
}
