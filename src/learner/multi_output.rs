//! Boosting rounds for `multi_strategy = multi_output_tree`: one vector-leaf
//! tree per round fits every output at once (XGBoost's `IsVectorLeaf` path),
//! for `gbtree` and DART, optionally growing its structure from reduced split
//! gradients supplied by the objective ([`Objective::split_gradient`]).

use super::model::BoostedModel;
use super::sampling::gradient_based_sample;
use super::train::{
    DART_SALT, EvalSet, dart_new_tree_weight, gradient_sampling, make_column_sampler,
    rescale_dropped, round_rng, sample_rows, select_dropout,
};
use crate::config::{BoosterKind, MultiStrategy, TrainingParams, TreeMethod};
use crate::data::ghist::GHistIndex;
use crate::data::{DMatrix, MetaInfo};
use crate::error::{HessboostError, Result};
use crate::objective::{GradPair, Objective, SplitGradient};
use crate::tree::RegTree;
use crate::tree::builder::{LeafRows, MultiTreeBuilder, VectorGradients};
use crate::tree::constraints::MonotoneConstraints;
use rand::rngs::StdRng;
use rayon::prelude::*;

/// Whether training grows vector-leaf trees: `multi_output_tree` with more
/// than one output. A single output always gets scalar trees, as in XGBoost.
pub(super) fn vector_leaf(params: &TrainingParams, n_outputs: usize) -> bool {
    params.multi_strategy == MultiStrategy::MultiOutputTree
        && params.booster != BoosterKind::GbLinear
        && n_outputs > 1
}

/// Configuration checks for `multi_output_tree`: like XGBoost, vector-leaf
/// trees are built by the histogram method only.
pub(super) fn validate(params: &TrainingParams) -> Result<()> {
    if params.multi_strategy == MultiStrategy::MultiOutputTree
        && params.booster != BoosterKind::GbLinear
        && !matches!(params.tree_method, TreeMethod::Hist | TreeMethod::Auto)
    {
        return Err(HessboostError::invalid_param(
            "multi_strategy",
            "`multi_output_tree` requires `tree_method=hist` (or `auto`)",
        ));
    }
    Ok(())
}

/// Reduced split gradients are defined for vector-leaf trees only: refuse an
/// objective that supplies them to any other booster or strategy.
pub(super) fn reject_split_gradient(
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
        return Err(HessboostError::DimensionMismatch {
            what: "split gradient length (n_rows * split n_targets)",
            expected: n_rows.saturating_mul(split.n_targets.max(1)),
            got: split.gpair.len(),
        });
    }
    if MonotoneConstraints::from_params(&params.monotone_constraints).is_active() {
        return Err(HessboostError::invalid_param(
            "monotone_constraints",
            "monotone constraints are not supported with reduced split gradients",
        ));
    }
    Ok(Some(split))
}

/// What every vector-leaf round reads.
pub(super) struct VectorRound<'a> {
    pub(super) params: &'a TrainingParams,
    pub(super) dtrain: &'a DMatrix,
    pub(super) ghist: &'a GHistIndex,
    pub(super) objective: &'a dyn Objective,
    pub(super) info: &'a MetaInfo<'a>,
    pub(super) evals: &'a [EvalSet<'a>],
}

/// One boosting round: grow a vector-leaf tree from the gradients at the
/// current margins (DART: the ensemble minus this round's dropout set), add
/// it to the model, and bring the train and eval margin caches up to date.
pub(super) fn boost_round(
    ctx: &VectorRound,
    model: &mut BoostedModel,
    round: usize,
    train_margin: &mut [f32],
    eval_margins: &mut [Vec<f32>],
    gpair: &mut [GradPair],
) -> Result<()> {
    let params = ctx.params;
    let n_out = model.n_outputs();
    if params.booster == BoosterKind::Dart {
        let mut rng = round_rng(params, round, DART_SALT);
        let (dropped, drop_indices) = select_dropout(model, params, &mut rng);
        let margin_excl = model.predict_margin_dropout(ctx.dtrain, &dropped);
        ctx.objective.gradient_info(&margin_excl, ctx.info, gpair);
        let split = split_gradient(ctx.objective, params, round, gpair, ctx.dtrain.n_rows())?;
        let rows = sample_rows(ctx.dtrain.n_rows(), params, &mut rng);
        let (tree, _) = fit_tree(ctx, gpair, split.as_ref(), &mut rng, &rows, n_out);
        model.push_tree_weighted(tree, dart_new_tree_weight(&drop_indices, params));
        rescale_dropped(model, &drop_indices, params);
        // Rescaled trees make the eval caches non-additive: recompute them.
        // (DART's gradients come from the ensemble, not `train_margin`.)
        for (margins, (d, _)) in eval_margins.iter_mut().zip(ctx.evals) {
            *margins = model.predict_margin_limited_unchecked(d, 0);
        }
        return Ok(());
    }

    ctx.objective.gradient_info(train_margin, ctx.info, gpair);
    let n = ctx.dtrain.n_rows();
    let split = split_gradient(ctx.objective, params, round, gpair, n)?;
    let mut rng = round_rng(params, round, 0);
    let rows = sample_rows(n, params, &mut rng);
    let (tree, leaf_rows) = fit_tree(ctx, gpair, split.as_ref(), &mut rng, &rows, n_out);
    // Leaf row lists identify every training row's leaf when all rows took
    // part in growing the tree.
    if rows.len() == n && !gradient_sampling(params) {
        add_leaf_rows(&tree, &leaf_rows, train_margin, n_out);
    } else {
        add_tree(&tree, ctx.dtrain, train_margin, n_out);
    }
    for (margins, (d, _)) in eval_margins.iter_mut().zip(ctx.evals) {
        add_tree(&tree, d, margins, n_out);
    }
    model.push_tree_weighted(tree, 1.0);
    Ok(())
}

/// Grow one vector-leaf tree: gradient-based row sampling (on the split
/// gradients, replayed on the value gradients), the tree's column sampler,
/// the build, and `eta` shrinkage of every leaf vector.
fn fit_tree(
    ctx: &VectorRound,
    gpair: &[GradPair],
    split: Option<&SplitGradient>,
    rng: &mut StdRng,
    rows: &[u32],
    n_out: usize,
) -> (RegTree, Vec<LeafRows>) {
    let params = ctx.params;
    let (split_gpair, n_split) = split.map_or((gpair, n_out), |s| (&s.gpair[..], s.n_targets));
    let sampled = if gradient_sampling(params) {
        gradient_based_sample(split_gpair, n_split, params.subsample, rng)
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
    let mut sampler = make_column_sampler(
        ctx.dtrain.n_cols(),
        ctx.dtrain.feature_weights(),
        params,
        rng,
    );
    let grad = VectorGradients {
        split: split_gpair,
        n_split,
        value,
        n_outputs: n_out,
    };
    let (mut tree, leaf_rows) =
        MultiTreeBuilder::new(params).build(ctx.ghist, &grad, rows, &mut sampler);
    tree.scale_leaves(params.eta as f32);
    (tree, leaf_rows)
}

/// Add a vector-leaf tree's leaf vector to every row's margins (`[row][k]`).
/// Rows are independent, so the parallel traversal keeps each row's
/// addition order.
fn add_tree(tree: &RegTree, data: &DMatrix, margins: &mut [f32], k: usize) {
    let update = |(row, margin): (usize, &mut [f32])| {
        let leaf = tree.leaf_id_with(|f| data.get(row, f as usize));
        for (m, &v) in margin.iter_mut().zip(tree.leaf_vector(leaf)) {
            *m += v;
        }
    };
    if data.n_rows() >= 4096 && rayon::current_num_threads() > 1 {
        margins
            .par_chunks_mut(k)
            .with_min_len(1024)
            .enumerate()
            .for_each(update);
    } else {
        margins.chunks_mut(k).enumerate().for_each(update);
    }
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
