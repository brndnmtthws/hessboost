//! The approximate mode's state: per-tree gradients, node statistics, and
//! row and column samples, built by replaying a model's trees (and, under
//! sampling, training's draws) over its training data.

use crate::config::TrainingParams;
use crate::data::DMatrix;
use crate::data::ghist::{Bins, GHistIndex};
use crate::data::quantile::HistCuts;
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::objective::{GradPair, Loss};
use crate::training::dart::{round_rng, round_salt};
use crate::training::margins::{TreeOutput, add_tree_margins};
use crate::training::row_sampling::{RowMeta, iteration_row_subsets, make_column_sampler};
use crate::tree::RegTree;
use crate::tree::builder::all_rows;
use crate::tree::gain::{GradStats, RegParams, calc_weight};
use crate::tree::sampler::ColumnSampler;

/// The approximate mode's state.
#[derive(Debug, Clone)]
pub(super) struct Cache {
    /// The split robustness tolerance (the approximate mode's).
    pub(super) tolerance: f64,
    pub(super) cuts: HistCuts,
    /// No row has a missing value (the builder then enumerates no missing
    /// directions).
    pub(super) dense: bool,
    pub(super) trees: Vec<TreeCache>,
    /// Under row sampling, the position of the next added row in the keyed
    /// streams that draw added rows' samples: the rows held so far, the
    /// training rows and every row added since, so no two rows share one.
    pub(super) next_row: u64,
}

#[derive(Debug, Clone)]
pub(super) struct TreeCache {
    /// Per node of the tree.
    pub(super) nodes: Vec<NodeCache>,
    /// The gradient pair each row of the current data contributes (zero
    /// for a row outside the tree's row sample).
    pub(super) grads: Vec<GradPair>,
    /// Under row sampling (`subsample < 1`), whether each row of the
    /// current data is in the tree's sample.
    pub(super) sample: Option<Vec<bool>>,
    /// Under `colsample_bytree < 1`, the tree's column sampler, which offers
    /// its pool at every node (`colsample_bylevel` and `colsample_bynode`
    /// are `1` in the approximate mode).
    pub(super) columns: Option<ColumnSampler>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct NodeCache {
    pub(super) stats: GradStats,
    /// Per-bin sums (internal nodes only).
    pub(super) hist: Vec<GradStats>,
}

impl Cache {
    /// Replay `model`'s trees over `data` (as training grew them), on the
    /// index training bins `data` with: its cached one, whose cuts (a
    /// source-cut view's source's) the model's splits were found on. Under
    /// row or column sampling each tree's samples are training's draws for
    /// its iteration, which depend on the seed and (row sampling) on the
    /// row count only.
    pub(super) fn build(
        model: &BoostedModel,
        params: &TrainingParams,
        data: &DMatrix,
        tolerance: f64,
    ) -> Result<Self> {
        let ghist = data.hist_index(params.max_bin);
        let cuts = ghist.cuts().clone();
        let objective = params.loss(1)?;
        let n = data.n_rows();
        let (row_sampled, col_sampled) = (params.subsample < 1.0, params.colsample_bytree < 1.0);
        let (meta, all) = (RowMeta::of(data, params), all_rows(n));
        let mut margins = vec![model.base_scores()[0]; n];
        let mut trees = Vec::with_capacity(model.num_trees());
        for (t, tree) in model.trees().iter().enumerate() {
            let (sample, columns) = if row_sampled || col_sampled {
                // The draws of `round::grow_round` for iteration `t`: its
                // row subset, then its tree's column sampler.
                let mut rng = round_rng(params, t, round_salt(params));
                let subsets = iteration_row_subsets(params, false, meta, &all, &mut rng);
                let sampler = make_column_sampler(data, params, &mut rng);
                let sample = row_sampled.then(|| {
                    let mut sample = vec![false; n];
                    for &row in subsets.rows(0) {
                        sample[row as usize] = true;
                    }
                    sample
                });
                (sample, col_sampled.then_some(sampler))
            } else {
                (None, None)
            };
            if let Some(columns) = &columns {
                check_column_sample(tree, t, &columns.tree_features())?;
            }
            let mut grads = gradients(objective.as_ref(), data, &margins);
            if let Some(sample) = &sample {
                for (g, _) in grads.iter_mut().zip(sample).filter(|(_, kept)| !**kept) {
                    *g = GradPair::default();
                }
            }
            let mut nodes = vec![NodeCache::default(); tree.num_nodes()];
            accumulate(
                tree,
                0,
                data,
                &ghist,
                grads
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|&(row, _)| sample.as_ref().is_none_or(|s| s[row])),
                &mut nodes,
            );
            // Without row sampling a later tree's leaves drift from the
            // replayed statistics once an update kept stale gradients (see
            // `check_newton_leaves`); with it every tree of a trained model
            // matches, and checking them all refuses replayed samples that
            // are not the ones the trees were grown on.
            if t == 0 || row_sampled {
                check_newton_leaves(tree, t, &nodes, params)?;
            }
            add_tree_margins(tree, data, &mut margins, 1, TreeOutput::Scalar(0));
            trees.push(TreeCache {
                nodes,
                grads,
                sample,
                columns,
            });
        }
        let dense = ghist.is_dense();
        Ok(Cache {
            tolerance,
            cuts,
            dense,
            trees,
            next_row: n as u64,
        })
    }
}

/// Refuse a tree splitting on a feature outside the column pool
/// `colsample_bytree` draws for it: a model trained with another seed or
/// other column sampling.
fn check_column_sample(tree: &RegTree, t: usize, pool: &[u32]) -> Result<()> {
    let outside = tree
        .nodes()
        .iter()
        .find(|node| !node.is_leaf() && pool.binary_search(&node.split_feature).is_err());
    match outside {
        Some(node) => Err(HessboostError::incompatible_model(
            "model",
            format!(
                "tree {t} splits on feature {}, outside the column sample these parameters \
                 (seed included) draw for it: the approximate mode updates only models \
                 trained with these parameters",
                node.split_feature
            ),
        )),
        None => Ok(()),
    }
}

/// Refuse a tree whose leaves are not the `eta`-scaled Newton steps
/// `params` gives on the replayed rows. Updates recompute every leaf they
/// touch that way, so a model whose leaves follow another convention (an
/// imported LightGBM model, whose first tree carries the label average) or
/// other parameters (another `eta`, `lambda`, `alpha`, another seed under
/// row sampling) or data would have its untouched and recomputed leaves
/// disagree. Without row sampling only the first tree is checked, because
/// its gradients, at the intercept, stay exact through approximate updates:
/// a model an update produced passes on its own data.
fn check_newton_leaves(
    tree: &RegTree,
    t: usize,
    nodes: &[NodeCache],
    params: &TrainingParams,
) -> Result<()> {
    let reg = RegParams::from_params(params);
    let eta = params.eta as f32;
    for (nid, cache) in nodes.iter().enumerate() {
        let node = tree.node(nid);
        if !node.is_leaf() || cache.stats.hess <= 0.0 {
            continue;
        }
        let expected = (calc_weight(cache.stats, &reg) as f32) * eta;
        let stored = node.leaf_value;
        // `false` for a NaN leaf, which is refused too.
        let close = (stored - expected).abs() <= 1e-3 * (1.0 + expected.abs());
        if !close {
            return Err(HessboostError::incompatible_model(
                "model",
                format!(
                    "leaf {nid} of tree {t} holds {stored}, but these parameters give \
                     {expected} on this data: the approximate mode updates only models \
                     trained with these parameters (seed included) on this data (not, for \
                     example, an imported LightGBM model, whose first tree carries the label \
                     average, or, under row sampling, a model an approximate update produced, \
                     whose row samples no replay of training's draws reproduces)"
                ),
            ));
        }
    }
    Ok(())
}

/// Gradient pairs of every row of `data` at `margins`.
fn gradients(objective: &dyn Loss, data: &DMatrix, margins: &[f32]) -> Vec<GradPair> {
    let mut out = vec![GradPair::default(); data.n_rows()];
    objective.gradient_info(margins, &data.info(), &mut out);
    out
}

pub(super) fn stats_of(g: GradPair) -> GradStats {
    GradStats::new(f64::from(g.grad), f64::from(g.hess))
}

/// Add `g` to `hist` at every bin of row `row` of `ghist`.
fn add_index_bins(hist: &mut [GradStats], ghist: &GHistIndex, row: usize, g: GradStats) {
    let (s, e) = (ghist.row_ptr()[row], ghist.row_ptr()[row + 1]);
    match ghist.bins() {
        Bins::U16(b) => b[s..e].iter().for_each(|&bin| hist[bin as usize].add(g)),
        Bins::U32(b) => b[s..e].iter().for_each(|&bin| hist[bin as usize].add(g)),
    }
}

/// Node statistics of `tree` below `root` from `rows` (row of `data` and
/// `ghist`, gradient).
pub(super) fn accumulate(
    tree: &RegTree,
    root: usize,
    data: &DMatrix,
    ghist: &GHistIndex,
    rows: impl Iterator<Item = (usize, GradPair)>,
    out: &mut [NodeCache],
) {
    let bins = ghist.total_bins();
    for (row, g) in rows {
        let g = stats_of(g);
        let mut nid = root;
        loop {
            let node = tree.node(nid);
            out[nid].stats.add(g);
            if node.is_leaf() {
                break;
            }
            if out[nid].hist.is_empty() {
                out[nid].hist = vec![GradStats::default(); bins];
            }
            add_index_bins(&mut out[nid].hist, ghist, row, g);
            nid = tree.child(nid, data.get(row, node.split_feature as usize));
        }
    }
}

/// Reset the caches of `root`'s subtree in `tree`.
pub(super) fn clear_subtree(tree: &RegTree, root: usize, nodes: &mut [NodeCache]) {
    let mut stack = vec![root];
    while let Some(nid) = stack.pop() {
        nodes[nid] = NodeCache::default();
        let node = tree.node(nid);
        if !node.is_leaf() {
            stack.push(node.left as usize);
            stack.push(node.right as usize);
        }
    }
}
