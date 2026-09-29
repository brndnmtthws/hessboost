//! The approximate mode's state: per-tree gradients and node statistics,
//! built by replaying a model's trees over its training data.

use crate::config::TrainingParams;
use crate::data::DMatrix;
use crate::data::ghist::{Bins, GHistIndex};
use crate::data::quantile::HistCuts;
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::objective::{GradPair, Loss};
use crate::training::margins::{TreeOutput, add_tree_margins};
use crate::tree::RegTree;
use crate::tree::gain::{GradStats, RegParams, calc_weight};

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
}

#[derive(Debug, Clone)]
pub(super) struct TreeCache {
    /// Per node of the tree.
    pub(super) nodes: Vec<NodeCache>,
    /// The gradient pair each row of the current data contributes.
    pub(super) grads: Vec<GradPair>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct NodeCache {
    pub(super) stats: GradStats,
    /// Per-bin sums (internal nodes only).
    pub(super) hist: Vec<GradStats>,
}

impl Cache {
    /// Replay `model`'s trees over `data` (as training grew them).
    pub(super) fn build(
        model: &BoostedModel,
        params: &TrainingParams,
        data: &DMatrix,
        tolerance: f64,
    ) -> Result<Self> {
        let cuts = HistCuts::from_dmatrix(data, params.max_bin);
        let ghist = GHistIndex::from_dmatrix(data, cuts.clone());
        let objective = params.loss(1)?;
        let mut margins = vec![model.base_scores()[0]; data.n_rows()];
        let mut trees = Vec::with_capacity(model.num_trees());
        for (t, tree) in model.trees().iter().enumerate() {
            let grads = gradients(objective.as_ref(), data, &margins);
            let mut nodes = vec![NodeCache::default(); tree.num_nodes()];
            accumulate(
                tree,
                0,
                data,
                &ghist,
                grads.iter().copied().enumerate(),
                &mut nodes,
            );
            if t == 0 {
                check_newton_leaves(tree, &nodes, params)?;
            }
            add_tree_margins(tree, data, &mut margins, 1, TreeOutput::Scalar(0));
            trees.push(TreeCache { nodes, grads });
        }
        let dense = ghist.dense_stride().is_some();
        Ok(Cache {
            tolerance,
            cuts,
            dense,
            trees,
        })
    }
}

/// Refuse a first tree whose leaves are not the `eta`-scaled Newton steps
/// `params` gives on the replayed rows. Updates recompute every leaf they
/// touch that way, so a model whose leaves follow another convention (an
/// imported LightGBM model, whose first tree carries the label average) or
/// other parameters (another `eta`, `lambda`, `alpha`) or data would have
/// its untouched and recomputed leaves disagree. The first tree is checked
/// because its gradients, at the intercept, stay exact through approximate
/// updates: a model an update produced passes on its own data.
fn check_newton_leaves(tree: &RegTree, nodes: &[NodeCache], params: &TrainingParams) -> Result<()> {
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
                    "leaf {nid} of the first tree holds {stored}, but these parameters give \
                     {expected} on this data: the approximate mode updates only models \
                     trained with these parameters on this data (not, for example, an \
                     imported LightGBM model, whose first tree carries the label average)"
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
