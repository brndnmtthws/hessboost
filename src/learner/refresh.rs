//! The refresh updater behind `process_type=update` (XGBoost's
//! `updater=refresh`, `src/tree/updater_refresh.cc`).
//!
//! Refreshing keeps a tree's split structure and recomputes its node
//! statistics from new gradients: every row is routed from the root to its
//! leaf, adding its gradient pair to each node on the path (`f64`
//! accumulation, like XGBoost's `GradStats`). Each node then gets
//!
//! * `sum_hess` = the accumulated Hessian (its cover),
//! * `split_gain` = `gain(left) + gain(right) - gain(node)` for internal nodes,
//! * with `refresh_leaf`, a leaf value `calc_weight(stats) * learning_rate`
//!   (the node's base weight, shrunk by the per-tree learning rate
//!   `eta / num_parallel_tree`); without it the leaf values stay as they are.
//!
//! No rows are subsampled. Unlike split finding, XGBoost's refresh applies
//! no `min_child_weight` floor: a node with any positive Hessian gets its
//! regularized weight (`CalcWeight` / `CalcGain` in `src/tree/param.h`).

use crate::config::TrainingParams;
use crate::data::DMatrix;
use crate::objective::GradPair;
use crate::tree::RegTree;
use crate::tree::gain::{GradStats, RegParams, calc_gain, calc_weight};
use rayon::prelude::*;

/// Rows per statistics block. Blocks are reduced in index order, so the sums
/// do not depend on the thread count.
const REFRESH_BLOCK_ROWS: usize = 4096;

/// Refresh `tree` in place from the per-row gradients `gpair` (one pair per
/// row of `data`) with `params`' regularization and `refresh_leaf`, shrinking
/// refreshed leaves by `learning_rate`. See the module docs for the
/// recomputed quantities.
pub(super) fn refresh_tree(
    tree: &mut RegTree,
    data: &DMatrix,
    gpair: &[GradPair],
    params: &TrainingParams,
    learning_rate: f32,
) {
    let reg = &RegParams {
        min_child_weight: 0.0,
        ..RegParams::from_params(params)
    };
    let refresh_leaf = params.refresh_leaf;
    let stats = node_stats(tree, data, gpair);
    for nid in 0..tree.num_nodes() {
        let node = *tree.node(nid);
        tree.set_sum_hess(nid, stats[nid].hess as f32);
        if node.is_leaf() {
            if refresh_leaf {
                let base_weight = calc_weight(stats[nid], reg) as f32;
                tree.set_leaf_value(nid, base_weight * learning_rate);
            }
        } else {
            let gain = calc_gain(stats[node.left as usize], reg)
                + calc_gain(stats[node.right as usize], reg)
                - calc_gain(stats[nid], reg);
            tree.set_split_gain(nid, gain as f32);
        }
    }
}

/// Per-node gradient statistics of `tree` over every row of `data`.
fn node_stats(tree: &RegTree, data: &DMatrix, gpair: &[GradPair]) -> Vec<GradStats> {
    let n_nodes = tree.num_nodes();
    let block_stats = |rows: std::ops::Range<usize>| {
        let mut stats = vec![GradStats::default(); n_nodes];
        for row in rows {
            let gp = GradStats::from_pair(gpair[row]);
            let mut nid = 0;
            stats[nid].add(gp);
            while !tree.node(nid).is_leaf() {
                let feature = tree.node(nid).split_feature as usize;
                nid = tree.child(nid, data.get(row, feature));
                stats[nid].add(gp);
            }
        }
        stats
    };
    let n = data.n_rows();
    let blocks: Vec<Vec<GradStats>> = (0..n.div_ceil(REFRESH_BLOCK_ROWS))
        .into_par_iter()
        .map(|b| block_stats(b * REFRESH_BLOCK_ROWS..((b + 1) * REFRESH_BLOCK_ROWS).min(n)))
        .collect();
    let mut total = vec![GradStats::default(); n_nodes];
    for block in blocks {
        for (acc, s) in total.iter_mut().zip(block) {
            acc.add(s);
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainingParams;

    /// Stump on feature 0 at 0.5 (missing left) with placeholder statistics.
    fn stump() -> RegTree {
        let mut t = RegTree::with_root(99.0);
        t.expand(0, 0, 0.5, true, 7.0, 99.0, -7.0, 99.0);
        t
    }

    #[test]
    fn refresh_recomputes_cover_gain_and_leaves_from_the_routed_rows() {
        // Rows 0, 1 and the missing row 3 go left; row 2 goes right.
        let data = DMatrix::from_dense(&[0.1, 0.2, 0.9, f32::NAN], 4, 1).unwrap();
        let gpair = [
            GradPair::new(1.0, 1.0),
            GradPair::new(2.0, 1.0),
            GradPair::new(-3.0, 2.0),
            GradPair::new(0.5, 0.5),
        ];
        // min_child_weight = 3 exceeds the right leaf's Hessian (2): refresh
        // still gives it a weight, as XGBoost's refresh does.
        let params = TrainingParams::builder()
            .lambda(1.0)
            .min_child_weight(3.0)
            .build()
            .unwrap();
        let reg = RegParams {
            min_child_weight: 0.0,
            ..RegParams::from_params(&params)
        };
        let mut tree = stump();
        refresh_tree(&mut tree, &data, &gpair, &params, 0.25);

        let (left, right, root) = (
            GradStats::new(3.5, 2.5),
            GradStats::new(-3.0, 2.0),
            GradStats::new(0.5, 4.5),
        );
        assert_eq!(tree.node(0).sum_hess, 4.5);
        assert_eq!(tree.node(1).sum_hess, 2.5);
        assert_eq!(tree.node(2).sum_hess, 2.0);
        let gain = calc_gain(left, &reg) + calc_gain(right, &reg) - calc_gain(root, &reg);
        assert_eq!(tree.node(0).split_gain, gain as f32);
        // Leaf = base weight -G / (H + lambda), shrunk by the learning rate.
        assert_eq!(tree.node(1).leaf_value, (-3.5f64 / 3.5) as f32 * 0.25);
        assert_eq!(tree.node(2).leaf_value, (3.0f64 / 3.0) as f32 * 0.25);

        // Without refresh_leaf the statistics change but the leaves do not.
        let mut kept = stump();
        let keep_leaves = TrainingParams {
            refresh_leaf: false,
            ..params
        };
        refresh_tree(&mut kept, &data, &gpair, &keep_leaves, 0.25);
        assert_eq!(kept.node(1).leaf_value, 7.0);
        assert_eq!(kept.node(2).leaf_value, -7.0);
        assert_eq!(kept.node(0).sum_hess, 4.5);
        assert_eq!(kept.node(0).split_gain, gain as f32);
    }

    #[test]
    fn a_leaf_no_row_reaches_gets_zero_weight() {
        let data = DMatrix::from_dense(&[0.1, 0.2], 2, 1).unwrap();
        let gpair = [GradPair::new(1.0, 1.0), GradPair::new(1.0, 1.0)];
        let mut tree = stump();
        refresh_tree(&mut tree, &data, &gpair, &TrainingParams::default(), 0.3);
        assert_eq!(tree.node(2).sum_hess, 0.0);
        assert_eq!(tree.node(2).leaf_value, 0.0);
    }
}
