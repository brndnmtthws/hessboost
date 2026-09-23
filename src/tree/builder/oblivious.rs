//! Symmetric (oblivious) tree growth, CatBoost-style (`grow_policy =
//! symmetric`, beyond XGBoost).
//!
//! A symmetric tree applies one split — the same feature, threshold, and
//! missing-value direction — to every node of a level. The split is chosen
//! greedily level by level: each candidate is scored across all nodes of the
//! level at once, and the candidate with the largest summed gain wins
//! (CatBoost's level-wise search). Candidate enumeration, per-node gain, and
//! leaf weights reuse the histogram builder's XGBoost arithmetic
//! ([`xgb_loss_chg`], `f32` gains, [`finalize_leaf_values`]), so `lambda`,
//! `alpha`, `max_delta_step`, and monotone bounds act per node exactly as they
//! do for `depthwise` trees.
//!
//! # Regularization and pruning
//!
//! A node takes the level split only when that split is valid *for the node*
//! under XGBoost's rules: both children have positive Hessian and meet
//! `min_child_weight`, the monotone direction holds, and the loss change is
//! positive (above `kRtEps`) and at least `gamma`. A node that fails stays a
//! leaf for good — deeper levels never split it — so the tree is a complete
//! `2^depth`-leaf tree with some subtrees collapsed into one leaf. A
//! candidate's level score is `Σ (loss_chg − gamma)` over the nodes it would
//! split, i.e. the regularized objective reduction of the whole level with
//! `gamma` charged once per new split; nodes it would not split add nothing.
//! Growth stops at `max_depth` or at the first level where no candidate splits
//! any node.
//!
//! # Other parameters
//!
//! - Missing values: every candidate fixes one default direction for the whole
//!   level. The forward sweep sends missing values right and, when any node of
//!   the level has missing values for the feature, the backward sweep sends
//!   them left (XGBoost's two enumeration passes). A node with no missing
//!   values scores a backward candidate as the equivalent forward partition.
//! - Column sampling: `colsample_bylevel` and `colsample_bynode` draw one
//!   subset per level, since a level makes a single split decision.
//! - Interaction constraints: every unpruned node of a level shares the same
//!   path features (the earlier levels' splits), so the permitted set is one
//!   per level.
//! - Categorical features and `tree_method = exact` are rejected before
//!   training ([`check_symmetric_input`]); `max_depth` must lie in
//!   `1..=`[`MAX_SYMMETRIC_DEPTH`](crate::config::MAX_SYMMETRIC_DEPTH).
//!
//! Trees are ordinary [`RegTree`]s (node ids breadth-first), so SHAP, JSON,
//! and XGBoost export treat them like any other tree; prediction recognizes
//! the shape and routes rows by bit pattern (`crate::tree::oblivious`).

use super::hist::{partition_rows, rayon_available};
use super::{
    BELOW_ALL_VALUES, BestSplit, InteractionState, K_RT_EPS, LeafRows, SplitPos,
    build_interaction_sets, finalize_leaf_values, next_allowed, permits, sum_rows, xgb_loss_chg,
    xgb_node_gain,
};
use crate::config::{TrainingParams, TreeMethod};
use crate::data::ghist::GHistIndex;
use crate::data::{DMatrix, FeatureType};
use crate::error::{HessboostError, Result};
use crate::objective::GradPair;
use crate::tree::constraints::{Bounds, MonotoneConstraints, child_bounds};
use crate::tree::gain::{GradStats, RegParams};
use crate::tree::hist::{CpuBackend, Histogram, HistogramBackend, subtract_in_place, zeroed};
use crate::tree::regtree::RegTree;
use crate::tree::sampler::ColumnSampler;
use rayon::prelude::*;

/// Combined level rows at which the level's child histograms are built
/// concurrently. Below this, the fork costs more than the scan.
const PARALLEL_LEVEL_ROWS: usize = 4096;

/// Histogram bins scanned per level (nodes × total bins) at which candidate
/// features are scored concurrently.
const PARALLEL_SCORE_BINS: usize = 8192;

/// Refuse inputs symmetric growth does not handle: the exact method (which
/// has no level-wide histogram search) and categorical features (a shared
/// category-set split has no single optimal partition across nodes).
pub(crate) fn check_symmetric_input(method: TreeMethod, dtrain: &DMatrix) -> Result<()> {
    if method == TreeMethod::Exact {
        return Err(HessboostError::invalid_param(
            "grow_policy",
            "`symmetric` growth requires `tree_method=hist` or `approx`",
        ));
    }
    if dtrain.feature_types().contains(&FeatureType::Categorical) {
        return Err(HessboostError::invalid_param(
            "grow_policy",
            "`symmetric` growth does not support categorical features",
        ));
    }
    Ok(())
}

/// A node of the level being split.
struct LevelNode {
    nid: usize,
    rows: Vec<u32>,
    /// Empty at the last level, whose nodes are never split.
    hist: Histogram,
    total: GradStats,
    bounds: Bounds,
    /// XGBoost's `root_gain` baseline for this node's candidates.
    root_gain: f32,
}

/// One level-wide split candidate.
#[derive(Debug, Clone, Copy)]
struct LevelSplit {
    feature: u32,
    /// Boundary offset within the feature's bins `fs..fe`. Forward (missing
    /// right): bins `<= fs + offset` go left. Backward (missing left): bins
    /// `>= fs + offset` go right.
    offset: usize,
    missing_left: bool,
    /// `Σ (loss_chg − gamma)` over the nodes the split applies to.
    score: f64,
}

impl LevelSplit {
    /// The histogram position the tree node and row partition use.
    fn pos(&self, fs: usize) -> SplitPos {
        match (self.missing_left, self.offset) {
            (false, b) => SplitPos::Bin(fs + b),
            (true, 0) => SplitPos::BelowBins,
            (true, b) => SplitPos::Bin(fs + b - 1),
        }
    }
}

/// A level split as applied to one node.
struct NodeSplit {
    left: GradStats,
    right: GradStats,
    loss_chg: f32,
    w_left: f32,
    w_right: f32,
}

/// Symmetric tree builder over the histogram index.
pub(super) struct SymmetricTreeBuilder<'a> {
    params: &'a TrainingParams,
    reg: RegParams,
    cons: MonotoneConstraints,
    interaction_sets: Option<Vec<Vec<u32>>>,
    backend: CpuBackend,
}

impl<'a> SymmetricTreeBuilder<'a> {
    pub(super) fn new(params: &'a TrainingParams) -> Self {
        SymmetricTreeBuilder {
            params,
            reg: RegParams::from_params(params),
            cons: MonotoneConstraints::from_params(&params.monotone_constraints),
            interaction_sets: build_interaction_sets(&params.interaction_constraints),
            backend: CpuBackend,
        }
    }

    /// Grow one symmetric tree. With `capture_rows`, also return every leaf's
    /// training rows (ascending) so the caller can update margins directly.
    pub(super) fn build(
        &self,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        row_subset: &[u32],
        sampler: &mut ColumnSampler,
        capture_rows: bool,
    ) -> (RegTree, Vec<LeafRows>) {
        let root_stats = sum_rows(gpair, row_subset);
        let mut root_hist = zeroed(ghist.total_bins());
        self.backend.build(ghist, row_subset, gpair, &mut root_hist);

        let mut tree = RegTree::with_root(root_stats.hess as f32);
        let mut stats = vec![root_stats];
        let mut bounds = vec![Bounds::default()];
        let mut leaf_rows = Vec::new();
        let mut record_leaf = |node: LevelNode| {
            if capture_rows {
                leaf_rows.push(LeafRows {
                    node: node.nid,
                    rows: node.rows,
                });
            }
        };

        let mut level = vec![self.level_node(
            0,
            row_subset.to_vec(),
            root_hist,
            root_stats,
            Bounds::default(),
        )];
        let mut allowed: Option<InteractionState> = None;
        let depth_limit = self.params.max_depth;
        for depth in 0..depth_limit {
            let features: Vec<u32> = sampler
                .sample(depth)
                .into_iter()
                .filter(|&f| permits(allowed.as_ref(), f))
                .collect();
            let Some(split) = self.best_level_split(ghist, &level, &features) else {
                break;
            };
            let cuts = ghist.cuts();
            let feature = split.feature as usize;
            let (fs, _) = cuts.feature_bins(feature);
            let pos = split.pos(fs);
            let threshold = match pos {
                SplitPos::Bin(bin) => cuts.cut_value(bin),
                _ => BELOW_ALL_VALUES,
            };
            let dir = self.cons.dir(feature);

            // Expand in node order, so child ids stay breadth-first.
            let mut pending = Vec::with_capacity(level.len());
            for node in level {
                let Some(s) = self.node_split(ghist, &node, &split) else {
                    record_leaf(node);
                    continue;
                };
                let (lb, rb) =
                    child_bounds(node.bounds, dir, f64::from(s.w_left), f64::from(s.w_right));
                let (left_id, right_id) = tree.expand(
                    node.nid,
                    split.feature,
                    threshold,
                    split.missing_left,
                    s.w_left,
                    s.left.hess as f32,
                    s.w_right,
                    s.right.hess as f32,
                );
                tree.set_split_gain(node.nid, s.loss_chg);
                debug_assert_eq!(left_id, stats.len());
                stats.extend([s.left, s.right]);
                bounds.extend([lb, rb]);
                pending.push((node, s, [left_id, right_id], [lb, rb]));
            }
            allowed = next_allowed(
                allowed.as_ref(),
                split.feature,
                self.interaction_sets.as_deref(),
            );

            let terminal = depth + 1 == depth_limit;
            if terminal && !capture_rows {
                level = Vec::new();
                break;
            }
            let routing = BestSplit::numeric(
                0.0,
                split.feature,
                pos,
                split.missing_left,
                GradStats::default(),
                GradStats::default(),
                0.0,
                0.0,
            );
            let parallel = pending.len() > 1
                && pending.iter().map(|(n, ..)| n.rows.len()).sum::<usize>() >= PARALLEL_LEVEL_ROWS
                && rayon_available();
            let build = |(node, s, ids, cb): (LevelNode, NodeSplit, [usize; 2], [Bounds; 2])| {
                self.children(ghist, gpair, &routing, node, &s, ids, cb, terminal)
            };
            let children: Vec<[LevelNode; 2]> = if parallel {
                pending.into_par_iter().map(build).collect()
            } else {
                pending.into_iter().map(build).collect()
            };
            level = children.into_iter().flatten().collect();
        }
        for node in level {
            record_leaf(node);
        }

        finalize_leaf_values(&mut tree, &stats, &bounds, &self.reg);
        (tree, leaf_rows)
    }

    fn level_node(
        &self,
        nid: usize,
        rows: Vec<u32>,
        hist: Histogram,
        total: GradStats,
        bounds: Bounds,
    ) -> LevelNode {
        LevelNode {
            nid,
            rows,
            hist,
            total,
            bounds,
            root_gain: xgb_node_gain(total, &self.reg, bounds),
        }
    }

    /// Partition a split node's rows and, below the last level, build the
    /// smaller child's histogram and derive the sibling by subtraction.
    #[allow(clippy::too_many_arguments)]
    fn children(
        &self,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        routing: &BestSplit,
        node: LevelNode,
        split: &NodeSplit,
        [left_id, right_id]: [usize; 2],
        [lb, rb]: [Bounds; 2],
        terminal: bool,
    ) -> [LevelNode; 2] {
        let (left_rows, right_rows) = partition_rows(ghist, &node.rows, routing);
        let mut parent_hist = node.hist;
        let (left_hist, right_hist) = if terminal {
            (Vec::new(), Vec::new())
        } else if left_rows.len() <= right_rows.len() {
            let mut lh = zeroed(parent_hist.len());
            self.backend.build(ghist, &left_rows, gpair, &mut lh);
            subtract_in_place(&mut parent_hist, &lh);
            (lh, parent_hist)
        } else {
            let mut rh = zeroed(parent_hist.len());
            self.backend.build(ghist, &right_rows, gpair, &mut rh);
            subtract_in_place(&mut parent_hist, &rh);
            (parent_hist, rh)
        };
        [
            self.level_node(left_id, left_rows, left_hist, split.left, lb),
            self.level_node(right_id, right_rows, right_hist, split.right, rb),
        ]
    }

    /// Per-node `(loss_chg, w_left, w_right)` of a candidate, or `None` when
    /// the node would not take it (see the module docs).
    #[inline]
    fn node_gain(
        &self,
        node: &LevelNode,
        left: GradStats,
        right: GradStats,
        dir: i8,
    ) -> Option<(f32, f32, f32)> {
        let (loss_chg, wl, wr) =
            xgb_loss_chg(left, right, node.root_gain, &self.reg, node.bounds, dir)?;
        let gain = f64::from(loss_chg);
        (loss_chg.is_finite() && gain > K_RT_EPS && gain >= self.params.gamma)
            .then_some((loss_chg, wl, wr))
    }

    /// The best level-wide candidate over `features`, or `None` when no
    /// candidate splits any node. Features are scored independently (in
    /// parallel for large levels) and reduced in ascending feature order,
    /// keeping the earlier candidate on ties.
    fn best_level_split(
        &self,
        ghist: &GHistIndex,
        level: &[LevelNode],
        features: &[u32],
    ) -> Option<LevelSplit> {
        let score = |&f: &u32| self.score_feature(ghist, level, f);
        let per_feature: Vec<Option<LevelSplit>> =
            if level.len() * ghist.total_bins() >= PARALLEL_SCORE_BINS && rayon_available() {
                features.par_iter().map(score).collect()
            } else {
                features.iter().map(score).collect()
            };
        per_feature
            .into_iter()
            .flatten()
            .fold(None, |best: Option<LevelSplit>, cand| match best {
                Some(b) if b.score >= cand.score => Some(b),
                _ => Some(cand),
            })
    }

    /// Score every boundary of feature `f` across the level's nodes and return
    /// the best one. Candidate order within a feature is XGBoost's: forward
    /// boundaries ascending, then (when some node has missing values)
    /// backward boundaries descending.
    fn score_feature(&self, ghist: &GHistIndex, level: &[LevelNode], f: u32) -> Option<LevelSplit> {
        let (fs, fe) = ghist.cuts().feature_bins(f as usize);
        if fe <= fs + 1 {
            return None; // degenerate feature, no interior boundary
        }
        let n_bins = fe - fs;
        let dense = ghist.dense_stride().is_some();
        let dir = self.cons.dir(f as usize);
        let gamma = self.params.gamma;
        // `(Σ (loss_chg − gamma), nodes split)` per boundary.
        let mut forward = vec![(0.0f64, 0u32); n_bins];
        let mut backward = if dense {
            Vec::new()
        } else {
            vec![(0.0f64, 0u32); n_bins]
        };
        let mut node_forward: Vec<Option<f64>> = if dense {
            Vec::new()
        } else {
            vec![None; n_bins]
        };
        let mut any_missing = false;
        let add = |slot: &mut (f64, u32), term: f64| {
            slot.0 += term;
            slot.1 += 1;
        };

        for node in level {
            let hist = &node.hist[fs..fe];
            let total = node.total;
            let mut acc = GradStats::default();
            for (b, &bin) in hist.iter().enumerate() {
                acc.add(bin);
                let term = self
                    .node_gain(node, acc, total.sub(acc), dir)
                    .map(|(loss_chg, ..)| f64::from(loss_chg) - gamma);
                if let Some(term) = term {
                    add(&mut forward[b], term);
                }
                if !dense {
                    node_forward[b] = term;
                }
            }
            if dense {
                continue;
            }
            if acc == total {
                // No missing values here: the backward boundary at `b` is
                // the forward partition at `b - 1` (and nothing at `0`).
                for b in 1..n_bins {
                    if let Some(term) = node_forward[b - 1] {
                        add(&mut backward[b], term);
                    }
                }
            } else {
                any_missing = true;
                let mut suffix = GradStats::default();
                for b in (0..n_bins).rev() {
                    suffix.add(hist[b]);
                    if let Some((loss_chg, ..)) =
                        self.node_gain(node, total.sub(suffix), suffix, dir)
                    {
                        add(&mut backward[b], f64::from(loss_chg) - gamma);
                    }
                }
            }
        }

        let mut best: Option<LevelSplit> = None;
        let mut consider = |(score, splits): (f64, u32), offset: usize, missing_left: bool| {
            if splits > 0 && best.is_none_or(|b| score > b.score) {
                best = Some(LevelSplit {
                    feature: f,
                    offset,
                    missing_left,
                    score,
                });
            }
        };
        for (b, &slot) in forward.iter().enumerate() {
            consider(slot, b, false);
        }
        if any_missing {
            for (b, &slot) in backward.iter().enumerate().rev() {
                consider(slot, b, true);
            }
        }
        best
    }

    /// The level split as applied to `node`, recomputed with the same
    /// accumulation order as [`Self::score_feature`] so the decision matches
    /// the score exactly; `None` when the node stays a leaf.
    fn node_split(
        &self,
        ghist: &GHistIndex,
        node: &LevelNode,
        split: &LevelSplit,
    ) -> Option<NodeSplit> {
        let feature = split.feature as usize;
        let (fs, fe) = ghist.cuts().feature_bins(feature);
        let hist = &node.hist[fs..fe];
        let total = node.total;
        let prefix = |last: usize| {
            let mut acc = GradStats::default();
            for &bin in &hist[..=last] {
                acc.add(bin);
            }
            acc
        };
        let (left, right) = if split.missing_left {
            let has_missing = ghist.dense_stride().is_none() && prefix(hist.len() - 1) != total;
            if has_missing {
                let mut suffix = GradStats::default();
                for &bin in hist[split.offset..].iter().rev() {
                    suffix.add(bin);
                }
                (total.sub(suffix), suffix)
            } else if split.offset == 0 {
                return None; // every row goes right
            } else {
                let left = prefix(split.offset - 1);
                (left, total.sub(left))
            }
        } else {
            let left = prefix(split.offset);
            (left, total.sub(left))
        };
        let (loss_chg, w_left, w_right) =
            self.node_gain(node, left, right, self.cons.dir(feature))?;
        Some(NodeSplit {
            left,
            right,
            loss_chg,
            w_left,
            w_right,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::gp;
    use super::*;
    use crate::config::{GrowPolicy, Monotone};
    use crate::data::quantile::HistCuts;
    use crate::learner::train;
    use crate::tree::builder::{HistTreeBuilder, all_rows};

    /// Deterministic pseudo-random value in `[0, 1)`.
    fn unit(i: usize) -> f32 {
        let h = (i as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .rotate_left(17)
            ^ 0x2545_F491;
        (h.wrapping_mul(0xBF58_476D_1CE4_E5B9) >> 40) as f32 / (1u64 << 24) as f32
    }

    /// `n × n_features` features plus a nonlinear target. With `missing`,
    /// every seventh value of the first two features is `NaN`.
    fn synthetic(n: usize, n_features: usize, missing: bool) -> (DMatrix, Vec<f32>) {
        let mut x: Vec<f32> = (0..n * n_features).map(unit).collect();
        let y: Vec<f32> = x
            .chunks_exact(n_features)
            .map(|r| (3.0 * r[0]).sin() + 2.0 * r[1] * r[2] + if r[3] > 0.6 { 1.0 } else { 0.0 })
            .collect();
        if missing {
            for (i, v) in x.iter_mut().enumerate() {
                if i % n_features < 2 && (i / n_features) % 7 == 3 {
                    *v = f32::NAN;
                }
            }
        }
        let d = DMatrix::from_dense(&x, n, n_features).unwrap();
        (d, y)
    }

    fn binned(data: &DMatrix, max_bin: usize) -> GHistIndex {
        GHistIndex::from_dmatrix(data, HistCuts::from_dmatrix(data, max_bin))
    }

    /// Squared-error gradients at a zero margin.
    fn residual_gradients(y: &[f32]) -> Vec<GradPair> {
        y.iter().map(|&v| gp(-v, 1.0)).collect()
    }

    fn symmetric() -> crate::config::TrainingParamsBuilder {
        TrainingParams::builder()
            .tree_method(TreeMethod::Hist)
            .grow_policy(GrowPolicy::Symmetric)
    }

    fn grow(params: &TrainingParams, data: &DMatrix, gpair: &[GradPair]) -> RegTree {
        let ghist = binned(data, 64);
        HistTreeBuilder::new(params).build(
            &ghist,
            gpair,
            &all_rows(data.n_rows()),
            &mut ColumnSampler::all(data.n_cols()),
        )
    }

    /// Assert the tree is symmetric: node ids are breadth-first by depth and
    /// every internal node at depth `d` carries level `d`'s split. Returns the
    /// `(feature, threshold bits, default_left)` of each level.
    fn levels(tree: &RegTree) -> Vec<(u32, u32, bool)> {
        let mut levels = Vec::new();
        let mut frontier = vec![0usize];
        while !frontier.is_empty() {
            let mut next = Vec::new();
            let mut level = None;
            for id in frontier {
                let n = tree.node(id);
                if n.is_leaf() {
                    continue;
                }
                let split = (n.split_feature, n.split_cond.to_bits(), n.default_left);
                assert_eq!(*level.get_or_insert(split), split, "node {id} breaks level");
                next.extend([n.left as usize, n.right as usize]);
            }
            levels.extend(level);
            frontier = next;
        }
        levels
    }

    #[test]
    fn full_tree_has_one_split_per_level() {
        // Five binary features covering every combination, and a target that
        // weighs them differently: each level splits on a new feature, and
        // every node of every level is non-constant. Without `lambda` such a
        // split always has positive gain, so no node collapses.
        let n = 32 * 20;
        let mut x = Vec::new();
        let mut gpair = Vec::new();
        for i in 0..n {
            let row: Vec<f32> = (0..5).map(|b| ((i >> b) & 1) as f32).collect();
            let y: f32 = row
                .iter()
                .zip([1.0, 2.0, 4.0, 8.0, 16.0])
                .map(|(v, w)| v * w)
                .sum();
            x.extend(row);
            gpair.push(gp(-y, 1.0));
        }
        let data = DMatrix::from_dense(&x, n, 5).unwrap();
        let params = symmetric().max_depth(5).lambda(0.0).build().unwrap();
        let tree = grow(&params, &data, &gpair);
        let lv = levels(&tree);
        let mut features: Vec<u32> = lv.iter().map(|l| l.0).collect();
        // The largest weight separates most, so it is taken first.
        assert_eq!(features, [4, 3, 2, 1, 0]);
        features.dedup();
        assert_eq!(features.len(), 5);
        assert_eq!(tree.num_leaves(), 32);
        assert_eq!(tree.num_nodes(), 63);
    }

    #[test]
    fn level_split_maximizes_summed_gain_not_each_node() {
        // Root: x0 halves the data. Left half: the target follows x1
        // strongly. Right half: it follows x2 weakly and ignores x1.
        // Depthwise picks x1 on the left and x2 on the right; the symmetric
        // level takes x1 (larger total) for both nodes.
        let n = 400;
        let mut x = Vec::new();
        let mut gpair = Vec::new();
        for i in 0..n {
            let (x0, x1, x2) = ((i % 2) as f32, ((i / 2) % 2) as f32, ((i / 4) % 2) as f32);
            x.extend([x0, x1, x2]);
            let y = if x0 == 0.0 { 5.0 * x1 } else { x2 - 2.5 };
            gpair.push(gp(-y, 1.0));
        }
        let data = DMatrix::from_dense(&x, n, 3).unwrap();
        let depthwise = TrainingParams::builder()
            .tree_method(TreeMethod::Hist)
            .max_depth(2)
            .build()
            .unwrap();
        let dw = grow(&depthwise, &data, &gpair);
        let (l, r) = (dw.node(1), dw.node(2));
        assert_eq!((l.split_feature, r.split_feature), (1, 2));

        let params = symmetric().max_depth(2).build().unwrap();
        let tree = grow(&params, &data, &gpair);
        let lv = levels(&tree);
        assert_eq!(lv.iter().map(|l| l.0).collect::<Vec<_>>(), [0, 1]);
        // x1 carries no signal on the right, so that node stays a leaf.
        assert!(!tree.node(1).is_leaf());
        assert!(tree.node(2).is_leaf());
    }

    #[test]
    fn nodes_failing_min_child_weight_or_gamma_stay_leaves() {
        let (data, y) = synthetic(600, 5, false);
        let gpair = residual_gradients(&y);
        let full = grow(
            &symmetric()
                .max_depth(6)
                .min_child_weight(0.0)
                .build()
                .unwrap(),
            &data,
            &gpair,
        );
        for (mcw, gamma) in [(40.0, 0.0), (0.0, 2.0)] {
            let params = symmetric()
                .max_depth(6)
                .min_child_weight(mcw)
                .gamma(gamma)
                .build()
                .unwrap();
            let tree = grow(&params, &data, &gpair);
            assert!(!levels(&tree).is_empty());
            assert!(tree.num_leaves() > 1 && tree.num_leaves() < full.num_leaves());
            for n in tree.nodes() {
                if n.is_leaf() {
                    assert!(f64::from(n.sum_hess) >= mcw);
                } else {
                    assert!(f64::from(n.split_gain) >= gamma);
                }
            }
        }
        let stump = grow(&symmetric().gamma(1e9).build().unwrap(), &data, &gpair);
        assert_eq!(stump.num_nodes(), 1);
    }

    #[test]
    fn missing_values_follow_one_direction_per_level_and_leaf_rows_match_routing() {
        let (data, y) = synthetic(3000, 5, true);
        let ghist = binned(&data, 32);
        assert!(ghist.dense_stride().is_none());
        let params = symmetric().max_depth(4).build().unwrap();
        let (tree, leaf_rows) = HistTreeBuilder::new(&params).build_with_leaf_rows(
            &ghist,
            &residual_gradients(&y),
            &all_rows(data.n_rows()),
            &mut ColumnSampler::all(data.n_cols()),
        );
        assert_eq!(levels(&tree).len(), 4);
        assert_eq!(leaf_rows.iter().map(|l| l.rows.len()).sum::<usize>(), 3000);
        assert_eq!(leaf_rows.len(), tree.num_leaves());
        for leaf in &leaf_rows {
            assert!(leaf.rows.is_sorted());
            for &r in &leaf.rows {
                let routed = tree.leaf_id_with(|f| data.get(r as usize, f as usize));
                assert_eq!(routed, leaf.node);
            }
        }
    }

    #[test]
    fn monotone_constraint_holds() {
        let (data, y) = synthetic(2000, 4, false);
        let params = symmetric()
            .max_depth(4)
            .monotone_constraints(vec![Monotone::Decreasing])
            .build()
            .unwrap();
        let model = train(&params, &data.clone().with_labels(&y).unwrap(), 20).unwrap();
        for i in 0..50 {
            let base: Vec<f32> = (1..4).map(|f| unit(10_000 + 4 * i + f)).collect();
            let preds: Vec<f32> = (0..=40)
                .map(|s| {
                    let mut row = vec![s as f32 / 40.0];
                    row.extend(&base);
                    let d = DMatrix::from_dense(&row, 1, 4).unwrap();
                    model.predict(&d).unwrap()[0]
                })
                .collect();
            assert!(preds.windows(2).all(|w| w[1] <= w[0]), "{preds:?}");
        }
    }

    #[test]
    fn interaction_constraints_bound_every_path() {
        let (data, y) = synthetic(2000, 6, false);
        let params = symmetric()
            .max_depth(4)
            .interaction_constraints(vec![vec![0, 1, 2], vec![3, 4]])
            .build()
            .unwrap();
        let model = train(&params, &data.with_labels(&y).unwrap(), 10).unwrap();
        for tree in model.trees() {
            let features: Vec<u32> = levels(tree).iter().map(|l| l.0).collect();
            let within = |g: &[u32]| features.iter().all(|f| g.contains(f));
            assert!(
                within(&[0, 1, 2]) || within(&[3, 4]) || features.len() <= 1,
                "{features:?}"
            );
        }
    }

    #[test]
    fn serial_and_parallel_growth_agree() {
        let (data, y) = synthetic(20_000, 8, true);
        let data = data.with_labels(&y).unwrap();
        let params = symmetric()
            .max_depth(6)
            .colsample_bylevel(0.75)
            .build()
            .unwrap();
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| train(&params, &data, 5).unwrap())
        };
        let (serial, parallel) = (run(1), run(8));
        assert_eq!(serial.trees(), parallel.trees());
        assert_eq!(serial.trees(), run(8).trees());
    }

    #[test]
    fn quality_is_close_to_depthwise() {
        let (dtrain, ytrain) = synthetic(8000, 6, false);
        let (dtest, ytest) = synthetic(2000, 6, false);
        let dtrain = dtrain.with_labels(&ytrain).unwrap();
        let rmse = |policy| {
            let params = TrainingParams::builder()
                .tree_method(TreeMethod::Hist)
                .grow_policy(policy)
                .max_depth(6)
                .eta(0.1)
                .build()
                .unwrap();
            let pred = train(&params, &dtrain, 200)
                .unwrap()
                .predict(&dtest)
                .unwrap();
            let se: f32 = pred.iter().zip(&ytest).map(|(p, y)| (p - y).powi(2)).sum();
            (se / ytest.len() as f32).sqrt()
        };
        let mean = ytest.iter().sum::<f32>() / ytest.len() as f32;
        let std =
            (ytest.iter().map(|y| (y - mean).powi(2)).sum::<f32>() / ytest.len() as f32).sqrt();
        let (sym, dw) = (rmse(GrowPolicy::Symmetric), rmse(GrowPolicy::DepthWise));
        assert!(sym < 0.2 * std, "symmetric rmse {sym} vs target std {std}");
        assert!(sym < 1.5 * dw, "symmetric rmse {sym} vs depthwise {dw}");
    }

    #[test]
    fn unsupported_configurations_are_rejected() {
        let (data, y) = synthetic(200, 4, false);
        let data = data.with_labels(&y).unwrap();
        for depth in [0, crate::config::MAX_SYMMETRIC_DEPTH + 1] {
            assert!(symmetric().max_depth(depth).build().is_err());
        }
        assert!(symmetric().max_leaves(8).build().is_err());
        let exact = symmetric().tree_method(TreeMethod::Exact).build().unwrap();
        assert!(train(&exact, &data, 1).is_err());
        let codes: Vec<f32> = (0..200).map(|i| (i % 3) as f32).collect();
        let categorical = DMatrix::from_dense(&codes, 200, 1)
            .unwrap()
            .with_labels(&y)
            .unwrap()
            .with_feature_types(&[FeatureType::Categorical])
            .unwrap();
        let err = train(&symmetric().build().unwrap(), &categorical, 1).unwrap_err();
        assert!(err.to_string().contains("categorical"), "{err}");
        let approx = symmetric().tree_method(TreeMethod::Approx).build().unwrap();
        let model = train(&approx, &data, 3).unwrap();
        assert!(model.trees().iter().all(|t| !levels(t).is_empty()));
    }
}
