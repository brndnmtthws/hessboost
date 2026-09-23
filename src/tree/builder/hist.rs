//! Histogram-based tree construction (XGBoost's `tree_method=hist`).
//!
//! Features are pre-binned once ([`GHistIndex`]). Growing a node reduces to
//! scanning its per-bin gradient histogram. Sibling histograms are obtained by
//! subtraction (`sibling = parent − smaller_child`), so only the smaller child
//! is ever built directly. Supports both `depthwise` and `lossguide` growth.

use super::{
    BELOW_ALL_VALUES, BestSplit, InteractionState, SplitPos, build_interaction_sets,
    finalize_leaf_values, next_allowed, permits, sum_rows, sweep_categorical, xgb_loss_chg,
    xgb_node_gain, xgb_update,
};
use crate::config::{GrowPolicy, TrainingParams};
use crate::data::ghist::{Bins, GHistIndex};
use crate::data::quantile::HistCuts;
use crate::objective::GradPair;
use crate::tree::constraints::{Bounds, MonotoneConstraints, child_bounds};
use crate::tree::gain::{GradStats, RegParams};
use crate::tree::hist::quantized::QuantNode;
use crate::tree::hist::{
    BinIndex, CpuBackend, Histogram, HistogramBackend, subtract_in_place, zeroed,
};
use crate::tree::regtree::RegTree;
use crate::tree::sampler::ColumnSampler;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Nodes with at least this many rows evaluate their two children's splits
/// concurrently. Smaller nodes appear in frontiers wide enough to keep the
/// pool busy, and their evaluation is too short to be worth a fork.
const PARALLEL_EVALUATE_ROWS: usize = 16_384;

/// Combined frontier rows at which depthwise growth builds a level's child
/// histograms concurrently. Below this, the fork costs more than the scan.
const PARALLEL_FRONTIER_ROWS: usize = 4096;

/// Whether the rayon pool has more than one thread, so parallelism can pay off.
pub(super) fn rayon_available() -> bool {
    rayon::current_num_threads() > 1
}

/// Training rows that reached a leaf during tree construction.
pub(crate) struct LeafRows {
    pub node: usize,
    pub rows: Vec<u32>,
}

/// A node awaiting or undergoing expansion.
struct NodeEntry {
    nid: usize,
    depth: usize,
    rows: Vec<u32>,
    hist: Histogram,
    best: BestSplit,
    bounds: Bounds,
    /// Features permitted for splits under this node given the interaction
    /// constraints and the split features on the path from the root. `None`
    /// means "all features allowed" (the root, and the inactive case).
    allowed: Option<InteractionState>,
    /// Quantized histogram (`use_quantized_grad`); `hist` then holds its
    /// dequantized copy for split evaluation.
    quant: Option<QuantNode>,
}

/// Tree expansion and sampling happen in node order, so the expensive row and
/// histogram work can then run independently for every split at a depth.
struct PendingSplit {
    entry: NodeEntry,
    left_id: usize,
    right_id: usize,
    left_bounds: Bounds,
    right_bounds: Bounds,
    left_features: Vec<u32>,
    right_features: Vec<u32>,
}

// Ordering for the loss-guided priority queue (max-heap on loss change).
impl PartialEq for NodeEntry {
    fn eq(&self, other: &Self) -> bool {
        self.best.loss_chg == other.best.loss_chg
    }
}
impl Eq for NodeEntry {}
impl PartialOrd for NodeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for NodeEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.best.loss_chg.total_cmp(&other.best.loss_chg)
    }
}

/// Histogram tree builder.
pub struct HistTreeBuilder<'a> {
    params: &'a TrainingParams,
    reg: RegParams,
    cons: MonotoneConstraints,
    /// Per-feature interaction sets: feature -> sorted set of features it may be
    /// combined with on a path (its interaction set). `None` means interaction
    /// constraints are inactive (no filtering). An unlisted feature may only
    /// interact with itself.
    interaction_sets: Option<Vec<Vec<u32>>>,
    backend: CpuBackend,
    /// Stream of the stochastic gradient rounding (`use_quantized_grad`).
    rounding_seed: u64,
}

impl<'a> HistTreeBuilder<'a> {
    /// Create a builder bound to a training configuration.
    pub fn new(params: &'a TrainingParams) -> Self {
        HistTreeBuilder {
            params,
            reg: RegParams::from_params(params),
            cons: MonotoneConstraints::from_params(&params.monotone_constraints),
            interaction_sets: build_interaction_sets(&params.interaction_constraints),
            backend: CpuBackend,
            rounding_seed: 0,
        }
    }

    /// Seed the stochastic rounding of quantized training
    /// (`use_quantized_grad`). The trainer passes a distinct seed per round
    /// and output so rounding noise is independent across trees.
    #[must_use]
    pub(crate) fn with_rounding_seed(mut self, seed: u64) -> Self {
        self.rounding_seed = seed;
        self
    }

    /// Grow one tree from the binned dataset.
    ///
    /// * `ghist`: the binned dataset (built once, reused across rounds).
    /// * `gpair`: per-row gradient/Hessian (length = dataset rows).
    /// * `row_subset`: sampled rows for this tree.
    /// * `sampler`: per-tree column sampler. A fresh subset is drawn per node.
    pub fn build(
        &self,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        row_subset: &[u32],
        sampler: &mut ColumnSampler,
    ) -> RegTree {
        self.build_inner(ghist, gpair, row_subset, sampler, false).0
    }

    /// Keep the final row partitions so training can update margins without
    /// traversing the tree again. Used for depthwise trees without row sampling.
    pub(crate) fn build_with_leaf_rows(
        &self,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        row_subset: &[u32],
        sampler: &mut ColumnSampler,
    ) -> (RegTree, Vec<LeafRows>) {
        debug_assert_eq!(self.params.grow_policy, GrowPolicy::DepthWise);
        self.build_inner(ghist, gpair, row_subset, sampler, true)
    }

    fn build_inner(
        &self,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        row_subset: &[u32],
        sampler: &mut ColumnSampler,
        capture_rows: bool,
    ) -> (RegTree, Vec<LeafRows>) {
        let total_bins = ghist.total_bins();
        // Leaf renewal recomputes leaf values from full-precision sums, which
        // needs every leaf's rows.
        let renew = self.params.use_quantized_grad && self.params.quant_train_renew_leaf;

        let (root_stats, root_hist, root_quant) = if self.params.use_quantized_grad {
            let (quant, stats, hist) =
                QuantNode::root(ghist, gpair, row_subset, self.params, self.rounding_seed);
            (stats, hist, Some(quant))
        } else {
            let root_stats = sum_rows(gpair, row_subset);
            let mut root_hist = zeroed(total_bins);
            self.backend.build(ghist, row_subset, gpair, &mut root_hist);
            (root_stats, root_hist, None)
        };

        let mut tree = RegTree::with_root(root_stats.hess as f32);
        let mut store = NodeStore {
            stats: vec![root_stats],
            bounds: vec![Bounds::default()],
            leaf_rows: (capture_rows || renew).then(Vec::new),
        };

        // Per-node column sampling (bylevel ∘ bynode) draws a fresh subset here.
        let root_feats = sampler.sample();
        let best = self.evaluate(
            ghist,
            &root_hist,
            root_stats,
            &root_feats,
            Bounds::default(),
            None,
        );
        let root = NodeEntry {
            nid: 0,
            depth: 0,
            rows: row_subset.to_vec(),
            hist: root_hist,
            best,
            bounds: Bounds::default(),
            allowed: None,
            quant: root_quant,
        };

        match self.params.grow_policy {
            GrowPolicy::DepthWise => {
                self.grow_depthwise(&mut tree, &mut store, ghist, gpair, sampler, root);
            }
            GrowPolicy::LossGuide => {
                self.grow_lossguide(&mut tree, &mut store, ghist, gpair, sampler, root);
            }
        }

        if renew && let Some(leaves) = &store.leaf_rows {
            for leaf in leaves {
                store.stats[leaf.node] = sum_rows(gpair, &leaf.rows);
            }
        }
        // Finalize leaf weights (respecting each leaf's monotone bounds).
        finalize_leaf_values(&mut tree, &store.stats, &store.bounds, &self.reg);
        let leaf_rows = if capture_rows {
            store.leaf_rows.unwrap_or_default()
        } else {
            Vec::new()
        };
        (tree, leaf_rows)
    }

    fn depth_limit(&self) -> usize {
        if self.params.max_depth == 0 {
            usize::MAX
        } else {
            self.params.max_depth
        }
    }

    fn grow_depthwise(
        &self,
        tree: &mut RegTree,
        store: &mut NodeStore,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        sampler: &mut ColumnSampler,
        root: NodeEntry,
    ) {
        let limit = self.depth_limit();
        let mut frontier = vec![root];
        let mut depth = 0;
        while depth < limit && !frontier.is_empty() {
            let parallel = frontier.len() > 1
                && frontier.iter().map(|entry| entry.rows.len()).sum::<usize>()
                    >= PARALLEL_FRONTIER_ROWS
                && rayon_available();
            let mut pending = Vec::with_capacity(frontier.len());
            for entry in frontier.drain(..) {
                if self.valid(&entry.best) {
                    if let Some(split) =
                        self.prepare_split(tree, store, ghist.cuts(), sampler, entry)
                    {
                        pending.push(split);
                    }
                } else {
                    store.record_leaf(entry);
                }
            }
            let build = |split| self.build_children(ghist, gpair, split);
            let children: Vec<_> = if parallel {
                pending.into_par_iter().map(build).collect()
            } else {
                pending.into_iter().map(build).collect()
            };
            frontier = children
                .into_iter()
                .flat_map(|(left, right)| [left, right])
                .collect();
            depth += 1;
        }
        for entry in frontier {
            store.record_leaf(entry);
        }
    }

    fn grow_lossguide(
        &self,
        tree: &mut RegTree,
        store: &mut NodeStore,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        sampler: &mut ColumnSampler,
        root: NodeEntry,
    ) {
        let limit = self.depth_limit();
        let max_leaves = if self.params.max_leaves == 0 {
            usize::MAX
        } else {
            self.params.max_leaves
        };
        let mut heap = BinaryHeap::new();
        heap.push(root);
        let mut n_leaves = 1usize;
        while let Some(entry) = heap.pop() {
            if n_leaves >= max_leaves {
                store.record_leaf(entry);
                break;
            }
            if entry.depth >= limit || !self.valid(&entry.best) {
                store.record_leaf(entry);
                continue; // permanent leaf
            }
            let children = self
                .prepare_split(tree, store, ghist.cuts(), sampler, entry)
                .map(|split| self.build_children(ghist, gpair, split));
            n_leaves += 1; // one leaf became two
            if let Some((l, r)) = children {
                heap.push(l);
                heap.push(r);
            }
        }
        for entry in heap {
            store.record_leaf(entry);
        }
    }

    /// Whether a node's best split should be taken.
    fn valid(&self, best: &BestSplit) -> bool {
        best.valid(self.params.gamma, self.reg.min_child_weight)
    }

    /// Expand a node and draw child features in traversal order. Children at the
    /// depth limit need only stored statistics to finalize their leaf weights.
    fn prepare_split(
        &self,
        tree: &mut RegTree,
        store: &mut NodeStore,
        cuts: &HistCuts,
        sampler: &mut ColumnSampler,
        entry: NodeEntry,
    ) -> Option<PendingSplit> {
        let b = &entry.best;

        // Monotone child bounds derived from the (bounded) child weights.
        let dir = self.cons.dir(b.feature as usize);
        let (lb_bounds, rb_bounds) = child_bounds(entry.bounds, dir, b.w_left, b.w_right);

        let (left_id, right_id) = if b.is_categorical {
            tree.expand_categorical(
                entry.nid,
                b.feature,
                &b.cat_left,
                b.default_left,
                b.w_left as f32,
                b.left.hess as f32,
                b.w_right as f32,
                b.right.hess as f32,
            )
        } else {
            let threshold = match b.split_bin {
                Some(bin) => cuts.cut_value(bin),
                None => BELOW_ALL_VALUES,
            };
            tree.expand(
                entry.nid,
                b.feature,
                threshold,
                b.default_left,
                b.w_left as f32,
                b.left.hess as f32,
                b.w_right as f32,
                b.right.hess as f32,
            )
        };
        tree.set_split_gain(entry.nid, b.loss_chg as f32);
        debug_assert_eq!(left_id, store.stats.len());
        store.push(b.left, lb_bounds);
        store.push(b.right, rb_bounds);

        if self.params.grow_policy == GrowPolicy::DepthWise
            && entry.depth + 1 >= self.depth_limit()
            && store.leaf_rows.is_none()
        {
            // Preserve the draws for these two nodes, including when callers
            // reuse the sampler. Their rows, histograms and candidate splits
            // cannot affect this tree, and leaf weights use the stored stats.
            sampler.sample();
            sampler.sample();
            return None;
        }

        Some(PendingSplit {
            entry,
            left_id,
            right_id,
            left_bounds: lb_bounds,
            right_bounds: rb_bounds,
            left_features: sampler.sample(),
            right_features: sampler.sample(),
        })
    }

    fn build_children(
        &self,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        split: PendingSplit,
    ) -> (NodeEntry, NodeEntry) {
        let PendingSplit {
            entry,
            left_id,
            right_id,
            left_bounds: lb_bounds,
            right_bounds: rb_bounds,
            left_features,
            right_features,
        } = split;
        let NodeEntry {
            depth: parent_depth,
            rows: parent_rows,
            hist: mut parent_hist,
            best,
            allowed: parent_allowed,
            quant: parent_quant,
            ..
        } = entry;
        let b = &best;

        let (left_rows, right_rows) = partition_rows(ghist, &parent_rows, b);
        drop(parent_rows);

        let terminal = self.params.grow_policy == GrowPolicy::DepthWise
            && parent_depth + 1 >= self.depth_limit();
        let total_bins = parent_hist.len();
        let (mut left_quant, mut right_quant) = (None, None);
        // Build the smaller child directly; derive the sibling by subtracting it
        // from the parent histogram in place. The parent's buffer is dead after
        // this node expands, so the sibling reuses it without a new allocation.
        let (left_hist, right_hist) = if terminal {
            (Vec::new(), Vec::new())
        } else if let Some(quant) = parent_quant {
            let ((lq, lh), (rq, rh)) = quant.children(ghist, &left_rows, &right_rows, parent_hist);
            (left_quant, right_quant) = (Some(lq), Some(rq));
            (lh, rh)
        } else if left_rows.len() <= right_rows.len() {
            let mut lh = zeroed(total_bins);
            self.backend.build(ghist, &left_rows, gpair, &mut lh);
            subtract_in_place(&mut parent_hist, &lh);
            (lh, parent_hist)
        } else {
            let mut rh = zeroed(total_bins);
            self.backend.build(ghist, &right_rows, gpair, &mut rh);
            subtract_in_place(&mut parent_hist, &rh);
            (parent_hist, rh)
        };

        // Both children share the state derived from the complete updated path:
        // path features plus groups containing every feature on that path.
        let child_allowed = next_allowed(
            parent_allowed.as_ref(),
            b.feature,
            self.interaction_sets.as_deref(),
        );

        // The children's split searches are independent; near the root, where
        // the frontier holds too few nodes to occupy the pool, running them
        // side by side halves the serial evaluation time. Each search keeps
        // its sequential candidate order, so the chosen split is identical.
        let (left_best, right_best) = if terminal {
            (BestSplit::none(), BestSplit::none())
        } else {
            let allowed = child_allowed.as_ref();
            let left = || {
                self.evaluate(
                    ghist,
                    &left_hist,
                    b.left,
                    &left_features,
                    lb_bounds,
                    allowed,
                )
            };
            let right = || {
                self.evaluate(
                    ghist,
                    &right_hist,
                    b.right,
                    &right_features,
                    rb_bounds,
                    allowed,
                )
            };
            if left_rows.len() + right_rows.len() >= PARALLEL_EVALUATE_ROWS && rayon_available() {
                rayon::join(left, right)
            } else {
                (left(), right())
            }
        };

        let left = NodeEntry {
            nid: left_id,
            depth: parent_depth + 1,
            rows: left_rows,
            hist: left_hist,
            best: left_best,
            bounds: lb_bounds,
            allowed: child_allowed.clone(),
            quant: left_quant,
        };
        let right = NodeEntry {
            nid: right_id,
            depth: parent_depth + 1,
            rows: right_rows,
            hist: right_hist,
            best: right_best,
            bounds: rb_bounds,
            allowed: child_allowed,
            quant: right_quant,
        };
        (left, right)
    }

    /// Find the best split for a node from its histogram, enumerating each
    /// sampled feature's bins as XGBoost's histogram evaluator does: a forward
    /// pass over every bin boundary (missing values right, including the last
    /// boundary that isolates the missing mass) and, only when the feature has
    /// missing values in this node, a backward pass (missing values left, down
    /// to the endpoint that routes only the missing mass left).
    /// Candidates are scored and compared with XGBoost's `f32` arithmetic and
    /// tie rule, so near-equal gains resolve the same way. Monotone bounds are
    /// honored through the bounded child weights.
    fn evaluate(
        &self,
        ghist: &GHistIndex,
        hist: &[GradStats],
        total: GradStats,
        feature_subset: &[u32],
        bounds: Bounds,
        allowed: Option<&InteractionState>,
    ) -> BestSplit {
        let cuts = ghist.cuts();
        let mut best = BestSplit::none();
        // A dense index has no missing entries: every feature's bins sum to
        // `total`, so the missing direction is never distinct and the
        // per-feature sums need not be computed.
        let dense = ghist.dense_stride().is_some();

        // Restrict the sampled features to those permitted by the interaction
        // constraints for this node. `allowed` is a sorted set; `None` means all
        // features are allowed (constraints inactive or unconstrained path).
        let filtered: Vec<u32>;
        let feature_subset: &[u32] = match allowed {
            Some(_) => {
                filtered = feature_subset
                    .iter()
                    .copied()
                    .filter(|&f| permits(allowed, f))
                    .collect();
                &filtered
            }
            None => feature_subset,
        };
        let constrained = self.cons.is_active();
        let root_gain = xgb_node_gain(total, &self.reg, bounds);

        for &f in feature_subset {
            let (fs, fe) = cuts.feature_bins(f as usize);
            if fe <= fs + 1 {
                continue; // degenerate feature, no interior boundary
            }
            let dir = self.cons.dir(f as usize);

            if cuts.is_categorical(f as usize) {
                // Only non-empty category bins can move; the sweep sorts them
                // by grad/hess ratio.
                let mut cats: Vec<(u32, GradStats)> = (fs..fe)
                    .filter(|&i| hist[i].hess > 0.0)
                    .map(|i| (cuts.cut_value(i) as u32, hist[i]))
                    .collect();
                sweep_categorical(
                    &mut best,
                    &mut cats,
                    total,
                    f64::from(root_gain),
                    bounds,
                    dir,
                    constrained,
                    &self.reg,
                    f,
                );
                continue;
            }

            let mut acc = GradStats::default();
            for (offset, &bin) in hist[fs..fe].iter().enumerate() {
                let i = fs + offset;
                acc.add(bin);
                let right = total.sub(acc);
                if let Some((loss_chg, wl, wr)) =
                    xgb_loss_chg(acc, right, root_gain, &self.reg, bounds, dir)
                {
                    xgb_update(
                        &mut best,
                        loss_chg,
                        f,
                        SplitPos::Bin(i),
                        false,
                        acc,
                        right,
                        wl,
                        wr,
                    );
                }
            }
            // Whether this feature has missing values in the node: XGBoost
            // compares the forward pass's final sum with the node statistics
            // exactly (`SplitContainsMissingValues`). A dense index never has
            // missing entries.
            if dense || acc == total {
                continue;
            }
            // Backward pass: bins `>= i` right, the rest (and missing) left.
            // The last candidate (`i == fs`, XGBoost's `NumericBinLowerBound`
            // at the feature's first bin) puts only the missing mass left. Its
            // children are the forward pass's last boundary swapped, so it is
            // distinct under a monotone constraint: the direction can reject
            // one orientation and accept the other. Without constraints the
            // gains tie and the earlier (forward) candidate is kept.
            let mut suffix = GradStats::default();
            for i in (fs..fe).rev() {
                suffix.add(hist[i]);
                let left = total.sub(suffix);
                if let Some((loss_chg, wl, wr)) =
                    xgb_loss_chg(left, suffix, root_gain, &self.reg, bounds, dir)
                {
                    let pos = if i == fs {
                        SplitPos::BelowBins
                    } else {
                        SplitPos::Bin(i - 1)
                    };
                    xgb_update(&mut best, loss_chg, f, pos, true, left, suffix, wl, wr);
                }
            }
        }
        best
    }
}

/// Split `rows` (kept in order) into the rows routed left and right by `best`.
fn partition_rows(ghist: &GHistIndex, rows: &[u32], best: &BestSplit) -> (Vec<u32>, Vec<u32>) {
    let cuts = ghist.cuts();
    let feature = best.feature as usize;
    if let (Some(columns), false) = (ghist.column_bins(), best.is_categorical) {
        let Some(split_bin) = best.split_bin else {
            // Missing-only-left split: a dense index has no missing rows.
            return (Vec::new(), rows.to_vec());
        };
        let n_rows = ghist.n_rows();
        return match columns {
            Bins::U16(bins) => route_dense(rows, &bins[feature * n_rows..][..n_rows], split_bin),
            Bins::U32(bins) => route_dense(rows, &bins[feature * n_rows..][..n_rows], split_bin),
        };
    }

    let (fs, fe) = cuts.feature_bins(feature);
    let mut left_rows = Vec::with_capacity(rows.len());
    let mut right_rows = Vec::with_capacity(rows.len());
    for &r in rows {
        let go_left = match ghist.feature_bin_at(r as usize, feature, fs, fe) {
            Some(bin) => {
                if best.is_categorical {
                    let cv = cuts.cut_value(bin as usize) as u32;
                    best.cat_left.contains(&cv)
                } else {
                    best.split_bin.is_some_and(|s| bin as usize <= s)
                }
            }
            None => best.default_left,
        };
        if go_left {
            left_rows.push(r);
        } else {
            right_rows.push(r);
        }
    }
    (left_rows, right_rows)
}

/// Rows per parallel partition chunk. Large nodes near the root are routed in
/// row-order chunks whose halves are concatenated, so the output order matches
/// the sequential loop exactly.
const PARTITION_CHUNK_ROWS: usize = 16_384;

/// Partition a dense index on a numeric split using the split feature's column
/// (`column[r]` is row `r`'s bin). Rows ascend, so the column is read as a
/// monotone stream the hardware prefetcher follows. Every row is written to
/// both output slots and only the matching length advances, keeping the loop
/// free of data-dependent branches. The outputs are written into spare
/// capacity, so neither buffer is zero-filled first.
fn route_dense<B: BinIndex>(rows: &[u32], column: &[B], split_bin: usize) -> (Vec<u32>, Vec<u32>) {
    let route = |rows: &[u32]| {
        let n = rows.len();
        let mut left: Vec<u32> = Vec::with_capacity(n);
        let mut right: Vec<u32> = Vec::with_capacity(n);
        let (mut nl, mut nr) = (0usize, 0usize);
        {
            let (lp, rp) = (left.spare_capacity_mut(), right.spare_capacity_mut());
            for &r in rows {
                let go_left = column[r as usize].index() <= split_bin;
                lp[nl].write(r);
                rp[nr].write(r);
                nl += usize::from(go_left);
                nr += usize::from(!go_left);
            }
        }
        // SAFETY: `nl + nr == n` and each side's slot `k` was written at the
        // iteration where its length was `k`, so `left[..nl]` and
        // `right[..nr]` are initialized and within the reserved capacity.
        unsafe {
            left.set_len(nl);
            right.set_len(nr);
        }
        (left, right)
    };
    if rows.len() < 2 * PARTITION_CHUNK_ROWS || !rayon_available() {
        return route(rows);
    }
    let chunks: Vec<(Vec<u32>, Vec<u32>)> =
        rows.par_chunks(PARTITION_CHUNK_ROWS).map(route).collect();
    let mut left = Vec::with_capacity(chunks.iter().map(|(l, _)| l.len()).sum());
    let mut right = Vec::with_capacity(chunks.iter().map(|(_, r)| r.len()).sum());
    for (l, r) in chunks {
        left.extend_from_slice(&l);
        right.extend_from_slice(&r);
    }
    (left, right)
}

/// Per-node statistics and monotone bounds, indexed by node id.
struct NodeStore {
    stats: Vec<GradStats>,
    bounds: Vec<Bounds>,
    leaf_rows: Option<Vec<LeafRows>>,
}

impl NodeStore {
    fn record_leaf(&mut self, entry: NodeEntry) {
        if let Some(leaves) = &mut self.leaf_rows {
            leaves.push(LeafRows {
                node: entry.nid,
                rows: entry.rows,
            });
        }
    }

    #[inline]
    fn push(&mut self, stats: GradStats, bounds: Bounds) {
        self.stats.push(stats);
        self.bounds.push(bounds);
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{gp, monotone_v_shape_data};
    use super::*;
    use crate::config::TrainingParams;
    use crate::data::DMatrix;
    use crate::tree::builder::all_rows;

    fn binned(data: &DMatrix, max_bin: usize) -> GHistIndex {
        let cuts = HistCuts::from_dmatrix(data, max_bin);
        GHistIndex::from_dmatrix(data, cuts)
    }

    #[test]
    fn splits_on_separating_feature() {
        let x = vec![0.0f32, 0.0, 1.0, 1.0];
        let data = DMatrix::from_dense(&x, 4, 1).unwrap();
        let ghist = binned(&data, 256);
        let gpair = vec![gp(1.0, 1.0), gp(1.0, 1.0), gp(-1.0, 1.0), gp(-1.0, 1.0)];
        let params = TrainingParams::builder()
            .max_depth(1)
            .lambda(0.0)
            .min_child_weight(0.0)
            .gamma(0.0)
            .build()
            .unwrap();
        let tree = HistTreeBuilder::new(&params).build(
            &ghist,
            &gpair,
            &all_rows(4),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );
        assert_eq!(tree.num_nodes(), 3);
        assert!((tree.predict_row(&data, 0) - (-1.0)).abs() < 1e-6);
        assert!((tree.predict_row(&data, 2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn depth_limit_preserves_reused_column_sampler_state() {
        let n = 256;
        let features = 8;
        let x: Vec<f32> = (0..n * features)
            .map(|i| ((i / features * 17 + i % features * 29) % 97) as f32 / 97.0)
            .collect();
        let gradients: Vec<GradPair> = x
            .chunks_exact(features)
            .map(|row| gp(row.iter().sum::<f32>() - 4.0, 1.0))
            .collect();
        let data = DMatrix::from_dense(&x, n, features).unwrap();
        let ghist = binned(&data, 64);
        let rows = all_rows(n);

        for depth in [1, 2, 4] {
            let params = TrainingParams::builder().max_depth(depth).build().unwrap();
            let builder = HistTreeBuilder::new(&params);
            let new_sampler = || ColumnSampler::new((0..features as u32).collect(), 0.75, 0.75, 42);
            let mut sampler = new_sampler();
            let mut expected = new_sampler();
            for _ in 0..3 {
                let tree = builder.build(&ghist, &gradients, &rows, &mut sampler);
                assert!(tree.num_nodes() > 1);
                // Every created node consumes one draw, including leaves whose
                // histogram and split search are skipped at the depth limit.
                for _ in 0..tree.num_nodes() {
                    expected.sample();
                }
                for _ in 0..4 {
                    assert_eq!(sampler.sample(), expected.sample());
                }
            }
        }
    }

    #[test]
    fn parallel_depthwise_preserves_tree_and_sampler() {
        use crate::config::Monotone;
        use crate::data::FeatureType;

        let n = 8192;
        let features = 8;
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for mode in ["dense", "missing", "categorical"] {
            let mut state = 123u64;
            let mut values = Vec::with_capacity(n * features);
            let mut gradients = Vec::with_capacity(n);
            for row in 0..n {
                let mut target = 0.0;
                for col in 0..features {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    let mut value = (state >> 33) as f32 / (1u32 << 31) as f32;
                    if mode == "categorical" && col < 2 {
                        value = (value * 4.0).floor();
                    }
                    target += value * (col + 1) as f32;
                    if mode == "missing" && (row * 13 + col * 7) % 11 < 2 {
                        value = f32::NAN;
                    }
                    values.push(value);
                }
                gradients.push(gp(18.0 - target, 1.0));
            }
            let mut data = DMatrix::from_dense(&values, n, features).unwrap();
            if mode == "categorical" {
                let mut types = vec![FeatureType::Numerical; features];
                types[..2].fill(FeatureType::Categorical);
                data = data.with_feature_types(&types).unwrap();
            }
            let ghist = binned(&data, 64);
            let rows = all_rows(n);
            let params = TrainingParams::builder()
                .max_depth(6)
                .alpha(0.1)
                .monotone_constraints(vec![Monotone::None, Monotone::None, Monotone::Increasing])
                .interaction_constraints(vec![vec![0, 1, 2, 3], vec![2, 4, 5, 6, 7]])
                .build()
                .unwrap();
            let builder = HistTreeBuilder::new(&params);
            let new_sampler = || ColumnSampler::new((0..features as u32).collect(), 0.75, 0.75, 91);
            // Three samplers from one seed, kept in lockstep by the draws below.
            let mut expected_sampler = new_sampler();
            let mut sampler = new_sampler();
            let mut captured_sampler = new_sampler();
            for _ in 0..3 {
                let expected = serial
                    .install(|| builder.build(&ghist, &gradients, &rows, &mut expected_sampler));
                let actual =
                    parallel.install(|| builder.build(&ghist, &gradients, &rows, &mut sampler));
                let (captured, leaves) = parallel.install(|| {
                    builder.build_with_leaf_rows(&ghist, &gradients, &rows, &mut captured_sampler)
                });
                assert_eq!(captured, expected, "{mode}");
                let mut seen = vec![false; n];
                for leaf in leaves {
                    assert!(captured.node(leaf.node).is_leaf());
                    for row in leaf.rows {
                        let row = row as usize;
                        assert!(!seen[row]);
                        seen[row] = true;
                        assert_eq!(
                            leaf.node,
                            captured.leaf_id_with(|f| data.get(row, f as usize))
                        );
                    }
                }
                assert!(seen.into_iter().all(|seen| seen));
                assert!(actual.num_nodes() > 7, "must exercise multiple depths");
                assert_eq!(actual, expected, "{mode}");
                let next = sampler.sample();
                assert_eq!(next, expected_sampler.sample(), "{mode}");
                assert_eq!(next, captured_sampler.sample(), "{mode}");
            }
        }
    }

    #[test]
    fn no_split_below_gamma() {
        let x = vec![0.0f32, 1.0];
        let data = DMatrix::from_dense(&x, 2, 1).unwrap();
        let ghist = binned(&data, 256);
        let gpair = vec![gp(1.0, 1.0), gp(-1.0, 1.0)];
        let params = TrainingParams::builder()
            .max_depth(3)
            .gamma(1e9)
            .build()
            .unwrap();
        let tree = HistTreeBuilder::new(&params).build(
            &ghist,
            &gpair,
            &all_rows(2),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );
        assert_eq!(tree.num_nodes(), 1);
        let (captured, leaves) = HistTreeBuilder::new(&params).build_with_leaf_rows(
            &ghist,
            &gpair,
            &all_rows(2),
            &mut ColumnSampler::all(1),
        );
        assert_eq!(captured, tree);
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].node, 0);
        assert_eq!(leaves[0].rows, all_rows(2));
    }

    #[test]
    fn lossguide_respects_max_leaves() {
        // Enough structure that greedy growth would exceed the leaf cap.
        let n = 64;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut y = Vec::new();
        for i in 0..n {
            y.push(gp(if i % 2 == 0 { 1.0 } else { -1.0 }, 1.0));
        }
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let ghist = binned(&data, 256);
        let params = TrainingParams::builder()
            .grow_policy(GrowPolicy::LossGuide)
            .max_leaves(4)
            .max_depth(0)
            .min_child_weight(0.0)
            .gamma(0.0)
            .lambda(0.0)
            .build()
            .unwrap();
        let tree = HistTreeBuilder::new(&params).build(
            &ghist,
            &y,
            &all_rows(n),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );
        assert!(tree.num_leaves() <= 4, "got {} leaves", tree.num_leaves());
    }

    #[test]
    fn monotone_increasing_is_enforced() {
        use crate::config::Monotone;
        let (data, gpair) = monotone_v_shape_data();
        let n = data.n_rows();
        let ghist = binned(&data, 256);
        let params = TrainingParams::builder()
            .max_depth(4)
            .min_child_weight(0.0)
            .gamma(0.0)
            .lambda(1.0)
            .monotone_constraints(vec![Monotone::Increasing])
            .build()
            .unwrap();
        let tree = HistTreeBuilder::new(&params).build(
            &ghist,
            &gpair,
            &all_rows(n),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );

        // Predictions must be non-decreasing in x under the increasing constraint.
        let mut prev = f32::NEG_INFINITY;
        for i in 0..n {
            let p = tree.predict_row(&data, i);
            assert!(
                p >= prev - 1e-5,
                "monotonicity violated at row {i}: {p} < {prev}"
            );
            prev = p;
        }
    }

    // Collect the split features along every root-to-leaf path.
    fn root_to_leaf_feature_sets(tree: &RegTree) -> Vec<Vec<u32>> {
        fn walk(tree: &RegTree, id: usize, path: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
            let node = tree.node(id);
            if node.is_leaf() {
                out.push(path.clone());
                return;
            }
            path.push(node.split_feature);
            walk(tree, node.left as usize, path, out);
            walk(tree, node.right as usize, path, out);
            path.pop();
        }
        let mut out = Vec::new();
        walk(tree, 0, &mut Vec::new(), &mut out);
        out
    }

    // A 4-feature dataset where every feature carries signal, so an
    // unconstrained tree would happily mix features across groups.
    fn four_feature_data() -> (DMatrix, Vec<GradPair>) {
        let n = 32;
        let mut x = vec![0.0f32; n * 4];
        let mut gpair = Vec::with_capacity(n);
        for i in 0..n {
            // Distinct-ish per-feature patterns so each is individually useful.
            x[i * 4] = (i % 2) as f32;
            x[i * 4 + 1] = (i % 4) as f32;
            x[i * 4 + 2] = (i % 8) as f32;
            x[i * 4 + 3] = (i % 16) as f32;
            let g = if i % 2 == 0 { 1.0 } else { -1.0 };
            gpair.push(gp(g, 1.0));
        }
        (DMatrix::from_dense(&x, n, 4).unwrap(), gpair)
    }

    #[test]
    fn interaction_constraints_confine_paths_to_one_group() {
        let (data, gpair) = four_feature_data();
        let ghist = binned(&data, 256);
        let params = TrainingParams::builder()
            .max_depth(4)
            .min_child_weight(0.0)
            .gamma(0.0)
            .lambda(0.0)
            .interaction_constraints(vec![vec![0, 1], vec![2, 3]])
            .build()
            .unwrap();
        let tree = HistTreeBuilder::new(&params).build(
            &ghist,
            &gpair,
            &all_rows(data.n_rows()),
            &mut crate::tree::sampler::ColumnSampler::all(4),
        );

        // Every path's split features must fit inside a single allowed group:
        // never both a {0,1} feature and a {2,3} feature on the same path.
        for path in root_to_leaf_feature_sets(&tree) {
            let has_ab = path.iter().any(|&f| f == 0 || f == 1);
            let has_cd = path.iter().any(|&f| f == 2 || f == 3);
            assert!(
                !(has_ab && has_cd),
                "path mixes interaction groups: {path:?}"
            );
        }
    }

    #[test]
    fn empty_interaction_constraints_leave_behavior_unchanged() {
        let (data, gpair) = four_feature_data();
        let ghist = binned(&data, 256);
        let base = TrainingParams::builder()
            .max_depth(4)
            .min_child_weight(0.0)
            .gamma(0.0)
            .lambda(0.0)
            .build()
            .unwrap();
        let with_empty = TrainingParams::builder()
            .max_depth(4)
            .min_child_weight(0.0)
            .gamma(0.0)
            .lambda(0.0)
            .interaction_constraints(Vec::new())
            .build()
            .unwrap();

        let t1 = HistTreeBuilder::new(&base).build(
            &ghist,
            &gpair,
            &all_rows(data.n_rows()),
            &mut crate::tree::sampler::ColumnSampler::all(4),
        );
        let t2 = HistTreeBuilder::new(&with_empty).build(
            &ghist,
            &gpair,
            &all_rows(data.n_rows()),
            &mut crate::tree::sampler::ColumnSampler::all(4),
        );

        assert_eq!(t1.num_nodes(), t2.num_nodes());
        for r in 0..data.n_rows() {
            assert!((t1.predict_row(&data, r) - t2.predict_row(&data, r)).abs() < 1e-9);
        }
    }

    #[test]
    fn hist_matches_exact_on_small_problem() {
        use crate::tree::builder::{ExactTreeBuilder, SortedColumns};
        // Random-ish separable-ish data; hist with enough bins should match exact.
        let n = 60;
        let mut x = Vec::new();
        let mut gpair = Vec::new();
        for i in 0..n {
            let xi = (i as f32) * 0.1;
            x.push(xi);
            gpair.push(gp((xi - 3.0).sin(), 1.0));
        }
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let params = TrainingParams::builder()
            .max_depth(3)
            .lambda(1.0)
            .min_child_weight(1.0)
            .gamma(0.0)
            .build()
            .unwrap();

        let cols = SortedColumns::from_dmatrix(&data);
        let exact = ExactTreeBuilder::new(&params).build(
            &cols,
            &data,
            &gpair,
            &all_rows(n),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );

        // 256 bins over 60 distinct values -> each value its own bin -> exact match.
        let ghist = binned(&data, 256);
        let hist = HistTreeBuilder::new(&params).build(
            &ghist,
            &gpair,
            &all_rows(n),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );

        for r in 0..n {
            let pe = exact.predict_row(&data, r);
            let ph = hist.predict_row(&data, r);
            assert!((pe - ph).abs() < 1e-5, "row {r}: exact {pe} vs hist {ph}");
        }
    }

    // Present rows share one gradient sign and missing rows the other. Under an
    // increasing constraint the forward endpoint (present left, missing right)
    // is rejected, and the only split with pure children is XGBoost's backward
    // endpoint: missing mass left, every present bin right.
    #[test]
    fn missing_only_left_split_under_monotone_constraint() {
        use crate::config::Monotone;
        let x = vec![0.0f32, 1.0, 2.0, 3.0, f32::NAN, f32::NAN];
        let n = x.len();
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let ghist = binned(&data, 256);
        assert!(ghist.dense_stride().is_none());
        let mut gpair = vec![gp(-1.0, 1.0); 4];
        gpair.extend([gp(1.0, 1.0), gp(1.0, 1.0)]);
        let params = TrainingParams::builder()
            .max_depth(1)
            .lambda(0.0)
            .min_child_weight(0.0)
            .gamma(0.0)
            .monotone_constraints(vec![Monotone::Increasing])
            .build()
            .unwrap();
        let tree = HistTreeBuilder::new(&params).build(
            &ghist,
            &gpair,
            &all_rows(n),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );
        assert_eq!(tree.num_nodes(), 3);
        let root = tree.node(0);
        assert!(root.default_left);
        assert!(root.split_cond.is_finite());
        let left = tree.node(root.left as usize);
        let right = tree.node(root.right as usize);
        assert!(
            (left.sum_hess - 2.0).abs() < 1e-6,
            "left cover {}",
            left.sum_hess
        );
        assert!(
            (right.sum_hess - 4.0).abs() < 1e-6,
            "right cover {}",
            right.sum_hess
        );
        for r in 0..4 {
            assert!(
                (tree.predict_row(&data, r) - 1.0).abs() < 1e-6,
                "present row {r}"
            );
        }
        for r in 4..6 {
            assert!(
                (tree.predict_row(&data, r) + 1.0).abs() < 1e-6,
                "missing row {r}"
            );
        }
    }

    // A sparse index whose split feature is fully present has no missing mass
    // to route: the backward pass never runs, so the split keeps the forward
    // orientation (missing right) even when the tie-breaking alternative
    // would be a missing-left split.
    #[test]
    fn fully_present_feature_never_splits_missing_left() {
        use crate::config::Monotone;
        // Feature 1 carries the NaN that makes the index sparse but is constant
        // otherwise, so only feature 0 can split.
        let x = vec![
            0.0f32,
            5.0,
            1.0,
            5.0,
            2.0,
            5.0,
            3.0,
            f32::NAN,
            4.0,
            5.0,
            5.0,
            5.0,
        ];
        let n = 6;
        let data = DMatrix::from_dense(&x, n, 2).unwrap();
        let ghist = binned(&data, 256);
        assert!(ghist.dense_stride().is_none());
        let gpair = vec![
            gp(1.0, 1.0),
            gp(1.0, 1.0),
            gp(1.0, 1.0),
            gp(-1.0, 1.0),
            gp(-1.0, 1.0),
            gp(-1.0, 1.0),
        ];
        let params = TrainingParams::builder()
            .max_depth(1)
            .lambda(0.0)
            .min_child_weight(0.0)
            .gamma(0.0)
            .monotone_constraints(vec![Monotone::Increasing, Monotone::None])
            .build()
            .unwrap();
        let tree = HistTreeBuilder::new(&params).build(
            &ghist,
            &gpair,
            &all_rows(n),
            &mut crate::tree::sampler::ColumnSampler::all(2),
        );
        assert_eq!(tree.num_nodes(), 3);
        let root = tree.node(0);
        assert_eq!(root.split_feature, 0);
        assert!(!root.default_left);
        assert!(
            root.split_cond > 2.0 && root.split_cond <= 3.0,
            "{}",
            root.split_cond
        );
        for r in 0..3 {
            assert!((tree.predict_row(&data, r) + 1.0).abs() < 1e-6, "row {r}");
        }
        for r in 3..6 {
            assert!((tree.predict_row(&data, r) - 1.0).abs() < 1e-6, "row {r}");
        }
    }
}
