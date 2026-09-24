//! Generalization-gated tree growth for budget-mode training
//! ([`train_with_budget`](crate::learner::budget::train_with_budget)), after
//! Perpetual's splitter and tree grower (`splitter.rs`,
//! `tree/core.rs`; Apache-2.0, re-implemented here).
//!
//! Every row belongs to one of five folds (`row % 5`, Perpetual's default
//! assignment), and every histogram bin keeps per-fold gradient, Hessian, and
//! row-count sums. A candidate split is scored five times: each fold in turn
//! is the validation part and the other four fit the child weights
//! `w = −G/(H + ε)`. With the second-order loss `G w + ½ H w²` per row, the
//! averaged in-fold (training) and out-of-fold (validation) losses give the
//! generalization ratio
//!
//! `gen = (parent − train) / (parent − valid)`,
//!
//! and a non-root split is only accepted when `gen` clears a floor of `1.0`
//! (relaxed by at most `0.01`/`0.016` for small, deep nodes; `0.99`-based for
//! categorical splits) and every fold has rows on both sides. The root always
//! splits when any positive-gain split exists. Accepted candidates are ranked
//! by their full-data gain damped by fold-weight stability and `gen`.
//!
//! Nodes are expanded best-first by their own score `G²/(H + ε)`. After each
//! split the actual loss reduction of the children's rows (from the
//! objective's pointwise loss, via a caller-supplied callback) is added to a
//! running per-row average, and growth stops once it exceeds the round's
//! target loss decrement ([`TreeStopper::StepSize`]), when no frontier node
//! has an acceptable split ([`TreeStopper::Generalization`]), or at
//! [`MAX_NODES`] nodes ([`TreeStopper::MaxNodes`]).
//!
//! Missing values: both directions are evaluated as complete candidates,
//! with the missing rows counted in every fold's statistics. (Perpetual's
//! imputing splitter adds the whole missing mass to the in-fold side only.)

use crate::data::ghist::{Bins, GHistIndex};
use crate::objective::GradPair;
use crate::tree::regtree::RegTree;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Number of generalization folds.
const N_FOLDS: usize = 5;
/// Hessian guard in every weight and score (Perpetual `HESSIAN_EPS`).
const HESSIAN_EPS: f64 = 1e-8;
/// Generalization floor for numeric splits (Perpetual `GENERALIZATION_THRESHOLD`).
const GENERALIZATION_THRESHOLD: f64 = 1.0;
/// Base floor for categorical splits (Perpetual `GENERALIZATION_THRESHOLD_RELAXED`).
const GENERALIZATION_THRESHOLD_RELAXED: f64 = 0.99;
/// Node cap per tree (Perpetual `N_NODES_ALLOC_MAX`).
pub(crate) const MAX_NODES: usize = 10_000;
/// Rows below which a node's histogram and loss update run serially.
const PARALLEL_ROWS: usize = 4096;
/// Upper bounds (rounded up) of the largest factor [`ranking_gain`] applies
/// to a split's gain: `1.05^0.1` for numeric and `1.12^0.3` for categorical
/// splits (every other factor is at most `1`).
const MAX_NUMERIC_RANK_FACTOR: f64 = 1.0049;
const MAX_CATEGORICAL_RANK_FACTOR: f64 = 1.0346;
/// Fixed chunk length of the parallel loss-decrement sums (thread-count
/// independent, so the summation order is deterministic).
const DECREMENT_CHUNK: usize = 4096;

/// Why a tree stopped growing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TreeStopper {
    /// No frontier node had a split passing the generalization check.
    Generalization,
    /// The round's target loss decrement was reached.
    StepSize,
    /// The tree reached [`MAX_NODES`] nodes.
    MaxNodes,
}

/// Per-fold sums for one histogram bin or row set.
#[derive(Debug, Clone, Copy, Default)]
struct FoldStats {
    grad: [f64; N_FOLDS],
    hess: [f64; N_FOLDS],
    count: [u32; N_FOLDS],
}

/// Whole-set sums of a [`FoldStats`].
#[derive(Debug, Clone, Copy)]
struct Totals {
    grad: f64,
    hess: f64,
    count: u32,
}

impl FoldStats {
    #[inline]
    fn add_row(&mut self, row: u32, gp: GradPair) {
        let fold = row as usize % N_FOLDS;
        self.grad[fold] += f64::from(gp.grad);
        self.hess[fold] += f64::from(gp.hess);
        self.count[fold] += 1;
    }

    #[inline]
    fn add(&mut self, other: &FoldStats) {
        for j in 0..N_FOLDS {
            self.grad[j] += other.grad[j];
            self.hess[j] += other.hess[j];
            self.count[j] += other.count[j];
        }
    }

    #[inline]
    fn sub(&self, other: &FoldStats) -> FoldStats {
        let mut out = *self;
        for j in 0..N_FOLDS {
            out.grad[j] -= other.grad[j];
            out.hess[j] -= other.hess[j];
            out.count[j] -= other.count[j];
        }
        out
    }

    fn totals(&self) -> Totals {
        Totals {
            grad: self.grad.iter().sum(),
            hess: self.hess.iter().sum(),
            count: self.count.iter().sum(),
        }
    }
}

/// Newton weight `−G/(H + ε)`.
#[inline]
fn weight(grad: f64, hess: f64) -> f64 {
    -grad / (hess + HESSIAN_EPS)
}

/// A leaf value: the Newton weight clamped to `±max_delta_step` (when
/// positive, as XGBoost's `CalcWeight` does) and shrunk by `eta`.
#[inline]
fn leaf_value(grad: f64, hess: f64, cfg: &GrowConfig) -> f32 {
    let mut w = weight(grad, hess);
    if cfg.max_delta_step > 0.0 {
        w = w.clamp(-cfg.max_delta_step, cfg.max_delta_step);
    }
    cfg.eta * w as f32
}

/// Score `G²/(H + ε)`, written as `−(2 G w + (H + ε) w²)` at the Newton weight
/// like Perpetual.
#[inline]
fn score(grad: f64, hess: f64) -> f64 {
    let w = weight(grad, hess);
    -(2.0 * grad * w + (hess + HESSIAN_EPS) * w * w)
}

fn mean(values: &[f64; N_FOLDS]) -> f64 {
    values.iter().sum::<f64>() / N_FOLDS as f64
}

/// Agreement of one child's five fold weights (splitter form):
/// `1 / (1 + σ/(|w̄| + 1e-6))` clamped to `[0.5, 1]`.
fn fold_weight_stability(weights: &[f64; N_FOLDS]) -> f64 {
    let m = mean(weights);
    let mean_abs = weights.iter().map(|w| w.abs()).sum::<f64>() / N_FOLDS as f64;
    if mean_abs <= f64::from(f32::EPSILON) {
        return 1.0;
    }
    let variance = weights.iter().map(|w| (w - m).powi(2)).sum::<f64>() / N_FOLDS as f64;
    (1.0 / (1.0 + variance.sqrt() / (mean_abs + 1e-6))).clamp(0.5, 1.0)
}

/// Root-mean-square of a child's fold weights.
fn fold_weight_energy(weights: &[f64; N_FOLDS]) -> f64 {
    (weights.iter().map(|w| w * w).sum::<f64>() / N_FOLDS as f64).sqrt()
}

fn split_weight_stability(left: &[f64; N_FOLDS], right: &[f64; N_FOLDS]) -> f64 {
    f64::midpoint(fold_weight_stability(left), fold_weight_stability(right))
}

/// The generalization floor a non-root split must reach: `1.0` (numeric) or
/// `0.99 + 0.004 (1 − stability)` (categorical), relaxed by up to `0.01`
/// (`0.006`) for nodes with few rows at depth, never below `0.98` (`0.984`).
fn generalization_floor(stability: f64, categorical: bool, depth: usize, count: usize) -> f64 {
    let floor = if categorical {
        GENERALIZATION_THRESHOLD_RELAXED + 0.004 * (1.0 - stability)
    } else {
        GENERALIZATION_THRESHOLD
    };
    let support_scale = (1.0 - count.min(2048) as f64 / 2048.0).clamp(0.0, 1.0);
    let depth_scale = (depth.min(3) as f64 / 3.0).clamp(0.0, 1.0);
    let max_relief = if categorical { 0.006 } else { 0.01 };
    let relief = max_relief * support_scale * (0.5 + 0.5 * depth_scale) * stability.clamp(0.5, 1.0);
    let min_floor = if categorical { 0.984 } else { 0.98 };
    (floor - relief).clamp(min_floor, GENERALIZATION_THRESHOLD)
}

/// The ranking score of an accepted split: its gain damped by fold-weight
/// stability and (weakly) by its generalization ratio.
fn ranking_gain(
    split_gain: f64,
    generalization: f64,
    left: &[f64; N_FOLDS],
    right: &[f64; N_FOLDS],
    categorical: bool,
) -> f64 {
    let stability = split_weight_stability(left, right);
    if categorical {
        let factor =
            generalization.clamp(generalization_floor(stability, true, 0, usize::MAX), 1.12);
        let energy = fold_weight_energy(left) + fold_weight_energy(right);
        split_gain * (0.75 + 0.25 * stability) * factor.powf(0.3) / (1.0 + 0.02 * energy)
    } else {
        let factor =
            generalization.clamp(generalization_floor(stability, false, 0, usize::MAX), 1.05);
        split_gain * (0.92 + 0.08 * stability) * factor.powf(0.1)
    }
}

/// A split candidate that passed the generalization check.
#[derive(Debug, Clone)]
struct Candidate {
    rank: f64,
    split_gain: f64,
    generalization: f64,
    left: FoldStats,
    right: FoldStats,
    left_weights: [f64; N_FOLDS],
    right_weights: [f64; N_FOLDS],
    feature: u32,
    /// Numeric: bins `<= split_bin` go left. Categorical: unused.
    split_bin: usize,
    /// Categorical: the global bins routed left (empty for numeric splits).
    cat_bins: Vec<usize>,
    default_left: bool,
}

/// What the split search needs to know about the node being split.
struct NodeCtx {
    is_root: bool,
    depth: usize,
    count: usize,
    gain: f64,
    stats: FoldStats,
}

impl NodeCtx {
    /// Score the partition `left`/`right` of this node; `None` when it fails
    /// the fold-coverage or generalization checks or has no positive gain.
    /// Partitions whose best possible rank (their gain times the largest
    /// ranking factor) cannot beat `rank_to_beat` are skipped before the
    /// fold evaluation; this only saves work, the chosen split is unchanged.
    fn evaluate(
        &self,
        left: &FoldStats,
        right: &FoldStats,
        categorical: bool,
        rank_to_beat: f64,
    ) -> Option<Scored> {
        let (lt, rt) = (left.totals(), right.totals());
        if lt.count == 0 || rt.count == 0 {
            return None;
        }
        let split_gain = score(lt.grad, lt.hess) + score(rt.grad, rt.hess) - self.gain;
        if split_gain.is_nan() || split_gain <= 0.0 {
            return None;
        }
        let max_factor = if categorical {
            MAX_CATEGORICAL_RANK_FACTOR
        } else {
            MAX_NUMERIC_RANK_FACTOR
        };
        if split_gain * max_factor <= rank_to_beat {
            return None;
        }
        let mut train = [0.0; N_FOLDS];
        let mut valid = [0.0; N_FOLDS];
        let mut left_weights = [0.0; N_FOLDS];
        let mut right_weights = [0.0; N_FOLDS];
        let mut n_folds = 0;
        for j in 0..N_FOLDS {
            let (lc_train, rc_train) = (lt.count - left.count[j], rt.count - right.count[j]);
            let (lc_valid, rc_valid) = (left.count[j], right.count[j]);
            if lc_train == 0 || rc_train == 0 || lc_valid == 0 || rc_valid == 0 {
                continue;
            }
            let (lg, lh) = (lt.grad - left.grad[j], lt.hess - left.hess[j]);
            let (rg, rh) = (rt.grad - right.grad[j], rt.hess - right.hess[j]);
            let (lw, rw) = (weight(lg, lh), weight(rg, rh));
            left_weights[j] = lw;
            right_weights[j] = rw;
            let valid_loss = |g: f64, h: f64, w: f64| g * w + (h + HESSIAN_EPS) * w * w / 2.0;
            valid[j] = (valid_loss(left.grad[j], left.hess[j], lw)
                + valid_loss(right.grad[j], right.hess[j], rw))
                / f64::from(lc_valid + rc_valid);
            train[j] = -0.5 * (score(lg, lh) + score(rg, rh)) / f64::from(lc_train + rc_train);
            n_folds += 1;
        }
        if n_folds < N_FOLDS && !self.is_root {
            return None;
        }
        let parent = -0.5 * self.gain / self.count as f64;
        let generalization = (parent - mean(&train)) / (parent - mean(&valid));
        // Undefined (`0/0`: a root without usable folds whose gradients sum
        // to zero) ranks nowhere, as in Perpetual, where a NaN rank never
        // beats the best; a root of at most eight rows then falls back to
        // `evaluate_tiny_root`.
        if generalization.is_nan() {
            return None;
        }
        if !self.is_root {
            let stability = split_weight_stability(&left_weights, &right_weights);
            let floor = generalization_floor(stability, categorical, self.depth, self.count);
            if generalization < floor {
                return None;
            }
        }
        Some(Scored {
            rank: ranking_gain(
                split_gain,
                generalization,
                &left_weights,
                &right_weights,
                categorical,
            ),
            split_gain,
            generalization,
            left_weights,
            right_weights,
        })
    }

    /// Perpetual's fallback for a root of at most eight rows, where the fold
    /// check cannot pass: plain positive-gain splits, generalization `1`.
    fn evaluate_tiny_root(&self, left: &FoldStats, right: &FoldStats) -> Option<Scored> {
        let (lt, rt) = (left.totals(), right.totals());
        if lt.count == 0 || rt.count == 0 || lt.hess <= 0.0 || rt.hess <= 0.0 {
            return None;
        }
        let split_gain = score(lt.grad, lt.hess) + score(rt.grad, rt.hess) - self.gain;
        (split_gain > 0.0).then(|| Scored {
            rank: split_gain,
            split_gain,
            generalization: 1.0,
            left_weights: [weight(lt.grad, lt.hess); N_FOLDS],
            right_weights: [weight(rt.grad, rt.hess); N_FOLDS],
        })
    }
}

/// The scores of one evaluated partition.
struct Scored {
    rank: f64,
    split_gain: f64,
    generalization: f64,
    left_weights: [f64; N_FOLDS],
    right_weights: [f64; N_FOLDS],
}

/// Keep the higher-ranked of `best` and a new partition (ties keep `best`,
/// which was found earlier in the fixed scan order).
fn offer(best: &mut Option<Candidate>, scored: Scored, make: impl FnOnce(Scored) -> Candidate) {
    if best.as_ref().is_none_or(|b| scored.rank > b.rank) {
        *best = Some(make(scored));
    }
}

/// One split recorded for the tree-level generalization score: the ratio of
/// the split that created a child plus that child's fold weights and rows.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChildRecord {
    /// Generalization ratio of the split that created this node.
    pub generalization: f64,
    /// The node's five fold weights (unscaled).
    pub fold_weights: [f64; N_FOLDS],
    /// Training rows in the node.
    pub count: usize,
}

/// A grown tree plus the training-time bookkeeping the booster needs.
pub(crate) struct GrownTree {
    /// The tree with leaves already shrunk by `eta`.
    pub tree: RegTree,
    /// Why growth stopped.
    pub stopper: TreeStopper,
    /// One record per node created by a split (two per split).
    pub children: Vec<ChildRecord>,
    /// Row index buffer; leaf `(nid, start, end)` owns `index[start..end]`.
    index: Vec<u32>,
    leaves: Vec<(usize, usize, usize)>,
}

impl GrownTree {
    /// Add each leaf's value to the training margins of its rows.
    pub fn apply(&self, margins: &mut [f32]) {
        for &(nid, start, end) in &self.leaves {
            let value = self.tree.node(nid).leaf_value;
            for &r in &self.index[start..end] {
                margins[r as usize] += value;
            }
        }
    }
}

/// Per-tree inputs.
pub(crate) struct GrowConfig<'a> {
    /// Learning rate applied to every node weight.
    pub eta: f32,
    /// Stop once the average per-row loss decrement exceeds this.
    pub target_loss_decrement: Option<f64>,
    /// Loss reduction of row `r` when its margin moves by `delta`
    /// (`loss_r(m_r) − loss_r(m_r + delta)`, sample-weighted).
    pub row_decrement: &'a (dyn Fn(u32, f32) -> f64 + Sync),
    /// Bound on every unshrunk leaf weight (`0`: unbounded), XGBoost
    /// `max_delta_step`.
    pub max_delta_step: f64,
}

/// A frontier node, ordered by its own score (ties: lower node id first).
struct Frontier {
    nid: usize,
    start: usize,
    end: usize,
    depth: usize,
    gain: f64,
    stats: FoldStats,
}

impl PartialEq for Frontier {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Frontier {}
impl PartialOrd for Frontier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Frontier {
    fn cmp(&self, other: &Self) -> Ordering {
        self.gain
            .total_cmp(&other.gain)
            .then_with(|| other.nid.cmp(&self.nid))
    }
}

/// Grow one tree from this round's gradients over all rows of `ghist`.
pub(crate) fn grow(ghist: &GHistIndex, gpair: &[GradPair], cfg: &GrowConfig) -> GrownTree {
    let n = ghist.n_rows();
    let mut index: Vec<u32> = (0..n as u32).collect();
    let mut root_stats = FoldStats::default();
    for &r in &index {
        root_stats.add_row(r, gpair[r as usize]);
    }
    let root_totals = root_stats.totals();
    let mut tree = RegTree::with_root(root_totals.hess as f32);
    tree.set_leaf_value(0, leaf_value(root_totals.grad, root_totals.hess, cfg));

    let mut heap = BinaryHeap::new();
    heap.push(Frontier {
        nid: 0,
        start: 0,
        end: n,
        depth: 0,
        gain: score(root_totals.grad, root_totals.hess),
        stats: root_stats,
    });
    let mut loss_decr = vec![0.0f64; n];
    let mut loss_decr_avg = 0.0f64;
    let mut stopper = TreeStopper::Generalization;
    let mut leaves = Vec::new();
    let mut children = Vec::new();
    let mut scratch = Vec::new();

    while !heap.is_empty() {
        if tree.num_nodes() + 2 > MAX_NODES {
            stopper = TreeStopper::MaxNodes;
            break;
        }
        if cfg
            .target_loss_decrement
            .is_some_and(|target| loss_decr_avg > target)
        {
            stopper = TreeStopper::StepSize;
            break;
        }
        let Some(node) = heap.pop() else { break };
        let ctx = NodeCtx {
            is_root: node.nid == 0,
            depth: node.depth,
            count: node.end - node.start,
            gain: node.gain,
            stats: node.stats,
        };
        let rows = &index[node.start..node.end];
        let hist = build_histogram(ghist, rows, gpair);
        let Some(best) = find_split(ghist, &hist, &ctx) else {
            leaves.push((node.nid, node.start, node.end));
            continue;
        };

        let n_left = partition(ghist, &mut index[node.start..node.end], &best, &mut scratch);
        let mid = node.start + n_left;
        let (lt, rt) = (best.left.totals(), best.right.totals());
        let left_value = leaf_value(lt.grad, lt.hess, cfg);
        let right_value = leaf_value(rt.grad, rt.hess, cfg);
        let (left_id, right_id) = if best.cat_bins.is_empty() {
            tree.expand(
                node.nid,
                best.feature,
                ghist.cuts().cut_value(best.split_bin),
                best.default_left,
                left_value,
                lt.hess as f32,
                right_value,
                rt.hess as f32,
            )
        } else {
            let mut cats: Vec<u32> = best
                .cat_bins
                .iter()
                .map(|&b| ghist.cuts().cut_value(b) as u32)
                .collect();
            cats.sort_unstable();
            tree.expand_categorical(
                node.nid,
                best.feature,
                &cats,
                best.default_left,
                left_value,
                lt.hess as f32,
                right_value,
                rt.hess as f32,
            )
        };
        tree.set_split_gain(node.nid, best.split_gain as f32);

        for (nid, start, end, value, stats, fold_weights) in [
            (
                left_id,
                node.start,
                mid,
                left_value,
                best.left,
                best.left_weights,
            ),
            (
                right_id,
                mid,
                node.end,
                right_value,
                best.right,
                best.right_weights,
            ),
        ] {
            if cfg.target_loss_decrement.is_some() {
                let delta =
                    update_decrement(&index[start..end], value, &mut loss_decr, cfg.row_decrement);
                loss_decr_avg += delta / n as f64;
            }
            let totals = stats.totals();
            children.push(ChildRecord {
                generalization: best.generalization,
                fold_weights,
                count: end - start,
            });
            heap.push(Frontier {
                nid,
                start,
                end,
                depth: node.depth + 1,
                gain: score(totals.grad, totals.hess),
                stats,
            });
        }
    }
    leaves.extend(heap.into_iter().map(|f| (f.nid, f.start, f.end)));

    GrownTree {
        tree,
        stopper,
        children,
        index,
        leaves,
    }
}

/// Recompute the loss decrement of `rows` now that their leaf value is
/// `value`, returning the change of the decrement sum. Parallel chunks are a
/// fixed length and summed in order, so the result is thread-count
/// independent.
fn update_decrement(
    rows: &[u32],
    value: f32,
    loss_decr: &mut [f64],
    row_decrement: &(dyn Fn(u32, f32) -> f64 + Sync),
) -> f64 {
    let serial = |chunk: &[u32]| -> Vec<(u32, f64)> {
        chunk
            .iter()
            .map(|&r| (r, row_decrement(r, value)))
            .collect()
    };
    let updates: Vec<Vec<(u32, f64)>> =
        if rows.len() >= PARALLEL_ROWS && rayon::current_num_threads() > 1 {
            rows.par_chunks(DECREMENT_CHUNK).map(serial).collect()
        } else {
            rows.chunks(DECREMENT_CHUNK).map(serial).collect()
        };
    let mut delta = 0.0;
    for (r, new) in updates.into_iter().flatten() {
        let old = &mut loss_decr[r as usize];
        delta += new - *old;
        *old = new;
    }
    delta
}

/// Per-fold histogram of `rows` over every global bin.
fn build_histogram(ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair]) -> Vec<FoldStats> {
    let mut hist = vec![FoldStats::default(); ghist.total_bins()];
    let parallel = rows.len() >= PARALLEL_ROWS && rayon::current_num_threads() > 1;
    if let (true, Some(columns)) = (parallel, ghist.column_bins()) {
        // Feature-parallel: each task owns one feature's bin range and adds
        // that feature's rows in the node's row order.
        let n_rows = ghist.n_rows();
        let cuts = ghist.cuts();
        let mut slices = Vec::with_capacity(ghist.n_cols());
        let mut rest = hist.as_mut_slice();
        for f in 0..ghist.n_cols() {
            let (fs, fe) = cuts.feature_bins(f);
            let (head, tail) = rest.split_at_mut(fe - fs);
            slices.push((f, fs, head));
            rest = tail;
        }
        slices
            .into_par_iter()
            .for_each(|(f, fs, slice)| match columns {
                Bins::U16(c) => {
                    accumulate_column(&c[f * n_rows..(f + 1) * n_rows], fs, rows, gpair, slice);
                }
                Bins::U32(c) => {
                    accumulate_column(&c[f * n_rows..(f + 1) * n_rows], fs, rows, gpair, slice);
                }
            });
    } else {
        match ghist.bins() {
            Bins::U16(bins) => accumulate_rows(bins, ghist.row_ptr(), rows, gpair, &mut hist),
            Bins::U32(bins) => accumulate_rows(bins, ghist.row_ptr(), rows, gpair, &mut hist),
        }
    }
    hist
}

fn accumulate_column<B: Copy + Into<u32>>(
    column: &[B],
    first_bin: usize,
    rows: &[u32],
    gpair: &[GradPair],
    out: &mut [FoldStats],
) {
    for &r in rows {
        let bin = column[r as usize].into() as usize - first_bin;
        out[bin].add_row(r, gpair[r as usize]);
    }
}

fn accumulate_rows<B: Copy + Into<u32>>(
    bins: &[B],
    row_ptr: &[usize],
    rows: &[u32],
    gpair: &[GradPair],
    out: &mut [FoldStats],
) {
    for &r in rows {
        let gp = gpair[r as usize];
        for &bin in &bins[row_ptr[r as usize]..row_ptr[r as usize + 1]] {
            out[bin.into() as usize].add_row(r, gp);
        }
    }
}

/// Best acceptable split of a node over all features (highest rank; ties go
/// to the lower feature index).
fn find_split(ghist: &GHistIndex, hist: &[FoldStats], ctx: &NodeCtx) -> Option<Candidate> {
    let n_features = ghist.n_cols();
    let per_feature = |f: usize| feature_split(ghist, hist, ctx, f);
    let found: Vec<Option<Candidate>> =
        if ctx.count >= PARALLEL_ROWS && rayon::current_num_threads() > 1 {
            (0..n_features).into_par_iter().map(per_feature).collect()
        } else {
            (0..n_features).map(per_feature).collect()
        };
    let mut best: Option<Candidate> = None;
    for candidate in found.into_iter().flatten() {
        if best.as_ref().is_none_or(|b| candidate.rank > b.rank) {
            best = Some(candidate);
        }
    }
    best
}

/// Best acceptable split on feature `f`.
fn feature_split(
    ghist: &GHistIndex,
    hist: &[FoldStats],
    ctx: &NodeCtx,
    f: usize,
) -> Option<Candidate> {
    let cuts = ghist.cuts();
    let (fs, fe) = cuts.feature_bins(f);
    let categorical = cuts.is_categorical(f);
    let bins = &hist[fs..fe];
    let mut present = FoldStats::default();
    for b in bins {
        present.add(b);
    }
    let missing = ctx.stats.sub(&present);
    let has_missing = missing.totals().count > 0;

    // Scan order: numeric bins ascending; categorical bins (non-empty only)
    // ascending by `G/(H + ε)` over all folds, as Perpetual sorts categories.
    let mut order: Vec<usize> = (0..bins.len())
        .filter(|&b| !categorical || bins[b].totals().count > 0)
        .collect();
    if categorical {
        let ratio = |b: usize| {
            let t = bins[b].totals();
            t.grad / (t.hess + HESSIAN_EPS)
        };
        order.sort_by(|&a, &b| ratio(a).total_cmp(&ratio(b)).then(a.cmp(&b)));
    }

    let feature = f as u32;
    let mut best: Option<Candidate> = None;
    let mut left = FoldStats::default();
    for (pos, &b) in order.iter().enumerate() {
        left.add(&bins[b]);
        let last = pos + 1 == order.len();
        // Empty numeric bins repeat the previous partition.
        if (!categorical && bins[b].count.iter().all(|&c| c == 0)) || (last && !has_missing) {
            continue;
        }
        let right = present.sub(&left);
        let cat_bins = || {
            if categorical {
                order[..=pos].iter().map(|&k| fs + k).collect()
            } else {
                Vec::new()
            }
        };
        let make = |l: FoldStats, r: FoldStats, default_left: bool| {
            move |s: Scored| Candidate {
                rank: s.rank,
                split_gain: s.split_gain,
                generalization: s.generalization,
                left: l,
                right: r,
                left_weights: s.left_weights,
                right_weights: s.right_weights,
                feature,
                split_bin: fs + b,
                cat_bins: cat_bins(),
                default_left,
            }
        };
        // Missing rows right (the only option without missing values).
        let rank_to_beat =
            |best: &Option<Candidate>| best.as_ref().map_or(f64::NEG_INFINITY, |c| c.rank);
        let mut missing_right = right;
        missing_right.add(&missing);
        if let Some(s) = ctx.evaluate(&left, &missing_right, categorical, rank_to_beat(&best)) {
            offer(&mut best, s, make(left, missing_right, false));
        }
        if has_missing && !last {
            let mut missing_left = left;
            missing_left.add(&missing);
            if let Some(s) = ctx.evaluate(&missing_left, &right, categorical, rank_to_beat(&best)) {
                offer(&mut best, s, make(missing_left, right, true));
            }
        }
    }

    if best.is_none() && ctx.is_root && !categorical && ctx.count <= 8 {
        let mut left = FoldStats::default();
        for (b, stats) in bins.iter().enumerate() {
            left.add(stats);
            if stats.totals().count == 0 || b + 1 == bins.len() {
                continue;
            }
            let mut right = present.sub(&left);
            right.add(&missing);
            if let Some(s) = ctx.evaluate_tiny_root(&left, &right) {
                offer(&mut best, s, |s| Candidate {
                    rank: s.rank,
                    split_gain: s.split_gain,
                    generalization: s.generalization,
                    left,
                    right,
                    left_weights: s.left_weights,
                    right_weights: s.right_weights,
                    feature,
                    split_bin: fs + b,
                    cat_bins: Vec::new(),
                    default_left: false,
                });
            }
        }
    }
    best
}

/// Stable in-place partition of a node's rows by the chosen split. Returns
/// the number of rows routed left.
fn partition(
    ghist: &GHistIndex,
    rows: &mut [u32],
    split: &Candidate,
    scratch: &mut Vec<u32>,
) -> usize {
    let cuts = ghist.cuts();
    let f = split.feature as usize;
    let (fs, fe) = cuts.feature_bins(f);
    let mut left_mask = vec![false; fe - fs];
    if split.cat_bins.is_empty() {
        left_mask[..=split.split_bin - fs].fill(true);
    } else {
        for &b in &split.cat_bins {
            left_mask[b - fs] = true;
        }
    }
    let goes_left = |r: u32| match ghist.feature_bin_at(r as usize, f, fs, fe) {
        Some(bin) => left_mask[bin as usize - fs],
        None => split.default_left,
    };
    scratch.clear();
    let mut n_left = 0;
    for i in 0..rows.len() {
        let r = rows[i];
        if goes_left(r) {
            rows[n_left] = r;
            n_left += 1;
        } else {
            scratch.push(r);
        }
    }
    rows[n_left..].copy_from_slice(scratch);
    n_left
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_is_strict_for_large_or_shallow_numeric_nodes() {
        // Enough rows: no relief at any depth.
        assert_eq!(generalization_floor(1.0, false, 3, 4096), 1.0);
        // Few rows, deep, stable: the full 0.01 relief.
        assert!((generalization_floor(1.0, false, 3, 0) - 0.99).abs() < 1e-12);
        // Relief never pushes the floor below its minimum.
        assert!(generalization_floor(1.0, true, 3, 0) >= 0.984);
    }

    #[test]
    fn stability_rewards_agreeing_fold_weights() {
        assert_eq!(fold_weight_stability(&[0.3; 5]), 1.0);
        let noisy = fold_weight_stability(&[1.0, -1.0, 1.0, -1.0, 1.0]);
        assert!(noisy < 0.6, "{noisy}");
    }
}
