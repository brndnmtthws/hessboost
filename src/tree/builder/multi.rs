//! Vector-leaf histogram tree construction (`multi_strategy =
//! multi_output_tree`), a port of XGBoost 3.4.2's CPU `MultiTargetHistBuilder`
//! and `HistMultiEvaluator`.
//!
//! One tree is grown for all `K` outputs at once: every node keeps a gradient
//! histogram per *split target*, and a candidate split's score is the sum of
//! the per-target regularized scores. Each leaf stores a weight vector (one
//! entry per output). The XGBoost details reproduced here:
//!
//! - a split is valid when the *mean* child Hessian over split targets is
//!   positive and at least `min_child_weight` (`sum_t H_t / K`), while the
//!   stored cover (`sum_hess`, used by SHAP and cover importance) is the plain
//!   sum over targets;
//! - unconstrained scores use the closed form `Tα(G)²/(H+λ)` per target in
//!   `f64`, parents the gain at their `f32` weight; loss changes are compared
//!   in `f32` with XGBoost's tie rule (lower feature index wins);
//! - numeric splits enumerate a forward pass (missing right) and, when the
//!   feature has missing values in the node, a backward pass (missing left);
//! - categorical features with fewer than `max_cat_to_onehot` (4, XGBoost's
//!   default) categories enumerate one-hot splits (each category against the
//!   rest, missing values on either side); larger ones use the partition
//!   search, with categories ordered by the projection of their weight vector
//!   on the parent's (`w_parent · w_category`) and at most `max_cat_threshold`
//!   (64, XGBoost's default) categories on the enumerated side;
//! - monotone constraints bound every target's weight; a child pair that
//!   violates the direction takes the pooled weight of both children instead
//!   of being rejected;
//! - growth follows XGBoost's `Driver` (depth-wise levels in node order,
//!   loss-guided by loss change with ties to the older node, both by
//!   XGBoost's node ids, which categorical splits swap relative to the tree's
//!   storage), and the child with the smaller summed Hessian gets its
//!   histogram built, the sibling by subtraction;
//! - with reduced (split) gradients, leaf weights are refit from the full
//!   value gradients of each leaf's rows once the structure is fixed.
//!
//! Leaf weights are left unscaled; training multiplies them by `eta`.

use super::hist::{partition_rows, rayon_available};
use super::{
    BELOW_ALL_VALUES, BestSplit, InteractionState, LeafRows, MAX_CAT_THRESHOLD, MAX_CAT_TO_ONEHOT,
    build_interaction_sets, limit_or_unbounded, need_replace, next_allowed, permits,
    xgb_calc_weight, xgb_gain_given_weight,
};
use crate::K_RT_EPS_F32;
use crate::config::{GrowPolicy, TrainingParams};
use crate::data::ghist::{Bins, GHistIndex};
use crate::objective::GradPair;
use crate::tree::constraints::MonotoneConstraints;
use crate::tree::gain::{GradStats, RegParams, threshold_l1};
use crate::tree::hist::{feature_slices, subtract_in_place};
use crate::tree::sampler::ColumnSampler;
use crate::tree::{ChildLeaf, RegTree, SplitRule};
use rayon::prelude::*;
use std::cmp::Ordering;

/// XGBoost's `Driver` batch size: at most this many nodes of one depth-wise
/// level are expanded together.
const MAX_NODE_BATCH: usize = 256;

/// Nodes with at least this many rows search their features in parallel.
const PARALLEL_EVALUATE_ROWS: usize = 16_384;

/// Row-feature entries at which a dense histogram is built feature-parallel.
const PARALLEL_HIST_ENTRIES: usize = 65_536;

/// The gradients one vector-leaf tree is grown from, row-major
/// `[row][target]`.
pub(crate) struct VectorGradients<'a> {
    /// Split gradients: they drive the histograms, the split search, and the
    /// internal weights. `n_split` pairs per row.
    pub(crate) split: &'a [GradPair],
    /// Targets per row of `split`.
    pub(crate) n_split: usize,
    /// Full value gradients (`n_outputs` pairs per row) the leaf weights are
    /// refit from (XGBoost's reduced-gradient training), or `None` when the
    /// split gradients are the full gradients.
    pub(crate) value: Option<&'a [GradPair]>,
    /// Outputs per leaf.
    pub(crate) n_outputs: usize,
}

/// Vector-leaf histogram tree builder.
pub(crate) struct MultiTreeBuilder<'a> {
    params: &'a TrainingParams,
    reg: RegParams,
    cons: MonotoneConstraints,
    interaction_sets: Option<Vec<Vec<u32>>>,
}

/// Where a candidate split falls, in XGBoost's orientation.
#[derive(Debug, Clone)]
enum Loc {
    /// Numeric: bins `<= b` go left.
    Bin(usize),
    /// Numeric: every present bin goes right, only missing values left.
    BelowBins,
    /// Categorical: these categories go right (XGBoost's category set).
    Cats(Vec<u32>),
}

/// XGBoost's `SplitEntryContainer` over vector statistics, in XGBoost's
/// orientation: for a categorical split `right` is the category set and
/// `default_left` sends missing values to the other side.
#[derive(Debug, Clone)]
struct Candidate {
    loss_chg: f32,
    feature: u32,
    default_left: bool,
    loc: Loc,
    left: Vec<GradStats>,
    right: Vec<GradStats>,
}

impl Candidate {
    fn none() -> Self {
        Candidate {
            loss_chg: 0.0,
            feature: 0,
            default_left: false,
            loc: Loc::BelowBins,
            left: Vec::new(),
            right: Vec::new(),
        }
    }

    fn update(
        &mut self,
        loss_chg: f32,
        feature: u32,
        default_left: bool,
        loc: impl FnOnce() -> Loc,
        left: &[GradStats],
        right: &[GradStats],
    ) -> bool {
        if !need_replace(self.loss_chg, self.feature, loss_chg, feature) {
            return false;
        }
        self.loss_chg = loss_chg;
        self.feature = feature;
        self.default_left = default_left;
        self.loc = loc();
        self.left.clear();
        self.left.extend_from_slice(left);
        self.right.clear();
        self.right.extend_from_slice(right);
        true
    }

    fn merge(&mut self, other: Candidate) {
        if need_replace(self.loss_chg, self.feature, other.loss_chg, other.feature) {
            *self = other;
        }
    }

    fn is_categorical(&self) -> bool {
        matches!(self.loc, Loc::Cats(_))
    }
}

/// One feature's bins `fs..fe` in the `[bin][target]` split-gradient
/// histogram of node `nid`: the input every split enumeration of the feature
/// scans.
#[derive(Clone, Copy)]
struct FeatureBins<'h> {
    nid: usize,
    hist: &'h [GradStats],
    feature: u32,
    fs: usize,
    fe: usize,
}

/// A node awaiting expansion.
struct Entry {
    nid: usize,
    /// XGBoost's id of this node: `nid`, except that the children of a
    /// categorical split trade ids (the tree stores XGBoost's right child
    /// first). XGBoost's driver orders nodes by this id.
    order: usize,
    depth: usize,
    rows: Vec<u32>,
    /// `[bin][target]` histogram (empty once no longer needed).
    hist: Vec<GradStats>,
    best: Candidate,
    allowed: Option<InteractionState>,
}

/// An expanded node whose children still need histograms and a split search.
struct Expanded {
    entry: Entry,
    /// Tree ids of XGBoost's (left, right) children.
    children: (usize, usize),
    /// Rows routed to XGBoost's (left, right) children.
    rows: (Vec<u32>, Vec<u32>),
    /// Whether the children may be split further (`Driver::IsChildValid`).
    child_valid: bool,
}

/// Mutable state of one tree's growth.
struct Grow<'g, 'a> {
    b: &'g MultiTreeBuilder<'a>,
    ghist: &'g GHistIndex,
    grad: &'g VectorGradients<'g>,
    tree: RegTree,
    /// `[node][split target]` statistics.
    stats: Vec<GradStats>,
    /// Per-node parent gain (`CalcGainGivenWeight` at the node's weight).
    gain: Vec<f64>,
    /// `[node][split target]` unscaled weights.
    weights: Vec<f32>,
    /// `[node][split target]` monotone bounds (only when constrained).
    lower: Vec<f32>,
    upper: Vec<f32>,
    leaf_rows: Vec<LeafRows>,
    num_leaves: usize,
}

impl<'a> MultiTreeBuilder<'a> {
    /// Create a builder bound to a training configuration.
    pub(crate) fn new(params: &'a TrainingParams) -> Self {
        MultiTreeBuilder {
            params,
            reg: RegParams::from_params(params),
            cons: MonotoneConstraints::from_params(&params.monotone_constraints),
            interaction_sets: build_interaction_sets(&params.interaction_constraints),
        }
    }

    /// Grow one vector-leaf tree over `rows` (ascending) and return it with
    /// the rows that reached each leaf. Leaf vectors are unscaled.
    pub(crate) fn build(
        &self,
        ghist: &GHistIndex,
        grad: &VectorGradients,
        rows: &[u32],
        sampler: &mut ColumnSampler,
    ) -> (RegTree, Vec<LeafRows>) {
        let s = grad.n_split;
        debug_assert!(grad.n_outputs > 1 && s >= 1);
        debug_assert_eq!(grad.split.len(), ghist.n_rows() * s);
        debug_assert!(grad.value.is_some() || s == grad.n_outputs);

        // Root statistics: every row's pairs, in row order.
        let mut root = vec![GradStats::default(); s];
        for &r in rows {
            let g = &grad.split[r as usize * s..][..s];
            for (acc, gp) in root.iter_mut().zip(g) {
                acc.add(GradStats::from_pair(*gp));
            }
        }
        // XGBoost sums the target Hessians of the root cover in `f32`.
        let root_hess = root.iter().fold(0.0f32, |acc, t| acc + t.hess as f32);
        let n_bounds = if self.cons.is_active() { s } else { 0 };
        let mut grow = Grow {
            b: self,
            ghist,
            grad,
            tree: RegTree::with_vector_root(grad.n_outputs, root_hess),
            stats: root.clone(),
            gain: Vec::new(),
            weights: Vec::new(),
            lower: vec![-f32::MAX; n_bounds],
            upper: vec![f32::MAX; n_bounds],
            leaf_rows: Vec::new(),
            num_leaves: 1,
        };
        let root_weight: Vec<f32> = (0..s).map(|t| grow.weight(0, t, root[t])).collect();
        grow.gain.push(self.gain_given_weights(&root, &root_weight));
        grow.weights.extend_from_slice(&root_weight);

        let root_hist = grow.build_hist(rows);
        let features = sampler.sample(0);
        let best = grow.evaluate(0, &root_hist, &features, None, rows.len());
        let mut queue = vec![Entry {
            nid: 0,
            order: 0,
            depth: 0,
            rows: rows.to_vec(),
            hist: root_hist,
            best,
            allowed: None,
        }];
        grow.run(&mut queue, sampler);
        for entry in queue {
            grow.record_leaf(entry.nid, entry.rows);
        }
        grow.finish()
    }

    /// Per-target weights' gain summed in `f64` (XGBoost's vector
    /// `CalcGainGivenWeight`).
    fn gain_given_weights(&self, stats: &[GradStats], weights: &[f32]) -> f64 {
        stats.iter().zip(weights).fold(0.0, |gain, (st, &w)| {
            gain + xgb_gain_given_weight(*st, &self.reg, w)
        })
    }
}

impl Grow<'_, '_> {
    fn n_split(&self) -> usize {
        self.grad.n_split
    }

    fn constrained(&self) -> bool {
        !self.lower.is_empty()
    }

    /// XGBoost's bounded `CalcWeight` for target `t` of node `nid`.
    fn weight(&self, nid: usize, t: usize, stats: GradStats) -> f32 {
        self.bound(nid, t, xgb_calc_weight(stats, &self.b.reg) as f32)
    }

    /// XGBoost's `ApplyBounds`.
    fn bound(&self, nid: usize, t: usize, w: f32) -> f32 {
        if !self.constrained() {
            return w;
        }
        let i = nid * self.n_split() + t;
        if w < self.lower[i] {
            self.lower[i]
        } else if w > self.upper[i] {
            self.upper[i]
        } else {
            w
        }
    }

    /// XGBoost's scalar `CalcSplitWeights` for target `t` of a split of
    /// `nid` on a feature with monotone direction `dir`: both children's
    /// bounded weights, or their pooled weight when they violate `dir`.
    fn split_weights(
        &self,
        nid: usize,
        t: usize,
        dir: i8,
        left: GradStats,
        right: GradStats,
    ) -> (f32, f32) {
        let wl = self.weight(nid, t, left);
        let wr = self.weight(nid, t, right);
        if !self.constrained() {
            return (wl, wr);
        }
        let ordered = dir == 0 || (dir > 0 && wl <= wr) || (dir < 0 && wl >= wr);
        if ordered {
            return (wl, wr);
        }
        // Two leaves share one value, each with its own regularization.
        let pooled_reg = RegParams {
            lambda: 2.0 * self.b.reg.lambda,
            alpha: 2.0 * self.b.reg.alpha,
            ..self.b.reg
        };
        let mut both = left;
        both.add(right);
        let pooled = self.bound(nid, t, xgb_calc_weight(both, &pooled_reg) as f32);
        (pooled, pooled)
    }

    /// XGBoost's vector `CalcSplitGain`: the summed child scores, or `-inf`
    /// when the mean child Hessian fails `min_child_weight`.
    fn split_gain(&self, nid: usize, dir: i8, left: &[GradStats], right: &[GradStats]) -> f64 {
        let reg = &self.b.reg;
        let constrained = self.constrained();
        let (mut left_hess, mut right_hess, mut gain) = (0.0f64, 0.0f64, 0.0f64);
        for (l, r) in left.iter().zip(right) {
            left_hess += l.hess;
            right_hess += r.hess;
            if !constrained {
                gain += calc_gain(reg, *l);
                gain += calc_gain(reg, *r);
            }
        }
        let k = left.len() as f64;
        let (lh, rh) = (left_hess / k, right_hess / k);
        let mcw = reg.min_child_weight;
        if !(lh > 0.0 && rh > 0.0 && lh >= mcw && rh >= mcw) {
            return f64::NEG_INFINITY;
        }
        if !constrained {
            return gain;
        }
        for (t, (l, r)) in left.iter().zip(right).enumerate() {
            let (wl, wr) = self.split_weights(nid, t, dir, *l, *r);
            gain += xgb_gain_given_weight(*l, reg, wl);
            gain += xgb_gain_given_weight(*r, reg, wr);
        }
        gain
    }

    fn node_stats(&self, nid: usize) -> &[GradStats] {
        let s = self.n_split();
        &self.stats[nid * s..(nid + 1) * s]
    }

    /// The `[bin][target]` split-gradient histogram of `rows`. Every bin adds
    /// its rows in ascending order, serially or feature-parallel alike.
    fn build_hist(&self, rows: &[u32]) -> Vec<GradStats> {
        let s = self.n_split();
        let ghist = self.ghist;
        let gp = self.grad.split;
        let mut hist = vec![GradStats::default(); ghist.total_bins() * s];
        let add = |slot: &mut [GradStats], g: &[GradPair]| {
            for (h, p) in slot.iter_mut().zip(g) {
                h.grad += f64::from(p.grad);
                h.hess += f64::from(p.hess);
            }
        };
        if let Some(columns) = ghist.column_bins()
            && rows.len().saturating_mul(ghist.n_cols()) >= PARALLEL_HIST_ENTRIES
            && rayon_available()
        {
            let n_rows = ghist.n_rows();
            feature_slices(ghist, &mut hist, s)
                .into_par_iter()
                .enumerate()
                .for_each(|(f, (fs, slice))| {
                    let column = |r: u32| -> usize {
                        match &columns {
                            Bins::U16(c) => usize::from(c[f * n_rows + r as usize]),
                            Bins::U32(c) => c[f * n_rows + r as usize] as usize,
                        }
                    };
                    for &r in rows {
                        let b = column(r) - fs;
                        add(&mut slice[b * s..(b + 1) * s], &gp[r as usize * s..][..s]);
                    }
                });
            return hist;
        }
        let row_ptr = ghist.row_ptr();
        let mut accumulate = |bins: &mut dyn Iterator<Item = usize>, g: &[GradPair]| {
            for b in bins {
                add(&mut hist[b * s..(b + 1) * s], g);
            }
        };
        match ghist.bins() {
            Bins::U16(bins) => {
                for &r in rows {
                    let r = r as usize;
                    accumulate(
                        &mut bins[row_ptr[r]..row_ptr[r + 1]]
                            .iter()
                            .map(|&b| usize::from(b)),
                        &gp[r * s..][..s],
                    );
                }
            }
            Bins::U32(bins) => {
                for &r in rows {
                    let r = r as usize;
                    accumulate(
                        &mut bins[row_ptr[r]..row_ptr[r + 1]].iter().map(|&b| b as usize),
                        &gp[r * s..][..s],
                    );
                }
            }
        }
        hist
    }

    /// XGBoost's `EvaluateSplits` for one node: the best candidate over
    /// `features` (restricted by the interaction constraints).
    fn evaluate(
        &self,
        nid: usize,
        hist: &[GradStats],
        features: &[u32],
        allowed: Option<&InteractionState>,
        n_rows: usize,
    ) -> Candidate {
        let features: Vec<u32> = features
            .iter()
            .copied()
            .filter(|&f| permits(allowed, f))
            .collect();
        let one = |f: u32| {
            let mut best = Candidate::none();
            self.evaluate_feature(nid, hist, f, &mut best);
            best
        };
        let mut best = Candidate::none();
        // Each feature starts from an empty candidate; merging them in
        // feature order picks exactly what one sequential sweep would.
        if n_rows >= PARALLEL_EVALUATE_ROWS && features.len() > 1 && rayon_available() {
            let per_feature: Vec<Candidate> = features.par_iter().map(|&f| one(f)).collect();
            for c in per_feature {
                best.merge(c);
            }
        } else {
            for &f in &features {
                best.merge(one(f));
            }
        }
        best
    }

    fn evaluate_feature(&self, nid: usize, hist: &[GradStats], f: u32, best: &mut Candidate) {
        let cuts = self.ghist.cuts();
        let (fs, fe) = cuts.feature_bins(f as usize);
        if fe <= fs {
            return;
        }
        let bins = FeatureBins {
            nid,
            hist,
            feature: f,
            fs,
            fe,
        };
        if cuts.is_categorical(f as usize) {
            if fe - fs < MAX_CAT_TO_ONEHOT {
                self.enumerate_one_hot(&bins, best);
            } else {
                self.enumerate_partition(&bins, best);
            }
        } else if self.enumerate_numeric(&bins, true, best) {
            self.enumerate_numeric(&bins, false, best);
        }
    }

    /// XGBoost's vector `EnumerateSplit`: the forward pass (`forward`,
    /// missing right) returns whether the feature has missing values in this
    /// node; the backward pass (missing left) runs only then.
    fn enumerate_numeric(&self, bins: &FeatureBins, forward: bool, best: &mut Candidate) -> bool {
        let FeatureBins {
            nid,
            hist,
            feature: f,
            fs,
            fe,
        } = *bins;
        let s = self.n_split();
        let parent = self.node_stats(nid);
        let parent_gain = self.gain[nid];
        let dir = self.b.cons.dir(f as usize);
        let mut acc = vec![GradStats::default(); s];
        let mut rest = vec![GradStats::default(); s];
        let bins: Box<dyn Iterator<Item = usize>> = if forward {
            Box::new(fs..fe)
        } else {
            Box::new((fs..fe).rev())
        };
        for i in bins {
            for t in 0..s {
                acc[t].add(hist[i * s + t]);
                rest[t] = parent[t].sub(acc[t]);
            }
            if forward {
                let loss = (self.split_gain(nid, dir, &acc, &rest) - parent_gain) as f32;
                best.update(loss, f, false, || Loc::Bin(i), &acc, &rest);
            } else {
                let loss = (self.split_gain(nid, dir, &rest, &acc) - parent_gain) as f32;
                let loc = || {
                    if i == fs {
                        Loc::BelowBins
                    } else {
                        Loc::Bin(i - 1)
                    }
                };
                best.update(loss, f, true, loc, &rest, &acc);
            }
        }
        // XGBoost compares the forward sums with the node totals exactly.
        forward && acc.as_slice() != parent
    }

    /// XGBoost's vector `EnumerateOneHot`: every category alone on the right
    /// child, first with missing values left (among the other categories),
    /// then with missing values right (with the category).
    fn enumerate_one_hot(&self, bins: &FeatureBins, best: &mut Candidate) {
        let FeatureBins {
            nid,
            hist,
            feature: f,
            fs,
            fe,
        } = *bins;
        let s = self.n_split();
        let parent = self.node_stats(nid);
        let parent_gain = self.gain[nid];
        let dir = self.b.cons.dir(f as usize);
        let cuts = self.ghist.cuts();
        // Per-target missing statistics: the node total minus every bin.
        let mut missing = parent.to_vec();
        for (t, m) in missing.iter_mut().enumerate() {
            let mut present = GradStats::default();
            for i in fs..fe {
                present.add(hist[i * s + t]);
            }
            *m = m.sub(present);
        }
        let mut left = vec![GradStats::default(); s];
        let mut right = vec![GradStats::default(); s];
        let mut local = Candidate::none();
        for i in fs..fe {
            let cat = || Loc::Cats(vec![cuts.cut_value(i) as u32]);
            for missing_left in [true, false] {
                for t in 0..s {
                    right[t] = hist[i * s + t];
                    if !missing_left {
                        right[t].add(missing[t]);
                    }
                    left[t] = parent[t].sub(right[t]);
                }
                let loss = (self.split_gain(nid, dir, &left, &right) - parent_gain) as f32;
                local.update(loss, f, missing_left, cat, &left, &right);
            }
        }
        if local.is_categorical() {
            best.merge(local);
        }
    }

    /// XGBoost's vector partition search for a categorical feature.
    fn enumerate_partition(&self, bins: &FeatureBins, best: &mut Candidate) {
        let FeatureBins {
            nid, hist, fs, fe, ..
        } = *bins;
        let s = self.n_split();
        let reg = &self.b.reg;
        let parent = self.node_stats(nid);
        let n_bins = fe - fs;
        // Order categories by `w_parent · w_category` (unbounded weights, as
        // XGBoost's `CalcWeightCat`), stably.
        let parent_w: Vec<f32> = parent
            .iter()
            .map(|&p| xgb_calc_weight(p, reg) as f32)
            .collect();
        let scores: Vec<f64> = (0..n_bins)
            .map(|b| {
                (0..s).fold(0.0f64, |sc, t| {
                    let w = xgb_calc_weight(hist[(fs + b) * s + t], reg) as f32;
                    sc + f64::from(parent_w[t] * w)
                })
            })
            .collect();
        let mut sorted: Vec<usize> = (0..n_bins).collect();
        sorted.sort_by(|&l, &r| scores[l].partial_cmp(&scores[r]).unwrap_or(Ordering::Equal));
        for forward in [true, false] {
            self.enumerate_part(bins, &sorted, forward, best);
        }
    }

    /// XGBoost's `EnumeratePart`: the forward direction moves the sorted
    /// prefix to the right child (missing left), the backward direction
    /// accumulates the sorted suffix on the left (missing right).
    fn enumerate_part(
        &self,
        bins: &FeatureBins,
        sorted: &[usize],
        forward: bool,
        best: &mut Candidate,
    ) {
        let FeatureBins {
            nid,
            hist,
            feature: f,
            fs,
            fe,
        } = *bins;
        let s = self.n_split();
        let parent = self.node_stats(nid);
        let parent_gain = self.gain[nid];
        let dir = self.b.cons.dir(f as usize);
        let n_bins_feature = fe - fs;
        let n_bins = MAX_CAT_THRESHOLD.min(n_bins_feature);
        let mut left = vec![GradStats::default(); s];
        let mut right = vec![GradStats::default(); s];
        let mut local = Candidate::none();
        // Number of sorted categories on the right child for the best step.
        let mut best_partition = None;
        for step in 0..n_bins.saturating_sub(1) {
            let j = if forward {
                step
            } else {
                n_bins_feature - 1 - step
            };
            let bin = fs + sorted[j];
            for t in 0..s {
                if forward {
                    right[t].add(hist[bin * s + t]);
                    left[t] = parent[t].sub(right[t]);
                } else {
                    left[t].add(hist[bin * s + t]);
                    right[t] = parent[t].sub(left[t]);
                }
            }
            let loss = (self.split_gain(nid, dir, &left, &right) - parent_gain) as f32;
            if local.update(loss, f, forward, || Loc::BelowBins, &left, &right) {
                best_partition = Some(if forward { step + 1 } else { j });
            }
        }
        if let Some(partition) = best_partition {
            let cuts = self.ghist.cuts();
            let mut cats: Vec<u32> = sorted[..partition]
                .iter()
                .map(|&c| cuts.cut_value(fs + c) as u32)
                .collect();
            cats.sort_unstable();
            local.loc = Loc::Cats(cats);
        }
        if local.is_categorical() {
            best.merge(local);
        }
    }

    /// XGBoost's `ApplyTreeSplit` plus `UpdatePosition` for one entry.
    fn apply(&mut self, entry: Entry, child_valid: bool) -> Expanded {
        let s = self.n_split();
        let nid = entry.nid;
        let best = &entry.best;
        let dir = self.b.cons.dir(best.feature as usize);
        let mut w_left = Vec::with_capacity(s);
        let mut w_right = Vec::with_capacity(s);
        for t in 0..s {
            let (l, r) = self.split_weights(nid, t, dir, best.left[t], best.right[t]);
            w_left.push(l);
            w_right.push(r);
        }
        let (route, (xgb_left, xgb_right)) = self.expand(nid, best);
        self.store_children(nid, dir, best, [&w_left, &w_right], [xgb_left, xgb_right]);

        let categorical = best.is_categorical();
        let (rows_tree_left, rows_tree_right) = partition_rows(self.ghist, &entry.rows, &route);
        let rows = if categorical {
            (rows_tree_right, rows_tree_left)
        } else {
            (rows_tree_left, rows_tree_right)
        };
        Expanded {
            entry: Entry {
                rows: Vec::new(),
                ..entry
            },
            children: (xgb_left, xgb_right),
            rows,
            child_valid,
        }
    }

    /// Expand node `nid` of the tree by `best`: the split the rows are
    /// routed by, and the ids of XGBoost's left and right children.
    /// hessboost's categorical nodes route their set to the tree's left
    /// child, which is XGBoost's right child.
    fn expand(&mut self, nid: usize, best: &Candidate) -> (BestSplit, (usize, usize)) {
        let left_hess: f64 = best.left.iter().map(|g| g.hess).sum();
        let right_hess: f64 = best.right.iter().map(|g| g.hess).sum();
        let mut route = BestSplit::none();
        route.feature = best.feature;
        let (tree_left, tree_right) = match &best.loc {
            Loc::Cats(cats) => {
                route.is_categorical = true;
                route.cat_left.clone_from(cats);
                route.default_left = !best.default_left;
                self.tree.expand(
                    nid,
                    SplitRule::categorical(best.feature, cats, !best.default_left),
                    ChildLeaf::new(0.0, right_hess as f32),
                    ChildLeaf::new(0.0, left_hess as f32),
                )
            }
            loc => {
                let threshold = match loc {
                    Loc::Bin(b) => {
                        route.split_bin = Some(*b);
                        self.ghist.cuts().cut_value(*b)
                    }
                    _ => BELOW_ALL_VALUES,
                };
                route.default_left = best.default_left;
                self.tree.expand(
                    nid,
                    SplitRule::numeric(best.feature, threshold, best.default_left),
                    ChildLeaf::new(0.0, left_hess as f32),
                    ChildLeaf::new(0.0, right_hess as f32),
                )
            }
        };
        self.tree.set_split_gain(nid, best.loss_chg);
        self.tree.set_sum_hess(nid, (left_hess + right_hess) as f32);
        let children = if best.is_categorical() {
            (tree_right, tree_left)
        } else {
            (tree_left, tree_right)
        };
        (route, children)
    }

    /// Per-node state of the new children `[left, right]` (XGBoost's
    /// orientation) of `nid`: statistics, weights `[w_left, w_right]`, gains,
    /// and, under monotone constraints, the weight bounds split at the
    /// children's midpoint by direction `dir`.
    fn store_children(
        &mut self,
        nid: usize,
        dir: i8,
        best: &Candidate,
        [w_left, w_right]: [&[f32]; 2],
        [xgb_left, xgb_right]: [usize; 2],
    ) {
        let s = self.n_split();
        let n_nodes = self.tree.num_nodes();
        self.stats.resize(n_nodes * s, GradStats::default());
        self.weights.resize(n_nodes * s, 0.0);
        self.gain.resize(n_nodes, 0.0);
        let gl = self.b.gain_given_weights(&best.left, w_left);
        let gr = self.b.gain_given_weights(&best.right, w_right);
        for (child, stats, w, gain) in [
            (xgb_left, &best.left, w_left, gl),
            (xgb_right, &best.right, w_right, gr),
        ] {
            self.stats[child * s..(child + 1) * s].copy_from_slice(stats);
            self.weights[child * s..(child + 1) * s].copy_from_slice(w);
            self.gain[child] = gain;
        }
        if self.constrained() {
            self.lower.resize(n_nodes * s, -f32::MAX);
            self.upper.resize(n_nodes * s, f32::MAX);
            for t in 0..s {
                let (lo, hi) = (self.lower[nid * s + t], self.upper[nid * s + t]);
                let mid = w_left[t] + 0.5 * (w_right[t] - w_left[t]);
                let (mut l_lo, mut l_hi, mut r_lo, mut r_hi) = (lo, hi, lo, hi);
                if dir < 0 {
                    l_lo = mid;
                    r_hi = mid;
                } else if dir > 0 {
                    l_hi = mid;
                    r_lo = mid;
                }
                self.lower[xgb_left * s + t] = l_lo;
                self.upper[xgb_left * s + t] = l_hi;
                self.lower[xgb_right * s + t] = r_lo;
                self.upper[xgb_right * s + t] = r_hi;
            }
        }
    }

    /// Children histograms (smaller summed Hessian built, sibling by
    /// subtraction) and split searches of one expanded node, as XGBoost's
    /// `AssignNodes` + `BuildHistLeftRight` + `EvaluateSplits`. `features`
    /// are sampled in XGBoost's child order (left, then right); the children
    /// are returned in the same order.
    fn children(&self, e: Expanded, features: [Vec<u32>; 2]) -> [Entry; 2] {
        let Expanded {
            entry,
            children: (xl, xr),
            rows: (rows_l, rows_r),
            ..
        } = e;
        let b = &entry.best;
        let left_hess: f64 = b.left.iter().map(|g| g.hess).sum();
        let right_hess: f64 = b.right.iter().map(|g| g.hess).sum();
        let mut sibling = entry.hist;
        let (hist_l, hist_r) = if right_hess < left_hess {
            let built = self.build_hist(&rows_r);
            subtract_in_place(&mut sibling, &built);
            (sibling, built)
        } else {
            let built = self.build_hist(&rows_l);
            subtract_in_place(&mut sibling, &built);
            (built, sibling)
        };
        let allowed = next_allowed(
            entry.allowed.as_ref(),
            b.feature,
            self.b.interaction_sets.as_deref(),
        );
        let [f_l, f_r] = features;
        let n_rows = rows_l.len() + rows_r.len();
        let eval_l = || self.evaluate(xl, &hist_l, &f_l, allowed.as_ref(), rows_l.len());
        let eval_r = || self.evaluate(xr, &hist_r, &f_r, allowed.as_ref(), rows_r.len());
        let (best_l, best_r) = if n_rows >= PARALLEL_EVALUATE_ROWS && rayon_available() {
            rayon::join(eval_l, eval_r)
        } else {
            (eval_l(), eval_r())
        };
        let depth = entry.depth + 1;
        // Expansions run in XGBoost's order, so both trees give each
        // expansion's children the same pair of consecutive ids, XGBoost's
        // left child the smaller one.
        let left = Entry {
            nid: xl,
            order: xl.min(xr),
            depth,
            rows: rows_l,
            hist: hist_l,
            best: best_l,
            allowed: allowed.clone(),
        };
        let right = Entry {
            nid: xr,
            order: xl.max(xr),
            depth,
            rows: rows_r,
            hist: hist_r,
            best: best_r,
            allowed,
        };
        [left, right]
    }

    /// XGBoost's `IsValidExpandEntry`.
    fn expandable(&self, e: &Entry) -> bool {
        let loss = e.best.loss_chg;
        !(loss <= K_RT_EPS_F32
            || loss < self.b.params.gamma as f32
            || e.depth == limit_or_unbounded(self.b.params.max_depth)
            || (self.b.params.max_leaves > 0 && self.num_leaves == self.b.params.max_leaves))
    }

    /// XGBoost's `Driver::IsChildValid`.
    fn child_valid(&self, e: &Entry) -> bool {
        !(e.depth + 1 >= limit_or_unbounded(self.b.params.max_depth)
            || (self.b.params.max_leaves > 0 && self.num_leaves >= self.b.params.max_leaves))
    }

    /// XGBoost's `Driver::Pop`. Popped entries that cannot expand become
    /// leaves.
    fn pop(&mut self, queue: &mut Vec<Entry>) -> Vec<Entry> {
        if queue.is_empty() {
            return Vec::new();
        }
        if self.b.params.grow_policy == GrowPolicy::LossGuide {
            let mut top = 0;
            for (i, e) in queue.iter().enumerate().skip(1) {
                let t = &queue[top];
                if e.best.loss_chg > t.best.loss_chg
                    || (e.best.loss_chg == t.best.loss_chg && e.order < t.order)
                {
                    top = i;
                }
            }
            let e = queue.swap_remove(top);
            if self.expandable(&e) {
                self.num_leaves += 1;
                return vec![e];
            }
            self.record_leaf(e.nid, e.rows);
            return Vec::new();
        }
        // Depth-wise: the lowest XGBoost node ids first, one level (and at
        // most `MAX_NODE_BATCH` expandable nodes) at a time.
        queue.sort_by_key(|e| std::cmp::Reverse(e.order));
        let level = queue[queue.len() - 1].depth;
        let mut result = Vec::new();
        while let Some(e) = queue.pop_if(|e| e.depth == level) {
            if self.expandable(&e) {
                self.num_leaves += 1;
                result.push(e);
            } else {
                self.record_leaf(e.nid, e.rows);
            }
            if result.len() >= MAX_NODE_BATCH {
                break;
            }
        }
        result
    }

    /// XGBoost's `UpdateTree` loop over the driver queue.
    fn run(&mut self, queue: &mut Vec<Entry>, sampler: &mut ColumnSampler) {
        let mut batch = self.pop(queue);
        while !batch.is_empty() {
            let mut expanded = Vec::with_capacity(batch.len());
            for entry in batch {
                let valid = self.child_valid(&entry);
                expanded.push(self.apply(entry, valid));
            }
            let mut work = Vec::new();
            for e in expanded {
                if e.child_valid {
                    let depth = e.entry.depth + 1;
                    let features = [sampler.sample(depth), sampler.sample(depth)];
                    work.push((e, features));
                } else {
                    let (xl, xr) = e.children;
                    let (rl, rr) = e.rows;
                    self.record_leaf(xl, rl);
                    self.record_leaf(xr, rr);
                }
            }
            let parallel = work.len() > 1 && rayon_available();
            let this = &*self;
            let children: Vec<[Entry; 2]> = if parallel {
                work.into_par_iter()
                    .map(|(e, f)| this.children(e, f))
                    .collect()
            } else {
                work.into_iter().map(|(e, f)| this.children(e, f)).collect()
            };
            for child in children.into_iter().flatten() {
                if child.best.loss_chg > K_RT_EPS_F32 {
                    queue.push(child);
                } else {
                    self.record_leaf(child.nid, child.rows);
                }
            }
            batch = self.pop(queue);
        }
    }

    fn record_leaf(&mut self, node: usize, rows: Vec<u32>) {
        self.leaf_rows.push(LeafRows { node, rows });
    }

    /// Write every leaf's weight vector: the split weights, or (reduced
    /// gradients) weights refit from the leaf rows' value gradients.
    fn finish(mut self) -> (RegTree, Vec<LeafRows>) {
        self.leaf_rows.sort_by_key(|l| l.node);
        let s = self.n_split();
        let k = self.grad.n_outputs;
        match self.grad.value {
            None => {
                for leaf in &self.leaf_rows {
                    let w = &self.weights[leaf.node * s..(leaf.node + 1) * s];
                    self.tree.set_leaf_vector(leaf.node, w);
                }
            }
            Some(value) => {
                let mut sums = vec![GradStats::default(); k];
                let mut w = vec![0.0f32; k];
                for leaf in &self.leaf_rows {
                    sums.fill(GradStats::default());
                    for &r in &leaf.rows {
                        for (acc, gp) in sums.iter_mut().zip(&value[r as usize * k..][..k]) {
                            acc.add(GradStats::from_pair(*gp));
                        }
                    }
                    for (w, st) in w.iter_mut().zip(&sums) {
                        *w = xgb_calc_weight(*st, &self.b.reg) as f32;
                    }
                    self.tree.set_leaf_vector(leaf.node, &w);
                }
            }
        }
        (self.tree, self.leaf_rows)
    }
}

/// XGBoost's `CalcGain` in `f64`: the closed form `Tα(G)²/(H+λ)`, or the
/// gain at the (`f64`) clamped weight when `max_delta_step` is set.
fn calc_gain(reg: &RegParams, st: GradStats) -> f64 {
    if st.hess <= 0.0 {
        return 0.0;
    }
    if reg.max_delta_step == 0.0 {
        let t = threshold_l1(st.grad, reg.alpha);
        t * t / (st.hess + reg.lambda)
    } else {
        let w = xgb_calc_weight(st, reg);
        -(2.0 * st.grad * w + (st.hess + reg.lambda) * (w * w) + 2.0 * reg.alpha * w.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DMatrix;
    use crate::tree::builder::{HistTreeBuilder, all_rows, test_support};

    fn binned(x: &[f32], n: usize, cols: usize) -> GHistIndex {
        test_support::binned(&DMatrix::from_dense(x, n, cols).unwrap(), 256)
    }

    fn build(
        params: &TrainingParams,
        ghist: &GHistIndex,
        split: &[GradPair],
        n_split: usize,
        value: Option<&[GradPair]>,
        n_outputs: usize,
    ) -> (RegTree, Vec<LeafRows>) {
        let grad = VectorGradients {
            split,
            n_split,
            value,
            n_outputs,
        };
        MultiTreeBuilder::new(params).build(
            ghist,
            &grad,
            &all_rows(ghist.n_rows()),
            &mut ColumnSampler::all(ghist.n_cols()),
        )
    }

    /// Two targets that each depend on a different feature.
    fn two_target_data(n: usize) -> (GHistIndex, Vec<GradPair>) {
        let mut x = Vec::new();
        let mut g = Vec::new();
        for i in 0..n {
            let a = (i % 17) as f32 / 17.0;
            let b = (i % 11) as f32 / 11.0;
            x.extend([a, b]);
            g.push(GradPair::new(if a < 0.5 { -1.0 } else { 1.0 }, 1.0));
            g.push(GradPair::new(if b < 0.3 { 2.0 } else { -0.5 }, 1.0));
        }
        (binned(&x, n, 2), g)
    }

    /// A single target's vector tree equals the scalar hist tree: same
    /// structure, and every leaf weight matches the scalar leaf value (the
    /// vector path's closed-form gains pick the same splits here).
    #[test]
    fn duplicated_target_matches_scalar_structure() {
        let (ghist, g2) = two_target_data(400);
        let g1: Vec<GradPair> = g2.iter().step_by(2).copied().collect();
        let dup: Vec<GradPair> = g1.iter().flat_map(|&g| [g, g]).collect();
        let params = TrainingParams::builder().max_depth(3).build().unwrap();
        let (vector, _) = build(&params, &ghist, &dup, 2, None, 2);
        let scalar = HistTreeBuilder::new(&params).build(
            &ghist,
            &g1,
            &all_rows(400),
            &mut ColumnSampler::all(2),
        );
        assert_eq!(vector.num_nodes(), scalar.num_nodes());
        for r in 0..400usize {
            let row = [(r % 17) as f32 / 17.0, (r % 11) as f32 / 11.0];
            let vl = vector.leaf_id_dense(&row, f32::NAN);
            let sl = scalar.leaf_id_dense(&row, f32::NAN);
            let w = vector.leaf_vector(vl);
            assert_eq!(w[0], w[1]);
            assert_eq!(w[0], scalar.node(sl).leaf_value);
        }
    }

    /// The shared structure serves both targets: each target's own feature
    /// is split on, and leaf vectors differ per target.
    #[test]
    fn splits_serve_every_target() {
        let (ghist, g) = two_target_data(400);
        let params = TrainingParams::builder().max_depth(2).build().unwrap();
        let (tree, leaves) = build(&params, &ghist, &g, 2, None, 2);
        let features: Vec<u32> = tree
            .nodes()
            .iter()
            .filter(|n| !n.is_leaf())
            .map(|n| n.split_feature)
            .collect();
        assert!(features.contains(&0) && features.contains(&1));
        // Leaf rows partition every row exactly once.
        let mut all: Vec<u32> = leaves.iter().flat_map(|l| l.rows.clone()).collect();
        all.sort_unstable();
        assert_eq!(all, (0..400).collect::<Vec<_>>());
        // Cover is the summed Hessian over targets.
        assert_eq!(tree.node(0).sum_hess, 800.0);
    }

    /// `min_child_weight` applies to the mean Hessian over targets: with a
    /// Hessian of 1 on one target and 0 on the other, a child of 30 rows has
    /// mean Hessian 15, so `min_child_weight = 20` forbids a 30-row child
    /// that a summed rule (30) would allow.
    #[test]
    fn min_child_weight_uses_mean_hessian() {
        let n = 60;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let ghist = binned(&x, n, 1);
        let g: Vec<GradPair> = (0..n)
            .flat_map(|i| {
                let gr = if i < 30 { -1.0 } else { 1.0 };
                [GradPair::new(gr, 1.0), GradPair::new(0.0, 0.0)]
            })
            .collect();
        let loose = TrainingParams::builder()
            .max_depth(1)
            .min_child_weight(15.0)
            .build()
            .unwrap();
        let (split, _) = build(&loose, &ghist, &g, 2, None, 2);
        assert_eq!(split.num_nodes(), 3);
        let strict = TrainingParams::builder()
            .max_depth(1)
            .min_child_weight(20.0)
            .build()
            .unwrap();
        let (stump, _) = build(&strict, &ghist, &g, 2, None, 2);
        assert_eq!(stump.num_nodes(), 1);
    }

    /// Reduced gradients: the structure follows the split gradient, while
    /// the leaf vectors are refit from the value gradients.
    #[test]
    fn reduced_gradients_refit_leaves_from_values() {
        let (ghist, g) = two_target_data(400);
        // Split on the mean of both targets.
        let mean: Vec<GradPair> = g
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[a, b]| GradPair::new(f32::midpoint(a.grad, b.grad), 1.0))
            .collect();
        let params = TrainingParams::builder()
            .max_depth(2)
            .lambda(0.0)
            .build()
            .unwrap();
        let (tree, leaves) = build(&params, &ghist, &mean, 1, Some(&g), 2);
        assert_eq!(tree.size_leaf_vector(), 2);
        for leaf in &leaves {
            let n = leaf.rows.len() as f32;
            for t in 0..2 {
                let sum: f32 = leaf.rows.iter().map(|&r| g[r as usize * 2 + t].grad).sum();
                let w = tree.leaf_vector(leaf.node)[t];
                assert!((w + sum / n).abs() < 1e-5, "leaf {} target {t}", leaf.node);
            }
        }
    }

    /// Loss-guided growth honors `max_leaves`.
    #[test]
    fn lossguide_respects_max_leaves() {
        let (ghist, g) = two_target_data(400);
        let params = TrainingParams::builder()
            .grow_policy(GrowPolicy::LossGuide)
            .max_depth(0)
            .max_leaves(3)
            .build()
            .unwrap();
        let (tree, leaves) = build(&params, &ghist, &g, 2, None, 2);
        assert_eq!(tree.num_leaves(), 3);
        assert_eq!(leaves.len(), 3);
    }
}
