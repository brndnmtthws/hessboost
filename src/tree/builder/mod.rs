//! Tree construction algorithms.
//!
//! Each builder grows a single [`crate::tree::RegTree`] from per-instance
//! gradients. The exact builder is the reference; the histogram builder shares
//! the same regularized gain math.

pub(crate) mod budget;
mod exact;
mod hist;
mod lightgbm;
mod multi;
mod oblivious;

pub(crate) use exact::{ExactTreeBuilder, SortedColumns, all_rows};
pub use hist::HistTreeBuilder;
pub(crate) use hist::LeafRows;
pub(crate) use multi::{MultiTreeBuilder, VectorGradients};
pub(crate) use oblivious::check_symmetric_input;

use std::collections::BTreeSet;

use crate::K_RT_EPS;
use crate::objective::GradPair;
use crate::tree::constraints::{
    Bounds, calc_weight_bounded, child_bounds, gain_at_weight, satisfies,
};
use crate::tree::gain::{GradStats, RegParams, calc_gain, threshold_l1};
use crate::tree::reuse::CategoricalPenalty;
use crate::tree::{ChildLeaf, RegTree, SplitRule};

/// The bound set by a `max_depth` / `max_leaves` style parameter, where `0`
/// means unlimited.
pub(super) fn limit_or_unbounded(limit: usize) -> usize {
    if limit == 0 { usize::MAX } else { limit }
}

/// The best split found so far for one node.
///
/// Both builders share this. `threshold` is the exact split value while
/// `split_bin` is the histogram global-bin boundary (`Some(b)`: bins `<= b` go
/// left; `None`: no present bin goes left, only missing values); each builder
/// writes its own location field and leaves the other at its default.
#[derive(Debug, Clone)]
pub(super) struct BestSplit {
    pub(super) loss_chg: f64,
    pub(super) feature: u32,
    pub(super) threshold: f32,
    pub(super) split_bin: Option<usize>,
    pub(super) default_left: bool,
    pub(super) left: GradStats,
    pub(super) right: GradStats,
    /// Bounded child weights (used to derive monotone child bounds).
    pub(super) w_left: f64,
    pub(super) w_right: f64,
    /// Whether this is a categorical (set-membership) split.
    pub(super) is_categorical: bool,
    /// For a categorical split, the category values routed left.
    pub(super) cat_left: Vec<u32>,
    /// Whether the children are XGBoost's children swapped: the XGBoost
    /// categorical search scores its set as the right child and records it
    /// as the tree's left one, so monotone bounds follow its orientation.
    pub(super) children_swapped: bool,
}

impl BestSplit {
    pub(super) fn none() -> Self {
        BestSplit {
            loss_chg: 0.0,
            feature: 0,
            threshold: 0.0,
            split_bin: None,
            default_left: true,
            left: GradStats::default(),
            right: GradStats::default(),
            w_left: 0.0,
            w_right: 0.0,
            is_categorical: false,
            cat_left: Vec::new(),
            children_swapped: false,
        }
    }

    /// A numeric split candidate; `pos` carries the split location (value-space
    /// threshold for exact search, global-bin boundary for histogram search).
    pub(super) fn numeric(feature: u32, pos: SplitPos, children: Children, score: Score) -> Self {
        let (threshold, split_bin) = match pos {
            SplitPos::Value(t) => (t, None),
            SplitPos::Bin(b) => (0.0, Some(b)),
            SplitPos::BelowBins => (0.0, None),
        };
        BestSplit {
            loss_chg: score.loss_chg,
            feature,
            threshold,
            split_bin,
            default_left: children.default_left,
            left: children.left,
            right: children.right,
            w_left: score.w_left,
            w_right: score.w_right,
            is_categorical: false,
            cat_left: Vec::new(),
            children_swapped: false,
        }
    }

    /// A categorical (set-membership) split candidate; `cat_left` holds the
    /// category values routed left, and `children.default_left` says where
    /// missing values go.
    pub(super) fn categorical(
        feature: u32,
        children: Children,
        score: Score,
        cat_left: Vec<u32>,
    ) -> Self {
        BestSplit {
            loss_chg: score.loss_chg,
            feature,
            threshold: 0.0,
            split_bin: None,
            default_left: children.default_left,
            left: children.left,
            right: children.right,
            w_left: score.w_left,
            w_right: score.w_right,
            is_categorical: true,
            cat_left,
            children_swapped: false,
        }
    }

    #[inline]
    pub(super) fn found(&self) -> bool {
        self.loss_chg > K_RT_EPS
    }

    /// Expand node `nid` of `tree` by this split (a numeric split at
    /// `threshold`), the children holding the bounded child weights as `f32`,
    /// and record the split's loss change. Returns the child ids.
    pub(super) fn expand(&self, tree: &mut RegTree, nid: usize, threshold: f32) -> (usize, usize) {
        let split = if self.is_categorical {
            SplitRule::categorical(self.feature, &self.cat_left, self.default_left)
        } else {
            SplitRule::numeric(self.feature, threshold, self.default_left)
        };
        let ids = tree.expand(
            nid,
            split,
            ChildLeaf::new(self.w_left as f32, self.left.hess as f32),
            ChildLeaf::new(self.w_right as f32, self.right.hess as f32),
        );
        tree.set_split_gain(nid, self.loss_chg as f32);
        ids
    }

    /// Whether this split should be taken: it was found, its loss change
    /// reaches `gamma` (XGBoost rejects `loss_chg < min_split_loss`), and both
    /// children have positive cover and meet `min_child_weight`.
    pub(super) fn valid(&self, gamma: f64, min_child_weight: f64) -> bool {
        self.found()
            && self.loss_chg >= gamma
            && children_valid(self.left, self.right, min_child_weight)
    }

    /// Monotone bounds of this split's children (left, right), derived in
    /// the orientation the split was scored in.
    pub(super) fn child_bounds(&self, parent: Bounds, dir: i8) -> (Bounds, Bounds) {
        if self.children_swapped {
            let (xgb_left, xgb_right) = child_bounds(parent, dir, self.w_right, self.w_left);
            (xgb_right, xgb_left)
        } else {
            child_bounds(parent, dir, self.w_left, self.w_right)
        }
    }
}

/// XGBoost's child validity: both children have positive Hessian and meet
/// `min_child_weight`.
#[inline]
pub(super) fn children_valid(left: GradStats, right: GradStats, min_child_weight: f64) -> bool {
    left.hess > 0.0
        && right.hess > 0.0
        && left.hess >= min_child_weight
        && right.hess >= min_child_weight
}

/// A candidate partition of a node's rows: where missing values go and the
/// gradient statistics of both children.
#[derive(Debug, Clone, Copy)]
pub(super) struct Children {
    pub(super) default_left: bool,
    pub(super) left: GradStats,
    pub(super) right: GradStats,
}

impl Children {
    #[inline]
    pub(super) fn new(default_left: bool, left: GradStats, right: GradStats) -> Self {
        Children {
            default_left,
            left,
            right,
        }
    }

    /// The same partition seen from the other side: children exchanged and
    /// missing values routed the other way.
    #[inline]
    fn swapped(self) -> Self {
        Children::new(!self.default_left, self.right, self.left)
    }
}

/// The score of a candidate partition: its loss change and both children's
/// bounded weights. `Score<f32>` is XGBoost's `f32` split arithmetic;
/// `Score` (`f64`) is what a [`BestSplit`] records.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Score<T = f64> {
    pub(super) loss_chg: T,
    pub(super) w_left: T,
    pub(super) w_right: T,
}

impl<T> Score<T> {
    /// The score of [`Children::swapped`].
    #[inline]
    fn swapped(self) -> Self {
        Score {
            loss_chg: self.loss_chg,
            w_left: self.w_right,
            w_right: self.w_left,
        }
    }
}

impl From<Score<f32>> for Score {
    #[inline]
    fn from(score: Score<f32>) -> Self {
        Score {
            loss_chg: f64::from(score.loss_chg),
            w_left: f64::from(score.w_left),
            w_right: f64::from(score.w_right),
        }
    }
}

/// Everything a candidate's score depends on besides its children: the
/// regularization, the node's `root_gain` baseline ([`xgb_node_gain`]) and
/// monotone bounds, and the candidate feature's monotone direction.
#[derive(Debug, Clone, Copy)]
pub(super) struct SplitScorer<'a> {
    pub(super) reg: &'a RegParams,
    pub(super) root_gain: f32,
    pub(super) bounds: Bounds,
    pub(super) dir: i8,
}

impl SplitScorer<'_> {
    /// XGBoost's scalar `SplitEvaluator::CalcSplitGain` minus the parent's
    /// `root_gain`, i.e. the `loss_chg` a candidate is compared and stored
    /// with. Returns `None` when the split is invalid (a child without
    /// positive Hessian or below `min_child_weight`) or violates the monotone
    /// direction, and otherwise the `f32` loss change plus both bounded child
    /// weights.
    #[inline]
    pub(super) fn loss_chg(&self, left: GradStats, right: GradStats) -> Option<Score<f32>> {
        let reg = self.reg;
        if !children_valid(left, right, reg.min_child_weight) {
            return None;
        }
        let wl = xgb_weight(left, reg, self.bounds);
        let wr = xgb_weight(right, reg, self.bounds);
        if !satisfies(self.dir, f64::from(wl), f64::from(wr)) {
            return None;
        }
        // Upstream's scalar `CalcGainGivenWeight` returns `float`: each child's
        // score is rounded before the two are added in `f32`.
        let gain = xgb_gain_given_weight(left, reg, wl) as f32
            + xgb_gain_given_weight(right, reg, wr) as f32;
        Some(Score {
            loss_chg: gain - self.root_gain,
            w_left: wl,
            w_right: wr,
        })
    }

    /// Gain of one candidate split in `f64` against the `root_gain` baseline,
    /// plus its bounded child weights, or `None` when a child is below
    /// `min_child_weight` or the monotone direction is violated. Unconstrained
    /// builds take the cheap closed-form path (weights unused).
    #[inline]
    pub(super) fn candidate_gain(
        &self,
        left: GradStats,
        right: GradStats,
        constrained: bool,
    ) -> Option<Score> {
        let reg = self.reg;
        if left.hess < reg.min_child_weight || right.hess < reg.min_child_weight {
            return None;
        }
        let parent = f64::from(self.root_gain);
        if constrained {
            let wl = calc_weight_bounded(left, reg, self.bounds);
            let wr = calc_weight_bounded(right, reg, self.bounds);
            if !satisfies(self.dir, wl, wr) {
                return None;
            }
            let g = gain_at_weight(left, reg, wl) + gain_at_weight(right, reg, wr) - parent;
            Some(Score {
                loss_chg: g,
                w_left: wl,
                w_right: wr,
            })
        } else {
            let g = calc_gain(left, reg) + calc_gain(right, reg) - parent;
            Some(Score {
                loss_chg: g,
                w_left: 0.0,
                w_right: 0.0,
            })
        }
    }
}

/// Every numeric boundary of one feature's histogram `bins` (global bins
/// from `first`) in XGBoost's order: a forward pass over every boundary
/// (bins `<= b` left, missing values right, including the last boundary that
/// isolates the missing mass) and, only when the feature has missing values
/// in the node, a backward pass (bins `>= b` right, missing values left).
/// The backward pass ends at `BelowBins` (XGBoost's `NumericBinLowerBound`
/// at the feature's first bin), which puts only the missing mass left. Its
/// children are the forward pass's last boundary swapped, so it is distinct
/// under a monotone constraint: the direction can reject one orientation and
/// accept the other. `offer(pos, children)` sees every candidate; a `dense`
/// index has no missing values.
#[inline]
pub(super) fn for_each_numeric_split(
    bins: &[GradStats],
    first: usize,
    total: GradStats,
    dense: bool,
    mut offer: impl FnMut(SplitPos, Children),
) {
    let mut acc = GradStats::default();
    for (offset, &bin) in bins.iter().enumerate() {
        acc.add(bin);
        offer(
            SplitPos::Bin(first + offset),
            Children::new(false, acc, total.sub(acc)),
        );
    }
    // XGBoost compares the forward pass's final sum with the node statistics
    // exactly (`SplitContainsMissingValues`).
    if dense || acc == total {
        return;
    }
    let mut suffix = GradStats::default();
    for offset in (0..bins.len()).rev() {
        suffix.add(bins[offset]);
        let pos = if offset == 0 {
            SplitPos::BelowBins
        } else {
            SplitPos::Bin(first + offset - 1)
        };
        offer(pos, Children::new(true, total.sub(suffix), suffix));
    }
}

/// Where a numeric candidate split falls. Exact search records the threshold
/// in value space; histogram search records the global-bin boundary (bins
/// `<= split_bin` go left) or `BelowBins`, XGBoost's backward-pass endpoint
/// (`NumericBinLowerBound` at the feature's first bin, `-inf`): every present
/// bin goes right and only missing values go left.
#[derive(Debug, Clone, Copy)]
pub(super) enum SplitPos {
    Value(f32),
    Bin(usize),
    BelowBins,
}

/// A finite threshold below every finite feature value: `x < BELOW_ALL_VALUES`
/// is false for all finite `x`, so a split at it routes every present row
/// right and only missing values (`default_left`) left. Stands in for the
/// `-inf` XGBoost stores (`NumericBinLowerBound` at a feature's first bin, or
/// an overflowed `ColMaker` endpoint) because trees here require a finite
/// `split_cond`.
pub(super) const BELOW_ALL_VALUES: f32 = f32::MIN;

/// XGBoost's `CalcWeight` in `f64`: `−Tα(G)/(H+λ)`, `0` without positive
/// Hessian, clamped to `max_delta_step` when set.
#[inline]
pub(super) fn xgb_calc_weight(stats: GradStats, reg: &RegParams) -> f64 {
    if stats.hess <= 0.0 {
        return 0.0;
    }
    let mut w = -threshold_l1(stats.grad, reg.alpha) / (stats.hess + reg.lambda);
    if reg.max_delta_step != 0.0 && w.abs() > reg.max_delta_step {
        w = reg.max_delta_step.copysign(w);
    }
    w
}

/// XGBoost's `SplitEvaluator::CalcWeight`: [`xgb_calc_weight`] rounded to
/// `f32`, then clamped to the node's monotone bounds. The `f32` rounding
/// happens before bounding, exactly as upstream.
#[inline]
pub(super) fn xgb_weight(stats: GradStats, reg: &RegParams, bounds: Bounds) -> f32 {
    let w = xgb_calc_weight(stats, reg) as f32;
    let (lower, upper) = (bounds.lower as f32, bounds.upper as f32);
    if w < lower {
        lower
    } else if w > upper {
        upper
    } else {
        w
    }
}

/// XGBoost's `CalcGainGivenWeight` with an `f32` weight: `−(2Gw + (H+λ)w² +
/// 2α|w|)` where `w²` is formed in `f32` (upstream `Sqr(float)`) and every
/// other operation runs in `f64`.
#[inline]
pub(super) fn xgb_gain_given_weight(stats: GradStats, reg: &RegParams, w: f32) -> f64 {
    -(2.0 * stats.grad * f64::from(w)
        + (stats.hess + reg.lambda) * f64::from(w * w)
        + 2.0 * reg.alpha * f64::from(w.abs()))
}

/// XGBoost's scalar `TreeEvaluator::CalcGain` for a node: the given-weight
/// gain at the `f32` (bounded) weight, rounded to `f32` as upstream stores
/// `root_gain`. The histogram and exact updaters both use this form, so the
/// parent baseline carries the same `f32` weight rounding as every candidate.
pub(super) fn xgb_node_gain(stats: GradStats, reg: &RegParams, bounds: Bounds) -> f32 {
    if stats.hess <= 0.0 {
        return 0.0;
    }
    xgb_gain_given_weight(stats, reg, xgb_weight(stats, reg, bounds)) as f32
}

/// XGBoost's `SplitEntry::NeedReplace`: a candidate replaces the incumbent
/// when its loss change is strictly better, or equal on a lower feature index.
/// Infinite loss changes are never taken.
pub(super) fn need_replace(
    incumbent: f32,
    incumbent_feature: u32,
    loss_chg: f32,
    feature: u32,
) -> bool {
    if loss_chg.is_infinite() {
        false
    } else if incumbent_feature <= feature {
        loss_chg > incumbent
    } else {
        incumbent.partial_cmp(&loss_chg) != Some(std::cmp::Ordering::Greater)
    }
}

/// XGBoost's `SplitEntry::Update`: replace the incumbent when
/// [`need_replace`] says so. `best.loss_chg` holds an `f32` value.
pub(super) fn xgb_update(
    best: &mut BestSplit,
    feature: u32,
    pos: SplitPos,
    children: Children,
    score: Score<f32>,
) -> bool {
    let replace = need_replace(best.loss_chg as f32, best.feature, score.loss_chg, feature);
    if replace {
        *best = BestSplit::numeric(feature, pos, children, score.into());
    }
    replace
}

/// XGBoost's default `max_cat_to_onehot`: categorical features with fewer
/// categories enumerate one-hot splits instead of partitions.
pub(super) const MAX_CAT_TO_ONEHOT: usize = 4;

/// XGBoost's default `max_cat_threshold`: the most categories one side of a
/// partition split enumerates.
pub(super) const MAX_CAT_THRESHOLD: usize = 64;

/// The best categorical split of one feature, in XGBoost's orientation: its
/// set of categories goes to the right child, `children.default_left` routes
/// missing values.
struct CatCandidate {
    children: Children,
    score: Score<f32>,
}

/// XGBoost's scalar categorical split search (`HistEvaluator`), shared by the
/// histogram and exact builders. `cats` holds every category of `feature` in
/// ascending order with its node statistics (zero when the node has none);
/// `total` is the node's statistics, including missing values.
///
/// - Fewer than [`MAX_CAT_TO_ONEHOT`] categories (`EnumerateOneHot`): each
///   category alone on one side, with missing values on either side.
/// - Otherwise (`EnumeratePart`): categories stably sorted by their weight
///   (`CalcWeightCat`), then scanned forward (a growing prefix of the order
///   on one side, missing values on the other) and backward (a growing
///   suffix on the other side, missing values with it), at most
///   [`MAX_CAT_THRESHOLD`] categories deep.
///
/// Candidates are scored with [`SplitScorer::loss_chg`] and compared with XGBoost's
/// tie rule ([`need_replace`]). XGBoost routes the chosen set to its right
/// child; the recorded split stores that set as the tree's left child, with
/// the children (and the missing direction) swapped to match. `penalty`
/// (opt-in reuse penalties) is subtracted from each candidate's loss change
/// before it competes; `None` leaves the search untouched.
pub(super) fn sweep_categorical(
    best: &mut BestSplit,
    cats: &[(u32, GradStats)],
    total: GradStats,
    scorer: &SplitScorer,
    feature: u32,
    penalty: Option<&dyn CategoricalPenalty>,
) {
    let n = cats.len();
    let reg = scorer.reg;
    let score = |children: Children, set: &dyn Fn() -> Vec<u32>| {
        let mut score = scorer.loss_chg(children.left, children.right)?;
        if let Some(penalty) = penalty {
            score.loss_chg -= penalty.categorical_penalty(feature, &set()) as f32;
        }
        Some(score)
    };
    // `SplitEntry::Update` on a per-feature entry that starts at zero.
    let offer =
        |local: &mut Option<CatCandidate>, children: Children, set: &dyn Fn() -> Vec<u32>| {
            let Some(score) = score(children, set) else {
                return false;
            };
            let incumbent = local.as_ref().map_or(0.0, |c| c.score.loss_chg);
            if !need_replace(incumbent, feature, score.loss_chg, feature) {
                return false;
            }
            *local = Some(CatCandidate { children, score });
            true
        };
    // `p_best->Update(best)`: the feature's split against the node's best.
    let merge = |best: &mut BestSplit, local: Option<CatCandidate>, mut set: Vec<u32>| {
        let Some(CatCandidate { children, score }) = local else {
            return;
        };
        if need_replace(best.loss_chg as f32, best.feature, score.loss_chg, feature) {
            set.sort_unstable();
            *best =
                BestSplit::categorical(feature, children.swapped(), score.swapped().into(), set);
            best.children_swapped = true;
        }
    };

    if n < MAX_CAT_TO_ONEHOT {
        let mut present = GradStats::default();
        for &(_, stats) in cats {
            present.add(stats);
        }
        let missing = total.sub(present);
        let mut local = None;
        let mut chosen = 0;
        for &(cat, stats) in cats {
            let single = || vec![cat];
            // Missing values with the other categories, then with this one.
            let mut right = stats;
            let missing_left = Children::new(true, total.sub(right), right);
            if offer(&mut local, missing_left, &single) {
                chosen = cat;
            }
            right.add(missing);
            let missing_right = Children::new(false, total.sub(right), right);
            if offer(&mut local, missing_right, &single) {
                chosen = cat;
            }
        }
        merge(best, local, vec![chosen]);
        return;
    }

    let weight = |s: GradStats| -> f32 {
        if s.hess < reg.min_child_weight {
            0.0
        } else {
            xgb_calc_weight(s, reg) as f32
        }
    };
    let keys: Vec<f32> = cats.iter().map(|&(_, s)| weight(s)).collect();
    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by(|&l, &r| {
        keys[l]
            .partial_cmp(&keys[r])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let set_of =
        |partition: usize| -> Vec<u32> { sorted[..partition].iter().map(|&c| cats[c].0).collect() };
    let depth = MAX_CAT_THRESHOLD.min(n);
    for forward in [true, false] {
        let mut acc = GradStats::default();
        let mut local = None;
        let mut best_partition = 0;
        for step in 0..depth - 1 {
            let j = if forward { step } else { n - 1 - step };
            acc.add(cats[sorted[j]].1);
            // The set (XGBoost's right child) is the first `partition`
            // sorted categories: the scanned prefix going forward, the
            // unscanned rest going backward.
            let partition = if forward { step + 1 } else { j };
            let (left, right) = if forward {
                (total.sub(acc), acc)
            } else {
                (acc, total.sub(acc))
            };
            if offer(&mut local, Children::new(forward, left, right), &|| {
                set_of(partition)
            }) {
                best_partition = partition;
            }
        }
        merge(best, local, set_of(best_partition));
    }
}

/// XGBoost interaction-constraint state for one node: every split feature on
/// the root-to-node path and the features still permitted there.
#[derive(Clone)]
pub(super) struct InteractionState {
    path: Vec<u32>,
    allowed: Vec<u32>,
}

/// Normalize configured interaction groups. `None` disables filtering.
pub(super) fn build_interaction_sets(groups: &[Vec<u32>]) -> Option<Vec<Vec<u32>>> {
    if groups.is_empty() {
        return None;
    }
    Some(
        groups
            .iter()
            .map(|group| {
                let mut group = group.clone();
                group.sort_unstable();
                group.dedup();
                group
            })
            .collect(),
    )
}

/// Derive each child's state after splitting `feature`. XGBoost permits every
/// feature already used on the path plus every member of a constraint group
/// containing the *entire* updated path.
pub(super) fn next_allowed(
    parent: Option<&InteractionState>,
    feature: u32,
    groups: Option<&[Vec<u32>]>,
) -> Option<InteractionState> {
    let groups = groups?;
    let mut path = parent.map_or_else(Vec::new, |state| state.path.clone());
    if let Err(pos) = path.binary_search(&feature) {
        path.insert(pos, feature);
    }
    let mut allowed: BTreeSet<u32> = path.iter().copied().collect();
    for group in groups {
        if path
            .iter()
            .all(|feature| group.binary_search(feature).is_ok())
        {
            allowed.extend(group.iter().copied());
        }
    }
    Some(InteractionState {
        path,
        allowed: allowed.into_iter().collect(),
    })
}

/// Whether `feature` is permitted at a node. `None` means constraints inactive
/// or the root (where every feature is allowed).
pub(super) fn permits(state: Option<&InteractionState>, feature: u32) -> bool {
    state.is_none_or(|state| state.allowed.binary_search(&feature).is_ok())
}

/// Sum the gradient pairs of `rows`, in row order. Shared by both builders'
/// root-statistics accumulation.
///
/// Kept out of line: inlined into `HistTreeBuilder::build_inner`, LLVM kept
/// the running sum in the caller's stack slot and paid a store-to-load round
/// trip per row.
#[inline(never)]
pub(super) fn sum_rows(gpair: &[GradPair], rows: &[u32]) -> GradStats {
    let mut total = GradStats::default();
    for &r in rows {
        total.add(GradStats::from_pair(gpair[r as usize]));
    }
    total
}

/// Set every leaf's weight from its stored statistics, respecting each leaf's
/// monotone bounds. Shared by both builders' final pass.
pub(super) fn finalize_leaf_values(
    tree: &mut RegTree,
    stats: &[GradStats],
    bounds: &[Bounds],
    reg: &RegParams,
) {
    #[allow(clippy::needless_range_loop)]
    for id in 0..tree.num_nodes() {
        if tree.node(id).is_leaf() {
            let w = calc_weight_bounded(stats[id], reg, bounds[id]);
            tree.set_leaf_value(id, w as f32);
        }
    }
}

#[cfg(test)]
mod test_support {
    use super::{ExactTreeBuilder, HistTreeBuilder, SortedColumns, all_rows};
    use crate::config::TrainingParams;
    use crate::data::DMatrix;
    use crate::data::ghist::GHistIndex;
    use crate::data::quantile::HistCuts;
    use crate::objective::GradPair;
    use crate::tree::regtree::RegTree;
    use crate::tree::sampler::ColumnSampler;

    pub(super) fn gp(g: f32, h: f32) -> GradPair {
        GradPair::new(g, h)
    }

    pub(super) fn binned(data: &DMatrix, max_bin: usize) -> GHistIndex {
        GHistIndex::from_dmatrix(data, HistCuts::from_dmatrix(data, max_bin))
    }

    /// One exact tree over every row and feature of `data`.
    pub(super) fn grow_exact(
        params: &TrainingParams,
        data: &DMatrix,
        gpair: &[GradPair],
    ) -> RegTree {
        ExactTreeBuilder::new(params).build(
            &SortedColumns::from_dmatrix(data),
            data,
            gpair,
            &all_rows(data.n_rows()),
            &mut ColumnSampler::all(data.n_cols()),
        )
    }

    /// One histogram tree over every row and feature of `ghist`.
    pub(super) fn grow_hist(
        params: &TrainingParams,
        ghist: &GHistIndex,
        gpair: &[GradPair],
    ) -> RegTree {
        HistTreeBuilder::new(params).build(
            ghist,
            gpair,
            &all_rows(ghist.n_rows()),
            &mut ColumnSampler::all(ghist.n_cols()),
        )
    }

    /// Whether `tree`'s predictions never decrease (beyond `1e-5`) from one
    /// row of `data` to the next.
    pub(super) fn non_decreasing(tree: &RegTree, data: &DMatrix) -> bool {
        let preds: Vec<f32> = (0..data.n_rows())
            .map(|r| tree.predict_row(data, r))
            .collect();
        preds.windows(2).all(|w| w[1] >= w[0] - 1e-5)
    }

    /// Data where the *unconstrained* fit would be non-monotone: a V shape.
    /// y dips in the middle, so an unconstrained tree would go down then up.
    pub(super) fn monotone_v_shape_data() -> (DMatrix, Vec<GradPair>) {
        let n = 60;
        let mut x = Vec::new();
        let mut gpair = Vec::new();
        for i in 0..n {
            let xi = i as f32 / n as f32;
            x.push(xi);
            let target = (xi - 0.5).abs(); // V shape, non-monotone
            gpair.push(gp(-(target - 0.25), 1.0)); // pseudo-residual around mean
        }
        (DMatrix::from_dense(&x, n, 1).unwrap(), gpair)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unregularized() -> RegParams {
        RegParams {
            lambda: 0.0,
            alpha: 0.0,
            max_delta_step: 0.0,
            min_child_weight: 0.0,
        }
    }

    fn sweep(cats: &[(u32, GradStats)], total: GradStats) -> BestSplit {
        let reg = unregularized();
        let mut best = BestSplit::none();
        let scorer = SplitScorer {
            reg: &reg,
            root_gain: xgb_node_gain(total, &reg, Bounds::default()),
            bounds: Bounds::default(),
            dir: 0,
        };
        sweep_categorical(&mut best, cats, total, &scorer, 0, None);
        best
    }

    /// A feature with a single category still separates it from the missing
    /// values (one-hot search, fewer than four categories).
    #[test]
    fn a_lone_category_splits_from_missing_values() {
        let best = sweep(&[(0, GradStats::new(2.0, 2.0))], GradStats::new(0.0, 4.0));
        assert!(best.is_categorical);
        assert_eq!(best.cat_left, [0]);
        assert!(!best.default_left, "missing values go to the other child");
        assert!((best.loss_chg - 4.0).abs() < 1e-6, "{}", best.loss_chg);
    }

    /// Four categories ordered by weight (`-2, -1, 1, 2`) plus missing values
    /// whose weight (`±3`) puts them with one end of that order: the search
    /// scans both directions, so the missing values join the side that fits
    /// them, and the recorded set is `{0, 1}` either way.
    #[test]
    fn missing_values_join_the_side_that_fits_them() {
        let cats = [
            (0, GradStats::new(2.0, 1.0)),
            (1, GradStats::new(1.0, 1.0)),
            (2, GradStats::new(-1.0, 1.0)),
            (3, GradStats::new(-2.0, 1.0)),
        ];
        // G = 3 and -6 over H = 2 and 3 against the parent's 9/5.
        let expected = 4.5 + 12.0 - 1.8;
        for (missing_grad, with_set) in [(-3.0, false), (3.0, true)] {
            let best = sweep(&cats, GradStats::new(missing_grad, 5.0));
            assert!(best.is_categorical);
            assert_eq!(best.cat_left, [0, 1], "missing gradient {missing_grad}");
            assert_eq!(
                best.default_left, with_set,
                "missing gradient {missing_grad}"
            );
            assert!(
                (best.loss_chg - expected).abs() < 1e-5,
                "missing gradient {missing_grad}: {}",
                best.loss_chg
            );
        }
    }
}
