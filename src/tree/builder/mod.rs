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

pub use exact::{ExactTreeBuilder, SortedColumns, all_features, all_rows};
pub use hist::HistTreeBuilder;
pub(crate) use hist::LeafRows;
pub(crate) use multi::{MultiTreeBuilder, VectorGradients};
pub(crate) use oblivious::check_symmetric_input;

use std::collections::BTreeSet;

use crate::K_RT_EPS;
use crate::objective::GradPair;
use crate::tree::constraints::{Bounds, calc_weight_bounded, gain_at_weight, satisfies};
use crate::tree::gain::{GradStats, RegParams, calc_gain, threshold_l1};
use crate::tree::regtree::RegTree;
use crate::tree::reuse::CategoricalPenalty;

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
        }
    }

    /// A numeric split candidate; `pos` carries the split location (value-space
    /// threshold for exact search, global-bin boundary for histogram search).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn numeric(
        loss_chg: f64,
        feature: u32,
        pos: SplitPos,
        default_left: bool,
        left: GradStats,
        right: GradStats,
        w_left: f64,
        w_right: f64,
    ) -> Self {
        let (threshold, split_bin) = match pos {
            SplitPos::Value(t) => (t, None),
            SplitPos::Bin(b) => (0.0, Some(b)),
            SplitPos::BelowBins => (0.0, None),
        };
        BestSplit {
            loss_chg,
            feature,
            threshold,
            split_bin,
            default_left,
            left,
            right,
            w_left,
            w_right,
            is_categorical: false,
            cat_left: Vec::new(),
        }
    }

    /// A categorical (set-membership) split candidate; `cat_left` holds the
    /// category values routed left.
    pub(super) fn categorical(
        loss_chg: f64,
        feature: u32,
        left: GradStats,
        right: GradStats,
        w_left: f64,
        w_right: f64,
        cat_left: Vec<u32>,
    ) -> Self {
        BestSplit {
            loss_chg,
            feature,
            threshold: 0.0,
            split_bin: None,
            // Present categories not in the left set (and missing) go
            // right, as XGBoost defaults for categorical features.
            default_left: false,
            left,
            right,
            w_left,
            w_right,
            is_categorical: true,
            cat_left,
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
        let (w_left, w_right) = (self.w_left as f32, self.w_right as f32);
        let (h_left, h_right) = (self.left.hess as f32, self.right.hess as f32);
        let ids = if self.is_categorical {
            tree.expand_categorical(
                nid,
                self.feature,
                &self.cat_left,
                self.default_left,
                w_left,
                h_left,
                w_right,
                h_right,
            )
        } else {
            tree.expand(
                nid,
                self.feature,
                threshold,
                self.default_left,
                w_left,
                h_left,
                w_right,
                h_right,
            )
        };
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

/// Gain of one candidate split plus its bounded child weights, or `None` when
/// a child is below `min_child_weight` or a monotone constraint is violated.
/// Unconstrained builds take the cheap closed-form path (weights unused).
#[inline]
pub(super) fn candidate_gain(
    left: GradStats,
    right: GradStats,
    parent: f64,
    bounds: Bounds,
    dir: i8,
    constrained: bool,
    reg: &RegParams,
) -> Option<(f64, f64, f64)> {
    if left.hess < reg.min_child_weight || right.hess < reg.min_child_weight {
        return None;
    }
    if constrained {
        let wl = calc_weight_bounded(left, reg, bounds);
        let wr = calc_weight_bounded(right, reg, bounds);
        if !satisfies(dir, wl, wr) {
            return None;
        }
        let g = gain_at_weight(left, reg, wl) + gain_at_weight(right, reg, wr) - parent;
        Some((g, wl, wr))
    } else {
        let g = calc_gain(left, reg) + calc_gain(right, reg) - parent;
        Some((g, 0.0, 0.0))
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
/// accept the other. `offer(pos, default_left, left, right)` sees every
/// candidate; a `dense` index has no missing values.
#[inline]
pub(super) fn for_each_numeric_split(
    bins: &[GradStats],
    first: usize,
    total: GradStats,
    dense: bool,
    mut offer: impl FnMut(SplitPos, bool, GradStats, GradStats),
) {
    let mut acc = GradStats::default();
    for (offset, &bin) in bins.iter().enumerate() {
        acc.add(bin);
        offer(SplitPos::Bin(first + offset), false, acc, total.sub(acc));
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
        offer(pos, true, total.sub(suffix), suffix);
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

/// XGBoost's scalar `SplitEvaluator::CalcSplitGain` minus the parent's
/// `root_gain`, i.e. the `loss_chg` a candidate is compared and stored with.
/// Returns `None` when the split is invalid (a child without positive Hessian
/// or below `min_child_weight`) or violates the monotone direction `dir`, and
/// otherwise the `f32` loss change plus both bounded child weights.
#[inline]
pub(super) fn xgb_loss_chg(
    left: GradStats,
    right: GradStats,
    root_gain: f32,
    reg: &RegParams,
    bounds: Bounds,
    dir: i8,
) -> Option<(f32, f32, f32)> {
    if !children_valid(left, right, reg.min_child_weight) {
        return None;
    }
    let wl = xgb_weight(left, reg, bounds);
    let wr = xgb_weight(right, reg, bounds);
    if !satisfies(dir, f64::from(wl), f64::from(wr)) {
        return None;
    }
    // Upstream's scalar `CalcGainGivenWeight` returns `float`: each child's
    // score is rounded before the two are added in `f32`.
    let gain =
        xgb_gain_given_weight(left, reg, wl) as f32 + xgb_gain_given_weight(right, reg, wr) as f32;
    Some((gain - root_gain, wl, wr))
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
#[allow(clippy::too_many_arguments)]
pub(super) fn xgb_update(
    best: &mut BestSplit,
    loss_chg: f32,
    feature: u32,
    pos: SplitPos,
    default_left: bool,
    left: GradStats,
    right: GradStats,
    w_left: f32,
    w_right: f32,
) -> bool {
    let replace = need_replace(best.loss_chg as f32, best.feature, loss_chg, feature);
    if replace {
        *best = BestSplit::numeric(
            f64::from(loss_chg),
            feature,
            pos,
            default_left,
            left,
            right,
            f64::from(w_left),
            f64::from(w_right),
        );
    }
    replace
}

/// Sweep prefix partitions of categories ordered by gradient/Hessian ratio
/// (XGBoost's sorted-partition strategy: the best subset is contiguous in
/// that order) and record the best set-membership split in `best`. Tied
/// ratios keep ascending category order, as XGBoost's `std::stable_sort`
/// over bin indices does, so the result never depends on the caller's
/// ordering of `cats`. Prefix categories form the left set; every other
/// present category — and missing — goes right. Callers supply the
/// `(category, stats)` pairs from their own
/// stat source (sorted-column map for exact search, histogram bins for
/// histogram search) and keep their own empty-bin filtering. `penalty`
/// (opt-in reuse penalties) is subtracted from each candidate's gain before
/// it competes; `None` leaves the sweep untouched.
#[allow(clippy::too_many_arguments)]
pub(super) fn sweep_categorical(
    best: &mut BestSplit,
    cats: &mut [(u32, GradStats)],
    total: GradStats,
    parent_gain: f64,
    bounds: Bounds,
    dir: i8,
    constrained: bool,
    reg: &RegParams,
    feature: u32,
    penalty: Option<&dyn CategoricalPenalty>,
) {
    if cats.len() < 2 {
        return; // no interior partition
    }
    let ratio = |s: GradStats| s.grad / (s.hess + reg.lambda);
    // Categories are distinct, so this is a total order.
    cats.sort_unstable_by(|a, b| ratio(a.1).total_cmp(&ratio(b.1)).then(a.0.cmp(&b.0)));
    sweep_prefixes(best, cats, total, feature, |left, right, cats_left| {
        let (mut g, wl, wr) =
            candidate_gain(left, right, parent_gain, bounds, dir, constrained, reg)?;
        if let Some(penalty) = penalty {
            g -= penalty.categorical_penalty(feature, cats_left);
        }
        Some((g, wl, wr))
    });
}

/// Offer every prefix of the ordered `cats` (at least one category always
/// stays right) as a set-membership split with the prefix on the left.
/// `total` includes any missing mass, which stays on the right.
/// `score(left, right, cats_left)` returns the candidate's gain and child
/// weights (`None`: invalid), and `best` takes it when it beats the incumbent
/// by more than `K_RT_EPS`.
pub(super) fn sweep_prefixes(
    best: &mut BestSplit,
    cats: &[(u32, GradStats)],
    total: GradStats,
    feature: u32,
    mut score: impl FnMut(GradStats, GradStats, &[u32]) -> Option<(f64, f64, f64)>,
) {
    let mut left = GradStats::default();
    let mut cats_left: Vec<u32> = Vec::new();
    for &(cat, stats) in &cats[..cats.len() - 1] {
        left.add(stats);
        cats_left.push(cat);
        let right = total.sub(left);
        if let Some((g, wl, wr)) = score(left, right, &cats_left)
            && g > best.loss_chg + K_RT_EPS
        {
            *best = BestSplit::categorical(g, feature, left, right, wl, wr, cats_left.clone());
        }
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

    /// Penalizes every left set except `{0, 2}`.
    struct OnlyZeroTwo;

    impl CategoricalPenalty for OnlyZeroTwo {
        fn categorical_penalty(&self, _feature: u32, cats_left: &[u32]) -> f64 {
            let mut set = cats_left.to_vec();
            set.sort_unstable();
            if set == [0, 2] { 0.0 } else { 10.0 }
        }
    }

    /// Categories 0 and 1 tie on gradient/Hessian ratio; the penalized sweep
    /// must break the tie by category id whatever order the caller supplies
    /// (the exact builder's order comes from a `HashMap`).
    #[test]
    fn categorical_ties_sweep_in_category_order() {
        let reg = RegParams {
            lambda: 0.0,
            alpha: 0.0,
            max_delta_step: 0.0,
            min_child_weight: 0.0,
        };
        let stats = |c: u32| {
            let g = [1.0, 1.0, -2.0][c as usize];
            (c, GradStats::new(g, 1.0))
        };
        let total = GradStats::new(0.0, 3.0);
        let sweep = |order: [u32; 3]| {
            let mut cats = order.map(stats);
            let mut best = BestSplit::none();
            sweep_categorical(
                &mut best,
                &mut cats,
                total,
                0.0,
                Bounds::default(),
                0,
                false,
                &reg,
                0,
                Some(&OnlyZeroTwo),
            );
            best
        };
        for order in [[0, 1, 2], [1, 0, 2], [2, 1, 0]] {
            let best = sweep(order);
            assert!(best.is_categorical, "order {order:?}");
            assert_eq!(best.cat_left, [2, 0], "order {order:?}");
            assert!((best.loss_chg - 1.5).abs() < 1e-9, "order {order:?}");
        }
    }
}
