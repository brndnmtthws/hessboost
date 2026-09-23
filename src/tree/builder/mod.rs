//! Tree construction algorithms.
//!
//! Each builder grows a single [`crate::tree::RegTree`] from per-instance
//! gradients. The exact builder is the reference; the histogram builder shares
//! the same regularized gain math.

mod exact;
mod hist;

pub use exact::{ExactTreeBuilder, SortedColumns, all_features, all_rows};
pub use hist::HistTreeBuilder;
pub(crate) use hist::LeafRows;

use std::collections::BTreeSet;

use crate::objective::GradPair;
use crate::tree::constraints::{Bounds, calc_weight_bounded, gain_at_weight, satisfies};
use crate::tree::gain::{GradStats, RegParams, calc_gain, threshold_l1};
use crate::tree::regtree::RegTree;
use crate::tree::reuse::CategoricalPenalty;

/// Tiny epsilon guarding against accepting numerically-zero-gain splits, mirror
/// of XGBoost's `kRtEps`.
pub(super) const K_RT_EPS: f64 = 1e-6;

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
    #[allow(clippy::too_many_arguments)]
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

    /// Whether this split should be taken: it was found, its loss change
    /// reaches `gamma` (XGBoost rejects `loss_chg < min_split_loss`), and both
    /// children have positive cover and meet `min_child_weight`.
    pub(super) fn valid(&self, gamma: f64, min_child_weight: f64) -> bool {
        self.found()
            && self.loss_chg >= gamma
            && self.left.hess > 0.0
            && self.right.hess > 0.0
            && self.left.hess >= min_child_weight
            && self.right.hess >= min_child_weight
    }
}

/// Gain of one candidate split plus its bounded child weights, or `None` when
/// a monotone constraint is violated. Unconstrained builds take the cheap
/// closed-form path (weights unused).
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

/// XGBoost's `SplitEvaluator::CalcWeight`: the regularized optimum computed in
/// `f64`, rounded to `f32`, then clamped to the node's monotone bounds. The
/// `f32` rounding happens before bounding, exactly as upstream.
#[inline]
pub(super) fn xgb_weight(stats: GradStats, reg: &RegParams, bounds: Bounds) -> f32 {
    let w = if stats.hess <= 0.0 {
        0.0
    } else {
        let mut w = -threshold_l1(stats.grad, reg.alpha) / (stats.hess + reg.lambda);
        if reg.max_delta_step != 0.0 && w.abs() > reg.max_delta_step {
            w = reg.max_delta_step.copysign(w);
        }
        w
    } as f32;
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
fn xgb_gain_given_weight(stats: GradStats, reg: &RegParams, w: f32) -> f64 {
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
    let mcw = reg.min_child_weight;
    if !(left.hess > 0.0 && right.hess > 0.0 && left.hess >= mcw && right.hess >= mcw) {
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

/// XGBoost's `SplitEntry::Update`: replace the incumbent when the candidate's
/// loss change is strictly better, or equal on a lower feature index. Infinite
/// loss changes are never taken. `best.loss_chg` holds an `f32` value.
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
    if loss_chg.is_infinite() {
        return false;
    }
    let incumbent = best.loss_chg as f32;
    let replace = if best.feature <= feature {
        loss_chg > incumbent
    } else {
        incumbent.partial_cmp(&loss_chg) != Some(std::cmp::Ordering::Greater)
    };
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
/// that order) and record the best set-membership split in `best`. Prefix
/// categories form the left set; every other present category — and missing
/// — goes right. Callers supply the `(category, stats)` pairs from their own
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
    cats.sort_by(|a, b| ratio(a.1).total_cmp(&ratio(b.1)));

    let mcw = reg.min_child_weight;
    let mut left = GradStats::default();
    let mut cats_left: Vec<u32> = Vec::new();
    // Sweep prefixes, always leaving at least one category on the right.
    for &(cat, s) in &cats[..cats.len() - 1] {
        left.add(s);
        cats_left.push(cat);
        // `total` includes any missing mass, which stays on the right.
        let right = total.sub(left);
        if left.hess < mcw || right.hess < mcw {
            continue;
        }
        let Some((mut g, wl, wr)) =
            candidate_gain(left, right, parent_gain, bounds, dir, constrained, reg)
        else {
            continue;
        };
        if let Some(penalty) = penalty {
            g -= penalty.categorical_penalty(feature, &cats_left);
        }
        if g > best.loss_chg + K_RT_EPS {
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
    use crate::data::DMatrix;
    use crate::objective::GradPair;

    pub(super) fn gp(g: f32, h: f32) -> GradPair {
        GradPair::new(g, h)
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
