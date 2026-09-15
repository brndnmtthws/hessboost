//! Tree construction algorithms.
//!
//! Each builder grows a single [`crate::tree::RegTree`] from per-instance
//! gradients. The exact builder is the reference. Approximate and histogram
//! builders (added in a later phase) share the same regularized gain math.

mod exact;
mod hist;

pub use exact::{all_features, all_rows, ExactTreeBuilder, SortedColumns};
pub use hist::HistTreeBuilder;
pub(crate) use hist::LeafRows;

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};

use crate::tree::constraints::{calc_weight_bounded, gain_at_weight, satisfies, Bounds};
use crate::tree::gain::{calc_gain, GradStats, RegParams};

/// Tiny epsilon guarding against accepting numerically-zero-gain splits, mirror
/// of XGBoost's `kRtEps`.
pub(super) const K_RT_EPS: f64 = 1e-6;

/// The best split found so far for one node.
///
/// Both builders share this. `threshold` is the exact split value while
/// `split_bin` is the histogram global-bin boundary (bins `<= split_bin` go
/// left); each builder writes its own location field and leaves the other at
/// its default.
#[derive(Debug, Clone)]
pub(super) struct BestSplit {
    pub(super) loss_chg: f64,
    pub(super) feature: u32,
    pub(super) threshold: f32,
    pub(super) split_bin: usize,
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
            split_bin: 0,
            default_left: true,
            left: GradStats::default(),
            right: GradStats::default(),
            w_left: 0.0,
            w_right: 0.0,
            is_categorical: false,
            cat_left: Vec::new(),
        }
    }

    #[inline]
    pub(super) fn found(&self) -> bool {
        self.loss_chg > K_RT_EPS
    }
}

/// Ordering on split loss change for the loss-guided priority queue.
#[inline]
pub(super) fn loss_ord(a: f64, b: f64) -> Ordering {
    a.total_cmp(&b)
}

/// Parent structure score subtracted from a split's gain: bounded when
/// constraints are active, closed-form otherwise.
#[inline]
pub(super) fn parent_gain(
    stats: GradStats,
    reg: &RegParams,
    bounds: Bounds,
    constrained: bool,
) -> f64 {
    if constrained {
        let w = calc_weight_bounded(stats, reg, bounds);
        gain_at_weight(stats, reg, w)
    } else {
        calc_gain(stats, reg)
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
/// Where a numeric candidate split falls. Exact search records the midpoint
/// threshold in value space; histogram search records the global-bin boundary
/// (bins `<= split_bin` go left). The accept path is otherwise identical.
#[derive(Debug, Clone, Copy)]
pub(super) enum SplitPos {
    Value(f32),
    Bin(usize),
}

/// Evaluate one numeric candidate split and record it in `best` when its gain
/// improves on the incumbent (beyond [`K_RT_EPS`]). Shared by both builders;
/// only the split location differs ([`SplitPos`]).
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn accept_numeric(
    best: &mut BestSplit,
    left: GradStats,
    right: GradStats,
    parent_gain: f64,
    bounds: Bounds,
    dir: i8,
    constrained: bool,
    reg: &RegParams,
    feature: u32,
    pos: SplitPos,
    default_left: bool,
) {
    let Some((g, wl, wr)) = candidate_gain(left, right, parent_gain, bounds, dir, constrained, reg)
    else {
        return;
    };
    if g > best.loss_chg + K_RT_EPS {
        let (threshold, split_bin) = match pos {
            SplitPos::Value(t) => (t, 0),
            SplitPos::Bin(b) => (0.0, b),
        };
        *best = BestSplit {
            loss_chg: g,
            feature,
            threshold,
            split_bin,
            default_left,
            left,
            right,
            w_left: wl,
            w_right: wr,
            is_categorical: false,
            cat_left: Vec::new(),
        };
    }
}

/// Evaluate one numeric boundary under both missing-value directions: missing
/// right, then missing left. The leftward direction is skipped when the caller
/// knows no missing mass exists (histogram search passes its `has_missing`
/// flag; exact search always passes `true`). Shared by both builders.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn eval_missing_directions(
    best: &mut BestSplit,
    left_present: GradStats,
    present: GradStats,
    total: GradStats,
    reg: &RegParams,
    parent_gain: f64,
    bounds: Bounds,
    dir: i8,
    constrained: bool,
    feature: u32,
    pos: SplitPos,
    eval_missing_left: bool,
) {
    let mcw = reg.min_child_weight;

    // Direction A: missing values go right. Left = present-so-far.
    let la = left_present;
    let ra = total.sub(left_present);
    if la.hess >= mcw && ra.hess >= mcw {
        accept_numeric(
            best,
            la,
            ra,
            parent_gain,
            bounds,
            dir,
            constrained,
            reg,
            feature,
            pos,
            false,
        );
    }

    // Direction B: missing values go left. Left = present-so-far + missing.
    if eval_missing_left {
        let mut lb = left_present;
        lb.add(total.sub(present));
        let rb = present.sub(left_present);
        if lb.hess >= mcw && rb.hess >= mcw {
            accept_numeric(
                best,
                lb,
                rb,
                parent_gain,
                bounds,
                dir,
                constrained,
                reg,
                feature,
                pos,
                true,
            );
        }
    }
}

/// Sweep prefix partitions of categories ordered by gradient/Hessian ratio
/// (XGBoost's sorted-partition strategy: the best subset is contiguous in
/// that order) and record the best set-membership split in `best`. Prefix
/// categories form the left set; every other present category — and missing
/// — goes right. Callers supply the `(category, stats)` pairs from their own
/// stat source (sorted-column map for exact search, histogram bins for
/// histogram search) and keep their own empty-bin filtering.
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
        let Some((g, wl, wr)) =
            candidate_gain(left, right, parent_gain, bounds, dir, constrained, reg)
        else {
            continue;
        };
        if g > best.loss_chg + K_RT_EPS {
            *best = BestSplit {
                loss_chg: g,
                feature,
                threshold: 0.0,
                split_bin: 0,
                // Present categories not in the left set (and missing) go
                // right, as XGBoost defaults for categorical features.
                default_left: false,
                left,
                right,
                w_left: wl,
                w_right: wr,
                is_categorical: true,
                cat_left: cats_left.clone(),
            };
        }
    }
}

/// Precompute per-feature interaction sets from the constraint groups.
///
/// The interaction set of a feature is the union of every group that contains
/// it (which includes the feature itself). A feature that appears in no group
/// may only interact with itself. Returns `None` when no constraints are
/// configured (the inactive, no-filtering case).
pub(super) fn build_interaction_sets(groups: &[Vec<u32>]) -> Option<HashMap<u32, Vec<u32>>> {
    if groups.is_empty() {
        return None;
    }
    let mut sets: HashMap<u32, BTreeSet<u32>> = HashMap::new();
    for group in groups {
        for &feature in group {
            sets.entry(feature)
                .or_default()
                .extend(group.iter().copied());
        }
    }
    Some(
        sets.into_iter()
            .map(|(feature, allowed)| (feature, allowed.into_iter().collect()))
            .collect(),
    )
}

/// Intersect a node's allowed set with a feature's interaction set. Both
/// operands are sorted. `None` denotes "all features"; inactive constraints
/// (`None` sets) stay `None`.
pub(super) fn next_allowed(
    parent: Option<&[u32]>,
    feature: u32,
    sets: Option<&HashMap<u32, Vec<u32>>>,
) -> Option<Vec<u32>> {
    let sets = sets?;
    let singleton = [feature];
    let feature_set = sets
        .get(&feature)
        .map_or(singleton.as_slice(), Vec::as_slice);
    Some(match parent {
        None => feature_set.to_vec(),
        Some(parent) => parent
            .iter()
            .copied()
            .filter(|f| feature_set.binary_search(f).is_ok())
            .collect(),
    })
}
