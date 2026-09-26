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

use std::cell::RefCell;
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

    /// [`Self::loss_chg`] of a run of candidates, written branch-free so it
    /// vectorizes: `acc_grad`/`acc_hess` are the accumulated statistics of
    /// each candidate's left child (`ACC_LEFT`) or right child, the other
    /// child is `total` minus them. Each `loss[i]` is the candidate's loss
    /// change, or `-inf` where [`Self::loss_chg`] returns `None`. The
    /// arithmetic is the scalar path's, operation for operation.
    #[inline]
    pub(super) fn score_run<const ACC_LEFT: bool>(
        &self,
        total: GradStats,
        acc_grad: &[f64],
        acc_hess: &[f64],
        loss: &mut [f32],
    ) {
        match self.dir {
            d if d > 0 => self.score_run_dir::<ACC_LEFT, 1>(total, acc_grad, acc_hess, loss),
            d if d < 0 => self.score_run_dir::<ACC_LEFT, { -1 }>(total, acc_grad, acc_hess, loss),
            _ => self.score_run_dir::<ACC_LEFT, 0>(total, acc_grad, acc_hess, loss),
        }
    }

    #[inline(always)]
    fn score_run_dir<const ACC_LEFT: bool, const DIR: i8>(
        &self,
        total: GradStats,
        acc_grad: &[f64],
        acc_hess: &[f64],
        loss: &mut [f32],
    ) {
        let RegParams {
            lambda,
            alpha,
            max_delta_step,
            min_child_weight,
        } = *self.reg;
        let (lower, upper) = (self.bounds.lower as f32, self.bounds.upper as f32);
        let root_gain = self.root_gain;
        // `xgb_weight` without its `hess <= 0` case: a candidate whose
        // child lacks positive Hessian is invalid, and its value discarded.
        let weight = |g: f64, h: f64| -> f32 {
            let t = if g > alpha {
                g - alpha
            } else if g < -alpha {
                g + alpha
            } else {
                0.0
            };
            let mut w = -t / (h + lambda);
            if max_delta_step != 0.0 && w.abs() > max_delta_step {
                w = max_delta_step.copysign(w);
            }
            let w = w as f32;
            if w < lower {
                lower
            } else if w > upper {
                upper
            } else {
                w
            }
        };
        let gain = |g: f64, h: f64, w: f32| -> f64 {
            -(2.0 * g * f64::from(w)
                + (h + lambda) * f64::from(w * w)
                + 2.0 * alpha * f64::from(w.abs()))
        };
        let n = loss.len();
        let (acc_grad, acc_hess) = (&acc_grad[..n], &acc_hess[..n]);
        for i in 0..n {
            let (ag, ah) = (acc_grad[i], acc_hess[i]);
            let (og, oh) = (total.grad - ag, total.hess - ah);
            let (lg, lh, rg, rh) = if ACC_LEFT {
                (ag, ah, og, oh)
            } else {
                (og, oh, ag, ah)
            };
            let valid = lh > 0.0 && rh > 0.0 && lh >= min_child_weight && rh >= min_child_weight;
            let wl = weight(lg, lh);
            let wr = weight(rg, rh);
            let monotone = match DIR {
                1 => f64::from(wl) <= f64::from(wr),
                -1 => f64::from(wl) >= f64::from(wr),
                _ => true,
            };
            let chg = (gain(lg, lh, wl) as f32 + gain(rg, rh, wr) as f32) - root_gain;
            loss[i] = if valid && monotone {
                chg
            } else {
                f32::NEG_INFINITY
            };
        }
    }

    /// Whether [`Self::approx_run`]'s error bound ([`APPROX_MARGIN`],
    /// [`UNDERFLOW_MARGIN`]) holds: no monotone direction or bounds, no
    /// `alpha` (whose soft threshold can cancel in the exact gain) or
    /// `max_delta_step`, and `H + λ >= 1e-3` for every valid child (`H >=
    /// min_child_weight`), so both scorers overflow only for large gains.
    #[inline]
    pub(super) fn approx_exact(&self) -> bool {
        let reg = self.reg;
        self.dir == 0
            && self.bounds.lower == f64::NEG_INFINITY
            && self.bounds.upper == f64::INFINITY
            && reg.alpha == 0.0
            && reg.max_delta_step == 0.0
            && reg.lambda + reg.min_child_weight >= 1e-3
            && self.root_gain.is_finite()
    }

    /// [`Screen::cannot_beat`] of this node: `false` whenever
    /// [`Self::approx_exact`] does not hold.
    #[cfg(test)]
    pub(super) fn cannot_beat(&self, left: GradStats, right: GradStats, incumbent: f64) -> bool {
        self.screen()
            .is_some_and(|screen| screen.cannot_beat(left, right, incumbent))
    }

    /// The node constants of the division-free screen, or `None` where
    /// [`Self::approx_exact`] does not hold (nothing may be screened): taken
    /// once so a scan over many candidates of one node screens each with a
    /// few multiplications.
    #[inline]
    pub(super) fn screen(&self) -> Option<Screen> {
        self.approx_exact().then(|| Screen {
            lambda: self.reg.lambda,
            root: f64::from(self.root_gain),
        })
    }
}

/// A node's division-free candidate screen ([`SplitScorer::screen`]).
#[derive(Debug, Clone, Copy)]
pub(super) struct Screen {
    lambda: f64,
    root: f64,
}

impl Screen {
    /// Whether the candidate `(left, right)` certainly cannot score a loss
    /// change above `incumbent` under [`SplitScorer::loss_chg`], decided
    /// without a division: `U = Σ G² / (H + λ)` bounds the exact loss change
    /// by `U - root_gain` within the exact scorer's share of the
    /// [`APPROX_MARGIN`] and [`UNDERFLOW_MARGIN`] analysis, and the
    /// comparison is cross-multiplied. `false` whenever `U` overflows, so a
    /// `true` never hides a winner. (A loss change that overflows or is NaN
    /// never replaces an incumbent from the same or an earlier feature, so a
    /// `true` for it is harmless.)
    #[cfg(test)]
    pub(super) fn cannot_beat(self, left: GradStats, right: GradStats, incumbent: f64) -> bool {
        let Screen { lambda, root } = self;
        let (hl, hr) = (left.hess + lambda, right.hess + lambda);
        if !(hl > 0.0 && hr > 0.0) {
            return false;
        }
        // The exact loss change is at most `U(1 + κ) - root + κ|root| + a`,
        // `a` the absolute allowance; it stays at most the incumbent while
        // `U(1 + κ) < incumbent + root - κ(|root| + |incumbent|) - a`.
        let absolute = UNDERFLOW_MARGIN * (hl + hr + 1.0);
        let bound = incumbent + root - SCREEN_KAPPA * (root.abs() + incumbent.abs()) - absolute;
        let n = left.grad * left.grad * hr + right.grad * right.grad * hl;
        n * (1.0 + SCREEN_KAPPA) < bound * (hl * hr)
    }

    /// `Self::cannot_beat` (the per-candidate bound the tests check against) of
    /// the node whose statistics have Hessian
    /// `total_hess`, against `incumbent`, with everything but the
    /// candidate's own terms computed once ([`ScreenBound::rules_out`]).
    #[inline]
    pub(super) fn bound(self, total_hess: f64, incumbent: f64) -> ScreenBound {
        // Every candidate's allowance `a = UNDERFLOW_MARGIN · (D_l + D_r +
        // 1)` is at most this one: its children have `H_l, H_r >= 0` with
        // `H_r = H - H_l` rounded, so `H_l + H_r <= H(1 + ε)` and the rounded
        // `D_l + D_r` stays within `(H + 2λ)(1 + 4ε)`; `2^-20` covers every
        // rounding here.
        const SLACK: f64 = 1.0 + 1.0 / 1_048_576.0;
        let Screen { lambda, root } = self;
        let allowance = UNDERFLOW_MARGIN * ((total_hess + 2.0 * lambda) * SLACK + 1.0) * SLACK;
        ScreenBound {
            lambda,
            limit: incumbent + root - SCREEN_KAPPA * (root.abs() + incumbent.abs()) - allowance,
        }
    }
}

/// `κ = 2^-19` of the division-free screen: the exact score is within
/// `4ε(U + |root|) + (D_l + D_r + 2)τ` of `U - root` ([`APPROX_MARGIN`]);
/// `κ` is eight times the relative part, and `UNDERFLOW_MARGIN · (D_l +
/// D_r + 1)` over thirty times the absolute one.
const SCREEN_KAPPA: f64 = 1.0 / 524_288.0;

/// The division-free screen bound to one node and incumbent
/// ([`Screen::bound`]).
#[derive(Debug, Clone, Copy)]
pub(super) struct ScreenBound {
    lambda: f64,
    /// `incumbent + root - κ(|root| + |incumbent|)` less the largest
    /// allowance of any candidate of the node.
    limit: f64,
}

impl ScreenBound {
    /// Whether the candidate `(left, right)`, whose children's Hessians are
    /// non-negative and sum to the node's (`right = total - left`, or the
    /// reverse), certainly cannot beat the incumbent. Implies
    /// `Screen::cannot_beat`: its limit is at most every candidate's own
    /// bound, and both sides of the comparison round monotonically.
    #[inline]
    pub(super) fn rules_out(self, left: GradStats, right: GradStats) -> bool {
        let (hl, hr) = (left.hess + self.lambda, right.hess + self.lambda);
        if !(hl > 0.0 && hr > 0.0) {
            return false;
        }
        let n = left.grad * left.grad * hr + right.grad * right.grad * hl;
        n * (1.0 + SCREEN_KAPPA) < self.limit * (hl * hr)
    }
}

impl SplitScorer<'_> {
    /// An `f32` approximation of [`Self::score_run`] (`acc` holds each
    /// candidate's accumulated statistics; `-inf` for invalid candidates,
    /// validity decided exactly): each child contributes `G · (G / (H +
    /// λ))`, the closed form of its gain at the optimal weight. Valid only
    /// under [`Self::approx_exact`].
    #[inline]
    pub(super) fn approx_run<const ACC_LEFT: bool>(
        &self,
        total: GradStats,
        acc: &[GradStats],
        approx: &mut [f32],
    ) {
        // A child is valid when `H > 0` and `H >= min_child_weight`: one
        // of the two tests implies the other, so each child needs one
        // compare.
        if self.reg.min_child_weight > 0.0 {
            self.approx_run_with::<ACC_LEFT, true>(total, acc, approx);
        } else {
            self.approx_run_with::<ACC_LEFT, false>(total, acc, approx);
        }
    }

    /// [`Self::approx_run`] with `MCW_POSITIVE` = `min_child_weight > 0`
    /// (a child is then valid when `H >= min_child_weight`, else when
    /// `H > 0`).
    #[inline(always)]
    #[allow(
        clippy::needless_bitwise_bool,
        reason = "non-short-circuit validity keeps the loop branch-free so it vectorizes"
    )]
    fn approx_run_with<const ACC_LEFT: bool, const MCW_POSITIVE: bool>(
        &self,
        total: GradStats,
        acc: &[GradStats],
        approx: &mut [f32],
    ) {
        let RegParams {
            lambda,
            min_child_weight,
            ..
        } = *self.reg;
        let root_gain = self.root_gain;
        let gain = |g: f64, h: f64| -> f32 {
            let g = g as f32;
            g * (g / (h + lambda) as f32)
        };
        let valid_child = |h: f64| {
            if MCW_POSITIVE {
                h >= min_child_weight
            } else {
                h > 0.0
            }
        };
        let n = approx.len();
        let acc = &acc[..n];
        for i in 0..n {
            let (ag, ah) = (acc[i].grad, acc[i].hess);
            let (og, oh) = (total.grad - ag, total.hess - ah);
            let (lg, lh, rg, rh) = if ACC_LEFT {
                (ag, ah, og, oh)
            } else {
                (og, oh, ag, ah)
            };
            // Non-short-circuit `&` keeps the loop free of branches, so it
            // vectorizes.
            let valid = valid_child(lh) & valid_child(rh);
            let chg = (gain(lg, lh) + gain(rg, rh)) - root_gain;
            approx[i] = if valid { chg } else { f32::NEG_INFINITY };
        }
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

/// Candidates scored per batch by [`scan_numeric_splits`].
const SCAN_RUN: usize = 64;

/// The outcome of [`scan_numeric_splits`] for one feature.
pub(super) enum NumericScan {
    /// No candidate has a finite loss change.
    Empty,
    /// The first candidate (in [`for_each_numeric_split`] order) with the
    /// largest finite loss change.
    Best {
        loss_chg: f32,
        pos: SplitPos,
        children: Children,
    },
    /// Some candidate scored NaN, whose replacement depends on the
    /// incumbent's feature ([`need_replace`]): replay the feature with
    /// [`for_each_numeric_split`].
    Nan,
}

/// [`SplitScorer::loss_chg`] of every candidate of one feature, batched: the
/// prefix sums of a run are formed first (in the same order), then every
/// candidate of the run is scored branch-free (invalid or monotone-violating
/// candidates as `-inf`) so the arithmetic vectorizes, and the run is
/// scanned for its first maximum.
///
/// Sequential [`xgb_update`] over one feature's candidates keeps the first
/// candidate with the largest finite loss change, if that one replaces the
/// incumbent, and never takes infinite ones: offering only the returned
/// [`NumericScan::Best`] to [`xgb_update`] picks the same split. NaN loss
/// changes are the exception ([`NumericScan::Nan`]).
pub(super) fn scan_numeric_splits(
    bins: &[GradStats],
    first: usize,
    total: GradStats,
    dense: bool,
    scorer: &SplitScorer,
    scratch: &mut ScanScratch,
) -> NumericScan {
    if bins.len() <= FILTER_BINS
        && scorer.approx_exact()
        && let Some(scan) = scan_filtered(bins, first, total, dense, scorer, scratch)
    {
        return scan;
    }
    scan_batched(bins, first, total, dense, scorer)
}

/// [`scan_numeric_splits`] without the approximate prefilter: every
/// candidate scored exactly, in runs of [`SCAN_RUN`].
fn scan_batched(
    bins: &[GradStats],
    first: usize,
    total: GradStats,
    dense: bool,
    scorer: &SplitScorer,
) -> NumericScan {
    let mut run = RunMax {
        best: f32::NEG_INFINITY,
        at: None,
    };
    let mut grad = [0f64; SCAN_RUN];
    let mut hess = [0f64; SCAN_RUN];
    let mut loss = [0f32; SCAN_RUN];
    let mut found = None;

    // Forward pass: bins `..= offset` left, missing values right.
    let mut acc = GradStats::default();
    for (index, chunk) in bins.chunks(SCAN_RUN).enumerate() {
        let n = chunk.len();
        for (k, &bin) in chunk.iter().enumerate() {
            acc.add(bin);
            grad[k] = acc.grad;
            hess[k] = acc.hess;
        }
        scorer.score_run::<true>(total, &grad[..n], &hess[..n], &mut loss[..n]);
        if !run.scan(&loss[..n], &grad, &hess, index * SCAN_RUN) {
            return NumericScan::Nan;
        }
    }
    if let Some((offset, left)) = run.at.take() {
        found = Some((
            SplitPos::Bin(first + offset),
            Children::new(false, left, total.sub(left)),
        ));
    }
    if !(dense || acc == total) {
        // Backward pass: bins `>= offset` right, missing values left.
        let mut suffix = GradStats::default();
        let mut end = bins.len();
        let mut base = 0;
        while end > 0 {
            let n = end.min(SCAN_RUN);
            for k in 0..n {
                suffix.add(bins[end - 1 - k]);
                grad[k] = suffix.grad;
                hess[k] = suffix.hess;
            }
            scorer.score_run::<false>(total, &grad[..n], &hess[..n], &mut loss[..n]);
            if !run.scan(&loss[..n], &grad, &hess, base) {
                return NumericScan::Nan;
            }
            base += n;
            end -= n;
        }
        if let Some((step, right)) = run.at {
            let offset = bins.len() - 1 - step;
            let pos = if offset == 0 {
                SplitPos::BelowBins
            } else {
                SplitPos::Bin(first + offset - 1)
            };
            found = Some((pos, Children::new(true, total.sub(right), right)));
        }
    }
    match found {
        Some((pos, children)) => NumericScan::Best {
            loss_chg: run.best,
            pos,
            children,
        },
        None => NumericScan::Empty,
    }
}

/// The most bins per feature [`scan_filtered`] handles (its candidate
/// buffers live on the stack); wider features take the exact batched scan.
const FILTER_BINS: usize = 256;

/// Candidates per run of [`scan_filtered`]'s vectorized threshold test.
const FILTER_RUN: usize = 16;

/// Relative error allowance of [`SplitScorer::approx_run`] against
/// [`SplitScorer::loss_chg`]: for a valid candidate the two differ by at most
/// `APPROX_MARGIN · (U + |root_gain|) + UNDERFLOW_MARGIN · (D_l + D_r + 1)`,
/// where `D = H + λ` per child (the `f64` sum both scorers start from) and
/// `U = Σ G² / D` is the real closed-form gain of the children.
///
/// Each `f32` operation or conversion gives `x(1 + δ) + η` with `|δ| <= ε =
/// 2^-24` and `|η| <= τ = 2^-150`, `η` only below the normal range (where
/// `f32` sums and differences are exact); `f64` rounding is far below both.
/// Under [`SplitScorer::approx_exact`] every `D >= 1e-3`, and
/// [`scan_filtered`] defers to the exact scan unless every `D` and every
/// candidate's gain is below `1e30`, so nothing overflows.
///
/// - Exact: at the `f32` weight `w = w*(1 + δ) + η` (`w* = -G/D`), the child
///   gain `-(2Gw + D·w²)` equals `G²/D - D(w - w*)²`, so the weight's
///   rounding enters only squared. What is left per child is the rounding of
///   `w²`, which `D` scales to `εU + Dτ` (large when `w²` is subnormal), and
///   of the gain (`εU + τ`); with the sum and the `root_gain` subtraction,
///   the exact score is within `4ε(U + |root|) + (D_l + D_r + 2)τ` of `U -
///   root_gain`.
/// - Approximation: per child two conversions, the quotient and the product
///   (`5εU`, plus `τ` from the product; a quotient's `τ` is scaled by `|G| <
///   D · 2^-126`, and a subnormal `G` leaves `U` and the product below
///   `2^-240`), then the sum and the subtraction: within `7ε(U + |root|) +
///   2τ`.
///
/// The relative parts total under `11ε ≈ 2^-20.5` and the absolute ones
/// `(D_l + D_r + 4)τ`; `2^-17` and [`UNDERFLOW_MARGIN`] `= 64τ` leave an
/// order of magnitude to spare.
const APPROX_MARGIN: f64 = 1.0 / 131_072.0;

/// Absolute error allowance of [`SplitScorer::approx_run`] against
/// [`SplitScorer::loss_chg`], per unit of `D_l + D_r + 1`: `2^-144` (derived
/// at [`APPROX_MARGIN`]).
const UNDERFLOW_MARGIN: f64 = f64::from_bits((1023 - 144) << 52);

/// [`scan_numeric_splits`] with an approximate prefilter: every candidate is
/// first scored by [`SplitScorer::approx_run`] (a single `f32` division per
/// child), and only candidates whose approximation lies within the error
/// allowance of the best approximation are scored exactly, in order. A
/// candidate with the largest exact loss change always passes the filter: its
/// approximation is within the allowance of its exact value, which is at
/// least the approximate maximum's exact value. The result is therefore the
/// exact scan's. `None` (non-finite approximations, or gains or `H + λ` too
/// large for the error bound) defers to the exact scan.
fn scan_filtered(
    bins: &[GradStats],
    first: usize,
    total: GradStats,
    dense: bool,
    scorer: &SplitScorer,
    scratch: &mut ScanScratch,
) -> Option<NumericScan> {
    let n = bins.len();
    let mut prefix = Prefix::default();
    for (&bin, s) in bins.iter().zip(&mut scratch.acc[..n]) {
        prefix.add(bin);
        *s = prefix.acc;
    }
    scan_filtered_rest(bins, first, total, dense, scorer, scratch, prefix)
}

/// A forward prefix sum in progress: the running statistics and whether
/// every bin so far has a non-negative Hessian (tracked alongside, where
/// the sum's dependent chain leaves the vector units idle).
#[derive(Clone, Copy)]
struct Prefix {
    acc: GradStats,
    nonneg: bool,
}

impl Default for Prefix {
    fn default() -> Self {
        Prefix {
            acc: GradStats::default(),
            nonneg: true,
        }
    }
}

impl Prefix {
    #[inline(always)]
    fn add(&mut self, bin: GradStats) {
        self.acc.add(bin);
        self.nonneg &= bin.hess >= 0.0;
    }
}

/// [`scan_numeric_splits`] of two features, whose forward prefix sums (one
/// dependent `f64` chain per feature) are formed in one interleaved loop so
/// the two chains overlap. Each feature's result is its own scan's.
pub(super) fn scan_numeric_pair(
    a: &NumericInput,
    b: &NumericInput,
    scratch: [&mut ScanScratch; 2],
) -> [NumericScan; 2] {
    let [sa, sb] = scratch;
    let filtered = |x: &NumericInput| x.bins.len() <= FILTER_BINS && x.scorer.approx_exact();
    if !(filtered(a) && filtered(b)) {
        return [a.scan(sa), b.scan(sb)];
    }
    let (na, nb) = (a.bins.len(), b.bins.len());
    let common = na.min(nb);
    let (mut acc_a, mut acc_b) = (Prefix::default(), Prefix::default());
    {
        let (sa, sb) = (&mut sa.acc[..common], &mut sb.acc[..common]);
        let (ba, bb) = (&a.bins[..common], &b.bins[..common]);
        for i in 0..common {
            acc_a.add(ba[i]);
            acc_b.add(bb[i]);
            (sa[i], sb[i]) = (acc_a.acc, acc_b.acc);
        }
    }
    for (x, s, acc) in [(a, &mut *sa, &mut acc_a), (b, &mut *sb, &mut acc_b)] {
        let n = x.bins.len();
        for i in common..n {
            acc.add(x.bins[i]);
            s.acc[i] = acc.acc;
        }
    }
    let finish = |x: &NumericInput, s: &mut ScanScratch, acc| {
        scan_filtered_rest(x.bins, x.first, x.total, x.dense, &x.scorer, s, acc)
            .unwrap_or_else(|| scan_batched(x.bins, x.first, x.total, x.dense, &x.scorer))
    };
    [finish(a, sa, acc_a), finish(b, sb, acc_b)]
}

/// One feature's histogram and scoring context, as [`scan_numeric_splits`]
/// takes them.
pub(super) struct NumericInput<'a> {
    /// The feature's bins (global bins from `first`).
    pub(super) bins: &'a [GradStats],
    pub(super) first: usize,
    /// The node's statistics.
    pub(super) total: GradStats,
    /// The index has no missing values.
    pub(super) dense: bool,
    pub(super) scorer: SplitScorer<'a>,
}

impl NumericInput<'_> {
    /// [`scan_numeric_splits`] of this feature.
    pub(super) fn scan(&self, scratch: &mut ScanScratch) -> NumericScan {
        scan_numeric_splits(
            self.bins,
            self.first,
            self.total,
            self.dense,
            &self.scorer,
            scratch,
        )
    }
}

/// [`scan_filtered`] after its forward prefix sums: `scratch.acc` holds
/// them for every bin, and `prefix` is their last value (with whether every
/// bin's Hessian is non-negative).
#[allow(
    clippy::needless_bitwise_bool,
    reason = "branch-free overflow and threshold tests vectorize"
)]
fn scan_filtered_rest(
    bins: &[GradStats],
    first: usize,
    total: GradStats,
    dense: bool,
    scorer: &SplitScorer,
    scratch: &mut ScanScratch,
    prefix: Prefix,
) -> Option<NumericScan> {
    let Prefix { acc, nonneg } = prefix;
    let n = bins.len();
    let ScanScratch { acc: stats, approx } = scratch;
    let (stats, approx) = (&mut stats[..2 * n], &mut approx[..2 * n]);
    // The extreme accumulated Hessians, which bound every child's `H`. With
    // every bin's Hessian non-negative (the usual case) each running sum
    // only grows, since adding a non-negative value never rounds below the
    // sum, so they are the first and last sums; otherwise a separate pass
    // finds them.
    let extremes = |sums: &[GradStats]| match (nonneg, sums.first(), sums.last()) {
        (true, Some(first), Some(last)) => (first.hess, last.hess),
        _ => min_max_hess(sums),
    };
    let (mut hess_lo, mut hess_hi) = extremes(&stats[..n]);
    scorer.approx_run::<true>(total, &stats[..n], &mut approx[..n]);
    // Candidates `n..2n` are the backward pass, whose accumulated statistics
    // are the right child's.
    let mut m = n;
    if !(dense || acc == total) {
        let mut suffix = GradStats::default();
        for (&bin, s) in bins.iter().rev().zip(&mut stats[n..]) {
            suffix.add(bin);
            *s = suffix;
        }
        let (lo, hi) = extremes(&stats[n..]);
        hess_lo = hess_lo.min(lo);
        hess_hi = hess_hi.max(hi);
        scorer.approx_run::<false>(total, &stats[n..], &mut approx[n..]);
        m = 2 * n;
    }
    let mut max = f32::NEG_INFINITY;
    let mut overflow = false;
    for &a in &approx[..m] {
        // Invalid candidates are `-inf`; valid ones are finite unless the
        // statistics overflow `f32`.
        overflow |= a.is_nan() | (a == f32::INFINITY);
        max = max.max(a);
    }
    if overflow {
        return None;
    }
    if max == f32::NEG_INFINITY {
        return Some(NumericScan::Empty);
    }
    let root = f64::from(scorer.root_gain);
    // `gain(left) + gain(right) + |root_gain|` of the largest candidate,
    // which bounds every candidate's relative error.
    let scale = (f64::from(max) + root + root.abs()).max(0.0);
    // The largest `H + λ` of any child, the other child's `H` being the
    // total's minus the accumulated one.
    let max_d = hess_hi.max(total.hess - hess_lo) + scorer.reg.lambda;
    if scale >= 1e30 || max_d.is_nan() || max_d >= 1e30 {
        return None;
    }
    // Each of the two candidates compared (the approximate and the exact
    // maximum) is off by at most its relative and absolute allowance.
    let absolute = UNDERFLOW_MARGIN * (2.0 * max_d + 1.0);
    let threshold = f64::from(max) - 2.0 * (APPROX_MARGIN * scale + absolute);
    // The largest `f32` at most `threshold`: comparing in `f32` against it
    // keeps every candidate the `f64` comparison keeps.
    let mut cutoff = threshold as f32;
    if f64::from(cutoff) > threshold {
        cutoff = cutoff.next_down();
    }

    let mut best = f32::NEG_INFINITY;
    let mut found = None;
    for (chunk, run) in approx[..m].chunks(FILTER_RUN).enumerate() {
        // Most runs hold no candidate near the maximum; this test vectorizes.
        if !run.iter().fold(false, |any, &a| any | (a >= cutoff)) {
            continue;
        }
        for (k, &a) in run.iter().enumerate() {
            if a < cutoff || f64::from(a) < threshold {
                continue;
            }
            let i = chunk * FILTER_RUN + k;
            let acc = stats[i];
            let (pos, children) = if i < n {
                (
                    SplitPos::Bin(first + i),
                    Children::new(false, acc, total.sub(acc)),
                )
            } else {
                let offset = n - 1 - (i - n);
                let pos = if offset == 0 {
                    SplitPos::BelowBins
                } else {
                    SplitPos::Bin(first + offset - 1)
                };
                (pos, Children::new(true, total.sub(acc), acc))
            };
            let Some(score) = scorer.loss_chg(children.left, children.right) else {
                continue;
            };
            let l = score.loss_chg;
            if l.is_nan() {
                return Some(NumericScan::Nan);
            }
            if l > best && l.is_finite() {
                best = l;
                found = Some((pos, children));
            }
        }
    }
    Some(match found {
        Some((pos, children)) => NumericScan::Best {
            loss_chg: best,
            pos,
            children,
        },
        None => NumericScan::Empty,
    })
}

/// The smallest and largest non-NaN Hessians of `stats` (`(inf, -inf)` when
/// there are none), over four independent lanes so the comparisons neither
/// form one long dependency chain nor depend on the order.
#[inline]
fn min_max_hess(stats: &[GradStats]) -> (f64, f64) {
    let mut lo = [f64::INFINITY; 4];
    let mut hi = [f64::NEG_INFINITY; 4];
    let (quads, rest) = stats.as_chunks::<4>();
    for quad in quads {
        for k in 0..4 {
            lo[k] = lo[k].min(quad[k].hess);
            hi[k] = hi[k].max(quad[k].hess);
        }
    }
    for (k, s) in rest.iter().enumerate() {
        lo[k] = lo[k].min(s.hess);
        hi[k] = hi[k].max(s.hess);
    }
    (
        lo[0].min(lo[1]).min(lo[2].min(lo[3])),
        hi[0].max(hi[1]).max(hi[2].max(hi[3])),
    )
}

thread_local! {
    /// Each thread's pair of [`ScanScratch`] buffers ([`with_scan_scratch`]).
    static SCAN_SCRATCH: RefCell<[ScanScratch; 2]> =
        RefCell::new([ScanScratch::new(), ScanScratch::new()]);
}

/// Run `f` with this thread's pair of scan buffers, allocated once per
/// thread instead of per node (fresh ones if a caller up the stack holds
/// them).
pub(super) fn with_scan_scratch<R>(f: impl FnOnce(&mut [ScanScratch; 2]) -> R) -> R {
    SCAN_SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut scratch) => f(&mut scratch),
        Err(_) => f(&mut [ScanScratch::new(), ScanScratch::new()]),
    })
}

/// Candidate buffers of [`scan_numeric_splits`], reused across features.
pub(super) struct ScanScratch {
    /// Each candidate's accumulated statistics (forward pass, then backward).
    acc: Vec<GradStats>,
    approx: Vec<f32>,
}

impl ScanScratch {
    pub(super) fn new() -> Self {
        ScanScratch {
            acc: vec![GradStats::default(); 2 * FILTER_BINS],
            approx: vec![0.0; 2 * FILTER_BINS],
        }
    }
}

/// The running first maximum of [`scan_numeric_splits`]: the best finite
/// loss change so far and, when the current pass reached it, the candidate's
/// index in the pass and its accumulated statistics.
struct RunMax {
    best: f32,
    at: Option<(usize, GradStats)>,
}

impl RunMax {
    /// Scan one scored run (candidates `base..`); `false` on a NaN.
    #[inline]
    fn scan(&mut self, loss: &[f32], grad: &[f64], hess: &[f64], base: usize) -> bool {
        for (k, &l) in loss.iter().enumerate() {
            if l.is_nan() {
                return false;
            }
            if l > self.best && l.is_finite() {
                self.best = l;
                self.at = Some((base + k, GradStats::new(grad[k], hess[k])));
            }
        }
        true
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
pub(crate) fn xgb_calc_weight(stats: GradStats, reg: &RegParams) -> f64 {
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
    let n_nodes = tree.num_nodes();
    for (id, (&stats, &bounds)) in stats[..n_nodes].iter().zip(&bounds[..n_nodes]).enumerate() {
        if tree.node(id).is_leaf() {
            let w = calc_weight_bounded(stats, reg, bounds);
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

    /// The split [`xgb_update`] picks from the sequential
    /// [`for_each_numeric_split`] search and from [`scan_numeric_splits`]'s
    /// result, as `(expected, actual)`. A [`NumericScan::Nan`] (which needs
    /// a candidate that scores NaN) is replayed sequentially, as the
    /// builders do.
    fn sequential_and_scanned(
        bins: &[GradStats],
        total: GradStats,
        dense: bool,
        scorer: &SplitScorer,
    ) -> (BestSplit, BestSplit) {
        let offer = |best: &mut BestSplit, pos, children: Children| {
            if let Some(score) = scorer.loss_chg(children.left, children.right) {
                xgb_update(best, 0, pos, children, score);
            }
        };
        let mut expected = BestSplit::none();
        let mut nan = false;
        for_each_numeric_split(bins, 0, total, dense, |pos, children| {
            nan |= scorer
                .loss_chg(children.left, children.right)
                .is_some_and(|score| score.loss_chg.is_nan());
            offer(&mut expected, pos, children);
        });
        let mut actual = BestSplit::none();
        match scan_numeric_splits(bins, 0, total, dense, scorer, &mut ScanScratch::new()) {
            NumericScan::Empty => {}
            NumericScan::Best { pos, children, .. } => offer(&mut actual, pos, children),
            NumericScan::Nan => {
                assert!(nan, "no candidate scores NaN");
                actual = expected.clone();
            }
        }
        (expected, actual)
    }

    /// Everything that identifies a recorded numeric split, bit for bit.
    fn split_key(b: &BestSplit) -> (u64, Option<usize>, bool, [u64; 4]) {
        (
            b.loss_chg.to_bits(),
            b.split_bin,
            b.default_left,
            [b.left.grad, b.left.hess, b.right.grad, b.right.hess].map(f64::to_bits),
        )
    }

    /// Random histogram bins of `n` bins, each the sum of up to 20 `f32`
    /// gradient pairs with gradients in `±2 · grad_scale` and Hessians in
    /// `[0.05, 1.05) · hess_scale` (a quarter of the bins empty), and on
    /// every fifth trial a repeated block (equal partial sums on both sides).
    fn random_bins(
        rng: &mut crate::rng::Rng,
        n: usize,
        (grad_scale, hess_scale): (f32, f32),
        repeat: bool,
    ) -> Vec<GradStats> {
        let mut bins = vec![GradStats::default(); n];
        for bin in &mut bins {
            if rng.below(4) == 0 {
                continue;
            }
            for _ in 0..rng.range(1..20) {
                let g = (rng.f32() * 4.0 - 2.0) * grad_scale;
                let h = (0.05 + rng.f32()) * hess_scale;
                bin.add(GradStats::from_pair(GradPair::new(g, h)));
            }
        }
        if repeat {
            let half = n / 2;
            for i in 0..half {
                bins[n - 1 - i] = bins[i];
            }
        }
        bins
    }

    /// The batched and prefiltered numeric scans pick the split the
    /// sequential [`for_each_numeric_split`] search picks, bit for bit, over
    /// random histograms (empty bins and repeated bins give tied
    /// candidates), missing mass, regularization, monotone directions, and
    /// features wider than the prefilter's buffers.
    #[test]
    fn numeric_scan_matches_sequential_search() {
        let mut rng = crate::rng::Rng::new(7);
        for trial in 0..4000 {
            let n = 2 + rng.range(0..300);
            let bins = random_bins(&mut rng, n, (1.0, 1.0), trial % 5 == 0);
            let mut total = GradStats::default();
            for &bin in &bins {
                total.add(bin);
            }
            let dense = trial % 3 == 0;
            if !dense && rng.below(2) == 0 {
                total.add(GradStats::new(f64::from(rng.f32()) * 8.0 - 4.0, 3.0));
            }
            let reg = RegParams {
                lambda: [0.0, 1.0, 0.1][rng.range(0..3)],
                alpha: if rng.below(6) == 0 { 0.5 } else { 0.0 },
                max_delta_step: if rng.below(6) == 0 { 0.7 } else { 0.0 },
                min_child_weight: [0.0, 1.0, 5.0][rng.range(0..3)],
            };
            let dir = [0, 0, 0, 1, -1][rng.range(0..5)];
            let scorer = SplitScorer {
                reg: &reg,
                root_gain: xgb_node_gain(total, &reg, Bounds::default()),
                bounds: Bounds::default(),
                dir,
            };
            let (expected, actual) = sequential_and_scanned(&bins, total, dense, &scorer);
            assert_eq!(split_key(&actual), split_key(&expected), "trial {trial}");
        }
    }

    /// [`numeric_scan_matches_sequential_search`] with gradients from `1e-30`
    /// to `1e30` and Hessians up to `1e38` (sums past `f32::MAX` included),
    /// where `f32` weights, their squares, and the prefilter's quotients
    /// underflow into subnormals or overflow.
    #[test]
    fn numeric_scan_matches_sequential_search_at_extreme_scales() {
        let mut rng = crate::rng::Rng::new(13);
        let scales = [1e-30f32, 1e-20, 1e-10, 1e-3, 1.0, 1e10, 1e20, 1e30];
        for trial in 0..4000 {
            let n = 2 + rng.range(0..40);
            let grad_scale = scales[rng.range(0..scales.len())];
            let hess_scale = [1.0f32, 1e10, 1e20, 1e30, 1e36, 1e38][rng.range(0..6)];
            let bins = random_bins(&mut rng, n, (grad_scale, hess_scale), trial % 5 == 0);
            let mut total = GradStats::default();
            for &bin in &bins {
                total.add(bin);
            }
            let dense = trial % 3 == 0;
            if !dense && rng.below(2) == 0 {
                let g = f64::from(rng.f32() * 8.0 - 4.0) * f64::from(grad_scale);
                total.add(GradStats::new(g, 3.0 * f64::from(hess_scale)));
            }
            let reg = RegParams {
                lambda: [1.0, 0.1][rng.range(0..2)],
                alpha: 0.0,
                max_delta_step: 0.0,
                min_child_weight: [0.0, 1.0][rng.range(0..2)],
            };
            let scorer = SplitScorer {
                reg: &reg,
                root_gain: xgb_node_gain(total, &reg, Bounds::default()),
                bounds: Bounds::default(),
                dir: 0,
            };
            let (expected, actual) = sequential_and_scanned(&bins, total, dense, &scorer);
            assert_eq!(split_key(&actual), split_key(&expected), "trial {trial}");
        }
    }

    /// Squared error with `x = [0, 1, 2]`, labels `[-1.02e-22, -1e-24,
    /// 1.03e-22]`, weights `1e38` and base score 0 under the default
    /// regularization, one bin per value.
    fn subnormal_weight_square_bins() -> (Vec<GradStats>, GradStats, RegParams) {
        let bins: Vec<GradStats> = [-1.02e-22f32, -1e-24, 1.03e-22]
            .iter()
            .map(|&label| GradStats::from_pair(GradPair::new((0.0 - label) * 1e38, 1e38)))
            .collect();
        let mut total = GradStats::default();
        for &bin in &bins {
            total.add(bin);
        }
        let reg = RegParams {
            lambda: 1.0,
            alpha: 0.0,
            max_delta_step: 0.0,
            min_child_weight: 1.0,
        };
        (bins, total, reg)
    }

    /// Each child's `f32` weight (about `1e-22`) squares to a subnormal, so
    /// the exact score `(H + λ) · w²` is off from `G² / (H + λ)` by far more
    /// than the relative error allowance: the true winner (bins `..= 0`
    /// left, loss change ~`1.5798e-6`) approximates below the runner-up
    /// (~`1.5011e-6` exact, ~`1.5913e-6` approximated). The prefilter must
    /// still keep it.
    #[test]
    fn numeric_scan_keeps_the_winner_when_weights_square_to_subnormals() {
        let (bins, total, reg) = subnormal_weight_square_bins();
        let scorer = SplitScorer {
            reg: &reg,
            root_gain: xgb_node_gain(total, &reg, Bounds::default()),
            bounds: Bounds::default(),
            dir: 0,
        };
        assert!(scorer.approx_exact());
        let (expected, actual) = sequential_and_scanned(&bins, total, true, &scorer);
        assert_eq!(expected.split_bin, Some(0));
        assert_eq!(split_key(&actual), split_key(&expected));
    }

    /// [`Screen::cannot_beat`] on the true winner of
    /// [`numeric_scan_keeps_the_winner_when_weights_square_to_subnormals`],
    /// whose exact loss change exceeds its `G² / (H + λ)` estimate by 1.2%:
    /// no incumbent below the exact loss change rules it out.
    #[test]
    fn cannot_beat_allows_for_subnormal_weight_squares() {
        let (bins, total, reg) = subnormal_weight_square_bins();
        let scorer = SplitScorer {
            reg: &reg,
            root_gain: xgb_node_gain(total, &reg, Bounds::default()),
            bounds: Bounds::default(),
            dir: 0,
        };
        let (left, right) = (bins[0], total.sub(bins[0]));
        let exact = scorer.loss_chg(left, right).unwrap().loss_chg;
        for incumbent in [exact * 0.98, exact * 0.99, exact.next_down()] {
            assert!(
                !scorer.cannot_beat(left, right, f64::from(incumbent)),
                "{incumbent} < {exact}"
            );
        }
    }

    /// [`Screen::cannot_beat`] never rules out a candidate whose exact
    /// loss change exceeds the incumbent, including incumbents a few `f32`
    /// steps around the exact value, and does rule out clearly worse ones.
    #[test]
    fn cannot_beat_is_conservative() {
        let mut rng = crate::rng::Rng::new(11);
        let mut ruled_out = 0;
        for _ in 0..20_000 {
            let scale = [1e-3f64, 1.0, 1e3][rng.range(0..3)];
            let stats = |rng: &mut crate::rng::Rng| {
                GradStats::new((rng.f64() * 2.0 - 1.0) * scale, 0.01 + rng.f64() * scale)
            };
            let (left, right) = (stats(&mut rng), stats(&mut rng));
            let total = GradStats::new(left.grad + right.grad, left.hess + right.hess);
            let reg = RegParams {
                lambda: [1.0, 0.1][rng.range(0..2)],
                alpha: 0.0,
                max_delta_step: 0.0,
                min_child_weight: 0.0,
            };
            let scorer = SplitScorer {
                reg: &reg,
                root_gain: xgb_node_gain(total, &reg, Bounds::default()),
                bounds: Bounds::default(),
                dir: 0,
            };
            let Some(score) = scorer.loss_chg(left, right) else {
                continue;
            };
            let exact = score.loss_chg;
            let mut incumbent = exact;
            for _ in 0..4 {
                incumbent = incumbent.next_down();
            }
            for _ in 0..8 {
                let beats = exact > incumbent;
                if scorer.cannot_beat(left, right, f64::from(incumbent)) {
                    assert!(!beats, "{left:?} {right:?} {incumbent} {exact}");
                }
                incumbent = incumbent.next_up();
            }
            if scorer.cannot_beat(left, right, f64::from(exact) + f64::from(exact.abs()) + 1.0) {
                ruled_out += 1;
            }
        }
        assert!(ruled_out > 10_000, "{ruled_out}");
    }

    /// [`ScreenBound::rules_out`], the exact builder's precomputed screen,
    /// rules out only candidates [`Screen::cannot_beat`] rules out, for
    /// children formed as the scan forms them (`right = total - left`, both
    /// Hessians non-negative), in either orientation, at magnitudes where
    /// the underflow allowance matters, and for incumbents around each
    /// candidate's own score.
    #[test]
    fn screen_bound_implies_cannot_beat() {
        let mut rng = crate::rng::Rng::new(23);
        let mut ruled_out = 0;
        for _ in 0..20_000 {
            let g_scale = [1e-30f64, 1e-3, 1.0, 1e3, 1e30][rng.range(0..5)];
            let h_scale = [1e-30f64, 1e-3, 1.0, 1e3, 1e30][rng.range(0..5)];
            let total = GradStats::new(
                (rng.f64() * 2.0 - 1.0) * g_scale,
                0.01 * h_scale + rng.f64() * h_scale,
            );
            let left = GradStats::new((rng.f64() * 2.0 - 1.0) * g_scale, rng.f64() * total.hess);
            let right = total.sub(left);
            if right.hess < 0.0 {
                continue;
            }
            let lambda = [1.0, 1e-3, 0.0][rng.range(0..3)];
            let reg = RegParams {
                lambda,
                alpha: 0.0,
                max_delta_step: 0.0,
                min_child_weight: 1e-3,
            };
            let scorer = SplitScorer {
                reg: &reg,
                root_gain: xgb_node_gain(total, &reg, Bounds::default()),
                bounds: Bounds::default(),
                dir: 0,
            };
            let Some(screen) = scorer.screen() else {
                continue;
            };
            for (l, r) in [(left, right), (right, left)] {
                let Some(score) = scorer.loss_chg(l, r) else {
                    continue;
                };
                let mut incumbent = score.loss_chg;
                for _ in 0..4 {
                    incumbent = incumbent.next_down();
                }
                for _ in 0..8 {
                    let incumbent64 = f64::from(incumbent);
                    if screen.bound(total.hess, incumbent64).rules_out(l, r) {
                        assert!(
                            screen.cannot_beat(l, r, incumbent64),
                            "{l:?} {r:?} {total:?} {incumbent}"
                        );
                    }
                    incumbent = incumbent.next_up();
                }
                let far = f64::from(score.loss_chg) + f64::from(score.loss_chg.abs()) + 1.0;
                if screen.bound(total.hess, far).rules_out(l, r) {
                    assert!(screen.cannot_beat(l, r, far));
                    ruled_out += 1;
                }
            }
        }
        assert!(ruled_out > 10_000, "{ruled_out}");
    }

    /// [`cannot_beat_is_conservative`] at gradient magnitudes from `1e-30`
    /// to `1e30` and Hessians up to `1e38`, where the `f32` weights and
    /// their squares underflow.
    #[test]
    fn cannot_beat_is_conservative_at_extreme_scales() {
        let mut rng = crate::rng::Rng::new(17);
        let grad_scales = [1e-30f64, 1e-20, 1e-10, 1.0, 1e10, 1e20, 1e30];
        let hess_scales = [1.0f64, 1e10, 1e20, 1e30, 1e38];
        for _ in 0..20_000 {
            let grad_scale = grad_scales[rng.range(0..grad_scales.len())];
            let hess_scale = hess_scales[rng.range(0..hess_scales.len())];
            let mut stats = || {
                GradStats::new(
                    (rng.f64() * 2.0 - 1.0) * grad_scale,
                    (0.01 + rng.f64()) * hess_scale,
                )
            };
            let (left, right) = (stats(), stats());
            let total = GradStats::new(left.grad + right.grad, left.hess + right.hess);
            let reg = RegParams {
                lambda: [1.0, 0.1][rng.range(0..2)],
                alpha: 0.0,
                max_delta_step: 0.0,
                min_child_weight: 0.0,
            };
            let scorer = SplitScorer {
                reg: &reg,
                root_gain: xgb_node_gain(total, &reg, Bounds::default()),
                bounds: Bounds::default(),
                dir: 0,
            };
            let Some(score) = scorer.loss_chg(left, right) else {
                continue;
            };
            let exact = score.loss_chg;
            if !exact.is_finite() {
                continue;
            }
            let mut incumbent = exact;
            for _ in 0..8 {
                incumbent = incumbent.next_down();
                assert!(
                    !scorer.cannot_beat(left, right, f64::from(incumbent)),
                    "{left:?} {right:?} {incumbent} {exact}"
                );
            }
        }
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
