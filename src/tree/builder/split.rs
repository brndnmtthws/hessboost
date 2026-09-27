//! Numeric split search shared by the builders: [`SplitScorer`] (XGBoost's
//! `f32` candidate score and its vectorized, approximate, and `f64` variants),
//! the candidate order of one feature's histogram ([`for_each_numeric_split`]),
//! and the batched scan [`scan_numeric_splits`] with its `f32` prefilter
//! ([`SplitScorer::approx_run`], [`APPROX_MARGIN`]) and exact's division-free
//! screen ([`Screen::bound`], [`ScreenBound::rules_out`]), both proven to keep
//! the sequential choice.

use std::cell::RefCell;

use super::shared::{apply_bounds, xgb_gain_given_weight, xgb_weight};
use super::{Children, Score, SplitPos, children_valid};
use crate::tree::constraints::{Bounds, calc_weight_bounded, gain_at_weight, satisfies};
use crate::tree::gain::{GradStats, RegParams, calc_gain};

/// Everything a candidate's score depends on besides its children: the
/// regularization, the node's `root_gain` baseline ([`xgb_node_gain`](super::shared::xgb_node_gain)) and
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
            apply_bounds(w as f32, lower, upper)
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
        offer(
            SplitPos::backward(first, offset),
            Children::new(true, total.sub(suffix), suffix),
        );
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
    /// incumbent's feature ([`need_replace`](super::need_replace)): replay the feature with
    /// [`for_each_numeric_split`].
    Nan,
}

/// [`SplitScorer::loss_chg`] of every candidate of one feature, batched: the
/// prefix sums of a run are formed first (in the same order), then every
/// candidate of the run is scored branch-free (invalid or monotone-violating
/// candidates as `-inf`) so the arithmetic vectorizes, and the run is
/// scanned for its first maximum.
///
/// Sequential [`xgb_update`](super::xgb_update) over one feature's candidates keeps the first
/// candidate with the largest finite loss change, if that one replaces the
/// incumbent, and never takes infinite ones: offering only the returned
/// [`NumericScan::Best`] to [`xgb_update`](super::xgb_update) picks the same split. NaN loss
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
            found = Some((
                SplitPos::backward(first, offset),
                Children::new(true, total.sub(right), right),
            ));
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
                (
                    SplitPos::backward(first, offset),
                    Children::new(true, total.sub(acc), acc),
                )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objective::GradPair;
    use crate::tree::builder::shared::xgb_node_gain;
    use crate::tree::builder::{BestSplit, SplitLocation, xgb_update};

    /// The split [`xgb_update`](super::xgb_update) picks from the sequential
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

    /// The last bin a recorded numeric split sends left (`None`: below the
    /// bins).
    fn numeric_bin(b: &BestSplit) -> Option<usize> {
        match &b.location {
            SplitLocation::Numeric(pos) => pos.bin(),
            SplitLocation::Categories(categories) => panic!("categorical split {categories:?}"),
        }
    }

    /// Everything that identifies a recorded numeric split, bit for bit.
    fn split_key(b: &BestSplit) -> (u64, Option<usize>, bool, [u64; 4]) {
        (
            b.loss_chg.to_bits(),
            numeric_bin(b),
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
        assert_eq!(numeric_bin(&expected), Some(0));
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
}
