//! Tree construction algorithms.
//!
//! Each builder grows a single [`crate::tree::RegTree`] from per-instance
//! gradients. The exact builder is the reference; the histogram builder shares
//! the same regularized gain math. This module holds the split record every
//! builder fills ([`BestSplit`], [`SplitLocation`]) and XGBoost's tie rule;
//! `split` (numeric scans), `categorical` (category sweeps), `shared`
//! (configuration, weights, interaction state), and `partition` (row routing,
//! child histograms) hold what several builders share.

pub(crate) mod budget;
mod categorical;
mod exact;
mod hist;
mod lightgbm;
mod multi;
mod oblivious;
pub(crate) mod online;
mod partition;
mod shared;
mod split;

pub(crate) use exact::{ExactTreeBuilder, SortedColumns, all_rows};
pub use hist::HistTreeBuilder;
pub(crate) use multi::{MultiTreeBuilder, VectorGradients};
pub(crate) use oblivious::check_symmetric_input;
pub(crate) use shared::{LeafRows, xgb_calc_weight};

use std::num::NonZeroUsize;

use crate::K_RT_EPS;
use crate::data::quantile::HistCuts;
use crate::tree::constraints::{Bounds, child_bounds};
use crate::tree::gain::GradStats;
use crate::tree::{ChildLeaf, RegTree, SplitRule};
use partition::SplitRoute;

/// The bound set by a `max_depth` / `max_leaves` style parameter, where
/// `None` means unlimited.
pub(super) fn limit_or_unbounded(limit: Option<NonZeroUsize>) -> usize {
    limit.map_or(usize::MAX, NonZeroUsize::get)
}

/// Where a split sends a node's present values: a numeric position, or a set
/// of categories.
#[derive(Debug, Clone)]
pub(super) enum SplitLocation {
    Numeric(SplitPos),
    /// Category values, ascending. A [`BestSplit`] routes them to the tree's
    /// left child.
    Categories(Vec<u32>),
}

impl SplitLocation {
    #[inline]
    pub(super) fn is_categorical(&self) -> bool {
        matches!(self, SplitLocation::Categories(_))
    }
}

/// The best split found so far for one node.
///
/// Every builder shares this; its `location` is a value-space threshold under
/// exact search and a global-bin boundary under histogram search.
#[derive(Debug, Clone)]
pub(super) struct BestSplit {
    pub(super) loss_chg: f64,
    pub(super) feature: u32,
    pub(super) location: SplitLocation,
    pub(super) default_left: bool,
    pub(super) left: GradStats,
    pub(super) right: GradStats,
    /// Bounded child weights (used to derive monotone child bounds).
    pub(super) w_left: f64,
    pub(super) w_right: f64,
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
            location: SplitLocation::Numeric(SplitPos::BelowBins),
            default_left: true,
            left: GradStats::default(),
            right: GradStats::default(),
            w_left: 0.0,
            w_right: 0.0,
            children_swapped: false,
        }
    }

    /// A numeric split candidate at `pos` (a value-space threshold for exact
    /// search, a global-bin boundary for histogram search).
    pub(super) fn numeric(feature: u32, pos: SplitPos, children: Children, score: Score) -> Self {
        BestSplit::new(feature, SplitLocation::Numeric(pos), children, score)
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
        BestSplit::new(
            feature,
            SplitLocation::Categories(cat_left),
            children,
            score,
        )
    }

    fn new(feature: u32, location: SplitLocation, children: Children, score: Score) -> Self {
        BestSplit {
            loss_chg: score.loss_chg,
            feature,
            location,
            default_left: children.default_left,
            left: children.left,
            right: children.right,
            w_left: score.w_left,
            w_right: score.w_right,
            children_swapped: false,
        }
    }

    #[inline]
    pub(super) fn found(&self) -> bool {
        self.loss_chg > K_RT_EPS
    }

    /// How this split routes rows.
    fn route(&self) -> SplitRoute<'_> {
        SplitRoute {
            feature: self.feature,
            location: &self.location,
            default_left: self.default_left,
        }
    }

    /// Expand node `nid` of `tree` by this split, tested by `rule`, the
    /// children holding the bounded child weights as `f32`, and record the
    /// split's loss change. Returns the child ids.
    pub(super) fn expand(&self, tree: &mut RegTree, nid: usize, rule: SplitRule) -> (usize, usize) {
        let ids = tree.expand(
            nid,
            rule,
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

/// Where a numeric candidate split falls. Exact search records the threshold
/// in value space; histogram search records the global-bin boundary (`Bin(b)`:
/// bins `<= b` go left) or `BelowBins`, XGBoost's backward-pass endpoint
/// (`NumericBinLowerBound` at the feature's first bin, `-inf`): every present
/// bin goes right and only missing values go left.
#[derive(Debug, Clone, Copy)]
pub(super) enum SplitPos {
    Value(f32),
    Bin(usize),
    BelowBins,
}

impl SplitPos {
    /// The boundary of a backward pass over bins from `first` (bins `first +
    /// offset..` right): the bin below it, or [`SplitPos::BelowBins`] at the
    /// first bin.
    #[inline]
    pub(super) fn backward(first: usize, offset: usize) -> Self {
        if offset == 0 {
            SplitPos::BelowBins
        } else {
            SplitPos::Bin(first + offset - 1)
        }
    }

    /// The last global bin that goes left: `None` for [`SplitPos::BelowBins`]
    /// (and for exact search's value thresholds, which have no bin).
    #[inline]
    pub(super) fn bin(self) -> Option<usize> {
        match self {
            SplitPos::Bin(bin) => Some(bin),
            SplitPos::BelowBins | SplitPos::Value(_) => None,
        }
    }

    /// The tree threshold of this position: a bin's upper cut value in
    /// `cuts`, [`BELOW_ALL_VALUES`] below the bins, or the value itself.
    #[inline]
    pub(super) fn threshold(self, cuts: &HistCuts) -> f32 {
        match self {
            SplitPos::Bin(bin) => cuts.cut_value(bin),
            SplitPos::BelowBins => BELOW_ALL_VALUES,
            SplitPos::Value(value) => value,
        }
    }
}

/// A finite threshold below every finite feature value: `x < BELOW_ALL_VALUES`
/// is false for all finite `x`, so a split at it routes every present row
/// right and only missing values (`default_left`) left. Stands in for the
/// `-inf` XGBoost stores (`NumericBinLowerBound` at a feature's first bin, or
/// an overflowed `ColMaker` endpoint) because trees here require a finite
/// `split_cond`.
pub(super) const BELOW_ALL_VALUES: f32 = f32::MIN;

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
