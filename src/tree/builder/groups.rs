//! `min_child_groups`
//! ([`TrainingParams::min_child_groups`](crate::config::TrainingParams::min_child_groups)):
//! a split is legal only when both children hold rows of at least `k`
//! distinct split groups
//! ([`DMatrix::with_split_groups`](crate::data::DMatrix::with_split_groups)).
//!
//! A child's groups follow from the rows that reach it, not from its
//! parent's and its sibling's (a union cannot be undone the way a sibling's
//! histogram is subtracted), so every searched node counts its own sampled
//! rows ([`NodeGroups`]), laid out group by group. One pass over a feature's
//! entries, group by group, summarizes each of the node's groups by the
//! lowest and the highest *position* among its rows that hold the feature
//! (a numeric feature's local bin, or a category's place in the order a
//! categorical search scans) and by whether any of its rows misses the
//! feature, and counts the groups by those positions. A candidate split
//! sends the positions below its *cut* to one side and the others to the
//! other side, missing values to either: the low side holds a group whose
//! lowest position is below the cut, the high side one whose highest
//! position is at or above it, and the side that takes missing values every
//! group with a missing row. Both counts are monotone in the cut, so each
//! missing direction's legal cuts form one interval ([`LegalCuts`]), read
//! off the `k`-th smallest lowest and `k`-th largest highest positions. A
//! one-hot candidate (one category against the rest) is not a cut of one
//! order; [`CategoryGroups::one_hot`] counts the groups by their sets of
//! categories instead. All of it is integer counting, the same in any row
//! order and on any number of threads, in memory that follows the node's
//! rows: the layout and its code tables are the node's, and a feature's
//! pass keeps only its counts.

use std::borrow::Cow;
use std::ops::Range;

use rayon::prelude::*;

use super::SplitPos;
use super::categorical::MAX_CAT_TO_ONEHOT;
use super::shared::rayon_available;
use crate::config::TrainingParams;
use crate::data::DMatrix;
use crate::data::ghist::{BinIndex, Bins, Column, Columns, GHistIndex, missing_bit};
use crate::objective::GradPair;

/// Training rows at which [`SplitGroups::of`] codes the groups in parallel.
const PARALLEL_CODE_ROWS: usize = 1 << 16;

/// Node rows times features at which [`NodeGroups::numeric_cuts`] counts
/// the features in parallel.
const PARALLEL_COUNT_ENTRIES: usize = 1 << 16;

/// A training run's split groups, coded `0..n_groups`, and the count each
/// child of a split must reach.
#[derive(Debug)]
pub(crate) struct SplitGroups {
    /// Each training row's group.
    codes: Vec<u32>,
    n_groups: usize,
    /// `min_child_groups`, at least 2.
    min: usize,
}

impl SplitGroups {
    /// The constraint `params` puts on `dtrain`'s split groups; `None` when
    /// `min_child_groups` constrains nothing or `dtrain` has no groups (which
    /// training refuses before it gets here).
    pub(crate) fn of(params: &TrainingParams, dtrain: &DMatrix) -> Option<Self> {
        if !params.split_groups_on() {
            return None;
        }
        Some(Self::new(dtrain.split_groups()?, params.min_child_groups))
    }

    /// Code `ids` (one per row) densely, in ascending id order.
    fn new(ids: &[u64], min: usize) -> Self {
        let parallel = ids.len() >= PARALLEL_CODE_ROWS && rayon_available();
        let mut distinct = ids.to_vec();
        if parallel {
            distinct.par_sort_unstable();
        } else {
            distinct.sort_unstable();
        }
        distinct.dedup();
        // The builders index rows by `u32`, and there is at most one group
        // per row.
        let code = |id: &u64| distinct.partition_point(|d| d < id) as u32;
        let codes = if parallel {
            ids.par_iter().map(code).collect()
        } else {
            ids.iter().map(code).collect()
        };
        SplitGroups {
            codes,
            n_groups: distinct.len(),
            min,
        }
    }
}

/// The cuts of one feature's positions that a candidate may use, inclusive:
/// empty when `lo > hi`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CutRange {
    lo: usize,
    hi: usize,
}

impl CutRange {
    const EMPTY: CutRange = CutRange { lo: 1, hi: 0 };

    /// `lo..=hi`, empty when either end is missing.
    fn new(lo: Option<usize>, hi: Option<usize>) -> Self {
        match (lo, hi) {
            (Some(lo), Some(hi)) => CutRange { lo, hi },
            _ => Self::EMPTY,
        }
    }

    #[inline]
    fn contains(self, cut: usize) -> bool {
        self.lo <= cut && cut <= self.hi
    }
}

/// The legal cuts of one feature in one node (see the module docs): a cut
/// `c` puts the positions below `c` on the low side, and a candidate is
/// legal when both sides hold enough groups with missing values on the side
/// the candidate sends them to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LegalCuts {
    /// With missing values on the high side.
    missing_high: CutRange,
    /// With missing values on the low side.
    missing_low: CutRange,
}

impl LegalCuts {
    /// No cut at all: a node with too few groups for any split.
    pub(super) const NONE: LegalCuts = LegalCuts {
        missing_high: CutRange::EMPTY,
        missing_low: CutRange::EMPTY,
    };

    /// The legal cuts `lo..=hi` per missing direction, `(missing high,
    /// missing low)`.
    #[cfg(test)]
    pub(super) fn from_ranges(missing_high: (usize, usize), missing_low: (usize, usize)) -> Self {
        let range = |(lo, hi)| CutRange { lo, hi };
        LegalCuts {
            missing_high: range(missing_high),
            missing_low: range(missing_low),
        }
    }

    /// Whether cut `cut`, missing values on the low side when `missing_low`,
    /// is legal.
    #[inline]
    pub(super) fn allows(&self, cut: usize, missing_low: bool) -> bool {
        if missing_low {
            self.missing_low.contains(cut)
        } else {
            self.missing_high.contains(cut)
        }
    }

    /// Whether the numeric candidate at `pos` of a feature whose bins start
    /// at global bin `first` (bins up to `pos` left), missing values left
    /// when `default_left`, is legal.
    #[inline]
    pub(super) fn allows_numeric(&self, first: usize, pos: SplitPos, default_left: bool) -> bool {
        let cut = match pos {
            SplitPos::Bin(bin) => bin + 1 - first,
            SplitPos::BelowBins => 0,
            SplitPos::Value(_) => unreachable!("histogram candidates sit at bin boundaries"),
        };
        self.allows(cut, default_left)
    }

    /// The forward-pass offsets (bins `..= offset` left, missing values
    /// right) of a feature of `n` bins that are legal (possibly none).
    pub(super) fn forward(&self, n: usize) -> Range<usize> {
        let CutRange { lo, hi } = self.missing_high;
        lo.saturating_sub(1)..hi.min(n)
    }

    /// The backward-pass offsets (bins `offset..` right, missing values
    /// left) of a feature of `n` bins that are legal (possibly none).
    pub(super) fn backward(&self, n: usize) -> Range<usize> {
        let CutRange { lo, hi } = self.missing_low;
        lo..(hi + 1).min(n)
    }
}

/// Whether categories may be one-hot candidates in one node: per category
/// (by local bin), whether it is legal alone on one side, with missing
/// values on the other, and with missing values beside it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct OneHot {
    alone: Vec<bool>,
    with_missing: Vec<bool>,
}

impl OneHot {
    /// Whether category `bin` alone on one side, missing values with it
    /// when `with_missing` and against it otherwise, is legal.
    pub(super) fn allows(&self, bin: usize, with_missing: bool) -> bool {
        if with_missing {
            self.with_missing[bin]
        } else {
            self.alone[bin]
        }
    }
}

/// One node's counted rows as split groups: those rows laid out group by
/// group (the groups numbered by first appearance, each group's rows in row
/// order) and, for an index that keeps no columns, their entries by feature
/// in that layout.
pub(super) struct NodeGroups {
    /// The node's counted rows, group by group.
    grouped: Vec<u32>,
    /// Group `g`'s rows are `grouped[starts[g]..starts[g + 1]]`.
    starts: Vec<usize>,
    /// The first group of each chunk of a feature's pass, then the group
    /// count ([`group_chunks`]).
    chunks: Vec<usize>,
    /// `min_child_groups`.
    min: usize,
    /// The present entries by feature, when the index keeps no columns (a
    /// sparse index under half full) to read a feature from.
    by_feature: Option<FeatureEntries>,
}

impl NodeGroups {
    /// The groups of the node holding `rows` of `ghist`, over the rows
    /// whose gradient pair in `gpair` [`counts`].
    pub(super) fn new(
        groups: &SplitGroups,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        rows: &[u32],
    ) -> Self {
        let parallel = rows.len() >= 2 * CHUNK_ROWS && rayon_available();
        let counted = |&r: &u32| counts(gpair[r as usize]);
        let all = if parallel {
            rows.par_iter().all(counted)
        } else {
            rows.iter().all(counted)
        };
        let rows: Cow<'_, [u32]> = match (all, parallel) {
            (true, _) => Cow::Borrowed(rows),
            (false, true) => Cow::Owned(rows.par_iter().copied().filter(|r| counted(r)).collect()),
            (false, false) => Cow::Owned(rows.iter().copied().filter(|r| counted(r)).collect()),
        };
        let (grouped, starts) = if rows.len() >= 2 * CHUNK_ROWS && rayon_available() {
            group_in_chunks(groups, &rows)
        } else {
            let grouping = group(groups, &rows);
            (grouping.grouped, grouping.starts)
        };
        let chunks = group_chunks(&starts);
        let by_feature = ghist
            .columns()
            .is_none()
            .then(|| FeatureEntries::new(ghist, &grouped));
        NodeGroups {
            grouped,
            starts,
            chunks,
            min: groups.min,
            by_feature,
        }
    }

    /// The number of the node's groups.
    fn n_groups(&self) -> usize {
        self.starts.len() - 1
    }

    /// Whether any split of the node can be legal: every child's groups are
    /// the node's, so it needs at least `min_child_groups` of them.
    pub(super) fn splittable(&self) -> bool {
        self.n_groups() >= self.min
    }

    /// The legal cuts of numeric feature `f` (positions are local bins).
    pub(super) fn numeric(&self, ghist: &GHistIndex, f: usize) -> LegalCuts {
        self.ranked(ghist, f, ghist.cuts().num_bins(f), |bin| bin)
    }

    /// [`Self::numeric`] of every numeric feature of `features` with an
    /// interior boundary, by feature id (`None` for the others), counted in
    /// parallel for large nodes.
    pub(super) fn numeric_cuts(
        &self,
        ghist: &GHistIndex,
        features: &[u32],
    ) -> Vec<Option<LegalCuts>> {
        let cuts = ghist.cuts();
        let numeric: Vec<u32> = features
            .iter()
            .copied()
            .filter(|&f| !cuts.is_categorical(f as usize) && cuts.num_bins(f as usize) > 1)
            .collect();
        let count = |&f: &u32| (f, self.numeric(ghist, f as usize));
        let counted: Vec<(u32, LegalCuts)> = if self.grouped.len().saturating_mul(numeric.len())
            >= PARALLEL_COUNT_ENTRIES
            && rayon_available()
        {
            numeric.par_iter().map(count).collect()
        } else {
            numeric.iter().map(count).collect()
        };
        let mut by_feature = vec![None; ghist.n_cols()];
        for (f, legal) in counted {
            by_feature[f as usize] = Some(legal);
        }
        by_feature
    }

    /// The legal cuts of feature `f` with its local bin `b` at position
    /// `position(b)`, at most `n_positions` (the position above every cut).
    fn ranked(
        &self,
        ghist: &GHistIndex,
        f: usize,
        n_positions: usize,
        position: impl Fn(u32) -> u32 + Sync,
    ) -> LegalCuts {
        self.fold(ghist, f, || Tally::new(n_positions, &position))
            .legal_cuts(self.min)
    }

    /// Fold feature `f`'s entries over the node's groups, one fold from
    /// `start` per chunk of the pass: in parallel when there are several,
    /// merged in chunk order.
    fn fold<F: GroupFold + Send>(
        &self,
        ghist: &GHistIndex,
        f: usize,
        start: impl Fn() -> F + Sync,
    ) -> F {
        if self.chunks.len() > 2 && rayon_available() {
            let folds: Vec<F> = self
                .chunks
                .par_windows(2)
                .map(|chunk| {
                    let mut fold = start();
                    self.fold_groups(ghist, f, chunk[0]..chunk[1], &mut fold);
                    fold
                })
                .collect();
            let mut folds = folds.into_iter();
            let mut total = folds.next().unwrap_or_else(&start);
            for fold in folds {
                total.merge(fold);
            }
            total
        } else {
            let mut fold = start();
            self.fold_groups(ghist, f, 0..self.n_groups(), &mut fold);
            fold
        }
    }

    /// Feed `fold` feature `f`'s entries of groups `groups`, group by
    /// group.
    fn fold_groups<F: GroupFold>(
        &self,
        ghist: &GHistIndex,
        f: usize,
        groups: Range<usize>,
        fold: &mut F,
    ) {
        let starts = &self.starts;
        if let Some(entries) = &self.by_feature {
            let entries = entries.feature(f);
            let first = starts[groups.start];
            let mut next = entries.partition_point(|&(i, _)| (i as usize) < first);
            for g in groups {
                let end = starts[g + 1];
                while let Some(&(i, bin)) = entries.get(next)
                    && (i as usize) < end
                {
                    fold.entry(bin);
                    next += 1;
                }
                fold.end_group(end - starts[g]);
            }
            return;
        }
        let grouped = &self.grouped;
        match ghist.columns() {
            Some(Columns::U8(set)) => fold_column(grouped, starts, groups, set.column(f), fold),
            Some(Columns::U16(set)) => fold_column(grouped, starts, groups, set.column(f), fold),
            Some(Columns::U32(set)) => fold_column(grouped, starts, groups, set.column(f), fold),
            None => unreachable!("an index without columns has its entries by feature"),
        }
    }
}

/// Whether a row with gradient pair `g` counts toward its group: its
/// Hessian is finite and nonzero. A row of zero weight adds nothing to a
/// child's fit (nor to `min_child_weight`), and the rows that count are
/// those the cuts were sketched from (`hist`'s rows of nonzero weight,
/// `approx`'s of nonzero Hessian), whose bins route them as their values
/// do; binning clamps an unsketched value beyond the last cut into the
/// last bin, and an unsketched category into the first.
#[inline]
fn counts(g: GradPair) -> bool {
    g.hess != 0.0 && g.hess.is_finite()
}

/// Feed `fold` the entries of one feature's `column` among the rows of
/// groups `groups` (laid out by `grouped` and `starts`), group by group.
#[inline(always)]
fn fold_column<B: BinIndex, F: GroupFold>(
    grouped: &[u32],
    starts: &[usize],
    groups: Range<usize>,
    column: Column<'_, B>,
    fold: &mut F,
) {
    let bins = column.bins;
    for g in groups {
        let rows = &grouped[starts[g]..starts[g + 1]];
        match column.missing {
            None => {
                for &r in rows {
                    fold.entry(bins[r as usize].index() as u32);
                }
            }
            Some(bits) => {
                for &r in rows {
                    if !missing_bit(bits, r as usize) {
                        fold.entry(bins[r as usize].index() as u32);
                    }
                }
            }
        }
        fold.end_group(rows.len());
    }
}

/// What a pass over one feature of a node gathers, group by group
/// ([`NodeGroups::fold`]).
trait GroupFold {
    /// A row of the current group holds the feature, in local bin `bin`.
    fn entry(&mut self, bin: u32);
    /// The current group, of `size` rows, ends.
    fn end_group(&mut self, size: usize);
    /// Add `other`, the fold of other groups.
    fn merge(&mut self, other: Self);
}

/// A [`GroupFold`] of a feature's positions (see the module docs): the
/// groups counted by their lowest and by their highest position, all of
/// them and the complete ones (none of whose rows misses the feature), and
/// the incomplete ones.
struct Tally<P> {
    position: P,
    n_positions: usize,
    /// The current group's lowest and highest positions and entries.
    low: u32,
    high: u32,
    count: usize,
    /// Groups by lowest position, the complete ones by lowest position,
    /// groups by highest position, and the complete ones by highest
    /// position, each over the positions `0..=n_positions`.
    counts: Vec<u32>,
    incomplete: usize,
}

impl<P: Fn(u32) -> u32> Tally<P> {
    fn new(n_positions: usize, position: P) -> Self {
        Tally {
            position,
            n_positions,
            low: u32::MAX,
            high: 0,
            count: 0,
            counts: vec![0; 4 * (n_positions + 1)],
            incomplete: 0,
        }
    }

    /// The legal cuts of the counted groups, each child needing `min` of
    /// them (see the module docs).
    fn legal_cuts(&self, min: usize) -> LegalCuts {
        let n_positions = self.n_positions;
        let slots = n_positions + 1;
        let (lows, rest) = self.counts.split_at(slots);
        let (complete_lows, rest) = rest.split_at(slots);
        let (highs, complete_highs) = rest.split_at(slots);
        let incomplete = self.incomplete;
        // The low side holds a group from cut `kth_smallest + 1` on, the
        // high side up to cut `kth_largest`; an incomplete group is on the
        // side that takes missing values at every cut.
        let from = |counts: &[u32], k: usize| kth_smallest(counts, k).map(|p| p + 1);
        let below_missing = |k: usize| k.saturating_sub(incomplete);
        LegalCuts {
            missing_high: CutRange::new(
                from(lows, min),
                if incomplete >= min {
                    Some(n_positions)
                } else {
                    kth_largest(complete_highs, below_missing(min))
                },
            ),
            missing_low: CutRange::new(
                if incomplete >= min {
                    Some(0)
                } else {
                    from(complete_lows, below_missing(min))
                },
                kth_largest(highs, min),
            ),
        }
    }
}

impl<P: Fn(u32) -> u32> GroupFold for Tally<P> {
    #[inline(always)]
    fn entry(&mut self, bin: u32) {
        let position = (self.position)(bin);
        self.low = self.low.min(position);
        self.high = self.high.max(position);
        self.count += 1;
    }

    #[inline(always)]
    fn end_group(&mut self, size: usize) {
        if self.count > 0 {
            let slots = self.n_positions + 1;
            let (low, high) = (self.low as usize, self.high as usize);
            self.counts[low] += 1;
            self.counts[2 * slots + high] += 1;
            if self.count == size {
                self.counts[slots + low] += 1;
                self.counts[3 * slots + high] += 1;
            }
        }
        if self.count < size {
            self.incomplete += 1;
        }
        (self.low, self.high, self.count) = (u32::MAX, 0, 0);
    }

    fn merge(&mut self, other: Self) {
        for (total, count) in self.counts.iter_mut().zip(other.counts) {
            *total += count;
        }
        self.incomplete += other.incomplete;
    }
}

/// The bit of missing values in a [`CategorySets`] set.
const MISSING_BIT: u32 = 1 << ONE_HOT_MAX;

/// A [`GroupFold`] of a one-hot feature (at most [`ONE_HOT_MAX`]
/// categories): the groups counted by the set of categories their rows
/// hold, with [`MISSING_BIT`] for a group some row of which misses the
/// feature.
#[derive(Default)]
struct CategorySets {
    /// The current group's categories and entries.
    mask: u32,
    count: usize,
    /// Groups by set.
    groups: [usize; 1 << (ONE_HOT_MAX + 1)],
}

impl GroupFold for CategorySets {
    #[inline(always)]
    fn entry(&mut self, bin: u32) {
        self.mask |= 1 << bin;
        self.count += 1;
    }

    #[inline(always)]
    fn end_group(&mut self, size: usize) {
        let missing = if self.count < size { MISSING_BIT } else { 0 };
        self.groups[(self.mask | missing) as usize] += 1;
        (self.mask, self.count) = (0, 0);
    }

    fn merge(&mut self, other: Self) {
        for (total, groups) in self.groups.iter_mut().zip(other.groups) {
            *total += groups;
        }
    }
}

/// One categorical feature of a node, as the categorical searches ask about
/// its split groups.
pub(super) struct CategoryGroups<'n> {
    pub(super) node: &'n NodeGroups,
    pub(super) ghist: &'n GHistIndex,
    pub(super) feature: usize,
}

impl CategoryGroups<'_> {
    /// The legal cuts of the feature's categories in `order` (local bins,
    /// each at most once): a cut `c` puts `order[..c]` on the low side, and
    /// categories `order` leaves out on the high side whatever the cut.
    pub(super) fn ordered(&self, order: &[usize]) -> LegalCuts {
        let n_bins = self.ghist.cuts().num_bins(self.feature);
        let n_positions = order.len();
        let mut position = vec![n_positions as u32; n_bins];
        for (rank, &bin) in order.iter().enumerate() {
            position[bin] = rank as u32;
        }
        self.node
            .ranked(self.ghist, self.feature, n_positions, |bin| {
                position[bin as usize]
            })
    }

    /// [`Self::ordered`] of the categories coded `codes` (category values,
    /// each one of the feature's), in that order.
    pub(super) fn ordered_categories(&self, codes: &[u32]) -> LegalCuts {
        let cuts = self.ghist.cuts();
        let (first, _) = cuts.feature_bins(self.feature);
        let order: Vec<usize> = codes
            .iter()
            .map(|&code| cuts.bin_of(self.feature, code as f32) as usize - first)
            .collect();
        self.ordered(&order)
    }

    /// Each category's legality as a one-hot candidate. The feature has at
    /// most [`ONE_HOT_MAX`] categories (the one-hot searches enumerate
    /// fewer than `max_cat_to_onehot`).
    pub(super) fn one_hot(&self) -> OneHot {
        let n_bins = self.ghist.cuts().num_bins(self.feature);
        assert!(n_bins <= ONE_HOT_MAX, "one-hot categories fit a mask");
        let node = self.node;
        let sets = node
            .fold(self.ghist, self.feature, CategorySets::default)
            .groups;
        // The groups holding a member of `side` (missing values one).
        let holding = |side: u32| -> usize {
            sets.iter()
                .enumerate()
                .filter(|&(set, _)| set as u32 & side != 0)
                .map(|(_, &groups)| groups)
                .sum()
        };
        let every = (1u32 << n_bins) - 1;
        let legal = |side: u32, rest: u32| holding(side) >= node.min && holding(rest) >= node.min;
        let mut one_hot = OneHot {
            alone: Vec::with_capacity(n_bins),
            with_missing: Vec::with_capacity(n_bins),
        };
        for bin in 0..n_bins {
            let category = 1u32 << bin;
            let others = every & !category;
            one_hot.alone.push(legal(category, others | MISSING_BIT));
            one_hot
                .with_missing
                .push(legal(category | MISSING_BIT, others));
        }
        one_hot
    }
}

/// The most categories [`CategoryGroups::one_hot`] takes: the one-hot
/// searches' (below XGBoost's `max_cat_to_onehot`).
pub(super) const ONE_HOT_MAX: usize = MAX_CAT_TO_ONEHOT - 1;

/// A node's present entries regrouped by feature: feature `f`'s are
/// `entries[start[f]..start[f + 1]]`, each its row's index in the node's
/// rows laid out group by group and the entry's local bin, in that order.
struct FeatureEntries {
    start: Vec<usize>,
    entries: Vec<(u32, u32)>,
}

impl FeatureEntries {
    fn new(ghist: &GHistIndex, grouped: &[u32]) -> Self {
        let cuts = ghist.cuts();
        let n_features = cuts.n_features();
        let mut start = vec![0usize; n_features + 1];
        for_each_row_bin(ghist, grouped, |_, bin| {
            start[cuts.feature_of_bin(bin) + 1] += 1;
        });
        for f in 0..n_features {
            start[f + 1] += start[f];
        }
        let mut cursor = start.clone();
        let mut entries = vec![(0u32, 0u32); start[n_features]];
        for_each_row_bin(ghist, grouped, |i, bin| {
            let f = cuts.feature_of_bin(bin);
            entries[cursor[f]] = (i as u32, (bin - cuts.feature_bins(f).0) as u32);
            cursor[f] += 1;
        });
        FeatureEntries { start, entries }
    }

    fn feature(&self, f: usize) -> &[(u32, u32)] {
        &self.entries[self.start[f]..self.start[f + 1]]
    }
}

/// Visit every stored entry of `rows` as its row's index in `rows` and its
/// global bin, rows in order.
fn for_each_row_bin(ghist: &GHistIndex, rows: &[u32], mut visit: impl FnMut(usize, usize)) {
    let row_ptr = ghist.row_ptr();
    match ghist.bins() {
        Bins::U16(bins) => {
            for (i, &r) in rows.iter().enumerate() {
                let r = r as usize;
                for &bin in &bins[row_ptr[r]..row_ptr[r + 1]] {
                    visit(i, usize::from(bin));
                }
            }
        }
        Bins::U32(bins) => {
            for (i, &r) in rows.iter().enumerate() {
                let r = r as usize;
                for &bin in &bins[row_ptr[r]..row_ptr[r + 1]] {
                    visit(i, bin as usize);
                }
            }
        }
    }
}

/// Node rows per chunk of [`group_in_chunks`], and the fewest rows of a
/// chunk of [`group_chunks`] but the last; small in unit tests, so their
/// nodes take the chunked paths.
const CHUNK_ROWS: usize = if cfg!(test) { 64 } else { 1 << 16 };

/// Group codes numbered by first appearance in an open-addressing table
/// sized for the most distinct codes one call can meet (so at most half
/// full): [`CodeNumbers`] for a run of many groups.
struct CodeTable {
    /// `code << 32 | number`, or [`CodeTable::EMPTY`] (which no entry is:
    /// a table numbering `u32::MAX` codes would need `2^33` slots).
    slots: Vec<u64>,
    /// 64 minus the slot count's bits: the hash keeps the product's top
    /// bits.
    shift: u32,
}

impl CodeTable {
    const EMPTY: u64 = u64::MAX;

    /// A table for at most `codes` distinct codes.
    fn new(codes: usize) -> Self {
        let len = (2 * codes).next_power_of_two().max(16);
        CodeTable {
            slots: vec![Self::EMPTY; len],
            shift: 64 - len.trailing_zeros(),
        }
    }

    /// The number of `code`; a code without one gets `next`.
    #[inline(always)]
    fn number(&mut self, code: u32, next: u32) -> u32 {
        let mask = self.slots.len() - 1;
        let mut slot = (u64::from(code).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize;
        loop {
            match self.slots[slot] {
                Self::EMPTY => {
                    self.slots[slot] = u64::from(code) << 32 | u64::from(next);
                    return next;
                }
                held if (held >> 32) as u32 == code => return held as u32,
                _ => slot = (slot + 1) & mask,
            }
        }
    }
}

/// Group codes numbered by first appearance for one call, in memory that
/// follows the rows it numbers: a slot per group of the run when there are
/// at most four of them per code the call can meet (codes, often in row
/// order, are then looked up in order), else a [`CodeTable`].
enum CodeNumbers {
    /// Each code's number, or `u32::MAX` while it has none (which no
    /// number is: that takes `2^32` groups, a row each).
    Direct(Vec<u32>),
    Hashed(CodeTable),
}

impl CodeNumbers {
    /// Numbers for at most `codes` distinct codes of a run of `n_groups`.
    fn new(codes: usize, n_groups: usize) -> Self {
        if n_groups <= 4 * codes {
            CodeNumbers::Direct(vec![u32::MAX; n_groups])
        } else {
            CodeNumbers::Hashed(CodeTable::new(codes))
        }
    }

    /// The number of `code`; a code without one gets `next`.
    #[inline(always)]
    fn number(&mut self, code: u32, next: u32) -> u32 {
        match self {
            CodeNumbers::Direct(numbers) => {
                let number = &mut numbers[code as usize];
                if *number == u32::MAX {
                    *number = next;
                }
                *number
            }
            CodeNumbers::Hashed(table) => table.number(code, next),
        }
    }
}

/// Rows laid out group by group ([`group`]).
struct Grouping {
    /// The rows, group by group, each group's rows in their order.
    grouped: Vec<u32>,
    /// Group `g`'s rows are `grouped[starts[g]..starts[g + 1]]`.
    starts: Vec<usize>,
    /// Each group's code.
    codes: Vec<u32>,
}

/// Lay `rows` out group by group, the groups numbered by first appearance.
fn group(groups: &SplitGroups, rows: &[u32]) -> Grouping {
    let mut numbers = CodeNumbers::new(rows.len().min(groups.n_groups), groups.n_groups);
    let mut codes = Vec::new();
    // Each group's rows, then where its rows end, then where they start.
    let mut starts: Vec<usize> = Vec::new();
    let local: Vec<u32> = rows
        .iter()
        .map(|&r| {
            let code = groups.codes[r as usize];
            let g = numbers.number(code, codes.len() as u32);
            if g as usize == codes.len() {
                codes.push(code);
                starts.push(0);
            }
            starts[g as usize] += 1;
            g
        })
        .collect();
    drop(numbers);
    let mut end = 0;
    for rows in &mut starts {
        end += *rows;
        *rows = end;
    }
    // Each row goes just before its group's end, the last rows first, so a
    // group keeps its rows' order and its end moves to its start.
    let mut grouped = vec![0; rows.len()];
    for (&r, &g) in rows.iter().zip(&local).rev() {
        let at = &mut starts[g as usize];
        *at -= 1;
        grouped[*at] = r;
    }
    starts.push(rows.len());
    Grouping {
        grouped,
        starts,
        codes,
    }
}

/// `0` and the running sums of `sizes`.
fn offsets(sizes: &[usize]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(sizes.len() + 1);
    let mut sum = 0;
    offsets.push(sum);
    for &size in sizes {
        sum += size;
        offsets.push(sum);
    }
    offsets
}

/// [`group`] in parallel chunks of rows: each chunk lays out its own groups
/// (its pieces of the node's), the node's groups are numbered by first
/// appearance across the chunks in order (as [`group`] numbers them), and
/// each gathers its pieces in chunk order. Returns [`group`]'s layout.
fn group_in_chunks(groups: &SplitGroups, rows: &[u32]) -> (Vec<u32>, Vec<usize>) {
    let chunks: Vec<Grouping> = rows
        .par_chunks(CHUNK_ROWS)
        .map(|chunk| group(groups, chunk))
        .collect();
    let pieces: usize = chunks.iter().map(|chunk| chunk.codes.len()).sum();
    // Each piece's node group, chunk after chunk.
    let mut numbers = CodeNumbers::new(pieces, groups.n_groups);
    let mut sizes: Vec<usize> = Vec::new();
    let mut node_group = Vec::with_capacity(pieces);
    for chunk in &chunks {
        for (piece, &code) in chunk.codes.iter().enumerate() {
            let g = numbers.number(code, sizes.len() as u32);
            if g as usize == sizes.len() {
                sizes.push(0);
            }
            sizes[g as usize] += chunk.starts[piece + 1] - chunk.starts[piece];
            node_group.push(g);
        }
    }
    drop(numbers);
    let starts = offsets(&sizes);
    if node_group.is_sorted() {
        // Every group's pieces are adjacent and in group order (its rows
        // run in consecutive chunks): the chunks' layouts joined end to end
        // are the node's.
        let mut grouped = Vec::with_capacity(rows.len());
        for chunk in &chunks {
            grouped.extend_from_slice(&chunk.grouped);
        }
        return (grouped, starts);
    }
    // Each group's pieces as `(chunk, piece)`, in chunk order: group `g`'s
    // are `located[first_piece[g]..first_piece[g + 1]]`.
    let mut first_piece = vec![0; sizes.len() + 1];
    for &g in &node_group {
        first_piece[g as usize + 1] += 1;
    }
    for g in 0..sizes.len() {
        first_piece[g + 1] += first_piece[g];
    }
    let mut cursor = first_piece.clone();
    let mut located = vec![(0u32, 0u32); pieces];
    let every_piece = chunks
        .iter()
        .enumerate()
        .flat_map(|(c, chunk)| (0..chunk.codes.len()).map(move |piece| (c as u32, piece as u32)));
    for (piece, &g) in every_piece.zip(&node_group) {
        let at = &mut cursor[g as usize];
        located[*at] = piece;
        *at += 1;
    }
    // Gather the pieces in parallel over chunks of whole groups, each
    // writing its own range of the layout.
    let mut grouped = vec![0; rows.len()];
    let mut outputs = Vec::new();
    let mut rest = grouped.as_mut_slice();
    for bounds in group_chunks(&starts).windows(2) {
        let (out, tail) =
            std::mem::take(&mut rest).split_at_mut(starts[bounds[1]] - starts[bounds[0]]);
        outputs.push((bounds[0]..bounds[1], out));
        rest = tail;
    }
    outputs.into_par_iter().for_each(|(range, out)| {
        let mut at = 0;
        for &(c, piece) in &located[first_piece[range.start]..first_piece[range.end]] {
            let chunk = &chunks[c as usize];
            let piece = piece as usize;
            let rows = &chunk.grouped[chunk.starts[piece]..chunk.starts[piece + 1]];
            out[at..at + rows.len()].copy_from_slice(rows);
            at += rows.len();
        }
    });
    (grouped, starts)
}

/// The first group of each chunk of whole groups laid out by `starts`, then
/// the group count: each chunk at least [`CHUNK_ROWS`] rows but the last.
fn group_chunks(starts: &[usize]) -> Vec<usize> {
    let n_groups = starts.len() - 1;
    let mut chunks = vec![0];
    let mut g = 0;
    while g < n_groups {
        let limit = starts[g] + CHUNK_ROWS;
        g = starts
            .partition_point(|&start| start < limit)
            .clamp(g + 1, n_groups);
        chunks.push(g);
    }
    chunks
}

/// The smallest position whose counts up to it sum to at least `k` (`k >=
/// 1`), if any.
fn kth_smallest(histogram: &[u32], k: usize) -> Option<usize> {
    let mut seen = 0usize;
    histogram.iter().position(|&n| {
        seen += n as usize;
        seen >= k
    })
}

/// The largest position whose counts from it on sum to at least `k` (`k >=
/// 1`), if any.
fn kth_largest(histogram: &[u32], k: usize) -> Option<usize> {
    let mut seen = 0usize;
    histogram.iter().rposition(|&n| {
        seen += n as usize;
        seen >= k
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::FeatureType;
    use crate::data::quantile::HistCuts;
    use crate::rng::Rng;
    use std::collections::BTreeSet;

    /// A matrix of `n` rows and `n_cols` features, every one a few levels
    /// (so bins hold several rows), its first `categorical` features
    /// categories `0..=cats`; with `missing`, a fifth of the entries
    /// missing, and with `sparse` stored as CSR two thirds empty (an index
    /// without columns).
    struct Layout {
        n_cols: usize,
        categorical: usize,
        cats: u32,
        missing: bool,
        sparse: bool,
    }

    impl Layout {
        fn index(&self, n: usize, rng: &mut Rng) -> GHistIndex {
            let Layout {
                n_cols,
                categorical,
                cats,
                missing,
                sparse,
            } = *self;
            let value = |c: usize, rng: &mut Rng| {
                if c < categorical {
                    rng.range(0..cats as usize + 1) as f32
                } else {
                    rng.range(0..12) as f32 * 0.5
                }
            };
            let data = if sparse {
                let (mut indptr, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
                for _ in 0..n {
                    for c in 0..n_cols {
                        if rng.below(3) == 0 {
                            indices.push(c as u32);
                            values.push(value(c, rng));
                        }
                    }
                    indptr.push(values.len());
                }
                DMatrix::from_csr(indptr, indices, values, n_cols).unwrap()
            } else {
                let values: Vec<f32> = (0..n * n_cols)
                    .map(|i| {
                        if missing && rng.below(5) == 0 {
                            f32::NAN
                        } else {
                            value(i % n_cols, rng)
                        }
                    })
                    .collect();
                DMatrix::from_dense(&values, n, n_cols).unwrap()
            };
            let types: Vec<FeatureType> = (0..n_cols)
                .map(|c| {
                    if c < categorical {
                        FeatureType::Categorical
                    } else {
                        FeatureType::Numerical
                    }
                })
                .collect();
            let data = data.with_feature_types(&types).unwrap();
            let index = GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, 64));
            assert_eq!(index.columns().is_none(), sparse, "the layout under test");
            index
        }
    }

    /// The distinct groups of `rows` on each side of a cut: rows whose
    /// local bin of `f` sits (`position`) below `cut` on the low side, and
    /// missing rows on the low side when `missing_low`.
    fn sides(
        ghist: &GHistIndex,
        f: usize,
        rows: &[u32],
        ids: &[u64],
        cut: usize,
        missing_low: bool,
        position: &dyn Fn(usize) -> usize,
    ) -> (usize, usize) {
        let (first, _) = ghist.cuts().feature_bins(f);
        let (mut low, mut high) = (BTreeSet::new(), BTreeSet::new());
        for &r in rows {
            let to_low = match ghist.feature_bin(r as usize, f) {
                Some(bin) => position(bin as usize - first) < cut,
                None => missing_low,
            };
            if to_low { &mut low } else { &mut high }.insert(ids[r as usize]);
        }
        (low.len(), high.len())
    }

    /// A random node of [`node`].
    struct Node {
        /// An ascending subset of the rows.
        rows: Vec<u32>,
        /// Every row's group id.
        ids: Vec<u64>,
        min: usize,
        /// Every row's gradient pair.
        gpair: Vec<GradPair>,
        /// The node's rows whose Hessian is finite and nonzero: the rows
        /// that count.
        counted: Vec<u32>,
    }

    /// Random nodes: an ascending subset of `n` rows, group ids drawn from
    /// `groups` values spread over the `u64` range (in runs of consecutive
    /// rows when `runs`), `min` in `2..=5`, and Hessians that are sometimes
    /// zero (of either sign), NaN, or negative.
    fn node(rng: &mut Rng, n: usize, groups: u64, runs: bool) -> Node {
        let ids: Vec<u64> = (0..n)
            .map(|r| {
                let g = if runs {
                    r as u64 * groups / n as u64
                } else {
                    rng.range(0..groups as usize) as u64
                };
                g.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            })
            .collect();
        let keep = 1 + rng.below(4);
        let rows: Vec<u32> = (0..n as u32).filter(|_| rng.below(4) < keep).collect();
        let gpair: Vec<GradPair> = (0..n)
            .map(|_| {
                let hess = match rng.below(12) {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f32::NAN,
                    3 => -0.5,
                    _ => 1.0,
                };
                GradPair::new(1.0, hess)
            })
            .collect();
        let counted = rows
            .iter()
            .copied()
            .filter(|&r| {
                let hess = gpair[r as usize].hess;
                hess != 0.0 && hess.is_finite()
            })
            .collect();
        Node {
            rows,
            ids,
            min: 2 + rng.range(0..4),
            gpair,
            counted,
        }
    }

    const LAYOUTS: [Layout; 3] = [
        Layout {
            n_cols: 4,
            categorical: 0,
            cats: 0,
            missing: true,
            sparse: false,
        },
        Layout {
            n_cols: 3,
            categorical: 0,
            cats: 0,
            missing: false,
            sparse: false,
        },
        Layout {
            n_cols: 6,
            categorical: 0,
            cats: 0,
            missing: false,
            sparse: true,
        },
    ];

    /// Every numeric cut, with missing values on either side, is legal
    /// exactly when both sides hold `min` distinct groups of the node's
    /// counted rows, in an index with a missing bitmap, one without, and
    /// one without columns; the scans' offset ranges agree.
    #[test]
    fn numeric_cuts_count_the_groups_of_both_sides() {
        let mut rng = Rng::new(3);
        for layout in &LAYOUTS {
            for trial in 0..60 {
                let n = 20 + rng.range(0..300);
                let ghist = layout.index(n, &mut rng);
                let trial_node = node(&mut rng, n, [2, 3, 6, 40][trial % 4], trial % 8 >= 4);
                let Node {
                    rows,
                    ids,
                    min,
                    gpair,
                    counted,
                } = &trial_node;
                let (min, groups) = (*min, SplitGroups::new(ids, *min));
                let node = NodeGroups::new(&groups, &ghist, gpair, rows);
                let distinct: BTreeSet<u64> = counted.iter().map(|&r| ids[r as usize]).collect();
                assert_eq!(node.splittable(), distinct.len() >= min);
                for f in 0..layout.n_cols {
                    let n_bins = ghist.cuts().num_bins(f);
                    let legal = node.numeric(&ghist, f);
                    for cut in 0..=n_bins {
                        for missing_low in [false, true] {
                            let (low, high) =
                                sides(&ghist, f, counted, ids, cut, missing_low, &|b| b);
                            assert_eq!(
                                legal.allows(cut, missing_low),
                                low >= min && high >= min,
                                "trial {trial} feature {f} cut {cut} missing low {missing_low}"
                            );
                        }
                    }
                    for offset in 0..n_bins {
                        assert_eq!(
                            legal.forward(n_bins).contains(&offset),
                            legal.allows(offset + 1, false)
                        );
                        assert_eq!(
                            legal.backward(n_bins).contains(&offset),
                            legal.allows(offset, true)
                        );
                    }
                }
            }
        }
    }

    /// Categories in a search's order (some left out, which stay high) and
    /// one-hot categories (alone, missing values with or against them) are
    /// legal exactly when both sides hold `min` distinct groups of the
    /// node's counted rows.
    #[test]
    fn categorical_candidates_count_the_groups_of_both_sides() {
        let mut rng = Rng::new(5);
        for (cats, sparse) in [(7, false), (7, true), (2, false), (1, true)] {
            let layout = Layout {
                n_cols: 2,
                categorical: 2,
                cats,
                missing: true,
                sparse,
            };
            for trial in 0..60 {
                let n = 20 + rng.range(0..300);
                let ghist = layout.index(n, &mut rng);
                let trial_node = node(&mut rng, n, [2, 3, 6, 40][trial % 4], trial % 8 >= 4);
                let Node {
                    rows,
                    ids,
                    min,
                    gpair,
                    counted,
                } = &trial_node;
                let (min, groups) = (*min, SplitGroups::new(ids, *min));
                let node = NodeGroups::new(&groups, &ghist, gpair, rows);
                for feature in 0..2 {
                    let category = CategoryGroups {
                        node: &node,
                        ghist: &ghist,
                        feature,
                    };
                    let n_bins = ghist.cuts().num_bins(feature);
                    let mut order: Vec<usize> = (0..n_bins).filter(|_| rng.below(4) != 0).collect();
                    for i in (1..order.len()).rev() {
                        order.swap(i, rng.range(0..i + 1));
                    }
                    let legal = category.ordered(&order);
                    let position =
                        |b: usize| order.iter().position(|&o| o == b).unwrap_or(order.len());
                    for cut in 0..=order.len() {
                        for missing_low in [false, true] {
                            let (low, high) =
                                sides(&ghist, feature, counted, ids, cut, missing_low, &position);
                            assert_eq!(
                                legal.allows(cut, missing_low),
                                low >= min && high >= min,
                                "trial {trial} order {order:?} cut {cut} missing low {missing_low}"
                            );
                        }
                    }
                    if n_bins > ONE_HOT_MAX {
                        continue;
                    }
                    let one_hot = category.one_hot();
                    for bin in 0..n_bins {
                        // The category alone below a cut of an order that
                        // starts with it.
                        let first = |b: usize| usize::from(b != bin);
                        for with_missing in [false, true] {
                            let (low, high) =
                                sides(&ghist, feature, counted, ids, 1, with_missing, &first);
                            assert_eq!(
                                one_hot.allows(bin, with_missing),
                                low >= min && high >= min,
                                "trial {trial} bin {bin} with missing {with_missing}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A large node's chunked layout is the sequential one: the same groups
    /// in the same order, each with the same rows, whether the groups run
    /// in consecutive rows (the chunks' layouts join end to end) or spread
    /// over the node (each group gathers its pieces).
    #[test]
    fn chunked_layout_is_the_sequential_one() {
        let mut rng = Rng::new(7);
        for trial in 0..40 {
            let n = 200 + rng.range(0..800);
            let ids = [1, 2, 9, 70, 1000][trial % 5];
            let Node { rows, ids, min, .. } = node(&mut rng, n, ids, trial % 2 == 0);
            let groups = SplitGroups::new(&ids, min);
            let sequential = group(&groups, &rows);
            assert_eq!(
                group_in_chunks(&groups, &rows),
                (sequential.grouped, sequential.starts),
                "trial {trial}"
            );
        }
    }
}
