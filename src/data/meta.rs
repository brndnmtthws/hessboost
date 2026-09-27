//! Feature and instance metadata carried alongside the feature matrix.

use serde::{Deserialize, Serialize};

use crate::error::{HessboostError, Result};
use std::num::NonZeroUsize;

/// How a feature column should be treated during split finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum FeatureType {
    /// Ordered numerical feature. Splits are `x < threshold`.
    #[default]
    Numerical,
    /// Unordered categorical feature. Splits partition category sets.
    Categorical,
}

/// Ranking group layout, stored as a prefix-sum (`group_ptr`) over rows.
///
/// `group_ptr` has `num_groups + 1` entries. Group `g` spans rows
/// `group_ptr[g]..group_ptr[g + 1]`. This matches XGBoost's CSR-style group
/// encoding for learning-to-rank objectives. Build with
/// [`GroupInfo::from_sizes`] (or set `group_ptr` on [`GroupInfo::default`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct GroupInfo {
    /// Prefix-sum group boundaries over the row index.
    pub group_ptr: Vec<usize>,
}

impl GroupInfo {
    /// Build a [`GroupInfo`] from per-group sizes (e.g. `[3, 2, 4]`).
    pub fn from_sizes(sizes: &[usize]) -> Self {
        let mut group_ptr = Vec::with_capacity(sizes.len() + 1);
        group_ptr.push(0);
        let mut acc = 0usize;
        for &s in sizes {
            // Saturating: sizes summing past `usize::MAX` (no real row
            // count) give a layout that `partitions` no dataset.
            acc = acc.saturating_add(s);
            group_ptr.push(acc);
        }
        GroupInfo { group_ptr }
    }

    /// Number of groups.
    pub fn num_groups(&self) -> usize {
        self.group_ptr.len().saturating_sub(1)
    }

    /// The total number of rows spanned by all groups.
    pub fn num_rows(&self) -> usize {
        self.group_ptr.last().copied().unwrap_or(0)
    }

    /// Iterate `(start, end)` row ranges, one per group.
    pub fn iter_ranges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.group_ptr.windows(2).map(|w| (w[0], w[1]))
    }

    /// Whether the groups split rows `0..n_rows` into consecutive ranges in
    /// order: `group_ptr` starts at `0`, never decreases, and ends at
    /// `n_rows` (empty groups allowed). Only then do the
    /// [`iter_ranges`](Self::iter_ranges) slice an `n_rows` buffer.
    pub(crate) fn partitions(&self, n_rows: usize) -> bool {
        self.group_ptr.first() == Some(&0)
            && self.group_ptr.last() == Some(&n_rows)
            && self.group_ptr.is_sorted()
    }
}

/// A dataset's labels: row-major `[row][target]` values with at least one
/// target per row.
#[derive(Debug, Clone, Copy)]
pub struct Labels<'a> {
    values: &'a [f32],
    n_targets: NonZeroUsize,
}

impl<'a> Labels<'a> {
    /// `values` laid out `[row][target]` with `n_targets` targets per row.
    /// Their count is checked against the rows where [`MetaInfo`] is read.
    pub fn new(values: &'a [f32], n_targets: NonZeroUsize) -> Self {
        Labels { values, n_targets }
    }

    /// One label per row.
    pub fn single(values: &'a [f32]) -> Self {
        Labels::new(values, NonZeroUsize::MIN)
    }

    /// The label values, `[row][target]`.
    #[inline]
    pub fn values(&self) -> &'a [f32] {
        self.values
    }

    /// Targets per row.
    #[inline]
    pub fn n_targets(&self) -> NonZeroUsize {
        self.n_targets
    }
}

/// Interval-censored label bounds: one lower and one upper bound per row,
/// always given together.
#[derive(Debug, Clone, Copy)]
pub struct LabelBounds<'a> {
    lower: &'a [f32],
    upper: &'a [f32],
}

impl<'a> LabelBounds<'a> {
    /// Per-row `lower` and `upper` bounds. Their lengths are checked against
    /// the rows where [`MetaInfo`] is read.
    pub fn new(lower: &'a [f32], upper: &'a [f32]) -> Self {
        LabelBounds { lower, upper }
    }

    /// Lower bounds, one per row.
    #[inline]
    pub fn lower(&self) -> &'a [f32] {
        self.lower
    }

    /// Upper bounds, one per row.
    #[inline]
    pub fn upper(&self) -> &'a [f32] {
        self.upper
    }
}

/// Borrowed per-row training metadata handed to objectives and metrics.
///
/// Built by [`DMatrix::info`](crate::data::DMatrix::info), by
/// [`MetaInfo::new`] for single-target callers that only have labels,
/// weights, and groups, or by [`MetaInfo::unlabeled`]; set the other fields
/// (a label matrix's [`Labels`], label bounds) on the result.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct MetaInfo<'a> {
    /// Number of rows (instances).
    pub n_rows: usize,
    /// Labels (`n_rows * n_targets` values), if the dataset has any.
    pub labels: Option<Labels<'a>>,
    /// Per-row weights (`n_rows`), if any.
    pub weights: Option<&'a [f32]>,
    /// Ranking groups, if any.
    pub group: Option<&'a GroupInfo>,
    /// Interval-censored label bounds (`n_rows` each), if any.
    pub bounds: Option<LabelBounds<'a>>,
}

impl<'a> MetaInfo<'a> {
    /// Single-target metadata: `n_rows = labels.len()`, one target, no
    /// label bounds.
    pub fn new(
        labels: &'a [f32],
        weights: Option<&'a [f32]>,
        group: Option<&'a GroupInfo>,
    ) -> Self {
        MetaInfo {
            n_rows: labels.len(),
            labels: Some(Labels::single(labels)),
            weights,
            group,
            bounds: None,
        }
    }

    /// Metadata of `n_rows` rows without labels, weights, groups, or bounds.
    pub fn unlabeled(n_rows: usize) -> Self {
        MetaInfo {
            n_rows,
            labels: None,
            weights: None,
            group: None,
            bounds: None,
        }
    }

    /// Targets per row: the labels' count, or one without labels.
    #[inline]
    pub fn n_targets(&self) -> usize {
        self.labels.map_or(1, |labels| labels.n_targets().get())
    }

    /// The label values, empty without labels: the slice the
    /// [`Loss`](crate::objective::Loss) and
    /// [`Metric`](crate::metric::Metric) slice hooks receive.
    #[inline]
    pub fn label_values(&self) -> &'a [f32] {
        self.labels.map_or(&[], |labels| labels.values())
    }

    /// Check the lengths the objective and metric hooks index by: labels
    /// (when present) `n_rows * n_targets` long, and weights and label
    /// bounds (when present) one per row.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] naming the inconsistent field.
    pub(crate) fn check_layout(&self) -> Result<()> {
        let n_targets = self.n_targets();
        let cells = self.n_rows.checked_mul(n_targets);
        if cells.is_none() {
            return Err(HessboostError::invalid_param(
                "n_targets",
                format!("dataset has {n_targets} targets for {} rows", self.n_rows),
            ));
        }
        if let Some(labels) = self.labels
            && cells != Some(labels.values().len())
        {
            return Err(self.label_count_error());
        }
        for (name, values) in [
            ("weights", self.weights),
            ("label_lower_bound", self.bounds.map(|b| b.lower())),
            ("label_upper_bound", self.bounds.map(|b| b.upper())),
        ] {
            if let Some(values) = values
                && values.len() != self.n_rows
            {
                return Err(HessboostError::invalid_param(
                    name,
                    format!(
                        "dataset has {} {name} for {} rows",
                        values.len(),
                        self.n_rows
                    ),
                ));
            }
        }
        Ok(())
    }

    /// The `labels` error of a label count that is not `n_rows * n_targets`.
    fn label_count_error(&self) -> HessboostError {
        HessboostError::invalid_param(
            "labels",
            format!(
                "dataset has {} labels for {} rows of {} targets",
                self.label_values().len(),
                self.n_rows,
                self.n_targets()
            ),
        )
    }

    /// The row weights repeated for each of the row's `n_targets` cells
    /// (`[row][target]`, length `n_rows * n_targets`), which is how XGBoost's
    /// elementwise objectives and metrics weight a label matrix; `None` when
    /// the rows are unweighted.
    ///
    /// # Errors
    ///
    /// The [`check_layout`](Self::check_layout) error when the lengths are
    /// inconsistent, and an `InvalidParameter` naming `labels` when there
    /// are no labels to back the cells (nothing is allocated on error).
    pub(crate) fn cell_weights(&self) -> Result<Option<Vec<f32>>> {
        self.check_layout()?;
        if self.labels.is_none() {
            return Err(self.label_count_error());
        }
        let n_targets = self.n_targets();
        Ok(self.weights.map(|w| {
            w.iter()
                .flat_map(|&wi| std::iter::repeat_n(wi, n_targets))
                .collect()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_prefix_sum() {
        let g = GroupInfo::from_sizes(&[3, 2, 4]);
        assert_eq!(g.group_ptr, vec![0, 3, 5, 9]);
        assert_eq!(g.num_groups(), 3);
        assert_eq!(g.num_rows(), 9);
        let ranges: Vec<_> = g.iter_ranges().collect();
        assert_eq!(ranges, vec![(0, 3), (3, 5), (5, 9)]);
    }

    /// `cell_weights` broadcasts only consistent metadata: a huge
    /// `n_targets` once overflowed the allocation.
    #[test]
    fn cell_weights_need_a_consistent_layout() {
        let labels = [1.0, 2.0, 3.0, 4.0];
        let weights = [1.0, 2.0];
        let matrix = |k: usize| Some(Labels::new(&labels, NonZeroUsize::new(k).unwrap()));
        let info = MetaInfo {
            n_rows: 2,
            labels: matrix(2),
            ..MetaInfo::new(&labels, Some(&weights), None)
        };
        assert_eq!(info.cell_weights().unwrap(), Some(vec![1.0, 1.0, 2.0, 2.0]));
        for bad in [
            MetaInfo {
                labels: matrix(usize::MAX),
                ..info
            },
            MetaInfo {
                labels: matrix(3),
                ..info
            },
            MetaInfo {
                labels: None,
                ..info
            },
            MetaInfo {
                weights: Some(&weights[..1]),
                ..info
            },
            MetaInfo {
                bounds: Some(LabelBounds::new(&labels, &weights)),
                ..info
            },
        ] {
            assert!(bad.cell_weights().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn partitions_needs_ordered_ptr_spanning_the_rows() {
        assert!(GroupInfo::from_sizes(&[2, 0, 1]).partitions(3));
        assert!(!GroupInfo::from_sizes(&[2]).partitions(3));
        assert!(!GroupInfo::default().partitions(0));
        let unordered = GroupInfo {
            group_ptr: vec![0, 3, 1, 3],
        };
        assert!(!unordered.partitions(3));
        // Sizes summing past `usize::MAX` saturate instead of overflowing.
        assert!(!GroupInfo::from_sizes(&[usize::MAX, 2]).partitions(1));
    }
}
