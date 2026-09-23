//! Feature and instance metadata carried alongside the feature matrix.

use serde::{Deserialize, Serialize};

/// How a feature column should be treated during split finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
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
/// encoding for learning-to-rank objectives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GroupInfo {
    /// Prefix-sum group boundaries over the row index.
    pub group_ptr: Vec<usize>,
}

impl GroupInfo {
    /// Build a [`GroupInfo`] from per-group sizes (e.g. `[3, 2, 4]`).
    pub fn from_sizes(sizes: &[usize]) -> Self {
        let mut group_ptr = Vec::with_capacity(sizes.len() + 1);
        group_ptr.push(0);
        let mut acc = 0;
        for &s in sizes {
            acc += s;
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
}

/// Borrowed per-row training metadata handed to objectives and metrics.
///
/// Built by [`DMatrix::info`](crate::data::DMatrix::info), or by
/// [`MetaInfo::new`] for single-target callers that only have labels,
/// weights, and groups.
#[derive(Debug, Clone, Copy)]
pub struct MetaInfo<'a> {
    /// Number of rows (instances).
    pub n_rows: usize,
    /// Labels, row-major `[row][target]` (`n_rows * n_targets`); empty when
    /// the dataset has no labels.
    pub labels: &'a [f32],
    /// Number of targets per row.
    pub n_targets: usize,
    /// Per-row weights (`n_rows`), if any.
    pub weights: Option<&'a [f32]>,
    /// Ranking groups, if any.
    pub group: Option<&'a GroupInfo>,
    /// Lower bounds of interval-censored labels (`n_rows`), if any.
    pub label_lower_bound: Option<&'a [f32]>,
    /// Upper bounds of interval-censored labels (`n_rows`), if any.
    pub label_upper_bound: Option<&'a [f32]>,
}

impl<'a> MetaInfo<'a> {
    /// Single-target metadata: `n_rows = labels.len()`, `n_targets = 1`, no
    /// label bounds.
    pub fn new(
        labels: &'a [f32],
        weights: Option<&'a [f32]>,
        group: Option<&'a GroupInfo>,
    ) -> Self {
        MetaInfo {
            n_rows: labels.len(),
            labels,
            n_targets: 1,
            weights,
            group,
            label_lower_bound: None,
            label_upper_bound: None,
        }
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
}
