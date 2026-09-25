//! The trees of a trained model ([`BoostedModel::trees`](crate::model::BoostedModel::trees)),
//! for inspection: [`RegTree`], its [`Node`]s, and the [`LinearLeaves`] of
//! `linear_tree` models.

pub(crate) mod builder;
pub(crate) mod compact;
pub(crate) mod constraints;
pub(crate) mod gain;
pub(crate) mod hist;
pub(crate) mod linear;
pub(crate) mod oblivious;
mod regtree;
pub(crate) mod reuse;
pub(crate) mod sampler;

pub use linear::LinearLeaves;
pub(crate) use regtree::{ChildLeaf, SplitRule, UncheckedRegTree};
pub use regtree::{Node, RegTree};

/// Output of scalar tree `t` in an ensemble of `n_outputs` outputs with
/// `num_parallel_tree` consecutive trees per output (XGBoost `tree_info`):
/// iteration `i` owns trees `i * n_outputs * num_parallel_tree ..`, grouped
/// by output.
#[inline]
pub(crate) fn scalar_tree_output(t: usize, num_parallel_tree: usize, n_outputs: usize) -> usize {
    (t / num_parallel_tree) % n_outputs
}

/// Whether the integer-coded category `v` (non-missing) is in the left set
/// `categories` of a categorical split.
#[inline]
pub(crate) fn in_category_set(categories: &[u32], v: f32) -> bool {
    categories.contains(&(v as u32))
}

/// What a split compares a present value with.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SplitTest<'a> {
    /// Rows with `value < threshold` go left, other present values right.
    Threshold(f32),
    /// Rows whose category is in the set go left, other present categories
    /// right.
    Categories(&'a [u32]),
}

/// Whether a row whose split value is `value` (`None` = missing) goes left:
/// a missing value follows `default_left`, a present one `test`. The one
/// routing rule of [`RegTree`] and the SHAP walk.
#[inline]
pub(crate) fn split_goes_left(value: Option<f32>, default_left: bool, test: SplitTest<'_>) -> bool {
    match (value, test) {
        (None, _) => default_left,
        (Some(v), SplitTest::Threshold(threshold)) => v < threshold,
        (Some(v), SplitTest::Categories(categories)) => in_category_set(categories, v),
    }
}
