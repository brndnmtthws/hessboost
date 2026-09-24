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
