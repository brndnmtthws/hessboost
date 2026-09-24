//! Decision-tree representation, split-scoring math, and construction.

pub mod builder;
pub(crate) mod compact;
pub mod constraints;
pub mod gain;
pub mod hist;
pub mod linear;
pub(crate) mod oblivious;
mod regtree;
pub(crate) mod reuse;
pub mod sampler;

pub use gain::{GradStats, RegParams, calc_gain, calc_weight};
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
