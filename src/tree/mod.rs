//! Decision-tree representation, split-scoring math, and construction.

pub mod builder;
pub(crate) mod compact;
pub mod constraints;
pub mod gain;
pub mod hist;
mod regtree;
pub(crate) mod reuse;
pub mod sampler;

pub use gain::{GradStats, RegParams, calc_gain, calc_weight};
pub use regtree::{Node, RegTree};
