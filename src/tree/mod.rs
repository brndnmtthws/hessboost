//! Decision-tree representation, split-scoring math, and construction.

pub mod builder;
pub(crate) mod compact;
pub mod constraints;
pub mod gain;
pub mod hist;
mod regtree;
pub mod sampler;

pub use gain::{calc_gain, calc_weight, GradStats, RegParams};
pub use regtree::{Node, RegTree};
