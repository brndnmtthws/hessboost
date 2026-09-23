//! Decision-tree representation, split-scoring math, and construction.

pub mod builder;
pub(crate) mod compact;
pub mod constraints;
pub mod gain;
pub mod hist;
pub(crate) mod oblivious;
mod regtree;
pub mod sampler;

pub use gain::{GradStats, RegParams, calc_gain, calc_weight};
pub use regtree::{Node, RegTree};
