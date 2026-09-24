//! Datasets: the [`DMatrix`] container, the [`MetaInfo`] view objectives and
//! metrics read, the libsvm/CSV loaders, and the opt-in [`target_stats`]
//! encoder for categorical columns.

mod dmatrix;
pub(crate) mod ghist;
mod loaders;
mod meta;
pub(crate) mod quantile;
mod sketch;
pub mod target_stats;

pub use dmatrix::DMatrix;
pub(crate) use dmatrix::{Entry, is_missing};
pub use loaders::{CsvOptions, load_csv, load_libsvm, read_csv, read_libsvm};
pub use meta::{FeatureType, GroupInfo, MetaInfo};
