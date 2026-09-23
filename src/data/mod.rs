//! Dataset containers, metadata, and loaders.

mod dmatrix;
pub mod ghist;
mod loaders;
mod meta;
pub mod quantile;
mod sketch;
pub mod target_stats;

pub(crate) use dmatrix::is_missing;
pub use dmatrix::{CscView, DMatrix, Entry};
pub use ghist::GHistIndex;
pub use loaders::{CsvOptions, load_csv, load_libsvm, read_csv, read_libsvm};
pub use meta::{FeatureType, GroupInfo, MetaInfo};
pub use quantile::HistCuts;
pub use target_stats::{
    FittedTargetEncoder, OrderedTargetEncoder, OrderedTargetEncoderBuilder, TargetKind,
};
