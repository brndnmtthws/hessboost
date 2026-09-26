//! Training configuration types.

mod groups;
mod params;
mod xgboost;

pub use groups::{
    Dart, DartBuilder, ExtraTrees, LinearTree, QuantizedGrad, QuantizedGradBuilder, Refresh,
};
pub use params::{
    BoosterKind, Device, GrowPolicy, MAX_SYMMETRIC_DEPTH, Monotone, MultiStrategy, ProcessType,
    SamplingMethod, TrainingParams, TrainingParamsBuilder, TreeMethod,
};
