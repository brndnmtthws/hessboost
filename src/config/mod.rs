//! Training configuration types.

mod groups;
mod params;
mod xgboost;

pub use groups::{
    BalancedBagging, Dart, DartBuilder, ExtraTrees, LinearTree, QuantizedGrad,
    QuantizedGradBuilder, Refresh,
};
pub use params::{
    BoosterKind, Device, GrowPolicy, MAX_SYMMETRIC_DEPTH, MaxDeltaStep, Monotone, MultiStrategy,
    ProcessType, SamplingMethod, TrainingParams, TrainingParamsBuilder, TreeMethod,
};
