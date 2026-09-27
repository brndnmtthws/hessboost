//! Training configuration types.

mod groups;
mod params;
mod xgboost;

pub use groups::{
    Dart, DartBuilder, ExtraTrees, Langevin, LangevinBuilder, LinearTree, ModelShrink,
    ModelShrinkBuilder, ModelShrinkMode, QuantizedGrad, QuantizedGradBuilder, Refresh,
};
pub use params::{
    BoosterKind, Device, GrowPolicy, MAX_SYMMETRIC_DEPTH, MaxDeltaStep, Monotone, MultiStrategy,
    ProcessType, SamplingMethod, TrainingParams, TrainingParamsBuilder, TreeMethod,
};
