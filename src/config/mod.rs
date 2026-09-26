//! Training configuration types.

mod params;
mod xgboost;

pub use params::{
    BoosterKind, Device, GrowPolicy, MAX_SYMMETRIC_DEPTH, Monotone, MultiStrategy, ProcessType,
    SamplingMethod, TrainingParams, TrainingParamsBuilder, TreeMethod,
};
