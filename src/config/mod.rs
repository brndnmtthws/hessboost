//! Training configuration types.

mod params;

pub use params::{
    AftDistribution, BoosterKind, DistGradient, GrowPolicy, MAX_SYMMETRIC_DEPTH, Monotone,
    MultiStrategy, ObjectiveParams, ProcessType, SamplingMethod, TrainingParams,
    TrainingParamsBuilder, TreeMethod,
};
