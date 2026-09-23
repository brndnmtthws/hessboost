//! Training configuration types.

mod params;

pub use params::{
    AftDistribution, BoosterKind, DistGradient, DistSplitDirection, GrowPolicy,
    MAX_SYMMETRIC_DEPTH, Monotone, MultiStrategy, ObjectiveParams, ProcessType, SamplingMethod,
    TrainingParams, TrainingParamsBuilder, TreeMethod,
};
