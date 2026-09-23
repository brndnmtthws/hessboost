//! Training configuration types.

mod params;

pub use params::{
    AftDistribution, BoosterKind, GrowPolicy, Monotone, MultiStrategy, ObjectiveParams,
    ProcessType, SamplingMethod, TrainingParams, TrainingParamsBuilder, TreeMethod,
};
