//! Training configuration types.

mod params;

pub(crate) use params::PartialObjectiveParams;
pub use params::{
    AftDistribution, BoosterKind, Device, DistGradient, DistSplitDirection, GrowPolicy,
    MAX_SYMMETRIC_DEPTH, ModelShrinkMode, Monotone, MultiStrategy, ObjectiveParams, ProcessType,
    SamplingMethod, TrainingParams, TrainingParamsBuilder, TreeMethod,
};
