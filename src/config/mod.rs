//! Training configuration types.

/// A chaining builder setter `name(v)` storing `v` at `self.<path>`, or
/// `Some(v)` for `=> Some(path)`; the doc attributes carry over.
macro_rules! setter {
    ($(#[$m:meta])* $name:ident: $ty:ty => Some($($path:ident).+)) => {
        $(#[$m])*
        #[must_use]
        pub fn $name(mut self, v: $ty) -> Self {
            self.$($path).+ = Some(v);
            self
        }
    };
    ($(#[$m:meta])* $name:ident: $ty:ty => $($path:ident).+) => {
        $(#[$m])*
        #[must_use]
        pub fn $name(mut self, v: $ty) -> Self {
            self.$($path).+ = v;
            self
        }
    };
}

mod groups;
mod params;
mod xgboost;

pub use groups::{
    BalancedBagging, Boulevard, BoulevardBuilder, Dart, DartBuilder, Ebm, EbmBuilder,
    EbmEarlyStopping, ExtraTrees, Langevin, LangevinBuilder, LinearTree, ModelShrink,
    ModelShrinkMode, QuantizedGrad, QuantizedGradBuilder, QueryBagging, Refresh,
};
pub use params::{
    BoosterKind, Device, GrowPolicy, MAX_SYMMETRIC_DEPTH, MaxDeltaStep, Monotone, MultiStrategy,
    ProcessType, SamplingMethod, TrainingParams, TrainingParamsBuilder, TreeMethod,
};
