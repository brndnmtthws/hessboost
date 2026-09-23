//! Top-level orchestration: the boosting loop, the trained model, prediction,
//! feature importance, and conformal prediction intervals.

pub mod conformal;
mod cv;
pub(crate) mod model;
mod multi_output;
mod sampling;
mod shap;
mod shap_multi;
mod train;

pub use cv::{CvResult, cv};
pub(crate) use model::LinearModel;
pub use model::{BoostedModel, ImportanceType};
pub use train::{
    EvalSet, RoundEval, TrainResult, train, train_with_custom_metric, train_with_eval,
    train_with_objective,
};
