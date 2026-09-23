//! Top-level orchestration: the boosting loop, the trained model, prediction,
//! feature importance, and conformal prediction intervals.

pub mod compact_model;
pub mod conformal;
mod continuation;
mod cv;
pub(crate) mod model;
mod refresh;
mod sampling;
mod shap;
mod train;

pub use cv::{CvResult, cv};
pub(crate) use model::LinearModel;
pub use model::{BoostedModel, ImportanceType};
pub use train::{
    EvalSet, RoundEval, TrainResult, train, train_continue, train_continue_with_eval,
    train_with_custom_metric, train_with_eval, train_with_objective,
};
