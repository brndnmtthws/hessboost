//! Top-level orchestration: the boosting loop, budget-mode training, the
//! trained model, prediction, feature importance, and conformal prediction
//! intervals.

pub mod budget;
pub mod conformal;
mod cv;
pub(crate) mod model;
mod shap;
mod train;

pub use budget::{BudgetConfig, BudgetResult, BudgetStop, train_with_budget};
pub use cv::{CvResult, cv};
pub(crate) use model::LinearModel;
pub use model::{BoostedModel, ImportanceType};
pub use train::{
    EvalSet, RoundEval, TrainResult, train, train_with_custom_metric, train_with_eval,
    train_with_objective,
};
