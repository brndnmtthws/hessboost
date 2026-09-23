//! # hessboost
//!
//! A faithful, fast, pure-Rust reimplementation of
//! [XGBoost](https://github.com/dmlc/xgboost) gradient boosting with no C/C++
//! dependency and no FFI.
//!
//! ## Quick start
//!
//! Build a [`DMatrix`], configure [`TrainingParams`] with a builder, call
//! [`train`], then [`predict`](prelude::BoostedModel::predict):
//!
//! ```
//! use hessboost::prelude::*;
//!
//! # fn main() -> Result<()> {
//! // 6 rows × 2 features, row-major, plus a label per row.
//! let x = [0.0, 0.0,  1.0, 0.0,  0.0, 1.0,  1.0, 1.0,  0.5, 0.5,  0.2, 0.9];
//! let y = [0.0,       1.0,       1.0,       0.0,       0.5,       0.7];
//! let dtrain = DMatrix::from_dense(&x, 6, 2)?.with_labels(&y)?;
//!
//! let params = TrainingParams::builder()
//!     .objective("reg:squarederror") // XGBoost-compatible names
//!     .tree_method(TreeMethod::Hist)
//!     .max_depth(3)
//!     .eta(0.1)
//!     .build()?;
//!
//! let model = train(&params, &dtrain, 50)?;
//! let preds = model.predict(&dtrain)?;
//! assert_eq!(preds.len(), 6);
//!
//! model.save_binary("model.bin")?;      // native format
//! # std::fs::remove_file("model.bin").ok();
//! # Ok(())
//! # }
//! ```
//!
//! ## What's here
//!
//! - **Boosters:** `gbtree`, `dart`, `gblinear`, and boosted random forests
//!   (`num_parallel_tree`).
//! - **Training lifecycle:** continued training from an existing model and
//!   `process_type=update` tree refresh ([`train_continue`]), model slicing
//!   ([`BoostedModel::slice`]) and `iteration_range` prediction
//!   ([`predict_margin_range`](prelude::BoostedModel::predict_margin_range) and
//!   siblings).
//! - **Tree methods:** `exact`, `hist`, and `approx`, with `depthwise` or
//!   `lossguide` growth.
//! - **Objectives:** regression, binary/multiclass classification, count
//!   (poisson/gamma/tweedie), learning-to-rank (LambdaMART), survival
//!   (`survival:cox`, `survival:aft` on censored label bounds), and a custom hook
//!   ([`train_with_objective`]).
//! - **Metrics:** rmse, mae, logloss, error, auc, aucpr, mlogloss, merror,
//!   ndcg/map, nloglik, cox/aft-nloglik, interval-regression-accuracy, and a
//!   custom hook ([`train_with_custom_metric`]).
//! - **Modeling:** monotone & interaction constraints, native categorical
//!   splits, early stopping, feature importance, TreeSHAP contributions and
//!   interaction values ([`BoostedModel::predict_contribs`] /
//!   [`predict_interactions`](prelude::BoostedModel::predict_interactions)).
//! - **I/O:** libsvm/CSV loaders, native binary + JSON model I/O, and
//!   XGBoost-format JSON and UBJSON model import/export ([`crate::model`]).
//! - **Validation:** cross-validation ([`cv`]).
//! - **Uncertainty:** split-conformal and conformalized-quantile prediction
//!   intervals with finite-sample marginal coverage
//!   ([`SplitConformal`](prelude::SplitConformal),
//!   [`ConformalizedQuantile`](prelude::ConformalizedQuantile); see
//!   [`learner::conformal`]).
//! - **Beyond XGBoost (opt-in):** CatBoost-style ordered target statistics
//!   for categorical columns ([`data::OrderedTargetEncoder`]); LightGBM tree
//!   options `extra_trees`, `path_smooth`, and `linear_tree` leaves
//!   ([`config::TrainingParams::extra_trees`], [`config::TrainingParams::path_smooth`],
//!   [`config::TrainingParams::linear_tree`], [`tree::linear`]);
//!   CatBoost-style symmetric (oblivious) trees
//!   ([`GrowPolicy::Symmetric`](config::GrowPolicy::Symmetric)), which batch
//!   prediction routes by bit pattern; and compact models after *Boosted Trees
//!   on a Diet*: feature/threshold reuse penalties (`toad_penalty_feature`,
//!   `toad_penalty_threshold`) and a bit-packed layout predicting bit-identical
//!   margins ([`learner::compact_model`]); and PerpetualBooster-style budget
//!   training, one `budget` number instead of tuning `eta`/depth/rounds
//!   ([`learner::budget`]).
//!
//! ## Where to look
//!
//! - Entry points: [`train`], [`train_with_eval`], [`train_with_objective`],
//!   [`train_with_custom_metric`], [`train_continue`] /
//!   [`train_continue_with_eval`], [`cv`].
//! - Core types: [`DMatrix`] (data), [`TrainingParams`] (config, mirrors
//!   XGBoost parameter names), [`BoostedModel`] (trained model).
//! - Runnable examples in the crate's `examples/` directory (e.g.
//!   `binary_classification`, `multiclass`, `ranking`, `shap`, `model_io`,
//!   `custom_objective`, `constraints`, `conformal`, `compact_model`). Run one with
//!   `cargo run --release --example binary_classification`.
//!
//! ## Compatibility notes
//!
//! Objective, metric, and parameter names mirror XGBoost, so configurations
//! transfer directly. Predictions match XGBoost's *model quality* (parity is
//! CI-tested) but are not bit-identical. The two histogram implementations pick
//! slightly different split points.
//!
//! [`DMatrix`]: prelude::DMatrix
//! [`TrainingParams`]: prelude::TrainingParams
//! [`BoostedModel`]: prelude::BoostedModel
//! [`BoostedModel::predict_contribs`]: prelude::BoostedModel::predict_contribs
//! [`train`]: prelude::train
//! [`train_with_eval`]: prelude::train_with_eval
//! [`train_with_objective`]: prelude::train_with_objective
//! [`train_with_custom_metric`]: prelude::train_with_custom_metric
//! [`train_continue`]: prelude::train_continue
//! [`train_continue_with_eval`]: prelude::train_continue_with_eval
//! [`BoostedModel::slice`]: prelude::BoostedModel::slice
//! [`cv`]: prelude::cv
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod booster;
pub mod config;
pub mod data;
pub mod error;
pub mod learner;
pub mod metric;
pub mod model;
pub mod objective;
mod simd;
pub mod tree;

/// Commonly used imports include `use hessboost::prelude::*;`.
///
/// Pulls in the data container, configuration, training entry points, the model
/// type, and the objective/metric hooks. This provides everything needed for the
/// typical train to predict workflow.
pub mod prelude {
    pub use crate::config::{
        AftDistribution, BoosterKind, GrowPolicy, Monotone, MultiStrategy, ProcessType,
        SamplingMethod, TrainingParams, TreeMethod,
    };
    pub use crate::data::{CsvOptions, DMatrix, FeatureType, MetaInfo};
    pub use crate::error::{HessboostError, Result};
    pub use crate::learner::budget::{BudgetConfig, BudgetResult, BudgetStop, train_with_budget};
    pub use crate::learner::compact_model::{CompactModel, ModelSizeReport};
    pub use crate::learner::conformal::{ConformalizedQuantile, SplitConformal};
    pub use crate::learner::{
        BoostedModel, CvResult, ImportanceType, TrainResult, cv, train, train_continue,
        train_continue_with_eval, train_with_custom_metric, train_with_eval, train_with_objective,
    };
    pub use crate::metric::{CustomMetric, Metric};
    pub use crate::objective::{CustomObjective, GradPair, Objective};
}
