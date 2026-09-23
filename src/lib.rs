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
//! - **Boosters:** `gbtree`, `dart`, `gblinear`.
//! - **Tree methods:** `exact`, `hist`, and `approx`, with `depthwise` or
//!   `lossguide` growth.
//! - **Objectives:** regression, binary/multiclass classification, count
//!   (poisson/gamma/tweedie), learning-to-rank (LambdaMART), and a custom hook
//!   ([`train_with_objective`]).
//! - **Metrics:** rmse, mae, logloss, error, auc, aucpr, mlogloss, merror,
//!   ndcg/map, nloglik, and a custom hook ([`train_with_custom_metric`]).
//! - **Modeling:** monotone & interaction constraints, native categorical
//!   splits, early stopping, feature importance, TreeSHAP contributions and
//!   interaction values ([`BoostedModel::predict_contribs`] /
//!   [`predict_interactions`](prelude::BoostedModel::predict_interactions)).
//! - **I/O:** libsvm/CSV loaders, native binary + JSON model I/O, and
//!   XGBoost-format JSON model import/export ([`crate::model`]).
//! - **Validation:** cross-validation ([`cv`]).
//! - **Beyond XGBoost (opt-in):** CatBoost-style ordered target statistics
//!   for categorical columns ([`data::OrderedTargetEncoder`]).
//!
//! ## Where to look
//!
//! - Entry points: [`train`], [`train_with_eval`], [`train_with_objective`],
//!   [`train_with_custom_metric`], [`cv`].
//! - Core types: [`DMatrix`] (data), [`TrainingParams`] (config, mirrors
//!   XGBoost parameter names), [`BoostedModel`] (trained model).
//! - Runnable examples in the crate's `examples/` directory (e.g.
//!   `binary_classification`, `multiclass`, `ranking`, `shap`, `model_io`,
//!   `custom_objective`, `constraints`). Run one with
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
    pub use crate::config::{BoosterKind, GrowPolicy, Monotone, TrainingParams, TreeMethod};
    pub use crate::data::{CsvOptions, DMatrix, FeatureType};
    pub use crate::error::{HessboostError, Result};
    pub use crate::learner::{
        BoostedModel, CvResult, ImportanceType, TrainResult, cv, train, train_with_custom_metric,
        train_with_eval, train_with_objective,
    };
    pub use crate::metric::{CustomMetric, Metric};
    pub use crate::objective::{CustomObjective, GradPair, Objective};
}
