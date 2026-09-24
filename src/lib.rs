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
//!   `lossguide` growth; uniform or `gradient_based` row sampling and
//!   column sampling, optionally weighted per feature
//!   ([`DMatrix::with_feature_weights`](prelude::DMatrix::with_feature_weights)).
//! - **Objectives:** regression (squared, squared-log, pseudo-Huber, smoothed
//!   absolute error, quantile and expectile alpha lists), binary
//!   (logistic, logitraw, hinge) and multiclass classification, count
//!   (poisson/gamma/tweedie), learning-to-rank (LambdaMART), survival
//!   (`survival:cox`, `survival:aft` on censored label bounds), and a custom
//!   hook ([`train_with_objective`]).
//! - **Multi-output:** multi-target label matrices
//!   ([`DMatrix::with_label_matrix`](prelude::DMatrix::with_label_matrix)),
//!   one tree per output or vector-leaf trees
//!   ([`MultiStrategy::MultiOutputTree`](prelude::MultiStrategy::MultiOutputTree)).
//! - **Metrics:** rmse, rmsle, mae, mape, mphe, logloss, error, auc, aucpr,
//!   mlogloss, merror, poisson/gamma/tweedie-nloglik, ndcg, map, pre,
//!   quantile, expectile, cox/aft-nloglik, interval-regression-accuracy, and
//!   a custom hook ([`train_with_custom_metric`]).
//! - **Modeling:** monotone & interaction constraints, native categorical
//!   splits, early stopping, feature importance, QuadratureTreeSHAP
//!   contributions and interaction values ([`BoostedModel::predict_contribs`] /
//!   [`predict_interactions`](prelude::BoostedModel::predict_interactions)).
//! - **I/O:** libsvm/CSV loaders, native binary + JSON model I/O, and
//!   XGBoost-format JSON and UBJSON model import/export ([`crate::model`]).
//! - **Validation:** cross-validation ([`cv`]).
//! - **Beyond XGBoost (opt-in):** split-conformal and conformalized-quantile
//!   prediction intervals with finite-sample marginal coverage
//!   ([`SplitConformal`](prelude::SplitConformal),
//!   [`ConformalizedQuantile`](prelude::ConformalizedQuantile); see
//!   [`learner::conformal`]); CatBoost-style ordered target statistics
//!   for categorical columns ([`data::OrderedTargetEncoder`]); LightGBM tree
//!   options `extra_trees`, `path_smooth`, and `linear_tree` leaves
//!   ([`config::TrainingParams::extra_trees`], [`config::TrainingParams::path_smooth`],
//!   [`config::TrainingParams::linear_tree`], [`tree::linear`]);
//!   CatBoost-style symmetric (oblivious) trees
//!   ([`GrowPolicy::Symmetric`](config::GrowPolicy::Symmetric)), which batch
//!   prediction routes by bit pattern; compact models after *Boosted Trees
//!   on a Diet*: feature/threshold reuse penalties (`toad_penalty_feature`,
//!   `toad_penalty_threshold`) and a bit-packed layout predicting bit-identical
//!   margins ([`learner::compact_model`]); LightGBM-style quantized-gradient
//!   training ([`config::TrainingParams::use_quantized_grad`]);
//!   PerpetualBooster-style budget training, one `budget` number instead of
//!   tuning `eta`/depth/rounds ([`learner::budget`]); and distributional
//!   boosting (NGBoost / XGBoostLSS style): `dist:normal`, `dist:lognormal`,
//!   `dist:gamma`, `dist:poisson`, `dist:negbinomial` predict a full
//!   conditional distribution per row
//!   ([`BoostedModel::predict_distribution`](prelude::BoostedModel::predict_distribution),
//!   [`objective::distributional`]), scored by `nll` / `crps`. None of them
//!   changes default training.
//!
//! ## Where to look
//!
//! - Entry points: [`train`], [`train_with_eval`], [`train_with_objective`],
//!   [`train_with_custom_metric`], [`train_continue`] /
//!   [`train_continue_with_eval`], [`cv`],
//!   [`train_with_budget`](prelude::train_with_budget).
//! - Core types: [`DMatrix`] (data), [`TrainingParams`] (config, mirrors
//!   XGBoost parameter names), [`BoostedModel`] (trained model).
//! - Runnable examples in the crate's `examples/` directory (e.g.
//!   `binary_classification`, `multiclass`, `ranking`, `shap`, `model_io`,
//!   `custom_objective`, `constraints`, `conformal`, `compact_model`,
//!   `distributional`). Run one with
//!   `cargo run --release --example binary_classification`.
//!
//! ## Compatibility notes
//!
//! Objective, metric, and parameter names mirror XGBoost, so configurations
//! transfer directly. Parity with XGBoost 3.4.2 is CI-tested: on the parity
//! fixtures, deterministic configurations reproduce XGBoost's predictions
//! within `1e-4` (quantile cuts bit for bit), and imported XGBoost models
//! predict and explain as XGBoost does. RNG-driven options (row/column
//! subsampling, DART) match only in model quality, because the random
//! streams differ.
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
mod rng;
mod simd;
#[cfg(test)]
mod test_support;
pub mod tree;

/// `1e-6` in `f64` arithmetic, where the crate compares against XGBoost's
/// `kRtEps` in double precision (the `f64` literal, not [`K_RT_EPS_F32`]
/// widened).
pub(crate) const K_RT_EPS: f64 = 1e-6;
/// XGBoost's `kRtEps` (`1e-6f`): the minimum gain improvement a split must
/// beat, and the floor of sampling weights and near-zero sums.
pub(crate) const K_RT_EPS_F32: f32 = 1e-6;

/// Commonly used imports include `use hessboost::prelude::*;`.
///
/// Pulls in the data container, configuration, training entry points, the model
/// type, and the objective/metric hooks. This provides everything needed for the
/// typical train to predict workflow.
pub mod prelude {
    pub use crate::config::{
        AftDistribution, BoosterKind, DistGradient, DistSplitDirection, GrowPolicy, Monotone,
        MultiStrategy, ProcessType, SamplingMethod, TrainingParams, TreeMethod,
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
    pub use crate::objective::{
        CustomObjective, Dist, DistFamily, GradPair, Objective, SplitGradient,
    };
}
