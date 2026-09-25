//! # hessboost
//!
//! [XGBoost](https://github.com/dmlc/xgboost) gradient boosting, implemented
//! in Rust. Its only C dependency is the official zstd library, which
//! compresses native model files; the opt-in `metal` feature additionally
//! binds Apple's Metal framework for GPU prediction and (bit-identical) GPU
//! histograms on macOS.
//!
//! ## Quick start
//!
//! Build a [`DMatrix`], configure [`TrainingParams`] with a builder, train
//! with [`train`] (or [`Trainer`] for eval sets, early stopping, custom
//! hooks, and continued training), then
//! [`predict`](model::BoostedModel::predict):
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
//! [`prelude`] holds only this workflow's items (including
//! [`TreeMethod`](config::TreeMethod)); everything else is imported from its
//! module.
//!
//! ## Modules
//!
//! - [`config`]: [`TrainingParams`], its builder, and the parameter enums
//!   ([`TreeMethod`](config::TreeMethod), [`GrowPolicy`](config::GrowPolicy),
//!   [`Monotone`](config::Monotone), ...).
//! - [`data`]: [`DMatrix`], [`MetaInfo`](data::MetaInfo), feature types, the
//!   CSV/libsvm loaders; [`data::target_stats`] (ordered target statistics).
//! - [`training`]: [`train`], [`Trainer`], [`cv`](training::cv);
//!   [`training::budget`] (budget-mode training).
//! - [`model`]: [`BoostedModel`] (prediction, SHAP, importance, slicing,
//!   native and XGBoost JSON/UBJSON formats); [`model::compact`]
//!   (bit-packed inference format).
//! - [`objective`]: the [`Objective`](objective::Objective) trait, the
//!   built-in objectives, [`CustomObjective`](objective::CustomObjective);
//!   [`objective::distributional`] (`dist:*` objectives).
//! - [`metric`]: the [`Metric`](metric::Metric) trait, the built-in metrics,
//!   [`CustomMetric`](metric::CustomMetric).
//! - [`conformal`]: split-conformal and conformalized-quantile prediction
//!   intervals.
//! - [`tree`]: [`RegTree`](tree::RegTree) and its nodes, for inspecting a
//!   trained model.
//! - [`error`]: [`HessboostError`](error::HessboostError) and
//!   [`Result`](error::Result).
//!
//! ## What's here
//!
//! - **Boosters:** `gbtree`, `dart`, `gblinear`, and boosted random forests
//!   (`num_parallel_tree`).
//! - **Training lifecycle:** continued training from an existing model and
//!   `process_type=update` tree refresh
//!   ([`Trainer::init_model`](training::Trainer::init_model)), model slicing
//!   ([`BoostedModel::slice`](model::BoostedModel::slice)) and
//!   `iteration_range` prediction as Rust ranges
//!   ([`predict_margin_range`](model::BoostedModel::predict_margin_range) and
//!   siblings).
//! - **Tree methods:** `exact`, `hist`, and `approx`, with `depthwise` or
//!   `lossguide` growth; uniform or `gradient_based` row sampling and
//!   column sampling, optionally weighted per feature
//!   ([`DMatrix::with_feature_weights`](data::DMatrix::with_feature_weights)).
//! - **Objectives:** regression (squared, squared-log, pseudo-Huber, smoothed
//!   absolute error, quantile and expectile alpha lists), binary
//!   (logistic, logitraw, hinge) and multiclass classification, count
//!   (poisson/gamma/tweedie), learning-to-rank (LambdaMART), survival
//!   (`survival:cox`, `survival:aft` on censored label bounds), and a custom
//!   hook ([`Trainer::objective`](training::Trainer::objective)).
//! - **Multi-output:** multi-target label matrices
//!   ([`DMatrix::with_label_matrix`](data::DMatrix::with_label_matrix)),
//!   one tree per output or vector-leaf trees
//!   ([`MultiStrategy::MultiOutputTree`](config::MultiStrategy::MultiOutputTree)).
//! - **Metrics:** rmse, rmsle, mae, mape, mphe, logloss, error, auc, aucpr,
//!   mlogloss, merror, poisson/gamma/tweedie-nloglik, ndcg, map, pre,
//!   quantile, expectile, cox/aft-nloglik, interval-regression-accuracy, and
//!   a custom hook ([`Trainer::custom_metric`](training::Trainer::custom_metric));
//!   `@k` cutoffs on the ranking metrics and `@rho` on tweedie-nloglik, any
//!   other suffix refused ([`create_metric`](metric::create_metric)).
//! - **Modeling:** monotone & interaction constraints, native categorical
//!   splits, early stopping, feature importance, QuadratureTreeSHAP
//!   contributions and interaction values
//!   ([`predict_contribs`](model::BoostedModel::predict_contribs) /
//!   [`predict_interactions`](model::BoostedModel::predict_interactions)).
//! - **I/O:** libsvm/CSV loaders, native binary + JSON model I/O, and
//!   XGBoost-format JSON and UBJSON model import/export
//!   ([XGBoost interchange](model#xgboost-interchange)).
//! - **Validation:** cross-validation ([`cv`](training::cv)), or over
//!   caller-supplied or forward-chaining (time-ordered, purged)
//!   [`Fold`](training::Fold)s with fold-mean early stopping
//!   ([`CrossValidation`](training::CrossValidation)).
//! - **Beyond XGBoost (opt-in, none changes default training):**
//!   - split-conformal and conformalized-quantile prediction intervals with
//!     finite-sample marginal coverage ([`conformal`]);
//!   - CatBoost-style ordered target statistics for categorical columns
//!     ([`data::target_stats`]);
//!   - LightGBM tree options `extra_trees`, `path_smooth`, and `linear_tree`
//!     leaves ([`TrainingParams::extra_trees`](config::TrainingParams::extra_trees),
//!     [`path_smooth`](config::TrainingParams::path_smooth),
//!     [`linear_tree`](config::TrainingParams::linear_tree),
//!     [`LinearLeaves`](tree::LinearLeaves));
//!   - CatBoost-style symmetric (oblivious) trees
//!     ([`GrowPolicy::Symmetric`](config::GrowPolicy::Symmetric)), which batch
//!     prediction routes by bit pattern;
//!   - compact models after *Boosted Trees on a Diet*: feature/threshold
//!     reuse penalties (`toad_penalty_feature`, `toad_penalty_threshold`) and
//!     a bit-packed layout predicting bit-identical margins
//!     ([`model::compact`]);
//!   - LightGBM-style quantized-gradient training
//!     ([`use_quantized_grad`](config::TrainingParams::use_quantized_grad));
//!   - PerpetualBooster-style budget training, one `budget` number instead of
//!     tuning `eta`/depth/rounds ([`training::budget`]);
//!   - distributional boosting (NGBoost / XGBoostLSS style): `dist:normal`,
//!     `dist:lognormal`, `dist:gamma`, `dist:poisson`, `dist:negbinomial`
//!     predict a full conditional distribution per row
//!     ([`predict_distribution`](model::BoostedModel::predict_distribution),
//!     [`objective::distributional`]), scored by `nll` / `crps`.
//!   - GPU acceleration on macOS 10.15+ through native Metal (`metal`
//!     feature): bit-identical GPU prediction
//!     ([`to_gpu`](model::BoostedModel::to_gpu), roughly 2.5x faster at
//!     scale) and bit-identical GPU histogram training
//!     ([`device`](config::TrainingParams::device) = `metal`; exact integer
//!     sums, with a CPU fallback outside their exact domain). The Metal API
//!     is documented only in macOS builds with the feature
//!     (`cargo doc --features metal`); elsewhere [`backend::metal`] is a
//!     stub.
//!
//! Runnable examples live in the crate's `examples/` directory (e.g.
//! `train_regression`, `binary_classification`, `multiclass`, `ranking`,
//! `shap`, `model_io`, `custom_objective`, `constraints`, `conformal`,
//! `compact_model`, `distributional`, `budget`, `ordered_target_stats`,
//! `pfn_boost`, `metal` (`--features metal`, macOS)). Run one with
//! `cargo run --release --example binary_classification`.
//!
//! ## Compatibility notes
//!
//! Objective, metric, and parameter names are XGBoost's, so an XGBoost
//! configuration carries over as is; a setting hessboost does not support
//! is refused with an error. Parity with XGBoost 3.4.2 is CI-tested: on the
//! parity fixtures, deterministic configurations reproduce XGBoost's
//! predictions within `1e-4` (quantile cuts bit for bit), and imported
//! XGBoost models predict and explain as XGBoost does. RNG-driven options
//! (row/column subsampling, forests, DART) match only in model quality,
//! because the random streams differ.
//!
//! ### Not implemented
//!
//! - Distributed and external-memory training; GPU training outside macOS.
//! - XGBoost options available at one setting only (so they are not
//!   [`TrainingParams`] fields): gblinear uses `updater = coord_descent`
//!   with `feature_selector = cyclic`; LambdaMART uses
//!   `lambdarank_pair_method = topk` (no `lambdarank_unbiased` or
//!   `ndcg_exp_gain`); DART has no `sample_type`, `normalize_type`, or
//!   `one_drop`; categorical splits use XGBoost's defaults
//!   `max_cat_to_onehot = 4` and `max_cat_threshold = 64`.
//! - The metrics `gamma-deviance`, `error@t` (XGBoost's classification
//!   threshold suffix), and the `-` variants of the ranking metrics
//!   (`ndcg-`, `ndcg@k-`, `map-`, `map@k-`); these names are refused.
//! - XGBoost import and export of gblinear models.
//!
//! [`DMatrix`]: data::DMatrix
//! [`TrainingParams`]: config::TrainingParams
//! [`BoostedModel`]: model::BoostedModel
//! [`train`]: training::train
//! [`Trainer`]: training::Trainer
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod backend;
pub mod config;
pub mod conformal;
pub mod data;
pub mod error;
pub mod metric;
pub mod model;
pub mod objective;
mod rng;
mod simd;
#[cfg(test)]
mod test_support;
pub mod training;
pub mod tree;
/// `1e-6` in `f64` arithmetic, where the crate compares against XGBoost's
/// `kRtEps` in double precision (the `f64` literal, not [`K_RT_EPS_F32`]
/// widened).
pub(crate) const K_RT_EPS: f64 = 1e-6;
/// XGBoost's `kRtEps` (`1e-6f`): the minimum gain improvement a split must
/// beat, and the floor of sampling weights and near-zero sums.
pub(crate) const K_RT_EPS_F32: f32 = 1e-6;
/// The train-and-predict workflow in one import: `use hessboost::prelude::*;`.
///
/// Holds the data container, the parameters, the training entry points, the
/// model, the error types, and the types their everyday methods take:
/// [`TreeMethod`](config::TreeMethod) (for
/// [`TrainingParamsBuilder::tree_method`](config::TrainingParamsBuilder::tree_method)),
/// [`ImportanceType`](model::ImportanceType) (for
/// [`BoostedModel::feature_importance`](model::BoostedModel::feature_importance)),
/// and [`ObjectiveParams`](config::ObjectiveParams) (a model's
/// [`objective_params`](model::BoostedModel::objective_params)). Everything
/// else (the other parameter enums, objectives, metrics, conformal
/// intervals, ...) is imported from its module.
pub mod prelude {
    pub use crate::config::{ObjectiveParams, TrainingParams, TreeMethod};
    pub use crate::data::DMatrix;
    pub use crate::error::{HessboostError, Result};
    pub use crate::model::{BoostedModel, ImportanceType};
    pub use crate::training::{Trainer, train};
}
/// Implementation details the crate's own benchmarks and parity tests
/// drive directly (histogram construction, tree growth, quantile cuts). Not
/// part of the public API: hidden from the docs and changed without notice.
#[doc(hidden)]
pub mod internals {
    pub use crate::data::ghist::GHistIndex;
    pub use crate::data::quantile::HistCuts;
    pub use crate::tree::builder::HistTreeBuilder;
    pub use crate::tree::hist::{CpuBackend, HistogramBackend, zeroed};
    pub use crate::tree::sampler::ColumnSampler;
}
