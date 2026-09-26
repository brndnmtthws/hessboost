//! # hessboost
//!
//! [XGBoost](https://github.com/dmlc/xgboost) gradient boosting in Rust. The
//! only C dependency is zstd (native model files); the opt-in `metal`
//! feature adds Apple's Metal framework for GPU prediction and bit-identical
//! GPU histograms on macOS.
//!
//! ## Quick start
//!
//! Build a [`DMatrix`], set [`TrainingParams`] with its builder, train with
//! [`train`] (or [`Trainer`] for eval sets, early stopping, custom hooks,
//! and continued training), then
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
//!     .objective(Objective::SquaredError) // XGBoost's `reg:squarederror`
//!     .tree_method(TreeMethod::Hist)
//!     .max_depth(3)
//!     .eta(0.1)
//!     .build()?;
//!
//! let model = train(&params, &dtrain, 50)?;
//! let preds = model.predict(&dtrain)?;
//! assert_eq!((preds.n_rows(), preds.width()), (6, 1)); // `[row][output]`
//!
//! model.save_binary("model.bin")?;      // native format
//! # std::fs::remove_file("model.bin").ok();
//! # Ok(())
//! # }
//! ```
//!
//! [`prelude`] holds only this workflow's items (including
//! [`TreeMethod`](config::TreeMethod) and [`Objective`](objective::Objective));
//! everything else is imported from its module.
//!
//! ## Modules
//!
//! - [`config`]: [`TrainingParams`], builder, parameter enums.
//! - [`data`]: [`DMatrix`], [`MetaInfo`](data::MetaInfo), feature types,
//!   CSV/libsvm loaders, [`data::target_stats`].
//! - [`training`]: [`train`], [`Trainer`], [`cv`](training::cv),
//!   [`training::budget`], [`training::online`].
//! - [`model`]: [`BoostedModel`] (prediction, SHAP, importance, slicing,
//!   native and XGBoost JSON/UBJSON, LightGBM text import);
//!   [`model::compact`], [`model::uncertainty`].
//! - [`objective`]: [`Objective`](objective::Objective) and its parameter
//!   types, the `Loss` trait, `CustomLoss`, [`objective::distributional`]
//!   (`dist:*` objectives).
//! - [`metric`]: [`EvalMetric`](metric::EvalMetric) (the built-in metrics),
//!   the `Metric` trait, `CustomMetric`.
//! - [`conformal`]: split-conformal and conformalized-quantile intervals.
//! - [`inference`]: Boulevard boosting's confidence and prediction intervals
//!   for `f(x)`, and a Boulevard EBM's shape-function bands.
//! - [`ebm`]: explainable boosting machines' terms and shape functions.
//! - [`diffusion`]: conditional diffusion and flow matching with GBDT score
//!   models, sampling a nonparametric `p(y | x)`.
//! - [`tree`]: [`RegTree`](tree::RegTree) and nodes, for model inspection.
//! - [`error`]: `HessboostError` and `Result`.
//!
//! ## What's here
//!
//! - **Boosters:** `gbtree`, `dart`, `gblinear`, boosted random forests
//!   (`num_parallel_tree`).
//! - **Lifecycle:** continued training and `process_type=update` refresh
//!   ([`Trainer::init_model`](training::Trainer::init_model)), a per-round
//!   hook for progress, custom stopping, and cancellation
//!   ([`Trainer::on_round`](training::Trainer::on_round)), slicing
//!   ([`BoostedModel::slice`](model::BoostedModel::slice)), `iteration_range`
//!   prediction as Rust ranges
//!   ([`predict_margin_range`](model::BoostedModel::predict_margin_range) and
//!   siblings).
//! - **Tree methods:** `exact`, `hist`, `approx`; `depthwise`/`lossguide`
//!   growth; uniform or `gradient_based` row sampling; column sampling with
//!   optional per-feature weights
//!   ([`DMatrix::with_feature_weights`](data::DMatrix::with_feature_weights)).
//! - **Objectives** ([`Objective`](objective::Objective), each with its
//!   parameters): regression (squared, squared-log, pseudo-Huber, smoothed
//!   absolute, quantile/expectile lists), binary (logistic, logitraw, hinge)
//!   and multiclass, counts, LambdaMART and XE-NDCG ranking, survival (`survival:cox`,
//!   `survival:aft` on censored bounds), plus custom losses
//!   ([`Objective::Custom`](objective::Objective::Custom), e.g. a
//!   [`CustomLoss`](objective::CustomLoss)).
//! - **Multi-output:** label matrices
//!   ([`DMatrix::with_label_matrix`](data::DMatrix::with_label_matrix)), one
//!   tree per output or vector-leaf trees
//!   ([`MultiStrategy::MultiOutputTree`](config::MultiStrategy::MultiOutputTree)).
//! - **Metrics** ([`EvalMetric`](metric::EvalMetric), each with its own
//!   parameters): rmse, rmsle, mae, mape, mphe, logloss, error, auc, aucpr,
//!   mlogloss, merror, poisson/gamma/tweedie-nloglik, ndcg, map, pre,
//!   quantile, expectile, cox/aft-nloglik, interval-regression-accuracy, plus
//!   a custom hook ([`Trainer::custom_metric`](training::Trainer::custom_metric),
//!   reported after them as in XGBoost). XGBoost's names parse through
//!   [`TrainingParams::from_xgboost`](config::TrainingParams::from_xgboost):
//!   `@k` ranking cutoffs and `@rho` on tweedie-nloglik, other suffixes
//!   refused.
//! - **Modeling:** monotone and interaction constraints, native categorical
//!   splits, early stopping, feature importance, QuadratureTreeSHAP values
//!   and interactions
//!   ([`predict_contribs`](model::BoostedModel::predict_contribs) /
//!   [`predict_interactions`](model::BoostedModel::predict_interactions)).
//! - **I/O:** libsvm/CSV loaders, native binary + JSON, XGBoost JSON and
//!   UBJSON import/export ([XGBoost interchange](model#xgboost-interchange)),
//!   LightGBM 4.x text model import
//!   ([`from_lightgbm_text`](model::BoostedModel::from_lightgbm_text); see
//!   [LightGBM import](model#lightgbm-import)).
//! - **Validation:** cross-validation ([`cv`](training::cv)), custom,
//!   forward-chaining (time-ordered, purged by a row gap), or purged forward
//!   (timestamped rows, purged by each label window,
//!   [`Fold::purged_forward`](training::Fold::purged_forward))
//!   [`Fold`](training::Fold)s with fold-mean early stopping
//!   ([`CrossValidation`](training::CrossValidation)).
//! - **Beyond XGBoost (opt-in, default training unchanged):**
//!   - split-conformal and conformalized-quantile intervals with
//!     finite-sample marginal coverage ([`conformal`]);
//!   - Boulevard boosting (Zhou & Hooker, JMLR 2022) and its dropout
//!     (BRAT-D) and parallel (BRAT-P) variants (Fang, Tan & Hooker, NeurIPS
//!     2025) with CLT-based confidence intervals for `f(x)`, prediction and
//!     reproduction intervals, an
//!     honest leaf refit, and exact or Nyström variance
//!     ([`BoosterKind::Boulevard`](config::BoosterKind::Boulevard),
//!     [`inference`]);
//!   - explainable boosting machines (GA²M: cyclic per-feature boosting,
//!     early-stopped outer bags, FAST pair terms, numerical and categorical
//!     terms; Lou et al., KDD 2012/2013, InterpretML)
//!     with per-term shape functions, and their Boulevard variant (Fang, Tan,
//!     Pipping & Hooker, AISTATS 2026) with confidence bands on every shape
//!     ([`BoosterKind::Ebm`](config::BoosterKind::Ebm), [`ebm`],
//!     [`EbmInference`](inference::EbmInference));
//!   - CatBoost-style ordered target statistics ([`data::target_stats`]);
//!   - LightGBM options `extra_trees`, `path_smooth`, `linear_tree` leaves
//!     ([`TrainingParams::extra_trees`](config::TrainingParams::extra_trees),
//!     [`path_smooth`](config::TrainingParams::path_smooth),
//!     [`linear_tree`](config::TrainingParams::linear_tree),
//!     [`LinearLeaves`](tree::LinearLeaves));
//!   - LightGBM class-balanced bagging for binary classification
//!     ([`BalancedBagging`](config::BalancedBagging): `pos_bagging_fraction`,
//!     `neg_bagging_fraction`), in place of `subsample`;
//!   - LightGBM XE-NDCG ranking
//!     ([`Objective::RankXendcg`](objective::Objective::RankXendcg); its keyed
//!     per-round draws differ from LightGBM's random stream) and query-level
//!     bagging ([`QueryBagging`](config::QueryBagging), `bagging_by_query`);
//!   - CatBoost-style symmetric trees
//!     ([`GrowPolicy::Symmetric`](config::GrowPolicy::Symmetric)), routed by
//!     bit pattern in batch prediction;
//!   - *Boosted Trees on a Diet* reuse penalties
//!     (`toad_penalty_feature`, `toad_penalty_threshold`) and a bit-packed
//!     layout with bit-identical margins ([`model::compact`]);
//!   - LightGBM-style quantized gradients
//!     ([`QuantizedGrad`](config::QuantizedGrad), `use_quantized_grad`);
//!   - PerpetualBooster-style budget training: one `budget` instead of
//!     `eta`/depth/rounds ([`training::budget`]);
//!   - in-place row addition and deletion (incremental learning and machine
//!     unlearning) for trained hist models, exact or approximate
//!     ([`training::online`]);
//!   - distributional boosting (NGBoost / XGBoostLSS style): `dist:normal`,
//!     `dist:lognormal`, `dist:gamma`, `dist:poisson`, `dist:negbinomial`
//!     per-row distributions
//!     ([`predict_distribution`](model::BoostedModel::predict_distribution),
//!     [`objective::distributional`]), scored by `nll` / `crps`;
//!   - CatBoost's Stochastic Gradient Langevin Boosting and model shrinkage
//!     ([`langevin`](config::TrainingParams::langevin),
//!     [`model_shrink`](config::TrainingParams::model_shrink),
//!     [`posterior_sampling`](config::TrainingParams::posterior_sampling))
//!     with virtual ensembles: knowledge, data, and total uncertainty from
//!     one model's exactly rebuilt truncations
//!     ([`predict_uncertainty`](model::BoostedModel::predict_uncertainty),
//!     [`model::uncertainty`]);
//!   - nonparametric `p(y | x)` by tree-based conditional diffusion
//!     (Treeffuser) and flow matching (DiffGBM) for scalar or vector
//!     labels, sampled deterministically ([`diffusion`]);
//!   - ForestFlow / ForestDiffusion tabular generation and imputation
//!     ([`diffusion::forest`]);
//!   - native Metal on macOS 10.15+ (`metal` feature): bit-identical GPU
//!     prediction ([`to_gpu`](model::BoostedModel::to_gpu), ~2.5x faster at
//!     scale) and bit-identical GPU histograms
//!     ([`device`](config::TrainingParams::device) = `metal`; exact integer
//!     sums, CPU fallback outside their exact domain). Documented only in
//!     macOS builds with the feature (`cargo doc --features metal`);
//!     elsewhere [`backend::metal`] is a stub.
//!
//! `examples/` has one program per topic (`train_regression`,
//! `binary_classification`, `multiclass`, `ranking`, `rank_xendcg`, `shap`,
//! `model_io`, `custom_objective`, `constraints`, `conformal`,
//! `boulevard_inference`, `ebm`, `compact_model`, `distributional`,
//! `virtual_ensembles`, `tree_diffusion`, `forest_flow`, `budget`,
//! `balanced_bagging`, `online_update`, `ordered_target_stats`, `pfn_boost`,
//! `metal` with `--features metal` on macOS). Run one with
//! `cargo run --release --example binary_classification`.
//!
//! ## Compatibility notes
//!
//! Parameter, objective, and metric names are XGBoost's, so an XGBoost
//! configuration carries over:
//! [`TrainingParams::from_xgboost`](config::TrainingParams::from_xgboost)
//! reads an XGBoost `params` dict (keys, aliases, and value spellings), and
//! unsupported settings are refused. Parity with
//! XGBoost 3.4.2 is CI-tested: deterministic fixtures reproduce XGBoost
//! within `1e-4` (quantile cuts bit for bit), and imported XGBoost models
//! predict and explain as XGBoost does. RNG-driven options (subsampling,
//! forests, DART) match in quality only — the streams differ.
//!
//! ### Not implemented
//!
//! - Distributed and external-memory training; GPU training outside macOS.
//! - XGBoost options available at one setting only (so they are not
//!   [`TrainingParams`] fields; `from_xgboost` accepts exactly that
//!   setting): gblinear uses `updater = coord_descent`
//!   with `feature_selector = cyclic`; LambdaMART uses
//!   `lambdarank_pair_method = topk` (no `lambdarank_unbiased` or
//!   `ndcg_exp_gain`); DART has no `sample_type` or `normalize_type` (it
//!   samples uniformly and normalizes by `tree`); categorical splits use
//!   XGBoost's defaults
//!   `max_cat_to_onehot = 4` and `max_cat_threshold = 64`.
//! - The metrics `gamma-deviance`, `error@t` (XGBoost's classification
//!   threshold suffix), and the `-` variants of the ranking metrics
//!   (`ndcg-`, `ndcg@k-`, `map-`, `map@k-`); these names are refused.
//! - XGBoost import and export of gblinear models.
//! - `scale_pos_weight` outside the logistic objectives: XGBoost also
//!   weights the positive rows of `reg:squarederror` and `reg:gamma` with
//!   it; [`TrainingParams::from_xgboost`](config::TrainingParams::from_xgboost)
//!   refuses it there, and XGBoost import drops it (predictions do not read
//!   it).
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
pub mod diffusion;
pub mod ebm;
pub mod error;
pub mod inference;
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
/// [`Objective`](objective::Objective) (for
/// [`TrainingParamsBuilder::objective`](config::TrainingParamsBuilder::objective)),
/// and [`EvalMetric`](metric::EvalMetric) (for
/// [`TrainingParamsBuilder::eval_metric`](config::TrainingParamsBuilder::eval_metric)).
/// Everything else (the other parameter enums, the objectives' and metrics'
/// parameters, conformal intervals, ...) is imported from its module.
pub mod prelude {
    pub use crate::config::{TrainingParams, TreeMethod};
    pub use crate::data::DMatrix;
    pub use crate::error::{HessboostError, Result};
    pub use crate::metric::EvalMetric;
    pub use crate::model::{BoostedModel, ImportanceType};
    pub use crate::objective::Objective;
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
