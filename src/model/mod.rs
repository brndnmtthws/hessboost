//! Trained models: [`BoostedModel`], its prediction, explanation, and
//! persistence, and the size-optimized [`compact`] form.
//!
//! # Prediction
//!
//! [`BoostedModel::predict`] applies the objective's output transform
//! (probabilities for `binary:logistic`, ...), [`predict_margin`] returns raw
//! margins, [`predict_class`] class indices or thresholded labels,
//! [`predict_leaf`] the leaf index each row reaches in every tree, and
//! [`predict_distribution`] the fitted conditional distribution of a `dist:*`
//! model. All but the last return [`Predictions`]: a row-major
//! `[row][output]` buffer with its [`n_rows`](Predictions::n_rows) and
//! [`width`](Predictions::width) (`num_class` for `multi:softprob`, 1 for
//! `multi:softmax`'s class index, the tree count for leaves), read by row or
//! element, or as one flat slice via [`as_slice`](Predictions::as_slice) /
//! [`into_vec`](Predictions::into_vec).
//!
//! A row of feature values predicts without a matrix:
//! [`predict_row`](BoostedModel::predict_row) /
//! [`predict_margin_row`](BoostedModel::predict_margin_row) return a
//! single-output model's value, and
//! [`predict_row_into`](BoostedModel::predict_row_into) /
//! [`predict_margin_row_into`](BoostedModel::predict_margin_row_into) write a
//! row of [`prediction_width`](BoostedModel::prediction_width) (or
//! [`n_outputs`](BoostedModel::n_outputs)) values into a caller's buffer,
//! allocating nothing; [`predict_rows`](BoostedModel::predict_rows) /
//! [`predict_margin_rows`](BoostedModel::predict_margin_rows) read many rows
//! in place. Every path gives the same bits as [`BoostedModel::predict`] of
//! any matrix holding the rows: the trees add up in the same order, and the
//! objective's transform applies to each value (each row, for softmax and
//! the sorted quantiles) on its own, with XGBoost's scalar `expf`, sigmoid,
//! and softmax. [`transform_margin`](BoostedModel::transform_margin) and
//! [`transform_margins_into`](BoostedModel::transform_margins_into) apply that
//! transform to margins: `predict` is exactly the transform of
//! `predict_margin`.
//!
//! Read-only accessors describe the model in memory, so callers never parse
//! a model file: its [`trees`](BoostedModel::trees) (nodes, category sets,
//! leaf vectors, and linear leaves), their
//! [`tree_weights`](BoostedModel::tree_weights), its
//! [`num_class`](BoostedModel::num_class), a `gblinear` model's
//! [`linear`](BoostedModel::linear) weights, and the
//! [`shrinkage`](BoostedModel::shrinkage) record of a model trained with
//! model shrinkage.
//!
//! Every prediction method takes the boosting iterations it uses
//! ([`Iterations`], XGBoost's `iteration_range`): [`Iterations::Best`] for
//! the effective iterations (through `best_iteration` after early stopping,
//! else all), or a Rust range of iterations: `..` for all, `..n` for the
//! first `n`, `2..5`. Leaf, contribution, and interaction ranges must start
//! at 0 (`Best` always does). [`BoostedModel::slice`] cuts a sub-model out
//! of an iteration range with a step.
//!
//! A model trained with per-iteration model shrinkage
//! ([`model_shrink`](crate::config::TrainingParams::model_shrink),
//! SGLB) rescales its whole ensemble every iteration, so the first `k`
//! iterations of it are not a prefix of its trees. Its predictions repeat
//! training's shrink-then-add arithmetic, so they are bit for bit the
//! margins training computed; `..k` ranges and
//! [`BoostedModel::slice`]`(..k, 1)` rebuild the model after `k` iterations
//! exactly, and ranges starting later are refused.
//! [`BoostedModel::predict_virtual_ensembles`] predicts with several such
//! truncations at once and decomposes their spread into knowledge and data
//! uncertainty ([`uncertainty`]).
//!
//! # Explanation
//!
//! [`predict_contribs`] gives per-feature SHAP contributions
//! (QuadratureTreeSHAP) as [`Contributions`]: `n_features + 1` values per row
//! and output with the bias last ([`Contributions::get`],
//! [`Contributions::bias`]); [`predict_interactions`] gives SHAP interaction
//! values as [`Interactions`]: an `(n_features + 1)^2` matrix per row and
//! output ([`Interactions::get`], [`Interactions::at`]).
//! [`BoostedModel::feature_importance`] scores features by
//! [`ImportanceType`] (XGBoost's `importance_type`).
//!
//! # Persistence
//!
//! Four verbs take a [`ModelFormat`]: [`BoostedModel::encode`] /
//! [`decode`] convert to and from bytes, [`save`] / [`load`] to and from a
//! file. [`ModelFormat::detect`] names the format of unknown bytes.
//!
//! - **Native binary** ([`ModelFormat::Binary`]): a zstd-compressed,
//!   checksummed section table holding everything the model needs,
//!   including linear leaves and gblinear weights. Files written by 0.2.0
//!   and later keep loading in every later release.
//! - **Native JSON** ([`ModelFormat::Json`]): the same model as readable
//!   JSON.
//! - **XGBoost JSON and UBJSON** ([`ModelFormat::XgboostJson`],
//!   [`ModelFormat::XgboostUbjson`]): see
//!   [XGBoost interchange](#xgboost-interchange).
//! - **LightGBM import** ([`ModelFormat::LightgbmText`], decode and load
//!   only): LightGBM 4.x text models (`model.txt`); see
//!   [LightGBM import](#lightgbm-import).
//!
//! ```
//! use hessboost::prelude::*;
//!
//! # fn main() -> Result<()> {
//! let x: Vec<f32> = (0..40).map(|i| i as f32).collect();
//! let dtrain = DMatrix::from_dense(&x, 40, 1)?.with_labels(&x)?;
//! let model = train(&TrainingParams::default(), &dtrain, 3)?;
//!
//! let bytes = model.encode(ModelFormat::XgboostUbjson)?;
//! assert_eq!(ModelFormat::detect(&bytes), Some(ModelFormat::XgboostUbjson));
//! let restored = BoostedModel::decode(&bytes, ModelFormat::XgboostUbjson)?;
//! assert_eq!(
//!     restored.predict(&dtrain, Iterations::Best)?,
//!     model.predict(&dtrain, Iterations::Best)?,
//! );
//! # Ok(())
//! # }
//! ```
//! - **Compact:** [`BoostedModel::to_compact`] builds a bit-packed
//!   [`CompactModel`](compact::CompactModel) predicting bit-identical margins
//!   in a fraction of the size; see [`compact`].
//! - **Embedded:** an [`EmbeddedModel`] in a `static` compiles a model file
//!   into the program ([`include_bytes!`]) and decodes it on first use, so a
//!   static binary ships without a model file.
//!
//! # LightGBM import
//!
//! [`BoostedModel::decode`] with [`ModelFormat::LightgbmText`] reads the text model LightGBM 4.x
//! writes with `Booster.save_model` / `model_to_string` (format `v4`).
//! The imported model predicts, explains ([`predict_contribs`] matches
//! LightGBM's `pred_contrib`), slices, and saves natively like any other,
//! and exports to XGBoost JSON/UBJSON unless it has linear leaves. Import
//! only: there is no LightGBM export. LightGBM's `dump_model` JSON is not
//! read: LightGBM cannot load it itself (the text model is its interchange
//! format, which every booster writes), and its nesting follows tree depth,
//! which deep trees take past `serde_json`'s recursion limit.
//!
//! Predictions are LightGBM's for **dense inputs with missing values as
//! `NaN`** and **categorical values as non-negative codes** (what
//! [`DMatrix::with_feature_types`](crate::data::DMatrix::with_feature_types)
//! accepts). hessboost reads absent sparse entries as missing where
//! LightGBM reads them as `0`, and reads a categorical value below 0 as
//! category 0 where LightGBM sends it right like `NaN`: pass `NaN` for
//! LightGBM's negative "missing" categories. LightGBM's dense-matrix
//! prediction also zeroes inputs with `|x| <= 1e-35` before the trees, while
//! the import follows the trees' own rule (as LightGBM's CSR path does);
//! the two differ only for such inputs at a split whose threshold lies in
//! that band. Leaf values are rounded to `f32` and summed in `f32`
//! (LightGBM: `f64`); the parity fixtures agree within `1e-5` relative.
//!
//! ## Mapping
//!
//! - **Layout:** tree `t` is iteration `t / num_tree_per_iteration`, output
//!   `t % num_tree_per_iteration`, hessboost's layout with one tree per
//!   output. `init_score` / `boost_from_average` live in the first trees,
//!   so the intercepts are 0. LightGBM's internal node `i` is node `i`, its
//!   leaf `j` node `num_leaves - 1 + j` ([`predict_leaf`] reports node ids).
//!   Covers (`sum_hess`) are the node data counts LightGBM's TreeSHAP
//!   weighs paths by; gains are `split_gain`.
//! - **Numeric splits:** LightGBM sends `x <= threshold` left, comparing the
//!   `f64` threshold with the input widened to `f64`. For every `f32` `x`
//!   that holds exactly when `x < c`, where `c` is the smallest `f32` above
//!   the threshold, so `c` becomes the split condition. A threshold at or
//!   above `f32::MAX` (LightGBM writes `inf` for "all values") sends every
//!   finite value left, which no finite `c` does: the node's children swap
//!   and `c = -f32::MAX` sends every finite value to the former left child
//!   (hessboost's matrices hold no infinities).
//! - **Missing types:** `None` (`NaN` read as `0`): missing values go where
//!   `0` goes. `NaN`: missing values take the default direction. `Zero`:
//!   missing values and `|x| <= 1e-35` take the default direction, which one
//!   threshold expresses only when that band borders the half-line on the
//!   default side (zeros left with a threshold at or above `-1e-35`, or
//!   right with one at or below `1e-35`); such splits map with the band
//!   folded into `c`, and any other `zero_as_missing` split is refused.
//! - **Categorical splits:** the `cat_threshold` bitset becomes the left
//!   category set; `NaN` goes right, and like LightGBM a value's integer
//!   part is looked up. Each split must own its bitset, as LightGBM writes
//!   them, and categories must stay below `2^31`.
//! - **Linear leaves** (`linear_tree`): `leaf_const`, `leaf_features` and
//!   `leaf_coeff` become the tree's [`LinearLeaves`](crate::tree::LinearLeaves),
//!   in `f64`; a row with a `NaN` feature of the leaf's model gets the
//!   leaf's constant value, LightGBM's rule. As in LightGBM, such models
//!   have no SHAP values.
//! - **Objectives** map by prediction transform (the loss also sets what
//!   continued training in hessboost optimizes; objective parameters come
//!   from the file's `parameters:` section, else LightGBM's defaults):
//!
//! |LightGBM|hessboost|transform|
//! |---|---|---|
//! |`regression`, `fair`|`reg:squarederror`|identity|
//! |`regression_l1`, `mape`|`reg:absoluteerror`|identity|
//! |`huber` (`alpha` as `huber_slope`)|`reg:pseudohubererror`|identity|
//! |`quantile` (`alpha`)|`reg:quantileerror`|identity|
//! |`poisson` (`poisson_max_delta_step`)|`count:poisson`|`exp`|
//! |`gamma`|`reg:gamma`|`exp`|
//! |`tweedie` (`tweedie_variance_power`)|`reg:tweedie`|`exp`|
//! |`binary` with `sigmoid:1`|`binary:logistic`|sigmoid|
//! |`cross_entropy`|`reg:logistic`|sigmoid|
//! |`multiclass`|`multi:softprob`|softmax|
//! |`multiclassova` with `sigmoid:1`|`binary:logistic` over `num_class` targets|sigmoid per class|
//! |`lambdarank`, `rank_xendcg`|`rank:ndcg`|identity|
//!
//! Refused with a [`HessboostError::ModelFormat`] naming the reason:
//! `sigmoid` other than 1 (`binary`, `multiclassova`), `reg_sqrt`,
//! `cross_entropy_lambda` (`log(1 + exp(x))`), models without an objective
//! (custom objectives), random forests (`average_output`: LightGBM averages
//! their trees in predictions but sums them in raw scores and SHAP), the
//! `zero_as_missing` splits above, versions other than `v4`, and anything
//! malformed or unknown (header or tree keys, decision-type bits, child
//! references, `tree_sizes` that disagree with the tree blocks, as in a
//! file converted to CRLF line ends, which LightGBM's loader rejects too).
//!
//! # XGBoost interchange
//!
//! The XGBoost formats target the XGBoost 3.4.2 schema (identical to
//! 3.4.1's) in both of XGBoost's encodings: JSON text
//! ([`ModelFormat::XgboostJson`], XGBoost's `m.json`) and Universal Binary
//! JSON ([`ModelFormat::XgboostUbjson`], XGBoost's `m.ubj` and
//! `save_raw("ubj")`). Both encodings carry the same
//! document and share one model mapping; UBJSON only changes how it is
//! serialized (see [UBJSON encoding](#ubjson-encoding)).
//!
//! XGBoost serializes a booster as a nested JSON document:
//!
//! ```text
//! {"version": [3, 4, 2],
//!  "learner": {
//!    "gradient_booster": {
//!      "name": "gbtree",
//!      "model": {"trees": [ {..per-tree arrays..} ], "tree_info": [..],
//!                "gbtree_model_param": {..}, "weight_drop": [..]?}},
//!    "learner_model_param": {"base_score", "num_class", "num_feature", ..},
//!    "objective": {"name": .., ..parameter block..}}}
//! ```
//!
//! Each tree is stored as a set of parallel, node-indexed arrays rather than a
//! nested structure: `left_children`, `right_children`, `split_indices`,
//! `split_conditions`, `default_left`, `base_weights`, `sum_hessian` and
//! `loss_changes`. A node `i` is a **leaf** when `left_children[i] == -1`. Its
//! weight is carried in `split_conditions[i]` (and, redundantly,
//! `base_weights[i]`). Numeric internal nodes route `x[split_indices[i]] <
//! split_conditions[i]`, sending missing values in the `default_left[i]`
//! direction, matching the exact semantics of [`RegTree`].
//! Categorical internal nodes (`split_type[i] == 1`) carry their category set
//! in the tree's `categories` / `categories_nodes` / `categories_segments` /
//! `categories_sizes` arrays. Import requires the first five arrays with an
//! entry per node (integers where XGBoost writes integers); `base_weights`,
//! `sum_hessian` and `loss_changes` may be absent, but not partial. A
//! malformed array is refused rather than defaulted.
//!
//! ## Scope and caveats
//!
//! Import targets a `gbtree` booster with a scalar, multiclass, or
//! multi-target objective, with scalar-leaf trees (`one_output_per_tree`) or
//! vector-leaf trees (`multi_output_tree`, see
//! [Vector-leaf trees](#vector-leaf-trees)).
//! XGBoost saves `booster=dart` as `gbtree` plus a per-tree
//! `model.weight_drop` array; those weights become the model's DART tree
//! weights on import, and a model with non-unit tree weights writes them back
//! as `weight_drop` on export. A model trained with model shrinkage instead
//! exports plain `gbtree` trees with each tree's closed-form weight
//! multiplied into its leaves (as CatBoost bakes its shrinkage): a sum of
//! trees cannot repeat the per-iteration rounding of training, so the
//! exported margins match within `f32` rounding rather than bit for bit.
//! The imported model has no shrinkage record, so its iteration
//! ranges are tree prefixes. Other booster kinds (`gblinear`) yield a clear
//! [`HessboostError::ModelFormat`]. Numeric and categorical splits both
//! round-trip in either direction. Export refuses what XGBoost cannot load:
//! `gblinear` models, linear-leaf trees (`linear_tree`), custom objectives,
//! and the distributional `dist:*` objectives (which import refuses as well).
//!
//! ## Tree layout (`tree_info`)
//!
//! XGBoost tags each tree with its output group in `model.tree_info` and lays
//! trees out per boosting iteration as `[g0 × num_parallel_tree, g1 × ...]`,
//! with `iteration_indptr` (or, when absent, `num_parallel_tree × groups`
//! trees per iteration) marking iteration boundaries. `hessboost` stores the
//! same layout ([`BoostedModel::num_parallel_tree`] trees per output and
//! iteration), so boosted random forests keep their iteration structure in
//! both directions and export writes `num_parallel_tree`, `tree_info` and
//! `iteration_indptr` accordingly. Import regroups an iteration whose trees
//! are tagged out of group order, preserving each group's order (per-output
//! predictions are sums over a group's trees, so this is lossless); a model
//! whose groups have unequal tree counts within an iteration, or whose
//! iterations differ in size, is rejected.
//!
//! ## Vector-leaf trees
//!
//! A `multi_output_tree` model stores XGBoost's `MultiTargetTree` layout:
//! `tree_param.size_leaf_vector = K`, the shared split structure in the usual
//! node-indexed arrays, and every leaf's `K` weights in `leaf_weights`
//! (leaves in node order), each leaf's `right_children` entry holding its
//! index into that array. Leaves and categorical nodes carry XGBoost's
//! `DftBadValue` split condition, the root's parent is `-1`, and every tree
//! belongs to group 0 of `tree_info` (one tree per iteration). hessboost does
//! not retain internal node weights: export writes zeros for internal nodes'
//! `base_weights` (leaves repeat their vectors), which XGBoost does not read
//! for prediction.
//!
//! ## Objective parameters
//!
//! The objective's parameter block (`reg_loss_param.scale_pos_weight`,
//! `poisson_regression_param.max_delta_step`,
//! `tweedie_regression_param.tweedie_variance_power`,
//! `pseudo_huber_param.huber_slope`,
//! `lambdarank_param.lambdarank_num_pair_per_sample`,
//! `quantile_loss_param.quantile_alpha`,
//! `expectile_loss_param.expectile_alpha`,
//! `aft_loss_param.{aft_loss_distribution, aft_loss_distribution_scale}`)
//! becomes the parameters of the model's [`Objective`](crate::objective::Objective)
//! ([`ModelObjective::built_in`]), `scale_pos_weight` for every `RegLossObj`
//! objective (`reg:squarederror`, `reg:gamma`, and the logistic ones); absent
//! fields take XGBoost's defaults, and parameters the objective does not read
//! are dropped (e.g. `reg_loss_param.scale_pos_weight` of
//! `reg:squaredlogerror`). The alpha lists are XGBoost's array strings
//! (`"[0.1,0.5,0.9]"`, `(..)` also read); `reg:absoluteerror` and
//! `survival:cox` have no block. A value that does not parse, or an invalid
//! parameter of the objective (e.g. an empty or unsorted alpha list), is a
//! format error. An objective hessboost does not implement imports by its
//! name alone ([`ModelObjective::built_in`] is `None`): its model predicts
//! margins.
//!
//! ## `base_score`
//!
//! XGBoost 3.x stores the intercept as a vector string, `"[v0,v1,...]"`, with
//! one entry per output (or a single entry that applies to every output), in
//! whatever space its objective's `ProbToMargin` maps from: raw margin for
//! `reg:squarederror`, `binary:logitraw` and `binary:hinge`, but
//! **probability** space for objectives with a link function (`0.5` for
//! `binary:logistic`, not its logit). `hessboost` stores per-output
//! intercepts in **margin** space, so on **import** the vector is mapped
//! through the objective's inverse link
//! ([`Loss::probs_to_margins`])
//! and on **export** the margin row is mapped back with
//! [`Loss::margins_to_probs`]
//! (the forward transform, except for `binary:hinge` and
//! `reg:quantileerror`, whose transforms (threshold, sort) are not their
//! links). Multiclass objectives (and any objective that cannot be
//! reconstructed) pass the values through unchanged, as XGBoost does:
//! softmax's inverse link is the identity, while its forward transform
//! normalizes across classes.
//!
//! `learner_model_param.num_target` is XGBoost's output count
//! (`ObjFunction::Targets`): [`BoostedModel::n_outputs`] for non-multiclass
//! models — one per label column for a multi-target model
//! (`one_output_per_tree` on a label matrix, `num_class` 0), one per alpha for
//! `reg:quantileerror` / `reg:expectileerror`, whose
//! [`BoostedModel::n_targets`] is the single label column — and
//! [`BoostedModel::n_targets`] (1) for multiclass. Import checks it against
//! the rebuilt objective's output count. Tree groups in `tree_info` and
//! `base_score` entries are per output, laid out exactly like multiclass
//! groups.
//!
//! ## UBJSON encoding
//!
//! XGBoost keeps the node-indexed tree arrays as typed arrays and writes them
//! to UBJSON in optimized form (`[$<type>#L<count>` plus big-endian
//! payloads). Encoding [`ModelFormat::XgboostUbjson`] does the same with XGBoost's element
//! types: float32 for `split_conditions`, `base_weights`, `loss_changes`,
//! `sum_hessian` (and `leaf_weights`, gblinear `weights`); int32 for
//! `left_children`, `right_children`, `parents`, `categories`,
//! `categories_nodes` and `split_indices` (int64 when a tree's `num_feature`
//! exceeds the int32 range, as in XGBoost); uint8 for `default_left` and
//! `split_type`; int64 for `categories_segments` and `categories_sizes`. The
//! category container's int32 `feature_segments` / `sorted_idx` / `offsets`
//! and its per-column `values` follow XGBoost too. Every other array is a
//! counted generic array, numbers are float32 and integers the narrowest
//! width, again as XGBoost writes them. Decoding it accepts the
//! optimized and the plain UBJSON container forms alike.
//!
//! [`predict_margin`]: BoostedModel::predict_margin
//! [`predict_class`]: BoostedModel::predict_class
//! [`predict_leaf`]: BoostedModel::predict_leaf
//! [`predict_distribution`]: BoostedModel::predict_distribution
//! [`predict_contribs`]: BoostedModel::predict_contribs
//! [`predict_interactions`]: BoostedModel::predict_interactions
//! [`decode`]: BoostedModel::decode
//! [`save`]: BoostedModel::save
//! [`load`]: BoostedModel::load

mod categories;
pub mod compact;
pub(crate) mod container;
mod embed;
mod io;
mod lightgbm;
pub(crate) mod native;
mod objective;
mod predict;
mod predictions;
pub(crate) mod sections;
mod serde;
mod shap;
mod shrinkage;
mod slice;
mod transform;
mod ubjson;
pub mod uncertainty;
mod validate;
mod xgboost;

pub use shrinkage::Shrinkage;
pub(crate) use shrinkage::shrink_margins;

pub use embed::EmbeddedModel;
pub use io::ModelFormat;
pub use objective::ModelObjective;
pub use predict::Iterations;
use predict::RowBlock;
pub(crate) use predict::initial_margins;
pub use predictions::{Contributions, Interactions, Predictions};
pub(crate) use transform::Transform;
pub(crate) use validate::{check_objective_width, validate_prediction_data};

use self::serde::UncheckedBoostedModel;
use crate::data::{DMatrix, Rows};
use crate::ebm::EbmInfo;
use crate::error::{HessboostError, Result};
use crate::inference::BoulevardInfo;
use crate::objective::{Loss, LossContext};
use crate::tree::compact::CompactForest;
use crate::tree::{RegTree, scalar_tree_output};
use ::serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::{Bound, Range, RangeBounds};
use std::sync::Arc;
use std::sync::OnceLock;

/// The kind of feature-importance score to compute, mirroring XGBoost's
/// `importance_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImportanceType {
    /// Number of times a feature is used to split.
    Weight,
    /// Total loss reduction attributed to splits on a feature.
    TotalGain,
    /// Average loss reduction per split on a feature.
    Gain,
    /// Total Hessian (cover) of splits on a feature.
    TotalCover,
    /// Average Hessian (cover) per split on a feature.
    Cover,
}

/// A gradient-boosted tree ensemble.
///
/// Leaf weights already include the learning rate (shrinkage), so a raw margin
/// prediction for output `k` is `base_score[k] + Σ w_t · tree_t(x)` over the
/// output's trees, where the contribution weight `w_t` is `1` except for DART
/// and model-shrinkage models. The
/// stored `objective` name drives the prediction transform (e.g. the logistic
/// sigmoid).
///
/// Trees are laid out as in XGBoost: boosting iteration `i` owns the
/// [`trees_per_iteration`](Self::trees_per_iteration) trees starting at
/// `i * trees_per_iteration`, grouped by output (`num_parallel_tree` trees
/// for output 0, then output 1, ...). Tree `t` therefore feeds output
/// `(t / num_parallel_tree) % n_outputs`.
///
/// The serde implementations are the native JSON format
/// ([`ModelFormat::Json`] in [`encode`](Self::encode) /
/// [`decode`](Self::decode)).
/// Deserializing validates the model like every loader does and refuses an
/// inconsistent one, so a model deserialized through serde directly is as
/// safe to predict with and train on as a loaded one.
#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "UncheckedBoostedModel")]
#[allow(
    clippy::unsafe_derive_deserialize,
    reason = "the type's only unsafe code is the optional Metal backend's \
              buffer handling; every serialized field is plain data"
)]
pub struct BoostedModel {
    trees: Vec<RegTree>,
    /// Per-output intercept in margin space (length `n_outputs`).
    base_score: Vec<f32>,
    /// The objective, which drives the prediction transform and XGBoost
    /// export.
    objective: ModelObjective,
    /// The `max_delta_step` training used (XGBoost stores it with
    /// `count:poisson`), kept as stored.
    max_delta_step: f64,
    /// The stored `num_class`: a multiclass objective's class count, `0`
    /// otherwise (a custom model keeps what it was saved with).
    num_class: usize,
    /// Raw outputs per instance: `num_class` for multiclass objectives, the
    /// objective's own output count otherwise (custom objectives may have
    /// several).
    n_outputs: usize,
    /// Label columns per training row (`1` unless trained on a label matrix).
    n_targets: usize,
    n_features: usize,
    /// The best iteration index selected by early stopping, if any.
    best_iteration: Option<usize>,
    /// Per-tree contribution weights: [`TreeWeights::Unit`] when every tree
    /// weighs `1.0` (e.g. imported or sliced `gbtree` models), otherwise one
    /// weight per tree. The DART booster stores fractional weights here so
    /// dropped trees can be rescaled.
    tree_weights: TreeWeights,
    /// Trees grown per output in each boosting iteration (XGBoost
    /// `num_parallel_tree`; `1` for ordinary boosting, more for boosted
    /// random forests).
    num_parallel_tree: usize,
    /// Linear (coordinate-descent) booster parameters. `Some` only for
    /// `gblinear` models, in which case predictions come from the linear model
    /// and the `trees` vector is empty.
    linear: Option<LinearModel>,
    /// The per-iteration model shrinkage record (`model_shrink_rate`,
    /// posterior sampling): `Some` exactly when training shrank the model,
    /// whose tree weights and intercepts it then determines.
    shrinkage: Option<Shrinkage>,
    /// How a `booster = boulevard` model was trained, which its statistical
    /// inference reads ([`crate::inference`]); `None` for every other model.
    /// Predictions do not depend on it.
    boulevard: Option<BoulevardInfo>,
    /// The terms of a `booster = ebm` model ([`crate::ebm`]); `None` for
    /// every other model. Predictions do not depend on it.
    ebm: Option<EbmInfo>,
    /// Prediction layout of `trees` ([`CompactForest`]), derived lazily and
    /// never serialized. Reset whenever `trees` changes.
    compact: OnceLock<CompactForest>,
    /// The prediction transform of `objective` ([`Transform`]), derived
    /// lazily and never serialized. Reset whenever `objective` changes.
    transform: OnceLock<Transform>,
}

/// Per-tree contribution weights of a [`BoostedModel`]. Every format stores
/// [`Self::Unit`] as an empty list and reads an empty list back as it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TreeWeights {
    /// Every tree weighs `1.0`.
    Unit,
    /// One weight per tree (never empty).
    Explicit(Vec<f32>),
}

impl TreeWeights {
    /// The weights stored as `weights`: [`Self::Unit`] when empty.
    pub(crate) fn from_vec(weights: Vec<f32>) -> Self {
        if weights.is_empty() {
            Self::Unit
        } else {
            Self::Explicit(weights)
        }
    }

    /// The stored form: empty for [`Self::Unit`].
    pub(crate) fn as_slice(&self) -> &[f32] {
        match self {
            Self::Unit => &[],
            Self::Explicit(weights) => weights,
        }
    }

    /// The stored weights in order (none for [`Self::Unit`]).
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, f32> {
        self.as_slice().iter()
    }

    /// Weight of tree `i`.
    #[inline]
    fn get(&self, i: usize) -> f32 {
        match self {
            Self::Unit => 1.0,
            Self::Explicit(weights) => weights[i],
        }
    }

    /// Append the weight of a new tree to a forest of `n_trees` trees.
    fn push(&mut self, n_trees: usize, weight: f32) {
        self.materialize(n_trees);
        match self {
            Self::Unit => *self = Self::Explicit(vec![weight]),
            Self::Explicit(weights) => weights.push(weight),
        }
    }

    /// Store one weight for each of `n_trees` trees (`1.0` where absent).
    fn materialize(&mut self, n_trees: usize) {
        match self {
            Self::Unit if n_trees > 0 => *self = Self::Explicit(vec![1.0; n_trees]),
            Self::Unit => {}
            Self::Explicit(weights) => weights.resize(n_trees, 1.0),
        }
    }

    /// Multiply tree `i`'s weight by `factor` (a no-op for [`Self::Unit`]).
    fn scale(&mut self, i: usize, factor: f32) {
        if let Self::Explicit(weights) = self
            && let Some(weight) = weights.get_mut(i)
        {
            *weight *= factor;
        }
    }

    /// The weights of the trees `layers` selects, in order.
    fn select(&self, layers: impl Iterator<Item = Range<usize>>) -> Self {
        match self {
            Self::Unit => Self::Unit,
            Self::Explicit(weights) => Self::from_vec(
                layers
                    .flat_map(|layer| weights[layer].iter().copied())
                    .collect(),
            ),
        }
    }
}

/// The parameters of a linear (`gblinear`) booster, read through
/// [`BoostedModel::linear`]: a weight per feature and output plus a bias per
/// output, fit by coordinate descent. Output `k`'s margin of a row is
/// `base_score[k] + bias[k] + Σ_f weights[f * n_outputs + k] · x[f]` over the
/// row's present features, each product formed in `f64` and added in `f32`
/// in feature order.
///
/// The owning model checks its lengths and values when it is trained or
/// loaded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearModel {
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl LinearModel {
    /// Assemble a linear model from its fitted `weights` (`[feature][output]`)
    /// and per-output `bias`.
    pub(crate) fn new(weights: Vec<f32>, bias: Vec<f32>) -> Self {
        LinearModel { weights, bias }
    }

    /// The per-output bias (`n_outputs` values).
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// The weights, `n_features * n_outputs` values laid out
    /// `[feature][output]`: feature `f`'s weight for output `k` is
    /// `weights()[f * n_outputs + k]`.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
}

/// Invoke `f(feature, value)` for each present feature of `row` in feature
/// order. Shared by the gblinear training and prediction paths; arithmetic
/// stays at each call site to preserve exact conversion points.
pub(crate) fn for_each_present_value(rows: Rows<'_>, row: usize, mut f: impl FnMut(usize, f32)) {
    for feat in 0..rows.n_cols() {
        if let Some(x) = rows.get(row, feat) {
            f(feat, x);
        }
    }
}

/// Validated prologue for the TreeSHAP paths: dimensions, effective trees,
/// and initial margins.
struct AttributionPrologue<'a> {
    n: usize,
    k: usize,
    nf: usize,
    width: usize,
    trees: &'a [RegTree],
    initial: Vec<f32>,
}

/// The metadata a model is assembled with: what it predicts and how its trees
/// are laid out. Shared by training and the XGBoost and LightGBM importers.
pub(crate) struct ModelSpec {
    /// The objective's XGBoost name (`Loss::name`).
    pub(crate) objective: ModelObjective,
    /// The `max_delta_step` training used.
    pub(crate) max_delta_step: f64,
    /// `num_class` (`0` for non-multiclass objectives).
    pub(crate) num_class: usize,
    /// Raw outputs per instance (`Loss::n_outputs`).
    pub(crate) n_outputs: usize,
    /// Label columns per training row ([`DMatrix::n_targets`]).
    pub(crate) n_targets: usize,
    pub(crate) n_features: usize,
}

impl BoostedModel {
    pub(crate) fn new(base_score: Vec<f32>, spec: ModelSpec) -> Self {
        Self::from_parts(Vec::new(), Vec::new(), base_score, spec)
    }

    /// Attach a fitted linear (`gblinear`) booster. Predictions then come from
    /// the linear model instead of the (empty) tree ensemble.
    pub(crate) fn set_linear(&mut self, linear: LinearModel) {
        self.linear = Some(linear);
    }

    /// A copy of this model's metadata (intercepts, objective, layout) with
    /// `trees` instead of its own, every tree weighing `1`, no
    /// `best_iteration`, and no shrinkage record (updates refuse shrunk
    /// models): the result of an in-place data update
    /// ([`crate::training::online`]).
    pub(crate) fn with_trees(&self, trees: Vec<RegTree>) -> BoostedModel {
        BoostedModel {
            trees,
            base_score: self.base_score.clone(),
            objective: self.objective.clone(),
            max_delta_step: self.max_delta_step,
            num_class: self.num_class,
            n_outputs: self.n_outputs,
            n_targets: self.n_targets,
            n_features: self.n_features,
            best_iteration: None,
            tree_weights: TreeWeights::Unit,
            num_parallel_tree: self.num_parallel_tree,
            linear: None,
            shrinkage: None,
            boulevard: None,
            ebm: None,
            compact: OnceLock::new(),
            transform: self.transform.clone(),
        }
    }

    /// Append a tree with an explicit contribution weight (`1.0` for plain
    /// `gbtree`; DART stores fractional weights so dropped trees can be
    /// rescaled).
    pub(crate) fn push_tree_weighted(&mut self, tree: RegTree, weight: f32) {
        self.tree_weights.push(self.trees.len(), weight);
        self.trees.push(tree);
        self.compact = OnceLock::new();
    }

    /// The prediction layout of the ensemble, built on first use and dropped
    /// whenever a tree is appended.
    pub(crate) fn compact_forest(&self) -> &CompactForest {
        self.compact
            .get_or_init(|| CompactForest::from_trees(&self.trees))
    }

    /// The prediction transform of the model's objective, built on first use
    /// and dropped whenever the objective changes.
    pub(crate) fn transform(&self) -> &Transform {
        self.transform
            .get_or_init(|| Transform::of(&self.objective, self.max_delta_step, self.n_targets))
    }

    /// Contribution weight of tree `i` (`1.0` when weights are absent, e.g. for
    /// imported models or plain `gbtree`).
    #[inline]
    pub(crate) fn tree_weight(&self, i: usize) -> f32 {
        self.tree_weights.get(i)
    }

    /// Every tree's contribution weight, in [`Self::trees`] order (one per
    /// tree): `1.0` for plain `gbtree` models, imported ones, and slices of
    /// them; DART's dropout-rescaled weights; and for a model trained with
    /// model shrinkage the closed-form weights its [`shrinkage`](Self::shrinkage)
    /// record determines. A margin is `base_score + Σ weight · tree(x)`
    /// over the iterations' trees, except that a shrunk model's predictions
    /// repeat training's shrink-then-add recurrence instead (equal up to
    /// `f32` rounding; TreeSHAP and XGBoost export use these weights).
    pub fn tree_weights(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        (0..self.trees.len()).map(|t| self.tree_weights.get(t))
    }

    /// Whether tree `t` stores a weight vector per leaf (vector-leaf trees).
    #[cfg(any(
        all(target_os = "macos", feature = "metal"),
        feature = "wgpu",
        all(target_os = "linux", feature = "cuda")
    ))]
    pub(crate) fn tree_is_vector_leaf(&self, t: usize) -> bool {
        self.trees[t].is_vector_leaf()
    }

    /// Invoke `f(feature, output, weight * x)` for each present feature of
    /// `row` and each output of the linear (`gblinear`) model, in feature
    /// order. The product is computed in f64, which is exact for f32 operands;
    /// callers round to their accumulation precision. No-op for tree
    /// ensembles.
    pub(crate) fn for_each_linear_contribution(
        &self,
        rows: Rows<'_>,
        row: usize,
        mut f: impl FnMut(usize, usize, f64),
    ) {
        let Some(lm) = &self.linear else {
            return;
        };
        let k = self.n_outputs();
        for_each_present_value(rows, row, |feat, x| {
            for c in 0..k {
                f(feat, c, f64::from(lm.weights[feat * k + c]) * f64::from(x));
            }
        });
    }

    /// Multiply tree `i`'s contribution weight by `factor` (DART rescaling).
    pub(crate) fn scale_tree_weight(&mut self, i: usize, factor: f32) {
        self.tree_weights.scale(i, factor);
    }

    /// Raw margin predictions that exclude the trees marked `true` in `dropped`
    /// (indexed by tree id). Used by the DART training loop to compute a round's
    /// gradients from the ensemble minus its dropout set. Output is laid out
    /// `[instance][output]`.
    pub(crate) fn predict_margin_dropout(&self, data: &DMatrix, dropped: &[bool]) -> Vec<f32> {
        let mut out = self.initial_margins(data.into());
        // Dropped trees contribute a zero weight, leaving per-cell accumulation
        // in ascending tree order (a `0.0` addend is a no-op).
        let weight = |ti: usize| {
            if dropped.get(ti).copied().unwrap_or(false) {
                0.0
            } else {
                self.tree_weight(ti)
            }
        };
        self.accumulate_forest(data.into(), &mut out, 0..self.trees.len(), weight);
        out
    }

    pub(crate) fn set_best_iteration(&mut self, it: Option<usize>) {
        self.best_iteration = it;
    }

    /// Record (or clear) how the model was trained by `booster = boulevard`.
    pub(crate) fn set_boulevard(&mut self, info: Option<BoulevardInfo>) {
        self.boulevard = info;
    }

    /// How this model was trained by `booster = boulevard`,
    /// which [`crate::inference::BoulevardInference`] reads; `None` for every
    /// other model, including a Boulevard model's [`slice`](Self::slice)s
    /// and its XGBoost-format or compact exports (which predict the same
    /// but are no longer Boulevard fits).
    pub fn boulevard(&self) -> Option<&BoulevardInfo> {
        self.boulevard.as_ref()
    }

    /// Record (or clear) the terms of a `booster = ebm` model.
    pub(crate) fn set_ebm(&mut self, info: Option<EbmInfo>) {
        self.ebm = info;
    }

    /// The terms of a `booster = ebm` model, which
    /// [`crate::ebm::shape_functions`] and
    /// [`crate::inference::EbmInference`] read; `None` for every other
    /// model, including an EBM's [`slice`](Self::slice)s and its
    /// XGBoost-format or compact exports (which predict the same but no
    /// longer know their terms).
    pub fn ebm(&self) -> Option<&EbmInfo> {
        self.ebm.as_ref()
    }

    /// Reassemble a model from its constituent parts. Used by the XGBoost and
    /// LightGBM importers, which build trees and metadata externally. `tree_weights`
    /// is either empty (every tree weighs `1.0`) or holds one DART weight per
    /// tree; [`BoostedModel::validate_structure`] enforces the length.
    pub(crate) fn from_parts(
        trees: Vec<RegTree>,
        tree_weights: Vec<f32>,
        base_score: Vec<f32>,
        spec: ModelSpec,
    ) -> Self {
        BoostedModel {
            trees,
            base_score,
            objective: spec.objective,
            max_delta_step: spec.max_delta_step,
            num_class: spec.num_class,
            n_outputs: spec.n_outputs,
            n_targets: spec.n_targets,
            n_features: spec.n_features,
            best_iteration: None,
            tree_weights: TreeWeights::from_vec(tree_weights),
            num_parallel_tree: 1,
            linear: None,
            shrinkage: None,
            boulevard: None,
            ebm: None,
            compact: OnceLock::new(),
            transform: OnceLock::new(),
        }
    }

    /// The configured `num_class`: a multiclass objective's class count
    /// (equal to [`Self::n_outputs`]), `0` for every other objective (a model
    /// with a custom objective keeps what it was trained or saved with).
    pub fn num_class(&self) -> usize {
        self.num_class
    }

    /// The `max_delta_step` training used (`0` when unbounded).
    pub(crate) fn max_delta_step(&self) -> f64 {
        self.max_delta_step
    }

    /// Number of raw outputs (margins) per row: `num_class` for multiclass,
    /// one per alpha for a `reg:quantileerror` or `reg:expectileerror` alpha
    /// list, one per natural parameter for `dist:*`,
    /// [`BoostedModel::n_targets`] for a label matrix, `1` for every other
    /// built-in objective, and what a custom objective declares.
    #[inline]
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// Number of label columns (targets) per row of the training data: `1`
    /// unless the model was trained on [`DMatrix::with_label_matrix`] labels.
    #[inline]
    pub fn n_targets(&self) -> usize {
        self.n_targets
    }

    /// Whether the ensemble consists of vector-leaf trees
    /// (`multi_strategy = multi_output_tree`): each tree predicts every
    /// output at once instead of one output per tree.
    pub fn has_vector_leaves(&self) -> bool {
        self.trees.first().is_some_and(RegTree::is_vector_leaf)
    }

    /// The first output's intercept (global bias) in margin space.
    ///
    /// Scalar-output models have exactly one value. For multiclass models use
    /// [`Self::base_scores`] to access every per-class intercept.
    pub fn base_score(&self) -> f32 {
        self.base_score[0]
    }

    /// Per-output intercepts in margin space, one per class/output.
    pub fn base_scores(&self) -> &[f32] {
        &self.base_score
    }

    /// The objective this model was trained (or imported) with.
    pub fn objective(&self) -> &ModelObjective {
        &self.objective
    }

    /// The best iteration chosen by early stopping, if applicable.
    pub fn best_iteration(&self) -> Option<usize> {
        self.best_iteration
    }

    /// Compute feature importance of the requested type, returned as a map from
    /// feature index to score in ascending feature order (features that never
    /// split are absent).
    pub fn feature_importance(&self, kind: ImportanceType) -> BTreeMap<usize, f64> {
        let value = |node: &crate::tree::Node| match kind {
            ImportanceType::Weight => 1.0,
            ImportanceType::Cover | ImportanceType::TotalCover => f64::from(node.sum_hess),
            ImportanceType::Gain | ImportanceType::TotalGain => f64::from(node.split_gain),
        };
        // Per feature: the total of `value` and the split count.
        let mut totals: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
        for tree in &self.trees {
            for node in tree.nodes() {
                if node.is_leaf() {
                    continue;
                }
                let (total, count) = totals.entry(node.split_feature as usize).or_default();
                *total += value(node);
                *count += 1.0;
            }
        }
        // Divide a total by the split count to get the per-split average.
        let average = matches!(kind, ImportanceType::Cover | ImportanceType::Gain);
        totals
            .into_iter()
            .map(|(f, (total, count))| (f, if average { total / count } else { total }))
            .collect()
    }

    /// Lay this model out for GPU batch prediction on Metal. Without the
    /// `metal` feature on macOS (the only supported platform today), this
    /// always returns an error; see
    /// [`backend`](crate::backend) for the accelerated path.
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    pub fn to_gpu(&self) -> Result<crate::backend::metal::GpuModel> {
        Err(HessboostError::gpu(
            "GPU prediction requires the `metal` feature on macOS",
        ))
    }

    /// Lay this model out for GPU batch prediction through wgpu. Without
    /// the `wgpu` feature this always returns an error; see
    /// [`backend`](crate::backend) for the accelerated path.
    #[cfg(not(feature = "wgpu"))]
    pub fn to_wgpu(&self) -> Result<crate::backend::wgpu::GpuModel> {
        Err(HessboostError::gpu(
            "wgpu prediction requires the `wgpu` feature",
        ))
    }

    /// Lay this model out for GPU batch prediction on CUDA device `ordinal`.
    /// Without the `cuda` feature on Linux this always returns an error;
    /// see [`backend::cuda`](crate::backend::cuda) for the accelerated path.
    #[cfg(not(all(target_os = "linux", feature = "cuda")))]
    pub fn to_cuda(&self, _ordinal: usize) -> Result<crate::backend::cuda::GpuModel> {
        Err(HessboostError::gpu(
            "CUDA prediction requires the `cuda` feature on Linux",
        ))
    }

    /// Read-only access to the trees (e.g. for serialization or SHAP).
    pub fn trees(&self) -> &[RegTree] {
        &self.trees
    }

    /// Number of features the model expects.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// The fitted weights and biases of a `gblinear` model, which predicts
    /// from them alone (it has no trees); `None` for every tree model.
    pub fn linear(&self) -> Option<&LinearModel> {
        self.linear.as_ref()
    }

    pub(crate) fn has_non_unit_tree_weights(&self) -> bool {
        (0..self.trees.len()).any(|i| self.tree_weight(i) != 1.0)
    }

    /// Margin buffer for `rows`: the per-output intercepts broadcast to every
    /// row, overridden by the dataset's per-instance `base_margin` when
    /// present (one value per row, or one per row and output). Shared by
    /// prediction, TreeSHAP, and the training margin caches.
    pub(crate) fn initial_margins(&self, rows: Rows<'_>) -> Vec<f32> {
        initial_margins(&self.base_score, rows)
    }

    /// Validated prologue for the TreeSHAP paths over the iterations in
    /// `iterations`, which must start at iteration `0` (as in XGBoost).
    ///
    /// Linear-leaf trees are refused: TreeSHAP attributes constant leaf values
    /// along decision paths and has no defined extension to leaves whose
    /// output varies with the row (LightGBM refuses SHAP for linear trees as
    /// well).
    fn attribution_prologue(
        &self,
        data: &DMatrix,
        iterations: Iterations,
        what: &str,
    ) -> Result<AttributionPrologue<'_>> {
        self.validate_prediction_data(data)?;
        let end = self.prefix_trees(iterations, what)?;
        self.refuse_partial_shrunk_range(&(0..end), what)?;
        let trees = &self.trees[..end];
        if trees.iter().any(|tree| tree.linear_leaves().is_some()) {
            return Err(HessboostError::incompatible_model(
                "linear_tree",
                "SHAP contributions and interactions are not defined for models with linear leaves",
            ));
        }
        let nf = self.n_features;
        Ok(AttributionPrologue {
            n: data.n_rows(),
            k: self.n_outputs(),
            nf,
            width: nf + 1,
            trees,
            initial: self.initial_margins(data.into()),
        })
    }

    /// Set the number of trees per output in each iteration. Only valid while
    /// the layout still holds whole iterations of that size
    /// ([`BoostedModel::validate_structure`] checks it).
    pub(crate) fn set_num_parallel_tree(&mut self, num_parallel_tree: usize) {
        self.num_parallel_tree = num_parallel_tree;
    }

    /// Replace the per-output intercepts (margin space).
    pub(crate) fn set_base_scores(&mut self, base_score: Vec<f32>) {
        self.base_score = base_score;
    }

    /// Replace the objective and the `max_delta_step` (continued training
    /// adopts the new configuration's parameters, as XGBoost's `set_param`
    /// does).
    pub(crate) fn set_objective(&mut self, objective: ModelObjective, max_delta_step: f64) {
        self.objective = objective;
        self.max_delta_step = max_delta_step;
        self.transform = OnceLock::new();
    }

    /// Give every tree an explicit contribution weight (`1.0` where absent),
    /// so appended trees' weights line up with their tree ids.
    pub(crate) fn materialize_tree_weights(&mut self) {
        self.tree_weights.materialize(self.trees.len());
    }

    /// Remove and return every tree (dropping the contribution weights),
    /// leaving an empty ensemble with the same metadata. Used by
    /// `process_type=update`, which re-appends the refreshed trees.
    pub(crate) fn take_trees(&mut self) -> Vec<RegTree> {
        self.tree_weights = TreeWeights::Unit;
        self.compact = OnceLock::new();
        std::mem::take(&mut self.trees)
    }

    /// Multiply every tree's leaves by `factor` (Boulevard's final `1/B`
    /// averaging scale).
    pub(crate) fn scale_all_leaves(&mut self, factor: f32) {
        for tree in self.trees_mut() {
            tree.scale_leaves(factor);
        }
    }

    /// The trees, for in-place edits of their values (the prediction layout
    /// is rebuilt on next use).
    pub(crate) fn trees_mut(&mut self) -> &mut [RegTree] {
        self.compact = OnceLock::new();
        &mut self.trees
    }

    /// Number of trees (boosting rounds × outputs × `num_parallel_tree`).
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Trees grown per output in each boosting iteration (XGBoost
    /// `num_parallel_tree`).
    #[inline]
    pub fn num_parallel_tree(&self) -> usize {
        self.num_parallel_tree
    }

    /// Trees per boosting iteration: `n_outputs × num_parallel_tree`, or
    /// `num_parallel_tree` vector-leaf trees (each feeds every output).
    #[inline]
    pub fn trees_per_iteration(&self) -> usize {
        if self.has_vector_leaves() {
            self.num_parallel_tree
        } else {
            self.n_outputs * self.num_parallel_tree
        }
    }

    /// Check that training `num_parallel_tree` trees per output for
    /// `n_outputs` outputs keeps the per-iteration tree count
    /// (`n_outputs × num_parallel_tree`) within `usize`, so every
    /// iteration-indexing product stays defined. The trainer calls it before
    /// assembling the model, as the loader's structural checks do for saved
    /// models.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] (`num_parallel_tree`) when the
    /// product overflows.
    pub(crate) fn check_iteration_size(n_outputs: usize, num_parallel_tree: usize) -> Result<()> {
        if n_outputs.checked_mul(num_parallel_tree).is_none() {
            return Err(HessboostError::invalid_param(
                "num_parallel_tree",
                format!(
                    "{num_parallel_tree} parallel trees for {n_outputs} outputs overflow the \
                     trees per iteration"
                ),
            ));
        }
        Ok(())
    }

    /// The output tree `t` contributes to (XGBoost `tree_info[t]`): `0` for
    /// every vector-leaf tree, whose leaves carry all outputs.
    #[inline]
    pub(crate) fn tree_output(&self, t: usize) -> usize {
        if self.has_vector_leaves() {
            0
        } else {
            scalar_tree_output(t, self.num_parallel_tree, self.n_outputs)
        }
    }

    /// Number of boosting iterations (`num_trees / trees_per_iteration`;
    /// XGBoost `num_boosted_rounds`).
    pub fn num_boost_rounds(&self) -> usize {
        self.trees.len() / self.trees_per_iteration()
    }

    /// Number of trees in the effective iterations (`[0, best_iteration +
    /// 1)` after early stopping, else all trees).
    pub(crate) fn effective_num_trees(&self) -> usize {
        self.best_iteration.map_or(self.trees.len(), |it| {
            ((it + 1) * self.trees_per_iteration()).min(self.trees.len())
        })
    }

    /// The bounds of `iterations`: [`Iterations::Best`] is
    /// `..best_iteration + 1` when early stopping selected an iteration,
    /// else every iteration (`..`, which is also the only range a
    /// `gblinear` model accepts).
    fn iteration_bounds(&self, iterations: Iterations) -> (Bound<usize>, Bound<usize>) {
        match iterations {
            Iterations::Best => {
                let end = self
                    .best_iteration
                    .map_or(Bound::Unbounded, |it| Bound::Excluded(it + 1));
                (Bound::Unbounded, end)
            }
            Iterations::Range { start, end } => (start, end),
        }
    }

    /// Resolve `iterations` against the model's iteration count.
    pub(crate) fn resolve_iterations(
        &self,
        iterations: Iterations,
        param: &'static str,
    ) -> Result<Range<usize>> {
        let iterations = self.iteration_bounds(iterations);
        if self.linear.is_some() {
            // No iterations to select: only the whole model (`..`) is a range.
            let whole = matches!(
                iterations.start_bound(),
                Bound::Unbounded | Bound::Included(0)
            ) && iterations.end_bound() == Bound::Unbounded;
            if !whole {
                return Err(HessboostError::incompatible_model(
                    param,
                    "gblinear models have no boosting iterations to select; pass `..`",
                ));
            }
            return Ok(0..0);
        }
        let rounds = self.num_boost_rounds();
        let overflow = || HessboostError::invalid_param(param, "range bound overflows");
        let begin = match iterations.start_bound() {
            Bound::Included(&b) => b,
            Bound::Excluded(&b) => b.checked_add(1).ok_or_else(overflow)?,
            Bound::Unbounded => 0,
        };
        let end = match iterations.end_bound() {
            Bound::Included(&e) => e.checked_add(1).ok_or_else(overflow)?,
            Bound::Excluded(&e) => e,
            Bound::Unbounded => rounds,
        };
        if end > rounds {
            return Err(HessboostError::incompatible_model(
                param,
                format!("{begin}..{end} is out of range for a model with {rounds} iterations"),
            ));
        }
        if begin > end {
            return Err(HessboostError::invalid_param(
                param,
                format!("{begin}..{end} is an inverted range"),
            ));
        }
        Ok(begin..end)
    }

    /// Tree ids covered by the (resolved) boosting `iterations`. Each
    /// iteration contributes its whole forest (every output and parallel
    /// tree).
    pub(crate) fn iteration_trees(&self, iterations: Range<usize>) -> Range<usize> {
        let per = self.trees_per_iteration();
        iterations.start * per..iterations.end * per
    }

    /// The trees of `iterations` for the attribution and leaf predictions,
    /// which (as in XGBoost) only accept ranges starting at iteration `0`;
    /// slice the model for a later start.
    fn prefix_trees(&self, iterations: Iterations, what: &str) -> Result<usize> {
        let trees = self.iteration_trees(self.resolve_iterations(iterations, "iterations")?);
        if trees.start != 0 {
            return Err(HessboostError::invalid_param(
                "iterations",
                format!(
                    "{what} supports only ranges starting at iteration 0; slice the model instead"
                ),
            ));
        }
        Ok(trees.end)
    }

    /// Record the shrinkage training applied (one coefficient per
    /// iteration, and the intercepts before it) and derive the tree weights
    /// and intercepts of the full model from it.
    pub(crate) fn set_shrinkage(&mut self, shrinkage: Shrinkage) {
        let (tree_weights, base_score) =
            shrinkage.scaling(self.num_boost_rounds(), self.trees_per_iteration());
        self.tree_weights = TreeWeights::from_vec(tree_weights);
        self.base_score = base_score;
        self.shrinkage = Some(shrinkage);
    }

    /// Cut a shrunk model back to its first `k` iterations (early stopping's
    /// best model, as CatBoost's `use_best_model` does); a no-op otherwise.
    pub(crate) fn truncate_shrunk(&mut self, k: usize) {
        if let Some(shrinkage) = &self.shrinkage
            && k < self.num_boost_rounds()
        {
            *self = self.shrunk_prefix(shrinkage, k);
        }
    }

    /// The per-iteration shrinkage record of a model trained with model
    /// shrinkage ([`model_shrink`](crate::config::TrainingParams::model_shrink),
    /// posterior sampling), from which its predictions are computed exactly as
    /// training computed its margins; `None` for every other model.
    pub fn shrinkage(&self) -> Option<&Shrinkage> {
        self.shrinkage.as_ref()
    }

    /// The iteration range the attribution predictions use by default (the
    /// effective iterations, like [`Self::predict_margin`]).
    pub(crate) fn validate_prediction_data(&self, data: &DMatrix) -> Result<()> {
        validate_prediction_data(self.n_features, self.n_outputs(), data)
    }

    /// The loss of the model's built-in objective (`None` for another
    /// objective, whose predictions are margins).
    pub(crate) fn rebuild_objective(&self) -> Option<Result<Arc<dyn Loss>>> {
        rebuild_objective(&self.objective, self.max_delta_step, self.n_targets)
    }
}

/// The loss of `objective` for a model with `n_targets` label columns that
/// trained with `max_delta_step`: `None` for an objective the crate does not
/// implement (custom losses).
fn rebuild_objective(
    objective: &ModelObjective,
    max_delta_step: f64,
    n_targets: usize,
) -> Option<Result<Arc<dyn Loss>>> {
    objective.built_in().map(|objective| {
        objective.build_loss(&LossContext {
            n_targets,
            max_delta_step,
            shared_tree_seed: None,
            // Keys training draws only (XE-NDCG); predictions never read it.
            seed: 0,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::BoostedModel;
    use crate::config::TrainingParams;
    use crate::data::DMatrix;
    use crate::error::HessboostError;
    use crate::model::Iterations;
    use crate::model::ModelFormat;
    use crate::objective::{Objective, RegLoss};
    use crate::test_support::labeled_dense;
    use crate::training::train;

    /// A two-output forest of `2^63` parallel trees overflows the trees per
    /// iteration: training refuses it as a parameter error instead of
    /// overflowing (or dividing by zero) in its iteration arithmetic, even
    /// for zero rounds.
    #[test]
    fn training_rejects_overflowing_iteration_size() {
        let x = [0.0f32, 1.0, 2.0, 3.0];
        let y = [0.0f32, 1.0, 1.0, 0.0, 2.0, 3.0, 3.0, 2.0];
        let d = DMatrix::from_dense(&x, 4, 1)
            .unwrap()
            .with_label_matrix(&y, 2)
            .unwrap();
        // Set directly, unvalidated: `train` itself must refuse the layout,
        // whatever the builder's own bounds.
        let params = TrainingParams {
            num_parallel_tree: 1usize << (usize::BITS - 1),
            ..TrainingParams::default()
        };
        for rounds in [0, 1] {
            assert!(matches!(
                train(&params, &d, rounds),
                Err(HessboostError::InvalidParameter { name, .. }) if name == "num_parallel_tree"
            ));
        }
    }

    /// Only unknown (custom) objective names skip the objective check: a
    /// built-in objective that cannot be rebuilt from the stored
    /// configuration (here a single-target objective with two label columns)
    /// is a format error, not a silently untransformed model.
    #[test]
    fn loading_propagates_invalid_builtin_objective() {
        let d = labeled_dense(&[0.0, 1.0], 2, 1, &[0.0, 1.0]);
        let model = train(&TrainingParams::default(), &d, 1).unwrap();
        let mut value: serde_json::Value =
            serde_json::from_slice(&model.encode(ModelFormat::Json).unwrap()).unwrap();
        value["n_targets"] = 2.into();
        value["objective"] = "count:poisson".into();
        assert!(matches!(
            BoostedModel::decode(value.to_string(), ModelFormat::Json),
            Err(HessboostError::ModelFormat(_))
        ));
        value["objective"] = "my:custom".into();
        let custom = BoostedModel::decode(value.to_string(), ModelFormat::Json).unwrap();
        assert_eq!(custom.objective().name(), "my:custom");
        assert_eq!(custom.objective().built_in(), None);
    }

    /// Non-finite gblinear parameters would save as JSON `null` (and
    /// predict NaN): the binary reader refuses them like non-finite trees.
    #[test]
    fn loading_refuses_non_finite_linear_parameters() {
        let x: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let d = labeled_dense(&x, 10, 2, &[1.0; 10]);
        let params = TrainingParams::builder()
            .booster(crate::config::BoosterKind::GbLinear)
            .build()
            .unwrap();
        let model = train(&params, &d, 2).unwrap();
        assert!(
            BoostedModel::decode(
                model.encode(ModelFormat::Binary).unwrap(),
                ModelFormat::Binary
            )
            .is_ok()
        );
        for (weight, bias) in [(f32::INFINITY, 0.0), (0.0, f32::NAN)] {
            let mut corrupt = model.clone();
            let linear = corrupt.linear.as_mut().unwrap();
            linear.weights[0] = weight;
            linear.bias[0] = bias;
            assert!(matches!(
                BoostedModel::decode(
                    corrupt.encode(ModelFormat::Binary).unwrap(),
                    ModelFormat::Binary
                ),
                Err(HessboostError::ModelFormat(_))
            ));
        }
    }

    /// A gblinear model and its native JSON document.
    fn gblinear_doc() -> (BoostedModel, serde_json::Value) {
        let x: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let d = labeled_dense(&x, 10, 2, &[1.0; 10]);
        let params = TrainingParams::builder()
            .booster(crate::config::BoosterKind::GbLinear)
            .build()
            .unwrap();
        let model = train(&params, &d, 2).unwrap();
        let doc = serde_json::from_slice(&model.encode(ModelFormat::Json).unwrap()).unwrap();
        (model, doc)
    }

    /// Deserializing through serde directly validates like decoding native JSON: an
    /// empty gblinear bias used to load and panic in prediction, and a cyclic
    /// tree used to load and loop forever in traversal.
    #[test]
    fn serde_deserialization_validates_the_model() {
        let (model, mut doc) = gblinear_doc();
        let valid: BoostedModel = serde_json::from_value(doc.clone()).unwrap();
        let d = DMatrix::from_dense(&[1.0, 2.0], 1, 2).unwrap();
        assert_eq!(
            valid.predict(&d, Iterations::Best).unwrap(),
            model.predict(&d, Iterations::Best).unwrap()
        );
        doc["linear"]["bias"] = serde_json::json!([]);
        assert!(serde_json::from_value::<BoostedModel>(doc.clone()).is_err());
        assert!(matches!(
            BoostedModel::decode(doc.to_string(), ModelFormat::Json),
            Err(HessboostError::ModelFormat(_))
        ));

        let d = labeled_dense(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 0.0, 1.0, 1.0]);
        let model = train(&TrainingParams::default(), &d, 1).unwrap();
        let mut doc: serde_json::Value =
            serde_json::from_slice(&model.encode(ModelFormat::Json).unwrap()).unwrap();
        assert!(doc["trees"][0]["nodes"].as_array().unwrap().len() > 1);
        doc["trees"][0]["nodes"][0]["left"] = 0.into();
        assert!(serde_json::from_value::<BoostedModel>(doc.clone()).is_err());
        assert!(matches!(
            BoostedModel::decode(doc.to_string(), ModelFormat::Json),
            Err(HessboostError::ModelFormat(_))
        ));
    }

    /// gblinear has no boosting iterations, so a stored `best_iteration`
    /// (which training never writes for it) is refused at load instead of
    /// making every plain prediction fail on its iteration range.
    #[test]
    fn gblinear_refuses_best_iteration() {
        let (model, mut doc) = gblinear_doc();
        doc["best_iteration"] = 0.into();
        assert!(matches!(
            BoostedModel::decode(doc.to_string(), ModelFormat::Json),
            Err(HessboostError::ModelFormat(_))
        ));
        let mut stopped = model.clone();
        stopped.set_best_iteration(Some(0));
        assert!(matches!(
            BoostedModel::decode(
                stopped.encode(ModelFormat::Binary).unwrap(),
                ModelFormat::Binary
            ),
            Err(HessboostError::ModelFormat(_))
        ));
    }

    /// `num_class` counts multiclass classes (XGBoost refuses it with
    /// several targets), so a two-target `binary:logistic` model with
    /// `num_class = 2` is neither trained nor loaded.
    #[test]
    fn num_class_applies_only_to_multiclass_objectives() {
        let x = [0.0f32, 1.0, 2.0, 3.0];
        let y = [0.0f32, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let d = DMatrix::from_dense(&x, 4, 1)
            .unwrap()
            .with_label_matrix(&y, 2)
            .unwrap();
        assert!(matches!(
            TrainingParams::from_xgboost([
                ("objective", serde_json::json!("binary:logistic")),
                ("num_class", serde_json::json!(2)),
            ]),
            Err(HessboostError::InvalidParameter { name, .. }) if name == "num_class"
        ));
        let params = TrainingParams::builder()
            .objective(Objective::BinaryLogistic(RegLoss::default()))
            .build()
            .unwrap();
        let model = train(&params, &d, 1).unwrap();
        let mut doc: serde_json::Value =
            serde_json::from_slice(&model.encode(ModelFormat::Json).unwrap()).unwrap();
        assert!(BoostedModel::decode(doc.to_string(), ModelFormat::Json).is_ok());
        doc["num_class"] = 2.into();
        assert!(matches!(
            BoostedModel::decode(doc.to_string(), ModelFormat::Json),
            Err(HessboostError::ModelFormat(_))
        ));
    }
}
