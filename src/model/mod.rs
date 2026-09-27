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
//! model. Multi-output predictions are row-major `[row][output]`.
//!
//! Each has a `_range` variant ([`predict_range`], [`predict_margin_range`],
//! [`predict_leaf_range`], [`predict_contribs_range`],
//! [`predict_interactions_range`], [`predict_distribution_range`]) taking
//! XGBoost's `iteration_range` as a Rust range of boosting iterations: `..`
//! for all, `..n` for the first `n`, `2..5`. Leaf, contribution, and
//! interaction ranges must start at 0. [`BoostedModel::slice`] cuts a
//! sub-model out of an iteration range with a step.
//!
//! A model trained with per-iteration model shrinkage
//! ([`model_shrink`](crate::config::TrainingParams::model_shrink),
//! SGLB) rescales its whole ensemble every iteration, so the first `k`
//! iterations of it are not a prefix of its trees: `..k` ranges and
//! [`BoostedModel::slice`]`(..k, 1)` rebuild the model after `k` iterations
//! exactly, and ranges starting later are refused.
//! [`BoostedModel::predict_virtual_ensembles`] predicts with several such
//! truncations at once and decomposes their spread into knowledge and data
//! uncertainty ([`uncertainty`]).
//!
//! # Explanation
//!
//! [`predict_contribs`] gives per-feature SHAP contributions (QuadratureTreeSHAP),
//! `[row][n_features + 1]` with the bias last; [`predict_interactions`] gives
//! SHAP interaction values, `[row][(n_features + 1)^2]`. Multi-output models
//! add an output axis. [`BoostedModel::feature_importance`] scores features by
//! [`ImportanceType`] (XGBoost's `importance_type`).
//!
//! # Persistence
//!
//! - **Native binary:** [`BoostedModel::to_bytes`] / [`from_bytes`],
//!   [`save_binary`] / [`load_binary`]: a zstd-compressed, checksummed
//!   section table holding everything the model needs, including linear
//!   leaves and gblinear weights. Files written by 0.2.0 and later keep
//!   loading in every later release.
//! - **Native JSON:** [`BoostedModel::to_json`] / [`from_json`],
//!   [`save_json`] / [`load_json`]: the same model as readable JSON.
//! - **XGBoost JSON and UBJSON:** [`BoostedModel::to_xgboost_json`] /
//!   [`from_xgboost_json`], [`save_xgboost_json`] / [`load_xgboost_json`],
//!   and the `_xgboost_ubjson` counterparts ([`to_xgboost_ubjson`],
//!   [`from_xgboost_ubjson`], [`save_xgboost_ubjson`],
//!   [`load_xgboost_ubjson`]); see [XGBoost interchange](#xgboost-interchange).
//! - **LightGBM import:** [`BoostedModel::from_lightgbm_text`] /
//!   [`BoostedModel::load_lightgbm_text`] read LightGBM 4.x text models
//!   (`model.txt`); see [LightGBM import](#lightgbm-import).
//! - **Compact:** [`BoostedModel::to_compact`] builds a bit-packed
//!   [`CompactModel`](compact::CompactModel) predicting bit-identical margins
//!   in a fraction of the size; see [`compact`].
//!
//! # LightGBM import
//!
//! [`BoostedModel::from_lightgbm_text`] reads the text model LightGBM 4.x
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
//! The XGBoost methods target the XGBoost 3.4.2 schema (identical to
//! 3.4.1's) in both of XGBoost's encodings: JSON text
//! ([`to_xgboost_json`] / [`from_xgboost_json`], XGBoost's `m.json`) and
//! Universal Binary JSON ([`to_xgboost_ubjson`] / [`from_xgboost_ubjson`],
//! XGBoost's `m.ubj` and `save_raw("ubj")`). Both encodings carry the same
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
//! `categories_sizes` arrays.
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
//! exports plain `gbtree` trees with each tree's weight multiplied into its
//! leaves (as CatBoost bakes its shrinkage), predicting the same margins bit
//! for bit; the imported model has no shrinkage record, so its iteration
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
//! becomes the parameters of the model's [`Objective`]
//! ([`ModelObjective::BuiltIn`]); absent fields take XGBoost's defaults, and
//! parameters the objective does not read are dropped (e.g.
//! `reg_loss_param.scale_pos_weight` of `reg:squarederror` or `reg:gamma`,
//! which hessboost does not apply). The alpha lists are XGBoost's array strings
//! (`"[0.1,0.5,0.9]"`, `(..)` also read); `reg:absoluteerror` and
//! `survival:cox` have no block. A value that does not parse, or an invalid
//! parameter of the objective (e.g. an empty or unsorted alpha list), is a
//! format error. An objective hessboost does not implement imports as
//! [`ModelObjective::Other`]: its model predicts margins.
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
//! payloads). [`to_xgboost_ubjson`] does the same with XGBoost's element
//! types: float32 for `split_conditions`, `base_weights`, `loss_changes`,
//! `sum_hessian` (and `leaf_weights`, gblinear `weights`); int32 for
//! `left_children`, `right_children`, `parents`, `categories`,
//! `categories_nodes` and `split_indices` (int64 when a tree's `num_feature`
//! exceeds the int32 range, as in XGBoost); uint8 for `default_left` and
//! `split_type`; int64 for `categories_segments` and `categories_sizes`. The
//! category container's int32 `feature_segments` / `sorted_idx` / `offsets`
//! and its per-column `values` follow XGBoost too. Every other array is a
//! counted generic array, numbers are float32 and integers the narrowest
//! width, again as XGBoost writes them. [`from_xgboost_ubjson`] accepts the
//! optimized and the plain UBJSON container forms alike.
//!
//! [`predict_margin`]: BoostedModel::predict_margin
//! [`predict_class`]: BoostedModel::predict_class
//! [`predict_leaf`]: BoostedModel::predict_leaf
//! [`predict_distribution`]: BoostedModel::predict_distribution
//! [`predict_contribs`]: BoostedModel::predict_contribs
//! [`predict_interactions`]: BoostedModel::predict_interactions
//! [`predict_range`]: BoostedModel::predict_range
//! [`predict_margin_range`]: BoostedModel::predict_margin_range
//! [`predict_leaf_range`]: BoostedModel::predict_leaf_range
//! [`predict_contribs_range`]: BoostedModel::predict_contribs_range
//! [`predict_interactions_range`]: BoostedModel::predict_interactions_range
//! [`predict_distribution_range`]: BoostedModel::predict_distribution_range
//! [`from_bytes`]: BoostedModel::from_bytes
//! [`save_binary`]: BoostedModel::save_binary
//! [`load_binary`]: BoostedModel::load_binary
//! [`from_json`]: BoostedModel::from_json
//! [`save_json`]: BoostedModel::save_json
//! [`load_json`]: BoostedModel::load_json
//! [`to_xgboost_json`]: BoostedModel::to_xgboost_json
//! [`from_xgboost_json`]: BoostedModel::from_xgboost_json
//! [`save_xgboost_json`]: BoostedModel::save_xgboost_json
//! [`load_xgboost_json`]: BoostedModel::load_xgboost_json
//! [`to_xgboost_ubjson`]: BoostedModel::to_xgboost_ubjson
//! [`from_xgboost_ubjson`]: BoostedModel::from_xgboost_ubjson
//! [`save_xgboost_ubjson`]: BoostedModel::save_xgboost_ubjson
//! [`load_xgboost_ubjson`]: BoostedModel::load_xgboost_ubjson

pub mod compact;
mod lightgbm;
mod native;
mod objective;
mod sections;
mod shap;
mod shrinkage;
mod ubjson;
pub mod uncertainty;
mod xgboost;

pub(crate) use shrinkage::Shrinkage;

pub use objective::ModelObjective;

use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::objective::distributional::Dist;
use crate::objective::{Loss, LossContext, Objective};
use crate::tree::compact::{CompactForest, FEATURE_LANES, LANES, LaneBlock, fill_lanes, key};
use crate::tree::{RegTree, UncheckedRegTree, scalar_tree_output};
use objective::{PartialStoredObjectiveParams, StoredObjectiveParams};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
/// ([`to_json`](Self::to_json) / [`from_json`](Self::from_json)).
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
    /// Per-tree contribution weights: empty when every tree weighs `1.0`
    /// (e.g. imported or sliced `gbtree` models), otherwise one weight per
    /// tree. The DART booster stores fractional weights here so dropped trees
    /// can be rescaled.
    tree_weights: Vec<f32>,
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
    /// Prediction layout of `trees` ([`CompactForest`]), derived lazily and
    /// never serialized. Reset whenever `trees` changes.
    compact: OnceLock<CompactForest>,
}

/// A [`BoostedModel`] as native JSON stores it (same names and order as
/// [`UncheckedBoostedModel`]), borrowed from the model.
#[derive(Serialize)]
struct SerializedBoostedModel<'a> {
    trees: &'a [RegTree],
    base_score: &'a [f32],
    objective: &'a str,
    objective_params: StoredObjectiveParams,
    num_class: usize,
    n_outputs: usize,
    n_targets: usize,
    n_features: usize,
    best_iteration: Option<usize>,
    tree_weights: &'a [f32],
    num_parallel_tree: usize,
    linear: &'a Option<LinearModel>,
    shrinkage: &'a Option<Shrinkage>,
}

impl Serialize for BoostedModel {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        SerializedBoostedModel {
            trees: &self.trees,
            base_score: &self.base_score,
            objective: self.objective.name(),
            objective_params: StoredObjectiveParams::of(&self.objective, self.max_delta_step),
            num_class: self.num_class,
            n_outputs: self.n_outputs,
            n_targets: self.n_targets,
            n_features: self.n_features,
            best_iteration: self.best_iteration,
            tree_weights: &self.tree_weights,
            num_parallel_tree: self.num_parallel_tree,
            linear: &self.linear,
            shrinkage: &self.shrinkage,
        }
        .serialize(serializer)
    }
}

/// The serialized fields of a [`BoostedModel`] (the native JSON format:
/// same names and layout), before validation. `BoostedModel`'s
/// `Deserialize` converts it with [`BoostedModel::try_from`], which runs
/// [`BoostedModel::validate_structure`].
///
/// Everything predictions depend on is required, including the nullable
/// `linear` (the gblinear weights), which a plain `Option` field would
/// default to `null` when absent; an absent `best_iteration` means none was
/// selected (every iteration predicts). Only the objective
/// parameters may be omitted (all of them, or any subset): each missing
/// one takes the recorded objective's default
/// ([`StoredObjectiveParams::defaults_for`]). A tree may omit `size_leaf_vector`
/// (scalar) and `leaf_vectors` (none), except that a multi-output model's
/// trees must state `size_leaf_vector`, since it decides whether they are
/// vector-leaf trees; each tree's `linear` is required
/// ([`UncheckedRegTree`]).
#[derive(Deserialize)]
struct UncheckedBoostedModel {
    trees: Vec<UncheckedRegTree>,
    base_score: Vec<f32>,
    objective: String,
    #[serde(default)]
    objective_params: PartialStoredObjectiveParams,
    num_class: usize,
    n_outputs: usize,
    n_targets: usize,
    n_features: usize,
    best_iteration: Option<usize>,
    tree_weights: Vec<f32>,
    num_parallel_tree: usize,
    #[serde(deserialize_with = "Option::deserialize")]
    linear: Option<LinearModel>,
    /// Absent in files written before model shrinkage existed: none.
    #[serde(default)]
    shrinkage: Option<Shrinkage>,
}

impl TryFrom<UncheckedBoostedModel> for BoostedModel {
    type Error = HessboostError;

    fn try_from(m: UncheckedBoostedModel) -> Result<Self> {
        if m.n_outputs > 1
            && let Some(tree) = m.trees.iter().position(|t| !t.states_leaf_width())
        {
            return Err(HessboostError::ModelFormat(format!(
                "tree {tree} of a {}-output model does not state size_leaf_vector",
                m.n_outputs
            )));
        }
        let stored = m.objective_params.fill(&m.objective);
        let objective = ModelObjective::from_stored(&m.objective, &stored, m.num_class)?;
        let model = BoostedModel {
            trees: m
                .trees
                .into_iter()
                .map(UncheckedRegTree::into_unchecked)
                .collect(),
            base_score: m.base_score,
            max_delta_step: stored.max_delta_step,
            objective,
            num_class: m.num_class,
            n_outputs: m.n_outputs,
            n_targets: m.n_targets,
            n_features: m.n_features,
            best_iteration: m.best_iteration,
            tree_weights: m.tree_weights,
            num_parallel_tree: m.num_parallel_tree,
            linear: m.linear,
            shrinkage: m.shrinkage,
            compact: OnceLock::new(),
        };
        model.validate_structure()?;
        Ok(model)
    }
}

/// The parameters of a linear (`gblinear`) booster: a per-output weight vector
/// plus a per-output bias, fit by coordinate descent.
///
/// `weights` has length `n_features * n_outputs` laid out `[feature][output]`
/// (the weight for feature `f`, output `k` is `weights[f * n_outputs + k]`).
/// `bias` has length `n_outputs`. Checked by the owning model
/// ([`BoostedModel::validate_structure`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LinearModel {
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl LinearModel {
    /// Assemble a linear model from its fitted `weights` (`[feature][output]`)
    /// and per-output `bias`.
    pub(crate) fn new(weights: Vec<f32>, bias: Vec<f32>) -> Self {
        LinearModel { weights, bias }
    }

    pub(crate) fn bias(&self) -> &[f32] {
        &self.bias
    }

    pub(crate) fn weights(&self) -> &[f32] {
        &self.weights
    }
}

/// Invoke `f(feature, value)` for each present feature of `row` in feature
/// order. Shared by the gblinear training and prediction paths; arithmetic
/// stays at each call site to preserve exact conversion points.
pub(crate) fn for_each_present_value(data: &DMatrix, row: usize, mut f: impl FnMut(usize, f32)) {
    for feat in 0..data.n_cols() {
        if let Some(x) = data.get(row, feat) {
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
            tree_weights: Vec::new(),
            num_parallel_tree: self.num_parallel_tree,
            linear: None,
            shrinkage: None,
            compact: OnceLock::new(),
        }
    }

    /// Append a tree with an explicit contribution weight (`1.0` for plain
    /// `gbtree`; DART stores fractional weights so dropped trees can be
    /// rescaled).
    pub(crate) fn push_tree_weighted(&mut self, tree: RegTree, weight: f32) {
        self.trees.push(tree);
        self.tree_weights.push(weight);
        self.compact = OnceLock::new();
    }

    /// The prediction layout of the ensemble, built on first use and dropped
    /// whenever a tree is appended.
    pub(crate) fn compact_forest(&self) -> &CompactForest {
        self.compact
            .get_or_init(|| CompactForest::from_trees(&self.trees))
    }

    /// Contribution weight of tree `i` (`1.0` when weights are absent, e.g. for
    /// imported models or plain `gbtree`).
    #[inline]
    pub(crate) fn tree_weight(&self, i: usize) -> f32 {
        self.tree_weights.get(i).copied().unwrap_or(1.0)
    }

    /// Whether this is a `gblinear` model, whose predictions come from the
    /// linear weights instead of the tree ensemble.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn is_gblinear(&self) -> bool {
        self.linear.is_some()
    }

    /// Whether any tree carries per-leaf linear models (`linear_tree`).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn has_linear_leaves(&self) -> bool {
        self.trees.iter().any(|tree| tree.linear_leaves().is_some())
    }

    /// Whether tree `t` stores a weight vector per leaf (vector-leaf trees).
    #[cfg(all(target_os = "macos", feature = "metal"))]
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
        data: &DMatrix,
        row: usize,
        mut f: impl FnMut(usize, usize, f64),
    ) {
        let Some(lm) = &self.linear else {
            return;
        };
        let k = self.n_outputs();
        for_each_present_value(data, row, |feat, x| {
            for c in 0..k {
                f(feat, c, f64::from(lm.weights[feat * k + c]) * f64::from(x));
            }
        });
    }

    /// Multiply tree `i`'s contribution weight by `factor` (DART rescaling).
    pub(crate) fn scale_tree_weight(&mut self, i: usize, factor: f32) {
        if i < self.tree_weights.len() {
            self.tree_weights[i] *= factor;
        }
    }

    /// Raw margin predictions that exclude the trees marked `true` in `dropped`
    /// (indexed by tree id). Used by the DART training loop to compute a round's
    /// gradients from the ensemble minus its dropout set. Output is laid out
    /// `[instance][output]`.
    pub(crate) fn predict_margin_dropout(&self, data: &DMatrix, dropped: &[bool]) -> Vec<f32> {
        let mut out = self.initial_margins(data);
        // Dropped trees contribute a zero weight, leaving per-cell accumulation
        // in ascending tree order (a `0.0` addend is a no-op).
        let weight = |ti: usize| {
            if dropped.get(ti).copied().unwrap_or(false) {
                0.0
            } else {
                self.tree_weight(ti)
            }
        };
        self.accumulate_forest(data, &mut out, 0..self.trees.len(), weight);
        out
    }

    pub(crate) fn set_best_iteration(&mut self, it: Option<usize>) {
        self.best_iteration = it;
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
            tree_weights,
            num_parallel_tree: 1,
            linear: None,
            shrinkage: None,
            compact: OnceLock::new(),
        }
    }

    /// The configured `num_class` (`0` for regression / binary objectives).
    pub(crate) fn num_class(&self) -> usize {
        self.num_class
    }

    /// The `max_delta_step` training used (`0` when unbounded).
    pub(crate) fn max_delta_step(&self) -> f64 {
        self.max_delta_step
    }

    /// Number of raw outputs per instance: `num_class` for multiclass, the
    /// objective's output count otherwise (`1` for every built-in scalar
    /// objective, [`BoostedModel::n_targets`] for a multi-target model; custom
    /// objectives may declare more).
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

    /// Margins from the trees `trees` (tree ids) without validating `data`.
    pub(crate) fn margin_from_trees(
        &self,
        data: &DMatrix,
        trees: std::ops::Range<usize>,
    ) -> Vec<f32> {
        let n = data.n_rows();
        let k = self.n_outputs();
        // Initialize from the dataset's per-instance base margin when present
        // (it overrides the per-output intercepts, matching XGBoost); otherwise
        // use the trained global bias.
        let mut out = self.initial_margins(data);
        // A gblinear model predicts from its linear parameters and ignores the
        // (empty) tree ensemble: margin(row, k) = base_score[k] + bias[k] +
        // Σ_f weights[f][k] * x[row, f], with missing features contributing 0.
        if let Some(lm) = &self.linear {
            // Each row adds its bias, then its features in ascending order.
            let add_row = |(row, margin): (usize, &mut [f32])| {
                for (m, &b) in margin.iter_mut().zip(&lm.bias) {
                    *m += b;
                }
                self.for_each_linear_contribution(data, row, |_f, c, v| {
                    margin[c] += v as f32;
                });
            };
            // Rows are independent, so parallel rows give the serial result.
            if n >= PREDICT_BLOCK_ROWS && rayon::current_num_threads() > 1 {
                out.par_chunks_mut(k)
                    .with_min_len(PREDICT_BLOCK_ROWS)
                    .enumerate()
                    .for_each(add_row);
            } else {
                out.chunks_mut(k).enumerate().for_each(add_row);
            }
            return out;
        }
        self.accumulate_forest(data, &mut out, trees, |ti| self.tree_weight(ti));
        out
    }

    /// Sum `weight(t) * leaf(row, t)` into `out[row * k + tree_output(t)]` for
    /// the trees in `trees`, where `k` is the output count. Rows are traversed
    /// in cache-friendly blocks (parallel across blocks); per (row, output)
    /// slot the trees are still summed in ascending order, so the result is
    /// bit-identical to the sequential tree-outer loop. Ensembles with linear
    /// leaves take a per-row path instead, since the compact forest stores
    /// constant leaf values only.
    fn accumulate_forest(
        &self,
        data: &DMatrix,
        out: &mut [f32],
        trees: std::ops::Range<usize>,
        weight: impl Fn(usize) -> f32 + Sync,
    ) {
        let k = self.n_outputs();
        if self.has_vector_leaves() {
            self.traverse_blocks(
                data,
                out,
                k,
                trees.clone(),
                |block, forest, r, out_row| {
                    block.accumulate_row_vector(forest, r, trees.clone(), &weight, out_row);
                },
                |block, forest, ti, rows, out_block, stride| {
                    block.accumulate_vector(forest, ti, rows, weight(ti), out_block, stride);
                },
            );
            return;
        }
        if self.trees[trees.clone()]
            .iter()
            .any(|tree| tree.linear_leaves().is_some())
        {
            crate::tree::linear::accumulate_forest(
                &self.trees,
                trees,
                |t| self.tree_output(t),
                data,
                out,
                k,
                weight,
            );
            return;
        }
        let parallel = self.num_parallel_tree;
        self.traverse_blocks(
            data,
            out,
            k,
            trees.clone(),
            |block, forest, r, out_row| {
                block.accumulate_row(forest, r, trees.clone(), parallel, &weight, out_row);
            },
            |block, forest, ti, rows, out_block, stride| {
                block.accumulate(
                    forest,
                    ti,
                    rows,
                    weight(ti),
                    &mut out_block[self.tree_output(ti)..],
                    stride,
                );
            },
        );
    }

    /// Block-parallel traversal of the trees in `trees` over `data`, writing
    /// into `out` laid out `[row][stride]`: `row_op` handles one loaded row of
    /// the small-batch path, `tree_op` one tree over a loaded block of rows.
    /// Tiny batches (online serving) skip the thread pool and overlap the
    /// trees of each row instead of the rows of each tree; larger inputs
    /// process rows in blocks whose feature rows stay in cache while every
    /// tree walks them, with blocks running in parallel.
    fn traverse_blocks<T: Send>(
        &self,
        data: &DMatrix,
        out: &mut [T],
        stride: usize,
        trees: std::ops::Range<usize>,
        row_op: impl Fn(&RowBlock, &CompactForest, usize, &mut [T]),
        tree_op: impl Fn(&RowBlock, &CompactForest, usize, usize, &mut [T], usize) + Sync,
    ) {
        let n = data.n_rows();
        let forest = self.compact_forest();
        if n < LANES {
            let mut block = RowBlock::new(data);
            block.load(0, n);
            for (r, out_row) in out.chunks_exact_mut(stride).enumerate() {
                row_op(&block, forest, r, out_row);
            }
            return;
        }
        out.par_chunks_mut(PREDICT_BLOCK_ROWS * stride)
            .enumerate()
            .for_each_init(
                || RowBlock::new(data),
                |block, (bi, out_block)| {
                    let start = bi * PREDICT_BLOCK_ROWS;
                    let rows = out_block.len() / stride;
                    block.load(start, rows);
                    for ti in trees.clone() {
                        tree_op(block, forest, ti, rows, out_block, stride);
                    }
                },
            );
    }

    /// [`Self::predict`] from the boosting `iterations` only (see
    /// [`Self::predict_margin_range`] for the range convention).
    pub fn predict_range(
        &self,
        data: &DMatrix,
        iterations: impl RangeBounds<usize>,
    ) -> Result<Vec<f32>> {
        let margin = self.predict_margin_range(data, iterations)?;
        Ok(transform_model_margins(
            &self.objective,
            self.max_delta_step,
            self.n_targets,
            self.n_outputs(),
            margin,
        ))
    }

    /// For multiclass, the predicted class index per row (argmax over classes).
    /// For single-output models this returns the transformed prediction rounded
    /// to the nearest class at 0.5. A multi-target model (label matrix) is
    /// multi-label: each target is thresholded at 0.5 independently, giving
    /// `n_rows × n_targets` decisions laid out `[row][target]`.
    pub fn predict_class(&self, data: &DMatrix) -> Result<Vec<u32>> {
        let probs = self.predict(data)?;
        let k = self.n_outputs();
        if k == 1 || self.n_targets > 1 {
            return Ok(probs.iter().map(|&p| u32::from(p > 0.5)).collect());
        }
        if self
            .objective
            .built_in()
            .is_some_and(Objective::predicts_class_index)
        {
            return Ok(probs.iter().map(|&class| class as u32).collect());
        }
        Ok(probs
            .chunks_exact(k)
            .map(|row| crate::simd::argmax_scalar(row) as u32)
            .collect())
    }

    /// [`Self::predict_leaf`] for the trees of the iterations in
    /// `iterations` (shape `n_rows × trees`). As in XGBoost the range
    /// must start at iteration `0`; use [`Self::slice`] for a later start.
    #[allow(
        clippy::redundant_closure_for_method_calls,
        reason = "the method path is not general enough over the block lifetime"
    )]
    pub fn predict_leaf_range(
        &self,
        data: &DMatrix,
        iterations: impl RangeBounds<usize>,
    ) -> Result<Vec<u32>> {
        self.validate_prediction_data(data)?;
        let t = self.prefix_trees(iterations, "leaf prediction")?;
        let n = data.n_rows();
        let mut out = vec![0u32; n * t];
        if t == 0 {
            return Ok(out);
        }
        self.traverse_blocks(
            data,
            &mut out,
            t,
            0..t,
            |block, forest, r, out_row| block.original_leaf_ids_for_row(forest, r, out_row),
            |block, forest, ti, rows, out_block, stride| {
                block.original_leaf_ids(forest, ti, rows, &mut out_block[ti..], stride);
            },
        );
        Ok(out)
    }

    /// Compute feature importance of the requested type, returned as a map from
    /// feature index to score (features that never split are absent).
    pub fn feature_importance(&self, kind: ImportanceType) -> HashMap<u32, f64> {
        let mut count: HashMap<u32, f64> = HashMap::new();
        let mut cover: HashMap<u32, f64> = HashMap::new();
        let mut gain: HashMap<u32, f64> = HashMap::new();
        for tree in &self.trees {
            for node in tree.nodes() {
                if node.is_leaf() {
                    continue;
                }
                *count.entry(node.split_feature).or_default() += 1.0;
                *cover.entry(node.split_feature).or_default() += f64::from(node.sum_hess);
                *gain.entry(node.split_feature).or_default() += f64::from(node.split_gain);
            }
        }
        // Divide a total by the split count to get the per-split average.
        let average = |totals: HashMap<u32, f64>| -> HashMap<u32, f64> {
            totals
                .into_iter()
                .map(|(f, t)| {
                    let n = count.get(&f).copied().unwrap_or(1.0);
                    (f, t / n)
                })
                .collect()
        };
        match kind {
            ImportanceType::Weight => count,
            ImportanceType::TotalCover => cover,
            ImportanceType::Cover => average(cover),
            ImportanceType::TotalGain => gain,
            ImportanceType::Gain => average(gain),
        }
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

    /// Read-only access to the trees (e.g. for serialization or SHAP).
    pub fn trees(&self) -> &[RegTree] {
        &self.trees
    }

    /// Number of features the model expects.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub(crate) fn linear(&self) -> Option<&LinearModel> {
        self.linear.as_ref()
    }

    pub(crate) fn has_non_unit_tree_weights(&self) -> bool {
        (0..self.trees.len()).any(|i| self.tree_weight(i) != 1.0)
    }

    /// Margin buffer for `data`: the per-output intercepts broadcast to every
    /// row, overridden by the dataset's per-instance `base_margin` when
    /// present (one value per row, or one per row and output). Shared by
    /// prediction, TreeSHAP, and the training margin caches.
    pub(crate) fn initial_margins(&self, data: &DMatrix) -> Vec<f32> {
        initial_margins(&self.base_score, data)
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
        iterations: impl RangeBounds<usize>,
        what: &str,
    ) -> Result<AttributionPrologue<'_>> {
        self.validate_prediction_data(data)?;
        let end = self.prefix_trees(iterations, what)?;
        self.refuse_partial_shrunk_range(&(0..end), what)?;
        let trees = &self.trees[..end];
        if trees.iter().any(|tree| tree.linear_leaves().is_some()) {
            return Err(HessboostError::invalid_param(
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
            initial: self.initial_margins(data),
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
    }

    /// Give every tree an explicit contribution weight (`1.0` where absent),
    /// so appended trees' weights line up with their tree ids.
    pub(crate) fn materialize_tree_weights(&mut self) {
        self.tree_weights.resize(self.trees.len(), 1.0);
    }

    /// Remove and return every tree (dropping the contribution weights),
    /// leaving an empty ensemble with the same metadata. Used by
    /// `process_type=update`, which re-appends the refreshed trees.
    pub(crate) fn take_trees(&mut self) -> Vec<RegTree> {
        self.tree_weights.clear();
        self.compact = OnceLock::new();
        std::mem::take(&mut self.trees)
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

    /// The iteration range plain prediction uses: `..best_iteration + 1`
    /// when early stopping selected an iteration, else every iteration
    /// (`..`, which is also the only range a `gblinear` model accepts).
    pub(crate) fn default_iteration_range(&self) -> (Bound<usize>, Bound<usize>) {
        let end = self
            .best_iteration
            .map_or(Bound::Unbounded, |it| Bound::Excluded(it + 1));
        (Bound::Unbounded, end)
    }

    /// Resolve `iterations`, any range of boosting iterations (`..`, `..end`,
    /// `begin..end`, ...), against the model's iteration count.
    pub(crate) fn resolve_iterations(
        &self,
        iterations: impl RangeBounds<usize>,
        param: &'static str,
    ) -> Result<Range<usize>> {
        if self.linear.is_some() {
            // No iterations to select: only the whole model (`..`) is a range.
            let whole = matches!(
                iterations.start_bound(),
                Bound::Unbounded | Bound::Included(0)
            ) && iterations.end_bound() == Bound::Unbounded;
            if !whole {
                return Err(HessboostError::invalid_param(
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
        if end > rounds || begin > end {
            return Err(HessboostError::invalid_param(
                param,
                format!("{begin}..{end} is out of range for a model with {rounds} iterations"),
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
    fn prefix_trees(&self, iterations: impl RangeBounds<usize>, what: &str) -> Result<usize> {
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

    /// Raw margin predictions using the effective iterations (`[0,
    /// best_iteration + 1)` after early stopping, else all). The output is
    /// laid out `[instance][output]` (length `n_rows × n_outputs`).
    pub fn predict_margin(&self, data: &DMatrix) -> Result<Vec<f32>> {
        self.predict_margin_range(data, self.default_iteration_range())
    }

    /// Raw margin predictions from the boosting `iterations` only (XGBoost's
    /// `iteration_range`), any range of iteration indices: `..` is the whole
    /// model regardless of early stopping, `..n` the first `n` iterations,
    /// `2..5` iterations 2 to 4. The intercept / dataset `base_margin` is
    /// always included, so an empty range predicts it alone. A `gblinear`
    /// model has no boosting iterations to select and accepts only `..`.
    ///
    /// For a model trained with model shrinkage
    /// ([`TrainingParams::model_shrink`](crate::config::TrainingParams::model_shrink)),
    /// `..n` is the model after `n` iterations, rebuilt exactly (bit for bit
    /// the predictions of the same training run stopped after `n` rounds):
    /// its trees reweighted by the shrinkage applied up to iteration `n` and
    /// its intercepts shrunk as far. Ranges starting after iteration 0 are
    /// refused, since the ensemble is rescaled every iteration.
    pub fn predict_margin_range(
        &self,
        data: &DMatrix,
        iterations: impl RangeBounds<usize>,
    ) -> Result<Vec<f32>> {
        self.validate_prediction_data(data)?;
        let iterations = self.resolve_iterations(iterations, "iterations")?;
        if let Some(shrinkage) = &self.shrinkage {
            // Every later iteration rescaled the earlier ones, so the
            // iterations `a..b` alone are no model.
            if iterations.start != 0 {
                return Err(HessboostError::invalid_param(
                    "iterations",
                    format!(
                        "a model trained with model shrinkage rescales its earlier iterations \
                         every iteration, so {}..{} has no meaning; use a range starting at 0",
                        iterations.start, iterations.end
                    ),
                ));
            }
            if iterations.end < self.num_boost_rounds() {
                let (weights, base) = shrinkage.scaling(iterations.end, self.trees_per_iteration());
                let mut out = initial_margins(&base, data);
                self.accumulate_forest(data, &mut out, 0..weights.len(), |t| weights[t]);
                return Ok(out);
            }
        }
        Ok(self.margin_from_trees(data, self.iteration_trees(iterations)))
    }

    /// Refuse anything but the whole ensemble of a shrunk model, for the
    /// predictions (`what`) that read the stored tree weights directly;
    /// [`Self::slice`]`(..k, 1)` builds the model after `k` iterations.
    pub(crate) fn refuse_partial_shrunk_range(
        &self,
        trees: &Range<usize>,
        what: &str,
    ) -> Result<()> {
        if self.shrinkage.is_none() || (trees.start == 0 && trees.end == self.trees.len()) {
            return Ok(());
        }
        Err(HessboostError::invalid_param(
            "iterations",
            format!(
                "{what} of a model trained with model shrinkage covers the whole ensemble \
                 only; `slice(..k, 1)` builds the model after `k` iterations"
            ),
        ))
    }

    /// Predictions in the objective's reported space. `multi:softprob` returns
    /// an `n_rows × num_class` probability matrix while `multi:softmax` returns
    /// one class index per row, encoded as `f32`. Uses the effective
    /// iterations, like [`Self::predict_margin`].
    pub fn predict(&self, data: &DMatrix) -> Result<Vec<f32>> {
        self.predict_range(data, self.default_iteration_range())
    }

    /// The predicted distribution of every row for a model trained with a
    /// distributional `dist:*` objective (beyond XGBoost, see
    /// [`crate::objective::distributional`]): one [`Dist`] per row, with its
    /// mean, variance, CDF, quantiles, log density, CRPS, intervals and
    /// sampling. The margins are mapped through the links in `f64`. Uses the
    /// effective iterations, like [`Self::predict`].
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] if the model's objective is not
    /// a `dist:*` objective, plus the errors of [`Self::predict_margin`].
    pub fn predict_distribution(&self, data: &DMatrix) -> Result<Vec<Dist>> {
        self.predict_distribution_range(data, self.default_iteration_range())
    }

    /// [`Self::predict_distribution`] from the boosting iterations in
    /// `iterations` only (see [`Self::predict_margin_range`]).
    pub fn predict_distribution_range(
        &self,
        data: &DMatrix,
        iterations: impl RangeBounds<usize>,
    ) -> Result<Vec<Dist>> {
        let family = self
            .objective
            .built_in()
            .and_then(Objective::dist_family)
            .ok_or_else(|| {
                HessboostError::invalid_param(
                    "objective",
                    format!(
                        "`{}` does not predict distributions; train with a `dist:*` objective",
                        self.objective.name()
                    ),
                )
            })?;
        let margin = self.predict_margin_range(data, iterations)?;
        Ok(margin
            .chunks_exact(family.n_params())
            .map(|row| {
                let mut eta = [0.0; 2];
                for (e, &m) in eta.iter_mut().zip(row) {
                    *e = f64::from(m);
                }
                family.dist_from_margins(&eta[..row.len()])
            })
            .collect())
    }

    /// Per-row leaf indices for each tree (shape `n_rows × num_trees`,
    /// row-major, trees in [`Self::trees`] order). Walks every tree,
    /// regardless of early stopping.
    pub fn predict_leaf(&self, data: &DMatrix) -> Result<Vec<u32>> {
        self.predict_leaf_range(data, ..)
    }

    /// A new model holding every `step`-th boosting iteration of
    /// `iterations` (Python's `booster[begin:end:step]`; e.g.
    /// `model.slice(2..8, 3)` keeps iterations 2 and 5, `model.slice(.., 1)`
    /// all of them). Each selected iteration keeps its whole
    /// forest (every output and parallel tree) together with its DART tree
    /// weights; the intercepts, objective and layout carry over unchanged and
    /// no refit happens. As in XGBoost the slice drops `best_iteration`, so
    /// it predicts with all of its iterations.
    ///
    /// A model trained with model shrinkage slices to prefixes only
    /// (`..k` with step 1): the result is the model after `k` iterations,
    /// bit for bit the model the same training run stopped after `k` rounds
    /// returns (see [`Self::predict_margin_range`]).
    ///
    /// `step` must be at least 1 and the range non-empty and within
    /// [`Self::num_boost_rounds`]. XGBoost 3.4.2 additionally trips an
    /// internal check when `end - begin` is not a multiple of `step`; here
    /// every step selects `ceil((end - begin) / step)` iterations, matching
    /// XGBoost wherever it succeeds.
    pub fn slice(&self, iterations: impl RangeBounds<usize>, step: usize) -> Result<BoostedModel> {
        if self.linear.is_some() {
            return Err(HessboostError::invalid_param(
                "slice",
                "gblinear models have no boosting iterations to slice",
            ));
        }
        if step == 0 {
            return Err(HessboostError::invalid_param(
                "slice",
                "step must be at least 1",
            ));
        }
        let Range { start: begin, end } = self.resolve_iterations(iterations, "slice")?;
        if begin == end {
            return Err(HessboostError::invalid_param(
                "slice",
                format!("empty slice {begin}..{end} is not allowed"),
            ));
        }
        if let Some(shrinkage) = &self.shrinkage {
            if begin != 0 || step != 1 {
                return Err(HessboostError::invalid_param(
                    "slice",
                    "a model trained with model shrinkage slices to prefixes only (`..k`, step 1)",
                ));
            }
            return Ok(self.shrunk_prefix(shrinkage, end));
        }
        let per = self.trees_per_iteration();
        let mut trees = Vec::with_capacity((end - begin).div_ceil(step) * per);
        let mut tree_weights = Vec::new();
        for it in (begin..end).step_by(step) {
            let layer = it * per..(it + 1) * per;
            trees.extend_from_slice(&self.trees[layer.clone()]);
            if !self.tree_weights.is_empty() {
                tree_weights.extend(layer.map(|t| self.tree_weight(t)));
            }
        }
        Ok(BoostedModel {
            trees,
            base_score: self.base_score.clone(),
            objective: self.objective.clone(),
            max_delta_step: self.max_delta_step,
            num_class: self.num_class,
            n_outputs: self.n_outputs,
            n_targets: self.n_targets,
            n_features: self.n_features,
            best_iteration: None,
            tree_weights,
            num_parallel_tree: self.num_parallel_tree,
            linear: None,
            shrinkage: None,
            compact: OnceLock::new(),
        })
    }

    /// The model after the first `k` iterations of this shrunk model:
    /// their trees, reweighted by the record's first `k` coefficients, and
    /// the intercepts shrunk as far ([`Shrinkage::scaling`]).
    fn shrunk_prefix(&self, shrinkage: &Shrinkage, k: usize) -> BoostedModel {
        let per = self.trees_per_iteration();
        let (tree_weights, base_score) = shrinkage.scaling(k, per);
        BoostedModel {
            trees: self.trees[..k * per].to_vec(),
            base_score,
            objective: self.objective.clone(),
            max_delta_step: self.max_delta_step,
            num_class: self.num_class,
            n_outputs: self.n_outputs,
            n_targets: self.n_targets,
            n_features: self.n_features,
            best_iteration: None,
            tree_weights,
            num_parallel_tree: self.num_parallel_tree,
            linear: None,
            shrinkage: Some(shrinkage.truncated(k)),
            compact: OnceLock::new(),
        }
    }

    /// Record the shrinkage training applied (one coefficient per
    /// iteration, and the intercepts before it) and derive the tree weights
    /// and intercepts of the full model from it.
    pub(crate) fn set_shrinkage(&mut self, shrinkage: Shrinkage) {
        let (tree_weights, base_score) =
            shrinkage.scaling(self.num_boost_rounds(), self.trees_per_iteration());
        self.tree_weights = tree_weights;
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

    /// The per-iteration shrinkage record, if the model was trained with
    /// model shrinkage.
    pub(crate) fn shrinkage(&self) -> Option<&Shrinkage> {
        self.shrinkage.as_ref()
    }

    /// The iteration range the attribution predictions use by default (the
    /// effective iterations, like [`Self::predict_margin`]).
    pub(crate) fn validate_prediction_data(&self, data: &DMatrix) -> Result<()> {
        validate_prediction_data(self.n_features, self.n_outputs(), data)
    }

    /// Serialize the model to the native binary format: a zstd-compressed
    /// container of named, typed sections holding the trees column-wise.
    /// Files written by this version keep loading in later ones.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        native::write(self)
    }

    /// Deserialize a model from bytes produced by [`BoostedModel::to_bytes`]
    /// of this or an earlier version. Malformed input, and files that need
    /// a feature this version lacks, are refused with
    /// [`HessboostError::ModelFormat`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let model = native::read(bytes)?;
        model.validate_structure()?;
        Ok(model)
    }

    /// Save the model to a file in the native binary format.
    pub fn save_binary(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        Ok(std::fs::write(path, self.to_bytes()?)?)
    }

    /// Load a model from a native binary file.
    pub fn load_binary(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Serialize the model to a (human-readable) JSON string: the model's
    /// fields by name.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize a model from a JSON string produced by
    /// [`BoostedModel::to_json`]. Malformed JSON is refused with
    /// [`HessboostError::Json`], an inconsistent model with
    /// [`HessboostError::ModelFormat`].
    pub fn from_json(s: &str) -> Result<Self> {
        Self::try_from(serde_json::from_str::<UncheckedBoostedModel>(s)?)
    }

    /// Check everything prediction and the formats rely on: the output
    /// layout, the objective and its parameters, and the stored values
    /// (in that order; the first failure is reported).
    pub(crate) fn validate_structure(&self) -> Result<()> {
        self.validate_layout()?;
        self.validate_objective()?;
        self.validate_values()
    }

    /// Feature count, outputs, targets, forest size, and tree kinds.
    fn validate_layout(&self) -> Result<()> {
        if self.n_features == 0 {
            return Err(HessboostError::model_format(
                "model has an invalid feature count",
            ));
        }
        let vector = self.has_vector_leaves();
        // Both factors come from the file: check the product before any
        // divisibility or round-count arithmetic relies on it.
        let per_iteration = if vector {
            Some(self.num_parallel_tree)
        } else {
            self.n_outputs.checked_mul(self.num_parallel_tree)
        };
        if self.n_outputs == 0
            || self.n_targets == 0
            || self.num_parallel_tree == 0
            || (self.num_class >= 2 && self.n_outputs != self.num_class)
            || per_iteration.is_none_or(|per| !self.trees.len().is_multiple_of(per))
            || self.trees.iter().any(|tree| {
                tree.is_vector_leaf() != vector
                    || (vector && tree.size_leaf_vector() != self.n_outputs)
            })
        {
            return Err(HessboostError::ModelFormat(format!(
                "invalid output layout: {} outputs, num_class {}, num_parallel_tree {}, {} trees",
                self.n_outputs,
                self.num_class,
                self.num_parallel_tree,
                self.trees.len()
            )));
        }
        Ok(())
    }

    /// The objective's output width and the stored `max_delta_step`
    /// (the objective's own parameters are valid by construction).
    fn validate_objective(&self) -> Result<()> {
        if !(self.max_delta_step.is_finite() && self.max_delta_step >= 0.0) {
            return Err(HessboostError::model_format(format!(
                "invalid objective parameters: max_delta_step {} is not finite and >= 0",
                self.max_delta_step
            )));
        }
        check_objective_width(
            &self.objective,
            self.max_delta_step,
            self.num_class,
            self.n_targets,
            self.n_outputs,
        )
    }

    /// `best_iteration`, intercepts, tree weights, trees, and the linear
    /// booster's parameters.
    fn validate_values(&self) -> Result<()> {
        // Early stopping selects iterations of a tree ensemble; gblinear has
        // none (training refuses early stopping for it), and a stored value
        // would make plain prediction ask it for an iteration range.
        if let Some(best) = self.best_iteration {
            if self.linear.is_some() {
                return Err(HessboostError::ModelFormat(format!(
                    "gblinear models have no boosting iterations, but best_iteration is {best}"
                )));
            }
            if best >= self.num_boost_rounds() {
                return Err(HessboostError::ModelFormat(format!(
                    "best_iteration {best} is out of range for {} iterations",
                    self.num_boost_rounds()
                )));
            }
        }
        if self.base_score.len() != self.n_outputs()
            || self.base_score.iter().any(|v| !v.is_finite())
        {
            return Err(HessboostError::ModelFormat(format!(
                "base_score must hold one finite value per output ({} outputs, got {:?})",
                self.n_outputs(),
                self.base_score
            )));
        }
        if !self.tree_weights.is_empty() && self.tree_weights.len() != self.trees.len() {
            return Err(HessboostError::model_format(
                "tree_weights length does not match trees",
            ));
        }
        if self.tree_weights.iter().any(|weight| !weight.is_finite()) {
            return Err(HessboostError::model_format("tree weights must be finite"));
        }
        for (tree_id, tree) in self.trees.iter().enumerate() {
            if !tree.is_valid_for_features(self.n_features) {
                return Err(HessboostError::ModelFormat(format!(
                    "tree {tree_id} contains invalid nodes"
                )));
            }
        }
        if let Some(linear) = &self.linear {
            let outputs = self.n_outputs();
            if linear.bias.len() != outputs
                || Some(linear.weights.len()) != self.n_features.checked_mul(outputs)
            {
                return Err(HessboostError::model_format(
                    "linear model dimensions are invalid",
                ));
            }
            if !linear
                .weights
                .iter()
                .chain(&linear.bias)
                .all(|v| v.is_finite())
            {
                return Err(HessboostError::model_format(
                    "linear model parameters must be finite",
                ));
            }
        }
        self.validate_shrinkage()
    }

    /// A shrinkage record must describe a tree model's iterations and
    /// agree bit for bit with the tree weights and intercepts it derives,
    /// and early stopping of a shrunk model keeps only the best iterations,
    /// so a `best_iteration` can only name the last one.
    fn validate_shrinkage(&self) -> Result<()> {
        let Some(shrinkage) = &self.shrinkage else {
            return Ok(());
        };
        if self.linear.is_some() {
            return Err(HessboostError::model_format(
                "gblinear models cannot carry a shrinkage record",
            ));
        }
        let rounds = self.num_boost_rounds();
        shrinkage.validate(rounds, self.n_outputs)?;
        if self.best_iteration.is_some_and(|best| best + 1 != rounds) {
            return Err(HessboostError::model_format(
                "a shrunk model's best_iteration must be its last iteration",
            ));
        }
        let (weights, base) = shrinkage.scaling(rounds, self.trees_per_iteration());
        let same = |a: f32, b: f32| a.to_bits() == b.to_bits();
        if !(weights
            .iter()
            .enumerate()
            .all(|(t, &w)| same(w, self.tree_weight(t)))
            && base.iter().zip(&self.base_score).all(|(&a, &b)| same(a, b)))
        {
            return Err(HessboostError::model_format(
                "the tree weights and intercepts do not match the shrinkage record",
            ));
        }
        Ok(())
    }

    /// Save the model to a JSON file.
    pub fn save_json(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        Ok(std::fs::write(path, self.to_json()?)?)
    }

    /// Load a model from a JSON file.
    pub fn load_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    /// Serialize the model to XGBoost's JSON model schema (the text form of
    /// `booster.save_model("m.json")`), so XGBoost-compatible tooling can read
    /// it. See [XGBoost interchange](crate::model#xgboost-interchange) for the
    /// mapping and caveats.
    pub fn to_xgboost_json(&self) -> Result<String> {
        crate::model::xgboost::export_xgboost_json(self)
    }

    /// Parse a model saved in XGBoost's JSON model schema: `gbtree` boosters
    /// (including DART) with scalar- or vector-leaf trees. See
    /// [XGBoost interchange](crate::model#xgboost-interchange) for the
    /// mapping and limitations.
    pub fn from_xgboost_json(json: &str) -> Result<Self> {
        crate::model::xgboost::import_xgboost_json(json)
    }

    /// Save the model to a file in XGBoost's JSON model format.
    pub fn save_xgboost_json(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        Ok(std::fs::write(path, self.to_xgboost_json()?)?)
    }

    /// Load a model from a file written in XGBoost's JSON model format.
    pub fn load_xgboost_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_xgboost_json(&std::fs::read_to_string(path)?)
    }

    /// Serialize the model to XGBoost's UBJSON model format (the bytes of
    /// `booster.save_model("m.ubj")` / `save_raw("ubj")`), carrying the same
    /// document as [`BoostedModel::to_xgboost_json`]. See
    /// [UBJSON encoding](crate::model#ubjson-encoding) for the element types.
    pub fn to_xgboost_ubjson(&self) -> Result<Vec<u8>> {
        crate::model::xgboost::export_xgboost_ubjson(self)
    }

    /// Parse a model saved in XGBoost's UBJSON model format (optimized or
    /// plain containers), with the same mapping and limitations as
    /// [`BoostedModel::from_xgboost_json`].
    pub fn from_xgboost_ubjson(bytes: &[u8]) -> Result<Self> {
        crate::model::xgboost::import_xgboost_ubjson(bytes)
    }

    /// Save the model to a file in XGBoost's UBJSON model format (XGBoost's
    /// `.ubj` files).
    pub fn save_xgboost_ubjson(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        Ok(std::fs::write(path, self.to_xgboost_ubjson()?)?)
    }

    /// Load a model from a file written in XGBoost's UBJSON model format.
    pub fn load_xgboost_ubjson(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_xgboost_ubjson(&std::fs::read(path)?)
    }

    /// Parse a LightGBM 4.x text model: the file `booster.save_model("model.txt")`
    /// writes, or `booster.model_to_string()`. The model predicts, explains,
    /// slices, and saves like any other; see
    /// [LightGBM import](crate::model#lightgbm-import) for the mapping, the
    /// input conventions it assumes (missing values as `NaN`, categories as
    /// non-negative codes), and the models it refuses with a
    /// [`HessboostError::ModelFormat`].
    pub fn from_lightgbm_text(text: &str) -> Result<Self> {
        crate::model::lightgbm::import_lightgbm_text(text)
    }

    /// Load a LightGBM 4.x text model file (`booster.save_model("model.txt")`);
    /// see [`BoostedModel::from_lightgbm_text`].
    pub fn load_lightgbm_text(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_lightgbm_text(&std::fs::read_to_string(path)?)
    }

    /// The loss of the model's built-in objective (`None` for another
    /// objective, whose predictions are margins).
    pub(crate) fn rebuild_objective(&self) -> Option<Result<Arc<dyn Loss>>> {
        rebuild_objective(&self.objective, self.max_delta_step, self.n_targets)
    }
}

/// Margin buffer for `data` (`[row][output]`): the per-output intercepts
/// `base_score` broadcast to every row, overridden by the dataset's
/// per-instance `base_margin` when present (one value per row, or one per row
/// and output). Shared by every model representation that predicts.
pub(crate) fn initial_margins(base_score: &[f32], data: &DMatrix) -> Vec<f32> {
    let n = data.n_rows();
    let k = base_score.len();
    match data.base_margin() {
        Some(bm) if bm.len() == n * k => bm.to_vec(),
        Some(bm) if bm.len() == n => bm.iter().flat_map(|&m| std::iter::repeat_n(m, k)).collect(),
        _ => {
            let mut out = Vec::with_capacity(n * k);
            for _ in 0..n {
                out.extend_from_slice(base_score);
            }
            out
        }
    }
}

/// Check that `data` fits a model with `n_features` inputs and `n_outputs`
/// outputs: matching column count and a per-row or per-row-and-output
/// `base_margin`.
pub(crate) fn validate_prediction_data(
    n_features: usize,
    n_outputs: usize,
    data: &DMatrix,
) -> Result<()> {
    if data.n_cols() != n_features {
        return Err(HessboostError::dimension_mismatch(
            "prediction feature count",
            n_features,
            data.n_cols(),
        ));
    }
    let n = data.n_rows();
    if let Some(margin) = data.base_margin()
        && margin.len() != n
        && margin.len() != n * n_outputs
    {
        return Err(HessboostError::dimension_mismatch(
            "prediction base_margin length",
            n * n_outputs,
            margin.len(),
        ));
    }
    Ok(())
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
        })
    })
}

/// Check that a built-in objective produces the model's `n_outputs`
/// outputs. Loaders call this before returning a model: the prediction
/// transform of a multi-output objective works on `[row][output]` blocks of
/// its own width, so a mismatched width would transform values of
/// neighboring rows together. Other objectives (custom losses) are skipped:
/// they predict margins. A built-in objective that cannot be rebuilt for the
/// stored layout (e.g. a distribution with a label matrix) is a format
/// error, since predicting without its transform would misreport every
/// output. So is a `num_class >= 2` on a built-in objective other than
/// `multi:softmax`/`multi:softprob`: XGBoost reads `num_class` as the
/// multiclass class count and refuses it together with several targets.
pub(crate) fn check_objective_width(
    objective: &ModelObjective,
    max_delta_step: f64,
    num_class: usize,
    n_targets: usize,
    n_outputs: usize,
) -> Result<()> {
    let name = objective.name();
    let multiclass = objective
        .built_in()
        .and_then(Objective::num_class)
        .is_some();
    match rebuild_objective(objective, max_delta_step, n_targets) {
        None => Ok(()),
        Some(Ok(rebuilt)) if rebuilt.n_outputs() == n_outputs => {
            if num_class >= 2 && !multiclass {
                Err(HessboostError::ModelFormat(format!(
                    "num_class {num_class} applies only to multiclass objectives, not `{name}`"
                )))
            } else {
                Ok(())
            }
        }
        Some(Ok(rebuilt)) => Err(HessboostError::ModelFormat(format!(
            "objective `{name}` has {} outputs but the model stores {n_outputs}",
            rebuilt.n_outputs()
        ))),
        Some(Err(e)) => Err(HessboostError::ModelFormat(format!(
            "objective `{name}` cannot be rebuilt from the stored configuration: {e}"
        ))),
    }
}

/// Turn raw margins (`[row][output]`, `n_outputs` wide) of a model with the
/// given objective into predictions in the objective's reported space: the
/// objective's transform (identity for an objective the crate does not
/// implement, e.g. a custom loss, mirroring how XGBoost returns margins
/// then), and for `multi:softmax` the per-row argmax class index encoded as
/// `f32`.
pub(crate) fn transform_model_margins(
    objective: &ModelObjective,
    max_delta_step: f64,
    n_targets: usize,
    n_outputs: usize,
    mut margin: Vec<f32>,
) -> Vec<f32> {
    if let Some(Ok(loss)) = rebuild_objective(objective, max_delta_step, n_targets) {
        loss.pred_transform(&mut margin);
    }
    if objective
        .built_in()
        .is_some_and(Objective::predicts_class_index)
    {
        return margin
            .chunks_exact(n_outputs)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map_or(0.0, |(i, _)| i as f32)
            })
            .collect();
    }
    margin
}

/// Rows per prediction block: the block's feature rows stay in cache while
/// every tree walks them. Must be a multiple of [`LANES`].
const PREDICT_BLOCK_ROWS: usize = 256;
const _: () = assert!(PREDICT_BLOCK_ROWS.is_multiple_of(LANES));

/// Widest CSR matrix that is densified block-by-block for prediction. Wider
/// matrices fall back to per-lookup row scans.
const MAX_DENSIFY_COLS: usize = 4096;

/// A block of consecutive rows exposed as dense feature vectors (`NaN` =
/// missing) for [`CompactForest`] traversal. Dense `NaN`-sentinel matrices are
/// viewed in place. Dense matrices with another sentinel and CSR rows are
/// materialized into a per-block scratch buffer so each node lookup is a single
/// indexed load. Both keep a lane-major copy of the full [`LANES`]-row groups
/// for the batch kernel.
enum RowBlock<'a> {
    View {
        data: &'a [f32],
        n_cols: usize,
        start: usize,
        lanes: Vec<u32>,
    },
    Scratch {
        source: &'a DMatrix,
        n_cols: usize,
        /// Row-major tail rows (those past the last full [`LANES`] group).
        scratch: Vec<f32>,
        lanes: Vec<u32>,
        /// Block row index of the first tail row.
        tail_start: usize,
    },
    /// Very wide sparse rows: route through `DMatrix::get` per lookup.
    Wide { data: &'a DMatrix, start: usize },
}

impl<'a> RowBlock<'a> {
    /// Blocks for batch traversal: wide CSR matrices stay sparse.
    fn new(data: &'a DMatrix) -> Self {
        Self::build(data, MAX_DENSIFY_COLS)
    }

    /// Blocks that are loaded one row at a time and always expose a dense row
    /// (for per-row algorithms such as TreeSHAP whose cost per row already
    /// scales with the feature count).
    fn single_rows(data: &'a DMatrix) -> Self {
        Self::build(data, usize::MAX)
    }

    fn build(data: &'a DMatrix, max_densify_cols: usize) -> Self {
        let n_cols = data.n_cols();
        match data.dense_values() {
            Some(dense) if data.missing().is_nan() => RowBlock::View {
                data: dense,
                n_cols,
                start: 0,
                lanes: Vec::new(),
            },
            None if n_cols > max_densify_cols => RowBlock::Wide { data, start: 0 },
            _ => RowBlock::Scratch {
                source: data,
                n_cols,
                scratch: Vec::new(),
                lanes: Vec::new(),
                tail_start: 0,
            },
        }
    }

    /// Point the block at rows `start..start + rows`.
    fn load(&mut self, start: usize, rows: usize) {
        match self {
            RowBlock::Wide { start: s, .. } => *s = start,
            RowBlock::View {
                data,
                n_cols,
                start: s,
                lanes,
            } => {
                *s = start;
                let n_cols = *n_cols;
                fill_lanes(
                    lanes,
                    &data[start * n_cols..(start + rows) * n_cols],
                    n_cols,
                );
            }
            RowBlock::Scratch {
                source,
                n_cols,
                scratch,
                lanes,
                tail_start,
            } => {
                let n_cols = *n_cols;
                let missing = source.missing();
                let groups = rows / LANES;
                *tail_start = groups * LANES;
                lanes.clear();
                lanes.resize(groups * FEATURE_LANES * n_cols, key(f32::NAN));
                scratch.clear();
                scratch.resize((rows - groups * LANES) * n_cols, f32::NAN);
                // Store feature `f` of block row `r` in its keyed lane slot
                // (the negated key follows `LANES` later) or its tail slot.
                let mut put = |r: usize, f: usize, v: f32| {
                    if r < groups * LANES {
                        let i =
                            (r / LANES) * FEATURE_LANES * n_cols + f * FEATURE_LANES + r % LANES;
                        lanes[i] = key(v);
                        lanes[i + LANES] = key(-v);
                    } else {
                        scratch[(r - groups * LANES) * n_cols + f] = v;
                    }
                };
                if let Some(dense) = source.dense_values() {
                    for r in 0..rows {
                        let src = &dense[(start + r) * n_cols..(start + r + 1) * n_cols];
                        for (f, &v) in src.iter().enumerate() {
                            put(r, f, if v == missing { f32::NAN } else { v });
                        }
                    }
                } else {
                    let (indptr, indices, values) =
                        source.csr_parts().expect("scratch blocks are dense or CSR");
                    for r in 0..rows {
                        let row = start + r;
                        // Reverse order so the first occurrence of a duplicated
                        // column wins, matching `DMatrix::get`.
                        for k in (indptr[row]..indptr[row + 1]).rev() {
                            let v = values[k];
                            let v = if crate::data::is_missing(v, missing) {
                                f32::NAN
                            } else {
                                v
                            };
                            put(r, indices[k] as usize, v);
                        }
                    }
                }
            }
        }
    }

    /// Loaded row `r` as a dense `NaN`-for-missing slice, or `None` for wide
    /// sparse blocks, which are never materialized. Scratch blocks only keep
    /// the tail rows (those past the last full [`LANES`] group) row-major.
    #[inline]
    fn row(&self, r: usize) -> Option<&[f32]> {
        match self {
            RowBlock::View {
                data,
                n_cols,
                start,
                ..
            } => Some(&data[(start + r) * n_cols..(start + r + 1) * n_cols]),
            RowBlock::Scratch {
                n_cols,
                scratch,
                tail_start,
                ..
            } => {
                let i = r
                    .checked_sub(*tail_start)
                    .expect("row-major access to a lane-major scratch row");
                Some(&scratch[i * n_cols..(i + 1) * n_cols])
            }
            RowBlock::Wide { .. } => None,
        }
    }

    /// Value of feature `f` in loaded row `r`, `None` when missing.
    #[inline]
    fn get(&self, r: usize, f: u32) -> Option<f32> {
        let Some(row) = self.row(r) else {
            let RowBlock::Wide { data, start } = self else {
                unreachable!("only wide blocks lack dense rows")
            };
            return data.get(start + r, f as usize);
        };
        let v = row[f as usize];
        if v.is_nan() { None } else { Some(v) }
    }

    /// Original leaf ids of loaded row `r` in trees `0..out.len()`, written to
    /// `out[t]`.
    fn original_leaf_ids_for_row(&self, forest: &CompactForest, r: usize, out: &mut [u32]) {
        match self.row(r) {
            Some(row) => forest.original_leaf_ids_for_row(row, out),
            None => self.wide_row_leaves(forest, r, 0..out.len(), |t, leaf| {
                out[t] = forest.original_id(leaf);
            }),
        }
    }

    /// The wide sparse fallback of the per-row walks: `sink(t, leaf)` with
    /// loaded row `r`'s arena leaf in each tree of `trees`, features looked
    /// up one at a time.
    #[inline]
    fn wide_row_leaves(
        &self,
        forest: &CompactForest,
        r: usize,
        trees: std::ops::Range<usize>,
        mut sink: impl FnMut(usize, u32),
    ) {
        for t in trees {
            sink(t, forest.leaf_id_with(t, |f| self.get(r, f)));
        }
    }

    /// The wide sparse fallback of the block walks: `sink(r, leaf)` with the
    /// arena leaf of each of the first `rows` loaded rows in tree `t`.
    #[inline]
    fn wide_leaves(
        &self,
        forest: &CompactForest,
        t: usize,
        rows: usize,
        mut sink: impl FnMut(usize, u32),
    ) {
        for r in 0..rows {
            sink(r, forest.leaf_id_with(t, |f| self.get(r, f)));
        }
    }

    /// `out[scalar_tree_output(t, parallel, k)] += weight(t) * leaf_value(row
    /// r, tree t)` for the trees `trees` of loaded row `r`, where `parallel`
    /// is the model's `num_parallel_tree`.
    fn accumulate_row(
        &self,
        forest: &CompactForest,
        r: usize,
        trees: std::ops::Range<usize>,
        parallel: usize,
        weight: impl Fn(usize) -> f32,
        out: &mut [f32],
    ) {
        if let Some(row) = self.row(r) {
            forest.accumulate_row(row, trees, parallel, weight, out);
        } else {
            let k = out.len();
            self.wide_row_leaves(forest, r, trees, |t, leaf| {
                out[scalar_tree_output(t, parallel, k)] += weight(t) * forest.leaf_value(leaf);
            });
        }
    }

    /// `out[j] += weight(t) * leaf_vector(row r, tree t)[j]` for the
    /// vector-leaf trees `trees` of loaded row `r`.
    fn accumulate_row_vector(
        &self,
        forest: &CompactForest,
        r: usize,
        trees: std::ops::Range<usize>,
        weight: impl Fn(usize) -> f32,
        out: &mut [f32],
    ) {
        if let Some(row) = self.row(r) {
            forest.accumulate_row_vector(row, trees, weight, out);
        } else {
            let k = out.len();
            self.wide_row_leaves(forest, r, trees, |t, leaf| {
                let w = weight(t);
                for (o, &v) in out.iter_mut().zip(forest.leaf_vector(leaf, k)) {
                    *o += w * v;
                }
            });
        }
    }

    /// The first `rows` loaded rows as a [`LaneBlock`] for the batch kernels,
    /// or `None` for wide sparse blocks.
    #[inline]
    fn lane_block(&self, rows: usize) -> Option<LaneBlock<'_>> {
        let tail_start = rows / LANES * LANES;
        match self {
            RowBlock::View {
                data,
                n_cols,
                start,
                lanes,
            } => Some(LaneBlock {
                lanes,
                tail: &data[(start + tail_start) * n_cols..(start + rows) * n_cols],
                n_cols: *n_cols,
                rows,
            }),
            RowBlock::Scratch {
                n_cols,
                scratch,
                lanes,
                ..
            } => Some(LaneBlock {
                lanes,
                tail: &scratch[..(rows - tail_start) * n_cols],
                n_cols: *n_cols,
                rows,
            }),
            RowBlock::Wide { .. } => None,
        }
    }

    /// `out[r * stride] = original leaf id of row r` in tree `t` over the
    /// loaded rows.
    fn original_leaf_ids(
        &self,
        forest: &CompactForest,
        t: usize,
        rows: usize,
        out: &mut [u32],
        stride: usize,
    ) {
        if let Some(block) = self.lane_block(rows) {
            forest.original_leaf_ids(t, block, out, stride);
        } else {
            self.wide_leaves(forest, t, rows, |r, leaf| {
                out[r * stride] = forest.original_id(leaf);
            });
        }
    }

    /// `out[r * stride + j] += weight * leaf_vector(row r)[j]` (all `stride`
    /// outputs) in vector-leaf tree `t` over the loaded rows.
    fn accumulate_vector(
        &self,
        forest: &CompactForest,
        t: usize,
        rows: usize,
        weight: f32,
        out: &mut [f32],
        stride: usize,
    ) {
        if let Some(block) = self.lane_block(rows) {
            forest.accumulate_vector(t, block, stride, weight, out, stride);
        } else {
            self.wide_leaves(forest, t, rows, |r, leaf| {
                let dst = &mut out[r * stride..(r + 1) * stride];
                for (o, &v) in dst.iter_mut().zip(forest.leaf_vector(leaf, stride)) {
                    *o += weight * v;
                }
            });
        }
    }

    /// `out[r * stride] += weight * leaf_value(row r)` in tree `t` over the
    /// loaded rows.
    fn accumulate(
        &self,
        forest: &CompactForest,
        t: usize,
        rows: usize,
        weight: f32,
        out: &mut [f32],
        stride: usize,
    ) {
        if let Some(block) = self.lane_block(rows) {
            forest.accumulate(t, block, weight, out, stride);
        } else {
            self.wide_leaves(forest, t, rows, |r, leaf| {
                out[r * stride] += weight * forest.leaf_value(leaf);
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BoostedModel;
    use crate::config::TrainingParams;
    use crate::data::DMatrix;
    use crate::error::HessboostError;
    use crate::objective::{Logistic, Objective};
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
        let mut value: serde_json::Value = serde_json::from_str(&model.to_json().unwrap()).unwrap();
        value["n_targets"] = 2.into();
        value["objective"] = "count:poisson".into();
        assert!(matches!(
            BoostedModel::from_json(&value.to_string()),
            Err(HessboostError::ModelFormat(_))
        ));
        value["objective"] = "my:custom".into();
        let custom = BoostedModel::from_json(&value.to_string()).unwrap();
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
        assert!(BoostedModel::from_bytes(&model.to_bytes().unwrap()).is_ok());
        for (weight, bias) in [(f32::INFINITY, 0.0), (0.0, f32::NAN)] {
            let mut corrupt = model.clone();
            let linear = corrupt.linear.as_mut().unwrap();
            linear.weights[0] = weight;
            linear.bias[0] = bias;
            assert!(matches!(
                BoostedModel::from_bytes(&corrupt.to_bytes().unwrap()),
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
        let doc = serde_json::from_str(&model.to_json().unwrap()).unwrap();
        (model, doc)
    }

    /// Deserializing through serde directly validates like `from_json`: an
    /// empty gblinear bias used to load and panic in prediction, and a cyclic
    /// tree used to load and loop forever in traversal.
    #[test]
    fn serde_deserialization_validates_the_model() {
        let (model, mut doc) = gblinear_doc();
        let valid: BoostedModel = serde_json::from_value(doc.clone()).unwrap();
        let d = DMatrix::from_dense(&[1.0, 2.0], 1, 2).unwrap();
        assert_eq!(valid.predict(&d).unwrap(), model.predict(&d).unwrap());
        doc["linear"]["bias"] = serde_json::json!([]);
        assert!(serde_json::from_value::<BoostedModel>(doc.clone()).is_err());
        assert!(matches!(
            BoostedModel::from_json(&doc.to_string()),
            Err(HessboostError::ModelFormat(_))
        ));

        let d = labeled_dense(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 0.0, 1.0, 1.0]);
        let model = train(&TrainingParams::default(), &d, 1).unwrap();
        let mut doc: serde_json::Value = serde_json::from_str(&model.to_json().unwrap()).unwrap();
        assert!(doc["trees"][0]["nodes"].as_array().unwrap().len() > 1);
        doc["trees"][0]["nodes"][0]["left"] = 0.into();
        assert!(serde_json::from_value::<BoostedModel>(doc.clone()).is_err());
        assert!(matches!(
            BoostedModel::from_json(&doc.to_string()),
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
            BoostedModel::from_json(&doc.to_string()),
            Err(HessboostError::ModelFormat(_))
        ));
        let mut stopped = model.clone();
        stopped.set_best_iteration(Some(0));
        assert!(matches!(
            BoostedModel::from_bytes(&stopped.to_bytes().unwrap()),
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
            .objective(Objective::BinaryLogistic(Logistic::default()))
            .build()
            .unwrap();
        let model = train(&params, &d, 1).unwrap();
        let mut doc: serde_json::Value = serde_json::from_str(&model.to_json().unwrap()).unwrap();
        assert!(BoostedModel::from_json(&doc.to_string()).is_ok());
        doc["num_class"] = 2.into();
        assert!(matches!(
            BoostedModel::from_json(&doc.to_string()),
            Err(HessboostError::ModelFormat(_))
        ));
    }
}
