//! XGBoost-format model import and export, targeting the XGBoost 3.4.2
//! schema (identical to 3.4.1's), in both of XGBoost's encodings: JSON text
//! ([`export_xgboost_json`] / [`import_xgboost_json`], XGBoost's `m.json`) and
//! Universal Binary JSON ([`export_xgboost_ubjson`] / [`import_xgboost_ubjson`],
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
//! direction, matching the exact semantics of [`crate::tree::RegTree`].
//! Categorical internal nodes (`split_type[i] == 1`) carry their category set
//! in the tree's `categories` / `categories_nodes` / `categories_segments` /
//! `categories_sizes` arrays.
//!
//! # Scope and caveats
//!
//! Import targets a `gbtree` booster with a scalar, multiclass, or
//! multi-target objective, with scalar-leaf trees (`one_output_per_tree`) or
//! vector-leaf trees (`multi_output_tree`, see
//! [Vector-leaf trees](#vector-leaf-trees)).
//! XGBoost saves `booster=dart` as `gbtree` plus a per-tree
//! `model.weight_drop` array; those weights become the model's DART tree
//! weights on import, and a model with non-unit tree weights writes them back
//! as `weight_drop` on export. Other booster kinds (`gblinear`) yield a clear
//! [`HessboostError::ModelFormat`]. Numeric and categorical splits both
//! round-trip in either direction.
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
//! round-trips through the model's [`ObjectiveParams`]; absent fields take
//! XGBoost's defaults. The alpha lists are XGBoost's array strings
//! (`"[0.1,0.5,0.9]"`, `(..)` also read); `reg:absoluteerror` and
//! `survival:cox` have no block. A supported objective whose parameters do
//! not rebuild it (e.g. an empty or unsorted alpha list) is a format error.
//! Export rejects models whose objective XGBoost cannot load (custom
//! objectives, and hessboost's distributional `dist:*` objectives, which
//! import rejects as well).
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
//! through the objective's inverse link ([`Objective::probs_to_margins`]) and
//! on **export** the margin row is mapped back with
//! [`Objective::margins_to_probs`] (the forward transform, except for
//! `binary:hinge`, whose threshold is not its link). Multiclass objectives
//! (and any objective we cannot reconstruct) pass the values through
//! unchanged, as XGBoost does: softmax's inverse link is the identity, while
//! its forward transform normalizes across classes.
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
//! payloads). [`export_xgboost_ubjson`] does the same with XGBoost's element
//! types: float32 for `split_conditions`, `base_weights`, `loss_changes`,
//! `sum_hessian` (and `leaf_weights`, gblinear `weights`); int32 for
//! `left_children`, `right_children`, `parents`, `categories`,
//! `categories_nodes` and `split_indices` (int64 when a tree's `num_feature`
//! exceeds the int32 range, as in XGBoost); uint8 for `default_left` and
//! `split_type`; int64 for `categories_segments` and `categories_sizes`. The
//! category container's int32 `feature_segments` / `sorted_idx` / `offsets`
//! and its per-column `values` follow XGBoost too. Every other array is a
//! counted generic array, numbers are float32 and integers the narrowest
//! width, again as XGBoost writes them. [`import_xgboost_ubjson`] accepts the
//! optimized and the plain UBJSON container forms alike.

use crate::config::{AftDistribution, ObjectiveParams};
use crate::error::{HessboostError, Result};
use crate::learner::BoostedModel;
use crate::learner::model::ModelSpec;
use crate::model::ubjson::{self, ElementType};
use crate::objective::{Objective, create_objective};
use crate::tree::{Node, RegTree};
use serde_json::{Map, Value, json};

/// Sentinel XGBoost writes for the parent of the root node (`kInvalidNodeId`).
const INVALID_NODE: i32 = i32::MAX;

/// Serialize a [`BoostedModel`] into XGBoost's JSON model schema.
///
/// The result is a pretty-printed JSON string equivalent to what
/// `xgboost.Booster.save_model("m.json")` produces for a `gbtree` model
/// (including DART tree weights as `weight_drop`), and is accepted by
/// [`import_xgboost_json`] as well as upstream XGBoost 3.4.2. See the module
/// docs (above) for the `base_score` space convention.
pub fn export_xgboost_json(model: &BoostedModel) -> Result<String> {
    Ok(serde_json::to_string_pretty(&model_to_value(model)?)?)
}

/// Serialize a [`BoostedModel`] into XGBoost's UBJSON model format.
///
/// The bytes hold the same document as [`export_xgboost_json`], encoded the
/// way `xgboost.Booster.save_model("m.ubj")` encodes it (typed tree arrays;
/// see [UBJSON encoding](self#ubjson-encoding)). They are accepted by
/// [`import_xgboost_ubjson`] and upstream XGBoost 3.4.2.
pub fn export_xgboost_ubjson(model: &BoostedModel) -> Result<Vec<u8>> {
    ubjson::encode(&model_to_value(model)?, &xgboost_typed_array)
}

/// Parse an XGBoost JSON model document into a [`BoostedModel`].
///
/// Accepts `gbtree` boosters, including XGBoost 3.4.1's DART layout
/// (`gbtree` plus `model.weight_drop`). Other booster kinds produce a
/// [`HessboostError::ModelFormat`]. See the module docs (above) for details and
/// the `base_score` space convention.
pub fn import_xgboost_json(json: &str) -> Result<BoostedModel> {
    model_from_value(&serde_json::from_str(json)?)
}

/// Parse an XGBoost UBJSON model (`save_model("m.ubj")` or
/// `save_raw("ubj")`) into a [`BoostedModel`].
///
/// Decoding accepts optimized (typed / counted) and plain UBJSON containers;
/// the decoded document then goes through the same mapping, with the same
/// support and errors, as [`import_xgboost_json`]. Malformed bytes produce a
/// [`HessboostError::ModelFormat`].
pub fn import_xgboost_ubjson(bytes: &[u8]) -> Result<BoostedModel> {
    model_from_value(&ubjson::decode(bytes)?)
}

/// The element type XGBoost stores the array member `key` of `object` with,
/// or `None` for a generic array. Mirrors the `F32Array` / `I32Array` /
/// `U8Array` / `I64Array` members of XGBoost's `RegTree::SaveModel`,
/// `MultiTargetTree::SaveModel`, `GBLinearModel::SaveModel` and
/// `CatContainer::Save`.
fn xgboost_typed_array(key: &str, object: &Map<String, Value>) -> Option<ElementType> {
    Some(match key {
        "split_conditions" | "base_weights" | "loss_changes" | "sum_hessian" | "leaf_weights"
        | "weights" => ElementType::F32,
        "left_children" | "right_children" | "parents" | "categories" | "categories_nodes"
        | "feature_segments" | "sorted_idx" | "offsets" => ElementType::I32,
        "split_indices" => {
            let num_feature = object
                .get("tree_param")
                .and_then(|p| p.get("num_feature"))
                .and_then(scalar_f64)
                .unwrap_or(0.0);
            if num_feature > f64::from(i32::MAX) {
                ElementType::I64
            } else {
                ElementType::I32
            }
        }
        "default_left" | "split_type" => ElementType::U8,
        "categories_segments" | "categories_sizes" => ElementType::I64,
        // A category column: string categories (`offsets` + int8 `values`)
        // or numeric ones tagged with XGBoost's `CatIndexType` code; unsigned
        // types are stored as the same-width signed bit pattern.
        "values" if object.contains_key("offsets") => ElementType::I8,
        "values" => match object.get("type").and_then(Value::as_i64)? {
            7 => ElementType::F32,
            8 => ElementType::F64,
            9 => ElementType::I8,
            10 => ElementType::U8,
            11 | 12 => ElementType::I16,
            13 | 14 => ElementType::I32,
            15 | 16 => ElementType::I64,
            _ => return None,
        },
        _ => return None,
    })
}

/// Build the XGBoost model document for `model`.
fn model_to_value(model: &BoostedModel) -> Result<Value> {
    if model.linear().is_some() {
        return Err(HessboostError::model_format(
            "XGBoost model export does not support gblinear models",
        ));
    }
    if model
        .trees()
        .iter()
        .any(|tree| tree.linear_leaves().is_some())
    {
        return Err(HessboostError::model_format(
            "XGBoost model export cannot represent linear-leaf trees (`linear_tree`); \
             save the model in the native binary or JSON format",
        ));
    }
    let num_feature = model.n_features();
    let num_class = model.num_class();
    let objective = model.objective().to_string();
    reject_extension_objective(&objective)?;
    // XGBoost can only load objectives it knows; a custom objective
    // (`train_with_objective`) has no XGBoost counterpart.
    let objective_impl = model.rebuild_objective().map_err(|_| {
        HessboostError::model_format(format!(
            "objective `{objective}` has no XGBoost equivalent; cannot export"
        ))
    })?;
    let n_trees = model.effective_num_trees();
    let per_iteration = model.trees_per_iteration();

    let trees: Vec<Value> = model.trees()[..n_trees]
        .iter()
        .enumerate()
        .map(|(id, t)| tree_to_json(id, t, num_feature))
        .collect();

    // `tree_info[t]` is the output group tree `t` contributes to; hessboost
    // lays iterations out like XGBoost (`num_parallel_tree` trees per group,
    // groups in order, and every vector-leaf tree in group 0), so the ids and
    // iteration boundaries carry over.
    let tree_info: Vec<Value> = (0..n_trees)
        .map(|t| json!(model.tree_output(t) as i32))
        .collect();
    let iteration_indptr: Vec<Value> = (0..=n_trees / per_iteration)
        .map(|i| json!(i * per_iteration))
        .collect();

    let mut booster_model = json!({
        "gbtree_model_param": {
            "num_parallel_tree": model.num_parallel_tree().to_string(),
            "num_trees": n_trees.to_string(),
        },
        "iteration_indptr": iteration_indptr,
        "tree_info": tree_info,
        "trees": trees,
    });
    // DART: XGBoost 3.4.1 keeps the booster name `gbtree` and stores the
    // per-tree weights alongside the trees.
    if model.has_non_unit_tree_weights() {
        let weight_drop: Vec<Value> = (0..n_trees).map(|t| json!(model.tree_weight(t))).collect();
        booster_model["weight_drop"] = Value::Array(weight_drop);
    }

    let base_score = format_base_score(model.base_scores(), &*objective_impl, num_class);

    let value = json!({
        "version": [3, 4, 2],
        "learner": {
            "attributes": {},
            "feature_names": [],
            "feature_types": [],
            "gradient_booster": {
                "name": "gbtree",
                "model": booster_model,
            },
            "learner_model_param": {
                "base_score": base_score,
                "boost_from_average": "0",
                "num_class": num_class.to_string(),
                "num_feature": num_feature.to_string(),
                // XGBoost counts outputs here (`ObjFunction::Targets`): one
                // per alpha for the alpha-list objectives; multiclass keeps 1.
                "num_target": if num_class >= 2 { model.n_targets() } else { model.n_outputs() }.to_string(),
            },
            "objective": objective_to_json(&objective, num_class, model.objective_params()),
        }
    });

    Ok(value)
}

/// Refuse hessboost's own objectives, which XGBoost does not define: the
/// distributional `dist:*` objectives. Their models are saved in the native
/// binary or JSON formats only.
fn reject_extension_objective(objective: &str) -> Result<()> {
    if crate::objective::DistFamily::from_objective(objective).is_some() {
        return Err(HessboostError::model_format(format!(
            "objective `{objective}` is a hessboost extension that XGBoost models cannot \
             carry; save the model in the native binary or JSON format"
        )));
    }
    Ok(())
}

/// Map an XGBoost model document (decoded from either encoding) to a
/// [`BoostedModel`].
fn model_from_value(root: &Value) -> Result<BoostedModel> {
    let learner = field(root, "learner")?;
    let booster = field(learner, "gradient_booster")?;

    let booster_name = booster
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("gbtree");
    if booster_name != "gbtree" {
        return Err(HessboostError::model_format(format!(
            "unsupported gradient_booster `{booster_name}`: only `gbtree` is supported"
        )));
    }

    let model = field(booster, "model")?;
    let lmp = field(learner, "learner_model_param")?;

    let num_feature = lmp
        .get("num_feature")
        .and_then(scalar_f64)
        .map(|v| v as usize)
        .ok_or_else(|| HessboostError::model_format("missing/invalid `num_feature`"))?;
    let num_class = lmp
        .get("num_class")
        .and_then(scalar_f64)
        .map_or(0, |v| v as usize);
    // XGBoost's `num_target` counts model outputs (`ObjFunction::Targets`):
    // label columns for most objectives, but one output per alpha for the
    // alpha-list objectives, which fit a single label column.
    let num_target = lmp
        .get("num_target")
        .and_then(scalar_f64)
        .map_or(1, |v| v as usize);

    let objective_json = learner.get("objective");
    let objective = objective_json
        .and_then(|o| o.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("reg:squarederror")
        .to_string();
    reject_extension_objective(&objective)?;

    // Parameter blocks come from the file: check them with the same rules as
    // a training configuration before the model rebuilds its objective.
    let objective_params = objective_params_from_json(&objective, objective_json);
    objective_params
        .training_params(&objective, num_class)
        .build()
        .map_err(|e| HessboostError::model_format(format!("invalid objective parameters: {e}")))?;
    let n_targets = match objective.as_str() {
        "reg:quantileerror" | "reg:expectileerror" => 1,
        _ => num_target,
    };
    let objective_impl = build_objective(&objective, num_class, n_targets, &objective_params)?;
    let n_outputs = match &objective_impl {
        Some(objective) => objective.n_outputs(),
        None if num_class >= 2 => num_class,
        None => num_target.max(1),
    };
    if num_class < 2 && num_target.max(1) != n_outputs {
        return Err(HessboostError::model_format(format!(
            "`num_target` {num_target} does not match the {n_outputs} outputs of objective `{objective}`"
        )));
    }

    let trees_json = model
        .get("trees")
        .and_then(Value::as_array)
        .ok_or_else(|| HessboostError::model_format("missing `model.trees` array"))?;
    // `order[i]` is the XGBoost index of hessboost tree `i` (the identity for
    // XGBoost's canonical group-ordered iterations). Vector-leaf trees
    // (`size_leaf_vector > 1`) form the single group 0.
    let vector_leaf = trees_json
        .first()
        .and_then(|t| t.pointer("/tree_param/size_leaf_vector"))
        .and_then(scalar_f64)
        .is_some_and(|k| k > 1.0);
    let groups = if vector_leaf { 1 } else { n_outputs };
    let (order, num_parallel_tree) = iteration_tree_order(model, trees_json.len(), groups)?;
    let mut trees = Vec::with_capacity(trees_json.len());
    for &i in &order {
        let tree = tree_from_json(&trees_json[i])
            .map_err(|e| HessboostError::model_format(format!("tree {i}: {e}")))?;
        trees.push(tree);
    }

    // DART weights (XGBoost 3.4.1 stores them next to the trees under the
    // `gbtree` name); absent for plain gbtree models. Indexed by XGBoost tree
    // position, so they follow their trees through the reordering.
    let tree_weights = match model.get("weight_drop") {
        None => Vec::new(),
        Some(wd) => {
            let entries = wd
                .as_array()
                .ok_or_else(|| HessboostError::model_format("`weight_drop` is not an array"))?;
            if entries.len() != trees.len() {
                return Err(HessboostError::model_format(format!(
                    "`weight_drop` has {} entries for {} trees",
                    entries.len(),
                    trees.len()
                )));
            }
            let weights = entries
                .iter()
                .map(|w| scalar_f64(w).map(|w| w as f32))
                .collect::<Option<Vec<f32>>>()
                .ok_or_else(|| {
                    HessboostError::model_format("`weight_drop` contains a non-numeric entry")
                })?;
            order.iter().map(|&i| weights[i]).collect()
        }
    };

    let base_score = lmp
        .get("base_score")
        .and_then(Value::as_str)
        .ok_or_else(|| HessboostError::model_format("missing/invalid `base_score`"))?;
    let base_margins = parse_base_score(base_score, objective_impl.as_deref(), n_outputs)?;

    let mut imported = BoostedModel::from_parts(
        trees,
        tree_weights,
        base_margins,
        ModelSpec {
            objective_params,
            objective,
            num_class,
            n_outputs,
            n_targets,
            n_features: num_feature,
        },
    );
    imported.set_num_parallel_tree(num_parallel_tree);
    imported.validate_structure()?;
    Ok(imported)
}

// ---------------------------------------------------------------------------
// Tree (de)serialization
// ---------------------------------------------------------------------------

/// Encode one [`RegTree`] as XGBoost's node-indexed array bundle.
fn tree_to_json(id: usize, tree: &RegTree, num_feature: usize) -> Value {
    if tree.is_vector_leaf() {
        return vector_tree_to_json(id, tree, num_feature);
    }
    let nodes = tree.nodes();
    let n = nodes.len();

    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    let mut split_indices = Vec::with_capacity(n);
    let mut split_conditions = Vec::with_capacity(n);
    let mut default_left = Vec::with_capacity(n);
    let mut base_weights = Vec::with_capacity(n);
    let mut loss_changes = Vec::with_capacity(n);
    let mut sum_hessian = Vec::with_capacity(n);
    let mut split_type = Vec::with_capacity(n);
    let mut categories = Vec::<i64>::new();
    let mut categories_nodes = Vec::<i64>::new();
    let mut categories_segments = Vec::<i64>::new();
    let mut categories_sizes = Vec::<i64>::new();
    let mut parents = vec![INVALID_NODE; n];
    for (i, node) in nodes.iter().enumerate() {
        if !node.is_leaf() {
            parents[node.left as usize] = i as i32;
            parents[node.right as usize] = i as i32;
        }
    }

    for (node_id, node) in nodes.iter().enumerate() {
        let categorical = node.is_categorical && !node.is_leaf();
        left.push(if categorical { node.right } else { node.left });
        right.push(if categorical { node.left } else { node.right });
        sum_hessian.push(node.sum_hess);
        split_type.push(u32::from(node.is_categorical));
        if node.is_categorical {
            let cats = &tree.categories()[node.cat_begin as usize..node.cat_end as usize];
            categories_nodes.push(node_id as i64);
            categories_segments.push(categories.len() as i64);
            categories_sizes.push(cats.len() as i64);
            categories.extend(cats.iter().map(|&category| i64::from(category)));
        }
        if node.is_leaf() {
            // XGBoost carries the leaf weight in both arrays for leaves.
            split_indices.push(0u32);
            split_conditions.push(node.leaf_value);
            base_weights.push(node.leaf_value);
            default_left.push(1i32);
            loss_changes.push(0.0f32);
        } else {
            split_indices.push(node.split_feature);
            split_conditions.push(node.split_cond);
            base_weights.push(0.0f32);
            default_left.push(i32::from(if node.is_categorical {
                !node.default_left
            } else {
                node.default_left
            }));
            loss_changes.push(node.split_gain);
        }
    }

    json!({
        "id": id,
        "tree_param": {
            "num_deleted": "0",
            "num_feature": num_feature.to_string(),
            "num_nodes": n.to_string(),
            "size_leaf_vector": "0",
        },
        "left_children": left,
        "right_children": right,
        "parents": parents,
        "split_indices": split_indices,
        "split_conditions": split_conditions,
        "default_left": default_left,
        "base_weights": base_weights,
        "loss_changes": loss_changes,
        "sum_hessian": sum_hessian,
        "split_type": split_type,
        "categories": categories,
        "categories_nodes": categories_nodes,
        "categories_segments": categories_segments,
        "categories_sizes": categories_sizes,
    })
}

/// Encode one vector-leaf [`RegTree`] as XGBoost's `MultiTargetTree` bundle
/// (`MultiTargetTree::SaveModel`): the shared structure in node-indexed
/// arrays, the leaf vectors in `leaf_weights` (`K` values per leaf, leaves in
/// node order) with each leaf's `right_children` entry holding its leaf index,
/// `parents[0] = -1`, and XGBoost's `DftBadValue` (the smallest subnormal) as
/// the split condition of leaves and categorical nodes. Internal weights are
/// not retained, so `base_weights` carries the leaf vectors and zeros for
/// internal nodes, as the scalar export does.
fn vector_tree_to_json(id: usize, tree: &RegTree, num_feature: usize) -> Value {
    const DFT_BAD_VALUE: f32 = f32::from_bits(1);
    let nodes = tree.nodes();
    let n = nodes.len();
    let k = tree.size_leaf_vector();
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    let mut parents = vec![-1i32; n];
    let mut split_indices = Vec::with_capacity(n);
    let mut split_conditions = Vec::with_capacity(n);
    let mut default_left = Vec::with_capacity(n);
    let mut base_weights = Vec::with_capacity(n * k);
    let mut leaf_weights = Vec::new();
    let mut loss_changes = Vec::with_capacity(n);
    let mut sum_hessian = Vec::with_capacity(n);
    let mut split_type = Vec::with_capacity(n);
    let mut categories = Vec::<i64>::new();
    let mut categories_nodes = Vec::<i64>::new();
    let mut categories_segments = Vec::<i64>::new();
    let mut categories_sizes = Vec::<i64>::new();
    for (i, node) in nodes.iter().enumerate() {
        if !node.is_leaf() {
            parents[node.left as usize] = i as i32;
            parents[node.right as usize] = i as i32;
        }
    }
    let mut n_leaves = 0i32;
    for (node_id, node) in nodes.iter().enumerate() {
        sum_hessian.push(node.sum_hess);
        split_type.push(u32::from(node.is_categorical));
        if node.is_leaf() {
            left.push(-1);
            right.push(n_leaves);
            n_leaves += 1;
            split_indices.push(0u32);
            split_conditions.push(DFT_BAD_VALUE);
            default_left.push(0i32);
            loss_changes.push(0.0f32);
            base_weights.extend_from_slice(tree.leaf_vector(node_id));
            leaf_weights.extend_from_slice(tree.leaf_vector(node_id));
            continue;
        }
        base_weights.extend(std::iter::repeat_n(0.0f32, k));
        split_indices.push(node.split_feature);
        loss_changes.push(node.split_gain);
        if node.is_categorical {
            // XGBoost sends the category set right; hessboost keeps it left.
            let cats = &tree.categories()[node.cat_begin as usize..node.cat_end as usize];
            categories_nodes.push(node_id as i64);
            categories_segments.push(categories.len() as i64);
            categories_sizes.push(cats.len() as i64);
            categories.extend(cats.iter().map(|&category| i64::from(category)));
            left.push(node.right);
            right.push(node.left);
            split_conditions.push(DFT_BAD_VALUE);
            default_left.push(i32::from(!node.default_left));
        } else {
            left.push(node.left);
            right.push(node.right);
            split_conditions.push(node.split_cond);
            default_left.push(i32::from(node.default_left));
        }
    }
    json!({
        "id": id,
        "tree_param": {
            "num_deleted": "0",
            "num_feature": num_feature.to_string(),
            "num_nodes": n.to_string(),
            "size_leaf_vector": k.to_string(),
        },
        "left_children": left,
        "right_children": right,
        "parents": parents,
        "split_indices": split_indices,
        "split_conditions": split_conditions,
        "default_left": default_left,
        "base_weights": base_weights,
        "leaf_weights": leaf_weights,
        "loss_changes": loss_changes,
        "sum_hessian": sum_hessian,
        "split_type": split_type,
        "categories": categories,
        "categories_nodes": categories_nodes,
        "categories_segments": categories_segments,
        "categories_sizes": categories_sizes,
    })
}

/// The leaf vectors of an XGBoost `MultiTargetTree` bundle, laid out
/// `[node][output]` (zeros for internal nodes): leaf `i`'s vector is
/// `leaf_weights[right_children[i] * k..][..k]`.
fn vector_leaves(tj: &Value, left: &[i32], right: &[i32], k: usize) -> Result<Vec<f32>> {
    let leaf_weights = arr(tj, "leaf_weights", scalar_f64)
        .ok_or_else(|| HessboostError::missing_field("leaf_weights"))?;
    let mut out = vec![0.0f32; left.len() * k];
    for (i, (&l, &r)) in left.iter().zip(right).enumerate() {
        if l != -1 {
            continue;
        }
        let slot = usize::try_from(r)
            .ok()
            .filter(|&slot| (slot + 1) * k <= leaf_weights.len())
            .ok_or_else(|| {
                HessboostError::model_format(format!("leaf {i} has an invalid leaf index {r}"))
            })?;
        for (dst, &w) in out[i * k..(i + 1) * k]
            .iter_mut()
            .zip(&leaf_weights[slot * k..(slot + 1) * k])
        {
            *dst = w as f32;
        }
    }
    Ok(out)
}

/// Decode one XGBoost tree object into a [`RegTree`].
fn tree_from_json(tj: &Value) -> Result<RegTree> {
    let left = i32_arr(tj, "left_children")
        .ok_or_else(|| HessboostError::missing_field("left_children"))?;
    let n = left.len();
    if n == 0 {
        return Err(HessboostError::model_format("tree contains no nodes"));
    }

    let right = i32_arr(tj, "right_children")
        .ok_or_else(|| HessboostError::missing_field("right_children"))?;
    if right.len() != n {
        return Err(HessboostError::model_format(
            "child arrays have different lengths",
        ));
    }

    let split_type = strict_nonnegative_integer_array(tj, "split_type")?;
    if !split_type.is_empty() && split_type.len() != n {
        return Err(HessboostError::model_format(
            "`split_type` length does not match the node count",
        ));
    }
    let categories = strict_nonnegative_integer_array(tj, "categories")?;
    let category_nodes = strict_nonnegative_integer_array(tj, "categories_nodes")?;
    let category_segments = strict_nonnegative_integer_array(tj, "categories_segments")?;
    let category_sizes = strict_nonnegative_integer_array(tj, "categories_sizes")?;
    if category_nodes.len() != category_segments.len()
        || category_nodes.len() != category_sizes.len()
    {
        return Err(HessboostError::model_format(
            "categorical node, segment, and size arrays have different lengths",
        ));
    }
    let mut node_categories: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut seen_category_node = vec![false; n];
    for slot in 0..category_nodes.len() {
        let node = category_nodes[slot] as usize;
        let begin = category_segments[slot] as usize;
        let size = category_sizes[slot] as usize;
        let end = begin
            .checked_add(size)
            .ok_or_else(|| HessboostError::model_format("categorical segment overflow"))?;
        if node >= n || seen_category_node[node] || size == 0 || end > categories.len() {
            return Err(HessboostError::model_format("invalid categorical arrays"));
        }
        seen_category_node[node] = true;
        node_categories[node] = categories[begin..end]
            .iter()
            .map(|&v| {
                u32::try_from(v).map_err(|_| HessboostError::model_format("category exceeds u32"))
            })
            .collect::<Result<Vec<_>>>()?;
    }

    let split_indices = arr_or_empty(tj, "split_indices");
    let split_conditions = arr(tj, "split_conditions", scalar_f64)
        .ok_or_else(|| HessboostError::missing_field("split_conditions"))?;
    let default_left = arr_or_empty(tj, "default_left");
    let base_weights = arr_or_empty(tj, "base_weights");
    let sum_hessian = arr_or_empty(tj, "sum_hessian");
    let loss_changes = arr_or_empty(tj, "loss_changes");

    let at = |v: &[f64], i: usize| v.get(i).copied().unwrap_or(0.0);
    let size_leaf_vector = tj
        .pointer("/tree_param/size_leaf_vector")
        .and_then(scalar_f64)
        .map_or(0, |k| k as usize);
    let leaf_vectors = if size_leaf_vector > 1 {
        vector_leaves(tj, &left, &right, size_leaf_vector)?
    } else {
        Vec::new()
    };

    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        let sum_hess = at(&sum_hessian, i) as f32;
        if left[i] == -1 && size_leaf_vector > 1 {
            // Vector leaf: the weights live in `leaf_vectors`.
            nodes.push(Node::leaf(0.0, sum_hess));
        } else if left[i] == -1 {
            // Leaf: prefer split_conditions, fall back to base_weights.
            let leaf_value = split_conditions
                .get(i)
                .copied()
                .or_else(|| base_weights.get(i).copied())
                .unwrap_or(0.0) as f32;
            nodes.push(Node::leaf(leaf_value, sum_hess));
        } else {
            if left[i] < 0 || right[i] < 0 || left[i] as usize >= n || right[i] as usize >= n {
                return Err(HessboostError::model_format(format!(
                    "node {i} has an invalid child index"
                )));
            }
            let is_categorical = split_type.get(i).copied().unwrap_or(0) != 0;
            if is_categorical && !seen_category_node[i] {
                return Err(HessboostError::model_format(format!(
                    "categorical node {i} has no category segment"
                )));
            }
            nodes.push(Node {
                split_feature: at(&split_indices, i) as u32,
                split_cond: at(&split_conditions, i) as f32,
                default_left: if is_categorical {
                    at(&default_left, i) == 0.0
                } else {
                    at(&default_left, i) != 0.0
                },
                left: if is_categorical { right[i] } else { left[i] },
                right: if is_categorical { left[i] } else { right[i] },
                leaf_value: 0.0,
                sum_hess,
                split_gain: at(&loss_changes, i) as f32,
                is_categorical,
                cat_begin: 0,
                cat_end: 0,
            });
        }
    }

    // Build RegTree through serde, flattening category lists in node order.
    let mut flat_categories: Vec<u32> = Vec::new();
    for (i, cats) in node_categories.iter().enumerate() {
        if !cats.is_empty() {
            nodes[i].cat_begin = flat_categories.len() as u32;
            flat_categories.extend(cats);
            nodes[i].cat_end = flat_categories.len() as u32;
        }
    }
    let tree: RegTree = serde_json::from_value(json!({
        "nodes": nodes,
        "categories": flat_categories,
        "size_leaf_vector": if size_leaf_vector > 1 { size_leaf_vector } else { 0 },
        "leaf_vectors": leaf_vectors,
    }))?;
    Ok(tree)
}

// ---------------------------------------------------------------------------
// base_score link handling
// ---------------------------------------------------------------------------

/// Reconstruct the objective from name, `num_class`, `n_targets` (label
/// columns), and the file's parameter blocks, if it is one we support. A
/// supported objective that rejects that configuration (too many targets,
/// an invalid alpha list) is a format error rather than a silently
/// untransformed model.
fn build_objective(
    name: &str,
    num_class: usize,
    n_targets: usize,
    params: &ObjectiveParams,
) -> Result<Option<Box<dyn Objective>>> {
    let params = params.training_params(name, num_class).build_unchecked();
    match create_objective(&params, n_targets) {
        Ok(objective) => Ok(Some(objective)),
        Err(HessboostError::Unknown { .. }) => Ok(None),
        Err(error) if n_targets > 1 => Err(HessboostError::model_format(format!(
            "`num_target` {n_targets}: {error}"
        ))),
        Err(error) => Err(HessboostError::model_format(format!(
            "objective `{name}`: {error}"
        ))),
    }
}

/// Render the per-output margin intercepts as XGBoost 3.x's `base_score`
/// vector string, `"[v0,v1,...]"`, in the space XGBoost stores it in: the
/// objective's inverse intercept link over the whole row
/// ([`Objective::margins_to_probs`]). Multiclass (softmax) values pass
/// through unchanged, since XGBoost's softmax inverse link is the identity
/// while its transform normalizes across classes.
fn format_base_score(margins: &[f32], objective: &dyn Objective, num_class: usize) -> String {
    let mut stored = margins.to_vec();
    if num_class < 2 {
        objective.margins_to_probs(&mut stored);
    }
    let entries: Vec<String> = stored.iter().map(f32::to_string).collect();
    format!("[{}]", entries.join(","))
}

/// Parse XGBoost 3.x's `base_score` vector string (`"[5E-1]"`,
/// `"[a,b,c]"`) into per-output margin intercepts. One entry applies to every
/// output (XGBoost `HandleOldFormat`); otherwise the length must equal
/// `n_outputs`. The vector is mapped through the objective's inverse link
/// (values pass through unchanged for an objective we cannot reconstruct).
fn parse_base_score(
    stored: &str,
    objective: Option<&dyn Objective>,
    n_outputs: usize,
) -> Result<Vec<f32>> {
    let invalid = || HessboostError::model_format(format!("invalid `base_score` `{stored}`"));
    let inner = stored
        .trim()
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(invalid)?;
    let values = inner
        .split(',')
        .map(|v| v.trim().parse::<f32>().ok())
        .collect::<Option<Vec<f32>>>()
        .ok_or_else(invalid)?;
    let mut values = match values.len() {
        1 => vec![values[0]; n_outputs],
        len if len == n_outputs => values,
        len => {
            return Err(HessboostError::model_format(format!(
                "`base_score` has {len} entries for {n_outputs} outputs"
            )));
        }
    };
    if let Some(obj) = objective {
        obj.probs_to_margins(&mut values);
    }
    Ok(values)
}

// ---------------------------------------------------------------------------
// Objective parameter blocks
// ---------------------------------------------------------------------------

/// `(block, key)` under `learner.objective` where XGBoost 3.4.1 keeps each
/// retained parameter (`SaveConfig` of the objective owning it).
const SCALE_POS_WEIGHT: (&str, &str) = ("reg_loss_param", "scale_pos_weight");
const MAX_DELTA_STEP: (&str, &str) = ("poisson_regression_param", "max_delta_step");
const TWEEDIE_VARIANCE_POWER: (&str, &str) = ("tweedie_regression_param", "tweedie_variance_power");
const HUBER_SLOPE: (&str, &str) = ("pseudo_huber_param", "huber_slope");
const LAMBDARANK_NUM_PAIR: (&str, &str) = ("lambdarank_param", "lambdarank_num_pair_per_sample");
const SOFTMAX_NUM_CLASS: (&str, &str) = ("softmax_multiclass_param", "num_class");
const QUANTILE_ALPHA: (&str, &str) = ("quantile_loss_param", "quantile_alpha");
const EXPECTILE_ALPHA: (&str, &str) = ("expectile_loss_param", "expectile_alpha");

/// Encode an alpha list as XGBoost's `ParamArray<float>` string, a JSON
/// array of the `f32` values (`"[0.1,0.5,0.9]"`).
fn format_param_array(values: &[f64]) -> String {
    let entries: Vec<String> = values.iter().map(|&v| (v as f32).to_string()).collect();
    format!("[{}]", entries.join(","))
}

/// Decode XGBoost's `ParamArray<float>` string: a JSON array or a single
/// number, with `(..)` accepted for `[..]` like XGBoost's reader. Entries are
/// rounded to `f32` as XGBoost stores them. `None` for anything else.
fn parse_param_array(text: &str) -> Option<Vec<f64>> {
    let text = text.trim();
    let text = match text.strip_prefix('(').and_then(|t| t.strip_suffix(')')) {
        Some(inner) => format!("[{inner}]"),
        None => text.to_string(),
    };
    let as_f32 = |v: &Value| v.as_f64().map(|v| f64::from(v as f32));
    match serde_json::from_str::<Value>(&text).ok()? {
        Value::Array(entries) => entries.iter().map(as_f32).collect(),
        number @ Value::Number(_) => as_f32(&number).map(|v| vec![v]),
        _ => None,
    }
}
const AFT_LOSS_PARAM: &str = "aft_loss_param";

/// Build the `objective` sub-document with the parameter block XGBoost 3.4.1
/// writes for each objective (its `SaveConfig`), so upstream XGBoost accepts
/// the file. The retained [`ObjectiveParams`] are written as XGBoost's
/// stringified numbers; LambdaRank parameters the model does not retain are
/// written at XGBoost's defaults, and the alpha lists as XGBoost's array
/// strings. Objectives without parameters (`reg:squaredlogerror`,
/// `binary:hinge`, `reg:absoluteerror`) write their name only.
fn objective_to_json(objective: &str, num_class: usize, params: &ObjectiveParams) -> Value {
    let mut out = Map::with_capacity(2);
    out.insert("name".to_string(), Value::String(objective.to_string()));
    match objective {
        // `CoxRegression::SaveConfig` writes the name only.
        "survival:cox" => return Value::Object(out),
        "survival:aft" => {
            let distribution = match params.aft_loss_distribution {
                AftDistribution::Normal => "normal",
                AftDistribution::Logistic => "logistic",
                AftDistribution::Extreme => "extreme",
            };
            let mut fields = Map::with_capacity(2);
            fields.insert(
                "aft_loss_distribution".to_string(),
                Value::String(distribution.to_string()),
            );
            fields.insert(
                "aft_loss_distribution_scale".to_string(),
                Value::String(params.aft_loss_distribution_scale.to_string()),
            );
            out.insert(AFT_LOSS_PARAM.to_string(), Value::Object(fields));
            return Value::Object(out);
        }
        _ => {}
    }
    let ((block, key), value) = match objective {
        "reg:squaredlogerror" | "binary:hinge" | "reg:absoluteerror" => return Value::Object(out),
        "reg:quantileerror" => (QUANTILE_ALPHA, format_param_array(&params.quantile_alpha)),
        "reg:expectileerror" => (EXPECTILE_ALPHA, format_param_array(&params.expectile_alpha)),
        "multi:softmax" | "multi:softprob" => (SOFTMAX_NUM_CLASS, num_class.to_string()),
        "count:poisson" => (MAX_DELTA_STEP, params.max_delta_step.to_string()),
        "reg:tweedie" => (
            TWEEDIE_VARIANCE_POWER,
            params.tweedie_variance_power.to_string(),
        ),
        "reg:pseudohubererror" => (HUBER_SLOPE, params.huber_slope.to_string()),
        "rank:pairwise" | "rank:ndcg" | "rank:map" => (
            LAMBDARANK_NUM_PAIR,
            params.lambdarank_num_pair_per_sample.to_string(),
        ),
        _ => (SCALE_POS_WEIGHT, params.scale_pos_weight.to_string()),
    };
    let mut fields = Map::new();
    if block == LAMBDARANK_NUM_PAIR.0 {
        // The LambdaRank settings the model does not retain, at XGBoost's
        // defaults (the pairing that hessboost implements is `topk`).
        for (k, v) in [
            ("lambdarank_bias_norm", "1"),
            ("lambdarank_normalization", "1"),
            ("lambdarank_pair_method", "topk"),
            ("lambdarank_score_normalization", "1"),
            ("lambdarank_unbiased", "0"),
            ("ndcg_exp_gain", "1"),
        ] {
            fields.insert(k.to_string(), Value::String(v.to_string()));
        }
    }
    fields.insert(key.to_string(), Value::String(value));
    out.insert(block.to_string(), Value::Object(fields));
    Value::Object(out)
}

/// Read the objective's parameter block (the inverse of [`objective_to_json`])
/// back into an [`ObjectiveParams`]. Missing blocks or fields keep XGBoost's
/// defaults for `objective` (e.g. `max_delta_step = 0.7` for `count:poisson`).
fn objective_params_from_json(objective: &str, obj: Option<&Value>) -> ObjectiveParams {
    let mut params = ObjectiveParams::defaults_for(objective);
    let Some(obj) = obj else {
        return params;
    };
    let get =
        |(block, key): (&str, &str)| obj.get(block).and_then(|b| b.get(key)).and_then(scalar_f64);
    if let Some(v) = get(SCALE_POS_WEIGHT) {
        params.scale_pos_weight = v;
    }
    if let Some(v) = get(MAX_DELTA_STEP) {
        params.max_delta_step = v;
    }
    if let Some(v) = get(TWEEDIE_VARIANCE_POWER) {
        params.tweedie_variance_power = v;
    }
    if let Some(v) = get(HUBER_SLOPE) {
        params.huber_slope = v;
    }
    // XGBoost writes `u32::MAX` (`LambdaRankParam::NotSet`) when unset; the
    // pair count then follows `lambdarank_pair_method`, whose `topk` default
    // is what `ObjectiveParams::default` already holds.
    if let Some(v) = get(LAMBDARANK_NUM_PAIR)
        && v >= 1.0
        && v != f64::from(u32::MAX)
    {
        params.lambdarank_num_pair_per_sample = v as usize;
    }
    for ((block, key), alpha) in [
        (QUANTILE_ALPHA, &mut params.quantile_alpha),
        (EXPECTILE_ALPHA, &mut params.expectile_alpha),
    ] {
        // An unreadable list stays empty, which the objective then rejects.
        if let Some(text) = obj
            .get(block)
            .and_then(|b| b.get(key))
            .and_then(Value::as_str)
        {
            *alpha = parse_param_array(text).unwrap_or_default();
        }
    }
    if let Some(aft) = obj.get(AFT_LOSS_PARAM) {
        match aft.get("aft_loss_distribution").and_then(Value::as_str) {
            Some("normal") => params.aft_loss_distribution = AftDistribution::Normal,
            Some("logistic") => params.aft_loss_distribution = AftDistribution::Logistic,
            Some("extreme") => params.aft_loss_distribution = AftDistribution::Extreme,
            _ => {}
        }
        if let Some(v) = aft.get("aft_loss_distribution_scale").and_then(scalar_f64) {
            params.aft_loss_distribution_scale = v;
        }
    }
    params
}

// ---------------------------------------------------------------------------
// Tree layout
// ---------------------------------------------------------------------------

/// Map XGBoost's tree layout onto hessboost's (iteration-major, then output,
/// then parallel tree: tree `t` feeds output `(t / num_parallel_tree) %
/// n_outputs`). Returns, for each hessboost tree position, the index of the
/// XGBoost tree that fills it, plus the model's `num_parallel_tree`.
///
/// XGBoost (`GBTreeModel::LoadModel`) stores each boosting iteration as
/// `[g0 × num_parallel_tree, g1 × num_parallel_tree, ...]` with `tree_info[t]`
/// naming tree `t`'s output group, and marks iteration boundaries with
/// `iteration_indptr` (derived as `num_parallel_tree × n_groups` trees per
/// iteration when absent, `MakeIndptr`). That canonical layout maps
/// one-to-one; an iteration whose trees are tagged out of group order is
/// regrouped, keeping each group's order, so every group's sum -- and hence
/// every prediction -- is unchanged. Iterations whose groups have unequal
/// tree counts, or whose forest size differs from another iteration's,
/// cannot be expressed and are rejected.
fn iteration_tree_order(
    model: &Value,
    n_trees: usize,
    n_outputs: usize,
) -> Result<(Vec<usize>, usize)> {
    field(model, "tree_info")?;
    let tree_info = strict_nonnegative_integer_array(model, "tree_info")?;
    if tree_info.len() != n_trees {
        return Err(HessboostError::model_format(format!(
            "`tree_info` has {} entries for {n_trees} trees",
            tree_info.len()
        )));
    }
    if let Some(bad) = tree_info.iter().find(|&&g| g >= n_outputs as u64) {
        return Err(HessboostError::model_format(format!(
            "`tree_info` group {bad} out of range for {n_outputs} outputs"
        )));
    }

    let num_parallel_tree = model
        .get("gbtree_model_param")
        .and_then(|p| p.get("num_parallel_tree"))
        .map_or(Some(1.0), scalar_f64)
        .filter(|&v| v >= 1.0 && v.fract() == 0.0)
        .ok_or_else(|| HessboostError::model_format("invalid `num_parallel_tree`"))?
        as usize;
    let indptr = if model.get("iteration_indptr").is_some() {
        let indptr = strict_nonnegative_integer_array(model, "iteration_indptr")?;
        let bounded = indptr.first() == Some(&0)
            && indptr.last() == Some(&(n_trees as u64))
            && indptr.windows(2).all(|w| w[0] <= w[1]);
        if !bounded {
            return Err(HessboostError::model_format(
                "`iteration_indptr` must run monotonically from 0 to the number of trees",
            ));
        }
        indptr.iter().map(|&i| i as usize).collect::<Vec<_>>()
    } else {
        let per_iteration = num_parallel_tree * n_outputs;
        if !n_trees.is_multiple_of(per_iteration) {
            return Err(HessboostError::model_format(format!(
                "{n_trees} trees do not form whole iterations of {per_iteration} \
                 (num_parallel_tree × outputs)"
            )));
        }
        (0..=n_trees / per_iteration)
            .map(|k| k * per_iteration)
            .collect()
    };

    let mut order = Vec::with_capacity(n_trees);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); n_outputs];
    // Trees per group, fixed by the first iteration (the model parameter
    // for a tree-less model).
    let mut per_group: Option<usize> = None;
    for (iteration, bounds) in indptr.windows(2).enumerate() {
        for group in &mut groups {
            group.clear();
        }
        for t in bounds[0]..bounds[1] {
            groups[tree_info[t] as usize].push(t);
        }
        let size = groups[0].len();
        if groups.iter().any(|g| g.len() != size) || *per_group.get_or_insert(size) != size {
            return Err(HessboostError::model_format(format!(
                "iteration {iteration}: outputs have unequal or varying tree counts; \
                 layout is not representable"
            )));
        }
        for group in &groups {
            order.extend_from_slice(group);
        }
    }
    match per_group {
        Some(0) => Err(HessboostError::model_format(
            "`iteration_indptr` contains an empty iteration",
        )),
        Some(size) => Ok((order, size)),
        None => Ok((order, num_parallel_tree)),
    }
}

/// Fetch a required object field, erroring with its name if absent.
fn field<'a>(v: &'a Value, key: &str) -> Result<&'a Value> {
    v.get(key).ok_or_else(|| HessboostError::missing_field(key))
}

/// Coerce a scalar JSON value (number, numeric string, or bool) to `f64`.
/// XGBoost writes learner/tree parameters as strings but arrays as numbers.
fn scalar_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Read a JSON array field, mapping each element through `f`. Returns `None` if
/// the field is missing or is not an array.
fn arr(v: &Value, key: &str, f: fn(&Value) -> Option<f64>) -> Option<Vec<f64>> {
    v.get(key)?
        .as_array()
        .map(|a| a.iter().map(|e| f(e).unwrap_or(0.0)).collect())
}

/// Read a JSON array field as `i32`s; `None` if the field is missing or is not
/// an array.
fn i32_arr(v: &Value, key: &str) -> Option<Vec<i32>> {
    arr(v, key, scalar_f64).map(|a| a.iter().map(|&x| x as i32).collect())
}

/// Read a JSON array field, defaulting to an empty vector when absent.
fn arr_or_empty(v: &Value, key: &str) -> Vec<f64> {
    arr(v, key, scalar_f64).unwrap_or_default()
}

/// Read a JSON array whose entries are finite, non-negative integers.
fn strict_nonnegative_integer_array(v: &Value, key: &str) -> Result<Vec<u64>> {
    let Some(value) = v.get(key) else {
        return Ok(Vec::new());
    };
    let entries = value
        .as_array()
        .ok_or_else(|| HessboostError::model_format(format!("`{key}` is not an array")))?;
    entries
        .iter()
        .map(|entry| {
            let value = scalar_f64(entry).ok_or_else(|| {
                HessboostError::model_format(format!("`{key}` contains a non-numeric entry"))
            })?;
            if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > u64::MAX as f64
            {
                return Err(HessboostError::model_format(format!(
                    "`{key}` contains an invalid integer {value}"
                )));
            }
            Ok(value as u64)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BoosterKind, TrainingParams};
    use crate::data::{DMatrix, FeatureType};
    use crate::learner::train;

    /// Train a small squared-error model on a noisy nonlinear signal.
    fn reg_model() -> (BoostedModel, DMatrix) {
        let n = 120;
        let mut x = Vec::with_capacity(n * 2);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let a = i as f32 / n as f32;
            let b = ((i * 7) % n) as f32 / n as f32;
            x.push(a);
            x.push(b);
            y.push(2.0 * a - 3.0 * b + if a > 0.5 { 1.0 } else { -1.0 });
        }
        let d = DMatrix::from_dense(&x, n, 2)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        (train(&params, &d, 15).unwrap(), d)
    }

    #[test]
    fn roundtrip_reg_preserves_predictions() {
        let (model, d) = reg_model();
        let before = model.predict(&d).unwrap();

        let json = export_xgboost_json(&model).unwrap();
        let restored = import_xgboost_json(&json).unwrap();
        let after = restored.predict(&d).unwrap();

        assert_eq!(restored.num_trees(), model.num_trees());
        assert_eq!(restored.n_features(), model.n_features());
        assert_eq!(restored.objective(), model.objective());
        assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(&after) {
            assert!((a - b).abs() < 1e-5, "pred drift: {a} vs {b}");
        }
    }

    #[test]
    fn roundtrip_binary_preserves_predictions() {
        // Binary logistic exercises the prob<->margin base_score link.
        let n = 80;
        let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let y: Vec<f32> = x.iter().map(|&v| f32::from(v > 0.4)).collect();
        let d = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("binary:logistic")
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 20).unwrap();
        let before = model.predict(&d).unwrap();

        let json = export_xgboost_json(&model).unwrap();
        let restored = import_xgboost_json(&json).unwrap();
        assert_eq!(restored.objective(), "binary:logistic");
        // base_score should round-trip through the logit/sigmoid link.
        assert!((restored.base_score() - model.base_score()).abs() < 1e-4);
        let after = restored.predict(&d).unwrap();
        for (a, b) in before.iter().zip(&after) {
            assert!((a - b).abs() < 1e-5, "pred drift: {a} vs {b}");
        }
    }

    /// A minimal, hand-written XGBoost 3.x stump: feature 0 with threshold
    /// 1.5, left leaf +10, right leaf -10, `base_score` `[0]` (raw margin).
    /// `objective`'s document is the 3.4.1 shape for `reg:squarederror`.
    fn hand_stump_json() -> &'static str {
        r#"{
          "version": [3, 4, 1],
          "learner": {
            "gradient_booster": {
              "name": "gbtree",
              "model": {
                "gbtree_model_param": {"num_parallel_tree": "1", "num_trees": "1"},
                "iteration_indptr": [0, 1],
                "tree_info": [0],
                "trees": [{
                  "id": 0,
                  "tree_param": {"num_nodes": "3", "num_feature": "1", "size_leaf_vector": "1"},
                  "left_children":  [1, -1, -1],
                  "right_children": [2, -1, -1],
                  "parents":        [2147483647, 0, 0],
                  "split_indices":  [0, 0, 0],
                  "split_conditions": [1.5, 10.0, -10.0],
                  "default_left":   [1, 0, 0],
                  "base_weights":   [0.0, 10.0, -10.0],
                  "loss_changes":   [42.0, 0.0, 0.0],
                  "sum_hessian":    [8.0, 5.0, 3.0],
                  "split_type":     [0, 0, 0]
                }]
              }
            },
            "learner_model_param": {
              "base_score": "[0E0]", "boost_from_average": "1",
              "num_class": "0", "num_feature": "1", "num_target": "1"
            },
            "objective": {"name": "reg:squarederror", "reg_loss_param": {"scale_pos_weight": "1"}}
          }
        }"#
    }

    #[test]
    fn import_hand_written_stump_routes_correctly() {
        let model = import_xgboost_json(hand_stump_json()).unwrap();
        assert_eq!(model.num_trees(), 1);
        assert_eq!(model.n_features(), 1);
        assert_eq!(model.base_score(), 0.0);

        // x=1.0 (< 1.5) -> left leaf +10 ; x=2.0 (>= 1.5) -> right leaf -10.
        let d = DMatrix::from_dense(&[1.0, 2.0], 2, 1).unwrap();
        let margins = model.predict_margin(&d).unwrap();
        assert!((margins[0] - 10.0).abs() < 1e-6, "got {}", margins[0]);
        assert!((margins[1] + 10.0).abs() < 1e-6, "got {}", margins[1]);

        // Missing value follows default_left = true -> left leaf.
        let dm = DMatrix::from_dense(&[f32::NAN], 1, 1).unwrap();
        let mm = model.predict_margin(&dm).unwrap();
        assert!(
            (mm[0] - 10.0).abs() < 1e-6,
            "missing routed wrong: {}",
            mm[0]
        );
    }

    /// Three constant stumps, one per class (`tree_info` round-robin), whose
    /// leaves are all zero, so `predict_margin` exposes the imported
    /// per-class intercepts. `multi:softprob` stores margins directly, so the
    /// vector must come through unchanged.
    fn three_class_json(base_score: &str) -> String {
        let stump = |id: usize| {
            format!(
                r#"{{"id": {id}, "tree_param": {{"num_nodes": "1", "num_feature": "2", "size_leaf_vector": "1"}},
                    "left_children": [-1], "right_children": [-1], "parents": [2147483647],
                    "split_indices": [0], "split_conditions": [0.0], "default_left": [0],
                    "base_weights": [0.0], "loss_changes": [0.0], "sum_hessian": [1.0], "split_type": [0]}}"#
            )
        };
        format!(
            r#"{{
              "version": [3, 4, 1],
              "learner": {{
                "gradient_booster": {{
                  "name": "gbtree",
                  "model": {{
                    "gbtree_model_param": {{"num_parallel_tree": "1", "num_trees": "3"}},
                    "tree_info": [0, 1, 2],
                    "trees": [{}, {}, {}]
                  }}
                }},
                "learner_model_param": {{
                  "base_score": "{base_score}", "boost_from_average": "1",
                  "num_class": "3", "num_feature": "2", "num_target": "1"
                }},
                "objective": {{"name": "multi:softprob", "softmax_multiclass_param": {{"num_class": "3"}}}}
              }}
            }}"#,
            stump(0),
            stump(1),
            stump(2)
        )
    }

    #[test]
    fn import_multiclass_vector_intercept_offsets_each_class() {
        let model = import_xgboost_json(&three_class_json(
            "[5.3293586E-2,-1.3475811E-1,8.146441E-2]",
        ))
        .unwrap();
        let expected = [5.329_358_6E-2f32, -1.347_581_1E-1, 8.146_441E-2];
        assert_eq!(model.base_scores(), &expected);
        let d = DMatrix::from_dense(&[0.0, 0.0, 1.0, 1.0], 2, 2).unwrap();
        let margins = model.predict_margin(&d).unwrap();
        assert_eq!(margins, [expected, expected].concat());

        // A single entry applies to every class (XGBoost's old-format rule).
        let uniform = import_xgboost_json(&three_class_json("[5E-1]")).unwrap();
        assert_eq!(uniform.base_scores(), &[0.5, 0.5, 0.5]);
    }

    #[test]
    fn malformed_base_score_is_rejected() {
        for bad in ["0.5", "[0.1,0.2]", "[a]", "[]", "[0.1,0.2,0.3,0.4]"] {
            let err = import_xgboost_json(&three_class_json(bad)).unwrap_err();
            assert!(
                matches!(err, HessboostError::ModelFormat(_)),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn export_writes_xgboost_3_learner_params() {
        let (model, _) = reg_model();
        let json: Value = serde_json::from_str(&export_xgboost_json(&model).unwrap()).unwrap();
        assert_eq!(json["version"], json!([3, 4, 2]));
        let lmp = &json["learner"]["learner_model_param"];
        assert_eq!(lmp["boost_from_average"], "0");
        assert_eq!(lmp["num_target"], "1");
        let expected = format!("[{}]", model.base_score());
        assert_eq!(lmp["base_score"], expected);
        assert_eq!(
            json["learner"]["objective"],
            json!({"name": "reg:squarederror", "reg_loss_param": {"scale_pos_weight": "1"}})
        );
        assert!(
            json["learner"]["gradient_booster"]["model"]
                .get("weight_drop")
                .is_none()
        );
    }

    #[test]
    fn unsupported_booster_is_rejected() {
        let js = r#"{"learner": {"gradient_booster": {"name": "gblinear"},
                     "learner_model_param": {"num_feature": "3", "base_score": "[0]"}}}"#;
        let err = import_xgboost_json(js).unwrap_err();
        assert!(matches!(err, HessboostError::ModelFormat(_)));
    }

    #[test]
    fn dart_roundtrips_through_weight_drop() {
        let (_, d) = reg_model();
        let params = TrainingParams::builder()
            .booster(BoosterKind::Dart)
            .rate_drop(0.5)
            .max_depth(3)
            .build()
            .unwrap();
        let model = train(&params, &d, 8).unwrap();
        assert!(model.has_non_unit_tree_weights());
        let before = model.predict(&d).unwrap();

        let exported = export_xgboost_json(&model).unwrap();
        let json: Value = serde_json::from_str(&exported).unwrap();
        assert_eq!(json["learner"]["gradient_booster"]["name"], "gbtree");
        let weight_drop = json["learner"]["gradient_booster"]["model"]["weight_drop"]
            .as_array()
            .unwrap();
        assert_eq!(weight_drop.len(), model.num_trees());

        let restored = import_xgboost_json(&exported).unwrap();
        for t in 0..model.num_trees() {
            assert_eq!(restored.tree_weight(t), model.tree_weight(t), "tree {t}");
        }
        assert_eq!(restored.predict(&d).unwrap(), before);
    }

    #[test]
    fn gblinear_export_is_rejected_and_categorical_roundtrips() {
        let (_, d) = reg_model();
        let params = TrainingParams::builder()
            .booster(BoosterKind::GbLinear)
            .build()
            .unwrap();
        let model = train(&params, &d, 3).unwrap();
        assert!(export_xgboost_json(&model).is_err());

        let categories = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let labels = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let categorical = DMatrix::from_dense(&categories, 6, 1)
            .unwrap()
            .with_labels(&labels)
            .unwrap()
            .with_feature_types(&[FeatureType::Categorical])
            .unwrap();
        let params = TrainingParams::builder().max_depth(2).build().unwrap();
        let model = train(&params, &categorical, 3).unwrap();
        assert!(model.trees().iter().any(|tree| tree.node(0).is_categorical));
        let before = model.predict(&categorical).unwrap();
        let restored = import_xgboost_json(&export_xgboost_json(&model).unwrap()).unwrap();
        assert_eq!(restored.predict(&categorical).unwrap(), before);
    }

    /// XGBoost 3.4.2 `save_raw("ubj")` / `save_raw("json")` of one booster
    /// with categorical splits and missing values, generated by:
    ///
    /// ```python
    /// rng = np.random.default_rng(7)
    /// x = rng.random((256, 3), dtype=np.float32)
    /// x[:, 0] = rng.integers(0, 6, 256)
    /// x[rng.random(x.shape) < 0.1] = np.nan
    /// y = ((x[:, 0] % 2 == 1) ^ (x[:, 1] > 0.5)).astype(np.float32)
    /// d = xgb.DMatrix(x, label=y, feature_types=["c", "q", "q"], enable_categorical=True)
    /// b = xgb.train({"max_depth": 2, "objective": "binary:logistic", "nthread": 1,
    ///                "max_cat_to_onehot": 1}, d, num_boost_round=3)
    /// ```
    const XGB_UBJ: &[u8] = include_bytes!("../../tests/data/xgboost-3.4.2-categorical.ubj");
    const XGB_JSON: &str = include_str!("../../tests/data/xgboost-3.4.2-categorical.json");

    #[test]
    fn xgboost_ubjson_reencodes_byte_for_byte() {
        // Decoding XGBoost's own bytes and encoding them again with the typed
        // array table reproduces the file exactly: same markers, integer
        // widths, typed element types, key order and big-endian payloads.
        let document = ubjson::decode(XGB_UBJ).unwrap();
        let reencoded = ubjson::encode(&document, &xgboost_typed_array).unwrap();
        assert!(reencoded == XGB_UBJ, "re-encoded XGBoost UBJSON differs");
    }

    #[test]
    fn xgboost_ubjson_imports_like_its_json_twin() {
        let from_ubj = import_xgboost_ubjson(XGB_UBJ).unwrap();
        let from_json = import_xgboost_json(XGB_JSON).unwrap();
        assert!(
            from_ubj
                .trees()
                .iter()
                .any(|t| t.nodes().iter().any(|n| n.is_categorical && !n.is_leaf()))
        );
        assert_eq!(from_ubj.to_bytes().unwrap(), from_json.to_bytes().unwrap());

        let truncated = &XGB_UBJ[..XGB_UBJ.len() / 2];
        assert!(matches!(
            import_xgboost_ubjson(truncated),
            Err(HessboostError::ModelFormat(_))
        ));
    }

    #[test]
    fn ubjson_export_is_the_json_document_with_typed_tree_arrays() {
        let (_, d) = reg_model();
        let dart = TrainingParams::builder()
            .booster(BoosterKind::Dart)
            .rate_drop(0.5)
            .max_depth(3)
            .build()
            .unwrap();
        let categories = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let categorical = DMatrix::from_dense(&categories, 6, 1)
            .unwrap()
            .with_labels(&[1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
            .unwrap()
            .with_feature_types(&[FeatureType::Categorical])
            .unwrap();
        let shallow = TrainingParams::builder().max_depth(2).build().unwrap();
        for (model, data) in [
            (reg_model().0, &d),
            (train(&dart, &d, 8).unwrap(), &d),
            (train(&shallow, &categorical, 3).unwrap(), &categorical),
        ] {
            let ubj = export_xgboost_ubjson(&model).unwrap();
            // The very document the JSON export prints (compared before text
            // formatting, which `serde_json` does not round-trip bit-exactly).
            assert_eq!(
                ubjson::decode(&ubj).unwrap(),
                model_to_value(&model).unwrap()
            );
            for header in [
                &b"split_conditions[$d#L"[..],
                b"base_weights[$d#L",
                b"loss_changes[$d#L",
                b"sum_hessian[$d#L",
                b"left_children[$l#L",
                b"right_children[$l#L",
                b"parents[$l#L",
                b"split_indices[$l#L",
                b"categories[$l#L",
                b"categories_nodes[$l#L",
                b"default_left[$U#L",
                b"split_type[$U#L",
                b"categories_segments[$L#L",
                b"categories_sizes[$L#L",
            ] {
                assert!(
                    ubj.windows(header.len()).any(|w| w == header),
                    "missing {}",
                    String::from_utf8_lossy(header)
                );
            }
            let restored = import_xgboost_ubjson(&ubj).unwrap();
            assert_eq!(
                restored.predict(data).unwrap(),
                model.predict(data).unwrap()
            );
            let via_json = import_xgboost_json(&export_xgboost_json(&model).unwrap()).unwrap();
            assert_eq!(restored.to_bytes().unwrap(), via_json.to_bytes().unwrap());
        }
    }

    /// A 3-class `gbtree` document of constant stumps whose leaf values are
    /// `leaves`, tagged with `tree_info`, laid out with `num_parallel_tree`
    /// parallel trees and optional `iteration_indptr` / `weight_drop` arrays.
    fn parallel_tree_json(
        num_parallel_tree: usize,
        tree_info: &[usize],
        leaves: &[f32],
        iteration_indptr: Option<&[usize]>,
        weight_drop: Option<&[f32]>,
    ) -> String {
        let trees: Vec<String> = leaves
            .iter()
            .enumerate()
            .map(|(id, leaf)| {
                format!(
                    r#"{{"id": {id}, "tree_param": {{"num_nodes": "1", "num_feature": "1", "size_leaf_vector": "1"}},
                    "left_children": [-1], "right_children": [-1], "parents": [2147483647],
                    "split_indices": [0], "split_conditions": [{leaf}], "default_left": [0],
                    "base_weights": [{leaf}], "loss_changes": [0.0], "sum_hessian": [1.0], "split_type": [0]}}"#
                )
            })
            .collect();
        let extra = |key: &str, values: Option<String>| {
            values.map_or(String::new(), |v| format!(r#""{key}": [{v}],"#))
        };
        let join = |v: &[String]| v.join(", ");
        let indptr = extra(
            "iteration_indptr",
            iteration_indptr.map(|v| join(&v.iter().map(usize::to_string).collect::<Vec<_>>())),
        );
        let weights = extra(
            "weight_drop",
            weight_drop.map(|v| join(&v.iter().map(f32::to_string).collect::<Vec<_>>())),
        );
        format!(
            r#"{{
              "version": [3, 4, 1],
              "learner": {{
                "gradient_booster": {{
                  "name": "gbtree",
                  "model": {{
                    "gbtree_model_param": {{"num_parallel_tree": "{num_parallel_tree}", "num_trees": "{}"}},
                    {indptr}
                    {weights}
                    "tree_info": [{}],
                    "trees": [{}]
                  }}
                }},
                "learner_model_param": {{
                  "base_score": "[0E0]", "boost_from_average": "1",
                  "num_class": "3", "num_feature": "1", "num_target": "1"
                }},
                "objective": {{"name": "multi:softprob", "softmax_multiclass_param": {{"num_class": "3"}}}}
              }}
            }}"#,
            leaves.len(),
            join(&tree_info.iter().map(usize::to_string).collect::<Vec<_>>()),
            join(&trees),
        )
    }

    /// Per-class margins of a one-row prediction through `json`.
    fn class_margins(json: &str) -> Vec<f32> {
        let model = import_xgboost_json(json).unwrap();
        let d = DMatrix::from_dense(&[0.0], 1, 1).unwrap();
        model.predict_margin(&d).unwrap()
    }

    #[test]
    fn import_keeps_parallel_trees_as_iterations() {
        // One iteration, two parallel trees per class: XGBoost order is
        // [c0, c0, c1, c1, c2, c2]; each class must receive its own leaves.
        let leaves = [1.0, 2.0, 10.0, 20.0, 100.0, 200.0];
        let tree_info = [0, 0, 1, 1, 2, 2];
        let derived = parallel_tree_json(2, &tree_info, &leaves, None, None);
        assert_eq!(class_margins(&derived), [3.0, 30.0, 300.0]);
        let model = import_xgboost_json(&derived).unwrap();
        assert_eq!(
            (model.num_parallel_tree(), model.num_boost_rounds()),
            (2, 1)
        );

        // Two iterations marked explicitly by `iteration_indptr`; the first
        // one's trees are tagged out of group order and get regrouped.
        let leaves2 = [
            [2.0, 1.0, 10.0, 200.0, 20.0, 100.0],
            [4.0, 8.0, 40.0, 80.0, 400.0, 800.0],
        ]
        .concat();
        let tree_info2 = [[0, 0, 1, 2, 1, 2], tree_info].concat();
        let explicit = parallel_tree_json(2, &tree_info2, &leaves2, Some(&[0, 6, 12]), None);
        assert_eq!(class_margins(&explicit), [15.0, 150.0, 1500.0]);
        let model = import_xgboost_json(&explicit).unwrap();
        assert_eq!(model.num_boost_rounds(), 2);
        let first = model.slice(0, 1, 1).unwrap();
        let d = DMatrix::from_dense(&[0.0], 1, 1).unwrap();
        assert_eq!(first.predict_margin(&d).unwrap(), [3.0, 30.0, 300.0]);

        // DART weights are indexed by XGBoost position and follow their trees.
        let weights = [1.0, 0.5, 1.0, 0.5, 1.0, 0.5];
        let dart = parallel_tree_json(2, &tree_info, &leaves, None, Some(&weights));
        assert_eq!(class_margins(&dart), [2.0, 20.0, 200.0]);
        // Re-export writes the forest back unchanged.
        let model = import_xgboost_json(&dart).unwrap();
        let doc: Value = serde_json::from_str(&export_xgboost_json(&model).unwrap()).unwrap();
        let booster = &doc["learner"]["gradient_booster"]["model"];
        assert_eq!(booster["gbtree_model_param"]["num_parallel_tree"], "2");
        assert_eq!(booster["tree_info"], json!(tree_info));
        assert_eq!(booster["iteration_indptr"], json!([0, 6]));
        assert_eq!(booster["weight_drop"], json!(weights));

        // Iterations of different forest sizes are unmappable.
        let uneven_info = [0, 0, 1, 2, 1, 2, 0, 1, 2];
        let uneven = parallel_tree_json(2, &uneven_info, &leaves2[..9], Some(&[0, 6, 9]), None);
        let err = import_xgboost_json(&uneven).unwrap_err();
        assert!(matches!(err, HessboostError::ModelFormat(_)), "{err}");

        // Groups with unequal tree counts in one iteration are unmappable.
        let lopsided = parallel_tree_json(2, &[0, 0, 1, 2, 2, 0], &leaves, None, None);
        let err = import_xgboost_json(&lopsided).unwrap_err();
        assert!(matches!(err, HessboostError::ModelFormat(_)), "{err}");
        let missing = parallel_tree_json(1, &tree_info, &leaves, None, None)
            .replace(r#""tree_info": [0, 0, 1, 1, 2, 2],"#, "");
        let err = import_xgboost_json(&missing).unwrap_err();
        assert!(matches!(err, HessboostError::ModelFormat(_)), "{err}");
    }

    /// Export `model`, assert the objective block's `key` equals `expected`,
    /// and hand back the re-imported model.
    fn roundtrip_objective_param(
        model: &BoostedModel,
        block: &str,
        key: &str,
        expected: &str,
    ) -> BoostedModel {
        let exported = export_xgboost_json(model).unwrap();
        let json: Value = serde_json::from_str(&exported).unwrap();
        assert_eq!(
            json["learner"]["objective"][block][key], expected,
            "{block}.{key}"
        );
        import_xgboost_json(&exported).unwrap()
    }

    #[test]
    fn objective_params_roundtrip_through_parameter_blocks() {
        let n = 40;
        let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let counts: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect();
        let data = |labels: &[f32]| {
            DMatrix::from_dense(&x, n, 1)
                .unwrap()
                .with_labels(labels)
                .unwrap()
        };
        let fit = |builder: crate::config::TrainingParamsBuilder, d: &DMatrix| {
            train(&builder.max_depth(2).build().unwrap(), d, 2).unwrap()
        };

        let d = data(&counts);
        let tweedie = fit(
            TrainingParams::builder()
                .objective("reg:tweedie")
                .tweedie_variance_power(1.2),
            &d,
        );
        let back = roundtrip_objective_param(
            &tweedie,
            "tweedie_regression_param",
            "tweedie_variance_power",
            "1.2",
        );
        assert_eq!(back.objective_params().tweedie_variance_power, 1.2);

        let poisson = fit(
            TrainingParams::builder()
                .objective("count:poisson")
                .max_delta_step(0.3),
            &d,
        );
        let back = roundtrip_objective_param(
            &poisson,
            "poisson_regression_param",
            "max_delta_step",
            "0.3",
        );
        assert_eq!(back.objective_params().max_delta_step, 0.3);

        let huber = fit(
            TrainingParams::builder()
                .objective("reg:pseudohubererror")
                .huber_slope(2.5),
            &d,
        );
        let back = roundtrip_objective_param(&huber, "pseudo_huber_param", "huber_slope", "2.5");
        assert_eq!(back.objective_params().huber_slope, 2.5);

        let binary: Vec<f32> = x.iter().map(|&v| f32::from(v > 0.6)).collect();
        let logistic = fit(
            TrainingParams::builder()
                .objective("binary:logistic")
                .scale_pos_weight(3.0),
            &data(&binary),
        );
        let back = roundtrip_objective_param(&logistic, "reg_loss_param", "scale_pos_weight", "3");
        assert_eq!(back.objective_params().scale_pos_weight, 3.0);

        let ranked = data(&counts).with_group_sizes(&[20, 20]).unwrap();
        let ranker = fit(
            TrainingParams::builder()
                .objective("rank:ndcg")
                .lambdarank_num_pair_per_sample(5),
            &ranked,
        );
        let back = roundtrip_objective_param(
            &ranker,
            "lambdarank_param",
            "lambdarank_num_pair_per_sample",
            "5",
        );
        assert_eq!(back.objective_params().lambdarank_num_pair_per_sample, 5);

        // XGBoost's own "not set" sentinel maps to the `topk` default.
        let exported = export_xgboost_json(&ranker).unwrap().replace(
            r#""lambdarank_num_pair_per_sample": "5""#,
            r#""lambdarank_num_pair_per_sample": "4294967295""#,
        );
        let unset = import_xgboost_json(&exported).unwrap();
        assert_eq!(unset.objective_params().lambdarank_num_pair_per_sample, 32);
    }

    /// Alpha lists travel as XGBoost's array strings (either bracket form),
    /// `num_target` counts the per-alpha outputs, and a list that does not
    /// rebuild the objective is a format error, never an untransformed model.
    #[test]
    fn alpha_list_objectives_roundtrip_and_reject_bad_blocks() {
        let n = 40;
        let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let y: Vec<f32> = x.iter().map(|&v| 3.0 * v + (v * 17.0).sin()).collect();
        let d = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("reg:quantileerror")
            .quantile_alpha(vec![0.1, 0.9])
            .max_depth(2)
            .build()
            .unwrap();
        let model = train(&params, &d, 3).unwrap();
        let exported = export_xgboost_json(&model).unwrap();
        let json: Value = serde_json::from_str(&exported).unwrap();
        assert_eq!(
            json["learner"]["objective"],
            json!({"name": "reg:quantileerror", "quantile_loss_param": {"quantile_alpha": "[0.1,0.9]"}})
        );
        assert_eq!(json["learner"]["learner_model_param"]["num_target"], "2");
        let back = import_xgboost_json(&exported).unwrap();
        assert_eq!(back.n_outputs(), 2);
        assert_eq!(back.n_targets(), 1);
        assert_eq!(back.predict(&d).unwrap(), model.predict(&d).unwrap());

        let parenthesized = exported.replace("[0.1,0.9]", "(0.1, 0.9)");
        let back = import_xgboost_json(&parenthesized).unwrap();
        assert_eq!(back.predict(&d).unwrap(), model.predict(&d).unwrap());
        for bad in ["0.5", "[0.9,0.1]", "[]", "nope"] {
            let err = import_xgboost_json(&exported.replace("[0.1,0.9]", bad)).unwrap_err();
            assert!(
                matches!(err, HessboostError::ModelFormat(_)),
                "{bad}: {err}"
            );
        }

        let mae = TrainingParams::builder()
            .objective("reg:absoluteerror")
            .max_depth(2)
            .build()
            .unwrap();
        let exported = export_xgboost_json(&train(&mae, &d, 2).unwrap()).unwrap();
        let json: Value = serde_json::from_str(&exported).unwrap();
        assert_eq!(
            json["learner"]["objective"],
            json!({"name": "reg:absoluteerror"})
        );
    }

    /// `survival:aft` keeps its distribution and scale in `aft_loss_param`
    /// and `survival:cox` writes no parameter block; both store `base_score`
    /// as `exp(margin)` and predict identically after the round trip.
    #[test]
    fn survival_objectives_roundtrip() {
        let n = 40;
        let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let times: Vec<f32> = (0..n).map(|i| 1.0 + (i % 7) as f32).collect();
        let upper: Vec<f32> = times
            .iter()
            .enumerate()
            .map(|(i, &t)| if i % 3 == 0 { f32::INFINITY } else { t })
            .collect();
        let d = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_label_bounds(&times, &upper)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("survival:aft")
            .aft_loss_distribution(AftDistribution::Extreme)
            .aft_loss_distribution_scale(1.5)
            .max_depth(2)
            .build()
            .unwrap();
        let aft = train(&params, &d, 3).unwrap();
        let exported = export_xgboost_json(&aft).unwrap();
        let json: Value = serde_json::from_str(&exported).unwrap();
        assert_eq!(
            json["learner"]["objective"],
            json!({"name": "survival:aft", "aft_loss_param": {
                "aft_loss_distribution": "extreme", "aft_loss_distribution_scale": "1.5"}})
        );
        assert_eq!(
            json["learner"]["learner_model_param"]["base_score"],
            "[0.5]"
        );
        let back = import_xgboost_json(&exported).unwrap();
        assert_eq!(
            back.objective_params().aft_loss_distribution,
            AftDistribution::Extreme
        );
        assert_eq!(back.objective_params().aft_loss_distribution_scale, 1.5);
        assert_eq!(back.predict(&d).unwrap(), aft.predict(&d).unwrap());

        let signed: Vec<f32> = times
            .iter()
            .zip(&upper)
            .map(|(&t, &u)| if u.is_infinite() { -t } else { t })
            .collect();
        let dc = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_labels(&signed)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("survival:cox")
            .max_depth(2)
            .build()
            .unwrap();
        let cox = train(&params, &dc, 3).unwrap();
        let exported = export_xgboost_json(&cox).unwrap();
        let json: Value = serde_json::from_str(&exported).unwrap();
        assert_eq!(
            json["learner"]["objective"],
            json!({"name": "survival:cox"})
        );
        let back = import_xgboost_json(&exported).unwrap();
        assert_eq!(back.base_scores(), cox.base_scores());
        assert_eq!(back.predict(&dc).unwrap(), cox.predict(&dc).unwrap());
    }

    #[test]
    fn custom_objective_export_is_rejected() {
        use crate::learner::train_with_objective;
        use crate::objective::{CustomObjective, GradPair};
        let (_, d) = reg_model();
        let params = TrainingParams::builder()
            .objective("custom:test")
            .max_depth(2)
            .build()
            .unwrap();
        let obj = CustomObjective::new("custom:test", 1, 0.0, "rmse", |preds, labels, w, out| {
            for i in 0..preds.len() {
                let wi = w.map_or(1.0, |ws| ws[i]);
                out[i] = GradPair::new((preds[i] - labels[i]) * wi, wi);
            }
        });
        let model = train_with_objective(&params, &d, 2, &obj).unwrap();
        let err = export_xgboost_json(&model).unwrap_err();
        assert!(matches!(err, HessboostError::ModelFormat(_)), "{err}");
    }
}
