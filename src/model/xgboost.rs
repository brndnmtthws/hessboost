//! XGBoost JSON/UBJSON schema mapping; user docs: `model` module, "XGBoost interchange".

use crate::config::{AftDistribution, ObjectiveParams};
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::model::ModelSpec;
use crate::model::ubjson::{self, ElementType};
use crate::objective::{Objective, create_objective};
use crate::tree::{Node, RegTree};
use serde::Deserialize;
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
    let objective = model.objective();
    reject_extension_objective(objective)?;
    // XGBoost can only load objectives it knows; a custom objective
    // (`Trainer::objective`) has no XGBoost counterpart.
    let objective_impl = model.rebuild_objective().map_err(|_| {
        HessboostError::model_format(format!(
            "objective `{objective}` has no XGBoost equivalent; cannot export"
        ))
    })?;
    // XGBoost refuses `num_class` beside several outputs (`LearnerModelParam`
    // allows `num_class > 1` only with one target), and a model of any other
    // objective with `num_class >= 2` has `num_class` outputs.
    if num_class >= 2 && !is_multiclass(objective) {
        return Err(HessboostError::model_format(format!(
            "`num_class` {num_class} with objective `{objective}` has no XGBoost equivalent; \
             cannot export"
        )));
    }
    let n_trees = model.effective_num_trees();
    let per_iteration = model.trees_per_iteration();

    // A shrunk model's contribution weights go into its leaves (as CatBoost
    // bakes its shrinkage), so XGBoost reads plain gbtree trees; the `f32`
    // product is the one prediction forms, so margins stay bit-identical.
    let baked: Vec<RegTree>;
    let exported = if model.shrinkage().is_some() {
        baked = model.trees()[..n_trees]
            .iter()
            .enumerate()
            .map(|(t, tree)| {
                let mut tree = tree.clone();
                tree.scale_leaves(model.tree_weight(t));
                tree
            })
            .collect();
        &baked[..]
    } else {
        &model.trees()[..n_trees]
    };
    let trees: Vec<Value> = exported
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
    if model.shrinkage().is_none() && model.has_non_unit_tree_weights() {
        let weight_drop: Vec<Value> = (0..n_trees).map(|t| json!(model.tree_weight(t))).collect();
        booster_model["weight_drop"] = Value::Array(weight_drop);
    }

    let base_score = format_base_score(model.base_scores(), &*objective_impl);

    Ok(json!({
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
                "num_target": if is_multiclass(objective) { model.n_targets() } else { model.n_outputs() }.to_string(),
            },
            "objective": objective_to_json(objective, num_class, model.objective_params()),
        }
    }))
}

/// Refuse hessboost's own objectives, which XGBoost does not define: the
/// distributional `dist:*` objectives. Their models are saved in the native
/// binary or JSON formats only.
fn reject_extension_objective(objective: &str) -> Result<()> {
    if crate::objective::distributional::DistFamily::from_objective(objective).is_some() {
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

    let booster_name = optional_str(booster, "name")?.unwrap_or("gbtree");
    if booster_name != "gbtree" {
        return Err(HessboostError::model_format(format!(
            "unsupported gradient_booster `{booster_name}`: only `gbtree` is supported"
        )));
    }

    let model = field(booster, "model")?;
    let lmp = field(learner, "learner_model_param")?;

    let num_feature = lmp
        .get("num_feature")
        .and_then(scalar_count)
        .ok_or_else(|| HessboostError::model_format("missing/invalid `num_feature`"))?;
    let num_class = count_param(lmp, "num_class", 0)?;
    // XGBoost's `num_target` counts model outputs (`ObjFunction::Targets`):
    // label columns for most objectives, but one output per alpha for the
    // alpha-list objectives, which fit a single label column.
    let num_target = count_param(lmp, "num_target", 1)?;

    let objective_json = learner.get("objective");
    if let Some(block) = objective_json
        && !block.is_object()
    {
        return Err(HessboostError::model_format(format!(
            "`objective` is not an object: {block}"
        )));
    }
    let objective = objective_json
        .map(|o| optional_str(o, "name"))
        .transpose()?
        .flatten()
        .unwrap_or("reg:squarederror")
        .to_string();
    reject_extension_objective(&objective)?;

    // Parameter blocks come from the file: check them with the same rules as
    // a training configuration before the model rebuilds its objective.
    let objective_params = objective_params_from_json(&objective, objective_json)?;
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
        let tree = tree_from_json(&trees_json[i], n_outputs)
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
            if !entries.iter().all(|w| scalar_f64(w).is_some()) {
                return Err(HessboostError::model_format(
                    "`weight_drop` contains a non-numeric entry",
                ));
            }
            order
                .iter()
                .map(|&i| scalar_f64(&entries[i]).map_or(0.0, |w| w as f32))
                .collect()
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
///
/// A vector-leaf tree is written as XGBoost's `MultiTargetTree` bundle
/// (`MultiTargetTree::SaveModel`): the leaf vectors in `leaf_weights` (`K`
/// values per leaf, leaves in node order) with each leaf's `right_children`
/// entry holding its leaf index, `parents[0] = -1`, and XGBoost's
/// `DftBadValue` (the smallest subnormal) as the split condition of leaves and
/// categorical nodes. Internal weights are not retained, so `base_weights`
/// carries the leaf values (vectors) and zeros for internal nodes.
fn tree_to_json(id: usize, tree: &RegTree, num_feature: usize) -> Value {
    const DFT_BAD_VALUE: f32 = f32::from_bits(1);
    let vector = tree.is_vector_leaf();
    let nodes = tree.nodes();
    let n = nodes.len();
    let k = tree.size_leaf_vector();

    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
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
    let mut parents = vec![if vector { -1 } else { INVALID_NODE }; n];
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
        // A scalar tree writes the category set of a categorical-flagged leaf
        // too; a vector-leaf tree only those of its split nodes.
        if node.is_categorical && !(vector && node.is_leaf()) {
            let cats = tree.node_categories(node);
            categories_nodes.push(node_id as i64);
            categories_segments.push(categories.len() as i64);
            categories_sizes.push(cats.len() as i64);
            categories.extend(cats.iter().map(|&category| i64::from(category)));
        }
        if node.is_leaf() {
            split_indices.push(0u32);
            loss_changes.push(0.0f32);
            if vector {
                left.push(-1);
                right.push(n_leaves);
                n_leaves += 1;
                split_conditions.push(DFT_BAD_VALUE);
                default_left.push(0i32);
                base_weights.extend_from_slice(tree.leaf_vector(node_id));
                leaf_weights.extend_from_slice(tree.leaf_vector(node_id));
            } else {
                // XGBoost carries the leaf weight in both arrays for leaves.
                left.push(node.left);
                right.push(node.right);
                split_conditions.push(node.leaf_value);
                base_weights.push(node.leaf_value);
                default_left.push(1i32);
            }
            continue;
        }
        split_indices.push(node.split_feature);
        loss_changes.push(node.split_gain);
        base_weights.extend(std::iter::repeat_n(0.0f32, k));
        if node.is_categorical {
            // XGBoost sends the category set right; hessboost keeps it left.
            left.push(node.right);
            right.push(node.left);
            split_conditions.push(if vector {
                DFT_BAD_VALUE
            } else {
                node.split_cond
            });
            default_left.push(i32::from(!node.default_left));
        } else {
            left.push(node.left);
            right.push(node.right);
            split_conditions.push(node.split_cond);
            default_left.push(i32::from(node.default_left));
        }
    }

    let mut bundle = json!({
        "id": id,
        "tree_param": {
            "num_deleted": "0",
            "num_feature": num_feature.to_string(),
            "num_nodes": n.to_string(),
            "size_leaf_vector": if vector { k.to_string() } else { "0".to_owned() },
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
    });
    if vector {
        bundle["leaf_weights"] = json!(leaf_weights);
    }
    bundle
}

/// The leaf vectors of an XGBoost `MultiTargetTree` bundle, laid out
/// `[node][output]` (zeros for internal nodes): leaf `i`'s vector is
/// `leaf_weights[right_children[i] * k..][..k]`. `k` has been checked
/// against the model's outputs. The `[node][output]` storage is sized from
/// the node count, so before allocating it the tree must have the node count
/// of a binary tree over its leaves (`2 * leaves - 1`) and the leaf weights
/// must hold a vector for every leaf: the storage is then smaller than twice
/// the serialized leaf weights.
fn vector_leaves(tj: &Value, left: &[i32], right: &[i32], k: usize) -> Result<Vec<f32>> {
    // Converted once: leaves may share a slot, so a lazy per-leaf read would
    // re-parse string entries once per referencing leaf.
    let leaf_weights = Scalars::required(tj, "leaf_weights")?.to_f32s();
    let n_leaves = left.iter().filter(|&&l| l == -1).count();
    if n_leaves.checked_mul(2) != left.len().checked_add(1) {
        return Err(HessboostError::model_format(format!(
            "tree has {} nodes, but a binary tree with {n_leaves} leaves has {}",
            left.len(),
            (2 * n_leaves).saturating_sub(1)
        )));
    }
    let n_vectors = leaf_weights.len() / k;
    if n_vectors < n_leaves {
        return Err(HessboostError::model_format(format!(
            "`leaf_weights` holds {} values, fewer than {n_leaves} leaf vectors of width {k}",
            leaf_weights.len()
        )));
    }
    let len = left
        .len()
        .checked_mul(k)
        .ok_or_else(|| HessboostError::model_format("leaf vector storage overflows"))?;
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| {
        HessboostError::model_format(format!("cannot allocate {len} leaf vector values"))
    })?;
    out.resize(len, 0.0f32);
    for (i, (&l, &r)) in left.iter().zip(right).enumerate() {
        if l != -1 {
            continue;
        }
        let slot = usize::try_from(r)
            .ok()
            .filter(|&slot| slot < n_vectors)
            .ok_or_else(|| {
                HessboostError::model_format(format!("leaf {i} has an invalid leaf index {r}"))
            })?;
        for (j, dst) in out[i * k..(i + 1) * k].iter_mut().enumerate() {
            *dst = leaf_weights[slot * k + j];
        }
    }
    Ok(out)
}

/// A tree's categorical splits: each node's segment of the `categories`
/// array, validated.
struct TreeCategories {
    values: Vec<u64>,
    /// `(begin, end)` in `values` per node, for the nodes that have one.
    segments: Vec<Option<(usize, usize)>>,
    /// Total segment length.
    total: usize,
}

impl TreeCategories {
    /// Read and check the categorical arrays of a tree of `n` nodes.
    fn read(tj: &Value, n: usize) -> Result<Self> {
        let values = strict_nonnegative_integer_array(tj, "categories")?;
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
        let mut segments = vec![None; n];
        // Every node gets its own copy of its segment, so overlapping
        // segments could expand the array quadratically. XGBoost writes
        // disjoint segments; together they hold at most the whole array.
        let mut total = 0usize;
        for slot in 0..category_nodes.len() {
            let node = category_nodes[slot] as usize;
            let begin = category_segments[slot] as usize;
            let size = category_sizes[slot] as usize;
            let end = begin
                .checked_add(size)
                .ok_or_else(|| HessboostError::model_format("categorical segment overflow"))?;
            total = total.saturating_add(size);
            if node >= n
                || segments[node].is_some()
                || size == 0
                || end > values.len()
                || total > values.len()
            {
                return Err(HessboostError::model_format("invalid categorical arrays"));
            }
            if values[begin..end]
                .iter()
                .any(|&v| u32::try_from(v).is_err())
            {
                return Err(HessboostError::model_format("category exceeds u32"));
            }
            segments[node] = Some((begin, end));
        }
        Ok(TreeCategories {
            values,
            segments,
            total,
        })
    }

    /// Whether node `i` has a category segment.
    fn has(&self, i: usize) -> bool {
        self.segments[i].is_some()
    }

    /// The category lists concatenated in node order, with each node's
    /// range recorded in `nodes`.
    fn flatten_into(&self, nodes: &mut [Node]) -> Vec<u32> {
        let mut flat = Vec::with_capacity(self.total);
        for (node, segment) in nodes.iter_mut().zip(&self.segments) {
            if let Some((begin, end)) = *segment {
                node.cat_begin = flat.len() as u32;
                flat.extend(self.values[begin..end].iter().map(|&v| v as u32));
                node.cat_end = flat.len() as u32;
            }
        }
        flat
    }
}

/// A tree's per-node arrays, as [`decode_nodes`] reads them.
struct NodeColumns<'a> {
    left: &'a [i32],
    right: &'a [i32],
    split_type: &'a [u64],
    split_indices: Scalars<'a>,
    split_conditions: Scalars<'a>,
    default_left: Scalars<'a>,
    base_weights: Scalars<'a>,
    sum_hessian: Scalars<'a>,
    loss_changes: Scalars<'a>,
}

/// `size_leaf_vector`: 0 or 1 for scalar trees and the model's output count
/// for vector-leaf trees (`MultiTargetTree`); anything else, including a
/// width too large to allocate, is malformed.
fn leaf_vector_width(tj: &Value, n_outputs: usize) -> Result<usize> {
    match tj.pointer("/tree_param/size_leaf_vector") {
        None => Ok(0),
        Some(value) => match scalar_f64(value) {
            Some(k) if k == 0.0 || k == 1.0 => Ok(k as usize),
            Some(k) if k == n_outputs as f64 => Ok(n_outputs),
            _ => Err(HessboostError::model_format(format!(
                "`size_leaf_vector` {value} is neither 0, 1, nor the model's {n_outputs} outputs"
            ))),
        },
    }
}

/// The nodes of a tree with `cols`, their categorical splits checked
/// against `categories` (whose ranges [`TreeCategories::flatten_into`]
/// fills in afterwards).
fn decode_nodes(
    cols: &NodeColumns,
    categories: &TreeCategories,
    size_leaf_vector: usize,
) -> Result<Vec<Node>> {
    let NodeColumns {
        left,
        right,
        split_type,
        ..
    } = *cols;
    let n = left.len();
    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        let sum_hess = cols.sum_hessian.at(i) as f32;
        if left[i] == -1 && size_leaf_vector > 1 {
            // Vector leaf: the weights live in `leaf_vectors`.
            nodes.push(Node::leaf(0.0, sum_hess));
        } else if left[i] == -1 {
            // Leaf: prefer split_conditions, fall back to base_weights.
            let leaf_value = cols
                .split_conditions
                .get(i)
                .or_else(|| cols.base_weights.get(i))
                .unwrap_or(0.0) as f32;
            nodes.push(Node::leaf(leaf_value, sum_hess));
        } else {
            if left[i] < 0 || right[i] < 0 || left[i] as usize >= n || right[i] as usize >= n {
                return Err(HessboostError::model_format(format!(
                    "node {i} has an invalid child index"
                )));
            }
            let is_categorical = split_type.get(i).copied().unwrap_or(0) != 0;
            if is_categorical && !categories.has(i) {
                return Err(HessboostError::model_format(format!(
                    "categorical node {i} has no category segment"
                )));
            }
            let default_left = cols.default_left.at(i);
            nodes.push(Node {
                split_feature: cols.split_indices.at(i) as u32,
                split_cond: cols.split_conditions.at(i) as f32,
                default_left: if is_categorical {
                    default_left == 0.0
                } else {
                    default_left != 0.0
                },
                left: if is_categorical { right[i] } else { left[i] },
                right: if is_categorical { left[i] } else { right[i] },
                leaf_value: 0.0,
                sum_hess,
                split_gain: cols.loss_changes.at(i) as f32,
                is_categorical,
                cat_begin: 0,
                cat_end: 0,
            });
        }
    }
    Ok(nodes)
}

/// Decode one XGBoost tree object into a [`RegTree`] of a model with
/// `n_outputs` outputs.
fn tree_from_json(tj: &Value, n_outputs: usize) -> Result<RegTree> {
    let left = Scalars::required(tj, "left_children")?.to_i32s();
    let n = left.len();
    if n == 0 {
        return Err(HessboostError::model_format("tree contains no nodes"));
    }

    let right = Scalars::required(tj, "right_children")?.to_i32s();
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
    let categories = TreeCategories::read(tj, n)?;
    let cols = NodeColumns {
        left: &left,
        right: &right,
        split_type: &split_type,
        split_indices: Scalars::optional(tj, "split_indices"),
        split_conditions: Scalars::required(tj, "split_conditions")?,
        default_left: Scalars::optional(tj, "default_left"),
        base_weights: Scalars::optional(tj, "base_weights"),
        sum_hessian: Scalars::optional(tj, "sum_hessian"),
        loss_changes: Scalars::optional(tj, "loss_changes"),
    };
    let size_leaf_vector = leaf_vector_width(tj, n_outputs)?;
    let mut nodes = decode_nodes(&cols, &categories, size_leaf_vector)?;
    let leaf_vectors = if size_leaf_vector > 1 {
        vector_leaves(tj, &left, &right, size_leaf_vector)?
    } else {
        Vec::new()
    };
    let flat_categories = categories.flatten_into(&mut nodes);
    // Unchecked here: the model's `validate_structure` checks every tree.
    Ok(RegTree::from_parts(
        nodes,
        flat_categories,
        if size_leaf_vector > 1 {
            size_leaf_vector
        } else {
            0
        },
        leaf_vectors,
        None,
    ))
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

/// Whether `objective` is one of XGBoost's multiclass (softmax) objectives,
/// whose outputs are the `num_class` classes.
fn is_multiclass(objective: &str) -> bool {
    matches!(objective, "multi:softmax" | "multi:softprob")
}

/// Render the per-output margin intercepts as XGBoost 3.x's `base_score`
/// vector string, `"[v0,v1,...]"`, in the space XGBoost stores it in: the
/// whole row mapped through [`Objective::margins_to_probs`], the inverse of
/// the objective's `ProbToMargin`. Multiclass (softmax) values pass through
/// unchanged, since XGBoost's softmax `ProbToMargin` is the identity while
/// its transform normalizes across classes.
fn format_base_score(margins: &[f32], objective: &dyn Objective) -> String {
    let mut stored = margins.to_vec();
    if !is_multiclass(objective.name()) {
        objective.margins_to_probs(&mut stored);
    }
    format_float_vector(stored)
}

/// Most outputs a single `base_score` entry is broadcast to. A document's
/// declared output counts (`num_target`, `num_class`) are bounded only by
/// its trees, and a tree-less document has none; past this, the intercepts
/// must be listed one per output (as XGBoost 3.x writes them), so the
/// imported model stays proportional to the document.
const MAX_BROADCAST_OUTPUTS: usize = 1 << 16;

/// Parse XGBoost 3.x's `base_score` vector string (`"[5E-1]"`,
/// `"[a,b,c]"`) into per-output margin intercepts. One entry applies to every
/// output (XGBoost `HandleOldFormat`) of a model with at most
/// [`MAX_BROADCAST_OUTPUTS`] outputs; otherwise the length must equal
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
        1 if n_outputs <= MAX_BROADCAST_OUTPUTS => vec![values[0]; n_outputs],
        len if len == n_outputs => values,
        1 => {
            return Err(HessboostError::model_format(format!(
                "`base_score` has one entry for {n_outputs} outputs; above \
                 {MAX_BROADCAST_OUTPUTS} outputs it must list one entry per output"
            )));
        }
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
/// The `survival:aft` block, holding `aft_loss_distribution` and
/// `aft_loss_distribution_scale`.
const AFT_LOSS_PARAM: &str = "aft_loss_param";

/// Encode `values` as XGBoost's float vector string (`"[0.1,0.5,0.9]"`): the
/// form of `base_score` and of `ParamArray<float>` alpha lists.
fn format_float_vector(values: impl IntoIterator<Item = f32>) -> String {
    use std::fmt::Write;
    let mut out = String::from("[");
    for (i, v) in values.into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        // Writing to a `String` cannot fail.
        let _ = write!(out, "{v}");
    }
    out.push(']');
    out
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

/// Build the `objective` sub-document with the parameter block XGBoost 3.4.1
/// writes for each objective (its `SaveConfig`), so upstream XGBoost accepts
/// the file. The retained [`ObjectiveParams`] are written as XGBoost's
/// stringified numbers; LambdaRank parameters the model does not retain are
/// written at XGBoost's defaults, and the alpha lists as XGBoost's array
/// strings. Objectives without parameters (`reg:squaredlogerror`,
/// `binary:hinge`, `reg:absoluteerror`, `survival:cox`) write their name
/// only.
fn objective_to_json(objective: &str, num_class: usize, params: &ObjectiveParams) -> Value {
    let mut out = Map::with_capacity(2);
    out.insert("name".to_string(), Value::String(objective.to_string()));
    if objective == "survival:aft" {
        // `AftDistribution` serializes as XGBoost's lowercase names.
        let fields = json!({
            "aft_loss_distribution": params.aft_loss_distribution,
            "aft_loss_distribution_scale": params.aft_loss_distribution_scale.to_string(),
        });
        out.insert(AFT_LOSS_PARAM.to_string(), fields);
        return Value::Object(out);
    }
    let ((block, key), value) = match objective {
        "survival:cox" | "reg:squaredlogerror" | "binary:hinge" | "reg:absoluteerror" => {
            return Value::Object(out);
        }
        "reg:quantileerror" => (
            QUANTILE_ALPHA,
            format_float_vector(params.quantile_alpha.iter().map(|&v| v as f32)),
        ),
        "reg:expectileerror" => (
            EXPECTILE_ALPHA,
            format_float_vector(params.expectile_alpha.iter().map(|&v| v as f32)),
        ),
        name if is_multiclass(name) => (SOFTMAX_NUM_CLASS, num_class.to_string()),
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
/// defaults for `objective` (e.g. `max_delta_step = 0.7` for `count:poisson`);
/// a present value that does not parse (or a block that is not an object) is
/// a format error, never a silent default.
fn objective_params_from_json(objective: &str, obj: Option<&Value>) -> Result<ObjectiveParams> {
    let mut params = ObjectiveParams::defaults_for(objective);
    let Some(obj) = obj else {
        return Ok(params);
    };
    let invalid =
        |key: &str, value: &Value| HessboostError::model_format(format!("invalid `{key}` {value}"));
    for (param, value) in [
        (SCALE_POS_WEIGHT, &mut params.scale_pos_weight),
        (MAX_DELTA_STEP, &mut params.max_delta_step),
        (TWEEDIE_VARIANCE_POWER, &mut params.tweedie_variance_power),
        (HUBER_SLOPE, &mut params.huber_slope),
        (
            (AFT_LOSS_PARAM, "aft_loss_distribution_scale"),
            &mut params.aft_loss_distribution_scale,
        ),
    ] {
        if let Some(v) = objective_param(obj, param)? {
            *value = scalar_f64(v).ok_or_else(|| invalid(param.1, v))?;
        }
    }
    // XGBoost writes `u32::MAX` (`LambdaRankParam::NotSet`) when unset; the
    // pair count then follows `lambdarank_pair_method`, whose `topk` default
    // is what `ObjectiveParams::default` already holds. Other counts,
    // including XGBoost's out-of-range `0`, are checked with the parameters.
    if let Some(v) = objective_param(obj, LAMBDARANK_NUM_PAIR)? {
        let count = scalar_count(v).ok_or_else(|| invalid(LAMBDARANK_NUM_PAIR.1, v))?;
        if count != u32::MAX as usize {
            params.lambdarank_num_pair_per_sample = count;
        }
    }
    for (param, alpha) in [
        (QUANTILE_ALPHA, &mut params.quantile_alpha),
        (EXPECTILE_ALPHA, &mut params.expectile_alpha),
    ] {
        if let Some(v) = objective_param(obj, param)? {
            *alpha = v
                .as_str()
                .and_then(parse_param_array)
                .ok_or_else(|| invalid(param.1, v))?;
        }
    }
    let distribution = (AFT_LOSS_PARAM, "aft_loss_distribution");
    if let Some(v) = objective_param(obj, distribution)? {
        params.aft_loss_distribution =
            AftDistribution::deserialize(v).map_err(|_| invalid(distribution.1, v))?;
    }
    Ok(params)
}

/// The value of `key` in the parameter block `block` of the objective
/// document `obj`, `None` when the block or the key is absent. A present
/// block that is not an object is malformed.
fn objective_param<'a>(obj: &'a Value, (block, key): (&str, &str)) -> Result<Option<&'a Value>> {
    match obj.get(block) {
        None => Ok(None),
        Some(Value::Object(fields)) => Ok(fields.get(key)),
        Some(other) => Err(HessboostError::model_format(format!(
            "objective parameter block `{block}` is not an object: {other}"
        ))),
    }
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

    let num_parallel_tree = num_parallel_tree_param(model)?;
    let indptr = iteration_indptr(model, n_trees, num_parallel_tree, n_outputs)?;

    if n_trees == 0 {
        // Every iteration of a tree-less model is empty.
        return if indptr.len() > 1 {
            Err(HessboostError::model_format(
                "`iteration_indptr` contains an empty iteration",
            ))
        } else {
            Ok((Vec::new(), num_parallel_tree))
        };
    }
    // An iteration holds at least one tree per output group, which also
    // bounds the per-group buffers below by the document's tree count.
    if n_outputs > n_trees {
        return Err(HessboostError::model_format(format!(
            "{n_trees} trees cannot hold one iteration of {n_outputs} outputs"
        )));
    }
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

/// `gbtree_model_param.num_parallel_tree`, `1` when absent.
fn num_parallel_tree_param(model: &Value) -> Result<usize> {
    Ok(model
        .get("gbtree_model_param")
        .and_then(|p| p.get("num_parallel_tree"))
        .map_or(Some(1.0), scalar_f64)
        .filter(|&v| v >= 1.0 && v.fract() == 0.0)
        .ok_or_else(|| HessboostError::model_format("invalid `num_parallel_tree`"))?
        as usize)
}

/// The iteration boundaries of `n_trees` trees: `iteration_indptr`, checked
/// to run monotonically from 0 to `n_trees`, or when absent XGBoost's
/// `MakeIndptr` of `num_parallel_tree × n_outputs` trees per iteration.
fn iteration_indptr(
    model: &Value,
    n_trees: usize,
    num_parallel_tree: usize,
    n_outputs: usize,
) -> Result<Vec<usize>> {
    if model.get("iteration_indptr").is_some() {
        let indptr = strict_nonnegative_integer_array(model, "iteration_indptr")?;
        let bounded = indptr.first() == Some(&0)
            && indptr.last() == Some(&(n_trees as u64))
            && indptr.windows(2).all(|w| w[0] <= w[1]);
        if !bounded {
            return Err(HessboostError::model_format(
                "`iteration_indptr` must run monotonically from 0 to the number of trees",
            ));
        }
        return Ok(indptr.iter().map(|&i| i as usize).collect());
    }
    let per_iteration = num_parallel_tree
        .checked_mul(n_outputs)
        .ok_or_else(|| HessboostError::model_format("trees per iteration overflow"))?;
    if !n_trees.is_multiple_of(per_iteration) {
        return Err(HessboostError::model_format(format!(
            "{n_trees} trees do not form whole iterations of {per_iteration} \
             (num_parallel_tree × outputs)"
        )));
    }
    Ok((0..=n_trees / per_iteration)
        .map(|k| k * per_iteration)
        .collect())
}

/// Fetch a required object field, erroring with its name if absent.
fn field<'a>(v: &'a Value, key: &str) -> Result<&'a Value> {
    v.get(key).ok_or_else(|| HessboostError::missing_field(key))
}

/// Read an optional string field, `None` when absent; a present value that
/// is not a string is malformed rather than ignored.
fn optional_str<'a>(v: &'a Value, key: &str) -> Result<Option<&'a str>> {
    v.get(key)
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| HessboostError::model_format(format!("invalid `{key}` {value}")))
        })
        .transpose()
}

/// Read an optional non-negative integer count (XGBoost writes them as
/// numeric strings), `default` when absent. Fractional, negative, or
/// non-representable values are malformed rather than truncated.
fn count_param(v: &Value, key: &str, default: usize) -> Result<usize> {
    let Some(value) = v.get(key) else {
        return Ok(default);
    };
    scalar_count(value)
        .ok_or_else(|| HessboostError::model_format(format!("invalid `{key}` {value}")))
}

/// A scalar JSON value as an exact non-negative integer: an integer number
/// or numeric string, or an integral value `f64` holds exactly (such as
/// `"4.0"`). Parsing every count through `f64` would round large ones.
fn scalar_count(value: &Value) -> Option<usize> {
    /// `2^53`: every integer up to it is exact in `f64`.
    const EXACT: f64 = 9_007_199_254_740_992.0;
    let integer = match value {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.parse::<u64>().ok(),
        _ => None,
    };
    integer
        .or_else(|| {
            scalar_f64(value)
                .filter(|&n| (0.0..=EXACT).contains(&n) && n.fract() == 0.0)
                .map(|n| n as u64)
        })
        .and_then(|n| usize::try_from(n).ok())
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

/// A JSON array field read element by element, each entry coerced with
/// [`scalar_f64`] (`0` for non-scalar entries), without copying the array.
#[derive(Clone, Copy)]
struct Scalars<'a>(&'a [Value]);

impl<'a> Scalars<'a> {
    /// The array field `key` of `v`, empty when absent or not an array.
    fn optional(v: &'a Value, key: &str) -> Self {
        Scalars(
            v.get(key)
                .and_then(Value::as_array)
                .map_or(&[], Vec::as_slice),
        )
    }

    /// The array field `key` of `v`; a missing or non-array field is a
    /// missing-field error naming `key`.
    fn required(v: &'a Value, key: &str) -> Result<Self> {
        v.get(key)
            .and_then(Value::as_array)
            .map(|a| Scalars(a))
            .ok_or_else(|| HessboostError::missing_field(key))
    }

    /// Entry `i`, `None` past the end.
    fn get(self, i: usize) -> Option<f64> {
        self.0.get(i).map(|e| scalar_f64(e).unwrap_or(0.0))
    }

    /// Entry `i`, `0` past the end.
    fn at(self, i: usize) -> f64 {
        self.get(i).unwrap_or(0.0)
    }

    /// Every entry as `f32`.
    fn to_f32s(self) -> Vec<f32> {
        self.0
            .iter()
            .map(|e| scalar_f64(e).unwrap_or(0.0) as f32)
            .collect()
    }

    /// Every entry as `i32` (truncated).
    fn to_i32s(self) -> Vec<i32> {
        self.0
            .iter()
            .map(|e| scalar_f64(e).unwrap_or(0.0) as i32)
            .collect()
    }
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
    use crate::test_support::labeled_dense;
    use crate::training::train;

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
        let d = labeled_dense(&x, n, 2, &y);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        (train(&params, &d, 15).unwrap(), d)
    }

    /// A DART model on `d` with non-unit tree weights.
    fn dart_model(d: &DMatrix) -> BoostedModel {
        let params = TrainingParams::builder()
            .booster(BoosterKind::Dart)
            .rate_drop(0.5)
            .max_depth(3)
            .build()
            .unwrap();
        train(&params, d, 8).unwrap()
    }

    /// A shallow model on one categorical feature, and its data.
    fn categorical_model() -> (BoostedModel, DMatrix) {
        let categorical = labeled_dense(
            &[0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            6,
            1,
            &[1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        )
        .with_feature_types(&[FeatureType::Categorical])
        .unwrap();
        let params = TrainingParams::builder().max_depth(2).build().unwrap();
        (train(&params, &categorical, 3).unwrap(), categorical)
    }

    /// The XGBoost JSON export of `model`, as text and parsed.
    fn export_json_document(model: &BoostedModel) -> (String, Value) {
        let text = export_xgboost_json(model).unwrap();
        let json = serde_json::from_str(&text).unwrap();
        (text, json)
    }

    /// Assert `result` is a [`HessboostError::ModelFormat`]; `context` labels
    /// a failure.
    fn assert_format_error<T: std::fmt::Debug>(result: Result<T>, context: impl std::fmt::Display) {
        let err = result.unwrap_err();
        assert!(
            matches!(err, HessboostError::ModelFormat(_)),
            "{context}: {err}"
        );
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
    fn feature_counts_round_trip_exactly() {
        // `num_feature` is written as an integer string; reading it through
        // `f64` rounded counts above 2^53 to a neighbor.
        let (model, _) = reg_model();
        for n_features in [(1usize << 53) + 1, usize::MAX] {
            let mut wide = model.clone();
            wide.n_features = n_features;
            for restored in [
                import_xgboost_json(&export_xgboost_json(&wide).unwrap()).unwrap(),
                import_xgboost_ubjson(&export_xgboost_ubjson(&wide).unwrap()).unwrap(),
            ] {
                assert_eq!(restored.n_features(), n_features);
            }
        }
        let lmp = |v: &str| serde_json::json!({ "num_feature": v });
        assert_eq!(count_param(&lmp("4.0"), "num_feature", 0).unwrap(), 4);
        assert!(count_param(&lmp("4.5"), "num_feature", 0).is_err());
        assert!(count_param(&lmp("-1"), "num_feature", 0).is_err());
    }

    #[test]
    fn roundtrip_binary_preserves_predictions() {
        // Binary logistic exercises the prob<->margin base_score link.
        let n = 80;
        let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let y: Vec<f32> = x.iter().map(|&v| f32::from(v > 0.4)).collect();
        let d = labeled_dense(&x, n, 1, &y);
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
        parallel_tree_json(1, &[0, 1, 2], &[0.0; 3], None, None).replace(
            r#""base_score": "[0E0]""#,
            &format!(r#""base_score": "{base_score}""#),
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
        let d = DMatrix::from_dense(&[0.0, 1.0], 2, 1).unwrap();
        let margins = model.predict_margin(&d).unwrap();
        assert_eq!(margins, [expected, expected].concat());

        // A single entry applies to every class (XGBoost's old-format rule).
        let uniform = import_xgboost_json(&three_class_json("[5E-1]")).unwrap();
        assert_eq!(uniform.base_scores(), &[0.5, 0.5, 0.5]);
    }

    #[test]
    fn malformed_base_score_is_rejected() {
        for bad in ["0.5", "[0.1,0.2]", "[a]", "[]", "[0.1,0.2,0.3,0.4]"] {
            assert_format_error(import_xgboost_json(&three_class_json(bad)), bad);
        }
    }

    /// A two-target model with one single-leaf vector tree of width
    /// `width` and the given `leaf_weights`.
    fn vector_stump_json(width: &str, leaf_weights: &str) -> String {
        format!(
            r#"{{
              "version": [3, 4, 1],
              "learner": {{
                "gradient_booster": {{
                  "name": "gbtree",
                  "model": {{
                    "gbtree_model_param": {{"num_parallel_tree": "1", "num_trees": "1"}},
                    "tree_info": [0],
                    "trees": [{{"id": 0,
                      "tree_param": {{"num_nodes": "1", "num_feature": "1", "size_leaf_vector": "{width}"}},
                      "left_children": [-1], "right_children": [0], "parents": [-1],
                      "split_indices": [0], "split_conditions": [0.0], "default_left": [0],
                      "base_weights": [], "leaf_weights": {leaf_weights},
                      "loss_changes": [0.0], "sum_hessian": [1.0], "split_type": [0]}}]
                  }}
                }},
                "learner_model_param": {{
                  "base_score": "[0E0]", "boost_from_average": "1",
                  "num_class": "0", "num_feature": "1", "num_target": "2"
                }},
                "objective": {{"name": "reg:squarederror", "reg_loss_param": {{"scale_pos_weight": "1"}}}}
              }}
            }}"#
        )
    }

    #[test]
    fn vector_leaf_width_is_validated_before_allocating() {
        let model = import_xgboost_json(&vector_stump_json("2", "[1.0, 2.0]")).unwrap();
        let d = DMatrix::from_dense(&[0.0], 1, 1).unwrap();
        assert_eq!(model.predict_margin(&d).unwrap(), [1.0, 2.0]);
        // A width that saturates `usize` must not panic allocating the leaf
        // storage: it, a width other than the model's outputs, a fractional
        // width, or one the leaf weights cannot fill is a format error.
        for (width, weights) in [
            ("1e30", "[]"),
            ("18446744073709551615", "[]"),
            ("3", "[1.0, 2.0, 3.0]"),
            ("2.5", "[1.0, 2.0, 3.0]"),
            ("2", "[1.0]"),
        ] {
            assert_format_error(
                import_xgboost_json(&vector_stump_json(width, weights)),
                width,
            );
        }
    }

    #[test]
    fn vector_leaf_storage_is_bounded_by_the_leaf_weights() {
        // 65,536 outputs over 65,535 nodes would expand to ~16 GiB of leaf
        // storage, so the node graph and leaf mapping must be checked before
        // any of it is allocated.
        const K: usize = 1 << 16;
        const N: usize = K - 1;
        let one_vector = vec![0.5f32; K];
        let tree = |left: Vec<i64>, right: Vec<i64>, leaf_weights: &[f32]| {
            json!({
                "tree_param": {"num_nodes": N.to_string(), "num_feature": "1",
                               "size_leaf_vector": K.to_string()},
                "left_children": left,
                "right_children": right,
                "split_conditions": vec![0.0f32; N],
                "leaf_weights": leaf_weights,
            })
        };
        // Heap-shaped binary tree: internal node `i` has children
        // `2i + 1, 2i + 2`; leaves are numbered in node order.
        let internal = N / 2;
        let heap_left: Vec<i64> = (0..N)
            .map(|i| if i < internal { 2 * i as i64 + 1 } else { -1 })
            .collect();
        let heap_right: Vec<i64> = (0..N)
            .map(|i| {
                if i < internal {
                    2 * i as i64 + 2
                } else {
                    (i - internal) as i64
                }
            })
            .collect();
        let cases = [
            // Every node a leaf sharing the one serialized vector.
            tree(vec![-1; N], vec![0; N], &one_vector),
            // Binary-tree shape, but internal children out of range.
            tree(
                heap_left
                    .iter()
                    .map(|&l| if l < 0 { l } else { i64::from(i32::MAX) })
                    .collect(),
                heap_right.clone(),
                &one_vector,
            ),
            // Valid graph, but one serialized vector for 32,768 leaves.
            tree(heap_left.clone(), heap_right.clone(), &one_vector),
        ];
        for (case, tj) in cases.iter().enumerate() {
            assert_format_error(tree_from_json(tj, K), case);
        }
        // The same valid graph with a vector per leaf decodes, each leaf
        // reading its own vector.
        let k = 2;
        let leaves = N - internal;
        let weights: Vec<f32> = (0..leaves * k).map(|v| v as f32).collect();
        let mut tj = tree(heap_left, heap_right, &weights);
        tj["tree_param"]["size_leaf_vector"] = json!(k.to_string());
        let decoded = tree_from_json(&tj, k).unwrap();
        assert_eq!(decoded.leaf_vector(internal), [0.0, 1.0]);
        assert_eq!(
            decoded.leaf_vector(N - 1),
            [(2 * leaves - 2) as f32, (2 * leaves - 1) as f32]
        );
    }

    /// [`hand_stump_json`] without its tree.
    fn treeless_json() -> String {
        let treeless = hand_stump_json()
            .replace(r#""num_trees": "1""#, r#""num_trees": "0""#)
            .replace(r#""iteration_indptr": [0, 1],"#, "")
            .replace(r#""tree_info": [0]"#, r#""tree_info": []"#);
        format!(
            "{}]{}",
            &treeless[..treeless.find(r#""trees": ["#).unwrap() + 10],
            &treeless[treeless.find("}]").unwrap() + 2..]
        )
    }

    #[test]
    fn output_count_is_validated_before_allocating() {
        // `num_target` saturating `usize` must not panic sizing the per-output
        // tree groups or broadcasting the intercept.
        let stump = hand_stump_json();
        let treeless = treeless_json();
        assert_eq!(import_xgboost_json(&treeless).unwrap().num_trees(), 0);
        for doc in [stump.to_string(), treeless] {
            for count in ["1e30", "18446744073709551615", "1e18", "2.5", "-1", "0"] {
                let doc = doc.replace(
                    r#""num_target": "1""#,
                    &format!(r#""num_target": "{count}""#),
                );
                assert_format_error(import_xgboost_json(&doc), count);
            }
        }
        // One tree cannot cover two outputs' groups.
        let two = stump.replace(r#""num_target": "1""#, r#""num_target": "2""#);
        assert_format_error(import_xgboost_json(&two), "two outputs");
    }

    #[test]
    fn treeless_intercept_broadcast_is_bounded() {
        // A tree-less document declares its outputs without backing them:
        // broadcasting one `base_score` entry to 2^29 of them would allocate
        // 2 GiB.
        let treeless = treeless_json();
        // The document declares `num_class` 0 and `num_target` 1.
        let declare = |key: &str, count: usize| {
            let declared = if key == "num_class" { "0" } else { "1" };
            treeless.replace(
                &format!(r#""{key}": "{declared}""#),
                &format!(r#""{key}": "{count}""#),
            )
        };
        let huge = declare("num_target", 536_870_912);
        assert_format_error(import_xgboost_json(&huge), "num_target 2^29");
        let classes = declare("num_class", 536_870_912).replace(
            r#""objective": {"name": "reg:squarederror", "reg_loss_param": {"scale_pos_weight": "1"}}"#,
            r#""objective": {"name": "multi:softprob",
                "softmax_multiclass_param": {"num_class": "536870912"}}"#,
        );
        assert_format_error(import_xgboost_json(&classes), "num_class 2^29");
        // Within the bound, and for an explicit per-output vector, the entry
        // still applies to every output.
        let wide = import_xgboost_json(&declare("num_target", MAX_BROADCAST_OUTPUTS)).unwrap();
        assert_eq!(wide.base_scores(), vec![0.0; MAX_BROADCAST_OUTPUTS]);
        let listed = declare("num_target", MAX_BROADCAST_OUTPUTS + 1).replace(
            r#""base_score": "[0E0]""#,
            &format!(
                r#""base_score": "{}""#,
                format_float_vector(vec![0.25; MAX_BROADCAST_OUTPUTS + 1])
            ),
        );
        let listed = import_xgboost_json(&listed).unwrap();
        assert_eq!(listed.base_scores(), vec![0.25; MAX_BROADCAST_OUTPUTS + 1]);
    }

    /// A one-split categorical stump whose categorical nodes take the
    /// segments `(begin, size)` of `categories`: node 0 splits to node 1 and
    /// leaf 2, node 1 to leaves 3 and 4.
    fn categorical_segments_json(categories: &str, segments: [(usize, usize); 2]) -> String {
        let tree = format!(
            r#""trees": [{{
              "id": 0,
              "tree_param": {{"num_nodes": "5", "num_feature": "1", "size_leaf_vector": "1"}},
              "left_children":  [1, 3, -1, -1, -1],
              "right_children": [2, 4, -1, -1, -1],
              "split_indices":  [0, 0, 0, 0, 0],
              "split_conditions": [0.0, 0.0, 1.0, 2.0, 3.0],
              "default_left":   [0, 0, 0, 0, 0],
              "split_type":     [1, 1, 0, 0, 0],
              "categories": {categories},
              "categories_nodes": [0, 1],
              "categories_segments": [{}, {}],
              "categories_sizes": [{}, {}]
            }}]"#,
            segments[0].0, segments[1].0, segments[0].1, segments[1].1
        );
        let stump = hand_stump_json();
        let start = stump.find(r#""trees": ["#).unwrap();
        let end = stump.find("}]").unwrap() + 2;
        format!("{}{tree}{}", &stump[..start], &stump[end..])
    }

    #[test]
    fn categorical_segments_cannot_expand_the_category_array() {
        // Each node copies its segment, so segments overlapping each other
        // would let `n` nodes expand the array `n`-fold.
        let disjoint = categorical_segments_json("[0, 1, 1, 2]", [(0, 2), (2, 2)]);
        let model = import_xgboost_json(&disjoint).unwrap();
        assert_eq!(model.trees()[0].categories().len(), 4);
        let overlapping = categorical_segments_json("[0, 1]", [(0, 2), (0, 2)]);
        assert_format_error(import_xgboost_json(&overlapping), "overlapping segments");
    }

    #[test]
    fn present_but_invalid_objective_parameters_are_refused() {
        let with_objective = |objective: &str| {
            hand_stump_json()
                .replace(
                    r#""objective": {"name": "reg:squarederror", "reg_loss_param": {"scale_pos_weight": "1"}}"#,
                    &format!(r#""objective": {objective}"#),
                )
                .replace(r#""base_score": "[0E0]""#, r#""base_score": "[5E-1]""#)
        };
        for objective in [
            r#"{"name": "survival:aft", "aft_loss_param": {"aft_loss_distribution": "unsupported"}}"#,
            r#"{"name": "survival:aft", "aft_loss_param": {"aft_loss_distribution": 1}}"#,
            r#"{"name": "survival:aft", "aft_loss_param": {"aft_loss_distribution_scale": "wide"}}"#,
            r#"{"name": "survival:aft", "aft_loss_param": "normal"}"#,
            r#"{"name": "reg:squarederror", "reg_loss_param": {"scale_pos_weight": "heavy"}}"#,
            r#"{"name": "count:poisson", "poisson_regression_param": {"max_delta_step": null}}"#,
            r#"{"name": "rank:ndcg", "lambdarank_param": {"lambdarank_num_pair_per_sample": "2.5"}}"#,
            r#"{"name": "rank:ndcg", "lambdarank_param": {"lambdarank_num_pair_per_sample": "0"}}"#,
            r#"{"name": "reg:quantileerror", "quantile_loss_param": {"quantile_alpha": 0.5}}"#,
            r#"{"name": "reg:quantileerror", "quantile_loss_param": {"quantile_alpha": "[a]"}}"#,
            r#"{"name": 7}"#,
            r#""reg:squarederror""#,
        ] {
            assert_format_error(import_xgboost_json(&with_objective(objective)), objective);
        }
        // Genuinely missing blocks and fields keep XGBoost's defaults.
        for objective in [
            r#"{"name": "survival:aft"}"#,
            r#"{"name": "survival:aft", "aft_loss_param": {}}"#,
        ] {
            let model = import_xgboost_json(&with_objective(objective)).unwrap();
            let params = model.objective_params();
            assert_eq!(params.aft_loss_distribution, AftDistribution::Normal);
            assert_eq!(params.aft_loss_distribution_scale, 1.0);
        }
        let unset = with_objective(
            r#"{"name": "rank:ndcg", "lambdarank_param": {"lambdarank_num_pair_per_sample": "4294967295"}}"#,
        );
        let model = import_xgboost_json(&unset).unwrap();
        assert_eq!(model.objective_params().lambdarank_num_pair_per_sample, 32);
    }

    #[test]
    fn export_refuses_num_class_beside_several_outputs() {
        // XGBoost has no model with `num_class` 2 and two binary targets
        // (`LearnerModelParam` refuses `num_class > 1` with `num_target >
        // 1`); exporting one wrote its margins as probabilities.
        let objective = "binary:logistic";
        let model = BoostedModel::from_parts(
            Vec::new(),
            Vec::new(),
            vec![0.0, 0.0],
            ModelSpec {
                objective_params: ObjectiveParams::defaults_for(objective),
                objective: objective.to_string(),
                num_class: 2,
                n_outputs: 2,
                n_targets: 2,
                n_features: 1,
            },
        );
        assert_format_error(export_xgboost_json(&model), "num_class with two targets");
        assert_format_error(export_xgboost_ubjson(&model), "num_class with two targets");
    }

    #[test]
    fn export_writes_xgboost_3_learner_params() {
        let (model, _) = reg_model();
        let (_, json) = export_json_document(&model);
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
        assert_format_error(import_xgboost_json(js), "gblinear");
    }

    #[test]
    fn dart_roundtrips_through_weight_drop() {
        let (_, d) = reg_model();
        let model = dart_model(&d);
        assert!(model.has_non_unit_tree_weights());
        let before = model.predict(&d).unwrap();

        let (exported, json) = export_json_document(&model);
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

        let (model, categorical) = categorical_model();
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
        assert_format_error(import_xgboost_ubjson(truncated), "truncated");
    }

    #[test]
    fn ubjson_export_is_the_json_document_with_typed_tree_arrays() {
        let (_, d) = reg_model();
        let (categorical_model, categorical) = categorical_model();
        for (model, data) in [
            (reg_model().0, &d),
            (dart_model(&d), &d),
            (categorical_model, &categorical),
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
        let first = model.slice(0..1, 1).unwrap();
        let d = DMatrix::from_dense(&[0.0], 1, 1).unwrap();
        assert_eq!(first.predict_margin(&d).unwrap(), [3.0, 30.0, 300.0]);

        // DART weights are indexed by XGBoost position and follow their trees.
        let weights = [1.0, 0.5, 1.0, 0.5, 1.0, 0.5];
        let dart = parallel_tree_json(2, &tree_info, &leaves, None, Some(&weights));
        assert_eq!(class_margins(&dart), [2.0, 20.0, 200.0]);
        // Re-export writes the forest back unchanged.
        let model = import_xgboost_json(&dart).unwrap();
        let (_, doc) = export_json_document(&model);
        let booster = &doc["learner"]["gradient_booster"]["model"];
        assert_eq!(booster["gbtree_model_param"]["num_parallel_tree"], "2");
        assert_eq!(booster["tree_info"], json!(tree_info));
        assert_eq!(booster["iteration_indptr"], json!([0, 6]));
        assert_eq!(booster["weight_drop"], json!(weights));

        // Iterations of different forest sizes are unmappable.
        let uneven_info = [0, 0, 1, 2, 1, 2, 0, 1, 2];
        let uneven = parallel_tree_json(2, &uneven_info, &leaves2[..9], Some(&[0, 6, 9]), None);
        assert_format_error(import_xgboost_json(&uneven), "uneven");

        // Groups with unequal tree counts in one iteration are unmappable.
        let lopsided = parallel_tree_json(2, &[0, 0, 1, 2, 2, 0], &leaves, None, None);
        assert_format_error(import_xgboost_json(&lopsided), "lopsided");
        let missing = parallel_tree_json(1, &tree_info, &leaves, None, None)
            .replace(r#""tree_info": [0, 0, 1, 1, 2, 2],"#, "");
        assert_format_error(import_xgboost_json(&missing), "no tree_info");
    }

    #[test]
    fn objective_params_roundtrip_through_parameter_blocks() {
        /// Reads the retained value of one case's parameter.
        type Retained = fn(&ObjectiveParams) -> f64;
        let n = 40;
        let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let counts: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect();
        let binary: Vec<f32> = x.iter().map(|&v| f32::from(v > 0.6)).collect();
        let d = labeled_dense(&x, n, 1, &counts);
        let binary = labeled_dense(&x, n, 1, &binary);
        let ranked = labeled_dense(&x, n, 1, &counts)
            .with_group_sizes(&[20, 20])
            .unwrap();
        let fit = |builder: crate::config::TrainingParamsBuilder, d: &DMatrix| {
            train(&builder.max_depth(2).build().unwrap(), d, 2).unwrap()
        };
        let b = TrainingParams::builder;
        let ranker = || b().objective("rank:ndcg").lambdarank_num_pair_per_sample(5);
        // (configuration, data, block, key, exported text, retained value)
        let cases: [(_, _, _, _, _, Retained); 5] = [
            (
                b().objective("reg:tweedie").tweedie_variance_power(1.2),
                &d,
                "tweedie_regression_param",
                "tweedie_variance_power",
                "1.2",
                |p| p.tweedie_variance_power,
            ),
            (
                b().objective("count:poisson").max_delta_step(0.3),
                &d,
                "poisson_regression_param",
                "max_delta_step",
                "0.3",
                |p| p.max_delta_step,
            ),
            (
                b().objective("reg:pseudohubererror").huber_slope(2.5),
                &d,
                "pseudo_huber_param",
                "huber_slope",
                "2.5",
                |p| p.huber_slope,
            ),
            (
                b().objective("binary:logistic").scale_pos_weight(3.0),
                &binary,
                "reg_loss_param",
                "scale_pos_weight",
                "3",
                |p| p.scale_pos_weight,
            ),
            (
                ranker(),
                &ranked,
                "lambdarank_param",
                "lambdarank_num_pair_per_sample",
                "5",
                |p| p.lambdarank_num_pair_per_sample as f64,
            ),
        ];
        for (builder, data, block, key, text, retained) in cases {
            let (exported, json) = export_json_document(&fit(builder, data));
            assert_eq!(
                json["learner"]["objective"][block][key], text,
                "{block}.{key}"
            );
            let back = import_xgboost_json(&exported).unwrap();
            let expected: f64 = text.parse().unwrap();
            assert_eq!(retained(back.objective_params()), expected, "{block}.{key}");
        }

        // XGBoost's own "not set" sentinel maps to the `topk` default.
        let exported = export_xgboost_json(&fit(ranker(), &ranked))
            .unwrap()
            .replace(
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
        let d = labeled_dense(&x, n, 1, &y);
        let params = TrainingParams::builder()
            .objective("reg:quantileerror")
            .quantile_alpha(vec![0.1, 0.9])
            .max_depth(2)
            .build()
            .unwrap();
        let model = train(&params, &d, 3).unwrap();
        let (exported, json) = export_json_document(&model);
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
            assert_format_error(
                import_xgboost_json(&exported.replace("[0.1,0.9]", bad)),
                bad,
            );
        }

        let mae = TrainingParams::builder()
            .objective("reg:absoluteerror")
            .max_depth(2)
            .build()
            .unwrap();
        let (_, json) = export_json_document(&train(&mae, &d, 2).unwrap());
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
        let (exported, json) = export_json_document(&aft);
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
        let dc = labeled_dense(&x, n, 1, &signed);
        let params = TrainingParams::builder()
            .objective("survival:cox")
            .max_depth(2)
            .build()
            .unwrap();
        let cox = train(&params, &dc, 3).unwrap();
        let (exported, json) = export_json_document(&cox);
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
        use crate::objective::{CustomObjective, GradPair};
        use crate::training::Trainer;
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
        let model = Trainer::new(&params, &d, 2)
            .objective(&obj)
            .train()
            .unwrap()
            .model;
        assert_format_error(export_xgboost_json(&model), "custom objective");
    }
}
