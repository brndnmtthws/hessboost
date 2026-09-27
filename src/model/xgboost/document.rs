//! The model-level XGBoost document: learner, booster, and tree layout (`tree_info`, `iteration_indptr`).

use super::objective::{
    build_objective, format_base_score, objective_params_from_json, objective_to_json,
    parse_base_score,
};
use super::parse::{
    count_param, field, optional_str, scalar_count, scalar_f64, strict_nonnegative_integer_array,
};
use super::tree::{tree_from_json, tree_to_json};
use crate::error::{HessboostError, Result};
use crate::model::objective::ModelObjective;
use crate::model::ubjson::ElementType;
use crate::model::{BoostedModel, ModelSpec};
use crate::objective::Objective;
use crate::tree::RegTree;
use serde_json::{Map, Value, json};

/// The element type XGBoost stores the array member `key` of `object` with,
/// or `None` for a generic array. Mirrors the `F32Array` / `I32Array` /
/// `U8Array` / `I64Array` members of XGBoost's `RegTree::SaveModel`,
/// `MultiTargetTree::SaveModel`, `GBLinearModel::SaveModel` and
/// `CatContainer::Save`.
pub(super) fn xgboost_typed_array(key: &str, object: &Map<String, Value>) -> Option<ElementType> {
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
pub(super) fn model_to_value(model: &BoostedModel) -> Result<Value> {
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
    let name = model.objective().name();
    reject_extension_objective(name)?;
    // XGBoost can only load objectives it knows; a custom loss has no
    // XGBoost counterpart.
    let no_equivalent = || {
        HessboostError::model_format(format!(
            "objective `{name}` has no XGBoost equivalent; cannot export"
        ))
    };
    let objective = model.objective().built_in().ok_or_else(no_equivalent)?;
    let objective_impl = model
        .rebuild_objective()
        .and_then(Result::ok)
        .ok_or_else(no_equivalent)?;
    // XGBoost refuses `num_class` beside several outputs (`LearnerModelParam`
    // allows `num_class > 1` only with one target), and a model of any other
    // objective with `num_class >= 2` has `num_class` outputs.
    if num_class >= 2 && objective.num_class().is_none() {
        return Err(HessboostError::model_format(format!(
            "`num_class` {num_class} with objective `{name}` has no XGBoost equivalent; \
             cannot export"
        )));
    }
    let n_trees = model.effective_num_trees();
    let per_iteration = model.trees_per_iteration();

    // A shrunk model's closed-form contribution weights go into its leaves
    // (as CatBoost bakes its shrinkage), so XGBoost reads plain gbtree
    // trees. A sum of trees cannot repeat training's per-iteration
    // rounding, so the exported margins match within `f32` rounding.
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
                "num_target": if objective.num_class().is_some() { model.n_targets() } else { model.n_outputs() }.to_string(),
            },
            "objective": objective_to_json(objective, model.max_delta_step()),
        }
    }))
}

/// Refuse hessboost's own objectives, which XGBoost does not define: the
/// distributional `dist:*` objectives and `rank:xendcg`. Their models are
/// saved in the native binary or JSON formats only.
pub(super) fn reject_extension_objective(objective: &str) -> Result<()> {
    if crate::objective::distributional::DistFamily::from_objective(objective).is_some()
        || objective == Objective::RankXendcg.name()
    {
        return Err(HessboostError::model_format(format!(
            "objective `{objective}` is a hessboost extension that XGBoost models cannot \
             carry; save the model in the native binary or JSON format"
        )));
    }
    Ok(())
}

/// Map an XGBoost model document (decoded from either encoding) to a
/// [`BoostedModel`].
pub(super) fn model_from_value(root: &Value) -> Result<BoostedModel> {
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

    // Parameter blocks come from the file: the objective's own parameters
    // are checked as its constructors check them; parameters it does not
    // read (e.g. `tweedie_regression_param` of `reg:squarederror`) are
    // dropped.
    let stored = objective_params_from_json(&objective, objective_json)?;
    let max_delta_step = stored.max_delta_step;
    let objective = ModelObjective::from_stored(&objective, &stored, num_class)?;
    let name = objective.name();
    let n_targets = match objective.built_in() {
        Some(Objective::Quantile(_) | Objective::Expectile(_)) => 1,
        _ => num_target,
    };
    let objective_impl = build_objective(&objective, n_targets, max_delta_step)?;
    let n_outputs = match &objective_impl {
        Some(objective) => objective.n_outputs(),
        None if num_class >= 2 => num_class,
        None => num_target.max(1),
    };
    if num_class < 2 && num_target.max(1) != n_outputs {
        return Err(HessboostError::model_format(format!(
            "`num_target` {num_target} does not match the {n_outputs} outputs of objective `{name}`"
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
            objective,
            max_delta_step,
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
pub(super) fn iteration_tree_order(
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
pub(super) fn num_parallel_tree_param(model: &Value) -> Result<usize> {
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
pub(super) fn iteration_indptr(
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
