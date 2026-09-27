//! XGBoost trees: node columns, categorical segments, vector leaves.

use super::parse::{
    column_len, float_column, integer_column, optional_float_column, scalar_f64,
    strict_nonnegative_integer_array,
};
use crate::error::{HessboostError, Result};
use crate::model::categories::{CategoryPool, PoolError};
use crate::tree::{Node, RegTree};
use serde_json::{Value, json};

/// Sentinel XGBoost writes for the parent of the root node (`kInvalidNodeId`).
pub(super) const INVALID_NODE: i32 = i32::MAX;

/// Encode one [`RegTree`] as XGBoost's node-indexed array bundle.
///
/// A vector-leaf tree is written as XGBoost's `MultiTargetTree` bundle
/// (`MultiTargetTree::SaveModel`): the leaf vectors in `leaf_weights` (`K`
/// values per leaf, leaves in node order) with each leaf's `right_children`
/// entry holding its leaf index, `parents[0] = -1`, and XGBoost's
/// `DftBadValue` (the smallest subnormal) as the split condition of leaves and
/// categorical nodes. Internal weights are not retained, so `base_weights`
/// carries the leaf values (vectors) and zeros for internal nodes.
pub(super) fn tree_to_json(id: usize, tree: &RegTree, num_feature: usize) -> Value {
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
        if let Some((left, right)) = node.children() {
            parents[left] = i as i32;
            parents[right] = i as i32;
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
                base_weights.extend_from_slice(tree.leaf_weights(node_id));
                leaf_weights.extend_from_slice(tree.leaf_weights(node_id));
            } else {
                // XGBoost carries the leaf weight in both arrays for leaves.
                let (l, r) = node.links();
                left.push(l);
                right.push(r);
                split_conditions.push(node.leaf_value);
                base_weights.push(node.leaf_value);
                default_left.push(1i32);
            }
            continue;
        }
        split_indices.push(node.split_feature);
        loss_changes.push(node.split_gain);
        base_weights.extend(std::iter::repeat_n(0.0f32, k));
        let (l, r) = node.links();
        if node.is_categorical {
            // XGBoost sends the category set right; hessboost keeps it left.
            left.push(r);
            right.push(l);
            split_conditions.push(if vector {
                DFT_BAD_VALUE
            } else {
                node.split_cond
            });
            default_left.push(i32::from(!node.default_left));
        } else {
            left.push(l);
            right.push(r);
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
pub(super) fn vector_leaves(tj: &Value, left: &[i32], right: &[i32], k: usize) -> Result<Vec<f32>> {
    // Converted once: leaves may share a slot, so a lazy per-leaf read would
    // re-parse string entries once per referencing leaf.
    let leaf_weights = float_column(tj, "leaf_weights", None)?;
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
pub(super) struct TreeCategories {
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
    fn flatten_into(&self, nodes: &mut [Node]) -> Result<Vec<u32>> {
        let mut pool = CategoryPool::with_capacity(self.total);
        for (node, segment) in nodes.iter_mut().zip(&self.segments) {
            if let Some((begin, end)) = *segment {
                // `read` checked every value against `u32`.
                let ids = self.values[begin..end].iter().map(|&v| v as u32);
                pool.push_split(node, ids).map_err(|e| {
                    HessboostError::model_format(match e {
                        PoolError::Empty => "invalid categorical arrays",
                        PoolError::TooMany => "too many categories",
                    })
                })?;
            }
        }
        Ok(pool.finish())
    }
}

/// A tree's per-node arrays, as [`decode_nodes`] reads them, each of the
/// tree's node count.
pub(super) struct NodeColumns<'a> {
    left: &'a [i32],
    right: &'a [i32],
    /// Empty when the tree has no `split_type`.
    split_type: &'a [u64],
    split_indices: Vec<u32>,
    split_conditions: Vec<f32>,
    default_left: Vec<u8>,
    /// Zeros when absent.
    sum_hessian: Vec<f32>,
    /// Zeros when absent.
    loss_changes: Vec<f32>,
}

impl<'a> NodeColumns<'a> {
    /// Read the columns of a tree of `left.len()` nodes. XGBoost 3.4.2
    /// writes every per-node array; the statistics XGBoost does not predict
    /// with (`base_weights`, `sum_hessian`, `loss_changes`) may be absent,
    /// but a present one must be whole and numeric. `base_weights` (`k`
    /// values per node in a vector-leaf tree of width `k`) is checked and
    /// dropped: leaves carry their values in `split_conditions`.
    fn read(
        tj: &Value,
        left: &'a [i32],
        right: &'a [i32],
        split_type: &'a [u64],
        k: usize,
    ) -> Result<Self> {
        let n = left.len();
        let statistic = |key| {
            optional_float_column(tj, key, n).map(|column| column.unwrap_or_else(|| vec![0.0; n]))
        };
        let weights = n
            .checked_mul(k.max(1))
            .ok_or_else(|| HessboostError::model_format("`base_weights` length overflows"))?;
        optional_float_column(tj, "base_weights", weights)?;
        Ok(NodeColumns {
            left,
            right,
            split_type,
            split_indices: integer_column(tj, "split_indices", n, 0..=i64::from(u32::MAX))?,
            split_conditions: float_column(tj, "split_conditions", Some(n))?,
            default_left: integer_column(tj, "default_left", n, 0..=1)?,
            sum_hessian: statistic("sum_hessian")?,
            loss_changes: statistic("loss_changes")?,
        })
    }
}

/// `size_leaf_vector`: 0 or 1 for scalar trees and the model's output count
/// for vector-leaf trees (`MultiTargetTree`); anything else, including a
/// width too large to allocate, is malformed.
pub(super) fn leaf_vector_width(tj: &Value, n_outputs: usize) -> Result<usize> {
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
pub(super) fn decode_nodes(
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
        let sum_hess = cols.sum_hessian[i];
        if left[i] == -1 && size_leaf_vector > 1 {
            // Vector leaf: the weights live in `leaf_vectors`.
            nodes.push(Node::leaf(0.0, sum_hess));
        } else if left[i] == -1 {
            // XGBoost carries a leaf's weight in `split_conditions`.
            nodes.push(Node::leaf(cols.split_conditions[i], sum_hess));
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
            let default_left = cols.default_left[i] == 1;
            let mut node = Node::leaf(0.0, sum_hess);
            node.split_feature = cols.split_indices[i];
            node.split_cond = cols.split_conditions[i];
            // XGBoost sends missing values of a categorical split the other
            // way round (its children are swapped).
            node.default_left = default_left != is_categorical;
            if is_categorical {
                node.set_links(right[i], left[i]);
            } else {
                node.set_links(left[i], right[i]);
            }
            node.split_gain = cols.loss_changes[i];
            node.is_categorical = is_categorical;
            nodes.push(node);
        }
    }
    Ok(nodes)
}

/// Decode one XGBoost tree object into a [`RegTree`] of a model with
/// `n_outputs` outputs.
pub(super) fn tree_from_json(tj: &Value, n_outputs: usize) -> Result<RegTree> {
    let child = -1..=i64::from(i32::MAX);
    let n = column_len(tj, "left_children")?;
    if n == 0 {
        return Err(HessboostError::model_format("tree contains no nodes"));
    }
    let left = integer_column(tj, "left_children", n, child.clone())?;
    let right = integer_column(tj, "right_children", n, child)?;

    let split_type = strict_nonnegative_integer_array(tj, "split_type")?;
    if !split_type.is_empty() && split_type.len() != n {
        return Err(HessboostError::model_format(
            "`split_type` length does not match the node count",
        ));
    }
    let categories = TreeCategories::read(tj, n)?;
    let size_leaf_vector = leaf_vector_width(tj, n_outputs)?;
    let cols = NodeColumns::read(tj, &left, &right, &split_type, size_leaf_vector)?;
    let mut nodes = decode_nodes(&cols, &categories, size_leaf_vector)?;
    let leaf_vectors = if size_leaf_vector > 1 {
        vector_leaves(tj, &left, &right, size_leaf_vector)?
    } else {
        Vec::new()
    };
    let flat_categories = categories.flatten_into(&mut nodes)?;
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
