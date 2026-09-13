//! Exact per-prediction feature attributions via TreeSHAP.
//!
//! This implements the path-dependent, exact TreeSHAP algorithm of Lundberg et
//! al., *"Consistent Individualized Feature Attribution for Tree Ensembles"*,
//! matching XGBoost's `pred_contribs=True`. For a single tree the returned
//! per-feature values sum to `f_tree(x) - E[f_tree]`, where the expectation is
//! taken over the tree's cover (Hessian) distribution. The missing offset
//! `E[f_tree]` is folded into the bias term. Summed over the whole ensemble the
//! contributions therefore satisfy the exact-additivity property
//!
//! ```text
//! Σ_j contribs[j] + bias == margin(x)
//! ```
//!
//! where `bias == base_score + Σ_tree E[f_tree]` and `margin(x)` is the model's
//! raw margin ([`BoostedModel::predict_margin`]).
//!
//! The core recursion is the standard `O(T · L · D²)` algorithm (`T` trees, `L`
//! leaves, `D` maximum depth): each root-to-leaf traversal maintains the set of
//! unique features seen so far together with their `zero`/`one` fractions and a
//! permutation weight, using the EXTEND / UNWIND operations to add and remove
//! features as the path forks.

use crate::data::DMatrix;
use crate::error::Result;
use crate::learner::model::{BoostedModel, RowBlock};
use crate::tree::RegTree;
use rayon::prelude::*;

/// One element of a decision path: a unique feature together with the fraction
/// of permutations in which it is "one" (present / in the coalition) and "zero"
/// (absent), plus the accumulated proportion of subset weights (`pweight`).
#[derive(Clone, Copy, Default)]
struct PathElement {
    /// Feature index for this path element, or `-1` for the root placeholder.
    feature_index: i64,
    /// Fraction of paths that flow through the "zero" (absent) branch.
    zero_fraction: f64,
    /// Fraction of paths that flow through the "one" (present) branch.
    one_fraction: f64,
    /// Accumulated proportion of the subset weights for this element.
    pweight: f64,
}

/// `1 / n` for the small integers that appear as path positions. f64 division
/// dominates TreeSHAP's inner loops, so the integer reciprocals are tabulated
/// and the data-dependent ones hoisted out of the loops.
const INV_LEN: usize = 128;
const INV: [f64; INV_LEN] = {
    let mut table = [0.0f64; INV_LEN];
    let mut i = 1;
    while i < INV_LEN {
        table[i] = 1.0 / i as f64;
        i += 1;
    }
    table
};

#[inline(always)]
fn inv(n: usize) -> f64 {
    if n < INV_LEN {
        INV[n]
    } else {
        1.0 / n as f64
    }
}

/// Grow the decision path `path[..len]` by one element (`path[len]`), updating
/// every existing element's `pweight` to account for one extra split in the
/// coalition ordering.
fn extend_path(
    path: &mut [PathElement],
    len: usize,
    zero_fraction: f64,
    one_fraction: f64,
    feature_index: i64,
) {
    let unique_depth = len; // index the new element will occupy
    path[unique_depth] = PathElement {
        feature_index,
        zero_fraction,
        one_fraction,
        pweight: if unique_depth == 0 { 1.0 } else { 0.0 },
    };
    let inv_denom = inv(unique_depth + 1);
    let one_scaled = one_fraction * inv_denom;
    let zero_scaled = zero_fraction * inv_denom;
    if one_fraction == 0.0 {
        // Cold edge: the new element takes no weight from its predecessors
        // (the `one_scaled` term is exactly zero).
        for i in (0..unique_depth).rev() {
            path[i].pweight = zero_scaled * path[i].pweight * (unique_depth - i) as f64;
        }
        return;
    }
    for i in (0..unique_depth).rev() {
        let pw_i = path[i].pweight;
        path[i + 1].pweight += one_scaled * pw_i * (i + 1) as f64;
        path[i].pweight = zero_scaled * pw_i * (unique_depth - i) as f64;
    }
}

/// Undo a previous [`extend_path`] on `path` (a full path of `path.len()`
/// elements), removing the element at `path_index` and restoring the `pweight`s
/// of the remaining elements. The path afterwards occupies `path[..len - 1]`.
fn unwind_path(path: &mut [PathElement], path_index: usize) {
    let unique_depth = path.len() - 1; // top index
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let denom = (unique_depth + 1) as f64;
    let inv_denom = inv(unique_depth + 1);
    if one_fraction != 0.0 {
        let mut next_one_portion = path[unique_depth].pweight;
        let scale = denom / one_fraction;
        let decay = scale * (zero_fraction * inv_denom);
        for i in (0..unique_depth).rev() {
            let inv_i = inv(i + 1);
            let tmp = path[i].pweight;
            path[i].pweight = next_one_portion * (scale * inv_i);
            next_one_portion = tmp - next_one_portion * (decay * inv_i * (unique_depth - i) as f64);
        }
    } else if zero_fraction != 0.0 {
        let scale = denom / zero_fraction;
        for i in (0..unique_depth).rev() {
            path[i].pweight *= scale * inv(unique_depth - i);
        }
    }
    for i in path_index..unique_depth {
        path[i].feature_index = path[i + 1].feature_index;
        path[i].zero_fraction = path[i + 1].zero_fraction;
        path[i].one_fraction = path[i + 1].one_fraction;
    }
}

/// The total permutation weight that element `path_index` would contribute if it
/// were unwound, computed without mutating `path`.
fn unwound_path_sum(path: &[PathElement], path_index: usize) -> f64 {
    let unique_depth = path.len() - 1; // top index
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let denom = (unique_depth + 1) as f64;
    let mut total = 0.0;
    if one_fraction != 0.0 {
        // next_i = pw_i - next_{i+1} * scale/(i+1) * zero/(D+1) * (D-i): the
        // coefficient of next_{i+1} is gathered off the dependency chain so
        // each step of the recurrence is one multiply-subtract.
        let mut next_one_portion = path[unique_depth].pweight;
        let scale = denom / one_fraction;
        let decay = scale * (zero_fraction * inv(unique_depth + 1));
        for i in (0..unique_depth).rev() {
            let inv_i = inv(i + 1);
            total += next_one_portion * (scale * inv_i);
            next_one_portion =
                path[i].pweight - next_one_portion * (decay * inv_i * (unique_depth - i) as f64);
        }
    } else if zero_fraction != 0.0 {
        let scale = denom / zero_fraction;
        for i in (0..unique_depth).rev() {
            total += path[i].pweight * scale * inv(unique_depth - i);
        }
    }
    total
}

/// Number of [`PathElement`]s a traversal of a tree of depth `depth` needs:
/// the path at tree level `d` holds at most `d + 1` elements and every level
/// owns its own region, so the regions sum to `(D + 1)(D + 2) / 2` for the
/// deepest level `D`.
fn arena_len(depth: usize) -> usize {
    (depth + 1) * (depth + 2) / 2
}

/// [`RegTree`] node marker for "no child".
const NO_CHILD: u32 = u32::MAX;

/// A tree node with everything TreeSHAP reads per visit precomputed: the
/// child cover ratio (`sum_hess / parent sum_hess`, which the recursion would
/// otherwise divide out at every visit) and the leaf value widened to `f64`.
#[derive(Clone, Copy)]
struct ShapNode {
    feature: u32,
    cond: f32,
    default_left: bool,
    /// Left child, or [`NO_CHILD`] for a leaf.
    left: u32,
    right: u32,
    /// This node's cover divided by its parent's (`0.0` when the parent has
    /// no cover, unused for the root).
    cover_fraction: f64,
    /// Leaf value (`0.0` for internal nodes).
    value: f64,
}

/// A [`RegTree`] prepared for TreeSHAP traversals.
struct ShapTree {
    nodes: Vec<ShapNode>,
    depth: usize,
}

impl ShapTree {
    fn from_tree(tree: &RegTree) -> Self {
        let src = tree.nodes();
        let mut nodes: Vec<ShapNode> = src
            .iter()
            .map(|n| ShapNode {
                feature: n.split_feature,
                cond: n.split_cond,
                default_left: n.default_left,
                left: if n.is_leaf() { NO_CHILD } else { n.left as u32 },
                right: if n.is_leaf() {
                    NO_CHILD
                } else {
                    n.right as u32
                },
                cover_fraction: 0.0,
                value: if n.is_leaf() {
                    n.leaf_value as f64
                } else {
                    0.0
                },
            })
            .collect();
        let mut depth = 0;
        let mut stack = vec![(0usize, 0usize)];
        while let Some((nid, d)) = stack.pop() {
            depth = depth.max(d);
            let n = &src[nid];
            if n.is_leaf() {
                continue;
            }
            let cover = n.sum_hess as f64;
            for child in [n.left as usize, n.right as usize] {
                nodes[child].cover_fraction = if cover > 0.0 {
                    src[child].sum_hess as f64 / cover
                } else {
                    0.0
                };
                stack.push((child, d + 1));
            }
        }
        ShapTree { nodes, depth }
    }
}

/// Per-instance TreeSHAP traversal state shared by the plain and conditioned
/// walks: the tree, the instance's dense feature row (`NaN` = missing), the
/// contribution accumulator, and the conditioning mode.
struct Walk<'a> {
    tree: &'a ShapTree,
    row: &'a [f32],
    phi: &'a mut [f64],
    /// `0`: ordinary contributions. `+1`: `condition_feature` fixed present
    /// (in the coalition). `-1`: fixed absent.
    condition: i32,
    condition_feature: i64,
}

/// Recursive TreeSHAP traversal of a single tree, accumulating per-feature
/// contributions into `walk.phi`.
///
/// `arena` is scratch for this node's decision path and every level below it:
/// its first `level + 1` elements are this node's region, holding the parent's
/// path in `arena[..parent_len]` on entry (copied by the caller, so the path
/// can be forked at each internal node without allocating). `condition_fraction`
/// is the running weight carried down the tree by the conditioning (it starts
/// at `1.0`). When conditioning is active the `condition_feature` is never
/// entered into the decision path, so it receives no attribution of its own.
/// The half-difference of the `+1` and `-1` runs yields the interaction of
/// `condition_feature` with every other feature.
#[allow(clippy::too_many_arguments)]
fn tree_shap_rec(
    walk: &mut Walk<'_>,
    node_index: usize,
    arena: &mut [PathElement],
    level: usize,
    parent_len: usize,
    parent_zero_fraction: f64,
    parent_one_fraction: f64,
    parent_feature_index: i64,
    condition_fraction: f64,
) {
    // No weight flows down this branch under the conditioning: nothing to do.
    if condition_fraction == 0.0 {
        return;
    }
    let (path, rest) = arena.split_at_mut(level + 1);
    let mut len = parent_len;

    // Extend the path with the parent split, unless we are conditioning on the
    // parent feature (in which case it is deliberately kept off the path).
    if walk.condition == 0 || walk.condition_feature != parent_feature_index {
        extend_path(
            path,
            len,
            parent_zero_fraction,
            parent_one_fraction,
            parent_feature_index,
        );
        len += 1;
    }
    let tree = walk.tree;
    let node = &tree.nodes[node_index];
    let unique_depth = len - 1;

    if node.left == NO_CHILD {
        let leaf = node.value;
        let path = &path[..len];
        // For an element with `one_fraction == 0`, `unwound_path_sum` is
        // `(D + 1) / zero_fraction * Σ_j pweight_j / (D - j)` and the leaf
        // factor `(one - zero)` is `-zero_fraction`, so the fraction cancels:
        // every such element contributes the same `-(D + 1) * Σ * leaf`. It is
        // computed once per leaf and added for each of them (most elements: a
        // leaf shares hot edges with the instance's own path only along their
        // common prefix). Elements with both fractions zero contribute nothing.
        let mut cold = None;
        for i in 1..=unique_depth {
            let el = path[i];
            if el.one_fraction != 0.0 {
                let w = unwound_path_sum(path, i);
                walk.phi[el.feature_index as usize] +=
                    w * (el.one_fraction - el.zero_fraction) * leaf * condition_fraction;
            } else if el.zero_fraction != 0.0 {
                let c = *cold.get_or_insert_with(|| {
                    let denom = (unique_depth + 1) as f64;
                    let mut sum = 0.0;
                    for j in (0..unique_depth).rev() {
                        sum += path[j].pweight * inv(unique_depth - j);
                    }
                    -(sum * denom) * leaf * condition_fraction
                });
                walk.phi[el.feature_index as usize] += c;
            }
        }
        return;
    }

    // Route the instance to determine the "hot" (taken) and "cold" child.
    let split = node.feature;
    let v = walk.row[split as usize];
    let go_left = if v.is_nan() {
        node.default_left
    } else {
        v < node.cond
    };
    let (hot, cold) = if go_left {
        (node.left as usize, node.right as usize)
    } else {
        (node.right as usize, node.left as usize)
    };

    // Cover-based child weights: hot/cold fraction = child_cover / node_cover.
    let hot_zero = tree.nodes[hot].cover_fraction;
    let cold_zero = tree.nodes[cold].cover_fraction;

    // If this feature is already on the path, unwind it first so it is not
    // double-counted, carrying its incoming fractions forward.
    let split_i = split as i64;
    let mut incoming_zero = 1.0;
    let mut incoming_one = 1.0;
    let found = path[1..len]
        .iter()
        .position(|e| e.feature_index == split_i)
        .map(|p| p + 1);
    if let Some(pi) = found {
        incoming_zero = path[pi].zero_fraction;
        incoming_one = path[pi].one_fraction;
        unwind_path(&mut path[..len], pi);
        len -= 1;
    }

    // Split the conditioning weight between the two children. When we condition
    // the split feature present, all weight follows the hot (taken) branch; when
    // we condition it absent, each branch keeps only its cover fraction.
    let mut hot_condition_fraction = condition_fraction;
    let mut cold_condition_fraction = condition_fraction;
    if walk.condition > 0 && split_i == walk.condition_feature {
        cold_condition_fraction = 0.0;
    } else if walk.condition < 0 && split_i == walk.condition_feature {
        hot_condition_fraction *= hot_zero;
        cold_condition_fraction *= cold_zero;
    }

    // Each child starts from its own copy of this path.
    rest[..len].copy_from_slice(&path[..len]);
    tree_shap_rec(
        walk,
        hot,
        rest,
        level + 1,
        len,
        hot_zero * incoming_zero,
        incoming_one,
        split_i,
        hot_condition_fraction,
    );
    rest[..len].copy_from_slice(&path[..len]);
    tree_shap_rec(
        walk,
        cold,
        rest,
        level + 1,
        len,
        cold_zero * incoming_zero,
        0.0,
        split_i,
        cold_condition_fraction,
    );
}

/// TreeSHAP contributions of `tree` for one instance, accumulated into `phi`
/// (which must be zeroed by the caller). `arena` must hold at least
/// [`arena_len`] elements for the tree's depth.
fn tree_shap(
    tree: &ShapTree,
    row: &[f32],
    phi: &mut [f64],
    arena: &mut [PathElement],
    condition: i32,
    condition_feature: i64,
) {
    let mut walk = Walk {
        tree,
        row,
        phi,
        condition,
        condition_feature,
    };
    tree_shap_rec(&mut walk, 0, arena, 0, 0, 1.0, 1.0, -1, 1.0);
}

/// Cover-weighted mean prediction of the subtree rooted at `node_index`. This is the
/// tree's expected output `E[f_tree]` when evaluated at the root. This is the
/// offset TreeSHAP folds into the bias term.
fn node_mean_value(tree: &RegTree, node_index: usize) -> f64 {
    let node = tree.node(node_index);
    if node.is_leaf() {
        return node.leaf_value as f64;
    }
    let cover = node.sum_hess as f64;
    if cover <= 0.0 {
        return 0.0;
    }
    let l = node.left as usize;
    let r = node.right as usize;
    let lc = tree.node(l).sum_hess as f64;
    let rc = tree.node(r).sum_hess as f64;
    (lc * node_mean_value(tree, l) + rc * node_mean_value(tree, r)) / cover
}

impl BoostedModel {
    /// Exact TreeSHAP feature contributions, matching XGBoost `pred_contribs=True`.
    ///
    /// For a single-output model the result is row-major with shape
    /// `n_rows × (n_features + 1)`: within each row, columns `0..n_features` are
    /// the per-feature contributions and the final column is the bias
    /// (`base_score` plus each tree's expected value).
    ///
    /// For a multiclass model (`n_outputs > 1`) the layout is
    /// `n_rows × n_outputs × (n_features + 1)`, row-major: the contributions for
    /// row `r`, output `c`, feature `j` live at
    /// `((r * n_outputs + c) * (n_features + 1)) + j`, with the bias at column
    /// `n_features`. Tree `t` contributes to output `t % n_outputs`.
    ///
    /// The key guarantee is exact additivity: for every row (and output) the sum
    /// of the `n_features + 1` values equals the raw margin from
    /// [`BoostedModel::predict_margin`].
    pub fn predict_contribs(&self, data: &DMatrix) -> Result<Vec<f32>> {
        self.validate_prediction_data(data)?;
        let n = data.n_rows();
        let k = self.n_outputs();
        let nf = self.n_features();
        let width = nf + 1;
        let trees = &self.trees()[..self.effective_ntrees()];

        // Each tree's root mean value is instance-independent; compute once.
        let tree_means: Vec<f64> = trees
            .iter()
            .enumerate()
            .map(|(i, t)| node_mean_value(t, 0) * self.tree_weight(i) as f64)
            .collect();
        let shap_trees: Vec<ShapTree> = trees.iter().map(ShapTree::from_tree).collect();
        let arena = arena_len(shap_trees.iter().map(|t| t.depth).max().unwrap_or(0));

        let mut out = vec![0f32; n * k * width];
        let initial = self.initial_margins(data);

        out.par_chunks_mut(k * width).enumerate().for_each_init(
            || {
                (
                    RowBlock::single_rows(data),
                    vec![0f64; k * width],
                    vec![0f64; nf],
                    vec![PathElement::default(); arena],
                )
            },
            |(rows, acc, scratch, arena), (row, out_row)| {
                for a in acc.iter_mut() {
                    *a = 0.0;
                }
                for c in 0..k {
                    acc[c * width + nf] = initial[row * k + c] as f64;
                }
                if let Some(linear) = self.linear() {
                    for f in 0..nf {
                        if let Some(x) = data.get(row, f) {
                            for c in 0..k {
                                acc[c * width + f] += linear.weights()[f * k + c] as f64 * x as f64;
                            }
                        }
                    }
                    for c in 0..k {
                        acc[c * width + nf] += linear.bias()[c] as f64;
                    }
                }
                rows.load(row, 1);
                let get = rows.row(0).expect("single-row blocks are dense");
                for (ti, tree) in shap_trees.iter().enumerate() {
                    let cls = ti % k;
                    let off = cls * width;
                    let weight = self.tree_weight(ti) as f64;
                    if weight == 1.0 {
                        // `1.0 * x` is exact, so unit-weight trees (gbtree)
                        // accumulate straight into the row.
                        tree_shap(tree, get, &mut acc[off..off + nf], arena, 0, -1);
                    } else {
                        scratch.fill(0.0);
                        tree_shap(tree, get, scratch, arena, 0, -1);
                        for f in 0..nf {
                            acc[off + f] += weight * scratch[f];
                        }
                    }
                    // Tree expected value folds into the bias column.
                    acc[off + nf] += tree_means[ti];
                }
                for (o, &v) in out_row.iter_mut().zip(acc.iter()) {
                    *o = v as f32;
                }
            },
        );
        Ok(out)
    }

    /// Exact TreeSHAP interaction values, matching XGBoost `pred_interactions=True`.
    ///
    /// For a single-output model the result is row-major with per-row shape
    /// `(n_features + 1) × (n_features + 1)`. Within a row's matrix `M`:
    ///
    /// * the off-diagonal entry `M[i][j]` (`i, j < n_features`) is the SHAP
    ///   interaction between features `i` and `j` (symmetric: `M[i][j] ==
    ///   M[j][i]`) and is split evenly between the two cells.
    /// * the diagonal entry `M[i][i]` is feature `i`'s *main* effect, set so that
    ///   the row sums to feature `i`'s full SHAP value (its
    ///   [`BoostedModel::predict_contribs`] contribution).
    /// * the final row/column (index `n_features`) carry the bias: `M[nf][nf]`
    ///   holds each tree's expected value `Σ E[f_tree]`, and the remaining bias
    ///   cells are zero.
    ///
    /// Consequently the whole matrix sums to the raw margin from
    /// [`BoostedModel::predict_margin`], including the applicable base margin.
    ///
    /// For a multiclass model (`n_outputs > 1`) the layout is
    /// `n_rows × n_outputs × (n_features + 1)^2`, row-major: the matrix for row
    /// `r`, output `c` occupies the `(n_features + 1)^2` values starting at
    /// `(r * n_outputs + c) * (n_features + 1)^2`. Tree `t` contributes to output
    /// `t % n_outputs`.
    #[allow(clippy::needless_range_loop)]
    pub fn predict_interactions(&self, data: &DMatrix) -> Result<Vec<f32>> {
        self.validate_prediction_data(data)?;
        let n = data.n_rows();
        let k = self.n_outputs();
        let nf = self.n_features();
        let width = nf + 1;
        let mwidth = width * width;
        let trees = &self.trees()[..self.effective_ntrees()];

        // Each tree's root mean value is instance-independent; compute once.
        let tree_means: Vec<f64> = trees
            .iter()
            .enumerate()
            .map(|(i, t)| node_mean_value(t, 0) * self.tree_weight(i) as f64)
            .collect();

        let shap_trees: Vec<ShapTree> = trees.iter().map(ShapTree::from_tree).collect();
        let arena = arena_len(shap_trees.iter().map(|t| t.depth).max().unwrap_or(0));
        let mut out = vec![0f32; n * k * mwidth];
        let initial = self.initial_margins(data);

        // Per-thread scratch: unconditioned contributions, condition = +1
        // (feature present) / -1 (absent) accumulators, the interaction
        // matrices, per-tree phi buffers, and the path arena.
        struct Scratch<'a> {
            rows: RowBlock<'a>,
            diag: Vec<f64>,
            on: Vec<f64>,
            off: Vec<f64>,
            mat: Vec<f64>,
            phi: Vec<f64>,
            phi_on: Vec<f64>,
            phi_off: Vec<f64>,
            arena: Vec<PathElement>,
        }

        out.par_chunks_mut(k * mwidth).enumerate().for_each_init(
            || Scratch {
                rows: RowBlock::single_rows(data),
                diag: vec![0f64; k * width],
                on: vec![0f64; k * width],
                off: vec![0f64; k * width],
                mat: vec![0f64; k * mwidth],
                phi: vec![0f64; nf],
                phi_on: vec![0f64; nf],
                phi_off: vec![0f64; nf],
                arena: vec![PathElement::default(); arena],
            },
            |s, (row, out_row)| {
                s.rows.load(row, 1);
                let get = s.rows.row(0).expect("single-row blocks are dense");
                let (diag, on, off, mat) = (&mut s.diag, &mut s.on, &mut s.off, &mut s.mat);
                diag.fill(0.0);
                mat.fill(0.0);

                for c in 0..k {
                    diag[c * width + nf] = initial[row * k + c] as f64;
                }
                if let Some(linear) = self.linear() {
                    for f in 0..nf {
                        if let Some(x) = data.get(row, f) {
                            for c in 0..k {
                                diag[c * width + f] +=
                                    linear.weights()[f * k + c] as f64 * x as f64;
                            }
                        }
                    }
                    for c in 0..k {
                        diag[c * width + nf] += linear.bias()[c] as f64;
                    }
                }
                for (ti, tree) in shap_trees.iter().enumerate() {
                    let cls = ti % k;
                    let base = cls * width;
                    s.phi.fill(0.0);
                    tree_shap(tree, get, &mut s.phi, &mut s.arena, 0, -1);
                    let weight = self.tree_weight(ti) as f64;
                    for f in 0..nf {
                        diag[base + f] += weight * s.phi[f];
                    }
                    diag[base + nf] += tree_means[ti];
                }
                for c in 0..k {
                    let mbase = c * mwidth;
                    let dbase = c * width;
                    for j in 0..width {
                        mat[mbase + j * width + j] = diag[dbase + j];
                    }
                }

                // 2. Interaction terms: for each feature `j`, the
                //    half-difference of the present/absent conditioned
                //    contributions gives the interaction with every other
                //    feature; the diagonal is reduced so the row keeps
                //    summing to feature `j`'s SHAP value.
                for j in 0..nf {
                    on.fill(0.0);
                    off.fill(0.0);
                    for (ti, tree) in shap_trees.iter().enumerate() {
                        let cls = ti % k;
                        let base = cls * width;
                        s.phi_on.fill(0.0);
                        s.phi_off.fill(0.0);
                        tree_shap(tree, get, &mut s.phi_on, &mut s.arena, 1, j as i64);
                        tree_shap(tree, get, &mut s.phi_off, &mut s.arena, -1, j as i64);
                        let weight = self.tree_weight(ti) as f64;
                        for f in 0..nf {
                            on[base + f] += weight * s.phi_on[f];
                            off[base + f] += weight * s.phi_off[f];
                        }
                    }
                    for c in 0..k {
                        let mbase = c * mwidth;
                        let dbase = c * width;
                        for kk in 0..width {
                            // The conditioned feature `j` never attributes to
                            // itself (on/off are zero there), so `kk == j`
                            // contributes 0.
                            let val = 0.5 * (on[dbase + kk] - off[dbase + kk]);
                            mat[mbase + j * width + kk] += val;
                            mat[mbase + j * width + j] -= val;
                        }
                    }
                }

                for (o, &v) in out_row.iter_mut().zip(mat.iter()) {
                    *o = v as f32;
                }
            },
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{BoosterKind, TrainingParams};
    use crate::data::DMatrix;
    use crate::learner::train;

    /// Build a small dense dataset with `nf` features. Features 0 and 1 carry
    /// signal, the rest are noise. Returns (data, n_rows).
    fn make_data(n: usize, nf: usize) -> DMatrix {
        let mut x = vec![0f32; n * nf];
        let mut y = vec![0f32; n];
        for i in 0..n {
            for j in 0..nf {
                // Deterministic pseudo-random values.
                let v = ((i * 31 + j * 17 + 7) % 97) as f32 / 97.0;
                x[i * nf + j] = v;
            }
            let f0 = x[i * nf];
            let f1 = x[i * nf + 1];
            y[i] = 2.0 * f0 - 1.5 * f1 + 0.3;
        }
        DMatrix::from_dense(&x, n, nf)
            .unwrap()
            .with_labels(&y)
            .unwrap()
    }

    #[test]
    fn additivity_single_output() {
        let n = 80;
        let nf = 5;
        let d = make_data(n, nf);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(4)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 30).unwrap();

        let contribs = model.predict_contribs(&d).unwrap();
        let margin = model.predict_margin(&d).unwrap();
        let width = nf + 1;
        assert_eq!(contribs.len(), n * width);

        let mut max_err = 0f64;
        for row in 0..n {
            let s: f64 = contribs[row * width..row * width + width]
                .iter()
                .map(|&v| v as f64)
                .sum();
            let err = (s - margin[row] as f64).abs();
            max_err = max_err.max(err);
        }
        assert!(
            max_err < 1e-4,
            "max additivity error {max_err} exceeded 1e-4"
        );
    }

    #[test]
    fn additivity_multiclass() {
        let n = 90;
        let nf = 4;
        let k = 3;
        let mut x = vec![0f32; n * nf];
        let mut y = vec![0f32; n];
        for i in 0..n {
            for j in 0..nf {
                x[i * nf + j] = ((i * 13 + j * 29 + 3) % 101) as f32 / 101.0;
            }
            y[i] = (i % k) as f32;
        }
        let d = DMatrix::from_dense(&x, n, nf)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("multi:softprob")
            .num_class(k)
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 15).unwrap();
        assert_eq!(model.n_outputs(), k);

        let contribs = model.predict_contribs(&d).unwrap();
        let margin = model.predict_margin(&d).unwrap();
        let width = nf + 1;
        assert_eq!(contribs.len(), n * k * width);

        let mut max_err = 0f64;
        for row in 0..n {
            for c in 0..k {
                let base = (row * k + c) * width;
                let s: f64 = contribs[base..base + width].iter().map(|&v| v as f64).sum();
                let err = (s - margin[row * k + c] as f64).abs();
                max_err = max_err.max(err);
            }
        }
        assert!(
            max_err < 1e-4,
            "max multiclass additivity error {max_err} exceeded 1e-4"
        );
    }

    #[test]
    fn unused_feature_has_zero_contribution() {
        // Feature 3 is pure constant noise (never predictive) AND we verify no
        // split uses it; its contribution must be ~0 for every row.
        let n = 70;
        let nf = 4;
        let mut x = vec![0f32; n * nf];
        let mut y = vec![0f32; n];
        for i in 0..n {
            x[i * nf] = ((i * 7 + 1) % 53) as f32 / 53.0;
            x[i * nf + 1] = ((i * 11 + 2) % 53) as f32 / 53.0;
            x[i * nf + 2] = ((i * 5 + 3) % 53) as f32 / 53.0;
            x[i * nf + 3] = 0.5; // constant -> never a useful split
            y[i] = 3.0 * x[i * nf] - 2.0 * x[i * nf + 1];
        }
        let d = DMatrix::from_dense(&x, n, nf)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(4)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 25).unwrap();

        // Sanity: feature 3 is never used in any split.
        let used = model
            .trees()
            .iter()
            .flat_map(|t| t.nodes().iter())
            .any(|nd| !nd.is_leaf() && nd.split_feature == 3);
        assert!(!used, "feature 3 unexpectedly used in a split");

        let contribs = model.predict_contribs(&d).unwrap();
        let width = nf + 1;
        let mut max_abs = 0f32;
        for row in 0..n {
            max_abs = max_abs.max(contribs[row * width + 3].abs());
        }
        assert!(
            max_abs < 1e-6,
            "unused feature contribution {max_abs} not ~0"
        );
    }

    #[test]
    fn interactions_single_output() {
        let n = 80;
        let nf = 5;
        let d = make_data(n, nf);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(4)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 30).unwrap();

        let width = nf + 1;
        let mwidth = width * width;
        let inter = model.predict_interactions(&d).unwrap();
        assert_eq!(inter.len(), n * mwidth);

        let contribs = model.predict_contribs(&d).unwrap();
        let margin = model.predict_margin(&d).unwrap();

        let mut max_row_err = 0f64;
        let mut max_eff_err = 0f64;
        let mut max_sym_err = 0f64;
        for row in 0..n {
            let m = &inter[row * mwidth..row * mwidth + mwidth];
            // Row consistency: each feature row sums to its SHAP contribution.
            for i in 0..nf {
                let s: f64 = (0..width).map(|j| m[i * width + j] as f64).sum();
                let cval = contribs[row * width + i] as f64;
                max_row_err = max_row_err.max((s - cval).abs());
            }
            // Efficiency: the whole matrix sums to the full margin.
            let total: f64 = m.iter().map(|&v| v as f64).sum();
            let target = margin[row] as f64;
            max_eff_err = max_eff_err.max((total - target).abs());
            // Symmetry.
            for i in 0..width {
                for j in 0..width {
                    let e = (m[i * width + j] as f64 - m[j * width + i] as f64).abs();
                    max_sym_err = max_sym_err.max(e);
                }
            }
        }
        assert!(max_row_err < 1e-4, "row-consistency error {max_row_err}");
        assert!(max_eff_err < 1e-4, "efficiency error {max_eff_err}");
        assert!(max_sym_err < 1e-5, "symmetry error {max_sym_err}");
    }

    #[test]
    fn interactions_multiclass() {
        let n = 90;
        let nf = 4;
        let k = 3;
        let mut x = vec![0f32; n * nf];
        let mut y = vec![0f32; n];
        for i in 0..n {
            for j in 0..nf {
                x[i * nf + j] = ((i * 13 + j * 29 + 3) % 101) as f32 / 101.0;
            }
            y[i] = (i % k) as f32;
        }
        let d = DMatrix::from_dense(&x, n, nf)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("multi:softprob")
            .num_class(k)
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 15).unwrap();
        assert_eq!(model.n_outputs(), k);

        let width = nf + 1;
        let mwidth = width * width;
        let inter = model.predict_interactions(&d).unwrap();
        assert_eq!(inter.len(), n * k * mwidth);

        let contribs = model.predict_contribs(&d).unwrap();
        let margin = model.predict_margin(&d).unwrap();

        let mut max_row_err = 0f64;
        let mut max_eff_err = 0f64;
        let mut max_sym_err = 0f64;
        for row in 0..n {
            for c in 0..k {
                let m = &inter[(row * k + c) * mwidth..(row * k + c) * mwidth + mwidth];
                let cbase = (row * k + c) * width;
                for i in 0..nf {
                    let s: f64 = (0..width).map(|j| m[i * width + j] as f64).sum();
                    let cval = contribs[cbase + i] as f64;
                    max_row_err = max_row_err.max((s - cval).abs());
                }
                let total: f64 = m.iter().map(|&v| v as f64).sum();
                let target = margin[row * k + c] as f64;
                max_eff_err = max_eff_err.max((total - target).abs());
                for i in 0..width {
                    for j in 0..width {
                        let e = (m[i * width + j] as f64 - m[j * width + i] as f64).abs();
                        max_sym_err = max_sym_err.max(e);
                    }
                }
            }
        }
        assert!(max_row_err < 1e-4, "row-consistency error {max_row_err}");
        assert!(max_eff_err < 1e-4, "efficiency error {max_eff_err}");
        assert!(max_sym_err < 1e-5, "symmetry error {max_sym_err}");
    }

    #[test]
    fn additivity_with_base_margins_dart_and_gblinear() {
        let n = 48;
        let x: Vec<f32> = (0..n)
            .flat_map(|row| [row as f32 / n as f32, (row % 7) as f32])
            .collect();
        let y: Vec<f32> = (0..n).map(|row| row as f32 / 10.0).collect();
        let base: Vec<f32> = (0..n).map(|row| row as f32 / 100.0).collect();
        let d = DMatrix::from_dense(&x, n, 2)
            .unwrap()
            .with_labels(&y)
            .unwrap()
            .with_base_margin(&base)
            .unwrap();

        for booster in [BoosterKind::Dart, BoosterKind::GbLinear] {
            let params = TrainingParams::builder()
                .booster(booster)
                .rate_drop(0.5)
                .eta(0.2)
                .max_depth(2)
                .build()
                .unwrap();
            let model = train(&params, &d, 8).unwrap();
            let margin = model.predict_margin(&d).unwrap();
            let contribs = model.predict_contribs(&d).unwrap();
            let interactions = model.predict_interactions(&d).unwrap();
            let width = d.n_cols() + 1;
            for row in 0..n {
                let contribution_sum: f32 = contribs[row * width..(row + 1) * width].iter().sum();
                let interaction_sum: f32 = interactions
                    [row * width * width..(row + 1) * width * width]
                    .iter()
                    .sum();
                assert!((contribution_sum - margin[row]).abs() < 1e-4);
                assert!((interaction_sum - margin[row]).abs() < 1e-4);
            }
        }
    }
}
