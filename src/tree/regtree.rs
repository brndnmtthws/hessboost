//! Regression tree representation and prediction.
//!
//! A tree is a flat array of [`Node`]s (node `0` is the root). Internal nodes
//! carry a numeric split `x[feature] < threshold`. Instances whose feature is
//! *missing* follow the node's `default_left` direction, implementing XGBoost's
//! sparsity-aware routing. Leaf nodes carry the raw leaf weight (the learning
//! rate is applied by the boosting loop, not baked into the tree).
//!
//! A *vector-leaf* tree (`multi_strategy = multi_output_tree`, XGBoost's
//! `MultiTargetTree`) shares one split structure across `K > 1` outputs and
//! stores a weight vector per leaf ([`RegTree::leaf_vector`]); its scalar
//! [`Node::leaf_value`]s are unused (zero).

use crate::data::DMatrix;
use crate::tree::in_category_set;
use crate::tree::linear::LinearLeaves;
use serde::{Deserialize, Serialize};

/// Sentinel used in child pointers to mark "no child" (i.e. a leaf).
const NO_CHILD: i32 = -1;

/// A single tree node.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// Split feature index (meaningful only for internal nodes).
    pub split_feature: u32,
    /// Split threshold: an instance goes left when `value < split_cond`.
    pub split_cond: f32,
    /// Direction taken by instances with a missing value at this node.
    pub default_left: bool,
    /// Left child index, or `-1` for a leaf.
    pub left: i32,
    /// Right child index, or `-1` for a leaf.
    pub right: i32,
    /// Leaf weight (used only for leaves).
    pub leaf_value: f32,
    /// Sum of Hessians routed through this node (for cover-based importance/SHAP).
    pub sum_hess: f32,
    /// Loss reduction (gain) achieved by this node's split (0 for leaves).
    pub split_gain: f32,
    /// Whether this internal node splits on a categorical feature by set
    /// membership rather than a numeric threshold. `false` for numeric splits
    /// and leaves.
    pub is_categorical: bool,
    /// For a categorical node, the start index into the owning tree's category
    /// list of the categories routed left. Unused (`0`) otherwise.
    pub cat_begin: u32,
    /// For a categorical node, the end index (exclusive) into the owning tree's
    /// category list of the categories routed left. Unused (`0`) otherwise.
    pub cat_end: u32,
}

impl Node {
    /// A fresh leaf node with the given weight and cover.
    pub(crate) fn leaf(value: f32, sum_hess: f32) -> Self {
        Node {
            split_feature: 0,
            split_cond: 0.0,
            default_left: true,
            left: NO_CHILD,
            right: NO_CHILD,
            leaf_value: value,
            sum_hess,
            split_gain: 0.0,
            is_categorical: false,
            cat_begin: 0,
            cat_end: 0,
        }
    }

    /// Whether this node is a leaf.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.left == NO_CHILD
    }
}

/// A regression tree: a flat node array with node `0` as the root.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct RegTree {
    nodes: Vec<Node>,
    /// Flat pool of category values routed left by categorical nodes. Node
    /// `n` (when `n.is_categorical`) owns `categories[n.cat_begin..n.cat_end]`.
    /// Empty for trees with no categorical splits.
    categories: Vec<u32>,
    /// Outputs per leaf of a vector-leaf tree (XGBoost's `size_leaf_vector`,
    /// `> 1`), or `0` for a scalar tree.
    size_leaf_vector: usize,
    /// Vector-leaf weights laid out `[node][output]` (`size_leaf_vector` per
    /// node; internal nodes hold zeros). Empty for scalar trees.
    leaf_vectors: Vec<f32>,
    /// Per-leaf linear models of a `linear_tree` tree ([`LinearLeaves`]);
    /// `None` for constant-leaf trees (every tree unless `linear_tree` is on).
    linear: Option<LinearLeaves>,
}

impl RegTree {
    /// Create an empty tree with a placeholder root leaf (node `0`), ready to
    /// be grown by a builder.
    pub(crate) fn with_root(sum_hess: f32) -> Self {
        Self::from_scalar_parts(vec![Node::leaf(0.0, sum_hess)], Vec::new())
    }

    /// Assemble a constant-leaf scalar tree from its node array and the flat
    /// category pool its categorical nodes index. The caller validates the
    /// result ([`RegTree::is_valid_for_features`]).
    pub(crate) fn from_scalar_parts(nodes: Vec<Node>, categories: Vec<u32>) -> Self {
        RegTree {
            nodes,
            categories,
            size_leaf_vector: 0,
            leaf_vectors: Vec::new(),
            linear: None,
        }
    }

    /// Assemble a tree from every stored part, as the native binary format
    /// keeps them: `size_leaf_vector` is `0` for a scalar tree, and
    /// `leaf_vectors` holds `size_leaf_vector` weights per node. The caller
    /// validates the result ([`RegTree::is_valid_for_features`]).
    pub(crate) fn from_parts(
        nodes: Vec<Node>,
        categories: Vec<u32>,
        size_leaf_vector: usize,
        leaf_vectors: Vec<f32>,
        linear: Option<LinearLeaves>,
    ) -> Self {
        RegTree {
            nodes,
            categories,
            size_leaf_vector,
            leaf_vectors,
            linear,
        }
    }

    /// The stored leaf-vector width (`0` for a scalar tree) and weights, as
    /// [`RegTree::from_parts`] takes them.
    pub(crate) fn leaf_vector_parts(&self) -> (usize, &[f32]) {
        (self.size_leaf_vector, &self.leaf_vectors)
    }

    /// Create a vector-leaf tree with `n_outputs > 1` weights per leaf and a
    /// placeholder (all-zero) root leaf.
    pub(crate) fn with_vector_root(n_outputs: usize, sum_hess: f32) -> Self {
        debug_assert!(n_outputs > 1);
        RegTree {
            nodes: vec![Node::leaf(0.0, sum_hess)],
            categories: Vec::new(),
            size_leaf_vector: n_outputs,
            leaf_vectors: vec![0.0; n_outputs],
            linear: None,
        }
    }

    /// Outputs per leaf: `1` for a scalar tree, `K > 1` for a vector-leaf
    /// tree.
    #[inline]
    pub fn size_leaf_vector(&self) -> usize {
        self.size_leaf_vector.max(1)
    }

    /// Whether this is a vector-leaf (multi-output) tree.
    #[inline]
    pub fn is_vector_leaf(&self) -> bool {
        self.size_leaf_vector > 1
    }

    /// The weights of leaf `nid`, one per output: the leaf's vector for a
    /// vector-leaf tree, the single [`Node::leaf_value`] otherwise.
    #[inline]
    pub fn leaf_vector(&self, nid: usize) -> &[f32] {
        if self.is_vector_leaf() {
            let k = self.size_leaf_vector;
            &self.leaf_vectors[nid * k..(nid + 1) * k]
        } else {
            std::slice::from_ref(&self.nodes[nid].leaf_value)
        }
    }

    /// Set the weight vector of vector-leaf tree node `nid`.
    pub(crate) fn set_leaf_vector(&mut self, nid: usize, values: &[f32]) {
        let k = self.size_leaf_vector;
        debug_assert!(k > 1 && values.len() == k);
        self.leaf_vectors[nid * k..(nid + 1) * k].copy_from_slice(values);
    }

    /// The scalar tree predicting output `output` of this vector-leaf tree:
    /// the same nodes (splits, covers, gains, categories) with each leaf's
    /// value taken from its vector.
    pub(crate) fn output_tree(&self, output: usize) -> RegTree {
        let k = self.size_leaf_vector;
        debug_assert!(k > 1 && output < k);
        let mut nodes = self.nodes.clone();
        for (id, node) in nodes.iter_mut().enumerate() {
            if node.is_leaf() {
                node.leaf_value = self.leaf_vectors[id * k + output];
            }
        }
        RegTree::from_scalar_parts(nodes, self.categories.clone())
    }

    /// Give two freshly pushed child nodes their (zero) leaf vectors.
    fn grow_leaf_vectors(&mut self) {
        if self.is_vector_leaf() {
            self.leaf_vectors
                .resize(self.nodes.len() * self.size_leaf_vector, 0.0);
        }
    }

    /// Number of nodes (internal + leaf).
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of leaf nodes.
    pub fn num_leaves(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    pub(crate) fn is_valid_for_features(&self, n_features: usize) -> bool {
        let locally_valid = !self.nodes.is_empty()
            && self.nodes.iter().all(|node| {
                node.sum_hess.is_finite()
                    && node.leaf_value.is_finite()
                    && node.split_cond.is_finite()
                    && node.split_gain.is_finite()
                    && (node.is_leaf()
                        || ((node.split_feature as usize) < n_features
                            && node.left >= 0
                            && node.right >= 0
                            && (node.left as usize) < self.nodes.len()
                            && (node.right as usize) < self.nodes.len()
                            && (!node.is_categorical
                                || (node.cat_begin <= node.cat_end
                                    && (node.cat_end as usize) <= self.categories.len()))))
            })
            && self
                .linear
                .as_ref()
                .is_none_or(|linear| linear.is_valid(&self.nodes, n_features));
        if !locally_valid {
            return false;
        }
        if self.is_vector_leaf()
            && (self.leaf_vectors.len() != self.nodes.len() * self.size_leaf_vector
                || self.leaf_vectors.iter().any(|w| !w.is_finite()))
        {
            return false;
        }
        // A leaf is a scalar constant, a vector, or a scalar linear model:
        // vector-leaf consumers ignore linear payloads, so both at once
        // would make them disagree with `predict_row`.
        if self.size_leaf_vector == 1
            || (!self.is_vector_leaf() && !self.leaf_vectors.is_empty())
            || (self.is_vector_leaf() && self.linear.is_some())
        {
            return false;
        }
        let mut seen = vec![false; self.nodes.len()];
        let mut stack = vec![0usize];
        while let Some(node_id) = stack.pop() {
            if seen[node_id] {
                return false;
            }
            seen[node_id] = true;
            let node = &self.nodes[node_id];
            if !node.is_leaf() {
                stack.push(node.left as usize);
                stack.push(node.right as usize);
            }
        }
        seen.into_iter().all(|visited| visited)
    }

    /// Read-only access to the node array.
    #[inline]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Flat pool of categories routed left by categorical nodes.
    #[inline]
    pub(crate) fn categories(&self) -> &[u32] {
        &self.categories
    }

    /// The categories categorical node `node` of this tree routes left.
    #[inline]
    pub(crate) fn node_categories(&self, node: &Node) -> &[u32] {
        &self.categories[node.cat_begin as usize..node.cat_end as usize]
    }

    /// The per-leaf linear models, when this is a linear-leaf tree (trained
    /// with `linear_tree`). Such a leaf predicts its linear model, or its
    /// constant `leaf_value` for rows missing one of the model's features.
    #[inline]
    pub fn linear_leaves(&self) -> Option<&LinearLeaves> {
        self.linear.as_ref()
    }

    /// Attach fitted leaf linear models.
    pub(crate) fn set_linear_leaves(&mut self, linear: LinearLeaves) {
        self.linear = Some(linear);
    }

    /// Access a node by id.
    #[inline]
    pub fn node(&self, id: usize) -> &Node {
        &self.nodes[id]
    }

    /// Turn leaf `nid` into an internal node by attaching two child leaves.
    /// Returns `(left_id, right_id)`.
    ///
    /// Both builders overwrite these child values in their finalize pass (which
    /// recomputes every leaf from stored stats and bounds), so the values here
    /// are placeholders on that path — but the parameters stay: directly built
    /// trees (tests, learners) rely on them as the real leaf weights.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn expand(
        &mut self,
        nid: usize,
        split_feature: u32,
        split_cond: f32,
        default_left: bool,
        left_value: f32,
        left_hess: f32,
        right_value: f32,
        right_hess: f32,
    ) -> (usize, usize) {
        let n = &mut self.nodes[nid];
        n.split_feature = split_feature;
        n.split_cond = split_cond;
        n.default_left = default_left;
        self.attach_children(nid, left_value, left_hess, right_value, right_hess)
    }

    /// Turn leaf `nid` into a categorical (set-membership) internal node.
    /// Instances whose value of `split_feature` is one of `cats_left` go to the
    /// left child, other present categories go right, and missing values
    /// follow `default_left`. Returns `(left_id, right_id)`.
    ///
    /// As in [`expand`](Self::expand), builders overwrite the child values when
    /// finalizing; the parameters serve directly built trees.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn expand_categorical(
        &mut self,
        nid: usize,
        split_feature: u32,
        cats_left: &[u32],
        default_left: bool,
        left_value: f32,
        left_hess: f32,
        right_value: f32,
        right_hess: f32,
    ) -> (usize, usize) {
        let begin = self.categories.len() as u32;
        self.categories.extend_from_slice(cats_left);
        let end = self.categories.len() as u32;
        let n = &mut self.nodes[nid];
        n.split_feature = split_feature;
        n.is_categorical = true;
        n.cat_begin = begin;
        n.cat_end = end;
        n.default_left = default_left;
        self.attach_children(nid, left_value, left_hess, right_value, right_hess)
    }

    /// Push two child leaves of `nid` (with their zero leaf vectors) and
    /// point `nid` at them. Returns `(left_id, right_id)`.
    fn attach_children(
        &mut self,
        nid: usize,
        left_value: f32,
        left_hess: f32,
        right_value: f32,
        right_hess: f32,
    ) -> (usize, usize) {
        let left_id = self.nodes.len();
        let right_id = left_id + 1;
        self.nodes.push(Node::leaf(left_value, left_hess));
        self.nodes.push(Node::leaf(right_value, right_hess));
        let n = &mut self.nodes[nid];
        n.left = left_id as i32;
        n.right = right_id as i32;
        self.grow_leaf_vectors();
        (left_id, right_id)
    }

    /// Set a leaf's weight (used to finalize leaf values after growth).
    pub(crate) fn set_leaf_value(&mut self, nid: usize, value: f32) {
        self.nodes[nid].leaf_value = value;
    }

    /// Record the loss reduction achieved by an internal node's split.
    pub(crate) fn set_split_gain(&mut self, nid: usize, gain: f32) {
        self.nodes[nid].split_gain = gain;
    }

    /// Record the Hessian sum (cover) of the instances reaching node `nid`.
    pub(crate) fn set_sum_hess(&mut self, nid: usize, sum_hess: f32) {
        self.nodes[nid].sum_hess = sum_hess;
    }

    /// Multiply every leaf weight by `factor`. Used to apply the learning rate
    /// (shrinkage) so that stored trees already carry their scaled contribution,
    /// matching XGBoost's saved-model semantics. Leaf linear models are scaled
    /// with them.
    pub fn scale_leaves(&mut self, factor: f32) {
        let k = self.size_leaf_vector;
        for (id, n) in self.nodes.iter_mut().enumerate() {
            if n.is_leaf() {
                n.leaf_value *= factor;
                if k > 1 {
                    for w in &mut self.leaf_vectors[id * k..(id + 1) * k] {
                        *w *= factor;
                    }
                }
            }
        }
        if let Some(linear) = &mut self.linear {
            linear.scale(f64::from(factor));
        }
    }

    /// Route a single feature vector (via an accessor) to its leaf id.
    ///
    /// `get` returns `None` for a missing feature. Generic over the accessor so
    /// the same code serves dense rows, sparse rows, and SHAP traversals. Each
    /// level loads its node once (re-indexing through `child` measured
    /// ~7% slower).
    pub fn leaf_id_with(&self, get: impl Fn(u32) -> Option<f32>) -> usize {
        let nodes = &self.nodes[..];
        let mut nid = 0usize;
        loop {
            let node = &nodes[nid];
            if node.is_leaf() {
                return nid;
            }
            nid = if self.goes_left(node, get(node.split_feature)) {
                node.left as usize
            } else {
                node.right as usize
            };
        }
    }

    /// The child of internal node `nid` that an instance whose split-feature
    /// value is `value` (`None` = missing) descends to.
    #[inline]
    pub(crate) fn child(&self, nid: usize, value: Option<f32>) -> usize {
        let node = &self.nodes[nid];
        if self.goes_left(node, value) {
            node.left as usize
        } else {
            node.right as usize
        }
    }

    /// Whether an instance whose split-feature value is `value` (`None` =
    /// missing) goes left at internal node `node` of this tree.
    #[inline]
    pub(crate) fn goes_left(&self, node: &Node, value: Option<f32>) -> bool {
        match value {
            // Categories are integer-coded; membership in the left set routes
            // left, everything else (present, not in set) right.
            Some(v) if node.is_categorical => in_category_set(self.node_categories(node), v),
            Some(v) => v < node.split_cond,
            None => node.default_left,
        }
    }

    /// Route a dense feature row (indexed by feature id, `missing` sentinel for
    /// absent values) to its leaf id.
    #[inline]
    pub fn leaf_id_dense(&self, row: &[f32], missing: f32) -> usize {
        self.leaf_id_with(|f| {
            let v = row[f as usize];
            (!crate::data::is_missing(v, missing)).then_some(v)
        })
    }

    /// Predict the raw output of row `row` of `data`: its leaf's weight, or
    /// the leaf's linear model for linear-leaf trees.
    pub fn predict_row(&self, data: &DMatrix, row: usize) -> f32 {
        let get = |f: u32| data.get(row, f as usize);
        let leaf = self.leaf_id_with(get);
        let constant = self.nodes[leaf].leaf_value;
        match &self.linear {
            Some(linear) => linear.predict(leaf, constant, get),
            None => constant,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small tree by hand:
    /// root: feature 0 < 0.5 ? left : right, missing -> left
    ///   left leaf = -1.0, right leaf = +2.0
    fn stump() -> RegTree {
        let mut t = RegTree::with_root(10.0);
        t.expand(0, 0, 0.5, true, -1.0, 5.0, 2.0, 5.0);
        t
    }

    #[test]
    fn routing_numeric() {
        let t = stump();
        // value 0.2 < 0.5 -> left leaf -1.0
        assert_eq!(t.leaf_id_with(|_| Some(0.2)), 1);
        // value 0.9 >= 0.5 -> right leaf +2.0
        assert_eq!(t.leaf_id_with(|_| Some(0.9)), 2);
    }

    #[test]
    fn routing_missing_follows_default() {
        let t = stump();
        // missing -> default_left = true -> left leaf
        assert_eq!(t.leaf_id_with(|_| None), 1);
        assert_eq!(t.node(1).leaf_value, -1.0);
    }

    #[test]
    fn routing_categorical_set_membership() {
        // Categorical split: categories {0, 2} go left, everything else right.
        let mut t = RegTree::with_root(10.0);
        t.expand_categorical(0, 0, &[0, 2], false, -1.0, 5.0, 2.0, 5.0);
        assert!(t.node(0).is_categorical);
        // In-set categories route left (leaf 1, value -1.0).
        assert_eq!(t.leaf_id_with(|_| Some(0.0)), 1);
        assert_eq!(t.leaf_id_with(|_| Some(2.0)), 1);
        // Out-of-set present categories route right (leaf 2, value 2.0).
        assert_eq!(t.leaf_id_with(|_| Some(1.0)), 2);
        assert_eq!(t.leaf_id_with(|_| Some(3.0)), 2);
        // Unseen category also routes right (not in the left set).
        assert_eq!(t.leaf_id_with(|_| Some(9.0)), 2);
        // Missing follows default_left = false -> right.
        assert_eq!(t.leaf_id_with(|_| None), 2);
    }

    #[test]
    fn predict_row_dense() {
        let t = stump();
        let d = DMatrix::from_dense(&[0.1, 0.9], 2, 1).unwrap();
        assert_eq!(t.predict_row(&d, 0), -1.0);
        assert_eq!(t.predict_row(&d, 1), 2.0);
        assert_eq!(t.num_leaves(), 2);
        assert_eq!(t.num_nodes(), 3);
    }

    /// A vector-leaf tree carrying a linear-leaf payload is invalid: vector
    /// consumers would drop the linear models `predict_row` uses.
    #[test]
    fn vector_leaves_refuse_linear_payload() {
        let linear: LinearLeaves = serde_json::from_str(
            r#"{"offsets":[0,1],"intercepts":[0.5],"features":[0],"coeffs":[2.0]}"#,
        )
        .unwrap();
        let mut scalar = RegTree::with_root(1.0);
        scalar.set_linear_leaves(linear.clone());
        assert!(scalar.is_valid_for_features(1));
        let mut vector = RegTree::with_vector_root(2, 1.0);
        assert!(vector.is_valid_for_features(1));
        vector.set_linear_leaves(linear);
        assert!(!vector.is_valid_for_features(1));
    }
}
