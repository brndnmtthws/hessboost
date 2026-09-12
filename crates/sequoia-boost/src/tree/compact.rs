//! Prediction-optimized tree layout.
//!
//! [`RegTree`] keeps every node attribute (gain, cover, categorical ranges) in
//! a 40-byte node, and traversal branches on the split outcome. For batch
//! prediction that is dominated by branch mispredictions: split directions are
//! data dependent and essentially unpredictable, so each level costs a pipeline
//! flush. [`CompactForest`] re-lays every tree of an ensemble out for a
//! branch-free walk, in one shared node arena:
//!
//! - nodes are renumbered breadth-first so the two children of a node are
//!   adjacent, letting the step compute `next = left + (go_right as u32)`
//!   arithmetically; child indices are absolute within the arena, so lanes
//!   walking different trees share one base pointer;
//! - every numeric split is expressed as one ordered compare `v' > cond'` that
//!   is false for missing values (`NaN`): a split whose missing values go left
//!   stores `cond' = next_below(cond)` (so `v > cond'` is `v >= cond`), and a
//!   split whose missing values go right stores the children mirrored with
//!   `cond' = -cond` and a sign mask that negates `v` (`-v > -cond` is
//!   `v < cond`), so a step is load, XOR, compare, add, with no select;
//! - leaves store `cond' = +inf` and point at themselves, so a walk can run a
//!   fixed number of steps (the tree depth) without testing for termination;
//!   several rows (or, for a single row, several trees) are walked in lockstep
//!   so their dependent load chains overlap.
//!
//! Leaf ids are arena indices; [`CompactForest::original_id`] maps them back
//! to [`RegTree`] node ids for `predict_leaf`.

use crate::tree::RegTree;

/// Rows (or trees) walked in lockstep by the fixed-depth kernel.
pub(crate) const LANES: usize = 16;

/// Deeper trees than this fall back to the early-exit walk: the fixed-depth
/// kernel would spend most steps parked on already-reached leaves.
const MAX_FIXED_DEPTH: u32 = 16;

/// `cond` bit pattern (`+inf`) marking a leaf: nothing compares greater, so a
/// leaf always selects `left`, which points at itself.
const LEAF_COND: u32 = f32::INFINITY.to_bits();
/// `aux` for a numeric node whose missing values go right: flips the sign of
/// the feature value so `v < cond` becomes `-v > -cond`.
const NEGATE: u32 = 1 << 31;
/// `aux` bit: set-membership split; `cond` holds `cat_begin` and
/// `aux >> CAT_END_SHIFT` holds `cat_end`. Numeric `aux` values are `0` or
/// [`NEGATE`], so this bit distinguishes them.
const CATEGORICAL: u32 = 1;
/// `aux` bit (categorical nodes): missing values go left.
const CAT_DEFAULT_LEFT: u32 = 2;
const CAT_END_SHIFT: u32 = 2;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct CNode {
    /// Split feature (kept unmasked: the row load depends on it).
    feat: u32,
    /// Numeric threshold in the "go right when greater" form (`+inf` for a
    /// leaf). For categorical nodes the bit pattern holds `cat_begin`.
    cond: f32,
    /// Arena index of the child selected when the compare is false; the other
    /// child is `left + 1`. A leaf points at itself.
    left: u32,
    /// Numeric node: sign mask XORed into the feature value (`0` or
    /// [`NEGATE`]). Leaf: the leaf value's bits. Categorical: [`CATEGORICAL`],
    /// [`CAT_DEFAULT_LEFT`], and the set end.
    aux: u32,
}

/// Largest finite value strictly below `c`, so that `v > next_below(c)` is
/// exactly `v >= c` for every non-`NaN` `v`.
fn next_below(c: f32) -> f32 {
    debug_assert!(c.is_finite());
    if c > 0.0 {
        f32::from_bits(c.to_bits() - 1)
    } else if c == 0.0 {
        -f32::from_bits(1)
    } else {
        f32::from_bits(c.to_bits() + 1)
    }
}

/// Per-tree summary of the arena region a tree occupies.
#[derive(Debug, Clone, Copy)]
struct TreeMeta {
    /// Arena index of the root.
    root: u32,
    depth: u32,
    has_categorical: bool,
    /// Largest split feature index (0 for a single leaf).
    max_feature: u32,
}

impl TreeMeta {
    /// Whether the fixed-depth lockstep kernel applies to this tree.
    #[inline]
    fn lockstep_ok(&self) -> bool {
        !self.has_categorical && self.depth <= MAX_FIXED_DEPTH
    }

    /// Check once that every split feature of this tree indexes a row of
    /// `n_cols` values, so the kernels may load `row[feature]` unchecked.
    #[inline]
    fn check_width(&self, n_cols: usize) {
        assert!(
            (self.max_feature as usize) < n_cols,
            "row has {n_cols} features but the tree splits on feature {}",
            self.max_feature
        );
    }
}

/// An ensemble of [`RegTree`]s re-laid out for prediction. See the module docs.
#[derive(Debug, Clone)]
pub(crate) struct CompactForest {
    nodes: Vec<CNode>,
    /// Arena index → original [`RegTree`] node id within its tree.
    orig_id: Vec<u32>,
    /// Every tree's category pool, concatenated; node ranges are absolute.
    categories: Vec<u32>,
    trees: Vec<TreeMeta>,
}

impl CompactForest {
    pub(crate) fn from_trees(trees: &[RegTree]) -> Self {
        let total: usize = trees.iter().map(RegTree::num_nodes).sum();
        let mut forest = CompactForest {
            nodes: Vec::with_capacity(total),
            orig_id: Vec::with_capacity(total),
            categories: Vec::new(),
            trees: Vec::with_capacity(trees.len()),
        };
        for tree in trees {
            forest.push_tree(tree);
        }
        forest
    }

    fn push_tree(&mut self, tree: &RegTree) {
        let src = tree.nodes();
        let base = self.nodes.len() as u32;
        let cat_base = self.categories.len() as u32;
        self.categories.extend_from_slice(tree.categories());
        // Breadth-first order: children are pushed as an adjacent pair, the
        // `false` child (missing values' destination) first.
        let mut order: Vec<u32> = Vec::with_capacity(src.len());
        let mut new_id = vec![u32::MAX; src.len()];
        let mut depth_of = vec![0u32; src.len()];
        order.push(0);
        new_id[0] = base;
        let mut i = 0;
        while i < order.len() {
            let old = order[i] as usize;
            let n = &src[old];
            if !n.is_leaf() {
                // Numeric splits whose missing values go right are stored
                // mirrored; categorical nodes keep (left, right).
                let (first, second) = if n.default_left || n.is_categorical {
                    (n.left, n.right)
                } else {
                    (n.right, n.left)
                };
                for child in [first as usize, second as usize] {
                    new_id[child] = base + order.len() as u32;
                    depth_of[child] = depth_of[old] + 1;
                    order.push(child as u32);
                }
            }
            i += 1;
        }
        let mut has_categorical = false;
        let mut depth = 0u32;
        let mut max_feature = 0u32;
        for &old in &order {
            let n = &src[old as usize];
            let id = base + self.nodes.len() as u32 - base;
            depth = depth.max(depth_of[old as usize]);
            let node = if n.is_leaf() {
                CNode {
                    feat: 0,
                    cond: f32::from_bits(LEAF_COND),
                    left: id,
                    aux: n.leaf_value.to_bits(),
                }
            } else if n.is_categorical {
                has_categorical = true;
                max_feature = max_feature.max(n.split_feature);
                let (begin, end) = (cat_base + n.cat_begin, cat_base + n.cat_end);
                assert!(
                    begin != LEAF_COND && end < (1 << (32 - CAT_END_SHIFT)),
                    "categorical set range does not fit the compact encoding"
                );
                let mut aux = CATEGORICAL | (end << CAT_END_SHIFT);
                if n.default_left {
                    aux |= CAT_DEFAULT_LEFT;
                }
                CNode {
                    feat: n.split_feature,
                    cond: f32::from_bits(begin),
                    left: new_id[n.left as usize],
                    aux,
                }
            } else if n.default_left {
                // go right (to `right`) iff v >= cond  <=>  v > next_below(cond)
                max_feature = max_feature.max(n.split_feature);
                CNode {
                    feat: n.split_feature,
                    cond: next_below(n.split_cond),
                    left: new_id[n.left as usize],
                    aux: 0,
                }
            } else {
                // children mirrored: go to `left` (second) iff v < cond
                //   <=>  -v > -cond
                max_feature = max_feature.max(n.split_feature);
                CNode {
                    feat: n.split_feature,
                    cond: -n.split_cond,
                    left: new_id[n.right as usize],
                    aux: NEGATE,
                }
            };
            self.nodes.push(node);
            self.orig_id.push(old);
        }
        self.trees.push(TreeMeta {
            root: base,
            depth,
            has_categorical,
            max_feature,
        });
    }

    /// Leaf value of arena node `id` (must be a leaf).
    #[inline]
    pub(crate) fn leaf_value(&self, id: u32) -> f32 {
        f32::from_bits(self.nodes[id as usize].aux)
    }

    /// Original [`RegTree`] node id (within its tree) of arena node `id`.
    #[inline]
    pub(crate) fn original_id(&self, id: u32) -> u32 {
        self.orig_id[id as usize]
    }

    #[inline]
    fn is_leaf(node: &CNode) -> bool {
        node.cond.to_bits() == LEAF_COND
    }

    /// Whether `v` (non-missing) belongs to the categorical node's left set.
    #[inline]
    fn in_left_set(&self, node: &CNode, v: f32) -> bool {
        let begin = node.cond.to_bits() as usize;
        let end = (node.aux >> CAT_END_SHIFT) as usize;
        let c = v as u32;
        self.categories[begin..end].contains(&c)
    }

    /// Child of internal `node` selected by value `v` (`NaN` = missing).
    /// Handles numeric and categorical splits.
    #[inline]
    fn next(&self, node: &CNode, v: f32) -> u32 {
        if node.aux & CATEGORICAL != 0 {
            let go_left = if v.is_nan() {
                node.aux & CAT_DEFAULT_LEFT != 0
            } else {
                self.in_left_set(node, v)
            };
            node.left + u32::from(!go_left)
        } else {
            Self::next_numeric(node, v) as u32
        }
    }

    /// [`Self::next`] for a numeric node: one ordered compare, no select (see
    /// the module docs for the encoding). Leaves yield themselves. The result
    /// is `usize`: with `u32` lane state LLVM emits a ~3x slower loop on
    /// aarch64.
    #[inline(always)]
    fn next_numeric(node: &CNode, v: f32) -> usize {
        let v = f32::from_bits(v.to_bits() ^ node.aux);
        node.left as usize + usize::from(v > node.cond)
    }

    /// Early-exit walk of a single row through tree `t`.
    #[inline]
    pub(crate) fn leaf_id(&self, t: usize, row: &[f32]) -> u32 {
        let mut nid = self.trees[t].root;
        loop {
            let node = &self.nodes[nid as usize];
            if Self::is_leaf(node) {
                return nid;
            }
            nid = self.next(node, row[node.feat as usize]);
        }
    }

    /// Early-exit walk through an accessor (`None` = missing), for rows that
    /// are not materialized densely.
    pub(crate) fn leaf_id_with(&self, t: usize, get: impl Fn(u32) -> Option<f32>) -> u32 {
        let mut nid = self.trees[t].root;
        loop {
            let node = &self.nodes[nid as usize];
            if Self::is_leaf(node) {
                return nid;
            }
            let v = get(node.feat).unwrap_or(f32::NAN);
            nid = self.next(node, v);
        }
    }

    /// Early-exit walk of one row stored with stride `stride` (`row[f * stride]`
    /// is feature `f`) through tree `t`.
    #[inline]
    fn leaf_id_strided(&self, t: usize, row: &[f32], stride: usize) -> u32 {
        let mut nid = self.trees[t].root;
        loop {
            let node = &self.nodes[nid as usize];
            if Self::is_leaf(node) {
                return nid;
            }
            nid = self.next(node, row[node.feat as usize * stride]);
        }
    }

    /// Walk `rows` dense rows (`NaN` = missing) through tree `t` and call
    /// `sink(r, leaf)` with each row's arena leaf id, in row order. The full
    /// [`LANES`]-row groups come from `lanes`, laid out `[group][feature][lane]`
    /// so a lane's value sits at a fixed immediate offset from the group's
    /// feature base (`n_cols * LANES` values per group); the remaining
    /// `rows % LANES` rows come from `tail`, row-major with stride `n_cols`.
    #[inline(always)]
    fn walk_block(
        &self,
        t: usize,
        lanes: &[f32],
        tail: &[f32],
        n_cols: usize,
        rows: usize,
        mut sink: impl FnMut(usize, u32),
    ) {
        let groups = rows / LANES;
        let group_len = LANES * n_cols;
        assert!(
            lanes.len() >= groups * group_len && tail.len() >= (rows - groups * LANES) * n_cols,
            "row block holds fewer rows than requested"
        );
        let meta = self.trees[t];
        if !meta.lockstep_ok() {
            for g in 0..groups {
                let grp = &lanes[g * group_len..(g + 1) * group_len];
                for j in 0..LANES {
                    sink(g * LANES + j, self.leaf_id_strided(t, &grp[j..], LANES));
                }
            }
        } else {
            meta.check_width(n_cols);
            let nodes = &self.nodes[..];
            let root = meta.root as usize;
            for g in 0..groups {
                let grp = &lanes[g * group_len..(g + 1) * group_len];
                let mut nid = [root; LANES];
                for _ in 0..meta.depth {
                    // Fully unrolled so the lane states live in registers; a
                    // rolled loop keeps them on the stack and serializes on
                    // the store-to-load round trip.
                    macro_rules! lane {
                        ($($j:literal)*) => {$(
                            // SAFETY: `nid[j]` is always an arena node id (the
                            // root, then children produced by
                            // `next_numeric`), `check_width` verified every
                            // split feature < n_cols, and `grp` holds n_cols
                            // features of LANES values.
                            let node = unsafe { nodes.get_unchecked(nid[$j]) };
                            let v = unsafe { *grp.get_unchecked(node.feat as usize * LANES + $j) };
                            nid[$j] = Self::next_numeric(node, v);
                        )*};
                    }
                    lane!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);
                }
                for (j, &id) in nid.iter().enumerate() {
                    sink(g * LANES + j, id as u32);
                }
            }
        }
        for (i, row) in tail
            .chunks_exact(n_cols)
            .take(rows - groups * LANES)
            .enumerate()
        {
            sink(groups * LANES + i, self.leaf_id(t, row));
        }
    }

    /// `out[r * stride] = original leaf id of row r` in tree `t` for `rows`
    /// dense rows given as lane-major groups plus a row-major tail (see
    /// [`Self::walk_block`]); `NaN` marks missing values.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn original_leaf_ids(
        &self,
        t: usize,
        lanes: &[f32],
        tail: &[f32],
        n_cols: usize,
        rows: usize,
        out: &mut [u32],
        stride: usize,
    ) {
        assert!(rows == 0 || out.len() > (rows - 1) * stride);
        let orig = &self.orig_id[..];
        self.walk_block(t, lanes, tail, n_cols, rows, |r, leaf| {
            // SAFETY: `r < rows` (asserted above against `out`) and `leaf` is
            // an arena node id produced by the walk.
            unsafe {
                *out.get_unchecked_mut(r * stride) = *orig.get_unchecked(leaf as usize);
            }
        });
    }

    /// `out[r * stride] += weight * leaf_value(row r)` in tree `t` for `rows`
    /// dense rows given as lane-major groups plus a row-major tail (see
    /// [`Self::walk_block`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn accumulate(
        &self,
        t: usize,
        lanes: &[f32],
        tail: &[f32],
        n_cols: usize,
        rows: usize,
        weight: f32,
        out: &mut [f32],
        stride: usize,
    ) {
        assert!(rows == 0 || out.len() > (rows - 1) * stride);
        let nodes = &self.nodes[..];
        self.walk_block(t, lanes, tail, n_cols, rows, |r, leaf| {
            // SAFETY: `r < rows` (asserted above against `out`) and `leaf` is
            // an arena node id produced by the walk.
            unsafe {
                *out.get_unchecked_mut(r * stride) +=
                    weight * f32::from_bits(nodes.get_unchecked(leaf as usize).aux);
            }
        });
    }

    /// Walk one dense `row` through trees `0..limit` and call `sink(t, leaf)`
    /// with each tree's arena leaf id, in tree order. Trees are walked
    /// [`LANES`] at a time in lockstep, so a single instance still overlaps its
    /// dependent load chains (the batch kernel overlaps rows instead).
    #[inline(always)]
    fn walk_row(&self, row: &[f32], limit: usize, mut sink: impl FnMut(usize, u32)) {
        assert!(limit <= self.trees.len());
        let nodes = &self.nodes[..];
        let full = limit / LANES * LANES;
        for g in 0..limit / LANES {
            let group = &self.trees[g * LANES..(g + 1) * LANES];
            let mut depth = 0u32;
            let mut ok = true;
            for meta in group {
                ok &= meta.lockstep_ok();
                depth = depth.max(meta.depth);
            }
            if !ok {
                for j in 0..LANES {
                    sink(g * LANES + j, self.leaf_id(g * LANES + j, row));
                }
                continue;
            }
            for meta in group {
                meta.check_width(row.len());
            }
            let mut nid = [0usize; LANES];
            for (n, meta) in nid.iter_mut().zip(group) {
                *n = meta.root as usize;
            }
            for _ in 0..depth {
                macro_rules! lane {
                    ($($j:literal)*) => {$(
                        // SAFETY: as in `walk_block`; `check_width` ran for
                        // every tree in the group.
                        let node = unsafe { nodes.get_unchecked(nid[$j]) };
                        let v = unsafe { *row.get_unchecked(node.feat as usize) };
                        nid[$j] = Self::next_numeric(node, v);
                    )*};
                }
                lane!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);
            }
            for (j, &id) in nid.iter().enumerate() {
                sink(g * LANES + j, id as u32);
            }
        }
        for t in full..limit {
            sink(t, self.leaf_id(t, row));
        }
    }

    /// Original leaf ids of one dense `row` in trees `0..out.len()`, written
    /// to `out[t]`.
    pub(crate) fn original_leaf_ids_for_row(&self, row: &[f32], out: &mut [u32]) {
        let orig = &self.orig_id[..];
        self.walk_row(row, out.len(), |t, leaf| out[t] = orig[leaf as usize]);
    }

    /// `out[t % k] += weight(t) * leaf_value(row, tree t)` for trees
    /// `0..limit` of one dense `row`.
    pub(crate) fn accumulate_row(
        &self,
        row: &[f32],
        limit: usize,
        weight: impl Fn(usize) -> f32,
        out: &mut [f32],
    ) {
        let k = out.len();
        let nodes = &self.nodes[..];
        self.walk_row(row, limit, |t, leaf| {
            out[t % k] += weight(t) * f32::from_bits(nodes[leaf as usize].aux);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> RegTree {
        // Root splits on f0 < 0.5 (missing left); left child splits on f1 < 2
        // (missing right) into leaves -1 / 1; right child is leaf 5.
        let mut t = RegTree::with_root(1.0);
        let (l, _r) = t.expand(0, 0, 0.5, true, 0.0, 1.0, 5.0, 1.0);
        t.expand(l, 1, 2.0, false, -1.0, 1.0, 1.0, 1.0);
        t
    }

    /// Row-major rows -> (lane-major full groups, row-major tail), the layout
    /// `walk_block` consumes.
    fn split_lanes(rows: &[f32], n_cols: usize) -> (Vec<f32>, &[f32]) {
        let groups = rows.len() / n_cols / LANES;
        let mut lanes = vec![0.0f32; groups * LANES * n_cols];
        for g in 0..groups {
            for j in 0..LANES {
                for f in 0..n_cols {
                    lanes[g * LANES * n_cols + f * LANES + j] = rows[(g * LANES + j) * n_cols + f];
                }
            }
        }
        (lanes, &rows[groups * LANES * n_cols..])
    }

    #[test]
    fn matches_reference_walk() {
        let t = tree();
        let f = CompactForest::from_trees(std::slice::from_ref(&t));
        assert_eq!(f.trees[0].depth, 2);
        let rows: Vec<[f32; 2]> = vec![
            [0.1, 1.0],
            [0.1, 3.0],
            [0.9, 0.0],
            [f32::NAN, 1.0],
            [f32::NAN, f32::NAN],
            [0.1, f32::NAN],
        ];
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let mut out = vec![0u32; rows.len()];
        f.original_leaf_ids(0, &[], &flat, 2, rows.len(), &mut out, 1);
        for (r, row) in rows.iter().enumerate() {
            let want = t.leaf_id_dense(row, f32::NAN);
            assert_eq!(out[r] as usize, want, "row {r}");
            let leaf = f.leaf_id(0, row);
            assert_eq!(f.original_id(leaf) as usize, want);
            assert_eq!(f.leaf_value(leaf), t.node(want).leaf_value);
        }
    }

    #[test]
    fn lockstep_matches_scalar_on_full_blocks() {
        let t = tree();
        let f = CompactForest::from_trees(&[t]);
        let n = 3 * LANES + 5;
        let block: Vec<f32> = (0..n * 2)
            .map(|i| {
                if i % 7 == 0 {
                    f32::NAN
                } else {
                    (i % 5) as f32 * 0.3
                }
            })
            .collect();
        let (lanes, tail) = split_lanes(&block, 2);
        let mut ids = vec![0u32; n * 3];
        f.original_leaf_ids(0, &lanes, tail, 2, n, &mut ids[1..], 3);
        let mut acc = vec![0.5f32; n * 2];
        f.accumulate(0, &lanes, tail, 2, n, 2.0, &mut acc, 2);
        for r in 0..n {
            let leaf = f.leaf_id(0, &block[r * 2..r * 2 + 2]);
            assert_eq!(ids[r * 3 + 1], f.original_id(leaf));
            assert_eq!(acc[r * 2], 0.5 + 2.0 * f.leaf_value(leaf));
            assert_eq!(acc[r * 2 + 1], 0.5);
        }
    }

    #[test]
    fn tree_lockstep_matches_per_tree_walk() {
        // 2 full lane groups plus a remainder, with a categorical tree forcing
        // one group onto the fallback path.
        let mut trees: Vec<RegTree> = (0..2 * LANES + 3)
            .map(|i| {
                let mut t = RegTree::with_root(1.0);
                let (l, _r) = t.expand(
                    0,
                    i as u32 % 3,
                    0.1 * i as f32,
                    i % 2 == 0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                );
                t.expand(l, 1, 0.5, i % 4 == 0, -1.0, 1.0, 2.0, 1.0);
                t
            })
            .collect();
        let mut cat = RegTree::with_root(1.0);
        cat.expand_categorical(0, 2, &[1, 3], false, -1.0, 1.0, 1.0, 1.0);
        trees[LANES + 1] = cat;
        let f = CompactForest::from_trees(&trees);
        let row = [0.7f32, f32::NAN, 3.0];
        let mut out = vec![0u32; trees.len()];
        f.original_leaf_ids_for_row(&row, &mut out);
        let mut acc = vec![0.25f32; 3];
        f.accumulate_row(&row, trees.len(), |t| 1.0 + t as f32, &mut acc);
        let mut want_acc = vec![0.25f32; 3];
        for (t, tree) in trees.iter().enumerate() {
            let want = tree.leaf_id_dense(&row, f32::NAN);
            assert_eq!(out[t] as usize, want, "tree {t}");
            want_acc[t % 3] += (1.0 + t as f32) * tree.node(want).leaf_value;
        }
        assert_eq!(acc, want_acc);
    }

    #[test]
    fn categorical_membership() {
        for default_left in [true, false] {
            let mut t = RegTree::with_root(1.0);
            t.expand_categorical(0, 0, &[2, 5], default_left, -1.0, 1.0, 1.0, 1.0);
            // A preceding tree with its own categories shifts the pool.
            let mut first = RegTree::with_root(1.0);
            first.expand_categorical(0, 0, &[9], true, 0.0, 1.0, 0.0, 1.0);
            let f = CompactForest::from_trees(&[first, t.clone()]);
            assert!(f.trees[1].has_categorical);
            for v in [0.0f32, 2.0, 5.0, 7.0, f32::NAN] {
                let want = t.leaf_id_dense(&[v], f32::NAN);
                let got = f.leaf_id(1, &[v]);
                assert_eq!(f.original_id(got) as usize, want, "v={v} dl={default_left}");
                assert_eq!(f.leaf_value(got), t.node(want).leaf_value);
            }
        }
    }
}
