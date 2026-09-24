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
//!   arithmetically. Child indices are absolute within the arena, so lanes
//!   walking different trees share one base pointer.
//! - every numeric split is expressed as one ordered compare `v' > cond'` that
//!   is false for missing values (`NaN`): a split whose missing values go left
//!   stores `cond' = next_below(cond)` (so `v > cond'` is `v >= cond`), and a
//!   split whose missing values go right stores the children mirrored with
//!   `cond' = -cond` and reads the negated feature value (`-v > -cond` is
//!   `v < cond`).
//! - values and thresholds are compared as monotone unsigned integer
//!   [`key`]s, which keeps the whole step in integer registers: load, load,
//!   compare, add. A missing value's key is `0`, below every real key, so it
//!   never compares greater. Batch rows are stored as keys of `v` and of `-v`
//!   side by side ([`FEATURE_LANES`] per feature), and a mirrored node simply
//!   addresses the negated half, so no per-step sign flip is needed.
//! - leaves store `cond' = +inf` and point at themselves, so a walk can run a
//!   fixed number of steps (the tree depth) without testing for termination.
//!   Several rows (or, for a single row, several trees) are walked in lockstep
//!   so their dependent load chains overlap.
//!
//! Leaf ids are arena indices, and [`CompactForest::original_id`] maps them back
//! to [`RegTree`] node ids for `predict_leaf`.
//!
//! A vector-leaf tree's leaves store, instead of their value's bits, the offset
//! of their weight vector in a shared arena ([`CompactForest::accumulate_vector`]);
//! the walk itself is unchanged.
//!
//! Symmetric trees (one split per level) are additionally indexed by
//! [`SymmetricTables`], whose bit-pattern walk replaces the per-node walk for
//! full lane groups and reaches the same arena leaves.

use crate::tree::oblivious::{ArenaNode, SymmetricTables};
use crate::tree::{RegTree, in_category_set, scalar_tree_output};

/// Rows (or trees) walked in lockstep by the fixed-depth kernel.
pub(crate) const LANES: usize = 16;
/// Key slots per feature in a lane group: the [`LANES`] keys of `v` followed
/// by the [`LANES`] keys of `-v`.
pub(crate) const FEATURE_LANES: usize = 2 * LANES;

/// Deeper trees than this fall back to the early-exit walk: the fixed-depth
/// kernel would spend most steps parked on already-reached leaves.
const MAX_FIXED_DEPTH: u32 = 16;

/// Largest feature index whose `slot` encoding fits a `u32`: the mirrored
/// variant stores `feature * FEATURE_LANES + LANES`, so anything larger
/// would wrap and address another feature's keys.
const MAX_SLOT_FEATURE: u32 = u32::MAX / FEATURE_LANES as u32;
const SIGN: u32 = 1 << 31;

/// Monotone unsigned key of an `f32`: `key(a) > key(b)` iff `a > b` for
/// non-`NaN` inputs (`-0.0` and `+0.0` share a key), and every `NaN` maps to
/// `0`, strictly below `key(-inf)`, so a missing value never compares greater
/// than a threshold.
#[inline(always)]
pub(crate) fn key(v: f32) -> u32 {
    if v.is_nan() {
        return 0;
    }
    // `-0.0 + 0.0` is `+0.0`; every other value is unchanged.
    let bits = (v + 0.0).to_bits();
    // Negative: complement all bits (reverses their order below zero).
    // Non-negative: set the sign bit (places them above every negative).
    bits ^ ((((bits as i32) >> 31) as u32) | SIGN)
}

/// Inverse of [`key`] up to the `NaN` payload and the sign of zero.
#[inline(always)]
fn unkey(key: u32) -> f32 {
    f32::from_bits(if key & SIGN != 0 { key ^ SIGN } else { !key })
}

/// Transpose the full [`LANES`]-row groups of the row-major `rows` into
/// `lanes` as `[group][feature][lane]` keys of `v` and of `-v`
/// ([`FEATURE_LANES`] per feature), so a lane's key sits at a fixed
/// immediate offset from the node's slot.
pub(crate) fn fill_lanes(lanes: &mut Vec<u32>, rows: &[f32], n_cols: usize) {
    let groups = rows.len() / n_cols / LANES;
    lanes.clear();
    lanes.resize(groups * FEATURE_LANES * n_cols, 0);
    for (g, dst) in lanes.chunks_exact_mut(FEATURE_LANES * n_cols).enumerate() {
        let src = &rows[g * LANES * n_cols..(g + 1) * LANES * n_cols];
        for (j, row) in src.chunks_exact(n_cols).enumerate() {
            for (f, &v) in row.iter().enumerate() {
                dst[f * FEATURE_LANES + j] = key(v);
                dst[f * FEATURE_LANES + LANES + j] = key(-v);
            }
        }
    }
}

/// Row-major rows -> (keyed lane-major full groups, row-major tail), the
/// layout the batch kernels consume.
#[cfg(test)]
pub(crate) fn split_lanes(rows: &[f32], n_cols: usize) -> (Vec<u32>, &[f32]) {
    let mut lanes = Vec::new();
    fill_lanes(&mut lanes, rows, n_cols);
    let groups = rows.len() / n_cols / LANES;
    (lanes, &rows[groups * LANES * n_cols..])
}

/// `key` of a leaf (`+inf`): no real key is greater, so a leaf always selects
/// `left`, which points at itself.
const LEAF_KEY: u32 = 0xFF80_0000;
const _: () = assert!(LEAF_KEY == f32::INFINITY.to_bits() | SIGN);

// `slot_key` reads `slot` and `key` as one little-endian `u64` on x86-64.
#[cfg(target_arch = "x86_64")]
const _: () = assert!(
    std::mem::size_of::<CNode>() == 16
        && std::mem::offset_of!(CNode, slot) == 0
        && std::mem::offset_of!(CNode, key) == 4
);
/// `aux` bit marking a set-membership split. `key` holds `cat_begin` and
/// `aux >> CAT_END_SHIFT` holds `cat_end`. Numeric `aux` is `0`, so this bit
/// distinguishes them.
const CATEGORICAL: u32 = 1;
/// `aux` bit (categorical nodes): missing values go left.
const CAT_DEFAULT_LEFT: u32 = 2;
const CAT_END_SHIFT: u32 = 2;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct CNode {
    /// Key slot of the split feature within a lane group:
    /// `feature * FEATURE_LANES`, plus `LANES` for a mirrored numeric node
    /// that reads the negated value. Kept unmasked: the row load depends on it.
    slot: u32,
    /// [`key`] of the numeric threshold in the "go right when greater" form
    /// ([`LEAF_KEY`] for a leaf). For categorical nodes it holds `cat_begin`.
    key: u32,
    /// Arena index of the child selected when the compare is false. The other
    /// child is `left + 1`. A leaf points at itself.
    left: u32,
    /// Numeric node: `0`. Leaf: the leaf value's bits, or for a vector-leaf
    /// tree the offset of its weight vector in `leaf_vectors`. Categorical:
    /// [`CATEGORICAL`], [`CAT_DEFAULT_LEFT`], and the set end.
    aux: u32,
}

impl CNode {
    #[inline(always)]
    fn feature(&self) -> usize {
        self.slot as usize / FEATURE_LANES
    }

    /// Sign mask a raw value is `XORed` with before keying: `SIGN` for a
    /// mirrored numeric node, `0` otherwise.
    #[inline(always)]
    fn negate_mask(&self) -> u32 {
        (self.slot & LANES as u32) << (31 - LANES.trailing_zeros())
    }
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

/// A block of dense rows as the batch kernels consume it: the full
/// [`LANES`]-row groups keyed lane-major (see [`fill_lanes`]) and the
/// remaining `rows % LANES` rows raw.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LaneBlock<'a> {
    /// The full groups, `[group][feature][lane]` as [`key`]s of `v` then of
    /// `-v` ([`FEATURE_LANES`] per feature, `n_cols * FEATURE_LANES` per
    /// group).
    pub(crate) lanes: &'a [u32],
    /// The remaining rows, row-major with stride `n_cols` (`NaN` = missing).
    pub(crate) tail: &'a [f32],
    /// Features per row.
    pub(crate) n_cols: usize,
    /// Rows in the block: the groups' plus the tail's.
    pub(crate) rows: usize,
}

/// An ensemble of [`RegTree`]s re-laid out for prediction. See the module docs.
#[derive(Debug, Clone)]
pub(crate) struct CompactForest {
    nodes: Vec<CNode>,
    /// Arena index → original [`RegTree`] node id within its tree.
    orig_id: Vec<u32>,
    /// Every tree's category pool, concatenated, so node ranges are absolute.
    categories: Vec<u32>,
    /// Every vector-leaf tree's leaf weight vectors, concatenated.
    leaf_vectors: Vec<f32>,
    trees: Vec<TreeMeta>,
    symmetric: SymmetricTables,
}

impl CompactForest {
    pub(crate) fn from_trees(trees: &[RegTree]) -> Self {
        let total: usize = trees.iter().map(RegTree::num_nodes).sum();
        let mut forest = CompactForest {
            nodes: Vec::with_capacity(total),
            orig_id: Vec::with_capacity(total),
            categories: Vec::new(),
            leaf_vectors: Vec::new(),
            trees: Vec::with_capacity(trees.len()),
            symmetric: SymmetricTables::default(),
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
            let id = self.nodes.len() as u32;
            depth = depth.max(depth_of[old as usize]);
            let node = if n.is_leaf() {
                let aux = if tree.is_vector_leaf() {
                    let offset = u32::try_from(self.leaf_vectors.len())
                        .expect("leaf vectors exceed the compact encoding");
                    self.leaf_vectors
                        .extend_from_slice(tree.leaf_vector(old as usize));
                    offset
                } else {
                    n.leaf_value.to_bits()
                };
                CNode {
                    slot: 0,
                    key: LEAF_KEY,
                    left: id,
                    aux,
                }
            } else if n.is_categorical {
                has_categorical = true;
                max_feature = max_feature.max(n.split_feature);
                let (begin, end) = (cat_base + n.cat_begin, cat_base + n.cat_end);
                assert!(
                    end < (1 << (32 - CAT_END_SHIFT)),
                    "categorical set range does not fit the compact encoding"
                );
                let mut aux = CATEGORICAL | (end << CAT_END_SHIFT);
                if n.default_left {
                    aux |= CAT_DEFAULT_LEFT;
                }
                CNode {
                    slot: n.split_feature * FEATURE_LANES as u32,
                    key: begin,
                    left: new_id[n.left as usize],
                    aux,
                }
            } else if n.default_left {
                // go right (to `right`) iff v >= cond  <=>  v > next_below(cond)
                max_feature = max_feature.max(n.split_feature);
                CNode {
                    slot: n.split_feature * FEATURE_LANES as u32,
                    key: key(next_below(n.split_cond)),
                    left: new_id[n.left as usize],
                    aux: 0,
                }
            } else {
                // children mirrored: go to `left` (second) iff v < cond
                //   <=>  -v > -cond, read from the negated key half
                max_feature = max_feature.max(n.split_feature);
                CNode {
                    slot: n.split_feature * FEATURE_LANES as u32 + LANES as u32,
                    key: key(-n.split_cond),
                    left: new_id[n.right as usize],
                    aux: 0,
                }
            };
            self.nodes.push(node);
            self.orig_id.push(old);
        }
        assert!(
            max_feature <= MAX_SLOT_FEATURE,
            "feature index {max_feature} does not fit the compact split encoding"
        );
        self.trees.push(TreeMeta {
            root: base,
            depth,
            has_categorical,
            max_feature,
        });
        let nodes = &self.nodes;
        self.symmetric.push(base, |id| {
            let node = &nodes[id as usize];
            if Self::is_leaf(node, id) {
                ArenaNode::Leaf(f32::from_bits(node.aux))
            } else if node.aux & CATEGORICAL != 0 {
                ArenaNode::Other
            } else {
                ArenaNode::Numeric {
                    slot: node.slot,
                    key: node.key,
                    first: node.left,
                }
            }
        });
    }

    /// Weight vector (`k` outputs) of vector-leaf arena node `id`.
    #[inline]
    pub(crate) fn leaf_vector(&self, id: u32, k: usize) -> &[f32] {
        let offset = self.nodes[id as usize].aux as usize;
        &self.leaf_vectors[offset..offset + k]
    }

    /// Whether tree `t` is walked by bit pattern ([`SymmetricTables`]).
    #[cfg(test)]
    pub(crate) fn is_symmetric(&self, t: usize) -> bool {
        self.symmetric.get(t).is_some()
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

    /// Leaves point at themselves. Children are laid out after their parent,
    /// so no internal node does. (A leaf's key is also [`LEAF_KEY`], but an
    /// internal `+inf` threshold shares that key, so it is not the test.)
    #[inline]
    fn is_leaf(node: &CNode, nid: u32) -> bool {
        node.left == nid
    }

    /// Whether `v` (non-missing) belongs to the categorical node's left set.
    #[inline]
    fn in_left_set(&self, node: &CNode, v: f32) -> bool {
        let begin = node.key as usize;
        let end = (node.aux >> CAT_END_SHIFT) as usize;
        in_category_set(&self.categories[begin..end], v)
    }

    /// Child of internal `node` selected by raw value `v` (`NaN` = missing).
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
            let v = f32::from_bits(v.to_bits() ^ node.negate_mask());
            Self::next_numeric(node, key(v)) as u32
        }
    }

    /// [`Self::next`] for a numeric node given the key of the (already
    /// sign-adjusted) feature value: one unsigned compare, no select (see the
    /// module docs for the encoding). Leaves yield themselves. The result is
    /// `usize`: with `u32` lane state LLVM emits a ~3x slower loop on aarch64.
    #[inline(always)]
    fn next_numeric(node: &CNode, key: u32) -> usize {
        node.left as usize + usize::from(key > node.key)
    }

    /// `(slot, key)` of a node. On x86-64 the lockstep kernel is bound by
    /// load-port throughput, so both fields are fetched with one 64-bit load
    /// (a load per lane and level saved). Other architectures, which were
    /// tuned with plain field loads, keep them.
    #[inline(always)]
    fn slot_key(node: &CNode) -> (usize, u32) {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: `CNode` is `repr(C)` with `slot` then `key` as its first
            // eight bytes (asserted above), x86-64 is little-endian, and an
            // unaligned read through a valid reference is sound.
            let packed = unsafe {
                std::ptr::from_ref::<CNode>(node)
                    .cast::<u64>()
                    .read_unaligned()
            };
            (packed as u32 as usize, (packed >> 32) as u32)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            (node.slot as usize, node.key)
        }
    }

    /// Early-exit walk of a single row through tree `t`.
    #[inline]
    pub(crate) fn leaf_id(&self, t: usize, row: &[f32]) -> u32 {
        self.leaf_id_with(t, |f| Some(row[f as usize]))
    }

    /// Early-exit walk through an accessor (`None` = missing), for rows that
    /// are not materialized densely.
    #[inline]
    pub(crate) fn leaf_id_with(&self, t: usize, get: impl Fn(u32) -> Option<f32>) -> u32 {
        let mut nid = self.trees[t].root;
        loop {
            let node = &self.nodes[nid as usize];
            if Self::is_leaf(node, nid) {
                return nid;
            }
            let v = get(node.feature() as u32).unwrap_or(f32::NAN);
            nid = self.next(node, v);
        }
    }

    /// Early-exit walk of one lane of a key group through tree `t`:
    /// `grp[node.slot + lane]` is the lane's key for the node's feature and
    /// sign. Categorical nodes recover the value from the unsigned half.
    #[inline]
    fn leaf_id_keyed(&self, t: usize, grp: &[u32], lane: usize) -> u32 {
        let mut nid = self.trees[t].root;
        loop {
            let node = &self.nodes[nid as usize];
            if Self::is_leaf(node, nid) {
                return nid;
            }
            nid = if node.aux & CATEGORICAL != 0 {
                self.next(node, unkey(grp[node.slot as usize + lane]))
            } else {
                Self::next_numeric(node, grp[node.slot as usize + lane]) as u32
            };
        }
    }

    /// Walk the rows of `block` through tree `t` and call `sink(r, leaf)`
    /// with each row's arena leaf id, in row order. The full [`LANES`]-row
    /// groups come from the keyed lanes, so a lane's key sits at a fixed
    /// immediate offset from the node's slot; the remaining rows come from
    /// the raw tail.
    #[inline(always)]
    fn walk_block(&self, t: usize, block: LaneBlock<'_>, mut sink: impl FnMut(usize, u32)) {
        let LaneBlock {
            lanes,
            tail,
            n_cols,
            rows,
        } = block;
        let groups = rows / LANES;
        let group_len = FEATURE_LANES * n_cols;
        assert!(
            lanes.len() >= groups * group_len && tail.len() >= (rows - groups * LANES) * n_cols,
            "row block holds fewer rows than requested"
        );
        let meta = self.trees[t];
        if let Some(symmetric) = self.symmetric.get(t) {
            meta.check_width(n_cols);
            symmetric.walk(lanes, groups, group_len, &mut sink);
        } else if meta.lockstep_ok() {
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
                            // features of FEATURE_LANES keys.
                            let node = unsafe { nodes.get_unchecked(nid[$j]) };
                            let (slot, key) = Self::slot_key(node);
                            // SAFETY: see above; `slot + j` indexes that
                            // feature's FEATURE_LANES keys.
                            let k = unsafe { *grp.get_unchecked(slot + $j) };
                            nid[$j] = node.left as usize + usize::from(k > key);
                        )*};
                    }
                    lane!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);
                }
                for (j, &id) in nid.iter().enumerate() {
                    sink(g * LANES + j, id as u32);
                }
            }
        } else {
            for g in 0..groups {
                let grp = &lanes[g * group_len..(g + 1) * group_len];
                for j in 0..LANES {
                    sink(g * LANES + j, self.leaf_id_keyed(t, grp, j));
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

    /// `out[r * stride] = original leaf id of row r` in tree `t` for the rows
    /// of `block`.
    pub(crate) fn original_leaf_ids(
        &self,
        t: usize,
        block: LaneBlock<'_>,
        out: &mut [u32],
        stride: usize,
    ) {
        let rows = block.rows;
        assert!(rows == 0 || out.len() > (rows - 1) * stride);
        let orig = &self.orig_id[..];
        self.walk_block(t, block, |r, leaf| {
            // SAFETY: `r < rows` (asserted above against `out`) and `leaf` is
            // an arena node id produced by the walk.
            unsafe {
                *out.get_unchecked_mut(r * stride) = *orig.get_unchecked(leaf as usize);
            }
        });
    }

    /// `out[r * stride] += weight * leaf_value(row r)` in tree `t` for the
    /// rows of `block`.
    pub(crate) fn accumulate(
        &self,
        t: usize,
        block: LaneBlock<'_>,
        weight: f32,
        out: &mut [f32],
        stride: usize,
    ) {
        let LaneBlock {
            lanes,
            tail,
            n_cols,
            rows,
        } = block;
        assert!(rows == 0 || out.len() > (rows - 1) * stride);
        if let Some(symmetric) = self.symmetric.get(t) {
            let groups = rows / LANES;
            assert!(
                lanes.len() >= groups * FEATURE_LANES * n_cols
                    && tail.len() >= (rows - groups * LANES) * n_cols,
                "row block holds fewer rows than requested"
            );
            self.trees[t].check_width(n_cols);
            symmetric.accumulate(lanes, groups, FEATURE_LANES * n_cols, weight, out, stride);
            for (i, row) in tail
                .chunks_exact(n_cols)
                .take(rows - groups * LANES)
                .enumerate()
            {
                out[(groups * LANES + i) * stride] +=
                    weight * self.leaf_value(self.leaf_id(t, row));
            }
            return;
        }
        let nodes = &self.nodes[..];
        self.walk_block(t, block, |r, leaf| {
            // SAFETY: `r < rows` (asserted above against `out`) and `leaf` is
            // an arena node id produced by the walk.
            unsafe {
                *out.get_unchecked_mut(r * stride) +=
                    weight * f32::from_bits(nodes.get_unchecked(leaf as usize).aux);
            }
        });
    }

    /// `out[r * stride + j] += weight * leaf_vector(row r)[j]` for `j < k` in
    /// vector-leaf tree `t`, over the rows of `block`.
    pub(crate) fn accumulate_vector(
        &self,
        t: usize,
        block: LaneBlock<'_>,
        k: usize,
        weight: f32,
        out: &mut [f32],
        stride: usize,
    ) {
        assert!(k <= stride && out.len() >= block.rows * stride);
        self.walk_block(t, block, |r, leaf| {
            let dst = &mut out[r * stride..r * stride + k];
            for (o, &w) in dst.iter_mut().zip(self.leaf_vector(leaf, k)) {
                *o += weight * w;
            }
        });
    }

    /// `out[j] += weight(t) * leaf_vector(row, tree t)[j]` for the vector-leaf
    /// trees `trees` of one dense `row` (`out` holds one value per output).
    pub(crate) fn accumulate_row_vector(
        &self,
        row: &[f32],
        trees: std::ops::Range<usize>,
        weight: impl Fn(usize) -> f32,
        out: &mut [f32],
    ) {
        let k = out.len();
        self.walk_row(row, trees, |t, leaf| {
            let w = weight(t);
            for (o, &v) in out.iter_mut().zip(self.leaf_vector(leaf, k)) {
                *o += w * v;
            }
        });
    }

    /// Walk one dense `row` through trees `trees` and call `sink(t, leaf)`
    /// with each tree's arena leaf id, in tree order. Trees are walked
    /// [`LANES`] at a time in lockstep, so a single instance still overlaps its
    /// dependent load chains (the batch kernel overlaps rows instead). The row
    /// is keyed once, `[feature][sign]`, so a node's slot maps to its key by a
    /// shift.
    #[inline(always)]
    fn walk_row(
        &self,
        row: &[f32],
        trees: std::ops::Range<usize>,
        mut sink: impl FnMut(usize, u32),
    ) {
        assert!(trees.end <= self.trees.len());
        let nodes = &self.nodes[..];
        let begin = trees.start;
        let groups = trees.len() / LANES;
        let full = begin + groups * LANES;
        let mut keys: Vec<u32> = Vec::new();
        for g in 0..groups {
            let first = begin + g * LANES;
            let group = &self.trees[first..first + LANES];
            let mut depth = 0u32;
            let mut ok = true;
            for meta in group {
                ok &= meta.lockstep_ok();
                depth = depth.max(meta.depth);
            }
            if !ok {
                for j in 0..LANES {
                    sink(first + j, self.leaf_id(first + j, row));
                }
                continue;
            }
            for meta in group {
                meta.check_width(row.len());
            }
            if keys.is_empty() {
                keys.reserve(2 * row.len());
                for &v in row {
                    keys.push(key(v));
                    keys.push(key(-v));
                }
            }
            let keys = &keys[..];
            let mut nid = [0usize; LANES];
            for (n, meta) in nid.iter_mut().zip(group) {
                *n = meta.root as usize;
            }
            for _ in 0..depth {
                macro_rules! lane {
                    ($($j:literal)*) => {$(
                        // SAFETY: as in `walk_block`; `check_width` ran for
                        // every tree in the group and `keys` holds two keys
                        // per feature, indexed by `slot / LANES`.
                        let node = unsafe { nodes.get_unchecked(nid[$j]) };
                        // SAFETY: see above.
                        let k = unsafe { *keys.get_unchecked(node.slot as usize / LANES) };
                        nid[$j] = Self::next_numeric(node, k);
                    )*};
                }
                lane!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);
            }
            for (j, &id) in nid.iter().enumerate() {
                sink(first + j, id as u32);
            }
        }
        for t in full..trees.end {
            sink(t, self.leaf_id(t, row));
        }
    }

    /// Original leaf ids of one dense `row` in trees `0..out.len()`, written
    /// to `out[t]`.
    pub(crate) fn original_leaf_ids_for_row(&self, row: &[f32], out: &mut [u32]) {
        self.walk_row(row, 0..out.len(), |t, leaf| out[t] = self.original_id(leaf));
    }

    /// `out[scalar_tree_output(t, parallel, k)] += weight(t) * leaf_value(row,
    /// tree t)` for the trees `trees` of one dense `row`, where
    /// `k = out.len()` and `parallel` is the number of consecutive trees per
    /// output (`num_parallel_tree`).
    pub(crate) fn accumulate_row(
        &self,
        row: &[f32],
        trees: std::ops::Range<usize>,
        parallel: usize,
        weight: impl Fn(usize) -> f32,
        out: &mut [f32],
    ) {
        let k = out.len();
        self.walk_row(row, trees, |t, leaf| {
            out[scalar_tree_output(t, parallel, k)] += weight(t) * self.leaf_value(leaf);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{ChildLeaf, SplitRule};

    fn tree() -> RegTree {
        // Root splits on f0 < 0.5 (missing left); left child splits on f1 < 2
        // (missing right) into leaves -1 / 1; right child is leaf 5.
        let mut t = RegTree::with_root(1.0);
        let (l, _r) = t.expand(
            0,
            SplitRule::numeric(0, 0.5, true),
            ChildLeaf::new(0.0, 1.0),
            ChildLeaf::new(5.0, 1.0),
        );
        t.expand(
            l,
            SplitRule::numeric(1, 2.0, false),
            ChildLeaf::new(-1.0, 1.0),
            ChildLeaf::new(1.0, 1.0),
        );
        t
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
        let block = LaneBlock {
            lanes: &[],
            tail: &flat,
            n_cols: 2,
            rows: rows.len(),
        };
        f.original_leaf_ids(0, block, &mut out, 1);
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
        let rows = LaneBlock {
            lanes: &lanes,
            tail,
            n_cols: 2,
            rows: n,
        };
        f.original_leaf_ids(0, rows, &mut ids[1..], 3);
        let mut acc = vec![0.5f32; n * 2];
        f.accumulate(0, rows, 2.0, &mut acc, 2);
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
                    SplitRule::numeric(i as u32 % 3, 0.1 * i as f32, i % 2 == 0),
                    ChildLeaf::new(0.0, 1.0),
                    ChildLeaf::new(1.0, 1.0),
                );
                t.expand(
                    l,
                    SplitRule::numeric(1, 0.5, i % 4 == 0),
                    ChildLeaf::new(-1.0, 1.0),
                    ChildLeaf::new(2.0, 1.0),
                );
                t
            })
            .collect();
        let mut cat = RegTree::with_root(1.0);
        cat.expand(
            0,
            SplitRule::categorical(2, &[1, 3], false),
            ChildLeaf::new(-1.0, 1.0),
            ChildLeaf::new(1.0, 1.0),
        );
        trees[LANES + 1] = cat;
        let f = CompactForest::from_trees(&trees);
        let row = [0.7f32, f32::NAN, 3.0];
        let mut out = vec![0u32; trees.len()];
        f.original_leaf_ids_for_row(&row, &mut out);
        // An offset range shifts the lockstep groups off the lane boundary;
        // pairs of consecutive trees share an output (`parallel = 2`).
        let mut acc = vec![0.25f32; 3];
        f.accumulate_row(&row, 3..trees.len(), 2, |t| 1.0 + t as f32, &mut acc);
        let mut want_acc = vec![0.25f32; 3];
        for (t, tree) in trees.iter().enumerate() {
            let want = tree.leaf_id_dense(&row, f32::NAN);
            assert_eq!(out[t] as usize, want, "tree {t}");
            if t >= 3 {
                want_acc[(t / 2) % 3] += (1.0 + t as f32) * tree.node(want).leaf_value;
            }
        }
        assert_eq!(acc, want_acc);
    }

    #[test]
    fn categorical_membership() {
        for default_left in [true, false] {
            let mut t = RegTree::with_root(1.0);
            t.expand(
                0,
                SplitRule::categorical(0, &[2, 5], default_left),
                ChildLeaf::new(-1.0, 1.0),
                ChildLeaf::new(1.0, 1.0),
            );
            // A preceding tree with its own categories shifts the pool.
            let mut first = RegTree::with_root(1.0);
            first.expand(
                0,
                SplitRule::categorical(0, &[9], true),
                ChildLeaf::new(0.0, 1.0),
                ChildLeaf::new(0.0, 1.0),
            );
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

    #[test]
    fn keys_order_like_floats_and_isolate_missing() {
        let values = [
            f32::NEG_INFINITY,
            -f32::MAX,
            -1.5,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            f32::from_bits(1),
            2.0,
            f32::MAX,
            f32::INFINITY,
        ];
        for (i, &a) in values.iter().enumerate() {
            for &b in &values[i..] {
                assert_eq!(key(a) > key(b), a > b, "{a} vs {b}");
                assert_eq!(key(a) == key(b), a == b, "{a} vs {b}");
            }
            assert!(key(a) > key(f32::NAN), "{a} must key above missing");
            assert!((key(f32::NAN) <= key(a)), "missing never compares greater");
            assert!(unkey(key(a)) == a, "{a} round trip");
        }
        assert_eq!(key(f32::NAN), 0);
        assert_eq!(key(-f32::NAN), 0);
        assert!(unkey(0).is_nan());
        assert_eq!(key(f32::INFINITY), LEAF_KEY);
    }

    #[test]
    fn boundary_values_match_reference_in_every_path() {
        // Thresholds at zero, negative, and large magnitudes, with both
        // missing directions, so mirrored nodes read negated keys and ties at
        // the threshold are exercised from both sides.
        let mut trees = Vec::new();
        for (cond, default_left) in [
            (0.0f32, true),
            (0.0, false),
            (-1.5, true),
            (-1.5, false),
            (f32::MAX, true),
            (-f32::MAX, false),
            (f32::MIN_POSITIVE, false),
        ] {
            let mut t = RegTree::with_root(1.0);
            let (l, r) = t.expand(
                0,
                SplitRule::numeric(0, cond, default_left),
                ChildLeaf::new(0.0, 1.0),
                ChildLeaf::new(0.0, 1.0),
            );
            t.expand(
                l,
                SplitRule::numeric(1, cond, !default_left),
                ChildLeaf::new(-1.0, 1.0),
                ChildLeaf::new(1.0, 1.0),
            );
            t.expand(
                r,
                SplitRule::numeric(1, -cond, default_left),
                ChildLeaf::new(2.0, 1.0),
                ChildLeaf::new(3.0, 1.0),
            );
            trees.push(t);
        }
        let f = CompactForest::from_trees(&trees);
        let probes = [
            f32::NEG_INFINITY,
            -f32::MAX,
            -1.5,
            -f32::from_bits(f32::MIN_POSITIVE.to_bits() + 1),
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            1.5,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
        ];
        // Every (f0, f1) pair plus one odd row, so the block has full lane
        // groups and a tail.
        let mut rows: Vec<f32> = vec![0.5, -0.5];
        for &a in &probes {
            for &b in &probes {
                rows.extend_from_slice(&[a, b]);
            }
        }
        let n = rows.len() / 2;
        assert!(!n.is_multiple_of(LANES), "layout must exercise a tail");
        let (lanes, tail) = split_lanes(&rows, 2);
        let block = LaneBlock {
            lanes: &lanes,
            tail,
            n_cols: 2,
            rows: n,
        };
        for (t, tree) in trees.iter().enumerate() {
            let mut ids = vec![0u32; n];
            f.original_leaf_ids(t, block, &mut ids, 1);
            for r in 0..n {
                let row = &rows[r * 2..r * 2 + 2];
                let want = tree.leaf_id_dense(row, f32::NAN);
                assert_eq!(ids[r] as usize, want, "tree {t} row {row:?} (block)");
                let leaf = f.leaf_id(t, row);
                assert_eq!(f.original_id(leaf) as usize, want, "tree {t} row {row:?}");
                let mut per_tree = vec![0u32; trees.len()];
                f.original_leaf_ids_for_row(row, &mut per_tree);
                assert_eq!(per_tree[t] as usize, want, "tree {t} row {row:?} (row)");
            }
        }
    }

    #[test]
    fn infinite_mirrored_threshold_is_not_mistaken_for_a_leaf() {
        // `v < -inf` with missing values right is stored as `-v > +inf`, whose
        // key equals a leaf's; the early-exit walkers must still descend.
        let mut t = RegTree::with_root(1.0);
        t.expand(
            0,
            SplitRule::numeric(0, f32::NEG_INFINITY, false),
            ChildLeaf::new(-1.0, 1.0),
            ChildLeaf::new(1.0, 1.0),
        );
        let f = CompactForest::from_trees(std::slice::from_ref(&t));
        for v in [f32::NEG_INFINITY, -1.0, 0.0, 1.0, f32::INFINITY, f32::NAN] {
            let want = t.leaf_id_dense(&[v], f32::NAN);
            assert_eq!(f.original_id(f.leaf_id(0, &[v])) as usize, want, "v={v}");
            let mut out = [0u32];
            f.original_leaf_ids_for_row(&[v], &mut out);
            assert_eq!(out[0] as usize, want, "v={v} (row)");
        }
    }
}
