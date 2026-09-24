//! Bit-pattern prediction for symmetric (oblivious) trees.
//!
//! In a symmetric tree every internal node of a level carries the same split,
//! so a row's path is fully described by one comparison per level: the leaf
//! is the bit pattern of the level outcomes (root level most significant),
//! looked up in a `2^depth` table. Collapsed subtrees (nodes that stayed a
//! leaf above the last level, see `grow_policy = symmetric`) fill every table
//! slot below them with their own leaf, so the table stays dense.
//!
//! The comparisons are exactly the [`CompactForest`](super::compact) node
//! compares (`key(v) > key'` against the same lane-major key block), and the
//! table maps each pattern to the arena leaf the generic walk reaches, so the
//! result is identical to the generic kernels by construction. Unlike the
//! generic walk, the levels do not form a dependent load chain: each level
//! reads one contiguous run of [`LANES`] keys and compares it against a
//! single threshold, which vectorizes.
//!
//! Detection is structural, so it also applies to imported trees that happen
//! to be symmetric. Trees shallower than two levels (nothing to gain) or
//! deeper than [`MAX_DEPTH`], and trees whose table would exceed four slots
//! per leaf (heavily collapsed trees), keep the generic walk. A slot holds the
//! arena leaf id (for `predict_leaf`) and the leaf value (for margins, saving
//! the dependent node load), so the tables take at most as much memory as the
//! arena, which spends two 16-byte nodes per leaf.

use super::compact::LANES;

/// Deepest tree given a table.
const MAX_DEPTH: usize = 16;

/// Table slots allowed per real leaf.
const MAX_SLOTS_PER_LEAF: usize = 4;

/// The part of a [`CompactForest`](super::compact) arena node detection needs.
pub(crate) enum ArenaNode {
    Leaf(f32),
    /// A numeric split: `first + u32::from(lane_key > key)` is the next node,
    /// where `lane_key` is the row's key at `slot` of a lane group.
    Numeric {
        slot: u32,
        key: u32,
        first: u32,
    },
    /// A split the table cannot express (categorical).
    Other,
}

/// Level splits and leaf table of one symmetric tree.
#[derive(Debug, Clone)]
pub(crate) struct SymmetricTree {
    /// `(slot, key)` per level, root first.
    levels: Vec<(u32, u32)>,
    /// Arena leaf id per bit pattern.
    leaves: Vec<u32>,
    /// Leaf value per bit pattern.
    values: Vec<f32>,
}

impl SymmetricTree {
    /// Recognize the tree rooted at arena node `root`, reading nodes through
    /// `node`. `None` when it is not symmetric or not worth a table.
    fn detect(root: u32, node: impl Fn(u32) -> ArenaNode) -> Option<Self> {
        let mut levels: Vec<(u32, u32)> = Vec::new();
        let mut n_leaves = 0usize;
        let mut frontier = vec![root];
        while !frontier.is_empty() {
            let mut next = Vec::with_capacity(2 * frontier.len());
            let mut level = None;
            for id in frontier {
                match node(id) {
                    ArenaNode::Leaf(_) => n_leaves += 1,
                    ArenaNode::Numeric { slot, key, first } => {
                        if *level.get_or_insert((slot, key)) != (slot, key) {
                            return None;
                        }
                        next.extend([first, first + 1]);
                    }
                    ArenaNode::Other => return None,
                }
            }
            if let Some(level) = level {
                if levels.len() == MAX_DEPTH {
                    return None;
                }
                levels.push(level);
            }
            frontier = next;
        }
        let depth = levels.len();
        if depth < 2 || (1usize << depth) > MAX_SLOTS_PER_LEAF * n_leaves {
            return None;
        }
        let leaves: Vec<u32> = (0..1u32 << depth)
            .map(|pattern| {
                let mut id = root;
                for d in 0..depth {
                    match node(id) {
                        ArenaNode::Numeric { first, .. } => {
                            id = first + ((pattern >> (depth - 1 - d)) & 1);
                        }
                        _ => break,
                    }
                }
                id
            })
            .collect();
        let values = leaves
            .iter()
            .map(|&id| match node(id) {
                ArenaNode::Leaf(value) => value,
                _ => unreachable!("patterns end at leaves"),
            })
            .collect();
        Some(SymmetricTree {
            levels,
            leaves,
            values,
        })
    }

    /// Call `sink(row, arena_leaf)` for the first `groups` [`LANES`]-row
    /// groups of `lanes` (`group_len` keys each, laid out as for
    /// [`CompactForest::accumulate`](super::compact)). The caller has checked
    /// that every split slot lies inside a group.
    #[inline]
    pub(crate) fn walk(
        &self,
        lanes: &[u32],
        groups: usize,
        group_len: usize,
        sink: &mut impl FnMut(usize, u32),
    ) {
        for (g, grp) in lanes.chunks_exact(group_len).take(groups).enumerate() {
            for (j, &p) in self.patterns(grp).iter().enumerate() {
                sink(g * LANES + j, self.leaves[p as usize]);
            }
        }
    }

    /// `out[r * stride] += weight * leaf_value(row r)` for the first `groups`
    /// lane groups, as [`Self::walk`] but reading the values directly.
    #[inline]
    pub(crate) fn accumulate(
        &self,
        lanes: &[u32],
        groups: usize,
        group_len: usize,
        weight: f32,
        out: &mut [f32],
        stride: usize,
    ) {
        for (g, grp) in lanes.chunks_exact(group_len).take(groups).enumerate() {
            for (j, &p) in self.patterns(grp).iter().enumerate() {
                out[(g * LANES + j) * stride] += weight * self.values[p as usize];
            }
        }
    }

    /// Bit pattern of each lane of one group.
    #[inline(always)]
    fn patterns(&self, grp: &[u32]) -> [u32; LANES] {
        let mut pattern = [0u32; LANES];
        for &(slot, key) in &self.levels {
            let keys: &[u32; LANES] = grp[slot as usize..]
                .first_chunk()
                .expect("split slot addresses a full lane run");
            for (p, &k) in pattern.iter_mut().zip(keys) {
                *p = (*p << 1) | u32::from(k > key);
            }
        }
        pattern
    }
}

/// Per-tree symmetric tables of a forest (`None` for other trees).
#[derive(Debug, Clone, Default)]
pub(crate) struct SymmetricTables {
    trees: Vec<Option<SymmetricTree>>,
}

impl SymmetricTables {
    /// Record the next tree of the forest, rooted at arena node `root`.
    pub(crate) fn push(&mut self, root: u32, node: impl Fn(u32) -> ArenaNode) {
        self.trees.push(SymmetricTree::detect(root, node));
    }

    /// The table of tree `t`, if it takes the bit-pattern path.
    #[inline]
    pub(crate) fn get(&self, t: usize) -> Option<&SymmetricTree> {
        self.trees[t].as_ref()
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{GrowPolicy, TrainingParams, TreeMethod};
    use crate::learner::train;
    use crate::test_support::labeled_dense;
    use crate::tree::RegTree;
    use crate::tree::compact::{CompactForest, LANES, split_lanes};

    /// Level 0: `f0 < 0.5`, missing left. Level 1: `f1 < 2`, missing right,
    /// on the left child only; the right child is a collapsed leaf. Level 2:
    /// `f2 < -1`, missing left, under both level-1 nodes.
    fn collapsed_tree() -> RegTree {
        let mut t = RegTree::with_root(1.0);
        let (l, _) = t.expand(0, 0, 0.5, true, 0.0, 1.0, 7.0, 1.0);
        let (ll, lr) = t.expand(l, 1, 2.0, false, 0.0, 1.0, 0.0, 1.0);
        t.expand(ll, 2, -1.0, true, -3.0, 1.0, -2.0, 1.0);
        t.expand(lr, 2, -1.0, true, 1.5, 1.0, 2.5, 1.0);
        t
    }

    /// A tree whose second level splits two nodes differently.
    fn asymmetric_tree() -> RegTree {
        let mut t = RegTree::with_root(1.0);
        let (l, r) = t.expand(0, 0, 0.5, true, 0.0, 1.0, 0.0, 1.0);
        t.expand(l, 1, 2.0, false, 1.0, 1.0, 2.0, 1.0);
        t.expand(r, 1, 3.0, false, 3.0, 1.0, 4.0, 1.0);
        t
    }

    /// A chain: every level has one internal node, but its `2^5` table
    /// would hold more than four slots per leaf.
    fn chain_tree(depth: usize) -> RegTree {
        let mut t = RegTree::with_root(1.0);
        let mut node = 0;
        for d in 0..depth {
            let (l, _) = t.expand(node, 0, d as f32, true, 0.0, 1.0, d as f32, 1.0);
            node = l;
        }
        t
    }

    /// Rows mixing missing values, exact thresholds, and their neighbors.
    fn rows(n: usize, n_cols: usize) -> Vec<f32> {
        let specials = [
            f32::NAN,
            0.5,
            0.5f32.next_down(),
            0.5f32.next_up(),
            2.0,
            2.0f32.next_down(),
            -1.0,
            (-1.0f32).next_up(),
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        (0..n * n_cols)
            .map(|i| {
                let h = (i as u64).wrapping_mul(crate::rng::GOLDEN) >> 40;
                if h.is_multiple_of(3) {
                    specials[(h / 3) as usize % specials.len()]
                } else {
                    (h % 1000) as f32 / 200.0 - 2.0
                }
            })
            .collect()
    }

    #[test]
    fn detection_accepts_level_uniform_trees_only() {
        let trees = [
            collapsed_tree(),
            asymmetric_tree(),
            chain_tree(1),
            chain_tree(4),
            chain_tree(5),
        ];
        let forest = CompactForest::from_trees(&trees);
        let symmetric: Vec<bool> = (0..trees.len()).map(|t| forest.is_symmetric(t)).collect();
        assert_eq!(symmetric, [true, false, false, true, false]);
    }

    #[test]
    fn bit_pattern_walk_matches_reference_routing() {
        let trees = [collapsed_tree(), chain_tree(4), asymmetric_tree()];
        let forest = CompactForest::from_trees(&trees);
        let n_cols = 3;
        let n = 5 * LANES + 7;
        let data = rows(n, n_cols);
        let (lanes, tail) = split_lanes(&data, n_cols);
        for (t, tree) in trees.iter().enumerate() {
            let mut margins = vec![0.25f32; n];
            forest.accumulate(t, &lanes, tail, n_cols, n, 1.0, &mut margins, 1);
            let mut leaves = vec![0u32; n];
            forest.original_leaf_ids(t, &lanes, tail, n_cols, n, &mut leaves, 1);
            for (r, row) in data.chunks_exact(n_cols).enumerate() {
                let leaf = tree.leaf_id_dense(row, f32::NAN);
                assert_eq!(leaves[r] as usize, leaf, "tree {t} row {r} {row:?}");
                let want = 0.25f32 + tree.node(leaf).leaf_value;
                assert_eq!(margins[r].to_bits(), want.to_bits());
            }
        }
    }

    #[test]
    fn trained_symmetric_model_predicts_bit_identically() {
        let (n, n_cols) = (3000, 5);
        // `DMatrix` rejects infinities; keep the other special values.
        let x: Vec<f32> = rows(n, n_cols)
            .into_iter()
            .map(|v| if v.is_infinite() { v.signum() * 3.0 } else { v })
            .collect();
        let y: Vec<f32> = x
            .chunks_exact(n_cols)
            .map(|r| r.iter().filter(|v| !v.is_nan()).map(|v| v.sin()).sum())
            .collect();
        let data = labeled_dense(&x, n, n_cols, &y);
        let params = TrainingParams::builder()
            .tree_method(TreeMethod::Hist)
            .grow_policy(GrowPolicy::Symmetric)
            .max_depth(5)
            .build()
            .unwrap();
        let model = train(&params, &data, 30).unwrap();
        let forest = CompactForest::from_trees(model.trees());
        assert!((0..model.num_trees()).all(|t| forest.is_symmetric(t)));

        let margins = model.predict_margin(&data).unwrap();
        let leaves = model.predict_leaf(&data).unwrap();
        for (r, row) in x.chunks_exact(n_cols).enumerate() {
            let mut want = model.base_score();
            for (t, tree) in model.trees().iter().enumerate() {
                let leaf = tree.leaf_id_dense(row, f32::NAN);
                assert_eq!(leaves[r * model.num_trees() + t] as usize, leaf);
                want += 1.0 * tree.node(leaf).leaf_value;
            }
            assert_eq!(margins[r].to_bits(), want.to_bits(), "row {r}");
        }
    }
}
