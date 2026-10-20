//! The row methods' layout: shallow numeric trees as complete binary trees.
//!
//! A single row has no other rows to overlap with, so its walk is bound by
//! how fast one core descends many trees, and by the cache lines those
//! trees occupy when it starts cold. [`RowForest`] cuts the ensemble into
//! chunks of [`CHUNK`] consecutive trees and lays out every chunk whose
//! trees are numeric (no categorical split), constant-leaf, at most
//! [`MAX_DEPTH`] levels deep, and split on features below `2^15` as
//! *complete* trees, unless padding would make the chunk more than
//! [`MAX_PADDING`] times its arena size:
//!
//! - each tree of the chunk is padded to the chunk's deepest tree (depth
//!   `D`) and stored in heap order, `2^D - 1` splits then `2^D` leaf slots,
//!   so node `a`'s children are `2a + 1` and `2a + 2`: no child links. A
//!   leaf above depth `D` fills every leaf slot of its subtree, so a walk
//!   always takes exactly `D` steps.
//! - a split is a `u32` threshold [`key`] and a `u16` *slot*,
//!   `2 * feature + mirrored`, in the arena's "second child when greater"
//!   form ([`NumericTest`]): a split whose missing values go right stores
//!   its children mirrored and compares the negated value, so one unsigned
//!   compare of the row's key for the slot routes every value, missing
//!   ones (key `0`) to the first child. A walk of at least one tree per
//!   [`FEATURES_PER_TREE`] features keys the row once, on the stack
//!   (`key(v)`, `key(-v)` per feature), when a step first reads a key; a
//!   shorter walk, or one of a model reading too many features for the
//!   stack, keys each value as a step reads it.
//! - the chunk's trees are walked in lockstep, one branch-free step
//!   ([`crate::simd::step_if_greater`]) per tree and level, so their
//!   dependent load chains overlap.
//! - a leaf slot is a `u32`: the leaf value's bits, or for vector leaves
//!   the offset of the leaf's weight vector, stored once however many slots
//!   it fills.
//!
//! A tree of depth `D` takes `10 * 2^D - 6` bytes (154 for depth 4) against
//! the arena's 16 per node (496 for a full depth-4 tree); a shallower tree
//! padded to its chunk's depth costs as much as a full one. The compare is
//! the arena's own (the same keys of the same values), so both walks reach
//! the same leaf. Every other chunk is walked through the model's arena
//! ([`CompactForest`]), with this walk's keys, or, when one of its trees
//! has linear leaves, through the [`RegTree`]s themselves. The walk
//! reports leaves in tree order either way.

use super::{CNode, CompactForest, NumericTest, RowKeys, key};
use crate::tree::RegTree;
use std::mem::size_of;
use std::ops::Range;

/// Consecutive trees a complete chunk holds and a row walks in lockstep.
const CHUNK: usize = 8;
/// Deepest tree a complete chunk holds: a deeper one would make its chunk
/// pad every tree to `2^depth` leaf slots.
const MAX_DEPTH: usize = 8;
/// Largest ratio of a complete chunk's split and leaf-slot bytes to its
/// trees' arena node bytes; a chunk padding past it stays on the arena.
const MAX_PADDING: usize = 2;
/// Largest split feature a complete split's `u16` slot encodes.
const MAX_FEATURE: u32 = (1 << 15) - 1;
/// Keys (two per feature) a walk keeps on the stack; a model that reads
/// more features keys each value a step reads, so the walk never allocates.
const INLINE_KEYS: usize = 1024;
/// Features per walked tree up to which a walk keys its row on the stack:
/// its trees read a key per level, and keying a value as a step reads it
/// costs several times making its key up front, so a walk of fewer trees
/// (one row through a few trees of a wide model) keys each value it reads.
const FEATURES_PER_TREE: usize = 8;
/// Key of a padding split (below a leaf above the chunk's depth): no row
/// key is greater, so its walk takes the first child, all of whose leaf
/// slots hold that leaf.
const PAD_KEY: u32 = u32::MAX;
/// Bytes a complete split takes: its key and its slot.
const SPLIT_BYTES: usize = size_of::<u32>() + size_of::<u16>();
/// Bytes a complete leaf slot takes.
const SLOT_BYTES: usize = size_of::<u32>();

/// How the row walk reaches one chunk's leaves.
#[derive(Debug, Clone, Copy)]
enum Chunk {
    /// Complete trees of depth `depth`: tree `j`'s splits start at
    /// `splits + j * (2^depth - 1)`, its leaf slots at
    /// `leaves + j * 2^depth`.
    Complete {
        depth: u32,
        splits: usize,
        leaves: usize,
    },
    /// Walked through the model's arena ([`CompactForest`]).
    Arena,
    /// Walked through the [`RegTree`]s (a tree with linear leaves).
    Tree,
}

/// A leaf the walk reached, by where its values are.
#[derive(Debug, Clone, Copy)]
enum Leaf {
    /// A complete leaf slot's word ([`RowForest::leaves`]).
    Complete(u32),
    /// Node of the arena.
    Arena(u32),
    /// The leaf the [`RegTree`] reaches.
    Tree,
}

/// An ensemble laid out for single-row walks (see the module docs), walked
/// through a [`RowWalk`].
#[derive(Debug, Clone)]
pub(crate) struct RowForest {
    chunks: Vec<Chunk>,
    n_trees: usize,
    /// Values per leaf: `1`, or a vector-leaf ensemble's outputs.
    width: usize,
    /// Complete splits' threshold keys, per tree in heap order.
    keys: Vec<u32>,
    /// Complete splits' slots (`2 * feature + mirrored`), as `keys`.
    slots: Vec<u16>,
    /// Complete trees' leaf slots: the leaf value's bits (`width == 1`) or
    /// the offset of its weight vector in `leaf_vectors`.
    leaves: Vec<u32>,
    /// Complete vector-leaf trees' weight vectors, one per leaf.
    leaf_vectors: Vec<f32>,
    /// Features the complete and arena trees split on: one past the largest
    /// (`0` when none splits), the keys a walk makes.
    features: usize,
    /// Whether a chunk walks the arena.
    arena: bool,
}

impl RowForest {
    /// The layout of `trees`, whose arena `arena` returns (called only when
    /// a chunk falls back to it: the features its trees split on, which the
    /// walk keys, are read from the arena's tree summaries).
    pub(crate) fn from_trees<'a>(trees: &[RegTree], arena: impl Fn() -> &'a CompactForest) -> Self {
        let width = trees.first().map_or(1, RegTree::size_leaf_vector);
        let mut forest = RowForest {
            chunks: Vec::with_capacity(trees.len().div_ceil(CHUNK)),
            n_trees: trees.len(),
            width,
            keys: Vec::new(),
            slots: Vec::new(),
            leaves: Vec::new(),
            leaf_vectors: Vec::new(),
            features: 0,
            arena: false,
        };
        for (c, chunk) in trees.chunks(CHUNK).enumerate() {
            // Validation refuses linear leaves in vector-leaf trees.
            let kind = if chunk.iter().any(|tree| tree.linear_leaves().is_some()) {
                Chunk::Tree
            } else if let Some(depth) = forest.scan(chunk) {
                forest.push_complete(chunk, depth)
            } else {
                let metas = &arena().trees[c * CHUNK..][..chunk.len()];
                for meta in metas.iter().filter(|meta| meta.depth > 0) {
                    forest.features = forest.features.max(meta.max_feature as usize + 1);
                }
                forest.arena = true;
                Chunk::Arena
            };
            forest.chunks.push(kind);
        }
        // The arrays grew chunk by chunk; the walk only reads them.
        forest.keys.shrink_to_fit();
        forest.slots.shrink_to_fit();
        forest.leaves.shrink_to_fit();
        forest.leaf_vectors.shrink_to_fit();
        forest
    }

    /// Whether a walk reads the model's arena ([`RowWalk::new`]).
    pub(crate) fn needs_arena(&self) -> bool {
        self.arena
    }

    /// The depth of the deepest tree of `chunk` when the chunk fits the
    /// complete layout (counting the features its trees split on into
    /// [`Self::features`]): numeric splits on features up to
    /// [`MAX_FEATURE`], at most [`MAX_DEPTH`] levels, the forest's leaf
    /// width, padding within [`MAX_PADDING`] of the arena, and leaf-vector
    /// offsets within `u32`. The scan stops at the first tree that does not
    /// fit.
    fn scan(&mut self, chunk: &[RegTree]) -> Option<u32> {
        let (mut deepest, mut nodes, mut values) = (0, 0, 0);
        for tree in chunk {
            if tree.size_leaf_vector() != self.width {
                return None;
            }
            deepest = deepest.max(depth_within(tree, 0, MAX_DEPTH)?);
            let mut leaves = 0;
            for node in tree.nodes() {
                if node.is_leaf() {
                    leaves += 1;
                } else if node.is_categorical || node.split_feature > MAX_FEATURE {
                    return None;
                } else {
                    self.features = self.features.max(node.split_feature as usize + 1);
                }
            }
            nodes += tree.num_nodes();
            values += leaves * self.width;
        }
        let slots = 1usize << deepest;
        let bytes = chunk.len() * ((slots - 1) * SPLIT_BYTES + slots * SLOT_BYTES);
        let offsets_fit =
            self.width == 1 || u32::try_from(self.leaf_vectors.len() + values).is_ok();
        // `deepest <= MAX_DEPTH`, so it fits.
        (bytes <= MAX_PADDING * nodes * size_of::<CNode>() && offsets_fit).then_some(deepest as u32)
    }

    /// Lay out `chunk` as complete trees of depth `depth`.
    fn push_complete(&mut self, chunk: &[RegTree], depth: u32) -> Chunk {
        let per = (1usize << depth) - 1;
        let kind = Chunk::Complete {
            depth,
            splits: self.keys.len(),
            leaves: self.leaves.len(),
        };
        for tree in chunk {
            let splits = self.keys.len();
            self.keys.resize(splits + per, PAD_KEY);
            self.slots.resize(splits + per, 0);
            let leaves = self.leaves.len();
            self.leaves.resize(leaves + per + 1, 0);
            Place {
                tree,
                keys: &mut self.keys[splits..],
                slots: &mut self.slots[splits..],
                leaves: &mut self.leaves[leaves..],
                leaf_vectors: &mut self.leaf_vectors,
                per,
            }
            .node(0, 0, depth);
        }
        kind
    }

    /// The leaf slot (`0..2^depth`) one row reaches in each tree of the
    /// complete chunk whose splits start at `splits` and which holds
    /// `count` trees (lanes past it repeat the last tree), with
    /// `key_of(slot)` the row's key for a split's slot.
    #[inline(always)]
    fn complete_leaves(
        &self,
        depth: u32,
        splits: usize,
        count: usize,
        key_of: impl Fn(u16) -> u32,
    ) -> [usize; CHUNK] {
        let per = (1usize << depth) - 1;
        let (keys, slots) = (&self.keys[..], &self.slots[..]);
        assert!(
            (1..=CHUNK).contains(&count) && splits + count * per <= keys.len(),
            "complete chunk outside the split arrays"
        );
        // Lane `j` walks absolute split indices of its tree, which starts at
        // `base[j]`: node `a`'s children `2a + 1 (+1)` are at
        // `2 * at + 1 - base[j] (+1)`.
        let mut base = [0usize; CHUNK];
        for (j, b) in base.iter_mut().enumerate() {
            *b = splits + j.min(count - 1) * per;
        }
        let shift = base.map(|b| 1usize.wrapping_sub(b));
        let mut at = base;
        for _ in 0..depth {
            // Fully unrolled so the lane states live in registers.
            macro_rules! lane {
                ($($j:literal)*) => {$(
                    // SAFETY: `at[j]` is a split of lane j's tree above the
                    // last level (the walk takes `depth` steps from its
                    // root), so `base[j] <= at[j] < base[j] + per`, inside
                    // both arrays (asserted above).
                    let (slot, threshold) = unsafe {
                        (*slots.get_unchecked(at[$j]), *keys.get_unchecked(at[$j]))
                    };
                    let first = shift[$j].wrapping_add(at[$j] << 1);
                    at[$j] = crate::simd::step_if_greater(first, key_of(slot), threshold);
                )*};
            }
            lane!(0 1 2 3 4 5 6 7);
        }
        for (a, b) in at.iter_mut().zip(base) {
            *a -= b + per;
        }
        at
    }
}

/// A [`RowForest`] with the trees it was built from and, when a chunk
/// needs it, their arena: what a row walk reads.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RowWalk<'a> {
    layout: &'a RowForest,
    trees: &'a [RegTree],
    arena: Option<&'a CompactForest>,
}

impl<'a> RowWalk<'a> {
    /// The walk of `layout`, built from `trees`, whose arena `arena` must be
    /// when `layout` [needs one](RowForest::needs_arena).
    #[inline]
    pub(crate) fn new(
        layout: &'a RowForest,
        trees: &'a [RegTree],
        arena: Option<&'a CompactForest>,
    ) -> Self {
        assert_eq!(trees.len(), layout.n_trees, "trees of another forest");
        assert!(
            !layout.arena || arena.is_some_and(|a| a.trees.len() == trees.len()),
            "row layout without its trees' arena"
        );
        RowWalk {
            layout,
            trees,
            arena,
        }
    }

    /// Call `sink(t, value)` for every tree `t` of `range` in order, where
    /// `value` is scalar tree `t`'s output for the dense `row` (`NaN` =
    /// missing), linear leaves included.
    #[inline]
    pub(crate) fn walk_scalar(
        &self,
        row: &[f32],
        range: Range<usize>,
        mut sink: impl FnMut(usize, f32),
    ) {
        self.walk(row, range, |t, leaf| {
            let value = match leaf {
                Leaf::Complete(bits) => f32::from_bits(bits),
                Leaf::Arena(id) => self.arena().leaf_value(id),
                Leaf::Tree => self.trees[t].predict_dense(row),
            };
            sink(t, value);
        });
    }

    /// Call `sink(t, values)` for every tree `t` of `range` in order, where
    /// `values` is vector-leaf tree `t`'s leaf vector for the dense `row`
    /// (`NaN` = missing).
    #[inline]
    pub(crate) fn walk_vector(
        &self,
        row: &[f32],
        range: Range<usize>,
        mut sink: impl FnMut(usize, &[f32]),
    ) {
        let RowForest {
            width: k,
            leaf_vectors,
            ..
        } = self.layout;
        self.walk(row, range, |t, leaf| {
            let values = match leaf {
                Leaf::Complete(offset) => &leaf_vectors[offset as usize..offset as usize + k],
                Leaf::Arena(id) => self.arena().leaf_vector(id, *k),
                Leaf::Tree => unreachable!("validation refuses linear leaves in vector-leaf trees"),
            };
            sink(t, values);
        });
    }

    /// The arena, which [`Self::new`] checked a layout with arena chunks
    /// has.
    #[inline]
    fn arena(&self) -> &'a CompactForest {
        self.arena.expect("RowWalk::new checked the arena")
    }

    /// Walk `row` through the trees `range` and call `sink(t, leaf)` in tree
    /// order.
    #[inline(always)]
    fn walk(&self, row: &[f32], range: Range<usize>, mut sink: impl FnMut(usize, Leaf)) {
        let layout = self.layout;
        let n_trees = layout.n_trees;
        assert!(range.end <= n_trees, "tree range past the forest");
        assert!(
            layout.features <= row.len(),
            "row has {} features but the forest splits on feature {}",
            row.len(),
            layout.features.saturating_sub(1)
        );
        let stack = range.len().saturating_mul(FEATURES_PER_TREE) >= layout.features;
        let mut keys = RowKeys::<INLINE_KEYS>::new(row, layout.features, stack);
        let mut t = range.start;
        while t < range.end {
            let c = t / CHUNK;
            let begin = c * CHUNK;
            let mut end = (begin + CHUNK).min(n_trees);
            match layout.chunks[c] {
                Chunk::Complete {
                    depth,
                    splits,
                    leaves,
                } => {
                    let count = end - begin;
                    let reached = match keys.get() {
                        // SAFETY: `slot < 2 * layout.features` (every complete
                        // slot, padding's `0` included: a padded chunk has a
                        // split), the keyed length (`layout.features <=
                        // row.len()`, asserted above).
                        Some(keys) => layout.complete_leaves(depth, splits, count, |slot| unsafe {
                            *keys.get_unchecked(usize::from(slot))
                        }),
                        // SAFETY: `slot / 2 < layout.features <= row.len()`
                        // (asserted above). The sign flip is `-v`.
                        None => layout.complete_leaves(depth, splits, count, |slot| unsafe {
                            let v = *row.get_unchecked(usize::from(slot >> 1));
                            key(f32::from_bits(v.to_bits() ^ (u32::from(slot & 1) << 31)))
                        }),
                    };
                    let stride = 1usize << depth;
                    for tt in t..end.min(range.end) {
                        let j = tt - begin;
                        sink(
                            tt,
                            Leaf::Complete(layout.leaves[leaves + j * stride + reached[j]]),
                        );
                    }
                }
                Chunk::Arena => {
                    // Consecutive arena chunks walk at once, in the arena's
                    // wider lockstep.
                    while end < range.end && matches!(layout.chunks[end / CHUNK], Chunk::Arena) {
                        end = (end + CHUNK).min(n_trees);
                    }
                    self.arena()
                        .walk_row_keyed(&mut keys, t..end.min(range.end), |tt, id| {
                            sink(tt, Leaf::Arena(id));
                        });
                }
                Chunk::Tree => {
                    for tt in t..end.min(range.end) {
                        sink(tt, Leaf::Tree);
                    }
                }
            }
            t = end.min(range.end);
        }
    }
}

/// The depth (splits on the longest path) of the subtree under node `nid`
/// of `tree`, when it is at most `limit`: [`RegTree::depth`] stopping past
/// the limit, without a heap stack.
fn depth_within(tree: &RegTree, nid: usize, limit: usize) -> Option<usize> {
    let node = tree.node(nid);
    if node.is_leaf() {
        return Some(0);
    }
    let limit = limit.checked_sub(1)?;
    let left = depth_within(tree, node.left as usize, limit)?;
    let right = depth_within(tree, node.right as usize, limit)?;
    Some(1 + left.max(right))
}

/// One tree's complete layout being written.
struct Place<'a> {
    tree: &'a RegTree,
    keys: &'a mut [u32],
    slots: &'a mut [u16],
    leaves: &'a mut [u32],
    leaf_vectors: &'a mut Vec<f32>,
    /// Splits per tree (`2^depth - 1`).
    per: usize,
}

impl Place<'_> {
    /// Place node `nid` of the tree at heap position `at`, with `below`
    /// levels under it.
    fn node(&mut self, nid: usize, at: usize, below: u32) {
        let node = self.tree.node(nid);
        if node.is_leaf() {
            let word = if self.tree.is_vector_leaf() {
                // `RowForest::complete_depth` checked that the offset fits.
                let offset = self.leaf_vectors.len() as u32;
                self.leaf_vectors
                    .extend_from_slice(self.tree.leaf_vector(nid));
                offset
            } else {
                node.leaf_value.to_bits()
            };
            // The leaf slots of the subtree under `at`.
            let first = ((at + 1) << below) - 1 - self.per;
            self.leaves[first..first + (1 << below)].fill(word);
            return;
        }
        let test = NumericTest::of(node);
        self.keys[at] = test.key;
        // `RowForest::complete_depth` checked that the feature fits 15 bits.
        self.slots[at] = ((node.split_feature as u16) << 1) | u16::from(test.mirrored);
        let (first, second) = test.children(node);
        self.node(first as usize, 2 * at + 1, below - 1);
        self.node(second as usize, 2 * at + 2, below - 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::tree::linear::LinearLeaves;
    use crate::tree::{ChildLeaf, SplitRule};

    /// A random numeric tree of at most `depth` levels over `n_features`
    /// features: irregular (a node below the root stops with probability
    /// `stop`, so leaves sit at every level), both missing directions,
    /// `width` values per leaf.
    fn random_tree(
        rng: &mut Rng,
        depth: usize,
        n_features: u32,
        width: usize,
        stop: f32,
    ) -> RegTree {
        let mut tree = if width > 1 {
            RegTree::with_vector_root(width, 1.0)
        } else {
            RegTree::with_root(1.0)
        };
        let mut frontier = vec![(0usize, 0usize)];
        while let Some((nid, level)) = frontier.pop() {
            if level == depth || (level > 0 && rng.f32() < stop) {
                if width > 1 {
                    let values: Vec<f32> = (0..width).map(|_| rng.f32() - 0.5).collect();
                    tree.set_leaf_vector(nid, &values);
                } else {
                    tree.set_leaf_value(nid, rng.f32() - 0.5);
                }
                continue;
            }
            let feature = (rng.next_u64() % u64::from(n_features)) as u32;
            // Thresholds on a coarse grid, so rows hit them exactly.
            let cond = (rng.next_u64() % 9) as f32 * 0.25 - 1.0;
            let (l, r) = tree.expand(
                nid,
                SplitRule::numeric(feature, cond, rng.f32() < 0.5),
                ChildLeaf::new(0.0, 1.0),
                ChildLeaf::new(0.0, 1.0),
            );
            frontier.push((l, level + 1));
            frontier.push((r, level + 1));
        }
        tree
    }

    /// Rows over `n_features` features: values on and between the
    /// thresholds' grid, signed zeros, and missing values.
    fn rows(rng: &mut Rng, n: usize, n_features: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| {
                (0..n_features)
                    .map(|_| match rng.next_u64() % 6 {
                        0 => f32::NAN,
                        1 => -0.0,
                        2 => (rng.next_u64() % 9) as f32 * 0.25 - 1.0,
                        _ => rng.f32() * 2.5 - 1.25,
                    })
                    .collect()
            })
            .collect()
    }

    /// Ranges of a forest of `n` trees starting and ending at chunk
    /// boundaries, off them, a single tree (too few for the stack keys of a
    /// model reading more than [`FEATURES_PER_TREE`] features), and empty.
    fn ranges(n: usize) -> Vec<Range<usize>> {
        [
            0..n,
            3..n,
            0..n - 5,
            CHUNK + 1..2 * CHUNK + 3,
            2..3,
            7..7,
            n..n,
        ]
        .into_iter()
        .map(|r| r.start.min(n)..r.end.min(n))
        .collect()
    }

    /// The forest of `trees` and their arena (walk them with [`walk_of`]).
    fn layouts(trees: &[RegTree]) -> (RowForest, CompactForest) {
        let arena = CompactForest::from_trees(trees);
        (RowForest::from_trees(trees, || &arena), arena)
    }

    fn walk_of<'a>(
        (forest, arena): &'a (RowForest, CompactForest),
        trees: &'a [RegTree],
    ) -> RowWalk<'a> {
        RowWalk::new(forest, trees, forest.needs_arena().then_some(arena))
    }

    fn check_scalar(trees: &[RegTree], rows: &[Vec<f32>]) -> RowForest {
        let layouts = layouts(trees);
        let walk = walk_of(&layouts, trees);
        for row in rows {
            for range in ranges(trees.len()) {
                let mut seen = Vec::new();
                walk.walk_scalar(row, range.clone(), |t, v| {
                    let want = trees[t].predict_dense(row);
                    assert_eq!(v.to_bits(), want.to_bits(), "tree {t} row {row:?}");
                    seen.push(t);
                });
                assert_eq!(seen, range.collect::<Vec<_>>(), "tree order");
            }
        }
        layouts.0
    }

    fn check_vector(trees: &[RegTree], rows: &[Vec<f32>]) -> RowForest {
        let layouts = layouts(trees);
        let walk = walk_of(&layouts, trees);
        for row in rows {
            for range in ranges(trees.len()) {
                let mut next = range.start;
                walk.walk_vector(row, range.clone(), |t, values| {
                    let want = trees[t].leaf_vector(trees[t].leaf_id_dense(row, f32::NAN));
                    assert_eq!(values, want, "tree {t} row {row:?}");
                    assert_eq!(t, next, "tree order");
                    next += 1;
                });
                assert_eq!(next, range.end);
            }
        }
        layouts.0
    }

    fn kinds(forest: &RowForest) -> Vec<&'static str> {
        forest
            .chunks
            .iter()
            .map(|c| match c {
                Chunk::Complete { .. } => "complete",
                Chunk::Arena => "arena",
                Chunk::Tree => "tree",
            })
            .collect()
    }

    /// Irregular trees of every depth up to the limit, a depth per chunk so
    /// chunks pad to different depths, reach the leaves the trees do, for
    /// boundary and missing values.
    #[test]
    fn complete_chunks_reach_every_trees_leaf() {
        let mut rng = Rng::new(3);
        let trees: Vec<RegTree> = (0..(MAX_DEPTH + 1) * CHUNK + 3)
            .map(|i| random_tree(&mut rng, (i / CHUNK) % (MAX_DEPTH + 1), 12, 1, 0.25))
            .collect();
        let rows = rows(&mut rng, 200, 12);
        let forest = check_scalar(&trees, &rows);
        assert!(kinds(&forest).iter().all(|&k| k == "complete"));
    }

    /// Single-leaf trees make a depth-0 chunk: no splits, one leaf each.
    #[test]
    fn single_leaf_trees_are_complete_chunks() {
        let trees: Vec<RegTree> = (0..CHUNK + 5)
            .map(|i| {
                let mut tree = RegTree::with_root(1.0);
                tree.set_leaf_value(0, i as f32 * 0.5 - 1.0);
                tree
            })
            .collect();
        let forest = check_scalar(&trees, &[vec![], vec![]]);
        assert!(kinds(&forest).iter().all(|&k| k == "complete"));
    }

    /// A path of `depth` splits, each with a leaf on one side: `depth + 1`
    /// leaves that a complete layout pads to `2^depth` slots.
    fn chain(depth: usize, width: usize) -> RegTree {
        let mut tree = if width > 1 {
            RegTree::with_vector_root(width, 1.0)
        } else {
            RegTree::with_root(1.0)
        };
        let mut nid = 0;
        for level in 0..depth {
            let cond = level as f32 * 0.25 - 1.0;
            let (l, r) = tree.expand(
                nid,
                SplitRule::numeric(level as u32 % 3, cond, level % 2 == 0),
                ChildLeaf::new(level as f32, 1.0),
                ChildLeaf::new(0.0, 1.0),
            );
            if width > 1 {
                tree.set_leaf_vector(l, &vec![level as f32; width]);
            }
            nid = r;
        }
        tree
    }

    /// A chunk whose padding would take more than [`MAX_PADDING`] times its
    /// arena bytes stays on the arena, scalar or vector (two chunks of them,
    /// so the arena's lockstep walk reads this walk's keys); full trees of
    /// the same depth stay complete.
    #[test]
    fn padding_past_the_bound_stays_on_the_arena() {
        for width in [1, 3] {
            let mut rng = Rng::new(11);
            let mut trees: Vec<RegTree> = (0..2 * CHUNK).map(|_| chain(MAX_DEPTH, width)).collect();
            trees.extend((0..CHUNK).map(|_| random_tree(&mut rng, MAX_DEPTH, 3, width, 0.0)));
            let rows = rows(&mut rng, 100, 3);
            let forest = if width == 1 {
                check_scalar(&trees, &rows)
            } else {
                check_vector(&trees, &rows)
            };
            assert_eq!(
                kinds(&forest),
                ["arena", "arena", "complete"],
                "width {width}"
            );
        }
    }

    /// A chunk with a tree past the depth limit, a categorical split, a
    /// feature past 15 bits, or linear leaves falls back; the others stay
    /// complete, and consecutive arena chunks walk as one run.
    #[test]
    fn chunks_the_layout_cannot_hold_fall_back() {
        let mut rng = Rng::new(5);
        let n_features = 1 << 15 | 3;
        let mut trees: Vec<RegTree> = (0..7 * CHUNK + 2)
            .map(|_| random_tree(&mut rng, 4, 20, 1, 0.25))
            .collect();
        trees[CHUNK + 2] = random_tree(&mut rng, MAX_DEPTH + 3, 20, 1, 0.1);
        while trees[CHUNK + 2].depth() <= MAX_DEPTH {
            trees[CHUNK + 2] = random_tree(&mut rng, MAX_DEPTH + 3, 20, 1, 0.1);
        }
        let mut categorical = RegTree::with_root(1.0);
        let (l, _) = categorical.expand(
            0,
            SplitRule::numeric(1, 0.5, true),
            ChildLeaf::new(0.0, 1.0),
            ChildLeaf::new(1.0, 1.0),
        );
        categorical.expand(
            l,
            SplitRule::categorical(2, &[1, 3], false),
            ChildLeaf::new(-1.0, 1.0),
            ChildLeaf::new(2.0, 1.0),
        );
        trees[2 * CHUNK + 7] = categorical;
        let mut wide = RegTree::with_root(1.0);
        wide.expand(
            0,
            SplitRule::numeric(MAX_FEATURE + 1, 0.0, false),
            ChildLeaf::new(-1.0, 1.0),
            ChildLeaf::new(1.0, 1.0),
        );
        trees[4 * CHUNK] = wide;
        let mut linear = RegTree::with_root(1.0);
        linear.expand(
            0,
            SplitRule::numeric(0, 0.25, true),
            ChildLeaf::new(0.5, 1.0),
            ChildLeaf::new(-0.5, 1.0),
        );
        // Leaf 1: 0.5 + 2 * x3, leaf 2: its constant.
        linear.set_linear_leaves(LinearLeaves::from_parts(
            vec![0, 0, 1, 1],
            vec![0.0, 0.5, 0.0],
            vec![3],
            vec![2.0],
        ));
        trees[6 * CHUNK + 1] = linear;
        let mut rows = rows(&mut rng, 100, n_features);
        for row in &mut rows {
            row[MAX_FEATURE as usize + 1] = rng.f32() - 0.5;
        }
        let forest = check_scalar(&trees, &rows);
        assert_eq!(
            kinds(&forest),
            [
                "complete", "arena", "arena", "complete", "arena", "complete", "tree", "complete"
            ]
        );
    }

    /// Vector leaves (two and three outputs) reach the trees' leaf vectors.
    #[test]
    fn vector_leaves_reach_every_trees_leaf_vector() {
        for width in [2, 3] {
            let mut rng = Rng::new(width as u64);
            let trees: Vec<RegTree> = (0..=6 * CHUNK)
                .map(|i| random_tree(&mut rng, 1 + (i / CHUNK) % 6, 9, width, 0.25))
                .collect();
            let forest = check_vector(&trees, &rows(&mut rng, 100, 9));
            assert!(kinds(&forest).iter().all(|&k| k == "complete"));
        }
    }

    /// A model reading more features than the stack keys hold keys each
    /// value as the walk reads it, with the same leaves.
    #[test]
    fn wide_rows_key_values_as_they_are_read() {
        let mut rng = Rng::new(9);
        let n_features = INLINE_KEYS / 2 + 40;
        let trees: Vec<RegTree> = (0..2 * CHUNK + 3)
            .map(|_| random_tree(&mut rng, 5, n_features as u32, 1, 0.25))
            .collect();
        let forest = check_scalar(&trees, &rows(&mut rng, 50, n_features));
        assert!(
            2 * forest.features > INLINE_KEYS,
            "the test must take the wide path"
        );
    }

    /// Thresholds at the edges of the float range, with both missing
    /// directions: the complete walk routes what the tree does.
    #[test]
    fn boundary_thresholds_route_like_the_tree() {
        let mut trees = Vec::new();
        for cond in [0.0f32, -1.5, f32::MAX, -f32::MAX, f32::MIN_POSITIVE] {
            for default_left in [true, false] {
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
        }
        let probes = [
            -f32::MAX,
            -1.5,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            1.5,
            f32::MAX,
            f32::NAN,
        ];
        let rows: Vec<Vec<f32>> = probes
            .iter()
            .flat_map(|&a| probes.iter().map(move |&b| vec![a, b]))
            .collect();
        check_scalar(&trees, &rows);
    }
}
