//! Batch prediction: the CPU compact forest's walk and its tree-order `f32`
//! additions. Objective transforms stay on the CPU.

use crate::{U32x2, ld, st};
use cuda_device::{kernel, thread};

/// A 16-byte node of the compact layout (`tree::compact`): split slot
/// (feature `slot / 32`, flag 16: compare the negated value), comparison
/// key or first category, left child (a leaf is its own left child), and
/// aux (categorical flags and category end, or the leaf's value bits or
/// vector offset).
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct Node16 {
    pub slot: u32,
    pub key: u32,
    pub left: u32,
    pub aux: u32,
}

/// A tree of the 16-byte layout: root node, output, and whether its leaves
/// are vectors.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct Tree {
    pub root: u32,
    pub output: u32,
    pub vector: u32,
    pub pad: u32,
}

/// The order-preserving key a numeric split compares: both zeros as one,
/// subnormals kept, NaN (missing) below every value.
#[inline(always)]
fn value_key(value: f32) -> u32 {
    if value.is_nan() {
        return 0;
    }
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    bits ^ if bits & 0x8000_0000 != 0 {
        u32::MAX
    } else {
        0x8000_0000
    }
}

/// [`predict16`]'s batch: rows `[0, n_rows)` of `n_cols` values, `outputs`
/// margins each, trees `[tree_begin, tree_end)`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Batch16 {
    pub n_rows: u32,
    pub n_cols: u32,
    pub outputs: u32,
    pub tree_begin: u32,
    pub tree_end: u32,
}

/// Every row's margins (`outputs` per row) plus trees `[tree_begin,
/// tree_end)` of the 16-byte layout, in tree order.
///
/// # Safety
///
/// A one-dimensional grid and blocks of any size (a grid-stride loop, one
/// thread per row). `rows` holds `n_rows * n_cols` values and `margins`
/// `n_rows * outputs`; `trees` holds trees `[tree_begin, tree_end)`; every
/// node reachable from a tree's root (its children `left` and `left + 1`)
/// is in `nodes`, every split's feature `slot / 32` is below `n_cols`, a
/// categorical split's categories `[key, aux >> 2)` are in `categories`, a
/// vector tree's leaves' `outputs` values from `aux` are in `vectors`, and
/// a scalar tree's `output` is below `outputs`.
#[kernel]
pub unsafe fn predict16(
    nodes: *const Node16,
    categories: *const u32,
    vectors: *const f32,
    trees: *const Tree,
    rows: *const f32,
    margins: *mut f32,
    batch: Batch16,
) {
    let Batch16 {
        n_rows,
        n_cols,
        outputs,
        tree_begin,
        tree_end,
    } = batch;
    let mut row = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
    while row < n_rows {
        // SAFETY: `row < n_rows` (loop condition): the row's `n_cols`
        // values are within `rows` (the contract).
        let input = unsafe { rows.add(row as usize * n_cols as usize) };
        // SAFETY: as for `input`, the row's `outputs` margins within
        // `margins`.
        let out = unsafe { margins.add(row as usize * outputs as usize) };
        let mut t = tree_begin;
        while t < tree_end {
            // SAFETY: `t < tree_end` (loop condition): tree `t` is in
            // `trees` (the contract).
            let tree = unsafe { ld(trees, u64::from(t)) };
            let mut id = tree.root;
            // SAFETY: the tree's root is in `nodes` (the contract).
            let mut node = unsafe { ld(nodes, u64::from(id)) };
            while node.left != id {
                // SAFETY: a split's feature `slot / 32` is below `n_cols`:
                // one of the row's values (the contract).
                let value = unsafe { ld(input, u64::from(node.slot / 32)) };
                id = if node.aux & 1 != 0 {
                    let mut left = node.aux & 2 != 0;
                    if !value.is_nan() {
                        // `as u32`: truncation, saturating negatives and
                        // overflow, the CPU's category code.
                        let code = value as u32;
                        left = false;
                        let mut c = node.key;
                        while c < node.aux >> 2 {
                            // SAFETY: `c` is in the split's category range
                            // `[key, aux >> 2)` (loop condition), within
                            // `categories` (the contract).
                            if unsafe { ld(categories, u64::from(c)) } == code {
                                left = true;
                                break;
                            }
                            c += 1;
                        }
                    }
                    node.left + u32::from(!left)
                } else {
                    let value = if node.slot & 16 != 0 { -value } else { value };
                    node.left + u32::from(value_key(value) > node.key)
                };
                // SAFETY: a split's children are reachable nodes, in `nodes`
                // (the contract).
                node = unsafe { ld(nodes, u64::from(id)) };
            }
            if tree.vector != 0 {
                let mut k = 0;
                while k < outputs {
                    // SAFETY: `k < outputs` (loop condition): the leaf's
                    // value `aux + k` is in `vectors` (the contract).
                    let leaf = unsafe { ld(vectors, u64::from(node.aux) + u64::from(k)) };
                    // SAFETY: `k < outputs`: one of the row's margins, which
                    // only this thread accesses (one thread per row).
                    unsafe { st(out, u64::from(k), ld(out, u64::from(k)) + leaf) };
                    k += 1;
                }
            } else {
                let o = u64::from(tree.output);
                // SAFETY: a scalar tree's `output` is below `outputs` (the
                // contract): one of the row's margins, which only this thread
                // accesses.
                unsafe { st(out, o, ld(out, o) + f32::from_bits(node.aux)) };
            }
            t += 1;
        }
        row += thread::blockDim_x() * thread::gridDim_x();
    }
}

/// [`predict8`]'s batch: rows `[0, n_rows)` of `n_cols` values, trees
/// `[tree_begin, tree_end)`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Batch8 {
    pub n_rows: u32,
    pub n_cols: u32,
    pub tree_begin: u32,
    pub tree_end: u32,
}

/// Every row's single margin plus trees `[tree_begin, tree_end)` of the
/// 8-byte layout (numeric splits only): `nodes[i]` is (threshold or leaf
/// bits, flags: bit 31 leaf, bit 30 negate, bits 15 to 29 feature, bits 0 to
/// 14 the left child's offset from the tree's root; the right child follows
/// it).
///
/// # Safety
///
/// A one-dimensional grid and blocks of any size (a grid-stride loop, one
/// thread per row). `rows` holds `n_rows * n_cols` values and `margins`
/// `n_rows`; `roots` holds trees `[tree_begin, tree_end)`; every node
/// reachable from a tree's root (a split's children at `root + offset` and
/// the next) is in `nodes`, and every split's feature is below `n_cols`.
#[kernel]
pub unsafe fn predict8(
    nodes: *const U32x2,
    roots: *const u32,
    rows: *const f32,
    margins: *mut f32,
    batch: Batch8,
) {
    let Batch8 {
        n_rows,
        n_cols,
        tree_begin,
        tree_end,
    } = batch;
    let mut row = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
    while row < n_rows {
        // SAFETY: `row < n_rows` (loop condition): the row's `n_cols`
        // values are within `rows` (the contract).
        let input = unsafe { rows.add(row as usize * n_cols as usize) };
        // SAFETY: `row < n_rows`: the row's margin, which only this thread
        // accesses (one thread per row).
        let mut out = unsafe { ld(margins, u64::from(row)) };
        let mut t = tree_begin;
        while t < tree_end {
            // SAFETY: `t < tree_end` (loop condition): tree `t`'s root is in
            // `roots` (the contract).
            let root = unsafe { ld(roots, u64::from(t)) };
            // SAFETY: the tree's root node is in `nodes` (the contract).
            let mut node = unsafe { ld(nodes, u64::from(root)) };
            while node.y & 0x8000_0000 == 0 {
                // SAFETY: a split's feature is below `n_cols`: one of the
                // row's values (the contract).
                let value = unsafe { ld(input, u64::from((node.y >> 15) & 0x7fff)) };
                let value = if node.y & 0x4000_0000 != 0 {
                    -value
                } else {
                    value
                };
                let next = root + (node.y & 0x7fff) + u32::from(value > f32::from_bits(node.x));
                // SAFETY: a split's children are reachable nodes, in `nodes`
                // (the contract).
                node = unsafe { ld(nodes, u64::from(next)) };
            }
            out += f32::from_bits(node.x);
            t += 1;
        }
        // SAFETY: as for the load of `out`.
        unsafe { st(margins, u64::from(row), out) };
        row += thread::blockDim_x() * thread::gridDim_x();
    }
}
