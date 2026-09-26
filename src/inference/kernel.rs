//! The leaf kernel of a tree ensemble over the rows it is evaluated on.
//!
//! Tree `t` sends every point to one leaf `A_t(x)`. With `N` trees and
//! `n_ℓ` kernel rows in leaf `ℓ`, the kernel between points is
//!
//! ```text
//! k(x, x') = (1/N) Σ_t 1[A_t(x) = A_t(x')] / (n_{A_t(x)} + κ)
//! ```
//!
//! (Zhou & Hooker's expected structure matrix `E[S_n]`, with the row
//! subsample's indicator replaced by its expectation, so that a leaf of `m`
//! sampled rows and L2 penalty `λ` predicting `Σ z / (m + λ)` contributes
//! `1 / (n_ℓ + κ)` with `κ = λ / ξ`). As a matrix over the kernel rows it is
//! `K = Φ Φᵀ` for the sparse feature map with one indicator per leaf, so it
//! is symmetric positive semidefinite with every eigenvalue in `[0, 1]`.

use rayon::prelude::*;

use crate::error::{HessboostError, Result};
use crate::tree::RegTree;

/// Sentinel of [`LeafKernel::node_leaf`] for internal nodes.
const INTERNAL: u32 = u32::MAX;

/// The leaf kernel of an ensemble over its `n` kernel rows.
pub(super) struct LeafKernel {
    n: usize,
    n_trees: usize,
    /// Start of tree `t`'s nodes in [`Self::node_leaf`].
    node_offset: Vec<usize>,
    /// Global leaf index of every node of every tree ([`INTERNAL`] for
    /// internal nodes).
    node_leaf: Vec<u32>,
    /// Kernel weight of each global leaf: `1 / (N (n_ℓ + κ))`, `0` for a
    /// leaf no kernel row reaches.
    leaf_weight: Vec<f64>,
    /// CSR offsets of [`Self::leaf_rows`] per global leaf.
    leaf_start: Vec<usize>,
    /// The kernel rows of each leaf, ascending.
    leaf_rows: Vec<u32>,
    /// Global leaf of every kernel row in every tree, `[row][tree]`.
    row_leaves: Vec<u32>,
}

impl LeafKernel {
    /// The kernel of `trees` over rows whose leaf node ids are `node_ids`
    /// (`[row][tree]`, as [`BoostedModel::predict_leaf`] returns them), with
    /// leaf-count offset `kappa`.
    ///
    /// Every leaf must hold at least as many kernel rows as its cover (the
    /// rows it was grown on, all of weight one): otherwise the rows are not
    /// the ones the trees were fitted to, and the error says so.
    ///
    /// [`BoostedModel::predict_leaf`]: crate::model::BoostedModel::predict_leaf
    pub(super) fn new(trees: &[RegTree], node_ids: &[u32], kappa: f64) -> Result<Self> {
        let n_trees = trees.len();
        let n = node_ids.len().checked_div(n_trees).unwrap_or(0);
        let mut node_offset = Vec::with_capacity(n_trees + 1);
        let mut node_leaf = Vec::new();
        let mut n_leaves = 0u32;
        for tree in trees {
            node_offset.push(node_leaf.len());
            for node in tree.nodes() {
                if node.is_leaf() {
                    node_leaf.push(n_leaves);
                    n_leaves += 1;
                } else {
                    node_leaf.push(INTERNAL);
                }
            }
        }
        node_offset.push(node_leaf.len());
        let mut row_leaves = vec![0u32; n * n_trees];
        let mut counts = vec![0usize; n_leaves as usize];
        for (row, ids) in node_ids.chunks_exact(n_trees.max(1)).enumerate() {
            for (t, &id) in ids.iter().enumerate() {
                let leaf = node_leaf[node_offset[t] + id as usize];
                row_leaves[row * n_trees + t] = leaf;
                counts[leaf as usize] += 1;
            }
        }
        for (t, tree) in trees.iter().enumerate() {
            for (id, node) in tree.nodes().iter().enumerate() {
                if !node.is_leaf() {
                    continue;
                }
                let count = counts[node_leaf[node_offset[t] + id] as usize];
                if (count as f64) < f64::from(node.sum_hess) {
                    return Err(HessboostError::invalid_param(
                        "train",
                        format!(
                            "leaf {id} of tree {t} was grown on {} rows but only {count} of these \
                             rows reach it: pass the rows the model was trained on (or refit on)",
                            node.sum_hess
                        ),
                    ));
                }
            }
        }
        let scale = n_trees as f64;
        let leaf_weight = counts
            .iter()
            .map(|&c| {
                if c == 0 {
                    0.0
                } else {
                    1.0 / (scale * (c as f64 + kappa))
                }
            })
            .collect();
        let mut leaf_start = Vec::with_capacity(counts.len() + 1);
        let mut at = 0usize;
        for &c in &counts {
            leaf_start.push(at);
            at += c;
        }
        leaf_start.push(at);
        let mut fill = leaf_start.clone();
        let mut leaf_rows = vec![0u32; at];
        for (row, leaves) in row_leaves.chunks_exact(n_trees.max(1)).enumerate() {
            for &leaf in leaves {
                let slot = &mut fill[leaf as usize];
                leaf_rows[*slot] = row as u32;
                *slot += 1;
            }
        }
        Ok(LeafKernel {
            n,
            n_trees,
            node_offset,
            node_leaf,
            leaf_weight,
            leaf_start,
            leaf_rows,
            row_leaves,
        })
    }

    /// Number of kernel rows.
    pub(super) fn n(&self) -> usize {
        self.n
    }

    /// Number of trees.
    pub(super) fn n_trees(&self) -> usize {
        self.n_trees
    }

    /// The global leaf of node `id` of tree `t`.
    fn leaf_of(&self, t: usize, id: u32) -> u32 {
        self.node_leaf[self.node_offset[t] + id as usize]
    }

    /// Add the kernel vector `k(x)` over the kernel rows of the point whose
    /// leaf node ids are `node_ids` (one per tree) to `out` (length `n`).
    pub(super) fn add_query(&self, node_ids: &[u32], out: &mut [f64]) {
        for (t, &id) in node_ids.iter().enumerate() {
            self.add_leaf(self.leaf_of(t, id), out);
        }
    }

    /// Add the kernel vector of kernel row `row` to `out`.
    pub(super) fn add_row(&self, row: usize, out: &mut [f64]) {
        let leaves = &self.row_leaves[row * self.n_trees..(row + 1) * self.n_trees];
        for &leaf in leaves {
            self.add_leaf(leaf, out);
        }
    }

    fn add_leaf(&self, leaf: u32, out: &mut [f64]) {
        let leaf = leaf as usize;
        let w = self.leaf_weight[leaf];
        for &row in &self.leaf_rows[self.leaf_start[leaf]..self.leaf_start[leaf + 1]] {
            out[row as usize] += w;
        }
    }

    /// The dense `n × n` kernel matrix, row-major (rows in parallel, each
    /// summed over its trees in tree order).
    pub(super) fn dense(&self) -> Vec<f64> {
        let n = self.n;
        let mut k = vec![0.0; n * n];
        k.par_chunks_mut(n.max(1))
            .enumerate()
            .for_each(|(row, out)| self.add_row(row, out));
        k
    }
}
