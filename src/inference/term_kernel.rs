//! The additive term kernel of one Boulevard EBM stage over its training
//! rows (Fang, Tan, Pipping & Hooker, AISTATS 2026, Section 4's
//! feature-specific kernels).
//!
//! Term `t`'s `N_t` trees each send every point to a leaf; with `n_ℓ`
//! training rows in leaf `ℓ` and `κ = lambda / subsample` (as in
//! [`super::kernel`]),
//!
//! ```text
//! K_t(x, x') = (1/N_t) Σ_{trees of t} 1[same leaf] / (n_ℓ + κ),     K = Σ_t K_t
//! ```
//!
//! and the ridge systems use the doubly centered `K̄ = J K J`
//! (`J = I − 11ᵀ/n`), since every tree is centered on the training rows.
//! Every term's trees are constant on the cells of its
//! [`TermGrid`](crate::ebm::grid::TermGrid), so a kernel row, a kernel
//! vector, and a product `K v` are computed on the grid with difference
//! arrays: `O(Σ_t (N_t + cells_t) + T n)` each, never touching the
//! `N_t n` leaf memberships.

use rayon::prelude::*;

use super::kernel::Kernel;
use crate::data::DMatrix;
use crate::ebm::grid::TermGrid;
use crate::error::{HessboostError, Result};
use crate::tree::RegTree;

/// One term's part of the kernel.
pub(super) struct TermPart<'a> {
    pub(super) grid: TermGrid,
    trees: Vec<&'a RegTree>,
    /// The grid cell of every training row.
    cell_of_row: Vec<u32>,
    /// `1 / (N_t (n_ℓ + κ))` of every leaf (`0` for a leaf no row reaches),
    /// in [`TermGrid::leaves`] order.
    weight: Vec<f64>,
    /// `K_t 1 / n` over the training rows.
    col_mean: Vec<f64>,
}

impl<'a> TermPart<'a> {
    /// The part of `trees` (splitting on `features`) over the rows of
    /// `train`. Every leaf must hold at least as many of the rows as it was
    /// grown on (its cover), or `train` is not the training data.
    pub(super) fn new(
        trees: Vec<&'a RegTree>,
        features: &[u32],
        train: &DMatrix,
        kappa: f64,
    ) -> Result<Self> {
        let grid = TermGrid::new(&trees, features);
        let n = train.n_rows();
        let cell_of_row: Vec<u32> = (0..n)
            .into_par_iter()
            .with_min_len(1024)
            .map(|row| grid.cell_of_row(train, row) as u32)
            .collect();
        let mut counts = vec![0.0; grid.len()];
        for &c in &cell_of_row {
            counts[c as usize] += 1.0;
        }
        let prefix = grid.prefix(&counts);
        let n_trees = trees.len() as f64;
        let mut weight = Vec::with_capacity(grid.leaves.len());
        for (t, tree) in trees.iter().enumerate() {
            for leaf in &grid.leaves[grid.leaf_start[t]..grid.leaf_start[t + 1]] {
                let rows: f64 = leaf.boxes.iter().map(|&b| grid.box_sum(&prefix, b)).sum();
                let cover = f64::from(tree.node(leaf.node as usize).sum_hess);
                if rows < cover {
                    return Err(HessboostError::invalid_data(
                        "train",
                        format!(
                            "a leaf of term {features:?} was grown on {cover} rows but only {rows} \
                             of these rows reach it: pass the rows the model was trained on"
                        ),
                    ));
                }
                weight.push(if rows > 0.0 {
                    1.0 / (n_trees * (rows + kappa))
                } else {
                    0.0
                });
            }
        }
        let mut part = TermPart {
            grid,
            trees,
            cell_of_row,
            weight,
            col_mean: Vec::new(),
        };
        // `K_t 1`: the per-cell sums of the ones vector are the row counts.
        let mut col_sum = vec![0.0; n];
        part.add_cell_sums_product(&counts, &mut col_sum);
        part.col_mean = col_sum.into_iter().map(|v| v / n as f64).collect();
        Ok(part)
    }

    /// `K_t` between grid cell `cell` and every grid cell, into
    /// `scratch.row`.
    fn cell_row(&self, cell: usize, scratch: &mut CellScratch) {
        let grid = &self.grid;
        grid.reset_diff(&mut scratch.diff);
        for (t, tree) in self.trees.iter().enumerate() {
            let leaf = grid.leaf_of_cell(tree, t, cell);
            let w = self.weight[leaf];
            if w != 0.0 {
                for &b in &grid.leaves[leaf].boxes {
                    grid.box_add(&mut scratch.diff, b, w);
                }
            }
        }
        grid.integrate_into(&scratch.diff, &mut scratch.row);
    }

    /// Add `K_t(x, ·)` over the training rows to `out`, for a point in
    /// grid cell `cell`.
    fn add_cell_vector(&self, cell: usize, scratch: &mut CellScratch, out: &mut [f64]) {
        self.cell_row(cell, scratch);
        for (o, &c) in out.iter_mut().zip(&self.cell_of_row) {
            *o += scratch.row[c as usize];
        }
    }

    /// `out += K_t v` over the training rows.
    fn add_product(&self, v: &[f64], out: &mut [f64]) {
        let mut sums = vec![0.0; self.grid.len()];
        for (&c, &vi) in self.cell_of_row.iter().zip(v) {
            sums[c as usize] += vi;
        }
        self.add_cell_sums_product(&sums, out);
    }

    /// `out += K_t v` over the training rows, for the `v` whose sums over
    /// the training rows of each grid cell are `sums`.
    fn add_cell_sums_product(&self, sums: &[f64], out: &mut [f64]) {
        let grid = &self.grid;
        let prefix = grid.prefix(sums);
        let mut diff = grid.diff();
        for (leaf, &w) in grid.leaves.iter().zip(&self.weight) {
            if w == 0.0 {
                continue;
            }
            let total: f64 = leaf.boxes.iter().map(|&b| grid.box_sum(&prefix, b)).sum();
            if total != 0.0 {
                for &b in &leaf.boxes {
                    grid.box_add(&mut diff, b, w * total);
                }
            }
        }
        let cells = grid.integrate(&diff);
        for (o, &c) in out.iter_mut().zip(&self.cell_of_row) {
            *o += cells[c as usize];
        }
    }

    /// The grid cell of row `row` of `data`.
    pub(super) fn cell_of(&self, data: &DMatrix, row: usize) -> usize {
        self.grid.cell_of_row(data, row)
    }
}

/// A kernel row on one term's grid and its difference array.
#[derive(Default)]
struct CellScratch {
    diff: Vec<f64>,
    row: Vec<f64>,
}

/// Working buffers of [`TermKernel::add_query`] and [`Kernel::add_row`],
/// reused across calls (every call resets what it reads).
#[derive(Default)]
pub(super) struct TermScratch {
    cells: CellScratch,
    /// A kernel vector over the training rows.
    k: Vec<f64>,
}

/// The centered additive kernel `K̄ = J (Σ_t K_t) J` of a stage's terms.
pub(super) struct TermKernel<'a> {
    n: usize,
    pub(super) parts: Vec<TermPart<'a>>,
    /// `K 1 / n`.
    col_mean: Vec<f64>,
    /// `1ᵀ K 1 / n²`.
    grand_mean: f64,
}

/// `v − mean(v)`, in place.
fn center(v: &mut [f64]) {
    let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
    for x in v {
        *x -= mean;
    }
}

impl<'a> TermKernel<'a> {
    pub(super) fn new(parts: Vec<TermPart<'a>>, n: usize) -> Self {
        let mut col_mean = vec![0.0; n];
        for part in &parts {
            for (a, &b) in col_mean.iter_mut().zip(&part.col_mean) {
                *a += b;
            }
        }
        let grand_mean = col_mean.iter().sum::<f64>() / n.max(1) as f64;
        TermKernel {
            n,
            parts,
            col_mean,
            grand_mean,
        }
    }

    /// Add `J k̃_t(x)` to `out`: part `t`'s kernel vector of a point in its
    /// grid cell `cell`, minus the part's `K_t 1 / n` (the tree centering),
    /// then centered.
    pub(super) fn add_query(
        &self,
        t: usize,
        cell: usize,
        scratch: &mut TermScratch,
        out: &mut [f64],
    ) {
        let part = &self.parts[t];
        let TermScratch { cells, k } = scratch;
        k.clear();
        k.resize(self.n, 0.0);
        part.add_cell_vector(cell, cells, k);
        for (v, &a) in k.iter_mut().zip(&part.col_mean) {
            *v -= a;
        }
        center(k);
        for (o, &v) in out.iter_mut().zip(k.iter()) {
            *o += v;
        }
    }

    /// `K̄ v`.
    pub(super) fn product(&self, v: &[f64]) -> Vec<f64> {
        let mut centered = v.to_vec();
        center(&mut centered);
        let mut out = vec![0.0; self.n];
        for part in &self.parts {
            part.add_product(&centered, &mut out);
        }
        center(&mut out);
        out
    }
}

impl Kernel for TermKernel<'_> {
    type Scratch = TermScratch;

    fn n(&self) -> usize {
        self.n
    }

    fn scratch(&self) -> TermScratch {
        TermScratch::default()
    }

    /// `K̄_i = K_i − K1/n − (K1/n)_i + 1ᵀK1/n²`, each term's row summed in
    /// term order.
    fn add_row(&self, row: usize, scratch: &mut TermScratch, out: &mut [f64]) {
        let TermScratch { cells, k } = scratch;
        k.clear();
        k.resize(self.n, 0.0);
        for part in &self.parts {
            part.add_cell_vector(part.cell_of_row[row] as usize, cells, k);
        }
        let shift = self.grand_mean - self.col_mean[row];
        for ((o, &v), &a) in out.iter_mut().zip(k.iter()).zip(&self.col_mean) {
            *o += v - a + shift;
        }
    }
}
