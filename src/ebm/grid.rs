//! The grid an EBM term's trees partition its features into.
//!
//! Every tree of a term splits only on the term's one or two features, at
//! thresholds. Along each feature (an [`Axis`]) the union of the term's
//! thresholds `e_0 < … < e_{m−2}` cuts the line into `m` cells
//! `(−∞, e_0), [e_0, e_1), …, [e_{m−2}, ∞)` plus a last cell for missing
//! values, so every tree is constant on every cell of the product grid and
//! each of its leaves covers a union of at most two intervals per axis (the
//! non-missing values it receives, and the missing cell when its path's
//! default directions lead there): at most four boxes. The shape functions
//! add leaf values over boxes, the term kernel adds leaf weights over them,
//! both with difference arrays (`O(leaves + cells)`).

use crate::data::DMatrix;
use crate::tree::RegTree;

/// One feature of a term and the thresholds its trees split it at.
#[derive(Debug, Clone)]
pub(crate) struct Axis {
    pub(crate) feature: u32,
    /// Sorted, distinct thresholds.
    pub(crate) edges: Vec<f32>,
}

impl Axis {
    /// Cells along the axis: the `edges.len() + 1` intervals, then missing.
    pub(crate) fn cells(&self) -> usize {
        self.edges.len() + 2
    }

    /// The cell of missing values (the last one).
    pub(crate) fn missing_cell(&self) -> usize {
        self.edges.len() + 1
    }

    /// The cell of `value` (`None` or NaN: missing).
    pub(crate) fn cell(&self, value: Option<f32>) -> usize {
        match value {
            Some(x) if !x.is_nan() => self.edges.partition_point(|&e| e <= x),
            _ => self.missing_cell(),
        }
    }

    /// A value inside cell `c` (`None` for the missing cell), which every
    /// tree of the term routes like all the cell's values.
    fn representative(&self, c: usize) -> Option<f32> {
        if c == self.missing_cell() {
            None
        } else if c == 0 {
            Some(f32::NEG_INFINITY)
        } else {
            Some(self.edges[c - 1])
        }
    }
}

/// A box of cells, `[lo0, hi0) × [lo1, hi1)` (the second axis is `[0, 1)`
/// for a one-feature term).
pub(crate) type CellBox = [usize; 4];

/// One leaf of one of the term's trees: its node id and the boxes of cells
/// it receives.
#[derive(Debug, Clone)]
pub(crate) struct Leaf {
    pub(crate) node: u32,
    pub(crate) boxes: Vec<CellBox>,
}

/// The cell grid of one term and the leaves of its trees on it.
#[derive(Debug, Clone)]
pub(crate) struct TermGrid {
    pub(crate) axes: Vec<Axis>,
    /// Cells along the first and second axis (`1` for a one-feature term).
    dims: [usize; 2],
    /// Every tree's leaves, tree by tree: tree `t` owns
    /// `leaves[leaf_start[t]..leaf_start[t + 1]]`.
    pub(crate) leaves: Vec<Leaf>,
    pub(crate) leaf_start: Vec<usize>,
    /// Per tree, the index into `leaves` of each of its nodes (`u32::MAX`
    /// for internal nodes), flattened with `node_start`.
    node_leaf: Vec<u32>,
    node_start: Vec<usize>,
}

/// The per-axis cell intervals a leaf receives: `[lo, hi)` of the non-missing
/// cells and whether the missing cell is included.
#[derive(Clone, Copy)]
struct Reach {
    lo: usize,
    hi: usize,
    missing: bool,
}

impl Reach {
    /// The reached cells as at most two intervals, merged when adjacent.
    fn intervals(self, missing_cell: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(2);
        if self.lo < self.hi {
            if self.missing && self.hi == missing_cell {
                out.push((self.lo, missing_cell + 1));
                return out;
            }
            out.push((self.lo, self.hi));
        }
        if self.missing {
            out.push((missing_cell, missing_cell + 1));
        }
        out
    }
}

impl TermGrid {
    /// The grid of `trees`, which split only on `features` (one or two,
    /// ascending) at numeric thresholds; [`crate::ebm::EbmInfo::validate`]
    /// checks that of every loaded model.
    pub(crate) fn new(trees: &[&RegTree], features: &[u32]) -> Self {
        let axes: Vec<Axis> = features
            .iter()
            .map(|&feature| {
                let mut edges: Vec<f32> = trees
                    .iter()
                    .flat_map(|t| t.nodes())
                    .filter(|n| !n.is_leaf() && n.split_feature == feature)
                    .map(|n| n.split_cond)
                    .collect();
                edges.sort_by(f32::total_cmp);
                edges.dedup();
                Axis { feature, edges }
            })
            .collect();
        let dims = [axes[0].cells(), axes.get(1).map_or(1, Axis::cells)];
        let mut leaves = Vec::new();
        let mut leaf_start = Vec::with_capacity(trees.len() + 1);
        let mut node_leaf = Vec::new();
        let mut node_start = Vec::with_capacity(trees.len() + 1);
        for tree in trees {
            leaf_start.push(leaves.len());
            node_start.push(node_leaf.len());
            let base = node_leaf.len();
            node_leaf.resize(base + tree.num_nodes(), u32::MAX);
            let full = |a: &Axis| Reach {
                lo: 0,
                hi: a.missing_cell(),
                missing: true,
            };
            let mut stack = vec![(
                0usize,
                [
                    full(&axes[0]),
                    axes.get(1).map_or(
                        Reach {
                            lo: 0,
                            hi: 1,
                            missing: false,
                        },
                        full,
                    ),
                ],
            )];
            while let Some((id, reach)) = stack.pop() {
                let node = tree.node(id);
                if node.is_leaf() {
                    let first = reach[0].intervals(axes[0].missing_cell());
                    let second = match axes.get(1) {
                        Some(a) => reach[1].intervals(a.missing_cell()),
                        None => vec![(0, 1)],
                    };
                    let boxes = first
                        .iter()
                        .flat_map(|&(lo0, hi0)| {
                            second.iter().map(move |&(lo1, hi1)| [lo0, hi0, lo1, hi1])
                        })
                        .collect();
                    node_leaf[base + id] = leaves.len() as u32;
                    leaves.push(Leaf {
                        node: id as u32,
                        boxes,
                    });
                    continue;
                }
                let a = usize::from(axes[0].feature != node.split_feature);
                let e = axes[a].edges.partition_point(|&v| v < node.split_cond);
                let r = reach[a];
                let mut left = reach;
                left[a] = Reach {
                    lo: r.lo,
                    hi: r.hi.min(e + 1),
                    missing: r.missing && node.default_left,
                };
                let mut right = reach;
                right[a] = Reach {
                    lo: r.lo.max(e + 1),
                    hi: r.hi,
                    missing: r.missing && !node.default_left,
                };
                stack.push((node.right as usize, right));
                stack.push((node.left as usize, left));
            }
        }
        leaf_start.push(leaves.len());
        node_start.push(node_leaf.len());
        TermGrid {
            axes,
            dims,
            leaves,
            leaf_start,
            node_leaf,
            node_start,
        }
    }

    /// Number of cells.
    pub(crate) fn len(&self) -> usize {
        self.dims[0] * self.dims[1]
    }

    /// The cell of row `row` of `data`.
    pub(crate) fn cell_of_row(&self, data: &DMatrix, row: usize) -> usize {
        let coord = |a: &Axis| a.cell(data.get(row, a.feature as usize));
        coord(&self.axes[0]) * self.dims[1] + self.axes.get(1).map_or(0, coord)
    }

    /// Index into [`Self::leaves`] of the leaf of tree `t` (`trees[t]` of
    /// [`Self::new`]) that cell `cell` reaches.
    pub(crate) fn leaf_of_cell(&self, tree: &RegTree, t: usize, cell: usize) -> usize {
        let (c0, c1) = (cell / self.dims[1], cell % self.dims[1]);
        let v0 = self.axes[0].representative(c0);
        let v1 = self.axes.get(1).and_then(|a| a.representative(c1));
        let f0 = self.axes[0].feature;
        let node = tree.leaf_id_with(|f| if f == f0 { v0 } else { v1 });
        self.node_leaf[self.node_start[t] + node] as usize
    }

    /// The per-cell sum over every leaf of `value(leaf)` on its boxes.
    pub(crate) fn paint(&self, mut value: impl FnMut(usize) -> f64) -> Vec<f64> {
        let mut diff = self.diff();
        for (i, leaf) in self.leaves.iter().enumerate() {
            let v = value(i);
            if v != 0.0 {
                for &b in &leaf.boxes {
                    self.box_add(&mut diff, b, v);
                }
            }
        }
        self.integrate(&diff)
    }

    /// A zeroed difference array (`(dims[0] + 1) × (dims[1] + 1)`).
    pub(crate) fn diff(&self) -> Vec<f64> {
        vec![0.0; (self.dims[0] + 1) * (self.dims[1] + 1)]
    }

    /// Add `v` over box `b` of the difference array `diff`.
    pub(crate) fn box_add(&self, diff: &mut [f64], b: CellBox, v: f64) {
        let s = self.dims[1] + 1;
        let [lo0, hi0, lo1, hi1] = b;
        diff[lo0 * s + lo1] += v;
        diff[lo0 * s + hi1] -= v;
        diff[hi0 * s + lo1] -= v;
        diff[hi0 * s + hi1] += v;
    }

    /// The cell values of the difference array `diff`.
    pub(crate) fn integrate(&self, diff: &[f64]) -> Vec<f64> {
        let [d0, d1] = self.dims;
        let s = d1 + 1;
        let mut out = vec![0.0; d0 * d1];
        for i in 0..d0 {
            let mut run = 0.0;
            for j in 0..d1 {
                run += diff[i * s + j];
                out[i * d1 + j] = run + if i > 0 { out[(i - 1) * d1 + j] } else { 0.0 };
            }
        }
        out
    }

    /// The inclusive-exclusive prefix sums `P[i][j] = Σ_{< i, < j}` of the
    /// cell values `cells`, `(dims[0] + 1) × (dims[1] + 1)`.
    pub(crate) fn prefix(&self, cells: &[f64]) -> Vec<f64> {
        let [d0, d1] = self.dims;
        let s = d1 + 1;
        let mut p = vec![0.0; (d0 + 1) * s];
        for i in 0..d0 {
            let mut run = 0.0;
            for j in 0..d1 {
                run += cells[i * d1 + j];
                p[(i + 1) * s + j + 1] = p[i * s + j + 1] + run;
            }
        }
        p
    }

    /// The sum over box `b` of the cells whose prefix sums are `p`.
    pub(crate) fn box_sum(&self, p: &[f64], b: CellBox) -> f64 {
        let s = self.dims[1] + 1;
        let [lo0, hi0, lo1, hi1] = b;
        p[hi0 * s + hi1] - p[lo0 * s + hi1] - p[hi0 * s + lo1] + p[lo0 * s + lo1]
    }
}
