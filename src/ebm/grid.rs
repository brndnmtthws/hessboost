//! The grid an EBM term's trees partition its features into.
//!
//! Every tree of a term splits only on the term's one or two features.
//! Along a numerical feature (an [`Axis`]) the union of the term's
//! thresholds `e_0 < … < e_{m−2}` cuts the line into `m` cells
//! `(−∞, e_0), [e_0, e_1), …, [e_{m−2}, ∞)`; along a categorical one every
//! category some split sends left gets a cell, then one cell holds every other
//! category. Either way a last cell holds missing values, every tree is
//! constant on every cell of the product grid, and each leaf receives a set
//! of cells per axis, stored as the runs of consecutive cells it covers
//! (for a numerical axis at most two: the values it receives, and the
//! missing cell when its path's default directions lead there). The shape
//! functions add leaf values over the leaves' boxes, the term kernel adds
//! leaf weights over them, both with difference arrays (`O(boxes + cells)`).

use super::TermAxis;
use crate::data::DMatrix;
use crate::tree::RegTree;

/// One feature of a term and its cells.
#[derive(Debug, Clone)]
pub(crate) struct Axis {
    pub(crate) feature: u32,
    pub(crate) kind: TermAxis,
}

impl Axis {
    /// Cells along the axis (the last one holds missing values).
    pub(crate) fn cells(&self) -> usize {
        self.kind.cells()
    }

    /// The cell of missing values (the last one).
    pub(crate) fn missing_cell(&self) -> usize {
        self.kind.cells() - 1
    }

    /// The cell of `value` (`None` or NaN: missing).
    pub(crate) fn cell(&self, value: Option<f32>) -> usize {
        self.kind.cell(value)
    }

    /// A value inside cell `c` (`None` for the missing cell), which every
    /// tree of the term routes like all the cell's values.
    fn representative(&self, c: usize) -> Option<f32> {
        if c == self.missing_cell() {
            return None;
        }
        match &self.kind {
            TermAxis::Numeric { edges } => Some(if c == 0 {
                f32::NEG_INFINITY
            } else {
                edges[c - 1]
            }),
            // Past the listed categories: the smallest code no split names
            // (at most `categories.len()`, so exact in `f32`; "largest + 1"
            // is not above 2^24, where it can round onto a listed code).
            TermAxis::Categorical { categories } => Some(match categories.get(c) {
                Some(&k) => k as f32,
                None => unnamed_code(categories) as f32,
            }),
        }
    }

    /// Whether a present value in non-missing cell `c` goes left at
    /// `node` (a split on this axis) of `tree`.
    fn goes_left(&self, tree: &RegTree, node: &crate::tree::Node, c: usize) -> bool {
        tree.goes_left(node, self.representative(c))
    }
}

/// The smallest category code not in the ascending, deduplicated
/// `categories`.
fn unnamed_code(categories: &[u32]) -> u32 {
    let mut code = 0u32;
    for &k in categories {
        if k != code {
            break;
        }
        code += 1;
    }
    code
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

/// The cells of each axis a leaf receives.
type Reach = [Vec<bool>; 2];

/// The runs of consecutive `true` cells, as `[lo, hi)` intervals.
fn runs(cells: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, &on) in cells.iter().chain(std::iter::once(&false)).enumerate() {
        match (on, start) {
            (true, None) => start = Some(i),
            (false, Some(lo)) => {
                out.push((lo, i));
                start = None;
            }
            _ => {}
        }
    }
    out
}

impl TermGrid {
    /// The grid of `trees`, which split only on `features` (one or two,
    /// ascending), each feature at numerical thresholds in every tree or on
    /// category sets in every tree; [`crate::ebm::EbmInfo::validate`] checks
    /// that of every loaded model.
    pub(crate) fn new(trees: &[&RegTree], features: &[u32]) -> Self {
        let axes: Vec<Axis> = features
            .iter()
            .map(|&feature| {
                let splits = || {
                    trees.iter().flat_map(|t| {
                        t.nodes()
                            .iter()
                            .filter(move |n| !n.is_leaf() && n.split_feature == feature)
                            .map(move |n| (*t, n))
                    })
                };
                let kind = if splits().any(|(_, n)| n.is_categorical) {
                    let mut categories: Vec<u32> = splits()
                        .flat_map(|(t, n)| t.node_categories(n).iter().copied())
                        .collect();
                    categories.sort_unstable();
                    categories.dedup();
                    TermAxis::Categorical { categories }
                } else {
                    let mut edges: Vec<f32> = splits().map(|(_, n)| n.split_cond).collect();
                    edges.sort_by(f32::total_cmp);
                    edges.dedup();
                    TermAxis::Numeric { edges }
                };
                Axis { feature, kind }
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
            let all: Reach = [vec![true; dims[0]], vec![true; dims[1]]];
            let mut stack = vec![(0usize, all)];
            while let Some((id, reach)) = stack.pop() {
                let node = tree.node(id);
                if node.is_leaf() {
                    let second = runs(&reach[1]);
                    let boxes = runs(&reach[0])
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
                let axis = &axes[a];
                let missing = axis.missing_cell();
                let (mut left, mut right) = (reach.clone(), reach);
                for c in 0..axis.cells() {
                    if !left[a][c] {
                        continue;
                    }
                    let goes_left = if c == missing {
                        node.default_left
                    } else {
                        axis.goes_left(tree, node, c)
                    };
                    if goes_left {
                        right[a][c] = false;
                    } else {
                        left[a][c] = false;
                    }
                }
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
        let mut diff = Vec::new();
        self.reset_diff(&mut diff);
        diff
    }

    /// Make `diff` a zeroed difference array, reusing its allocation.
    pub(crate) fn reset_diff(&self, diff: &mut Vec<f64>) {
        diff.clear();
        diff.resize((self.dims[0] + 1) * (self.dims[1] + 1), 0.0);
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
        let mut out = Vec::new();
        self.integrate_into(diff, &mut out);
        out
    }

    /// [`Self::integrate`] into `out`, reusing its allocation.
    pub(crate) fn integrate_into(&self, diff: &[f64], out: &mut Vec<f64>) {
        let [d0, d1] = self.dims;
        let s = d1 + 1;
        out.clear();
        out.resize(d0 * d1, 0.0);
        for i in 0..d0 {
            let mut run = 0.0;
            for j in 0..d1 {
                run += diff[i * s + j];
                out[i * d1 + j] = run + if i > 0 { out[(i - 1) * d1 + j] } else { 0.0 };
            }
        }
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
