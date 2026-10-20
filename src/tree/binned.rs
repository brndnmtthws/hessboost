//! A finished tree routed by the bins of a [`GHistIndex`]: each split becomes
//! a test on its feature's local bins, so one pass over the index's narrow
//! feature columns (or a sparse index's row bins) finds every row's leaf
//! without reading a raw value.
//!
//! A row's bins route it as its values do when the tree's numeric thresholds
//! are cuts of the index (or lie below every finite value) and the cuts were
//! sketched from the row ([`SketchedRows`](crate::data::SketchedRows)): a
//! sketched value lies in the bin its neighbouring cuts bound, below the
//! feature's last cut, and a sketched category has a bin of its own.

use crate::data::ghist::{BinIndex, Bins, Column, ColumnSet, Columns, GHistIndex, missing_bit};
use crate::tree::RegTree;
use std::ops::Range;

/// One node of a [`BinnedTree`].
#[derive(Debug, Clone, Copy)]
struct BinNode {
    /// The split feature (`0` at a leaf, which tests nothing).
    feature: u32,
    /// The split feature's first global bin, and one past its last.
    first_bin: u32,
    end_bin: u32,
    /// A numeric split's limit (present local bins below it go left), or
    /// where a categorical split's per-bin table starts in
    /// [`BinnedTree::categories_left`].
    test: u32,
    categorical: bool,
    default_left: bool,
    leaf: bool,
    left: u32,
    right: u32,
}

impl BinNode {
    /// Whether a present row whose local bin of the split feature is `bin`
    /// goes left.
    #[inline(always)]
    fn present_left(&self, categories_left: &[bool], bin: usize) -> bool {
        if self.categorical {
            categories_left[self.test as usize + bin]
        } else {
            bin < self.test as usize
        }
    }

    /// The child a row takes.
    #[inline(always)]
    fn child(&self, left: bool) -> usize {
        (if left { self.left } else { self.right }) as usize
    }
}

/// A tree's splits as tests on one index's bins ([`BinnedTree::new`]).
pub(crate) struct BinnedTree<'a> {
    index: &'a GHistIndex,
    nodes: Vec<BinNode>,
    /// Whether each local bin of a categorical split's feature goes left, one
    /// run per categorical split.
    categories_left: Vec<bool>,
}

impl<'a> BinnedTree<'a> {
    /// `tree`'s splits as tests on `index`'s bins. `None` when a split does
    /// not fall on the index's cuts: a numeric threshold inside a bin, or a
    /// split whose kind its feature's cuts do not have. A tree grown on the
    /// index always does.
    pub(crate) fn new(tree: &RegTree, index: &'a GHistIndex) -> Option<Self> {
        let cuts = index.cuts();
        let mut categories_left = Vec::new();
        let nodes = tree
            .nodes()
            .iter()
            .map(|node| {
                let mut binned = BinNode {
                    feature: 0,
                    first_bin: 0,
                    end_bin: 0,
                    test: 0,
                    categorical: false,
                    default_left: false,
                    leaf: node.is_leaf(),
                    left: 0,
                    right: 0,
                };
                if binned.leaf {
                    return Some(binned);
                }
                let f = node.split_feature as usize;
                if f >= cuts.n_features() || node.is_categorical != cuts.is_categorical(f) {
                    return None;
                }
                let test = if node.is_categorical {
                    let start = categories_left.len();
                    categories_left.extend(cuts.categories_left(f, tree.node_categories(node)));
                    start
                } else {
                    cuts.bins_left_of(f, node.split_cond)?
                };
                let (first, end) = cuts.feature_bins(f);
                binned.feature = node.split_feature;
                binned.first_bin = u32::try_from(first).ok()?;
                binned.end_bin = u32::try_from(end).ok()?;
                binned.test = u32::try_from(test).ok()?;
                binned.categorical = node.is_categorical;
                binned.default_left = node.default_left;
                binned.left = u32::try_from(node.left).ok()?;
                binned.right = u32::try_from(node.right).ok()?;
                Some(binned)
            })
            .collect::<Option<Vec<_>>>()?;
        Some(BinnedTree {
            index,
            nodes,
            categories_left,
        })
    }

    /// Call `visit(row, leaf)` for every row of `rows` (below the index's
    /// row count), in order, with the id of the leaf the row's bins reach.
    pub(crate) fn for_each_leaf(&self, rows: Range<usize>, visit: impl FnMut(usize, usize)) {
        match self.index.columns() {
            Some(Columns::U8(set)) => self.walk_columns(set, rows, visit),
            Some(Columns::U16(set)) => self.walk_columns(set, rows, visit),
            Some(Columns::U32(set)) => self.walk_columns(set, rows, visit),
            // A sparse index less than half full keeps only its rows' bins.
            None => match self.index.bins() {
                Bins::U16(bins) => self.walk_rows(bins, rows, visit),
                Bins::U32(bins) => self.walk_rows(bins, rows, visit),
            },
        }
    }

    /// [`Self::for_each_leaf`] over feature columns of one width.
    #[inline(always)]
    #[allow(
        clippy::needless_bitwise_bool,
        reason = "branch-free routing predicates keep the walk free of data-dependent branches"
    )]
    fn walk_columns<B: BinIndex>(
        &self,
        set: ColumnSet<'a, B>,
        rows: Range<usize>,
        mut visit: impl FnMut(usize, usize),
    ) {
        // Each node's column, resolved once for the whole range.
        let steps: Vec<(BinNode, Column<'a, B>)> = self
            .nodes
            .iter()
            .map(|&node| (node, set.column(node.feature as usize)))
            .collect();
        for row in rows {
            let mut nid = 0;
            loop {
                let (node, column) = &steps[nid];
                if node.leaf {
                    break;
                }
                let missing = column.missing.is_some_and(|bits| missing_bit(bits, row));
                let present_left =
                    node.present_left(&self.categories_left, column.bins[row].index());
                nid = node.child((present_left & !missing) | (missing & node.default_left));
            }
            visit(row, nid);
        }
    }

    /// [`Self::for_each_leaf`] over each row's stored global bins: a row
    /// takes its first bin in the split feature's range (as
    /// [`DMatrix::get`](crate::data::DMatrix::get) takes a row's first entry
    /// of a feature), and lacking one it is missing there.
    fn walk_rows<B: BinIndex>(
        &self,
        bins: &[B],
        rows: Range<usize>,
        mut visit: impl FnMut(usize, usize),
    ) {
        let row_ptr = self.index.row_ptr();
        for row in rows {
            let stored = &bins[row_ptr[row]..row_ptr[row + 1]];
            let mut nid = 0;
            loop {
                let node = &self.nodes[nid];
                if node.leaf {
                    break;
                }
                let range = node.first_bin as usize..node.end_bin as usize;
                let left = match stored.iter().map(|b| b.index()).find(|b| range.contains(b)) {
                    Some(bin) => node.present_left(&self.categories_left, bin - range.start),
                    None => node.default_left,
                };
                nid = node.child(left);
            }
            visit(row, nid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainingParams;
    use crate::data::{DMatrix, FeatureType};
    use crate::training::train;

    /// Whether feature `f` of a row with hash `h` is absent.
    type Absent = fn(usize, usize) -> bool;

    /// Every row's bins reach the leaf its values reach, on numeric splits
    /// of a dense index, of one with missing values (columns with a missing
    /// bitmap) and of a sparse one too empty for columns (row bins), and on
    /// categorical splits.
    #[test]
    fn bins_reach_the_leaves_values_reach() {
        let n = 2_000;
        let layouts: [(&str, Absent); 3] = [
            ("dense", |_, _| false),
            ("missing values", |f, h| f == 1 && h.is_multiple_of(3)),
            ("sparse", |f, h| f != 2 && !h.is_multiple_of(5)),
        ];
        for (layout, absent) in layouts {
            let value = |i: usize, f: usize| -> Option<f32> {
                let h = (i * 2_654_435_761 + f * 40_503) % 1_000;
                (!absent(f, h)).then(|| match f {
                    // A category code.
                    2 => (h % 7) as f32,
                    _ => h as f32 / 10.0,
                })
            };
            let (mut indptr, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
            for i in 0..n {
                for f in 0..3 {
                    if let Some(v) = value(i, f) {
                        indices.push(f as u32);
                        values.push(v);
                    }
                }
                indptr.push(values.len());
            }
            let y: Vec<f32> = (0..n)
                .map(|i| {
                    let category = value(i, 2).unwrap_or(0.0) as u32;
                    let shift = if category.is_multiple_of(2) {
                        3.0
                    } else {
                        -2.0
                    };
                    value(i, 0).unwrap_or(50.0) * 0.1 + value(i, 1).unwrap_or(0.0) * 0.01 + shift
                })
                .collect();
            let data = DMatrix::from_csr(indptr, indices, values, 3)
                .unwrap()
                .with_labels(&y)
                .unwrap()
                .with_feature_types(&[
                    FeatureType::Numerical,
                    FeatureType::Numerical,
                    FeatureType::Categorical,
                ])
                .unwrap();
            let params = TrainingParams::builder()
                .max_depth(5)
                .max_bin(32)
                .build()
                .unwrap();
            let model = train(&params, &data, 8).unwrap();
            let index = data.hist_index(params.max_bin);
            let has_missing = index.columns().map(|columns| columns.has_missing());
            let expected = match layout {
                "dense" => Some(false),
                "missing values" => Some(true),
                _ => None,
            };
            assert_eq!(has_missing, expected, "{layout}");
            let mut categorical_splits = 0;
            for tree in model.trees() {
                categorical_splits += tree.nodes().iter().filter(|n| n.is_categorical).count();
                let binned = BinnedTree::new(tree, &index).expect("splits on the index's cuts");
                binned.for_each_leaf(0..n, |row, leaf| {
                    let by_value = tree.leaf_id_with(|f| data.get(row, f as usize));
                    assert_eq!(leaf, by_value, "row {row} of the {layout} index");
                });
            }
            assert!(categorical_splits > 0, "{layout}: no categorical split");
        }
    }

    /// A threshold inside a bin cannot be followed by bins.
    #[test]
    fn thresholds_off_the_cuts_are_refused() {
        let x: Vec<f32> = (0..400).map(|i| i as f32).collect();
        let data = DMatrix::from_dense(&x, 400, 1).unwrap();
        let index = data.hist_index(16);
        let cuts = index.cuts();
        let (start, _) = cuts.feature_bins(0);
        let on_cut = cuts.cut_value(start + 3);
        assert_eq!(cuts.bins_left_of(0, on_cut), Some(4));
        assert_eq!(cuts.bins_left_of(0, f32::MIN), Some(0));
        assert_eq!(cuts.bins_left_of(0, on_cut + 0.5), None);
        assert_eq!(cuts.bins_left_of(0, f32::NAN), None);
    }
}
