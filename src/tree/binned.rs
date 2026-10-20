//! A finished tree routed by the bins of a [`GHistIndex`]: each split becomes
//! a test on its feature's local bins, so one pass over the index's narrow
//! feature columns finds every row's leaf without reading a raw value.
//!
//! A row's bins route it as its values do when the tree's numeric thresholds
//! are cuts of the index (or lie below every finite value) and the cuts were
//! sketched from the row ([`SketchedRows`](crate::data::SketchedRows)): a
//! sketched value lies in the bin its neighbouring cuts bound, below the
//! feature's last cut, and a sketched category has a bin of its own.

use crate::data::ghist::{BinIndex, Column, ColumnSet, Columns, GHistIndex, missing_bit};
use crate::tree::RegTree;
use std::ops::Range;

/// One node of a [`BinnedTree`].
#[derive(Debug, Clone, Copy)]
struct BinNode {
    /// The split feature (`0` at a leaf, which tests nothing).
    feature: u32,
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

/// A tree's splits as tests on one index's feature columns
/// ([`BinnedTree::new`]).
pub(crate) struct BinnedTree<'a> {
    columns: Columns<'a>,
    nodes: Vec<BinNode>,
    /// Whether each local bin of a categorical split's feature goes left, one
    /// run per categorical split.
    categories_left: Vec<bool>,
}

impl<'a> BinnedTree<'a> {
    /// `tree`'s splits as tests on `index`'s bins. `None` when the index
    /// keeps no feature columns (a sparse one less than half full) or a split
    /// does not fall on the index's cuts: a numeric threshold inside a bin,
    /// or a split whose kind its feature's cuts do not have.
    pub(crate) fn new(tree: &RegTree, index: &'a GHistIndex) -> Option<Self> {
        let columns = index.columns()?;
        let cuts = index.cuts();
        let mut categories_left = Vec::new();
        let nodes = tree
            .nodes()
            .iter()
            .map(|node| {
                let mut binned = BinNode {
                    feature: 0,
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
                binned.feature = node.split_feature;
                binned.test = u32::try_from(test).ok()?;
                binned.categorical = node.is_categorical;
                binned.default_left = node.default_left;
                binned.left = u32::try_from(node.left).ok()?;
                binned.right = u32::try_from(node.right).ok()?;
                Some(binned)
            })
            .collect::<Option<Vec<_>>>()?;
        Some(BinnedTree {
            columns,
            nodes,
            categories_left,
        })
    }

    /// Call `visit(row, leaf)` for every row of `rows` (below the index's
    /// row count), in order, with the id of the leaf the row's bins reach.
    pub(crate) fn for_each_leaf(&self, rows: Range<usize>, visit: impl FnMut(usize, usize)) {
        match self.columns {
            Columns::U8(set) => self.walk(set, rows, visit),
            Columns::U16(set) => self.walk(set, rows, visit),
            Columns::U32(set) => self.walk(set, rows, visit),
        }
    }

    /// [`Self::for_each_leaf`] over columns of one width.
    #[inline(always)]
    #[allow(
        clippy::needless_bitwise_bool,
        reason = "branch-free routing predicates keep the walk free of data-dependent branches"
    )]
    fn walk<B: BinIndex>(
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
                let bin = column.bins[row].index();
                let present_left = if node.categorical {
                    self.categories_left[node.test as usize + bin]
                } else {
                    bin < node.test as usize
                };
                let left = (present_left & !missing) | (missing & node.default_left);
                nid = if left { node.left } else { node.right } as usize;
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

    /// Every row's bins reach the leaf its values reach, on numeric splits
    /// of a dense index and of one with missing values, and on categorical
    /// splits.
    #[test]
    fn bins_reach_the_leaves_values_reach() {
        let n = 2_000;
        for with_missing in [false, true] {
            let value = |i: usize, f: usize| -> f32 {
                let h = (i * 2_654_435_761 + f * 40_503) % 1_000;
                match f {
                    1 if with_missing && h.is_multiple_of(3) => f32::NAN,
                    // A category code.
                    2 => (h % 7) as f32,
                    _ => h as f32 / 10.0,
                }
            };
            let x: Vec<f32> = (0..n)
                .flat_map(|i| (0..3).map(move |f| value(i, f)))
                .collect();
            let y: Vec<f32> = (0..n)
                .map(|i| {
                    let category = if (value(i, 2) as u32).is_multiple_of(2) {
                        3.0
                    } else {
                        -2.0
                    };
                    value(i, 0) * 0.1 + value(i, 1).max(0.0) * 0.01 + category
                })
                .collect();
            let data = DMatrix::from_dense(&x, n, 3)
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
            let columns = index.columns().expect("a dense or nearly full index");
            assert_eq!(columns.has_missing(), with_missing);
            let mut categorical_splits = 0;
            for tree in model.trees() {
                categorical_splits += tree.nodes().iter().filter(|n| n.is_categorical).count();
                let binned = BinnedTree::new(tree, &index).expect("splits on the index's cuts");
                binned.for_each_leaf(0..n, |row, leaf| {
                    let by_value = tree.leaf_id_with(|f| data.get(row, f as usize));
                    assert_eq!(leaf, by_value, "row {row}, missing values: {with_missing}");
                });
            }
            assert!(categorical_splits > 0, "no categorical split exercised");
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
