//! The margin caches training keeps current, and how a tree adds to them.

use super::eval::EvalSet;
use crate::data::ghist::GHistIndex;
use crate::data::{DMatrix, SketchedRows};
use crate::model::{BoostedModel, shrink_margins};
use crate::tree::RegTree;
use crate::tree::binned::BinnedTree;
use crate::tree::builder::LeafRows;
use rayon::prelude::*;

/// Apply `update` to every `(row, row margins)` pair of `margins`
/// (`[row][n_out]`), in parallel for large inputs. Rows are independent, so
/// parallel traversal preserves each row's floating-point addition order.
fn for_each_row_margins(
    margins: &mut [f32],
    n_out: usize,
    update: impl Fn((usize, &mut [f32])) + Sync + Send,
) {
    if margins.len() / n_out >= 4096 && rayon::current_num_threads() > 1 {
        margins
            .par_chunks_mut(n_out)
            .with_min_len(1024)
            .enumerate()
            .for_each(update);
    } else {
        margins.chunks_mut(n_out).enumerate().for_each(update);
    }
}

/// Rows per parallel task of the leaf-value margin updates.
const MARGIN_CHUNK_ROWS: usize = 8192;

/// Call `update(first, chunk)` on consecutive row chunks of `margins`
/// (`[row][n_out]`), `chunk` holding the margins of rows `first..`, in
/// parallel for large inputs.
fn for_each_row_chunk(
    margins: &mut [f32],
    n_out: usize,
    update: impl Fn(usize, &mut [f32]) + Sync + Send,
) {
    if margins.len() / n_out < 2 * MARGIN_CHUNK_ROWS || rayon::current_num_threads() <= 1 {
        update(0, margins);
    } else {
        margins
            .par_chunks_mut(MARGIN_CHUNK_ROWS * n_out)
            .enumerate()
            .for_each(|(chunk, margins)| update(chunk * MARGIN_CHUNK_ROWS, margins));
    }
}

/// How a finished tree finds the training rows' leaves when no list of them
/// was kept.
#[derive(Clone, Copy)]
pub(super) enum RowRouting<'a> {
    /// Through the rows' bins in the training index, which route every row
    /// as its values do ([`SketchedRows::All`]).
    Bins(&'a GHistIndex),
    /// Through the rows' values.
    Values,
}

impl<'a> RowRouting<'a> {
    /// The routing of the training rows of `index` sketched as `sketched`
    /// says: by bins only when every row was.
    pub(super) fn of(index: &'a GHistIndex, sketched: SketchedRows) -> Self {
        match sketched {
            SketchedRows::All => RowRouting::Bins(index),
            SketchedRows::Weighted | SketchedRows::Partial => RowRouting::Values,
        }
    }

    /// `tree` routed by bins, when this routing has them and the tree splits
    /// on their cuts.
    pub(super) fn binned(self, tree: &RegTree) -> Option<BinnedTree<'a>> {
        match self {
            RowRouting::Bins(index) => BinnedTree::new(tree, index),
            RowRouting::Values => None,
        }
    }
}

/// Which margins of a row a tree adds to.
#[derive(Clone, Copy)]
pub(super) enum TreeOutput {
    /// A scalar-leaf tree feeding one output.
    Scalar(usize),
    /// A vector-leaf tree feeding every output.
    Vector,
}

/// The margins training keeps current, `[row][n_out]`: the training
/// matrix's and each eval set's, starting from the model's full current
/// predictions (a dataset's per-instance `base_margin`, when present,
/// overrides the per-output intercepts). Each new tree adds to every cell
/// once, so every cell sums its trees in ensemble order.
pub(super) struct MarginCaches<'a> {
    dtrain: &'a DMatrix,
    eval_sets: &'a [EvalSet<'a>],
    n_out: usize,
    /// The training matrix's margins.
    pub(super) train: Vec<f32>,
    /// Each eval set's margins, in eval-set order.
    pub(super) evals: Vec<Vec<f32>>,
}

impl<'a> MarginCaches<'a> {
    pub(super) fn new(
        model: &BoostedModel,
        dtrain: &'a DMatrix,
        eval_sets: &'a [EvalSet<'a>],
    ) -> Self {
        let trees = 0..model.num_trees();
        MarginCaches {
            dtrain,
            eval_sets,
            n_out: model.n_outputs(),
            train: model.margin_from_trees(dtrain.into(), trees.clone()),
            evals: eval_sets
                .iter()
                .map(|set| model.margin_from_trees(set.data.into(), trees.clone()))
                .collect(),
        }
    }

    /// Add `tree`'s predictions to every cache. `leaf_rows`, when given,
    /// lists the leaf of every training row; otherwise the training rows are
    /// routed as `routing` says.
    pub(super) fn add_tree(
        &mut self,
        tree: &RegTree,
        output: TreeOutput,
        leaf_rows: Option<&[LeafRows]>,
        routing: RowRouting,
    ) {
        let (train, n_out) = (&mut self.train, self.n_out);
        match leaf_rows {
            Some(leaf_rows) => apply_leaf_rows(tree, self.dtrain, leaf_rows, train, n_out, output),
            None => add_routed_margins(tree, self.dtrain, routing, train, n_out, output),
        }
        for (margins, set) in self.evals.iter_mut().zip(self.eval_sets) {
            add_tree_margins(tree, set.data, margins, n_out, output);
        }
    }

    /// Add `tree`'s predictions to the eval sets' caches only: the training
    /// margins are kept elsewhere (on a GPU, by device-resident rounds).
    pub(super) fn add_tree_to_evals(&mut self, tree: &RegTree, output: TreeOutput) {
        for (margins, set) in self.evals.iter_mut().zip(self.eval_sets) {
            add_tree_margins(tree, set.data, margins, self.n_out, output);
        }
    }

    /// Recompute the training matrix's cache from `model`.
    pub(super) fn recompute_train(&mut self, model: &BoostedModel) {
        self.train = model.margin_from_trees(self.dtrain.into(), 0..model.num_trees());
    }

    /// Multiply every cached margin by `factor` (model shrinkage,
    /// [`shrink_margins`], the step prediction repeats). Cells are
    /// independent, so the parallel pass gives the serial result.
    pub(super) fn scale(&mut self, factor: f64) {
        let n_out = self.n_out;
        for margins in std::iter::once(&mut self.train).chain(&mut self.evals) {
            for_each_row_margins(margins, n_out, |(_, row)| shrink_margins(row, factor));
        }
    }

    /// Recompute every cache from `model` (after a DART rescaling, which
    /// makes them non-additive).
    pub(super) fn recompute(&mut self, model: &BoostedModel) {
        self.train = model.margin_from_trees(self.dtrain.into(), 0..model.num_trees());
        for (margins, set) in self.evals.iter_mut().zip(self.eval_sets) {
            *margins = model.margin_from_trees(set.data.into(), 0..model.num_trees());
        }
    }

    /// The eval sets' matrices, in eval-set order.
    pub(super) fn eval_data(&self) -> impl Iterator<Item = &'a DMatrix> + 'a {
        self.eval_sets.iter().map(|set| set.data)
    }
}

/// Add `tree`'s prediction of every row of `data` to `margins` (training's
/// margin caches and the online state's replay).
pub(super) fn add_tree_margins(
    tree: &RegTree,
    data: &DMatrix,
    margins: &mut [f32],
    n_out: usize,
    output: TreeOutput,
) {
    match output {
        TreeOutput::Scalar(k) => for_each_row_margins(margins, n_out, |(row, margin)| {
            margin[k] += tree.predict_row(data, row);
        }),
        TreeOutput::Vector => for_each_row_margins(margins, n_out, |(row, margin)| {
            let leaf = tree.leaf_id_with(|f| data.get(row, f as usize));
            for (m, &v) in margin.iter_mut().zip(tree.leaf_vector(leaf)) {
                *m += v;
            }
        }),
    }
}

/// [`add_tree_margins`] of the training rows `data`, routed as `routing`
/// says: by bins unless the tree's leaves are linear (their models read the
/// values) or its splits miss the bins' cuts. Each row still receives one
/// addition of its leaf's value, so the margins are the routing's by value
/// bit for bit.
fn add_routed_margins(
    tree: &RegTree,
    data: &DMatrix,
    routing: RowRouting,
    margins: &mut [f32],
    n_out: usize,
    output: TreeOutput,
) {
    let binned = match tree.linear_leaves() {
        None => routing.binned(tree),
        Some(_) => None,
    };
    let Some(binned) = binned else {
        return add_tree_margins(tree, data, margins, n_out, output);
    };
    for_each_row_chunk(margins, n_out, |first, margins| {
        let rows = first..first + margins.len() / n_out;
        match output {
            TreeOutput::Scalar(k) => binned.for_each_leaf(rows, |row, leaf| {
                margins[(row - first) * n_out + k] += tree.node(leaf).leaf_value;
            }),
            TreeOutput::Vector => binned.for_each_leaf(rows, |row, leaf| {
                let at = (row - first) * n_out;
                for (m, &v) in margins[at..at + n_out]
                    .iter_mut()
                    .zip(tree.leaf_vector(leaf))
                {
                    *m += v;
                }
            }),
        }
    });
}

/// Add each leaf's value (or vector), or its linear model's output, to the
/// margins of the training rows of `data` that reached it.
fn apply_leaf_rows(
    tree: &RegTree,
    data: &DMatrix,
    leaf_rows: &[LeafRows],
    margins: &mut [f32],
    n_out: usize,
    output: TreeOutput,
) {
    match (output, tree.linear_leaves()) {
        (TreeOutput::Scalar(k), None) => apply_leaf_values(
            leaf_rows,
            margins,
            n_out,
            |node| tree.node(node).leaf_value,
            |margins, base, _, value| margins[base + k] += value,
        ),
        // `RegTree::predict_row`'s linear-leaf arithmetic, without routing.
        (TreeOutput::Scalar(k), Some(linear)) => apply_leaf_values(
            leaf_rows,
            margins,
            n_out,
            |node| node,
            |margins, base, row, node| {
                let get = |f: u32| data.get(row as usize, f as usize);
                margins[base + k] += linear.predict(node, tree.node(node).leaf_value, get);
            },
        ),
        (TreeOutput::Vector, _) => apply_leaf_values(
            leaf_rows,
            margins,
            n_out,
            |node| tree.leaf_vector(node),
            |margins, base, _, value: &[f32]| {
                for (m, &v) in margins[base..base + n_out].iter_mut().zip(value) {
                    *m += v;
                }
            },
        ),
    }
}

/// `add(margins, row * n_out, row, value(leaf))` for every row of every leaf of
/// `leaf_rows`, in parallel row chunks for large inputs. Leaf row lists are
/// ascending, so each chunk locates its slice of every leaf by binary
/// search; each row still receives one addition per tree.
fn apply_leaf_values<V: Copy + Send + Sync>(
    leaf_rows: &[LeafRows],
    margins: &mut [f32],
    n_out: usize,
    value: impl Fn(usize) -> V + Sync,
    add: impl Fn(&mut [f32], usize, u32, V) + Sync,
) {
    for_each_row_chunk(margins, n_out, |first, margins| {
        let (first, last) = (first as u32, (first + margins.len() / n_out) as u32);
        for leaf in leaf_rows {
            let v = value(leaf.node);
            let start = leaf.rows.partition_point(|&row| row < first);
            let end = start + leaf.rows[start..].partition_point(|&row| row < last);
            for &row in &leaf.rows[start..end] {
                add(margins, (row - first) as usize * n_out, row, v);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objective::GradPair;
    use crate::tree::{ChildLeaf, SplitRule};

    #[test]
    fn parallel_margin_updates_preserve_output_columns() {
        let n = 4103;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                if i % 11 == 0 {
                    f32::NAN
                } else {
                    (i % 7) as f32
                }
            })
            .collect();
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let mut tree = RegTree::with_root(n as f32);
        tree.expand(
            0,
            SplitRule::numeric(0, 3.0, true),
            ChildLeaf::new(-0.25, 1.0),
            ChildLeaf::new(0.75, 1.0),
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for outputs in [1, 3] {
            let mut actual: Vec<f32> = (0..n * outputs).map(|i| i as f32 / 100.0).collect();
            let mut expected = actual.clone();
            for output in 0..outputs {
                for row in 0..n {
                    expected[row * outputs + output] += tree.predict_row(&data, row);
                }
                pool.install(|| {
                    add_tree_margins(
                        &tree,
                        &data,
                        &mut actual,
                        outputs,
                        TreeOutput::Scalar(output),
                    );
                });
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn chunked_leaf_rows_match_the_serial_traversal() {
        // Enough rows for the chunked parallel path (two 8192-row chunks
        // and a partial third); leaves interleave across chunk boundaries.
        let n = 20_000;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                if i % 13 == 0 {
                    f32::NAN
                } else {
                    (i % 7) as f32
                }
            })
            .collect();
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let leaf_rows_of = |tree: &RegTree| {
            let mut leaves: Vec<LeafRows> = Vec::new();
            for row in 0..n {
                let node = tree.leaf_id_with(|f| data.get(row, f as usize));
                match leaves.iter_mut().find(|l| l.node == node) {
                    Some(leaf) => leaf.rows.push(row as u32),
                    None => leaves.push(LeafRows {
                        node,
                        rows: vec![row as u32],
                    }),
                }
            }
            leaves
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let start = |outputs: usize| -> Vec<f32> {
            (0..n * outputs).map(|i| (i % 97) as f32 / 8.0).collect()
        };

        let mut scalar = RegTree::with_root(n as f32);
        scalar.expand(
            0,
            SplitRule::numeric(0, 3.0, true),
            ChildLeaf::new(-0.25, 1.0),
            ChildLeaf::new(0.75, 1.0),
        );
        let leaves = leaf_rows_of(&scalar);
        let mut linear = scalar.clone();
        let gpair: Vec<GradPair> = x
            .iter()
            .map(|&v| GradPair::new(-(2.0 * v.max(0.0) + 1.0), 1.0))
            .collect();
        let rows: Vec<u32> = (0..n as u32).collect();
        crate::tree::linear_fit::fit_linear_leaves(&mut linear, &data, &gpair, &rows, 0.5);
        assert!(linear.linear_leaves().is_some());
        for tree in [&scalar, &linear] {
            let output = TreeOutput::Scalar(1);
            let mut expected = start(3);
            add_tree_margins(tree, &data, &mut expected, 3, output);
            let mut actual = start(3);
            pool.install(|| apply_leaf_rows(tree, &data, &leaves, &mut actual, 3, output));
            assert_eq!(actual, expected);
        }

        let mut vector = RegTree::with_vector_root(3, n as f32);
        let (left, right) = vector.expand(
            0,
            SplitRule::numeric(0, 3.0, false),
            ChildLeaf::new(0.0, 1.0),
            ChildLeaf::new(0.0, 1.0),
        );
        let (ll, lr) = vector.expand(
            left,
            SplitRule::numeric(0, 1.0, true),
            ChildLeaf::new(0.0, 1.0),
            ChildLeaf::new(0.0, 1.0),
        );
        vector.set_leaf_vector(ll, &[0.5, -1.0, 0.125]);
        vector.set_leaf_vector(lr, &[-0.75, 0.25, 2.0]);
        vector.set_leaf_vector(right, &[1.5, 0.0625, -0.5]);
        let leaves = leaf_rows_of(&vector);
        let mut expected = start(3);
        add_tree_margins(&vector, &data, &mut expected, 3, TreeOutput::Vector);
        let mut actual = start(3);
        pool.install(|| {
            apply_leaf_rows(&vector, &data, &leaves, &mut actual, 3, TreeOutput::Vector);
        });
        assert_eq!(actual, expected);
    }
}
