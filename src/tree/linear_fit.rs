//! Fitting the linear leaves of [`LinearLeaves`](super::LinearLeaves): every
//! leaf's ridge-regularized Newton step on the numerical features along its
//! path, as the [`linear`](super::linear) module describes, with each term's
//! input range when the leaves clamp their inputs.

use crate::config::LinearTree;
use crate::data::{DMatrix, FeatureType};
use crate::objective::GradPair;
use crate::tree::builder::LeafRows;
use crate::tree::linear::LinearLeaves;
use crate::tree::regtree::{Node, RegTree};
use rayon::prelude::*;

/// Coefficients at or below this magnitude are dropped (LightGBM
/// `kZeroThreshold`, `1e-35f`).
const ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;

/// Fit a linear model in every leaf of `tree` from the training rows `rows`
/// of `data` and their gradients (`gpair`, indexed by row), as `linear`
/// configures them (slope penalty, input clamping), before shrinkage.
/// Single-leaf trees are left constant.
pub(crate) fn fit_linear_leaves(
    tree: &mut RegTree,
    data: &DMatrix,
    gpair: &[GradPair],
    rows: &[u32],
    linear: LinearTree,
) {
    let nodes = tree.nodes();
    if nodes.len() == 1 {
        return;
    }
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); nodes.len()];
    let leaves: Vec<u32> = rows
        .par_iter()
        .with_min_len(4096)
        .map(|&r| tree.leaf_id_with(|f| data.get(r as usize, f as usize)) as u32)
        .collect();
    for (&r, &leaf) in rows.iter().zip(&leaves) {
        members[leaf as usize].push(r);
    }
    fit_leaf_members(tree, data, gpair, &members, linear);
}

/// [`fit_linear_leaves`] from the rows the tree builder partitioned into
/// each leaf (ascending, as routing `rows` in order gives them), instead of
/// routing every row again.
pub(crate) fn fit_captured_linear_leaves(
    tree: &mut RegTree,
    data: &DMatrix,
    gpair: &[GradPair],
    leaf_rows: &[LeafRows],
    linear: LinearTree,
) {
    let n_nodes = tree.num_nodes();
    if n_nodes == 1 {
        return;
    }
    let mut members: Vec<&[u32]> = vec![&[]; n_nodes];
    for leaf in leaf_rows {
        members[leaf.node] = &leaf.rows;
    }
    fit_leaf_members(tree, data, gpair, &members, linear);
}

/// One leaf's model: its intercept, its non-negligible `(feature, slope)`
/// terms, and, when the leaves clamp, each term's input range (empty
/// otherwise).
struct LeafModel {
    intercept: f64,
    terms: Vec<(u32, f64)>,
    ranges: Vec<[f32; 2]>,
}

impl LeafModel {
    /// A leaf (or internal node) without terms.
    fn constant(intercept: f64) -> Self {
        LeafModel {
            intercept,
            terms: Vec::new(),
            ranges: Vec::new(),
        }
    }
}

/// Fit every leaf's model from its training rows, `members[leaf]`.
fn fit_leaf_members(
    tree: &mut RegTree,
    data: &DMatrix,
    gpair: &[GradPair],
    members: &[impl AsRef<[u32]> + Sync],
    linear: LinearTree,
) {
    let nodes = tree.nodes();
    let features = path_features(nodes, data.feature_types());
    let models: Vec<LeafModel> = (0..nodes.len())
        .into_par_iter()
        .map(|id| {
            if !nodes[id].is_leaf() {
                return LeafModel::constant(0.0);
            }
            fit_leaf(&features[id], members[id].as_ref(), data, gpair, linear)
                .unwrap_or_else(|| LeafModel::constant(f64::from(nodes[id].leaf_value)))
        })
        .collect();

    let mut offsets = Vec::with_capacity(nodes.len() + 1);
    let mut intercepts = Vec::with_capacity(nodes.len());
    let mut term_features = Vec::new();
    let mut term_coeffs = Vec::new();
    let mut term_ranges = Vec::new();
    offsets.push(0);
    for model in models {
        intercepts.push(model.intercept);
        for (f, c) in model.terms {
            term_features.push(f);
            term_coeffs.push(c);
        }
        term_ranges.extend(model.ranges);
        offsets.push(term_features.len() as u32);
    }
    tree.set_linear_leaves(LinearLeaves::from_parts(
        offsets,
        intercepts,
        term_features,
        term_coeffs,
        linear.clamp().then_some(term_ranges),
    ));
}

/// Sorted distinct numerical features split on along each leaf's path
/// (empty for internal nodes).
fn path_features(nodes: &[Node], types: &[FeatureType]) -> Vec<Vec<u32>> {
    let mut out = vec![Vec::new(); nodes.len()];
    let mut stack: Vec<(usize, Vec<u32>)> = vec![(0, Vec::new())];
    while let Some((id, mut path)) = stack.pop() {
        let node = &nodes[id];
        if node.is_leaf() {
            out[id] = path;
            continue;
        }
        let f = node.split_feature;
        let numerical = !node.is_categorical
            && types
                .get(f as usize)
                .is_none_or(|t| *t == FeatureType::Numerical);
        if numerical && let Err(pos) = path.binary_search(&f) {
            path.insert(pos, f);
        }
        stack.push((node.left as usize, path.clone()));
        stack.push((node.right as usize, path));
    }
    out
}

/// Solve one leaf's ridge system. `None` keeps the constant leaf: too few
/// complete rows, or a singular / non-finite solution. Otherwise returns the
/// intercept, the non-negligible `(feature, slope)` terms, and, when
/// `linear` clamps, their input ranges: each feature's smallest and largest
/// value among the complete rows with a nonzero gradient or Hessian.
fn fit_leaf(
    features: &[u32],
    rows: &[u32],
    data: &DMatrix,
    gpair: &[GradPair],
    linear: LinearTree,
) -> Option<LeafModel> {
    let p = features.len();
    let dim = p + 1;
    // Upper triangle of XᵀHX (row-major) and Xᵀg, accumulated in row order in
    // f64 from f32 inputs as LightGBM does.
    let mut xthx = vec![0.0f64; dim * (dim + 1) / 2];
    let mut xtg = vec![0.0f64; dim];
    let mut x = vec![0.0f32; dim];
    x[p] = 1.0;
    // Widened by comparison, so equal values (`-0.0` and `0.0`) keep the
    // first row's bits whatever the platform's `min`/`max`.
    let mut ranges = vec![[f32::INFINITY, f32::NEG_INFINITY]; if linear.clamp() { p } else { 0 }];
    let mut complete = 0usize;
    'rows: for &r in rows {
        for (slot, &f) in x.iter_mut().zip(features) {
            match data.get(r as usize, f as usize) {
                Some(v) => *slot = v,
                None => continue 'rows,
            }
        }
        complete += 1;
        let GradPair { grad, hess } = gpair[r as usize];
        // A row whose pair is zero (a zero-weight row) adds nothing to the
        // system: the model is not fit on its values.
        if grad != 0.0 || hess != 0.0 {
            for (range, &v) in ranges.iter_mut().zip(&x) {
                if v < range[0] {
                    range[0] = v;
                }
                if v > range[1] {
                    range[1] = v;
                }
            }
        }
        let mut k = 0;
        for i in 0..dim {
            let xi = f64::from(x[i]);
            xtg[i] += xi * f64::from(grad);
            let xih = xi * f64::from(hess);
            for &xj in &x[i..] {
                xthx[k] += xih * f64::from(xj);
                k += 1;
            }
        }
    }
    if complete < dim {
        return None;
    }
    let mut a = vec![0.0f64; dim * dim];
    let mut k = 0;
    for i in 0..dim {
        for j in i..dim {
            a[i * dim + j] = xthx[k];
            a[j * dim + i] = xthx[k];
            k += 1;
        }
        if i < p {
            a[i * dim + i] += linear.lambda();
        }
    }
    let rhs: Vec<f64> = xtg.iter().map(|v| -v).collect();
    // A solution exists only if some row's pair was nonzero (otherwise the
    // intercept's row of the system is zero) and every value was finite (an
    // infinite one makes the solution non-finite), so a fitted leaf's ranges
    // are finite and ordered.
    let beta = solve_full_pivot(a, rhs)?;
    let mut model = LeafModel::constant(beta[p]);
    for (i, (&f, &c)) in features.iter().zip(&beta).enumerate() {
        if c.abs() > ZERO_THRESHOLD {
            model.terms.push((f, c));
            model.ranges.extend(ranges.get(i));
        }
    }
    Some(model)
}

/// Solve `A x = b` for a square row-major `A` by Gaussian elimination with
/// full pivoting. Returns `None` when `A` is numerically singular (a pivot
/// below `ε·n` times the largest one, Eigen `FullPivLU`'s rank threshold) or
/// the solution is not finite.
fn solve_full_pivot(mut a: Vec<f64>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    // `cols[j]` is the unknown held in column `j` after column swaps.
    let mut cols: Vec<usize> = (0..n).collect();
    let mut first_pivot = 0.0f64;
    for k in 0..n {
        let (mut pr, mut pc, mut pv) = (k, k, 0.0f64);
        for r in k..n {
            for c in k..n {
                let v = a[r * n + c].abs();
                if v > pv {
                    (pr, pc, pv) = (r, c, v);
                }
            }
        }
        if k == 0 {
            first_pivot = pv;
        }
        // `pv` only ever takes compared (non-NaN) magnitudes.
        if pv <= first_pivot * f64::EPSILON * n as f64 {
            return None;
        }
        if pr != k {
            for c in 0..n {
                a.swap(pr * n + c, k * n + c);
            }
            b.swap(pr, k);
        }
        if pc != k {
            for r in 0..n {
                a.swap(r * n + pc, r * n + k);
            }
            cols.swap(pc, k);
        }
        let pivot = a[k * n + k];
        for r in k + 1..n {
            let factor = a[r * n + k] / pivot;
            if factor == 0.0 {
                continue;
            }
            for c in k..n {
                a[r * n + c] -= factor * a[k * n + c];
            }
            b[r] -= factor * b[k];
        }
    }
    let mut y = vec![0.0f64; n];
    for k in (0..n).rev() {
        let mut s = b[k];
        for c in k + 1..n {
            s -= a[k * n + c] * y[c];
        }
        y[k] = s / a[k * n + k];
    }
    let mut x = vec![0.0f64; n];
    for (j, &unknown) in cols.iter().enumerate() {
        x[unknown] = y[j];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{ChildLeaf, SplitRule};

    #[test]
    fn full_pivot_solver_matches_a_known_solution() {
        // The first pivot of this system sits off the diagonal.
        let a = vec![1.0, 2.0, 3.0, 2.0, 5.0, 3.0, 1.0, 0.0, 8.0];
        let want = [-40.0, 16.0, 5.0];
        let b: Vec<f64> = (0..3)
            .map(|r| (0..3).map(|c| a[r * 3 + c] * want[c]).sum())
            .collect();
        let got = solve_full_pivot(a, b).unwrap();
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-9, "{got:?}");
        }
        assert!(solve_full_pivot(vec![1.0, 2.0, 2.0, 4.0], vec![1.0, 2.0]).is_none());
    }

    /// A single root split of feature 0 at `x < 0`, missing values following
    /// `default_left`, with constant leaves `-0.5` / `0.25`.
    fn stump(default_left: bool) -> RegTree {
        let mut tree = RegTree::with_root(1.0);
        tree.expand(
            0,
            SplitRule::numeric(0, 0.0, default_left),
            ChildLeaf::new(-0.5, 1.0),
            ChildLeaf::new(0.25, 1.0),
        );
        tree
    }

    fn column(x: &[f32]) -> DMatrix {
        DMatrix::from_dense(x, x.len(), 1).unwrap()
    }

    /// Squared-error gradients at a zero margin for `y = 2x + 1`.
    fn line_gradients(x: &[f32]) -> Vec<GradPair> {
        x.iter()
            .map(|&v| GradPair::new(-(2.0 * v + 1.0), 1.0))
            .collect()
    }

    #[test]
    fn leaf_fit_is_the_ridge_newton_step() {
        // Right-leaf rows x = 1..4 with h = 1: the normal equations are
        // [[Σx² + λ, Σx], [Σx, n]] β = [Σx·y, Σy] (no penalty on the
        // intercept). The single left row cannot fit two coefficients.
        let x = [-1.0f32, 1.0, 2.0, 3.0, 4.0];
        let data = column(&x);
        let gpair = line_gradients(&x);
        let rows = [0, 1, 2, 3, 4];
        let lambda = 2.0;
        let mut tree = stump(true);
        fit_linear_leaves(
            &mut tree,
            &data,
            &gpair,
            &rows,
            LinearTree::new(lambda).unwrap(),
        );
        let (sxx, sx, n, sxy, sy) = (30.0 + lambda, 10.0, 4.0, 70.0, 24.0);
        let det = sxx * n - sx * sx;
        let slope = (n * sxy - sx * sy) / det;
        let intercept = (sxx * sy - sx * sxy) / det;
        let linear = tree.linear_leaves().unwrap();
        let (features, coeffs) = linear.terms(2);
        assert_eq!(features, &[0]);
        assert!((coeffs[0] - slope).abs() < 1e-12, "{coeffs:?} vs {slope}");
        assert!((linear.intercept(2) - intercept).abs() < 1e-12);
        assert_eq!(linear.terms(1).0, [] as [u32; 0]);
        assert_eq!(tree.predict_row(&data, 0), -0.5);
        // Without a penalty the fit recovers the line.
        let mut exact = stump(true);
        fit_linear_leaves(&mut exact, &data, &gpair, &rows, LinearTree::default());
        for (r, &v) in x.iter().enumerate().skip(1) {
            assert!((exact.predict_row(&data, r) - (2.0 * v + 1.0)).abs() < 1e-5);
        }
    }

    #[test]
    fn missing_features_are_skipped_in_the_fit_and_predict_the_constant() {
        // Missing values route right, into the fitted leaf.
        let x = [-1.0f32, 1.0, 2.0, f32::NAN, 3.0, 4.0];
        let data = column(&x);
        let mut gpair = line_gradients(&x);
        // The missing row's gradient would dominate the fit if it were used.
        gpair[3] = GradPair::new(1e6, 1.0);
        let mut tree = stump(false);
        let all = [0, 1, 2, 3, 4, 5];
        fit_linear_leaves(&mut tree, &data, &gpair, &all, LinearTree::default());
        assert_eq!(tree.predict_row(&data, 3), 0.25);
        for r in [1, 2, 4, 5] {
            let want = 2.0 * x[r] + 1.0;
            assert!((tree.predict_row(&data, r) - want).abs() < 1e-5, "row {r}");
        }
    }

    /// A clamping fit records each term's range over the rows that entered
    /// it (a zero-weight row's value does not widen it), predicts those rows
    /// bit for bit as the unclamped fit does, and is flat beyond the range.
    #[test]
    fn clamped_fits_record_their_rows_ranges_and_stay_flat_beyond_them() {
        // Right-leaf rows x = 1, 2, 3, 4, and a zero-weight row at x = 50.
        let x = [-1.0f32, 1.0, 2.0, 3.0, 50.0, 4.0];
        let data = column(&x);
        let mut gpair = line_gradients(&x);
        gpair[4] = GradPair::new(0.0, 0.0);
        let rows = [0, 1, 2, 3, 4, 5];
        let mut plain = stump(true);
        fit_linear_leaves(&mut plain, &data, &gpair, &rows, LinearTree::default());
        let mut clamped = stump(true);
        let clamp = LinearTree::default().with_clamp(true);
        fit_linear_leaves(&mut clamped, &data, &gpair, &rows, clamp);
        let (linear, unclamped) = (
            clamped.linear_leaves().unwrap(),
            plain.linear_leaves().unwrap(),
        );
        assert_eq!(linear.ranges(2), Some(&[[1.0f32, 4.0]][..]));
        // The single left row cannot fit a model: no terms, no ranges.
        assert_eq!(linear.ranges(1), Some(&[][..]));
        assert_eq!(unclamped.ranges(2), None);
        assert_eq!(linear.terms(2), unclamped.terms(2));
        for r in [0, 1, 2, 3, 5] {
            assert_eq!(
                clamped.predict_row(&data, r).to_bits(),
                plain.predict_row(&data, r).to_bits(),
                "row {r}"
            );
        }
        // The zero-weight row lies past the range: it gets the value at 4,
        // not the line's 101.
        assert_eq!(clamped.predict_row(&data, 4), clamped.predict_row(&data, 5));
        assert!((plain.predict_row(&data, 4) - 101.0).abs() < 1e-3);
    }

    #[test]
    fn categorical_path_features_route_but_stay_out_of_the_model() {
        // Root splits categorical feature 0; each child splits numeric 1.
        let mut tree = RegTree::with_root(8.0);
        let (l, r) = tree.expand(
            0,
            SplitRule::categorical(0, &[1], false),
            ChildLeaf::new(0.0, 4.0),
            ChildLeaf::new(0.0, 4.0),
        );
        for child in [l, r] {
            tree.expand(
                child,
                SplitRule::numeric(1, 0.5, true),
                ChildLeaf::new(0.0, 2.0),
                ChildLeaf::new(0.0, 2.0),
            );
        }
        let types = [FeatureType::Categorical, FeatureType::Numerical];
        let paths = path_features(tree.nodes(), &types);
        for (id, node) in tree.nodes().iter().enumerate() {
            let want: &[u32] = if node.is_leaf() { &[1] } else { &[] };
            assert_eq!(paths[id], want, "node {id}");
        }
    }
}
