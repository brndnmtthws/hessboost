//! Linear leaves: LightGBM's `linear_tree` (opt-in, beyond XGBoost).
//!
//! After a tree's structure is grown, every leaf fits a ridge-regularized
//! linear model on the numerical features split on along its root-to-leaf path
//! (LightGBM `linear_tree_learner.cpp`, after Shi et al., "Gradient Boosting
//! With Piece-Wise Linear Regression Trees"). With `X` the leaf's rows over
//! those features plus a trailing column of ones, `H = diag(h)` and `g` the
//! rows' gradients, the coefficients are the Newton step
//!
//! `β = −(XᵀHX + Λ)⁻¹ Xᵀg`,  `Λ = diag(λ, …, λ, 0)`,
//!
//! so `linear_lambda` (`λ`) penalizes the slopes but not the intercept.
//! Following LightGBM:
//!
//! - categorical features route rows but never enter a leaf model;
//! - rows with a missing value in any of the leaf's features are left out of
//!   its fit, and at prediction such rows get the ordinary constant leaf
//!   value instead (hessboost treats absent sparse entries as missing, as it
//!   does everywhere);
//! - a leaf with fewer complete rows than coefficients keeps its constant
//!   value (so does a leaf whose system is singular, which LightGBM's
//!   `fullPivLu().inverse()` leaves undefined);
//! - slopes with magnitude `<= 1e-35` (`kZeroThreshold`) are dropped;
//! - trees of the first boosting round and single-leaf trees stay constant.
//!
//! Learning-rate shrinkage scales intercepts and slopes together with the
//! constant leaf values ([`RegTree::scale_leaves`]). Split finding is
//! unchanged, so monotone constraints bound only the constant values, not the
//! fitted slopes (as in LightGBM). Numerical features should be on comparable
//! scales, since the slope penalty is not scale invariant.

use crate::data::{DMatrix, FeatureType};
use crate::objective::GradPair;
use crate::tree::regtree::{Node, RegTree};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Coefficients at or below this magnitude are dropped (LightGBM
/// `kZeroThreshold`, `1e-35f`).
const ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;

/// The per-leaf linear models of one tree, indexed by node id.
///
/// Leaf `n` predicts `intercept(n) + Σ coeff·x[feature]` over its
/// [`terms`](Self::terms), or the node's constant `leaf_value` when any of
/// those features is missing. Internal nodes hold no terms.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LinearLeaves {
    /// Node `n`'s terms are `features[offsets[n]..offsets[n + 1]]` (and the
    /// matching `coeffs`); length `num_nodes + 1`.
    offsets: Vec<u32>,
    /// Per-node intercept (`0` for internal nodes).
    intercepts: Vec<f64>,
    features: Vec<u32>,
    coeffs: Vec<f64>,
}

impl LinearLeaves {
    /// The intercept of leaf `node`.
    #[inline]
    pub fn intercept(&self, node: usize) -> f64 {
        self.intercepts[node]
    }

    /// The `(features, slopes)` of leaf `node`, features ascending.
    #[inline]
    pub fn terms(&self, node: usize) -> (&[u32], &[f64]) {
        let range = self.offsets[node] as usize..self.offsets[node + 1] as usize;
        (&self.features[range.clone()], &self.coeffs[range])
    }

    /// Output of leaf `node` for a row read through `get` (`None` = missing):
    /// the linear model, or `constant` when one of its features is missing.
    #[inline]
    pub(crate) fn predict(
        &self,
        node: usize,
        constant: f32,
        get: impl Fn(u32) -> Option<f32>,
    ) -> f32 {
        let (features, coeffs) = self.terms(node);
        let mut out = self.intercepts[node];
        for (&f, &c) in features.iter().zip(coeffs) {
            match get(f) {
                Some(x) => out += c * f64::from(x),
                None => return constant,
            }
        }
        out as f32
    }

    /// Multiply every intercept and slope by `factor` (shrinkage).
    pub(crate) fn scale(&mut self, factor: f64) {
        for v in self.intercepts.iter_mut().chain(&mut self.coeffs) {
            *v *= factor;
        }
    }

    /// The stored arrays `(offsets, intercepts, features, coeffs)`, as
    /// [`LinearLeaves::from_parts`] takes them.
    pub(crate) fn parts(&self) -> (&[u32], &[f64], &[u32], &[f64]) {
        (
            &self.offsets,
            &self.intercepts,
            &self.features,
            &self.coeffs,
        )
    }

    /// Assemble leaf models from their stored arrays; the owning tree checks
    /// them with [`LinearLeaves::is_valid`].
    pub(crate) fn from_parts(
        offsets: Vec<u32>,
        intercepts: Vec<f64>,
        features: Vec<u32>,
        coeffs: Vec<f64>,
    ) -> Self {
        LinearLeaves {
            offsets,
            intercepts,
            features,
            coeffs,
        }
    }

    /// Structural validity against the owning tree's nodes.
    pub(crate) fn is_valid(&self, nodes: &[Node], n_features: usize) -> bool {
        let n = nodes.len();
        self.offsets.len() == n + 1
            && self.intercepts.len() == n
            && self.offsets.first() == Some(&0)
            && self
                .offsets
                .last()
                .is_some_and(|&end| end as usize == self.features.len())
            && self.features.len() == self.coeffs.len()
            && self
                .offsets
                .windows(2)
                .zip(nodes)
                .all(|(w, node)| w[0] <= w[1] && (node.is_leaf() || w[0] == w[1]))
            && self.features.iter().all(|&f| (f as usize) < n_features)
            && self
                .intercepts
                .iter()
                .chain(&self.coeffs)
                .all(|v| v.is_finite())
    }
}

/// Fit a linear model in every leaf of `tree` from the training rows `rows`
/// of `data` and their gradients (`gpair`, indexed by row), with slope
/// penalty `lambda`, before shrinkage. Single-leaf trees are left constant.
pub(crate) fn fit_linear_leaves(
    tree: &mut RegTree,
    data: &DMatrix,
    gpair: &[GradPair],
    rows: &[u32],
    lambda: f64,
) {
    let nodes = tree.nodes();
    if nodes.len() == 1 {
        return;
    }
    let features = path_features(nodes, data.feature_types());
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); nodes.len()];
    let leaves: Vec<u32> = rows
        .par_iter()
        .with_min_len(4096)
        .map(|&r| tree.leaf_id_with(|f| data.get(r as usize, f as usize)) as u32)
        .collect();
    for (&r, &leaf) in rows.iter().zip(&leaves) {
        members[leaf as usize].push(r);
    }
    let models: Vec<(f64, Vec<(u32, f64)>)> = (0..nodes.len())
        .into_par_iter()
        .map(|id| {
            if !nodes[id].is_leaf() {
                return (0.0, Vec::new());
            }
            fit_leaf(&features[id], &members[id], data, gpair, lambda)
                .unwrap_or((f64::from(nodes[id].leaf_value), Vec::new()))
        })
        .collect();

    let mut linear = LinearLeaves {
        offsets: Vec::with_capacity(nodes.len() + 1),
        intercepts: Vec::with_capacity(nodes.len()),
        features: Vec::new(),
        coeffs: Vec::new(),
    };
    linear.offsets.push(0);
    for (intercept, terms) in models {
        linear.intercepts.push(intercept);
        for (f, c) in terms {
            linear.features.push(f);
            linear.coeffs.push(c);
        }
        linear.offsets.push(linear.features.len() as u32);
    }
    tree.set_linear_leaves(linear);
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
/// intercept and the non-negligible `(feature, slope)` terms.
fn fit_leaf(
    features: &[u32],
    rows: &[u32],
    data: &DMatrix,
    gpair: &[GradPair],
    lambda: f64,
) -> Option<(f64, Vec<(u32, f64)>)> {
    let p = features.len();
    let dim = p + 1;
    // Upper triangle of XᵀHX (row-major) and Xᵀg, accumulated in row order in
    // f64 from f32 inputs as LightGBM does.
    let mut xthx = vec![0.0f64; dim * (dim + 1) / 2];
    let mut xtg = vec![0.0f64; dim];
    let mut x = vec![0.0f32; dim];
    x[p] = 1.0;
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
            a[i * dim + i] += lambda;
        }
    }
    let rhs: Vec<f64> = xtg.iter().map(|v| -v).collect();
    let beta = solve_full_pivot(a, rhs)?;
    let terms = features
        .iter()
        .zip(&beta)
        .filter(|&(_, c)| c.abs() > ZERO_THRESHOLD)
        .map(|(&f, &c)| (f, c))
        .collect();
    Some((beta[p], terms))
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

/// Add `weight(t) · tree_t(row)` into `out[row * k + output(t)]` for every
/// tree index `t` in `range` (of `trees`), in ascending tree order per slot:
/// the prediction path for ensembles that contain linear leaves (the compact
/// forest stores constant leaf values only). `output` maps a tree to the
/// output it feeds (the model's tree layout).
pub(crate) fn accumulate_forest(
    trees: &[RegTree],
    range: std::ops::Range<usize>,
    output: impl Fn(usize) -> usize + Sync,
    data: &DMatrix,
    out: &mut [f32],
    k: usize,
    weight: impl Fn(usize) -> f32 + Sync,
) {
    let row = |(r, out_row): (usize, &mut [f32])| {
        for t in range.clone() {
            out_row[output(t)] += weight(t) * trees[t].predict_row(data, r);
        }
    };
    if data.n_rows() >= 1024 && rayon::current_num_threads() > 1 {
        out.par_chunks_mut(k)
            .with_min_len(256)
            .enumerate()
            .for_each(row);
    } else {
        out.chunks_mut(k).enumerate().for_each(row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        tree.expand(0, 0, 0.0, default_left, -0.5, 1.0, 0.25, 1.0);
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
        fit_linear_leaves(&mut tree, &data, &gpair, &rows, lambda);
        let (sxx, sx, n, sxy, sy) = (30.0 + lambda, 10.0, 4.0, 70.0, 24.0);
        let det = sxx * n - sx * sx;
        let slope = (n * sxy - sx * sy) / det;
        let intercept = (sxx * sy - sx * sxy) / det;
        let linear = tree.linear_leaves().unwrap();
        let (features, coeffs) = linear.terms(2);
        assert_eq!(features, &[0]);
        assert!((coeffs[0] - slope).abs() < 1e-12, "{coeffs:?} vs {slope}");
        assert!((linear.intercept(2) - intercept).abs() < 1e-12);
        assert!(linear.terms(1).0.is_empty());
        assert_eq!(tree.predict_row(&data, 0), -0.5);
        // Without a penalty the fit recovers the line.
        let mut exact = stump(true);
        fit_linear_leaves(&mut exact, &data, &gpair, &rows, 0.0);
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
        fit_linear_leaves(&mut tree, &data, &gpair, &[0, 1, 2, 3, 4, 5], 0.0);
        assert_eq!(tree.predict_row(&data, 3), 0.25);
        for r in [1, 2, 4, 5] {
            let want = 2.0 * x[r] + 1.0;
            assert!((tree.predict_row(&data, r) - want).abs() < 1e-5, "row {r}");
        }
    }

    #[test]
    fn categorical_path_features_route_but_stay_out_of_the_model() {
        // Root splits categorical feature 0; each child splits numeric 1.
        let mut tree = RegTree::with_root(8.0);
        let (l, r) = tree.expand_categorical(0, 0, &[1], false, 0.0, 4.0, 0.0, 4.0);
        tree.expand(l, 1, 0.5, true, 0.0, 2.0, 0.0, 2.0);
        tree.expand(r, 1, 0.5, true, 0.0, 2.0, 0.0, 2.0);
        let types = [FeatureType::Categorical, FeatureType::Numerical];
        let paths = path_features(tree.nodes(), &types);
        for (id, node) in tree.nodes().iter().enumerate() {
            let want: &[u32] = if node.is_leaf() { &[1] } else { &[] };
            assert_eq!(paths[id], want, "node {id}");
        }
    }
}
