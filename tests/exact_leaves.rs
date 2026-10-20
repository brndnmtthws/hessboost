//! Exact quantile leaves (`leaf_estimation_method = exact`): every leaf of
//! every tree is `eta` times the residual quantile of the training rows
//! prediction routes to it, checked against a brute-force reference with
//! missing values, row and column sampling, sample weights, several levels,
//! vector leaves, forests, label matrices, and `max_delta_step`; models do
//! not depend on the thread count, and the combinations that would set the
//! leaves otherwise are refused.

use hessboost::config::{
    BoosterKind, Dart, GrowPolicy, LeafEstimation, LinearTree, MaxDeltaStep, Monotone,
    MultiStrategy, TrainingParamsBuilder,
};
use hessboost::objective::{Objective, Quantiles};
use hessboost::prelude::*;
use hessboost::tree::RegTree;

mod common;
use common::{invalid_param, lcg, with_threads};

const ROWS: usize = 3000;
const COLS: usize = 4;
const ETA: f32 = 0.3;

/// `ROWS × COLS` features, a quarter missing, and a heavy-tailed label that
/// depends on the first two.
fn features() -> (Vec<f32>, Vec<f32>) {
    let mut next = lcg(17);
    let x: Vec<f32> = (0..ROWS * COLS)
        .map(|_| {
            let v = next();
            if next() < 0.25 { f32::NAN } else { v }
        })
        .collect();
    let y = (0..ROWS)
        .map(|r| {
            let row = &x[r * COLS..][..COLS];
            // `max`/`min` read a missing value as their other operand.
            let signal = 3.0 * row[0].max(0.0) - 2.0 * row[1].min(0.5);
            // A Cauchy-like tail: the quantiles and the mean differ.
            signal + (next() - 0.5) / (next() + 0.05)
        })
        .collect();
    (x, y)
}

fn params() -> TrainingParamsBuilder {
    TrainingParams::builder()
        .max_depth(3)
        .eta(f64::from(ETA))
        .subsample(0.6)
        .colsample_bytree(0.75)
        .leaf_estimation(LeafEstimation::Exact)
        .seed(5)
}

/// XGBoost `common::Quantile` of `values` (any order): interpolation on the
/// `(n + 1)α` grid, clamped to the extremes.
fn interpolated(mut values: Vec<f32>, alpha: f32) -> f32 {
    values.sort_by(f32::total_cmp);
    let n = values.len() as f64;
    let alpha = f64::from(alpha);
    if alpha <= 1.0 / (n + 1.0) {
        return values[0];
    }
    if alpha >= n / (n + 1.0) {
        return values[values.len() - 1];
    }
    let x = alpha * (n + 1.0);
    let k = x.floor() - 1.0;
    let d = (x - 1.0) - k;
    let (v0, v1) = (values[k as usize], values[k as usize + 1]);
    (f64::from(v0) + d * f64::from(v1 - v0)) as f32
}

/// XGBoost `common::WeightedQuantile` of `(value, weight)` pairs (rows in
/// ascending order): the first value, in a stable sort, whose `f32`
/// cumulative weight reaches `α` of the total; `0` without weight.
fn weighted(mut pairs: Vec<(f32, f32)>, alpha: f32) -> f32 {
    if pairs.iter().all(|&(_, w)| w == 0.0) {
        return 0.0;
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut cdf = Vec::with_capacity(pairs.len());
    let mut acc = 0.0f32;
    for (i, &(_, w)) in pairs.iter().enumerate() {
        acc = if i == 0 { w } else { acc + w };
        cdf.push(acc);
    }
    let thresh = (f64::from(acc) * f64::from(alpha)) as f32;
    let at = cdf
        .iter()
        .position(|&c| c >= thresh)
        .unwrap_or(pairs.len() - 1);
    pairs[at].0
}

/// What the exact leaves of one model read: the matrix and its raw
/// features, the sample weights, every output's level and label column, and
/// the `max_delta_step` bound.
struct Reference<'a> {
    data: &'a DMatrix,
    x: &'a [f32],
    weights: Option<&'a [f32]>,
    levels: &'a [f32],
    columns: &'a [usize],
    bound: Option<f32>,
}

impl Reference<'_> {
    /// Every leaf of `model`, tree by tree, holds `eta / num_parallel_tree`
    /// times the (bound-clamped) quantile at `levels[output]` of
    /// `label[column] - margin[output]` over the rows routed to it, at the
    /// margins of the iterations before its tree's: every tree of a forest
    /// fits the same margins.
    fn check(&self, model: &BoostedModel) {
        let n_out = model.n_outputs();
        let label_cols = self.data.n_targets();
        let labels = self.data.labels().unwrap();
        let per_iteration = model.trees_per_iteration();
        let parallel = model.num_parallel_tree();
        let eta = ETA / parallel as f32;
        for (t, tree) in model.trees().iter().enumerate() {
            let margins = model
                .predict_margin(self.data, ..t / per_iteration)
                .unwrap()
                .into_vec();
            let value = |rows: &[usize], output: usize| {
                let residual = |r: usize| {
                    labels[r * label_cols + self.columns[output]] - margins[r * n_out + output]
                };
                let alpha = self.levels[output];
                let q = match self.weights {
                    None if rows.is_empty() => 0.0,
                    None => interpolated(rows.iter().map(|&r| residual(r)).collect(), alpha),
                    Some(w) => weighted(rows.iter().map(|&r| (residual(r), w[r])).collect(), alpha),
                };
                self.bound.map_or(q, |b| q.clamp(-b, b)) * eta
            };
            for (leaf, rows) in leaf_rows(tree, self.x).iter().enumerate() {
                if !tree.node(leaf).is_leaf() {
                    continue;
                }
                if tree.is_vector_leaf() {
                    for (output, &got) in tree.leaf_vector(leaf).iter().enumerate() {
                        let want = value(rows, output);
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "tree {t} leaf {leaf} output {output}"
                        );
                    }
                } else {
                    // Scalar trees are grouped by output, a forest per output.
                    let want = value(rows, t % per_iteration / parallel);
                    let got = tree.node(leaf).leaf_value;
                    assert_eq!(got.to_bits(), want.to_bits(), "tree {t} leaf {leaf}");
                }
            }
        }
    }
}

/// Every row of `x` (`COLS` features) routed by raw value, missing values
/// taking the default branch: the rows reaching each node.
fn leaf_rows(tree: &RegTree, x: &[f32]) -> Vec<Vec<usize>> {
    let mut rows = vec![Vec::new(); tree.num_nodes()];
    for (r, row) in x.as_chunks::<COLS>().0.iter().enumerate() {
        rows[tree.leaf_id_dense(row, f32::NAN)].push(r);
    }
    rows
}

fn quantiles(levels: &[f32]) -> Objective {
    Objective::Quantile(Quantiles::new(levels.iter().map(|&a| f64::from(a))).unwrap())
}

#[test]
fn scalar_leaves_are_the_residual_quantiles_of_every_row() {
    let (x, y) = features();
    let data = DMatrix::from_dense(&x, ROWS, COLS)
        .unwrap()
        .with_labels(&y)
        .unwrap();
    let levels = [0.1, 0.5, 0.9];
    // Every builder hands its finished structure over before `eta`.
    for (method, policy) in [
        (TreeMethod::Hist, GrowPolicy::DepthWise),
        (TreeMethod::Hist, GrowPolicy::Symmetric),
        (TreeMethod::Approx, GrowPolicy::DepthWise),
        (TreeMethod::Exact, GrowPolicy::DepthWise),
    ] {
        let p = params()
            .objective(quantiles(&levels))
            .tree_method(method)
            .grow_policy(policy)
            .build()
            .unwrap();
        let model = train(&p, &data, 4).unwrap();
        assert_eq!(model.num_trees(), 12);
        Reference {
            data: &data,
            x: &x,
            weights: None,
            levels: &levels,
            columns: &[0, 0, 0],
            bound: None,
        }
        .check(&model);
    }
}

#[test]
fn weighted_leaves_are_the_weighted_residual_quantiles() {
    let (x, y) = features();
    let mut next = lcg(29);
    // Some rows weigh nothing, as recency or inclusion weights can.
    let w: Vec<f32> = (0..ROWS)
        .map(|_| {
            if next() < 0.1 {
                0.0
            } else {
                0.2 + 3.0 * next()
            }
        })
        .collect();
    let data = DMatrix::from_dense(&x, ROWS, COLS)
        .unwrap()
        .with_labels(&y)
        .unwrap()
        .with_weights(&w)
        .unwrap();
    let levels = [0.2, 0.85];
    let model = train(
        &params().objective(quantiles(&levels)).build().unwrap(),
        &data,
        3,
    )
    .unwrap();
    Reference {
        data: &data,
        x: &x,
        weights: Some(&w),
        levels: &levels,
        columns: &[0, 0],
        bound: None,
    }
    .check(&model);
}

#[test]
fn vector_leaves_hold_every_levels_quantile() {
    let (x, y) = features();
    let data = DMatrix::from_dense(&x, ROWS, COLS)
        .unwrap()
        .with_labels(&y)
        .unwrap();
    let levels = [0.1, 0.9];
    let p = params()
        .objective(quantiles(&levels))
        .multi_strategy(MultiStrategy::MultiOutputTree)
        .build()
        .unwrap();
    let model = train(&p, &data, 4).unwrap();
    assert!(model.has_vector_leaves());
    Reference {
        data: &data,
        x: &x,
        weights: None,
        levels: &levels,
        columns: &[0, 0],
        bound: None,
    }
    .check(&model);
}

#[test]
fn every_tree_of_a_forest_fits_the_iterations_starting_margins() {
    let (x, y) = features();
    let data = DMatrix::from_dense(&x, ROWS, COLS)
        .unwrap()
        .with_labels(&y)
        .unwrap();
    let levels = [0.1, 0.9];
    for strategy in [
        MultiStrategy::OneOutputPerTree,
        MultiStrategy::MultiOutputTree,
    ] {
        let p = params()
            .objective(quantiles(&levels))
            .multi_strategy(strategy)
            .num_parallel_tree(3)
            .build()
            .unwrap();
        let model = train(&p, &data, 3).unwrap();
        assert_eq!(model.num_parallel_tree(), 3);
        Reference {
            data: &data,
            x: &x,
            weights: None,
            levels: &levels,
            columns: &[0, 0],
            bound: None,
        }
        .check(&model);
    }
}

#[test]
fn absolute_error_leaves_are_each_columns_median() {
    let (x, y) = features();
    let mut next = lcg(41);
    let targets: Vec<f32> = y.iter().flat_map(|&v| [v, -0.5 * v + next()]).collect();
    let data = DMatrix::from_dense(&x, ROWS, COLS)
        .unwrap()
        .with_label_matrix(&targets, 2)
        .unwrap();
    let p = params()
        .objective(Objective::AbsoluteError)
        .build()
        .unwrap();
    let model = train(&p, &data, 3).unwrap();
    Reference {
        data: &data,
        x: &x,
        weights: None,
        levels: &[0.5, 0.5],
        columns: &[0, 1],
        bound: None,
    }
    .check(&model);
}

#[test]
fn max_delta_step_bounds_the_quantile_before_eta() {
    let (x, y) = features();
    let data = DMatrix::from_dense(&x, ROWS, COLS)
        .unwrap()
        .with_labels(&y)
        .unwrap();
    let p = params()
        .objective(quantiles(&[0.05]))
        .max_delta_step(MaxDeltaStep::Bounded(0.4))
        .build()
        .unwrap();
    let model = train(&p, &data, 3).unwrap();
    Reference {
        data: &data,
        x: &x,
        weights: None,
        levels: &[0.05],
        columns: &[0],
        bound: Some(0.4),
    }
    .check(&model);
    // The bound binds: some leaf sits on it.
    let bound = 0.4 * ETA;
    let at_bound = model
        .trees()
        .iter()
        .flat_map(RegTree::nodes)
        .any(|n| n.is_leaf() && n.leaf_value.abs() == bound);
    assert!(at_bound);
}

#[test]
fn exact_leaves_do_not_depend_on_the_thread_count() {
    let (x, y) = features();
    let data = DMatrix::from_dense(&x, ROWS, COLS)
        .unwrap()
        .with_labels(&y)
        .unwrap();
    let p = params().objective(quantiles(&[0.1, 0.9])).build().unwrap();
    let fit = |threads| {
        with_threads(threads, || {
            train(&p, &data, 5)
                .unwrap()
                .encode(ModelFormat::Binary)
                .unwrap()
        })
    };
    assert_eq!(fit(1), fit(4));
}

#[test]
fn exact_leaves_need_a_closed_form_loss_and_trees_whose_leaves_they_set() {
    let median = || params().objective(quantiles(&[0.5]));
    let unsampled = TrainingParams::builder()
        .objective(quantiles(&[0.5]))
        .leaf_estimation(LeafEstimation::Exact);
    for refused in [
        params().objective(Objective::default()),
        median().booster(BoosterKind::Dart(Dart::default())),
        unsampled.booster(BoosterKind::GbLinear),
        median().monotone_constraints(vec![Monotone::Increasing]),
        median().linear_tree(LinearTree::default()),
        median().path_smooth(1.0),
        median().posterior_sampling(true),
    ] {
        assert_eq!(invalid_param(refused.build()), "leaf_estimation_method");
    }
}
