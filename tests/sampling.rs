//! Training-level contracts of row and column sampling:
//! `sampling_method=gradient_based` (XGBoost's CPU MVS sampler) and
//! feature-weighted column sampling (`DMatrix::with_feature_weights`).

use hessboost::config::{BoosterKind, SamplingMethod, TrainingParamsBuilder, TreeMethod};
use hessboost::prelude::*;
use hessboost::tree::RegTree;

mod common;
use common::{invalid_param, labeled_dense, lcg};

/// Deterministic `n × f` regression data whose label depends on every
/// feature, with a few large-residual rows so gradient magnitudes vary.
fn dataset(n: usize, f: usize) -> DMatrix {
    let mut next = lcg(0x1234_5678);
    let mut x = Vec::with_capacity(n * f);
    let mut y = Vec::with_capacity(n);
    for row in 0..n {
        let mut target = 0.0;
        for col in 0..f {
            let v = next();
            target += v * (col as f32 + 1.0);
            x.push(v);
        }
        if row % 37 == 0 {
            target += 10.0;
        }
        y.push(target);
    }
    labeled_dense(&x, f, &y)
}

fn mvs_params(seed: u64) -> TrainingParams {
    TrainingParams::builder()
        .tree_method(TreeMethod::Hist)
        .sampling_method(SamplingMethod::GradientBased)
        .subsample(0.3)
        .max_depth(4)
        .seed(seed)
        .build()
        .unwrap()
}

fn predictions(params: &TrainingParams, data: &DMatrix, rounds: usize) -> Vec<f32> {
    train(params, data, rounds).unwrap().predict(data).unwrap()
}

/// Every feature any split of `tree` uses on some root-to-leaf path, one set
/// per path.
fn path_features(tree: &RegTree) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    let mut stack = vec![(0usize, Vec::new())];
    while let Some((nid, mut path)) = stack.pop() {
        let node = tree.node(nid);
        if node.is_leaf() {
            out.push(path);
            continue;
        }
        path.push(node.split_feature);
        stack.push((node.left as usize, path.clone()));
        stack.push((node.right as usize, path));
    }
    out
}

fn split_features(model: &BoostedModel) -> Vec<u32> {
    let mut all: Vec<u32> = model
        .trees()
        .iter()
        .flat_map(|t| path_features(t).into_iter().flatten())
        .collect();
    all.sort_unstable();
    all.dedup();
    all
}

#[test]
fn gradient_based_sampling_is_seeded_and_thread_count_independent() {
    let data = dataset(6000, 5);
    let run = |threads: usize, seed: u64| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| predictions(&mvs_params(seed), &data, 8))
    };
    let serial = run(1, 3);
    assert_eq!(serial, run(4, 3), "serial and parallel training differ");
    assert_ne!(serial, run(1, 4), "the seed drives the sample");
    let full = predictions(
        &TrainingParams::builder().max_depth(4).build().unwrap(),
        &data,
        8,
    );
    assert_ne!(serial, full, "gradient-based sampling had no effect");
}

/// XGBoost bypasses the sampler when `trunc(n * subsample) == n`, so a
/// gradient-based model at `subsample = 1` equals the default model.
#[test]
fn gradient_based_at_full_subsample_is_the_unsampled_model() {
    let data = dataset(500, 3);
    let base = || TrainingParams::builder().max_depth(3).colsample_bynode(0.7);
    let uniform = predictions(&base().build().unwrap(), &data, 5);
    let mvs = predictions(
        &base()
            .sampling_method(SamplingMethod::GradientBased)
            .build()
            .unwrap(),
        &data,
        5,
    );
    assert_eq!(uniform, mvs);
}

/// Only the histogram-based updaters implement gradient-based sampling;
/// `exact` refuses it once rows are actually subsampled, like XGBoost's
/// `colmaker`.
#[test]
fn gradient_based_sampling_tree_method_support() {
    let data = dataset(300, 3);
    let with = |method: TreeMethod, subsample: f64| {
        TrainingParams::builder()
            .tree_method(method)
            .sampling_method(SamplingMethod::GradientBased)
            .subsample(subsample)
            .build()
            .unwrap()
    };
    assert_eq!(
        invalid_param(train(&with(TreeMethod::Exact, 0.5), &data, 2)),
        "sampling_method"
    );
    assert!(train(&with(TreeMethod::Exact, 1.0), &data, 2).is_ok());
    for method in [TreeMethod::Hist, TreeMethod::Approx, TreeMethod::Auto] {
        let model = train(&with(method, 0.4), &data, 3).unwrap();
        assert_eq!(model.num_trees(), 3, "{method:?}");
    }
    // DART grows its trees through the same per-tree sampling.
    let dart = TrainingParams::builder()
        .booster(BoosterKind::Dart)
        .sampling_method(SamplingMethod::GradientBased)
        .subsample(0.4)
        .build()
        .unwrap();
    assert!(train(&dart, &data, 3).is_ok());
}

/// Zero weights are epsilon weights (floored at 1e-6, as in XGBoost): against
/// weights far above the floor they practically never win, so on these fixed
/// seeds a stage that keeps as many features as have positive weight never
/// splits on a zero-weight feature, for every tree method and sampling stage.
#[test]
fn zero_weight_features_are_practically_never_split_on() {
    let data = dataset(800, 4)
        .with_feature_weights(&[0.0, 3.0, 0.0, 1.0])
        .unwrap();
    let base = |method: TreeMethod| {
        TrainingParams::builder()
            .tree_method(method)
            .max_depth(4)
            .seed(11)
    };
    for method in [TreeMethod::Hist, TreeMethod::Approx, TreeMethod::Exact] {
        // 4 * 0.5 = 2 features per tree, level, or node.
        for params in [
            base(method).colsample_bytree(0.5).build().unwrap(),
            base(method).colsample_bylevel(0.5).build().unwrap(),
            base(method).colsample_bynode(0.5).build().unwrap(),
        ] {
            let model = train(&params, &data, 10).unwrap();
            assert_eq!(split_features(&model), vec![1, 3], "{method:?}");
        }
    }
    // Without weights the same configuration uses every feature.
    let unweighted = dataset(800, 4);
    let model = train(
        &base(TreeMethod::Hist)
            .colsample_bynode(0.5)
            .build()
            .unwrap(),
        &unweighted,
        10,
    )
    .unwrap();
    assert_eq!(split_features(&model), vec![0, 1, 2, 3]);
}

/// Heavier features are chosen more often: with one feature per tree, the
/// share of trees using each feature follows its weight.
#[test]
fn tree_feature_shares_follow_weights() {
    let weights = [1.0f32, 2.0, 5.0];
    let data = dataset(400, 3).with_feature_weights(&weights).unwrap();
    let params = TrainingParams::builder()
        .colsample_bytree(0.34) // 3 * 0.34 = 1.02 -> 1 feature per tree
        .max_depth(2)
        .eta(0.01)
        .seed(5)
        .build()
        .unwrap();
    let model = train(&params, &data, 800).unwrap();
    let mut counts = [0usize; 3];
    for tree in model.trees() {
        let used: Vec<u32> = path_features(tree).into_iter().flatten().collect();
        assert!(
            used.windows(2).all(|w| w[0] == w[1]),
            "one feature per tree"
        );
        if let Some(&f) = used.first() {
            counts[f as usize] += 1;
        }
    }
    let total: usize = counts.iter().sum();
    assert!(total > 700, "trees must split");
    for (f, &count) in counts.iter().enumerate() {
        let share = count as f64 / total as f64;
        let expected = f64::from(weights[f]) / 8.0;
        assert!(
            (share - expected).abs() < 0.06,
            "feature {f}: {share} vs {expected}"
        );
    }
}

/// Interaction constraints filter the weighted sample: a feature the sampler
/// did not draw is never reintroduced, and each path stays in one
/// constraint group.
#[test]
fn interaction_constraints_filter_the_weighted_sample() {
    let data = dataset(1500, 4)
        .with_feature_weights(&[1.0, 1.0, 0.0, 0.0])
        .unwrap();
    for method in [TreeMethod::Hist, TreeMethod::Exact] {
        let params = TrainingParams::builder()
            .tree_method(method)
            .colsample_bynode(0.5)
            .interaction_constraints(vec![vec![0, 2], vec![1, 3]])
            .max_depth(4)
            .seed(2)
            .build()
            .unwrap();
        let model = train(&params, &data, 10).unwrap();
        let mut used_any = [false; 4];
        for tree in model.trees() {
            for path in path_features(tree) {
                // Sampled {0, 1} ∩ one group = a single feature per path.
                assert!(
                    path.windows(2).all(|w| w[0] == w[1]),
                    "{method:?}: {path:?}"
                );
                for f in path {
                    used_any[f as usize] = true;
                }
            }
        }
        assert_eq!(used_any, [true, true, false, false], "{method:?}");
    }
}

#[test]
fn weighted_column_sampling_is_seeded() {
    let data = dataset(600, 6)
        .with_feature_weights(&[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        .unwrap();
    let params = |seed: u64| {
        TrainingParams::builder()
            .colsample_bytree(0.8)
            .colsample_bylevel(0.8)
            .colsample_bynode(0.6)
            .max_depth(4)
            .seed(seed)
            .build()
            .unwrap()
    };
    let a = predictions(&params(1), &data, 6);
    assert_eq!(a, predictions(&params(1), &data, 6));
    assert_ne!(a, predictions(&params(2), &data, 6));
}

/// Eight single-feature rows (`x` per `feature`) with labels
/// `[0, 0, 0, 0, 1, 1, 1, 1]`.
fn step_rows(feature: impl Fn(usize) -> f32) -> DMatrix {
    let x: Vec<f32> = (0..8).map(feature).collect();
    let y: Vec<f32> = (0..8).map(|i| if i < 4 { 0.0 } else { 1.0 }).collect();
    labeled_dense(&x, 1, &y)
}

/// Unregularized unit-rate squared-error parameters from a zero margin.
fn plain(method: TreeMethod, booster: BoosterKind) -> TrainingParamsBuilder {
    TrainingParams::builder()
        .tree_method(method)
        .booster(booster)
        .base_score(0.0)
        .eta(1.0)
        .lambda(0.0)
        .min_child_weight(0.0)
}

/// XGBoost's approx updater samples once per output forest
/// (`GlobalApproxUpdater::Update`), so on a constant feature every parallel
/// tree of an iteration holds the same leaf; the hist updater samples each
/// tree, so its forests' leaves differ.
#[test]
fn approx_parallel_trees_share_one_row_sample() {
    let data = step_rows(|_| 0.0);
    for booster in [BoosterKind::GbTree, BoosterKind::Dart] {
        for (sampling, subsample) in [
            (SamplingMethod::GradientBased, 0.25),
            (SamplingMethod::Uniform, 0.5),
        ] {
            let leaves = |method: TreeMethod| {
                let params = plain(method, booster)
                    .sampling_method(sampling)
                    .subsample(subsample)
                    .num_parallel_tree(4)
                    .seed(13)
                    .build()
                    .unwrap();
                let model = train(&params, &data, 3).unwrap();
                let leaves: Vec<f32> = model.trees().iter().map(|t| t.node(0).leaf_value).collect();
                assert_eq!(leaves.len(), 12);
                leaves
            };
            let differs = |leaves: &[f32]| {
                leaves
                    .chunks(4)
                    .any(|forest| forest.iter().any(|&v| v != forest[0]))
            };
            let approx = leaves(TreeMethod::Approx);
            assert!(!differs(&approx), "{booster:?} {sampling:?}: {approx:?}");
            let hist = leaves(TreeMethod::Hist);
            assert!(differs(&hist), "{booster:?} {sampling:?}: {hist:?}");
        }
    }
}

/// Continuing `approx` training under gradient-based sampling reproduces the
/// uninterrupted run: the constant-Hessian cuts an uninterrupted run caches
/// from its first tree's sample are rebuilt on resume, not taken from the
/// resumed round's sample.
#[test]
fn approx_gradient_sampling_continuation_matches_uninterrupted_training() {
    let step = step_rows(|i| i as f32);
    let wide = dataset(400, 3);
    for booster in [BoosterKind::GbTree, BoosterKind::Dart] {
        for (data, depth, split) in [(&step, 1, [1, 1]), (&wide, 3, [2, 3])] {
            let params = plain(TreeMethod::Approx, booster)
                .sampling_method(SamplingMethod::GradientBased)
                .subsample(0.25)
                .max_depth(depth)
                .seed(13)
                .build()
                .unwrap();
            let full = train(&params, data, split[0] + split[1]).unwrap();
            let head = train(&params, data, split[0]).unwrap();
            let resumed = Trainer::new(&params, data, split[1])
                .init_model(&head)
                .train()
                .unwrap()
                .model;
            assert_eq!(
                full.predict(data).unwrap(),
                resumed.predict(data).unwrap(),
                "{booster:?} depth {depth}"
            );
        }
    }
}
