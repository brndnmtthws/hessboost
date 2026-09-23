//! Training-level contracts of row and column sampling:
//! `sampling_method=gradient_based` (XGBoost's CPU MVS sampler) and
//! feature-weighted column sampling (`DMatrix::with_feature_weights`).

use hessboost::prelude::*;
use hessboost::tree::RegTree;

/// Deterministic `n × f` regression data whose label depends on every
/// feature, with a few large-residual rows so gradient magnitudes vary.
fn dataset(n: usize, f: usize) -> DMatrix {
    let mut state = 0x1234_5678_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
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
    DMatrix::from_dense(&x, n, f)
        .unwrap()
        .with_labels(&y)
        .unwrap()
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
    match train(&with(TreeMethod::Exact, 0.5), &data, 2) {
        Err(HessboostError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "sampling_method");
            assert!(reason.contains("hist"), "{reason}");
        }
        other => panic!("expected a sampling_method error, got {other:?}"),
    }
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

/// A stage that keeps as many features as have positive weight never picks a
/// zero-weight feature, for every tree method and sampling stage.
#[test]
fn zero_weight_features_are_never_split_on() {
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
