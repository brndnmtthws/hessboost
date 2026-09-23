//! End-to-end behavior of the opt-in LightGBM tree options: `extra_trees`,
//! `path_smooth`, and `linear_tree` / `linear_lambda`.

use hessboost::config::TrainingParamsBuilder;
use hessboost::prelude::*;

/// Deterministic pseudo-random values in `[0, 1)`.
fn uniform(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

/// Two features; the target is piecewise linear: a jump at `x0 = 0.5` plus
/// slopes in both features, with a little noise.
fn piecewise_linear(n: usize, seed: u64) -> DMatrix {
    let x = uniform(2 * n, seed);
    let noise = uniform(n, seed ^ 0xABCD);
    let y: Vec<f32> = x
        .as_chunks::<2>()
        .0
        .iter()
        .zip(&noise)
        .map(|(row, e)| {
            let jump = if row[0] < 0.5 { 0.0 } else { 3.0 };
            jump + 2.0 * row[0] - 1.5 * row[1] + 0.05 * (e - 0.5)
        })
        .collect();
    DMatrix::from_dense(&x, n, 2)
        .unwrap()
        .with_labels(&y)
        .unwrap()
}

fn rmse(model: &BoostedModel, data: &DMatrix) -> f64 {
    let preds = model.predict(data).unwrap();
    let labels = data.labels().unwrap();
    let sse: f64 = preds
        .iter()
        .zip(labels)
        .map(|(p, y)| f64::from(p - y).powi(2))
        .sum();
    (sse / preds.len() as f64).sqrt()
}

fn base() -> TrainingParamsBuilder {
    TrainingParams::builder().max_depth(3).eta(0.3)
}

#[test]
fn disabled_options_keep_the_default_model_bit_for_bit() {
    let data = piecewise_linear(500, 1);
    let default = train(&base().build().unwrap(), &data, 10).unwrap();
    let explicit = base()
        .extra_trees(false)
        .extra_seed(99)
        .path_smooth(0.0)
        .linear_tree(false)
        .linear_lambda(3.0)
        .build()
        .unwrap();
    let off = train(&explicit, &data, 10).unwrap();
    assert_eq!(default.to_bytes().unwrap(), off.to_bytes().unwrap());
}

#[test]
fn every_option_changes_the_model_and_is_reproducible() {
    let data = piecewise_linear(500, 2);
    let default = train(&base().build().unwrap(), &data, 10)
        .unwrap()
        .to_bytes()
        .unwrap();
    for params in [
        base().extra_trees(true).build().unwrap(),
        base().path_smooth(5.0).build().unwrap(),
        base().linear_tree(true).build().unwrap(),
    ] {
        let a = train(&params, &data, 10).unwrap().to_bytes().unwrap();
        assert_eq!(a, train(&params, &data, 10).unwrap().to_bytes().unwrap());
        assert_ne!(a, default);
    }
}

#[test]
fn options_grow_the_same_model_serially_and_in_parallel() {
    // Large enough for parallel frontiers, split searches, and partitions.
    let data = piecewise_linear(40_000, 3);
    let params = base()
        .max_depth(5)
        .extra_trees(true)
        .path_smooth(2.0)
        .linear_tree(true)
        .linear_lambda(0.1)
        .subsample(0.8)
        .build()
        .unwrap();
    let fit = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| train(&params, &data, 4).unwrap().to_bytes().unwrap())
    };
    assert_eq!(fit(1), fit(4));
}

#[test]
fn linear_leaves_fit_piecewise_linear_targets_far_better() {
    let (train_set, test_set) = (piecewise_linear(2000, 4), piecewise_linear(1000, 5));
    let fit = |linear: bool| {
        let params = base()
            .max_depth(2)
            .eta(0.5)
            .linear_tree(linear)
            .build()
            .unwrap();
        train(&params, &train_set, 10).unwrap()
    };
    let (constant, linear) = (fit(false), fit(true));
    let (rc, rl) = (rmse(&constant, &test_set), rmse(&linear, &test_set));
    assert!(rl < 0.4 * rc, "held-out RMSE linear {rl} vs constant {rc}");
    // The first round keeps constant leaves; later rounds fit linear ones.
    assert!(linear.trees()[0].linear_leaves().is_none());
    assert!(
        linear.trees()[1..]
            .iter()
            .all(|t| t.linear_leaves().is_some())
    );
}

#[test]
fn linear_models_round_trip_natively_and_refuse_xgboost_formats_and_shap() {
    let x = uniform(600, 6);
    let mut values = x.clone();
    for v in values.iter_mut().step_by(7) {
        *v = f32::NAN; // exercise the constant fallback on missing values
    }
    let y: Vec<f32> = x
        .as_chunks::<2>()
        .0
        .iter()
        .map(|r| 3.0 * r[0] - r[1])
        .collect();
    let data = DMatrix::from_dense(&values, 300, 2)
        .unwrap()
        .with_labels(&y)
        .unwrap();
    let params = base().linear_tree(true).build().unwrap();
    let model = train(&params, &data, 6).unwrap();
    let before = model.predict(&data).unwrap();

    let from_bytes = BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
    assert_eq!(from_bytes.predict(&data).unwrap(), before);
    let from_json = BoostedModel::from_json(&model.to_json().unwrap()).unwrap();
    assert_eq!(from_json.predict(&data).unwrap(), before);

    assert!(matches!(
        model.to_xgboost_json(),
        Err(HessboostError::ModelFormat(_))
    ));
    assert!(matches!(
        model.to_xgboost_ubjson(),
        Err(HessboostError::ModelFormat(_))
    ));
    for result in [
        model.predict_contribs(&data),
        model.predict_interactions(&data),
    ] {
        assert!(matches!(
            result,
            Err(HessboostError::InvalidParameter {
                name: "linear_tree",
                ..
            })
        ));
    }
}

#[test]
fn incompatible_configurations_are_rejected() {
    let invalid = |builder: TrainingParamsBuilder, name: &str| match builder.build() {
        Err(HessboostError::InvalidParameter { name: got, .. }) => assert_eq!(got, name),
        other => panic!("expected `{name}` to be rejected, got {other:?}"),
    };
    invalid(base().path_smooth(-1.0), "path_smooth");
    invalid(base().linear_lambda(f64::NAN), "linear_lambda");
    invalid(
        base().extra_trees(true).tree_method(TreeMethod::Exact),
        "extra_trees",
    );
    invalid(
        base().path_smooth(1.0).booster(BoosterKind::GbLinear),
        "path_smooth",
    );
    invalid(
        base()
            .linear_tree(true)
            .multi_strategy(MultiStrategy::MultiOutputTree),
        "linear_tree",
    );
    invalid(
        base().linear_tree(true).objective("reg:absoluteerror"),
        "linear_tree",
    );
    // The histogram-based `approx` builder accepts every option.
    base()
        .tree_method(TreeMethod::Approx)
        .extra_trees(true)
        .path_smooth(1.0)
        .linear_tree(true)
        .build()
        .unwrap();
}
