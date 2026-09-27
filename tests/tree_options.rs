//! End-to-end behavior of the opt-in LightGBM tree options: `extra_trees`,
//! `path_smooth`, and `linear_tree` / `linear_lambda`.

use hessboost::config::{
    BoosterKind, ExtraTrees, LinearTree, Monotone, MultiStrategy, TrainingParamsBuilder,
};
use hessboost::data::FeatureType;
use hessboost::prelude::*;

mod common;
use common::{invalid_param, labeled_dense, lcg, rmse};

/// `n` deterministic pseudo-random values in `[0, 1)`.
fn uniform(n: usize, seed: u64) -> Vec<f32> {
    let mut next = lcg(seed);
    (0..n).map(|_| next()).collect()
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
    labeled_dense(&x, 2, &y)
}

fn base() -> TrainingParamsBuilder {
    TrainingParams::builder().max_depth(3).eta(0.3)
}

#[test]
fn disabled_options_keep_the_default_model_bit_for_bit() {
    let data = piecewise_linear(500, 1);
    let default = train(&base().build().unwrap(), &data, 10).unwrap();
    // `extra_seed` and `linear_lambda` live inside their switches, so with
    // the switches off only `path_smooth` can be spelled out.
    let explicit = base().path_smooth(0.0).build().unwrap();
    let off = train(&explicit, &data, 10).unwrap();
    assert_eq!(
        default.encode(ModelFormat::Binary).unwrap(),
        off.encode(ModelFormat::Binary).unwrap()
    );
}

#[test]
fn every_option_changes_the_model_and_is_reproducible() {
    let data = piecewise_linear(500, 2);
    let default = train(&base().build().unwrap(), &data, 10)
        .unwrap()
        .encode(ModelFormat::Binary)
        .unwrap();
    for params in [
        base().extra_trees(ExtraTrees::default()).build().unwrap(),
        base().path_smooth(5.0).build().unwrap(),
        base().linear_tree(LinearTree::default()).build().unwrap(),
    ] {
        let a = train(&params, &data, 10)
            .unwrap()
            .encode(ModelFormat::Binary)
            .unwrap();
        assert_eq!(
            a,
            train(&params, &data, 10)
                .unwrap()
                .encode(ModelFormat::Binary)
                .unwrap()
        );
        assert_ne!(a, default);
    }
}

#[test]
fn options_grow_the_same_model_serially_and_in_parallel() {
    // Large enough for parallel frontiers, split searches, and partitions.
    let data = piecewise_linear(40_000, 3);
    let params = base()
        .max_depth(5)
        .extra_trees(ExtraTrees::default())
        .path_smooth(2.0)
        .linear_tree(LinearTree::new(0.1).unwrap())
        .subsample(0.8)
        .build()
        .unwrap();
    let fit = |threads: usize| {
        common::with_threads(threads, || {
            train(&params, &data, 4)
                .unwrap()
                .encode(ModelFormat::Binary)
                .unwrap()
        })
    };
    assert_eq!(fit(1), fit(4));
}

#[test]
fn linear_leaves_fit_piecewise_linear_targets_far_better() {
    let (train_set, test_set) = (piecewise_linear(2000, 4), piecewise_linear(1000, 5));
    let fit = |linear: bool| {
        let mut builder = base().max_depth(2).eta(0.5);
        if linear {
            builder = builder.linear_tree(LinearTree::default());
        }
        let params = builder.build().unwrap();
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
    let data = labeled_dense(&values, 2, &y);
    let params = base().linear_tree(LinearTree::default()).build().unwrap();
    let model = train(&params, &data, 6).unwrap();
    let before = model.predict(&data, Iterations::Best).unwrap();

    let from_bytes = BoostedModel::decode(
        model.encode(ModelFormat::Binary).unwrap(),
        ModelFormat::Binary,
    )
    .unwrap();
    assert_eq!(from_bytes.predict(&data, Iterations::Best).unwrap(), before);
    let from_json =
        BoostedModel::decode(model.encode(ModelFormat::Json).unwrap(), ModelFormat::Json).unwrap();
    assert_eq!(from_json.predict(&data, Iterations::Best).unwrap(), before);

    assert!(matches!(
        model.encode(ModelFormat::XgboostJson),
        Err(HessboostError::ModelFormat(_))
    ));
    assert!(matches!(
        model.encode(ModelFormat::XgboostUbjson),
        Err(HessboostError::ModelFormat(_))
    ));
    assert_eq!(
        invalid_param(model.predict_contribs(&data, Iterations::Best)),
        "linear_tree"
    );
    assert_eq!(
        invalid_param(model.predict_interactions(&data, Iterations::Best)),
        "linear_tree"
    );
}

#[test]
fn incompatible_configurations_are_rejected() {
    let invalid =
        |builder: TrainingParamsBuilder, name| assert_eq!(invalid_param(builder.build()), name);
    invalid(base().path_smooth(-1.0), "path_smooth");
    assert_eq!(invalid_param(LinearTree::new(f64::NAN)), "linear_lambda");
    invalid(
        base()
            .extra_trees(ExtraTrees::default())
            .tree_method(TreeMethod::Exact),
        "extra_trees",
    );
    invalid(
        base().path_smooth(1.0).booster(BoosterKind::GbLinear),
        "path_smooth",
    );
    invalid(
        base()
            .linear_tree(LinearTree::default())
            .multi_strategy(MultiStrategy::MultiOutputTree),
        "linear_tree",
    );
    invalid(
        base()
            .linear_tree(LinearTree::default())
            .objective(Objective::AbsoluteError),
        "linear_tree",
    );
    // The histogram-based `approx` builder accepts every option.
    base()
        .tree_method(TreeMethod::Approx)
        .extra_trees(ExtraTrees::default())
        .path_smooth(1.0)
        .linear_tree(LinearTree::default())
        .build()
        .unwrap();
}

/// A tiny positive `path_smooth` approaches the unsmoothed tree instead of
/// overflowing the `n / s` count ratio into a NaN gain that discards every
/// split.
#[test]
fn vanishing_path_smoothing_approaches_the_unsmoothed_tree() {
    let data = labeled_dense(&[0.0, 1.0], 1, &[-1.0, 1.0]);
    let params = TrainingParams::builder()
        .tree_method(TreeMethod::Hist)
        .path_smooth(1e-308)
        .lambda(0.0)
        .eta(1.0)
        .max_depth(1)
        .min_child_weight(0.0)
        .base_score(0.0)
        .build()
        .unwrap();
    let preds = train(&params, &data, 1)
        .unwrap()
        .predict(&data, Iterations::Best)
        .unwrap()
        .into_vec(); // one value per row
    assert!(
        (preds[0] + 1.0).abs() < 1e-6 && (preds[1] - 1.0).abs() < 1e-6,
        "{preds:?}"
    );
}

/// A monotone constraint on a categorical feature bounds a split's children
/// in the orientation the split was scored in (XGBoost's search records its
/// children swapped; the `extra_trees` search does not), so a fit that
/// satisfies the direction keeps its leaf values.
#[test]
fn categorical_splits_keep_their_monotone_leaves() {
    let data = labeled_dense(&[0.0, 0.0, 1.0, 1.0], 1, &[0.0, 0.0, 2.0, 2.0])
        .with_feature_types(&[FeatureType::Categorical])
        .unwrap();
    for extra_trees in [false, true] {
        let mut builder = TrainingParams::builder().tree_method(TreeMethod::Hist);
        if extra_trees {
            builder = builder.extra_trees(ExtraTrees::default());
        }
        let params = builder
            .monotone_constraints(vec![Monotone::Decreasing])
            .lambda(0.0)
            .eta(1.0)
            .max_depth(1)
            .base_score(0.0)
            .build()
            .unwrap();
        let preds = train(&params, &data, 1)
            .unwrap()
            .predict(&data, Iterations::Best)
            .unwrap();
        assert_eq!(
            preds.as_slice(),
            [0.0, 0.0, 2.0, 2.0],
            "extra_trees {extra_trees}"
        );
    }
}

/// Separating the `±1e20` labels gains about `1e40`, beyond `f32`: the
/// LightGBM searches skip that candidate on a categorical feature as they do
/// on a numerical one, so training succeeds with the same (unsplit) model.
#[test]
fn categorical_splits_with_unrepresentable_gains_are_skipped() {
    let x = [0.0, 1.0];
    let y = [-1e20, 1e20];
    let categorical = labeled_dense(&x, 1, &y)
        .with_feature_types(&[FeatureType::Categorical])
        .unwrap();
    let numerical = labeled_dense(&x, 1, &y);
    for builder in [
        base().extra_trees(ExtraTrees::default()),
        base().path_smooth(1.0),
    ] {
        let params = builder.base_score(0.0).build().unwrap();
        let preds = |data: &DMatrix| {
            train(&params, data, 1)
                .unwrap()
                .predict(data, Iterations::Best)
                .unwrap()
        };
        assert_eq!(preds(&categorical), preds(&numerical));
    }
}
