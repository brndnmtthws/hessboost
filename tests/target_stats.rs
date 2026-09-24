//! Held-out quality of ordered target statistics against native categorical
//! splits on a high-cardinality feature with few rows per category.

use hessboost::config::TreeMethod;
use hessboost::data::FeatureType;
use hessboost::data::target_stats::OrderedTargetEncoder;
use hessboost::prelude::*;

mod common;
use common::{labeled_dense, rmse};

const CATEGORIES: usize = 600;
const TRAIN_ROWS: usize = 2400; // ~4 rows per category
const TEST_ROWS: usize = 4000;

/// 64-bit LCG returning uniforms in `[0, 1)`.
fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `y = effect[category] + 2·x + noise`; column 0 is the category code,
/// column 1 the numeric feature.
fn sample(rows: usize, effects: &[f64], rng: &mut impl FnMut() -> f64) -> DMatrix {
    let mut x = Vec::with_capacity(rows * 2);
    let mut y = Vec::with_capacity(rows);
    for _ in 0..rows {
        let cat = ((rng() * CATEGORIES as f64) as usize).min(CATEGORIES - 1);
        let num = rng();
        let noise = (0..4).map(|_| rng() - 0.5).sum::<f64>();
        x.extend([cat as f32, num as f32]);
        y.push((effects[cat] + 2.0 * num + noise) as f32);
    }
    labeled_dense(&x, 2, &y)
        .with_feature_types(&[FeatureType::Categorical, FeatureType::Numerical])
        .unwrap()
}

#[test]
fn ordered_target_stats_beat_native_splits_on_sparse_categories() {
    let mut rng = lcg(2026);
    let effects: Vec<f64> = (0..CATEGORIES).map(|_| 4.0 * rng() - 2.0).collect();
    let train_set = sample(TRAIN_ROWS, &effects, &mut rng);
    let test_set = sample(TEST_ROWS, &effects, &mut rng);

    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .tree_method(TreeMethod::Hist)
        .max_depth(4)
        .eta(0.1)
        .build()
        .unwrap();
    let rounds = 150;

    let native = train(&params, &train_set, rounds).unwrap();
    let native_rmse = rmse(&native, &test_set);

    let encoder = OrderedTargetEncoder::builder().seed(1).build().unwrap();
    let (encoded_train, fitted) = encoder.fit_transform(&train_set, &[0]).unwrap();
    let encoded_test = fitted.transform(&test_set).unwrap();
    let ordered = train(&params, &encoded_train, rounds).unwrap();
    let ordered_rmse = rmse(&ordered, &encoded_test);

    println!("held-out RMSE: native {native_rmse:.4}, ordered TS {ordered_rmse:.4}");
    assert!(
        ordered_rmse <= native_rmse,
        "ordered TS {ordered_rmse} should not lose to native splits {native_rmse}"
    );
}
