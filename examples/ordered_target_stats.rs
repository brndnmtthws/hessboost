//! Opt-in CatBoost-style ordered target statistics (beyond XGBoost) for a
//! high-cardinality categorical feature with few rows per category, compared
//! with native categorical splits and a leaky in-sample target mean.
//! Run: `cargo run --release --example ordered_target_stats`.

use hessboost::data::FeatureType;
use hessboost::data::target_stats::{FittedTargetEncoder, OrderedTargetEncoder};
use hessboost::metric::{Metric, Rmse};
use hessboost::prelude::*;

mod common;
use common::lcg;

const CATEGORIES: usize = 800;

/// `y = effect[category] + 2·x + noise`; column 0 holds the category code,
/// column 1 a numeric feature. Returns `(features, labels)`.
fn sample(rows: usize, effects: &[f32], rng: &mut impl FnMut() -> f32) -> (Vec<f32>, Vec<f32>) {
    let (mut x, mut y) = (Vec::with_capacity(rows * 2), Vec::with_capacity(rows));
    for _ in 0..rows {
        let cat = ((rng() * CATEGORIES as f32) as usize).min(CATEGORIES - 1);
        let num = rng();
        let noise: f32 = 3.0 * (0..4).map(|_| rng() - 0.5).sum::<f32>();
        x.extend([cat as f32, num]);
        y.push(effects[cat] + 2.0 * num + noise);
    }
    (x, y)
}

fn matrix(x: &[f32], y: &[f32]) -> Result<DMatrix> {
    DMatrix::from_dense(x, y.len(), 2)?
        .with_labels(y)?
        .with_feature_types(&[FeatureType::Categorical, FeatureType::Numerical])
}

/// `(train, held-out)` RMSE of a model trained on `train_set`.
fn fit_rmse(
    params: &TrainingParams,
    train_set: &DMatrix,
    test_set: &DMatrix,
) -> Result<(f64, f64)> {
    let model = train(params, train_set, 200)?;
    let rmse = |data: &DMatrix| -> Result<f64> {
        Ok(Rmse.eval(
            &model.predict(data)?,
            data.labels().unwrap_or_default(),
            None,
        ))
    };
    Ok((rmse(train_set)?, rmse(test_set)?))
}

/// The naive encoding: each category's in-sample mean *including* the row's
/// own target, which leaks the label into the feature.
fn naive_mean(x: &[f32], y: &[f32], test_x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut sum = vec![0f32; CATEGORIES];
    let mut count = vec![0f32; CATEGORIES];
    for (&[cat, _], &label) in x.as_chunks::<2>().0.iter().zip(y) {
        sum[cat as usize] += label;
        count[cat as usize] += 1.0;
    }
    let prior = y.iter().sum::<f32>() / y.len() as f32;
    let encode = |data: &[f32]| {
        data.as_chunks::<2>()
            .0
            .iter()
            .flat_map(|&[cat, num]| {
                let c = cat as usize;
                [(sum[c] + prior) / (count[c] + 1.0), num]
            })
            .collect()
    };
    (encode(x), encode(test_x))
}

fn main() -> Result<()> {
    let mut rng = lcg(7);
    let effects: Vec<f32> = (0..CATEGORIES).map(|_| 4.0 * rng() - 2.0).collect();
    // ~4 training rows per category.
    let (train_x, train_y) = sample(3200, &effects, &mut rng);
    let (test_x, test_y) = sample(4000, &effects, &mut rng);
    let (dtrain, dtest) = (matrix(&train_x, &train_y)?, matrix(&test_x, &test_y)?);

    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .max_depth(4)
        .eta(0.1)
        .build()?;

    // 1. Native categorical partition splits on the raw codes.
    let native = fit_rmse(&params, &dtrain, &dtest)?;

    // 2. In-sample target mean: the row's own label leaks into its feature.
    let (naive_train, naive_test) = naive_mean(&train_x, &train_y, &test_x);
    let naive = fit_rmse(
        &params,
        &DMatrix::from_dense(&naive_train, train_y.len(), 2)?.with_labels(&train_y)?,
        &DMatrix::from_dense(&naive_test, test_y.len(), 2)?.with_labels(&test_y)?,
    )?;

    // 3. Ordered target statistics: each training row only sees the rows before
    //    it in a seeded permutation; test rows use all training statistics.
    let encoder = OrderedTargetEncoder::builder()
        .prior_weight(1.0)
        .seed(0)
        .build()?;
    let (encoded_train, fitted) = encoder.fit_transform(&dtrain, &[0])?;
    let ordered = fit_rmse(&params, &encoded_train, &fitted.transform(&dtest)?)?;

    println!("RMSE (train / held-out), {CATEGORIES} categories x ~4 rows each:");
    for (name, (train_rmse, test_rmse)) in [
        ("native categorical splits", native),
        ("in-sample target mean (leaky)", naive),
        ("ordered target statistics", ordered),
    ] {
        println!("  {name:<30} {train_rmse:.4} / {test_rmse:.4}");
    }

    // The fitted encoder is serde-serializable: store it next to the model and
    // reload it to encode new data exactly as during training.
    let json = serde_json::to_string(&fitted)?;
    let reloaded: FittedTargetEncoder = serde_json::from_str(&json)?;
    assert_eq!(reloaded, fitted);
    println!(
        "fitted encoder: {} bytes of JSON, prior {:.4}, unseen category -> {:?}",
        json.len(),
        reloaded.prior(),
        reloaded.encode(0, 1_000_000),
    );
    Ok(())
}
