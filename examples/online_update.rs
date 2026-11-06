//! In-place data updates: add and delete training rows of
//! a trained model without retraining it from scratch, and unlearn rows
//! exactly when that is required.
//!
//! Trains a regression model on 20,000 rows, then adds 200 new rows and
//! deletes 200 old ones with the approximate mode (split robustness
//! tolerance 0.1), comparing time and test error with retraining; then
//! checks that the exact mode (`OnlineMode::Exact`) reproduces retraining
//! bit for bit, also on recency-weighted rows with row and column
//! subsampling.
//!
//! Run with: `cargo run --release --example online_update`

use std::time::Instant;

use hessboost::prelude::*;
use hessboost::training::online::{OnlineMode, OnlineModel, OnlineParams};

mod common;
use common::lcg;

const COLS: usize = 10;

/// Friedman #1: `10 sin(π x0 x1) + 20 (x2 - 0.5)² + 10 x3 + 5 x4 + noise`.
fn friedman(n: usize, seed: u64) -> Result<DMatrix> {
    let mut next = lcg(seed);
    let (mut x, mut y) = (Vec::with_capacity(n * COLS), Vec::with_capacity(n));
    for _ in 0..n {
        let row: Vec<f32> = (0..COLS).map(|_| next()).collect();
        let f = 10.0 * (std::f32::consts::PI * row[0] * row[1]).sin()
            + 20.0 * (row[2] - 0.5).powi(2)
            + 10.0 * row[3]
            + 5.0 * row[4];
        y.push(f + 2.0 * (next() + next() + next() - 1.5));
        x.extend(row);
    }
    DMatrix::from_dense(&x, n, COLS)?.with_labels(&y)
}

fn rmse(model: &BoostedModel, data: &DMatrix) -> Result<f64> {
    let preds = model.predict(data, Iterations::Best)?;
    let metric = EvalMetric::Rmse.build(1)?;
    Ok(metric.eval(preds.as_slice(), data.labels().unwrap_or_default(), None))
}

fn main() -> Result<()> {
    let (data, test, new_rows) = (friedman(20_000, 1)?, friedman(5000, 2)?, friedman(200, 3)?);
    let params = TrainingParams::builder()
        .tree_method(TreeMethod::Hist)
        .max_depth(6)
        .eta(0.1)
        .build()?;
    let rounds = 100;
    let deletions: Vec<usize> = (0..200).map(|i| i * 97).collect();

    let start = Instant::now();
    let mut online = OnlineModel::train(&params, &data, rounds, OnlineParams::default())?;
    println!("trained with update state in {:.1?}", start.elapsed());
    let original = online.model().clone();

    let start = Instant::now();
    let report = online.update(Some(&new_rows), &deletions)?;
    let update_time = start.elapsed();
    let start = Instant::now();
    let retrained = train(&params, online.data(), rounds)?;
    let retrain_time = start.elapsed();
    println!(
        "add 200 + delete 200 rows: update {update_time:.1?} vs retrain {retrain_time:.1?} \
         ({:.1}x); kept {} nodes, regrew {} subtrees, refreshed {} rows",
        retrain_time.as_secs_f64() / update_time.as_secs_f64(),
        report.nodes_kept,
        report.subtrees_regrown,
        report.rows_refreshed,
    );
    println!(
        "test RMSE: original {:.4}, updated {:.4}, retrained {:.4}",
        rmse(&original, &test)?,
        rmse(online.model(), &test)?,
        rmse(&retrained, &test)?,
    );

    // Exact unlearning: the exact mode equals retraining on the remaining rows.
    let mut exact = OnlineModel::train(&params, &data, rounds, OnlineParams::exact())?;
    assert_eq!(exact.online_params().mode(), OnlineMode::Exact);
    exact.update(None, &deletions)?;
    let reference = train(&params, exact.data(), rounds)?;
    println!(
        "exact mode after deleting 200 rows equals retraining bit for bit: {}",
        exact.model().encode(ModelFormat::Json)? == reference.encode(ModelFormat::Json)?
    );

    // Recency-weighted rows and a subsampled ensemble: the approximate mode
    // keeps each tree's row and column samples; the exact mode redraws them
    // over the updated rows, as a retrain with the same seed does.
    let recency: Vec<f32> = (0..data.n_rows())
        .map(|i| 0.5 + i as f32 / data.n_rows() as f32)
        .collect();
    let weighted = data.clone().with_weights(&recency)?;
    let new_weighted = new_rows.clone().with_weights(&[1.5; 200])?;
    let mut sampled = params.clone();
    sampled.subsample = 0.5;
    sampled.colsample_bytree = 0.5;
    let mut approximate = OnlineModel::train(&sampled, &weighted, rounds, OnlineParams::default())?;
    approximate.update(Some(&new_weighted), &deletions)?;
    let reference = train(&sampled, approximate.data(), rounds)?;
    println!(
        "weighted, subsampled: test RMSE updated {:.4}, retrained {:.4}",
        rmse(approximate.model(), &test)?,
        rmse(&reference, &test)?,
    );
    let mut exact = OnlineModel::train(&sampled, &weighted, rounds, OnlineParams::exact())?;
    exact.update(Some(&new_weighted), &deletions)?;
    println!(
        "weighted, subsampled exact mode equals retraining bit for bit: {}",
        exact.model().encode(ModelFormat::Json)? == reference.encode(ModelFormat::Json)?
    );
    Ok(())
}
