//! In-place data updates (beyond XGBoost): add and delete training rows of
//! a trained model without retraining it from scratch, and unlearn rows
//! exactly when that is required.
//!
//! Trains a regression model on 20,000 rows, then adds 200 new rows and
//! deletes 200 old ones with the approximate mode (split robustness
//! tolerance 0.1), comparing time and test error with retraining; then
//! checks that the exact mode (tolerance 0) reproduces retraining bit for
//! bit.
//!
//! Run with: `cargo run --release --example online_update`

use std::time::Instant;

use hessboost::prelude::*;
use hessboost::training::online::{OnlineModel, OnlineParams};

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
    let labels = data.labels().unwrap_or_default();
    let sse: f64 = preds
        .as_slice()
        .iter()
        .zip(labels)
        .map(|(p, y)| f64::from(p - y).powi(2))
        .sum();
    Ok((sse / labels.len() as f64).sqrt())
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

    // Exact unlearning: tolerance 0 equals retraining on the remaining rows.
    let mut exact = OnlineModel::train(&params, &data, rounds, OnlineParams::exact())?;
    exact.update(None, &deletions)?;
    let reference = train(&params, exact.data(), rounds)?;
    println!(
        "exact mode after deleting 200 rows equals retraining bit for bit: {}",
        exact.model().to_json()? == reference.to_json()?
    );
    Ok(())
}
