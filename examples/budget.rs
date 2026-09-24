//! Budget-mode training (Perpetual's algorithm): one `budget` number
//! instead of tuning the learning rate, tree size, and round count.
//!
//! Compares several budgets against default XGBoost-style training and a
//! validation-tuned run (early stopping on a held-out set) on synthetic
//! regression (Friedman #1) and binary classification.
//!
//! Run with: `cargo run --release --example budget`

use hessboost::metric::{LogLoss, Metric, Rmse};
use hessboost::prelude::*;
use hessboost::training::budget::{BudgetConfig, train_with_budget};
use std::time::Instant;

mod common;
use common::lcg;

const N_FEATURES: usize = 10;

/// Friedman #1 features (10 uniforms, 5 informative), the noiseless target
/// `10 sin(π x0 x1) + 20 (x2 − ½)² + 10 x3 + 5 x4`, and a noisy copy.
fn friedman(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut next = lcg(seed);
    let mut x = Vec::with_capacity(n * N_FEATURES);
    let (mut clean, mut noisy) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let row: Vec<f32> = (0..N_FEATURES).map(|_| next()).collect();
        let f = 10.0 * (std::f32::consts::PI * row[0] * row[1]).sin()
            + 20.0 * (row[2] - 0.5).powi(2)
            + 10.0 * row[3]
            + 5.0 * row[4];
        let eps: f32 = (0..12).map(|_| next()).sum::<f32>() - 6.0;
        x.extend_from_slice(&row);
        clean.push(f);
        noisy.push(f + eps);
    }
    (x, clean, noisy)
}

fn regression(n: usize, seed: u64) -> Result<DMatrix> {
    let (x, _, y) = friedman(n, seed);
    DMatrix::from_dense(&x, n, N_FEATURES)?.with_labels(&y)
}

/// Binary labels drawn with probability `σ((f − 14) / 2)` from the Friedman
/// function, so the Bayes classifier is imperfect.
fn binary(n: usize, seed: u64) -> Result<DMatrix> {
    let (x, f, _) = friedman(n, seed);
    let mut next = lcg(seed ^ 0xB1);
    let y: Vec<f32> = f
        .iter()
        .map(|v| f32::from(next() < 1.0 / (1.0 + (-(v - 14.0) / 2.0).exp())))
        .collect();
    DMatrix::from_dense(&x, n, N_FEATURES)?.with_labels(&y)
}

/// Mean leaves per tree over the first `trees` trees.
fn mean_leaves(model: &BoostedModel, trees: usize) -> f64 {
    let leaves: usize = model.trees()[..trees]
        .iter()
        .map(hessboost::tree::RegTree::num_leaves)
        .sum();
    leaves as f64 / trees.max(1) as f64
}

fn row(label: &str, model: &BoostedModel, trees: usize, score: f64, seconds: f64) {
    let leaves = mean_leaves(model, trees);
    println!("  {label:<34} {trees:>6} {leaves:>7.1} {score:>9.4} {seconds:>8.2}");
}

fn report(
    task: &str,
    objective: &str,
    dtrain: &DMatrix,
    dvalid: &DMatrix,
    dtest: &DMatrix,
    metric: &dyn Metric,
) -> Result<()> {
    let labels = dtest.labels().unwrap_or_default();
    let score = |preds: &[f32]| metric.eval(preds, labels, None);
    let params = TrainingParams::builder().objective(objective).build()?;
    println!("{task} (test {}):", metric.name());
    println!(
        "  {:<34} {:>6} {:>7} {:>9} {:>8}",
        "method", "trees", "leaves", "score", "seconds"
    );

    let start = Instant::now();
    let model = train(&params, dtrain, 100)?;
    let seconds = start.elapsed().as_secs_f64();
    let value = score(&model.predict(dtest)?);
    row(
        "default (eta 0.3, depth 6, 100)",
        &model,
        model.num_trees(),
        value,
        seconds,
    );

    // Tuned: a smaller learning rate with early stopping on the validation
    // set (the tuning budget mode replaces).
    let tuned_params = TrainingParams::builder()
        .objective(objective)
        .eta(0.05)
        .build()?;
    let start = Instant::now();
    let tuned = Trainer::new(&tuned_params, dtrain, 2000)
        .eval(dvalid, "valid")
        .early_stopping_rounds(50)
        .train()?
        .model;
    let seconds = start.elapsed().as_secs_f64();
    let rounds = tuned.best_iteration().map_or(tuned.num_trees(), |b| b + 1);
    let value = score(&tuned.predict(dtest)?);
    row(
        "tuned (eta 0.05, early stopping)",
        &tuned,
        rounds,
        value,
        seconds,
    );

    for budget in [0.5, 1.0, 1.5] {
        let start = Instant::now();
        let result = train_with_budget(&params, dtrain, &BudgetConfig::new(budget))?;
        let seconds = start.elapsed().as_secs_f64();
        let value = score(&result.model.predict(dtest)?);
        let label = format!("budget {budget} (eta {:.3}, {:?})", result.eta, result.stop);
        row(
            &label,
            &result.model,
            result.model.num_trees(),
            value,
            seconds,
        );
    }
    println!();
    Ok(())
}

fn main() -> Result<()> {
    report(
        "Friedman #1 regression, 5000 rows",
        "reg:squarederror",
        &regression(5000, 1)?,
        &regression(2000, 2)?,
        &regression(10_000, 3)?,
        &Rmse,
    )?;
    report(
        "Friedman #1 binary classification, 5000 rows",
        "binary:logistic",
        &binary(5000, 4)?,
        &binary(2000, 5)?,
        &binary(10_000, 6)?,
        &LogLoss,
    )?;
    Ok(())
}
