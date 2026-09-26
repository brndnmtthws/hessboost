//! Virtual ensembles (beyond XGBoost): SGLB posterior sampling turns one
//! model into an ensemble of its own truncations, whose disagreement is
//! knowledge (epistemic) uncertainty. It grows where the training data is
//! missing. A distributional (`dist:normal`) model adds data (aleatoric)
//! uncertainty, which tracks the noise level where the model has seen data.
//!
//! Run with: `cargo run --release --example virtual_ensembles`

use hessboost::prelude::*;

mod common;
use common::lcg;

/// The noise scale of the target at `x0`.
fn noise_sd(x0: f32) -> f32 {
    0.1 + 0.3 * x0
}

/// `n` rows of `y = sin(2π x0) + x1 + noise_sd(x0) ε` over the unit
/// square, leaving out the corner `x0, x1 > 0.6`.
fn dataset(n: usize, seed: u64) -> Result<DMatrix> {
    let mut next = lcg(seed);
    let (mut x, mut y) = (Vec::with_capacity(2 * n), Vec::with_capacity(n));
    while y.len() < n {
        let (x0, x1) = (next(), next());
        if x0 > 0.6 && x1 > 0.6 {
            continue;
        }
        // A sum of twelve uniforms: roughly standard normal noise.
        let eps = (0..12).map(|_| next()).sum::<f32>() - 6.0;
        x.extend_from_slice(&[x0, x1]);
        y.push((std::f32::consts::TAU * x0).sin() + x1 + noise_sd(x0) * eps);
    }
    DMatrix::from_dense(&x, n, 2)?.with_labels(&y)
}

/// `n` unlabeled points drawn uniformly from the box `lo..hi`.
fn region(n: usize, seed: u64, lo: [f32; 2], hi: [f32; 2]) -> Result<DMatrix> {
    let mut next = lcg(seed);
    let x: Vec<f32> = (0..2 * n)
        .map(|i| lo[i % 2] + (hi[i % 2] - lo[i % 2]) * next())
        .collect();
    DMatrix::from_dense(&x, n, 2)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// The share of (out, in) pairs where the out-of-distribution point is the
/// more uncertain one: the ROC AUC of knowledge uncertainty as a detector.
fn detection_auc(inside: &[f64], outside: &[f64]) -> f64 {
    let wins: usize = outside
        .iter()
        .map(|&o| inside.iter().filter(|&&i| o > i).count())
        .sum();
    wins as f64 / (inside.len() * outside.len()) as f64
}

fn main() -> Result<()> {
    let dtrain = dataset(2000, 1)?;
    let test = dataset(1000, 2)?;
    let corner = region(1000, 3, [0.6, 0.6], [1.0, 1.0])?;
    let beyond = region(1000, 4, [1.2, 0.0], [2.0, 1.0])?;

    // `posterior_sampling` sets Langevin noise at temperature N and model
    // shrinkage at rate 1/(2N) (N = training rows), as CatBoost does.
    let params = |objective: &str| {
        TrainingParams::builder()
            .objective(objective)
            .tree_method(TreeMethod::Hist)
            .max_depth(4)
            .eta(0.1)
            .posterior_sampling(true)
            .seed(7)
            .build()
    };

    let model = train(&params("reg:squarederror")?, &dtrain, 1000)?;
    let members = model.predict_virtual_ensembles(&test, 10)?;
    println!(
        "reg:squarederror, {} members: the models after iterations {:?}",
        members.n_members(),
        members.iterations()
    );
    let inside = model.predict_uncertainty(&test, 10)?.knowledge;
    println!(
        "\n{:<34} {:>10} {:>8} {:>6}",
        "inputs", "knowledge", "ratio", "AUC"
    );
    println!(
        "{:<34} {:>10.2e}",
        "test set (like the training data)",
        mean(&inside)
    );
    for (name, data) in [
        ("left-out corner x0, x1 > 0.6", &corner),
        ("beyond the range, x0 in [1.2, 2]", &beyond),
    ] {
        let knowledge = model.predict_uncertainty(data, 10)?.knowledge;
        println!(
            "{name:<34} {:>10.2e} {:>7.1}x {:>6.3}",
            mean(&knowledge),
            mean(&knowledge) / mean(&inside),
            detection_auc(&inside, &knowledge)
        );
    }

    // A `dist:normal` model predicts a variance per row: data uncertainty
    // (its mean over the members) against the true noise variance.
    let dist = train(&params("dist:normal")?, &dtrain, 100)?;
    println!(
        "\ndist:normal: {:<22} {:>10} {:>10} {:>10} {:>10}",
        "inputs", "knowledge", "data", "total", "true var"
    );
    for (name, x0) in [
        ("x0 in [0, 0.2]", [0.0, 0.2]),
        ("x0 in [0.4, 0.6]", [0.4, 0.6]),
    ] {
        let data = region(1000, 5, [x0[0], 0.0], [x0[1], 0.6])?;
        let u = dist.predict_uncertainty(&data, 10)?;
        let truth = (0..=100)
            .map(|i| f64::from(noise_sd(x0[0] + (x0[1] - x0[0]) * i as f32 / 100.0)).powi(2))
            .sum::<f64>()
            / 101.0;
        println!(
            "             {name:<22} {:>10.2e} {:>10.4} {:>10.4} {:>10.4}",
            mean(&u.knowledge),
            mean(u.data.as_deref().unwrap_or_default()),
            mean(u.total.as_deref().unwrap_or_default()),
            truth
        );
    }
    Ok(())
}
