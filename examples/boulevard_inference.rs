//! Boulevard boosting with statistical inference on the regression function:
//! confidence intervals for `f(x)`, prediction intervals for new labels,
//! and a variable-importance test, on data with a known `f`.
//!
//! The model is trained with `booster = boulevard` (BRAT-D: dropout
//! Boulevard), its leaves are refitted on an independent sample
//! (`honest_refit`, the structure–value isolation the theory assumes), and
//! `BoulevardInference` turns its leaf kernel into standard errors.
//!
//! Run with: `cargo run --release --example boulevard_inference`

use hessboost::config::{BoosterKind, TrainingParams};
use hessboost::inference::{
    BoulevardInference, KernelSolver, NoiseVariance, honest_refit, importance_test,
};
use hessboost::prelude::*;

mod common;
use common::lcg;

/// The true regression function: `f(x) = sin(2π x0) + x0² / 2` (Fang, Tan &
/// Hooker's Figure 1). `x1` is pure noise.
fn f(x0: f32) -> f64 {
    let x0 = f64::from(x0);
    (std::f64::consts::TAU * x0).sin() + 0.5 * x0 * x0
}

/// `n` rows of `(x0, x1)` uniform on the unit square with
/// `y = f(x0) + w x1 + ε`, `ε` of variance `0.25`.
fn dataset(n: usize, w: f32, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut next = lcg(seed);
    let mut x = Vec::with_capacity(2 * n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let (x0, x1) = (next(), next());
        // Sum of 12 uniforms minus 6: mean 0, variance 1.
        let eps: f32 = (0..12).map(|_| next()).sum::<f32>() - 6.0;
        x.extend_from_slice(&[x0, x1]);
        y.push(f(x0) as f32 + w * x1 + 0.5 * eps);
    }
    (x, y)
}

fn matrix(x: &[f32], y: &[f32], cols: usize) -> Result<DMatrix> {
    DMatrix::from_dense(x, y.len(), cols)?.with_labels(y)
}

/// The first column of a two-column row-major matrix.
fn first_column(x: &[f32]) -> Vec<f32> {
    x.as_chunks::<2>().0.iter().map(|r| r[0]).collect()
}

fn params() -> Result<TrainingParams> {
    TrainingParams::builder()
        .booster(BoosterKind::Boulevard)
        .eta(0.6)
        .boulevard_dropout(0.6)
        .subsample(0.6)
        .max_depth(8)
        .min_child_weight(5.0)
        .lambda(0.0)
        .seed(1)
        .build()
}

fn main() -> Result<()> {
    let n = 1000;
    let (xs, ys) = dataset(n, 0.0, 1); // grows the tree structures
    let (xv, yv) = dataset(n, 0.0, 2); // refits the leaves; the kernel's rows
    let (xc, yc) = dataset(n / 2, 0.0, 3); // estimates the noise variance
    let (dstruct, dvalues, dcal) = (
        matrix(&xs, &ys, 2)?,
        matrix(&xv, &yv, 2)?,
        matrix(&xc, &yc, 2)?,
    );

    let trained = train(&params()?, &dstruct, 200)?;
    let model = honest_refit(&trained, &dvalues)?;
    let inference = BoulevardInference::fit(
        &model,
        &dvalues,
        NoiseVariance::Holdout(&dcal),
        KernelSolver::Exact,
    )?;
    println!(
        "BRAT-D, {} trees; noise variance {:.3} (true 0.250)",
        model.num_trees(),
        inference.noise_variance()
    );

    // Intervals on a grid of x0 (x1 fixed at 0.5).
    let grid: Vec<f32> = (0..9)
        .flat_map(|i| [0.05 + 0.1125 * i as f32, 0.5])
        .collect();
    let dgrid = DMatrix::from_dense(&grid, 9, 2)?;
    let preds = model.predict(&dgrid)?;
    let se = inference.standard_errors(&dgrid)?;
    let ci = inference.confidence_intervals(&dgrid, 0.05)?;
    let pi = inference.prediction_intervals(&dgrid, 0.05)?;
    println!("\n    x0   f(x0)   f̂(x0)     se          95% CI for f          95% PI for y");
    for i in 0..9 {
        println!(
            "  {:.3}  {:+.3}  {:+.3}  {:.3}  [{:+.3}, {:+.3}]  [{:+.3}, {:+.3}]",
            grid[2 * i],
            f(grid[2 * i]),
            preds[i],
            se[i],
            ci[i].0,
            ci[i].1,
            pi[i].0,
            pi[i].1
        );
    }

    // Pointwise coverage of f on fresh test points (one fitted model, so
    // this is a single draw of the marginal coverage).
    let (xt, yt) = dataset(500, 0.0, 4);
    let dtest = matrix(&xt, &yt, 2)?;
    for alpha in [0.10, 0.05] {
        let ci = inference.confidence_intervals(&dtest, alpha)?;
        let pi = inference.prediction_intervals(&dtest, alpha)?;
        let covered_f = ci
            .iter()
            .zip(xt.as_chunks::<2>().0)
            .filter(|&(&(lo, hi), x)| lo <= f(x[0]) && f(x[0]) <= hi)
            .count();
        let covered_y = pi
            .iter()
            .zip(&yt)
            .filter(|&(&(lo, hi), &y)| lo <= f64::from(y) && f64::from(y) <= hi)
            .count();
        let width: f64 = ci.iter().map(|(lo, hi)| hi - lo).sum::<f64>() / ci.len() as f64;
        println!(
            "\nnominal {:.0}%: CI covers f at {:.1}% of test points (mean width {width:.3}), PI \
             covers y at {:.1}%",
            100.0 * (1.0 - alpha),
            100.0 * covered_f as f64 / ci.len() as f64,
            100.0 * covered_y as f64 / pi.len() as f64
        );
    }

    // Variable importance: is x1 needed? Full model on (x0, x1), reduced on
    // x0 alone, each trained and refitted on its own half of the data.
    println!("\nvariable-importance test of x1 (20 test points):");
    let points: Vec<f32> = (0..20)
        .flat_map(|i| [0.025 + 0.05 * i as f32, ((i * 7) % 20) as f32 / 20.0])
        .collect();
    let (dfull_points, dreduced_points) = (
        DMatrix::from_dense(&points, 20, 2)?,
        DMatrix::from_dense(&first_column(&points), 20, 1)?,
    );
    for w in [0.0, 1.0] {
        let data = |seed| dataset(n, w, seed);
        let ((x1s, y1s), (x1v, y1v), (x2s, y2s), (x2v, y2v), (xh, yh)) =
            (data(11), data(12), data(13), data(14), data(15));
        let full_values = matrix(&x1v, &y1v, 2)?;
        let full = honest_refit(
            &train(&params()?, &matrix(&x1s, &y1s, 2)?, 200)?,
            &full_values,
        )?;
        let reduced_values = matrix(&first_column(&x2v), &y2v, 1)?;
        let reduced = honest_refit(
            &train(&params()?, &matrix(&first_column(&x2s), &y2s, 1)?, 200)?,
            &reduced_values,
        )?;
        let holdout = matrix(&xh, &yh, 2)?;
        let full_inf = BoulevardInference::fit(
            &full,
            &full_values,
            NoiseVariance::Holdout(&holdout),
            KernelSolver::Exact,
        )?;
        let reduced_inf = BoulevardInference::fit(
            &reduced,
            &reduced_values,
            NoiseVariance::TrainingResiduals,
            KernelSolver::Exact,
        )?;
        let test = importance_test(&full_inf, &dfull_points, &reduced_inf, &dreduced_points)?;
        println!(
            "  y = f(x0) + {w} x1 + ε: chi² = {:.1} on {} df, p = {:.4}",
            test.statistic, test.degrees_of_freedom, test.p_value
        );
    }

    // The Boulevard record survives the native formats.
    let reloaded = BoostedModel::from_bytes(&model.to_bytes()?)?;
    assert_eq!(reloaded.boulevard(), model.boulevard());
    Ok(())
}
