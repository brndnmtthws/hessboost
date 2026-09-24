//! Distribution-free prediction intervals: split conformal and conformalized
//! quantile regression (CQR) on heteroscedastic data.
//!
//! Both methods calibrate a fitted model on a held-out set and guarantee
//! marginal coverage `P(Y in C(X)) >= 1 - alpha` on exchangeable test data.
//! Split conformal gives constant-width intervals; CQR keeps the adaptive
//! width of a quantile model.
//!
//! Run with: `cargo run --release --example conformal`

use hessboost::conformal::{ConformalizedQuantile, SplitConformal};
use hessboost::objective::{CustomObjective, GradPair};
use hessboost::prelude::*;

mod common;
use common::lcg;

/// `y = sin(2π x0) + (0.1 + x1) · ε` with `ε` roughly standard normal: the
/// noise scale grows with `x1`.
fn dataset(n: usize, seed: u64) -> Result<DMatrix> {
    let mut next = lcg(seed);
    let mut x = Vec::with_capacity(2 * n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let (x0, x1) = (next(), next());
        // Sum of 12 uniforms minus 6: mean 0, variance 1.
        let eps: f32 = (0..12).map(|_| next()).sum::<f32>() - 6.0;
        x.extend_from_slice(&[x0, x1]);
        y.push((std::f32::consts::TAU * x0).sin() + (0.1 + x1) * eps);
    }
    DMatrix::from_dense(&x, n, 2)?.with_labels(&y)
}

/// Fraction of rows whose label lies in its interval, and the mean width.
fn summarize(intervals: &[(f32, f32)], data: &DMatrix) -> (f64, f64) {
    let labels = data.labels().unwrap_or_default();
    let covered = intervals
        .iter()
        .zip(labels)
        .filter(|&(&(lo, hi), &y)| lo <= y && y <= hi)
        .count();
    let width: f64 = intervals.iter().map(|(lo, hi)| f64::from(hi - lo)).sum();
    let n = intervals.len() as f64;
    (covered as f64 / n, width / n)
}

fn main() -> Result<()> {
    let alpha = 0.1;
    // Three disjoint samples: the calibration set must not be used for training.
    let dtrain = dataset(4000, 1)?;
    let dcal = dataset(1000, 2)?;
    let dtest = dataset(5000, 3)?;
    let params = TrainingParams::builder().max_depth(4).eta(0.3).build()?;

    // --- Split conformal around a squared-error point model. ---
    let point = train(&params, &dtrain, 50)?;
    let split = SplitConformal::calibrate(&point, &dcal, alpha)?;
    let split_iv = split.predict_interval(&dtest)?;
    let (cov, width) = summarize(&split_iv, &dtest);
    println!(
        "split conformal: half-width Q = {:.3}, coverage {cov:.3}, mean width {width:.3}",
        split.half_width()
    );

    // --- CQR around a two-output quantile model. ---
    // The intended pairing is XGBoost's `reg:quantileerror` with
    // `quantile_alpha = [alpha / 2, 1 - alpha / 2]`. Here the same pinball
    // loss is supplied through the custom-objective hook: output j fits
    // quantile level taus[j].
    let taus = [(alpha / 2.0) as f32, (1.0 - alpha / 2.0) as f32];
    let pinball = CustomObjective::new("pinball", 2, 0.0, "mae", move |p, y, _w, out| {
        for (i, &yi) in y.iter().enumerate() {
            for (j, tau) in taus.iter().enumerate() {
                let g = if p[2 * i + j] > yi { 1.0 - tau } else { -tau };
                out[2 * i + j] = GradPair::new(g, 1.0);
            }
        }
    });
    let quantiles = Trainer::new(&params, &dtrain, 200)
        .objective(&pinball)
        .train()?
        .model;
    // The uncalibrated band, for comparison: predictions are `[row][output]`.
    let preds = quantiles.predict(&dtest)?;
    let band: Vec<(f32, f32)> = preds
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[lo, hi]| (lo, hi))
        .collect();
    let (cov, width) = summarize(&band, &dtest);
    println!("raw quantiles:   coverage {cov:.3}, mean width {width:.3}");
    let cqr = ConformalizedQuantile::calibrate_outputs(&quantiles, 0, 1, &dcal, alpha)?;
    let cqr_iv = cqr.predict_interval(&dtest)?;
    let (cov, width) = summarize(&cqr_iv, &dtest);
    println!(
        "CQR:             correction Q = {:+.3}, coverage {cov:.3}, mean width {width:.3}",
        cqr.correction()
    );

    // Width by noise level: split conformal is constant, CQR adapts.
    let probe = DMatrix::from_dense(&[0.25, 0.05, 0.25, 0.5, 0.25, 0.95], 3, 2)?;
    let split_probe = split.predict_interval(&probe)?;
    let cqr_probe = cqr.predict_interval(&probe)?;
    println!("\n  x1   noise sd   split width   CQR width");
    for (i, x1) in [0.05f32, 0.5, 0.95].into_iter().enumerate() {
        println!(
            "{x1:5.2}   {:8.2}   {:11.3}   {:9.3}",
            0.1 + x1,
            split_probe[i].1 - split_probe[i].0,
            cqr_probe[i].1 - cqr_probe[i].0
        );
    }
    // The guarantee averages over calibration draws: for this one calibration
    // set the realized coverage is Beta-distributed around k / (n + 1), with a
    // standard deviation of about sqrt(alpha (1 - alpha) / n) ≈ 0.01.
    let n_cal = dcal.n_rows() as f64;
    println!(
        "\nBoth guarantee marginal coverage in [{:.2}, {:.4}] (no ties) averaged over \
         calibration sets;\na single calibration set of {n_cal} rows fluctuates by about ±{:.3}.",
        1.0 - alpha,
        1.0 - alpha + 1.0 / (n_cal + 1.0),
        (alpha * (1.0 - alpha) / n_cal).sqrt()
    );
    Ok(())
}
