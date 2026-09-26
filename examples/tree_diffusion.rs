//! Tree-based conditional diffusion and flow matching (beyond XGBoost):
//! learn the whole conditional distribution `p(y | x)` without a parametric
//! family and sample from it.
//!
//! On a bimodal target (`y = ±(1 + x) + noise`, the sign a hidden coin) and
//! a skewed heteroscedastic one, fit score diffusion (DiffGBM's recipe) and
//! flow matching, print sample quantiles and a text histogram, and compare
//! the held-out CRPS with a `dist:normal` model. Then sample a
//! two-dimensional label jointly and round-trip a model through both
//! formats.
//!
//! Run with: `cargo run --release --example tree_diffusion`

use std::time::Instant;

use hessboost::diffusion::{DiffusionModel, DiffusionParams, Samples};
use hessboost::prelude::*;

mod common;
use common::lcg;

/// A standard normal from two uniforms (Box–Muller).
fn normal(next: &mut impl FnMut() -> f32) -> f32 {
    let (u1, u2) = (1.0 - next(), next());
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// `y = ±(1 + x) + 0.1 ε`: two branches whose gap grows with `x`.
fn bimodal(n: usize, seed: u64) -> Result<DMatrix> {
    let mut next = lcg(seed);
    let (mut x, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let xi = next();
        let sign = if next() < 0.5 { -1.0 } else { 1.0 };
        x.push(xi);
        y.push(sign * (1.0 + xi) + 0.1 * normal(&mut next));
    }
    DMatrix::from_dense(&x, n, 1)?.with_labels(&y)
}

/// `y = sin(2π x0) + (0.1 + x1) · (E - 1)`, `E ~ Exp(1)`: right-skewed
/// noise whose scale grows with `x1`.
fn heteroscedastic(n: usize, seed: u64) -> Result<DMatrix> {
    let mut next = lcg(seed);
    let (mut x, mut y) = (Vec::with_capacity(2 * n), Vec::with_capacity(n));
    for _ in 0..n {
        let (x0, x1) = (next(), next());
        let e = -(1.0 - next()).ln();
        x.extend_from_slice(&[x0, x1]);
        y.push((std::f32::consts::TAU * x0).sin() + (0.1 + x1) * (e - 1.0));
    }
    DMatrix::from_dense(&x, n, 2)?.with_labels(&y)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Held-out CRPS of a `dist:normal` model, the parametric baseline.
fn normal_crps(train: &DMatrix, test: &DMatrix) -> Result<f64> {
    let params = TrainingParams::builder()
        .objective("dist:normal")
        .tree_method(TreeMethod::Hist)
        .eta(0.05)
        .max_depth(4)
        .build()?;
    let model = Trainer::new(&params, train, 1000)
        .eval(test, "test")
        .early_stopping_rounds(50)
        .train()?
        .model;
    let labels = test.labels().unwrap_or_default();
    let crps: Vec<f64> = model
        .predict_distribution(test)?
        .iter()
        .zip(labels)
        .map(|(d, &y)| d.crps(f64::from(y)))
        .collect();
    Ok(mean(&crps))
}

/// Fraction of labels inside the samples' central 90% interval.
fn coverage90(samples: &Samples, labels: &[f32]) -> Result<f64> {
    let q = samples.quantiles(&[0.05, 0.95])?;
    let inside = labels
        .iter()
        .zip(q.as_chunks::<2>().0)
        .filter(|&(&y, band)| band[0] <= f64::from(y) && f64::from(y) <= band[1])
        .count();
    Ok(inside as f64 / labels.len() as f64)
}

/// A one-line histogram of the draws of row `row` on `[-2.5, 2.5]`.
fn histogram(samples: &Samples, row: usize) -> String {
    const BINS: usize = 40;
    let mut counts = [0usize; BINS];
    for &v in samples.row(row).unwrap_or_default() {
        let bin = ((f64::from(v) + 2.5) / 5.0 * BINS as f64).floor();
        if (0.0..BINS as f64).contains(&bin) {
            counts[bin as usize] += 1;
        }
    }
    let max = counts.iter().copied().max().unwrap_or(1).max(1);
    counts
        .iter()
        .map(|&c| [' ', '.', ':', '|', '#'][(c * 4).div_ceil(max)])
        .collect()
}

fn report(name: &str, params: &DiffusionParams, train: &DMatrix, test: &DMatrix) -> Result<()> {
    let start = Instant::now();
    let model = DiffusionModel::fit(params, train)?;
    let fit_time = start.elapsed();
    let start = Instant::now();
    let samples = model.sample(test, 100, 1)?;
    let sample_time = start.elapsed();
    let labels = test.labels().unwrap_or_default();
    println!(
        "  {name:<14} CRPS {:.4}  90% coverage {:.3}  ({} rounds, fit {:.1?}, 100 samples × {} rows {:.1?})",
        mean(&samples.crps(labels)?),
        coverage90(&samples, labels)?,
        model.regressor().num_boost_rounds(),
        fit_time,
        test.n_rows(),
        sample_time,
    );
    Ok(())
}

fn main() -> Result<()> {
    let (train, test) = (bimodal(2000, 1)?, bimodal(500, 2)?);
    println!("bimodal: y = ±(1 + x) + 0.1 ε");
    println!("  dist:normal    CRPS {:.4}", normal_crps(&train, &test)?);
    report("score (EDM)", &DiffusionParams::default(), &train, &test)?;
    report("treeffuser", &DiffusionParams::treeffuser(), &train, &test)?;
    report(
        "flow matching",
        &DiffusionParams::flow_matching(),
        &train,
        &test,
    )?;

    // Where do the draws land? Two probe rows, x = 0.1 and x = 0.9.
    let model = DiffusionModel::fit(&DiffusionParams::default(), &train)?;
    let probes = DMatrix::from_dense(&[0.1, 0.9], 2, 1)?;
    let samples = model.sample(&probes, 2000, 3)?;
    let q = samples.quantiles(&[0.1, 0.25, 0.5, 0.75, 0.9])?;
    for (row, x) in [0.1, 0.9].into_iter().enumerate() {
        println!(
            "  x = {x}: quantiles 10/25/50/75/90% = {:?}, modes at ±{:.1}",
            q[row * 5..row * 5 + 5]
                .iter()
                .map(|v| (v * 100.0).round() / 100.0)
                .collect::<Vec<_>>(),
            1.0 + x,
        );
        println!("    [-2.5 {} 2.5]", histogram(&samples, row));
    }

    let (train, test) = (heteroscedastic(2000, 3)?, heteroscedastic(500, 4)?);
    println!("heteroscedastic, right-skewed: y = sin(2π x0) + (0.1 + x1)(E - 1)");
    println!("  dist:normal    CRPS {:.4}", normal_crps(&train, &test)?);
    report("score (EDM)", &DiffusionParams::default(), &train, &test)?;
    report(
        "flow matching",
        &DiffusionParams::flow_matching(),
        &train,
        &test,
    )?;

    // A two-dimensional label sampled jointly: y = (u, u² + 0.05 ε) with a
    // shared latent u ~ N(x, 0.5²), so the columns are dependent.
    let n = 1500;
    let mut next = lcg(5);
    let (mut x, mut y) = (Vec::with_capacity(n), Vec::with_capacity(2 * n));
    for _ in 0..n {
        let xi = next();
        let u = xi + 0.5 * normal(&mut next);
        x.push(xi);
        y.extend_from_slice(&[u, u * u + 0.05 * normal(&mut next)]);
    }
    let data = DMatrix::from_dense(&x, n, 1)?.with_label_matrix(&y, 2)?;
    let model = DiffusionModel::fit(&DiffusionParams::treeffuser(), &data)?;
    let probe = DMatrix::from_dense(&[0.5], 1, 1)?;
    let draws = model.sample(&probe, 2000, 4)?;
    let pairs: Vec<(f64, f64)> = draws
        .values()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| (f64::from(p[0]), f64::from(p[1])))
        .collect();
    // |y2 - y1²| within each draw, and with y2 taken from the next draw:
    // joint samples keep the relation, independent marginals would not.
    let median_gap = |shift: usize| {
        let mut gaps: Vec<f64> = (0..pairs.len())
            .map(|i| (pairs[(i + shift) % pairs.len()].1 - pairs[i].0.powi(2)).abs())
            .collect();
        gaps.sort_by(f64::total_cmp);
        gaps[gaps.len() / 2]
    };
    println!(
        "2-D label at x = 0.5: mean {:?} (true [0.5, 0.5]), median |y2 - y1²| {:.3} within draws \
         vs {:.3} across draws (true noise alone: 0.034)",
        draws
            .mean()
            .iter()
            .map(|v| (v * 100.0).round() / 100.0)
            .collect::<Vec<_>>(),
        median_gap(0),
        median_gap(1),
    );

    // Both formats reproduce the samples exactly.
    let from_bytes = DiffusionModel::from_bytes(&model.to_bytes()?)?;
    let from_json = DiffusionModel::from_json(&model.to_json()?)?;
    assert_eq!(
        from_bytes.sample(&probe, 50, 9)?,
        model.sample(&probe, 50, 9)?
    );
    assert_eq!(
        from_json.sample(&probe, 50, 9)?,
        model.sample(&probe, 50, 9)?
    );
    println!(
        "saved: {} bytes native, {} bytes JSON; reloaded samples match",
        model.to_bytes()?.len(),
        model.to_json()?.len()
    );
    Ok(())
}
