//! ForestFlow / ForestDiffusion: generate synthetic tabular
//! rows and impute missing entries with per-noise-level boosted trees.
//!
//! A mixed table (two correlated continuous columns, an integer count, a
//! three-level categorical) with a class label: fit ForestFlow per class,
//! compare the synthetic rows' statistics with the real ones, then impute a
//! fifth of the entries with ForestDiffusion and compare with mean
//! imputation. Finally round-trip a model through both formats.
//!
//! Run with: `cargo run --release --example forest_flow`

use std::num::NonZeroUsize;

use hessboost::diffusion::forest::{
    ColumnKind, ForestModel, ForestParams, ImputeOptions, NoiseLevels, Repaint,
};
use hessboost::prelude::*;

mod common;
use common::lcg;

const COLS: usize = 4;

/// Rows `[a, b, count, category]` with a class label: class 1 shifts `a`,
/// raises the count, and favors category 2.
fn table(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut next = lcg(seed);
    let mut normal = move || {
        let (u1, u2) = (1.0 - next(), next());
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    };
    let (mut x, mut y) = (Vec::with_capacity(n * COLS), Vec::with_capacity(n));
    for i in 0..n {
        let class = (i % 2) as f32;
        let a = normal() + 2.0 * class;
        let b = 0.8 * a + 0.3 * normal();
        let count = (1.0 + 2.0 * class + normal().abs() * 2.0).round();
        let u = normal();
        let category = if u + class > 0.8 {
            2.0
        } else if u < -0.5 {
            0.0
        } else {
            1.0
        };
        x.extend_from_slice(&[a, b, count, category]);
        y.push(class);
    }
    (x, y)
}

/// Per class: column means, corr(a, b), and the category-2 share.
fn summary(x: &[f32], y: &[f32]) -> Vec<String> {
    (0..2)
        .map(|class| {
            let rows: Vec<&[f32]> = x
                .as_chunks::<COLS>().0.iter()
                .zip(y)
                .filter(|(_, l)| **l == class as f32)
                .map(|(r, _)| &r[..])
                .collect();
            let n = rows.len() as f64;
            let mean = |j: usize| rows.iter().map(|r| f64::from(r[j])).sum::<f64>() / n;
            let (ma, mb) = (mean(0), mean(1));
            let cov = |i: usize, mi: f64, j: usize, mj: f64| {
                rows.iter()
                    .map(|r| (f64::from(r[i]) - mi) * (f64::from(r[j]) - mj))
                    .sum::<f64>()
                    / n
            };
            let corr = cov(0, ma, 1, mb) / (cov(0, ma, 0, ma) * cov(1, mb, 1, mb)).sqrt();
            let cat2 = rows.iter().filter(|r| r[3] == 2.0).count() as f64 / n;
            format!(
                "class {class}: mean a {ma:.2}, b {mb:.2}, count {:.2}; corr(a,b) {corr:.2}; P(cat=2) {cat2:.2}",
                mean(2)
            )
        })
        .collect()
}

fn main() -> Result<()> {
    let n = 600;
    let (x, y) = table(n, 1);
    let data = DMatrix::from_dense(&x, n, COLS)?.with_labels(&y)?;
    let kinds = vec![
        ColumnKind::Continuous,
        ColumnKind::Continuous,
        ColumnKind::Integer,
        ColumnKind::Categorical,
    ];

    let mut params = ForestParams::default();
    params.column_kinds = Some(kinds.clone());
    params.n_t = NoiseLevels::new(20).unwrap();
    params.duplicate_k = NonZeroUsize::new(50).unwrap();
    let start = std::time::Instant::now();
    let flow = ForestModel::fit(&params, &data)?;
    println!("ForestFlow: fitted in {:.1?}", start.elapsed());
    let synthetic = flow.sample(n, 7)?;
    let labels = synthetic.labels().unwrap_or_default();
    println!("real:");
    for line in summary(&x, &y) {
        println!("  {line}");
    }
    println!("synthetic:");
    for line in summary(synthetic.as_slice(), labels) {
        println!("  {line}");
    }
    let integral = synthetic
        .as_slice()
        .as_chunks::<COLS>()
        .0
        .iter()
        .all(|r| r[2].fract() == 0.0 && [0.0, 1.0, 2.0].contains(&r[3]));
    println!("  integer and categorical columns decode to valid values: {integral}");

    // Impute 20% of the entries (never a whole row) with ForestDiffusion.
    let mut next = lcg(9);
    let masked: Vec<f32> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if i % COLS != 0 && next() < 0.27 {
                f32::NAN
            } else {
                v
            }
        })
        .collect();
    let holes = masked.iter().filter(|v| v.is_nan()).count();
    let mut params = ForestParams::forest_diffusion();
    params.column_kinds = Some(kinds);
    params.n_t = NoiseLevels::new(20).unwrap();
    params.duplicate_k = NonZeroUsize::new(50).unwrap();
    let start = std::time::Instant::now();
    let diffusion = ForestModel::fit(
        &params,
        &DMatrix::from_dense(&masked, n, COLS)?.with_labels(&y)?,
    )?;
    println!(
        "ForestDiffusion on {holes} missing entries: fitted in {:.1?}",
        start.elapsed()
    );
    let incomplete = DMatrix::from_dense(&masked, n, COLS)?.with_labels(&y)?;
    let imputations = diffusion.impute(
        &incomplete,
        1,
        &ImputeOptions::seeded(3).with_repaint(Repaint::default()),
    )?;
    let imputed = imputations.as_slice(); // one imputation: [row][column]
    let observed_mean = |j: usize| {
        let v: Vec<f64> = masked
            .as_chunks::<COLS>()
            .0
            .iter()
            .map(|r| f64::from(r[j]))
            .filter(|v| !v.is_nan())
            .collect();
        v.iter().sum::<f64>() / v.len() as f64
    };
    for (j, name) in [(1, "b"), (2, "count")] {
        let (mut se_forest, mut se_mean, mut k) = (0.0, 0.0, 0.0);
        for (r, row) in masked.as_chunks::<COLS>().0.iter().enumerate() {
            if row[j].is_nan() {
                let truth = f64::from(x[r * COLS + j]);
                se_forest += (f64::from(imputed[r * COLS + j]) - truth).powi(2);
                se_mean += (observed_mean(j) - truth).powi(2);
                k += 1.0;
            }
        }
        println!(
            "  RMSE on missing `{name}`: ForestDiffusion {:.3}, mean imputation {:.3}",
            (se_forest / k).sqrt(),
            (se_mean / k).sqrt()
        );
    }
    let kept = masked
        .iter()
        .zip(imputed)
        .all(|(m, i)| m.is_nan() || m == i);
    println!("  observed entries kept: {kept}");

    let from_bytes = ForestModel::from_bytes(&flow.to_bytes()?)?;
    let from_json = ForestModel::from_json(&flow.to_json()?)?;
    assert_eq!(from_bytes.sample(50, 1)?, flow.sample(50, 1)?);
    assert_eq!(from_json.sample(50, 1)?, flow.sample(50, 1)?);
    println!(
        "saved ForestFlow: {} bytes native, {} GBDTs; reloaded models sample the same rows",
        flow.to_bytes()?.len(),
        2 * flow.n_t().get()
    );
    Ok(())
}
