//! Explainable boosting machines: a classic cyclic EBM with early-stopped
//! outer bags, a categorical feature, and a FAST-selected pair term, its
//! shape functions, and a Boulevard EBM's confidence bands (refitted on an
//! independent sample).
//!
//! `cargo run --release --example ebm`

use hessboost::config::{BoosterKind, Ebm, GrowPolicy};
use hessboost::data::FeatureType;
use hessboost::ebm::{TermAxis, shape_functions};
use hessboost::inference::{EbmInference, KernelSolver, NoiseVariance, honest_refit};
use hessboost::prelude::*;

mod common;
use common::lcg;

/// Effects of the four categories of feature 3.
const CATEGORY_EFFECT: [f32; 4] = [0.0, 1.0, -0.5, 0.5];

/// `n` rows of three uniform features (the third missing on every tenth
/// row) and a four-category one with
/// `y = sin(2π x0) + 2 (x1 − ½)² + 1[x2 > ½] + 4 (x0 − ½)(x1 − ½) + e[x3]`
/// plus Gaussian noise of standard deviation ½.
fn simulate(n: usize, seed: u64) -> Result<DMatrix> {
    let mut uniform = lcg(seed);
    let mut x = Vec::with_capacity(4 * n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let (a, b, c) = (uniform(), uniform(), uniform());
        let category = ((uniform() * 4.0) as usize).min(3);
        let c = if i % 10 == 0 { f32::NAN } else { c };
        let step = if c > 0.5 { 1.0 } else { 0.0 };
        let f = (std::f32::consts::TAU * a).sin()
            + 2.0 * (b - 0.5).powi(2)
            + step
            + 4.0 * (a - 0.5) * (b - 0.5)
            + CATEGORY_EFFECT[category];
        x.extend_from_slice(&[a, b, c, category as f32]);
        // Box–Muller.
        let (u, v) = (uniform().max(f32::MIN_POSITIVE), uniform());
        let normal = (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos();
        y.push(f + 0.5 * normal);
    }
    DMatrix::from_dense(&x, n, 4)?
        .with_labels(&y)?
        .with_feature_types(&[
            FeatureType::Numerical,
            FeatureType::Numerical,
            FeatureType::Numerical,
            FeatureType::Categorical,
        ])
}

fn main() -> Result<()> {
    let dtrain = simulate(2000, 1)?;
    let dtest = simulate(2000, 2)?;

    // Classic EBM (InterpretML-like): cyclic three-leaf trees, 8 bags of
    // 85% of the rows, each stopped on its other 15% after 50 rounds
    // without improvement, and one FAST pair.
    let params = TrainingParams::builder()
        .booster(BoosterKind::Ebm(
            Ebm::builder()
                .outer_bags(8)
                .bag_fraction(0.85)
                .early_stopping_rounds(50)
                .interactions(1)
                .build()?,
        ))
        .eta(0.04)
        .grow_policy(GrowPolicy::LossGuide)
        .max_leaves(3)
        .min_child_weight(4.0)
        .build()?;
    let model = train(&params, &dtrain, 5000)?;
    let rmse = |m: &BoostedModel| -> Result<f64> {
        let preds = m.predict(&dtest)?;
        let labels = dtest.labels().unwrap_or_default();
        let sse: f64 = preds
            .iter()
            .zip(labels)
            .map(|(p, y)| f64::from(p - y).powi(2))
            .sum();
        Ok((sse / labels.len() as f64).sqrt())
    };
    println!(
        "classic EBM: {} trees, test RMSE {:.3}",
        model.num_trees(),
        rmse(&model)?
    );
    let shapes = shape_functions(&model)?;
    println!("intercept {:.3}", shapes.intercept);
    for term in &shapes.terms {
        let (lo, hi) = term
            .values()
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
                (l.min(v), h.max(v))
            });
        let cells: Vec<usize> = term.axes().iter().map(TermAxis::cells).collect();
        println!(
            "  term {:?}: cells {cells:?}, range [{lo:.3}, {hi:.3}]",
            term.features()
        );
    }
    let step = &shapes.terms[2];
    println!(
        "  f2(0.25) = {:.3}, f2(0.75) = {:.3}, f2(missing) = {:.3}",
        step.value(&[0.25])?,
        step.value(&[0.75])?,
        step.value(&[f32::NAN])?
    );
    let categorical = &shapes.terms[3];
    let base = categorical.value(&[0.0])?;
    for (code, effect) in CATEGORY_EFFECT.iter().enumerate() {
        println!(
            "  f3(category {code}) − f3(category 0) = {:+.3} (true {effect:+.1})",
            categorical.value(&[code as f32])? - base
        );
    }

    // Boulevard EBM: structures from `dtrain`, leaves refitted on an
    // independent sample, bands from the refit.
    let params = TrainingParams::builder()
        .booster(BoosterKind::Ebm(
            Ebm::builder().boulevard(true).interactions(1).build()?,
        ))
        .eta(1.0)
        .subsample(0.8)
        .grow_policy(GrowPolicy::LossGuide)
        .max_leaves(32)
        .min_child_weight(5.0)
        .max_bin(64)
        .build()?;
    let values = simulate(2000, 3)?;
    let model = honest_refit(&train(&params, &dtrain, 300)?, &values)?;
    println!("Boulevard EBM (refitted): test RMSE {:.3}", rmse(&model)?);
    let inference = EbmInference::fit(
        &model,
        &values,
        NoiseVariance::TrainingResiduals,
        KernelSolver::Exact,
    )?;
    println!(
        "  noise variance {:.3}, intercept standard error {:.4}",
        inference.noise_variance(),
        inference.intercept_standard_error()
    );
    let bands = inference.term_bands(0, 0.05)?;
    println!("  f0 = sin(2π x0) − mean, 95% band:");
    for x in [0.1f32, 0.25, 0.5, 0.75, 0.9] {
        let cell = bands.shape.cell(&[x])?;
        println!(
            "    x0 = {x:.2}: {:+.3} in [{:+.3}, {:+.3}] (true {:+.3})",
            bands.shape.values()[cell],
            bands.lower[cell],
            bands.upper[cell],
            (std::f32::consts::TAU * x).sin()
        );
    }
    let first = dtest.select_rows(&[0, 1, 2])?;
    for (i, (lo, hi)) in inference
        .confidence_intervals(&first, 0.05)?
        .iter()
        .enumerate()
    {
        println!("  test row {i}: f(x) in [{lo:.3}, {hi:.3}]");
    }
    Ok(())
}
