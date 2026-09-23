//! Distributional boosting (beyond XGBoost): predict a full Normal
//! distribution `N(μ(x), σ(x)²)` per row on heteroscedastic data, read off
//! prediction intervals, and compare the held-out negative log-likelihood
//! with a homoscedastic baseline (a point model plus one global residual
//! deviation). Finally, conformalize the predicted central band with CQR.
//!
//! Run with: `cargo run --release --example distributional`

use hessboost::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

mod common;
use common::lcg;

/// `y = 2 sin(2π x0) + (0.1 + x1) · ε`, `ε ~ N(0, 1)`: the noise scale grows
/// with `x1`.
fn dataset(n: usize, seed: u64) -> Result<DMatrix> {
    let mut next = lcg(seed);
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Dist::Normal {
        mu: 0.0,
        sigma: 1.0,
    };
    let mut x = Vec::with_capacity(2 * n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let (x0, x1) = (next(), next());
        let eps = noise.sample(&mut rng) as f32;
        x.extend_from_slice(&[x0, x1]);
        y.push(2.0 * (std::f32::consts::TAU * x0).sin() + (0.1 + x1) * eps);
    }
    DMatrix::from_dense(&x, n, 2)?.with_labels(&y)
}

fn mean_nll(dists: &[Dist], data: &DMatrix) -> f64 {
    let labels = data.labels().unwrap_or_default();
    let total: f64 = dists
        .iter()
        .zip(labels)
        .map(|(d, &y)| -d.log_prob(f64::from(y)))
        .sum();
    total / labels.len() as f64
}

/// Coverage and mean width of `intervals` on `data`.
fn summarize(intervals: &[(f64, f64)], data: &DMatrix) -> (f64, f64) {
    let labels = data.labels().unwrap_or_default();
    let covered = intervals
        .iter()
        .zip(labels)
        .filter(|&(&(lo, hi), &y)| lo <= f64::from(y) && f64::from(y) <= hi)
        .count();
    let width: f64 = intervals.iter().map(|(lo, hi)| hi - lo).sum();
    (
        covered as f64 / labels.len() as f64,
        width / labels.len() as f64,
    )
}

fn main() -> Result<()> {
    let dtrain = dataset(6000, 1)?;
    let dvalid = dataset(2000, 2)?;
    let dcal = dataset(2000, 3)?;
    let dtest = dataset(6000, 4)?;
    let fit = |objective: &str| -> Result<BoostedModel> {
        let params = TrainingParams::builder()
            .objective(objective)
            .tree_method(TreeMethod::Hist)
            .max_depth(3)
            .eta(0.1)
            .build()?;
        // Early stopping on the validation NLL (the `dist:*` default metric).
        Ok(train_with_eval(&params, &dtrain, 1000, &[(&dvalid, "valid")], Some(20))?.model)
    };

    // One tree per distribution parameter and round: (μ, ln σ).
    let model = fit("dist:normal")?;
    let dists = model.predict_distribution(&dtest)?;
    println!(
        "dist:normal: {} rounds, first test rows:",
        model.best_iteration().map_or(0, |b| b + 1)
    );
    for d in &dists[..3] {
        let (lo, hi) = d.interval(0.9);
        println!(
            "  mean {:+.3}  sd {:.3}  90% interval [{lo:+.3}, {hi:+.3}]",
            d.mean(),
            d.std_dev()
        );
    }

    // Homoscedastic baseline: squared-error point model, one global sigma.
    let point = fit("reg:squarederror")?;
    let fitted = point.predict(&dtrain)?;
    let labels = dtrain.labels().unwrap_or_default();
    let sigma = (fitted
        .iter()
        .zip(labels)
        .map(|(&p, &y)| f64::from(y - p).powi(2))
        .sum::<f64>()
        / labels.len() as f64)
        .sqrt();
    let baseline: Vec<Dist> = point
        .predict(&dtest)?
        .iter()
        .map(|&mu| Dist::Normal {
            mu: f64::from(mu),
            sigma,
        })
        .collect();

    println!("\nheld-out mean NLL (lower is better):");
    println!("  dist:normal            {:.4}", mean_nll(&dists, &dtest));
    println!(
        "  homoscedastic baseline {:.4}",
        mean_nll(&baseline, &dtest)
    );

    println!("\n90% intervals on the test set (coverage, mean width):");
    let raw: Vec<(f64, f64)> = dists.iter().map(|d| d.interval(0.9)).collect();
    let (c, w) = summarize(&raw, &dtest);
    println!("  predicted distribution {c:.3}  {w:.3}");
    let flat: Vec<(f64, f64)> = baseline.iter().map(|d| d.interval(0.9)).collect();
    let (c, w) = summarize(&flat, &dtest);
    println!("  homoscedastic baseline {c:.3}  {w:.3}");
    // CQR on the predicted 5% / 95% quantiles: finite-sample marginal
    // coverage regardless of how well the Normal family fits.
    let cqr = ConformalizedQuantile::calibrate_distribution(&model, &dcal, 0.1)?;
    let conformal: Vec<(f64, f64)> = cqr
        .predict_interval(&dtest)?
        .into_iter()
        .map(|(lo, hi)| (f64::from(lo), f64::from(hi)))
        .collect();
    let (c, w) = summarize(&conformal, &dtest);
    println!(
        "  conformalized (CQR)    {c:.3}  {w:.3}  (correction {:+.4})",
        cqr.correction()
    );
    Ok(())
}
