//! Model explainability: QuadratureTreeSHAP feature contributions and interaction values.
//! Run: `cargo run --release --example shap`.

use hessboost::prelude::*;

mod common;
use common::{fill_random, lcg};

fn main() -> Result<()> {
    // y = 2*x0 - 3*x1 + a small x0*x2 interaction.
    let (n, f) = (1000usize, 3usize);
    let mut rng = lcg(3);
    let mut x = vec![0f32; n * f];
    let mut y = vec![0f32; n];
    fill_random(&mut rng, &mut x);
    for i in 0..n {
        y[i] = 2.0 * x[i * f] - 3.0 * x[i * f + 1] + x[i * f] * x[i * f + 2];
    }
    let d = DMatrix::from_dense(&x, n, f)?.with_labels(&y)?;
    let params = TrainingParams::builder()
        .objective(Objective::SquaredError(RegLoss::default()))
        .max_depth(4)
        .eta(0.2)
        .build()?;
    let model = train(&params, &d, 80)?;

    // Contributions: n_features + 1 values per row and output, bias last.
    // Their sum equals the raw margin prediction (SHAP additivity).
    let contribs = model.predict_contribs(&d)?;
    let row0 = contribs.get(0, 0).expect("row 0 exists");
    let margin0 = *model.predict_margin(&d)?.get(0, 0).expect("row 0 exists");
    let sum0: f32 = row0.iter().sum();
    println!("row 0 SHAP contributions {row0:?}");
    println!("  sum {sum0:.4} ≈ margin {margin0:.4}");

    // Interaction values: an (n_features + 1)^2 matrix per row and output.
    // Off-diagonal (0,2) should be non-trivial thanks to the x0*x2 term.
    let inter = model.predict_interactions(&d)?;
    let value = inter.at(0, 0, 0, 2).expect("row 0, features 0 and 2 exist");
    println!("row 0 interaction[0][2] = {value:.4}");
    Ok(())
}
