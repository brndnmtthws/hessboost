//! Bring-your-own loss and metric: the custom-objective and custom-metric hooks.
//! Run: `cargo run --release --example custom_objective`.

use hessboost::metric::{CustomMetric, Metric, Rmse};
use hessboost::objective::{CustomObjective, GradPair};
use hessboost::prelude::*;

mod common;
use common::{fill_random, lcg};

fn main() -> Result<()> {
    let (n, f) = (800usize, 3usize);
    let mut rng = lcg(5);
    let mut x = vec![0f32; n * f];
    let mut y = vec![0f32; n];
    fill_random(&mut rng, &mut x);
    for i in 0..n {
        y[i] = 1.5 * x[i * f] - x[i * f + 1];
    }
    let d = DMatrix::from_dense(&x, n, f)?.with_labels(&y)?;
    let params = TrainingParams::builder().max_depth(3).eta(0.2).build()?;

    // --- Custom objective: squared error via first/second-order gradients. ---
    // Signature: (raw_margins, labels, optional_weights, out_gradients).
    let obj = CustomObjective::new(
        "my:squarederror",
        1,
        0.0,
        "rmse",
        |preds, labels, w, out| {
            for i in 0..preds.len() {
                let wi = w.map_or(1.0, |ws| ws[i]);
                out[i] = GradPair::new((preds[i] - labels[i]) * wi, wi); // grad, hess
            }
        },
    );
    let model = Trainer::new(&params, &d, 60).objective(&obj).train()?.model;
    let preds = model.predict(&d)?;
    let rmse = Rmse.eval(&preds, &y, None);
    println!("custom-objective RMSE: {rmse:.4}");

    // --- Custom metric: mean absolute error, used for early stopping. ---
    // Signature: (predictions, labels, optional_weights) -> f64; `maximize=false`.
    let mae = CustomMetric::new("my:mae", false, |p, l, _w| {
        p.iter()
            .zip(l)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
            .sum::<f64>()
            / p.len() as f64
    });
    let builtin = TrainingParams::builder()
        .objective("reg:squarederror")
        .max_depth(3)
        .eta(0.2)
        .build()?;
    let out = Trainer::new(&builtin, &d, 100)
        .eval(&d, "train")
        .early_stopping_rounds(10)
        .custom_metric(Box::new(mae))
        .train()?;
    println!(
        "custom-metric run: {} trees, last MAE = {:.4}",
        out.model.num_trees(),
        out.history.last().unwrap().scores.last().unwrap().2
    );
    Ok(())
}
