//! Binary classification with `binary:logistic`, a watched eval set, and early
//! stopping. Run: `cargo run --release --example binary_classification`.

use hessboost::metric::{Auc, Metric};
use hessboost::prelude::*;

mod common;
use common::{accuracy, fill_random, lcg};

fn main() -> Result<()> {
    // Synthetic separable-ish data: label depends on a logit of two features.
    let (n, f) = (2000usize, 4usize);
    let mut rng = lcg(42);
    let mut x = vec![0f32; n * f];
    let mut y = vec![0f32; n];
    for i in 0..n {
        fill_random(&mut rng, &mut x[i * f..(i + 1) * f]);
        let logit = 3.0 * x[i * f] - 2.0 * x[i * f + 1] - 0.5;
        let p = 1.0 / (1.0 + (-logit).exp());
        y[i] = if p > rng() { 1.0 } else { 0.0 };
    }
    // Simple train/valid split.
    let split = 1600 * f;
    let dtrain = DMatrix::from_dense(&x[..split], 1600, f)?.with_labels(&y[..1600])?;
    let dvalid = DMatrix::from_dense(&x[split..], 400, f)?.with_labels(&y[1600..])?;

    let params = TrainingParams::builder()
        .objective("binary:logistic")
        .eval_metric("logloss")
        .eval_metric("auc")
        .max_depth(4)
        .eta(0.1)
        .subsample(0.9)
        .build()?;

    // Watch `dvalid`; stop after 20 rounds without improvement on the last metric.
    let out = Trainer::new(&params, &dtrain, 500)
        .eval(&dvalid, "valid")
        .early_stopping_rounds(20)
        .train()?;
    let model = out.model;
    println!(
        "stopped at {} trees (best iteration {:?})",
        model.num_trees(),
        model.best_iteration()
    );

    let probs = model.predict(&dvalid)?; // probabilities in [0, 1]
    let classes = model.predict_class(&dvalid)?; // hard 0/1 labels
    let acc = accuracy(&classes, dvalid.labels().unwrap());
    let auc = Auc.eval(&probs, dvalid.labels().unwrap(), None);
    println!("valid accuracy {acc:.3}, AUC {auc:.3}");
    Ok(())
}
