//! Compact models for memory-constrained devices ("Boosted Trees on a Diet",
//! Herrmann et al., ICLR 2026): reuse penalties during training plus the
//! bit-packed compact layout.
//!
//! Trains a binary classifier on synthetic sensor data with increasing
//! feature (`ι`) and threshold (`ξ`) reuse penalties and reports test
//! accuracy against model size. The compact model predicts bit-identical
//! margins to the tree ensemble it was built from.
//!
//! Run with: `cargo run --release --example compact_model`

use hessboost::model::compact::CompactModel;
use hessboost::prelude::*;

mod common;
use common::{accuracy, lcg};

const N_FEATURES: usize = 16;

/// Sixteen sensor channels: integer counters, half-degree temperatures and
/// continuous readings, six of them informative.
fn dataset(n: usize, seed: u64) -> Result<DMatrix> {
    let mut next = lcg(seed);
    let mut x = Vec::with_capacity(n * N_FEATURES);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let row: Vec<f32> = (0..N_FEATURES)
            .map(|j| match j % 4 {
                0 => (next() * 16.0).floor(),             // counter 0..15
                1 => (next() * 60.0).round() * 0.5 - 5.0, // temperature, 0.5 °C steps
                _ => next() * 2.0 - 1.0,                  // continuous reading
            })
            .collect();
        let score = 0.4 * (row[0] - 7.5) / 4.0
            + (row[1] - 10.0) / 8.0
            + 1.5 * row[2] * row[3]
            + row[5] / 6.0
            + (row[6] * 3.0).sin()
            + 0.5 * f32::from(row[8] > 9.0)
            + 0.6 * (next() - 0.5);
        y.push(f32::from(score > 0.0));
        x.extend_from_slice(&row);
    }
    DMatrix::from_dense(&x, n, N_FEATURES)?.with_labels(&y)
}

fn main() -> Result<()> {
    let dtrain = dataset(8000, 7)?;
    let dtest = dataset(4000, 11)?;
    let labels = dtest.labels().unwrap_or_default().to_vec();

    println!(
        "{:>6} {:>6} | {:>8} | {:>5} {:>6} | {:>9} {:>9} {:>6}",
        "iota", "xi", "accuracy", "feats", "thresh", "native B", "compact B", "ratio"
    );
    for (iota, xi) in [
        (0.0, 0.0),
        (1.0, 1.0),
        (4.0, 4.0),
        (16.0, 16.0),
        (64.0, 16.0),
        (64.0, 64.0),
    ] {
        let params = TrainingParams::builder()
            .objective("binary:logistic")
            .max_depth(3)
            .eta(0.3)
            .toad_penalty_feature(iota)
            .toad_penalty_threshold(xi)
            .build()?;
        let model = train(&params, &dtrain, 100)?;
        let acc = accuracy(&model.predict_class(&dtest)?, &labels);

        let compact = CompactModel::from_bytes(&model.to_compact_bytes()?)?;
        assert_eq!(
            compact.predict_margin(&dtest)?,
            model.predict_margin(&dtest)?,
            "compact margins are bit-identical"
        );

        let r = model.size_report()?;
        println!(
            "{iota:>6} {xi:>6} | {acc:>8.4} | {:>5} {:>6} | {:>9} {:>9} {:>5.1}x",
            r.used_features,
            r.thresholds,
            r.native_bytes,
            r.compact_bytes,
            r.compression_ratio()
        );
    }
    Ok(())
}
