//! Fit-time and held-out quality measurements on shared binary datasets.
//!
//! Driven by `scripts/bench_xgb.py`. File I/O and test-data preparation are
//! outside the timer, and each fit constructs a fresh training DMatrix.

use sequoia_boost::metric::create_metric;
use sequoia_boost::prelude::*;
use serde::Deserialize;
use std::path::Path;
use std::time::Instant;

#[derive(Deserialize)]
struct Dataset {
    n_rows: usize,
    n_test: usize,
    n_cols: usize,
    num_round: usize,
    objective: String,
    num_class: usize,
    metric: String,
    max_depth: usize,
    eta: f64,
    lambda: f64,
    max_bin: usize,
    base_score: f64,
    seed: u64,
}

fn read_f32(path: &Path) -> std::io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "dataset byte count must be divisible by four",
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("BENCH_DIR")?;
    let dir = Path::new(&dir);
    let meta: Dataset = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    let x = read_f32(&dir.join("X.bin"))?;
    let y = read_f32(&dir.join("y.bin"))?;
    let x_test = read_f32(&dir.join("X_test.bin"))?;
    let y_test = read_f32(&dir.join("y_test.bin"))?;
    let dtest = DMatrix::from_dense(&x_test, meta.n_test, meta.n_cols)?.with_labels(&y_test)?;
    let params = TrainingParams::builder()
        .objective(meta.objective)
        .num_class(meta.num_class)
        .tree_method(TreeMethod::Hist)
        .grow_policy(GrowPolicy::DepthWise)
        .max_depth(meta.max_depth)
        .eta(meta.eta)
        .lambda(meta.lambda)
        .max_bin(meta.max_bin)
        .base_score(meta.base_score)
        .seed(meta.seed)
        .build()?;
    let metric = create_metric(&meta.metric, meta.num_class)?;
    let repeats: usize = std::env::var("BENCH_REPEATS")
        .unwrap_or_else(|_| "3".to_owned())
        .parse()?;
    if repeats == 0 {
        return Err("BENCH_REPEATS must be positive".into());
    }
    let mut samples = Vec::with_capacity(repeats);
    let mut score = 0.0;
    // The first complete fit warms the allocator and Rayon pool.
    for run in 0..=repeats {
        let start = Instant::now();
        let dtrain = DMatrix::from_dense(&x, meta.n_rows, meta.n_cols)?.with_labels(&y)?;
        let model = train(&params, &dtrain, meta.num_round)?;
        let elapsed = start.elapsed().as_secs_f64();
        if run > 0 {
            samples.push(elapsed);
        }
        if run == repeats {
            let preds = model.predict(&dtest)?;
            score = metric.eval(&preds, &y_test, None);
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "engine": "sequoia-boost",
            "threads": rayon::current_num_threads(),
            "fit_seconds": samples,
            "test_metric": meta.metric,
            "test_score": score,
        })
    );
    Ok(())
}
