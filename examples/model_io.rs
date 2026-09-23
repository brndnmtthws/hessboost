//! Saving and loading models: native binary, native JSON, and XGBoost-format
//! JSON and UBJSON (interoperable with real XGBoost). Run:
//! `cargo run --release --example model_io`.

use hessboost::prelude::*;
use std::path::Path;

mod common;
use common::{fill_random, lcg};

fn main() -> Result<()> {
    let (n, f) = (500usize, 4usize);
    let mut rng = lcg(99);
    let mut x = vec![0f32; n * f];
    let mut y = vec![0f32; n];
    fill_random(&mut rng, &mut x);
    for i in 0..n {
        y[i] = x[i * f] - 2.0 * x[i * f + 1];
    }
    let d = DMatrix::from_dense(&x, n, f)?.with_labels(&y)?;
    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .max_depth(3)
        .eta(0.2)
        .build()?;
    let model = train(&params, &d, 40)?;
    let before = model.predict(&d)?;

    let dir = std::env::temp_dir();
    let bin = dir.join("hessboost_model.bin");
    let json = dir.join("hessboost_model.json");
    let xgb = dir.join("hessboost_xgb.json");
    let ubj = dir.join("hessboost_xgb.ubj");

    // 1) Native binary (compact) round-trip.
    model.save_binary(&bin)?;
    let m_bin = BoostedModel::load_binary(&bin)?;

    // 2) Native JSON (human-readable) round-trip.
    model.save_json(&json)?;
    let m_json = BoostedModel::load_json(&json)?;

    // 3) XGBoost-format JSON, readable by real XGBoost's `Booster.load_model`.
    model.save_xgboost_json(&xgb)?;
    let m_xgb = BoostedModel::load_xgboost_json(&xgb)?;

    // 4) XGBoost-format UBJSON (binary JSON, XGBoost's `.ubj`), the same
    //    document in XGBoost's compact encoding.
    model.save_xgboost_ubjson(&ubj)?;
    let m_ubj = BoostedModel::load_xgboost_ubjson(&ubj)?;

    for (label, m) in [
        ("binary", &m_bin),
        ("json", &m_json),
        ("xgboost-json", &m_xgb),
        ("xgboost-ubj", &m_ubj),
    ] {
        let after = m.predict(&d)?;
        let max_diff = before
            .iter()
            .zip(&after)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("{label:<13} round-trip max |Δ| = {max_diff:.2e}");
    }

    for p in [&bin, &json, &xgb, &ubj] {
        let _ = std::fs::remove_file(Path::new(p));
    }
    Ok(())
}
