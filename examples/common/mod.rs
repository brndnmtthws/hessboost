//! Helpers shared by the crate's examples.
//!
//! Every example is its own crate root and pulls this in with `mod common;`
//! (`examples/common/mod.rs` is not auto-discovered as an example target).
//! Not every example uses every helper.
#![allow(dead_code)]

use serde::Deserialize;
use std::path::Path;

/// A tiny deterministic RNG so the examples need no dependency.
pub fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 33) as f32) / (1u32 << 31) as f32
    }
}

/// Fill `out` with successive draws from `rng`, in order.
pub fn fill_random(mut rng: impl FnMut() -> f32, out: &mut [f32]) {
    for v in out {
        *v = rng();
    }
}

/// Fraction of rows where the predicted class equals the label.
pub fn accuracy(classes: &[u32], labels: &[f32]) -> f32 {
    classes
        .iter()
        .zip(labels)
        .filter(|(c, l)| **c as f32 == **l)
        .count() as f32
        / labels.len() as f32
}

/// Typed `meta.json` prologue of a generated bench dataset (`scripts/bench_xgb.py`).
#[derive(Deserialize)]
pub struct Dataset {
    pub n_rows: usize,
    pub n_test: usize,
    pub n_cols: usize,
    pub num_round: usize,
    pub objective: String,
    pub num_class: usize,
    pub metric: String,
    pub max_depth: usize,
    pub eta: f64,
    pub lambda: f64,
    pub max_bin: usize,
    pub base_score: f64,
    pub seed: u64,
}

/// Read a little-endian `f32` blob written by the bench-data generator.
pub fn read_f32(path: &Path) -> std::io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "dataset byte count must be divisible by four",
        ));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Load the `meta.json` of a bench dataset directory.
pub fn load_meta(dir: &Path) -> Result<Dataset, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&std::fs::read(
        dir.join("meta.json"),
    )?)?)
}
