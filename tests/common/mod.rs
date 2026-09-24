//! Helpers shared by the integration tests; each test crate pulls this in
//! with `mod common;` (`tests/common/mod.rs` is not a test target itself).
#![allow(dead_code, reason = "each test crate uses a subset of the helpers")]

use hessboost::prelude::{BoostedModel, DMatrix, HessboostError, Result};

/// A dense matrix of `labels.len()` rows × `n_cols` features with `labels`.
pub fn labeled_dense(x: &[f32], n_cols: usize, labels: &[f32]) -> DMatrix {
    DMatrix::from_dense(x, labels.len(), n_cols)
        .unwrap()
        .with_labels(labels)
        .unwrap()
}

/// 64-bit LCG returning uniforms in `[0, 1)` (the top 24 bits of the state).
pub fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Row `i` of a deterministic four-feature table: values in `[0, 1)`, with
/// feature 3 missing on every 7th row.
pub fn four_features(i: usize) -> [f32; 4] {
    let a = ((i * 37) % 101) as f32 / 101.0;
    let b = ((i * 53) % 97) as f32 / 97.0;
    let c = ((i * 11) % 89) as f32 / 89.0;
    let d = if i.is_multiple_of(7) {
        f32::NAN
    } else {
        ((i * 29) % 83) as f32 / 83.0
    };
    [a, b, c, d]
}

/// The parameter name of the invalid-parameter error `result` must hold.
pub fn invalid_param<T: std::fmt::Debug>(result: Result<T>) -> &'static str {
    match result {
        Err(HessboostError::InvalidParameter { name, .. }) => name,
        other => panic!("expected an invalid-parameter error, got {other:?}"),
    }
}

/// Root mean squared error of `model`'s predictions on the labels of `data`
/// (every label cell of a label matrix).
pub fn rmse(model: &BoostedModel, data: &DMatrix) -> f64 {
    let preds = model.predict(data).unwrap();
    let labels = data.labels().unwrap();
    let sse: f64 = preds
        .iter()
        .zip(labels)
        .map(|(p, y)| f64::from(p - y).powi(2))
        .sum();
    (sse / labels.len() as f64).sqrt()
}
