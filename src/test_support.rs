//! Crate-wide unit-test helpers (compiled only for `cargo test`).

use crate::data::DMatrix;

/// A dense `rows × cols` matrix with one label per row.
pub(crate) fn labeled_dense(x: &[f32], rows: usize, cols: usize, y: &[f32]) -> DMatrix {
    DMatrix::from_dense(x, rows, cols)
        .unwrap()
        .with_labels(y)
        .unwrap()
}
