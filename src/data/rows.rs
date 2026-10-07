//! [`Rows`]: the feature rows a prediction reads, a matrix's or borrowed.

use super::DMatrix;

/// The rows a prediction reads: a [`DMatrix`]'s, or borrowed dense rows
/// (`NaN` marks a missing value) without metadata, which predict exactly
/// as the same values in a matrix from [`DMatrix::from_dense`].
#[derive(Clone, Copy)]
pub(crate) enum Rows<'a> {
    Matrix(&'a DMatrix),
    Dense { values: &'a [f32], n_cols: usize },
}

impl<'a> Rows<'a> {
    pub(crate) fn n_rows(self) -> usize {
        match self {
            Rows::Matrix(data) => data.n_rows(),
            Rows::Dense { values, n_cols } => values.len() / n_cols,
        }
    }

    pub(crate) fn n_cols(self) -> usize {
        match self {
            Rows::Matrix(data) => data.n_cols(),
            Rows::Dense { n_cols, .. } => n_cols,
        }
    }

    /// The value of feature `f` in row `row`, `None` when missing.
    #[inline]
    pub(crate) fn get(self, row: usize, f: usize) -> Option<f32> {
        match self {
            Rows::Matrix(data) => data.get(row, f),
            Rows::Dense { values, n_cols } => {
                let v = values[row * n_cols + f];
                (!v.is_nan()).then_some(v)
            }
        }
    }

    /// The per-row (or per-row-and-output) margins replacing a model's
    /// intercepts: the matrix's `base_margin`; dense rows have none.
    pub(crate) fn base_margin(self) -> Option<&'a [f32]> {
        match self {
            Rows::Matrix(data) => data.base_margin(),
            Rows::Dense { .. } => None,
        }
    }
}

impl<'a> From<&'a DMatrix> for Rows<'a> {
    fn from(data: &'a DMatrix) -> Self {
        Rows::Matrix(data)
    }
}
