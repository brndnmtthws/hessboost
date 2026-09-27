//! Typed, row-major containers returned by the prediction methods. Each owns
//! the buffer the prediction computed, without copying it.

/// Dense row-major predictions: [`n_rows`](Self::n_rows) rows of
/// [`width`](Self::width) values each.
///
/// The width is the model's outputs for margins and transformed predictions
/// (`num_class` for `multi:softprob`, the alpha count for
/// `reg:quantileerror`), 1 for `multi:softmax`'s class index, and the tree
/// count for leaf indices. [`row`](Self::row), [`rows`](Self::rows), and
/// [`get`](Self::get) read by position; [`as_slice`](Self::as_slice) and
/// [`into_vec`](Self::into_vec) expose the flat `[row][column]` buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct Predictions<T = f32> {
    values: Vec<T>,
    n_rows: usize,
    width: usize,
}

impl<T> Predictions<T> {
    /// Wraps `values`, laid out `[row][column]` with `width` columns.
    pub(crate) fn new(values: Vec<T>, n_rows: usize, width: usize) -> Self {
        debug_assert_eq!(values.len(), n_rows * width);
        Self {
            values,
            n_rows,
            width,
        }
    }

    /// Number of rows (the input's rows).
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of values per row.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Row `row`, or `None` past the last row.
    pub fn row(&self, row: usize) -> Option<&[T]> {
        if row >= self.n_rows {
            return None;
        }
        let start = row * self.width;
        Some(&self.values[start..start + self.width])
    }

    /// The rows in order.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &[T]> {
        (0..self.n_rows).map(move |row| &self.values[row * self.width..(row + 1) * self.width])
    }

    /// Value `column` of row `row`, or `None` if either is out of range.
    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        if column >= self.width {
            return None;
        }
        self.row(row).map(|values| &values[column])
    }

    /// The flat `[row][column]` buffer.
    pub fn as_slice(&self) -> &[T] {
        &self.values
    }

    /// The flat `[row][column]` buffer, without copying.
    pub fn into_vec(self) -> Vec<T> {
        self.values
    }
}

impl<T> AsRef<[T]> for Predictions<T> {
    fn as_ref(&self) -> &[T] {
        &self.values
    }
}

/// SHAP contributions: per row and output, one value per feature followed by
/// the bias, laid out `[row][output][n_features + 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Contributions {
    values: Vec<f32>,
    n_rows: usize,
    n_outputs: usize,
    n_features: usize,
}

impl Contributions {
    /// Wraps `values`, laid out `[row][output][n_features + 1]`.
    pub(crate) fn new(
        values: Vec<f32>,
        n_rows: usize,
        n_outputs: usize,
        n_features: usize,
    ) -> Self {
        debug_assert_eq!(values.len(), n_rows * n_outputs * (n_features + 1));
        Self {
            values,
            n_rows,
            n_outputs,
            n_features,
        }
    }

    /// Number of rows (the input's rows).
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of model outputs per row.
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// Number of features (each row and output has one more value, the bias).
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// The `n_features + 1` contributions of `output` for `row`, bias last,
    /// or `None` if either is out of range.
    pub fn get(&self, row: usize, output: usize) -> Option<&[f32]> {
        if row >= self.n_rows || output >= self.n_outputs {
            return None;
        }
        let width = self.n_features + 1;
        let start = (row * self.n_outputs + output) * width;
        Some(&self.values[start..start + width])
    }

    /// The bias term of `output` for `row`, or `None` if either is out of
    /// range.
    pub fn bias(&self, row: usize, output: usize) -> Option<f32> {
        self.get(row, output).map(|values| values[self.n_features])
    }

    /// The flat `[row][output][n_features + 1]` buffer.
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// The flat `[row][output][n_features + 1]` buffer, without copying.
    pub fn into_vec(self) -> Vec<f32> {
        self.values
    }
}

impl AsRef<[f32]> for Contributions {
    fn as_ref(&self) -> &[f32] {
        &self.values
    }
}

/// SHAP interaction values: per row and output, an `(n_features + 1)^2`
/// row-major matrix over the features and the bias (last), laid out
/// `[row][output][n_features + 1][n_features + 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Interactions {
    values: Vec<f32>,
    n_rows: usize,
    n_outputs: usize,
    n_features: usize,
}

impl Interactions {
    /// Wraps `values`, laid out `[row][output][n_features + 1][n_features + 1]`.
    pub(crate) fn new(
        values: Vec<f32>,
        n_rows: usize,
        n_outputs: usize,
        n_features: usize,
    ) -> Self {
        let width = n_features + 1;
        debug_assert_eq!(values.len(), n_rows * n_outputs * width * width);
        Self {
            values,
            n_rows,
            n_outputs,
            n_features,
        }
    }

    /// Number of rows (the input's rows).
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of model outputs per row.
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// Number of features (each matrix side has one more entry, the bias).
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// The row-major `(n_features + 1)^2` matrix of `output` for `row`, or
    /// `None` if either is out of range.
    pub fn get(&self, row: usize, output: usize) -> Option<&[f32]> {
        if row >= self.n_rows || output >= self.n_outputs {
            return None;
        }
        let size = (self.n_features + 1) * (self.n_features + 1);
        let start = (row * self.n_outputs + output) * size;
        Some(&self.values[start..start + size])
    }

    /// Entry `[i][j]` of the matrix of `output` for `row` (index
    /// `n_features` is the bias), or `None` if any index is out of range.
    pub fn at(&self, row: usize, output: usize, i: usize, j: usize) -> Option<f32> {
        let width = self.n_features + 1;
        if i >= width || j >= width {
            return None;
        }
        self.get(row, output).map(|matrix| matrix[i * width + j])
    }

    /// The flat `[row][output][n_features + 1][n_features + 1]` buffer.
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// The flat `[row][output][n_features + 1][n_features + 1]` buffer,
    /// without copying.
    pub fn into_vec(self) -> Vec<f32> {
        self.values
    }
}

impl AsRef<[f32]> for Interactions {
    fn as_ref(&self) -> &[f32] {
        &self.values
    }
}

#[cfg(test)]
mod tests {
    use super::{Contributions, Interactions, Predictions};

    #[test]
    fn predictions_index_rows_and_columns_within_bounds() {
        let predictions = Predictions::new(vec![1, 2, 3, 4, 5, 6], 2, 3);
        assert_eq!(predictions.row(1), Some(&[4, 5, 6][..]));
        assert_eq!(predictions.row(2), None);
        assert_eq!(predictions.get(1, 2), Some(&6));
        assert_eq!(predictions.get(2, 0), None);
        assert_eq!(predictions.get(0, 3), None);
        assert_eq!(
            predictions.rows().collect::<Vec<_>>(),
            [&[1, 2, 3][..], &[4, 5, 6][..]]
        );
        // Zero-width rows (leaf indices of a tree-less model) still count.
        let empty = Predictions::<u32>::new(Vec::new(), 2, 0);
        assert_eq!(empty.rows().len(), 2);
        assert_eq!(empty.row(1), Some(&[][..]));
        assert_eq!(empty.get(0, 0), None);
    }

    #[test]
    fn shap_accessors_keep_output_axes_and_bias_position() {
        let contributions = Contributions::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1, 2, 2);
        assert_eq!(contributions.get(0, 1), Some(&[4.0, 5.0, 6.0][..]));
        assert_eq!(contributions.bias(0, 1), Some(6.0));
        assert_eq!(contributions.get(1, 0), None);
        assert_eq!(contributions.get(0, 2), None);
        let interactions = Interactions::new((0..18).map(|value| value as f32).collect(), 1, 2, 2);
        assert_eq!(
            interactions.get(0, 1),
            Some(&[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0][..])
        );
        assert_eq!(interactions.at(0, 1, 2, 1), Some(16.0));
        assert_eq!(interactions.at(0, 1, 3, 0), None);
        assert_eq!(interactions.at(0, 2, 0, 0), None);
    }
}
