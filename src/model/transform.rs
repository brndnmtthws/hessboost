//! How a model's margins become its predictions: [`Transform`].

use super::{ModelObjective, Predictions, rebuild_objective};
use crate::objective::{Loss, Objective};
use rayon::prelude::*;
use std::sync::Arc;

/// Values at and above which [`Transform::apply`] transforms rows in
/// parallel (row-aligned chunks; every transform is row-local, so the
/// result does not depend on the split).
const PARALLEL_VALUES: usize = 1 << 15;

/// Rows per parallel chunk of [`Transform::apply`].
const CHUNK_ROWS: usize = 4096;

/// The prediction transform of a model's objective, which turns a row of
/// `n_outputs` margins into the row [`BoostedModel::predict`] reports. Every
/// variant transforms each row on its own, one value or one row at a time,
/// so a row's predictions never depend on the batch it is predicted in.
///
/// [`BoostedModel::predict`]: super::BoostedModel::predict
#[derive(Clone)]
pub(crate) enum Transform {
    /// Margins are the predictions: an objective the crate does not
    /// implement (a custom loss), whose model predicts margins, as XGBoost
    /// returns them.
    Margins,
    /// The built-in loss's [`Loss::pred_transform`] (identity for the
    /// objectives without a link).
    Loss(Arc<dyn Loss>),
    /// `multi:softmax`: the index of each row's largest margin, as `f32`
    /// (XGBoost's `FindMaxIndex`: the first of equal maxima, never `NaN`
    /// past the first value).
    ClassIndex,
}

impl std::fmt::Debug for Transform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Transform::Margins => f.write_str("Margins"),
            Transform::Loss(loss) => write!(f, "Loss({})", loss.name()),
            Transform::ClassIndex => f.write_str("ClassIndex"),
        }
    }
}

impl Transform {
    /// The transform of a model whose objective is `objective`, trained with
    /// `max_delta_step` on `n_targets` label columns. A built-in objective
    /// the loaders accepted always rebuilds ([`super::check_objective_width`]).
    pub(crate) fn of(objective: &ModelObjective, max_delta_step: f64, n_targets: usize) -> Self {
        if objective
            .built_in()
            .is_some_and(Objective::predicts_class_index)
        {
            return Transform::ClassIndex;
        }
        match rebuild_objective(objective, max_delta_step, n_targets) {
            Some(Ok(loss)) => Transform::Loss(loss),
            Some(Err(_)) | None => Transform::Margins,
        }
    }

    /// Values per predicted row from `n_outputs` margins: `1` for a class
    /// index, `n_outputs` otherwise.
    pub(crate) fn width(&self, n_outputs: usize) -> usize {
        match self {
            Transform::ClassIndex => 1,
            Transform::Margins | Transform::Loss(_) => n_outputs,
        }
    }

    /// Transform the `n_outputs`-wide margin rows `values` in place and
    /// return the prediction width: the first `values.len() / n_outputs *
    /// width` values then hold the predictions (for a class index, row `r`'s
    /// class lands at index `r` and the rest are left over). Inputs of more
    /// than one [`CHUNK_ROWS`] chunk and at least [`PARALLEL_VALUES`] values
    /// are transformed in parallel, by whole chunks; anything smaller, a
    /// single row included, on the calling thread.
    pub(crate) fn apply(&self, values: &mut [f32], n_outputs: usize) -> usize {
        match self {
            Transform::Margins => {}
            Transform::Loss(loss) => {
                let chunk = CHUNK_ROWS * n_outputs;
                if values.len() >= PARALLEL_VALUES
                    && values.len() > chunk
                    && rayon::current_num_threads() > 1
                {
                    values
                        .par_chunks_mut(chunk)
                        .for_each(|rows| loss.pred_transform(rows));
                } else {
                    loss.pred_transform(values);
                }
            }
            Transform::ClassIndex => {
                // Row `r`'s class lands at index `r`, at or before the row's
                // own values, so every row is read before it is overwritten.
                for r in 0..values.len() / n_outputs {
                    let class = crate::simd::argmax_scalar(&values[r * n_outputs..][..n_outputs]);
                    values[r] = class as f32;
                }
            }
        }
        self.width(n_outputs)
    }

    /// The predictions of the margin rows `margin` (its width is the
    /// model's output count), in the buffer that held them.
    pub(crate) fn predictions(&self, margin: Predictions) -> Predictions {
        let (n_rows, n_outputs) = (margin.n_rows(), margin.width());
        let mut values = margin.into_vec();
        let width = self.apply(&mut values, n_outputs);
        values.truncate(n_rows * width);
        Predictions::new(values, n_rows, width)
    }

    /// [`Self::apply`] from `margins` (`n_outputs`-wide rows) into `out` (the
    /// same rows at the prediction width), leaving `margins` as they are.
    pub(crate) fn apply_into(&self, margins: &[f32], out: &mut [f32], n_outputs: usize) {
        match self {
            Transform::ClassIndex => {
                for (class, row) in out.iter_mut().zip(margins.chunks_exact(n_outputs)) {
                    *class = crate::simd::argmax_scalar(row) as f32;
                }
            }
            Transform::Margins | Transform::Loss(_) => {
                out.copy_from_slice(margins);
                self.apply(out, n_outputs);
            }
        }
    }
}
