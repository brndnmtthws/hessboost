use super::{BoostedModel, Predictions, Shrinkage, Transform, shrink_margins};
use crate::data::DMatrix;
use crate::data::check_len;
use crate::error::{HessboostError, Result};
use crate::objective::Objective;
use crate::objective::distributional::Dist;
use crate::tree::compact::{CompactForest, FEATURE_LANES, LANES, LaneBlock, fill_lanes, key};
use crate::tree::scalar_tree_output;
use rayon::prelude::*;
use std::ops::{Bound, Range, RangeBounds};

/// Classes whose margins [`BoostedModel::predict_row_into`] forms at once,
/// on the stack, for a `multi:softmax` row (more are formed chunk by chunk).
const CLASS_CHUNK: usize = 64;

/// The boosting iterations a prediction uses (XGBoost's `iteration_range`).
///
/// Every prediction method of [`BoostedModel`] takes one as
/// `impl Into<Iterations>`, so a call passes either [`Iterations::Best`] or
/// any Rust range of iteration indices: `..` is the whole model regardless
/// of early stopping, `..n` the first `n` iterations, `2..5` iterations 2
/// to 4 (`2..=4` and `2..` work too). The intercept / dataset
/// `base_margin` is always included, so an empty range predicts it alone.
///
/// Leaf, contribution, and interaction predictions accept only ranges
/// starting at iteration 0 (as in XGBoost); [`BoostedModel::slice`] cuts
/// out a later start. [`Iterations::Best`] always starts there, so every
/// prediction accepts it. A `gblinear` model has no boosting iterations to
/// select and accepts only `..` (and [`Iterations::Best`], which is `..`
/// for it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Iterations {
    /// The effective iterations: `..best_iteration + 1` after early
    /// stopping, else every iteration (`..`).
    #[default]
    Best,
    /// The iterations between two bounds, as any Rust range describes
    /// them (the [`From`] conversion builds this from one).
    Range {
        /// First iteration (`Unbounded`: iteration 0).
        start: Bound<usize>,
        /// End of the range (`Unbounded`: through the last iteration).
        end: Bound<usize>,
    },
}

impl<R: RangeBounds<usize>> From<R> for Iterations {
    fn from(range: R) -> Self {
        Iterations::Range {
            start: range.start_bound().cloned(),
            end: range.end_bound().cloned(),
        }
    }
}

impl BoostedModel {
    /// Margins from the trees `trees` (tree ids) without validating `data`.
    pub(crate) fn margin_from_trees(
        &self,
        data: &DMatrix,
        trees: std::ops::Range<usize>,
    ) -> Vec<f32> {
        let n = data.n_rows();
        let k = self.n_outputs();
        // Initialize from the dataset's per-instance base margin when present
        // (it overrides the per-output intercepts, matching XGBoost); otherwise
        // use the trained global bias.
        let mut out = self.initial_margins(data);
        // A gblinear model predicts from its linear parameters and ignores the
        // (empty) tree ensemble: margin(row, k) = base_score[k] + bias[k] +
        // Σ_f weights[f][k] * x[row, f], with missing features contributing 0.
        if let Some(lm) = &self.linear {
            // Each row adds its bias, then its features in ascending order.
            let add_row = |(row, margin): (usize, &mut [f32])| {
                for (m, &b) in margin.iter_mut().zip(&lm.bias) {
                    *m += b;
                }
                self.for_each_linear_contribution(data, row, |_f, c, v| {
                    margin[c] += v as f32;
                });
            };
            // Rows are independent, so parallel rows give the serial result.
            if n >= PREDICT_BLOCK_ROWS && rayon::current_num_threads() > 1 {
                out.par_chunks_mut(k)
                    .with_min_len(PREDICT_BLOCK_ROWS)
                    .enumerate()
                    .for_each(add_row);
            } else {
                out.chunks_mut(k).enumerate().for_each(add_row);
            }
            return out;
        }
        self.accumulate_forest(data, &mut out, trees, |ti| self.tree_weight(ti));
        out
    }

    /// Sum `weight(t) * leaf(row, t)` into `out[row * k + tree_output(t)]` for
    /// the trees in `trees`, where `k` is the output count. Rows are traversed
    /// in cache-friendly blocks (parallel across blocks); per (row, output)
    /// slot the trees are still summed in ascending order, so the result is
    /// bit-identical to the sequential tree-outer loop. Ensembles with linear
    /// leaves take a per-row path instead, since the compact forest stores
    /// constant leaf values only.
    pub(super) fn accumulate_forest(
        &self,
        data: &DMatrix,
        out: &mut [f32],
        trees: std::ops::Range<usize>,
        weight: impl Fn(usize) -> f32 + Sync,
    ) {
        let k = self.n_outputs();
        if self.has_vector_leaves() {
            self.traverse_blocks(
                data,
                out,
                k,
                trees.clone(),
                |block, forest, r, out_row| {
                    block.accumulate_row_vector(forest, r, trees.clone(), &weight, out_row);
                },
                |block, forest, ti, rows, out_block, stride| {
                    block.accumulate_vector(forest, ti, rows, weight(ti), out_block, stride);
                },
            );
            return;
        }
        if self.trees[trees.clone()]
            .iter()
            .any(|tree| tree.linear_leaves().is_some())
        {
            crate::tree::linear::accumulate_forest(
                &self.trees,
                trees,
                |t| self.tree_output(t),
                data,
                out,
                k,
                weight,
            );
            return;
        }
        let parallel = self.num_parallel_tree;
        self.traverse_blocks(
            data,
            out,
            k,
            trees.clone(),
            |block, forest, r, out_row| {
                block.accumulate_row(forest, r, trees.clone(), parallel, &weight, out_row);
            },
            |block, forest, ti, rows, out_block, stride| {
                block.accumulate(
                    forest,
                    ti,
                    rows,
                    weight(ti),
                    &mut out_block[self.tree_output(ti)..],
                    stride,
                );
            },
        );
    }

    /// Block-parallel traversal of the trees in `trees` over `data`, writing
    /// into `out` laid out `[row][stride]`: `row_op` handles one loaded row of
    /// the small-batch path, `tree_op` one tree over a loaded block of rows.
    /// Tiny batches (online serving) skip the thread pool and overlap the
    /// trees of each row instead of the rows of each tree; larger inputs
    /// process rows in blocks whose feature rows stay in cache while every
    /// tree walks them, with blocks running in parallel.
    fn traverse_blocks<T: Send>(
        &self,
        data: &DMatrix,
        out: &mut [T],
        stride: usize,
        trees: std::ops::Range<usize>,
        row_op: impl Fn(&RowBlock, &CompactForest, usize, &mut [T]),
        tree_op: impl Fn(&RowBlock, &CompactForest, usize, usize, &mut [T], usize) + Sync,
    ) {
        let n = data.n_rows();
        let forest = self.compact_forest();
        if n < LANES {
            let mut block = RowBlock::new(data);
            block.load(0, n);
            for (r, out_row) in out.chunks_exact_mut(stride).enumerate() {
                row_op(&block, forest, r, out_row);
            }
            return;
        }
        out.par_chunks_mut(PREDICT_BLOCK_ROWS * stride)
            .enumerate()
            .for_each_init(
                || RowBlock::new(data),
                |block, (bi, out_block)| {
                    let start = bi * PREDICT_BLOCK_ROWS;
                    let rows = out_block.len() / stride;
                    block.load(start, rows);
                    for ti in trees.clone() {
                        tree_op(block, forest, ti, rows, out_block, stride);
                    }
                },
            );
    }

    /// Predictions in the objective's reported space from the boosting
    /// `iterations` ([`Iterations`]; [`Iterations::Best`] for the effective
    /// ones). `multi:softprob` returns an `n_rows × num_class` probability
    /// matrix while `multi:softmax` returns one class index per row,
    /// encoded as `f32` (the first of equal largest margins, as XGBoost).
    ///
    /// They are the margins of [`Self::predict_margin`] through
    /// [`Self::transform_margins_into`], which transforms every value (every
    /// row, for softmax and the sorted quantiles) on its own: a row's
    /// predictions are bit for bit the same in every batch, including
    /// [`Self::predict_row`]'s single row.
    ///
    /// # Errors
    ///
    /// As for [`Self::predict_margin`].
    pub fn predict(
        &self,
        data: &DMatrix,
        iterations: impl Into<Iterations>,
    ) -> Result<Predictions> {
        Ok(self.transform_margins(self.predict_margin(data, iterations)?))
    }

    /// For multiclass, the predicted class index per row (argmax over classes).
    /// For single-output models this returns the transformed prediction rounded
    /// to the nearest class at 0.5. A multi-target model (label matrix) is
    /// multi-label: each target is thresholded at 0.5 independently, giving
    /// `n_rows × n_targets` decisions laid out `[row][target]`. Predicts
    /// from the boosting `iterations` ([`Iterations`]).
    ///
    /// # Errors
    ///
    /// As for [`Self::predict_margin`].
    pub fn predict_class(
        &self,
        data: &DMatrix,
        iterations: impl Into<Iterations>,
    ) -> Result<Predictions<u32>> {
        Ok(self.classes(&self.predict(data, iterations)?))
    }

    /// The predictions of [`Self::predict`] from the margins of
    /// [`Self::predict_margin`] (shared with the GPU predictors).
    pub(crate) fn transform_margins(&self, margin: Predictions) -> Predictions {
        self.transform().predictions(margin)
    }

    /// Values per row of [`Self::predict`] (and of
    /// [`Self::predict_row_into`]'s output): [`Self::n_outputs`], except `1`
    /// for `multi:softmax`, which predicts one class index per row.
    pub fn prediction_width(&self) -> usize {
        self.transform().width(self.n_outputs)
    }

    /// The prediction of one margin of a single-output model, by the
    /// objective's transform (the sigmoid of `binary:logistic`, the `exp` of
    /// the log-link objectives, the identity for squared error, ...): bit
    /// for bit what [`Self::predict`] reports for a row with that margin.
    ///
    /// # Errors
    ///
    /// [`HessboostError::IncompatibleModel`] (`outputs`) for a model with
    /// several outputs; use [`Self::transform_margins_into`].
    pub fn transform_margin(&self, margin: f32) -> Result<f32> {
        one_value_per_row(self.n_outputs, "transform_margins_into")?;
        let mut value = [margin];
        self.transform().apply(&mut value, 1);
        Ok(value[0])
    }

    /// Write the predictions of rows of margins into `out`: `margins` holds
    /// whole rows of [`Self::n_outputs`] margins (`[row][output]`, as
    /// [`Self::predict_margin`] lays them out) and `out` the same rows of
    /// [`Self::prediction_width`] values. The transform of
    /// [`Self::predict`], without allocating: `predict` of any batch is bit
    /// for bit this transform of its `predict_margin`.
    ///
    /// # Errors
    ///
    /// [`HessboostError::DimensionMismatch`] when `margins` is not whole rows
    /// or `out` does not hold one prediction row per margin row.
    pub fn transform_margins_into(&self, margins: &[f32], out: &mut [f32]) -> Result<()> {
        let k = self.n_outputs;
        if !margins.len().is_multiple_of(k) {
            return Err(HessboostError::dimension_mismatch(
                "margins (whole rows of the model's outputs)",
                margins.len().next_multiple_of(k),
                margins.len(),
            ));
        }
        let transform = self.transform();
        let rows = margins.len() / k;
        check_len(
            "transformed predictions",
            out.len(),
            rows * transform.width(k),
        )?;
        transform.apply_into(margins, out, k);
        Ok(())
    }

    /// The margin of one row of feature values, for a single-output model:
    /// [`Self::predict_margin_row_into`] into one value.
    ///
    /// # Errors
    ///
    /// [`HessboostError::IncompatibleModel`] (`outputs`) for a model with
    /// several outputs, plus the errors of [`Self::predict_margin_row_into`].
    pub fn predict_margin_row(
        &self,
        row: &[f32],
        iterations: impl Into<Iterations>,
    ) -> Result<f32> {
        one_value_per_row(self.n_outputs, "predict_margin_row_into")?;
        let mut out = [0.0];
        self.predict_margin_row_into(row, iterations, &mut out)?;
        Ok(out[0])
    }

    /// Write the margins of one row of feature values into `out` (one per
    /// output, [`Self::n_outputs`]): `row` holds one value per feature, `NaN`
    /// for a missing one, as in a matrix from [`DMatrix::from_dense`]. Bit for
    /// bit that row's [`Self::predict_margin`] in any batch (tree weights,
    /// model shrinkage, missing-value routing, linear leaves and
    /// [`Iterations::Best`] included), without building a matrix and without
    /// allocating once the model's first prediction has laid out its trees.
    /// The intercepts are the model's: a row has no `base_margin`.
    ///
    /// # Errors
    ///
    /// [`HessboostError::DimensionMismatch`] when `row` does not hold one
    /// value per feature or `out` one per output,
    /// [`HessboostError::InvalidData`] (`row`) for an infinite value, plus the
    /// errors of [`Self::predict_margin`] for `iterations`.
    pub fn predict_margin_row_into(
        &self,
        row: &[f32],
        iterations: impl Into<Iterations>,
        out: &mut [f32],
    ) -> Result<()> {
        let iterations = self.row_request(row, iterations.into())?;
        check_len("row margins", out.len(), self.n_outputs)?;
        self.margin_row(row, iterations, 0..self.n_outputs, out);
        Ok(())
    }

    /// The prediction of one row of feature values, for a model that
    /// predicts one value per row: [`Self::predict_row_into`] into one value.
    ///
    /// # Errors
    ///
    /// [`HessboostError::IncompatibleModel`] (`outputs`) for a model whose
    /// [`Self::prediction_width`] is not 1, plus the errors of
    /// [`Self::predict_row_into`].
    pub fn predict_row(&self, row: &[f32], iterations: impl Into<Iterations>) -> Result<f32> {
        one_value_per_row(self.prediction_width(), "predict_row_into")?;
        let mut out = [0.0];
        self.predict_row_into(row, iterations, &mut out)?;
        Ok(out[0])
    }

    /// Write the predictions of one row of feature values into `out`
    /// ([`Self::prediction_width`] values: one per output, a `dist:*` model's
    /// natural parameters, a `multi:softprob` row's probabilities, or
    /// `multi:softmax`'s class index): [`Self::predict_margin_row_into`]'s
    /// margins through [`Self::transform_margins_into`], so bit for bit that
    /// row's [`Self::predict`] in any batch, without allocating.
    ///
    /// # Errors
    ///
    /// [`HessboostError::DimensionMismatch`] when `row` does not hold one
    /// value per feature or `out` one per predicted value,
    /// [`HessboostError::InvalidData`] (`row`) for an infinite value, plus the
    /// errors of [`Self::predict_margin`] for `iterations`.
    ///
    /// ```
    /// use hessboost::objective::{Multiclass, Objective};
    /// use hessboost::prelude::*;
    ///
    /// # fn main() -> Result<()> {
    /// let x: Vec<f32> = (0..60).map(|i| (i % 30) as f32).collect();
    /// let y: Vec<f32> = (0..30).map(|i| (i % 3) as f32).collect();
    /// let dtrain = DMatrix::from_dense(&x, 30, 2)?.with_labels(&y)?;
    /// let params = TrainingParams::builder()
    ///     .objective(Objective::Softprob(Multiclass::new(3)?))
    ///     .build()?;
    /// let model = train(&params, &dtrain, 5)?;
    ///
    /// let mut probabilities = [0.0; 3];
    /// model.predict_row_into(&[4.0, f32::NAN], Iterations::Best, &mut probabilities)?;
    /// let batch = model.predict(&DMatrix::from_dense(&[4.0, f32::NAN], 1, 2)?, Iterations::Best)?;
    /// assert_eq!(probabilities, batch.as_slice());
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict_row_into(
        &self,
        row: &[f32],
        iterations: impl Into<Iterations>,
        out: &mut [f32],
    ) -> Result<()> {
        let iterations = self.row_request(row, iterations.into())?;
        let transform = self.transform();
        check_len(
            "row predictions",
            out.len(),
            transform.width(self.n_outputs),
        )?;
        if let Transform::ClassIndex = transform {
            out[0] = self.row_class(row, iterations) as f32;
        } else {
            self.margin_row(row, iterations, 0..self.n_outputs, out);
            transform.apply(out, self.n_outputs);
        }
        Ok(())
    }

    /// Check one row of features (one finite or `NaN` value per feature, as
    /// a dense matrix requires) and resolve `iterations` for it as
    /// [`Self::predict_margin`] does.
    fn row_request(&self, row: &[f32], iterations: Iterations) -> Result<Range<usize>> {
        check_len("prediction feature count", row.len(), self.n_features)?;
        if row.iter().any(|v| v.is_infinite()) {
            return Err(HessboostError::invalid_data(
                "row",
                "non-missing feature values must be finite",
            ));
        }
        self.margin_iterations(iterations)
    }

    /// `multi:softmax`'s class of one row from the resolved `iterations`:
    /// XGBoost's `FindMaxIndex` over its margins (the first of equal maxima,
    /// as [`Transform::ClassIndex`] picks it), whose margins are formed
    /// [`CLASS_CHUNK`] classes at a time on the stack.
    fn row_class(&self, row: &[f32], iterations: Range<usize>) -> usize {
        let k = self.n_outputs;
        let mut chunk = [0.0f32; CLASS_CHUNK];
        let (mut best, mut best_margin) = (0, f32::NAN);
        for start in (0..k).step_by(CLASS_CHUNK) {
            let end = (start + CLASS_CHUNK).min(k);
            let margins = &mut chunk[..end - start];
            self.margin_row(row, iterations.clone(), start..end, margins);
            for (class, &margin) in (start..end).zip(margins.iter()) {
                if class == 0 || margin > best_margin {
                    (best, best_margin) = (class, margin);
                }
            }
        }
        best
    }

    /// The margins of outputs `outputs` of one dense `row` (`NaN` = missing)
    /// from the resolved `iterations`, into `out` (one per output), with the
    /// batch paths' arithmetic cell by cell: the intercept, then gblinear's
    /// bias and its feature terms in feature order, or every tree's weighted
    /// value in tree order (for a shrunk model, each iteration's shrink
    /// before its trees).
    fn margin_row(
        &self,
        row: &[f32],
        iterations: Range<usize>,
        outputs: Range<usize>,
        out: &mut [f32],
    ) {
        debug_assert_eq!(out.len(), outputs.len());
        let k = self.n_outputs;
        if let Some(linear) = &self.linear {
            let base = &self.base_score[outputs.clone()];
            for ((m, &b), &bias) in out
                .iter_mut()
                .zip(base)
                .zip(&linear.bias()[outputs.clone()])
            {
                *m = b + bias;
            }
            for (f, &x) in row.iter().enumerate() {
                if x.is_nan() {
                    continue;
                }
                let weights = &linear.weights()[f * k..(f + 1) * k][outputs.clone()];
                for (m, &w) in out.iter_mut().zip(weights) {
                    *m += (f64::from(w) * f64::from(x)) as f32;
                }
            }
            return;
        }
        let per = self.trees_per_iteration();
        if let Some(shrinkage) = &self.shrinkage {
            out.copy_from_slice(&shrinkage.base_scores()[outputs.clone()]);
            for i in iterations {
                shrink_margins(out, shrinkage.factors()[i]);
                self.add_row_trees(row, i * per..(i + 1) * per, &outputs, |_| 1.0, out);
            }
            return;
        }
        out.copy_from_slice(&self.base_score[outputs.clone()]);
        let weight = |t| self.tree_weight(t);
        let trees = self.iteration_trees(iterations);
        if outputs.len() == k {
            self.add_row_trees(row, trees, &outputs, weight, out);
        } else {
            for i in trees.start / per..trees.end / per {
                self.add_row_trees(row, i * per..(i + 1) * per, &outputs, weight, out);
            }
        }
    }

    /// `out[o - outputs.start] += weight(t) * tree_t(row)[o]` for the outputs
    /// `o` in `outputs` and the trees `t` of `trees` (whole iterations, or
    /// one iteration when `outputs` is not every output) feeding them, in
    /// tree order per output.
    fn add_row_trees(
        &self,
        row: &[f32],
        trees: Range<usize>,
        outputs: &Range<usize>,
        weight: impl Fn(usize) -> f32,
        out: &mut [f32],
    ) {
        let forest = self.compact_forest();
        let k = self.n_outputs;
        if self.has_vector_leaves() {
            forest.walk_row(row, trees, |t, leaf| {
                let w = weight(t);
                let values = &forest.leaf_vector(leaf, k)[outputs.clone()];
                for (o, &v) in out.iter_mut().zip(values) {
                    *o += w * v;
                }
            });
            return;
        }
        // A scalar tree feeds one output; within an iteration the trees of
        // the outputs `outputs` are contiguous.
        let parallel = self.num_parallel_tree;
        let trees = if outputs.len() == k {
            trees
        } else {
            trees.start + outputs.start * parallel..trees.start + outputs.end * parallel
        };
        let slot = |t: usize| scalar_tree_output(t, parallel, k) - outputs.start;
        if self.trees[trees.clone()]
            .iter()
            .any(|tree| tree.linear_leaves().is_some())
        {
            for t in trees {
                out[slot(t)] += weight(t) * self.trees[t].predict_dense(row);
            }
        } else {
            forest.walk_row(row, trees, |t, leaf| {
                out[slot(t)] += weight(t) * forest.leaf_value(leaf);
            });
        }
    }

    /// The class decisions of [`Self::predict_class`] from the
    /// predictions `probs` of [`Self::predict`] (shared with the GPU
    /// predictor).
    pub(crate) fn classes(&self, probs: &Predictions) -> Predictions<u32> {
        let k = self.n_outputs();
        if k == 1 || self.n_targets > 1 {
            let values = probs
                .as_slice()
                .iter()
                .map(|&p| u32::from(p > 0.5))
                .collect();
            return Predictions::new(values, probs.n_rows(), probs.width());
        }
        if self
            .objective
            .built_in()
            .is_some_and(Objective::predicts_class_index)
        {
            let values = probs.as_slice().iter().map(|&class| class as u32).collect();
            return Predictions::new(values, probs.n_rows(), 1);
        }
        let values = probs
            .rows()
            .map(|row| crate::simd::argmax_scalar(row) as u32)
            .collect();
        Predictions::new(values, probs.n_rows(), 1)
    }

    /// Per-row leaf indices for each tree of the boosting `iterations`
    /// (shape `n_rows × trees`, row-major, trees in [`Self::trees`] order).
    /// `..` walks every tree regardless of early stopping;
    /// [`Iterations::Best`] only the trees through `best_iteration`, like
    /// the other predictions. As in XGBoost a range must start at
    /// iteration `0`; use [`Self::slice`] for a later start.
    ///
    /// # Errors
    ///
    /// [`HessboostError::DimensionMismatch`] when `data` does not fit the
    /// model, [`HessboostError::IncompatibleModel`] (`iterations`) for
    /// `iterations` past the model's, [`HessboostError::InvalidParameter`]
    /// (`iterations`) for an inverted range or one starting after
    /// iteration `0`.
    #[allow(
        clippy::redundant_closure_for_method_calls,
        reason = "the method path is not general enough over the block lifetime"
    )]
    pub fn predict_leaf(
        &self,
        data: &DMatrix,
        iterations: impl Into<Iterations>,
    ) -> Result<Predictions<u32>> {
        self.validate_prediction_data(data)?;
        let t = self.prefix_trees(iterations.into(), "leaf prediction")?;
        let n = data.n_rows();
        let mut out = vec![0u32; n * t];
        if t == 0 {
            return Ok(Predictions::new(out, n, t));
        }
        self.traverse_blocks(
            data,
            &mut out,
            t,
            0..t,
            |block, forest, r, out_row| block.original_leaf_ids_for_row(forest, r, out_row),
            |block, forest, ti, rows, out_block, stride| {
                block.original_leaf_ids(forest, ti, rows, &mut out_block[ti..], stride);
            },
        );
        Ok(Predictions::new(out, n, t))
    }

    /// Raw margin predictions from the boosting `iterations`
    /// ([`Iterations`]: [`Iterations::Best`] for the effective iterations,
    /// `[0, best_iteration + 1)` after early stopping, else all; or a range
    /// such as `..`, `..n`, `2..5`), laid out `[row][output]` (`n_outputs`
    /// wide).
    ///
    /// For a model trained with model shrinkage
    /// ([`TrainingParams::model_shrink`](crate::config::TrainingParams::model_shrink)),
    /// `..n` is the model after `n` iterations, predicted with training's
    /// arithmetic (bit for bit the margins training reached after `n`
    /// rounds, and the predictions of the same run stopped there): from the
    /// unshrunk intercepts, every iteration shrinks the margins and adds its
    /// trees. A dataset's `base_margin` replaces the shrunk intercepts (it
    /// is added to the trees' shrunk sum). Ranges starting after iteration 0
    /// are refused, since the ensemble is rescaled every iteration.
    ///
    /// # Errors
    ///
    /// [`HessboostError::DimensionMismatch`] when `data` does not fit the
    /// model, [`HessboostError::IncompatibleModel`] (`iterations`) for
    /// `iterations` past the model's (or, for a shrunk model, one starting
    /// after iteration 0), [`HessboostError::InvalidParameter`]
    /// (`iterations`) for an inverted or overflowing range.
    pub fn predict_margin(
        &self,
        data: &DMatrix,
        iterations: impl Into<Iterations>,
    ) -> Result<Predictions> {
        self.validate_prediction_data(data)?;
        let iterations = self.margin_iterations(iterations.into())?;
        let values = match &self.shrinkage {
            Some(shrinkage) => self.shrunk_margins(shrinkage, data, iterations.end),
            None => self.margin_from_trees(data, self.iteration_trees(iterations)),
        };
        Ok(Predictions::new(values, data.n_rows(), self.n_outputs()))
    }

    /// The iterations of a margin prediction ([`Self::resolve_iterations`]),
    /// refusing a shrunk model's ranges that start after iteration 0: every
    /// later iteration rescaled the earlier ones, so `a..b` alone is no
    /// model.
    fn margin_iterations(&self, iterations: Iterations) -> Result<Range<usize>> {
        let iterations = self.resolve_iterations(iterations, "iterations")?;
        if self.shrinkage.is_some() && iterations.start != 0 {
            return Err(HessboostError::incompatible_model(
                "iterations",
                format!(
                    "a model trained with model shrinkage rescales its earlier iterations \
                     every iteration, so {}..{} has no meaning; use a range starting at 0",
                    iterations.start, iterations.end
                ),
            ));
        }
        Ok(iterations)
    }

    /// The margins of a shrunk model after its first `k` iterations, with
    /// training's arithmetic ([`shrinkage`]): from
    /// [`Shrinkage::start_margins`], every iteration shrinks every margin
    /// ([`shrink_margins`]) and then adds its trees, each once per cell in
    /// tree order (the blocked traversal keeps that order per cell).
    fn shrunk_margins(&self, shrinkage: &Shrinkage, data: &DMatrix, k: usize) -> Vec<f32> {
        let mut out = shrinkage.start_margins(data);
        self.shrink_and_add(shrinkage, data, &mut out, 0..k);
        shrinkage.finish_margins(data, &mut out);
        out
    }

    /// Continue the shrunk margins `out` (before
    /// [`Shrinkage::finish_margins`]) through `iterations`: every iteration
    /// shrinks every margin and then adds its trees. Running `a..b` and then
    /// `b..c` is running `a..c`: each cell sees the same operations in the
    /// same order.
    fn shrink_and_add(
        &self,
        shrinkage: &Shrinkage,
        data: &DMatrix,
        out: &mut [f32],
        iterations: Range<usize>,
    ) {
        let factors = shrinkage.factors();
        let per = self.trees_per_iteration();
        let n_out = self.n_outputs();
        let trees = iterations.start * per..iterations.end * per;
        let unit = |_: usize| 1.0f32;
        if self.trees[trees.clone()]
            .iter()
            .any(|tree| tree.linear_leaves().is_some())
        {
            for i in iterations {
                shrink_margins(out, factors[i]);
                crate::tree::linear::accumulate_forest(
                    &self.trees,
                    i * per..(i + 1) * per,
                    |t| self.tree_output(t),
                    data,
                    out,
                    n_out,
                    unit,
                );
            }
        } else {
            let vector = self.has_vector_leaves();
            let parallel = self.num_parallel_tree;
            self.traverse_blocks(
                data,
                out,
                n_out,
                trees,
                |block, forest, r, out_row| {
                    for i in iterations.clone() {
                        shrink_margins(out_row, factors[i]);
                        let layer = i * per..(i + 1) * per;
                        if vector {
                            block.accumulate_row_vector(forest, r, layer, unit, out_row);
                        } else {
                            block.accumulate_row(forest, r, layer, parallel, unit, out_row);
                        }
                    }
                },
                |block, forest, ti, rows, out_block, stride| {
                    if ti % per == 0 {
                        shrink_margins(&mut out_block[..rows * stride], factors[ti / per]);
                    }
                    if vector {
                        block.accumulate_vector(forest, ti, rows, 1.0, out_block, stride);
                    } else {
                        let out = &mut out_block[self.tree_output(ti)..];
                        block.accumulate(forest, ti, rows, 1.0, out, stride);
                    }
                },
            );
        }
    }

    /// The margins of the models after each of the `ends` iterations
    /// (ascending, each at most [`Self::num_boost_rounds`]), concatenated
    /// `[member][row][output]`: bit for bit
    /// [`Self::predict_margin`]`(data, ..end)` of every `end`, from one
    /// pass over the trees. The trees of `..ends[0]`, then those up to
    /// `ends[1]`, and so on, add onto the running margins, which are copied
    /// out at every end; the accumulation adds each tree once per cell in
    /// tree order either way (for a shrunk model, each iteration shrinks
    /// before it adds), so a split pass sums what one pass over `..end`
    /// does.
    pub(crate) fn prefix_margins(&self, data: &DMatrix, ends: &[usize]) -> Result<Vec<f32>> {
        self.validate_prediction_data(data)?;
        let cells = data.n_rows() * self.n_outputs();
        let mut margins = Vec::with_capacity(ends.len() * cells);
        let mut start = 0;
        if let Some(shrinkage) = &self.shrinkage {
            let mut out = shrinkage.start_margins(data);
            for &end in ends {
                self.shrink_and_add(shrinkage, data, &mut out, start..end);
                let member = margins.len();
                margins.extend_from_slice(&out);
                shrinkage.finish_margins(data, &mut margins[member..]);
                start = end;
            }
        } else {
            let mut out = self.initial_margins(data);
            for &end in ends {
                let trees = self.iteration_trees(start..end);
                self.accumulate_forest(data, &mut out, trees, |ti| self.tree_weight(ti));
                margins.extend_from_slice(&out);
                start = end;
            }
        }
        Ok(margins)
    }

    /// Refuse anything but the whole ensemble of a shrunk model, for the
    /// predictions (`what`) that read the stored tree weights directly;
    /// [`Self::slice`]`(..k, 1)` builds the model after `k` iterations.
    pub(crate) fn refuse_partial_shrunk_range(
        &self,
        trees: &Range<usize>,
        what: &str,
    ) -> Result<()> {
        if self.shrinkage.is_none() || (trees.start == 0 && trees.end == self.trees.len()) {
            return Ok(());
        }
        Err(HessboostError::incompatible_model(
            "iterations",
            format!(
                "{what} of a model trained with model shrinkage covers the whole ensemble \
                 only; `slice(..k, 1)` builds the model after `k` iterations"
            ),
        ))
    }

    /// The predicted distribution of every row for a model trained with a
    /// distributional `dist:*` objective (see
    /// [`crate::objective::distributional`]): one [`Dist`] per row, with its
    /// mean, variance, CDF, quantiles, log density, CRPS, intervals and
    /// sampling. The margins, from the boosting `iterations`
    /// ([`Iterations`]), are mapped through the links in `f64`.
    ///
    /// # Errors
    ///
    /// [`HessboostError::IncompatibleModel`] (`objective`) if the model's
    /// objective is not a `dist:*` objective, plus the errors of
    /// [`Self::predict_margin`].
    pub fn predict_distribution(
        &self,
        data: &DMatrix,
        iterations: impl Into<Iterations>,
    ) -> Result<Vec<Dist>> {
        let family = self
            .objective
            .built_in()
            .and_then(Objective::dist_family)
            .ok_or_else(|| {
                HessboostError::incompatible_model(
                    "objective",
                    format!(
                        "`{}` does not predict distributions; train with a `dist:*` objective",
                        self.objective.name()
                    ),
                )
            })?;
        let margin = self.predict_margin(data, iterations)?;
        Ok(margin
            .as_slice()
            .chunks_exact(family.n_params())
            .map(|row| {
                let mut eta = [0.0; 2];
                for (e, &m) in eta.iter_mut().zip(row) {
                    *e = f64::from(m);
                }
                family.dist_from_margins(&eta[..row.len()])
            })
            .collect())
    }
}

/// Margin buffer for `data` (`[row][output]`): the per-output intercepts
/// `base_score` broadcast to every row, overridden by the dataset's
/// per-instance `base_margin` when present (one value per row, or one per row
/// and output). Shared by every model representation that predicts.
pub(crate) fn initial_margins(base_score: &[f32], data: &DMatrix) -> Vec<f32> {
    let n = data.n_rows();
    let k = base_score.len();
    match data.base_margin() {
        Some(bm) if bm.len() == n * k => bm.to_vec(),
        Some(bm) if bm.len() == n => bm.iter().flat_map(|&m| std::iter::repeat_n(m, k)).collect(),
        _ => {
            let mut out = Vec::with_capacity(n * k);
            for _ in 0..n {
                out.extend_from_slice(base_score);
            }
            out
        }
    }
}

/// Refuse a model with other than one value per row of what a scalar method
/// returns (`width`: its outputs, or its prediction width), naming the
/// `_into` method that serves it.
fn one_value_per_row(width: usize, instead: &str) -> Result<()> {
    if width == 1 {
        return Ok(());
    }
    Err(HessboostError::incompatible_model(
        "outputs",
        format!("the model gives {width} values per row; use `{instead}`"),
    ))
}

/// Rows per prediction block: the block's feature rows stay in cache while
/// every tree walks them. Must be a multiple of [`LANES`].
const PREDICT_BLOCK_ROWS: usize = 256;
const _: () = assert!(PREDICT_BLOCK_ROWS.is_multiple_of(LANES));

/// Widest CSR matrix that is densified block-by-block for prediction. Wider
/// matrices fall back to per-lookup row scans.
const MAX_DENSIFY_COLS: usize = 4096;

/// Which CSR matrices [`RowBlock::build`] densifies into a scratch buffer.
#[derive(Clone, Copy)]
enum Densify {
    /// Matrices up to this many columns; wider ones stay sparse.
    UpTo(usize),
    /// Every matrix, whatever its width.
    Always,
}

/// A block of consecutive rows exposed as dense feature vectors (`NaN` =
/// missing) for [`CompactForest`] traversal. Dense `NaN`-sentinel matrices are
/// viewed in place. Dense matrices with another sentinel and CSR rows are
/// materialized into a per-block scratch buffer so each node lookup is a single
/// indexed load. Both keep a lane-major copy of the full [`LANES`]-row groups
/// for the batch kernel.
pub(super) enum RowBlock<'a> {
    View {
        data: &'a [f32],
        n_cols: usize,
        start: usize,
        lanes: Vec<u32>,
    },
    Scratch {
        source: &'a DMatrix,
        n_cols: usize,
        /// Row-major tail rows (those past the last full [`LANES`] group).
        scratch: Vec<f32>,
        lanes: Vec<u32>,
        /// Block row index of the first tail row.
        tail_start: usize,
    },
    /// Very wide sparse rows: route through `DMatrix::get` per lookup.
    Wide { data: &'a DMatrix, start: usize },
}

impl<'a> RowBlock<'a> {
    /// Blocks for batch traversal: wide CSR matrices stay sparse.
    fn new(data: &'a DMatrix) -> Self {
        Self::build(data, Densify::UpTo(MAX_DENSIFY_COLS))
    }

    /// Blocks that are loaded one row at a time and always expose a dense row
    /// (for per-row algorithms such as TreeSHAP whose cost per row already
    /// scales with the feature count).
    pub(super) fn single_rows(data: &'a DMatrix) -> Self {
        Self::build(data, Densify::Always)
    }

    fn build(data: &'a DMatrix, densify: Densify) -> Self {
        let n_cols = data.n_cols();
        match data.dense_values() {
            Some(dense) if data.missing().is_nan() => RowBlock::View {
                data: dense,
                n_cols,
                start: 0,
                lanes: Vec::new(),
            },
            None if matches!(densify, Densify::UpTo(max) if n_cols > max) => {
                RowBlock::Wide { data, start: 0 }
            }
            _ => RowBlock::Scratch {
                source: data,
                n_cols,
                scratch: Vec::new(),
                lanes: Vec::new(),
                tail_start: 0,
            },
        }
    }

    /// Point the block at rows `start..start + rows`.
    pub(super) fn load(&mut self, start: usize, rows: usize) {
        match self {
            RowBlock::Wide { start: s, .. } => *s = start,
            RowBlock::View {
                data,
                n_cols,
                start: s,
                lanes,
            } => {
                *s = start;
                let n_cols = *n_cols;
                fill_lanes(
                    lanes,
                    &data[start * n_cols..(start + rows) * n_cols],
                    n_cols,
                );
            }
            RowBlock::Scratch {
                source,
                n_cols,
                scratch,
                lanes,
                tail_start,
            } => {
                let n_cols = *n_cols;
                let missing = source.missing();
                let groups = rows / LANES;
                *tail_start = groups * LANES;
                lanes.clear();
                lanes.resize(groups * FEATURE_LANES * n_cols, key(f32::NAN));
                scratch.clear();
                scratch.resize((rows - groups * LANES) * n_cols, f32::NAN);
                // Store feature `f` of block row `r` in its keyed lane slot
                // (the negated key follows `LANES` later) or its tail slot.
                let mut put = |r: usize, f: usize, v: f32| {
                    if r < groups * LANES {
                        let i =
                            (r / LANES) * FEATURE_LANES * n_cols + f * FEATURE_LANES + r % LANES;
                        lanes[i] = key(v);
                        lanes[i + LANES] = key(-v);
                    } else {
                        scratch[(r - groups * LANES) * n_cols + f] = v;
                    }
                };
                if let Some(dense) = source.dense_values() {
                    for r in 0..rows {
                        let src = &dense[(start + r) * n_cols..(start + r + 1) * n_cols];
                        for (f, &v) in src.iter().enumerate() {
                            put(r, f, if v == missing { f32::NAN } else { v });
                        }
                    }
                } else {
                    let (indptr, indices, values) =
                        source.csr_parts().expect("scratch blocks are dense or CSR");
                    for r in 0..rows {
                        let row = start + r;
                        // Reverse order so the first occurrence of a duplicated
                        // column wins, matching `DMatrix::get`.
                        for k in (indptr[row]..indptr[row + 1]).rev() {
                            let v = values[k];
                            let v = if crate::data::is_missing(v, missing) {
                                f32::NAN
                            } else {
                                v
                            };
                            put(r, indices[k] as usize, v);
                        }
                    }
                }
            }
        }
    }

    /// Loaded row `r` as a dense `NaN`-for-missing slice, or `None` for wide
    /// sparse blocks, which are never materialized. Scratch blocks only keep
    /// the tail rows (those past the last full [`LANES`] group) row-major.
    #[inline]
    pub(super) fn row(&self, r: usize) -> Option<&[f32]> {
        match self {
            RowBlock::View {
                data,
                n_cols,
                start,
                ..
            } => Some(&data[(start + r) * n_cols..(start + r + 1) * n_cols]),
            RowBlock::Scratch {
                n_cols,
                scratch,
                tail_start,
                ..
            } => {
                let i = r
                    .checked_sub(*tail_start)
                    .expect("row-major access to a lane-major scratch row");
                Some(&scratch[i * n_cols..(i + 1) * n_cols])
            }
            RowBlock::Wide { .. } => None,
        }
    }

    /// Value of feature `f` in loaded row `r`, `None` when missing.
    #[inline]
    fn get(&self, r: usize, f: u32) -> Option<f32> {
        let Some(row) = self.row(r) else {
            let RowBlock::Wide { data, start } = self else {
                unreachable!("only wide blocks lack dense rows")
            };
            return data.get(start + r, f as usize);
        };
        let v = row[f as usize];
        if v.is_nan() { None } else { Some(v) }
    }

    /// Original leaf ids of loaded row `r` in trees `0..out.len()`, written to
    /// `out[t]`.
    fn original_leaf_ids_for_row(&self, forest: &CompactForest, r: usize, out: &mut [u32]) {
        match self.row(r) {
            Some(row) => forest.original_leaf_ids_for_row(row, out),
            None => self.wide_row_leaves(forest, r, 0..out.len(), |t, leaf| {
                out[t] = forest.original_id(leaf);
            }),
        }
    }

    /// The wide sparse fallback of the per-row walks: `sink(t, leaf)` with
    /// loaded row `r`'s arena leaf in each tree of `trees`, features looked
    /// up one at a time.
    #[inline]
    fn wide_row_leaves(
        &self,
        forest: &CompactForest,
        r: usize,
        trees: std::ops::Range<usize>,
        mut sink: impl FnMut(usize, u32),
    ) {
        for t in trees {
            sink(t, forest.leaf_id_with(t, |f| self.get(r, f)));
        }
    }

    /// The wide sparse fallback of the block walks: `sink(r, leaf)` with the
    /// arena leaf of each of the first `rows` loaded rows in tree `t`.
    #[inline]
    fn wide_leaves(
        &self,
        forest: &CompactForest,
        t: usize,
        rows: usize,
        mut sink: impl FnMut(usize, u32),
    ) {
        for r in 0..rows {
            sink(r, forest.leaf_id_with(t, |f| self.get(r, f)));
        }
    }

    /// `out[scalar_tree_output(t, parallel, k)] += weight(t) * leaf_value(row
    /// r, tree t)` for the trees `trees` of loaded row `r`, where `parallel`
    /// is the model's `num_parallel_tree`.
    fn accumulate_row(
        &self,
        forest: &CompactForest,
        r: usize,
        trees: std::ops::Range<usize>,
        parallel: usize,
        weight: impl Fn(usize) -> f32,
        out: &mut [f32],
    ) {
        if let Some(row) = self.row(r) {
            forest.accumulate_row(row, trees, parallel, weight, out);
        } else {
            let k = out.len();
            self.wide_row_leaves(forest, r, trees, |t, leaf| {
                out[scalar_tree_output(t, parallel, k)] += weight(t) * forest.leaf_value(leaf);
            });
        }
    }

    /// `out[j] += weight(t) * leaf_vector(row r, tree t)[j]` for the
    /// vector-leaf trees `trees` of loaded row `r`.
    fn accumulate_row_vector(
        &self,
        forest: &CompactForest,
        r: usize,
        trees: std::ops::Range<usize>,
        weight: impl Fn(usize) -> f32,
        out: &mut [f32],
    ) {
        if let Some(row) = self.row(r) {
            forest.accumulate_row_vector(row, trees, weight, out);
        } else {
            let k = out.len();
            self.wide_row_leaves(forest, r, trees, |t, leaf| {
                let w = weight(t);
                for (o, &v) in out.iter_mut().zip(forest.leaf_vector(leaf, k)) {
                    *o += w * v;
                }
            });
        }
    }

    /// The first `rows` loaded rows as a [`LaneBlock`] for the batch kernels,
    /// or `None` for wide sparse blocks.
    #[inline]
    fn lane_block(&self, rows: usize) -> Option<LaneBlock<'_>> {
        let tail_start = rows / LANES * LANES;
        match self {
            RowBlock::View {
                data,
                n_cols,
                start,
                lanes,
            } => Some(LaneBlock {
                lanes,
                tail: &data[(start + tail_start) * n_cols..(start + rows) * n_cols],
                n_cols: *n_cols,
                rows,
            }),
            RowBlock::Scratch {
                n_cols,
                scratch,
                lanes,
                ..
            } => Some(LaneBlock {
                lanes,
                tail: &scratch[..(rows - tail_start) * n_cols],
                n_cols: *n_cols,
                rows,
            }),
            RowBlock::Wide { .. } => None,
        }
    }

    /// `out[r * stride] = original leaf id of row r` in tree `t` over the
    /// loaded rows.
    fn original_leaf_ids(
        &self,
        forest: &CompactForest,
        t: usize,
        rows: usize,
        out: &mut [u32],
        stride: usize,
    ) {
        if let Some(block) = self.lane_block(rows) {
            forest.original_leaf_ids(t, block, out, stride);
        } else {
            self.wide_leaves(forest, t, rows, |r, leaf| {
                out[r * stride] = forest.original_id(leaf);
            });
        }
    }

    /// `out[r * stride + j] += weight * leaf_vector(row r)[j]` (all `stride`
    /// outputs) in vector-leaf tree `t` over the loaded rows.
    fn accumulate_vector(
        &self,
        forest: &CompactForest,
        t: usize,
        rows: usize,
        weight: f32,
        out: &mut [f32],
        stride: usize,
    ) {
        if let Some(block) = self.lane_block(rows) {
            forest.accumulate_vector(t, block, stride, weight, out, stride);
        } else {
            self.wide_leaves(forest, t, rows, |r, leaf| {
                let dst = &mut out[r * stride..(r + 1) * stride];
                for (o, &v) in dst.iter_mut().zip(forest.leaf_vector(leaf, stride)) {
                    *o += weight * v;
                }
            });
        }
    }

    /// `out[r * stride] += weight * leaf_value(row r)` in tree `t` over the
    /// loaded rows.
    fn accumulate(
        &self,
        forest: &CompactForest,
        t: usize,
        rows: usize,
        weight: f32,
        out: &mut [f32],
        stride: usize,
    ) {
        if let Some(block) = self.lane_block(rows) {
            forest.accumulate(t, block, weight, out, stride);
        } else {
            self.wide_leaves(forest, t, rows, |r, leaf| {
                out[r * stride] += weight * forest.leaf_value(leaf);
            });
        }
    }
}
