use super::{
    BoostedModel, ModelObjective, Predictions, Shrinkage, rebuild_objective, shrink_margins,
};
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::objective::Objective;
use crate::objective::distributional::Dist;
use crate::tree::compact::{CompactForest, FEATURE_LANES, LANES, LaneBlock, fill_lanes, key};
use crate::tree::scalar_tree_output;
use rayon::prelude::*;
use std::ops::{Bound, Range, RangeBounds};

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
    /// encoded as `f32`.
    ///
    /// # Errors
    ///
    /// As for [`Self::predict_margin`].
    pub fn predict(
        &self,
        data: &DMatrix,
        iterations: impl Into<Iterations>,
    ) -> Result<Predictions> {
        let margin = self.predict_margin(data, iterations)?;
        Ok(transform_model_margins(
            &self.objective,
            self.max_delta_step,
            self.n_targets,
            margin,
        ))
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
        let iterations = self.resolve_iterations(iterations.into(), "iterations")?;
        if let Some(shrinkage) = &self.shrinkage {
            // Every later iteration rescaled the earlier ones, so the
            // iterations `a..b` alone are no model.
            if iterations.start != 0 {
                return Err(HessboostError::incompatible_model(
                    "iterations",
                    format!(
                        "a model trained with model shrinkage rescales its earlier iterations \
                         every iteration, so {}..{} has no meaning; use a range starting at 0",
                        iterations.start, iterations.end
                    ),
                ));
            }
            let values = self.shrunk_margins(shrinkage, data, iterations.end);
            return Ok(Predictions::new(values, data.n_rows(), self.n_outputs()));
        }
        let values = self.margin_from_trees(data, self.iteration_trees(iterations));
        Ok(Predictions::new(values, data.n_rows(), self.n_outputs()))
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
    /// distributional `dist:*` objective (beyond XGBoost, see
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

/// Turn raw margins (`[row][output]`) of a model with the given objective
/// into predictions in the objective's reported space: the objective's
/// transform (identity for an objective the crate does not implement, e.g. a
/// custom loss, mirroring how XGBoost returns margins then), and for
/// `multi:softmax` the per-row argmax class index encoded as `f32` (width 1).
pub(crate) fn transform_model_margins(
    objective: &ModelObjective,
    max_delta_step: f64,
    n_targets: usize,
    margin: Predictions,
) -> Predictions {
    let (n_rows, n_outputs) = (margin.n_rows(), margin.width());
    let mut values = margin.into_vec();
    let width =
        transform_margins_in_place(objective, max_delta_step, n_targets, &mut values, n_outputs);
    values.truncate(n_rows * width);
    Predictions::new(values, n_rows, width)
}

/// [`transform_model_margins`] in place: turns the margins `values`
/// (`[row][output]`, `n_outputs` wide) into predictions and returns their
/// width. For `multi:softmax` (width 1) the class indices fill the first
/// `n_rows` values and the rest are left over.
pub(crate) fn transform_margins_in_place(
    objective: &ModelObjective,
    max_delta_step: f64,
    n_targets: usize,
    values: &mut [f32],
    n_outputs: usize,
) -> usize {
    if let Some(Ok(loss)) = rebuild_objective(objective, max_delta_step, n_targets) {
        loss.pred_transform(values);
    }
    if objective
        .built_in()
        .is_some_and(Objective::predicts_class_index)
    {
        // Row `r`'s class lands at index `r`, at or before the row's own
        // values, so every row is read before it is overwritten.
        for r in 0..values.len() / n_outputs {
            let class = values[r * n_outputs..(r + 1) * n_outputs]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map_or(0.0, |(i, _)| i as f32);
            values[r] = class;
        }
        return 1;
    }
    n_outputs
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
