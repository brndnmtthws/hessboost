//! The trained model: an ensemble of trees plus the metadata needed to turn
//! their sum into calibrated predictions.

use crate::config::ObjectiveParams;
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::objective::{Dist, DistFamily, create_objective};
use crate::tree::compact::{CompactForest, FEATURE_LANES, LANES, fill_lanes, key};
use crate::tree::{RegTree, scalar_tree_output};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

/// The kind of feature-importance score to compute, mirroring XGBoost's
/// `importance_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportanceType {
    /// Number of times a feature is used to split.
    Weight,
    /// Total loss reduction attributed to splits on a feature.
    TotalGain,
    /// Average loss reduction per split on a feature.
    Gain,
    /// Total Hessian (cover) of splits on a feature.
    TotalCover,
    /// Average Hessian (cover) per split on a feature.
    Cover,
}

/// A gradient-boosted tree ensemble.
///
/// Leaf weights already include the learning rate (shrinkage), so a raw margin
/// prediction for output `k` is simply `base_score[k] + Σ tree_k(x)`. The
/// stored `objective` name drives the prediction transform (e.g. the logistic
/// sigmoid).
///
/// Trees are laid out as in XGBoost: boosting iteration `i` owns the
/// [`trees_per_iteration`](Self::trees_per_iteration) trees starting at
/// `i * trees_per_iteration`, grouped by output (`num_parallel_tree` trees
/// for output 0, then output 1, ...). Tree `t` therefore feeds output
/// `(t / num_parallel_tree) % n_outputs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoostedModel {
    trees: Vec<RegTree>,
    /// Per-output intercept in margin space (length `n_outputs`).
    base_score: Vec<f32>,
    /// The objective's XGBoost name (`Objective::name`), which drives the
    /// prediction transform.
    objective: String,
    /// Objective hyper-parameters, retained for XGBoost-format export and for
    /// rebuilding the objective.
    objective_params: ObjectiveParams,
    /// The configured `num_class` (`0` for scalar objectives).
    num_class: usize,
    /// Raw outputs per instance: `num_class` for multiclass objectives, the
    /// objective's own output count otherwise (custom objectives may have
    /// several).
    n_outputs: usize,
    /// Label columns per training row (`1` unless trained on a label matrix).
    n_targets: usize,
    n_features: usize,
    /// The best iteration index selected by early stopping, if any.
    best_iteration: Option<usize>,
    /// Per-tree contribution weights: empty when every tree weighs `1.0`
    /// (e.g. imported or sliced `gbtree` models), otherwise one weight per
    /// tree. The DART booster stores fractional weights here so dropped trees
    /// can be rescaled.
    tree_weights: Vec<f32>,
    /// Trees grown per output in each boosting iteration (XGBoost
    /// `num_parallel_tree`; `1` for ordinary boosting, more for boosted
    /// random forests).
    num_parallel_tree: usize,
    /// Linear (coordinate-descent) booster parameters. `Some` only for
    /// `gblinear` models, in which case predictions come from the linear model
    /// and the `trees` vector is empty.
    linear: Option<LinearModel>,
    /// Prediction layout of `trees` ([`CompactForest`]), derived lazily and
    /// never serialized. Reset whenever `trees` changes.
    #[serde(skip)]
    compact: OnceLock<CompactForest>,
}

/// The parameters of a linear (`gblinear`) booster: a per-output weight vector
/// plus a per-output bias, fit by coordinate descent.
///
/// `weights` has length `n_features * n_outputs` laid out `[feature][output]`
/// (the weight for feature `f`, output `k` is `weights[f * n_outputs + k]`).
/// `bias` has length `n_outputs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearModel {
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl LinearModel {
    /// Assemble a linear model from its fitted `weights` (`[feature][output]`)
    /// and per-output `bias`.
    pub(crate) fn new(weights: Vec<f32>, bias: Vec<f32>) -> Self {
        LinearModel { weights, bias }
    }

    pub(crate) fn bias(&self) -> &[f32] {
        &self.bias
    }

    pub(crate) fn weights(&self) -> &[f32] {
        &self.weights
    }
}

/// Invoke `f(feature, value)` for each present feature of `row` in feature
/// order. Shared by the gblinear training and prediction paths; arithmetic
/// stays at each call site to preserve exact conversion points.
pub(crate) fn for_each_present_value(data: &DMatrix, row: usize, mut f: impl FnMut(usize, f32)) {
    for feat in 0..data.n_cols() {
        if let Some(x) = data.get(row, feat) {
            f(feat, x);
        }
    }
}

/// Validated prologue for the TreeSHAP paths: dimensions, effective trees,
/// and initial margins.
pub(super) struct AttributionPrologue<'a> {
    pub(super) n: usize,
    pub(super) k: usize,
    pub(super) nf: usize,
    pub(super) width: usize,
    pub(super) trees: &'a [RegTree],
    pub(super) initial: Vec<f32>,
}

/// The metadata a model is assembled with: what it predicts and how its trees
/// are laid out. Shared by training and the XGBoost-JSON importer.
pub(crate) struct ModelSpec {
    /// The objective's XGBoost name (`Objective::name`).
    pub(crate) objective: String,
    pub(crate) objective_params: ObjectiveParams,
    /// Configured `num_class` (`0` for scalar objectives).
    pub(crate) num_class: usize,
    /// Raw outputs per instance (`Objective::n_outputs`).
    pub(crate) n_outputs: usize,
    /// Label columns per training row ([`DMatrix::n_targets`]).
    pub(crate) n_targets: usize,
    pub(crate) n_features: usize,
}

impl BoostedModel {
    pub(crate) fn new(base_score: Vec<f32>, spec: ModelSpec) -> Self {
        Self::from_parts(Vec::new(), Vec::new(), base_score, spec)
    }

    /// Attach a fitted linear (`gblinear`) booster. Predictions then come from
    /// the linear model instead of the (empty) tree ensemble.
    pub(crate) fn set_linear(&mut self, linear: LinearModel) {
        self.linear = Some(linear);
    }

    /// Append a tree with an explicit contribution weight (`1.0` for plain
    /// `gbtree`; DART stores fractional weights so dropped trees can be
    /// rescaled).
    pub(crate) fn push_tree_weighted(&mut self, tree: RegTree, weight: f32) {
        self.trees.push(tree);
        self.tree_weights.push(weight);
        self.compact = OnceLock::new();
    }

    /// The prediction layout of the ensemble, built on first use and dropped
    /// whenever a tree is appended.
    fn compact_forest(&self) -> &CompactForest {
        self.compact
            .get_or_init(|| CompactForest::from_trees(&self.trees))
    }

    /// Contribution weight of tree `i` (`1.0` when weights are absent, e.g. for
    /// imported models or plain `gbtree`).
    #[inline]
    pub(crate) fn tree_weight(&self, i: usize) -> f32 {
        self.tree_weights.get(i).copied().unwrap_or(1.0)
    }

    /// Invoke `f(feature, output, weight * x)` for each present feature of
    /// `row` and each output of the linear (`gblinear`) model, in feature
    /// order. The product is computed in f64, which is exact for f32 operands;
    /// callers round to their accumulation precision. No-op for tree
    /// ensembles.
    pub(crate) fn for_each_linear_contribution(
        &self,
        data: &DMatrix,
        row: usize,
        mut f: impl FnMut(usize, usize, f64),
    ) {
        let Some(lm) = &self.linear else {
            return;
        };
        let k = self.n_outputs();
        for_each_present_value(data, row, |feat, x| {
            for c in 0..k {
                f(feat, c, f64::from(lm.weights[feat * k + c]) * f64::from(x));
            }
        });
    }

    /// Multiply tree `i`'s contribution weight by `factor` (DART rescaling).
    pub(crate) fn scale_tree_weight(&mut self, i: usize, factor: f32) {
        if i < self.tree_weights.len() {
            self.tree_weights[i] *= factor;
        }
    }

    /// Raw margin predictions that exclude the trees marked `true` in `dropped`
    /// (indexed by tree id). Used by the DART training loop to compute a round's
    /// gradients from the ensemble minus its dropout set. Output is laid out
    /// `[instance][output]`.
    pub(crate) fn predict_margin_dropout(&self, data: &DMatrix, dropped: &[bool]) -> Vec<f32> {
        let mut out = self.initial_margins(data);
        // Dropped trees contribute a zero weight, leaving per-cell accumulation
        // in ascending tree order (a `0.0` addend is a no-op).
        let weight = |ti: usize| {
            if dropped.get(ti).copied().unwrap_or(false) {
                0.0
            } else {
                self.tree_weight(ti)
            }
        };
        self.accumulate_forest(data, &mut out, 0..self.trees.len(), weight);
        out
    }

    pub(crate) fn set_best_iteration(&mut self, it: Option<usize>) {
        self.best_iteration = it;
    }

    /// Reassemble a model from its constituent parts. Used by the XGBoost-JSON
    /// importer, which builds trees and metadata externally. `tree_weights`
    /// is either empty (every tree weighs `1.0`) or holds one DART weight per
    /// tree; [`BoostedModel::validate_structure`] enforces the length.
    pub(crate) fn from_parts(
        trees: Vec<RegTree>,
        tree_weights: Vec<f32>,
        base_score: Vec<f32>,
        spec: ModelSpec,
    ) -> Self {
        BoostedModel {
            trees,
            base_score,
            objective: spec.objective,
            objective_params: spec.objective_params,
            num_class: spec.num_class,
            n_outputs: spec.n_outputs,
            n_targets: spec.n_targets,
            n_features: spec.n_features,
            best_iteration: None,
            tree_weights,
            num_parallel_tree: 1,
            linear: None,
            compact: OnceLock::new(),
        }
    }

    /// The configured `num_class` (`0` for regression / binary objectives).
    pub(crate) fn num_class(&self) -> usize {
        self.num_class
    }

    /// The objective hyper-parameters the model was trained with.
    pub fn objective_params(&self) -> &ObjectiveParams {
        &self.objective_params
    }

    /// Number of raw outputs per instance: `num_class` for multiclass, the
    /// objective's output count otherwise (`1` for every built-in scalar
    /// objective, [`BoostedModel::n_targets`] for a multi-target model; custom
    /// objectives may declare more).
    #[inline]
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// Number of label columns (targets) per row of the training data: `1`
    /// unless the model was trained on [`DMatrix::with_label_matrix`] labels.
    #[inline]
    pub fn n_targets(&self) -> usize {
        self.n_targets
    }

    /// Whether the ensemble consists of vector-leaf trees
    /// (`multi_strategy = multi_output_tree`): each tree predicts every
    /// output at once instead of one output per tree.
    pub fn has_vector_leaves(&self) -> bool {
        self.trees.first().is_some_and(RegTree::is_vector_leaf)
    }

    /// The first output's intercept (global bias) in margin space.
    ///
    /// Scalar-output models have exactly one value. For multiclass models use
    /// [`Self::base_scores`] to access every per-class intercept.
    pub fn base_score(&self) -> f32 {
        self.base_score[0]
    }

    /// Per-output intercepts in margin space, one per class/output.
    pub fn base_scores(&self) -> &[f32] {
        &self.base_score
    }

    /// The objective name this model was trained with.
    pub fn objective(&self) -> &str {
        &self.objective
    }

    /// The best iteration chosen by early stopping, if applicable.
    pub fn best_iteration(&self) -> Option<usize> {
        self.best_iteration
    }

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
            for row in 0..n {
                for c in 0..k {
                    out[row * k + c] += lm.bias[c];
                }
                self.for_each_linear_contribution(data, row, |_f, c, v| {
                    out[row * k + c] += v as f32;
                });
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
    fn accumulate_forest(
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

    /// [`Self::predict`] from the boosting iterations in `iteration_range`
    /// only (see [`Self::predict_margin_range`] for the range convention).
    pub fn predict_range(
        &self,
        data: &DMatrix,
        iteration_range: (usize, usize),
    ) -> Result<Vec<f32>> {
        let margin = self.predict_margin_range(data, iteration_range)?;
        Ok(transform_model_margins(
            &self.objective,
            &self.objective_params,
            self.num_class,
            self.n_targets,
            self.n_outputs(),
            margin,
        ))
    }

    /// For multiclass, the predicted class index per row (argmax over classes).
    /// For single-output models this returns the transformed prediction rounded
    /// to the nearest class at 0.5. A multi-target model (label matrix) is
    /// multi-label: each target is thresholded at 0.5 independently, giving
    /// `n_rows × n_targets` decisions laid out `[row][target]`.
    pub fn predict_class(&self, data: &DMatrix) -> Result<Vec<u32>> {
        let probs = self.predict(data)?;
        let k = self.n_outputs();
        if k == 1 || self.n_targets > 1 {
            return Ok(probs.iter().map(|&p| u32::from(p > 0.5)).collect());
        }
        if self.objective == "multi:softmax" {
            return Ok(probs.iter().map(|&class| class as u32).collect());
        }
        Ok(probs
            .chunks_exact(k)
            .map(|row| crate::simd::argmax_scalar(row) as u32)
            .collect())
    }

    /// [`Self::predict_leaf`] for the trees of the iterations in
    /// `iteration_range` (shape `n_rows × trees`). As in XGBoost the range
    /// must start at iteration `0`; use [`Self::slice`] for a later start.
    #[allow(
        clippy::redundant_closure_for_method_calls,
        reason = "the method path is not general enough over the block lifetime"
    )]
    pub fn predict_leaf_range(
        &self,
        data: &DMatrix,
        iteration_range: (usize, usize),
    ) -> Result<Vec<u32>> {
        self.validate_prediction_data(data)?;
        let t = self.prefix_trees(iteration_range, "leaf prediction")?;
        let n = data.n_rows();
        let mut out = vec![0u32; n * t];
        if t == 0 {
            return Ok(out);
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
        Ok(out)
    }

    /// Compute feature importance of the requested type, returned as a map from
    /// feature index to score (features that never split are absent).
    pub fn feature_importance(&self, kind: ImportanceType) -> HashMap<u32, f64> {
        let mut count: HashMap<u32, f64> = HashMap::new();
        let mut cover: HashMap<u32, f64> = HashMap::new();
        let mut gain: HashMap<u32, f64> = HashMap::new();
        for tree in &self.trees {
            for node in tree.nodes() {
                if node.is_leaf() {
                    continue;
                }
                *count.entry(node.split_feature).or_default() += 1.0;
                *cover.entry(node.split_feature).or_default() += f64::from(node.sum_hess);
                *gain.entry(node.split_feature).or_default() += f64::from(node.split_gain);
            }
        }
        // Divide a total by the split count to get the per-split average.
        let average = |totals: HashMap<u32, f64>| -> HashMap<u32, f64> {
            totals
                .into_iter()
                .map(|(f, t)| {
                    let n = count.get(&f).copied().unwrap_or(1.0);
                    (f, t / n)
                })
                .collect()
        };
        match kind {
            ImportanceType::Weight => count,
            ImportanceType::TotalCover => cover,
            ImportanceType::Cover => average(cover),
            ImportanceType::TotalGain => gain,
            ImportanceType::Gain => average(gain),
        }
    }

    /// Read-only access to the trees (e.g. for serialization or SHAP).
    pub fn trees(&self) -> &[RegTree] {
        &self.trees
    }

    /// Number of features the model expects.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub(crate) fn linear(&self) -> Option<&LinearModel> {
        self.linear.as_ref()
    }

    pub(crate) fn has_non_unit_tree_weights(&self) -> bool {
        (0..self.trees.len()).any(|i| self.tree_weight(i) != 1.0)
    }

    /// Margin buffer for `data`: the per-output intercepts broadcast to every
    /// row, overridden by the dataset's per-instance `base_margin` when
    /// present (one value per row, or one per row and output). Shared by
    /// prediction, TreeSHAP, and the training margin caches.
    pub(crate) fn initial_margins(&self, data: &DMatrix) -> Vec<f32> {
        initial_margins(&self.base_score, data)
    }

    /// Validated prologue for the TreeSHAP paths over the iterations in
    /// `iteration_range`, which must start at iteration `0` (as in XGBoost).
    ///
    /// Linear-leaf trees are refused: TreeSHAP attributes constant leaf values
    /// along decision paths and has no defined extension to leaves whose
    /// output varies with the row (LightGBM refuses SHAP for linear trees as
    /// well).
    pub(super) fn attribution_prologue(
        &self,
        data: &DMatrix,
        iteration_range: (usize, usize),
        what: &str,
    ) -> Result<AttributionPrologue<'_>> {
        self.validate_prediction_data(data)?;
        let end = self.prefix_trees(iteration_range, what)?;
        let trees = &self.trees[..end];
        if trees.iter().any(|tree| tree.linear_leaves().is_some()) {
            return Err(HessboostError::invalid_param(
                "linear_tree",
                "SHAP contributions and interactions are not defined for models with linear leaves",
            ));
        }
        let nf = self.n_features;
        Ok(AttributionPrologue {
            n: data.n_rows(),
            k: self.n_outputs(),
            nf,
            width: nf + 1,
            trees,
            initial: self.initial_margins(data),
        })
    }

    /// Set the number of trees per output in each iteration. Only valid while
    /// the layout still holds whole iterations of that size
    /// ([`BoostedModel::validate_structure`] checks it).
    pub(crate) fn set_num_parallel_tree(&mut self, num_parallel_tree: usize) {
        self.num_parallel_tree = num_parallel_tree;
    }

    /// Replace the per-output intercepts (margin space).
    pub(crate) fn set_base_scores(&mut self, base_score: Vec<f32>) {
        self.base_score = base_score;
    }

    /// Replace the objective hyper-parameters (continued training adopts the
    /// new configuration, as XGBoost's `set_param` does).
    pub(crate) fn set_objective_params(&mut self, params: ObjectiveParams) {
        self.objective_params = params;
    }

    /// Give every tree an explicit contribution weight (`1.0` where absent),
    /// so appended trees' weights line up with their tree ids.
    pub(crate) fn materialize_tree_weights(&mut self) {
        self.tree_weights.resize(self.trees.len(), 1.0);
    }

    /// Remove and return every tree (dropping the contribution weights),
    /// leaving an empty ensemble with the same metadata. Used by
    /// `process_type=update`, which re-appends the refreshed trees.
    pub(crate) fn take_trees(&mut self) -> Vec<RegTree> {
        self.tree_weights.clear();
        self.compact = OnceLock::new();
        std::mem::take(&mut self.trees)
    }

    /// Number of trees (boosting rounds × outputs × `num_parallel_tree`).
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Trees grown per output in each boosting iteration (XGBoost
    /// `num_parallel_tree`).
    #[inline]
    pub fn num_parallel_tree(&self) -> usize {
        self.num_parallel_tree
    }

    /// Trees per boosting iteration: `n_outputs × num_parallel_tree`, or
    /// `num_parallel_tree` vector-leaf trees (each feeds every output).
    #[inline]
    pub fn trees_per_iteration(&self) -> usize {
        if self.has_vector_leaves() {
            self.num_parallel_tree
        } else {
            self.n_outputs * self.num_parallel_tree
        }
    }

    /// Check that training `num_parallel_tree` trees per output for
    /// `n_outputs` outputs keeps the per-iteration tree count
    /// (`n_outputs × num_parallel_tree`) within `usize`, so every
    /// iteration-indexing product stays defined. The trainer calls it before
    /// assembling the model, as the loader's structural checks do for saved
    /// models.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] (`num_parallel_tree`) when the
    /// product overflows.
    pub(crate) fn check_iteration_size(n_outputs: usize, num_parallel_tree: usize) -> Result<()> {
        if n_outputs.checked_mul(num_parallel_tree).is_none() {
            return Err(HessboostError::invalid_param(
                "num_parallel_tree",
                format!(
                    "{num_parallel_tree} parallel trees for {n_outputs} outputs overflow the \
                     trees per iteration"
                ),
            ));
        }
        Ok(())
    }

    /// The output tree `t` contributes to (XGBoost `tree_info[t]`): `0` for
    /// every vector-leaf tree, whose leaves carry all outputs.
    #[inline]
    pub(crate) fn tree_output(&self, t: usize) -> usize {
        if self.has_vector_leaves() {
            0
        } else {
            scalar_tree_output(t, self.num_parallel_tree, self.n_outputs)
        }
    }

    /// Number of boosting iterations (`num_trees / trees_per_iteration`;
    /// XGBoost `num_boosted_rounds`).
    pub fn num_boost_rounds(&self) -> usize {
        self.trees.len() / self.trees_per_iteration()
    }

    /// Number of trees in the effective iterations (`[0, best_iteration +
    /// 1)` after early stopping, else all trees).
    pub(crate) fn effective_num_trees(&self) -> usize {
        self.best_iteration.map_or(self.trees.len(), |it| {
            ((it + 1) * self.trees_per_iteration()).min(self.trees.len())
        })
    }

    /// The iteration range plain prediction uses: `[0, best_iteration + 1)`
    /// when early stopping selected an iteration, else every iteration.
    fn default_iteration_range(&self) -> (usize, usize) {
        (0, self.best_iteration.map_or(0, |it| it + 1))
    }

    /// Tree ids covered by `iteration_range`, XGBoost's half-open `[begin,
    /// end)` over boosting iterations with `end == 0` meaning "through the
    /// last iteration". Each iteration contributes its whole forest (every
    /// output and parallel tree).
    pub(crate) fn iteration_trees(
        &self,
        iteration_range: (usize, usize),
    ) -> Result<std::ops::Range<usize>> {
        let (begin, end) = iteration_range;
        let rounds = self.num_boost_rounds();
        let end = if end == 0 { rounds } else { end };
        if self.linear.is_some() && iteration_range != (0, 0) {
            return Err(HessboostError::invalid_param(
                "iteration_range",
                "gblinear models have no boosting iterations to select",
            ));
        }
        if end > rounds || begin > end {
            return Err(HessboostError::invalid_param(
                "iteration_range",
                format!("[{begin}, {end}) is out of range for a model with {rounds} iterations"),
            ));
        }
        let per = self.trees_per_iteration();
        Ok(begin * per..end * per)
    }

    /// Like [`Self::iteration_trees`] for the attribution and leaf
    /// predictions, which (as in XGBoost) only accept ranges starting at
    /// iteration `0`; slice the model for a later start.
    fn prefix_trees(&self, iteration_range: (usize, usize), what: &str) -> Result<usize> {
        let trees = self.iteration_trees(iteration_range)?;
        if trees.start != 0 {
            return Err(HessboostError::invalid_param(
                "iteration_range",
                format!(
                    "{what} supports only ranges starting at iteration 0; slice the model instead"
                ),
            ));
        }
        Ok(trees.end)
    }

    /// Raw margin predictions using the effective iterations (`[0,
    /// best_iteration + 1)` after early stopping, else all). The output is
    /// laid out `[instance][output]` (length `n_rows × n_outputs`).
    pub fn predict_margin(&self, data: &DMatrix) -> Result<Vec<f32>> {
        self.predict_margin_range(data, self.default_iteration_range())
    }

    /// Raw margin predictions from the boosting iterations in
    /// `iteration_range` only (XGBoost `iteration_range`: half-open `[begin,
    /// end)`, `end == 0` = through the last iteration, `(0, 0)` = the whole
    /// model regardless of early stopping). The intercept / dataset
    /// `base_margin` is always included.
    pub fn predict_margin_range(
        &self,
        data: &DMatrix,
        iteration_range: (usize, usize),
    ) -> Result<Vec<f32>> {
        self.validate_prediction_data(data)?;
        let trees = self.iteration_trees(iteration_range)?;
        Ok(self.margin_from_trees(data, trees))
    }

    /// Predictions in the objective's reported space. `multi:softprob` returns
    /// an `n_rows × num_class` probability matrix while `multi:softmax` returns
    /// one class index per row, encoded as `f32`. Uses the effective
    /// iterations, like [`Self::predict_margin`].
    pub fn predict(&self, data: &DMatrix) -> Result<Vec<f32>> {
        self.predict_range(data, self.default_iteration_range())
    }

    /// The predicted distribution of every row for a model trained with a
    /// distributional `dist:*` objective (beyond XGBoost, see
    /// [`crate::objective::distributional`]): one [`Dist`] per row, with its
    /// mean, variance, CDF, quantiles, log density, CRPS, intervals and
    /// sampling. The margins are mapped through the links in `f64`. Uses the
    /// effective iterations, like [`Self::predict`].
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] if the model's objective is not
    /// a `dist:*` objective, plus the errors of [`Self::predict_margin`].
    pub fn predict_distribution(&self, data: &DMatrix) -> Result<Vec<Dist>> {
        self.predict_distribution_range(data, self.default_iteration_range())
    }

    /// [`Self::predict_distribution`] from the boosting iterations in
    /// `iteration_range` only (see [`Self::predict_margin_range`]).
    pub fn predict_distribution_range(
        &self,
        data: &DMatrix,
        iteration_range: (usize, usize),
    ) -> Result<Vec<Dist>> {
        let family = DistFamily::from_objective(&self.objective).ok_or_else(|| {
            HessboostError::invalid_param(
                "objective",
                format!(
                    "`{}` does not predict distributions; train with a `dist:*` objective",
                    self.objective
                ),
            )
        })?;
        let margin = self.predict_margin_range(data, iteration_range)?;
        Ok(margin
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

    /// Per-row leaf indices for each tree (shape `n_rows × num_trees`,
    /// row-major, trees in [`Self::trees`] order). Walks every tree,
    /// regardless of early stopping.
    pub fn predict_leaf(&self, data: &DMatrix) -> Result<Vec<u32>> {
        self.predict_leaf_range(data, (0, 0))
    }

    /// A new model holding boosting iterations `begin, begin + step, ...`
    /// below `end` (Python's `booster[begin:end:step]`; `end == 0` means
    /// through the last iteration). Each selected iteration keeps its whole
    /// forest (every output and parallel tree) together with its DART tree
    /// weights; the intercepts, objective and layout carry over unchanged and
    /// no refit happens. As in XGBoost the slice drops `best_iteration`, so
    /// it predicts with all of its iterations.
    ///
    /// `step` must be at least 1 and the range non-empty and within
    /// [`Self::num_boost_rounds`]. XGBoost 3.4.2 additionally trips an
    /// internal check when `end - begin` is not a multiple of `step`; here
    /// every step selects `ceil((end - begin) / step)` iterations, matching
    /// XGBoost wherever it succeeds.
    pub fn slice(&self, begin: usize, end: usize, step: usize) -> Result<BoostedModel> {
        if self.linear.is_some() {
            return Err(HessboostError::invalid_param(
                "slice",
                "gblinear models have no boosting iterations to slice",
            ));
        }
        if step == 0 {
            return Err(HessboostError::invalid_param(
                "slice",
                "step must be at least 1",
            ));
        }
        let rounds = self.num_boost_rounds();
        let end = if end == 0 { rounds } else { end };
        if begin >= end {
            return Err(HessboostError::invalid_param(
                "slice",
                format!("empty slice [{begin}:{end}] is not allowed"),
            ));
        }
        if end > rounds {
            return Err(HessboostError::invalid_param(
                "slice",
                format!("end {end} is out of range for a model with {rounds} iterations"),
            ));
        }
        let per = self.trees_per_iteration();
        let mut trees = Vec::with_capacity((end - begin).div_ceil(step) * per);
        let mut tree_weights = Vec::new();
        for it in (begin..end).step_by(step) {
            let layer = it * per..(it + 1) * per;
            trees.extend_from_slice(&self.trees[layer.clone()]);
            if !self.tree_weights.is_empty() {
                tree_weights.extend(layer.map(|t| self.tree_weight(t)));
            }
        }
        Ok(BoostedModel {
            trees,
            base_score: self.base_score.clone(),
            objective: self.objective.clone(),
            objective_params: self.objective_params.clone(),
            num_class: self.num_class,
            n_outputs: self.n_outputs,
            n_targets: self.n_targets,
            n_features: self.n_features,
            best_iteration: None,
            tree_weights,
            num_parallel_tree: self.num_parallel_tree,
            linear: None,
            compact: OnceLock::new(),
        })
    }

    /// The iteration range the attribution predictions use by default (the
    /// effective iterations, like [`Self::predict_margin`]).
    pub(super) fn attribution_default_range(&self) -> (usize, usize) {
        self.default_iteration_range()
    }

    pub(crate) fn validate_prediction_data(&self, data: &DMatrix) -> Result<()> {
        validate_prediction_data(self.n_features, self.n_outputs(), data)
    }

    /// Serialize the model to the native binary format: a zstd-compressed
    /// container of named, typed sections holding the trees column-wise.
    /// Files written by this version keep loading in later ones.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        super::native::write(&super::native::StoredRef {
            trees: &self.trees,
            base_score: &self.base_score,
            objective: &self.objective,
            objective_params: &self.objective_params,
            num_class: self.num_class,
            n_outputs: self.n_outputs,
            n_targets: self.n_targets,
            n_features: self.n_features,
            best_iteration: self.best_iteration,
            tree_weights: &self.tree_weights,
            num_parallel_tree: self.num_parallel_tree,
            linear: self.linear.as_ref(),
        })
    }

    /// Deserialize a model from bytes produced by [`BoostedModel::to_bytes`]
    /// of this or an earlier version. Malformed input, and files that need
    /// a feature this version lacks, are refused with
    /// [`HessboostError::ModelFormat`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let m = super::native::read(bytes)?;
        let model = BoostedModel {
            trees: m.trees,
            base_score: m.base_score,
            objective: m.objective,
            objective_params: m.objective_params,
            num_class: m.num_class,
            n_outputs: m.n_outputs,
            n_targets: m.n_targets,
            n_features: m.n_features,
            best_iteration: m.best_iteration,
            tree_weights: m.tree_weights,
            num_parallel_tree: m.num_parallel_tree,
            linear: m.linear,
            compact: OnceLock::new(),
        };
        model.validate_structure()?;
        Ok(model)
    }

    /// Save the model to a file in the native binary format.
    pub fn save_binary(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        Ok(std::fs::write(path, self.to_bytes()?)?)
    }

    /// Load a model from a native binary file.
    pub fn load_binary(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Serialize the model to a (human-readable) JSON string: the model's
    /// fields by name.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize a model from a JSON string produced by
    /// [`BoostedModel::to_json`].
    pub fn from_json(s: &str) -> Result<Self> {
        let model: Self = serde_json::from_str(s)?;
        model.validate_structure()?;
        Ok(model)
    }

    pub(crate) fn validate_structure(&self) -> Result<()> {
        if self.n_features == 0 {
            return Err(HessboostError::model_format(
                "model has an invalid feature count",
            ));
        }
        let vector = self.has_vector_leaves();
        // Both factors come from the file: check the product before any
        // divisibility or round-count arithmetic relies on it.
        let per_iteration = if vector {
            Some(self.num_parallel_tree)
        } else {
            self.n_outputs.checked_mul(self.num_parallel_tree)
        };
        if self.n_outputs == 0
            || self.n_targets == 0
            || self.num_parallel_tree == 0
            || (self.num_class >= 2 && self.n_outputs != self.num_class)
            || per_iteration.is_none_or(|per| !self.trees.len().is_multiple_of(per))
            || self.trees.iter().any(|tree| {
                tree.is_vector_leaf() != vector
                    || (vector && tree.size_leaf_vector() != self.n_outputs)
            })
        {
            return Err(HessboostError::ModelFormat(format!(
                "invalid output layout: {} outputs, num_class {}, num_parallel_tree {}, {} trees",
                self.n_outputs,
                self.num_class,
                self.num_parallel_tree,
                self.trees.len()
            )));
        }
        check_objective_width(
            &self.objective,
            &self.objective_params,
            self.num_class,
            self.n_targets,
            self.n_outputs,
        )?;
        if let Some(best) = self.best_iteration
            && self.linear.is_none()
            && best >= self.num_boost_rounds()
        {
            return Err(HessboostError::ModelFormat(format!(
                "best_iteration {best} is out of range for {} iterations",
                self.num_boost_rounds()
            )));
        }
        if self.base_score.len() != self.n_outputs()
            || self.base_score.iter().any(|v| !v.is_finite())
        {
            return Err(HessboostError::ModelFormat(format!(
                "base_score must hold one finite value per output ({} outputs, got {:?})",
                self.n_outputs(),
                self.base_score
            )));
        }
        if !self.tree_weights.is_empty() && self.tree_weights.len() != self.trees.len() {
            return Err(HessboostError::model_format(
                "tree_weights length does not match trees",
            ));
        }
        if self.tree_weights.iter().any(|weight| !weight.is_finite()) {
            return Err(HessboostError::model_format("tree weights must be finite"));
        }
        for (tree_id, tree) in self.trees.iter().enumerate() {
            if !tree.is_valid_for_features(self.n_features) {
                return Err(HessboostError::ModelFormat(format!(
                    "tree {tree_id} contains invalid nodes"
                )));
            }
        }
        if let Some(linear) = &self.linear {
            let outputs = self.n_outputs();
            if linear.bias.len() != outputs || linear.weights.len() != self.n_features * outputs {
                return Err(HessboostError::model_format(
                    "linear model dimensions are invalid",
                ));
            }
        }
        Ok(())
    }

    /// Save the model to a JSON file.
    pub fn save_json(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        Ok(std::fs::write(path, self.to_json()?)?)
    }

    /// Load a model from a JSON file.
    pub fn load_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    /// Serialize the model to XGBoost's JSON model schema (the text form of
    /// `booster.save_model("m.json")`), so XGBoost-compatible tooling can read
    /// it. See [`crate::model::export_xgboost_json`] for details and caveats.
    pub fn to_xgboost_json(&self) -> Result<String> {
        crate::model::export_xgboost_json(self)
    }

    /// Parse a model saved in XGBoost's JSON model schema. Best-effort for
    /// `gbtree` boosters and common objectives. See
    /// [`crate::model::import_xgboost_json`] for the mapping and limitations.
    pub fn from_xgboost_json(json: &str) -> Result<Self> {
        crate::model::import_xgboost_json(json)
    }

    /// Save the model to a file in XGBoost's JSON model format.
    pub fn save_xgboost_json(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        Ok(std::fs::write(path, self.to_xgboost_json()?)?)
    }

    /// Load a model from a file written in XGBoost's JSON model format.
    pub fn load_xgboost_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_xgboost_json(&std::fs::read_to_string(path)?)
    }

    /// Serialize the model to XGBoost's UBJSON model format (the bytes of
    /// `booster.save_model("m.ubj")` / `save_raw("ubj")`), carrying the same
    /// document as [`BoostedModel::to_xgboost_json`]. See
    /// [`crate::model::export_xgboost_ubjson`] for details.
    pub fn to_xgboost_ubjson(&self) -> Result<Vec<u8>> {
        crate::model::export_xgboost_ubjson(self)
    }

    /// Parse a model saved in XGBoost's UBJSON model format, with the same
    /// mapping and limitations as [`BoostedModel::from_xgboost_json`]. See
    /// [`crate::model::import_xgboost_ubjson`].
    pub fn from_xgboost_ubjson(bytes: &[u8]) -> Result<Self> {
        crate::model::import_xgboost_ubjson(bytes)
    }

    /// Save the model to a file in XGBoost's UBJSON model format (XGBoost's
    /// `.ubj` files).
    pub fn save_xgboost_ubjson(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        Ok(std::fs::write(path, self.to_xgboost_ubjson()?)?)
    }

    /// Load a model from a file written in XGBoost's UBJSON model format.
    pub fn load_xgboost_ubjson(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_xgboost_ubjson(&std::fs::read(path)?)
    }

    /// The objective the model was trained with, rebuilt from its name,
    /// `num_class` and retained parameters. Fails for objectives the crate
    /// cannot construct by name (custom objectives).
    pub(crate) fn rebuild_objective(&self) -> Result<Box<dyn crate::objective::Objective>> {
        rebuild_objective(
            &self.objective,
            &self.objective_params,
            self.num_class,
            self.n_targets,
        )
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

/// Check that `data` fits a model with `n_features` inputs and `n_outputs`
/// outputs: matching column count and a per-row or per-row-and-output
/// `base_margin`.
pub(crate) fn validate_prediction_data(
    n_features: usize,
    n_outputs: usize,
    data: &DMatrix,
) -> Result<()> {
    if data.n_cols() != n_features {
        return Err(HessboostError::dimension_mismatch(
            "prediction feature count",
            n_features,
            data.n_cols(),
        ));
    }
    let n = data.n_rows();
    if let Some(margin) = data.base_margin()
        && margin.len() != n
        && margin.len() != n * n_outputs
    {
        return Err(HessboostError::dimension_mismatch(
            "prediction base_margin length",
            n * n_outputs,
            margin.len(),
        ));
    }
    Ok(())
}

/// The objective named `objective`, rebuilt from its retained parameters.
/// Fails for objectives the crate cannot construct by name (custom
/// objectives).
fn rebuild_objective(
    objective: &str,
    params: &ObjectiveParams,
    num_class: usize,
    n_targets: usize,
) -> Result<Box<dyn crate::objective::Objective>> {
    let params = params
        .training_params(objective, num_class)
        .build_unchecked();
    create_objective(&params, n_targets)
}

/// Check that the objective a model names, when the crate can rebuild it,
/// produces the model's `n_outputs` outputs. Loaders call this before
/// returning a model: the prediction transform of a multi-output objective
/// works on `[row][output]` blocks of its own width, so a mismatched width
/// would transform values of neighboring rows together. Only objectives the
/// crate does not know by name (custom objectives) are skipped: they predict
/// margins. A built-in objective that cannot be rebuilt from the stored
/// configuration (e.g. a distribution with a label matrix) is a format error,
/// since predicting without its transform would misreport every output.
pub(crate) fn check_objective_width(
    objective: &str,
    params: &ObjectiveParams,
    num_class: usize,
    n_targets: usize,
    n_outputs: usize,
) -> Result<()> {
    match rebuild_objective(objective, params, num_class, n_targets) {
        Ok(rebuilt) if rebuilt.n_outputs() != n_outputs => {
            Err(HessboostError::ModelFormat(format!(
                "objective `{objective}` has {} outputs but the model stores {n_outputs}",
                rebuilt.n_outputs()
            )))
        }
        Ok(_)
        | Err(HessboostError::Unknown {
            kind: "objective", ..
        }) => Ok(()),
        Err(e) => Err(HessboostError::ModelFormat(format!(
            "objective `{objective}` cannot be rebuilt from the stored configuration: {e}"
        ))),
    }
}

/// Turn raw margins (`[row][output]`, `n_outputs` wide) of a model with the
/// given objective metadata into predictions in the objective's reported
/// space: the objective's transform (identity when it cannot be rebuilt,
/// e.g. a custom objective, mirroring how XGBoost returns margins then), and
/// for `multi:softmax` the per-row argmax class index encoded as `f32`.
pub(crate) fn transform_model_margins(
    objective: &str,
    params: &ObjectiveParams,
    num_class: usize,
    n_targets: usize,
    n_outputs: usize,
    mut margin: Vec<f32>,
) -> Vec<f32> {
    if let Ok(obj) = rebuild_objective(objective, params, num_class, n_targets) {
        obj.pred_transform(&mut margin);
    }
    if objective == "multi:softmax" {
        return margin
            .chunks_exact(n_outputs)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map_or(0.0, |(i, _)| i as f32)
            })
            .collect();
    }
    margin
}

/// Rows per prediction block: the block's feature rows stay in cache while
/// every tree walks them. Must be a multiple of [`LANES`].
const PREDICT_BLOCK_ROWS: usize = 256;
const _: () = assert!(PREDICT_BLOCK_ROWS.is_multiple_of(LANES));

/// Widest CSR matrix that is densified block-by-block for prediction. Wider
/// matrices fall back to per-lookup row scans.
const MAX_DENSIFY_COLS: usize = 4096;

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
    pub(super) fn new(data: &'a DMatrix) -> Self {
        Self::build(data, MAX_DENSIFY_COLS)
    }

    /// Blocks that are loaded one row at a time and always expose a dense row
    /// (for per-row algorithms such as TreeSHAP whose cost per row already
    /// scales with the feature count).
    pub(super) fn single_rows(data: &'a DMatrix) -> Self {
        Self::build(data, usize::MAX)
    }

    fn build(data: &'a DMatrix, max_densify_cols: usize) -> Self {
        let n_cols = data.n_cols();
        match data.dense_values() {
            Some(dense) if data.missing().is_nan() => RowBlock::View {
                data: dense,
                n_cols,
                start: 0,
                lanes: Vec::new(),
            },
            None if n_cols > max_densify_cols => RowBlock::Wide { data, start: 0 },
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
    pub(super) fn get(&self, r: usize, f: u32) -> Option<f32> {
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
            None => {
                for (t, slot) in out.iter_mut().enumerate() {
                    *slot = forest.original_id(forest.leaf_id_with(t, |f| self.get(r, f)));
                }
            }
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
            for t in trees {
                let leaf = forest.leaf_id_with(t, |f| self.get(r, f));
                out[scalar_tree_output(t, parallel, k)] += weight(t) * forest.leaf_value(leaf);
            }
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
            for t in trees {
                let leaf = forest.leaf_id_with(t, |f| self.get(r, f));
                let w = weight(t);
                for (o, &v) in out.iter_mut().zip(forest.leaf_vector(leaf, k)) {
                    *o += w * v;
                }
            }
        }
    }

    /// The loaded rows for the batch kernel: the full [`LANES`]-row groups in
    /// lane-major layout plus the remaining rows row-major (see
    /// [`CompactForest::accumulate`]), or `None` for wide sparse blocks.
    #[inline]
    fn lane_block(&self, rows: usize) -> Option<(&[u32], &[f32], usize)> {
        let tail_start = rows / LANES * LANES;
        match self {
            RowBlock::View {
                data,
                n_cols,
                start,
                lanes,
            } => Some((
                lanes,
                &data[(start + tail_start) * n_cols..(start + rows) * n_cols],
                *n_cols,
            )),
            RowBlock::Scratch {
                n_cols,
                scratch,
                lanes,
                ..
            } => Some((lanes, &scratch[..(rows - tail_start) * n_cols], *n_cols)),
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
        if let Some((lanes, tail, n_cols)) = self.lane_block(rows) {
            forest.original_leaf_ids(t, lanes, tail, n_cols, rows, out, stride);
        } else {
            for r in 0..rows {
                let leaf = forest.leaf_id_with(t, |f| self.get(r, f));
                out[r * stride] = forest.original_id(leaf);
            }
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
        if let Some((lanes, tail, n_cols)) = self.lane_block(rows) {
            forest.accumulate_vector(t, lanes, tail, n_cols, rows, stride, weight, out, stride);
        } else {
            for r in 0..rows {
                let leaf = forest.leaf_id_with(t, |f| self.get(r, f));
                let dst = &mut out[r * stride..(r + 1) * stride];
                for (o, &v) in dst.iter_mut().zip(forest.leaf_vector(leaf, stride)) {
                    *o += weight * v;
                }
            }
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
        if let Some((lanes, tail, n_cols)) = self.lane_block(rows) {
            forest.accumulate(t, lanes, tail, n_cols, rows, weight, out, stride);
        } else {
            for r in 0..rows {
                let leaf = forest.leaf_id_with(t, |f| self.get(r, f));
                out[r * stride] += weight * forest.leaf_value(leaf);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BoostedModel;
    use crate::config::TrainingParams;
    use crate::data::DMatrix;
    use crate::error::HessboostError;
    use crate::learner::train;
    use crate::test_support::labeled_dense;

    /// A two-output forest of `2^63` parallel trees overflows the trees per
    /// iteration: training refuses it as a parameter error instead of
    /// overflowing (or dividing by zero) in its iteration arithmetic, even
    /// for zero rounds.
    #[test]
    fn training_rejects_overflowing_iteration_size() {
        let x = [0.0f32, 1.0, 2.0, 3.0];
        let y = [0.0f32, 1.0, 1.0, 0.0, 2.0, 3.0, 3.0, 2.0];
        let d = DMatrix::from_dense(&x, 4, 1)
            .unwrap()
            .with_label_matrix(&y, 2)
            .unwrap();
        let params = TrainingParams::builder()
            .num_parallel_tree(1usize << (usize::BITS - 1))
            .build()
            .unwrap();
        for rounds in [0, 1] {
            assert!(matches!(
                train(&params, &d, rounds),
                Err(HessboostError::InvalidParameter { name, .. }) if name == "num_parallel_tree"
            ));
        }
    }

    /// Only unknown (custom) objective names skip the objective check: a
    /// built-in objective that cannot be rebuilt from the stored
    /// configuration (here a single-target objective with two label columns)
    /// is a format error, not a silently untransformed model.
    #[test]
    fn loading_propagates_invalid_builtin_objective() {
        let d = labeled_dense(&[0.0, 1.0], 2, 1, &[0.0, 1.0]);
        let model = train(&TrainingParams::default(), &d, 1).unwrap();
        let mut value: serde_json::Value = serde_json::from_str(&model.to_json().unwrap()).unwrap();
        value["n_targets"] = 2.into();
        value["objective"] = "count:poisson".into();
        assert!(matches!(
            BoostedModel::from_json(&value.to_string()),
            Err(HessboostError::ModelFormat(_))
        ));
        value["objective"] = "my:custom".into();
        let custom = BoostedModel::from_json(&value.to_string()).unwrap();
        assert_eq!(custom.objective(), "my:custom");
    }
}
