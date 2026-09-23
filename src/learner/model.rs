//! The trained model: an ensemble of trees plus the metadata needed to turn
//! their sum into calibrated predictions.

use crate::config::ObjectiveParams;
use crate::data::DMatrix;
use crate::error::Result;
use crate::objective::create_objective;
use crate::tree::RegTree;
use crate::tree::compact::{CompactForest, FEATURE_LANES, LANES, key};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

// Native binary format marker; changing it breaks loading existing models.
const NATIVE_MAGIC: &[u8; 4] = b"SQB\0";
const NATIVE_VERSION: u8 = 1;

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoostedModel {
    trees: Vec<RegTree>,
    /// Per-output intercept in margin space (length `n_outputs`).
    base_score: Vec<f32>,
    /// The objective's XGBoost name (`Objective::name`), which drives the
    /// prediction transform.
    objective: String,
    /// Objective hyper-parameters, retained for XGBoost-format export and for
    /// rebuilding the objective (XGBoost defaults when absent).
    #[serde(default)]
    objective_params: ObjectiveParams,
    /// The configured `num_class` (`0` for scalar objectives).
    num_class: usize,
    /// Raw outputs per instance: `num_class` for multiclass objectives, the
    /// objective's own output count otherwise (custom objectives may have
    /// several). Trees are laid out round-robin over outputs.
    n_outputs: usize,
    n_features: usize,
    /// The best iteration index selected by early stopping, if any.
    best_iteration: Option<usize>,
    /// Per-tree contribution weights. For a plain `gbtree` model every weight is
    /// `1.0`. The DART booster stores fractional weights here so dropped trees
    /// can be rescaled. Defaults to empty for models serialized before this
    /// field existed, in which case every tree is treated as weight `1.0`.
    #[serde(default)]
    tree_weights: Vec<f32>,
    /// Linear (coordinate-descent) booster parameters. `Some` only for
    /// `gblinear` models, in which case predictions come from the linear model
    /// and the `trees` vector is empty. Defaults to `None` for tree ensembles
    /// and for models serialized before this field existed.
    #[serde(default)]
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
    /// legacy models or plain `gbtree`).
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
        self.accumulate_forest(data, &mut out, self.trees.len(), weight);
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
            n_features: spec.n_features,
            best_iteration: None,
            tree_weights,
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

    /// Number of trees (boosting rounds × outputs).
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Number of raw outputs per instance: `num_class` for multiclass, the
    /// objective's output count otherwise (`1` for every built-in scalar
    /// objective; custom objectives may declare more).
    #[inline]
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// Number of boosting rounds (`num_trees / n_outputs`).
    pub fn num_boost_rounds(&self) -> usize {
        self.trees.len() / self.n_outputs()
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

    /// Number of trees to use at prediction time: `(best_iteration + 1) ×
    /// n_outputs` when early stopping selected one, else all trees.
    pub(crate) fn effective_ntrees(&self) -> usize {
        match self.best_iteration {
            Some(it) => (it + 1) * self.n_outputs(),
            None => self.trees.len(),
        }
    }

    /// Raw margin predictions using the effective tree count. The output is laid
    /// out `[instance][output]` (length `n_rows × n_outputs`).
    pub fn predict_margin(&self, data: &DMatrix) -> Result<Vec<f32>> {
        self.predict_margin_limited(data, self.effective_ntrees())
    }

    /// Raw margin predictions using only the first `ntree_limit` trees
    /// (`0` = all trees, ignoring early stopping). Tree `t` contributes to
    /// output `t % n_outputs`.
    pub fn predict_margin_limited(&self, data: &DMatrix, ntree_limit: usize) -> Result<Vec<f32>> {
        self.validate_prediction_data(data)?;
        Ok(self.predict_margin_limited_unchecked(data, ntree_limit))
    }

    pub(crate) fn predict_margin_limited_unchecked(
        &self,
        data: &DMatrix,
        ntree_limit: usize,
    ) -> Vec<f32> {
        let n = data.n_rows();
        let k = self.n_outputs();
        // A gblinear model predicts from its linear parameters and ignores the
        // (empty) tree ensemble: margin(row, k) = base_score[k] + bias[k] +
        // Σ_f weights[f][k] * x[row, f], with missing features contributing 0.
        if let Some(lm) = &self.linear {
            let mut out = self.initial_margins(data);
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
        let limit = if ntree_limit == 0 {
            self.trees.len()
        } else {
            ntree_limit.min(self.trees.len())
        };
        // Initialize from the dataset's per-instance base margin when present
        // (it overrides the per-output intercepts, matching XGBoost); otherwise
        // use the trained global bias.
        let mut out = self.initial_margins(data);
        self.accumulate_forest(data, &mut out, limit, |ti| self.tree_weight(ti));
        out
    }

    /// Sum `weight(t) * leaf(row, t)` into `out[row * k + t % k]` for trees
    /// `0..limit`, where `k` is the output count. Rows are traversed in
    /// cache-friendly blocks (parallel across blocks); per (row, output) slot
    /// the trees are still summed in ascending order, so the result is
    /// bit-identical to the sequential tree-outer loop.
    fn accumulate_forest(
        &self,
        data: &DMatrix,
        out: &mut [f32],
        limit: usize,
        weight: impl Fn(usize) -> f32 + Sync,
    ) {
        let k = self.n_outputs();
        self.traverse_blocks(
            data,
            out,
            k,
            limit,
            |block, forest, r, out_row| block.accumulate_row(forest, r, limit, &weight, out_row),
            |block, forest, ti, rows, out_block, stride| {
                block.accumulate(
                    forest,
                    ti,
                    rows,
                    weight(ti),
                    &mut out_block[ti % k..],
                    stride,
                );
            },
        );
    }

    /// Block-parallel traversal of trees `0..limit` over `data`, writing into
    /// `out` laid out `[row][stride]`: `row_op` handles one loaded row of the
    /// small-batch path, `tree_op` one tree over a loaded block of rows. Tiny
    /// batches (online serving) skip the thread pool and overlap the trees of
    /// each row instead of the rows of each tree; larger inputs process rows
    /// in blocks whose feature rows stay in cache while every tree walks
    /// them, with blocks running in parallel.
    fn traverse_blocks<T: Send>(
        &self,
        data: &DMatrix,
        out: &mut [T],
        stride: usize,
        limit: usize,
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
                    for ti in 0..limit {
                        tree_op(block, forest, ti, rows, out_block, stride);
                    }
                },
            );
    }

    /// Predictions in the objective's reported space. `multi:softprob` returns
    /// an `n_rows × num_class` probability matrix while `multi:softmax` returns
    /// one class index per row, encoded as `f32`.
    pub fn predict(&self, data: &DMatrix) -> Result<Vec<f32>> {
        let mut margin = self.predict_margin(data)?;
        // A model trained with a custom objective cannot reconstruct its
        // transform from the name; fall back to the identity (raw margins),
        // mirroring how XGBoost returns margins for custom objectives.
        if let Ok(obj) = self.rebuild_objective() {
            obj.pred_transform(&mut margin);
        }
        if self.objective == "multi:softmax" {
            let k = self.n_outputs();
            return Ok(margin
                .chunks_exact(k)
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map_or(0.0, |(i, _)| i as f32)
                })
                .collect());
        }
        Ok(margin)
    }

    /// For multiclass, the predicted class index per row (argmax over classes).
    /// For single-output models this returns the transformed prediction rounded
    /// to the nearest class at 0.5.
    pub fn predict_class(&self, data: &DMatrix) -> Result<Vec<u32>> {
        let probs = self.predict(data)?;
        let k = self.n_outputs();
        let n = data.n_rows();
        let mut out = vec![0u32; n];
        if k == 1 {
            for (i, o) in out.iter_mut().enumerate() {
                *o = u32::from(probs[i] > 0.5);
            }
        } else if self.objective == "multi:softmax" {
            for (dst, &class) in out.iter_mut().zip(&probs) {
                *dst = class as u32;
            }
        } else {
            for i in 0..n {
                out[i] = crate::simd::argmax_scalar(&probs[i * k..i * k + k]) as u32;
            }
        }
        Ok(out)
    }

    /// Per-row leaf indices for each tree (shape `n_rows × num_trees`, row-major).
    #[allow(
        clippy::redundant_closure_for_method_calls,
        reason = "the method path is not general enough over the block lifetime"
    )]
    pub fn predict_leaf(&self, data: &DMatrix) -> Result<Vec<u32>> {
        self.validate_prediction_data(data)?;
        let n = data.n_rows();
        let t = self.trees.len();
        let mut out = vec![0u32; n * t];
        if t == 0 {
            return Ok(out);
        }
        self.traverse_blocks(
            data,
            &mut out,
            t,
            t,
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
        let n = data.n_rows();
        let k = self.n_outputs();
        match data.base_margin() {
            Some(bm) if bm.len() == n * k => bm.to_vec(),
            Some(bm) if bm.len() == n => {
                bm.iter().flat_map(|&m| std::iter::repeat_n(m, k)).collect()
            }
            _ => {
                let mut out = Vec::with_capacity(n * k);
                for _ in 0..n {
                    out.extend_from_slice(&self.base_score);
                }
                out
            }
        }
    }

    /// Validated prologue for the TreeSHAP paths. `predict_leaf` is excluded:
    /// it walks all trees (not the effective prefix) and needs no margins.
    pub(super) fn attribution_prologue(&self, data: &DMatrix) -> Result<AttributionPrologue<'_>> {
        self.validate_prediction_data(data)?;
        let n = data.n_rows();
        let k = self.n_outputs();
        let nf = self.n_features;
        Ok(AttributionPrologue {
            n,
            k,
            nf,
            width: nf + 1,
            trees: &self.trees[..self.effective_ntrees()],
            initial: self.initial_margins(data),
        })
    }

    pub(crate) fn validate_prediction_data(&self, data: &DMatrix) -> Result<()> {
        if data.n_cols() != self.n_features {
            return Err(crate::error::HessboostError::DimensionMismatch {
                what: "prediction feature count",
                expected: self.n_features,
                got: data.n_cols(),
            });
        }
        let n = data.n_rows();
        let k = self.n_outputs();
        if let Some(margin) = data.base_margin()
            && margin.len() != n
            && margin.len() != n * k
        {
            return Err(crate::error::HessboostError::DimensionMismatch {
                what: "prediction base_margin length",
                expected: n * k,
                got: margin.len(),
            });
        }
        Ok(())
    }

    /// Serialize the model to a compact Postcard binary blob.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let payload = postcard::to_stdvec(self)
            .map_err(|e| crate::error::HessboostError::ModelFormat(e.to_string()))?;
        let mut bytes = Vec::with_capacity(NATIVE_MAGIC.len() + 1 + payload.len());
        bytes.extend_from_slice(NATIVE_MAGIC);
        bytes.push(NATIVE_VERSION);
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    /// Deserialize a model from a binary blob produced by [`BoostedModel::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < NATIVE_MAGIC.len() + 1 || &bytes[..NATIVE_MAGIC.len()] != NATIVE_MAGIC {
            return Err(crate::error::HessboostError::ModelFormat(
                "invalid native model header".to_string(),
            ));
        }
        if bytes[NATIVE_MAGIC.len()] != NATIVE_VERSION {
            return Err(crate::error::HessboostError::ModelFormat(format!(
                "unsupported native model version {}",
                bytes[NATIVE_MAGIC.len()]
            )));
        }
        let model: Self = postcard::from_bytes(&bytes[NATIVE_MAGIC.len() + 1..])
            .map_err(|e| crate::error::HessboostError::ModelFormat(e.to_string()))?;
        model.validate_structure()?;
        Ok(model)
    }

    /// Save the model to a file in the native binary format.
    pub fn save_binary(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path, self.to_bytes()?)?;
        Ok(())
    }

    /// Load a model from a native binary file.
    pub fn load_binary(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Serialize the model to a (human-readable) JSON string.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize a model from a JSON string.
    pub fn from_json(s: &str) -> Result<Self> {
        let model: Self = serde_json::from_str(s)?;
        model.validate_structure()?;
        Ok(model)
    }

    pub(crate) fn validate_structure(&self) -> Result<()> {
        use crate::error::HessboostError;

        if self.n_features == 0 {
            return Err(HessboostError::ModelFormat(
                "model has an invalid feature count".to_string(),
            ));
        }
        if self.n_outputs == 0
            || (self.num_class >= 2 && self.n_outputs != self.num_class)
            || !self.trees.len().is_multiple_of(self.n_outputs)
        {
            return Err(HessboostError::ModelFormat(format!(
                "invalid output layout: {} outputs, num_class {}, {} trees",
                self.n_outputs,
                self.num_class,
                self.trees.len()
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
            return Err(HessboostError::ModelFormat(
                "tree_weights length does not match trees".to_string(),
            ));
        }
        if self.tree_weights.iter().any(|weight| !weight.is_finite()) {
            return Err(HessboostError::ModelFormat(
                "tree weights must be finite".to_string(),
            ));
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
                return Err(HessboostError::ModelFormat(
                    "linear model dimensions are invalid".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Save the model to a JSON file.
    pub fn save_json(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path, self.to_json()?)?;
        Ok(())
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
        std::fs::write(path, self.to_xgboost_json()?)?;
        Ok(())
    }

    /// Load a model from a file written in XGBoost's JSON model format.
    pub fn load_xgboost_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_xgboost_json(&std::fs::read_to_string(path)?)
    }

    /// The objective the model was trained with, rebuilt from its name,
    /// `num_class` and retained parameters. Fails for objectives the crate
    /// cannot construct by name (custom objectives).
    pub(crate) fn rebuild_objective(&self) -> Result<Box<dyn crate::objective::Objective>> {
        let params = self
            .objective_params
            .training_params(&self.objective, self.num_class)
            .build_unchecked();
        create_objective(&params)
    }
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
            Some(_) => RowBlock::Scratch {
                source: data,
                n_cols,
                scratch: Vec::new(),
                lanes: Vec::new(),
                tail_start: 0,
            },
            None if n_cols <= max_densify_cols => RowBlock::Scratch {
                source: data,
                n_cols,
                scratch: Vec::new(),
                lanes: Vec::new(),
                tail_start: 0,
            },
            None => RowBlock::Wide { data, start: 0 },
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
                Self::fill_lanes(
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
                // Destination of feature `f` of block row `r`: a keyed lane
                // slot (its negated key follows `LANES` later) or a tail slot.
                let slot = |r: usize, f: usize| -> (bool, usize) {
                    if r < groups * LANES {
                        (
                            true,
                            (r / LANES) * FEATURE_LANES * n_cols + f * FEATURE_LANES + r % LANES,
                        )
                    } else {
                        (false, (r - groups * LANES) * n_cols + f)
                    }
                };
                if let Some(dense) = source.dense_values() {
                    for r in 0..rows {
                        let src = &dense[(start + r) * n_cols..(start + r + 1) * n_cols];
                        for (f, &v) in src.iter().enumerate() {
                            let v = if v == missing { f32::NAN } else { v };
                            match slot(r, f) {
                                (true, i) => {
                                    lanes[i] = key(v);
                                    lanes[i + LANES] = key(-v);
                                }
                                (false, i) => scratch[i] = v,
                            }
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
                            match slot(r, indices[k] as usize) {
                                (true, i) => {
                                    lanes[i] = key(v);
                                    lanes[i + LANES] = key(-v);
                                }
                                (false, i) => scratch[i] = v,
                            }
                        }
                    }
                }
            }
        }
    }

    /// Transpose the full [`LANES`]-row groups of the row-major `rows` into
    /// `lanes` as `[group][feature][lane]` keys of `v` and of `-v`
    /// ([`FEATURE_LANES`] per feature), so a lane's key sits at a fixed
    /// immediate offset from the node's slot.
    fn fill_lanes(lanes: &mut Vec<u32>, rows: &[f32], n_cols: usize) {
        let groups = rows.len() / n_cols / LANES;
        lanes.clear();
        lanes.resize(groups * FEATURE_LANES * n_cols, 0);
        for (g, dst) in lanes.chunks_exact_mut(FEATURE_LANES * n_cols).enumerate() {
            let src = &rows[g * LANES * n_cols..(g + 1) * LANES * n_cols];
            for (j, row) in src.chunks_exact(n_cols).enumerate() {
                for (f, &v) in row.iter().enumerate() {
                    dst[f * FEATURE_LANES + j] = key(v);
                    dst[f * FEATURE_LANES + LANES + j] = key(-v);
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

    /// `out[t % k] += weight(t) * leaf_value(row r, tree t)` for trees
    /// `0..limit` of loaded row `r`.
    fn accumulate_row(
        &self,
        forest: &CompactForest,
        r: usize,
        limit: usize,
        weight: impl Fn(usize) -> f32,
        out: &mut [f32],
    ) {
        if let Some(row) = self.row(r) {
            forest.accumulate_row(row, limit, weight, out);
        } else {
            let k = out.len();
            for t in 0..limit {
                let leaf = forest.leaf_id_with(t, |f| self.get(r, f));
                out[t % k] += weight(t) * forest.leaf_value(leaf);
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
    use crate::learner::train;

    fn small_model() -> (BoostedModel, DMatrix) {
        let n = 60;
        let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let y: Vec<f32> = x.iter().map(|&v| if v > 0.5 { 1.0 } else { 0.0 }).collect();
        let d = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("binary:logistic")
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        (train(&params, &d, 20).unwrap(), d)
    }

    #[test]
    fn binary_roundtrip_preserves_predictions() {
        let (model, d) = small_model();
        let before = model.predict(&d).unwrap();
        let bytes = model.to_bytes().unwrap();
        let restored = BoostedModel::from_bytes(&bytes).unwrap();
        let after = restored.predict(&d).unwrap();
        assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(&after) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(restored.num_trees(), model.num_trees());
        assert_eq!(restored.objective(), model.objective());
    }

    #[test]
    fn json_roundtrip_preserves_predictions() {
        let (model, d) = small_model();
        let before = model.predict(&d).unwrap();
        let json = model.to_json().unwrap();
        let restored = BoostedModel::from_json(&json).unwrap();
        let after = restored.predict(&d).unwrap();
        for (a, b) in before.iter().zip(&after) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
