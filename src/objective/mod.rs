//! Learning objectives: gradients, Hessians, prediction transforms, and base
//! score estimation.
//!
//! Every objective implements [`Objective`]. Boosting works in *margin* space
//! (raw additive scores). The [`Objective::pred_transform`] maps margins to the
//! reported prediction (e.g. the logistic sigmoid). This mirrors XGBoost's
//! separation of `GetGradient` / `PredTransform`.

mod absolute;
mod classification;
mod count;
mod custom;
pub mod distributional;
mod multi_target;
mod multiclass;
mod quantile;
mod ranking;
mod regression;
mod survival;

pub use absolute::AbsoluteError;
pub use classification::{Hinge, Logistic};
pub use count::{Gamma, Poisson, Tweedie};
pub use custom::CustomObjective;
pub use multiclass::Softmax;
pub use quantile::{Expectile, Quantile};
pub use ranking::LambdaMart;
pub use regression::{PseudoHuber, SquaredError, SquaredLogError};

pub(crate) use quantile::validate_alphas;

pub use survival::{Aft, Cox};

pub(crate) use survival::{abs_label_order, aft_nloglik};

use rayon::prelude::*;

use crate::config::TrainingParams;
use crate::data::MetaInfo;
use crate::error::{HessboostError, Result};
use distributional::{DistFamily, DistObjective};

/// A first- and second-order gradient for one instance/output: a fixed
/// pair, built with [`GradPair::new`] or a struct literal.
///
/// Stored as `f32` to match XGBoost's memory layout and to keep histogram
/// accumulation cache-friendly.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct GradPair {
    /// First-order gradient of the loss w.r.t. the margin.
    pub grad: f32,
    /// Second-order gradient (Hessian) of the loss w.r.t. the margin.
    pub hess: f32,
}

impl GradPair {
    /// Construct a gradient pair.
    #[inline]
    pub fn new(grad: f32, hess: f32) -> Self {
        GradPair { grad, hess }
    }
}

/// A per-row loss `ℓ(margin, label)`, unweighted by the sample weight, whose
/// first and second derivatives with respect to the margin are the gradient
/// pairs the objective produces (up to Hessian safeguards such as
/// `max_delta_step`). Returned by [`Objective::pointwise_loss`].
pub type PointwiseLoss<'a> = Box<dyn Fn(f32, f32) -> f64 + Send + Sync + 'a>;

/// Reduced gradients a custom objective supplies for the *split search* of
/// vector-leaf trees (see [`Objective::split_gradient`]). Build with
/// [`SplitGradient::new`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SplitGradient {
    /// Row-major `[row][target]` gradient pairs, `n_targets` per row.
    pub gpair: Vec<GradPair>,
    /// Split targets per row (at least `1`; usually far fewer than the
    /// model's outputs).
    pub n_targets: usize,
}

impl SplitGradient {
    /// Row-major `[row][target]` gradient pairs with `n_targets` per row.
    pub fn new(gpair: Vec<GradPair>, n_targets: usize) -> Self {
        SplitGradient { gpair, n_targets }
    }
}

/// Rows per parallel gradient chunk. A multiple of every vector kernel's block
/// (4 rows, and 4 values for any class count), so chunk boundaries fall where
/// the kernels' block boundaries already are and every element is computed
/// by the same path as in one whole-batch call.
const GRADIENT_CHUNK_ROWS: usize = 8192;

/// Lower bound on any per-instance Hessian, matching XGBoost's guard, so that
/// confidently-classified instances still contribute a positive Hessian.
pub(crate) const MIN_HESS: f32 = 1e-16;
/// XGBoost's `f64` Hessian floor (AFT `kMinHessian`, LambdaRank `Eps64`):
/// the `f64` literal, not [`MIN_HESS`] widened.
pub(crate) const MIN_HESS_F64: f64 = 1e-16;

/// Run a row-independent gradient `kernel` over `n_rows` instances with
/// `n_outputs` values each, in parallel row chunks when the batch is large and
/// a thread pool is available. Every row's outputs depend only on that row,
/// and the chunking is fixed (not thread-count dependent): a short final
/// chunk is folded into the last full chunk so every row takes the same
/// vector/scalar path as in one whole-batch call, and the result is
/// identical. Debug builds first assert the shapes with
/// [`check_gradient_inputs`].
pub(crate) fn rowwise_gradient<K>(
    n_rows: usize,
    n_outputs: usize,
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    out: &mut [GradPair],
    kernel: K,
) where
    K: Fn(&[f32], &[f32], Option<&[f32]>, &mut [GradPair]) + Sync,
{
    check_gradient_inputs(n_rows, n_outputs, preds, labels, weights, out);
    let complete = n_rows
        .checked_mul(n_outputs)
        .is_some_and(|values| preds.len() == values && out.len() == values)
        && labels.len() == n_rows
        && weights.is_none_or(|w| w.len() == n_rows);
    if !complete
        || n_rows == 0
        || n_outputs == 0
        || n_rows < 2 * GRADIENT_CHUNK_ROWS
        || rayon::current_num_threads() <= 1
    {
        kernel(preds, labels, weights, out);
        return;
    }
    let run_chunk = |first: usize, out: &mut [GradPair]| {
        let rows = out.len() / n_outputs;
        kernel(
            &preds[first * n_outputs..(first + rows) * n_outputs],
            &labels[first..first + rows],
            weights.map(|w| &w[first..first + rows]),
            out,
        );
    };
    let chunk_values = GRADIENT_CHUNK_ROWS * n_outputs;
    if out.len().is_multiple_of(chunk_values) {
        out.par_chunks_mut(chunk_values)
            .enumerate()
            .for_each(|(index, out)| run_chunk(index * GRADIENT_CHUNK_ROWS, out));
        return;
    }
    // A separate short tail chunk would compute its rows on the scalar path
    // where a whole-batch call vectorizes them (or vice versa); fold it into
    // the last full chunk so every row keeps the whole-batch vector/scalar
    // split — chunk starts stay multiples of every kernel block.
    let head_rows = (n_rows / GRADIENT_CHUNK_ROWS - 1) * GRADIENT_CHUNK_ROWS;
    let (head, tail) = out.split_at_mut(head_rows * n_outputs);
    rayon::join(
        || {
            head.par_chunks_mut(chunk_values)
                .enumerate()
                .for_each(|(index, out)| run_chunk(index * GRADIENT_CHUNK_ROWS, out));
        },
        || run_chunk(head_rows, tail),
    );
}

/// [`rowwise_gradient`] for a single-output objective whose pair depends only
/// on the row's margin, label, and weight (`1` without weights):
/// `out[i] = pair(preds[i], labels[i], w_i)`.
pub(crate) fn elementwise_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    out: &mut [GradPair],
    pair: impl Fn(f32, f32, f32) -> GradPair + Sync,
) {
    rowwise_gradient(
        labels.len(),
        1,
        preds,
        labels,
        weights,
        out,
        |preds, labels, weights, out| {
            for i in 0..preds.len() {
                let w = weights.map_or(1.0, |ws| ws[i]);
                out[i] = pair(preds[i], labels[i], w);
            }
        },
    );
}

/// The pairs `objective.gradient` writes for `preds` (one per margin).
#[cfg(test)]
fn gradient_pairs(
    objective: &dyn Objective,
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
) -> Vec<GradPair> {
    let mut out = vec![GradPair::default(); preds.len()];
    objective.gradient(preds, labels, weights, &mut out);
    out
}

/// The intercepts `objective.base_margins_info` estimates from single-target
/// labels and weights.
#[cfg(test)]
fn base_margins(objective: &dyn Objective, labels: &[f32], weights: Option<&[f32]>) -> Vec<f32> {
    objective.base_margins_info(&MetaInfo::new(labels, weights, None))
}

/// Debug-only shape check shared by every [`Objective::gradient`]: `preds` and
/// `out` hold `n_rows * n_outputs` values while `labels` (and `weights`, when
/// present) hold one per row. Release builds skip it; [`rowwise_gradient`]
/// runs it and re-validates the same shapes at runtime for its chunking
/// decision.
pub(crate) fn check_gradient_inputs(
    n_rows: usize,
    n_outputs: usize,
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    out: &[GradPair],
) {
    debug_assert_eq!(preds.len(), n_rows * n_outputs);
    debug_assert_eq!(out.len(), n_rows * n_outputs);
    debug_assert_eq!(labels.len(), n_rows);
    if let Some(w) = weights {
        debug_assert_eq!(w.len(), n_rows);
    }
}

/// A differentiable learning objective.
///
/// Implementors are `Send + Sync` so gradient computation can be parallelized.
pub trait Objective: Send + Sync {
    /// The XGBoost-compatible objective name (e.g. `"reg:squarederror"`).
    fn name(&self) -> &str;

    /// Number of raw outputs produced per instance. `1` for regression and
    /// binary classification. It is `num_class` for multiclass objectives and
    /// the label-column count for a multi-target (label matrix) objective.
    fn n_outputs(&self) -> usize {
        1
    }

    /// Compute per-instance gradients and Hessians.
    ///
    /// `preds` holds raw margins laid out as `n_rows * n_outputs` (row-major by
    /// instance). `out` is written in the same layout. `weights`, if present,
    /// scales each instance's contribution.
    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    );

    /// Compute gradients with optional query-group structure.
    ///
    /// Learning-to-rank objectives (LambdaMART) override this to form document
    /// pairs *within* each group supplied by `group`. The default forwards to
    /// [`Objective::gradient`], ignoring the grouping. This is correct for all
    /// non-ranking objectives.
    fn gradient_grouped(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        _group: Option<&crate::data::GroupInfo>,
        out: &mut [GradPair],
    ) {
        self.gradient(preds, labels, weights, out);
    }

    /// Compute gradients from a dataset's full metadata view.
    ///
    /// This is the entry point the training loops call. The default forwards
    /// the labels, weights, and groups to [`Objective::gradient_grouped`];
    /// objectives that read other metadata (label bounds, several targets per
    /// row) override it.
    fn gradient_info(&self, preds: &[f32], info: &MetaInfo, out: &mut [GradPair]) {
        self.gradient_grouped(preds, info.labels, info.weights, info.group, out);
    }

    /// Whether the Hessian is constant across margins (XGBoost
    /// `ObjInfo::const_hess`); only `reg:squarederror` returns `true`.
    fn const_hess(&self) -> bool {
        false
    }

    /// Transform raw margins into reported predictions, in place. Default is the
    /// identity (used by squared-error regression). An objective whose
    /// transform is an inverse link also overrides
    /// [`Objective::probs_to_margins`] with the link, which the intercept
    /// estimate and a user-supplied `base_score` go through.
    fn pred_transform(&self, _preds: &mut [f32]) {}

    /// Estimate the per-output intercepts in *margin* space from a dataset's
    /// full metadata view; the single hook training uses to initialize the
    /// model's `base_score` when the user does not supply one. Returns
    /// exactly [`Objective::n_outputs`] values. To estimate from bare
    /// labels, weights, and groups, pass [`MetaInfo::new`].
    ///
    /// The default is XGBoost's `FitIntercept::InitEstimation`: one Newton
    /// step from all-zero margins over [`Objective::gradient_info`] on `info`
    /// itself (so label bounds and label matrices reach the gradient),
    /// `w_k = -Σg_k / max(Σh_k, 1e-6)` (sums in `f64`, step rounded to
    /// `f32`), mapped through [`Objective::pred_transform`] and back through
    /// [`Objective::probs_to_margins`] to reproduce XGBoost's `f32` rounding;
    /// `NaN` intercepts (which training refuses) when `info`'s lengths are
    /// inconsistent. Objectives whose optimal constant has a closed form
    /// (label mean, class frequencies) override it.
    fn base_margins_info(&self, info: &MetaInfo) -> Vec<f32> {
        newton_intercepts(self, info)
    }

    /// Transform raw margins into the values evaluation metrics receive
    /// (XGBoost `EvalTransform`), in place. Defaults to
    /// [`Objective::pred_transform`].
    fn eval_transform(&self, preds: &mut [f32]) {
        self.pred_transform(preds);
    }

    /// Map one row of prediction-space intercepts (length
    /// [`Objective::n_outputs`]) to margin space, in place: XGBoost
    /// `ProbToMargin` on the base-score vector, in `f32`. Training applies
    /// it to a user-supplied `base_score`, XGBoost-model import to the
    /// stored one, and the default [`Objective::base_margins_info`] to its
    /// Newton step. Default is the identity; objectives with a link
    /// function (e.g. the logit of `binary:logistic`) override it.
    fn probs_to_margins(&self, _scores: &mut [f32]) {}

    /// Map one row of margin-space intercepts back to the prediction space
    /// XGBoost stores `base_score` in (the inverse of
    /// [`Objective::probs_to_margins`]), in place; used by XGBoost-JSON
    /// export. Defaults to [`Objective::pred_transform`], which inverts the
    /// link of every objective whose transform is its link; objectives whose
    /// transform is not the inverse link (`binary:hinge` thresholds and
    /// `reg:quantileerror` sorts, while their `ProbToMargin` is the identity)
    /// override it.
    fn margins_to_probs(&self, margins: &mut [f32]) {
        self.pred_transform(margins);
    }

    /// Validate a dataset's labels and metadata for this objective. Training
    /// calls it for the training matrix and every evaluation set before the
    /// first round. The default accepts everything.
    ///
    /// Error messages refer to the data as "dataset"; training inserts the
    /// dataset's name after that word.
    fn validate_info(&self, _info: &MetaInfo) -> Result<()> {
        Ok(())
    }

    /// Whether the objective needs ordinary labels. `true` by default;
    /// objectives that learn from other metadata only (e.g. label bounds)
    /// return `false`, and training then accepts datasets without labels.
    fn requires_labels(&self) -> bool {
        true
    }

    /// Reduced split gradients for vector-leaf trees (XGBoost 3.2+'s
    /// `TreeObjective.split_grad`, the idea of `SketchBoost`).
    ///
    /// With `multi_strategy = multi_output_tree`, training calls this every
    /// round (`iteration` counts from 0) with that round's full gradients
    /// `gpair` (`[row][output]`, [`Objective::n_outputs`] pairs per row, row
    /// weights applied). Returning `Some` grows the tree's structure — its
    /// histograms, split search and internal weights — from the returned
    /// (typically much narrower) gradients, while every leaf's weight vector
    /// is still fit from `gpair` over the rows that reach it. `None`, the
    /// default and what every built-in objective returns, grows the tree
    /// from the full gradients. Training rejects a `Some` for the other
    /// strategies and together with monotone constraints, as XGBoost does.
    fn split_gradient(&self, _iteration: usize, _gpair: &[GradPair]) -> Option<SplitGradient> {
        None
    }
    /// The objective's per-row loss, for trainers that measure the actual
    /// loss reduction of a tree (budget-mode training,
    /// [`train_with_budget`](crate::training::budget::train_with_budget)).
    /// Label-dependent reweighting the gradient applies (e.g.
    /// `scale_pos_weight`) is part of the loss; the sample weight is not.
    /// Losses are shifted so a perfect prediction of a hard label scores `0`
    /// (deviance form), which makes relative loss reductions meaningful.
    /// `None` (the default) when the objective has no single-row loss, e.g.
    /// ranking, multi-output, or custom objectives.
    fn pointwise_loss(&self) -> Option<PointwiseLoss<'_>> {
        None
    }

    /// The default evaluation metric for this objective, as XGBoost's
    /// `DefaultEvalMetric` names it — including any configuration-dependent
    /// suffix such as `ndcg@32` (LambdaRank's top-k) or `tweedie-nloglik@1.5`.
    fn default_metric(&self) -> String;
}

/// One Newton step from all-zero margins, per output: `w_k = -Σg_k /
/// max(Σh_k, 1e-6)` with the sums in `f64` and the step rounded to `f32`, then
/// mapped through [`Objective::pred_transform`] and back through
/// [`Objective::probs_to_margins`]. XGBoost (`FitIntercept::InitEstimation` +
/// `tree::FitStump`) stores the intercept in prediction space and re-applies
/// the link on use; taking the same round trip reproduces its `f32` rounding.
/// `NaN` for every output when `info` fails [`MetaInfo::check_layout`] or
/// the margin buffer would overflow, before anything is allocated.
pub(crate) fn newton_intercepts<O: Objective + ?Sized>(objective: &O, info: &MetaInfo) -> Vec<f32> {
    let k = objective.n_outputs();
    let Some(len) = info
        .n_rows
        .checked_mul(k)
        .filter(|_| info.check_layout().is_ok())
    else {
        return vec![f32::NAN; k];
    };
    let zeros = vec![0.0f32; len];
    let mut gpair = vec![GradPair::default(); len];
    objective.gradient_info(&zeros, info, &mut gpair);
    let mut out = fit_stump(&gpair, k);
    objective.pred_transform(&mut out);
    objective.probs_to_margins(&mut out);
    out
}

/// XGBoost's `tree::FitStump`: the unregularized Newton step `-Σg_k /
/// max(Σh_k, 1e-6)` per output `k` of a `[row][output]` gradient buffer with
/// `k` outputs, summed in `f64` and rounded once to `f32`.
pub(crate) fn fit_stump(gpair: &[GradPair], k: usize) -> Vec<f32> {
    let mut sum_grad = vec![0.0f64; k];
    let mut sum_hess = vec![0.0f64; k];
    for row in gpair.chunks_exact(k) {
        for (c, gp) in row.iter().enumerate() {
            sum_grad[c] += f64::from(gp.grad);
            sum_hess[c] += f64::from(gp.hess);
        }
    }
    sum_grad
        .iter()
        .zip(&sum_hess)
        .map(|(g, h)| (-g / h.max(crate::K_RT_EPS)) as f32)
        .collect()
}

/// The log link's [`Objective::probs_to_margins`] (XGBoost `ProbToMargin`
/// of the log-link objectives): `ln(v)` of every entry, in `f32`.
pub(crate) fn log_link(scores: &mut [f32]) {
    for s in scores {
        *s = s.ln();
    }
}

/// Shared [`Objective::validate_info`] label-domain check: reject the dataset
/// when any label satisfies `invalid`.
pub(crate) fn check_label_domain(info: &MetaInfo, invalid: impl Fn(f32) -> bool) -> Result<()> {
    if info.labels.iter().any(|&y| invalid(y)) {
        return Err(HessboostError::invalid_param(
            "labels",
            "dataset has labels outside the objective's valid domain",
        ));
    }
    Ok(())
}

/// Shared [`Objective::validate_info`] label-width check: reject the dataset
/// unless it carries the `n_targets` label columns the objective's outputs
/// are paired with (objectives passed to training directly are not sized
/// from the dataset, unlike [`create_objective`]'s).
pub(crate) fn check_label_width(info: &MetaInfo, n_targets: usize) -> Result<()> {
    if info.n_targets != n_targets {
        return Err(HessboostError::invalid_param(
            "labels",
            format!(
                "dataset has {} label columns but the objective models {n_targets}",
                info.n_targets
            ),
        ));
    }
    Ok(())
}

/// Objectives XGBoost 3.4.2 trains on a label matrix: elementwise losses whose
/// output `j` fits label column `j` (`Targets(info) = labels.Shape(1)`).
const MULTI_TARGET_OBJECTIVES: &[&str] = &[
    "reg:squarederror",
    "reg:pseudohubererror",
    "reg:logistic",
    "binary:logistic",
];

/// Fit `n_targets` label columns with `objective`: one per output through
/// [`multi_target::MultiTarget`] for the objectives in
/// [`MULTI_TARGET_OBJECTIVES`], unchanged for one column, and an
/// `invalid parameter "labels"` error for any other objective.
fn with_targets(objective: Box<dyn Objective>, n_targets: usize) -> Result<Box<dyn Objective>> {
    if n_targets <= 1 {
        return Ok(objective);
    }
    if MULTI_TARGET_OBJECTIVES.contains(&objective.name()) {
        return Ok(Box::new(multi_target::MultiTarget::new(
            objective, n_targets,
        )));
    }
    Err(HessboostError::invalid_param(
        "labels",
        format!(
            "objective `{}` supports one target per row, got {n_targets}",
            objective.name()
        ),
    ))
}

/// Weighted mean of `labels`, or the plain mean when `weights` is `None`, as
/// the `f32` intercept XGBoost's `FitInterceptGlmLike` stores. Accumulates
/// `Σ yᵢ/n` (or `Σ yᵢ/Σw · wᵢ`) in `f64` exactly like `common::SampleMean` /
/// `WeightedSampleMean`, then rounds once to `f32`. Empty or zero-weight input
/// yields `0.0`.
pub(crate) fn weighted_label_mean(labels: &[f32], weights: Option<&[f32]>) -> f32 {
    let mean = match weights {
        Some(w) => {
            let sum_w: f64 = w.iter().map(|&wi| f64::from(wi)).sum();
            if sum_w > 0.0 {
                labels
                    .iter()
                    .zip(w)
                    .map(|(&y, &wi)| f64::from(y) / sum_w * f64::from(wi))
                    .sum()
            } else {
                0.0
            }
        }
        None if labels.is_empty() => 0.0,
        None => {
            let n = labels.len() as f64;
            labels.iter().map(|&y| f64::from(y) / n).sum()
        }
    };
    mean as f32
}

/// Resolve an objective by name, configured from `params`, for a dataset with
/// `n_targets` label columns per row.
///
/// `reg:squarederror`, `reg:pseudohubererror`, `reg:logistic`,
/// `binary:logistic`, and `reg:absoluteerror` accept a label matrix and give
/// one output per label column, as in XGBoost. `reg:quantileerror` /
/// `reg:expectileerror` produce one output per `quantile_alpha` /
/// `expectile_alpha` entry and reject an empty, unsorted, or out-of-`[0, 1]`
/// list. The distributional `dist:*` objectives (beyond XGBoost, see
/// [`distributional`]) give one output per distribution parameter. Every
/// other objective models a single target and rejects `n_targets > 1` with
/// an `invalid parameter "labels"` error.
pub fn create_objective(params: &TrainingParams, n_targets: usize) -> Result<Box<dyn Objective>> {
    let objective: Box<dyn Objective> = match params.objective.as_str() {
        "reg:squarederror" | "reg:linear" => Box::new(SquaredError),
        "reg:pseudohubererror" => Box::new(PseudoHuber::new(params.huber_slope as f32)),
        "binary:logistic" => Box::new(Logistic::new(params.scale_pos_weight as f32)),
        "binary:logitraw" => Box::new(Logistic::raw(params.scale_pos_weight as f32)),
        "binary:hinge" => Box::new(Hinge),
        "reg:squaredlogerror" => Box::new(SquaredLogError),
        "reg:logistic" => Box::new(Logistic::regression(params.scale_pos_weight as f32)),
        "multi:softmax" | "multi:softprob" => {
            if params.num_class < 2 {
                return Err(HessboostError::invalid_param(
                    "num_class",
                    "multiclass objectives require num_class >= 2",
                ));
            }
            let prob = params.objective == "multi:softprob";
            Box::new(Softmax::new(params.num_class, prob))
        }
        "count:poisson" => Box::new(Poisson::new(params.effective_max_delta_step() as f32)),
        "reg:gamma" => Box::new(Gamma),
        "reg:tweedie" => Box::new(Tweedie::new(params.tweedie_variance_power as f32)),
        "reg:quantileerror" => Box::new(Quantile::new(&params.quantile_alpha)?),
        "reg:expectileerror" => Box::new(Expectile::new(&params.expectile_alpha)?),
        "reg:absoluteerror" => return Ok(Box::new(AbsoluteError::new(n_targets))),
        "rank:pairwise" => Box::new(LambdaMart::pairwise(params.lambdarank_num_pair_per_sample)),
        "rank:ndcg" => Box::new(LambdaMart::ndcg(params.lambdarank_num_pair_per_sample)),
        "rank:map" => Box::new(LambdaMart::map(params.lambdarank_num_pair_per_sample)),
        "survival:cox" => Box::new(Cox),
        "survival:aft" => Box::new(Aft::new(
            params.aft_loss_distribution,
            params.aft_loss_distribution_scale as f32,
        )),
        other => match DistFamily::from_objective(other) {
            Some(family) => {
                let objective = DistObjective::new(family, params.dist_gradient);
                Box::new(
                    if params.multi_strategy == crate::config::MultiStrategy::MultiOutputTree {
                        objective.with_split_direction(params.dist_split_direction, params.seed)
                    } else {
                        objective
                    },
                )
            }
            None => return Err(HessboostError::unknown("objective", other)),
        },
    };
    with_targets(objective, n_targets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_mean_basic() {
        let labels = [1.0f32, 3.0];
        assert_eq!(weighted_label_mean(&labels, None), 2.0);
        let w = [3.0f32, 1.0];
        // (1*3 + 3*1) / 4 = 1.5
        assert_eq!(weighted_label_mean(&labels, Some(&w)), 1.5);
    }

    /// The default intercept is one Newton step from zero margins, mapped
    /// through the link and back. For the logistic loss with
    /// `scale_pos_weight != 1` that is `-Σg/Σh` with `g = (½ − y)·w`,
    /// `h = ¼·w`, then sigmoid then logit (XGBoost's Newton fallback).
    #[test]
    fn default_base_margins_is_newton_step_through_link() {
        let obj = Logistic::new(2.0);
        let labels = [1.0f32, 0.0, 0.0, 0.0];
        let margins = base_margins(&obj, &labels, None);
        assert_eq!(margins.len(), 1);
        // g = -0.5*2 + 3*0.5 = 0.5, h = 0.25*(2 + 3) = 1.25 -> w = -0.4.
        let mut through_link = [-0.4f32];
        obj.pred_transform(&mut through_link);
        obj.probs_to_margins(&mut through_link);
        assert_eq!(margins[0], through_link[0]);
        assert!((margins[0] + 0.4).abs() < 1e-6, "got {}", margins[0]);
    }

    /// An objective that learns from label bounds through `gradient_info`
    /// only: the default `base_margins_info` must take its Newton step on
    /// the dataset's own metadata (it once rebuilt label-only metadata, so
    /// the gradient saw no bounds), and give NaN, not a panic, for a
    /// `n_targets` the lengths do not match.
    #[test]
    fn default_intercept_reads_the_original_metadata() {
        struct Midpoint;
        impl Objective for Midpoint {
            fn name(&self) -> &'static str {
                "test:midpoint"
            }
            fn gradient(&self, _: &[f32], _: &[f32], _: Option<&[f32]>, out: &mut [GradPair]) {
                out.fill(GradPair::new(f32::NAN, 1.0));
            }
            fn gradient_info(&self, preds: &[f32], info: &MetaInfo, out: &mut [GradPair]) {
                let (Some(lo), Some(hi)) = (info.label_lower_bound, info.label_upper_bound) else {
                    return self.gradient(preds, info.labels, info.weights, out);
                };
                for (i, g) in out.iter_mut().enumerate() {
                    *g = GradPair::new(preds[i] - f32::midpoint(lo[i], hi[i]), 1.0);
                }
            }
            fn default_metric(&self) -> String {
                "rmse".to_string()
            }
        }
        let (lower, upper) = ([0.0f32, 4.0], [2.0f32, 6.0]);
        let info = MetaInfo {
            n_rows: 2,
            label_lower_bound: Some(&lower),
            label_upper_bound: Some(&upper),
            ..MetaInfo::new(&[], None, None)
        };
        assert_eq!(Midpoint.base_margins_info(&info), vec![3.0]);
        let inconsistent = MetaInfo {
            n_targets: usize::MAX,
            ..info
        };
        assert!(Midpoint.base_margins_info(&inconsistent)[0].is_nan());
    }

    #[test]
    fn factory_resolves_known_and_rejects_unknown() {
        let p = TrainingParams::builder()
            .objective("reg:squarederror")
            .build_unchecked();
        assert_eq!(create_objective(&p, 1).unwrap().name(), "reg:squarederror");
        let p = TrainingParams::builder()
            .objective("nope:whatever")
            .build_unchecked();
        assert!(create_objective(&p, 1).is_err());
    }

    /// Only XGBoost's elementwise multi-target objectives (and
    /// `reg:absoluteerror`, which models label matrices itself) accept a label
    /// matrix (one output per column); every other built-in objective
    /// rejects two label columns with a parameter error naming `labels`,
    /// never a silently wrong model.
    #[test]
    fn factory_accepts_label_matrices_only_for_elementwise_objectives() {
        for name in [
            "reg:squarederror",
            "reg:linear",
            "reg:pseudohubererror",
            "binary:logistic",
            "reg:logistic",
            "reg:absoluteerror",
        ] {
            let p = TrainingParams::builder().objective(name).build_unchecked();
            assert_eq!(create_objective(&p, 3).unwrap().n_outputs(), 3, "{name}");
        }
        for name in [
            "multi:softprob",
            "count:poisson",
            "reg:gamma",
            "reg:tweedie",
            "reg:quantileerror",
            "reg:expectileerror",
            "rank:ndcg",
        ] {
            let p = TrainingParams::builder()
                .objective(name)
                .num_class(3)
                .quantile_alpha(vec![0.5])
                .expectile_alpha(vec![0.5])
                .build_unchecked();
            assert!(create_objective(&p, 1).is_ok(), "{name}");
            match create_objective(&p, 2) {
                Err(HessboostError::InvalidParameter { name: param, .. }) => {
                    assert_eq!(param, "labels", "{name}");
                }
                Err(other) => panic!("{name}: unexpected error {other}"),
                Ok(_) => panic!("{name}: accepted two targets"),
            }
        }
    }

    /// The alpha-list objectives need their list: one output per alpha, and a
    /// missing or invalid list is an error naming the parameter.
    #[test]
    fn factory_sizes_alpha_objectives_and_requires_alphas() {
        for (name, param) in [
            ("reg:quantileerror", "quantile_alpha"),
            ("reg:expectileerror", "expectile_alpha"),
        ] {
            let p = TrainingParams::builder().objective(name).build_unchecked();
            match create_objective(&p, 1) {
                Err(HessboostError::InvalidParameter { name: got, .. }) => assert_eq!(got, param),
                Err(other) => panic!("{name}: unexpected error {other}"),
                Ok(_) => panic!("{name}: accepted an empty alpha list"),
            }
            let p = TrainingParams::builder()
                .objective(name)
                .quantile_alpha(vec![0.1, 0.5, 0.9])
                .expectile_alpha(vec![0.1, 0.5, 0.9])
                .build_unchecked();
            assert_eq!(create_objective(&p, 1).unwrap().n_outputs(), 3, "{name}");
        }
    }

    /// Parallel row chunks must reproduce the whole-batch gradient bit for bit
    /// for every objective routed through the chunked helper. Lengths are not
    /// multiples of the chunk or of any vector block. The logistic case sweeps
    /// every short-tail residue r in 1..=15, where a separate final chunk
    /// would fall below the vector dispatch length and compute its rows on
    /// the scalar path. The sweep also pins the structural invariant the fold
    /// relies on: chunk boundaries stay multiples of every kernel block, so
    /// a uniform `chunk + tail` chunking would fail here.
    #[test]
    fn chunked_gradients_match_whole_batch() {
        let c = GRADIENT_CHUNK_ROWS;
        let objectives: Vec<(Box<dyn Objective>, usize, Vec<usize>)> = vec![
            (Box::new(SquaredError), 1, vec![2 * c + 4097]),
            (
                Box::new(Logistic::new(1.5)),
                1,
                (1..=15).map(|r| 2 * c + r).collect(),
            ),
            (Box::new(Softmax::new(2, true)), 2, vec![2 * c + 4]),
            (Box::new(Softmax::new(3, true)), 3, vec![2 * c + 4]),
            (Box::new(Softmax::new(9, false)), 9, vec![2 * c + 1]),
            // Per-output residual scales are global reductions: chunking the
            // row kernel must not change them.
            (
                Box::new(Quantile::new(&[0.1, 0.5, 0.9]).unwrap()),
                3,
                vec![2 * c + 3],
            ),
            (
                Box::new(Expectile::new(&[0.2, 0.8]).unwrap()),
                2,
                vec![2 * c + 3],
            ),
            (Box::new(AbsoluteError::new(1)), 1, vec![2 * c + 5]),
        ];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for (objective, k, ns) in objectives {
            for n in ns {
                let preds: Vec<f32> = (0..n * k)
                    .map(|i| ((i * 7919) % 2003) as f32 / 97.0 - 10.0)
                    .collect();
                let labels: Vec<f32> = (0..n).map(|i| (i % k.max(2)) as f32).collect();
                let weights: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.25).collect();
                for weights in [None, Some(weights.as_slice())] {
                    let mut whole = vec![GradPair::default(); n * k];
                    // A single-thread pool takes the whole-batch path.
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(1)
                        .build()
                        .unwrap()
                        .install(|| objective.gradient(&preds, &labels, weights, &mut whole));
                    let mut chunked = vec![GradPair::default(); n * k];
                    pool.install(|| objective.gradient(&preds, &labels, weights, &mut chunked));
                    for (i, (a, b)) in whole.iter().zip(&chunked).enumerate() {
                        assert_eq!(
                            a.grad.to_bits(),
                            b.grad.to_bits(),
                            "{} grad {i}",
                            objective.name()
                        );
                        assert_eq!(
                            a.hess.to_bits(),
                            b.hess.to_bits(),
                            "{} hess {i}",
                            objective.name()
                        );
                    }
                }
            }
        }
    }
}
