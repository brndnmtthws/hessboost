//! Learning objectives: gradients, Hessians, prediction transforms, and base
//! score estimation.
//!
//! Every objective trains through a [`Loss`]. Boosting works in *margin*
//! space (raw additive scores). The [`Loss::pred_transform`] maps margins to
//! the reported prediction (e.g. the logistic sigmoid). This mirrors
//! XGBoost's separation of `GetGradient` / `PredTransform`.
//!
//! A configuration names its objective with [`Objective`]: a built-in
//! XGBoost objective with its parameters ([`RegLoss`], [`Multiclass`],
//! [`PseudoHuber`], [`Quantiles`], [`Expectiles`], [`Tweedie`],
//! [`LambdaRank`], [`Aft`], [`distributional::Distributional`]; each
//! validated when constructed, several shared with the metrics that read
//! them), or [`Objective::Custom`] with any [`Loss`] such as
//! [`CustomLoss`].
//! [`TrainingParams::loss`](crate::config::TrainingParams::loss) builds the
//! loss a configuration trains with.

/// Stable names of the enums a model file stores, matching their serde
/// (and XGBoost) spellings. Defined before the submodules so they can use it.
macro_rules! stored_names {
    ($($ty:ident { $($variant:ident => $name:literal),+ $(,)? })+) => {$(
        impl $ty {
            /// The variant's stored name.
            pub(crate) fn name(self) -> &'static str {
                match self {
                    $($ty::$variant => $name,)+
                }
            }

            /// The variant stored as `name`, if any.
            pub(crate) fn from_name(name: &str) -> Option<Self> {
                match name {
                    $($name => Some($ty::$variant),)+
                    _ => None,
                }
            }
        }
    )+};
}

mod absolute;
mod classification;
mod count;
mod custom;
pub mod distributional;
mod multi_target;
mod multiclass;
mod params;
mod quantile;
mod query;
mod ranking;
mod regression;
mod spec;
mod survival;
mod xendcg;

pub(crate) use absolute::AbsoluteError;
pub(crate) use classification::{Hinge, LogisticLoss};
pub(crate) use count::{Gamma, Poisson, TweedieLoss};
pub use custom::CustomLoss;
pub(crate) use multiclass::Softmax;
pub use params::{
    Aft, AftDistribution, Expectiles, LambdaRank, Multiclass, PseudoHuber, Quantiles, RegLoss,
    Tweedie,
};
pub(crate) use quantile::{Expectile, Quantile};
pub(crate) use ranking::LambdaMart;
pub(crate) use regression::{PseudoHuberLoss, SquaredError, SquaredLogError};
pub use spec::Objective;
pub(crate) use spec::{LossContext, OBJECTIVE_PARAMS, ObjectiveParts};
pub(crate) use xendcg::Xendcg;

pub(crate) use survival::{AftLoss, Cox};

pub(crate) use survival::{abs_label_order, aft_nloglik};

use rayon::prelude::*;

use crate::data::MetaInfo;
use crate::error::{HessboostError, Result};

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
/// `max_delta_step`). Returned by [`Loss::pointwise_loss`].
pub type PointwiseLoss<'a> = Box<dyn Fn(f32, f32) -> f64 + Send + Sync + 'a>;

/// Reduced gradients a custom objective supplies for the *split search* of
/// vector-leaf trees (see [`Loss::split_gradient`]). Build with
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
/// `n_outputs` values each and one label per row, in parallel row chunks when
/// the batch is large and a thread pool is available
/// ([`rowwise_cells`] with one label column).
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
    let shape = RowShape {
        n_rows,
        n_outputs,
        label_cols: 1,
    };
    rowwise_cells(shape, preds, labels, weights, out, kernel);
}

/// The per-row layout of a [`rowwise_cells`] batch: `preds` and `out` hold
/// `n_outputs` values per row, `labels` hold `label_cols`, and weights one.
#[derive(Clone, Copy)]
pub(crate) struct RowShape {
    pub(crate) n_rows: usize,
    pub(crate) n_outputs: usize,
    pub(crate) label_cols: usize,
}

/// Run a row-independent gradient `kernel` over `shape.n_rows` rows, in
/// parallel row chunks when the batch is large and a thread pool is
/// available. Every row's outputs depend only on that row, and the chunking
/// is fixed (not thread-count dependent): a short final chunk is folded into
/// the last full chunk so every row takes the same vector/scalar path as in
/// one whole-batch call, and the result is identical. Mis-shaped inputs run
/// as one whole-batch call.
pub(crate) fn rowwise_cells<K>(
    shape: RowShape,
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    out: &mut [GradPair],
    kernel: K,
) where
    K: Fn(&[f32], &[f32], Option<&[f32]>, &mut [GradPair]) + Sync,
{
    let RowShape {
        n_rows,
        n_outputs,
        label_cols,
    } = shape;
    let complete = n_rows
        .checked_mul(n_outputs)
        .is_some_and(|values| preds.len() == values && out.len() == values)
        && n_rows
            .checked_mul(label_cols)
            .is_some_and(|values| labels.len() == values)
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
            &labels[first * label_cols..(first + rows) * label_cols],
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
    objective: &dyn Loss,
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
fn base_margins(objective: &dyn Loss, labels: &[f32], weights: Option<&[f32]>) -> Vec<f32> {
    objective.base_margins_info(&MetaInfo::new(labels, weights, None))
}

/// Debug-only shape check shared by every [`Loss::gradient`]: `preds` and
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

/// A differentiable training loss: gradients, Hessians, the prediction
/// transform, and the intercept estimate of one learning objective. Every
/// built-in objective is one, and [`CustomLoss`] (or any other
/// implementation) trains as [`Objective::Custom`] (built with
/// [`Objective::custom`]).
///
/// Implementors are `Send + Sync` so gradient computation can be parallelized.
pub trait Loss: Send + Sync {
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
    /// [`Loss::gradient`], ignoring the grouping. This is correct for all
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
    /// the labels, weights, and groups to [`Loss::gradient_grouped`];
    /// objectives that read other metadata (label bounds, several targets per
    /// row) override it.
    fn gradient_info(&self, preds: &[f32], info: &MetaInfo, out: &mut [GradPair]) {
        self.gradient_grouped(preds, info.label_values(), info.weights, info.group, out);
    }

    /// Compute the gradients of boosting round `iteration` (counted from
    /// the model's first round, so continued training carries on) from a
    /// dataset's full metadata view: what training calls. Losses whose
    /// gradients are randomized per round (XE-NDCG's targets) override it;
    /// the default forwards to [`Loss::gradient_info`].
    fn gradient_info_at(
        &self,
        preds: &[f32],
        info: &MetaInfo,
        out: &mut [GradPair],
        _iteration: usize,
    ) {
        self.gradient_info(preds, info, out);
    }

    /// Whether the Hessian is constant across margins (XGBoost
    /// `ObjInfo::const_hess`); only `reg:squarederror` returns `true`.
    fn const_hess(&self) -> bool {
        false
    }

    /// Transform raw margins into reported predictions, in place. Default is the
    /// identity (used by squared-error regression). An objective whose
    /// transform is an inverse link also overrides
    /// [`Loss::probs_to_margins`] with the link, which the intercept
    /// estimate and a user-supplied `base_score` go through.
    fn pred_transform(&self, _preds: &mut [f32]) {}

    /// Estimate the per-output intercepts in *margin* space from a dataset's
    /// full metadata view; the single hook training uses to initialize the
    /// model's `base_score` when the user does not supply one. Returns
    /// exactly [`Loss::n_outputs`] values. To estimate from bare
    /// labels, weights, and groups, pass [`MetaInfo::new`].
    ///
    /// The default is XGBoost's `FitIntercept::InitEstimation`: one Newton
    /// step from all-zero margins over [`Loss::gradient_info`] on `info`
    /// itself (so label bounds and label matrices reach the gradient),
    /// `w_k = -Σg_k / max(Σh_k, 1e-6)` (sums in `f64`, step rounded to
    /// `f32`), mapped through [`Loss::pred_transform`] and back through
    /// [`Loss::probs_to_margins`] to reproduce XGBoost's `f32` rounding;
    /// `NaN` intercepts (which training refuses) when `info`'s lengths are
    /// inconsistent. Objectives whose optimal constant has a closed form
    /// (label mean, class frequencies) override it.
    fn base_margins_info(&self, info: &MetaInfo) -> Vec<f32> {
        newton_intercepts(self, info)
    }

    /// Transform raw margins into the values evaluation metrics receive
    /// (XGBoost `EvalTransform`), in place. Defaults to
    /// [`Loss::pred_transform`].
    fn eval_transform(&self, preds: &mut [f32]) {
        self.pred_transform(preds);
    }

    /// Map one row of prediction-space intercepts (length
    /// [`Loss::n_outputs`]) to margin space, in place: XGBoost
    /// `ProbToMargin` on the base-score vector, in `f32`. Training applies
    /// it to a user-supplied `base_score`, XGBoost-model import to the
    /// stored one, and the default [`Loss::base_margins_info`] to its
    /// Newton step. Default is the identity; objectives with a link
    /// function (e.g. the logit of `binary:logistic`) override it.
    fn probs_to_margins(&self, _scores: &mut [f32]) {}

    /// Refuse a user-supplied `base_score` (prediction space) this loss's
    /// [`Loss::probs_to_margins`] cannot map to a margin, e.g. a probability
    /// outside `(0, 1)` for `binary:logistic`. Training calls it on the loss
    /// it trains, before the link is applied. The default accepts every
    /// value (training refuses non-finite intercepts after the link).
    fn validate_base_score(&self, _base_score: f64) -> Result<()> {
        Ok(())
    }

    /// Map one row of margin-space intercepts back to the prediction space
    /// XGBoost stores `base_score` in (the inverse of
    /// [`Loss::probs_to_margins`]), in place; used by XGBoost-JSON
    /// export. Defaults to [`Loss::pred_transform`], which inverts the
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
    /// `gpair` (`[row][output]`, [`Loss::n_outputs`] pairs per row, row
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

    /// The metric training evaluates when no `eval_metric` is configured:
    /// XGBoost's `DefaultEvalMetric`, with the parameters it gives it (e.g.
    /// `ndcg@32` from LambdaRank's top-k, `tweedie-nloglik@1.5` from the
    /// variance power, the loss's own alphas or slope).
    fn default_metric(&self) -> crate::metric::EvalMetric;
}

/// One Newton step from all-zero margins, per output: `w_k = -Σg_k /
/// max(Σh_k, 1e-6)` with the sums in `f64` and the step rounded to `f32`, then
/// mapped through [`Loss::pred_transform`] and back through
/// [`Loss::probs_to_margins`]. XGBoost (`FitIntercept::InitEstimation` +
/// `tree::FitStump`) stores the intercept in prediction space and re-applies
/// the link on use; taking the same round trip reproduces its `f32` rounding.
/// `NaN` for every output when `info` fails [`MetaInfo::check_layout`] or
/// the margin buffer would overflow, before anything is allocated.
pub(crate) fn newton_intercepts<O: Loss + ?Sized>(objective: &O, info: &MetaInfo) -> Vec<f32> {
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

/// The log link's [`Loss::probs_to_margins`] (XGBoost `ProbToMargin`
/// of the log-link objectives): `ln(v)` of every entry, in `f32`.
pub(crate) fn log_link(scores: &mut [f32]) {
    for s in scores {
        *s = s.ln();
    }
}

/// Shared [`Loss::validate_base_score`] check: refuse a `base_score`
/// outside the output domain of the objective, `(0, 1)` for the logistic
/// link and `(0, ∞)` for the log link.
pub(crate) fn check_base_score_domain(base_score: f64, domain: OutputDomain) -> Result<()> {
    let inside = match domain {
        OutputDomain::Probability => 0.0 < base_score && base_score < 1.0,
        OutputDomain::Positive => base_score > 0.0,
    };
    if inside {
        Ok(())
    } else {
        Err(HessboostError::invalid_param(
            "base_score",
            "is outside the objective's valid output domain",
        ))
    }
}

/// The prediction space a link maps from ([`check_base_score_domain`]).
#[derive(Debug, Clone, Copy)]
pub(crate) enum OutputDomain {
    /// `(0, 1)`: the logistic link.
    Probability,
    /// `(0, ∞)`: the log link.
    Positive,
}

/// Shared [`Loss::validate_info`] label-domain check: reject the dataset
/// when any label satisfies `invalid`.
pub(crate) fn check_label_domain(info: &MetaInfo, invalid: impl Fn(f32) -> bool) -> Result<()> {
    if info.label_values().iter().any(|&y| invalid(y)) {
        return Err(HessboostError::invalid_param(
            "labels",
            "dataset has labels outside the objective's valid domain",
        ));
    }
    Ok(())
}

/// Shared [`Loss::validate_info`] label-width check: reject the dataset
/// unless it carries the `n_targets` label columns the objective's outputs
/// are paired with (custom losses are not sized from the dataset, unlike
/// the built-in objectives'
/// [`TrainingParams::loss`](crate::config::TrainingParams::loss)).
pub(crate) fn check_label_width(info: &MetaInfo, n_targets: usize) -> Result<()> {
    if info.n_targets() != n_targets {
        return Err(HessboostError::invalid_param(
            "labels",
            format!(
                "dataset has {} label columns but the objective models {n_targets}",
                info.n_targets()
            ),
        ));
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainingParams;
    use serde_json::json;

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
        let obj = LogisticLoss::new(2.0);
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
        impl Loss for Midpoint {
            fn name(&self) -> &'static str {
                "test:midpoint"
            }
            fn gradient(&self, _: &[f32], _: &[f32], _: Option<&[f32]>, out: &mut [GradPair]) {
                out.fill(GradPair::new(f32::NAN, 1.0));
            }
            fn gradient_info(&self, preds: &[f32], info: &MetaInfo, out: &mut [GradPair]) {
                let Some((lo, hi)) = info.bounds.map(|b| (b.lower(), b.upper())) else {
                    return self.gradient(preds, info.label_values(), info.weights, out);
                };
                for (i, g) in out.iter_mut().enumerate() {
                    *g = GradPair::new(preds[i] - f32::midpoint(lo[i], hi[i]), 1.0);
                }
            }
            fn default_metric(&self) -> crate::metric::EvalMetric {
                crate::metric::EvalMetric::Rmse
            }
        }
        let (lower, upper) = ([0.0f32, 4.0], [2.0f32, 6.0]);
        let info = MetaInfo {
            n_rows: 2,
            bounds: Some(crate::data::LabelBounds::new(&lower, &upper)),
            weights: None,
            ..MetaInfo::unlabeled(0)
        };
        assert_eq!(Midpoint.base_margins_info(&info), vec![3.0]);
        let inconsistent = MetaInfo {
            labels: Some(crate::data::Labels::new(&[], std::num::NonZeroUsize::MAX)),
            ..info
        };
        assert!(Midpoint.base_margins_info(&inconsistent)[0].is_nan());
    }

    #[test]
    fn factory_resolves_known_and_rejects_unknown() {
        let p = TrainingParams::from_xgboost([("objective", json!("reg:squarederror"))]).unwrap();
        assert_eq!(p.loss(1).unwrap().name(), "reg:squarederror");
        assert!(TrainingParams::from_xgboost([("objective", json!("nope:whatever"))]).is_err());
    }

    /// The params with objective `objective`.
    fn with_objective(objective: Objective) -> TrainingParams {
        TrainingParams {
            objective,
            ..TrainingParams::default()
        }
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
            let p = TrainingParams::from_xgboost([("objective", json!(name))]).unwrap();
            assert_eq!(p.loss(3).unwrap().n_outputs(), 3, "{name}");
        }
        for objective in [
            Objective::Softprob(Multiclass::new(3).unwrap()),
            Objective::Poisson,
            Objective::Gamma(RegLoss::default()),
            Objective::Tweedie(Tweedie::default()),
            Objective::Quantile(Quantiles::new([0.5]).unwrap()),
            Objective::Expectile(Expectiles::new([0.5]).unwrap()),
            Objective::RankNdcg(LambdaRank::default()),
        ] {
            let name = objective.name().to_owned();
            let p = with_objective(objective);
            assert!(p.loss(1).is_ok(), "{name}");
            match p.loss(2) {
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
            match TrainingParams::from_xgboost([("objective", json!(name))]) {
                Err(HessboostError::InvalidParameter { name: got, .. }) => assert_eq!(got, param),
                Err(other) => panic!("{name}: unexpected error {other}"),
                Ok(_) => panic!("{name}: accepted an empty alpha list"),
            }
            let p = TrainingParams::from_xgboost([
                ("objective", json!(name)),
                (param, json!([0.1, 0.5, 0.9])),
            ])
            .unwrap();
            assert_eq!(p.loss(1).unwrap().n_outputs(), 3, "{name}");
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
        let objectives: Vec<(Box<dyn Loss>, usize, Vec<usize>)> = vec![
            (Box::new(SquaredError::new(1.5)), 1, vec![2 * c + 4097]),
            (
                Box::new(LogisticLoss::new(1.5)),
                1,
                (1..=15).map(|r| 2 * c + r).collect(),
            ),
            (
                Box::new(Poisson::new(0.7)),
                1,
                (1..=15).map(|r| 2 * c + r).collect(),
            ),
            (
                Box::new(Gamma::new(1.5)),
                1,
                (1..=15).map(|r| 2 * c + r).collect(),
            ),
            (
                Box::new(TweedieLoss::new(Tweedie::new(1.3).unwrap())),
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
