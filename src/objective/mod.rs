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
mod multi_target;
mod multiclass;
mod quantile;
mod ranking;
mod regression;

pub use absolute::AbsoluteErrorObjective;
pub use classification::{HingeObjective, LogisticObjective};
pub use count::{GammaObjective, PoissonObjective, TweedieObjective};
pub use custom::CustomObjective;
pub use multiclass::SoftmaxObjective;
pub use quantile::{ExpectileObjective, QuantileObjective};
pub use ranking::LambdaMartObjective;
pub use regression::{PseudoHuberObjective, SquaredErrorObjective, SquaredLogErrorObjective};

pub(crate) use quantile::validate_alphas;

use rayon::prelude::*;

use crate::config::TrainingParams;
use crate::data::MetaInfo;
use crate::error::{HessboostError, Result};

/// A first- and second-order gradient for one instance/output.
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

/// Rows per parallel gradient chunk. A multiple of every vector kernel's block
/// (4 rows, and 4 values for any class count), so chunk boundaries fall where
/// the kernels' block boundaries already are and every element is computed
/// by the same path as in one whole-batch call.
const GRADIENT_CHUNK_ROWS: usize = 8192;

/// Lower bound on any per-instance Hessian, matching XGBoost's guard, so that
/// confidently-classified instances still contribute a positive Hessian.
pub(crate) const MIN_HESS: f32 = 1e-16;

/// Run a row-independent gradient `kernel` over `n_rows` instances with
/// `n_outputs` values each, in parallel row chunks when the batch is large and
/// a thread pool is available. Every row's outputs depend only on that row,
/// and the chunking is fixed (not thread-count dependent): a short final
/// chunk is folded into the last full chunk so every row takes the same
/// vector/scalar path as in one whole-batch call, and the result is
/// identical.
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

/// Debug-only shape check shared by every [`Objective::gradient`]: `preds` and
/// `out` hold `n_rows * n_outputs` values while `labels` (and `weights`, when
/// present) hold one per row. Release builds skip it, like the
/// `debug_assert_eq!`s it replaces; [`rowwise_gradient`] re-validates the same
/// shapes at runtime for its chunking decision.
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
    /// identity (used by squared-error regression).
    fn pred_transform(&self, _preds: &mut [f32]) {}

    /// Estimate the per-output intercepts in *margin* space from the training
    /// labels; used to initialize the model's `base_score` when the user does
    /// not supply one. Returns exactly [`Objective::n_outputs`] values.
    ///
    /// The default is XGBoost's `FitIntercept::InitEstimation`: one Newton
    /// step from all-zero margins, `w_k = -Σg_k / max(Σh_k, 1e-6)` (sums in
    /// `f64`, step rounded to `f32`), mapped through
    /// [`Objective::pred_transform`] and back through
    /// [`Objective::prob_to_margin`] to reproduce XGBoost's `f32` rounding.
    /// Objectives whose optimal constant has a closed form (label mean, class
    /// frequencies) override it.
    fn base_margins(
        &self,
        labels: &[f32],
        weights: Option<&[f32]>,
        group: Option<&crate::data::GroupInfo>,
    ) -> Vec<f32> {
        newton_intercepts(self, &MetaInfo::new(labels, weights, group))
    }

    /// Estimate the per-output intercepts from a dataset's full metadata
    /// view; the entry point training uses. The default forwards to
    /// [`Objective::base_margins`].
    fn base_margins_info(&self, info: &MetaInfo) -> Vec<f32> {
        self.base_margins(info.labels, info.weights, info.group)
    }

    /// Transform raw margins into the values evaluation metrics receive
    /// (XGBoost `EvalTransform`), in place. Defaults to
    /// [`Objective::pred_transform`].
    fn eval_transform(&self, preds: &mut [f32]) {
        self.pred_transform(preds);
    }

    /// Convert a `base_score` given in prediction space into margin space via
    /// the objective's inverse link, in `f32` like XGBoost's `ProbToMargin`.
    /// Default is the identity; objectives with a link function (e.g.
    /// logistic) override it.
    fn prob_to_margin(&self, base_score: f32) -> f32 {
        base_score
    }

    /// Map one row of prediction-space intercepts (length
    /// [`Objective::n_outputs`]) to margin space, in place (XGBoost
    /// `ProbToMargin` on the base-score vector). Defaults to
    /// [`Objective::prob_to_margin`] per entry.
    fn probs_to_margins(&self, scores: &mut [f32]) {
        for s in scores {
            *s = self.prob_to_margin(*s);
        }
    }

    /// Map one row of margin-space intercepts back to the prediction space
    /// XGBoost stores `base_score` in (the inverse of
    /// [`Objective::probs_to_margins`]), in place; used by XGBoost-JSON
    /// export. Defaults to [`Objective::pred_transform`], which inverts the
    /// link of every objective whose transform is its link; objectives whose
    /// transform is not the inverse link (`binary:hinge` thresholds, while its
    /// `ProbToMargin` is the identity) override it.
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
pub(crate) fn newton_intercepts<O: Objective + ?Sized>(objective: &O, info: &MetaInfo) -> Vec<f32> {
    let k = objective.n_outputs();
    let n = info.n_rows;
    let zeros = vec![0.0f32; n * k];
    let mut gpair = vec![GradPair::default(); n * k];
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
        .map(|(g, h)| (-g / h.max(1e-6)) as f32)
        .collect()
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
/// list. Every other objective models a single target and rejects
/// `n_targets > 1` with an `invalid parameter "labels"` error.
pub fn create_objective(params: &TrainingParams, n_targets: usize) -> Result<Box<dyn Objective>> {
    let objective: Box<dyn Objective> = match params.objective.as_str() {
        "reg:squarederror" | "reg:linear" => Box::new(SquaredErrorObjective),
        "reg:pseudohubererror" => Box::new(PseudoHuberObjective::new(params.huber_slope as f32)),
        "binary:logistic" => Box::new(LogisticObjective::new(params.scale_pos_weight as f32)),
        "binary:logitraw" => Box::new(LogisticObjective::raw(params.scale_pos_weight as f32)),
        "binary:hinge" => Box::new(HingeObjective),
        "reg:squaredlogerror" => Box::new(SquaredLogErrorObjective),
        "reg:logistic" => Box::new(LogisticObjective::regression(
            params.scale_pos_weight as f32,
        )),
        "multi:softmax" | "multi:softprob" => {
            if params.num_class < 2 {
                return Err(HessboostError::invalid_param(
                    "num_class",
                    "multiclass objectives require num_class >= 2",
                ));
            }
            let prob = params.objective == "multi:softprob";
            Box::new(SoftmaxObjective::new(params.num_class, prob))
        }
        "count:poisson" => Box::new(PoissonObjective::new(
            params.effective_max_delta_step() as f32
        )),
        "reg:gamma" => Box::new(GammaObjective),
        "reg:tweedie" => Box::new(TweedieObjective::new(params.tweedie_variance_power as f32)),
        "reg:quantileerror" => Box::new(QuantileObjective::new(&params.quantile_alpha)?),
        "reg:expectileerror" => Box::new(ExpectileObjective::new(&params.expectile_alpha)?),
        "reg:absoluteerror" => return Ok(Box::new(AbsoluteErrorObjective::new(n_targets))),
        "rank:pairwise" => Box::new(LambdaMartObjective::pairwise(
            params.lambdarank_num_pair_per_sample,
        )),
        "rank:ndcg" => Box::new(LambdaMartObjective::ndcg(
            params.lambdarank_num_pair_per_sample,
        )),
        "rank:map" => Box::new(LambdaMartObjective::map(
            params.lambdarank_num_pair_per_sample,
        )),
        other => return Err(HessboostError::unknown("objective", other)),
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
        let obj = LogisticObjective::new(2.0);
        let labels = [1.0f32, 0.0, 0.0, 0.0];
        let margins = obj.base_margins(&labels, None, None);
        assert_eq!(margins.len(), 1);
        // g = -0.5*2 + 3*0.5 = 0.5, h = 0.25*(2 + 3) = 1.25 -> w = -0.4.
        let mut through_link = [-0.4f32];
        obj.pred_transform(&mut through_link);
        let expected = obj.prob_to_margin(through_link[0]);
        assert_eq!(margins[0], expected);
        assert!((margins[0] + 0.4).abs() < 1e-6, "got {}", margins[0]);
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
            (Box::new(SquaredErrorObjective), 1, vec![2 * c + 4097]),
            (
                Box::new(LogisticObjective::new(1.5)),
                1,
                (1..=15).map(|r| 2 * c + r).collect(),
            ),
            (Box::new(SoftmaxObjective::new(2, true)), 2, vec![2 * c + 4]),
            (Box::new(SoftmaxObjective::new(3, true)), 3, vec![2 * c + 4]),
            (
                Box::new(SoftmaxObjective::new(9, false)),
                9,
                vec![2 * c + 1],
            ),
            // Per-output residual scales are global reductions: chunking the
            // row kernel must not change them.
            (
                Box::new(QuantileObjective::new(&[0.1, 0.5, 0.9]).unwrap()),
                3,
                vec![2 * c + 3],
            ),
            (
                Box::new(ExpectileObjective::new(&[0.2, 0.8]).unwrap()),
                2,
                vec![2 * c + 3],
            ),
            (Box::new(AbsoluteErrorObjective::new(1)), 1, vec![2 * c + 5]),
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
                let labels: Vec<f32> = (0..n)
                    .map(|i| {
                        if k == 1 {
                            (i % 2) as f32
                        } else {
                            (i % k) as f32
                        }
                    })
                    .collect();
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
