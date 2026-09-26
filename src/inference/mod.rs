//! Statistical inference for Boulevard boosting: confidence intervals for
//! the regression function `f(x)`, prediction intervals for new labels,
//! reproduction intervals, and a variable-importance test, with asymptotic
//! (central-limit) guarantees. Beyond XGBoost and opt-in: train with
//! [`BoosterKind::Boulevard`](crate::config::BoosterKind::Boulevard), then
//! fit a [`BoulevardInference`] on the training rows.
//!
//! Unlike [`crate::conformal`], whose intervals have finite-sample
//! *marginal* coverage of the label, these intervals are about `f` itself
//! and hold pointwise (conditionally on `x`), but only asymptotically and
//! under the assumptions below.
//!
//! # The algorithms
//!
//! Boulevard (Zhou & Hooker, *Boulevard: Regularized Stochastic Gradient
//! Boosted Trees and Their Limiting Distribution*, JMLR 23, 2022) averages
//! its trees instead of summing them: after `b` rounds the ensemble is
//! `f_b = (λ / b) Σ_{i ≤ b} t_i`, each tree fitted to the residuals of the
//! current average on a row subsample. Fang, Tan & Hooker (*Statistical
//! Inference for Gradient Boosting Regression*, NeurIPS 2025) add two
//! variants that recover more of the signal, both implemented here:
//!
//! - **BRAT-D** (their Algorithm 1; `num_parallel_tree = 1`): round `b`
//!   drops each earlier tree independently with probability `p`
//!   ([`boulevard_dropout`](crate::config::TrainingParams::boulevard_dropout);
//!   `p = 0` is Zhou & Hooker's Boulevard) and fits the new tree to
//!   `y − μ − (λ / (b−1)) Σ_{kept} t_s(x)`, dividing by every earlier tree,
//!   not only the kept ones. The model predicts `μ + ((1 + λq) / B) Σ t_b`
//!   with `q = 1 − p` and learning rate `λ = eta ∈ (0, 1]`.
//! - **BRAT-P** (Algorithm 2; `num_parallel_tree = K ≥ 2`): the first
//!   iteration boosts `K` trees in sequence; afterwards tree `k` of round
//!   `b` fits `y − μ − Σ_{l ≠ k} ā_l(x)`, where `ā_l` averages slot `l`'s
//!   earlier trees, so a round's trees grow in parallel. The model predicts
//!   `μ + (1/B) Σ_{b,k} t_{b,k}`.
//!
//! `μ` is the intercept (the label mean unless `base_score` is set). With
//! [`boulevard_truncation`](crate::config::TrainingParams::boulevard_truncation)
//! `M > 0` the subtracted ensemble part is clipped to `[−M, M]`, the `Γ_M`
//! of the convergence proofs. The trained trees are ordinary
//! [`RegTree`](crate::tree::RegTree)s whose leaves already carry the final
//! scale, so prediction, SHAP, slicing, and every export work as for a
//! `gbtree` model; the native binary and JSON formats also keep the
//! [`BoulevardInfo`] inference reads.
//!
//! # The variance
//!
//! As the number of rounds grows, both algorithms converge to a kernel ridge
//! regression `f̂(x) = μ + s k(x)ᵀ (c I + K)⁻¹ (y − μ 1)` in the leaf kernel
//! of the ensemble: `K_ij` averages, over the trees, `1 / (n_ℓ + κ)` when
//! rows `i` and `j` share a leaf `ℓ` holding `n_ℓ` training rows
//! (`κ = lambda / subsample`, so a leaf's value `Σ z / (m + lambda)` over
//! its `m ≈ ξ n_ℓ` sampled rows is matched), and `k(x)` is the same average
//! between `x` and the training rows. BRAT-D has `c = 1 / (λq)` and
//! `s = (1 + λq) / (λq)`; BRAT-P `c = 1 / (K−1)` and `s = K / (K−1)`. The
//! estimate is linear in `y` with weights `w(x) = s u(x) + γ(x) 1`, where
//! `u = (c I + K)⁻¹ k(x)` and, for a label-mean intercept,
//! `γ = (1 − s 1ᵀu) / n`, so under the regression model `y = f(x) + ε` with
//! independent noise of variance `σ²`,
//!
//! ```text
//! f̂(x) ≈ N(f(x), σ² ‖w(x)‖²)     (Fang, Tan & Hooker, Theorem 2)
//! ```
//!
//! The intervals ([`BoulevardInference::confidence_intervals`] and
//! siblings) plug in an estimate `σ̂²` ([`NoiseVariance`]) and the normal
//! quantile. The kernel is estimated by the trained trees themselves (the
//! paper's equations (1)–(2)), with each tree's row sample replaced by its
//! expectation, as the authors' reference implementation does.
//! [`KernelSolver::Exact`] factors the `n × n` system (`O(n³)` time,
//! `O(n²)` memory); [`KernelSolver::Nystrom`] uses the paper's
//! Appendix A Nyström approximation from `s` uniformly sampled landmark
//! rows (`O(n s²)` time, `O(n s)` memory).
//!
//! # Assumptions
//!
//! The asymptotic guarantees rest on the papers' conditions; the estimates
//! are computed regardless, so read them as approximations when these fail:
//!
//! - **Regression with squared error**, `y = f(x) + ε` with independent,
//!   homoscedastic, sub-Gaussian noise. `booster = boulevard` refuses every
//!   other objective, row weights, and base margins, and every option that
//!   makes leaf values nonlinear in the labels (see
//!   [`TrainingParams::validate`](crate::config::TrainingParams::validate)).
//! - **Structure–value isolation** (the tree structures independent of the
//!   labels the leaves average): not true of trees grown greedily on the
//!   same labels. [`honest_refit`] provides it, refitting every leaf on an
//!   independent sample through the same Boulevard recursion (the NeurIPS
//!   paper's "integrity"); fit the inference on that sample.
//! - **Non-adaptivity** (tree structures eventually drawn from a fixed
//!   distribution), bounded leaf diameters and a minimal leaf size growing
//!   with `n` (set `min_child_weight`, which counts rows here), a row
//!   subsample, and, for BRAT-P, balanced splits. Both papers report that
//!   the intervals behave well in practice without enforcing all of them.
//! - Enough rounds that the ensemble is near its limit: the variance is the
//!   limit's.
//!
//! # Example
//!
//! ```
//! use hessboost::config::BoosterKind;
//! use hessboost::inference::{BoulevardInference, KernelSolver, NoiseVariance};
//! use hessboost::prelude::*;
//!
//! # fn main() -> Result<()> {
//! let n = 300;
//! let x: Vec<f32> = (0..n).map(|i| ((i * 37) % n) as f32 / n as f32).collect();
//! let y: Vec<f32> = x
//!     .iter()
//!     .enumerate()
//!     .map(|(i, v)| (6.0 * v).sin() + 0.1 * ((i * 7919 % 101) as f32 / 50.0 - 1.0))
//!     .collect();
//! let all = DMatrix::from_dense(&x, n, 1)?.with_labels(&y)?;
//! let (fit_rows, cal_rows): (Vec<usize>, Vec<usize>) = (0..n).partition(|i| i % 3 != 0);
//! let (dtrain, dcal) = (all.select_rows(&fit_rows)?, all.select_rows(&cal_rows)?);
//!
//! let params = TrainingParams::builder()
//!     .booster(BoosterKind::Boulevard)
//!     .eta(0.8)
//!     .boulevard_dropout(0.5)
//!     .subsample(0.8)
//!     .max_depth(3)
//!     .min_child_weight(5.0)
//!     .build()?;
//! let model = train(&params, &dtrain, 100)?;
//!
//! let inference = BoulevardInference::fit(
//!     &model,
//!     &dtrain,
//!     NoiseVariance::Holdout(&dcal),
//!     KernelSolver::Exact,
//! )?;
//! let ci = inference.confidence_intervals(&dcal, 0.05)?;
//! let pi = inference.prediction_intervals(&dcal, 0.05)?;
//! assert!(ci.iter().zip(&pi).all(|(c, p)| p.0 < c.0 && c.1 < p.1));
//! # Ok(())
//! # }
//! ```

mod kernel;
mod linalg;
mod refit;
mod solver;

use serde::{Deserialize, Serialize};

use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::objective::distributional::special::{gamma_q, norm_ppf};
use kernel::LeafKernel;
use linalg::{forward_solve, pivoted_cholesky};
use rayon::prelude::*;
use solver::RidgeSolver;

pub use refit::honest_refit;

/// Query points solved together: one block of right-hand sides.
const QUERY_BLOCK: usize = 32;

/// Relative diagonal tolerance of the importance test's covariance: test
/// points whose weight vectors are (numerically) combinations of the others'
/// are dropped, and the degrees of freedom count the rest.
const TEST_POINT_TOL: f64 = 1e-9;

/// Largest training set [`KernelSolver::Exact`] factors (its `n × n`
/// system takes `8 n²` bytes: 512 MiB here).
pub const MAX_EXACT_ROWS: usize = 8192;

/// How a `booster = boulevard` model was trained: the settings its
/// inference and [`honest_refit`] read, recorded by training
/// ([`BoostedModel::boulevard`]). Whether it is BRAT-D or BRAT-P follows
/// from the model's
/// [`num_parallel_tree`](BoostedModel::num_parallel_tree) (`1` or more).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BoulevardInfo {
    /// BRAT-D's dropout probability `p`
    /// ([`boulevard_dropout`](crate::config::TrainingParams::boulevard_dropout);
    /// `0` for BRAT-P).
    pub dropout: f64,
    /// The learning rate `λ` (`eta`; `1` for BRAT-P).
    pub learning_rate: f64,
    /// The row subsample ratio `ξ` (`subsample`).
    pub subsample: f64,
    /// The L2 leaf penalty (`lambda`).
    pub reg_lambda: f64,
    /// The residual truncation level `M`
    /// ([`boulevard_truncation`](crate::config::TrainingParams::boulevard_truncation);
    /// `0` = none).
    pub truncation: f64,
    /// The training seed, which [`honest_refit`] derives its draws from.
    pub seed: u64,
    /// Whether the intercept is the training-label mean (`base_score`
    /// unset), which the variance then includes.
    pub intercept_from_labels: bool,
}

impl BoulevardInfo {
    /// Check the recorded settings against their ranges and `model`'s
    /// layout (one output, squared error, unweighted scalar trees).
    pub(crate) fn validate(&self, model: &BoostedModel) -> Result<()> {
        let fail = |reason: &str| {
            Err(HessboostError::model_format(format!(
                "invalid Boulevard record: {reason}"
            )))
        };
        if !(self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout)) {
            return fail("dropout must be in [0, 1)");
        }
        if !(self.learning_rate > 0.0 && self.learning_rate <= 1.0) {
            return fail("learning_rate must be in (0, 1]");
        }
        if !(self.subsample > 0.0 && self.subsample <= 1.0) {
            return fail("subsample must be in (0, 1]");
        }
        if !(self.reg_lambda.is_finite() && self.reg_lambda >= 0.0) {
            return fail("reg_lambda must be finite and >= 0");
        }
        if !(self.truncation.is_finite() && self.truncation >= 0.0) {
            return fail("truncation must be finite and >= 0");
        }
        if model.num_parallel_tree() > 1 && (self.dropout != 0.0 || self.learning_rate != 1.0) {
            return fail("BRAT-P (num_parallel_tree > 1) needs dropout 0 and learning_rate 1");
        }
        if model.objective() != "reg:squarederror"
            || model.n_outputs() != 1
            || model.has_vector_leaves()
            || model.has_non_unit_tree_weights()
            || model.trees().iter().any(|t| t.linear_leaves().is_some())
        {
            return fail("only single-output reg:squarederror tree ensembles are Boulevard fits");
        }
        Ok(())
    }

    /// `(c, s)` of the kernel ridge limit for `parallel` trees per round:
    /// `f̂ = μ + s kᵀ (c I + K)⁻¹ (y − μ)`.
    fn ridge(&self, parallel: usize) -> (f64, f64) {
        if parallel > 1 {
            let k = parallel as f64;
            (1.0 / (k - 1.0), k / (k - 1.0))
        } else {
            let lq = self.learning_rate * (1.0 - self.dropout);
            (1.0 / lq, (1.0 + lq) / lq)
        }
    }

    /// The leaf-count offset `κ = lambda / subsample` of the kernel.
    fn kappa(&self) -> f64 {
        self.reg_lambda / self.subsample
    }
}

/// How [`BoulevardInference::fit`] solves the kernel ridge systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum KernelSolver {
    /// Factor the `n × n` system exactly (`O(n³)` time, `8 n²` bytes); at
    /// most [`MAX_EXACT_ROWS`] training rows.
    #[default]
    Exact,
    /// The Nyström approximation of Fang, Tan & Hooker's Appendix A:
    /// `landmarks` training rows drawn uniformly without replacement with
    /// `seed` (all of them when `landmarks >= n`, which then reproduces
    /// [`Exact`](Self::Exact) up to rounding), those numerically dependent
    /// on the others dropped. `O(n s²)` time and `8 n s` bytes for
    /// `s = landmarks`.
    Nystrom {
        /// Number of landmark rows (`>= 1`).
        landmarks: usize,
        /// Seed of the landmark draw.
        seed: u64,
    },
}

/// Where [`BoulevardInference::fit`] takes the noise variance `σ²` from.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum NoiseVariance<'a> {
    /// The mean squared residual on held-out labelled rows (the paper's
    /// estimator; also enables
    /// [`calibrated_prediction_intervals`](BoulevardInference::calibrated_prediction_intervals)).
    /// Slightly conservative: the residuals also carry the estimate's own
    /// variance and bias.
    Holdout(&'a DMatrix),
    /// The mean squared residual on the training rows themselves: no data
    /// held out, but biased low (the model has fitted some of the noise).
    TrainingResiduals,
    /// A known `σ² > 0` (simulations).
    Known(f64),
}

/// The variance machinery of one Boulevard model: its leaf kernel over the
/// training rows, the factored ridge system, and a noise estimate. Fit once
/// with [`fit`](Self::fit), then query any rows.
pub struct BoulevardInference<'a> {
    model: &'a BoostedModel,
    kernel: LeafKernel,
    solver: RidgeSolver,
    /// Ridge `c` and scale `s` of the kernel ridge limit.
    c: f64,
    s: f64,
    intercept_from_labels: bool,
    noise_variance: f64,
    holdout: Option<&'a DMatrix>,
}

impl std::fmt::Debug for BoulevardInference<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoulevardInference")
            .field("rows", &self.kernel.n())
            .field("trees", &self.kernel.n_trees())
            .field("ridge", &self.c)
            .field("scale", &self.s)
            .field("noise_variance", &self.noise_variance)
            .finish_non_exhaustive()
    }
}

/// Refuse `data` unless it has `model`'s features, one label column when
/// `labelled`, and neither row weights other than 1 nor base margins (which
/// Boulevard training refuses too).
fn check_data(
    model: &BoostedModel,
    data: &DMatrix,
    what: &'static str,
    labelled: bool,
) -> Result<()> {
    if data.n_rows() == 0 {
        return Err(HessboostError::EmptyDataset(what));
    }
    if data.n_cols() != model.n_features() {
        return Err(HessboostError::dimension_mismatch(
            what,
            model.n_features(),
            data.n_cols(),
        ));
    }
    if labelled && data.labels().is_none() {
        return Err(HessboostError::invalid_param(what, "needs labels"));
    }
    if labelled && data.n_targets() != 1 {
        return Err(HessboostError::invalid_param(
            what,
            "needs one label column",
        ));
    }
    if data.weights().is_some_and(|w| w.iter().any(|&v| v != 1.0)) {
        return Err(HessboostError::invalid_param(
            what,
            "row weights other than 1 are not supported: Boulevard inference assumes equal \
             noise per row",
        ));
    }
    if data.base_margin().is_some() {
        return Err(HessboostError::invalid_param(
            what,
            "base margins are not supported by Boulevard inference",
        ));
    }
    Ok(())
}

/// `alpha` must be a miscoverage level in `(0, 1)`.
fn check_alpha(alpha: f64) -> Result<()> {
    if alpha > 0.0 && alpha < 1.0 {
        Ok(())
    } else {
        Err(HessboostError::invalid_param(
            "alpha",
            format!("must be in (0, 1), got {alpha}"),
        ))
    }
}

/// The two-sided normal quantile `z_{1 − α/2}`.
fn z_value(alpha: f64) -> f64 {
    norm_ppf(1.0 - alpha / 2.0)
}

/// Mean squared residual of `model` on the labelled `data`.
fn mean_squared_residual(model: &BoostedModel, data: &DMatrix) -> Result<f64> {
    let preds = model.predict(data)?;
    let labels = data.labels().unwrap_or_default();
    let sum: f64 = preds
        .iter()
        .zip(labels)
        .map(|(&p, &y)| (f64::from(y) - f64::from(p)).powi(2))
        .sum();
    Ok(sum / labels.len() as f64)
}

impl<'a> BoulevardInference<'a> {
    /// Build the leaf kernel of `model` over `train`, the rows it was
    /// trained on (or, after [`honest_refit`], refitted on), factor the
    /// ridge system with `solver`, and estimate the noise variance.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] when `model` is not a Boulevard
    /// fit ([`BoostedModel::boulevard`] is `None`), when `train` is not its
    /// training data (a leaf holds fewer of its rows than it was grown on),
    /// has row weights or base margins, or (for the noise estimate) lacks
    /// labels; when [`KernelSolver::Exact`] gets more than
    /// [`MAX_EXACT_ROWS`] rows or `Nystrom` no landmark; when a noise
    /// variance is not finite and positive.
    pub fn fit(
        model: &'a BoostedModel,
        train: &DMatrix,
        noise: NoiseVariance<'a>,
        solver: KernelSolver,
    ) -> Result<Self> {
        let info = model.boulevard().ok_or_else(|| {
            HessboostError::invalid_param(
                "model",
                "not a Boulevard fit: train it with `booster = boulevard`",
            )
        })?;
        check_data(
            model,
            train,
            "train",
            matches!(noise, NoiseVariance::TrainingResiduals),
        )?;
        let n = train.n_rows();
        let noise_variance = match noise {
            NoiseVariance::Holdout(holdout) => {
                check_data(model, holdout, "holdout", true)?;
                mean_squared_residual(model, holdout)?
            }
            NoiseVariance::TrainingResiduals => mean_squared_residual(model, train)?,
            NoiseVariance::Known(v) => v,
        };
        if !(noise_variance.is_finite() && noise_variance > 0.0) {
            return Err(HessboostError::invalid_param(
                "noise",
                format!("the noise variance must be finite and > 0, got {noise_variance}"),
            ));
        }
        let (c, s) = info.ridge(model.num_parallel_tree());
        let leaves = model.predict_leaf_range(train, ..)?;
        let kernel = LeafKernel::new(model.trees(), &leaves, info.kappa())?;
        let solver = match solver {
            KernelSolver::Exact => {
                if n > MAX_EXACT_ROWS {
                    return Err(HessboostError::invalid_param(
                        "solver",
                        format!(
                            "the exact solver factors at most {MAX_EXACT_ROWS} rows, got {n}; use \
                             `KernelSolver::Nystrom`"
                        ),
                    ));
                }
                RidgeSolver::exact(&kernel, c)?
            }
            KernelSolver::Nystrom { landmarks, seed } => {
                if landmarks == 0 {
                    return Err(HessboostError::invalid_param(
                        "solver",
                        "the Nyström solver needs at least one landmark",
                    ));
                }
                RidgeSolver::nystrom(&kernel, c, landmarks, seed)?
            }
        };
        Ok(BoulevardInference {
            model,
            kernel,
            solver,
            c,
            s,
            intercept_from_labels: info.intercept_from_labels,
            noise_variance,
            holdout: match noise {
                NoiseVariance::Holdout(h) => Some(h),
                _ => None,
            },
        })
    }

    /// The noise variance estimate `σ̂²`.
    pub fn noise_variance(&self) -> f64 {
        self.noise_variance
    }

    /// The Gram matrix `w(x_a)ᵀ w(x_b)` of the estimate's weight vectors at
    /// the rows of `data` (`m × m`), from the solver's `u`-Gram and sums.
    fn weight_gram(&self, gram: &[f64], sums: &[f64], m: usize) -> Vec<f64> {
        let s = self.s;
        let n = self.kernel.n() as f64;
        let gamma: Vec<f64> = sums
            .iter()
            .map(|&su| {
                if self.intercept_from_labels {
                    (1.0 - s * su) / n
                } else {
                    0.0
                }
            })
            .collect();
        let mut out = vec![0.0; m * m];
        for a in 0..m {
            for b in 0..m {
                out[a * m + b] = s * s * gram[a * m + b]
                    + s * (gamma[a] * sums[b] + gamma[b] * sums[a])
                    + n * gamma[a] * gamma[b];
            }
        }
        out
    }

    /// The leaf node ids of `data`'s rows, `[row][tree]`.
    fn leaves(&self, data: &DMatrix) -> Result<Vec<u32>> {
        check_data(self.model, data, "data", false)?;
        self.model.predict_leaf_range(data, ..)
    }

    /// The kernel vectors of the `rows` of `leaves` (`[row][tree]`), one
    /// per row of the result (`rows.len() × n`).
    fn kernel_vectors(&self, leaves: &[u32], rows: std::ops::Range<usize>) -> Vec<f64> {
        let (n, t) = (self.kernel.n(), self.kernel.n_trees());
        let mut k = vec![0.0; rows.len() * n];
        for (out, row) in k.chunks_exact_mut(n).zip(rows) {
            self.kernel.add_query(&leaves[row * t..(row + 1) * t], out);
        }
        k
    }

    /// `‖w(x)‖²` for every row of `data`, in blocks solved in parallel.
    fn weight_norms(&self, data: &DMatrix) -> Result<Vec<f64>> {
        let leaves = self.leaves(data)?;
        let rows = data.n_rows();
        let blocks: Vec<Vec<f64>> = (0..rows.div_ceil(QUERY_BLOCK))
            .into_par_iter()
            .map(|b| {
                let range = b * QUERY_BLOCK..((b + 1) * QUERY_BLOCK).min(rows);
                let m = range.len();
                let k = self.kernel_vectors(&leaves, range);
                let solved = self.solver.solve(&k, m, self.c);
                let g = self.weight_gram(&solved.gram, &solved.sums, m);
                (0..m).map(|a| g[a * m + a].max(0.0)).collect()
            })
            .collect();
        Ok(blocks.concat())
    }

    /// The standard error `σ̂ ‖w(x)‖` of the model's prediction at every row
    /// of `data`: the estimated standard deviation of `f̂(x)` over new
    /// training samples.
    ///
    /// # Errors
    ///
    /// When `data` does not have the model's features, or has row weights
    /// or base margins.
    pub fn standard_errors(&self, data: &DMatrix) -> Result<Vec<f64>> {
        let sigma = self.noise_variance.sqrt();
        Ok(self
            .weight_norms(data)?
            .into_iter()
            .map(|w2| sigma * w2.sqrt())
            .collect())
    }

    /// `(prediction, half width)` of every row with half widths
    /// `z · width(‖w‖²)`.
    fn intervals(
        &self,
        data: &DMatrix,
        alpha: f64,
        width: impl Fn(f64) -> f64,
    ) -> Result<Vec<(f64, f64)>> {
        check_alpha(alpha)?;
        let z = z_value(alpha);
        let norms = self.weight_norms(data)?;
        let preds = self.model.predict(data)?;
        Ok(preds
            .iter()
            .zip(norms)
            .map(|(&p, w2)| {
                let (center, half) = (f64::from(p), z * width(w2));
                (center - half, center + half)
            })
            .collect())
    }

    /// Confidence intervals for the regression function `f(x)` at every row
    /// of `data` at miscoverage `alpha`: `f̂(x) ± z_{1−α/2} σ̂ ‖w(x)‖`
    /// (Fang, Tan & Hooker, equation (3)), with asymptotic pointwise
    /// coverage `1 − alpha`.
    ///
    /// # Errors
    ///
    /// When `alpha` is not in `(0, 1)`, plus those of
    /// [`standard_errors`](Self::standard_errors).
    pub fn confidence_intervals(&self, data: &DMatrix, alpha: f64) -> Result<Vec<(f64, f64)>> {
        let sigma2 = self.noise_variance;
        self.intervals(data, alpha, |w2| (sigma2 * w2).sqrt())
    }

    /// Prediction intervals for a new label `y` at every row of `data`:
    /// `f̂(x) ± z_{1−α/2} sqrt(σ̂² + σ̂² ‖w(x)‖²)`, covering `y | x` with
    /// asymptotic probability `1 − alpha` (conditionally on `x`, unlike
    /// [`crate::conformal`]'s marginal guarantee).
    ///
    /// The paper's display scales the noise term by BRAT-D's `(1 + λq) / λ`
    /// as well; that factor belongs to the estimate only (the new label's
    /// noise is not rescaled), so it is applied to `‖w‖` alone, as in the
    /// authors' reference implementation.
    ///
    /// # Errors
    ///
    /// As [`confidence_intervals`](Self::confidence_intervals).
    pub fn prediction_intervals(&self, data: &DMatrix, alpha: f64) -> Result<Vec<(f64, f64)>> {
        let sigma2 = self.noise_variance;
        self.intervals(data, alpha, |w2| (sigma2 * (1.0 + w2)).sqrt())
    }

    /// Reproduction intervals: where the prediction of the same procedure
    /// retrained on an independent sample falls, `f̂(x) ± z √2 σ̂ ‖w(x)‖`
    /// (Zhou & Hooker; the difference of two independent estimates has
    /// twice the variance).
    ///
    /// # Errors
    ///
    /// As [`confidence_intervals`](Self::confidence_intervals).
    pub fn reproduction_intervals(&self, data: &DMatrix, alpha: f64) -> Result<Vec<(f64, f64)>> {
        let sigma2 = self.noise_variance;
        self.intervals(data, alpha, |w2| (2.0 * sigma2 * w2).sqrt())
    }

    /// [`prediction_intervals`](Self::prediction_intervals) whose widths are
    /// all scaled by one factor chosen on the [`NoiseVariance::Holdout`]
    /// rows: the split-conformal quantile (rank `⌈(h + 1)(1 − α)⌉` of `h`)
    /// of `|y − f̂(x)| / half_width(x)`. Fang, Tan & Hooker's Section 6
    /// adjustment for finite samples; it keeps the per-point widths'
    /// shape. The holdout rows also estimated `σ̂²`, so the conformal
    /// guarantee holds only approximately. When the holdout set is too
    /// small for the level (`⌈(h + 1)(1 − α)⌉ > h`), every interval is
    /// `(−∞, ∞)`.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] unless the noise came from a
    /// holdout set, plus those of
    /// [`prediction_intervals`](Self::prediction_intervals).
    pub fn calibrated_prediction_intervals(
        &self,
        data: &DMatrix,
        alpha: f64,
    ) -> Result<Vec<(f64, f64)>> {
        let holdout = self.holdout.ok_or_else(|| {
            HessboostError::invalid_param(
                "noise",
                "calibrated intervals need `NoiseVariance::Holdout` rows",
            )
        })?;
        let calibration = self.prediction_intervals(holdout, alpha)?;
        let labels = holdout.labels().unwrap_or_default();
        let mut ratios: Vec<f64> = calibration
            .iter()
            .zip(labels)
            .map(|(&(lo, hi), &y)| {
                let half = (hi - lo) / 2.0;
                let center = f64::midpoint(hi, lo);
                (f64::from(y) - center).abs() / half.max(f64::MIN_POSITIVE)
            })
            .collect();
        let scale = match crate::conformal::conformal_rank(ratios.len(), alpha) {
            Some(k) => *ratios.select_nth_unstable_by(k - 1, f64::total_cmp).1,
            None => f64::INFINITY,
        };
        Ok(self
            .prediction_intervals(data, alpha)?
            .into_iter()
            .map(|(lo, hi)| {
                let (center, half) = (f64::midpoint(hi, lo), (hi - lo) / 2.0 * scale);
                if half.is_finite() {
                    (center - half, center + half)
                } else {
                    (f64::NEG_INFINITY, f64::INFINITY)
                }
            })
            .collect())
    }
}

/// The outcome of [`importance_test`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ImportanceTest {
    /// The chi-squared statistic `σ̂⁻² dᵀ Ξ⁻¹ d`.
    pub statistic: f64,
    /// Its degrees of freedom: the number of test points kept (those whose
    /// weight vectors are not numerically combinations of the others').
    pub degrees_of_freedom: usize,
    /// `P(χ²_df ≥ statistic)`.
    pub p_value: f64,
}

/// Fang, Tan & Hooker's variable-importance test (Section 4): does `f`
/// depend on features that a reduced model leaves out?
///
/// Split the training data into two independent halves; train `full` (all
/// features) on the first and `reduced` (the features kept under the null)
/// on the second, and fit each one's [`BoulevardInference`] on its own half.
/// At `m` test points (`full_points` and `reduced_points`: the same rows in
/// each model's feature layout) the difference of the predictions `d` is,
/// under `H₀: f = g` (the projection of `f` on the kept features),
/// asymptotically `N(0, σ² Ξ)` with `Ξ = W₁ W₁ᵀ + W₂ W₂ᵀ` (the two
/// estimates' weight vectors, independent by the split), so
/// `σ̂⁻² dᵀ Ξ⁻¹ d ~ χ²_m`. `σ̂²` is the full model's noise estimate. Test
/// points that duplicate others (numerically) are dropped from `Ξ` and the
/// degrees of freedom. Keep `m` well below the training sizes; the cost is
/// that of `m` variance queries plus `O(m³)`.
///
/// # Errors
///
/// [`HessboostError::DimensionMismatch`] when the two point sets differ in
/// row count, plus those of [`BoulevardInference::standard_errors`] for
/// either model's points.
pub fn importance_test(
    full: &BoulevardInference,
    full_points: &DMatrix,
    reduced: &BoulevardInference,
    reduced_points: &DMatrix,
) -> Result<ImportanceTest> {
    let m = full_points.n_rows();
    if reduced_points.n_rows() != m {
        return Err(HessboostError::dimension_mismatch(
            "importance test points",
            m,
            reduced_points.n_rows(),
        ));
    }
    let gram = |inf: &BoulevardInference, data: &DMatrix| -> Result<Vec<f64>> {
        let leaves = inf.leaves(data)?;
        let k = inf.kernel_vectors(&leaves, 0..m);
        let solved = inf.solver.solve(&k, m, inf.c);
        Ok(inf.weight_gram(&solved.gram, &solved.sums, m))
    };
    let (g1, g2) = (gram(full, full_points)?, gram(reduced, reduced_points)?);
    let xi: Vec<f64> = g1.iter().zip(&g2).map(|(a, b)| a + b).collect();
    let p1 = full.model.predict(full_points)?;
    let p2 = reduced.model.predict(reduced_points)?;
    let d: Vec<f64> = p1
        .iter()
        .zip(&p2)
        .map(|(&a, &b)| f64::from(a) - f64::from(b))
        .collect();
    let pivoted = pivoted_cholesky(&xi, m, TEST_POINT_TOL);
    let r = pivoted.rank();
    let mut y: Vec<f64> = pivoted.pivots.iter().map(|&p| d[p]).collect();
    forward_solve(&pivoted.factor, r, &mut y, 1);
    let statistic = y.iter().map(|v| v * v).sum::<f64>() / full.noise_variance;
    let p_value = if r == 0 {
        1.0
    } else {
        gamma_q(r as f64 / 2.0, statistic / 2.0)
    };
    Ok(ImportanceTest {
        statistic,
        degrees_of_freedom: r,
        p_value,
    })
}
