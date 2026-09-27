//! Confidence bands on a Boulevard EBM's shape functions (Fang, Tan,
//! Pipping & Hooker, *Statistical Inference for Explainable Boosting
//! Machines*, AISTATS 2026).

use rayon::prelude::*;

use super::solver::RidgeSolver;
use super::term_kernel::{TermKernel, TermPart};
use super::{
    KernelSolver, NoiseVariance, QUERY_BLOCK, build_solver, check_alpha, check_data,
    noise_estimate, z_value,
};
use crate::conformal::Interval;
use crate::data::DMatrix;
use crate::ebm::{EbmInfo, TermShape, shape_functions};
use crate::error::{HessboostError, Result};
use crate::model::{BoostedModel, Predictions};

/// Pointwise confidence bands of one term's shape function on its grid
/// ([`EbmInference::term_bands`]), aligned with [`TermShape::values`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TermBands {
    /// The term's shape function.
    pub shape: TermShape,
    /// `σ̂ (1 + λ)/λ ‖r_t(x)‖` at every cell.
    pub standard_errors: Vec<f64>,
    /// `shape − z_{1−α/2} · standard error` at every cell.
    pub lower: Vec<f64>,
    /// `shape + z_{1−α/2} · standard error` at every cell.
    pub upper: Vec<f64>,
}

/// One Boulevard stage: the centered additive kernel of its terms over
/// the training rows, and the factored `c I + K̄`.
struct Stage<'a> {
    kernel: TermKernel<'a>,
    solver: RidgeSolver,
}

/// The variance machinery of a Boulevard EBM
/// ([`Ebm::boulevard`](crate::config::Ebm::boulevard)): the
/// additive term kernels over the training rows, their factored ridge
/// systems, and a noise estimate. Fit once with [`fit`](Self::fit), then
/// query shape-function bands ([`term_bands`](Self::term_bands)) or
/// intervals for `f(x)`.
///
/// # The limit and the bands
///
/// A stage of Algorithm 1 over terms `t` (see [`crate::ebm`]) converges, as
/// the rounds grow, to the feature-wise kernel ridge regression of the
/// paper's Theorem 4.8. With `K_t` the expected tree kernel of term `t` over
/// the `n` training rows (the trees' row samples replaced by their
/// expectation, as in [`BoulevardInference`](super::BoulevardInference)),
/// `K̄ = J (Σ_t K_t) J`, `c = 1/λ`, and `s = (1 + λ)/λ`, each shape function
/// is linear in the labels,
///
/// ```text
/// ŝ_t(x) = s · r_t(x)ᵀ y,   r_t(x) = (c I + K̄)⁻¹ J k̃_t(x),
/// ```
///
/// where `k̃_t(x)` is `K_t`'s kernel vector at `x` minus its training mean
/// (the tree centering), and the CLT (Theorem 4.12) gives
/// `ŝ_t(x) ≈ N(s c_A f_t(x) + bias, σ² s² ‖r_t(x)‖²)` with `c_A = λ/(1+λ)`,
/// so the band is `ŝ_t(x) ± z_{1−α/2} σ̂ s ‖r_t(x)‖`. The intercept is the
/// label mean with variance `σ²/n`, orthogonal to the terms, and the whole
/// prediction's weight vector is `1/n + s Σ_t r_t(x)`.
///
/// Pair terms form a second stage fitted to the first stage's residuals
/// `(I − H₁) y` (`H₁` the first stage's hat matrix on the training rows), so
/// a pair's weight vector is `(I − H₁) r_p(x) = r_p − s (c I + K̄₁)⁻¹ K̄₁ r_p`,
/// and the prediction's is `1/n + s Σ_t r_t(x) + s (I − H₁) Σ_p r_p(x)`.
/// The paper covers main effects; the pair stage is this crate's extension
/// along its "isolated interaction terms" remark.
///
/// # Assumptions
///
/// Those of [`crate::inference`] (squared error with independent
/// homoscedastic noise, structure–value isolation, non-adaptive trees,
/// enough rounds), plus the paper's GAM ones: an additive truth, and for
/// the bands of individual terms Assumption 4.7 (the terms' kernels act on
/// nearly orthogonal subspaces, e.g. independent features), under which
/// every term converges separately. The variance is conditional on the
/// trees' structures, which are grown on the same labels: across training
/// samples the spread of `ŝ_t` can exceed the estimate, most where the
/// structures vary most.
///
/// # Validation
///
/// Simulated `y = sin(2π x0) + 2 (x1 − ½)² + 1[x2 > ½] + ε`, `x ~ U[0,1]³`,
/// `σ = ½`, 30 seeds, 19 points per term, `eta = 1`, `subsample = 0.8`,
/// 32 loss-guided leaves, `min_child_weight = 5`, `max_bin = 64`, 300
/// rounds, `NoiseVariance::TrainingResiduals`; coverage of the three main
/// terms' 95% bands:
///
/// | n | in-sample | [`honest_refit`](super::honest_refit) on `n` more rows |
/// |---|---|---|
/// | 500 | 0.57–0.65 | 0.78–0.80 |
/// | 1000 | 0.64–0.67 | 0.81–0.85 |
/// | 2000 | 0.65–0.71 | 0.80–0.84 |
/// | 4000 | 0.65–0.69 | 0.76–0.86 |
///
/// With the structures held fixed (same structure sample, fresh noise on
/// the refit rows) the spread of `ŝ_t` across replications matches the
/// estimated variance (ratio 1.0–1.2, pair terms included); across training
/// samples it is about twice the estimate after an honest refit and four
/// to five times in-sample, the structure variance the estimate leaves
/// out. Pair bands (a `4 (x0 − ½)(x1 − ½)` pair added) covered about 0.70
/// with an honest refit, below the main terms: the pair stage's bias is
/// larger. Treat the bands as lower bounds on the uncertainty.
///
/// The paper computes `‖r_t‖` per feature in bin space (an `m × m` system
/// per feature, ignoring the other features' kernels); this implementation
/// solves the joint `n × n` system (exactly, or with the Nyström
/// approximation), which needs no orthogonality to be the limit's variance.
///
/// # Example
///
/// ```
/// use hessboost::config::{BoosterKind, Ebm, GrowPolicy};
/// use hessboost::inference::{EbmInference, KernelSolver, NoiseVariance};
/// use hessboost::prelude::*;
///
/// # fn main() -> Result<()> {
/// let n = 300;
/// let x: Vec<f32> = (0..n * 2).map(|i| ((i * 37) % 101) as f32 / 101.0).collect();
/// let y: Vec<f32> = x
///     .chunks(2)
///     .enumerate()
///     .map(|(i, r)| (6.0 * r[0]).sin() + r[1] + 0.1 * ((i * 7919 % 101) as f32 / 50.0 - 1.0))
///     .collect();
/// let dtrain = DMatrix::from_dense(&x, n, 2)?.with_labels(&y)?;
/// let params = TrainingParams::builder()
///     .booster(BoosterKind::Ebm(Ebm::builder().boulevard(true).build()?))
///     .eta(0.5)
///     .subsample(0.8)
///     .grow_policy(GrowPolicy::LossGuide)
///     .max_leaves(8)
///     .min_child_weight(10.0)
///     .build()?;
/// let model = train(&params, &dtrain, 100)?;
/// let inference = EbmInference::fit(&model, &dtrain, NoiseVariance::Known(0.01), KernelSolver::Exact)?;
/// let bands = inference.term_bands(0, 0.05)?;
/// assert!(bands.lower.iter().zip(&bands.upper).all(|(lo, hi)| lo <= hi));
/// # Ok(())
/// # }
/// ```
pub struct EbmInference<'a> {
    model: &'a BoostedModel,
    info: &'a EbmInfo,
    stages: Vec<Stage<'a>>,
    /// The stage and part of every term.
    term_slot: Vec<(usize, usize)>,
    c: f64,
    s: f64,
    n: usize,
    noise_variance: f64,
}

impl std::fmt::Debug for EbmInference<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EbmInference")
            .field("rows", &self.n)
            .field("terms", &self.info.terms.len())
            .field("ridge", &self.c)
            .field("scale", &self.s)
            .field("noise_variance", &self.noise_variance)
            .finish_non_exhaustive()
    }
}

impl<'a> EbmInference<'a> {
    /// Build the term kernels of `model` over `train`, the rows it was
    /// trained on, factor each stage's ridge system with `solver` (the pair
    /// stage, when there is one, has its own), and estimate the noise
    /// variance.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] when `model` is not a Boulevard
    /// EBM ([`BoostedModel::ebm`] is `None` or has no
    /// [`boulevard`](EbmInfo::boulevard) record), when `train` is not its
    /// training data (a leaf holds fewer of its rows than it was grown on),
    /// has row weights or base margins, or (for the noise estimate) lacks
    /// labels; as [`BoulevardInference::fit`](super::BoulevardInference::fit)
    /// for the solver and the noise variance.
    pub fn fit(
        model: &'a BoostedModel,
        train: &DMatrix,
        noise: NoiseVariance<'a>,
        solver: KernelSolver,
    ) -> Result<Self> {
        let not_boulevard = || {
            HessboostError::invalid_param(
                "model",
                "not a Boulevard EBM: train it with `booster = ebm` and `ebm_boulevard`",
            )
        };
        let info = model.ebm().ok_or_else(not_boulevard)?;
        let settings = info.boulevard.ok_or_else(not_boulevard)?;
        check_data(
            model,
            train,
            "train",
            matches!(noise, NoiseVariance::TrainingResiduals),
        )?;
        let noise_variance = noise_estimate(model, train, noise)?;
        let lambda = settings.learning_rate;
        let (c, s) = (1.0 / lambda, (1.0 + lambda) / lambda);
        let kappa = settings.reg_lambda / settings.subsample;
        let n = train.n_rows();
        let mut term_slot = vec![(0, 0); info.terms.len()];
        let mut stages = Vec::new();
        for size in [1, 2] {
            let terms: Vec<usize> = (0..info.terms.len())
                .filter(|&t| info.terms[t].len() == size)
                .collect();
            if terms.is_empty() {
                continue;
            }
            let parts = terms
                .iter()
                .map(|&t| TermPart::new(info.term_trees(model, t), &info.terms[t], train, kappa))
                .collect::<Result<Vec<_>>>()?;
            for (k, &t) in terms.iter().enumerate() {
                term_slot[t] = (stages.len(), k);
            }
            let kernel = TermKernel::new(parts, n);
            let solver = build_solver(&kernel, solver, c)?;
            stages.push(Stage { kernel, solver });
        }
        Ok(EbmInference {
            model,
            info,
            stages,
            term_slot,
            c,
            s,
            n,
            noise_variance,
        })
    }

    /// The noise variance estimate `σ̂²`.
    pub fn noise_variance(&self) -> f64 {
        self.noise_variance
    }

    /// The standard error `σ̂ / √n` of the intercept (the label mean).
    pub fn intercept_standard_error(&self) -> f64 {
        (self.noise_variance / self.n as f64).sqrt()
    }

    /// Turn second-stage solutions `u` (`m × n`) into their weight vectors
    /// on the labels, `(I − H₁) u = u − s (c I + K̄₁)⁻¹ K̄₁ u`.
    fn through_first_stage(&self, u: &mut [f64], m: usize) {
        let first = &self.stages[0];
        let n = self.n;
        let mut v: Vec<f64> = u
            .par_chunks(n)
            .flat_map_iter(|row| first.kernel.product(row))
            .collect();
        first.solver.solve_vectors(&mut v, m, self.c);
        for (x, y) in u.iter_mut().zip(v) {
            *x -= self.s * y;
        }
    }

    /// `‖r(x)‖²` for query points given as `rhs(a, out)`, which adds the
    /// right-hand side of point `a` of stage `stage` to `out`; second-stage
    /// weights go through the first stage.
    fn norms(&self, stage: usize, m: usize, rhs: impl Fn(usize, &mut [f64]) + Sync) -> Vec<f64> {
        let n = self.n;
        (0..m.div_ceil(QUERY_BLOCK))
            .into_par_iter()
            .flat_map_iter(|b| {
                let range = b * QUERY_BLOCK..((b + 1) * QUERY_BLOCK).min(m);
                let k = range.len();
                let mut u = vec![0.0; k * n];
                for (out, a) in u.chunks_exact_mut(n).zip(range) {
                    rhs(a, out);
                }
                self.stages[stage].solver.solve_vectors(&mut u, k, self.c);
                if stage > 0 {
                    self.through_first_stage(&mut u, k);
                }
                u.chunks_exact(n)
                    .map(|w| w.iter().map(|v| v * v).sum::<f64>())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// The stage and part of term `term`, refusing unknown terms.
    fn slot(&self, term: usize) -> Result<(usize, usize)> {
        self.term_slot.get(term).copied().ok_or_else(|| {
            HessboostError::invalid_param(
                "term",
                format!("the model has {} terms, got {term}", self.term_slot.len()),
            )
        })
    }

    /// Standard errors of term `term`'s shape function at grid cells
    /// `cells`.
    fn cell_standard_errors(&self, term: usize, cells: &[usize]) -> Result<Vec<f64>> {
        let (stage, part) = self.slot(term)?;
        let kernel = &self.stages[stage].kernel;
        let scale = self.s * self.noise_variance.sqrt();
        Ok(self
            .norms(stage, cells.len(), |a, out| {
                kernel.add_query(part, cells[a], out);
            })
            .into_iter()
            .map(|w2| scale * w2.max(0.0).sqrt())
            .collect())
    }

    /// Term `term`'s shape function (as [`shape_functions`] gives it) with
    /// pointwise bands at miscoverage `alpha` on every cell of its grid:
    /// `shape ± z_{1−α/2} σ̂ (1 + λ)/λ ‖r_t‖`, asymptotically covering the
    /// term's centered limit `f_t` (see the type docs). One ridge solve per
    /// cell (a pair's grid can have many cells; see
    /// [`term_standard_errors`](Self::term_standard_errors) for chosen
    /// points).
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] when `term` is not a term of the
    /// model or `alpha` is not in `(0, 1)`.
    pub fn term_bands(&self, term: usize, alpha: f64) -> Result<TermBands> {
        check_alpha(alpha)?;
        self.slot(term)?;
        let shape = shape_functions(self.model)?.terms.swap_remove(term);
        let cells: Vec<usize> = (0..shape.values().len()).collect();
        let standard_errors = self.cell_standard_errors(term, &cells)?;
        let z = z_value(alpha);
        let (lower, upper) = shape
            .values()
            .iter()
            .zip(&standard_errors)
            .map(|(&v, &se)| (v - z * se, v + z * se))
            .unzip();
        Ok(TermBands {
            shape,
            standard_errors,
            lower,
            upper,
        })
    }

    /// The standard errors of term `term`'s shape function at every row of
    /// `data` (which has the model's features).
    ///
    /// # Errors
    ///
    /// When `term` is not a term of the model, or `data` does not have the
    /// model's features or has row weights or base margins.
    pub fn term_standard_errors(&self, term: usize, data: &DMatrix) -> Result<Predictions<f64>> {
        check_data(self.model, data, "data", false)?;
        let (stage, part) = self.slot(term)?;
        let part = &self.stages[stage].kernel.parts[part];
        let cells: Vec<usize> = (0..data.n_rows()).map(|r| part.cell_of(data, r)).collect();
        let se = self.cell_standard_errors(term, &cells)?;
        Ok(Predictions::new(se, data.n_rows(), 1))
    }

    /// `‖w(x)‖²` of the whole prediction at every row of `data`:
    /// `1/n + ‖s u₁ + s (I − H₁) u₂‖²`, with `u₁` the main stage's solution
    /// for the sum of its terms' right-hand sides and `u₂` the pair stage's
    /// (absent without pairs).
    fn prediction_norms(&self, data: &DMatrix) -> Result<Vec<f64>> {
        check_data(self.model, data, "data", false)?;
        Ok(self.joint_norms(data))
    }

    /// [`Self::prediction_norms`] of checked `data`.
    fn joint_norms(&self, data: &DMatrix) -> Vec<f64> {
        let (n, rows) = (self.n, data.n_rows());
        let cells: Vec<Vec<Vec<usize>>> = self
            .stages
            .iter()
            .map(|stage| {
                stage
                    .kernel
                    .parts
                    .iter()
                    .map(|p| (0..rows).map(|r| p.cell_of(data, r)).collect())
                    .collect()
            })
            .collect();
        (0..rows.div_ceil(QUERY_BLOCK))
            .into_par_iter()
            .flat_map_iter(|b| {
                let range = b * QUERY_BLOCK..((b + 1) * QUERY_BLOCK).min(rows);
                let k = range.len();
                let mut sides: Vec<Vec<f64>> = self
                    .stages
                    .iter()
                    .zip(&cells)
                    .map(|(stage, cells)| {
                        let mut u = vec![0.0; k * n];
                        for (out, a) in u.chunks_exact_mut(n).zip(range.clone()) {
                            for (p, cells) in cells.iter().enumerate() {
                                stage.kernel.add_query(p, cells[a], out);
                            }
                        }
                        stage.solver.solve_vectors(&mut u, k, self.c);
                        u
                    })
                    .collect();
                let mut w = sides.swap_remove(0);
                if let Some(mut second) = sides.pop() {
                    self.through_first_stage(&mut second, k);
                    for (a, b) in w.iter_mut().zip(second) {
                        *a += b;
                    }
                }
                w.chunks_exact(n)
                    .map(|w| {
                        let w2: f64 = w.iter().map(|v| v * v).sum();
                        1.0 / n as f64 + self.s * self.s * w2
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// The standard error `σ̂ ‖w(x)‖` of the model's prediction at every row
    /// of `data` (intercept and every term).
    ///
    /// # Errors
    ///
    /// When `data` does not have the model's features, or has row weights
    /// or base margins.
    pub fn standard_errors(&self, data: &DMatrix) -> Result<Predictions<f64>> {
        let sigma = self.noise_variance.sqrt();
        let se: Vec<f64> = self
            .prediction_norms(data)?
            .into_iter()
            .map(|w2| sigma * w2.max(0.0).sqrt())
            .collect();
        Ok(Predictions::new(se, data.n_rows(), 1))
    }

    /// The interval `prediction ± z · width(‖w‖²)` of every row.
    fn intervals(
        &self,
        data: &DMatrix,
        alpha: f64,
        width: impl Fn(f64) -> f64,
    ) -> Result<Vec<Interval<f64>>> {
        check_alpha(alpha)?;
        let z = z_value(alpha);
        let norms = self.prediction_norms(data)?;
        let preds = self.model.predict(data)?;
        Ok(preds
            .as_slice()
            .iter()
            .zip(norms)
            .map(|(&p, w2)| {
                let (center, half) = (f64::from(p), z * width(w2.max(0.0)));
                Interval {
                    lower: center - half,
                    upper: center + half,
                }
            })
            .collect())
    }

    /// Confidence intervals for `f(x)` at every row of `data`:
    /// `f̂(x) ± z_{1−α/2} σ̂ ‖w(x)‖` (the paper's Section 5 interval for the
    /// overall response, with the exact joint weight vector instead of its
    /// sum-of-terms bound).
    ///
    /// # Errors
    ///
    /// When `alpha` is not in `(0, 1)`, plus those of
    /// [`standard_errors`](Self::standard_errors).
    pub fn confidence_intervals(&self, data: &DMatrix, alpha: f64) -> Result<Vec<Interval<f64>>> {
        let sigma2 = self.noise_variance;
        self.intervals(data, alpha, |w2| (sigma2 * w2).sqrt())
    }

    /// Prediction intervals for a new label at every row of `data`:
    /// `f̂(x) ± z_{1−α/2} sqrt(σ̂² + σ̂² ‖w(x)‖²)`. The new label's own noise
    /// enters with the normal quantile, so the coverage holds only for
    /// Gaussian noise (the CLT covers the estimate, not the label).
    ///
    /// # Errors
    ///
    /// As [`confidence_intervals`](Self::confidence_intervals).
    pub fn prediction_intervals(&self, data: &DMatrix, alpha: f64) -> Result<Vec<Interval<f64>>> {
        let sigma2 = self.noise_variance;
        self.intervals(data, alpha, |w2| (sigma2 * (1.0 + w2)).sqrt())
    }
}
