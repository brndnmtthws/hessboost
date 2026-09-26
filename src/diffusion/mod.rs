//! Nonparametric probabilistic regression: conditional diffusion and flow
//! matching with boosted trees as the score or velocity model (beyond
//! XGBoost, opt-in).
//!
//! A [`DiffusionModel`] learns the whole conditional distribution `p(y | x)`
//! of a scalar or vector label, with no parametric family: multimodal,
//! skewed, heavy-tailed, heteroscedastic and correlated multivariate labels
//! are all in reach, which complements the parametric `dist:*` objectives of
//! [`objective::distributional`](crate::objective::distributional). Instead
//! of a density it returns samples ([`DiffusionModel::sample`]); means,
//! quantiles and CRPS are Monte Carlo estimates over them ([`Samples`]).
//!
//! Training reduces to ordinary squared-error boosting: every labelled row
//! is repeated [`n_repeats`](DiffusionParams::n_repeats) times, its label
//! `y₀` noised to `y_t` at a random time `t`, and one multi-output GBDT
//! (trained through [`Trainer`](crate::training::Trainer)) learns to map
//! `(y_t, x, t)` to the target that reconstructs the score `∇ log p_t(y_t |
//! x)` or the flow's velocity. Sampling integrates the reverse-time SDE or
//! ODE from the prior down to `t = 10⁻⁵`, one batch prediction per step over
//! every `(row, sample)` pair.
//!
//! # Methods
//!
//! [`Method::Score`] is score-based diffusion ([`ScoreConfig`]). The noising
//! process is a Gaussian SDE ([`Sde`]): variance exploding (the default,
//! `σ(t) = σ_min (σ_max/σ_min)^t`), variance preserving, or sub-variance
//! preserving. Sampling runs Euler–Maruyama on the reverse SDE. Two
//! parameterizations of the regression target are offered:
//!
//! - [`Parameterization::Noise`], Treeffuser's: the GBDT predicts `-z`, the
//!   negated noise of `y_t = α(t) y₀ + σ(t) z`, and the score is its
//!   prediction divided by `σ(t)`.
//! - [`Parameterization::Edm`], DiffGBM's default: EDM preconditioning
//!   (Karras et al., 2022). The GBDT sees `c_in(σ) y_t` and predicts the
//!   residual `(y₀ - c_skip y_t) / c_out`, which keeps the target at unit
//!   scale at every noise level; the denoiser `D = c_skip y_t + c_out F` gives
//!   the score `(α D - y_t) / σ²`.
//!
//! [`Method::FlowMatching`] is conditional flow matching ([`FlowMatchingConfig`]):
//! along a Gaussian path `y_t = a(t) y₀ + b(t) z` ([`FlowPath`]: linear,
//! trigonometric, or variance preserving), the GBDT regresses the velocity
//! `a'(t) y₀ + b'(t) z`, and sampling integrates the deterministic
//! reverse-time ODE with Euler or Heun steps ([`OdeSolver`]). Five Heun steps
//! suffice (DiffGBM's operating point), which makes it about ten times
//! cheaper to sample than 50-step score diffusion.
//!
//! Both methods draw training times uniformly on `[10⁻⁵, 1]` or, EDM-style,
//! with a normal log noise level ([`TimeSampling`]), which moves histogram
//! bins (and so the trees' capacity) toward the noise levels that shape the
//! distribution. Score diffusion can add `ln σ(t)` as an explicit feature
//! ([`ScoreConfig::noise_level_feature`]).
//!
//! # Preprocessing
//!
//! Labels are standardized per column (mean `0`, standard deviation `1`;
//! a constant column keeps scale `1`). With a [`Residualizer`] (DiffGBM's
//! conditional-mean residualization), `k`-fold cross-fitted GBDTs estimate
//! `E[y | x]`; the diffusion then learns the distribution of the
//! out-of-fold residuals, centered and divided by their 1%/99%-winsorized
//! standard deviation, and sampling adds the fold models' averaged mean
//! back. This moves the conditional location out of the diffusion's target.
//! Features are used as given: tree splits do not change under the
//! per-feature affine maps Treeffuser standardizes them with.
//!
//! # Configurations
//!
//! - [`DiffusionParams::default`]: DiffGBM's score-side recipe (the corner
//!   its "score-flex" search selected most often): VE SDE, EDM
//!   preconditioning, a log-noise feature, log-σ time sampling,
//!   conditional-mean residualization, 50 Euler–Maruyama steps.
//! - [`DiffusionParams::treeffuser`]: the published Treeffuser recipe:
//!   noise parameterization, raw time feature, uniform time sampling, no
//!   residualization.
//! - [`DiffusionParams::flow_matching`]: DiffGBM's flow-matching corner: VP
//!   path, log-noise time sampling, residualization, 5 Heun steps.
//!
//! Every configuration boosts with LightGBM's defaults as Treeffuser and
//! DiffGBM use them (leaf-wise growth with 31 leaves, learning rate 0.1, 20
//! rows per leaf, no L2 penalty, 255 bins) for up to 3000 rounds, stopped
//! after 50 rounds without improvement on a 10% validation split of the
//! original rows (split before repetition, so no row's noisy copies straddle
//! it). Every one of these is a field of [`DiffusionParams`].
//!
//! # Multi-output labels
//!
//! A label matrix ([`DMatrix::with_label_matrix`]) with `d` columns is
//! modelled jointly: the GBDT sees all `d` noisy coordinates and predicts
//! the `d`-dimensional target through hessboost's multi-output boosting.
//! With the default [`MultiStrategy::OneOutputPerTree`](crate::config::MultiStrategy::OneOutputPerTree)
//! each output gets its own trees, as Treeffuser fits one regressor per
//! output; [`MultiStrategy::MultiOutputTree`](crate::config::MultiStrategy::MultiOutputTree)
//! in [`DiffusionParams::training`] shares vector-leaf trees across outputs
//! instead (fewer trees, cheaper sampling, one tree structure for all score
//! coordinates). Either way the outputs share one early-stopping round (the
//! mean validation RMSE), where Treeffuser stops each output separately.
//!
//! # Sampling
//!
//! [`DiffusionModel::sample`] returns `n_samples` draws for every row of a
//! feature matrix, laid out row-major `[row][sample][output]`. The noise of
//! draw `s` of row `r` comes from a counter-based SplitMix64 stream keyed by
//! the seed, `r` and `s`, so the result depends only on the model, the
//! matrix, `n_samples` and the seed: never on the thread count or on how the
//! sampler batches pairs, and the first `k` samples of a row are the same for
//! every `n_samples ≥ k`.
//!
//! # Persistence
//!
//! [`DiffusionModel::to_bytes`] writes a zstd-compressed section container
//! (magic `HBDM`) holding the method, the standardization, the residualizer
//! and the GBDTs as embedded native containers;
//! [`DiffusionModel::to_json`] writes the same content as JSON, with each
//! GBDT in the native JSON format. Both readers validate the model.
//!
//! # Refusals
//!
//! Fitting refuses data without labels, instance weights, base margins,
//! ranking groups, label bounds, or feature weights; an objective other than
//! `reg:squarederror`; residualization with fewer than 80 rows; and
//! non-positive or non-finite process parameters. Sampling refuses a matrix
//! whose feature count differs from the training data's, or one with base
//! margins.
//!
//! # Sources
//!
//! - N. Beltran-Velez, A. A. Grande, A. Nazaret, A. Kucukelbir, D. Blei,
//!   *Treeffuser: Probabilistic Predictions via Conditional Diffusions with
//!   Gradient-Boosted Trees*, NeurIPS 2024
//!   (<https://arxiv.org/abs/2406.07658>, reference code
//!   <https://github.com/blei-lab/treeffuser>).
//! - S. Koemen, *Conditioning Tree-Based Diffusions and Flows for
//!   Probabilistic Tabular Regression* (DiffGBM), 2026
//!   (<https://arxiv.org/abs/2607.28864>, reference code
//!   <https://github.com/silaskoemen/diffgbm>).
//! - T. Karras, M. Aittala, T. Aila, S. Laine, *Elucidating the Design Space
//!   of Diffusion-Based Generative Models* (EDM), NeurIPS 2022.
//! - Y. Song et al., *Score-Based Generative Modeling through Stochastic
//!   Differential Equations*, ICLR 2021 (the VE, VP and sub-VP SDEs).
//!
//! Deviations from the reference code: the validation split is taken
//! before repetition (DiffGBM's fix; Treeffuser's code repeats the full data,
//! validation rows included); the outputs share one early-stopping round;
//! residualization refuses fewer than 80 rows (DiffGBM warns and skips it);
//! and the random streams are hessboost's, so results match the references
//! in quality, not draw for draw.
//!
//! # Example
//!
//! ```
//! use hessboost::diffusion::{DiffusionModel, DiffusionParams};
//! use hessboost::prelude::*;
//!
//! # fn main() -> Result<()> {
//! // A bimodal label: y = ±1 (by a hidden coin) plus a little noise.
//! let n = 200;
//! let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
//! let y: Vec<f32> = (0..n)
//!     .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 } + 0.05 * ((i * 7 % 11) as f32 - 5.0) / 5.0)
//!     .collect();
//! let data = DMatrix::from_dense(&x, n, 1)?.with_labels(&y)?;
//!
//! let mut params = DiffusionParams::flow_matching();
//! params.n_repeats = 5;
//! params.num_boost_round = 50;
//! let model = DiffusionModel::fit(&params, &data)?;
//!
//! let samples = model.sample(&data, 20, 7)?; // [row][sample][output]
//! assert_eq!(samples.values().len(), n * 20);
//! let q = samples.quantiles(&[0.1, 0.9])?; // [row][level][output]
//! assert!(q[0] < q[1]);
//!
//! let restored = DiffusionModel::from_bytes(&model.to_bytes()?)?;
//! assert_eq!(restored.sample(&data, 20, 7)?, samples);
//! # Ok(())
//! # }
//! ```

mod fit;
mod format;
mod process;
mod sample;

use serde::{Deserialize, Serialize};

use crate::config::{GrowPolicy, TrainingParams, TreeMethod};
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;

pub use sample::Samples;

/// What the GBDT learns and how sampling integrates it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Method {
    /// Score-based diffusion: the GBDT reconstructs the score of a Gaussian
    /// SDE's marginals; sampling runs Euler–Maruyama on the reverse SDE.
    Score(ScoreConfig),
    /// Conditional flow matching: the GBDT regresses a Gaussian path's
    /// velocity; sampling integrates the reverse ODE.
    FlowMatching(FlowMatchingConfig),
}

/// Settings of [`Method::Score`]. [`Default`] is DiffGBM's score-side
/// recipe; [`ScoreConfig::treeffuser`] Treeffuser's.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ScoreConfig {
    /// The noising SDE.
    pub sde: Sde,
    /// The GBDT's regression target and how the score is rebuilt from it.
    pub parameterization: Parameterization,
    /// Add `ln σ(t)`, the log noise scale, as a feature after `t`
    /// (DiffGBM's `raw_time_log_std` layout; Treeffuser's `raw_time` without
    /// it).
    pub noise_level_feature: bool,
    /// Distribution of the training times.
    pub time_sampling: TimeSampling,
}

impl Default for ScoreConfig {
    /// VE SDE (`σ_min = 0.01`, `σ_max = 20`), EDM with `σ_data = 1`, the
    /// log-noise feature, and `ln σ(t) ~ N(-1.2, 1.2²)` training times.
    fn default() -> Self {
        ScoreConfig {
            sde: Sde::default(),
            parameterization: Parameterization::Edm { sigma_data: 1.0 },
            noise_level_feature: true,
            time_sampling: TimeSampling::default(),
        }
    }
}

impl ScoreConfig {
    /// Treeffuser's published configuration: VE SDE (`σ_min = 0.01`,
    /// `σ_max = 20`), noise parameterization, no noise-level feature,
    /// uniform training times.
    pub fn treeffuser() -> Self {
        ScoreConfig {
            sde: Sde::default(),
            parameterization: Parameterization::Noise,
            noise_level_feature: false,
            time_sampling: TimeSampling::Uniform,
        }
    }
}

/// Settings of [`Method::FlowMatching`]. [`Default`] is DiffGBM's
/// flow-matching configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FlowMatchingConfig {
    /// The Gaussian probability path from data (`t = 0`) to `N(0, I)`
    /// (`t = 1`).
    pub path: FlowPath,
    /// Distribution of the training times. [`TimeSampling::Uniform`] also
    /// sets 5% of the rows to `t = 1` (DiffGBM's endpoint anchor).
    pub time_sampling: TimeSampling,
    /// The reverse-ODE integrator.
    pub solver: OdeSolver,
}

impl Default for FlowMatchingConfig {
    /// The VP path (`β_min = 0.1`, `β_max = 20`), `ln b(t) ~ N(-1.2, 1.2²)`
    /// training times (clipped at `ln b(1)`), and Heun steps.
    fn default() -> Self {
        FlowMatchingConfig {
            path: FlowPath::VariancePreserving {
                beta_min: 0.1,
                beta_max: 20.0,
            },
            time_sampling: TimeSampling::default(),
            solver: OdeSolver::Heun,
        }
    }
}

/// The noising SDE of [`Method::Score`], with Treeffuser's schedules.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Sde {
    /// `dY = √(2σσ') dW` with `σ(t) = σ_min (σ_max/σ_min)^t`: kernel
    /// `N(y₀, σ(t)² - σ_min²)`, prior `N(0, σ_max²)`. Needs
    /// `0 < sigma_min < sigma_max`.
    VarianceExploding {
        /// `σ(0)`.
        sigma_min: f64,
        /// `σ(1)`, the prior's standard deviation.
        sigma_max: f64,
    },
    /// `dY = -½β Y dt + √β dW` with `β(t) = β_min + (β_max - β_min) t`:
    /// kernel `N(e^{-B/2} y₀, 1 - e^{-B})` with `B = ∫₀ᵗ β`, prior
    /// `N(0, 1)`. Needs `0 < beta_min < beta_max`.
    VariancePreserving {
        /// `β(0)`.
        beta_min: f64,
        /// `β(1)`.
        beta_max: f64,
    },
    /// `dY = -½β Y dt + √(β (1 - e^{-2B})) dW`: kernel
    /// `N(e^{-B/2} y₀, (1 - e^{-B})²)`, prior `N(0, 1)`. Needs
    /// `0 < beta_min < beta_max`.
    SubVariancePreserving {
        /// `β(0)`.
        beta_min: f64,
        /// `β(1)`.
        beta_max: f64,
    },
}

impl Default for Sde {
    /// The VE SDE with Treeffuser's `σ_min = 0.01`, `σ_max = 20`.
    fn default() -> Self {
        Sde::VarianceExploding {
            sigma_min: 0.01,
            sigma_max: 20.0,
        }
    }
}

/// The regression target of [`Method::Score`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Parameterization {
    /// Predict the negated noise `-z`; score `= prediction / σ(t)`
    /// (Treeffuser).
    Noise,
    /// EDM preconditioning with data scale `sigma_data` (`> 0`; the
    /// standardized labels make `1` natural): input `c_in y_t`, target
    /// `(y₀ - c_skip y_t) / c_out` with `c_skip = σ_d²/(σ² + σ_d²)`,
    /// `c_out = σ σ_d/√(σ² + σ_d²)`, `c_in = 1/√(σ² + σ_d²)`. For the VP
    /// SDEs this remains a valid preconditioned `y₀` target, though the
    /// coefficients are no longer the variance-optimal ones.
    Edm {
        /// `σ_d`.
        sigma_data: f64,
    },
}

/// Distribution of the training times `t`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TimeSampling {
    /// Uniform on `[10⁻⁵, 1]` (Treeffuser).
    Uniform,
    /// The log noise scale (`ln σ(t)` of an SDE, `ln b(t)` of a flow path)
    /// normal with `mean` and `std` (`> 0`), clipped to its range on
    /// `[10⁻⁵, 1]` and mapped back to `t` through a 1024-point table
    /// (EDM's log-normal noise levels, as DiffGBM applies them).
    LogNoiseNormal {
        /// Mean of the log noise scale.
        mean: f64,
        /// Standard deviation of the log noise scale.
        std: f64,
    },
}

impl Default for TimeSampling {
    /// EDM's `P_mean = -1.2`, `P_std = 1.2`.
    fn default() -> Self {
        TimeSampling::LogNoiseNormal {
            mean: -1.2,
            std: 1.2,
        }
    }
}

/// The Gaussian path `y_t = a(t) y₀ + b(t) z` of [`Method::FlowMatching`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FlowPath {
    /// Rectified flow: `a = 1 - t`, `b = t`.
    Linear,
    /// `a = cos(πt/2)`, `b = sin(πt/2)` (variance preserving).
    Trigonometric,
    /// DDPM's linear-β schedule: `a = √ᾱ`, `b = √(1 - ᾱ)` with
    /// `ᾱ = exp(-(β_min t/2 + (β_max - β_min) t²/4))`. Needs
    /// `0 < beta_min < beta_max`.
    VariancePreserving {
        /// `β(0)`.
        beta_min: f64,
        /// `β(1)`.
        beta_max: f64,
    },
}

/// The reverse-ODE integrator of [`Method::FlowMatching`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum OdeSolver {
    /// Explicit Euler: one GBDT evaluation per step.
    Euler,
    /// Heun's second-order predictor–corrector: two evaluations per step.
    Heun,
}

/// Early stopping of the score/velocity GBDT on a validation split of the
/// original rows.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct EarlyStopping {
    /// Rounds without improvement of the validation RMSE before stopping
    /// (`> 0`).
    pub rounds: usize,
    /// Fraction of the rows held out (in `(0, 1)`; `ceil(fraction · n)`
    /// rows, at least one row on each side).
    pub eval_fraction: f64,
}

impl Default for EarlyStopping {
    /// Treeffuser's 50 rounds on 10% of the rows.
    fn default() -> Self {
        EarlyStopping {
            rounds: 50,
            eval_fraction: 0.1,
        }
    }
}

/// Cross-fitted conditional-mean residualization (DiffGBM's
/// `residualize = "mean"`).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Residualizer {
    /// Number of cross-fitting folds (`>= 2`). With `n` rows,
    /// `min(folds, max(2, n / 40))` are used, so every fold keeps about 40
    /// rows or more; fewer than 80 rows are refused.
    pub folds: usize,
    /// Parameters of the fold models (objective `reg:squarederror`).
    pub training: TrainingParams,
    /// Boosting rounds of each fold model (`> 0`).
    pub num_boost_round: usize,
}

impl Default for Residualizer {
    /// DiffGBM's default residualizer: 5 folds of 100 rounds, learning rate
    /// 0.05, depth 6, 31 leaves, 20 rows per leaf.
    fn default() -> Self {
        Residualizer {
            folds: 5,
            training: TrainingParams {
                eta: 0.05,
                max_depth: 6,
                ..lightgbm_like()
            },
            num_boost_round: 100,
        }
    }
}

/// Configuration of [`DiffusionModel::fit`]. [`Default`] is DiffGBM's
/// score-side recipe; see the [module docs](self#configurations) for the
/// presets.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DiffusionParams {
    /// Score diffusion or flow matching, with its settings.
    pub method: Method,
    /// Noisy copies of each training row (`> 0`; Treeffuser's
    /// `n_repeats`).
    pub n_repeats: usize,
    /// Integration steps of the sampler (`> 0`), stored with the model.
    pub n_steps: usize,
    /// Parameters of the score/velocity GBDT (objective
    /// `reg:squarederror`).
    pub training: TrainingParams,
    /// Maximum boosting rounds of the score/velocity GBDT (`> 0`).
    pub num_boost_round: usize,
    /// Early stopping on a validation split; `None` trains every round on
    /// all rows.
    pub early_stopping: Option<EarlyStopping>,
    /// Conditional-mean residualization; `None` diffuses the standardized
    /// labels themselves.
    pub residualizer: Option<Residualizer>,
    /// Seed of the validation split, the residualizer's folds, and the
    /// training noise and times. The GBDTs' own sampling uses their
    /// [`TrainingParams::seed`].
    pub seed: u64,
}

impl Default for DiffusionParams {
    fn default() -> Self {
        DiffusionParams {
            method: Method::Score(ScoreConfig::default()),
            n_repeats: 30,
            n_steps: 50,
            training: lightgbm_like(),
            num_boost_round: 3000,
            early_stopping: Some(EarlyStopping::default()),
            residualizer: Some(Residualizer::default()),
            seed: 0,
        }
    }
}

impl DiffusionParams {
    /// Treeffuser's published recipe: [`ScoreConfig::treeffuser`], no
    /// residualization, 50 Euler–Maruyama steps.
    pub fn treeffuser() -> Self {
        DiffusionParams {
            method: Method::Score(ScoreConfig::treeffuser()),
            residualizer: None,
            ..DiffusionParams::default()
        }
    }

    /// DiffGBM's flow-matching configuration: [`FlowMatchingConfig::default`]
    /// with residualization and 5 Heun steps.
    pub fn flow_matching() -> Self {
        DiffusionParams {
            method: Method::FlowMatching(FlowMatchingConfig::default()),
            n_steps: 5,
            ..DiffusionParams::default()
        }
    }

    /// Check every setting, [`DiffusionModel::fit`]'s first step.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] for a zero count, an
    /// out-of-range fraction or process parameter, or GBDT parameters that
    /// fail [`TrainingParams::validate`] or name an objective other than
    /// `reg:squarederror`.
    pub fn validate(&self) -> Result<()> {
        self.method.validate()?;
        positive_count("n_repeats", self.n_repeats)?;
        positive_count("n_steps", self.n_steps)?;
        positive_count("num_boost_round", self.num_boost_round)?;
        validate_regressor_params("training", &self.training)?;
        if let Some(stop) = &self.early_stopping {
            positive_count("early_stopping.rounds", stop.rounds)?;
            let f = stop.eval_fraction;
            if !(f.is_finite() && f > 0.0 && f < 1.0) {
                return Err(HessboostError::invalid_param(
                    "early_stopping.eval_fraction",
                    format!("must be in (0, 1), got {f}"),
                ));
            }
        }
        if let Some(r) = &self.residualizer {
            if r.folds < 2 {
                return Err(HessboostError::invalid_param(
                    "residualizer.folds",
                    format!("must be at least 2, got {}", r.folds),
                ));
            }
            positive_count("residualizer.num_boost_round", r.num_boost_round)?;
            validate_regressor_params("residualizer.training", &r.training)?;
        }
        Ok(())
    }
}

/// LightGBM's defaults as Treeffuser and DiffGBM train with them: leaf-wise
/// growth to 31 leaves without a depth limit, learning rate 0.1, 20 rows
/// per leaf (unit Hessians), no L2 penalty, 255 bins.
fn lightgbm_like() -> TrainingParams {
    TrainingParams {
        tree_method: TreeMethod::Hist,
        grow_policy: GrowPolicy::LossGuide,
        max_depth: 0,
        max_leaves: 31,
        eta: 0.1,
        min_child_weight: 20.0,
        lambda: 0.0,
        max_bin: 255,
        ..TrainingParams::default()
    }
}

fn positive_count(name: &'static str, v: usize) -> Result<()> {
    if v == 0 {
        return Err(HessboostError::invalid_param(name, "must be at least 1"));
    }
    Ok(())
}

/// `params` validate and regress with squared error, the loss every
/// diffusion target is fit with.
fn validate_regressor_params(name: &'static str, params: &TrainingParams) -> Result<()> {
    params.validate()?;
    if params.objective != "reg:squarederror" {
        return Err(HessboostError::invalid_param(
            name,
            format!(
                "the diffusion targets are regressed with squared error: objective must be \
                 `reg:squarederror`, got `{}`",
                params.objective
            ),
        ));
    }
    Ok(())
}

/// `v` is finite and `> 0`.
fn check_positive(name: &'static str, v: f64) -> Result<()> {
    if !(v.is_finite() && v > 0.0) {
        return Err(HessboostError::invalid_param(
            name,
            format!("must be finite and > 0, got {v}"),
        ));
    }
    Ok(())
}

/// `0 < lo < hi`, both finite.
fn check_schedule(name: &'static str, lo: f64, hi: f64) -> Result<()> {
    check_positive(name, lo)?;
    check_positive(name, hi)?;
    if lo >= hi {
        return Err(HessboostError::invalid_param(
            name,
            format!("the minimum ({lo}) must be below the maximum ({hi})"),
        ));
    }
    Ok(())
}

impl Method {
    /// Check the process parameters (shared by [`DiffusionParams::validate`]
    /// and the model readers).
    fn validate(&self) -> Result<()> {
        match self {
            Method::Score(score) => {
                match score.sde {
                    Sde::VarianceExploding {
                        sigma_min,
                        sigma_max,
                    } => check_schedule("sde", sigma_min, sigma_max)?,
                    Sde::VariancePreserving { beta_min, beta_max }
                    | Sde::SubVariancePreserving { beta_min, beta_max } => {
                        check_schedule("sde", beta_min, beta_max)?;
                    }
                }
                if let Parameterization::Edm { sigma_data } = score.parameterization {
                    check_positive("sigma_data", sigma_data)?;
                    // The coefficients divide by `σ_d²`-sized terms.
                    check_positive("sigma_data", sigma_data * sigma_data)?;
                }
                // Finite parameters can still overflow the kernel (e.g. a VE
                // `σ_max²` beyond `f64::MAX`); every quantity is monotone in
                // `t`, so the endpoints bound it on `[T_EPS, 1]`.
                let finite = [process::T_EPS, 1.0].iter().all(|&t| {
                    let (alpha, std) = score.sde.marginal(t);
                    let (c, g2) = score.sde.drift_diffusion(t);
                    [alpha, std, std.ln(), c, g2].iter().all(|v| v.is_finite())
                }) && score.sde.prior_std().is_finite();
                if !finite {
                    return Err(HessboostError::invalid_param(
                        "sde",
                        "the schedule's noise scale or drift is zero or overflows on [1e-5, 1]",
                    ));
                }
                score.time_sampling.validate()
            }
            Method::FlowMatching(flow) => {
                if let FlowPath::VariancePreserving { beta_min, beta_max } = flow.path {
                    check_schedule("path", beta_min, beta_max)?;
                }
                let finite = [process::T_EPS, 1.0].iter().all(|&t| {
                    let c = flow.path.coefficients(t);
                    [c.a, c.b, c.b.ln(), c.da, c.db]
                        .iter()
                        .all(|v| v.is_finite())
                });
                if !finite {
                    return Err(HessboostError::invalid_param(
                        "path",
                        "the path's noise scale or velocity is zero or overflows on [1e-5, 1]",
                    ));
                }
                flow.time_sampling.validate()
            }
        }
    }

    /// Number of noise-level columns after the features: `t`, plus `ln σ(t)`
    /// with the noise-level feature.
    fn time_columns(&self) -> usize {
        match self {
            Method::Score(score) => 1 + usize::from(score.noise_level_feature),
            Method::FlowMatching(_) => 1,
        }
    }
}

impl TimeSampling {
    fn validate(self) -> Result<()> {
        if let TimeSampling::LogNoiseNormal { mean, std } = self {
            if !mean.is_finite() {
                return Err(HessboostError::invalid_param(
                    "time_sampling",
                    format!("mean must be finite, got {mean}"),
                ));
            }
            check_positive("time_sampling", std)?;
        }
        Ok(())
    }
}

/// The conditional-mean residualizer of a fitted model: `y_std = mean(x) +
/// scale · u + center`, `mean(x)` the fold models' average.
#[derive(Debug, Clone, Serialize)]
struct FittedResidualizer {
    models: Vec<BoostedModel>,
    center: Vec<f64>,
    scale: Vec<f64>,
}

/// A fitted conditional diffusion or flow-matching model: the
/// score/velocity GBDT, the label standardization, the optional
/// residualizer, and the sampler settings. See the [module docs](self).
///
/// The serde implementations are the JSON format ([`Self::to_json`] /
/// [`Self::from_json`]); deserializing validates the model like the loaders
/// do.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "format::UncheckedDiffusionModel")]
pub struct DiffusionModel {
    method: Method,
    n_steps: usize,
    /// Feature columns of the data (`x`).
    n_features: usize,
    /// Label columns (`y`).
    n_outputs: usize,
    /// Per-column label mean.
    target_mean: Vec<f64>,
    /// Per-column label standard deviation (`1` for a constant column).
    target_scale: Vec<f64>,
    residualizer: Option<FittedResidualizer>,
    /// The GBDT on `[y_t (n_outputs), x (n_features), t, (ln σ(t))]`.
    regressor: BoostedModel,
}

impl DiffusionModel {
    /// Fit a model of `p(y | x)` to `data` (features and a label vector or
    /// label matrix): standardize the labels, residualize them if
    /// configured, build the noisy training set, and boost the
    /// score/velocity GBDT. Deterministic for fixed `params` and data at any
    /// thread count.
    ///
    /// # Errors
    ///
    /// Everything [`DiffusionParams::validate`] refuses, plus
    /// [`HessboostError::InvalidParameter`] for data without labels, with
    /// weights, base margins, groups, label bounds, or feature weights, too
    /// few rows for the validation split (2) or the residualizer (80), and
    /// the errors of training.
    pub fn fit(params: &DiffusionParams, data: &DMatrix) -> Result<Self> {
        fit::fit(params, data)
    }

    /// Draw `n_samples` labels from the model's `p(y | x)` for every row of
    /// `data` (features only; labels are ignored), laid out
    /// `[row][sample][output]`. Deterministic for a given `seed` at any
    /// thread count; see the [module docs](self#sampling).
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] for `n_samples == 0`, a base
    /// margin on `data`, or a sampler that diverges to non-finite values;
    /// [`HessboostError::DimensionMismatch`] when `data`'s feature count
    /// differs from the training data's.
    pub fn sample(&self, data: &DMatrix, n_samples: usize, seed: u64) -> Result<Samples> {
        sample::sample(self, data, n_samples, seed)
    }

    /// The method and its settings.
    pub fn method(&self) -> &Method {
        &self.method
    }

    /// Integration steps [`Self::sample`] takes.
    pub fn n_steps(&self) -> usize {
        self.n_steps
    }

    /// Change the number of integration steps (`> 0`): more steps follow
    /// the learned dynamics more closely at a proportional cost.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] for `0`.
    pub fn set_n_steps(&mut self, n_steps: usize) -> Result<()> {
        positive_count("n_steps", n_steps)?;
        self.n_steps = n_steps;
        Ok(())
    }

    /// Feature columns the model conditions on.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Label columns the model samples.
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// Whether the model was fitted with a [`Residualizer`].
    pub fn is_residualized(&self) -> bool {
        self.residualizer.is_some()
    }

    /// The score/velocity GBDT, on the columns `[y_t (n_outputs), x
    /// (n_features), t]` plus `ln σ(t)` with
    /// [`ScoreConfig::noise_level_feature`], all in the standardized
    /// (and residualized) label space.
    pub fn regressor(&self) -> &BoostedModel {
        &self.regressor
    }

    /// Serialize to the native binary format: a zstd-compressed section
    /// container (magic `HBDM`) embedding the GBDTs' native containers.
    ///
    /// # Errors
    ///
    /// [`HessboostError::ModelFormat`] for a GBDT too large for the native
    /// format; [`HessboostError::Io`] if compression fails.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        format::write(self)
    }

    /// Deserialize a model written by [`Self::to_bytes`].
    ///
    /// # Errors
    ///
    /// [`HessboostError::ModelFormat`] for malformed or inconsistent input,
    /// and for files needing a feature this version lacks.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        format::read(bytes)
    }

    /// Save to a file in the native binary format.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::to_bytes`] and of writing the file.
    pub fn save_binary(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        Ok(std::fs::write(path, self.to_bytes()?)?)
    }

    /// Load a native binary file.
    ///
    /// # Errors
    ///
    /// The errors of reading the file and of [`Self::from_bytes`].
    pub fn load_binary(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Serialize to JSON: the method, the standardization, the residualizer,
    /// and each GBDT in the native JSON format
    /// ([`BoostedModel::to_json`]).
    ///
    /// # Errors
    ///
    /// [`HessboostError::Json`] if serialization fails.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize a model written by [`Self::to_json`].
    ///
    /// # Errors
    ///
    /// [`HessboostError::Json`] for malformed JSON (a missing field
    /// included), [`HessboostError::ModelFormat`] for an inconsistent model.
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    /// Save to a file as JSON.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::to_json`] and of writing the file.
    pub fn save_json(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        Ok(std::fs::write(path, self.to_json()?)?)
    }

    /// Load a JSON file.
    ///
    /// # Errors
    ///
    /// The errors of reading the file and of [`Self::from_json`].
    pub fn load_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    /// Check what sampling and the formats rely on.
    fn validate(&self) -> Result<()> {
        self.method
            .validate()
            .map_err(|e| HessboostError::model_format(e.to_string()))?;
        let bad = |msg: String| Err(HessboostError::model_format(msg));
        if self.n_steps == 0 {
            return bad("n_steps must be at least 1".into());
        }
        let d = self.n_outputs;
        if d == 0 || self.n_features == 0 {
            return bad("a diffusion model needs at least one feature and one output".into());
        }
        if self.target_mean.len() != d
            || self.target_scale.len() != d
            || !self.target_mean.iter().all(|v| v.is_finite())
            || !self.target_scale.iter().all(|v| v.is_finite() && *v > 0.0)
        {
            return bad(format!(
                "the label standardization needs {d} finite means and positive scales"
            ));
        }
        let Some(regressor_features) = d
            .checked_add(self.n_features)
            .and_then(|n| n.checked_add(self.method.time_columns()))
        else {
            return bad("the feature count overflows usize".into());
        };
        check_regressor("regressor", &self.regressor, regressor_features, d)?;
        if let Some(r) = &self.residualizer {
            if r.models.len() < 2 {
                return bad("the residualizer needs at least two fold models".into());
            }
            for model in &r.models {
                check_regressor("residualizer model", model, self.n_features, d)?;
            }
            if r.center.len() != d
                || r.scale.len() != d
                || !r.center.iter().all(|v| v.is_finite())
                || !r.scale.iter().all(|v| v.is_finite() && *v > 0.0)
            {
                return bad(format!(
                    "the residualizer needs {d} finite centers and positive scales"
                ));
            }
        }
        Ok(())
    }
}

/// `model` is a squared-error regressor on `n_features` columns with
/// `n_outputs` outputs.
fn check_regressor(
    what: &str,
    model: &BoostedModel,
    n_features: usize,
    n_outputs: usize,
) -> Result<()> {
    if model.objective() != "reg:squarederror"
        || model.n_features() != n_features
        || model.n_outputs() != n_outputs
    {
        return Err(HessboostError::model_format(format!(
            "the {what} must be a reg:squarederror model with {n_features} features and \
             {n_outputs} outputs, got `{}` with {} and {}",
            model.objective(),
            model.n_features(),
            model.n_outputs()
        )));
    }
    Ok(())
}
