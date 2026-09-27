//! ForestFlow and ForestDiffusion: generating synthetic tabular rows from
//! `p(x)` and imputing missing entries with boosted trees (beyond XGBoost,
//! opt-in).
//!
//! A [`ForestModel`] learns the joint distribution of a table's columns,
//! optionally per class of a label, and then
//!
//! - [`generate`](ForestModel::generate)s new rows (with labels drawn from
//!   the training proportions, or for given labels with
//!   [`generate_for_labels`](ForestModel::generate_for_labels)), and
//! - [`impute`](ForestModel::impute)s the missing entries of rows, any number
//!   of times, keeping the observed entries (diffusion only).
//!
//! This follows Jolicoeur-Martineau, Fatras and Kilian (*Generating and
//! Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted
//! Trees*, AISTATS 2024, <https://arxiv.org/abs/2309.09968>; reference code
//! <https://github.com/SamsungSAILMontreal/ForestDiffusion>) and its
//! defaults.
//!
//! # Method
//!
//! Columns are encoded as the reference does: a categorical column with `K`
//! categories becomes `K - 1` dummy columns (the first category is the
//! baseline), and every encoded column is min–max scaled to `[-1, 1]`. Each
//! row is duplicated [`duplicate_k`](ForestParams::duplicate_k) times with its
//! own standard-normal noise `x₀`, shared by all noise levels.
//!
//! The noise levels are `n_t` evenly spaced times on `[10⁻³, 1]`, and **one
//! GBDT is trained per noise level** (and per class). The paper found this
//! beats a single model with `t` as a feature: a boosted tree with `t` as an
//! input must spend splits locating the noise level, while each per-level
//! model only fits one, much simpler, regression. Without missing values
//! one multi-output GBDT predicts every encoded column
//! ([`MultiStrategy::OneOutputPerTree`](crate::config::MultiStrategy) by
//! default, the reference's XGBoost setting); with missing values each
//! column gets its own GBDT, trained on the rows where it is observed
//! (the reference's `p_in_one = False`).
//!
//! - [`ForestMethod::Flow`] (ForestFlow, the paper's recommended generator):
//!   `x_t = t x₁ + (1 - t) x₀` with the data `x₁` at `t = 1`; the GBDTs
//!   regress the velocity `x₁ - x₀`, and generation integrates it from noise
//!   with `n_t - 1` Euler steps.
//! - [`ForestMethod::Diffusion`] (ForestDiffusion): the VP SDE
//!   ([`Sde::VariancePreserving`], here with
//!   `β` from 0.1 to 8), with the data at `t = 0`; the GBDTs predict the noise
//!   `x₀`, the score is `-prediction / σ(t)`, and generation runs
//!   Euler–Maruyama on the reverse SDE over the `n_t` levels, then a Tweedie
//!   denoising step at `t = 10⁻³`.
//!
//! Decoding inverts the scaling, takes the most likely category of each
//! categorical column (a dummy wins over the baseline above `0.5`), rounds
//! [`ColumnKind::Integer`] columns (binary columns included), and clips every
//! column to its training range.
//!
//! # Imputation
//!
//! [`ForestModel::impute`] (diffusion only, as in the reference) runs the
//! reverse SDE on the missing entries while the observed ones are re-noised
//! to the current level at every step, so every prediction sees a
//! consistent noisy row. With [`Repaint`] (Lugmayr et al., *RePaint*, CVPR
//! 2022; the paper's "REPAINT"), every `jump` steps the missing entries are
//! pushed back up the forward SDE and re-denoised, `resample - 1` times per
//! segment, harmonizing them with the observed entries.
//!
//! # Determinism
//!
//! Training noise is drawn from `rng::Rng` in a fixed order; the GBDTs train
//! in parallel, each deterministically. Generation and imputation draw from
//! SplitMix64 streams keyed by the seed, the row, the step and the column,
//! so the result depends only on the model, the input, and the seed: never
//! on the thread count.
//!
//! # Model size
//!
//! A model holds `max(classes, 1) × n_t` GBDTs, times the number of encoded
//! columns when the training data has missing values (one GBDT per column
//! instead of one multi-output GBDT). Each GBDT has `num_boost_round` trees
//! per encoded column, each up to `2^max_depth` leaves, so the size grows
//! as `classes × n_t × num_boost_round × encoded columns × 2^max_depth`.
//! The `forest_flow` example (2 classes, `n_t = 20`, 5 encoded columns, 100
//! rounds of depth 7: 40 GBDTs, 20,000 trees) saves to 38.6 MB; halve
//! `n_t`, the rounds, or the depth (each step of depth halves the leaves)
//! to shrink it. Generation and imputation cost one batch prediction per
//! level, proportional to the same tree count.
//!
//! # Persistence
//!
//! [`ForestModel::to_bytes`] writes the diffusion container framing with its
//! own magic `HBFF` (see [`super`]'s `HBDM`), embedding every GBDT as a native
//! container; [`ForestModel::to_json`] writes the same content as JSON.
//!
//! # Deviations from the reference
//!
//! - A missing categorical entry stays missing in its dummies (the
//!   reference's `get_dummies` turns it into the baseline category), which
//!   is what imputation needs.
//! - [`impute`](ForestModel::impute) takes raw rows and encodes them itself
//!   (the reference expects already-scaled data).
//! - Not ported: zero-shot classification, extra covariates (`X_covs`),
//!   the data iterator (`n_batch`), and the non-XGBoost regressors.
//!
//! # Example
//!
//! ```
//! use hessboost::diffusion::forest::{ColumnKind, ForestModel, ForestParams};
//! use hessboost::prelude::*;
//!
//! # fn main() -> Result<()> {
//! // Two columns: a continuous one and a binary one that follows it.
//! let n = 60;
//! let x: Vec<f32> = (0..n)
//!     .flat_map(|i| {
//!         let v = i as f32 / n as f32;
//!         [v, if v > 0.5 { 1.0 } else { 0.0 }]
//!     })
//!     .collect();
//! let data = DMatrix::from_dense(&x, n, 2)?;
//!
//! let mut params = ForestParams::default();
//! params.column_kinds = vec![ColumnKind::Continuous, ColumnKind::Integer];
//! params.n_t = 10;
//! params.duplicate_k = 10;
//! params.num_boost_round = 20;
//! let model = ForestModel::fit(&params, &data)?;
//!
//! let synthetic = model.generate(100, 7)?; // [row][column]
//! assert_eq!(synthetic.values().len(), 100 * 2);
//! assert!(synthetic.values().chunks(2).all(|r| r[1] == 0.0 || r[1] == 1.0));
//! # Ok(())
//! # }
//! ```

use std::num::NonZeroUsize;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::Sde;
use super::process::{keyed_normal, try_filled};
use super::{check_regressor, positive_count, validate_regressor_params};
use crate::config::{TrainingParams, TreeMethod};
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::rng::{GOLDEN, Rng, mix64, splitmix64};
use crate::training::train;

mod format;

/// Smallest noise level (the reference's `eps`).
const EPS: f64 = 1e-3;
/// Stream of the training noise.
const NOISE_STREAM: u64 = 0xF0E5_0001;
/// Streams of the samplers: the prior, the reverse-SDE noise, the re-noised
/// observed entries, RePaint's forward steps, and the drawn labels.
const PRIOR_STREAM: u64 = 0xF0E5_0002;
const STEP_STREAM: u64 = 0xF0E5_0003;
const KNOWN_STREAM: u64 = 0xF0E5_0004;
const REPAINT_STREAM: u64 = 0xF0E5_0005;
const LABEL_STREAM: u64 = 0xF0E5_0006;

/// The generative process.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ForestMethod {
    /// ForestFlow: conditional flow matching on the straight path from
    /// noise (`t = 0`) to data (`t = 1`). Generation only.
    Flow,
    /// ForestDiffusion: the VP SDE with `β(t) = β_min + (β_max - β_min) t`
    /// (`0 < beta_min < beta_max`). Generation and imputation.
    Diffusion {
        /// `β(0)`.
        beta_min: f64,
        /// `β(1)`.
        beta_max: f64,
    },
}

impl ForestMethod {
    /// ForestDiffusion with the reference's `β_min = 0.1`, `β_max = 8`.
    pub fn diffusion() -> Self {
        ForestMethod::Diffusion {
            beta_min: 0.1,
            beta_max: 8.0,
        }
    }

    fn sde(self) -> Option<Sde> {
        match self {
            ForestMethod::Flow => None,
            ForestMethod::Diffusion { beta_min, beta_max } => {
                Some(Sde::VariancePreserving { beta_min, beta_max })
            }
        }
    }

    fn validate(self) -> Result<()> {
        if let Some(sde) = self.sde() {
            super::Method::Score(super::ScoreConfig {
                sde,
                ..super::ScoreConfig::treeffuser()
            })
            .validate()
            .map_err(|e| match e {
                HessboostError::InvalidParameter { reason, .. } => {
                    HessboostError::invalid_param("method", reason)
                }
                other => other,
            })?;
        }
        Ok(())
    }
}

/// How a column is encoded and decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ColumnKind {
    /// A real value, clipped to its training range.
    Continuous,
    /// An ordinal or binary value: rounded to the nearest integer, then
    /// clipped.
    Integer,
    /// Categories (any distinct values): dummy-coded against the first
    /// (smallest) category, decoded to the most likely one.
    Categorical,
}

/// RePaint resampling for [`ForestModel::impute`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Repaint {
    /// Passes over each segment (`>= 1`; `1` disables resampling). The
    /// reference's `r`.
    pub resample: usize,
    /// Segment length as a fraction of `n_t` (in `(0, 1]`; `ceil(jump ·
    /// n_t)` steps). The reference's `j`.
    pub jump: f64,
}

impl Default for Repaint {
    /// The reference's `r = 5`, `j = 0.1`.
    fn default() -> Self {
        Repaint {
            resample: 5,
            jump: 0.1,
        }
    }
}

/// Configuration of [`ForestModel::fit`]. [`Default`] is the reference's
/// ForestFlow configuration.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ForestParams {
    /// Flow matching (default) or VP diffusion.
    pub method: ForestMethod,
    /// Number of noise levels, and of GBDTs per class (`>= 2`; 50).
    pub n_t: usize,
    /// Noisy copies of each row (`> 0`; 100).
    pub duplicate_k: usize,
    /// How each column is encoded; empty means every column is
    /// [`ColumnKind::Continuous`], otherwise one entry per column.
    pub column_kinds: Vec<ColumnKind>,
    /// Parameters of every GBDT (objective `reg:squarederror`). Default:
    /// the reference's XGBoost settings, `hist`, depth 7, `eta = 0.3`,
    /// `lambda = 0`.
    pub training: TrainingParams,
    /// Boosting rounds of each GBDT (`> 0`; 100).
    pub num_boost_round: usize,
    /// Seed of the training noise.
    pub seed: u64,
}

impl Default for ForestParams {
    fn default() -> Self {
        ForestParams {
            method: ForestMethod::Flow,
            n_t: 50,
            duplicate_k: 100,
            column_kinds: Vec::new(),
            training: TrainingParams {
                tree_method: TreeMethod::Hist,
                max_depth: NonZeroUsize::new(7),
                eta: 0.3,
                lambda: 0.0,
                ..TrainingParams::default()
            },
            num_boost_round: 100,
            seed: 0,
        }
    }
}

impl ForestParams {
    /// The reference's ForestDiffusion configuration: [`Self::default`] with
    /// [`ForestMethod::diffusion`].
    pub fn diffusion() -> Self {
        ForestParams {
            method: ForestMethod::diffusion(),
            ..ForestParams::default()
        }
    }

    /// Check every setting, [`ForestModel::fit`]'s first step.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] for `n_t < 2`, a zero count,
    /// invalid `β`, or GBDT parameters that fail
    /// [`TrainingParams::validate`], name an objective other than
    /// `reg:squarederror`, or set `process_type` to `update`.
    pub fn validate(&self) -> Result<()> {
        self.method.validate()?;
        if self.n_t < 2 {
            return Err(HessboostError::invalid_param(
                "n_t",
                format!("needs at least 2 noise levels, got {}", self.n_t),
            ));
        }
        positive_count("duplicate_k", self.duplicate_k)?;
        positive_count("num_boost_round", self.num_boost_round)?;
        validate_regressor_params("training", &self.training)
    }
}

/// One encoded column's origin and min–max scaling.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Scale {
    /// Minimum of the encoded column.
    min: f64,
    /// Its range (`1` when constant, as scikit-learn's `MinMaxScaler`).
    range: f64,
}

impl Scale {
    fn forward(self, v: f64) -> f64 {
        (v - self.min) * 2.0 / self.range - 1.0
    }

    fn inverse(self, v: f64) -> f64 {
        (v + 1.0) * self.range / 2.0 + self.min
    }
}

/// An original column: its kind, training range, and categories.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Column {
    kind: ColumnKind,
    min: f64,
    max: f64,
    /// Sorted categories of a categorical column (else empty); the column
    /// encodes as `categories.len() - 1` dummies.
    categories: Vec<f64>,
}

impl Column {
    fn width(&self) -> usize {
        match self.kind {
            ColumnKind::Categorical => self.categories.len().saturating_sub(1),
            ColumnKind::Continuous | ColumnKind::Integer => 1,
        }
    }
}

/// Synthetic rows from [`ForestModel::generate`], row-major
/// `[row][column]`, with their labels for a class-conditional model.
#[derive(Debug, Clone, PartialEq)]
pub struct Synthetic {
    values: Vec<f32>,
    labels: Option<Vec<f32>>,
    n_columns: usize,
}

impl Synthetic {
    /// The rows, `[row][column]`.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Each row's class label (a class-conditional model only).
    pub fn labels(&self) -> Option<&[f32]> {
        self.labels.as_deref()
    }

    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        self.values.len() / self.n_columns
    }

    /// Number of columns.
    pub fn n_columns(&self) -> usize {
        self.n_columns
    }

    /// The rows as a matrix (with the labels attached, if any), e.g. to
    /// train on synthetic data.
    ///
    /// # Errors
    ///
    /// The errors of [`DMatrix::from_dense`].
    pub fn to_dmatrix(&self) -> Result<DMatrix> {
        let m = DMatrix::from_dense(&self.values, self.n_rows(), self.n_columns)?;
        match &self.labels {
            Some(labels) => m.with_labels(labels),
            None => Ok(m),
        }
    }
}

/// A fitted ForestFlow / ForestDiffusion model. See the
/// [module docs](self).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "format::UncheckedForestModel")]
pub struct ForestModel {
    method: ForestMethod,
    n_t: usize,
    columns: Vec<Column>,
    /// Scaling of every encoded column.
    scales: Vec<Scale>,
    /// The class labels (empty for an unconditional model) and their
    /// training proportions.
    classes: Vec<f64>,
    class_probs: Vec<f64>,
    /// One multi-output GBDT per level (`false`) or one per level and
    /// encoded column (`true`).
    per_output: bool,
    /// `[class][level]` or `[class][level][encoded column]`.
    models: Vec<BoostedModel>,
}

impl ForestModel {
    /// Fit a model of the rows of `data` (its features). Labels, if any,
    /// are class labels: one set of GBDTs is trained per distinct label, and
    /// generation reproduces the label proportions. Rows missing every
    /// feature are dropped.
    ///
    /// # Errors
    ///
    /// Everything [`ForestParams::validate`] refuses, plus
    /// [`HessboostError::InvalidParameter`] for weights, base margins,
    /// groups, label bounds, feature weights, a label matrix,
    /// `column_kinds` of the wrong length, a categorical column with
    /// non-integral or out-of-range values... (see [`ColumnKind`]), a column
    /// with no observed value (overall or in a class, with missing values),
    /// and the errors of training: each level's GBDT regresses a label
    /// matrix of every column, which `booster = boulevard` and
    /// `booster = ebm` refuse for a table of several columns.
    pub fn fit(params: &ForestParams, data: &DMatrix) -> Result<Self> {
        params.validate()?;
        refuse_metadata(data)?;
        let p = data.n_cols();
        let kinds = if params.column_kinds.is_empty() {
            vec![ColumnKind::Continuous; p]
        } else if params.column_kinds.len() == p {
            params.column_kinds.clone()
        } else {
            return Err(HessboostError::dimension_mismatch(
                "column_kinds",
                p,
                params.column_kinds.len(),
            ));
        };

        // Rows with at least one observed value, densely with NaN.
        let raw = super::fit::dense_features(data);
        let keep: Vec<usize> = (0..data.n_rows())
            .filter(|&r| raw[r * p..(r + 1) * p].iter().any(|v| !v.is_nan()))
            .collect();
        if keep.is_empty() {
            return Err(HessboostError::invalid_param(
                "data",
                "every row is entirely missing",
            ));
        }
        let rows: Vec<f64> = keep
            .iter()
            .flat_map(|&r| raw[r * p..(r + 1) * p].iter().map(|&v| f64::from(v)))
            .collect();
        let n = keep.len();

        let columns = describe_columns(&rows, p, &kinds)?;
        let c: usize = columns.iter().map(Column::width).sum();
        if c == 0 {
            return Err(HessboostError::invalid_param(
                "data",
                "every column is a single-category categorical: nothing to model",
            ));
        }
        let mut encoded = try_filled(n * c, 0.0, "data")?;
        for (row, out) in rows.chunks_exact(p).zip(encoded.chunks_exact_mut(c)) {
            encode_row(&columns, row, out);
        }
        let scales: Vec<Scale> = (0..c)
            .map(|j| {
                let (lo, hi) = encoded
                    .chunks_exact(c)
                    .map(|r| r[j])
                    .filter(|v| !v.is_nan())
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                        (lo.min(v), hi.max(v))
                    });
                let range = hi - lo;
                Scale {
                    min: lo,
                    range: if range > 0.0 { range } else { 1.0 },
                }
            })
            .collect();
        for row in encoded.chunks_exact_mut(c) {
            for (v, s) in row.iter_mut().zip(&scales) {
                *v = s.forward(*v);
            }
        }
        let per_output = encoded.iter().any(|v| v.is_nan());

        // Classes, as sorted distinct labels.
        let labels: Option<Vec<f64>> = data
            .labels()
            .map(|l| keep.iter().map(|&r| f64::from(l[r])).collect());
        let (classes, class_of) = match &labels {
            Some(l) => {
                let mut classes = l.clone();
                classes.sort_by(f64::total_cmp);
                classes.dedup();
                let class_of: Vec<usize> = l
                    .iter()
                    .map(|v| classes.partition_point(|c| c < v))
                    .collect();
                (classes, class_of)
            }
            None => (Vec::new(), vec![0; n]),
        };
        let n_classes = classes.len().max(1);
        let class_probs = if classes.is_empty() {
            Vec::new()
        } else {
            let mut counts = vec![0.0; n_classes];
            for &k in &class_of {
                counts[k] += 1.0;
            }
            counts.iter().map(|&k| k / n as f64).collect()
        };

        // Noise shared by every level, per duplicated row.
        let k = params.duplicate_k;
        let n_dup = n
            .checked_mul(k)
            .filter(|m| m.checked_mul(c).is_some())
            .ok_or_else(|| {
                HessboostError::invalid_param(
                    "duplicate_k",
                    "the training set size overflows usize",
                )
            })?;
        let mut noise = try_filled(n_dup * c, 0.0, "duplicate_k")?;
        let mut rng = Rng::new(splitmix64(params.seed ^ NOISE_STREAM));
        let mut normal = super::process::Normal::default();
        for v in &mut noise {
            *v = normal.draw(&mut rng);
        }

        let levels: Vec<f64> = (0..params.n_t)
            .map(|i| EPS + (1.0 - EPS) * i as f64 / (params.n_t - 1) as f64)
            .collect();
        let outputs = if per_output { c } else { 1 };
        let jobs: Vec<(usize, usize, usize)> = (0..n_classes)
            .flat_map(|class| {
                (0..params.n_t).flat_map(move |level| (0..outputs).map(move |o| (class, level, o)))
            })
            .collect();
        let set = LevelSet {
            method: params.method,
            data: &encoded,
            noise: &noise,
            class_of: &class_of,
            n,
            c,
            k,
        };
        let models = jobs
            .into_par_iter()
            .map(|(class, level, output)| {
                let dtrain = set.build(class, levels[level], per_output.then_some(output))?;
                train(&params.training, &dtrain, params.num_boost_round)
            })
            .collect::<Result<Vec<_>>>()?;

        let model = ForestModel {
            method: params.method,
            n_t: params.n_t,
            columns,
            scales,
            classes,
            class_probs,
            per_output,
            models,
        };
        model.validate()?;
        Ok(model)
    }

    /// Generate `n_rows` synthetic rows, `[row][column]`. A class-conditional
    /// model draws each row's label from the training proportions.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] for `n_rows == 0`, a request too
    /// large to allocate, or a sampler that diverges.
    pub fn generate(&self, n_rows: usize, seed: u64) -> Result<Synthetic> {
        positive_count("n_rows", n_rows)?;
        let mut class_of: Vec<usize> = try_filled(n_rows, 0, "n_rows")?;
        if !self.classes.is_empty() {
            let key = splitmix64(seed ^ LABEL_STREAM);
            for (r, class) in class_of.iter_mut().enumerate() {
                let bits = mix64(key.wrapping_add((r as u64 + 1).wrapping_mul(GOLDEN)));
                let u = (bits >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
                let mut acc = 0.0;
                *class = self
                    .class_probs
                    .iter()
                    .position(|&p| {
                        acc += p;
                        u < acc
                    })
                    .unwrap_or(self.classes.len() - 1);
            }
        }
        self.generate_classes(&class_of, seed)
    }

    /// Generate one row per entry of `labels` from that class's model,
    /// `[row][column]`.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] for an unconditional model, an
    /// empty `labels`, or a label the model was not trained on; the errors
    /// of [`Self::generate`].
    pub fn generate_for_labels(&self, labels: &[f32], seed: u64) -> Result<Synthetic> {
        positive_count("labels", labels.len())?;
        let class_of = self.class_indices(labels)?;
        self.generate_classes(&class_of, seed)
    }

    /// Impute the missing entries of every row of `data`, `n_imputations`
    /// times, laid out `[imputation][row][column]`; observed entries are
    /// kept (rounded and clipped like generated values). A class-conditional
    /// model reads each row's class from `data`'s labels.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] for a flow model (the reference
    /// imputes with diffusion only), `n_imputations == 0`, an invalid
    /// [`Repaint`], metadata [`Self::fit`] refuses (weights, base margins,
    /// groups, label bounds, feature weights, a label matrix), missing or
    /// unknown labels on a class-conditional model, a categorical value the
    /// model has not seen, or a sampler that diverges;
    /// [`HessboostError::DimensionMismatch`] for the wrong column count.
    pub fn impute(
        &self,
        data: &DMatrix,
        n_imputations: usize,
        repaint: Option<Repaint>,
        seed: u64,
    ) -> Result<Vec<f32>> {
        let Some(sde) = self.method.sde() else {
            return Err(HessboostError::invalid_param(
                "method",
                "imputation needs ForestMethod::Diffusion (flow matching cannot condition \
                 on the observed entries)",
            ));
        };
        positive_count("n_imputations", n_imputations)?;
        let (resample, jump) = match repaint {
            None => (1, self.n_t),
            Some(r) => {
                positive_count("repaint.resample", r.resample)?;
                if !(r.jump.is_finite() && r.jump > 0.0 && r.jump <= 1.0) {
                    return Err(HessboostError::invalid_param(
                        "repaint.jump",
                        format!("must be in (0, 1], got {}", r.jump),
                    ));
                }
                (
                    r.resample,
                    ((r.jump * self.n_t as f64).ceil() as usize).max(1),
                )
            }
        };
        let p = self.columns.len();
        if data.n_cols() != p {
            return Err(HessboostError::dimension_mismatch(
                "imputation column count",
                p,
                data.n_cols(),
            ));
        }
        refuse_metadata(data)?;
        let class_of = if self.classes.is_empty() {
            vec![0; data.n_rows()]
        } else {
            let labels = data.labels().ok_or_else(|| {
                HessboostError::invalid_param(
                    "data",
                    "a class-conditional model imputes rows of known class: attach labels",
                )
            })?;
            self.class_indices(labels)?
        };
        let raw = super::fit::dense_features(data);
        let c = self.scales.len();
        let mut known = try_filled(data.n_rows() * c, 0.0, "data")?;
        for (row, out) in raw.chunks_exact(p).zip(known.chunks_exact_mut(c)) {
            let row: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
            for (j, (column, &v)) in self.columns.iter().zip(&row).enumerate() {
                if column.kind == ColumnKind::Categorical
                    && !v.is_nan()
                    && column
                        .categories
                        .binary_search_by(|x| x.total_cmp(&v))
                        .is_err()
                {
                    return Err(HessboostError::invalid_param(
                        "data",
                        format!("column {j} has category {v}, unseen in training"),
                    ));
                }
            }
            encode_row(&self.columns, &row, out);
            for (v, s) in out.iter_mut().zip(&self.scales) {
                *v = s.forward(*v);
            }
        }
        let n_rows = data.n_rows();
        let total = n_imputations
            .checked_mul(n_rows)
            .and_then(|m| m.checked_mul(p))
            .unwrap_or(usize::MAX);
        let mut out = try_filled(total, 0.0f32, "n_imputations")?;
        for (i, dest) in out.chunks_exact_mut(n_rows * p).enumerate() {
            let key = splitmix64(seed ^ splitmix64(i as u64));
            let sampler = Sampler {
                model: self,
                class_of: &class_of,
                key,
            };
            let x = sampler.reverse_sde(sde, Some(&known), (resample, jump))?;
            self.decode(&x, dest)?;
        }
        Ok(out)
    }

    /// The method.
    pub fn method(&self) -> ForestMethod {
        self.method
    }

    /// Number of noise levels.
    pub fn n_t(&self) -> usize {
        self.n_t
    }

    /// Number of (original) columns.
    pub fn n_columns(&self) -> usize {
        self.columns.len()
    }

    /// The class labels of a class-conditional model (else empty), sorted.
    pub fn classes(&self) -> &[f64] {
        &self.classes
    }

    /// Serialize to the binary format (magic `HBFF`).
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
    /// [`HessboostError::ModelFormat`] for malformed or inconsistent input.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        format::read(bytes)
    }

    /// Serialize to JSON, with every GBDT in the native JSON format.
    ///
    /// # Errors
    ///
    /// [`HessboostError::Json`] if serialization fails.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Deserialize a model written by [`Self::to_json`].
    ///
    /// # Errors
    ///
    /// [`HessboostError::Json`] for malformed JSON,
    /// [`HessboostError::ModelFormat`] for an inconsistent model.
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    /// Save to a file in the binary format.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::to_bytes`] and of writing the file.
    pub fn save_binary(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        Ok(std::fs::write(path, self.to_bytes()?)?)
    }

    /// Load a binary file.
    ///
    /// # Errors
    ///
    /// The errors of reading the file and of [`Self::from_bytes`].
    pub fn load_binary(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
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

    /// Every GBDT, in `[class][level]` order, or `[class][level][encoded
    /// column]` for a model fitted with missing values; see the
    /// [module docs](self#model-size) for their count. Each takes the
    /// scaled encoded columns as features.
    pub fn gbdts(&self) -> &[BoostedModel] {
        &self.models
    }

    /// Indices into [`Self::classes`] of `labels`.
    fn class_indices(&self, labels: &[f32]) -> Result<Vec<usize>> {
        if self.classes.is_empty() {
            return Err(HessboostError::invalid_param(
                "labels",
                "the model was fitted without class labels",
            ));
        }
        labels
            .iter()
            .map(|&l| {
                let l = f64::from(l);
                self.classes
                    .binary_search_by(|c| c.total_cmp(&l))
                    .map_err(|_| {
                        HessboostError::invalid_param(
                            "labels",
                            format!("label {l} was not a training class"),
                        )
                    })
            })
            .collect()
    }

    fn generate_classes(&self, class_of: &[usize], seed: u64) -> Result<Synthetic> {
        let n_rows = class_of.len();
        let p = self.columns.len();
        let total = n_rows.saturating_mul(p);
        let mut values = try_filled(total, 0.0f32, "n_rows")?;
        let sampler = Sampler {
            model: self,
            class_of,
            key: seed,
        };
        let x = match self.method.sde() {
            None => sampler.euler_flow()?,
            Some(sde) => sampler.reverse_sde(sde, None, (1, self.n_t))?,
        };
        self.decode(&x, &mut values)?;
        let labels = (!self.classes.is_empty())
            .then(|| class_of.iter().map(|&k| self.classes[k] as f32).collect());
        Ok(Synthetic {
            values,
            labels,
            n_columns: p,
        })
    }

    /// Scaled encoded rows `x` (`[row][c]`) back to columns in `out`
    /// (`[row][p]`).
    fn decode(&self, x: &[f64], out: &mut [f32]) -> Result<()> {
        let (c, p) = (self.scales.len(), self.columns.len());
        for (row, dest) in x.chunks_exact(c).zip(out.chunks_exact_mut(p)) {
            let mut at = 0;
            for (column, d) in self.columns.iter().zip(dest.iter_mut()) {
                let width = column.width();
                let value = match column.kind {
                    ColumnKind::Categorical => {
                        // The baseline competes at 0.5; the first maximum wins.
                        let mut best = (0, 0.5);
                        for m in 0..width {
                            let v = self.scales[at + m].inverse(row[at + m]);
                            if v > best.1 {
                                best = (m + 1, v);
                            }
                        }
                        column.categories.get(best.0).copied().unwrap_or(column.min)
                    }
                    ColumnKind::Integer => self.scales[at].inverse(row[at]).round_ties_even(),
                    ColumnKind::Continuous => self.scales[at].inverse(row[at]),
                };
                at += width;
                let v = value.max(column.min).min(column.max) as f32;
                if !v.is_finite() {
                    return Err(diverged());
                }
                *d = v;
            }
        }
        Ok(())
    }

    /// The GBDTs of `class` at noise level `level`.
    fn level_models(&self, class: usize, level: usize) -> &[BoostedModel] {
        let per = if self.per_output {
            self.scales.len()
        } else {
            1
        };
        let start = (class * self.n_t + level) * per;
        &self.models[start..start + per]
    }

    /// Check what sampling and the formats rely on.
    fn validate(&self) -> Result<()> {
        let bad = |msg: String| Err(HessboostError::model_format(msg));
        self.method
            .validate()
            .map_err(|e| HessboostError::model_format(e.to_string()))?;
        if self.n_t < 2 || self.columns.is_empty() {
            return bad("a forest model needs 2 noise levels and a column".into());
        }
        let c: usize = self.columns.iter().map(Column::width).sum();
        if c == 0 || self.scales.len() != c {
            return bad(format!(
                "{} scales for {c} encoded columns",
                self.scales.len()
            ));
        }
        for column in &self.columns {
            let sorted = column.categories.windows(2).all(|w| w[0] < w[1]);
            // Ranges and categories come from `f32` data and decode back
            // to `f32`: anything else is not a model fit here produced.
            let valid = f32_exact(column.min)
                && f32_exact(column.max)
                && column.min <= column.max
                && column.categories.iter().all(|&v| f32_exact(v))
                && sorted
                && (column.kind == ColumnKind::Categorical) != column.categories.is_empty();
            if !valid {
                return bad("a column's range or categories are invalid".into());
            }
        }
        if !self
            .scales
            .iter()
            .all(|s| s.min.is_finite() && s.range.is_finite() && s.range > 0.0)
        {
            return bad("an encoded column's scale is invalid".into());
        }
        let classes_sorted = self.classes.windows(2).all(|w| w[0] < w[1]);
        if !classes_sorted
            || self.classes.len() != self.class_probs.len()
            || !self.classes.iter().all(|&v| f32_exact(v))
            || !self.class_probs.iter().all(|p| p.is_finite() && *p >= 0.0)
        {
            return bad("the classes are invalid".into());
        }
        let per = if self.per_output { c } else { 1 };
        let expected = self
            .classes
            .len()
            .max(1)
            .checked_mul(self.n_t)
            .and_then(|m| m.checked_mul(per));
        if expected != Some(self.models.len()) {
            return bad(format!(
                "{} GBDTs, expected one per class, level{}",
                self.models.len(),
                if self.per_output { " and column" } else { "" }
            ));
        }
        let outputs = if self.per_output { 1 } else { c };
        for model in &self.models {
            check_regressor("forest GBDT", model, c, outputs)?;
        }
        Ok(())
    }
}

/// Refuse the metadata a forest model cannot honor.
fn refuse_metadata(data: &DMatrix) -> Result<()> {
    let refuse = |what: &str| {
        Err(HessboostError::invalid_param(
            "data",
            format!("forest models do not support {what}"),
        ))
    };
    if data.weights().is_some() {
        return refuse("instance weights");
    }
    if data.base_margin().is_some() {
        return refuse("base margins");
    }
    if data.group().is_some() {
        return refuse("ranking groups");
    }
    if data.label_lower_bound().is_some() || data.label_upper_bound().is_some() {
        return refuse("label bounds");
    }
    if data.feature_weights().is_some() {
        return refuse("feature weights");
    }
    if data.labels().is_some() && data.n_targets() != 1 {
        return refuse("label matrices (labels are class labels)");
    }
    Ok(())
}

/// Ranges and categories of the `[row][p]` values.
fn describe_columns(rows: &[f64], p: usize, kinds: &[ColumnKind]) -> Result<Vec<Column>> {
    kinds
        .iter()
        .enumerate()
        .map(|(j, &kind)| {
            let mut observed: Vec<f64> = rows
                .chunks_exact(p)
                .map(|r| r[j])
                .filter(|v| !v.is_nan())
                .collect();
            if observed.is_empty() {
                return Err(HessboostError::invalid_param(
                    "data",
                    format!("column {j} has no observed value"),
                ));
            }
            observed.sort_by(f64::total_cmp);
            let (min, max) = (observed[0], observed[observed.len() - 1]);
            let categories = if kind == ColumnKind::Categorical {
                observed.dedup();
                observed
            } else {
                Vec::new()
            };
            Ok(Column {
                kind,
                min,
                max,
                categories,
            })
        })
        .collect()
}

/// Encode one row (`p` values, NaN missing) into `out` (`c` values).
fn encode_row(columns: &[Column], row: &[f64], out: &mut [f64]) {
    let mut at = 0;
    for (column, &v) in columns.iter().zip(row) {
        match column.kind {
            ColumnKind::Categorical => {
                for (m, category) in column.categories.iter().skip(1).enumerate() {
                    out[at + m] = if v.is_nan() {
                        f64::NAN
                    } else {
                        f64::from(u8::from(v == *category))
                    };
                }
            }
            ColumnKind::Continuous | ColumnKind::Integer => out[at] = v,
        }
        at += column.width();
    }
}

/// The inputs of one noise level's training set.
struct LevelSet<'a> {
    method: ForestMethod,
    /// Scaled encoded rows `x₁`, `[row][c]`, NaN missing.
    data: &'a [f64],
    /// `x₀` of every duplicated row, `[copy][row][c]`.
    noise: &'a [f64],
    class_of: &'a [usize],
    n: usize,
    c: usize,
    k: usize,
}

impl LevelSet<'_> {
    /// The training matrix of `class` at time `t`: features `x_t`, labels
    /// the flow's velocity `x₁ - x₀` or the diffusion's noise `x₀`, for
    /// every column or just `output` (on the rows where it is observed).
    fn build(&self, class: usize, t: f64, output: Option<usize>) -> Result<DMatrix> {
        let c = self.c;
        let (alpha, std) = match self.method.sde() {
            None => (t, 1.0 - t),
            Some(sde) => sde.marginal(t),
        };
        let width = if output.is_some() { 1 } else { c };
        let mut features = Vec::new();
        let mut labels = Vec::new();
        for copy in 0..self.k {
            for row in (0..self.n).filter(|&r| self.class_of[r] == class) {
                let x1 = &self.data[row * c..(row + 1) * c];
                if output.is_some_and(|o| x1[o].is_nan()) {
                    continue;
                }
                let x0 = &self.noise[(copy * self.n + row) * c..(copy * self.n + row + 1) * c];
                for (&a, &z) in x1.iter().zip(x0) {
                    features.push((alpha * a + std * z) as f32);
                }
                let target = |j: usize| match self.method {
                    ForestMethod::Flow => x1[j] - x0[j],
                    ForestMethod::Diffusion { .. } => x0[j],
                };
                match output {
                    Some(o) => labels.push(target(o) as f32),
                    None => labels.extend((0..c).map(|j| target(j) as f32)),
                }
            }
        }
        let n_rows = labels.len() / width;
        if n_rows == 0 {
            return Err(HessboostError::invalid_param(
                "data",
                format!(
                    "class {class} has no observed value in encoded column {}",
                    output.unwrap_or(0)
                ),
            ));
        }
        DMatrix::from_dense_vec(features, n_rows, c)?.with_label_matrix(&labels, width)
    }
}

/// Generation and imputation state: the model, each row's class, and the
/// seed key.
struct Sampler<'a> {
    model: &'a ForestModel,
    class_of: &'a [usize],
    key: u64,
}

impl Sampler<'_> {
    /// Draw `counter` of `stream` for encoded entry `(row, j)`.
    fn noise(&self, stream: u64, row: usize, counter: u64) -> f64 {
        keyed_normal(
            splitmix64(splitmix64(self.key ^ stream) ^ row as u64),
            counter,
        )
    }

    /// The prior `N(0, I)` over every encoded entry.
    fn prior(&self) -> Result<Vec<f64>> {
        let c = self.model.scales.len();
        let mut x = try_filled(self.class_of.len() * c, 0.0, "n_rows")?;
        for (row, values) in x.chunks_exact_mut(c).enumerate() {
            for (j, v) in values.iter_mut().enumerate() {
                *v = self.noise(PRIOR_STREAM, row, j as u64);
            }
        }
        Ok(x)
    }

    /// The level-`t` GBDTs' predictions at `x` (`[row][c]`).
    fn predict(&self, x: &[f64], t: f64) -> Result<Vec<f64>> {
        let model = self.model;
        let c = model.scales.len();
        let level = ((t * (model.n_t - 1) as f64).round() as usize).min(model.n_t - 1);
        let mut out = vec![0.0; x.len()];
        for class in 0..model.classes.len().max(1) {
            let rows: Vec<usize> = (0..self.class_of.len())
                .filter(|&r| self.class_of[r] == class)
                .collect();
            if rows.is_empty() {
                continue;
            }
            let values: Vec<f32> = rows
                .iter()
                .flat_map(|&r| x[r * c..(r + 1) * c].iter().map(|&v| v as f32))
                .collect();
            let input = DMatrix::from_dense_vec(values, rows.len(), c)?;
            let models = model.level_models(class, level);
            for (m, gbdt) in models.iter().enumerate() {
                let pred = gbdt.predict_margin(&input)?;
                for (&r, p) in rows.iter().zip(pred.rows()) {
                    if model.per_output {
                        out[r * c + m] = f64::from(p[0]);
                    } else {
                        for (o, &v) in out[r * c..(r + 1) * c].iter_mut().zip(p) {
                            *o = f64::from(v);
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// ForestFlow: Euler steps of the learned velocity from `t = 0` to `1`.
    fn euler_flow(&self) -> Result<Vec<f64>> {
        let n_t = self.model.n_t;
        let h = 1.0 / (n_t - 1) as f64;
        let mut x = self.prior()?;
        for step in 0..n_t - 1 {
            let v = self.predict(&x, step as f64 * h)?;
            for (s, &d) in x.iter_mut().zip(&v) {
                *s += h * d;
            }
            check_finite(&x)?;
        }
        Ok(x)
    }

    /// The score at `x` and time `t`; with `known` (imputation), the
    /// observed entries are first re-noised to level `t` with the draws of
    /// evaluation `eval`.
    fn score(
        &self,
        sde: Sde,
        x: &mut [f64],
        known: Option<&[f64]>,
        t: f64,
        eval: u64,
    ) -> Result<Vec<f64>> {
        let (alpha, std) = sde.marginal(t);
        if let Some(known) = known {
            let c = self.model.scales.len();
            for (row, (state, obs)) in x.chunks_exact_mut(c).zip(known.chunks_exact(c)).enumerate()
            {
                for (j, (s, &o)) in state.iter_mut().zip(obs).enumerate() {
                    if !o.is_nan() {
                        *s = alpha * o
                            + std * self.noise(KNOWN_STREAM, row, eval * c as u64 + j as u64);
                    }
                }
            }
        }
        let mut out = self.predict(x, t)?;
        for v in &mut out {
            *v = -*v / std;
        }
        Ok(out)
    }

    /// ForestDiffusion: Euler–Maruyama on the reverse VP SDE over the `n_t`
    /// levels from `t = 1` to `10⁻³`, with RePaint's `(resample, jump)` and a
    /// final Tweedie denoising step.
    fn reverse_sde(
        &self,
        sde: Sde,
        known: Option<&[f64]>,
        (resample, jump): (usize, usize),
    ) -> Result<Vec<f64>> {
        let n_t = self.model.n_t;
        let c = self.model.scales.len() as u64;
        let times: Vec<f64> = (0..n_t)
            .map(|i| 1.0 - (1.0 - EPS) * i as f64 / (n_t - 1) as f64)
            .collect();
        let step = |i: usize| times[i] - times.get(i + 1).copied().unwrap_or(0.0);
        let mut x = self.prior()?;
        let mut eval = 0u64;
        let (mut i, mut passes) = (0usize, 0usize);
        while i < n_t - 1 {
            let t = times[i];
            let h = step(i);
            let score = self.score(sde, &mut x, known, t, eval)?;
            let (drift, g2) = sde.drift_diffusion(t);
            let noise_scale = (g2 * h).sqrt();
            for (e, (v, &s)) in x.iter_mut().zip(&score).enumerate() {
                let (row, j) = (e / c as usize, e as u64 % c);
                let reverse = drift * *v - g2 * s;
                *v = *v - reverse * h + noise_scale * self.noise(STEP_STREAM, row, eval * c + j);
            }
            check_finite(&x)?;
            eval += 1;
            if (i + 1).is_multiple_of(jump) && passes + 1 < resample && i + 1 >= jump {
                // Back up the forward SDE over the last `jump` steps.
                let span: f64 = (i + 1 - jump..=i).map(step).sum();
                let (drift, g2) = sde.drift_diffusion(t);
                let scale = (g2 * span).sqrt();
                for (e, v) in x.iter_mut().enumerate() {
                    let (row, j) = (e / c as usize, e as u64 % c);
                    *v += drift * *v * span + scale * self.noise(REPAINT_STREAM, row, eval * c + j);
                }
                passes += 1;
                i = i + 1 - jump;
                continue;
            }
            if (i + 1).is_multiple_of(jump) {
                passes = 0;
            }
            i += 1;
        }
        let (_, std) = sde.marginal(EPS);
        let score = self.score(sde, &mut x, known, EPS, eval)?;
        for (v, &s) in x.iter_mut().zip(&score) {
            *v += std * std * s;
        }
        if let Some(known) = known {
            for (v, &o) in x.iter_mut().zip(known) {
                if !o.is_nan() {
                    *v = o;
                }
            }
        }
        check_finite(&x)?;
        Ok(x)
    }
}

fn check_finite(x: &[f64]) -> Result<()> {
    if x.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(diverged())
    }
}

fn diverged() -> HessboostError {
    HessboostError::invalid_param("n_t", "sampling diverged to non-finite values")
}

/// `v` is a finite `f32` value stored in `f64` (every label, category and
/// range a fit records is one).
fn f32_exact(v: f64) -> bool {
    let narrowed = v as f32;
    narrowed.is_finite() && f64::from(narrowed) == v
}
