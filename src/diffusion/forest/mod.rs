//! ForestFlow and ForestDiffusion: generating synthetic tabular rows from
//! `p(x)` and imputing missing entries with boosted trees (beyond XGBoost,
//! opt-in).
//!
//! A [`ForestModel`] learns the joint distribution of a table's columns,
//! optionally per class of a label, and then
//!
//! - [`sample`](ForestModel::sample)s new rows (with labels drawn from
//!   the training proportions, or for given labels with
//!   [`sample_for_labels`](ForestModel::sample_for_labels)), and
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
//! use std::num::NonZeroUsize;
//!
//! use hessboost::diffusion::forest::{ColumnKind, ForestModel, ForestParams, NoiseLevels};
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
//! params.column_kinds = Some(vec![ColumnKind::Continuous, ColumnKind::Integer]);
//! params.n_t = NoiseLevels::new(10).unwrap();
//! params.duplicate_k = NonZeroUsize::new(10).unwrap();
//! params.num_boost_round = NonZeroUsize::new(20).unwrap();
//! let model = ForestModel::fit(&params, &data)?;
//!
//! let synthetic = model.sample(100, 7)?; // [row][column]
//! assert_eq!(synthetic.n_rows(), 100);
//! assert!(synthetic.rows().all(|r| r[1] == 0.0 || r[1] == 1.0));
//! # Ok(())
//! # }
//! ```

use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

use super::Sde;
use super::process::{keyed_normal, try_filled};
use super::{check_regressor, validate_regressor_params};
use crate::config::{TrainingParams, TreeMethod};
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::rng::{Rng, keyed_unit, splitmix64};
use crate::rng::{GOLDEN, mix64, splitmix64};
use encoding::encode_row;

mod encoding;
mod fit;
mod format;

/// Smallest noise level (the reference's `eps`).
const EPS: f64 = 1e-3;

/// The time of noise level `level` of `n_t`, evenly spaced on `[EPS, 1]`
/// (the reference's `t_levels`). Training and sampling both index the
/// levels, so every step evaluates the GBDTs trained at its own time.
fn level_time(n_t: usize, level: usize) -> f64 {
    EPS + (1.0 - EPS) * level as f64 / (n_t - 1) as f64
}
/// Stream of the training noise.
const NOISE_STREAM: u64 = 0xF0E5_0001;
/// Streams of the samplers: the prior, the reverse-SDE noise, the re-noised
/// observed entries, RePaint's forward steps, and the drawn labels.
const PRIOR_STREAM: u64 = 0xF0E5_0002;
const STEP_STREAM: u64 = 0xF0E5_0003;
const KNOWN_STREAM: u64 = 0xF0E5_0004;
const REPAINT_STREAM: u64 = 0xF0E5_0005;
const LABEL_STREAM: u64 = 0xF0E5_0006;

/// The number of noise levels of a forest model: at least 2 (the data end
/// and the noise end). Each level has its own GBDTs; generation takes one
/// step per pair of adjacent levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "usize", into = "usize")]
pub struct NoiseLevels(usize);

impl NoiseLevels {
    /// `n` levels, or `None` below 2.
    pub const fn new(n: usize) -> Option<Self> {
        if n >= 2 { Some(NoiseLevels(n)) } else { None }
    }

    /// The number of levels.
    pub const fn get(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for NoiseLevels {
    type Error = HessboostError;

    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] below 2.
    fn try_from(n: usize) -> Result<Self> {
        NoiseLevels::new(n).ok_or_else(|| {
            HessboostError::invalid_param("n_t", format!("needs at least 2 noise levels, got {n}"))
        })
    }
}

impl From<NoiseLevels> for usize {
    fn from(n: NoiseLevels) -> usize {
        n.0
    }
}

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
    pub fn forest_diffusion() -> Self {
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

/// Settings of one [`ForestModel::impute`] call.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[non_exhaustive]
pub struct ImputeOptions {
    /// Seed of the draws: the same seed and data give the same imputations
    /// at any thread count.
    pub seed: u64,
    /// RePaint resampling; `None` runs the reverse SDE once.
    pub repaint: Option<Repaint>,
}

impl ImputeOptions {
    /// Options imputing with `seed` and no resampling.
    pub fn seeded(seed: u64) -> Self {
        ImputeOptions {
            seed,
            repaint: None,
        }
    }

    /// These options with RePaint resampling `repaint`.
    #[must_use]
    pub fn with_repaint(mut self, repaint: Repaint) -> Self {
        self.repaint = Some(repaint);
        self
    }
}

/// RePaint resampling for [`ForestModel::impute`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Repaint {
    /// Passes over each segment (`1` disables resampling). The reference's
    /// `r`.
    pub resample: NonZeroUsize,
    /// Segment length as a fraction of `n_t` (in `(0, 1]`; `ceil(jump ·
    /// n_t)` steps). The reference's `j`.
    pub jump: f64,
}

impl Default for Repaint {
    /// The reference's `r = 5`, `j = 0.1`.
    fn default() -> Self {
        Repaint {
            resample: const { NonZeroUsize::new(5).unwrap() },
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
    /// Number of noise levels, and of GBDTs per class (50).
    pub n_t: NoiseLevels,
    /// Noisy copies of each row (100).
    pub duplicate_k: NonZeroUsize,
    /// How each column is encoded, one entry per column; `None` makes every
    /// column [`ColumnKind::Continuous`].
    pub column_kinds: Option<Vec<ColumnKind>>,
    /// Parameters of every GBDT (objective `reg:squarederror` with
    /// `scale_pos_weight = 1`). Default:
    /// the reference's XGBoost settings, `hist`, depth 7, `eta = 0.3`,
    /// `lambda = 0`.
    pub training: TrainingParams,
    /// Boosting rounds of each GBDT (100).
    pub num_boost_round: NonZeroUsize,
    /// Seed of the training noise.
    pub seed: u64,
}

impl Default for ForestParams {
    fn default() -> Self {
        ForestParams {
            method: ForestMethod::Flow,
            n_t: const { NoiseLevels::new(50).unwrap() },
            duplicate_k: const { NonZeroUsize::new(100).unwrap() },
            column_kinds: None,
            training: TrainingParams {
                tree_method: TreeMethod::Hist,
                max_depth: NonZeroUsize::new(7),
                eta: 0.3,
                lambda: 0.0,
                ..TrainingParams::default()
            },
            num_boost_round: const { NonZeroUsize::new(100).unwrap() },
            seed: 0,
        }
    }
}

impl ForestParams {
    /// The reference's ForestFlow configuration (the [`Default`]).
    pub fn forest_flow() -> Self {
        ForestParams::default()
    }

    /// The reference's ForestDiffusion configuration: [`Self::forest_flow`]
    /// with [`ForestMethod::forest_diffusion`].
    pub fn forest_diffusion() -> Self {
        ForestParams {
            method: ForestMethod::forest_diffusion(),
            ..ForestParams::default()
        }
    }

    /// Check every setting, [`ForestModel::fit`]'s first step.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] for invalid `β`, or
    /// GBDT parameters that fail [`TrainingParams::validate`], name an
    /// objective other than `reg:squarederror` with `scale_pos_weight = 1`,
    /// or set `process_type` to `update`.
    pub fn validate(&self) -> Result<()> {
        self.method.validate()?;
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

/// How a model's GBDTs cover the encoded columns at each class and level.
/// Stored as `per_output`: `false` (format `0`) or `true` (`1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "bool", into = "bool")]
enum OutputLayout {
    /// One multi-output GBDT per level, predicting every encoded column.
    Joint,
    /// One single-output GBDT per level and encoded column, each trained on
    /// the rows observing its column (a table with missing values).
    PerColumn,
}

impl OutputLayout {
    /// The GBDTs of one class and level.
    fn gbdts_per_level(self, c: usize) -> usize {
        match self {
            OutputLayout::Joint => 1,
            OutputLayout::PerColumn => c,
        }
    }

    /// The outputs of each GBDT.
    fn gbdt_outputs(self, c: usize) -> usize {
        match self {
            OutputLayout::Joint => c,
            OutputLayout::PerColumn => 1,
        }
    }
}

impl From<bool> for OutputLayout {
    fn from(per_output: bool) -> Self {
        if per_output {
            OutputLayout::PerColumn
        } else {
            OutputLayout::Joint
        }
    }
}

impl From<OutputLayout> for bool {
    fn from(layout: OutputLayout) -> bool {
        layout == OutputLayout::PerColumn
    }
}

/// Synthetic rows from [`ForestModel::sample`], row-major
/// `[row][column]`, with their labels for a class-conditional model.
#[derive(Debug, Clone, PartialEq)]
pub struct Synthetic {
    values: Vec<f32>,
    labels: Option<Vec<f32>>,
    n_columns: usize,
}

impl Synthetic {
    /// Row `row`, or `None` past the last row.
    pub fn row(&self, row: usize) -> Option<&[f32]> {
        let start = row.checked_mul(self.n_columns)?;
        self.values.get(start..start.checked_add(self.n_columns)?)
    }

    /// The rows in order.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &[f32]> {
        self.values.chunks_exact(self.n_columns)
    }

    /// The flat `[row][column]` buffer.
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// The flat `[row][column]` buffer and the labels, without copying.
    pub fn into_parts(self) -> (Vec<f32>, Option<Vec<f32>>) {
        (self.values, self.labels)
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

/// Imputed rows from [`ForestModel::impute`]: per imputation, one row per
/// input row, laid out `[imputation][row][column]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Imputations {
    values: Vec<f32>,
    draws: usize,
    n_rows: usize,
    n_columns: usize,
}

impl Imputations {
    /// Number of imputations.
    pub fn n_imputations(&self) -> usize {
        self.draws
    }

    /// Rows per imputation (the input's rows).
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns.
    pub fn n_columns(&self) -> usize {
        self.n_columns
    }

    /// Row `row` of imputation `imputation`, or `None` if either is out of
    /// range.
    pub fn get(&self, imputation: usize, row: usize) -> Option<&[f32]> {
        if imputation >= self.draws || row >= self.n_rows {
            return None;
        }
        let start = (imputation * self.n_rows + row) * self.n_columns;
        Some(&self.values[start..start + self.n_columns])
    }

    /// The flat `[imputation][row][column]` buffer.
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// The flat `[imputation][row][column]` buffer, without copying.
    pub fn into_vec(self) -> Vec<f32> {
        self.values
    }
}

impl AsRef<[f32]> for Imputations {
    fn as_ref(&self) -> &[f32] {
        &self.values
    }
}

/// A fitted ForestFlow / ForestDiffusion model. See the
/// [module docs](self).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "format::UncheckedForestModel")]
pub struct ForestModel {
    method: ForestMethod,
    n_t: NoiseLevels,
    columns: Vec<Column>,
    /// Scaling of every encoded column.
    scales: Vec<Scale>,
    /// The class labels (empty for an unconditional model) and their
    /// training proportions.
    classes: Vec<f64>,
    class_probs: Vec<f64>,
    /// One multi-output GBDT per level, or one per level and encoded
    /// column.
    #[serde(rename = "per_output")]
    layout: OutputLayout,
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
        fit::fit(params, data)
    }

    /// Draw `n_rows` synthetic rows, `[row][column]`. A class-conditional
    /// model draws each row's label from the training proportions.
    /// Deterministic for a given `seed` at any thread count.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] for `n_rows == 0`, a request too
    /// large to allocate, or a sampler that diverges.
    pub fn sample(&self, n_rows: usize, seed: u64) -> Result<Synthetic> {
        positive_count("n_rows", n_rows)?;
        let mut class_of: Vec<usize> = try_filled(n_rows, 0, "n_rows")?;
        if !self.classes.is_empty() {
            let key = splitmix64(seed ^ LABEL_STREAM);
            for (r, class) in class_of.iter_mut().enumerate() {
                let u = keyed_unit(key, r as u64);
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
        self.sample_classes(&class_of, seed)
    }

    /// Draw one row per entry of `labels` from that class's model,
    /// `[row][column]`, labelled with them.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] for an unconditional model, an
    /// empty `labels`, or a label the model was not trained on; the errors
    /// of [`Self::sample`].
    pub fn sample_for_labels(&self, labels: &[f32], seed: u64) -> Result<Synthetic> {
        positive_count("labels", labels.len())?;
        let class_of = self.class_indices(labels)?;
        self.sample_classes(&class_of, seed)
    }

    /// Impute the missing entries of every row of `data`, `n_imputations`
    /// times with `options` (seed, [`Repaint`]), laid out
    /// `[imputation][row][column]`; observed entries are
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
        options: &ImputeOptions,
    ) -> Result<Imputations> {
        let ImputeOptions { seed, repaint } = *options;
        let Some(sde) = self.method.sde() else {
            return Err(HessboostError::invalid_param(
                "method",
                "imputation needs ForestMethod::Diffusion (flow matching cannot condition \
                 on the observed entries)",
            ));
        };
        positive_count("n_imputations", n_imputations)?;
        let (resample, jump) = match repaint {
            None => (1, self.n_t.get()),
            Some(r) => {
                if !(r.jump.is_finite() && r.jump > 0.0 && r.jump <= 1.0) {
                    return Err(HessboostError::invalid_param(
                        "repaint.jump",
                        format!("must be in (0, 1], got {}", r.jump),
                    ));
                }
                (
                    r.resample.get(),
                    ((r.jump * self.n_t.get() as f64).ceil() as usize).max(1),
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
            for (j, (column, &v)) in self.columns.iter().zip(row).enumerate() {
                let v = f64::from(v);
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
            encode_row(&self.columns, row, out);
            for (v, s) in out.iter_mut().zip(&self.scales) {
                *v = s.forward(*v);
            }
        }
        let n_rows = data.n_rows();
        let total = n_imputations
            .checked_mul(n_rows)
            .and_then(|m| m.checked_mul(p))
            .ok_or_else(|| {
                HessboostError::invalid_param("n_imputations", "the imputed table overflows usize")
            })?;
        let mut out = try_filled(total, 0.0f32, "n_imputations")?;
        let mut sampler = Sampler::new(self, &class_of, 0)?;
        for (i, dest) in out.chunks_exact_mut(n_rows * p).enumerate() {
            sampler.key = splitmix64(seed ^ splitmix64(i as u64));
            let x = sampler.reverse_sde(sde, Some(&known), (resample, jump))?;
            self.decode(&x, dest)?;
        }
        Ok(Imputations {
            values: out,
            draws: n_imputations,
            n_rows,
            n_columns: p,
        })
    }

    /// The method.
    pub fn method(&self) -> ForestMethod {
        self.method
    }

    /// Number of noise levels.
    pub fn n_t(&self) -> NoiseLevels {
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
        Ok(serde_json::to_string_pretty(self)?)
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

    fn sample_classes(&self, class_of: &[usize], seed: u64) -> Result<Synthetic> {
        let n_rows = class_of.len();
        let p = self.columns.len();
        let total = n_rows.checked_mul(p).ok_or_else(|| {
            HessboostError::invalid_param("n_rows", "the synthetic table overflows usize")
        })?;
        let mut values = try_filled(total, 0.0f32, "n_rows")?;
        let mut sampler = Sampler::new(self, class_of, seed)?;
        let x = match self.method.sde() {
            None => sampler.euler_flow()?,
            Some(sde) => sampler.reverse_sde(sde, None, (1, self.n_t.get()))?,
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
        let per = self.layout.gbdts_per_level(self.scales.len());
        let start = (class * self.n_t.get() + level) * per;
        &self.models[start..start + per]
    }

    /// Check what sampling and the formats rely on.
    fn validate(&self) -> Result<()> {
        let bad = |msg: String| Err(HessboostError::model_format(msg));
        self.method
            .validate()
            .map_err(|e| HessboostError::model_format(e.to_string()))?;
        if self.columns.is_empty() {
            return bad("a forest model needs a column".into());
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
        let per = self.layout.gbdts_per_level(c);
        let expected = self
            .classes
            .len()
            .max(1)
            .checked_mul(self.n_t.get())
            .and_then(|m| m.checked_mul(per));
        if expected != Some(self.models.len()) {
            return bad(format!(
                "{} GBDTs, expected one per class, level{}",
                self.models.len(),
                match self.layout {
                    OutputLayout::Joint => "",
                    OutputLayout::PerColumn => " and column",
                }
            ));
        }
        let outputs = self.layout.gbdt_outputs(c);
        for model in &self.models {
            check_regressor("forest GBDT", model, c, outputs)?;
        }
        Ok(())
    }
}

/// Refuse the metadata a forest model cannot honor.
fn refuse_metadata(data: &DMatrix) -> Result<()> {
    super::fit::refuse_unsupported_metadata(data, "forest")?;
    if data.labels().is_some() && data.n_targets() != 1 {
        return Err(HessboostError::invalid_param(
            "data",
            "forest models do not support label matrices (labels are class labels)",
        ));
    }
    Ok(())
}

/// Generation and imputation state: the model, the seed key, and each
/// class's rows with the input buffer their GBDTs predict from.
struct Sampler<'a> {
    model: &'a ForestModel,
    key: u64,
    n_rows: usize,
    batches: Vec<ClassBatch>,
}

/// The rows of one class and the dense `[row][c]` matrix, rewritten at each
/// noise level, that its GBDTs predict.
struct ClassBatch {
    class: usize,
    rows: Vec<usize>,
    input: DMatrix,
}

impl<'a> Sampler<'a> {
    /// A sampler of the rows `class_of` (each row's class) keyed by `key`.
    fn new(model: &'a ForestModel, class_of: &[usize], key: u64) -> Result<Self> {
        let c = model.scales.len();
        let mut rows = vec![Vec::new(); model.classes.len().max(1)];
        for (r, &class) in class_of.iter().enumerate() {
            rows[class].push(r);
        }
        let batches = rows
            .into_iter()
            .enumerate()
            .filter(|(_, rows)| !rows.is_empty())
            .map(|(class, rows)| {
                let values = try_filled(rows.len() * c, 0.0f32, "n_rows")?;
                let input = DMatrix::from_dense_vec(values, rows.len(), c)?;
                Ok(ClassBatch { class, rows, input })
            })
            .collect::<Result<_>>()?;
        Ok(Sampler {
            model,
            key,
            n_rows: class_of.len(),
            batches,
        })
    }

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
        let mut x = try_filled(self.n_rows * c, 0.0, "n_rows")?;
        for (row, values) in x.chunks_exact_mut(c).enumerate() {
            for (j, v) in values.iter_mut().enumerate() {
                *v = self.noise(PRIOR_STREAM, row, j as u64);
            }
        }
        Ok(x)
    }

    /// The predictions of the GBDTs of noise level `level` at `x`
    /// (`[row][c]`).
    fn predict(&mut self, x: &[f64], level: usize) -> Result<Vec<f64>> {
        let model = self.model;
        let c = model.scales.len();
        let mut out = vec![0.0; x.len()];
        for batch in &mut self.batches {
            let values = batch
                .input
                .dense_values_mut()
                .ok_or_else(|| HessboostError::model_format("sampling input must be dense"))?;
            for (&r, dest) in batch.rows.iter().zip(values.chunks_exact_mut(c)) {
                for (d, &v) in dest.iter_mut().zip(&x[r * c..(r + 1) * c]) {
                    *d = v as f32;
                }
            }
            // The matrix's invariant, which building it checked before.
            if values.iter().any(|v| v.is_infinite()) {
                return Err(HessboostError::invalid_param(
                    "dense data",
                    "non-missing feature values must be finite",
                ));
            }
            let models = model.level_models(batch.class, level);
            for (m, gbdt) in models.iter().enumerate() {
                let pred = gbdt.predict_margin(&batch.input)?;
                for (&r, p) in batch.rows.iter().zip(pred.rows()) {
                    match model.layout {
                        OutputLayout::PerColumn => out[r * c + m] = f64::from(p[0]),
                        OutputLayout::Joint => {
                            for (o, &v) in out[r * c..(r + 1) * c].iter_mut().zip(p) {
                                *o = f64::from(v);
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// ForestFlow: Euler steps of the learned velocity from `t = 0` to `1`.
    fn euler_flow(&mut self) -> Result<Vec<f64>> {
        let n_t = self.model.n_t.get();
        let h = 1.0 / (n_t - 1) as f64;
        let mut x = self.prior()?;
        for step in 0..n_t - 1 {
            let v = self.predict(&x, step)?;
            for (s, &d) in x.iter_mut().zip(&v) {
                *s += h * d;
            }
            check_finite(&x)?;
        }
        Ok(x)
    }

    /// The score at `x` and noise level `level`; with `known` (imputation),
    /// the observed entries are first re-noised to that level with the draws
    /// of evaluation `eval`.
    fn score(
        &mut self,
        sde: Sde,
        x: &mut [f64],
        known: Option<&[f64]>,
        level: usize,
        eval: u64,
    ) -> Result<Vec<f64>> {
        let (alpha, std) = sde.marginal(level_time(self.model.n_t.get(), level));
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
        let mut out = self.predict(x, level)?;
        for v in &mut out {
            *v = -*v / std;
        }
        Ok(out)
    }

    /// ForestDiffusion: Euler–Maruyama on the reverse VP SDE over the `n_t`
    /// levels from `t = 1` to `10⁻³`, with RePaint's `(resample, jump)` and a
    /// final Tweedie denoising step.
    fn reverse_sde(
        &mut self,
        sde: Sde,
        known: Option<&[f64]>,
        (resample, jump): (usize, usize),
    ) -> Result<Vec<f64>> {
        let n_t = self.model.n_t.get();
        let c = self.model.scales.len() as u64;
        // Step `i` runs from level `n_t - 1 - i` to the one below it.
        let level = |i: usize| n_t - 1 - i;
        let step = |i: usize| level_time(n_t, level(i)) - level_time(n_t, level(i) - 1);
        let mut x = self.prior()?;
        let mut eval = 0u64;
        let (mut i, mut passes) = (0usize, 0usize);
        while i < n_t - 1 {
            let t = level_time(n_t, level(i));
            let h = step(i);
            let score = self.score(sde, &mut x, known, level(i), eval)?;
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
        let score = self.score(sde, &mut x, known, 0, eval)?;
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

fn positive_count(name: &'static str, v: usize) -> Result<()> {
    if v == 0 {
        return Err(HessboostError::invalid_param(name, "must be at least 1"));
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::train;

    /// A one-column diffusion model whose level-`L` GBDT predicts `L`, so
    /// the draws show which level each step evaluates.
    fn level_indexed_model(n_t: usize) -> ForestModel {
        let data = DMatrix::from_dense(&[0.0, 1.0], 2, 1).unwrap();
        let params = TrainingParams::default();
        let models = (0..n_t)
            .map(|level| {
                let dtrain = data.clone().with_labels(&[level as f32; 2]).unwrap();
                train(&params, &dtrain, 1).unwrap()
            })
            .collect();
        ForestModel {
            method: ForestMethod::forest_diffusion(),
            n_t: NoiseLevels::new(n_t).unwrap(),
            columns: vec![Column {
                kind: ColumnKind::Continuous,
                min: -1e30,
                max: 1e30,
                categories: Vec::new(),
            }],
            scales: vec![Scale {
                min: -1.0,
                range: 2.0,
            }],
            classes: Vec::new(),
            class_probs: Vec::new(),
            layout: OutputLayout::Joint,
            models,
        }
    }

    /// Every reverse-SDE step evaluates the GBDTs trained at its own time,
    /// at any `n_t`: finding the level by rounding `t (n_t - 1)` picked the
    /// next one up once `EPS · i` reached `½` (from step 500 on).
    #[test]
    fn reverse_steps_use_the_level_trained_at_their_time() {
        let n_t = 600;
        let model = level_indexed_model(n_t);
        let sde = model.method.sde().unwrap();
        let mut sampler = Sampler::new(&model, &[0, 0, 0], 7).unwrap();
        let drawn = sampler.reverse_sde(sde, None, (1, n_t)).unwrap();

        // The same Euler–Maruyama steps with each step's level named.
        let mut x = sampler.prior().unwrap();
        for (eval, level) in (1..n_t).rev().enumerate() {
            let t = level_time(n_t, level);
            let h = t - level_time(n_t, level - 1);
            let (_, std) = sde.marginal(t);
            let (drift, g2) = sde.drift_diffusion(t);
            for (row, v) in x.iter_mut().enumerate() {
                let score = -(level as f64) / std;
                let noise = sampler.noise(STEP_STREAM, row, eval as u64);
                *v = *v - (drift * *v - g2 * score) * h + (g2 * h).sqrt() * noise;
            }
        }
        // Level 0's GBDT predicts 0: the final denoising step adds nothing.
        for (a, b) in drawn.iter().zip(&x) {
            assert!((a - b).abs() <= 1e-9 * b.abs().max(1.0), "{a} vs {b}");
        }
    }
}
