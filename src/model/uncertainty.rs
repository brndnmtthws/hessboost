//! Virtual ensembles and uncertainty decomposition for models trained with
//! SGLB posterior sampling (beyond XGBoost; CatBoost's
//! `virtual_ensembles_predict`).
//!
//! A model trained with
//! [`posterior_sampling`](crate::config::TrainingParams::posterior_sampling)
//! (Stochastic Gradient Langevin Boosting with model shrinkage; Ustimenko and
//! Prokhorenkova, *SGLB: Stochastic Gradient Langevin Boosting*, ICML 2021)
//! draws its later iterates from the Bayesian posterior of the ensemble.
//! Malinin, Prokhorenkova and Ustimenko (*Uncertainty in Gradient Boosting
//! via Ensembles*, ICLR 2021) turn one such model into a *virtual ensemble*:
//! the models after several iterations of its second half act as posterior
//! samples, and the spread of their predictions measures knowledge
//! (epistemic) uncertainty, which grows away from the training data.
//!
//! [`BoostedModel::predict_virtual_ensembles`] returns the members'
//! predictions ([`VirtualEnsembles`]), [`BoostedModel::predict_uncertainty`]
//! their decomposition ([`Uncertainty`]).
//!
//! # Members
//!
//! As in CatBoost's `ApplyVirtualEnsembles`
//! (`catboost/private/libs/algo/apply.cpp`), `K` members of a model with `T`
//! (effective) iterations are the models after `b + p`, `b + 2p`, ..., `T`
//! iterations, with period `p = ⌊T / (2K)⌋` and `b = T − pK`; `p` must be
//! positive and `b` too (at least `2K` iterations, roughly). Because model
//! shrinkage rescales the ensemble every iteration, a member is not a prefix
//! of the trees: CatBoost multiplies the prefix by `(1 − rate · eta)^(k − T)`
//! in `f32`, which holds for the `constant` shrink mode only. hessboost
//! rebuilds each member from the stored shrinkage coefficients with the
//! arithmetic of training (see
//! [`predict_margin_range`](BoostedModel::predict_margin_range)), so member
//! `k` predicts bit for bit what the same run stopped after `k` rounds
//! predicts, in either shrink mode. A model trained without shrinkage has
//! plain prefixes as members.
//!
//! # Decomposition
//!
//! Following CatBoost (`CalcRegressionUncertaitny`,
//! `CalcClassificationUncertainty`, `CalcMulticlassUncertainty` in
//! `catboost/libs/eval_result/eval_helpers.cpp`; the CatBoost *Uncertainty*
//! reference), with `K` members:
//!
//! - **Regression** (`reg:*`, `count:*`, `survival:*`): knowledge
//!   uncertainty is the variance (over members, divided by `K`) of the
//!   members' predictions, per output. There is no data uncertainty.
//! - **Distributional** (`dist:*`, the analogue of CatBoost's
//!   `RMSEWithUncertainty` for `dist:normal`): knowledge uncertainty is the
//!   variance of the members' predicted means, data uncertainty the mean of
//!   their predicted variances, and the total their sum (the law of total
//!   variance).
//! - **Classification** (`binary:logistic`, `binary:logitraw`,
//!   `reg:logistic`, per label column for multi-label models, and
//!   `multi:softprob` / `multi:softmax` over the classes): total uncertainty
//!   is the entropy (in nats) of the mean predicted distribution, data
//!   uncertainty the mean of the members' entropies, and knowledge
//!   uncertainty their difference (the mutual information between the
//!   prediction and the ensemble member), non-negative up to rounding.
//!
//! Other objectives (ranking, `binary:hinge`, custom objectives) are
//! refused.
//!
//! ```
//! use hessboost::prelude::*;
//!
//! # fn main() -> Result<()> {
//! let x: Vec<f32> = (0..200).map(|i| (i % 50) as f32 / 50.0).collect();
//! let y: Vec<f32> = x.iter().map(|v| 2.0 * v).collect();
//! let dtrain = DMatrix::from_dense(&x, 200, 1)?.with_labels(&y)?;
//! let params = TrainingParams::builder()
//!     .posterior_sampling(true)
//!     .eta(0.1)
//!     .max_depth(3)
//!     .build()?;
//! let model = train(&params, &dtrain, 100)?;
//!
//! let ensembles = model.predict_virtual_ensembles(&dtrain, 10)?;
//! assert_eq!(ensembles.iterations(), &[55, 60, 65, 70, 75, 80, 85, 90, 95, 100]);
//! let uncertainty = model.predict_uncertainty(&dtrain, 10)?;
//! assert_eq!(uncertainty.knowledge.len(), 200);
//! assert!(uncertainty.data.is_none()); // squared error has no data uncertainty
//! # Ok(())
//! # }
//! ```

use super::{BoostedModel, transform_model_margins};
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::objective::distributional::{Dist, DistFamily};

/// The predictions of a model's virtual ensemble
/// ([`BoostedModel::predict_virtual_ensembles`]): one model per member, the
/// model after [`iterations`](Self::iterations)`()[m]` boosting iterations.
#[derive(Debug, Clone)]
pub struct VirtualEnsembles {
    iterations: Vec<usize>,
    n_rows: usize,
    n_outputs: usize,
    width: usize,
    margins: Vec<f32>,
    predictions: Vec<f32>,
}

impl VirtualEnsembles {
    /// Number of members.
    pub fn n_members(&self) -> usize {
        self.iterations.len()
    }

    /// The iteration count of every member's model, ascending; the last is
    /// the whole (effective) model.
    pub fn iterations(&self) -> &[usize] {
        &self.iterations
    }

    /// Number of predicted rows.
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Values per row of a member's predictions (the width of
    /// [`BoostedModel::predict`]'s layout).
    pub fn width(&self) -> usize {
        self.width
    }

    /// Member `m`'s predictions in [`BoostedModel::predict`]'s layout
    /// (`n_rows * width`), or `None` past the last member.
    pub fn member_predictions(&self, m: usize) -> Option<&[f32]> {
        if m >= self.n_members() {
            return None;
        }
        let len = self.n_rows * self.width;
        self.predictions.get(m * len..(m + 1) * len)
    }

    /// Member `m`'s raw margins in [`BoostedModel::predict_margin`]'s layout
    /// (`n_rows * n_outputs`), or `None` past the last member.
    pub fn member_margins(&self, m: usize) -> Option<&[f32]> {
        if m >= self.n_members() {
            return None;
        }
        let len = self.n_rows * self.n_outputs;
        self.margins.get(m * len..(m + 1) * len)
    }

    /// Every member's predictions, member-major: `[member][row][value]`.
    pub fn predictions(&self) -> &[f32] {
        &self.predictions
    }

    /// Every member's raw margins, member-major: `[member][row][output]`.
    pub fn margins(&self) -> &[f32] {
        &self.margins
    }
}

/// A virtual ensemble's uncertainty decomposition
/// ([`BoostedModel::predict_uncertainty`]; see the
/// [module docs](self#decomposition) for the per-objective definitions).
///
/// `knowledge`, `data`, and `total` hold [`columns`](Self::columns) values
/// per row, row-major: one per row, except one per output for multi-output
/// regression and one per label column for multi-label classification.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Uncertainty {
    /// Values per row of `knowledge`, `data`, and `total`.
    pub columns: usize,
    /// The members' mean prediction: predictions for regression, predicted
    /// means for `dist:*`, probabilities for classification
    /// (`multi:softmax` too: one per class), row-major.
    pub mean: Vec<f64>,
    /// Knowledge (epistemic) uncertainty.
    pub knowledge: Vec<f64>,
    /// Data (aleatoric) uncertainty; `None` for plain regression, whose
    /// models predict no spread.
    pub data: Option<Vec<f64>>,
    /// Total uncertainty (`data + knowledge`); `None` when `data` is.
    pub total: Option<Vec<f64>>,
}

/// How [`BoostedModel::predict_uncertainty`] reads an objective's members.
enum Decomposition {
    Regression,
    Distributional(DistFamily),
    Binary,
    Multiclass,
}

impl Decomposition {
    fn of(objective: &str) -> Result<Self> {
        if let Some(family) = DistFamily::from_objective(objective) {
            return Ok(Decomposition::Distributional(family));
        }
        Ok(match objective {
            "binary:logistic" | "binary:logitraw" | "reg:logistic" => Decomposition::Binary,
            "multi:softprob" | "multi:softmax" => Decomposition::Multiclass,
            _ if ["reg:", "count:", "survival:"]
                .iter()
                .any(|prefix| objective.starts_with(prefix)) =>
            {
                Decomposition::Regression
            }
            _ => {
                return Err(HessboostError::invalid_param(
                    "objective",
                    format!(
                        "uncertainty is defined for regression, `dist:*`, and probabilistic \
                         classification objectives, not `{objective}`"
                    ),
                ));
            }
        })
    }
}

/// Binary entropy in nats, `0 ln 0 = 0`.
fn binary_entropy(p: f64) -> f64 {
    let term = |q: f64| if q > 0.0 { -q * q.ln() } else { 0.0 };
    term(p) + term(1.0 - p)
}

/// Mean and population variance of `values`.
fn mean_variance(values: impl Iterator<Item = f64> + Clone) -> (f64, f64) {
    let n = values.clone().count() as f64;
    let mean = values.clone().sum::<f64>() / n;
    let variance = values.map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    (mean, variance)
}

impl BoostedModel {
    /// The iteration counts of a virtual ensemble of `count` members over
    /// the effective iterations (CatBoost's `ApplyVirtualEnsembles`).
    fn virtual_ensemble_iterations(&self, count: usize) -> Result<Vec<usize>> {
        if self.linear.is_some() {
            return Err(HessboostError::invalid_param(
                "model",
                "gblinear models have no boosting iterations to form virtual ensembles from",
            ));
        }
        if count == 0 {
            return Err(HessboostError::invalid_param(
                "virtual_ensembles_count",
                "must be at least 1",
            ));
        }
        let end = self.effective_num_trees() / self.trees_per_iteration();
        let period = end / count.saturating_mul(2);
        if period == 0 || period * count >= end {
            return Err(HessboostError::invalid_param(
                "virtual_ensembles_count",
                format!(
                    "{count} virtual ensembles need a model of at least {} iterations, this one \
                     has {end}",
                    count.saturating_mul(2)
                ),
            ));
        }
        let begin = end - period * count;
        Ok((1..=count).map(|m| begin + m * period).collect())
    }

    /// Predict `data` with a virtual ensemble of `count` members (CatBoost's
    /// `virtual_ensembles_predict`, default count 10): the models after
    /// several iterations of the second half of the effective iterations
    /// (see the [`uncertainty`](super::uncertainty) module docs). Meant for
    /// models trained with
    /// [`posterior_sampling`](crate::config::TrainingParams::posterior_sampling);
    /// any tree model is accepted.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] for `count == 0`, a model with
    /// fewer iterations than `count` members need, or a `gblinear` model,
    /// plus the errors of [`Self::predict_margin`].
    pub fn predict_virtual_ensembles(
        &self,
        data: &DMatrix,
        count: usize,
    ) -> Result<VirtualEnsembles> {
        let iterations = self.virtual_ensemble_iterations(count)?;
        let mut margins = Vec::new();
        let mut predictions = Vec::new();
        for &k in &iterations {
            let margin = self.predict_margin_range(data, ..k)?;
            margins.extend_from_slice(&margin);
            predictions.extend(transform_model_margins(
                &self.objective,
                &self.objective_params,
                self.num_class,
                self.n_targets,
                self.n_outputs,
                margin,
            ));
        }
        Ok(VirtualEnsembles {
            // `predict`'s layout: one class index per row for `multi:softmax`.
            width: if self.objective == "multi:softmax" {
                1
            } else {
                self.n_outputs
            },
            iterations,
            n_rows: data.n_rows(),
            n_outputs: self.n_outputs,
            margins,
            predictions,
        })
    }

    /// The knowledge, data, and total uncertainty of `data` under a virtual
    /// ensemble of `count` members ([`Self::predict_virtual_ensembles`];
    /// decomposition per objective in the
    /// [`uncertainty`](super::uncertainty#decomposition) module docs).
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] for objectives without a
    /// decomposition (ranking, `binary:hinge`, custom objectives), plus the
    /// errors of [`Self::predict_virtual_ensembles`].
    pub fn predict_uncertainty(&self, data: &DMatrix, count: usize) -> Result<Uncertainty> {
        let decomposition = Decomposition::of(&self.objective)?;
        let ensembles = self.predict_virtual_ensembles(data, count)?;
        let n = ensembles.n_rows;
        let k = self.n_outputs;
        let members = || 0..ensembles.n_members();
        let margin = |m: usize, row: usize, out: usize| {
            f64::from(ensembles.margins[(m * n + row) * k + out])
        };
        Ok(match decomposition {
            Decomposition::Regression => {
                let width = ensembles.width;
                let mut mean = Vec::with_capacity(n * width);
                let mut knowledge = Vec::with_capacity(n * width);
                for cell in 0..n * width {
                    let values =
                        members().map(|m| f64::from(ensembles.predictions[m * n * width + cell]));
                    let (mu, variance) = mean_variance(values);
                    mean.push(mu);
                    knowledge.push(variance);
                }
                Uncertainty {
                    columns: width,
                    mean,
                    knowledge,
                    data: None,
                    total: None,
                }
            }
            Decomposition::Distributional(family) => {
                let mut mean = Vec::with_capacity(n);
                let mut knowledge = Vec::with_capacity(n);
                let mut aleatoric = Vec::with_capacity(n);
                for row in 0..n {
                    let dists: Vec<_> = members()
                        .map(|m| {
                            let eta: Vec<f64> = (0..k).map(|out| margin(m, row, out)).collect();
                            family.dist_from_margins(&eta)
                        })
                        .collect();
                    let (mu, variance) = mean_variance(dists.iter().map(Dist::mean));
                    mean.push(mu);
                    knowledge.push(variance);
                    aleatoric.push(dists.iter().map(Dist::variance).sum::<f64>() / count as f64);
                }
                let total = knowledge
                    .iter()
                    .zip(&aleatoric)
                    .map(|(a, b)| a + b)
                    .collect();
                Uncertainty {
                    columns: 1,
                    mean,
                    knowledge,
                    data: Some(aleatoric),
                    total: Some(total),
                }
            }
            Decomposition::Binary => {
                let mut mean = Vec::with_capacity(n * k);
                let mut aleatoric = Vec::with_capacity(n * k);
                let mut total = Vec::with_capacity(n * k);
                for row in 0..n {
                    for out in 0..k {
                        let probs = members().map(|m| 1.0 / (1.0 + (-margin(m, row, out)).exp()));
                        let p = probs.clone().sum::<f64>() / count as f64;
                        mean.push(p);
                        aleatoric.push(probs.map(binary_entropy).sum::<f64>() / count as f64);
                        total.push(binary_entropy(p));
                    }
                }
                let knowledge = total.iter().zip(&aleatoric).map(|(t, d)| t - d).collect();
                Uncertainty {
                    columns: k,
                    mean,
                    knowledge,
                    data: Some(aleatoric),
                    total: Some(total),
                }
            }
            Decomposition::Multiclass => {
                let mut mean = vec![0.0; n * k];
                let mut aleatoric = Vec::with_capacity(n);
                let mut total = Vec::with_capacity(n);
                let mut probs = vec![0.0; k];
                for row in 0..n {
                    let row_mean = &mut mean[row * k..(row + 1) * k];
                    let mut entropy_sum = 0.0;
                    for m in members() {
                        let max = (0..k)
                            .map(|c| margin(m, row, c))
                            .fold(f64::NEG_INFINITY, f64::max);
                        for (c, p) in probs.iter_mut().enumerate() {
                            *p = (margin(m, row, c) - max).exp();
                        }
                        let sum: f64 = probs.iter().sum();
                        for (p, acc) in probs.iter_mut().zip(row_mean.iter_mut()) {
                            *p /= sum;
                            *acc += *p;
                            if *p > 0.0 {
                                entropy_sum -= *p * p.ln();
                            }
                        }
                    }
                    let mut entropy_of_mean = 0.0;
                    for p in row_mean.iter_mut() {
                        *p /= count as f64;
                        if *p > 0.0 {
                            entropy_of_mean -= *p * p.ln();
                        }
                    }
                    aleatoric.push(entropy_sum / count as f64);
                    total.push(entropy_of_mean);
                }
                let knowledge = total.iter().zip(&aleatoric).map(|(t, d)| t - d).collect();
                Uncertainty {
                    columns: 1,
                    mean,
                    knowledge,
                    data: Some(aleatoric),
                    total: Some(total),
                }
            }
        })
    }
}
