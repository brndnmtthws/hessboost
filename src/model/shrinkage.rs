//! How a model trained with per-iteration model shrinkage (SGLB; CatBoost
//! `model_shrink_rate`) stores the shrinkage, and how it and any truncation
//! of it predict exactly what training computed.
//!
//! CatBoost (`TrainOneIteration` in `catboost/private/libs/algo/train.cpp`)
//! multiplies every approximation, the starting approximation (the
//! intercept) included, by the iteration's coefficient `s_i` before the
//! iteration's gradients, recording `s_0 = 1` and every later coefficient in
//! `ModelShrinkHistory`; after training (`train_model.cpp`) it bakes the
//! product of the later coefficients into each tree's leaf values. The
//! model after `k` iterations is therefore
//!
//! ```text
//! F_k(x) = b0 · P(1, k) + Σ_{i<k} P(i+1, k) · f_i(x),   P(a, k) = Π_{a<=j<k} s_j
//! ```
//!
//! with `f_i` iteration `i`'s trees as grown (leaves already scaled by
//! `eta`) and `b0` the intercept before shrinkage. hessboost keeps the trees
//! unscaled and stores the coefficients and `b0` instead ([`Shrinkage`]).
//!
//! Training evaluates `F_k` by its recurrence in `f32` margins: starting
//! from `b0`, every iteration multiplies each margin by `s_i` (in `f64`,
//! rounded once, [`shrink_margins`]) and then adds its trees. Prediction
//! ([`BoostedModel::predict_margin`](super::BoostedModel::predict_margin),
//! slices, virtual ensembles, the compact format) runs the same recurrence
//! over the same record, so the model after any `k` iterations predicts
//! bit for bit the margins training reached after `k` rounds (and those of
//! the run stopped there), rather than the prefix rescaled by
//! `1 / P(k, T)` as CatBoost's `ApplyVirtualEnsembles` computes it.
//!
//! The model also stores the closed form's coefficients rounded to `f32`
//! ([`Shrinkage::scaling`]): per-tree contribution weights `P(i+1, T)` and
//! intercepts `b0 · P(1, T)`. They define the same function up to `f32`
//! rounding, and serve where a model must be a weighted sum of its trees:
//! TreeSHAP and XGBoost export (the weights baked into the leaves).

use serde::{Deserialize, Serialize};

use crate::data::DMatrix;
use crate::error::{HessboostError, Result};

use super::initial_margins;

/// Multiply every margin by the shrink coefficient `factor`, in `f64` and
/// rounded to `f32` once per cell: the shrink step training and prediction
/// share (`1` leaves the margins as they are).
pub(crate) fn shrink_margins(margins: &mut [f32], factor: f64) {
    if factor != 1.0 {
        for m in margins {
            *m = (f64::from(*m) * factor) as f32;
        }
    }
}

/// The per-iteration shrinkage record of a model: the coefficient `s_i`
/// applied at the start of every iteration `i` (`s_0 = 1`) and the
/// intercepts before shrinkage, in margin space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Shrinkage {
    factors: Vec<f64>,
    base_score: Vec<f32>,
}

impl Shrinkage {
    /// The record of the coefficients `factors` (one per iteration) and the
    /// unshrunk intercepts `base_score`; [`Self::validate`] checks it.
    pub(crate) fn new(factors: Vec<f64>, base_score: Vec<f32>) -> Self {
        Shrinkage {
            factors,
            base_score,
        }
    }

    /// The coefficient of every iteration.
    pub(crate) fn factors(&self) -> &[f64] {
        &self.factors
    }

    /// The intercepts before shrinkage.
    pub(crate) fn base_score(&self) -> &[f32] {
        &self.base_score
    }

    /// The record of the first `k` iterations.
    pub(crate) fn truncated(&self, k: usize) -> Shrinkage {
        Shrinkage::new(self.factors[..k].to_vec(), self.base_score.clone())
    }

    /// The closed form of the model after `k` iterations (`k` at most the
    /// recorded count), rounded to `f32`: the contribution weight of each of
    /// its `k * trees_per_iteration` trees, `P(i + 1, k)` for every tree of
    /// iteration `i`, and its intercepts `b0 · P(1, k)`. The products run
    /// from the last iteration backwards in `f64` and are rounded once.
    /// Prediction does not use them (see the [module docs](self)).
    pub(crate) fn scaling(&self, k: usize, trees_per_iteration: usize) -> (Vec<f32>, Vec<f32>) {
        let mut weights = vec![0.0f32; k * trees_per_iteration];
        let mut product = 1.0f64;
        for (i, layer) in weights
            .chunks_exact_mut(trees_per_iteration)
            .enumerate()
            .rev()
        {
            layer.fill(product as f32);
            product *= self.factors[i];
        }
        let base = self
            .base_score
            .iter()
            .map(|&b| (f64::from(b) * product) as f32)
            .collect();
        (weights, base)
    }

    /// Whether `tree_weight` (by tree id) and `base_score` are, bit for bit,
    /// the closed form of the whole recorded model ([`Self::scaling`]).
    pub(crate) fn matches(
        &self,
        trees_per_iteration: usize,
        tree_weight: impl Fn(usize) -> f32,
        base_score: &[f32],
    ) -> bool {
        let (weights, base) = self.scaling(self.factors.len(), trees_per_iteration);
        let same = |a: f32, b: f32| a.to_bits() == b.to_bits();
        weights
            .iter()
            .enumerate()
            .all(|(t, &w)| same(w, tree_weight(t)))
            && base.len() == base_score.len()
            && base.iter().zip(base_score).all(|(&a, &b)| same(a, b))
    }

    /// The margins `[row][output]` the recurrence starts from for `data`:
    /// the unshrunk intercepts, or zeros when `data` carries a
    /// `base_margin`, which replaces the shrunk intercepts and is added by
    /// [`Self::finish_margins`] (training refuses `base_margin` with
    /// shrinkage, so prediction does not shrink it).
    pub(crate) fn start_margins(&self, data: &DMatrix) -> Vec<f32> {
        if data.base_margin().is_some() {
            vec![0.0; data.n_rows() * self.base_score.len()]
        } else {
            initial_margins(&self.base_score, data)
        }
    }

    /// Add `data`'s `base_margin`, if any, to the shrunk trees' `margins`
    /// (see [`Self::start_margins`]).
    pub(crate) fn finish_margins(&self, data: &DMatrix, margins: &mut [f32]) {
        if data.base_margin().is_some() {
            let base = initial_margins(&self.base_score, data);
            for (m, b) in margins.iter_mut().zip(base) {
                *m += b;
            }
        }
    }

    /// Check the record against the model holding it: one coefficient per
    /// iteration, each finite and in `(0, 1]`, the first `1`; one finite
    /// unshrunk intercept per output.
    pub(crate) fn validate(&self, iterations: usize, n_outputs: usize) -> Result<()> {
        if self.factors.len() != iterations {
            return Err(HessboostError::ModelFormat(format!(
                "the shrinkage record holds {} coefficients for {iterations} iterations",
                self.factors.len()
            )));
        }
        if self
            .factors
            .iter()
            .any(|&s| !(s.is_finite() && s > 0.0 && s <= 1.0))
            || self.factors.first().is_some_and(|&s| s != 1.0)
        {
            return Err(HessboostError::model_format(
                "shrinkage coefficients must be in (0, 1], the first one 1",
            ));
        }
        if self.base_score.len() != n_outputs || self.base_score.iter().any(|b| !b.is_finite()) {
            return Err(HessboostError::ModelFormat(format!(
                "the shrinkage record must hold one finite intercept per output ({n_outputs} \
                 outputs, got {:?})",
                self.base_score
            )));
        }
        Ok(())
    }
}
