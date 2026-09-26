//! How a model trained with per-iteration model shrinkage (SGLB; CatBoost
//! `model_shrink_rate`) stores the shrinkage, and how any truncation of it is
//! reconstructed exactly.
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
//! unscaled and stores the coefficients and `b0` instead
//! ([`Shrinkage`]): the full model's per-tree contribution weights are
//! `P(i+1, T)` and its intercepts `b0 · P(1, T)`, and the model after any
//! `k < T` iterations is rebuilt from the same record with the same
//! arithmetic ([`Shrinkage::scaling`]) that training ran for `k` rounds
//! uses. A prefix is thus bit for bit the model trained for `k` rounds,
//! rather than the prefix rescaled by `1 / P(k, T)` as CatBoost's
//! `ApplyVirtualEnsembles` computes it.

use serde::{Deserialize, Serialize};

use crate::error::{HessboostError, Result};

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

    /// The model after `k` iterations (`k` at most the recorded count): the
    /// contribution weight of each of its `k * trees_per_iteration` trees,
    /// `P(i + 1, k)` for every tree of iteration `i`, and its intercepts
    /// `b0 · P(1, k)`. The products run from the last iteration backwards in
    /// `f64` and are rounded to `f32` once, so every `k` repeats training's
    /// arithmetic for a `k`-round run exactly.
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
