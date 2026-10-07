//! Linear leaves: LightGBM's `linear_tree` (opt-in).
//!
//! After a tree's structure is grown, every leaf fits a ridge-regularized
//! linear model on the numerical features split on along its root-to-leaf path
//! (LightGBM `linear_tree_learner.cpp`, after Shi et al., "Gradient Boosting
//! With Piece-Wise Linear Regression Trees"). With `X` the leaf's rows over
//! those features plus a trailing column of ones, `H = diag(h)` and `g` the
//! rows' gradients, the coefficients are the Newton step
//!
//! `β = −(XᵀHX + Λ)⁻¹ Xᵀg`,  `Λ = diag(λ, …, λ, 0)`,
//!
//! so `linear_lambda` (`λ`) penalizes the slopes but not the intercept.
//! `g` and `h` are whatever the objective trains on: for `reg:absoluteerror`
//! and `reg:quantileerror` their smooth surrogates' gradients and curvatures,
//! which makes the L1 fit an iteratively reweighted least-squares step.
//! Following LightGBM:
//!
//! - categorical features route rows but never enter a leaf model;
//! - rows with a missing value in any of the leaf's features are left out of
//!   its fit, and at prediction such rows get the ordinary constant leaf
//!   value instead (hessboost treats absent sparse entries as missing, as it
//!   does everywhere);
//! - a leaf with fewer complete rows than coefficients keeps its constant
//!   value (so does a leaf whose system is singular, which LightGBM's
//!   `fullPivLu().inverse()` leaves undefined);
//! - slopes with magnitude `<= 1e-35` (`kZeroThreshold`) are dropped;
//! - trees of the first boosting round and single-leaf trees stay constant.
//!
//! Learning-rate shrinkage scales intercepts and slopes together with the
//! constant leaf values ([`RegTree::scale_leaves`]). Split finding is
//! unchanged, so monotone constraints bound only the constant values, not the
//! fitted slopes (as in LightGBM). Numerical features should be on comparable
//! scales, since the slope penalty is not scale invariant.

use crate::data::Rows;
use crate::error::HessboostError;
use crate::tree::regtree::{Node, RegTree};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// The per-leaf linear models of one tree, indexed by node id: what
/// [`TrainingParams::linear_tree`](crate::config::TrainingParams::linear_tree)
/// fits in every leaf, read through [`RegTree::linear_leaves`].
///
/// Leaf `n` predicts `intercept(n) + Σ coeff·x[feature]` over its
/// [`terms`](Self::terms), or the node's constant `leaf_value` when any of
/// those features is missing. Internal nodes hold no terms.
///
/// Serde deserialization checks that the arrays are consistent (one
/// intercept and one term range per node, ranges inside the term arrays,
/// finite values) and refuses them otherwise; the owning [`RegTree`] checks
/// them against its nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(try_from = "UncheckedLinearLeaves")]
pub struct LinearLeaves {
    /// Node `n`'s terms are `features[offsets[n]..offsets[n + 1]]` (and the
    /// matching `coeffs`); length `num_nodes + 1`.
    offsets: Vec<u32>,
    /// Per-node intercept (`0` for internal nodes).
    intercepts: Vec<f64>,
    features: Vec<u32>,
    coeffs: Vec<f64>,
}

/// The serialized fields of [`LinearLeaves`] (same names and layout), before
/// validation. A serialized [`RegTree`] reads its leaf models through it, so
/// the tree checks them against its nodes in one place.
#[derive(Deserialize)]
pub(crate) struct UncheckedLinearLeaves {
    offsets: Vec<u32>,
    intercepts: Vec<f64>,
    features: Vec<u32>,
    coeffs: Vec<f64>,
}

impl UncheckedLinearLeaves {
    /// The leaf models as stored, unvalidated: the caller validates them
    /// ([`LinearLeaves::is_valid`]).
    pub(crate) fn into_unchecked(self) -> LinearLeaves {
        LinearLeaves::from_parts(self.offsets, self.intercepts, self.features, self.coeffs)
    }
}

impl TryFrom<UncheckedLinearLeaves> for LinearLeaves {
    type Error = HessboostError;

    fn try_from(unchecked: UncheckedLinearLeaves) -> Result<Self, Self::Error> {
        let linear = unchecked.into_unchecked();
        if linear.is_consistent() {
            Ok(linear)
        } else {
            Err(HessboostError::model_format(
                "linear leaf models are inconsistent",
            ))
        }
    }
}

impl LinearLeaves {
    /// The intercept of leaf `node`.
    #[inline]
    pub fn intercept(&self, node: usize) -> f64 {
        self.intercepts[node]
    }

    /// The `(features, slopes)` of leaf `node`, in evaluation order
    /// (features ascending in the models hessboost trains).
    #[inline]
    pub fn terms(&self, node: usize) -> (&[u32], &[f64]) {
        let range = self.offsets[node] as usize..self.offsets[node + 1] as usize;
        (&self.features[range.clone()], &self.coeffs[range])
    }

    /// Output of leaf `node` for a row read through `get` (`None` = missing):
    /// the linear model, or `constant` when one of its features is missing.
    #[inline]
    pub(crate) fn predict(
        &self,
        node: usize,
        constant: f32,
        get: impl Fn(u32) -> Option<f32>,
    ) -> f32 {
        let (features, coeffs) = self.terms(node);
        let mut out = self.intercepts[node];
        for (&f, &c) in features.iter().zip(coeffs) {
            match get(f) {
                Some(x) => out += c * f64::from(x),
                None => return constant,
            }
        }
        out as f32
    }

    /// Multiply every intercept and slope by `factor` (shrinkage).
    pub(crate) fn scale(&mut self, factor: f64) {
        for v in self.intercepts.iter_mut().chain(&mut self.coeffs) {
            *v *= factor;
        }
    }

    /// The stored arrays `(offsets, intercepts, features, coeffs)`, as
    /// [`LinearLeaves::from_parts`] takes them.
    pub(crate) fn parts(&self) -> (&[u32], &[f64], &[u32], &[f64]) {
        (
            &self.offsets,
            &self.intercepts,
            &self.features,
            &self.coeffs,
        )
    }

    /// Assemble leaf models from their stored arrays; the owning tree checks
    /// them with [`LinearLeaves::is_valid`].
    pub(crate) fn from_parts(
        offsets: Vec<u32>,
        intercepts: Vec<f64>,
        features: Vec<u32>,
        coeffs: Vec<f64>,
    ) -> Self {
        LinearLeaves {
            offsets,
            intercepts,
            features,
            coeffs,
        }
    }

    /// Consistency of the arrays on their own: one intercept and one
    /// non-decreasing term range per node, the ranges covering `features`
    /// and `coeffs` exactly, and finite values.
    fn is_consistent(&self) -> bool {
        self.offsets.len() == self.intercepts.len() + 1
            && self.offsets.first() == Some(&0)
            && self
                .offsets
                .last()
                .is_some_and(|&end| end as usize == self.features.len())
            && self.features.len() == self.coeffs.len()
            && self.offsets.windows(2).all(|w| w[0] <= w[1])
            && self
                .intercepts
                .iter()
                .chain(&self.coeffs)
                .all(|v| v.is_finite())
    }

    /// Structural validity against the owning tree's nodes.
    pub(crate) fn is_valid(&self, nodes: &[Node], n_features: usize) -> bool {
        self.is_consistent()
            && self.intercepts.len() == nodes.len()
            && self
                .offsets
                .windows(2)
                .zip(nodes)
                .all(|(w, node)| node.is_leaf() || w[0] == w[1])
            && self.features.iter().all(|&f| (f as usize) < n_features)
    }
}

/// Add `weight(t) · tree_t(row)` into `out[row * k + output(t)]` for every
/// tree index `t` in `range` (of `trees`), in ascending tree order per slot:
/// the prediction path for ensembles that contain linear leaves (the compact
/// forest stores constant leaf values only). `output` maps a tree to the
/// output it feeds (the model's tree layout).
pub(crate) fn accumulate_forest(
    trees: &[RegTree],
    range: std::ops::Range<usize>,
    output: impl Fn(usize) -> usize + Sync,
    rows: Rows<'_>,
    out: &mut [f32],
    k: usize,
    weight: impl Fn(usize) -> f32 + Sync,
) {
    let row = |(r, out_row): (usize, &mut [f32])| {
        for t in range.clone() {
            out_row[output(t)] += weight(t) * trees[t].predict_with(|f| rows.get(r, f as usize));
        }
    };
    if rows.n_rows() >= 1024 && rayon::current_num_threads() > 1 {
        out.par_chunks_mut(k)
            .with_min_len(256)
            .enumerate()
            .for_each(row);
    } else {
        out.chunks_mut(k).enumerate().for_each(row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deserialized leaf models are consistent: a term range past the term
    /// arrays (which `terms` would slice out of bounds) is refused.
    #[test]
    fn deserialization_refuses_inconsistent_arrays() {
        let doc = |offsets: &str| {
            format!(r#"{{"offsets":{offsets},"intercepts":[0.5],"features":[0],"coeffs":[2.0]}}"#)
        };
        let linear: LinearLeaves = serde_json::from_str(&doc("[0,1]")).unwrap();
        assert_eq!(linear.terms(0), (&[0u32][..], &[2.0][..]));
        for offsets in ["[0,2]", "[1,1]", "[0]", "[0,1,1]"] {
            assert!(
                serde_json::from_str::<LinearLeaves>(&doc(offsets)).is_err(),
                "{offsets}"
            );
        }
    }
}
