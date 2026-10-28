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
//! With [`LinearTree::clamp`](crate::config::LinearTree::clamp)
//! (`linear_clamp`, which LightGBM lacks), every fitted leaf also records
//! each term's input range: the smallest and largest value of its feature
//! among the rows the fit read with a nonzero gradient or Hessian (rows with
//! both zero, such as zero-weight rows, add nothing to `XᵀHX` or `Xᵀg`).
//! Prediction clamps a row's value into the range before multiplying it by
//! the slope, so beyond the values its fit saw a leaf stays flat instead of
//! extrapolating. The clamp compares rather than calling `min`/`max`: a value
//! inside the range, signed zeros included, passes through bit for bit, and
//! the result never depends on how a platform orders `-0.0` and `0.0`.
//!
//! Learning-rate shrinkage scales intercepts and slopes together with the
//! constant leaf values ([`RegTree::scale_leaves`]), leaving the input
//! ranges as they are. Split finding is unchanged, so monotone constraints
//! bound only the constant values, not the fitted slopes (as in LightGBM).
//! Numerical features should be on comparable scales, since the slope
//! penalty is not scale invariant.

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
/// those features is missing. When the models clamp their inputs
/// ([`ranges`](Self::ranges) is `Some`), each `x` is first clamped to its
/// term's `[lower, upper]`. Internal nodes hold no terms.
///
/// Serde deserialization checks that the arrays are consistent (one
/// intercept and one term range per node, ranges inside the term arrays,
/// finite values, one ordered input range per term when there are any) and
/// refuses them otherwise; the owning [`RegTree`] checks them against its
/// nodes.
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
    /// Every term's input range `[lower, upper]`, aligned with `features`,
    /// when the models clamp their inputs (`linear_clamp`); `None` when they
    /// do not (LightGBM's leaves).
    ranges: Option<Vec<[f32; 2]>>,
}

/// The serialized fields of [`LinearLeaves`] (same names and layout), before
/// validation. A serialized [`RegTree`] reads its leaf models through it, so
/// the tree checks them against its nodes in one place. `ranges` may be
/// absent: files written before clamping existed hold unclamped models.
#[derive(Deserialize)]
pub(crate) struct UncheckedLinearLeaves {
    offsets: Vec<u32>,
    intercepts: Vec<f64>,
    features: Vec<u32>,
    coeffs: Vec<f64>,
    #[serde(default)]
    ranges: Option<Vec<[f32; 2]>>,
}

impl UncheckedLinearLeaves {
    /// The leaf models as stored, unvalidated: the caller validates them
    /// ([`LinearLeaves::is_valid`]).
    pub(crate) fn into_unchecked(self) -> LinearLeaves {
        LinearLeaves::from_parts(
            self.offsets,
            self.intercepts,
            self.features,
            self.coeffs,
            self.ranges,
        )
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
        let range = self.term_range(node);
        (&self.features[range.clone()], &self.coeffs[range])
    }

    /// The input range `[lower, upper]` of each of leaf `node`'s
    /// [`terms`](Self::terms): the smallest and largest value of the term's
    /// feature among the rows the leaf's model was fit on, which prediction
    /// clamps the row's value to. `None` when the models do not clamp their
    /// inputs (trained without
    /// [`LinearTree::clamp`](crate::config::LinearTree::clamp), or imported
    /// from LightGBM).
    #[inline]
    pub fn ranges(&self, node: usize) -> Option<&[[f32; 2]]> {
        let ranges = self.ranges.as_deref()?;
        Some(&ranges[self.term_range(node)])
    }

    /// Where leaf `node`'s terms sit in the term arrays.
    #[inline]
    fn term_range(&self, node: usize) -> std::ops::Range<usize> {
        self.offsets[node] as usize..self.offsets[node + 1] as usize
    }

    /// Output of leaf `node` for a row read through `get` (`None` = missing):
    /// the linear model over the row's values (each clamped to its term's
    /// input range, when the models have them), or `constant` when one of
    /// its features is missing.
    #[inline]
    pub(crate) fn predict(
        &self,
        node: usize,
        constant: f32,
        get: impl Fn(u32) -> Option<f32>,
    ) -> f32 {
        let (features, coeffs) = self.terms(node);
        let mut out = self.intercepts[node];
        match self.ranges(node) {
            None => {
                for (&f, &c) in features.iter().zip(coeffs) {
                    let Some(x) = get(f) else {
                        return constant;
                    };
                    out += c * f64::from(x);
                }
            }
            Some(ranges) => {
                for ((&f, &c), &[lower, upper]) in features.iter().zip(coeffs).zip(ranges) {
                    let Some(x) = get(f) else {
                        return constant;
                    };
                    out += c * f64::from(clamp(x, lower, upper));
                }
            }
        }
        out as f32
    }

    /// Replace every intercept and slope `v` by `map(v)` (shrinkage, model
    /// averaging).
    pub(crate) fn map_coefficients(&mut self, map: impl Fn(f64) -> f64) {
        for v in self.intercepts.iter_mut().chain(&mut self.coeffs) {
            *v = map(*v);
        }
    }

    /// The stored arrays, as [`LinearLeaves::from_parts`] takes them.
    pub(crate) fn parts(&self) -> LinearParts<'_> {
        LinearParts {
            offsets: &self.offsets,
            intercepts: &self.intercepts,
            features: &self.features,
            coeffs: &self.coeffs,
            ranges: self.ranges.as_deref(),
        }
    }

    /// Assemble leaf models from their stored arrays (`ranges`: every
    /// term's input range, or `None` for unclamped models); the owning tree
    /// checks them with [`LinearLeaves::is_valid`].
    pub(crate) fn from_parts(
        offsets: Vec<u32>,
        intercepts: Vec<f64>,
        features: Vec<u32>,
        coeffs: Vec<f64>,
        ranges: Option<Vec<[f32; 2]>>,
    ) -> Self {
        LinearLeaves {
            offsets,
            intercepts,
            features,
            coeffs,
            ranges,
        }
    }

    /// Consistency of the arrays on their own: one intercept and one
    /// non-decreasing term range per node, the ranges covering `features`
    /// and `coeffs` exactly, finite values, and, when the models clamp, one
    /// finite input range `lower <= upper` per term.
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
            && self.ranges.as_ref().is_none_or(|ranges| {
                ranges.len() == self.features.len()
                    && ranges.iter().all(|&[lower, upper]| {
                        lower.is_finite() && upper.is_finite() && lower <= upper
                    })
            })
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

/// The stored arrays of one tree's [`LinearLeaves`] ([`LinearLeaves::parts`]),
/// as the native format writes them.
pub(crate) struct LinearParts<'a> {
    pub(crate) offsets: &'a [u32],
    pub(crate) intercepts: &'a [f64],
    pub(crate) features: &'a [u32],
    pub(crate) coeffs: &'a [f64],
    /// Every term's input range, `None` for unclamped models.
    pub(crate) ranges: Option<&'a [[f32; 2]]>,
}

/// `x` clamped into `[lower, upper]` by comparisons: a value inside the
/// range (a zero of either sign against a zero bound too) is returned bit
/// for bit, anything below or above it (infinities included) is the nearer
/// bound, and `NaN` passes through.
#[inline]
fn clamp(x: f32, lower: f32, upper: f32) -> f32 {
    if x < lower {
        lower
    } else if x > upper {
        upper
    } else {
        x
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
    /// arrays (which `terms` would slice out of bounds) is refused, and so
    /// are input ranges that do not give every term one ordered range.
    #[test]
    fn deserialization_refuses_inconsistent_arrays() {
        let doc = |offsets: &str| {
            format!(r#"{{"offsets":{offsets},"intercepts":[0.5],"features":[0],"coeffs":[2.0]}}"#)
        };
        let linear: LinearLeaves = serde_json::from_str(&doc("[0,1]")).unwrap();
        assert_eq!(linear.terms(0), (&[0u32][..], &[2.0][..]));
        // Files written before input ranges existed hold unclamped models.
        assert_eq!(linear.ranges(0), None);
        for offsets in ["[0,2]", "[1,1]", "[0]", "[0,1,1]"] {
            assert!(
                serde_json::from_str::<LinearLeaves>(&doc(offsets)).is_err(),
                "{offsets}"
            );
        }
        let ranged = |ranges: &str| {
            let doc = doc("[0,1]");
            format!(r#"{},"ranges":{ranges}}}"#, &doc[..doc.len() - 1])
        };
        let linear: LinearLeaves = serde_json::from_str(&ranged("[[-1.5,2.0]]")).unwrap();
        assert_eq!(linear.ranges(0), Some(&[[-1.5f32, 2.0]][..]));
        let unclamped: LinearLeaves = serde_json::from_str(&ranged("null")).unwrap();
        assert_eq!(unclamped.ranges(0), None);
        for ranges in ["[]", "[[0.0,1.0],[0.0,1.0]]", "[[1.0,0.0]]"] {
            assert!(
                serde_json::from_str::<LinearLeaves>(&ranged(ranges)).is_err(),
                "{ranges}"
            );
        }
        for bad in [[f32::NAN, 1.0], [0.0, f32::INFINITY]] {
            let linear = LinearLeaves::from_parts(
                vec![0, 1],
                vec![0.5],
                vec![0],
                vec![2.0],
                Some(vec![bad]),
            );
            assert!(!linear.is_consistent(), "{bad:?}");
        }
    }

    /// A clamped leaf keeps its model inside its input ranges and stays flat
    /// outside them, infinities included; a missing feature still gives the
    /// constant.
    #[test]
    fn clamped_leaves_stay_flat_outside_their_ranges() {
        // Leaf 0: 1 + 2·x0 − x1 with x0 in [0, 1], x1 in [−2, 2].
        let linear: LinearLeaves = serde_json::from_str(
            r#"{"offsets":[0,2],"intercepts":[1.0],"features":[0,1],"coeffs":[2.0,-1.0],
                "ranges":[[0.0,1.0],[-2.0,2.0]]}"#,
        )
        .unwrap();
        let at = |x0: f32, x1: Option<f32>| {
            linear.predict(0, -9.0, |f| if f == 0 { Some(x0) } else { x1 })
        };
        assert_eq!(at(0.5, Some(1.0)), 1.0);
        assert_eq!(at(100.0, Some(1.0)), 2.0);
        assert_eq!(at(-100.0, Some(-50.0)), 3.0);
        assert_eq!(at(f32::INFINITY, Some(f32::NEG_INFINITY)), 5.0);
        assert_eq!(at(0.5, None), -9.0);
    }

    /// Clamping compares: values inside the range keep their bits (a zero
    /// of either sign against zero bounds too), values outside take the
    /// nearer bound.
    #[test]
    fn clamping_keeps_in_range_bits() {
        for (x, lower, upper, want) in [
            (0.25f32, 0.0, 1.0, 0.25f32),
            (-0.0, 0.0, 1.0, -0.0),
            (0.0, -1.0, -0.0, 0.0),
            (-3.0, -1.0, 1.0, -1.0),
            (7.0, -1.0, 1.0, 1.0),
        ] {
            assert_eq!(
                clamp(x, lower, upper).to_bits(),
                want.to_bits(),
                "{x} in [{lower}, {upper}]"
            );
        }
    }
}
