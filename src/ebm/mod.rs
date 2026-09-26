//! Explainable boosting machines (EBMs): cyclic GA²M boosting, per-term
//! shape functions, and (with Boulevard averaging) confidence bands on them.
//! Beyond XGBoost and opt-in: train with
//! [`BoosterKind::Ebm`](crate::config::BoosterKind::Ebm), read the shape
//! functions with [`shape_functions`], and, for an
//! [`ebm_boulevard`](crate::config::TrainingParams::ebm_boulevard) model,
//! their bands with [`EbmInference`](crate::inference::EbmInference).
//!
//! An EBM (Lou, Caruana & Gehrke, KDD 2012; Nori et al., *InterpretML*,
//! 2019) is a generalized additive model with pairwise interactions (a
//! GA²M): `g(E[y | x]) = β + Σ_j f_j(x_j) + Σ_{(j,k)} f_jk(x_j, x_k)`, each
//! *term* `f` learned as a sum of small trees that split only on the term's
//! features. The trees are ordinary [`RegTree`]s, so prediction, SHAP, and
//! every export work as for a `gbtree` model; [`EbmInfo`] records which term
//! each tree belongs to.
//!
//! # Training
//!
//! `num_boost_round` counts EBM rounds; each round grows one tree per term,
//! restricted to the term's features (the tree shape follows `max_depth`,
//! `max_leaves`, `grow_policy`, `min_child_weight`, `lambda`, `subsample`,
//! ...; InterpretML's defaults are about `max_leaves = 3` under loss-guided
//! growth, `eta = 0.01`, and thousands of rounds). The model counts every
//! tree as one iteration ([`BoostedModel::num_boost_rounds`] is the tree
//! count).
//!
//! - **Classic** (the default): cyclic boosting as in InterpretML. Within a
//!   round the terms take turns in feature order, each tree fitted to the
//!   gradients of the model so far (including the round's earlier terms)
//!   and added with learning rate `eta`. Any single-output objective works;
//!   the shapes are on the margin scale.
//! - **Outer bags** ([`ebm_outer_bags`](crate::config::TrainingParams::ebm_outer_bags)
//!   `= B`): each bag boosts every term on its own row sample
//!   ([`ebm_bag_fraction`](crate::config::TrainingParams::ebm_bag_fraction))
//!   and the model averages the bags (each bag's trees carry `1/B`). The
//!   bags train in parallel and are combined in bag order.
//! - **Interactions** ([`ebm_interactions`](crate::config::TrainingParams::ebm_interactions)
//!   `= k`): after the main effects, FAST (Lou, Caruana, Gehrke & Hooker,
//!   *Accurate intelligible models with pairwise interactions*, KDD 2013)
//!   ranks every pair of features by the best four-quadrant split of the
//!   main-effect model's gradients on their histogram bins (`max_bin`
//!   quantile bins; rows missing either feature sit out), scored as
//!   `Σ_q G_q² / (H_q + lambda) − G² / (H + lambda)`. The top `k` pairs
//!   (ties by feature order) become terms, boosted with the main effects
//!   frozen, as InterpretML does.
//! - **Boulevard** ([`ebm_boulevard`](crate::config::TrainingParams::ebm_boulevard)):
//!   Fang, Tan, Pipping & Hooker's inferable EBM (*Statistical Inference for
//!   Explainable Boosting Machines*, AISTATS 2026, Algorithm 1). Round `b`
//!   fits every term's tree to the same residuals
//!   `y − ȳ − Σ_t f_t^{(b−1)}(x)` (in parallel), centers it on the training
//!   rows, `t̃ = t − (1/n) Σ_i t(x_i)`, and averages it into its term,
//!   `f_t^{(b)} = ((b − 1)/b) f_t^{(b−1)} + (λ/b) t̃` with `λ = eta`; the
//!   model predicts `ȳ + ((1 + λ)/λ) Σ_t f_t^{(B)}`. Pairs run as a second
//!   Boulevard stage on the residuals of the first. See
//!   [`crate::inference`] for the limit and the bands.
//!
//! # Shape functions
//!
//! [`shape_functions`] merges every term's trees into one piecewise-constant
//! function on the grid the union of their thresholds cuts the term's
//! features into, missing values included ([`TermShape`]), centered to mean
//! zero over the training rows; the intercept collects `base_score` and the
//! terms' training means, so `intercept + Σ_t shape_t(x)` is the model's
//! margin.
//!
//! # Refusals
//!
//! `booster = ebm` needs one output, numerical features, and no
//! `init_model`, eval sets, or early stopping (the terms of one run are
//! fixed); it refuses `num_parallel_tree > 1`, column sampling, interaction
//! constraints (the terms fix every tree's features), linear leaves, the
//! reuse penalties, `process_type = update`, feature weights, and base
//! margins (the shapes and their centering assume the intercept alone).
//! With `ebm_boulevard` also everything Boulevard inference refuses
//! (non-squared-error objectives, row weights, L1 or clipped
//! leaves, quantized gradients, smoothed leaves, gradient-based sampling),
//! outer bags, and `base_score`. See
//! [`TrainingParams::validate`](crate::config::TrainingParams::validate).
//!
//! # Deviations from InterpretML
//!
//! No early stopping or inner bags; pairs use the main effects' `max_bin`
//! bins rather than a separate `max_interaction_bins`, FAST runs once on the
//! bag-averaged main effects rather than per bag, and the shapes are not
//! purified (Lengerich et al., AISTATS 2020): a pair term keeps whatever
//! main-effect part its trees fit.
//!
//! # Example
//!
//! ```
//! use hessboost::config::{BoosterKind, GrowPolicy};
//! use hessboost::ebm::shape_functions;
//! use hessboost::prelude::*;
//!
//! # fn main() -> Result<()> {
//! let n = 400;
//! let x: Vec<f32> = (0..n * 2).map(|i| ((i * 37) % 101) as f32 / 101.0).collect();
//! let y: Vec<f32> = x.chunks(2).map(|r| (6.0 * r[0]).sin() + r[1] * r[1]).collect();
//! let dtrain = DMatrix::from_dense(&x, n, 2)?.with_labels(&y)?;
//! let params = TrainingParams::builder()
//!     .booster(BoosterKind::Ebm)
//!     .eta(0.1)
//!     .grow_policy(GrowPolicy::LossGuide)
//!     .max_leaves(3)
//!     .build()?;
//! let model = train(&params, &dtrain, 50)?;
//! let shapes = shape_functions(&model)?;
//! assert_eq!(shapes.terms.len(), 2);
//! let margin = shapes.intercept + shapes.terms[0].value(&[0.3])? + shapes.terms[1].value(&[0.5])?;
//! let direct = model.predict(&DMatrix::from_dense(&[0.3, 0.5], 1, 2)?)?[0];
//! assert!((margin - f64::from(direct)).abs() < 1e-4);
//! # Ok(())
//! # }
//! ```
//!
//! [`RegTree`]: crate::tree::RegTree

pub(crate) mod grid;

use serde::{Deserialize, Serialize};

use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::tree::RegTree;
use grid::TermGrid;

/// How a `booster = ebm` model was trained: the features of every term and
/// which term each tree belongs to, recorded by training
/// ([`BoostedModel::ebm`]) and read by [`shape_functions`] and
/// [`EbmInference`](crate::inference::EbmInference).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EbmInfo {
    /// The features of each term: one main term per feature (ascending),
    /// then the pair terms in FAST order (each pair ascending).
    pub terms: Vec<Vec<u32>>,
    /// The term of each tree.
    pub tree_terms: Vec<u32>,
    /// The mean of each term's raw contribution over the training rows,
    /// which [`shape_functions`] moves into the intercept.
    pub term_means: Vec<f64>,
    /// The Boulevard settings of an
    /// [`ebm_boulevard`](crate::config::TrainingParams::ebm_boulevard) fit;
    /// `None` for a classic EBM.
    #[serde(deserialize_with = "Option::deserialize")]
    pub boulevard: Option<EbmBoulevard>,
}

/// The settings of a Boulevard EBM its inference reads.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EbmBoulevard {
    /// The learning rate `λ` (`eta`).
    pub learning_rate: f64,
    /// The row subsample ratio `ξ` (`subsample`).
    pub subsample: f64,
    /// The L2 leaf penalty (`lambda`).
    pub reg_lambda: f64,
}

impl EbmInfo {
    /// Check the record against `model`: one term per tree, terms of one or
    /// two distinct features, every tree splitting numerically on its
    /// term's features only, and (for a Boulevard fit) the settings'
    /// ranges.
    pub(crate) fn validate(&self, model: &BoostedModel) -> Result<()> {
        let fail = |reason: String| {
            Err(HessboostError::model_format(format!(
                "invalid EBM record: {reason}"
            )))
        };
        if model.n_outputs() != 1
            || model.has_vector_leaves()
            || model.has_non_unit_tree_weights()
            || model.num_parallel_tree() != 1
            || model.trees().iter().any(|t| t.linear_leaves().is_some())
        {
            return fail("only single-output scalar tree ensembles are EBMs".into());
        }
        if self.tree_terms.len() != model.num_trees() {
            return fail(format!(
                "{} tree terms for {} trees",
                self.tree_terms.len(),
                model.num_trees()
            ));
        }
        if self.term_means.len() != self.terms.len() {
            return fail(format!(
                "{} term means for {} terms",
                self.term_means.len(),
                self.terms.len()
            ));
        }
        if self.term_means.iter().any(|m| !m.is_finite()) {
            return fail("term means must be finite".into());
        }
        for (t, features) in self.terms.iter().enumerate() {
            let sorted = features.windows(2).all(|w| w[0] < w[1]);
            if !(1..=2).contains(&features.len())
                || !sorted
                || features.iter().any(|&f| f as usize >= model.n_features())
            {
                return fail(format!(
                    "term {t} must name one or two ascending features below {}",
                    model.n_features()
                ));
            }
        }
        for (i, (&term, tree)) in self.tree_terms.iter().zip(model.trees()).enumerate() {
            let Some(features) = self.terms.get(term as usize) else {
                return fail(format!(
                    "tree {i} names term {term} of {}",
                    self.terms.len()
                ));
            };
            if tree
                .nodes()
                .iter()
                .any(|n| !n.is_leaf() && (n.is_categorical || !features.contains(&n.split_feature)))
            {
                return fail(format!(
                    "tree {i} splits outside its term's features or categorically"
                ));
            }
        }
        if let Some(b) = &self.boulevard {
            if !(b.learning_rate > 0.0 && b.learning_rate <= 1.0) {
                return fail("learning_rate must be in (0, 1]".into());
            }
            if !(b.subsample > 0.0 && b.subsample <= 1.0) {
                return fail("subsample must be in (0, 1]".into());
            }
            if !(b.reg_lambda.is_finite() && b.reg_lambda >= 0.0) {
                return fail("reg_lambda must be finite and >= 0".into());
            }
            if model.objective() != "reg:squarederror" {
                return fail("a Boulevard EBM is a reg:squarederror model".into());
            }
        }
        Ok(())
    }

    /// The trees of term `term`, in model order.
    pub(crate) fn term_trees<'m>(&self, model: &'m BoostedModel, term: usize) -> Vec<&'m RegTree> {
        self.tree_terms
            .iter()
            .zip(model.trees())
            .filter(|&(&t, _)| t as usize == term)
            .map(|(_, tree)| tree)
            .collect()
    }

    /// The grid of term `term`.
    pub(crate) fn grid(&self, model: &BoostedModel, term: usize) -> TermGrid {
        TermGrid::new(&self.term_trees(model, term), &self.terms[term])
    }

    /// Every term's mean raw contribution over the rows of `data`.
    pub(crate) fn term_means_on(&self, model: &BoostedModel, data: &DMatrix) -> Vec<f64> {
        let n = data.n_rows();
        (0..self.terms.len())
            .map(|t| {
                let grid = self.grid(model, t);
                let values = self.raw_values(model, t, &grid);
                let sum: f64 = (0..n).map(|row| values[grid.cell_of_row(data, row)]).sum();
                sum / n as f64
            })
            .collect()
    }

    /// The raw (uncentered) cell values of term `term` on `grid`.
    pub(crate) fn raw_values(
        &self,
        model: &BoostedModel,
        term: usize,
        grid: &TermGrid,
    ) -> Vec<f64> {
        let trees = self.term_trees(model, term);
        let mut leaf_values = Vec::with_capacity(grid.leaves.len());
        for (t, tree) in trees.iter().enumerate() {
            for leaf in &grid.leaves[grid.leaf_start[t]..grid.leaf_start[t + 1]] {
                leaf_values.push(f64::from(tree.node(leaf.node as usize).leaf_value));
            }
        }
        grid.paint(|i| leaf_values[i])
    }
}

/// Every term's shape function and the intercept of an EBM
/// ([`shape_functions`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ShapeFunctions {
    /// `base_score` plus every term's training mean: the margin of a point
    /// at which every shape is zero.
    pub intercept: f64,
    /// One shape per term, in [`EbmInfo::terms`] order.
    pub terms: Vec<TermShape>,
}

/// One term's shape function: piecewise constant on the grid its trees'
/// thresholds cut its features into, centered to mean zero over the
/// training rows.
///
/// Along feature `a` (`features[a]`) the `edges[a]` `e_0 < … < e_{m−2}`
/// give `m + 1` cells: the intervals `(−∞, e_0), [e_0, e_1), …,
/// [e_{m−2}, ∞)` and then the cell of missing values. `values` holds one
/// value per grid cell, row-major (the first feature's cell slowest), so a
/// main term has `edges[0].len() + 2` values and a pair
/// `(edges[0].len() + 2) × (edges[1].len() + 2)`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TermShape {
    /// The term's features (one, or two ascending).
    pub features: Vec<u32>,
    /// The thresholds along each feature.
    pub edges: Vec<Vec<f32>>,
    /// The shape's value on every cell.
    pub values: Vec<f64>,
}

impl TermShape {
    /// The index into [`values`](Self::values) of the cell holding the
    /// feature values `x` (one per feature of the term; NaN is missing).
    ///
    /// # Errors
    ///
    /// [`HessboostError::DimensionMismatch`] unless `x` has one value per
    /// feature of the term.
    pub fn cell(&self, x: &[f32]) -> Result<usize> {
        if x.len() != self.features.len() {
            return Err(HessboostError::dimension_mismatch(
                "term feature values",
                self.features.len(),
                x.len(),
            ));
        }
        let cell = |a: usize| {
            let edges = &self.edges[a];
            if x[a].is_nan() {
                edges.len() + 1
            } else {
                edges.partition_point(|&e| e <= x[a])
            }
        };
        Ok(match self.edges.len() {
            1 => cell(0),
            _ => cell(0) * (self.edges[1].len() + 2) + cell(1),
        })
    }

    /// The shape's value at the feature values `x` (as [`cell`](Self::cell)).
    ///
    /// # Errors
    ///
    /// As [`cell`](Self::cell).
    pub fn value(&self, x: &[f32]) -> Result<f64> {
        Ok(self.values[self.cell(x)?])
    }
}

/// The shape functions and intercept of `model`, an EBM
/// ([`BoosterKind::Ebm`](crate::config::BoosterKind::Ebm)): every term's
/// trees merged into one piecewise-constant function of its features (see
/// [`TermShape`]), missing values included. `intercept + Σ_t shape_t(x)` is
/// the model's margin (up to `f32` rounding of the tree sum).
///
/// # Errors
///
/// [`HessboostError::InvalidParameter`] when `model` is not an EBM
/// ([`BoostedModel::ebm`] is `None`).
pub fn shape_functions(model: &BoostedModel) -> Result<ShapeFunctions> {
    let info = model.ebm().ok_or_else(|| {
        HessboostError::invalid_param("model", "not an EBM: train it with `booster = ebm`")
    })?;
    let terms = (0..info.terms.len())
        .map(|t| {
            let grid = info.grid(model, t);
            let mean = info.term_means[t];
            let values = info
                .raw_values(model, t, &grid)
                .into_iter()
                .map(|v| v - mean)
                .collect();
            TermShape {
                features: info.terms[t].clone(),
                edges: grid.axes.iter().map(|a| a.edges.clone()).collect(),
                values,
            }
        })
        .collect();
    Ok(ShapeFunctions {
        intercept: f64::from(model.base_scores()[0]) + info.term_means.iter().sum::<f64>(),
        terms,
    })
}
