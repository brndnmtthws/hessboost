//! Explainable boosting machines (EBMs): cyclic GA²M boosting, per-term
//! shape functions, and (with Boulevard averaging) confidence bands on them.
//! Beyond XGBoost and opt-in: train with
//! [`BoosterKind::Ebm`](crate::config::BoosterKind::Ebm), read the shape
//! functions with [`shape_functions`], and, for an
//! [`Ebm::boulevard`](crate::config::Ebm::boulevard) model,
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
//!   the shapes are on the margin scale. Categorical features
//!   ([`DMatrix::with_feature_types`](crate::data::DMatrix::with_feature_types))
//!   get the builders' native set-membership splits.
//! - **Outer bags** ([`Ebm::outer_bags`](crate::config::Ebm::outer_bags)
//!   `= B`): each bag boosts every term on its own row sample
//!   ([`Ebm::bag_fraction`](crate::config::Ebm::bag_fraction))
//!   and the model averages the bags (each bag's trees carry `1/B`). The
//!   bags train in parallel and are combined in bag order. Each tree
//!   subsamples its bag's rows (`subsample`, by class under
//!   [`BalancedBagging`](crate::config::BalancedBagging), or by query under
//!   [`QueryBagging`](crate::config::QueryBagging)).
//! - **Early stopping** ([`Ebm::early_stopping`](crate::config::Ebm::early_stopping),
//!   an [`EbmEarlyStopping`](crate::config::EbmEarlyStopping)):
//!   InterpretML's rule. Every bag scores the rows it does not train on
//!   after every tree, stops a stage once the last `rounds × terms` trees
//!   failed to beat its best earlier score by
//!   its tolerance (relative), and keeps its trees up to its best score, so the bags
//!   stop at different rounds and `num_boost_round` only caps them.
//! - **Interactions** ([`Ebm::interactions`](crate::config::Ebm::interactions)
//!   `= k`): after the main effects, FAST (Lou, Caruana, Gehrke & Hooker,
//!   *Accurate intelligible models with pairwise interactions*, KDD 2013)
//!   ranks every pair of features by the best four-quadrant split of the
//!   main-effect model's gradients on their histogram bins (`max_bin`
//!   quantile bins, one bin per category for categorical features, ordered
//!   by the category's mean gradient; rows missing either feature sit
//!   out), scored as
//!   `Σ_q G_q² / (H_q + lambda) − G² / (H + lambda)`. The top `k` pairs
//!   (ties by feature order) become terms, boosted with the main effects
//!   frozen, as InterpretML does.
//! - **Boulevard** ([`Ebm::boulevard`](crate::config::Ebm::boulevard)):
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
//! function on the grid their splits cut the term's features into: the
//! union of the thresholds of a numerical feature, one cell per category a
//! split sends left (plus one for every other category) of a categorical
//! one, and a missing-value cell on each ([`TermShape`], [`TermAxis`]),
//! centered to mean
//! zero over the training rows; the intercept collects `base_score` and the
//! terms' training means, so `intercept + Σ_t shape_t(x)` is the model's
//! margin.
//!
//! # Refusals
//!
//! `booster = ebm` needs one output and no `init_model`, eval sets, or
//! `Trainer::early_stopping_rounds` (the terms of one run are fixed; stop
//! each bag with [`Ebm::early_stopping`](crate::config::Ebm::early_stopping),
//! flat `ebm_early_stopping_rounds`, instead); it refuses `num_parallel_tree > 1`, column sampling, interaction
//! constraints (the terms fix every tree's features), linear leaves, the
//! reuse penalties, `process_type = update`, feature weights, and base
//! margins (the shapes and their centering assume the intercept alone).
//! With `ebm_boulevard` also everything Boulevard inference refuses
//! (non-squared-error objectives, row weights, L1 or clipped
//! leaves, quantized gradients, smoothed leaves, gradient-based sampling),
//! outer bags, early stopping, and `base_score`. See
//! [`TrainingParams::validate`](crate::config::TrainingParams::validate).
//!
//! # Deviations from InterpretML
//!
//! No inner bags, smoothing rounds, or greedy rounds, and early stopping
//! keeps each bag's best model per stage (InterpretML's stopping rule and
//! tolerance) without its per-step greedy term selection; pairs use the
//! main effects' `max_bin`
//! bins rather than a separate `max_interaction_bins`, FAST runs once on the
//! bag-averaged main effects rather than per bag, and the shapes are not
//! purified (Lengerich et al., AISTATS 2020): a pair term keeps whatever
//! main-effect part its trees fit.
//!
//! # Example
//!
//! ```
//! use hessboost::config::{BoosterKind, Ebm, GrowPolicy};
//! use hessboost::ebm::shape_functions;
//! use hessboost::prelude::*;
//!
//! # fn main() -> Result<()> {
//! let n = 400;
//! let x: Vec<f32> = (0..n * 2).map(|i| ((i * 37) % 101) as f32 / 101.0).collect();
//! let y: Vec<f32> = x.chunks(2).map(|r| (6.0 * r[0]).sin() + r[1] * r[1]).collect();
//! let dtrain = DMatrix::from_dense(&x, n, 2)?.with_labels(&y)?;
//! let params = TrainingParams::builder()
//!     .booster(BoosterKind::Ebm(Ebm::default()))
//!     .eta(0.1)
//!     .grow_policy(GrowPolicy::LossGuide)
//!     .max_leaves(3)
//!     .build()?;
//! let model = train(&params, &dtrain, 50)?;
//! let shapes = shape_functions(&model)?;
//! assert_eq!(shapes.terms.len(), 2);
//! let margin = shapes.intercept + shapes.terms[0].value(&[0.3])? + shapes.terms[1].value(&[0.5])?;
//! let direct = model.predict(&DMatrix::from_dense(&[0.3, 0.5], 1, 2)?, Iterations::Best)?;
//! let direct = *direct.get(0, 0).expect("one row, one output");
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
use crate::objective::Objective;
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
    /// [`Ebm::boulevard`](crate::config::Ebm::boulevard) fit;
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
        // Per term and axis: whether its splits are categorical, once seen.
        let mut categorical: Vec<[Option<bool>; 2]> = vec![[None; 2]; self.terms.len()];
        for (i, (&term, tree)) in self.tree_terms.iter().zip(model.trees()).enumerate() {
            let Some(features) = self.terms.get(term as usize) else {
                return fail(format!(
                    "tree {i} names term {term} of {}",
                    self.terms.len()
                ));
            };
            for n in tree.nodes().iter().filter(|n| !n.is_leaf()) {
                let Some(a) = features.iter().position(|&f| f == n.split_feature) else {
                    return fail(format!("tree {i} splits outside its term's features"));
                };
                let kind = &mut categorical[term as usize][a];
                if *kind.get_or_insert(n.is_categorical) != n.is_categorical {
                    return fail(format!(
                        "term {term} splits feature {} both numerically and categorically",
                        n.split_feature
                    ));
                }
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
            if !model
                .objective()
                .built_in()
                .is_some_and(Objective::is_unweighted_squared_error)
            {
                return fail("a Boulevard EBM is a reg:squarederror model".into());
            }
            self.validate_stages().or_else(fail)?;
        }
        Ok(())
    }

    /// The layout a Boulevard EBM's inference and refit read: the main
    /// terms first, then the pairs, and each stage's trees contiguous,
    /// round by round, one per term of the stage in term order.
    fn validate_stages(&self) -> std::result::Result<(), String> {
        let mains = self.terms.iter().take_while(|t| t.len() == 1).count();
        if self.terms[mains..].iter().any(|t| t.len() == 1) {
            return Err("a Boulevard EBM lists its main terms before its pairs".into());
        }
        let mut at = 0;
        for stage in [0..mains, mains..self.terms.len()] {
            let k = stage.len();
            let trees = self.tree_terms[at..]
                .iter()
                .take_while(|&&t| stage.contains(&(t as usize)))
                .count();
            if k == 0 {
                continue;
            }
            let round_robin = self.tree_terms[at..at + trees]
                .iter()
                .enumerate()
                .all(|(i, &t)| t as usize == stage.start + i % k);
            if trees % k != 0 || !round_robin {
                return Err(format!(
                    "a Boulevard EBM stage must hold whole rounds of its {k} terms in term order"
                ));
            }
            at += trees;
        }
        if at != self.tree_terms.len() {
            return Err(
                "a Boulevard EBM's trees must be its main stage, then its pair stage".into(),
            );
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

/// The cells of one feature of a term ([`TermShape::axes`]). Either way
/// the last cell holds missing values.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TermAxis {
    /// A numerical feature: its thresholds `e_0 < … < e_{m−2}` give the
    /// intervals `(−∞, e_0), [e_0, e_1), …, [e_{m−2}, ∞)`, then the
    /// missing cell (`edges.len() + 2` cells).
    #[non_exhaustive]
    Numeric {
        /// The sorted thresholds the term's trees split at.
        edges: Vec<f32>,
    },
    /// A categorical feature: one cell per category some split of the
    /// term sends left (ascending codes), one for every other category
    /// (including categories every split sends right), then
    /// the missing cell (`categories.len() + 2` cells). Values are
    /// category codes as [`DMatrix`] stores them (truncated to integers, as
    /// the trees route them).
    #[non_exhaustive]
    Categorical {
        /// The category codes the term's trees split on, ascending.
        categories: Vec<u32>,
    },
}

impl TermAxis {
    /// Number of cells, the missing cell included.
    pub fn cells(&self) -> usize {
        match self {
            TermAxis::Numeric { edges } => edges.len() + 2,
            TermAxis::Categorical { categories } => categories.len() + 2,
        }
    }

    /// The cell of `value` (`None` or NaN: missing).
    pub(crate) fn cell(&self, value: Option<f32>) -> usize {
        match value {
            Some(x) if !x.is_nan() => match self {
                TermAxis::Numeric { edges } => edges.partition_point(|&e| e <= x),
                TermAxis::Categorical { categories } => {
                    // The trees' own truncation of the code.
                    let code = x as u32;
                    categories.binary_search(&code).unwrap_or(categories.len())
                }
            },
            _ => self.cells() - 1,
        }
    }
}

/// One term's shape function: piecewise constant on the grid its trees cut
/// its features into, centered to mean zero over the training rows.
///
/// Feature `a` of the term ([`features`](Self::features)) has the cells of
/// [`axes`](Self::axes)`[a]`; [`values`](Self::values) holds one value per
/// grid cell, row-major (the first feature's cell slowest), so a main term
/// has `axes[0].cells()` values and a pair `axes[0].cells() ×
/// axes[1].cells()`. Built only by [`shape_functions`], so the three always
/// agree.
#[derive(Debug, Clone, PartialEq)]
pub struct TermShape {
    features: Vec<u32>,
    axes: Vec<TermAxis>,
    values: Vec<f64>,
}

impl TermShape {
    /// The term's features (one, or two ascending).
    pub fn features(&self) -> &[u32] {
        &self.features
    }

    /// The cells along each feature.
    pub fn axes(&self) -> &[TermAxis] {
        &self.axes
    }

    /// The shape's value on every cell, row-major.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The index into [`values`](Self::values) of the cell holding the
    /// feature values `x` (one per feature of the term; NaN is missing, a
    /// category no split names is the axis's "other" cell).
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
        Ok(self.axes.iter().zip(x).fold(0, |cell, (axis, &v)| {
            cell * axis.cells() + axis.cell(Some(v))
        }))
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
                axes: grid.axes.iter().map(|a| a.kind.clone()).collect(),
                values,
            }
        })
        .collect();
    Ok(ShapeFunctions {
        intercept: f64::from(model.base_scores()[0]) + info.term_means.iter().sum::<f64>(),
        terms,
    })
}
