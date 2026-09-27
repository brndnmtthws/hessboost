//! In-place data addition and deletion for trained GBDT models
//! (incremental and decremental learning, machine unlearning; beyond
//! XGBoost, opt-in).
//!
//! An [`OnlineModel`] keeps a model together with its training data and,
//! when approximate updates are enabled, the node statistics that let it
//! [`update`](OnlineModel::update) the model after rows are added to or
//! deleted from that data without retraining from scratch. It follows Lin,
//! Chung, Lao and Zhao, *Online Gradient Boosting Decision Tree: In-Place
//! Updates for Efficient Adding/Deleting Data* (arXiv 2502.01634; reference
//! code <https://github.com/huawei-lin/InplaceOnlineGBDT>), which extends
//! their decremental method (*Machine Unlearning in Gradient Boosting
//! Decision Trees*, KDD 2023) to additions.
//!
//! # Method
//!
//! The iterations are walked in order. In each tree the nodes are visited
//! from the root down; the rows that changed (added, deleted, or whose
//! gradient is refreshed, below) update the node's cached per-bin gradient
//! histogram, and the node's current split is ranked among every candidate
//! of the updated histogram with the histogram builder's own scoring. With
//! [`OnlineParams::approximate`]'s tolerance `σ`, a split ranked within the top
//! `max(1, ⌊σ · candidates⌋)` is kept (the paper's split robustness
//! tolerance); otherwise the subtree under the node is regrown with the
//! histogram builder on the rows now reaching it. Leaves whose statistics
//! changed get their weights recomputed.
//!
//! Gradients are refreshed lazily (the paper's adaptive lazy update): rows
//! that were added, or that reached a regrown subtree in an earlier tree,
//! get their gradients recomputed from the updated model; every other row
//! keeps the gradient cached for it, since its margin only drifts by the
//! (small) changes of the leaf weights it reaches. The histogram's bin
//! boundaries and the model's intercept stay those of the original
//! training, so an added value at or above a feature's top bin boundary
//! (beyond the training data's range) is refused in this mode.
//!
//! # Exactness
//!
//! [`OnlineParams::exact`] is the exact mode: every node's split must still be the
//! best one on recomputed statistics, bin boundaries and intercept, which is
//! by definition what retraining produces, so the exact mode defers every
//! tree to the builder: the result equals [`train`](super::train) on
//! [`OnlineModel::data`] bit for bit, at about the cost of retraining (a
//! reference to measure the approximate mode against, and the answer when
//! unlearning must be exact). [`OnlineParams::approximate`] is approximate: the kept
//! splits, the fixed bins and intercept, and the lazily refreshed gradients
//! make the model differ from a retrain, by a gap that grows with the
//! fraction of rows changed and with `σ`, in exchange for touching only the
//! changed rows and the regrown subtrees.
//!
//! # Supported configurations
//!
//! Updating is sound only where retraining the same parameters on the
//! updated data is deterministic in the data alone and every node's
//! subtree depends on the node's rows only: `gbtree` with `tree_method =
//! hist` (or `auto`), depth-wise growth with a positive `max_depth` and no
//! `max_leaves`, one output, `num_parallel_tree = 1`, no row or column
//! sampling (a retrain draws its samples sequentially over the rows, so
//! they would change with any added or deleted row), no monotone or
//! interaction constraints, none of the beyond-XGBoost split options
//! (`extra_trees`, `path_smooth`, `linear_tree`, quantized gradients, reuse
//! penalties), CPU, and a built-in objective whose gradients are per row
//! and whose leaves are plain Newton steps (not ranking, `survival:cox`,
//! `reg:absoluteerror`, `reg:quantileerror`, or a custom loss). The data
//! may not carry weights, base margins, groups, label bounds, or feature
//! weights, and the approximate mode needs numerical features. Everything
//! else is refused.
//!
//! # Memory
//!
//! The approximate mode caches, per tree, one gradient pair per row and one
//! histogram (`total_bins` pairs of `f64`) per internal node: `trees × (8 ·
//! rows + 16 · total_bins · internal nodes)` bytes. An update works on a
//! copy of it, swapped in on success, so an abandoned update restores the
//! exact state it started from; the copy briefly doubles that memory.
//!
//! # Accuracy and speed
//!
//! On 20,000-row synthetic tasks (Friedman #1 regression, its thresholded
//! classification, and a 30-feature variant; 100 trees of depth 6, `σ =
//! 0.1`), updates of 0.1% and 1% of the rows ran 1.3–4.8x faster than
//! retraining, with test losses within about 0.5% of the retrained
//! model's; at 5% they were slower than retraining (regrown subtrees
//! refresh most rows). Deleting rows raises the model's loss on them, but
//! by a fraction of what retraining does (e.g. regression RMSE on 200
//! deleted rows 0.877 → 0.903, retrained 1.080): the approximate mode
//! forgets only partially, so unlearning that must be complete needs the
//! exact mode.
//!
//! # Interruption
//!
//! [`OnlineModel::train_with`] and [`OnlineModel::update_with`] call a
//! per-iteration hook as [`Trainer::on_round`] does. Breaking stops
//! training after the iteration (as it stops [`Trainer`]) and abandons an
//! update, leaving the model, data and state unchanged.
//! [`OnlineModel::update_with_commit`] also asks for a last confirmation
//! once the update is computed, before it is applied.
//!
//! # Example
//!
//! ```
//! use hessboost::prelude::*;
//! use hessboost::training::online::{OnlineModel, OnlineParams};
//!
//! # fn main() -> Result<()> {
//! let x: Vec<f32> = (0..200).map(|i| (i % 50) as f32 / 50.0).collect();
//! let y: Vec<f32> = x.iter().map(|v| 3.0 * v).collect();
//! let data = DMatrix::from_dense(&x, 200, 1)?.with_labels(&y)?;
//! let params = TrainingParams::builder()
//!     .tree_method(TreeMethod::Hist)
//!     .max_depth(3)
//!     .build()?;
//!
//! let mut online = OnlineModel::train(&params, &data, 20, OnlineParams::default())?;
//! // Forget the first ten rows, learn two new ones.
//! let new = DMatrix::from_dense(&[0.5, 0.25], 2, 1)?.with_labels(&[1.5, 0.75])?;
//! let report = online.update(Some(&new), &(0..10).collect::<Vec<_>>())?;
//! assert_eq!(online.data().n_rows(), 192);
//! assert!(report.nodes_kept > 0);
//! # Ok(())
//! # }
//! ```

mod cache;
mod update;

use std::num::NonZeroUsize;
use std::ops::ControlFlow;

use self::cache::Cache;
use self::update::Incremental;
use super::api::{RoundEval, Trainer};
use super::eval::configured_metrics;
use super::train::{initial_intercepts, with_thread_pool};
use super::validate::{validate_trained_model, validate_training_data};
use crate::config::{
    BoosterKind, Device, GrowPolicy, ProcessType, SamplingMethod, TrainingParams, TreeMethod,
};
use crate::data::quantile::HistCuts;
use crate::data::{DMatrix, FeatureType};
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::objective::Objective;
use crate::tree::RegTree;

/// Settings of [`OnlineModel`]: its [`OnlineMode`], the exact mode
/// ([`Self::exact`], see the [module docs](self#exactness)) or the
/// approximate one with a split robustness tolerance
/// ([`Self::approximate`]). The default is approximate at `0.1`, the
/// paper's recommendation.
///
/// ```
/// use hessboost::training::online::{OnlineMode, OnlineParams};
///
/// # fn main() -> hessboost::error::Result<()> {
/// let OnlineMode::Approximate { tolerance, .. } = OnlineParams::default().mode() else {
///     unreachable!("the default is approximate");
/// };
/// assert_eq!(tolerance, 0.1);
/// assert_eq!(OnlineParams::exact().mode(), OnlineMode::Exact);
/// // The exact mode is `exact()`, not a tolerance of 0.
/// assert!(OnlineParams::approximate(0.0).is_err());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnlineParams {
    mode: OnlineMode,
}

/// How an [`OnlineModel`] updates (see the [module docs](self#exactness)),
/// read from [`OnlineParams::mode`]; built by [`OnlineParams::exact`] and
/// [`OnlineParams::approximate`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum OnlineMode {
    /// Every update reproduces retraining bit for bit, at about its cost.
    Exact,
    /// Splits ranked within the tolerance are kept, the rest regrown, on the
    /// original training's bins and intercept, with lazily refreshed
    /// gradients.
    #[non_exhaustive]
    Approximate {
        /// The split robustness tolerance `σ`, in `(0, 1]`.
        tolerance: f64,
    },
}

impl Default for OnlineParams {
    fn default() -> Self {
        OnlineParams {
            mode: OnlineMode::Approximate { tolerance: 0.1 },
        }
    }
}

impl OnlineParams {
    /// The exact mode: every update reproduces retraining bit for bit.
    pub fn exact() -> Self {
        OnlineParams {
            mode: OnlineMode::Exact,
        }
    }

    /// The approximate mode with split robustness tolerance `σ` in
    /// `(0, 1]`: a node keeps its split while it ranks within the top
    /// `max(1, ⌊σ · candidates⌋)` candidates; `1` regrows only nodes whose
    /// split stopped being a valid candidate (a child below
    /// `min_child_weight`, a gain below `gamma`).
    ///
    /// # Errors
    ///
    /// `tolerance` outside `(0, 1]` (for the exact mode use
    /// [`Self::exact`]), named `tolerance`.
    pub fn approximate(tolerance: f64) -> Result<Self> {
        if !(tolerance > 0.0 && tolerance <= 1.0) {
            return Err(HessboostError::invalid_param(
                "tolerance",
                format!("must be in (0, 1] (use the exact mode for none), got {tolerance}"),
            ));
        }
        Ok(OnlineParams {
            mode: OnlineMode::Approximate { tolerance },
        })
    }

    /// The update mode.
    pub fn mode(&self) -> OnlineMode {
        self.mode
    }

    /// Whether updates are approximate (and keep the approximate mode's
    /// state).
    fn is_approximate(self) -> bool {
        matches!(self.mode, OnlineMode::Approximate { .. })
    }
}

/// What an [`OnlineModel::update`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct UpdateReport {
    /// Nodes whose split (or leaf) was kept, over all trees.
    pub nodes_kept: usize,
    /// Subtrees regrown by the builder (the exact mode regrows every tree).
    pub subtrees_regrown: usize,
    /// Rows whose gradients were recomputed in at least one tree.
    pub rows_refreshed: usize,
}

/// A trained model with its training data, updatable in place. See the
/// [module docs](self).
#[derive(Debug, Clone)]
pub struct OnlineModel {
    params: TrainingParams,
    online: OnlineParams,
    data: DMatrix,
    model: BoostedModel,
    cache: Option<Cache>,
}

impl OnlineModel {
    /// Train `num_boost_round` iterations on `data` (as
    /// [`train`](super::train) does) and keep what updates need.
    ///
    /// # Errors
    ///
    /// The refusals of the [module docs](self#supported-configurations) and
    /// the errors of training.
    pub fn train(
        params: &TrainingParams,
        data: &DMatrix,
        num_boost_round: usize,
        online: OnlineParams,
    ) -> Result<Self> {
        Self::train_with(params, data, num_boost_round, online, |_| {
            ControlFlow::Continue(())
        })
    }

    /// [`Self::train`] calling `on_round` after every iteration, as
    /// [`Trainer::on_round`] does: [`ControlFlow::Break`] stops training
    /// after the iteration, and the online model keeps the iterations so
    /// far (updates then retrain or update that many).
    ///
    /// # Errors
    ///
    /// Those of [`Self::train`].
    pub fn train_with(
        params: &TrainingParams,
        data: &DMatrix,
        num_boost_round: usize,
        online: OnlineParams,
        on_round: impl FnMut(RoundEval<'_>) -> ControlFlow<()> + Send,
    ) -> Result<Self> {
        check_supported(params, data, online)?;
        let model = Trainer::new(params, data, num_boost_round)
            .on_round(on_round)
            .train()?
            .model;
        Self::from_model(model, params, data, online)
    }

    /// Resume from a `model` trained with `params` on `data` (for example one
    /// loaded from a file), rebuilding the update state by replaying its
    /// trees over `data`.
    ///
    /// # Errors
    ///
    /// The refusals of [`Self::train`], and
    /// [`HessboostError::InvalidParameter`] for a model that `params` could
    /// not have trained (another objective or `max_delta_step`, several
    /// outputs, weighted trees, categorical trees in the approximate mode,
    /// linear leaves, a `gblinear`, `boulevard`, or `ebm` booster, for
    /// example from an imported LightGBM `linear_tree` model, model
    /// shrinkage, a different feature count), for an early-stopped model
    /// (`best_iteration` set: slice it to its best iterations first), and
    /// for `eval_metric`s training would refuse.
    pub fn from_model(
        model: BoostedModel,
        params: &TrainingParams,
        data: &DMatrix,
        online: OnlineParams,
    ) -> Result<Self> {
        check_supported(params, data, online)?;
        // Training refuses metrics the objective cannot score before any
        // round; a resumed model gets the same check.
        configured_metrics(params, params.loss(1)?.as_ref())?;
        let categorical = model
            .trees()
            .iter()
            .any(|t| t.nodes().iter().any(|n| n.is_categorical));
        if model.objective().built_in() != Some(&params.objective)
            || model.n_outputs() != 1
            || model.num_parallel_tree() != 1
            || model.n_features() != data.n_cols()
            // The approximate mode replays numeric splits only; the exact
            // mode retrains, so categorical trees are fine there.
            || (categorical && online.is_approximate())
            // The objective's `max_delta_step` shapes every leaf: a model
            // trained with another one is not a model of these parameters.
            || model.max_delta_step() != params.effective_max_delta_step()
            || model.linear().is_some()
            || model.boulevard().is_some()
            || model.ebm().is_some()
            || model.trees().iter().any(|t| t.linear_leaves().is_some())
            || model
                .trees()
                .iter()
                .any(|t| splits_below(t, params.max_depth))
            || (0..model.num_trees()).any(|t| model.tree_weight(t) != 1.0)
            // Model shrinkage rescales every earlier tree each round, so
            // an update of one node's subtree would not reproduce a retrain.
            || model.shrinkage().is_some()
        {
            return Err(HessboostError::invalid_param(
                "model",
                "not a single-output, unweighted, unshrunk, numeric gbtree model (not \
                 Boulevard or EBM) with constant leaves of these parameters and data",
            ));
        }
        if let Some(best) = model.best_iteration() {
            // Early stopping keeps every trained iteration but predicts with
            // the first `best + 1`; an update updates (and predicts with)
            // them all, and a model without `best_iteration`.
            return Err(HessboostError::invalid_param(
                "model",
                format!(
                    "an early-stopped model (best_iteration {best} of {} iterations) is not \
                     updatable: slice it to its best iterations first (`slice(..{}, 1)`)",
                    model.num_boost_rounds(),
                    best + 1
                ),
            ));
        }
        let cache = match online.mode {
            OnlineMode::Exact => None,
            OnlineMode::Approximate { tolerance } => Some(with_thread_pool(params, || {
                Cache::build(&model, params, data, tolerance)
            })?),
        };
        Ok(OnlineModel {
            params: params.clone(),
            online,
            data: data.clone(),
            model,
            cache,
        })
    }

    /// Add the rows of `additions` (features and labels, like the training
    /// data) and delete the training rows `deletions` (indices into
    /// [`Self::data`]), then update the model. The new data is the kept rows
    /// in their order followed by the added ones.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] for out-of-range or repeated
    /// deletions, deleting every row, additions without labels or with
    /// metadata, or of another shape, and (approximate mode) added values
    /// beyond the training data's bins; whatever training refuses on the
    /// updated data (such as labels outside the objective's domain), checked
    /// before anything changes; [`HessboostError::ModelFormat`] for an
    /// update whose arithmetic overflows `f32`, as training refuses such a
    /// model; the errors of training.
    pub fn update(
        &mut self,
        additions: Option<&DMatrix>,
        deletions: &[usize],
    ) -> Result<UpdateReport> {
        self.update_with(additions, deletions, |_| ControlFlow::Continue(()))
    }

    /// [`Self::update`] calling `on_round` after every updated iteration, as
    /// [`Trainer::on_round`] does during training. [`ControlFlow::Break`]
    /// abandons the update: the model and data stay as they were and
    /// [`HessboostError::InvalidParameter`] (`on_round`) is returned.
    ///
    /// # Errors
    ///
    /// Those of [`Self::update`], and the interruption.
    pub fn update_with(
        &mut self,
        additions: Option<&DMatrix>,
        deletions: &[usize],
        on_round: impl FnMut(RoundEval<'_>) -> ControlFlow<()> + Send,
    ) -> Result<UpdateReport> {
        self.update_with_commit(additions, deletions, on_round, || ControlFlow::Continue(()))
    }

    /// [`Self::update_with`] asking `commit` once the update is computed,
    /// just before it is applied: [`ControlFlow::Break`] abandons it as a
    /// break from `on_round` does (for a caller whose interruption can
    /// arrive after the last iteration's hook, such as a signal).
    ///
    /// # Errors
    ///
    /// Those of [`Self::update_with`].
    pub fn update_with_commit(
        &mut self,
        additions: Option<&DMatrix>,
        deletions: &[usize],
        mut on_round: impl FnMut(RoundEval<'_>) -> ControlFlow<()> + Send,
        commit: impl FnOnce() -> ControlFlow<()>,
    ) -> Result<UpdateReport> {
        let deleted = self.check_change(additions, deletions)?;
        let updated = compose(&self.data, &deleted, additions)?;
        // Refuse what retraining on `updated` refuses (e.g. labels outside
        // the loss's domain) before any state changes: the approximate mode
        // computes gradients without going through the trainer's checks.
        validate_training_data(&self.params, &updated)?;
        // Retraining estimates the intercept from the labels (unless
        // `base_score` is set) and refuses a non-finite one; the approximate
        // mode keeps the original intercept, so check it here.
        initial_intercepts(
            &self.params,
            self.params.loss(1)?.as_ref(),
            &updated.info(),
            1,
        )?;
        let rounds = self.model.num_boost_rounds();
        // The approximate mode updates a copy of the cache, swapped in on
        // success, so an abandoned update leaves exactly the state it started
        // from (a rebuild would recompute the bins and gradients earlier
        // updates keep fixed).
        let Some(cache) = self.cache.as_ref() else {
            let mut stopped = false;
            let model = Trainer::new(&self.params, &updated, rounds)
                .on_round(|round| {
                    let flow = on_round(round);
                    stopped |= flow.is_break();
                    flow
                })
                .train()?
                .model;
            if stopped || model.num_boost_rounds() != rounds || commit().is_break() {
                return Err(interrupted());
            }
            let report = UpdateReport {
                nodes_kept: kept_nodes(&self.model, &model),
                subtrees_regrown: rounds,
                rows_refreshed: updated.n_rows(),
            };
            self.model = model;
            self.data = updated;
            return Ok(report);
        };
        let mut cache = cache.clone();
        let run = Incremental {
            params: &self.params,
            tolerance: cache.tolerance,
            old: &self.data,
            new: &updated,
            deleted: &deleted,
            model: &self.model,
        };
        // On `nthread` threads, as training runs.
        let outcome = with_thread_pool(&self.params, || run.run(&mut cache, &mut on_round))
            .and_then(|done| match commit() {
                ControlFlow::Continue(()) => Ok(done),
                ControlFlow::Break(()) => Err(interrupted()),
            });
        let (trees, report) = outcome?;
        // Arithmetic that overflows `f32` (extreme labels or margins) leaves
        // non-finite leaves the model formats refuse: refuse the update, as
        // training refuses such a model, before anything changes.
        let model = self.model.with_trees(trees);
        validate_trained_model(&model)?;
        self.model = model;
        self.data = updated;
        self.cache = Some(cache);
        Ok(report)
    }

    /// The current model.
    pub fn model(&self) -> &BoostedModel {
        &self.model
    }

    /// The current training data: the original rows minus deletions plus
    /// additions, in update order.
    pub fn data(&self) -> &DMatrix {
        &self.data
    }

    /// The training parameters.
    pub fn params(&self) -> &TrainingParams {
        &self.params
    }

    /// The update settings.
    pub fn online_params(&self) -> OnlineParams {
        self.online
    }

    /// The model, dropping the data and update state.
    pub fn into_model(self) -> BoostedModel {
        self.model
    }

    /// Validate a change; returns the deletion mask over [`Self::data`].
    fn check_change(&self, additions: Option<&DMatrix>, deletions: &[usize]) -> Result<Vec<bool>> {
        let n = self.data.n_rows();
        let mut deleted = vec![false; n];
        for &row in deletions {
            if row >= n {
                return Err(HessboostError::invalid_param(
                    "deletions",
                    format!("row {row} is out of range for {n} rows"),
                ));
            }
            if std::mem::replace(&mut deleted[row], true) {
                return Err(HessboostError::invalid_param(
                    "deletions",
                    format!("row {row} is deleted twice"),
                ));
            }
        }
        let added = additions.map_or(0, DMatrix::n_rows);
        if deletions.len() == n && added == 0 {
            return Err(HessboostError::invalid_param(
                "deletions",
                "an update must leave at least one row",
            ));
        }
        if let Some(a) = additions {
            check_data(a, "additions")?;
            if a.n_cols() != self.data.n_cols() || a.feature_types() != self.data.feature_types() {
                return Err(HessboostError::invalid_param(
                    "additions",
                    "added rows need the training data's columns and feature types",
                ));
            }
            if let Some(cache) = &self.cache {
                check_within_cuts(&cache.cuts, a)?;
            }
        }
        Ok(deleted)
    }
}

/// Refuse an added value at or above its feature's top cut. The approximate
/// mode keeps the original training's bins, whose last bin ends at that cut:
/// a value past it would count in the last bin of the histograms that rank
/// splits while prediction routes it right of a split at the top cut, so
/// the ranked and the actual children would differ.
fn check_within_cuts(cuts: &HistCuts, additions: &DMatrix) -> Result<()> {
    let mut outside = None;
    for row in 0..additions.n_rows() {
        additions.for_row_entry(row, |c, v| {
            let (start, end) = cuts.feature_bins(c as usize);
            if outside.is_none() && (end == start || v >= cuts.cut_value(end - 1)) {
                outside = Some((row, c, v));
            }
        });
        if let Some((row, c, v)) = outside {
            return Err(HessboostError::invalid_param(
                "additions",
                format!(
                    "added row {row} has feature {c} = {v}, beyond the training data's bins, \
                     which the approximate mode keeps fixed; use the exact mode (`OnlineParams::exact`, \
                     Python `mode=Exact()`), \
                     or rebuild the state on data covering it with `OnlineModel::from_model`"
                ),
            ));
        }
    }
    Ok(())
}

fn interrupted() -> HessboostError {
    HessboostError::invalid_param("on_round", "the update was interrupted; nothing changed")
}

/// Nodes of `after` equal (split or leaf, same position) to `before`'s.
fn kept_nodes(before: &BoostedModel, after: &BoostedModel) -> usize {
    before
        .trees()
        .iter()
        .zip(after.trees())
        .map(|(a, b)| {
            a.nodes()
                .iter()
                .zip(b.nodes())
                .filter(|(x, y)| {
                    x.children() == y.children()
                        && x.split_feature == y.split_feature
                        && x.split_cond == y.split_cond
                        && x.default_left == y.default_left
                })
                .count()
        })
        .sum()
}

/// The refusals of the module docs.
fn check_supported(params: &TrainingParams, data: &DMatrix, online: OnlineParams) -> Result<()> {
    params.validate()?;
    let refuse = |name: &'static str, why: &str| {
        Err(HessboostError::invalid_param(
            name,
            format!("in-place updates need {why}"),
        ))
    };
    if params.booster != BoosterKind::GbTree {
        return refuse("booster", "booster = gbtree");
    }
    if !matches!(params.tree_method, TreeMethod::Hist | TreeMethod::Auto) {
        return refuse("tree_method", "tree_method = hist");
    }
    if params.grow_policy != GrowPolicy::DepthWise
        || params.max_leaves.is_some()
        || params.max_depth.is_none()
    {
        return refuse(
            "grow_policy",
            "depth-wise growth with a max_depth and no max_leaves (a node's subtree must \
             depend on its rows only)",
        );
    }
    if params.subsample != 1.0
        || params.sampling_method != SamplingMethod::Uniform
        || params.colsample_bytree != 1.0
        || params.colsample_bylevel != 1.0
        || params.colsample_bynode != 1.0
    {
        return refuse(
            "subsample",
            "no row or column sampling (a retrain would draw different samples)",
        );
    }
    if params.num_parallel_tree != 1 {
        return refuse("num_parallel_tree", "num_parallel_tree = 1");
    }
    if params.process_type != ProcessType::Default {
        return refuse("process_type", "process_type = default");
    }
    if !params.monotone_constraints.is_empty() || !params.interaction_constraints.is_empty() {
        return refuse(
            "monotone_constraints",
            "no monotone or interaction constraints",
        );
    }
    if params.extra_trees.is_some()
        || params.path_smooth != 0.0
        || params.linear_tree.is_some()
        || params.quantized.is_some()
        || params.toad_penalty_feature != 0.0
        || params.toad_penalty_threshold != 0.0
    {
        return refuse("extra_trees", "none of the beyond-XGBoost split options");
    }
    if params.device != Device::Cpu {
        return refuse("device", "device = cpu");
    }
    // Every other setting, including any added later, stays at its default:
    // a new training option is refused until it is shown sound here.
    let reference = TrainingParams {
        booster: params.booster,
        nthread: params.nthread,
        seed: params.seed,
        device: params.device,
        objective: params.objective.clone(),
        base_score: params.base_score,
        eval_metric: params.eval_metric.clone(),
        eta: params.eta,
        gamma: params.gamma,
        max_depth: params.max_depth,
        min_child_weight: params.min_child_weight,
        max_delta_step: params.max_delta_step,
        lambda: params.lambda,
        alpha: params.alpha,
        tree_method: params.tree_method,
        max_bin: params.max_bin,
        multi_strategy: params.multi_strategy,
        ..TrainingParams::default()
    };
    params.refuse_changes_from(
        &reference,
        "params",
        "in-place updates support only the settings they are proven sound for",
    )?;
    // Exhaustive, so a new objective has to be classified here.
    let per_row_newton = match &params.objective {
        Objective::SquaredError(_)
        | Objective::SquaredLogError
        | Objective::PseudoHuber(_)
        | Objective::Expectile(_)
        | Objective::RegLogistic(_)
        | Objective::BinaryLogistic(_)
        | Objective::BinaryLogitRaw(_)
        | Objective::BinaryHinge
        | Objective::Softmax(_)
        | Objective::Softprob(_)
        | Objective::Poisson
        | Objective::Gamma(_)
        | Objective::Tweedie(_)
        | Objective::Aft(_)
        | Objective::Dist(_) => true,
        Objective::AbsoluteError
        | Objective::Quantile(_)
        | Objective::RankPairwise(_)
        | Objective::RankNdcg(_)
        | Objective::RankMap(_)
        | Objective::RankXendcg
        | Objective::Cox
        | Objective::Custom(_) => false,
    };
    if !per_row_newton {
        return refuse(
            "objective",
            "a built-in objective with per-row gradients and Newton-step leaves (not \
             ranking, survival:cox, reg:absoluteerror, reg:quantileerror, or a custom loss)",
        );
    }
    let objective = params.loss(1)?;
    if objective.n_outputs() != 1 {
        return refuse("objective", "a single-output objective");
    }
    check_data(data, "data")?;
    if online.is_approximate() && data.feature_types().contains(&FeatureType::Categorical) {
        return refuse(
            "data",
            "numerical features in the approximate mode (the exact mode accepts categorical ones)",
        );
    }
    validate_training_data(params, data)
}

/// Labels and none of the metadata updates cannot honor.
fn check_data(data: &DMatrix, name: &'static str) -> Result<()> {
    if data.labels().is_none() || data.n_targets() != 1 {
        return Err(HessboostError::invalid_param(
            name,
            "in-place updates need one label per row",
        ));
    }
    if data.weights().is_some()
        || data.base_margin().is_some()
        || data.group().is_some()
        || data.label_lower_bound().is_some()
        || data.feature_weights().is_some()
    {
        return Err(HessboostError::invalid_param(
            name,
            "in-place updates do not support weights, base margins, groups, label bounds, or \
             feature weights",
        ));
    }
    Ok(())
}

/// The rows of `data` not in `deleted`, then those of `additions`: dense
/// (missing as NaN) when `data` is dense, else compressed sparse rows.
fn compose(data: &DMatrix, deleted: &[bool], additions: Option<&DMatrix>) -> Result<DMatrix> {
    let p = data.n_cols();
    let rows = deleted
        .iter()
        .enumerate()
        .filter(|(_, d)| !**d)
        .map(|(r, _)| (data, r))
        .chain(
            additions
                .into_iter()
                .flat_map(|a| (0..a.n_rows()).map(move |r| (a, r))),
        );
    let mut labels = Vec::new();
    let mut out = if data.dense_values().is_some() {
        let mut values = Vec::new();
        for (m, row) in rows {
            let start = values.len();
            values.resize(start + p, f32::NAN);
            m.for_row_entry(row, |c, v| values[start + c as usize] = v);
            labels.push(m.labels().map_or(0.0, |l| l[row]));
        }
        let n = labels.len();
        DMatrix::from_dense(&values, n, p)?
    } else {
        let mut indptr = vec![0usize];
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        for (m, row) in rows {
            m.for_row_entry(row, |c, v| {
                indices.push(c);
                values.push(v);
            });
            indptr.push(values.len());
            labels.push(m.labels().map_or(0.0, |l| l[row]));
        }
        DMatrix::from_csr(indptr, indices, values, p)?
    };
    if data.feature_types().contains(&FeatureType::Categorical) {
        out = out.with_feature_types(data.feature_types())?;
    }
    out.with_labels(&labels)
}

/// Whether `tree` splits a node at depth `max_depth` or deeper (`None`: no
/// limit), which depth-wise training to that depth never does.
fn splits_below(tree: &RegTree, max_depth: Option<NonZeroUsize>) -> bool {
    let Some(limit) = max_depth else {
        return false;
    };
    let mut stack = vec![(0usize, 0usize)];
    while let Some((id, depth)) = stack.pop() {
        let Some((left, right)) = tree.node_at(id).children() else {
            continue;
        };
        if depth >= limit.get() {
            return true;
        }
        stack.push((left, depth + 1));
        stack.push((right, depth + 1));
    }
    false
}
