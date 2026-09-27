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
//! [`OnlineParams::tolerance`] `σ`, a split ranked within the top
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
//! training.
//!
//! # Exactness
//!
//! `tolerance = 0` is the exact mode: every node's split must still be the
//! best one on recomputed statistics, bin boundaries and intercept, which is
//! by definition what retraining produces, so the exact mode defers every
//! tree to the builder: the result equals [`train`](super::train) on
//! [`OnlineModel::data`] bit for bit, at about the cost of retraining (a
//! reference to measure the approximate mode against, and the answer when
//! unlearning must be exact). `tolerance > 0` is approximate: the kept
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

use std::num::NonZeroUsize;
use std::ops::ControlFlow;

use super::train::{RoundEval, Trainer, validate_training_data};
use crate::config::{
    BoosterKind, Device, GrowPolicy, ProcessType, SamplingMethod, TrainingParams, TreeMethod,
};
use crate::data::ghist::{Bins, GHistIndex};
use crate::data::quantile::HistCuts;
use crate::data::{DMatrix, FeatureType};
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::objective::{GradPair, Loss, Objective};
use crate::tree::builder::HistTreeBuilder;
use crate::tree::builder::online::rank_split;
use crate::tree::gain::{GradStats, RegParams, calc_weight};
use crate::tree::sampler::ColumnSampler;
use crate::tree::{ChildLeaf, RegTree, SplitRule};

/// Settings of [`OnlineModel`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct OnlineParams {
    /// Split robustness tolerance `σ` in `[0, 1]`: a node keeps its split
    /// while it ranks within the top `max(1, ⌊σ · candidates⌋)` candidates.
    /// `0` is the exact mode (see the [module docs](self#exactness)), `1`
    /// regrows only nodes whose split stopped being a valid candidate (a
    /// child below `min_child_weight`, a gain below `gamma`). Default `0.1`, the paper's recommendation.
    pub tolerance: f64,
}

impl Default for OnlineParams {
    fn default() -> Self {
        OnlineParams { tolerance: 0.1 }
    }
}

impl OnlineParams {
    /// Settings with split robustness tolerance `tolerance` (`0` is exact).
    pub fn with_tolerance(tolerance: f64) -> Self {
        OnlineParams { tolerance }
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

/// The approximate mode's state.
#[derive(Debug, Clone)]
struct Cache {
    cuts: HistCuts,
    /// No row has a missing value (the builder then enumerates no missing
    /// directions).
    dense: bool,
    trees: Vec<TreeCache>,
}

#[derive(Debug, Clone)]
struct TreeCache {
    /// Per node of the tree.
    nodes: Vec<NodeCache>,
    /// The gradient pair each row of the current data contributes.
    grads: Vec<GradPair>,
}

#[derive(Debug, Clone, Default)]
struct NodeCache {
    stats: GradStats,
    /// Per-bin sums (internal nodes only).
    hist: Vec<GradStats>,
}

impl OnlineModel {
    /// Train `num_boost_round` iterations on `data` (as
    /// [`train`](super::train) does) and keep what updates need.
    ///
    /// # Errors
    ///
    /// The refusals of the [module docs](self#supported-configurations), an
    /// invalid [`OnlineParams::tolerance`], and the errors of training.
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
        on_round: impl FnMut(&RoundEval) -> ControlFlow<()> + Send,
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
    /// not have trained (another objective, several outputs, weighted or
    /// categorical trees, linear leaves or a `gblinear` booster, for
    /// example from an imported LightGBM `linear_tree` model, model
    /// shrinkage, a different feature count).
    pub fn from_model(
        model: BoostedModel,
        params: &TrainingParams,
        data: &DMatrix,
        online: OnlineParams,
    ) -> Result<Self> {
        check_supported(params, data, online)?;
        let categorical = model
            .trees()
            .iter()
            .any(|t| t.nodes().iter().any(|n| n.is_categorical));
        if model.objective().built_in() != Some(&params.objective)
            || model.n_outputs() != 1
            || model.num_parallel_tree() != 1
            || model.n_features() != data.n_cols()
            || categorical
            || model.linear().is_some()
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
                "not a single-output, unweighted, unshrunk, numeric gbtree model with constant \
                 leaves of these parameters and data",
            ));
        }
        let cache = if online.tolerance > 0.0 {
            Some(Cache::build(&model, params, data)?)
        } else {
            None
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
    /// metadata, or of another shape; whatever training refuses on the
    /// updated data (such as labels outside the objective's domain), checked
    /// before anything changes; the errors of training.
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
        on_round: impl FnMut(&RoundEval) -> ControlFlow<()> + Send,
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
        mut on_round: impl FnMut(&RoundEval) -> ControlFlow<()> + Send,
        commit: impl FnOnce() -> ControlFlow<()>,
    ) -> Result<UpdateReport> {
        let deleted = self.check_change(additions, deletions)?;
        let updated = compose(&self.data, &deleted, additions)?;
        // Refuse what retraining on `updated` refuses (e.g. labels outside
        // the loss's domain) before any state changes: the approximate mode
        // computes gradients without going through the trainer's checks.
        validate_training_data(&self.params, &updated)?;
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
            tolerance: self.online.tolerance,
            old: &self.data,
            new: &updated,
            deleted: &deleted,
            model: &self.model,
        };
        let outcome = run
            .run(&mut cache, &mut on_round)
            .and_then(|done| match commit() {
                ControlFlow::Continue(()) => Ok(done),
                ControlFlow::Break(()) => Err(interrupted()),
            });
        let (trees, report) = outcome?;
        self.model = self.model.with_trees(trees);
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
        }
        Ok(deleted)
    }
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
                    x.left == y.left
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
    let t = online.tolerance;
    if !(t.is_finite() && (0.0..=1.0).contains(&t)) {
        return Err(HessboostError::invalid_param(
            "tolerance",
            format!("must be in [0, 1], got {t}"),
        ));
    }
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
        Objective::SquaredError
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
        | Objective::Gamma
        | Objective::Tweedie(_)
        | Objective::Aft(_)
        | Objective::Dist(_) => true,
        Objective::AbsoluteError
        | Objective::Quantile(_)
        | Objective::RankPairwise(_)
        | Objective::RankNdcg(_)
        | Objective::RankMap(_)
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
    if t > 0.0 && data.feature_types().contains(&FeatureType::Categorical) {
        return refuse(
            "data",
            "numerical features for tolerance > 0 (the exact mode accepts categorical ones)",
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

/// Gradient pairs of every row of `data` at `margins`.
fn gradients(objective: &dyn Loss, data: &DMatrix, margins: &[f32]) -> Vec<GradPair> {
    let mut out = vec![GradPair::default(); data.n_rows()];
    objective.gradient_info(margins, &data.info(), &mut out);
    out
}

fn stats_of(g: GradPair) -> GradStats {
    GradStats::new(f64::from(g.grad), f64::from(g.hess))
}

fn negate(g: GradStats) -> GradStats {
    GradStats::new(-g.grad, -g.hess)
}

/// Add `g` to `hist` at every bin of row `row` of `ghist`.
fn add_index_bins(hist: &mut [GradStats], ghist: &GHistIndex, row: usize, g: GradStats) {
    let (s, e) = (ghist.row_ptr()[row], ghist.row_ptr()[row + 1]);
    match ghist.bins() {
        Bins::U16(b) => b[s..e].iter().for_each(|&bin| hist[bin as usize].add(g)),
        Bins::U32(b) => b[s..e].iter().for_each(|&bin| hist[bin as usize].add(g)),
    }
}

/// The global bins of row `row` of `data` under `cuts` (present features).
fn row_bins(cuts: &HistCuts, data: &DMatrix, row: usize) -> Vec<u32> {
    let mut bins = Vec::with_capacity(data.n_cols());
    data.for_row_entry(row, |c, v| bins.push(cuts.bin_of(c as usize, v)));
    bins
}

/// Node statistics of `tree` below `root` from `rows` (row of `data` and
/// `ghist`, gradient).
fn accumulate(
    tree: &RegTree,
    root: usize,
    data: &DMatrix,
    ghist: &GHistIndex,
    rows: impl Iterator<Item = (usize, GradPair)>,
    out: &mut [NodeCache],
) {
    let bins = ghist.total_bins();
    for (row, g) in rows {
        let g = stats_of(g);
        let mut nid = root;
        loop {
            let node = tree.node(nid);
            out[nid].stats.add(g);
            if node.is_leaf() {
                break;
            }
            if out[nid].hist.is_empty() {
                out[nid].hist = vec![GradStats::default(); bins];
            }
            add_index_bins(&mut out[nid].hist, ghist, row, g);
            nid = tree.child(nid, data.get(row, node.split_feature as usize));
        }
    }
}

impl Cache {
    /// Replay `model`'s trees over `data` (as training grew them).
    fn build(model: &BoostedModel, params: &TrainingParams, data: &DMatrix) -> Result<Self> {
        let cuts = HistCuts::from_dmatrix(data, params.max_bin);
        let ghist = GHistIndex::from_dmatrix(data, cuts.clone());
        let objective = params.loss(1)?;
        let mut margins = vec![model.base_scores()[0]; data.n_rows()];
        let mut trees = Vec::with_capacity(model.num_trees());
        for tree in model.trees() {
            let grads = gradients(objective.as_ref(), data, &margins);
            let mut nodes = vec![NodeCache::default(); tree.num_nodes()];
            accumulate(
                tree,
                0,
                data,
                &ghist,
                grads.iter().copied().enumerate(),
                &mut nodes,
            );
            for (row, m) in margins.iter_mut().enumerate() {
                *m += tree.predict_row(data, row);
            }
            trees.push(TreeCache { nodes, grads });
        }
        let dense = ghist.dense_stride().is_some();
        Ok(Cache { cuts, dense, trees })
    }
}

/// One approximate update.
struct Incremental<'a> {
    params: &'a TrainingParams,
    tolerance: f64,
    old: &'a DMatrix,
    new: &'a DMatrix,
    deleted: &'a [bool],
    model: &'a BoostedModel,
}

/// A changed row's contribution routed down a tree.
struct Delta {
    /// Row of the deleted-rows matrix (`true`) or of the new data.
    deleted: bool,
    row: usize,
    /// The change of the row's gradient pair.
    g: GradStats,
    /// The row's global bins.
    bins: Vec<u32>,
}

impl Incremental<'_> {
    fn run(
        &self,
        cache: &mut Cache,
        on_round: &mut dyn FnMut(&RoundEval) -> ControlFlow<()>,
    ) -> Result<(Vec<RegTree>, UpdateReport)> {
        let (old, new) = (self.old, self.new);
        let deleted_rows: Vec<usize> = (0..old.n_rows()).filter(|&r| self.deleted[r]).collect();
        let gone = if deleted_rows.is_empty() {
            None
        } else {
            Some(old.select_rows(&deleted_rows)?)
        };
        let n_added = new.n_rows() + deleted_rows.len() - old.n_rows();
        let new_to_old: Vec<Option<usize>> = (0..old.n_rows())
            .filter(|&r| !self.deleted[r])
            .map(Some)
            .chain(std::iter::repeat_n(None, n_added))
            .collect();
        let objective = self.params.loss(1)?;
        let reg = RegParams::from_params(self.params);
        let eta = self.params.eta as f32;
        let base = self.model.base_scores()[0];
        // Margins under the updated trees, kept for fresh rows only.
        let mut fresh: Vec<bool> = new_to_old.iter().map(Option::is_none).collect();
        let mut margins = vec![base; new.n_rows()];
        let mut ghist: Option<GHistIndex> = None;
        let mut report = UpdateReport::default();
        cache.dense &= (0..new.n_rows())
            .filter(|&i| new_to_old[i].is_none())
            .all(|i| {
                let mut present = 0;
                new.for_row_entry(i, |_, _| present += 1);
                present == new.n_cols()
            });
        let mut trees: Vec<RegTree> = Vec::with_capacity(self.model.num_trees());
        for (m, old_tree) in self.model.trees().iter().enumerate() {
            let tc = &mut cache.trees[m];
            let old_grads = std::mem::take(&mut tc.grads);
            let g_new = gradients(objective.as_ref(), new, &margins);
            let cur: Vec<GradPair> = (0..new.n_rows())
                .map(|i| match new_to_old[i] {
                    Some(r) if !fresh[i] => old_grads[r],
                    _ => g_new[i],
                })
                .collect();
            let cuts = &cache.cuts;
            let mut deltas: Vec<Delta> = Vec::new();
            if let Some(gone) = &gone {
                for (k, &r) in deleted_rows.iter().enumerate() {
                    deltas.push(Delta {
                        deleted: true,
                        row: k,
                        g: negate(stats_of(old_grads[r])),
                        bins: row_bins(cuts, gone, k),
                    });
                }
            }
            for i in (0..new.n_rows()).filter(|&i| fresh[i]) {
                let g = match new_to_old[i] {
                    // Kept rows sit on the same path in the kept structure:
                    // their change is the difference.
                    Some(r) => stats_of(cur[i]).sub(stats_of(old_grads[r])),
                    None => stats_of(cur[i]),
                };
                deltas.push(Delta {
                    deleted: false,
                    row: i,
                    g,
                    bins: row_bins(cuts, new, i),
                });
            }
            let ctx = TreeUpdate {
                run: self,
                dense: cache.dense,
                reg: &reg,
                eta,
                old_tree,
                gone: gone.as_ref(),
                cur: &cur,
                cuts,
            };
            let old_nodes = std::mem::take(&mut tc.nodes);
            let (tree, nodes, regrown_rows) =
                ctx.update(old_nodes, deltas, &mut ghist, &mut report);
            tc.nodes = nodes;
            tc.grads = cur;
            for (i, v) in margins.iter_mut().enumerate() {
                if fresh[i] {
                    *v += tree.predict_row(new, i);
                }
            }
            trees.push(tree);
            for i in regrown_rows {
                if !std::mem::replace(&mut fresh[i], true) {
                    margins[i] = base + trees.iter().map(|t| t.predict_row(new, i)).sum::<f32>();
                }
            }
            let round = RoundEval {
                iteration: m,
                scores: Vec::new(),
            };
            if on_round(&round).is_break() {
                return Err(interrupted());
            }
        }
        report.rows_refreshed = fresh.iter().filter(|&&f| f).count();
        Ok((trees, report))
    }
}

/// The update of one tree.
struct TreeUpdate<'a> {
    run: &'a Incremental<'a>,
    /// Whether the data (old and new) has no missing value.
    dense: bool,
    reg: &'a RegParams,
    eta: f32,
    old_tree: &'a RegTree,
    /// The deleted rows.
    gone: Option<&'a DMatrix>,
    /// The gradient every row of the new data contributes now.
    cur: &'a [GradPair],
    cuts: &'a HistCuts,
}

impl TreeUpdate<'_> {
    fn value(&self, d: &Delta, feature: usize) -> Option<f32> {
        match (d.deleted, self.gone) {
            (true, Some(m)) => m.get(d.row, feature),
            _ => self.run.new.get(d.row, feature),
        }
    }

    /// The updated tree, its node caches, and the rows that reached a
    /// regrown subtree. `ghist`, the new data's index, is built on the first
    /// regrowth.
    fn update(
        &self,
        mut old_nodes: Vec<NodeCache>,
        deltas: Vec<Delta>,
        ghist: &mut Option<GHistIndex>,
        report: &mut UpdateReport,
    ) -> (RegTree, Vec<NodeCache>, Vec<usize>) {
        let params = self.run.params;
        let new = self.run.new;
        let mut tree = RegTree::with_root(0.0);
        let mut nodes: Vec<NodeCache> = vec![NodeCache::default()];
        let mut regrown_rows = Vec::new();
        // (old node, new node, depth, deltas reaching it)
        let mut queue = std::collections::VecDeque::from([(0usize, 0usize, 0usize, deltas)]);
        while let Some((old_id, new_id, depth, deltas)) = queue.pop_front() {
            let old = *self.old_tree.node(old_id);
            let mut cache = std::mem::take(&mut old_nodes[old_id]);
            for d in &deltas {
                cache.stats.add(d.g);
                if !old.is_leaf() {
                    for &bin in &d.bins {
                        cache.hist[bin as usize].add(d.g);
                    }
                }
            }
            let touched = !deltas.is_empty();
            if old.is_leaf() {
                let value = if touched {
                    (calc_weight(cache.stats, self.reg) as f32) * self.eta
                } else {
                    old.leaf_value
                };
                tree.set_leaf_value(new_id, value);
                tree.set_sum_hess(new_id, cache.stats.hess as f32);
                nodes[new_id] = cache;
                report.nodes_kept += 1;
                continue;
            }
            // An untouched node's histogram, and so its ranking, is as
            // before: its split stays.
            let (keep, gain) = if touched {
                let rank = rank_split(
                    self.reg,
                    self.cuts,
                    &cache.hist,
                    cache.stats,
                    self.dense,
                    (old.split_feature, old.split_cond, old.default_left),
                );
                let allowed = ((self.run.tolerance * rank.candidates as f64) as usize).max(1);
                let keep = rank.better.is_some_and(|b| b < allowed)
                    && f64::from(rank.loss_chg) >= params.gamma;
                (keep, rank.loss_chg)
            } else {
                (true, old.split_gain)
            };
            if keep {
                let (l, r) = tree.expand(
                    new_id,
                    SplitRule::numeric(old.split_feature, old.split_cond, old.default_left),
                    ChildLeaf::new(0.0, 0.0),
                    ChildLeaf::new(0.0, 0.0),
                );
                tree.set_sum_hess(new_id, cache.stats.hess as f32);
                tree.set_split_gain(new_id, gain);
                nodes[new_id] = cache;
                nodes.resize(tree.num_nodes(), NodeCache::default());
                let (mut left, mut right) = (Vec::new(), Vec::new());
                for d in deltas {
                    let value = self.value(&d, old.split_feature as usize);
                    if self.old_tree.child(old_id, value) == old.left as usize {
                        left.push(d);
                    } else {
                        right.push(d);
                    }
                }
                queue.push_back((old.left as usize, l, depth + 1, left));
                queue.push_back((old.right as usize, r, depth + 1, right));
                report.nodes_kept += 1;
                continue;
            }
            // Regrow: every row of the new data now reaching this node, with
            // the builder on the new data's index.
            let index = match ghist {
                Some(index) => &*index,
                None => ghist.insert(GHistIndex::from_dmatrix(new, self.cuts.clone())),
            };
            let rows: Vec<u32> = (0..new.n_rows())
                .filter(|&i| tree.leaf_id_with(|f| new.get(i, f as usize)) == new_id)
                .map(|i| i as u32)
                .collect();
            // A split node lies above `max_depth` (`check_supported` requires
            // one, `from_model` refuses deeper trees), so its subtree keeps at
            // least one level.
            let sub_params = TrainingParams {
                max_depth: params
                    .max_depth
                    .and_then(|limit| NonZeroUsize::new(limit.get().saturating_sub(depth))),
                ..params.clone()
            };
            let mut sampler = ColumnSampler::new(new.n_cols(), None, 1.0, 1.0, 1.0, params.seed);
            let sub = HistTreeBuilder::new(&sub_params).build(index, self.cur, &rows, &mut sampler);
            graft(&mut tree, new_id, &sub, 0, self.eta);
            nodes.resize(tree.num_nodes(), NodeCache::default());
            clear_subtree(&tree, new_id, &mut nodes);
            accumulate(
                &tree,
                new_id,
                new,
                index,
                rows.iter().map(|&i| (i as usize, self.cur[i as usize])),
                &mut nodes,
            );
            regrown_rows.extend(rows.iter().map(|&i| i as usize));
            report.subtrees_regrown += 1;
        }
        (tree, nodes, regrown_rows)
    }
}

/// Whether `tree` splits a node at depth `max_depth` or deeper (`None`: no
/// limit), which depth-wise training to that depth never does.
fn splits_below(tree: &RegTree, max_depth: Option<NonZeroUsize>) -> bool {
    let Some(limit) = max_depth else {
        return false;
    };
    let mut stack = vec![(0usize, 0usize)];
    while let Some((id, depth)) = stack.pop() {
        let node = tree.node(id);
        if node.is_leaf() {
            continue;
        }
        if depth >= limit.get() {
            return true;
        }
        stack.push((node.left as usize, depth + 1));
        stack.push((node.right as usize, depth + 1));
    }
    false
}

/// Reset the caches of `root`'s subtree in `tree`.
fn clear_subtree(tree: &RegTree, root: usize, nodes: &mut [NodeCache]) {
    let mut stack = vec![root];
    while let Some(nid) = stack.pop() {
        nodes[nid] = NodeCache::default();
        let node = tree.node(nid);
        if !node.is_leaf() {
            stack.push(node.left as usize);
            stack.push(node.right as usize);
        }
    }
}

/// Copy `src`'s subtree at `src_id` into `dst` at leaf `dst_id`, scaling
/// leaf weights by `eta` (the builder leaves them unshrunk).
fn graft(dst: &mut RegTree, dst_id: usize, src: &RegTree, src_id: usize, eta: f32) {
    let node = *src.node(src_id);
    dst.set_sum_hess(dst_id, node.sum_hess);
    if node.is_leaf() {
        dst.set_leaf_value(dst_id, node.leaf_value * eta);
        return;
    }
    let (l, r) = dst.expand(
        dst_id,
        SplitRule::numeric(node.split_feature, node.split_cond, node.default_left),
        ChildLeaf::new(0.0, 0.0),
        ChildLeaf::new(0.0, 0.0),
    );
    dst.set_split_gain(dst_id, node.split_gain);
    graft(dst, l, src, node.left as usize, eta);
    graft(dst, r, src, node.right as usize, eta);
}
