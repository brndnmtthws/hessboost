//! Budget-mode training: one `budget` number instead of hyperparameter
//! tuning, after [PerpetualBooster](https://github.com/perpetual-ml/perpetual)
//! (Apache-2.0; the algorithm is re-implemented here, no code is copied).
//!
//! Beyond XGBoost and opt-in: nothing here runs unless you call
//! [`train_with_budget`]. The trained model is an ordinary gbtree
//! [`BoostedModel`], so prediction, SHAP, and every model format (including
//! XGBoost JSON/UBJSON export) work unchanged.
//!
//! # Algorithm
//!
//! With budget `b` (Perpetual's defaults and general-case schedules, taken
//! from its `booster/core.rs` and `constants.rs`):
//!
//! * **Learning rate** `η = 10^(−b)` for `b ≤ 1`, else `10^(−(1 + 0.65 (b − 1)))`.
//! * **Target loss decrement per tree.** With `u = max(b, 0.1)`, `n = 10/u`,
//!   `c = (n − 2)/(n (n − 1))`, a tree stops growing once the average
//!   per-row loss reduction it achieves exceeds `c · 10^(−min(u, 3)) · L̄`.
//!   For `b ≤ 0.2`, `L̄` is the initial average loss; otherwise it is the blend
//!   `α L̄₀ + (1 − α) L̄_prev` with `α = clamp(1 − 0.1 b, 0.85, 0.995)` and the
//!   target is scaled by `1.2`. After more than `stopping_rounds + 1`
//!   consecutive trees that did not reach it, trees grow without a target.
//! * **Splits** are gated by a five-fold generalization check. Rows fall in
//!   fold `row % 5`; each fold in turn validates child weights fitted on the
//!   other four, and with the second-order loss `G w + ½ H w²` the averaged
//!   in-fold and out-of-fold losses give `gen = (parent − train) / (parent −
//!   valid)`. A non-root split needs `gen ≥ 1` (relaxed by at most 0.01 for
//!   small nodes at depth; `0.99`-based for categorical splits) and rows on
//!   both sides in every fold; the root splits whenever a positive-gain
//!   split exists. Accepted splits are ranked by gain damped by fold-weight
//!   stability. Nodes grow best-first by their own score `G²/(H + 10⁻⁸)`;
//!   leaves use unregularized Newton weights `−G/(H + 10⁻⁸)`, clamped to
//!   `±max_delta_step` for `count:poisson` (XGBoost's default `0.7`, as its
//!   leaves are in regular training) and shrunk by `η`.
//!   Missing values try both directions (each counted in every fold); a tree
//!   holds at most 10 000 nodes.
//! * **Stopping.** A tree with at most one split whose generalization score
//!   is below `0.99` (and that did not stop on the loss target) counts as a
//!   weak round; boosting stops after `stopping_rounds` weak rounds, right
//!   after a tree whose root could not be split, after `stopping_rounds`
//!   rounds without a lower training loss, at the iteration cap, or before
//!   a tree that would make the training loss non-finite (it is not added).
//!   `stopping_rounds` defaults to `⌈3 · clamp(10^(0.5 (b − 1)⁺), 1, 6)⌉` (3
//!   for `b ≤ 1`); the hard **iteration cap** is
//!   `round(1000 · clamp(10^(0.35 (b − 1)⁺), 1, 4))` rounds (1000 for `b ≤ 1`,
//!   at most 4000), optionally lowered by [`BudgetConfig::iteration_limit`].
//!
//! Perpetual's dataset-regime heuristics (automatic row/column subsampling,
//! class reweighting, leaf-value refinement, linear heads, best-iteration
//! truncation, structural-plateau stopping, and objective/shape-specific
//! adjustments of the schedules above) are not reproduced, so results match
//! Perpetual's behavior in kind, not number for number. Losses are the
//! objectives' [`pointwise_loss`](crate::objective::Objective::pointwise_loss)
//! (deviance form for the log-link objectives, where Perpetual uses the
//! unshifted negative log-likelihood).
//!
//! # Parameters
//!
//! Budget mode derives the learning rate, tree size, and round count itself.
//! It reads `objective`, `num_class`, `base_score`, `max_bin`, `nthread`,
//! `missing`, and the one objective parameter the trained objective consumes
//! (`scale_pos_weight` for the logistic objectives, `huber_slope` for
//! `reg:pseudohubererror`, `tweedie_variance_power` for `reg:tweedie`,
//! `max_delta_step` for `count:poisson`); every other [`TrainingParams`]
//! field must keep its default, or training fails naming the fields.
//! Supported objectives are the single-output ones with a pointwise loss:
//! `reg:squarederror`, `reg:pseudohubererror`, `binary:logistic`,
//! `binary:logitraw`, `reg:logistic`, `count:poisson`, `reg:gamma`, and
//! `reg:tweedie`. Training is
//! deterministic (no random numbers are drawn) and independent of the thread
//! count.

use crate::config::TrainingParams;
use crate::data::DMatrix;
use crate::data::ghist::GHistIndex;
use crate::data::quantile::HistCuts;
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::objective::{GradPair, create_objective};
use crate::training::train::{
    initial_intercepts, new_model, reject_missing_param, validate_dataset, with_thread_pool,
};
use crate::tree::builder::budget::{
    ChildRecord, GENERALIZATION_THRESHOLD_RELAXED, GrowConfig, N_FOLDS, TreeStopper,
    fold_weight_spread, grow,
};

/// Perpetual's default budget.
pub const DEFAULT_BUDGET: f64 = 0.5;
/// Largest admissible budget (exclusive): for `b ≥ 5` the target loss
/// decrement formula is not positive.
pub const MAX_BUDGET: f64 = 5.0;
/// Weak/non-improving rounds tolerated before stopping (Perpetual
/// `STOPPING_ROUNDS`), before the budget scaling.
const STOPPING_ROUNDS: usize = 3;
/// Base iteration cap (Perpetual `ITER_LIMIT`), before the budget scaling.
const ITER_LIMIT: usize = 1000;

/// Configuration of [`train_with_budget`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetConfig {
    /// The fitting budget `b`, in `(0, 5)`. Larger budgets use a smaller
    /// learning rate and a smaller per-tree loss target, so they train more
    /// trees and fit more closely. Perpetual's default is `0.5`; `1.0` and
    /// `1.5` are common choices.
    pub budget: f64,
    /// Lower the hard iteration cap (`None`: the budget-derived cap).
    pub iteration_limit: Option<usize>,
    /// Override the number of weak or non-improving rounds that stop
    /// training (`None`: the budget-derived value).
    pub stopping_rounds: Option<usize>,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        BudgetConfig::new(DEFAULT_BUDGET)
    }
}

impl BudgetConfig {
    /// A configuration with the given budget and derived limits.
    pub fn new(budget: f64) -> Self {
        BudgetConfig {
            budget,
            iteration_limit: None,
            stopping_rounds: None,
        }
    }

    /// Set [`BudgetConfig::iteration_limit`].
    #[must_use]
    pub fn iteration_limit(mut self, limit: usize) -> Self {
        self.iteration_limit = Some(limit);
        self
    }

    /// Set [`BudgetConfig::stopping_rounds`].
    #[must_use]
    pub fn stopping_rounds(mut self, rounds: usize) -> Self {
        self.stopping_rounds = Some(rounds);
        self
    }

    /// Reject budgets outside `(0, 5)` and zero limits.
    pub fn validate(&self) -> Result<()> {
        if !(self.budget > 0.0 && self.budget < MAX_BUDGET) {
            return Err(HessboostError::invalid_param(
                "budget",
                format!("must be in (0, {MAX_BUDGET}), got {}", self.budget),
            ));
        }
        if self.iteration_limit == Some(0) {
            return Err(HessboostError::invalid_param(
                "iteration_limit",
                "must be at least 1",
            ));
        }
        if self.stopping_rounds == Some(0) {
            return Err(HessboostError::invalid_param(
                "stopping_rounds",
                "must be at least 1",
            ));
        }
        Ok(())
    }

    /// The learning rate the budget implies.
    pub fn eta(&self) -> f64 {
        let b = self.budget.max(0.0);
        let power = if b <= 1.0 { b } else { 1.0 + 0.65 * (b - 1.0) };
        10f64.powf(-power)
    }

    /// Growth factor `clamp(10^(exponent · (b − 1)⁺), 1, max)` of the base
    /// limits for budgets above 1.
    fn scale(&self, exponent: f64, max: f64) -> f64 {
        10f64
            .powf((self.budget - 1.0).max(0.0) * exponent)
            .clamp(1.0, max)
    }

    /// The effective number of weak or non-improving rounds that stop
    /// training.
    pub(crate) fn effective_stopping_rounds(&self) -> usize {
        self.stopping_rounds
            .unwrap_or_else(|| (STOPPING_ROUNDS as f64 * self.scale(0.5, 6.0)).ceil() as usize)
    }

    /// The effective hard cap on boosting rounds.
    pub(crate) fn effective_iteration_limit(&self) -> usize {
        let derived = (ITER_LIMIT as f64 * self.scale(0.35, 4.0)).round() as usize;
        self.iteration_limit
            .map_or(derived, |limit| limit.min(derived))
    }

    /// Target average per-row loss decrement for a loss level `loss_avg`.
    fn base_target(&self, loss_avg: f64) -> f64 {
        let u = self.budget.max(0.1);
        let n = 10.0 / u;
        let c = (n - 2.0) / (n * (n - 1.0));
        c * 10f64.powf(-u.min(3.0)) * loss_avg.max(f64::from(f32::EPSILON))
    }

    /// The round's target from the initial and the previous round's average
    /// loss.
    fn target(&self, initial_loss: f64, previous_loss: f64) -> f64 {
        if self.budget <= 0.2 {
            return self.base_target(initial_loss);
        }
        let alpha = (1.0 - 0.1 * self.budget).clamp(0.85, 0.995);
        self.base_target(alpha * initial_loss + (1.0 - alpha) * previous_loss) * 1.2
    }
}

/// Why budget-mode training stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetStop {
    /// The last tree's root had no split that passed the generalization
    /// check.
    RootUnsplittable,
    /// `stopping_rounds` trees with at most one weakly generalizing split.
    WeakTrees,
    /// The training loss did not improve for `stopping_rounds` rounds.
    NoImprovement,
    /// The iteration cap was reached.
    IterationLimit,
    /// The next tree would have made the training loss non-finite (an
    /// overflowing step); it was not added.
    NonFiniteLoss,
}

/// The result of [`train_with_budget`].
#[derive(Debug)]
pub struct BudgetResult {
    /// The trained model (one tree per round, leaves already shrunk by
    /// [`BudgetResult::eta`]).
    pub model: BoostedModel,
    /// The learning rate the budget implied.
    pub eta: f64,
    /// Why training stopped.
    pub stop: BudgetStop,
}

/// Train with a fitting budget instead of a learning rate, tree limits, and
/// a round count (see the [module docs](self) for the algorithm and the
/// parameters it reads).
///
/// ```
/// use hessboost::prelude::*;
/// use hessboost::training::budget::{BudgetConfig, train_with_budget};
///
/// let x: Vec<f32> = (0..400).map(|i| (i % 20) as f32).collect();
/// let y: Vec<f32> = x.iter().map(|v| (v * 0.3).sin()).collect();
/// let data = DMatrix::from_dense(&x, 400, 1)?.with_labels(&y)?;
/// let result = train_with_budget(&TrainingParams::default(), &data, &BudgetConfig::new(1.0))?;
/// assert!(result.model.num_trees() > 1);
/// # Ok::<(), HessboostError>(())
/// ```
pub fn train_with_budget(
    params: &TrainingParams,
    dtrain: &DMatrix,
    config: &BudgetConfig,
) -> Result<BudgetResult> {
    with_thread_pool(params, || train_budget_inner(params, dtrain, config))
}

/// Refuse every [`TrainingParams`] field budget mode does not read (they are
/// derived from the budget or have no budget-mode meaning), comparing the
/// serialized configuration against the defaults so newly added fields are
/// covered too. Of the objective parameters only the one the supported
/// objective consumes may differ from its default.
fn reject_tuned_params(params: &TrainingParams) -> Result<()> {
    let mut reference = TrainingParams::builder()
        .objective(&params.objective)
        .num_class(params.num_class)
        .build_unchecked();
    match params.objective.as_str() {
        "binary:logistic" | "binary:logitraw" | "reg:logistic" => {
            reference.scale_pos_weight = params.scale_pos_weight;
        }
        "reg:pseudohubererror" => reference.huber_slope = params.huber_slope,
        "reg:tweedie" => reference.tweedie_variance_power = params.tweedie_variance_power,
        // The Hessian safeguard and leaf-step bound of `count:poisson`.
        "count:poisson" => reference.max_delta_step = params.max_delta_step,
        _ => {}
    }
    reference.base_score = params.base_score;
    reference.max_bin = params.max_bin;
    reference.nthread = params.nthread;
    let (Ok(serde_json::Value::Object(set)), Ok(serde_json::Value::Object(allowed))) = (
        serde_json::to_value(params),
        serde_json::to_value(&reference),
    ) else {
        return Err(HessboostError::invalid_param(
            "budget",
            "training parameters could not be compared",
        ));
    };
    let changed: Vec<&str> = set
        .iter()
        .filter(|(key, value)| allowed.get(*key) != Some(*value))
        .map(|(key, _)| key.as_str())
        .collect();
    if changed.is_empty() {
        Ok(())
    } else {
        Err(HessboostError::invalid_param(
            "budget",
            format!(
                "budget mode derives the learning rate, tree shape, and round count itself; \
                 leave {} at the default",
                changed
                    .iter()
                    .map(|k| format!("`{k}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ))
    }
}

fn train_budget_inner(
    params: &TrainingParams,
    dtrain: &DMatrix,
    config: &BudgetConfig,
) -> Result<BudgetResult> {
    config.validate()?;
    params.validate()?;
    reject_missing_param(params)?;
    if dtrain.feature_weights().is_some() {
        return Err(HessboostError::invalid_param(
            "feature_weights",
            "budget mode does not sample columns",
        ));
    }
    let objective = create_objective(params, dtrain.n_targets())?;
    let n_out = objective.n_outputs();
    let loss_fn = match objective.pointwise_loss() {
        Some(loss) if n_out == 1 => loss,
        _ => {
            return Err(HessboostError::invalid_param(
                "objective",
                format!(
                    "`{}` is not supported by budget mode (it needs a single-output \
                     objective with a pointwise loss)",
                    objective.name()
                ),
            ));
        }
    };
    reject_tuned_params(params)?;
    let Some(labels) = dtrain.labels() else {
        return Err(HessboostError::EmptyDataset(
            "train_with_budget: dtrain has no labels",
        ));
    };
    let n = dtrain.n_rows();
    let n_features = dtrain.n_cols();
    validate_dataset(
        objective.as_ref(),
        dtrain,
        dtrain.n_targets(),
        n_features,
        n_out,
        "dtrain",
    )?;
    let info = dtrain.info();
    let base_margins = initial_intercepts(params, objective.as_ref(), &info, n_out)?;
    let mut model = new_model(params, objective.as_ref(), dtrain, base_margins);

    let ghist = GHistIndex::from_dmatrix(dtrain, HistCuts::from_dmatrix(dtrain, params.max_bin));
    let weights = dtrain.weights();
    let weight_of = |r: usize| weights.map_or(1.0, |w| f64::from(w[r]));
    let row_loss = |r: usize, margin: f32| weight_of(r) * loss_fn(margin, labels[r]);
    let mut margins = model.initial_margins(dtrain);
    let mut loss: Vec<f64> = (0..n).map(|r| row_loss(r, margins[r])).collect();
    let average = |loss: &[f64]| loss.iter().sum::<f64>() / n.max(1) as f64;

    let eta = config.eta();
    let stopping_rounds = config.effective_stopping_rounds();
    let regression_like = matches!(
        objective.name(),
        "reg:squarederror" | "reg:pseudohubererror"
    );
    let initial_loss = average(&loss);
    let mut previous_loss = initial_loss;
    let mut best_loss = initial_loss;
    let mut weak_rounds = 0usize;
    let mut untargeted_rounds = 0usize;
    let mut no_improvement = 0usize;
    let mut gpair = vec![GradPair::default(); n];
    let mut stop = BudgetStop::IterationLimit;

    for _ in 0..config.effective_iteration_limit() {
        let target = (untargeted_rounds <= stopping_rounds.saturating_add(1))
            .then(|| config.target(initial_loss, previous_loss));
        objective.gradient_info(&margins, &info, &mut gpair);
        let row_decrement = |r: u32, delta: f32| {
            let r = r as usize;
            loss[r] - row_loss(r, margins[r] + delta)
        };
        let grown = grow(
            &ghist,
            &gpair,
            &GrowConfig {
                eta: eta as f32,
                target_loss_decrement: target,
                row_decrement: &row_decrement,
                max_delta_step: params.effective_max_delta_step(),
            },
        );
        grown.apply(&mut margins);

        let n_nodes = grown.tree.num_nodes();
        let generalization = tree_generalization(&grown.children, regression_like);
        let mut stop_now = false;
        if n_nodes < 5
            && generalization < GENERALIZATION_THRESHOLD_RELAXED
            && grown.stopper != TreeStopper::StepSize
        {
            weak_rounds += 1;
            stop_now = n_nodes == 1;
        }
        if grown.stopper == TreeStopper::StepSize {
            untargeted_rounds = 0;
        } else {
            untargeted_rounds += 1;
        }

        for (r, l) in loss.iter_mut().enumerate() {
            *l = row_loss(r, margins[r]);
        }
        let current_loss = average(&loss);
        if !current_loss.is_finite() {
            // The step overflowed the loss (and would poison the next
            // round's gradients): keep the model built so far.
            stop = BudgetStop::NonFiniteLoss;
            break;
        }
        previous_loss = current_loss;
        if current_loss < best_loss {
            best_loss = current_loss;
            no_improvement = 0;
        } else {
            no_improvement += 1;
        }
        model.push_tree_weighted(grown.tree, 1.0);

        if stop_now {
            stop = BudgetStop::RootUnsplittable;
            break;
        }
        if weak_rounds >= stopping_rounds {
            stop = BudgetStop::WeakTrees;
            break;
        }
        if no_improvement >= stopping_rounds {
            stop = BudgetStop::NoImprovement;
            break;
        }
    }

    Ok(BudgetResult { model, eta, stop })
}

/// Sign agreement and spread of a node's fold weights (booster form):
/// `max(share≥0, share<0) / (1 + σ/|w̄|)` clamped to `[0.5, 1]`.
fn fold_weight_reliability(weights: &[f64; N_FOLDS]) -> f64 {
    fold_weight_spread(weights).map_or(1.0, |(mean_abs, std_dev)| {
        let positive = weights.iter().filter(|&&w| w >= 0.0).count() as f64 / N_FOLDS as f64;
        (positive.max(1.0 - positive) / (1.0 + std_dev / mean_abs)).clamp(0.5, 1.0)
    })
}

/// A tree's generalization score: the best node score for classification
/// and count objectives; for regression, the node-size- and
/// stability-weighted mean of node scores bounded to `[0.95, 1.05]`. A tree
/// without splits scores `0`.
fn tree_generalization(children: &[ChildRecord], regression_like: bool) -> f64 {
    let mut best = 0.0f64;
    let (mut weighted, mut total) = (0.0, 0.0);
    for child in children {
        let stability = fold_weight_reliability(&child.fold_weights);
        let node_score = child.generalization * (0.99 + 0.01 * stability);
        best = best.max(node_score);
        let node_weight = (child.count.max(1) as f64).sqrt() * stability;
        weighted += node_score.clamp(0.95, 1.05) * node_weight;
        total += node_weight;
    }
    match (regression_like, total > 0.0) {
        (true, true) => weighted / total,
        (true, false) => 0.0,
        (false, _) => best,
    }
}
