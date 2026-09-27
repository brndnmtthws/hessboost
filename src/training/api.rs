//! The public training API: [`Trainer`], [`train`], and the evaluation history.

use super::eval::EvalSet;
use super::train::{train_impl, with_thread_pool};
use super::validate::validate_trained_model;

use crate::config::TrainingParams;
use crate::data::DMatrix;
use crate::error::Result;
use crate::metric::Metric;
use crate::model::BoostedModel;
use std::ops::ControlFlow;

/// One row of the evaluation history: the metric values computed at the end of
/// a boosting round.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RoundEval {
    /// The 0-based boosting iteration of the model (after continued training,
    /// counted from the start of the initial model).
    pub iteration: usize,
    /// Every eval set's metric values, in eval-set order and, within a set,
    /// in metric order (a [`Trainer::custom_metric`] last).
    pub scores: Vec<Score>,
}

impl RoundEval {
    /// The value of `metric` on the eval set named `dataset`, if recorded.
    pub fn score(&self, dataset: &str, metric: &str) -> Option<f64> {
        self.scores
            .iter()
            .find(|score| score.dataset == dataset && score.metric == metric)
            .map(|score| score.value)
    }
}

/// One metric value on one eval set in a [`RoundEval`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Score {
    /// The eval set's name, as passed to [`Trainer::eval`].
    pub dataset: String,
    /// The metric's name: XGBoost's `evals_result` key (`rmse`, `ndcg@5`).
    pub metric: String,
    /// The metric's value.
    pub value: f64,
}

/// The result of [`Trainer::train`]: the model plus the per-round evaluation
/// history.
#[derive(Debug)]
#[non_exhaustive]
pub struct TrainResult {
    /// The trained model.
    pub model: BoostedModel,
    /// Evaluation history (empty when no eval sets were supplied). Each
    /// entry's `iteration` is the model's absolute iteration index, which
    /// after continued training starts at the initial model's
    /// [`num_boost_rounds`](BoostedModel::num_boost_rounds).
    pub history: Vec<RoundEval>,
    /// With [`early_stopping_rounds`](Trainer::early_stopping_rounds), the
    /// watched metric's value at the model's
    /// [`best_iteration`](BoostedModel::best_iteration) (XGBoost's
    /// `best_score`), whether or not training stopped early; `None` without
    /// early stopping or when no round ran.
    pub best_score: Option<f64>,
}

/// Train a model for `num_boost_round` iterations with no eval sets, early
/// stopping, or custom hooks. Shorthand for
/// `Trainer::new(params, dtrain, num_boost_round).train()?.model`; use
/// [`Trainer`] for everything else.
pub fn train(
    params: &TrainingParams,
    dtrain: &DMatrix,
    num_boost_round: usize,
) -> Result<BoostedModel> {
    Ok(Trainer::new(params, dtrain, num_boost_round).train()?.model)
}

/// Configures one training run: XGBoost's `xgb.train` with its optional
/// arguments as builder methods.
///
/// ```
/// use hessboost::prelude::*;
///
/// # fn main() -> Result<()> {
/// let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let dtrain = DMatrix::from_dense(&x[..6], 6, 1)?.with_labels(&x[..6])?;
/// let dvalid = DMatrix::from_dense(&x[6..], 2, 1)?.with_labels(&x[6..])?;
/// let params = TrainingParams::builder().max_depth(2).build()?;
///
/// let result = Trainer::new(&params, &dtrain, 100)
///     .eval(&dvalid, "valid")
///     .early_stopping_rounds(5)
///     .train()?;
/// assert!(!result.history.is_empty());
///
/// // Continue boosting from the trained model.
/// let more = Trainer::new(&params, &dtrain, 10)
///     .init_model(&result.model)
///     .train()?
///     .model;
/// assert_eq!(more.num_boost_rounds(), result.model.num_boost_rounds() + 10);
/// # Ok(())
/// # }
/// ```
pub struct Trainer<'a> {
    pub(super) params: &'a TrainingParams,
    pub(super) dtrain: &'a DMatrix,
    pub(super) num_boost_round: usize,
    pub(super) evals: Vec<EvalSet<'a>>,
    pub(super) early_stopping_rounds: Option<usize>,
    pub(super) metric: Option<Box<dyn Metric>>,
    pub(super) init_model: Option<&'a BoostedModel>,
    pub(super) on_round: Option<RoundHook<'a>>,
}

/// The per-round hook of [`Trainer::on_round`].
pub(super) type RoundHook<'a> = Box<dyn FnMut(&RoundEval) -> ControlFlow<()> + Send + 'a>;

impl<'a> Trainer<'a> {
    /// Train on `dtrain` for (at most) `num_boost_round` iterations with
    /// `params`, which name the objective and metrics.
    pub fn new(params: &'a TrainingParams, dtrain: &'a DMatrix, num_boost_round: usize) -> Self {
        Trainer {
            params,
            dtrain,
            num_boost_round,
            evals: Vec::new(),
            early_stopping_rounds: None,
            metric: None,
            init_model: None,
            on_round: None,
        }
    }

    /// Evaluate the metrics on `data` after every round, reporting them
    /// under `name` in [`TrainResult::history`]. Call once per eval set; the
    /// order is kept.
    #[must_use]
    pub fn eval(mut self, data: &'a DMatrix, name: &'a str) -> Self {
        self.evals.push(EvalSet { data, name });
        self
    }

    /// Stop when the watched metric fails to improve for `rounds`
    /// consecutive rounds. As in XGBoost, the watched metric is the **last**
    /// metric of the **last** eval set. Needs at least one
    /// [`eval`](Self::eval) set and `rounds > 0`.
    ///
    /// The model's [`best_iteration`](BoostedModel::best_iteration) and
    /// [`TrainResult::best_score`] record the best round whether training
    /// stopped early or ran all `rounds` (as XGBoost's `best_iteration`
    /// does), so plain prediction uses the iterations up to the best one
    /// either way. When the metric never improves (it is NaN), the best
    /// round is this run's first.
    ///
    /// A model trained with model shrinkage
    /// ([`model_shrink`](crate::config::TrainingParams::model_shrink),
    /// posterior sampling) is instead cut back to its best iteration, as
    /// CatBoost's `use_best_model` does: every later iteration rescaled the
    /// earlier ones, so the returned model is the one the run held after the
    /// best iteration, and its `best_iteration` is its last.
    ///
    /// After [`init_model`](Self::init_model) the early-stopping state starts
    /// fresh; `best_iteration` and the history's iterations are absolute
    /// iteration indices of the continued model (XGBoost's `starting_round`
    /// offset).
    #[must_use]
    pub fn early_stopping_rounds(mut self, rounds: usize) -> Self {
        self.early_stopping_rounds = Some(rounds);
        self
    }

    /// Also report `metric` (the custom-metric hook, e.g. a
    /// [`CustomMetric`](crate::metric::CustomMetric)), as XGBoost's
    /// `xgb.train(custom_metric=...)` does: every eval set reports the
    /// `eval_metric` list (or the loss's default metric) and then `metric`,
    /// which, being last, drives early stopping (per its
    /// [`maximize`](Metric::maximize)).
    #[must_use]
    pub fn custom_metric(mut self, metric: Box<dyn Metric>) -> Self {
        self.metric = Some(metric);
        self
    }

    /// Continue training `model` for `num_boost_round` more iterations
    /// (XGBoost's `xgb.train(..., xgb_model=model)`).
    ///
    /// The new iterations start from `model`'s full current margins (every
    /// tree, whatever its `best_iteration`) and are appended to a copy of it.
    /// The copy keeps the model's intercepts unless `params.base_score` is
    /// set, which replaces them (as XGBoost's `set_param` does); the
    /// intercept is never re-estimated. `params` must use the model's
    /// objective, `num_class`, `num_parallel_tree`, booster family (tree or
    /// `gblinear`), feature count and label width; they otherwise drive the
    /// new iterations, including the objective's hyper-parameters, which the
    /// result records. The per-round RNG continues from the model's iteration
    /// count, so training `a` rounds and continuing for `b` grows the same
    /// trees as training `a + b` rounds with the same parameters. DART tree
    /// weights carry over and are rescaled by later dropouts. The copy's
    /// `best_iteration` is cleared.
    ///
    /// With `process_type=update` the model's trees are not extended but
    /// refreshed on `dtrain` (XGBoost's `updater=refresh`): round `i`
    /// recomputes the statistics, and with `refresh_leaf` the leaf values, of
    /// iteration `i`'s trees from the gradients of the already refreshed
    /// iterations. The result holds exactly the `num_boost_round` refreshed
    /// iterations (at most the model's count), as in XGBoost. Update mode
    /// needs a gbtree model without DART weights or linear leaves, no
    /// monotone constraints, and no feature weights on `dtrain`; settings
    /// refresh does not read (row and column sampling, symmetric growth,
    /// DART dropout, the beyond-XGBoost tree options) must keep their
    /// defaults, while XGBoost's tree-shape settings (`tree_method`,
    /// `max_depth`, `min_child_weight`, ...) are accepted.
    ///
    /// Model shrinkage is refused on both sides, as in CatBoost: a model
    /// trained with it cannot be continued, and shrinkage parameters cannot
    /// continue a model. Langevin noise without shrinkage (an explicit
    /// `model_shrink_rate` of `0`) continues exactly: its draws are keyed by
    /// the absolute iteration.
    ///
    /// The model must be structurally valid, as every loaded model is.
    #[must_use]
    pub fn init_model(mut self, model: &'a BoostedModel) -> Self {
        self.init_model = Some(model);
        self
    }

    /// Call `hook` after every boosting round, once the round's eval sets
    /// are scored: progress reporting, custom stopping rules, or
    /// cancellation. It runs on the training thread, in round order, and
    /// sees the round as [`TrainResult::history`] records it (with empty
    /// `scores` when there are no eval sets). Returning
    /// [`ControlFlow::Break`] ends training after that round; the result
    /// keeps every completed round and is otherwise what training for that
    /// many rounds would have produced.
    ///
    /// With [`early_stopping_rounds`](Self::early_stopping_rounds), the
    /// hook also sees the round on which patience runs out, and a `Break`
    /// still records the best round so far as
    /// [`best_iteration`](BoostedModel::best_iteration) (with its
    /// [`TrainResult::best_score`]). With `process_type=update` the model
    /// holds the iterations refreshed so far; with `booster = boulevard` it
    /// is the Boulevard average of the rounds run so far. For `gblinear`, which stores
    /// no boosting iterations, [`RoundEval::iteration`] counts this run's
    /// rounds from 0. For `booster = ebm` it counts EBM rounds from 0
    /// through the main-effect stage and on through the pair stage (up to
    /// `num_boost_round` each, fewer once every bag has early-stopped); a
    /// `Break` keeps the completed rounds (each bag's best ones under
    /// `ebm_early_stopping_rounds`), a stopped main-effect stage gets no
    /// pair terms, and with interactions the result is then not a shorter
    /// run's model.
    ///
    /// Observing never changes the model: training with a hook that always
    /// continues gives the same result as training without one.
    ///
    /// ```
    /// use hessboost::prelude::*;
    /// use std::ops::ControlFlow;
    ///
    /// # fn main() -> Result<()> {
    /// let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    /// let dtrain = DMatrix::from_dense(&x, 6, 1)?.with_labels(&x)?;
    /// let params = TrainingParams::builder().max_depth(2).build()?;
    /// let mut seen = Vec::new();
    /// let result = Trainer::new(&params, &dtrain, 100)
    ///     .on_round(|round| {
    ///         seen.push(round.iteration);
    ///         if round.iteration == 4 { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
    ///     })
    ///     .train()?;
    /// assert_eq!(result.model.num_boost_rounds(), 5);
    /// assert_eq!(seen, [0, 1, 2, 3, 4]);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn on_round(mut self, hook: impl FnMut(&RoundEval) -> ControlFlow<()> + Send + 'a) -> Self {
        self.on_round = Some(Box::new(hook));
        self
    }

    /// Run the configured training.
    pub fn train(self) -> Result<TrainResult> {
        let loss = self.params.loss(self.dtrain.n_targets())?;
        let params = self.params;
        let result = with_thread_pool(params, || train_impl(self, loss.as_ref()))?;
        validate_trained_model(&result.model)?;
        Ok(result)
    }
}
