//! The public training API: [`Trainer`], [`train`], and the evaluation history.

use super::eval::EvalSet;
use super::train::{train_impl, with_thread_pool};
use super::validate::validate_trained_model;

use crate::config::TrainingParams;
use crate::data::DMatrix;
use crate::error::Result;
use crate::metric::Metric;
use crate::model::BoostedModel;
use std::num::NonZeroUsize;
use std::ops::ControlFlow;

/// The evaluation history of a training run: every eval set's metric values
/// after each boosting round, stored once per round as a
/// `[dataset][metric]` block of values with the names kept once.
///
/// Rounds are contiguous: round `r` of [`rounds`](Self::rounds) is the
/// model's iteration [`first_iteration`](Self::first_iteration)` + r`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EvalHistory {
    datasets: Vec<String>,
    metrics: Vec<String>,
    first_iteration: usize,
    /// `[round][dataset][metric]`.
    values: Vec<f64>,
}

impl EvalHistory {
    /// An empty history of `datasets` × `metrics` whose first round will be
    /// iteration `first_iteration`.
    pub(super) fn new(datasets: Vec<String>, metrics: Vec<String>, first_iteration: usize) -> Self {
        EvalHistory {
            datasets,
            metrics,
            first_iteration,
            values: Vec::new(),
        }
    }

    /// Append one round's values, `[dataset][metric]`.
    pub(super) fn push_round(&mut self, values: impl IntoIterator<Item = f64>) {
        let before = self.values.len();
        self.values.extend(values);
        debug_assert_eq!(self.values.len() - before, self.round_width());
    }

    /// The eval sets' names, in [`Trainer::eval`] order.
    pub fn datasets(&self) -> &[String] {
        &self.datasets
    }

    /// The metrics' names (XGBoost's `evals_result` keys, `rmse`,
    /// `ndcg@5`), in the order every eval set reports them (a
    /// [`Trainer::custom_metric`] last).
    pub fn metrics(&self) -> &[String] {
        &self.metrics
    }

    /// The model iteration of the first recorded round (after continued
    /// training, counted from the start of the initial model).
    pub fn first_iteration(&self) -> usize {
        self.first_iteration
    }

    /// The number of recorded rounds.
    pub fn len(&self) -> usize {
        match self.round_width() {
            0 => 0,
            width => self.values.len() / width,
        }
    }

    /// Whether no round was recorded (always so without eval sets).
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The recorded rounds, in order.
    pub fn rounds(&self) -> impl ExactSizeIterator<Item = RoundEval<'_>> + DoubleEndedIterator {
        (0..self.len()).map(|round| self.view(round))
    }

    /// The recorded round of model iteration `iteration`, if any.
    pub fn round(&self, iteration: usize) -> Option<RoundEval<'_>> {
        let round = iteration.checked_sub(self.first_iteration)?;
        (round < self.len()).then(|| self.view(round))
    }

    /// The last recorded round, if any.
    pub fn last(&self) -> Option<RoundEval<'_>> {
        self.len().checked_sub(1).map(|round| self.view(round))
    }

    /// The value of `metric` on the eval set named `dataset` in every
    /// recorded round, in order; `None` when either name is not recorded.
    pub fn series(
        &self,
        dataset: &str,
        metric: &str,
    ) -> Option<impl ExactSizeIterator<Item = f64> + '_> {
        let cell = self.cell(dataset, metric)?;
        let width = self.round_width();
        Some((0..self.len()).map(move |round| self.values[round * width + cell]))
    }

    /// Values per round: one per (eval set, metric).
    fn round_width(&self) -> usize {
        self.datasets.len() * self.metrics.len()
    }

    /// The position of (`dataset`, `metric`) within a round's values.
    fn cell(&self, dataset: &str, metric: &str) -> Option<usize> {
        let d = self.datasets.iter().position(|name| name == dataset)?;
        let m = self.metrics.iter().position(|name| name == metric)?;
        Some(d * self.metrics.len() + m)
    }

    /// The view of recorded round `round` (0-based).
    fn view(&self, round: usize) -> RoundEval<'_> {
        let width = self.round_width();
        RoundEval {
            iteration: self.first_iteration + round,
            history: Some(self),
            values: &self.values[round * width..(round + 1) * width],
        }
    }
}

/// One round of an [`EvalHistory`], borrowed: the metric values computed at
/// the end of a boosting round, which [`Trainer::on_round`] also sees.
#[derive(Debug, Clone, Copy)]
pub struct RoundEval<'a> {
    iteration: usize,
    /// The history holding the names (`None` for a round without eval sets).
    history: Option<&'a EvalHistory>,
    /// `[dataset][metric]`.
    values: &'a [f64],
}

impl<'a> RoundEval<'a> {
    /// A round without eval sets: no values.
    pub(super) fn unscored(iteration: usize) -> Self {
        RoundEval {
            iteration,
            history: None,
            values: &[],
        }
    }

    /// The 0-based boosting iteration of the model (after continued training,
    /// counted from the start of the initial model).
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Every eval set's metric values, in eval-set order and, within a set,
    /// in [`EvalHistory::metrics`] order (empty without eval sets). The last
    /// one is the early-stopping metric's on the last eval set.
    pub fn values(&self) -> &'a [f64] {
        self.values
    }

    /// The value of `metric` on the eval set named `dataset`, if recorded.
    pub fn score(&self, dataset: &str, metric: &str) -> Option<f64> {
        let cell = self.history?.cell(dataset, metric)?;
        Some(self.values[cell])
    }

    /// Every `(dataset, metric, value)`, in [`values`](Self::values) order.
    pub fn scores(&self) -> impl Iterator<Item = (&'a str, &'a str, f64)> + 'a {
        let (datasets, metrics): (&[String], &[String]) = match self.history {
            Some(history) => (&history.datasets, &history.metrics),
            None => (&[], &[]),
        };
        datasets
            .iter()
            .flat_map(move |dataset| metrics.iter().map(move |metric| (dataset, metric)))
            .zip(self.values)
            .map(|((dataset, metric), &value)| (dataset.as_str(), metric.as_str(), value))
    }
}

/// The result of [`Trainer::train`]: the model plus the per-round evaluation
/// history.
#[derive(Debug)]
#[non_exhaustive]
pub struct TrainResult {
    /// The trained model.
    pub model: BoostedModel,
    /// Evaluation history (empty when no eval sets were supplied). Its
    /// iterations are the model's absolute iteration indices, which after
    /// continued training start at the initial model's
    /// [`num_boost_rounds`](BoostedModel::num_boost_rounds).
    pub history: EvalHistory,
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
/// use std::num::NonZeroUsize;
///
/// # fn main() -> Result<()> {
/// let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let dtrain = DMatrix::from_dense(&x[..6], 6, 1)?.with_labels(&x[..6])?;
/// let dvalid = DMatrix::from_dense(&x[6..], 2, 1)?.with_labels(&x[6..])?;
/// let params = TrainingParams::builder().max_depth(2).build()?;
///
/// let result = Trainer::new(&params, &dtrain, 100)
///     .eval(&dvalid, "valid")
///     .early_stopping_rounds(NonZeroUsize::new(5).unwrap())
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
    pub(super) early_stopping_rounds: Option<NonZeroUsize>,
    pub(super) metric: Option<Box<dyn Metric>>,
    pub(super) init_model: Option<&'a BoostedModel>,
    pub(super) on_round: Option<RoundHook<'a>>,
}

/// The per-round hook of [`Trainer::on_round`].
pub(super) type RoundHook<'a> = Box<dyn FnMut(RoundEval<'_>) -> ControlFlow<()> + Send + 'a>;

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
    /// [`eval`](Self::eval) set.
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
    pub fn early_stopping_rounds(mut self, rounds: NonZeroUsize) -> Self {
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
    /// sees the round as [`TrainResult::history`] records it (with no
    /// [`values`](RoundEval::values) when there are no eval sets). Returning
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
    ///         seen.push(round.iteration());
    ///         if round.iteration() == 4 { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
    ///     })
    ///     .train()?;
    /// assert_eq!(result.model.num_boost_rounds(), 5);
    /// assert_eq!(seen, [0, 1, 2, 3, 4]);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn on_round(
        mut self,
        hook: impl FnMut(RoundEval<'_>) -> ControlFlow<()> + Send + 'a,
    ) -> Self {
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
