//! Evaluation during training: eval sets, metrics, early stopping, and the
//! per-round reporter.

use super::api::{EvalHistory, RoundEval, RoundHook, TrainResult};
use super::margins::MarginCaches;
use crate::config::TrainingParams;
use crate::data::{DMatrix, MetaInfo};
use crate::error::{HessboostError, Result};
use crate::metric::Metric;
use crate::model::BoostedModel;
use crate::objective::Loss;
use std::num::NonZeroUsize;
use std::ops::ControlFlow;

/// A named evaluation dataset watched during training.
#[derive(Clone, Copy)]
pub(super) struct EvalSet<'a> {
    pub(super) data: &'a DMatrix,
    /// The name its scores are reported under.
    pub(super) name: &'a str,
}

/// The eval sets' metrics and the buffers their predictions are transformed
/// and scored in, reused every round.
pub(super) struct EvalPlan<'a> {
    objective: &'a dyn Loss,
    evals: &'a [EvalSet<'a>],
    infos: Vec<MetaInfo<'a>>,
    pub(super) metrics: Vec<Box<dyn Metric>>,
    preds: Vec<f32>,
    /// One round's values, `[dataset][metric]`.
    scores: Vec<f64>,
}

impl<'a> EvalPlan<'a> {
    /// The metrics every eval set reports: the configured or default ones,
    /// then `metric_override` (as XGBoost's `xgb.train` appends its
    /// `custom_metric`), refusing a metric that cannot read the label
    /// layout or the model's prediction width.
    pub(super) fn new(
        params: &TrainingParams,
        objective: &'a dyn Loss,
        metric_override: Option<Box<dyn Metric>>,
        evals: &'a [EvalSet<'a>],
        n_targets: usize,
    ) -> Result<Self> {
        let n_out = objective.n_outputs();
        let mut metrics = configured_metrics(params, objective)?;
        metrics.extend(metric_override);
        if n_targets > 1
            && let Some(metric) = metrics.iter().find(|m| !m.supports_label_matrix())
        {
            return Err(HessboostError::invalid_param(
                "eval_metric",
                format!(
                    "metric `{}` does not support multi-target labels",
                    metric.name()
                ),
            ));
        }
        let infos: Vec<MetaInfo> = evals.iter().map(|set| set.data.info()).collect();
        for (info, set) in infos.iter().zip(evals) {
            for metric in &metrics {
                metric
                    .validate_info(info)
                    .map_err(|error| error.in_dataset(set.name))?;
                check_prediction_width(metric.as_ref(), info, n_out, set.name)?;
            }
        }
        Ok(EvalPlan {
            objective,
            evals,
            infos,
            metrics,
            preds: Vec::new(),
            scores: Vec::new(),
        })
    }

    /// Whether the early-stopping metric (the last one) is maximized.
    pub(super) fn maximize(&self) -> bool {
        self.metrics.last().is_some_and(|m| m.maximize())
    }

    /// An empty history of these eval sets and metrics, starting at model
    /// iteration `first_iteration`.
    fn history(&self, first_iteration: usize) -> EvalHistory {
        EvalHistory::new(
            self.evals.iter().map(|set| set.name.to_owned()).collect(),
            self.metrics.iter().map(|m| m.name().to_owned()).collect(),
            first_iteration,
        )
    }

    /// Evaluate every metric on every eval set's `margins`, append the
    /// round to `history`, and return the last value (the early-stopping
    /// metric of the last eval set).
    fn record(&mut self, margins: &MarginCaches, history: &mut EvalHistory) -> f64 {
        self.scores.clear();
        let mut last_metric_value = 0.0;
        for ei in 0..self.evals.len() {
            self.preds.clear();
            self.preds.extend_from_slice(&margins.evals[ei]);
            self.objective.eval_transform(&mut self.preds);
            for m in &self.metrics {
                let v = m.eval_info(&self.preds, &self.infos[ei]);
                self.scores.push(v);
                last_metric_value = v;
            }
        }
        history.push_round(self.scores.iter().copied());
        last_metric_value
    }
}

/// XGBoost's early-stopping rule: a round improves on the best score so
/// far only strictly (so a NaN score never does), and training stops after
/// `patience` rounds without improvement. Shared by [`Trainer`](super::Trainer) and
/// [`CrossValidation`](crate::training::CrossValidation).
pub(super) struct EarlyStopping {
    patience: NonZeroUsize,
    maximize: bool,
    best_score: f64,
    best_round: usize,
    since_improved: usize,
}

impl EarlyStopping {
    /// Tracking that starts at round `first_round`, which stays the best
    /// one when no score ever improves.
    pub(crate) fn new(patience: NonZeroUsize, maximize: bool, first_round: usize) -> Self {
        EarlyStopping {
            patience,
            maximize,
            best_score: if maximize {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            },
            best_round: first_round,
            since_improved: 0,
        }
    }

    /// Record `round`'s `score`; `true` once patience has run out.
    pub(crate) fn observe(&mut self, round: usize, score: f64) -> bool {
        let improved = if self.maximize {
            score > self.best_score
        } else {
            score < self.best_score
        };
        if improved {
            self.best_score = score;
            self.best_round = round;
            self.since_improved = 0;
            false
        } else {
            self.since_improved += 1;
            self.since_improved >= self.patience.get()
        }
    }

    /// The best round so far.
    pub(crate) fn best_round(&self) -> usize {
        self.best_round
    }
}

/// The metrics `params` configure for `loss`: `params.eval_metric`, or the
/// loss's default, each built for its output count. Without a custom metric
/// the last one is the early-stopping metric.
pub(super) fn configured_metrics(
    params: &TrainingParams,
    loss: &dyn Loss,
) -> Result<Vec<Box<dyn Metric>>> {
    let n_outputs = loss.n_outputs();
    if params.eval_metric.is_empty() {
        Ok(vec![loss.default_metric().build(n_outputs)?])
    } else {
        params
            .eval_metric
            .iter()
            .map(|metric| metric.build(n_outputs))
            .collect()
    }
}

/// Refuse a metric that reads a different number of predictions per row
/// than the model's `n_out` outputs (XGBoost's "label and prediction size
/// not match"): an elementwise metric on an alpha-list, multiclass, or
/// distributional model, a multiclass metric on a single-output model, and
/// so on. A metric of any width ([`Metric::prediction_width`] `None`, the
/// custom-metric hook) needs a whole number of outputs per label column.
/// Fails with [`HessboostError::InvalidParameter`] (`eval_metric`), the
/// reason naming `dataset`.
///
/// [`Metric::prediction_width`]: crate::metric::Metric::prediction_width
fn check_prediction_width(
    metric: &dyn crate::metric::Metric,
    info: &MetaInfo,
    n_out: usize,
    dataset: &str,
) -> Result<()> {
    let reason = match metric.prediction_width(info) {
        Some(width) if width != n_out => format!(
            "metric `{}` reads {width} prediction(s) per row of dataset `{dataset}`, but the \
             model has {n_out} outputs",
            metric.name()
        ),
        None if !n_out.is_multiple_of(info.n_targets().max(1)) => format!(
            "metric `{}` needs a whole number of the model's {n_out} outputs per label \
             column of dataset `{dataset}` ({} columns)",
            metric.name(),
            info.n_targets()
        ),
        _ => return Ok(()),
    };
    Err(HessboostError::invalid_param("eval_metric", reason))
}

/// What a run reports after each round: the eval sets' scores (appended to
/// the history), the early-stopping state, and the [`Trainer::on_round`](super::Trainer::on_round)
/// hook, in that order.
pub(super) struct RoundReporter<'a> {
    /// The eval sets' metrics; `None` without eval sets.
    eval_plan: Option<EvalPlan<'a>>,
    history: EvalHistory,
    stopping: Option<EarlyStopping>,
    on_round: Option<RoundHook<'a>>,
}

impl<'a> RoundReporter<'a> {
    /// A reporter that only runs `on_round` (with empty scores) until
    /// [`Self::watch`] gives it eval sets.
    pub(super) fn new(on_round: Option<RoundHook<'a>>) -> Self {
        RoundReporter {
            eval_plan: None,
            history: EvalHistory::default(),
            stopping: None,
            on_round,
        }
    }

    /// Score `plan`'s eval sets after every round (when it has any), and
    /// with `early_stopping_rounds` track the watched metric from
    /// `first_round`.
    pub(super) fn watch(
        &mut self,
        plan: EvalPlan<'a>,
        early_stopping_rounds: Option<NonZeroUsize>,
        first_round: usize,
    ) {
        self.stopping = early_stopping_rounds
            .map(|patience| EarlyStopping::new(patience, plan.maximize(), first_round));
        if !plan.evals.is_empty() {
            self.history = plan.history(first_round);
            self.eval_plan = Some(plan);
        }
    }

    /// Report round `iteration`: score the eval sets on `margins` (a round
    /// without eval sets passes `None`), update early stopping, and call the
    /// hook. `Break` once patience runs out or the hook breaks.
    pub(super) fn finish_round(
        &mut self,
        iteration: usize,
        margins: Option<&MarginCaches>,
    ) -> ControlFlow<()> {
        let mut stop = false;
        let mut scored = false;
        if let (Some(plan), Some(margins)) = (&mut self.eval_plan, margins) {
            let score = plan.record(margins, &mut self.history);
            scored = true;
            if let Some(stopping) = &mut self.stopping {
                stop = stopping.observe(iteration, score);
            }
        }
        if let Some(hook) = &mut self.on_round {
            let round = match self.history.last() {
                Some(round) if scored => round,
                _ => RoundEval::unscored(iteration),
            };
            debug_assert_eq!(round.iteration(), iteration);
            stop |= hook(round).is_break();
        }
        if stop {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    /// The run's result for the trained `model`, with the history and, under
    /// early stopping, the best round. XGBoost records the best iteration
    /// whenever early stopping is on, not only when patience runs out.
    pub(super) fn into_result(self, mut model: BoostedModel) -> TrainResult {
        let mut best_score = None;
        if let Some(stopping) = &self.stopping
            && !self.history.is_empty()
        {
            let best_iter = stopping.best_round();
            best_score = self
                .history
                .round(best_iter)
                .and_then(|round| round.values().last().copied());
            // A shrunk model's later iterations rescaled the best one, so keep
            // the model as it was after the best iteration (CatBoost's
            // `use_best_model`) instead of hiding the rest behind the selection.
            model.truncate_shrunk(best_iter + 1);
            model.set_best_iteration(Some(best_iter));
        }
        TrainResult {
            model,
            history: self.history,
            best_score,
        }
    }
}
