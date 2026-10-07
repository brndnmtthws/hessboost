//! Cross-validation, mirroring `xgboost.cv`: shuffled k-fold ([`cv`]) or
//! caller-supplied folds ([`CrossValidation`], [`Fold`]), including
//! forward-chaining folds for time-ordered rows, forward folds purged by
//! each row's label window ([`Fold::purged_forward`]), whole-query folds of
//! ranking data, per-fold ordered target statistics
//! ([`CrossValidation::target_stats`], optionally fitted on a separate
//! per-row target), continuing a model in every fold
//! ([`CrossValidation::init_model`]), and retraining on every row for the
//! round count cross-validation chose ([`CrossValidation::refit`],
//! [`CvRefit`]).
//!
//! Rounds are boosting rounds as [`Trainer::on_round`] counts them; for
//! `booster = ebm` they are EBM rounds, counted on through both stages.

mod fold;

pub use fold::Fold;

use crate::config::{BoosterKind, TrainingParams};
use crate::data::DMatrix;
use crate::data::target_stats::{FittedTargetEncoder, OrderedTargetEncoder};
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::training::Trainer;
use crate::training::eval::{EarlyStopping, configured_metrics};
use std::num::NonZeroUsize;
use std::ops::ControlFlow;

/// Per-metric cross-validation history, aggregated across folds.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CvResult {
    /// Metric name.
    pub metric: String,
    /// The held-out metric per boosting round, aggregated across folds.
    pub rounds: Vec<CvRound>,
}

/// One boosting round of a [`CvResult`]: the held-out metric across folds.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct CvRound {
    /// The mean over folds.
    pub mean: f64,
    /// The standard deviation (population, like XGBoost's) over folds.
    pub std: f64,
}

/// What [`CrossValidation::refit`] returns: the cross-validation results
/// and the model retrained on every row for the chosen round count.
#[derive(Debug)]
#[non_exhaustive]
pub struct CvRefit {
    /// The cross-validation results, as [`CrossValidation::run`] returns
    /// them.
    pub results: Vec<CvResult>,
    /// The chosen round count: the length of every result's `rounds` (the
    /// best round's index plus one under early stopping, else every round
    /// the folds ran).
    pub num_boost_round: usize,
    /// The model trained on every row for `num_boost_round` rounds
    /// (continuing [`CrossValidation::init_model`] when set).
    pub model: BoostedModel,
    /// With [`CrossValidation::target_stats`], the encoder fitted on every
    /// row, whose encoding `model` was trained on: encode new data with it
    /// before predicting. `None` otherwise.
    pub target_encoder: Option<FittedTargetEncoder>,
}

/// Cross-validation over caller-supplied [`Fold`]s (XGBoost's `cv(...,
/// folds=...)`), optionally with early stopping.
///
/// Each fold trains `params` for `num_boost_round` rounds on its training
/// rows and evaluates the metrics (`params.eval_metric`, or the objective's
/// default) on its test rows after every round; [`run`](Self::run)
/// averages them across folds, and [`refit`](Self::refit) also retrains on
/// every row for the chosen round count.
///
/// On ranking data (query groups attached), each fold's training and test
/// rows must be whole query groups ([`DMatrix::select_rows`]); the folds
/// keep them as their own groups.
///
/// ```
/// use hessboost::training::{CrossValidation, Fold};
/// use hessboost::prelude::*;
/// use std::num::NonZeroUsize;
///
/// # fn main() -> Result<()> {
/// // 60 time-ordered rows; labels look 3 rows ahead.
/// let x: Vec<f32> = (0..60).map(|i| i as f32).collect();
/// let y: Vec<f32> = (0..60).map(|i| ((i + 3) % 7) as f32).collect();
/// let data = DMatrix::from_dense(&x, 60, 1)?.with_labels(&y)?;
/// let params = TrainingParams::builder().max_depth(2).build()?;
///
/// let folds = Fold::forward_chaining(data.n_rows(), 3, 3)?;
/// let results = CrossValidation::new(&params, &data, 20, folds)
///     .early_stopping_rounds(NonZeroUsize::new(3).unwrap())
///     .run()?;
/// // With early stopping, the last round reported is the best one.
/// let rmse = &results[0];
/// assert_eq!(rmse.metric, "rmse");
/// assert!(rmse.rounds.len() <= 20);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct CrossValidation<'a> {
    params: &'a TrainingParams,
    data: &'a DMatrix,
    num_boost_round: usize,
    folds: Vec<Fold>,
    early_stopping_rounds: Option<NonZeroUsize>,
    target_stats: Option<(OrderedTargetEncoder, Vec<usize>)>,
    target_stats_label: Option<&'a [f32]>,
    init_model: Option<&'a BoostedModel>,
}

impl<'a> CrossValidation<'a> {
    /// Cross-validate `params` on `data` for `num_boost_round` rounds over
    /// `folds`.
    pub fn new(
        params: &'a TrainingParams,
        data: &'a DMatrix,
        num_boost_round: usize,
        folds: Vec<Fold>,
    ) -> Self {
        CrossValidation {
            params,
            data,
            num_boost_round,
            folds,
            early_stopping_rounds: None,
            target_stats: None,
            target_stats_label: None,
            init_model: None,
        }
    }

    /// Early stopping on the fold-averaged metric, as in `xgboost.cv`: the
    /// watched metric is the last one, and the best round is the one with
    /// the best mean after which the mean fails to improve for `rounds`
    /// consecutive rounds (or the best within `num_boost_round`). The
    /// results end at the best round, so `rounds.len() - 1` is its
    /// index. Unlike `xgboost.cv`, which truncates only when patience runs
    /// out, they are truncated whenever early stopping is on.
    ///
    /// The folds still train every round: the stopping point depends on
    /// all folds' metrics.
    #[must_use]
    pub fn early_stopping_rounds(mut self, rounds: NonZeroUsize) -> Self {
        self.early_stopping_rounds = Some(rounds);
        self
    }

    /// Encode categorical `columns` with ordered target statistics inside
    /// each fold: `encoder` is fitted on the fold's training rows only
    /// ([`OrderedTargetEncoder::fit_transform`]) and the fitted statistics
    /// encode its test rows ([`FittedTargetEncoder::transform`]), so no
    /// held-out label enters any encoding the fold trains or is scored on.
    /// Encoding the whole matrix before cross-validating would leak every
    /// test row's label into its own encoding.
    ///
    /// The statistics are taken over the data's labels (one per row), or
    /// over [`target_stats_label`](Self::target_stats_label) when set.
    #[must_use]
    pub fn target_stats(mut self, encoder: OrderedTargetEncoder, columns: Vec<usize>) -> Self {
        self.target_stats = Some((encoder, columns));
        self
    }

    /// The per-row target the [`target_stats`](Self::target_stats) encoder
    /// is fitted on instead of the data's labels, which stay the training
    /// target: one column of a multi-target matrix, or a class's 0/1
    /// indicator. Each fold passes the values of its training rows to
    /// [`OrderedTargetEncoder::fit_transform_with_labels`] (which checks
    /// them), and [`refit`](Self::refit) all of them.
    ///
    /// [`run`](Self::run) and `refit` refuse it without `target_stats`
    /// ([`HessboostError::InvalidParameter`] `target_stats_label`) and
    /// unless it holds one value per row of `data`
    /// ([`HessboostError::DimensionMismatch`]).
    #[must_use]
    pub fn target_stats_label(mut self, labels: &'a [f32]) -> Self {
        self.target_stats_label = Some(labels);
        self
    }

    /// Continue `model` in every fold (and in [`refit`](Self::refit)) as
    /// [`Trainer::init_model`] does, with its checks of `params` and the
    /// data against the model. The results count this run's rounds from 0.
    ///
    /// [`run`](Self::run) and `refit` refuse it together with
    /// [`target_stats`](Self::target_stats)
    /// ([`HessboostError::InvalidParameter`] `target_stats`): the model was
    /// trained on its own encoding of those columns, which statistics
    /// refitted per fold would change.
    #[must_use]
    pub fn init_model(mut self, model: &'a BoostedModel) -> Self {
        self.init_model = Some(model);
        self
    }

    /// Train and evaluate every fold, returning one [`CvResult`] per metric
    /// in the configured metric order.
    ///
    /// Fails when there are no folds, a fold's training or test rows are
    /// empty, out of bounds, or (on ranking data) not whole query groups,
    /// when [`target_stats_label`](Self::target_stats_label) or
    /// [`init_model`](Self::init_model) is refused, or when encoding or
    /// training a fold fails. An EBM that stops its bags early
    /// ([`Ebm::early_stopping`](crate::config::Ebm::early_stopping)) is refused
    /// ([`HessboostError::InvalidParameter`] `ebm_early_stopping_rounds`):
    /// each fold would end its stages at a different round, so the folds'
    /// rounds would not line up.
    /// [`early_stopping_rounds`](Self::early_stopping_rounds) stops on the
    /// fold means instead.
    pub fn run(self) -> Result<Vec<CvResult>> {
        self.results()
    }

    /// Cross-validate as [`run`](Self::run) does, then retrain on every row
    /// of `data` for the chosen round count
    /// ([`CvRefit::num_boost_round`]: through the best round under
    /// [`early_stopping_rounds`](Self::early_stopping_rounds), else every
    /// round the folds ran). The retraining continues
    /// [`init_model`](Self::init_model) when set and has no eval sets. With
    /// [`target_stats`](Self::target_stats), it trains on `data` encoded by
    /// the encoder fitted on every row (over
    /// [`target_stats_label`](Self::target_stats_label) when set), which is
    /// returned as [`CvRefit::target_encoder`].
    ///
    /// The retraining is configured for all `num_boost_round` rounds and
    /// stopped through [`Trainer::on_round`] after the chosen count, so its
    /// model equals one trained on `data` for that many rounds. For
    /// `booster = ebm` the count is of EBM rounds across both stages
    /// (`num_boost_round` still caps each stage), so the model is the one
    /// after that many rounds of the full run.
    ///
    /// Fails as `run` does, or when encoding or retraining on `data` fails.
    ///
    /// ```
    /// use hessboost::training::{CrossValidation, Fold};
    /// use hessboost::prelude::*;
    /// use std::num::NonZeroUsize;
    ///
    /// # fn main() -> Result<()> {
    /// let x: Vec<f32> = (0..90).map(|i| (i % 30) as f32).collect();
    /// let y: Vec<f32> = x.iter().map(|v| (v / 10.0).floor()).collect();
    /// let data = DMatrix::from_dense(&x, 90, 1)?.with_labels(&y)?;
    /// let params = TrainingParams::builder().max_depth(2).build()?;
    ///
    /// let refit = CrossValidation::new(&params, &data, 50, Fold::k_fold(90, 3, 0)?)
    ///     .early_stopping_rounds(NonZeroUsize::new(5).unwrap())
    ///     .refit()?;
    /// // Trained on all 90 rows for as many rounds as the results report.
    /// assert_eq!(refit.num_boost_round, refit.results[0].rounds.len());
    /// assert_eq!(refit.model.num_boost_rounds(), refit.num_boost_round);
    /// let predictions = refit.model.predict(&data, Iterations::Best)?;
    /// assert_eq!(predictions.as_slice().len(), 90);
    /// # Ok(())
    /// # }
    /// ```
    pub fn refit(self) -> Result<CvRefit> {
        let results = self.results()?;
        let rounds = results.first().map_or(0, |result| result.rounds.len());
        let (encoded, target_encoder) = match &self.target_stats {
            Some((encoder, columns)) => {
                let (encoded, fitted) =
                    fit_encoder(encoder, columns, self.data, self.target_stats_label)?;
                (Some(encoded), Some(fitted))
            }
            None => (None, None),
        };
        let dtrain = encoded.as_ref().unwrap_or(self.data);
        // The hook cannot stop before the first round.
        let num_boost_round = if rounds == 0 { 0 } else { self.num_boost_round };
        let mut completed = 0;
        let model = self
            .trainer(dtrain, num_boost_round)
            .on_round(move |_| {
                completed += 1;
                if completed == rounds {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .train()?
            .model;
        Ok(CvRefit {
            results,
            num_boost_round: rounds,
            model,
            target_encoder,
        })
    }

    /// The validated cross-validation results [`run`](Self::run) and
    /// [`refit`](Self::refit) share: every fold's scores aggregated per
    /// round, truncated at the best round under early stopping.
    fn results(&self) -> Result<Vec<CvResult>> {
        if self.folds.is_empty() {
            return Err(HessboostError::invalid_param("folds", "no folds"));
        }
        if let BoosterKind::Ebm(ebm) = &self.params.booster
            && ebm.early_stopping().is_some()
        {
            return Err(HessboostError::invalid_param(
                "ebm_early_stopping_rounds",
                "cross-validation averages the folds round by round, but stopping each \
                 fold's bags on its own held-out rows ends the folds' stages at different \
                 rounds; stop on the fold means with early_stopping_rounds instead",
            ));
        }
        if let Some(labels) = self.target_stats_label {
            if self.target_stats.is_none() {
                return Err(HessboostError::invalid_param(
                    "target_stats_label",
                    "needs target_stats: it is the target the encoder is fitted on",
                ));
            }
            if labels.len() != self.data.n_rows() {
                return Err(HessboostError::dimension_mismatch(
                    "target_stats_label length (one per row)",
                    self.data.n_rows(),
                    labels.len(),
                ));
            }
        }
        if self.init_model.is_some() && self.target_stats.is_some() {
            return Err(HessboostError::invalid_param(
                "target_stats",
                "cannot be refitted per fold when continuing init_model, which was trained on \
                 its own encoding of those columns",
            ));
        }
        validate_folds(&self.folds, self.data)?;
        let objective = self.params.loss(self.data.n_targets())?;
        let metrics = configured_metrics(self.params, objective.as_ref())?;
        let maximize = metrics.last().is_some_and(|m| m.maximize());

        let values = self.scores(metrics.len())?;
        let mut out: Vec<CvResult> = metrics
            .iter()
            .zip(values)
            .map(|(metric, per_round)| aggregate(metric.name(), &per_round))
            .collect();

        if let Some(patience) = self.early_stopping_rounds
            && let Some(watched) = out.last()
            && !watched.rounds.is_empty()
        {
            let means = watched.rounds.iter().map(|round| round.mean);
            let end = best_round(means, patience, maximize) + 1;
            for result in &mut out {
                result.rounds.truncate(end);
            }
        }
        Ok(out)
    }

    /// Train on every fold in order and collect its test scores as
    /// `values[metric][round][fold]`, grown as rounds arrive (not sized by
    /// `num_boost_round`, which is caller input).
    fn scores(&self, n_metrics: usize) -> Result<Vec<Vec<Vec<f64>>>> {
        let mut values: Vec<Vec<Vec<f64>>> = vec![Vec::new(); n_metrics];
        for fold in &self.folds {
            let (dtrain, dtest) = self.fold_data(fold)?;
            let res = self
                .trainer(&dtrain, self.num_boost_round)
                .eval(&dtest, "test")
                .train()?;
            for (round, eval) in res.history.rounds().enumerate() {
                for (per_round, &value) in values.iter_mut().zip(eval.values()) {
                    if per_round.len() == round {
                        per_round.push(Vec::with_capacity(self.folds.len()));
                    }
                    per_round[round].push(value);
                }
            }
        }
        Ok(values)
    }

    /// A [`Trainer`] of `params` on `dtrain`, continuing the initial model
    /// when set.
    fn trainer<'b>(&self, dtrain: &'b DMatrix, num_boost_round: usize) -> Trainer<'b>
    where
        'a: 'b,
    {
        let trainer = Trainer::new(self.params, dtrain, num_boost_round);
        match self.init_model {
            Some(model) => trainer.init_model(model),
            None => trainer,
        }
    }

    /// A fold's training and test matrices, target-encoded with statistics
    /// of its training rows when configured.
    fn fold_data(&self, fold: &Fold) -> Result<(DMatrix, DMatrix)> {
        let dtrain = self.data.select_rows(&fold.train)?;
        let dtest = self.data.select_rows(&fold.test)?;
        let Some((encoder, columns)) = &self.target_stats else {
            return Ok((dtrain, dtest));
        };
        // `results` checked the length and `validate_folds` the rows.
        let labels: Option<Vec<f32>> = self
            .target_stats_label
            .map(|labels| fold.train.iter().map(|&row| labels[row]).collect());
        let (dtrain, fitted) = fit_encoder(encoder, columns, &dtrain, labels.as_deref())?;
        let dtest = fitted.transform(&dtest)?;
        Ok((dtrain, dtest))
    }
}

/// `encoder` fitted on `data` with its encoding of `columns`, over `labels`
/// when given, else over `data`'s own labels.
fn fit_encoder(
    encoder: &OrderedTargetEncoder,
    columns: &[usize],
    data: &DMatrix,
    labels: Option<&[f32]>,
) -> Result<(DMatrix, FittedTargetEncoder)> {
    match labels {
        Some(labels) => encoder.fit_transform_with_labels(data, columns, labels),
        None => encoder.fit_transform(data, columns),
    }
}

/// Refuse a fold without training or test rows, naming a row past the
/// data's, or (on ranking data) not selecting whole query groups.
fn validate_folds(folds: &[Fold], data: &DMatrix) -> Result<()> {
    let n = data.n_rows();
    for (f, fold) in folds.iter().enumerate() {
        for (name, rows) in [("training", &fold.train), ("test", &fold.test)] {
            if rows.is_empty() {
                return Err(HessboostError::invalid_param(
                    "folds",
                    format!("fold {f} has no {name} rows"),
                ));
            }
            if let Some(&row) = rows.iter().find(|&&row| row >= n) {
                return Err(HessboostError::invalid_param(
                    "folds",
                    format!("fold {f}: {name} row {row} is out of bounds for {n} rows"),
                ));
            }
            data.selected_group_sizes(rows).map_err(|reason| {
                HessboostError::invalid_param(
                    "folds",
                    format!("fold {f}: its {name} rows {reason}"),
                )
            })?;
        }
    }
    Ok(())
}

/// A metric's per-round fold mean and (population) standard deviation.
fn aggregate(metric: &str, per_round: &[Vec<f64>]) -> CvResult {
    let rounds = per_round
        .iter()
        .map(|vals| {
            let len = vals.len() as f64;
            let mean = vals.iter().sum::<f64>() / len;
            let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / len;
            CvRound {
                mean,
                std: var.sqrt(),
            }
        })
        .collect();
    CvResult {
        metric: metric.to_string(),
        rounds,
    }
}

/// The round early stopping selects on `scores`: the same rule as
/// [`Trainer::early_stopping_rounds`] ([`EarlyStopping`]; round 0 when no
/// score ever improves, e.g. NaN).
fn best_round(scores: impl Iterator<Item = f64>, patience: NonZeroUsize, maximize: bool) -> usize {
    let mut stopping = EarlyStopping::new(patience, maximize, 0);
    for (round, score) in scores.enumerate() {
        if stopping.observe(round, score) {
            break;
        }
    }
    stopping.best_round()
}

/// Shuffled `nfold` cross-validation ([`Fold::k_fold`] folds), returning one
/// [`CvResult`] per evaluation metric. Every fold trains for the full
/// `num_boost_round` rounds (no early stopping). The metric list comes from
/// `params.eval_metric` or the objective's default.
///
/// For time-ordered or grouped rows, supply the folds (and optionally early
/// stopping and per-fold target statistics) through [`CrossValidation`].
pub fn cv(
    params: &TrainingParams,
    data: &DMatrix,
    num_boost_round: usize,
    nfold: usize,
    seed: u64,
) -> Result<Vec<CvResult>> {
    let folds = Fold::k_fold(data.n_rows(), nfold, seed)?;
    CrossValidation::new(params, data, num_boost_round, folds).run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objective::{Objective, RegLoss};
    use crate::test_support::labeled_dense;

    #[test]
    fn cv_reports_decreasing_rmse() {
        // Simple learnable data.
        let n = 200;
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..n {
            let xi = i as f32 / n as f32;
            x.push(xi);
            y.push(if xi > 0.5 { 1.0 } else { 0.0 });
        }
        let d = labeled_dense(&x, n, 1, &y);
        let params = TrainingParams::builder()
            .objective(Objective::SquaredError(RegLoss::default()))
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();

        let results = cv(&params, &d, 30, 5, 42).unwrap();
        assert_eq!(results.len(), 1);
        let rmse = &results[0];
        assert_eq!(rmse.metric, "rmse");
        assert_eq!(rmse.rounds.len(), 30);
        // Held-out error should drop from first to last round.
        assert!(rmse.rounds[29].mean < rmse.rounds[0].mean);
        // Std is non-negative and finite.
        assert!(
            rmse.rounds
                .iter()
                .all(|r| r.std.is_finite() && r.std >= 0.0)
        );
    }

    fn step_data(n: usize) -> DMatrix {
        let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let y: Vec<f32> = x.iter().map(|&v| if v > 0.5 { 1.0 } else { 0.0 }).collect();
        labeled_dense(&x, n, 1, &y)
    }

    #[test]
    fn forward_chaining_trains_only_on_rows_before_the_gap() {
        // 23 rows, 3 splits: blocks of 5, the first test block starting after
        // the 8 leading rows (the remainder goes to the first training set).
        let folds = Fold::forward_chaining(23, 3, 2).unwrap();
        let expect = [(0..6, 8..13), (0..11, 13..18), (0..16, 18..23)];
        assert_eq!(folds.len(), 3);
        for (fold, (train, test)) in folds.iter().zip(expect) {
            assert_eq!(fold.train, train.collect::<Vec<_>>());
            assert_eq!(fold.test, test.collect::<Vec<_>>());
        }
        // The largest gap that still leaves one training row, and the first
        // that does not.
        assert_eq!(Fold::forward_chaining(23, 3, 7).unwrap()[0].train, [0]);
        assert!(Fold::forward_chaining(23, 3, 8).is_err());
        // One row per block is the most splits the rows allow; more are
        // refused, up to the largest count (whose `+ 1` would overflow).
        let tight = Fold::forward_chaining(4, 3, 0).unwrap();
        assert_eq!(tight[0], Fold::new(vec![0], vec![1]));
        assert_eq!(tight[2], Fold::new(vec![0, 1, 2], vec![3]));
        assert!(Fold::forward_chaining(3, 3, 0).is_err());
        assert!(Fold::forward_chaining(10, usize::MAX, 0).is_err());
        assert!(Fold::forward_chaining(10, 2, usize::MAX).is_err());
        assert!(Fold::forward_chaining(10, 0, 0).is_err());
    }

    #[test]
    fn caller_folds_are_checked() {
        let d = step_data(20);
        let params = TrainingParams::default();
        let run = |folds: Vec<Fold>| CrossValidation::new(&params, &d, 2, folds).run();
        assert!(run(Vec::new()).is_err());
        assert!(run(vec![Fold::new(vec![], vec![1])]).is_err());
        assert!(run(vec![Fold::new(vec![0], vec![])]).is_err());
        assert!(run(vec![Fold::new(vec![0, 20], vec![1])]).is_err());
        assert!(run(vec![Fold::new(vec![0, 1], vec![19])]).is_ok());
    }

    #[test]
    fn metrics_keep_their_configured_order() {
        let d = step_data(60);
        let params = TrainingParams::builder()
            .eval_metric(crate::metric::EvalMetric::Rmse)
            .eval_metric(crate::metric::EvalMetric::Mae)
            .build()
            .unwrap();
        let results = cv(&params, &d, 3, 3, 1).unwrap();
        let names: Vec<&str> = results.iter().map(|r| r.metric.as_str()).collect();
        assert_eq!(names, ["rmse", "mae"]);
    }

    #[test]
    fn early_stopping_ends_at_the_best_mean_round() {
        let d = step_data(120);
        let params = TrainingParams::builder()
            .eval_metric(crate::metric::EvalMetric::Mae)
            .eval_metric(crate::metric::EvalMetric::Rmse)
            .max_depth(6)
            .eta(0.8)
            .build()
            .unwrap();
        let folds = Fold::k_fold(d.n_rows(), 4, 3).unwrap();
        let full = CrossValidation::new(&params, &d, 40, folds.clone())
            .run()
            .unwrap();
        let rmse: Vec<f64> = full[1].rounds.iter().map(|r| r.mean).collect();
        let best = (0..rmse.len())
            .min_by(|&a, &b| rmse[a].total_cmp(&rmse[b]))
            .unwrap();
        let stopped = CrossValidation::new(&params, &d, 40, folds)
            .early_stopping_rounds(NonZeroUsize::new(5).unwrap())
            .run()
            .unwrap();
        // The results end at the last round that improved on every earlier
        // one before 5 rounds without improvement.
        let end = stopped[1].rounds.len();
        assert!(end + 5 <= 40, "ended after {end} rounds");
        let b = end - 1;
        assert!(rmse[..b].iter().all(|&v| v > rmse[b]));
        assert!(rmse[b + 1..=b + 5].iter().all(|&v| v >= rmse[b]));
        for (s, f) in stopped.iter().zip(&full) {
            assert_eq!(s.rounds, f.rounds[..end]);
        }
        // Patience longer than the run still ends at the best round.
        let folds = Fold::k_fold(d.n_rows(), 4, 3).unwrap();
        let long = CrossValidation::new(&params, &d, 40, folds)
            .early_stopping_rounds(NonZeroUsize::new(100).unwrap())
            .run()
            .unwrap();
        assert_eq!(long[1].rounds, full[1].rounds[..=best]);
    }

    /// `n_groups` query groups of 4 rows, relevance learnable from the one
    /// feature.
    fn ranking_data(n_groups: usize) -> DMatrix {
        let x: Vec<f32> = (0..n_groups * 4).map(|i| ((i * 5) % 7) as f32).collect();
        let y: Vec<f32> = x.iter().map(|&v| (v / 2.0).floor()).collect();
        labeled_dense(&x, x.len(), 1, &y)
            .with_group_sizes(&vec![4; n_groups])
            .unwrap()
    }

    #[test]
    fn ranking_folds_keep_whole_query_groups() {
        let d = ranking_data(6);
        let params = TrainingParams::builder()
            .objective(Objective::RankNdcg(crate::objective::LambdaRank::default()))
            .max_depth(2)
            .build()
            .unwrap();
        // Groups {0, 1}, {2, 3}, {4, 5} held out in turn.
        let folds: Vec<Fold> = (0..3)
            .map(|f| {
                let (test, train) = (0..24).partition(|&row| row / 8 == f);
                Fold::new(train, test)
            })
            .collect();
        let results = CrossValidation::new(&params, &d, 3, folds).run().unwrap();
        assert_eq!(results[0].metric, "ndcg@32");
        assert!(results[0].rounds.iter().all(|r| r.mean.is_finite()));
        // Shuffled row folds split the groups: refused before training.
        let shuffled = Fold::k_fold(24, 3, 0).unwrap();
        assert!(matches!(
            CrossValidation::new(&params, &d, 3, shuffled).run(),
            Err(HessboostError::InvalidParameter { name: "folds", .. })
        ));
    }

    #[test]
    fn target_stats_are_fitted_on_each_folds_training_rows() {
        use crate::data::FeatureType;
        // Column 0: one of 5 categories whose label mean differs; column 1
        // numeric noise.
        let n = 90;
        let x: Vec<f32> = (0..n)
            .flat_map(|i| [(i % 5) as f32, ((i * 13) % 11) as f32])
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|i| (i % 5) as f32 + ((i * 7) % 3) as f32)
            .collect();
        let d = labeled_dense(&x, n, 2, &y)
            .with_feature_types(&[FeatureType::Categorical, FeatureType::Numerical])
            .unwrap();
        let params = TrainingParams::builder().max_depth(2).build().unwrap();
        let encoder = OrderedTargetEncoder::builder().seed(4).build().unwrap();
        let folds = Fold::k_fold(n, 3, 7).unwrap();
        let results = CrossValidation::new(&params, &d, 5, folds.clone())
            .target_stats(encoder.clone(), vec![0])
            .run()
            .unwrap();

        // The same folds encoded by hand from their training rows alone.
        let mut last = Vec::new();
        for fold in &folds {
            let (dtrain, fitted) = encoder
                .fit_transform(&d.select_rows(&fold.train).unwrap(), &[0])
                .unwrap();
            let dtest = fitted
                .transform(&d.select_rows(&fold.test).unwrap())
                .unwrap();
            let res = Trainer::new(&params, &dtrain, 5)
                .eval(&dtest, "test")
                .train()
                .unwrap();
            last.push(res.history.rounds().last().unwrap().values()[0]);
        }
        let mean = last.iter().sum::<f64>() / last.len() as f64;
        assert_eq!(results[0].rounds[4].mean, mean);
        // A column that is not categorical is refused by the encoder.
        let folds = Fold::k_fold(n, 3, 7).unwrap();
        assert!(matches!(
            CrossValidation::new(&params, &d, 5, folds)
                .target_stats(encoder, vec![1])
                .run(),
            Err(HessboostError::InvalidParameter {
                name: "columns",
                ..
            })
        ));
    }
}
