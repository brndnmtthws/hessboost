//! Cross-validation, mirroring `xgboost.cv`: shuffled k-fold ([`cv`]) or
//! caller-supplied folds ([`CrossValidation`], [`Fold`]), including
//! forward-chaining folds for time-ordered rows, forward folds purged by
//! each row's label window ([`Fold::purged_forward`], or at given test-block
//! starts with [`Fold::purged_forward_at`]), whole-query folds of
//! ranking data, per-fold ordered target statistics
//! ([`CrossValidation::target_stats`], optionally fitted on a separate
//! per-row target), continuing a model in every fold
//! ([`CrossValidation::init_model`]), every held-out row's out-of-fold
//! margin and each fold's round count ([`CvReport`]), leakage-safe
//! per-fold early stopping on a purged tail of each fold's own training
//! rows ([`CvStopping::per_fold`], [`PurgedTail`]), and retraining on
//! every row for the round count cross-validation chose
//! ([`CrossValidation::refit`], [`CvRefit`]).
//!
//! Rounds are boosting rounds as [`Trainer::on_round`] counts them; for
//! `booster = ebm` they are EBM rounds, counted on through both stages.

mod fold;

pub use fold::{Fold, PurgedTail};

use crate::config::{BoosterKind, TrainingParams};
use crate::data::DMatrix;
use crate::data::target_stats::{FittedTargetEncoder, OrderedTargetEncoder};
use crate::error::{HessboostError, Result};
use crate::metric::Metric;
use crate::model::{BoostedModel, Predictions};
use crate::training::eval::{EarlyStopping, configured_metrics};
use crate::training::{EvalHistory, TrainResult, Trainer};
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

/// How [`CrossValidation`] chooses the round count of each fold's model
/// ([`CrossValidation::early_stopping`]); without one, every fold trains
/// every round.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum CvStopping<'a> {
    /// Early stopping on the fold-averaged metric, as in `xgboost.cv`: the
    /// watched metric is the last one, and the best round is the one with
    /// the best mean after which the mean fails to improve for `patience`
    /// consecutive rounds (or the best within `num_boost_round`). The
    /// history ([`CvReport::results`]) ends at the best round, so
    /// `rounds.len() - 1` is its index, and every fold's model is its
    /// iterations through that round. Unlike `xgboost.cv`, which truncates
    /// only when patience runs out, the history is truncated whenever early
    /// stopping is on.
    ///
    /// The folds still train every round: the stopping point depends on all
    /// folds' metrics, so it was chosen with every fold's held-out labels.
    #[non_exhaustive]
    FoldMean {
        /// Rounds without improvement of the fold mean before stopping.
        patience: NonZeroUsize,
    },
    /// Leakage-safe early stopping per fold: each fold splits its own
    /// training rows by `tail` ([`PurgedTail::split`]), trains on the
    /// split's training rows while scoring the tail (the watched metric is
    /// the last one, with [`Trainer::early_stopping_rounds`]'s rule), and
    /// then trains its model on all its training rows for the best round's
    /// count. No fold's round count depends on its held-out rows or on any
    /// other fold, so with forward folds ([`Fold::purged_forward`]) no fold
    /// uses rounds chosen with later labels. The fold-mean history is empty
    /// (the folds' models run different round counts); each fold's scores
    /// are in [`FoldReport::scores`].
    #[non_exhaustive]
    PerFold {
        /// Rounds without improvement on a fold's tail before stopping.
        patience: NonZeroUsize,
        /// The times the tails are cut by, one per row of the data.
        tail: PurgedTail<'a>,
    },
}

impl<'a> CvStopping<'a> {
    /// [`CvStopping::FoldMean`] with `patience`.
    pub fn fold_mean(patience: NonZeroUsize) -> Self {
        CvStopping::FoldMean { patience }
    }

    /// [`CvStopping::PerFold`] with `patience`, on tails cut by `tail`.
    pub fn per_fold(patience: NonZeroUsize, tail: PurgedTail<'a>) -> Self {
        CvStopping::PerFold { patience, tail }
    }
}

/// What [`CrossValidation::run`] returns: the fold-mean history, and each
/// fold's model's round count, test scores, and test-row margins.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CvReport {
    /// The metrics' names (as [`CvResult::metric`]), in the configured
    /// order, which [`FoldReport::scores`] follows; the last one is
    /// watched by early stopping.
    pub metrics: Vec<String>,
    /// One [`CvResult`] per metric in the configured metric order: the
    /// held-out metric per round across folds. Empty under
    /// [`CvStopping::PerFold`], whose folds run different round counts.
    pub results: Vec<CvResult>,
    /// One report per fold, in fold order (none under
    /// [`CrossValidation::history_only`]).
    pub folds: Vec<FoldReport>,
    /// The cross-validated data's row count.
    n_rows: usize,
}

/// One fold of a [`CvReport`]: what its model was and predicted.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FoldReport {
    /// The fold's test rows, as the fold listed them.
    pub test: Vec<usize>,
    /// The rounds the fold's model used: every round the fold trained
    /// without early stopping, the best fold-mean round's count under
    /// [`CvStopping::FoldMean`], the best tail round's count under
    /// [`CvStopping::PerFold`].
    pub num_boost_round: usize,
    /// The model's test metrics after its last round, in the configured
    /// metric order (empty for a model of no rounds).
    pub scores: Vec<f64>,
    /// The model's margins on the test rows, one row per entry of
    /// [`test`](Self::test) (target-encoded by the fold's own statistics
    /// under [`CrossValidation::target_stats`]): bit for bit those of the
    /// fold's configured run stopped through [`Trainer::on_round`] after
    /// `num_boost_round` rounds (the same as a run configured for that
    /// many, except under `booster = ebm`, whose configured count caps each
    /// stage). Under [`CvStopping::FoldMean`] they are a gbtree fold's first
    /// iterations; other boosters, whose later rounds rescale earlier trees
    /// or add several trees per round, retrain the fold for that many
    /// rounds.
    pub margins: Predictions,
    /// Under [`CvStopping::PerFold`], the fold's tail split and its best
    /// score; `None` otherwise.
    pub stopping: Option<TailStopping>,
}

/// How [`CvStopping::PerFold`] sized one model: its tail split and the
/// watched metric there.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct TailStopping {
    /// The split time ([`PurgedTail::split`]): a decision time of the rows
    /// split.
    pub split_at: i64,
    /// The rows the early-stopping fit trained on.
    pub fit_rows: usize,
    /// The tail rows it was scored on.
    pub tail_rows: usize,
    /// The watched metric on the tail at the best round (`NaN` for no
    /// rounds).
    pub best_score: f64,
}

/// Every row's out-of-fold margins ([`CvReport::out_of_fold`]).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OutOfFold {
    /// One row of margins per data row, from the fold that held the row
    /// out; `NaN` for a row no fold held out.
    pub margins: Predictions,
    /// Per data row, the fold that held it out (`None`: no fold did).
    pub fold: Vec<Option<usize>>,
}

impl CvReport {
    /// Every row's margins from the fold that held it out, combining the
    /// folds' [`margins`](FoldReport::margins) by their test rows. A row no
    /// fold held out (forward folds never test the first rows) gets `NaN`
    /// margins and no fold; a row one fold lists several times gets that
    /// fold's margins.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] (`folds`) when two folds hold
    /// out the same row, which then has no single out-of-fold margin (each
    /// fold's own margins stay in [`folds`](Self::folds)), or (in a report
    /// edited through its public fields) a test row is out of bounds;
    /// [`HessboostError::DimensionMismatch`] when a fold's margins do not
    /// hold one row per test row, or not as many outputs as fold 0's.
    pub fn out_of_fold(&self) -> Result<OutOfFold> {
        let width = self.folds.first().map_or(1, |fold| fold.margins.width());
        // The fields are public: check what the report's own folds always
        // satisfy before indexing by them.
        for (f, fold) in self.folds.iter().enumerate() {
            if fold.margins.n_rows() != fold.test.len() {
                return Err(HessboostError::dimension_mismatch(
                    "fold margin rows (one per test row)",
                    fold.test.len(),
                    fold.margins.n_rows(),
                ));
            }
            if fold.margins.width() != width {
                return Err(HessboostError::dimension_mismatch(
                    "fold margin width (fold 0's)",
                    width,
                    fold.margins.width(),
                ));
            }
            if let Some(&row) = fold.test.iter().find(|&&row| row >= self.n_rows) {
                return Err(HessboostError::invalid_param(
                    "folds",
                    format!(
                        "fold {f}: test row {row} is out of bounds for {} rows",
                        self.n_rows
                    ),
                ));
            }
        }
        let mut margins = vec![f32::NAN; self.n_rows * width];
        let mut fold_of = vec![None; self.n_rows];
        for (f, fold) in self.folds.iter().enumerate() {
            for (&row, values) in fold.test.iter().zip(fold.margins.rows()) {
                if let Some(other) = fold_of[row]
                    && other != f
                {
                    return Err(HessboostError::invalid_param(
                        "folds",
                        format!(
                            "row {row} is held out by folds {other} and {f}, so it has no single \
                             out-of-fold margin"
                        ),
                    ));
                }
                fold_of[row] = Some(f);
                margins[row * width..(row + 1) * width].copy_from_slice(values);
            }
        }
        Ok(OutOfFold {
            margins: Predictions::new(margins, self.n_rows, width),
            fold: fold_of,
        })
    }
}

/// What [`CrossValidation::refit`] returns: the cross-validation report
/// and the model retrained on every row for the chosen round count.
#[derive(Debug)]
#[non_exhaustive]
pub struct CvRefit {
    /// The cross-validation report, as [`CrossValidation::run`] returns
    /// it.
    pub report: CvReport,
    /// The chosen round count: the best round's index plus one under
    /// [`CvStopping::FoldMean`] (the length of every result's `rounds`),
    /// the best round on the purged tail of every row under
    /// [`CvStopping::PerFold`], else every round the folds ran.
    pub num_boost_round: usize,
    /// The model trained on every row for `num_boost_round` rounds
    /// (continuing [`CrossValidation::init_model`] when set).
    pub model: BoostedModel,
    /// With [`CrossValidation::target_stats`], the encoder fitted on every
    /// row, whose encoding `model` was trained on: encode new data with it
    /// before predicting. `None` otherwise.
    pub target_encoder: Option<FittedTargetEncoder>,
    /// Under [`CvStopping::PerFold`], the tail split of every row that
    /// sized `model`; `None` otherwise.
    pub stopping: Option<TailStopping>,
}

/// Cross-validation over caller-supplied [`Fold`]s (XGBoost's `cv(...,
/// folds=...)`), optionally with early stopping.
///
/// Each fold trains `params` for `num_boost_round` rounds on its training
/// rows and evaluates the metrics (`params.eval_metric`, or the objective's
/// default) on its test rows after every round; [`run`](Self::run)
/// averages them across folds and reports each fold's round count, scores,
/// and test-row margins ([`CvReport`], whose
/// [`out_of_fold`](CvReport::out_of_fold) gives every held-out row's
/// margins), and [`refit`](Self::refit) also retrains on every row for the
/// chosen round count. [`early_stopping`](Self::early_stopping) chooses the
/// round counts on the fold means or, leakage-free, per fold on a purged
/// tail of its training rows.
///
/// On ranking data (query groups attached), each fold's training and test
/// rows (and each purged tail's sides) must be whole query groups
/// ([`DMatrix::select_rows`]); the folds keep them as their own groups.
///
/// ```
/// use hessboost::training::{CrossValidation, CvStopping, Fold};
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
/// let report = CrossValidation::new(&params, &data, 20, folds)
///     .early_stopping(CvStopping::fold_mean(NonZeroUsize::new(3).unwrap()))
///     .run()?;
/// // With early stopping, the last round reported is the best one.
/// let rmse = &report.results[0];
/// assert_eq!(rmse.metric, "rmse");
/// assert!(rmse.rounds.len() <= 20);
/// // The first 15 rows only ever train: they have no out-of-fold margin.
/// let oof = report.out_of_fold()?;
/// assert!(oof.fold[..15].iter().all(Option::is_none));
/// assert!(oof.margins.as_slice()[15..].iter().all(|m| m.is_finite()));
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct CrossValidation<'a> {
    params: &'a TrainingParams,
    data: &'a DMatrix,
    num_boost_round: usize,
    folds: Vec<Fold>,
    stopping: Option<CvStopping<'a>>,
    target_stats: Option<(OrderedTargetEncoder, Vec<usize>)>,
    target_stats_label: Option<&'a [f32]>,
    init_model: Option<&'a BoostedModel>,
    history_only: bool,
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
            stopping: None,
            target_stats: None,
            target_stats_label: None,
            init_model: None,
            history_only: false,
        }
    }

    /// Choose each fold's round count (at most the rounds the folds train) by
    /// `stopping`: on the fold means ([`CvStopping::fold_mean`], as
    /// `xgboost.cv`'s `early_stopping_rounds`), or per fold on a purged tail
    /// of its own training rows ([`CvStopping::per_fold`]).
    ///
    /// [`run`](Self::run) and [`refit`](Self::refit) refuse per-fold
    /// stopping unless its tail times hold one value per row of `data`
    /// ([`HessboostError::DimensionMismatch`]), for a Boulevard fit
    /// (`booster = boulevard` or a Boulevard EBM, which do not early-stop:
    /// [`HessboostError::InvalidParameter`] `early_stopping`), and for a fold
    /// (or, in `refit`, the data) whose tail split leaves either side empty
    /// ([`HessboostError::InvalidParameter`] `purged tail`, as
    /// [`PurgedTail::split`]).
    #[must_use]
    pub fn early_stopping(mut self, stopping: CvStopping<'a>) -> Self {
        self.stopping = Some(stopping);
        self
    }

    /// Encode categorical `columns` with ordered target statistics inside
    /// each fold: `encoder` is fitted on the fold's training rows only
    /// ([`OrderedTargetEncoder::fit_transform`]) and the fitted statistics
    /// encode its test rows ([`FittedTargetEncoder::transform`]), so no
    /// held-out label enters any encoding the fold trains or is scored on.
    /// Per-fold early stopping fits its own encoder on the tail split's
    /// training rows likewise. Encoding the whole matrix before
    /// cross-validating would leak every test row's label into its own
    /// encoding.
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
    /// data against the model. The results count this run's rounds from 0;
    /// the folds' margins include the model's own iterations.
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

    /// Report only the fold-mean history ([`CvReport::results`]), as [`cv`]
    /// and Python's `hessboost.cv` do: [`CvReport::folds`] stays empty, so
    /// no fold's model outlives its run, and none is predicted on or (under
    /// [`CvStopping::FoldMean`], for boosters other than gbtree) retrained
    /// for its margins.
    ///
    /// [`run`](Self::run) and [`refit`](Self::refit) refuse it with
    /// [`CvStopping::per_fold`] ([`HessboostError::InvalidParameter`]
    /// `early_stopping`), whose folds report different round counts and
    /// no fold-mean history.
    #[must_use]
    pub fn history_only(mut self) -> Self {
        self.history_only = true;
        self
    }

    /// Train and evaluate every fold, returning the fold-mean history and
    /// every fold's report ([`CvReport`]).
    ///
    /// Fails when there are no folds, a fold's training or test rows are
    /// empty, out of bounds, or (on ranking data) not whole query groups,
    /// when [`early_stopping`](Self::early_stopping) (alone or with
    /// [`history_only`](Self::history_only)),
    /// [`target_stats_label`](Self::target_stats_label) or
    /// [`init_model`](Self::init_model) is refused, or when encoding,
    /// training, or predicting a fold fails. An EBM that stops its bags
    /// early
    /// ([`Ebm::early_stopping`](crate::config::Ebm::early_stopping)) is
    /// refused ([`HessboostError::InvalidParameter`]
    /// `ebm_early_stopping_rounds`): each fold would end its stages at a
    /// different round, so the folds' rounds would not line up.
    /// [`CvStopping::fold_mean`] stops on the fold means instead.
    pub fn run(self) -> Result<CvReport> {
        let metrics = self.checked_metrics()?;
        self.report(&metrics)
    }

    /// Cross-validate as [`run`](Self::run) does, then retrain on every row
    /// of `data` for the chosen round count ([`CvRefit::num_boost_round`]:
    /// through the best fold-mean round under [`CvStopping::FoldMean`],
    /// the best round on the purged tail of every row under
    /// [`CvStopping::PerFold`] (sized exactly as each fold is), else every
    /// round the folds ran). The retraining continues
    /// [`init_model`](Self::init_model) when set and has no eval sets. With
    /// [`target_stats`](Self::target_stats), it trains on `data` encoded by
    /// the encoder fitted on every row (over
    /// [`target_stats_label`](Self::target_stats_label) when set), which is
    /// returned as [`CvRefit::target_encoder`].
    ///
    /// The retraining is configured for all `num_boost_round` rounds and
    /// stopped through [`Trainer::on_round`] after the chosen count, so its
    /// model equals one trained on `data` for that many rounds; the folds'
    /// models under per-fold stopping are trained the same way. For
    /// `booster = ebm` the count is of EBM rounds across both stages
    /// (`num_boost_round` still caps each stage), so the model is the one
    /// after that many rounds of the full run.
    ///
    /// Fails as `run` does, or when encoding or retraining on `data` fails.
    ///
    /// ```
    /// use hessboost::training::{CrossValidation, CvStopping, Fold};
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
    ///     .early_stopping(CvStopping::fold_mean(NonZeroUsize::new(5).unwrap()))
    ///     .refit()?;
    /// // Trained on all 90 rows for as many rounds as the results report.
    /// assert_eq!(refit.num_boost_round, refit.report.results[0].rounds.len());
    /// assert_eq!(refit.model.num_boost_rounds(), refit.num_boost_round);
    /// let predictions = refit.model.predict(&data, Iterations::Best)?;
    /// assert_eq!(predictions.as_slice().len(), 90);
    /// # Ok(())
    /// # }
    /// ```
    pub fn refit(self) -> Result<CvRefit> {
        let metrics = self.checked_metrics()?;
        let report = self.report(&metrics)?;
        let (rounds, stopping) = match self.stopping {
            Some(CvStopping::PerFold { patience, tail }) => {
                let rows: Vec<usize> = (0..self.data.n_rows()).collect();
                let (stopping, rounds) =
                    self.tail_rounds(&rows, patience, &tail, maximizes(&metrics), "all rows")?;
                (rounds, Some(stopping))
            }
            _ => (
                report
                    .results
                    .first()
                    .map_or(0, |result| result.rounds.len()),
                None,
            ),
        };
        let (encoded, target_encoder) = match &self.target_stats {
            Some((encoder, columns)) => {
                let (encoded, fitted) =
                    fit_encoder(encoder, columns, self.data, self.target_stats_label)?;
                (Some(encoded), Some(fitted))
            }
            None => (None, None),
        };
        let dtrain = encoded.as_ref().unwrap_or(self.data);
        let model = self.train_rounds(dtrain, None, rounds)?.model;
        Ok(CvRefit {
            report,
            num_boost_round: rounds,
            model,
            target_encoder,
            stopping,
        })
    }

    /// Check the configuration against the data and return its metrics
    /// (the configured ones, or the objective's default; the last one is
    /// watched).
    fn checked_metrics(&self) -> Result<Vec<Box<dyn Metric>>> {
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
                 rounds; stop on the fold means instead (fold-mean early stopping)",
            ));
        }
        let boulevard = match &self.params.booster {
            BoosterKind::Boulevard(_) => true,
            BoosterKind::Ebm(ebm) => ebm.boulevard(),
            _ => false,
        };
        if boulevard && matches!(self.stopping, Some(CvStopping::PerFold { .. })) {
            return Err(HessboostError::invalid_param(
                "early_stopping",
                "per-fold stopping early-stops a trial on each fold's tail, but Boulevard \
                 fits do not early-stop; cross-validate them without it or on the fold means",
            ));
        }
        if self.history_only && matches!(self.stopping, Some(CvStopping::PerFold { .. })) {
            return Err(HessboostError::invalid_param(
                "early_stopping",
                "per-fold stopping has no fold-mean history, only the folds' reports; run it \
                 without history_only",
            ));
        }
        if let Some(CvStopping::PerFold { tail, .. }) = &self.stopping
            && tail.n_rows() != self.data.n_rows()
        {
            return Err(HessboostError::dimension_mismatch(
                "per-fold early stopping times (one per row)",
                self.data.n_rows(),
                tail.n_rows(),
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
        configured_metrics(self.params, objective.as_ref())
    }

    /// The report [`run`](Self::run) and [`refit`](Self::refit) share.
    fn report(&self, metrics: &[Box<dyn Metric>]) -> Result<CvReport> {
        let (results, folds) = match self.stopping {
            Some(CvStopping::PerFold { patience, tail }) => (
                Vec::new(),
                self.per_fold(patience, &tail, maximizes(metrics))?,
            ),
            Some(CvStopping::FoldMean { patience }) => self.fold_mean(metrics, Some(patience))?,
            None => self.fold_mean(metrics, None)?,
        };
        Ok(CvReport {
            metrics: metrics
                .iter()
                .map(|metric| metric.name().to_string())
                .collect(),
            results,
            folds,
            n_rows: self.data.n_rows(),
        })
    }

    /// Every fold trained for every round and scored on its test rows,
    /// aggregated per round and, under `patience`, truncated at the best
    /// fold-mean round, which then sizes every fold's report (none when
    /// history-only).
    fn fold_mean(
        &self,
        metrics: &[Box<dyn Metric>],
        patience: Option<NonZeroUsize>,
    ) -> Result<(Vec<CvResult>, Vec<FoldReport>)> {
        // `values[metric][round][fold]`, grown as rounds arrive (not sized
        // by `num_boost_round`, which is caller input).
        let mut values: Vec<Vec<Vec<f64>>> = vec![Vec::new(); metrics.len()];
        // Each fold's history and margins (or what they come from).
        let mut runs = Vec::with_capacity(self.folds.len());
        for fold in &self.folds {
            let (dtrain, encoder) = self.train_data(&fold.train)?;
            let dtest = self.held_out_data(&fold.test, encoder.as_ref())?;
            let TrainResult { model, history, .. } = self
                .trainer(&dtrain, self.num_boost_round)
                .eval(&dtest, "test")
                .train()?;
            for (round, eval) in history.rounds().enumerate() {
                for (per_round, &value) in values.iter_mut().zip(eval.values()) {
                    if per_round.len() == round {
                        per_round.push(Vec::with_capacity(self.folds.len()));
                    }
                    per_round[round].push(value);
                }
            }
            // A history-only run keeps nothing of the fold past its scores.
            // Without early stopping every round counts: predict now, so the
            // model goes before the next fold trains instead of every fold's
            // being kept to the end.
            if !self.history_only {
                let margins = match patience {
                    None => FoldMargins::Ready(model.predict_margin(&dtest, ..)?),
                    Some(_) => FoldMargins::Pending(Box::new(model), encoder),
                };
                runs.push((history, margins));
            }
        }
        let mut results: Vec<CvResult> = metrics
            .iter()
            .zip(values)
            .map(|(metric, per_round)| aggregate(metric.name(), &per_round))
            .collect();
        if let Some(patience) = patience
            && let Some(watched) = results.last()
            && !watched.rounds.is_empty()
        {
            let means = watched.rounds.iter().map(|round| round.mean);
            let end = best_round(means, patience, maximizes(metrics)) + 1;
            for result in &mut results {
                result.rounds.truncate(end);
            }
        }
        let rounds = results.first().map_or(0, |result| result.rounds.len());
        let reports = self
            .folds
            .iter()
            .zip(runs)
            .map(|(fold, (history, margins))| {
                let used = rounds.min(history.len());
                let last = used
                    .checked_sub(1)
                    .and_then(|round| history.rounds().nth(round));
                let margins = match margins {
                    FoldMargins::Ready(margins) => margins,
                    FoldMargins::Pending(model, encoder) => {
                        let dtest = self.held_out_data(&fold.test, encoder.as_ref())?;
                        match last {
                            _ if used == history.len() => model.predict_margin(&dtest, ..)?,
                            // A gbtree round is one iteration, and its first
                            // iterations are the shorter run's model (shrunk
                            // ones included).
                            Some(round) if matches!(self.params.booster, BoosterKind::GbTree) => {
                                model.predict_margin(&dtest, ..round.iteration() + 1)?
                            }
                            // DART rescales earlier trees and an EBM round
                            // adds a tree per term (and bag): replay the
                            // shorter run.
                            _ => {
                                let (dtrain, _) = self.train_data(&fold.train)?;
                                self.train_rounds(&dtrain, None, used)?
                                    .model
                                    .predict_margin(&dtest, ..)?
                            }
                        }
                    }
                };
                Ok(FoldReport {
                    test: fold.test.clone(),
                    num_boost_round: used,
                    scores: last.map_or_else(Vec::new, |round| round.values().to_vec()),
                    margins,
                    stopping: None,
                })
            })
            .collect::<Result<_>>()?;
        Ok((results, reports))
    }

    /// Every fold sized on its own purged tail and retrained on all its
    /// training rows for that many rounds ([`CvStopping::PerFold`]).
    fn per_fold(
        &self,
        patience: NonZeroUsize,
        tail: &PurgedTail<'_>,
        maximize: bool,
    ) -> Result<Vec<FoldReport>> {
        self.folds
            .iter()
            .enumerate()
            .map(|(f, fold)| {
                let (stopping, rounds) =
                    self.tail_rounds(&fold.train, patience, tail, maximize, &format!("fold {f}"))?;
                let (dtrain, encoder) = self.train_data(&fold.train)?;
                let dtest = self.held_out_data(&fold.test, encoder.as_ref())?;
                let TrainResult { model, history, .. } =
                    self.train_rounds(&dtrain, Some(&dtest), rounds)?;
                Ok(FoldReport {
                    test: fold.test.clone(),
                    num_boost_round: rounds,
                    scores: last_scores(&history),
                    margins: model.predict_margin(&dtest, ..)?,
                    stopping: Some(stopping),
                })
            })
            .collect()
    }

    /// The round count early stopping on `rows`' purged tail selects, with
    /// the split: train on the split's training rows, score the tail, and
    /// take the best round's index plus one (0 for no rounds). `whose`
    /// names the rows in a refusal.
    fn tail_rounds(
        &self,
        rows: &[usize],
        patience: NonZeroUsize,
        tail: &PurgedTail<'_>,
        maximize: bool,
        whose: &str,
    ) -> Result<(TailStopping, usize)> {
        let (split_at, split) = tail.split_rows(rows).map_err(|reason| {
            HessboostError::invalid_param("purged tail", format!("{whose}: {reason}"))
        })?;
        let (dfit, encoder) = self.train_data(&split.train)?;
        let dtail = self.held_out_data(&split.test, encoder.as_ref())?;
        let trial = self
            .trainer(&dfit, self.num_boost_round)
            .eval(&dtail, "early_stopping")
            .early_stopping_rounds(patience)
            .train()?;
        // Re-run the stopping rule over the trial's rounds rather than read
        // its `best_iteration`, which counts trees for an EBM (a tree per
        // term and bag each round), not rounds.
        let watched: Vec<f64> = trial
            .history
            .rounds()
            .map(|round| round.values().last().copied().unwrap_or(f64::NAN))
            .collect();
        let best = best_round(watched.iter().copied(), patience, maximize);
        let rounds = if watched.is_empty() { 0 } else { best + 1 };
        let stopping = TailStopping {
            split_at,
            fit_rows: split.train.len(),
            tail_rows: split.test.len(),
            best_score: watched.get(best).copied().unwrap_or(f64::NAN),
        };
        Ok((stopping, rounds))
    }

    /// `params` trained on `dtrain` for the first `rounds` rounds of the
    /// full `num_boost_round` run (stopped through [`Trainer::on_round`]),
    /// scoring `eval` when given.
    fn train_rounds(
        &self,
        dtrain: &DMatrix,
        eval: Option<&DMatrix>,
        rounds: usize,
    ) -> Result<TrainResult> {
        // The hook cannot stop before the first round.
        let num_boost_round = if rounds == 0 { 0 } else { self.num_boost_round };
        let mut completed = 0;
        let mut trainer = self.trainer(dtrain, num_boost_round).on_round(move |_| {
            completed += 1;
            if completed == rounds {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        if let Some(eval) = eval {
            trainer = trainer.eval(eval, "test");
        }
        trainer.train()
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

    /// The matrix of `rows` to train on, target-encoded with statistics of
    /// those rows when configured (with the fitted encoder).
    fn train_data(&self, rows: &[usize]) -> Result<(DMatrix, Option<FittedTargetEncoder>)> {
        let dtrain = self.data.select_rows(rows)?;
        let Some((encoder, columns)) = &self.target_stats else {
            return Ok((dtrain, None));
        };
        // `checked_metrics` checked the length and the folds and tail
        // splits hold rows of the data.
        let labels: Option<Vec<f32>> = self
            .target_stats_label
            .map(|labels| rows.iter().map(|&row| labels[row]).collect());
        let (dtrain, fitted) = fit_encoder(encoder, columns, &dtrain, labels.as_deref())?;
        Ok((dtrain, Some(fitted)))
    }

    /// The matrix of held-out `rows`, encoded by the statistics `encoder`
    /// fitted on the training rows ([`train_data`](Self::train_data)).
    fn held_out_data(
        &self,
        rows: &[usize],
        encoder: Option<&FittedTargetEncoder>,
    ) -> Result<DMatrix> {
        let data = self.data.select_rows(rows)?;
        match encoder {
            Some(fitted) => fitted.transform(&data),
            None => Ok(data),
        }
    }
}

/// A fold's test margins in [`CrossValidation::fold_mean`]: taken as soon as
/// its run ends when every round counts, else its model (with the fold's
/// encoder) kept until the fold-mean round count is known.
enum FoldMargins {
    Ready(Predictions),
    Pending(Box<BoostedModel>, Option<FittedTargetEncoder>),
}

/// Whether the watched (last) metric is maximized.
fn maximizes(metrics: &[Box<dyn Metric>]) -> bool {
    metrics.last().is_some_and(|metric| metric.maximize())
}

/// The scores of `history`'s last round (none for no rounds).
fn last_scores(history: &EvalHistory) -> Vec<f64> {
    history
        .last()
        .map_or_else(Vec::new, |round| round.values().to_vec())
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
/// stopping and per-fold target statistics) through [`CrossValidation`],
/// which also reports each fold's margins.
pub fn cv(
    params: &TrainingParams,
    data: &DMatrix,
    num_boost_round: usize,
    nfold: usize,
    seed: u64,
) -> Result<Vec<CvResult>> {
    let folds = Fold::k_fold(data.n_rows(), nfold, seed)?;
    Ok(CrossValidation::new(params, data, num_boost_round, folds)
        .history_only()
        .run()?
        .results)
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
            .unwrap()
            .results;
        let rmse: Vec<f64> = full[1].rounds.iter().map(|r| r.mean).collect();
        let best = (0..rmse.len())
            .min_by(|&a, &b| rmse[a].total_cmp(&rmse[b]))
            .unwrap();
        let stopped = CrossValidation::new(&params, &d, 40, folds)
            .early_stopping(CvStopping::fold_mean(NonZeroUsize::new(5).unwrap()))
            .run()
            .unwrap()
            .results;
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
            .early_stopping(CvStopping::fold_mean(NonZeroUsize::new(100).unwrap()))
            .run()
            .unwrap()
            .results;
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
        let results = CrossValidation::new(&params, &d, 3, folds)
            .run()
            .unwrap()
            .results;
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
            .unwrap()
            .results;

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
