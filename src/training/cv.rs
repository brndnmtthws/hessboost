//! Cross-validation, mirroring `xgboost.cv`: shuffled k-fold ([`cv`]) or
//! caller-supplied folds ([`CrossValidation`], [`Fold`]), including
//! forward-chaining folds for time-ordered rows.

use crate::config::TrainingParams;
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::objective::create_objective;
use crate::rng::Rng;
use crate::training::Trainer;
use crate::training::train::configured_metrics;

/// Per-metric cross-validation history, aggregated across folds.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CvResult {
    /// Metric name.
    pub metric: String,
    /// Mean held-out metric value per boosting round.
    pub test_mean: Vec<f64>,
    /// Standard deviation (population, like XGBoost's) of the held-out
    /// metric across folds per round.
    pub test_std: Vec<f64>,
}

/// The training and held-out (test) rows of one cross-validation fold, as
/// row indices into the cross-validated [`DMatrix`]: XGBoost's `folds=`
/// entries.
///
/// Build folds with [`Fold::new`] (grouped, purged, or any other custom
/// split), [`Fold::k_fold`] (what [`cv`] uses), or
/// [`Fold::forward_chaining`] (time-ordered rows).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Fold {
    /// Rows the fold trains on, in this order.
    pub train: Vec<usize>,
    /// Rows the fold is evaluated on.
    pub test: Vec<usize>,
}

impl Fold {
    /// A fold training on `train` and evaluated on `test`. The two may
    /// overlap and repeat rows; [`CrossValidation::run`] only requires both
    /// to be non-empty and in bounds.
    pub fn new(train: Vec<usize>, test: Vec<usize>) -> Self {
        Fold { train, test }
    }

    /// Shuffled k-fold over `n_rows` rows: the rows are shuffled with
    /// `seed` and dealt to the `nfold` test sets round-robin; each fold
    /// trains on the other folds' rows. Suitable only for exchangeable rows:
    /// on serially dependent data (time series, overlapping label windows)
    /// every training set holds rows from after its test rows; use
    /// [`Fold::forward_chaining`] there.
    pub fn k_fold(n_rows: usize, nfold: usize, seed: u64) -> Result<Vec<Fold>> {
        if nfold < 2 {
            return Err(HessboostError::invalid_param("nfold", "must be >= 2"));
        }
        if n_rows < nfold {
            return Err(HessboostError::invalid_param(
                "nfold",
                "more folds than rows",
            ));
        }
        let mut order: Vec<usize> = (0..n_rows).collect();
        Rng::new(seed).shuffle(&mut order);
        let mut tests: Vec<Vec<usize>> = vec![Vec::new(); nfold];
        for (i, &row) in order.iter().enumerate() {
            tests[i % nfold].push(row);
        }
        Ok((0..nfold)
            .map(|f| {
                let train = tests
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != f)
                    .flat_map(|(_, rows)| rows.iter().copied())
                    .collect();
                Fold::new(train, tests[f].clone())
            })
            .collect())
    }

    /// Forward-chaining (expanding-window) folds over `n_rows` rows in time
    /// order (row `i` precedes row `i + 1`), like scikit-learn's
    /// `TimeSeriesSplit`: the last `n_splits × s` rows, `s = n_rows /
    /// (n_splits + 1)`, form `n_splits` consecutive test blocks of `s` rows;
    /// each fold trains on every row before its test block except the `gap`
    /// rows immediately before it. A `gap` of at least the label horizon
    /// (e.g. 24 for hourly rows whose labels look 24 hours ahead) purges the
    /// training rows whose labels overlap the test block. No training row
    /// follows a test row, so no embargo is needed.
    ///
    /// Fails when `n_splits` is 0, when the rows do not fill `n_splits + 1`
    /// blocks, or when `gap` leaves the first fold no training rows.
    pub fn forward_chaining(n_rows: usize, n_splits: usize, gap: usize) -> Result<Vec<Fold>> {
        if n_splits == 0 {
            return Err(HessboostError::invalid_param("n_splits", "must be >= 1"));
        }
        // `n_splits < n_rows` keeps `n_splits + 1` from overflowing and every
        // block non-empty.
        if n_splits >= n_rows {
            return Err(HessboostError::invalid_param(
                "n_splits",
                format!("{n_rows} rows do not fill {n_splits} + 1 blocks"),
            ));
        }
        let block = n_rows / (n_splits + 1);
        let first_test = n_rows - n_splits * block;
        if first_test <= gap {
            return Err(HessboostError::invalid_param(
                "gap",
                format!("a gap of {gap} rows leaves the first fold no training rows"),
            ));
        }
        Ok((0..n_splits)
            .map(|i| {
                let start = first_test + i * block;
                Fold::new((0..start - gap).collect(), (start..start + block).collect())
            })
            .collect())
    }
}

/// Cross-validation over caller-supplied [`Fold`]s (XGBoost's `cv(...,
/// folds=...)`), optionally with early stopping.
///
/// Each fold trains `params` for `num_boost_round` rounds on its training
/// rows and evaluates the metrics (`params.eval_metric`, or the objective's
/// default) on its test rows after every round; [`run`](Self::run)
/// averages them across folds.
///
/// ```
/// use hessboost::training::{CrossValidation, Fold};
/// use hessboost::prelude::*;
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
///     .early_stopping_rounds(3)
///     .run()?;
/// // With early stopping, the last round reported is the best one.
/// let rmse = &results[0];
/// assert_eq!(rmse.metric, "rmse");
/// assert!(rmse.test_mean.len() <= 20);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct CrossValidation<'a> {
    params: &'a TrainingParams,
    data: &'a DMatrix,
    num_boost_round: usize,
    folds: Vec<Fold>,
    early_stopping_rounds: Option<usize>,
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
        }
    }

    /// Early stopping on the fold-averaged metric, as in `xgboost.cv`: the
    /// watched metric is the last one, and the best round is the one with
    /// the best mean after which the mean fails to improve for `rounds`
    /// consecutive rounds (or the best within `num_boost_round`). The
    /// results end at the best round, so `test_mean.len() - 1` is its
    /// index. Unlike `xgboost.cv`, which truncates only when patience runs
    /// out, they are truncated whenever early stopping is on.
    ///
    /// The folds still train every round: the stopping point depends on
    /// all folds' metrics.
    #[must_use]
    pub fn early_stopping_rounds(mut self, rounds: usize) -> Self {
        self.early_stopping_rounds = Some(rounds);
        self
    }

    /// Train and evaluate every fold, returning one [`CvResult`] per metric
    /// in the configured metric order.
    ///
    /// Fails when there are no folds, a fold's training or test rows are
    /// empty or out of bounds, `early_stopping_rounds` is 0, or training a
    /// fold fails.
    pub fn run(self) -> Result<Vec<CvResult>> {
        let CrossValidation {
            params,
            data,
            num_boost_round,
            folds,
            early_stopping_rounds,
        } = self;
        if folds.is_empty() {
            return Err(HessboostError::invalid_param("folds", "no folds"));
        }
        if early_stopping_rounds == Some(0) {
            return Err(HessboostError::invalid_param(
                "early_stopping_rounds",
                "must be greater than zero",
            ));
        }
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
            }
        }
        let objective = create_objective(params, data.n_targets())?;
        let metrics = configured_metrics(params, objective.as_ref())?;
        let maximize = metrics.last().is_some_and(|m| m.maximize());

        // values[metric][round][fold], grown as rounds arrive (not sized by
        // `num_boost_round`, which is caller input).
        let mut values: Vec<Vec<Vec<f64>>> = vec![Vec::new(); metrics.len()];
        for fold in &folds {
            let dtrain = data.select_rows(&fold.train)?;
            let dtest = data.select_rows(&fold.test)?;
            let res = Trainer::new(params, &dtrain, num_boost_round)
                .eval(&dtest, "test")
                .train()?;
            for (round, eval) in res.history.iter().enumerate() {
                for (per_round, (_, _, value)) in values.iter_mut().zip(&eval.scores) {
                    if per_round.len() == round {
                        per_round.push(Vec::with_capacity(folds.len()));
                    }
                    per_round[round].push(*value);
                }
            }
        }

        let mut out: Vec<CvResult> = metrics
            .iter()
            .zip(values)
            .map(|(metric, per_round)| {
                let (test_mean, test_std) = per_round
                    .iter()
                    .map(|vals| {
                        let len = vals.len() as f64;
                        let mean = vals.iter().sum::<f64>() / len;
                        let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / len;
                        (mean, var.sqrt())
                    })
                    .unzip();
                CvResult {
                    metric: metric.name().to_string(),
                    test_mean,
                    test_std,
                }
            })
            .collect();

        if let Some(patience) = early_stopping_rounds
            && let Some(watched) = out.last()
            && !watched.test_mean.is_empty()
        {
            let end = best_round(&watched.test_mean, patience, maximize) + 1;
            for result in &mut out {
                result.test_mean.truncate(end);
                result.test_std.truncate(end);
            }
        }
        Ok(out)
    }
}

/// The round early stopping selects on `scores`: the same rule as
/// [`Trainer::early_stopping_rounds`] (strict improvement; round 0 when no
/// score ever improves, e.g. NaN).
fn best_round(scores: &[f64], patience: usize, maximize: bool) -> usize {
    let mut best = if maximize {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    let mut best_round = 0;
    let mut since = 0;
    for (round, &score) in scores.iter().enumerate() {
        let improved = if maximize { score > best } else { score < best };
        if improved {
            best = score;
            best_round = round;
            since = 0;
        } else {
            since += 1;
            if since >= patience {
                break;
            }
        }
    }
    best_round
}

/// Shuffled `nfold` cross-validation ([`Fold::k_fold`] folds), returning one
/// [`CvResult`] per evaluation metric. Every fold trains for the full
/// `num_boost_round` rounds (no early stopping). The metric list comes from
/// `params.eval_metric` or the objective's default.
///
/// For time-ordered or grouped rows, supply the folds (and optionally early
/// stopping) through [`CrossValidation`].
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
            .objective("reg:squarederror")
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();

        let results = cv(&params, &d, 30, 5, 42).unwrap();
        assert_eq!(results.len(), 1);
        let rmse = &results[0];
        assert_eq!(rmse.metric, "rmse");
        assert_eq!(rmse.test_mean.len(), 30);
        // Held-out error should drop from first to last round.
        assert!(rmse.test_mean[29] < rmse.test_mean[0]);
        // Std is non-negative and finite.
        assert!(rmse.test_std.iter().all(|s| s.is_finite() && *s >= 0.0));
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
        let folds = Fold::forward_chaining(20, 2, 0).unwrap();
        assert!(
            CrossValidation::new(&params, &d, 2, folds)
                .early_stopping_rounds(0)
                .run()
                .is_err()
        );
    }

    #[test]
    fn metrics_keep_their_configured_order() {
        let d = step_data(60);
        let params = TrainingParams::builder()
            .eval_metric("rmse")
            .eval_metric("mae")
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
            .eval_metric("mae")
            .eval_metric("rmse")
            .max_depth(6)
            .eta(0.8)
            .build()
            .unwrap();
        let folds = Fold::k_fold(d.n_rows(), 4, 3).unwrap();
        let full = CrossValidation::new(&params, &d, 40, folds.clone())
            .run()
            .unwrap();
        let rmse = &full[1].test_mean;
        let best = (0..rmse.len())
            .min_by(|&a, &b| rmse[a].total_cmp(&rmse[b]))
            .unwrap();
        let stopped = CrossValidation::new(&params, &d, 40, folds)
            .early_stopping_rounds(5)
            .run()
            .unwrap();
        // The results end at the last round that improved on every earlier
        // one before 5 rounds without improvement.
        let end = stopped[1].test_mean.len();
        assert!(end + 5 <= 40, "ended after {end} rounds");
        let b = end - 1;
        assert!(rmse[..b].iter().all(|&v| v > rmse[b]));
        assert!(rmse[b + 1..=b + 5].iter().all(|&v| v >= rmse[b]));
        for (s, f) in stopped.iter().zip(&full) {
            assert_eq!(s.test_mean, f.test_mean[..end]);
            assert_eq!(s.test_std, f.test_std[..end]);
        }
        // Patience longer than the run still ends at the best round.
        let folds = Fold::k_fold(d.n_rows(), 4, 3).unwrap();
        let long = CrossValidation::new(&params, &d, 40, folds)
            .early_stopping_rounds(100)
            .run()
            .unwrap();
        assert_eq!(long[1].test_mean, rmse[..=best]);
    }
}
