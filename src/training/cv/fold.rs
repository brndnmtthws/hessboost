//! [`Fold`] builders: caller-supplied, shuffled k-fold, forward-chaining,
//! and purged forward folds.

use crate::error::{HessboostError, Result};
use crate::rng::Rng;

/// The training and held-out (test) rows of one cross-validation fold, as
/// row indices into the cross-validated [`DMatrix`](crate::data::DMatrix): XGBoost's `folds=`
/// entries.
///
/// Build folds with [`Fold::new`] (grouped or any other custom split),
/// [`Fold::k_fold`] (what [`cv`](super::cv()) uses), [`Fold::forward_chaining`]
/// (time-ordered rows), or [`Fold::purged_forward`] (timestamped rows with
/// label windows).
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
    /// overlap and repeat rows; [`CrossValidation::run`](super::CrossValidation::run) only requires both
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

impl Fold {
    /// Forward, purged folds over rows grouped by timestamp, for rows whose
    /// labels span a time window (forecast horizons, overlapping returns).
    ///
    /// `decision_at[i]` is row `i`'s decision (feature) time and
    /// `label_end[i]` the end of its label window plus any embargo, on one
    /// integer time scale (e.g. epoch seconds); rows may come in any order
    /// and share times. The last `validation_fraction` of the distinct
    /// decision times (from index `min(n - 1, floor((1 - fraction) * n))` of
    /// the `n` sorted distinct times) is cut into `blocks` contiguous test
    /// blocks. Fold `j` tests on every row decided inside block `j` and
    /// trains on every row decided before the block whose `label_end` is at
    /// or before the block's first decision time: a label ending exactly at
    /// the block start is kept, one ending a tick later is purged. A
    /// decision time is never split between training and test rows, and
    /// every test row is decided strictly after every training row of its
    /// fold. Unlike [`Fold::forward_chaining`], which purges a fixed number
    /// of rows, this purges by each row's own label window, so irregular
    /// schedules and horizons that vary by row purge exactly the overlapping
    /// rows.
    ///
    /// When the first block's purge would leave fewer than `min_train`
    /// training rows, its start moves later one decision time at a time
    /// (shrinking the test tail) until it leaves `min_train`; the tail is
    /// then cut into the blocks.
    ///
    /// Fails when the slices differ in length or are empty, a label ends
    /// before its decision, `validation_fraction` is not in `(0, 1)`,
    /// `blocks` is 0, no start leaves `min_train` training rows and a
    /// decision time per block, or a fold would have no training or test
    /// rows.
    ///
    /// ```
    /// use hessboost::training::Fold;
    ///
    /// # fn main() -> hessboost::error::Result<()> {
    /// // Two rows per day for 10 days; each label ends two days later.
    /// let day = 86_400;
    /// let decision_at: Vec<i64> = (0..20).map(|i| (i / 2) * day).collect();
    /// let label_end: Vec<i64> = decision_at.iter().map(|at| at + 2 * day).collect();
    /// let folds = Fold::purged_forward(&decision_at, &label_end, 0.2, 1, 1)?;
    /// // Days 8 and 9 test. Day 7's labels run past day 8 and are purged;
    /// // day 6's end exactly at day 8 and train.
    /// assert_eq!(folds[0].test, (16..20).collect::<Vec<_>>());
    /// assert_eq!(folds[0].train, (0..14).collect::<Vec<_>>());
    /// # Ok(())
    /// # }
    /// ```
    pub fn purged_forward(
        decision_at: &[i64],
        label_end: &[i64],
        validation_fraction: f64,
        blocks: usize,
        min_train: usize,
    ) -> Result<Vec<Fold>> {
        let refuse = |reason: String| HessboostError::invalid_param("purged folds", reason);
        if decision_at.len() != label_end.len() || decision_at.is_empty() {
            return Err(refuse(format!(
                "need one decision time and one label end per row, got {} and {}",
                decision_at.len(),
                label_end.len()
            )));
        }
        if !(validation_fraction > 0.0 && validation_fraction < 1.0) || blocks == 0 {
            return Err(refuse(format!(
                "need a validation fraction in (0, 1) and at least one block, got \
                 {validation_fraction} and {blocks}"
            )));
        }
        if let Some(row) = (0..decision_at.len()).find(|&row| label_end[row] < decision_at[row]) {
            return Err(refuse(format!(
                "row {row}'s label ends before its decision"
            )));
        }
        let mut times = decision_at.to_vec();
        times.sort_unstable();
        times.dedup();
        let count = times.len();
        let mut split = (count - 1).min(((1.0 - validation_fraction) * count as f64) as usize);
        // A row trains a fold starting at `start` when `at < start` and
        // `end <= start`; as `end >= at`, that is `start >= end`, strictly
        // when the label ends at its decision. Sorted by `(end, strict)`,
        // the rows a start trains are a prefix, and the candidate starts
        // only grow, so one pass counts them all.
        let mut keys: Vec<(i64, bool)> = decision_at
            .iter()
            .zip(label_end)
            .map(|(&at, &end)| (end, end == at))
            .collect();
        keys.sort_unstable();
        let mut trained = 0;
        let mut trainable = |start: i64| {
            while let Some(&(end, strict)) = keys.get(trained)
                && (end < start || (end == start && !strict))
            {
                trained += 1;
            }
            trained
        };
        while trainable(times[split]) < min_train {
            split += 1;
            if split + blocks > count {
                return Err(refuse(format!(
                    "no test start leaves {min_train} purged training rows and {blocks} test \
                     decision times"
                )));
            }
        }
        let tail = &times[split..];
        if tail.len() < blocks {
            return Err(refuse(format!(
                "{blocks} test blocks need {blocks} distinct decision times; the tail has {}",
                tail.len()
            )));
        }
        let starts: Vec<i64> = (0..blocks)
            .map(|block| tail[block * tail.len() / blocks])
            .collect();
        let mut folds = Vec::with_capacity(blocks);
        for (block, &start) in starts.iter().enumerate() {
            let end = starts.get(block + 1).copied();
            let test: Vec<usize> = (0..decision_at.len())
                .filter(|&row| {
                    decision_at[row] >= start && end.is_none_or(|end| decision_at[row] < end)
                })
                .collect();
            let train: Vec<usize> = (0..decision_at.len())
                .filter(|&row| decision_at[row] < start && label_end[row] <= start)
                .collect();
            if train.is_empty() || test.is_empty() {
                return Err(refuse(format!(
                    "test block {block} leaves no training or test rows"
                )));
            }
            folds.push(Fold::new(train, test));
        }
        Ok(folds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Six rows per daily decision over 20 days, labels ending `horizon`
    /// hours (plus a one-day embargo) after their decision.
    fn schedule(horizon: i64) -> (Vec<i64>, Vec<i64>) {
        let day = 86_400;
        let decisions: Vec<i64> = (0..20)
            .flat_map(|index| std::iter::repeat_n(index * day, 6))
            .collect();
        let ends = decisions
            .iter()
            .map(|at| at + horizon * 3_600 + day)
            .collect();
        (decisions, ends)
    }

    fn times(decisions: &[i64], rows: &[usize]) -> Vec<i64> {
        rows.iter().map(|&row| decisions[row]).collect()
    }

    #[test]
    fn purged_folds_keep_decision_times_whole_and_test_after_training() {
        let day = 86_400;
        let (decisions, ends) = schedule(72);
        for blocks in [1, 2, 4] {
            let folds = Fold::purged_forward(&decisions, &ends, 0.2, blocks, 1).unwrap();
            assert_eq!(folds.len(), blocks);
            let mut tested = std::collections::BTreeSet::new();
            for fold in &folds {
                let (train, test) = (
                    times(&decisions, &fold.train),
                    times(&decisions, &fold.test),
                );
                assert!(train.iter().max() < test.iter().min());
                // A used decision time keeps all six of its rows.
                for rows in [&train, &test] {
                    for at in rows {
                        assert_eq!(rows.iter().filter(|other| *other == at).count(), 6);
                    }
                }
                tested.extend(test);
            }
            // The blocks tile the last 20% of the days.
            assert_eq!(
                tested.into_iter().collect::<Vec<_>>(),
                [16, 17, 18, 19].map(|d| d * day)
            );
        }
        // Row order is irrelevant: folds hold indices, not positions.
        let reversed: Vec<i64> = decisions.iter().rev().copied().collect();
        let reversed_ends: Vec<i64> = ends.iter().rev().copied().collect();
        let fold = &Fold::purged_forward(&reversed, &reversed_ends, 0.2, 1, 1).unwrap()[0];
        assert!(fold.test.iter().all(|&row| reversed[row] >= 16 * day));
    }

    #[test]
    fn purge_keeps_a_label_ending_at_the_block_start_and_drops_one_a_tick_later() {
        let day = 86_400;
        let (decisions, mut ends) = schedule(72);
        // Day 12's labels (72h + one day) end exactly at the day-16 start.
        let fold = &Fold::purged_forward(&decisions, &ends, 0.2, 1, 1).unwrap()[0];
        assert_eq!(
            times(&decisions, &fold.train).into_iter().max(),
            Some(12 * day)
        );
        for (end, &at) in ends.iter_mut().zip(&decisions) {
            if at == 12 * day {
                *end += 1;
            }
        }
        let fold = &Fold::purged_forward(&decisions, &ends, 0.2, 1, 1).unwrap()[0];
        assert_eq!(
            times(&decisions, &fold.train).into_iter().max(),
            Some(11 * day)
        );
        // Row-varying windows: only the rows whose own label overlaps go.
        let (decisions, mut ends) = schedule(72);
        for row in (0..decisions.len()).filter(|row| row % 6 == 0) {
            ends[row] = decisions[row] + day;
        }
        let fold = &Fold::purged_forward(&decisions, &ends, 0.2, 1, 1).unwrap()[0];
        for at in [13, 14, 15] {
            let kept = times(&decisions, &fold.train)
                .into_iter()
                .filter(|&t| t == at * day)
                .count();
            assert_eq!(kept, 1, "day {at}: only its short-label row trains");
        }
    }

    #[test]
    fn purged_folds_move_the_start_to_leave_min_train_rows() {
        let day = 86_400;
        // 168h + one day purges eight days before the block.
        let (decisions, ends) = schedule(168);
        let fold = &Fold::purged_forward(&decisions, &ends, 0.2, 1, 1).unwrap()[0];
        assert_eq!(
            times(&decisions, &fold.train).into_iter().max(),
            Some(8 * day)
        );
        // Days 0..=8 hold 54 rows; 60 moves the start one day, to day 17.
        let fold = &Fold::purged_forward(&decisions, &ends, 0.2, 1, 60).unwrap()[0];
        assert_eq!(fold.train.len(), 60);
        assert_eq!(
            times(&decisions, &fold.test).into_iter().min(),
            Some(17 * day)
        );
        let folds = Fold::purged_forward(&decisions, &ends, 0.2, 2, 60).unwrap();
        assert_eq!(
            times(&decisions, &folds[1].test).into_iter().min(),
            Some(18 * day)
        );
        for (blocks, min_train) in [(3, 66), (1, 80), (5, 1)] {
            assert!(Fold::purged_forward(&decisions, &ends, 0.2, blocks, min_train).is_err());
        }
        // A tail whose purge leaves nothing moves until a row trains: day 7's
        // start keeps no row, day 8's keeps day 0's six.
        let fold = &Fold::purged_forward(&decisions, &ends, 0.65, 1, 1).unwrap()[0];
        assert_eq!(fold.train, [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn purged_folds_refuse_inconsistent_inputs() {
        let (decisions, ends) = schedule(24);
        for (fraction, blocks) in [(0.0, 1), (1.0, 1), (f64::NAN, 1), (0.2, 0)] {
            assert!(Fold::purged_forward(&decisions, &ends, fraction, blocks, 1).is_err());
        }
        assert!(Fold::purged_forward(&decisions, &ends[1..], 0.2, 1, 1).is_err());
        assert!(Fold::purged_forward(&[], &[], 0.2, 1, 1).is_err());
        assert!(Fold::purged_forward(&[5, 6], &[4, 7], 0.5, 1, 1).is_err());
    }
}
