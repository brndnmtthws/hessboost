//! `CrossValidation`: refitting on every row, continuing a model in every
//! fold, per-fold target statistics over a separate per-row target, each
//! fold's margins and round count, and per-fold early stopping; purged
//! forward folds at given block starts.

use hessboost::data::FeatureType;
use hessboost::data::target_stats::OrderedTargetEncoder;
use hessboost::model::{Iterations, ModelFormat};
use hessboost::objective::{Objective, Quantiles};
use hessboost::prelude::{BoostedModel, DMatrix, HessboostError, Trainer, TrainingParams};
use hessboost::training::{CrossValidation, CvResult, CvStopping, Fold, PurgedTail, TailCut};
use std::num::NonZeroUsize;

mod common;
use common::{four_features, invalid_data, invalid_param, lcg};

const ROWS: usize = 150;

/// Four features and a noisy label, so deep fast trees overfit.
fn regression() -> DMatrix {
    let mut noise = lcg(9);
    let x: Vec<f32> = (0..ROWS).flat_map(four_features).collect();
    let y: Vec<f32> = (0..ROWS)
        .map(|i| {
            let [a, b, ..] = four_features(i);
            3.0 * a - b + 2.0 * noise()
        })
        .collect();
    DMatrix::from_dense(&x, ROWS, 4)
        .unwrap()
        .with_labels(&y)
        .unwrap()
}

/// Column 0 one of 6 categories, column 1 numeric; two label columns, the
/// first depending on the category.
fn categorical(n_targets: usize) -> DMatrix {
    let x: Vec<f32> = (0..ROWS)
        .flat_map(|i| [(i % 6) as f32, ((i * 13) % 11) as f32])
        .collect();
    let y: Vec<f32> = (0..ROWS)
        .flat_map(|i| {
            let targets = [(i % 6) as f32 + ((i * 7) % 3) as f32, ((i * 5) % 4) as f32];
            targets.into_iter().take(n_targets)
        })
        .collect();
    DMatrix::from_dense(&x, ROWS, 2)
        .unwrap()
        .with_label_matrix(&y, n_targets)
        .unwrap()
        .with_feature_types(&[FeatureType::Categorical, FeatureType::Numerical])
        .unwrap()
}

fn params() -> TrainingParams {
    TrainingParams::builder()
        .max_depth(6)
        .eta(0.8)
        .subsample(0.8)
        .seed(3)
        .build()
        .unwrap()
}

fn folds() -> Vec<Fold> {
    Fold::k_fold(ROWS, 3, 5).unwrap()
}

fn bytes(model: &BoostedModel) -> Vec<u8> {
    model.encode(ModelFormat::Binary).unwrap()
}

/// `model` equals `expected` byte for byte and in its predictions on `data`.
fn assert_same_model(model: &BoostedModel, expected: &BoostedModel, data: &DMatrix) {
    assert_eq!(bytes(model), bytes(expected));
    let predict = |m: &BoostedModel| m.predict(data, Iterations::Best).unwrap().into_vec();
    assert_eq!(predict(model), predict(expected));
}

/// The first metric's per-round fold means of the histories `per_fold`
/// (`[fold][round]`), summed in fold order as cross-validation does.
fn fold_means(per_fold: &[Vec<f64>]) -> Vec<f64> {
    (0..per_fold[0].len())
        .map(|r| per_fold.iter().map(|f| f[r]).sum::<f64>() / per_fold.len() as f64)
        .collect()
}

fn means(result: &CvResult) -> Vec<f64> {
    result.rounds.iter().map(|r| r.mean).collect()
}

/// `trainer`'s test history of the first metric.
fn history(trainer: Trainer<'_>) -> Vec<f64> {
    let res = trainer.train().unwrap();
    res.history.rounds().map(|r| r.values()[0]).collect()
}

#[test]
fn refit_trains_every_row_for_the_chosen_rounds() {
    let data = regression();
    let params = params();
    let refit = CrossValidation::new(&params, &data, 60, folds())
        .early_stopping(CvStopping::fold_mean(NonZeroUsize::new(4).unwrap()))
        .refit()
        .unwrap();
    let rounds = refit.num_boost_round;
    assert!((1..56).contains(&rounds), "stopped after {rounds}");
    assert_eq!(refit.report.results[0].rounds.len(), rounds);
    let run = CrossValidation::new(&params, &data, 60, folds())
        .early_stopping(CvStopping::fold_mean(NonZeroUsize::new(4).unwrap()))
        .run()
        .unwrap();
    assert_eq!(refit.report.results[0].rounds, run.results[0].rounds);
    assert!(refit.target_encoder.is_none());
    let expected = Trainer::new(&params, &data, rounds).train().unwrap().model;
    assert_same_model(&refit.model, &expected, &data);

    // Without early stopping, every round.
    let refit = CrossValidation::new(&params, &data, 12, folds())
        .refit()
        .unwrap();
    assert_eq!(refit.num_boost_round, 12);
    let expected = Trainer::new(&params, &data, 12).train().unwrap().model;
    assert_same_model(&refit.model, &expected, &data);
}

#[test]
fn refit_encodes_every_row_with_the_target_stats_it_returns() {
    let data = categorical(1);
    let params = params();
    let encoder = OrderedTargetEncoder::builder().seed(4).build().unwrap();
    let refit = CrossValidation::new(&params, &data, 30, folds())
        .early_stopping(CvStopping::fold_mean(NonZeroUsize::new(3).unwrap()))
        .target_stats(encoder.clone(), vec![0])
        .refit()
        .unwrap();
    let (encoded, fitted) = encoder.fit_transform(&data, &[0]).unwrap();
    assert_eq!(refit.target_encoder, Some(fitted));
    let expected = Trainer::new(&params, &encoded, refit.num_boost_round)
        .train()
        .unwrap()
        .model;
    assert_same_model(&refit.model, &expected, &encoded);
}

#[test]
fn target_stats_label_encodes_a_multi_target_matrix() {
    let data = categorical(2);
    let params = params();
    let encoder = OrderedTargetEncoder::builder().seed(4).build().unwrap();
    let first: Vec<f32> = data.labels().unwrap().iter().step_by(2).copied().collect();

    // The data's own labels are two per row: refused by the encoder.
    let (input, _) = invalid_data(
        CrossValidation::new(&params, &data, 4, folds())
            .target_stats(encoder.clone(), vec![0])
            .run(),
    );
    assert_eq!(input, "labels");

    let refit = CrossValidation::new(&params, &data, 4, folds())
        .target_stats(encoder.clone(), vec![0])
        .target_stats_label(&first)
        .refit()
        .unwrap();
    let per_fold: Vec<Vec<f64>> = folds()
        .iter()
        .map(|fold| {
            let labels: Vec<f32> = fold.train.iter().map(|&row| first[row]).collect();
            let (dtrain, fitted) = encoder
                .fit_transform_with_labels(&data.select_rows(&fold.train).unwrap(), &[0], &labels)
                .unwrap();
            let dtest = fitted
                .transform(&data.select_rows(&fold.test).unwrap())
                .unwrap();
            history(Trainer::new(&params, &dtrain, 4).eval(&dtest, "test"))
        })
        .collect();
    assert_eq!(means(&refit.report.results[0]), fold_means(&per_fold));
    let (encoded, fitted) = encoder
        .fit_transform_with_labels(&data, &[0], &first)
        .unwrap();
    assert_eq!(refit.target_encoder, Some(fitted));
    let expected = Trainer::new(&params, &encoded, 4).train().unwrap().model;
    assert_same_model(&refit.model, &expected, &encoded);
}

#[test]
fn init_model_is_continued_in_every_fold_and_the_refit() {
    let data = regression();
    let params = params();
    let base = Trainer::new(&params, &data, 5).train().unwrap().model;
    let refit = CrossValidation::new(&params, &data, 4, folds())
        .init_model(&base)
        .refit()
        .unwrap();
    let per_fold: Vec<Vec<f64>> = folds()
        .iter()
        .map(|fold| {
            let dtrain = data.select_rows(&fold.train).unwrap();
            let dtest = data.select_rows(&fold.test).unwrap();
            history(
                Trainer::new(&params, &dtrain, 4)
                    .init_model(&base)
                    .eval(&dtest, "test"),
            )
        })
        .collect();
    assert_eq!(means(&refit.report.results[0]), fold_means(&per_fold));
    assert_eq!(refit.num_boost_round, 4);
    assert_eq!(refit.model.num_boost_rounds(), 9);
    let expected = Trainer::new(&params, &data, 4)
        .init_model(&base)
        .train()
        .unwrap()
        .model;
    assert_same_model(&refit.model, &expected, &data);
}

#[test]
fn conflicting_target_options_are_refused() {
    let data = categorical(1);
    let params = params();
    let encoder = OrderedTargetEncoder::builder().build().unwrap();
    let labels = vec![0.0; ROWS];
    let cv = || CrossValidation::new(&params, &data, 2, folds());

    assert_eq!(
        invalid_param(cv().target_stats_label(&labels).run()),
        "target_stats_label"
    );
    assert!(matches!(
        cv().target_stats(encoder.clone(), vec![0])
            .target_stats_label(&labels[1..])
            .refit(),
        Err(HessboostError::DimensionMismatch { expected: ROWS, got, .. }) if got == ROWS - 1
    ));
    let base = Trainer::new(&params, &data, 2).train().unwrap().model;
    assert_eq!(
        invalid_param(cv().init_model(&base).target_stats(encoder, vec![0]).run()),
        "target_stats"
    );
}

/// [`regression`]'s rows decided one minute apart, each label ending three
/// minutes later.
fn schedule() -> (Vec<i64>, Vec<i64>) {
    let decision_at: Vec<i64> = (0..ROWS as i64).map(|i| i * 60).collect();
    let label_end = decision_at.iter().map(|at| at + 180).collect();
    (decision_at, label_end)
}

/// `params` trained on `rows` of `data` for `rounds` rounds, with its test
/// metrics on `test` after the last one.
fn trained(
    params: &TrainingParams,
    data: &DMatrix,
    rows: &[usize],
    test: &[usize],
    rounds: usize,
) -> (BoostedModel, Vec<f64>) {
    let dtrain = data.select_rows(rows).unwrap();
    let dtest = data.select_rows(test).unwrap();
    let res = Trainer::new(params, &dtrain, rounds)
        .eval(&dtest, "test")
        .train()
        .unwrap();
    let scores = res.history.last().unwrap().values().to_vec();
    (res.model, scores)
}

/// What per-fold stopping does by hand: early stopping on the purged tail
/// of `rows`, then `(rounds, best score, split time, fit rows, tail rows)`.
fn tail_stopping(
    params: &TrainingParams,
    data: &DMatrix,
    tail: &PurgedTail<'_>,
    rows: &[usize],
) -> (usize, f64, i64, usize, usize) {
    let (split_at, split) = tail.split(rows).unwrap();
    let dfit = data.select_rows(&split.train).unwrap();
    let dtail = data.select_rows(&split.test).unwrap();
    let trial = Trainer::new(params, &dfit, 40)
        .eval(&dtail, "early_stopping")
        .early_stopping_rounds(NonZeroUsize::new(3).unwrap())
        .train()
        .unwrap();
    let rounds = trial.model.best_iteration().unwrap() + 1;
    let best = trial.best_score.unwrap();
    (rounds, best, split_at, split.train.len(), split.test.len())
}

/// Per-fold stopping sizes each purged forward fold on the purged tail of
/// its own training rows and retrains it on all of them: its round count,
/// scores, and test margins are those of the folds trained by hand, bit for
/// bit, and so is the refit, sized on the tail of every row.
#[test]
fn per_fold_stopping_sizes_each_fold_on_its_own_tail() {
    let data = regression();
    let params = params();
    let (decision_at, label_end) = schedule();
    let folds = || Fold::purged_forward(&decision_at, &label_end, 0.4, 3, 20).unwrap();
    let tail = PurgedTail::new(&decision_at, &label_end, 0.3).unwrap();
    let stopping = CvStopping::per_fold(NonZeroUsize::new(3).unwrap(), tail);
    let refit = CrossValidation::new(&params, &data, 40, folds())
        .early_stopping(stopping)
        .refit()
        .unwrap();
    let report = &refit.report;
    assert!(report.results.is_empty());
    let mut counts = Vec::new();
    for (fold, got) in folds().iter().zip(&report.folds) {
        let (rounds, best, split_at, fit_rows, tail_rows) =
            tail_stopping(&params, &data, &tail, &fold.train);
        let (model, scores) = trained(&params, &data, &fold.train, &fold.test, rounds);
        let dtest = data.select_rows(&fold.test).unwrap();
        assert_eq!(got.test, fold.test);
        assert_eq!(got.num_boost_round, rounds);
        assert_eq!(got.scores, scores);
        assert_eq!(
            got.margins,
            model.predict_margin(&dtest, Iterations::Best).unwrap()
        );
        let tail_stop = got.stopping.unwrap();
        assert_eq!(
            (tail_stop.split_at, tail_stop.fit_rows, tail_stop.tail_rows),
            (split_at, fit_rows, tail_rows)
        );
        assert_eq!(tail_stop.best_score, best);
        counts.push(rounds);
    }
    assert!(
        counts.iter().any(|&rounds| rounds != counts[0]),
        "{counts:?}"
    );

    let all: Vec<usize> = (0..ROWS).collect();
    let (rounds, ..) = tail_stopping(&params, &data, &tail, &all);
    assert_eq!(refit.num_boost_round, rounds);
    assert!(refit.stopping.is_some());
    let expected = Trainer::new(&params, &data, rounds).train().unwrap().model;
    assert_same_model(&refit.model, &expected, &data);
}

/// Per-fold stopping cuts each fold's tail as its `PurgedTail` says: on
/// rows whose density changes, a row-count tail sizes each fold on its
/// row-count split, which is not the span cut's.
#[test]
fn per_fold_stopping_honors_the_row_count_cut() {
    let data = regression();
    let params = params();
    // Half the rows ten minutes apart, then the rest one minute apart.
    let decision_at: Vec<i64> = (0..ROWS as i64)
        .map(|i| {
            if i < 75 {
                i * 600
            } else {
                75 * 600 + (i - 75) * 60
            }
        })
        .collect();
    let label_end: Vec<i64> = decision_at.iter().map(|at| at + 180).collect();
    let folds = || Fold::purged_forward(&decision_at, &label_end, 0.4, 3, 20).unwrap();
    let tail = PurgedTail::with_cut(&decision_at, &label_end, 0.3, TailCut::Rows).unwrap();
    let span = PurgedTail::new(&decision_at, &label_end, 0.3).unwrap();
    let report = CrossValidation::new(&params, &data, 40, folds())
        .early_stopping(CvStopping::per_fold(NonZeroUsize::new(3).unwrap(), tail))
        .run()
        .unwrap();
    let mut differs = false;
    for (fold, got) in folds().iter().zip(&report.folds) {
        let (rounds, best, split_at, fit_rows, tail_rows) =
            tail_stopping(&params, &data, &tail, &fold.train);
        assert_eq!(got.num_boost_round, rounds);
        let tail_stop = got.stopping.unwrap();
        assert_eq!(
            (tail_stop.split_at, tail_stop.fit_rows, tail_stop.tail_rows),
            (split_at, fit_rows, tail_rows)
        );
        assert_eq!(tail_stop.best_score, best);
        differs |= span.split(&fold.train).unwrap().0 != split_at;
    }
    assert!(differs);
}

/// Under fold-mean stopping every fold's margins are its model through the
/// chosen round, which every fold reports; without stopping, through every
/// round. Shuffled k-fold holds every row out once, so every row has an
/// out-of-fold margin, its fold's.
#[test]
fn fold_mean_margins_are_each_fold_through_the_chosen_round() {
    let data = regression();
    let params = params();
    for stopping in [
        Some(CvStopping::fold_mean(NonZeroUsize::new(4).unwrap())),
        None,
    ] {
        let mut cv = CrossValidation::new(&params, &data, 30, folds());
        if let Some(stopping) = stopping {
            cv = cv.early_stopping(stopping);
        }
        let report = cv.run().unwrap();
        let rounds = report.results[0].rounds.len();
        assert_eq!(rounds < 30, stopping.is_some(), "{rounds} rounds");
        let oof = report.out_of_fold().unwrap();
        for (f, (fold, got)) in folds().iter().zip(&report.folds).enumerate() {
            let (model, scores) = trained(&params, &data, &fold.train, &fold.test, rounds);
            let dtest = data.select_rows(&fold.test).unwrap();
            let margins = model.predict_margin(&dtest, Iterations::Best).unwrap();
            assert_eq!(got.num_boost_round, rounds);
            assert_eq!(got.scores, scores);
            assert_eq!(got.margins, margins);
            assert!(got.stopping.is_none());
            for (&row, margin) in fold.test.iter().zip(margins.as_slice()) {
                assert_eq!(oof.fold[row], Some(f));
                assert_eq!(oof.margins.as_slice()[row].to_bits(), margin.to_bits());
            }
        }
        assert!(oof.fold.iter().all(Option::is_some));
    }
}

/// Rows no fold holds out have no out-of-fold margin; a row two folds hold
/// out has none either and is refused, while one fold listing a row twice
/// is not.
#[test]
fn out_of_fold_margins_account_for_every_row() {
    let data = regression();
    let params = params();
    let run = |folds: Vec<Fold>| {
        CrossValidation::new(&params, &data, 3, folds)
            .run()
            .unwrap()
    };
    let train: Vec<usize> = (0..100).collect();
    let report = run(vec![
        Fold::new(train.clone(), vec![100, 101, 101]),
        Fold::new(train.clone(), vec![102]),
    ]);
    let oof = report.out_of_fold().unwrap();
    assert_eq!(oof.margins.n_rows(), ROWS);
    for row in 0..ROWS {
        let expected = match row {
            100 | 101 => Some(0),
            102 => Some(1),
            _ => None,
        };
        assert_eq!(oof.fold[row], expected, "row {row}");
        assert_eq!(oof.margins.as_slice()[row].is_nan(), expected.is_none());
    }
    let overlapping = run(vec![
        Fold::new(train.clone(), vec![100, 101]),
        Fold::new(train, vec![101]),
    ]);
    assert_eq!(invalid_param(overlapping.out_of_fold()), "folds");
    assert_eq!(overlapping.folds[1].margins.n_rows(), 1);

    // An edited report is refused, not indexed out of bounds.
    let mut edited = report.clone();
    edited.folds[0].test[0] = usize::MAX;
    assert_eq!(invalid_param(edited.out_of_fold()), "folds");
    let mut edited = report.clone();
    edited.folds[0].test.pop();
    assert!(matches!(
        edited.out_of_fold(),
        Err(HessboostError::DimensionMismatch { .. })
    ));
    let mut edited = report.clone();
    let two_outputs = TrainingParams::builder()
        .objective(Objective::Quantile(Quantiles::new(vec![0.2, 0.8]).unwrap()))
        .build()
        .unwrap();
    edited.folds[1].margins = hessboost::training::train(&two_outputs, &data, 1)
        .unwrap()
        .predict_margin(&data.select_rows(&[102]).unwrap(), Iterations::Best)
        .unwrap();
    assert!(matches!(
        edited.out_of_fold(),
        Err(HessboostError::DimensionMismatch { .. })
    ));
}

/// Per-fold stopping needs times for every row and a tail split with rows
/// on both sides for every fold.
#[test]
fn per_fold_stopping_refuses_unusable_tails() {
    let data = regression();
    let params = params();
    let (decision_at, label_end) = schedule();
    let patience = NonZeroUsize::new(3).unwrap();
    let short = PurgedTail::new(&decision_at[1..], &label_end[1..], 0.3).unwrap();
    assert!(matches!(
        CrossValidation::new(&params, &data, 5, folds())
            .early_stopping(CvStopping::per_fold(patience, short))
            .run(),
        Err(HessboostError::DimensionMismatch { expected: ROWS, got, .. }) if got == ROWS - 1
    ));
    // Two training rows decided together: one decision time, no fit rows.
    let tail = PurgedTail::new(&decision_at, &label_end, 0.3).unwrap();
    let mut together = decision_at.clone();
    together[1] = together[0];
    let same_time = PurgedTail::new(&together, &label_end, 0.3).unwrap();
    for (tail, train) in [(tail, vec![0, 1]), (same_time, vec![0, 1])] {
        let cv = CrossValidation::new(&params, &data, 5, vec![Fold::new(train, vec![5])])
            .early_stopping(CvStopping::per_fold(patience, tail));
        assert_eq!(invalid_param(cv.run()), "purged tail");
    }
}

/// Under per-fold stopping with target statistics, the trial's encoder is
/// fitted on the tail split's fit rows only, and the fold's on all its
/// training rows: bit for bit the folds encoded and trained by hand.
#[test]
fn per_fold_stopping_encodes_the_trial_on_its_fit_rows() {
    let data = categorical(1);
    let params = params();
    let encoder = OrderedTargetEncoder::builder().seed(4).build().unwrap();
    let (decision_at, label_end) = schedule();
    let folds = || Fold::purged_forward(&decision_at, &label_end, 0.4, 2, 20).unwrap();
    let tail = PurgedTail::new(&decision_at, &label_end, 0.3).unwrap();
    let patience = NonZeroUsize::new(3).unwrap();
    let report = CrossValidation::new(&params, &data, 40, folds())
        .early_stopping(CvStopping::per_fold(patience, tail))
        .target_stats(encoder.clone(), vec![0])
        .run()
        .unwrap();
    for (fold, got) in folds().iter().zip(&report.folds) {
        let (_, split) = tail.split(&fold.train).unwrap();
        let (dfit, fitted) = encoder
            .fit_transform(&data.select_rows(&split.train).unwrap(), &[0])
            .unwrap();
        let dtail = fitted
            .transform(&data.select_rows(&split.test).unwrap())
            .unwrap();
        let trial = Trainer::new(&params, &dfit, 40)
            .eval(&dtail, "early_stopping")
            .early_stopping_rounds(patience)
            .train()
            .unwrap();
        let rounds = trial.model.best_iteration().unwrap() + 1;
        let (dtrain, fitted) = encoder
            .fit_transform(&data.select_rows(&fold.train).unwrap(), &[0])
            .unwrap();
        let dtest = fitted
            .transform(&data.select_rows(&fold.test).unwrap())
            .unwrap();
        let model = Trainer::new(&params, &dtrain, rounds)
            .train()
            .unwrap()
            .model;
        assert_eq!(got.num_boost_round, rounds);
        assert_eq!(got.stopping.unwrap().best_score, trial.best_score.unwrap());
        assert_eq!(
            got.margins,
            model.predict_margin(&dtest, Iterations::Best).unwrap()
        );
    }
}

const DAY: i64 = 86_400;

/// Two rows a day for 10 days (rows `2d` and `2d + 1` decided on day `d`),
/// each label ending two days after its decision.
fn daily() -> (Vec<i64>, Vec<i64>) {
    let decision_at: Vec<i64> = (0..20).map(|i| (i / 2) * DAY).collect();
    let label_end = decision_at.iter().map(|at| at + 2 * DAY).collect();
    (decision_at, label_end)
}

/// [`daily`]'s rows decided on days `first` to `end - 1`.
fn days(first: usize, end: usize) -> Vec<usize> {
    (2 * first..2 * end).collect()
}

/// Each block tests the rows decided from its start to the next block's
/// (the last, to the end) and trains on the rows decided before it whose
/// labels end by its start, in row order. Rows decided before the first
/// start are never tested, and starts need not be decision times.
#[test]
fn purged_forward_at_tests_each_block_and_trains_before_it() {
    let (decision_at, label_end) = daily();
    let starts = [5 * DAY, 7 * DAY, 8 * DAY];
    let folds = Fold::purged_forward_at(&decision_at, &label_end, &starts, 1).unwrap();
    // Labels end two days later: a block starting on day d trains days 0 to
    // d - 2.
    assert_eq!(
        folds,
        [
            Fold::new(days(0, 4), days(5, 7)),
            Fold::new(days(0, 6), days(7, 8)),
            Fold::new(days(0, 7), days(8, 10)),
        ]
    );
    // Days 0 to 4 are never tested.
    let tested: Vec<usize> = folds
        .iter()
        .flat_map(|fold| fold.test.iter().copied())
        .collect();
    assert_eq!(tested, days(5, 10));
    // A start at noon on day 4 tests from day 5 and trains labels ending
    // by then: days 0 to 2.
    let noon = 4 * DAY + DAY / 2;
    let folds = Fold::purged_forward_at(&decision_at, &label_end, &[noon], 1).unwrap();
    assert_eq!(folds, [Fold::new(days(0, 3), days(5, 10))]);
    // Row order is irrelevant: reversed, row `i` is row `19 - i`.
    let reversed: Vec<i64> = decision_at.iter().rev().copied().collect();
    let reversed_ends: Vec<i64> = label_end.iter().rev().copied().collect();
    let folds = Fold::purged_forward_at(&reversed, &reversed_ends, &[8 * DAY], 1).unwrap();
    assert_eq!(folds, [Fold::new((6..20).collect(), (0..4).collect())]);
}

/// The purge keeps a label ending exactly at a block start and drops one
/// ending a tick later, from that block's fold alone.
#[test]
fn purged_forward_at_keeps_a_label_ending_at_the_block_start() {
    let (decision_at, mut label_end) = daily();
    let starts = [5 * DAY, 7 * DAY];
    // Row 7, decided on day 3, ends exactly at the first block's start.
    let folds = Fold::purged_forward_at(&decision_at, &label_end, &starts, 1).unwrap();
    assert_eq!(folds[0].train, days(0, 4));
    label_end[7] += 1;
    let folds = Fold::purged_forward_at(&decision_at, &label_end, &starts, 1).unwrap();
    assert_eq!(folds[0].train, [0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(folds[1].train, days(0, 6));
}

/// Given the starts `purged_forward` chooses (its folds' first test
/// decision times), `purged_forward_at` returns its folds, also when
/// `min_train` moved the start and rows share decision times.
#[test]
fn purged_forward_at_reproduces_purged_forward() {
    let (minutes, minute_ends) = schedule();
    // Six rows a day for 20 days, labels ending eight days later: 60
    // training rows move the start from day 16 to day 17.
    let six_a_day: Vec<i64> = (0..120).map(|i| (i / 6) * DAY).collect();
    let six_a_day_ends: Vec<i64> = six_a_day.iter().map(|at| at + 8 * DAY).collect();
    for (decision_at, label_end, fraction, blocks, min_train) in [
        (&minutes, &minute_ends, 0.4, 3, 20),
        (&six_a_day, &six_a_day_ends, 0.2, 1, 1),
        (&six_a_day, &six_a_day_ends, 0.2, 2, 60),
    ] {
        let forward =
            Fold::purged_forward(decision_at, label_end, fraction, blocks, min_train).unwrap();
        let starts: Vec<i64> = forward
            .iter()
            .map(|fold| fold.test.iter().map(|&row| decision_at[row]).min().unwrap())
            .collect();
        let at = Fold::purged_forward_at(decision_at, label_end, &starts, min_train).unwrap();
        assert_eq!(at, forward, "{starts:?}");
    }
}

/// Unusable times and blocks are refused, each case breaking one rule.
#[test]
fn purged_forward_at_refuses_unusable_blocks() {
    let (decision_at, label_end) = daily();
    let at = |starts: &[i64], min_train: usize| {
        Fold::purged_forward_at(&decision_at, &label_end, starts, min_train)
    };
    assert!(matches!(
        Fold::purged_forward_at(&decision_at, &label_end[1..], &[5 * DAY], 1),
        Err(HessboostError::DimensionMismatch {
            expected: 20,
            got: 19,
            ..
        })
    ));
    assert_eq!(
        invalid_param(Fold::purged_forward_at(&[], &[], &[0], 1)),
        "purged folds"
    );
    // Row 1's label ends before its decision; its block trains row 0.
    assert_eq!(
        invalid_param(Fold::purged_forward_at(&[5, 6], &[5, 5], &[6], 1)),
        "purged folds"
    );
    assert!(Fold::purged_forward_at(&[5, 6], &[5, 6], &[6], 1).is_ok());
    assert_eq!(invalid_param(at(&[], 1)), "purged folds");
    // A repeated or earlier start empties the block before it, but the same
    // starts deduplicated and sorted are accepted, so their order is at fault.
    for (starts, ordered) in [
        ([5 * DAY, 7 * DAY, 7 * DAY], &[5 * DAY, 7 * DAY][..]),
        (
            [5 * DAY, 7 * DAY, 6 * DAY],
            &[5 * DAY, 6 * DAY, 7 * DAY][..],
        ),
    ] {
        assert_eq!(invalid_param(at(&starts, 1)), "purged folds");
        assert!(at(ordered, 1).is_ok());
    }
    // Nothing is decided from a second after day 5 to day 6, nor after
    // day 9; every other block tests and trains rows.
    for starts in [
        [5 * DAY, 5 * DAY + 1, 6 * DAY],
        [5 * DAY, 6 * DAY, 10 * DAY],
    ] {
        assert_eq!(invalid_param(at(&starts, 1)), "purged folds");
    }
    // A block on day 5 trains days 0 to 3, eight rows; one on day 1 none,
    // which even a `min_train` of 0 refuses. The other block trains enough.
    let folds = at(&[5 * DAY], 8).unwrap();
    assert_eq!(folds[0].train.len(), 8);
    for (starts, min_train) in [([5 * DAY, 7 * DAY], 9), ([DAY, 5 * DAY], 0)] {
        assert_eq!(invalid_param(at(&starts, min_train)), "purged folds");
    }
}

/// Folds at given block starts cross-validate as `purged_forward`'s do:
/// per-fold stopping sizes each on the purged tail of its own training
/// rows, bit for bit the folds trained by hand.
#[test]
fn per_fold_stopping_sizes_folds_at_given_starts() {
    let data = regression();
    let params = params();
    let (decision_at, label_end) = schedule();
    // Blocks of 20 minutes from minute 90.
    let starts = [90 * 60, 110 * 60, 130 * 60];
    let folds = || Fold::purged_forward_at(&decision_at, &label_end, &starts, 20).unwrap();
    let tail = PurgedTail::new(&decision_at, &label_end, 0.3).unwrap();
    let patience = NonZeroUsize::new(3).unwrap();
    let report = CrossValidation::new(&params, &data, 40, folds())
        .early_stopping(CvStopping::per_fold(patience, tail))
        .run()
        .unwrap();
    assert_eq!(report.folds.len(), starts.len());
    for (fold, got) in folds().iter().zip(&report.folds) {
        let (rounds, best, split_at, fit_rows, tail_rows) =
            tail_stopping(&params, &data, &tail, &fold.train);
        let (model, scores) = trained(&params, &data, &fold.train, &fold.test, rounds);
        let dtest = data.select_rows(&fold.test).unwrap();
        assert_eq!(got.test, fold.test);
        assert_eq!(got.num_boost_round, rounds);
        assert_eq!(got.scores, scores);
        assert_eq!(
            got.margins,
            model.predict_margin(&dtest, Iterations::Best).unwrap()
        );
        let tail_stop = got.stopping.unwrap();
        assert_eq!(
            (tail_stop.split_at, tail_stop.fit_rows, tail_stop.tail_rows),
            (split_at, fit_rows, tail_rows)
        );
        assert_eq!(tail_stop.best_score, best);
    }
}

/// Boulevard fits do not early-stop, so per-fold stopping refuses them
/// before training.
#[test]
fn per_fold_stopping_refuses_boulevard() {
    use hessboost::config::{BoosterKind, Boulevard};
    let data = regression();
    let params = TrainingParams::builder()
        .booster(BoosterKind::Boulevard(
            Boulevard::builder().build().unwrap(),
        ))
        .eta(0.8)
        .subsample(0.8)
        .build()
        .unwrap();
    let (decision_at, label_end) = schedule();
    let tail = PurgedTail::new(&decision_at, &label_end, 0.3).unwrap();
    let cv = CrossValidation::new(&params, &data, 5, folds())
        .early_stopping(CvStopping::per_fold(NonZeroUsize::new(2).unwrap(), tail));
    assert_eq!(invalid_param(cv.run()), "early_stopping");
}

/// DART rescales earlier trees every round, so a fold stopped early on the
/// fold means is the fold retrained for the chosen rounds, not a prefix of
/// its full run: its margins equal that retrained fold's bit for bit.
#[test]
fn fold_mean_margins_of_dart_are_the_retrained_folds() {
    use hessboost::config::{BoosterKind, Dart};
    let data = regression();
    let params = TrainingParams::builder()
        .booster(BoosterKind::Dart(
            Dart::builder().rate_drop(0.3).build().unwrap(),
        ))
        .max_depth(6)
        .eta(0.8)
        .subsample(0.8)
        .seed(3)
        .build()
        .unwrap();
    let report = CrossValidation::new(&params, &data, 30, folds())
        .early_stopping(CvStopping::fold_mean(NonZeroUsize::new(3).unwrap()))
        .run()
        .unwrap();
    let rounds = report.results[0].rounds.len();
    assert!(rounds < 30, "stopped after {rounds} rounds");
    for (fold, got) in folds().iter().zip(&report.folds) {
        let (model, scores) = trained(&params, &data, &fold.train, &fold.test, rounds);
        let dtest = data.select_rows(&fold.test).unwrap();
        assert_eq!(got.num_boost_round, rounds);
        assert_eq!(got.scores, scores);
        assert_eq!(
            got.margins,
            model.predict_margin(&dtest, Iterations::Best).unwrap()
        );
    }
}

/// A history-only run reports the full run's fold-mean history without
/// any fold (so a DART fold stopped early is not retrained for margins),
/// and its refit is the full refit's model. Per-fold stopping, which has no
/// fold-mean history, is refused.
#[test]
fn history_only_reports_the_fold_mean_history_alone() {
    use hessboost::config::{BoosterKind, Dart};
    let data = regression();
    let params = TrainingParams::builder()
        .booster(BoosterKind::Dart(
            Dart::builder().rate_drop(0.3).build().unwrap(),
        ))
        .max_depth(6)
        .eta(0.8)
        .subsample(0.8)
        .seed(3)
        .build()
        .unwrap();
    let stopping = CvStopping::fold_mean(NonZeroUsize::new(3).unwrap());
    let cv = || CrossValidation::new(&params, &data, 30, folds()).early_stopping(stopping);
    let full = cv().refit().unwrap();
    let history = cv().history_only().refit().unwrap();
    assert!(full.num_boost_round < 30, "{} rounds", full.num_boost_round);
    assert!(history.report.folds.is_empty());
    assert_eq!(history.report.metrics, full.report.metrics);
    for (got, expected) in history.report.results.iter().zip(&full.report.results) {
        assert_eq!(got.rounds, expected.rounds);
    }
    assert_eq!(history.num_boost_round, full.num_boost_round);
    assert_same_model(&history.model, &full.model, &data);

    let (decision_at, label_end) = schedule();
    let tail = PurgedTail::new(&decision_at, &label_end, 0.3).unwrap();
    let per_fold = CrossValidation::new(&params, &data, 5, folds())
        .early_stopping(CvStopping::per_fold(NonZeroUsize::new(2).unwrap(), tail))
        .history_only();
    assert_eq!(invalid_param(per_fold.run()), "early_stopping");
}
