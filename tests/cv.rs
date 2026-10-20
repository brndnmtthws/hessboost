//! `CrossValidation`: refitting on every row, continuing a model in every
//! fold, per-fold target statistics over a separate per-row target, each
//! fold's margins and round count, and per-fold early stopping.

use hessboost::data::FeatureType;
use hessboost::data::target_stats::OrderedTargetEncoder;
use hessboost::model::{Iterations, ModelFormat};
use hessboost::prelude::{BoostedModel, DMatrix, HessboostError, Trainer, TrainingParams};
use hessboost::training::{CrossValidation, CvResult, CvStopping, Fold, PurgedTail};
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
        assert_eq!(invalid_param(cv.run()), "early_stopping");
    }
}
