//! `CrossValidation`: refitting on every row, continuing a model in every
//! fold, and per-fold target statistics over a separate per-row target.

use hessboost::data::FeatureType;
use hessboost::data::target_stats::OrderedTargetEncoder;
use hessboost::model::{Iterations, ModelFormat};
use hessboost::prelude::{BoostedModel, DMatrix, HessboostError, Trainer, TrainingParams};
use hessboost::training::{CrossValidation, CvResult, Fold};
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
        .early_stopping_rounds(NonZeroUsize::new(4).unwrap())
        .refit()
        .unwrap();
    let rounds = refit.num_boost_round;
    assert!((1..56).contains(&rounds), "stopped after {rounds}");
    assert_eq!(refit.results[0].rounds.len(), rounds);
    let run = CrossValidation::new(&params, &data, 60, folds())
        .early_stopping_rounds(NonZeroUsize::new(4).unwrap())
        .run()
        .unwrap();
    assert_eq!(refit.results[0].rounds, run[0].rounds);
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
        .early_stopping_rounds(NonZeroUsize::new(3).unwrap())
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
    assert_eq!(means(&refit.results[0]), fold_means(&per_fold));
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
    assert_eq!(means(&refit.results[0]), fold_means(&per_fold));
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
