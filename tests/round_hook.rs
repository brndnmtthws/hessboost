//! `Trainer::on_round`: the per-round hook sees every round in order, never
//! changes the model, and a `Break` ends training with the rounds so far.

use hessboost::config::{BoosterKind, ProcessType};
use hessboost::prelude::{BoostedModel, DMatrix, Trainer, TrainingParams, train};
use hessboost::training::RoundEval;
use std::ops::ControlFlow;

mod common;
use common::labeled_dense;

/// `n` rows of 4 features with a noisy smooth target (`salt` varies the
/// noise), so a validation set eventually stops improving.
fn data(n: usize, salt: usize) -> DMatrix {
    let mut x = Vec::with_capacity(n * 4);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let f: Vec<f32> = (0..4)
            .map(|j| ((i * (7 + 3 * j) + 11 * j) % 97) as f32 / 97.0)
            .collect();
        let noise = (((i + salt) * 2_654_435_761) % 1000) as f32 / 1000.0 - 0.5;
        y.push(2.0 * f[0] - 3.0 * f[1] * f[1] + 0.5 * f[2] + noise);
        x.extend(f);
    }
    labeled_dense(&x, 4, &y)
}

fn bytes(model: &BoostedModel) -> Vec<u8> {
    model.to_bytes().unwrap()
}

/// Row sampling and DART dropout: a hook must not disturb the RNG streams.
fn sampled() -> TrainingParams {
    TrainingParams::builder()
        .booster(BoosterKind::Dart)
        .rate_drop(0.2)
        .subsample(0.7)
        .colsample_bynode(0.8)
        .seed(9)
        .max_depth(3)
        .build()
        .unwrap()
}

fn stop_at(last: usize) -> impl FnMut(&RoundEval) -> ControlFlow<()> + Send {
    move |round| {
        if round.iteration == last {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

#[test]
fn a_continuing_hook_changes_nothing_and_sees_every_round_in_order() {
    let (dtrain, dvalid) = (data(300, 0), data(200, 1));
    let params = sampled();
    let plain = Trainer::new(&params, &dtrain, 12)
        .eval(&dvalid, "valid")
        .train()
        .unwrap();
    let mut seen = Vec::new();
    let hooked = Trainer::new(&params, &dtrain, 12)
        .eval(&dvalid, "valid")
        .on_round(|round| {
            seen.push((round.iteration, round.scores.clone()));
            ControlFlow::Continue(())
        })
        .train()
        .unwrap();
    assert_eq!(bytes(&hooked.model), bytes(&plain.model));
    let history: Vec<_> = plain
        .history
        .iter()
        .map(|round| (round.iteration, round.scores.clone()))
        .collect();
    assert_eq!(seen, history);
    // Without eval sets the hook still runs every round, with no scores.
    let mut iterations = Vec::new();
    let unscored = Trainer::new(&params, &dtrain, 5)
        .on_round(|round| {
            assert!(round.scores.is_empty());
            iterations.push(round.iteration);
            ControlFlow::Continue(())
        })
        .train()
        .unwrap();
    assert_eq!(iterations, [0, 1, 2, 3, 4]);
    assert!(unscored.history.is_empty());
}

#[test]
fn break_keeps_the_rounds_so_far_as_a_shorter_run_would() {
    let (dtrain, dvalid) = (data(300, 0), data(200, 1));
    let params = sampled();
    let stopped = Trainer::new(&params, &dtrain, 100)
        .eval(&dvalid, "valid")
        .on_round(stop_at(6))
        .train()
        .unwrap();
    assert_eq!(stopped.model.num_boost_rounds(), 7);
    assert_eq!(stopped.history.len(), 7);
    assert_eq!(stopped.model.best_iteration(), None);
    assert_eq!(stopped.best_score, None);
    let short = Trainer::new(&params, &dtrain, 7)
        .eval(&dvalid, "valid")
        .train()
        .unwrap();
    assert_eq!(bytes(&stopped.model), bytes(&short.model));
    // Continued training numbers rounds after the model's iterations.
    let mut seen = Vec::new();
    let continued = Trainer::new(&params, &dtrain, 10)
        .init_model(&short.model)
        .on_round(|round| {
            seen.push(round.iteration);
            ControlFlow::Break(())
        })
        .train()
        .unwrap();
    assert_eq!(seen, [7]);
    assert_eq!(continued.model.num_boost_rounds(), 8);
}

#[test]
fn break_under_early_stopping_records_the_best_round_so_far() {
    let (dtrain, dvalid) = (data(300, 0), data(200, 1));
    let params = TrainingParams::builder()
        .eta(0.5)
        .max_depth(4)
        .build()
        .unwrap();
    let full = Trainer::new(&params, &dtrain, 200)
        .eval(&dvalid, "valid")
        .early_stopping_rounds(3)
        .train()
        .unwrap();
    let best = full.model.best_iteration().unwrap();
    let last = full.model.num_boost_rounds() - 1;
    assert!(best >= 2 && last == best + 3, "best {best}, last {last}");
    // The hook sees the round on which patience runs out.
    let mut seen = Vec::new();
    let watched = Trainer::new(&params, &dtrain, 200)
        .eval(&dvalid, "valid")
        .early_stopping_rounds(3)
        .on_round(|round| {
            seen.push(round.iteration);
            ControlFlow::Continue(())
        })
        .train()
        .unwrap();
    assert_eq!(seen, (0..=last).collect::<Vec<_>>());
    assert_eq!(bytes(&watched.model), bytes(&full.model));
    // Breaking before patience runs out keeps the best round seen so far.
    let cut = best - 1;
    let stopped = Trainer::new(&params, &dtrain, 200)
        .eval(&dvalid, "valid")
        .early_stopping_rounds(3)
        .on_round(stop_at(cut))
        .train()
        .unwrap();
    assert_eq!(stopped.model.num_boost_rounds(), cut + 1);
    let scores: Vec<f64> = stopped
        .history
        .iter()
        .map(|round| round.scores[0].2)
        .collect();
    let best_so_far = (0..scores.len())
        .min_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .unwrap();
    assert_eq!(stopped.model.best_iteration(), Some(best_so_far));
    assert_eq!(stopped.best_score, Some(scores[best_so_far]));
}

#[test]
fn refresh_and_gblinear_stop_on_break_too() {
    let dtrain = data(300, 0);
    let base = TrainingParams::builder().max_depth(3).build().unwrap();
    let old = train(&base, &data(300, 5), 6).unwrap();
    let update = TrainingParams::builder()
        .max_depth(3)
        .process_type(ProcessType::Update)
        .build()
        .unwrap();
    let refreshed = Trainer::new(&update, &dtrain, 6)
        .init_model(&old)
        .on_round(stop_at(2))
        .train()
        .unwrap();
    let short = Trainer::new(&update, &dtrain, 3)
        .init_model(&old)
        .train()
        .unwrap();
    assert_eq!(refreshed.model.num_boost_rounds(), 3);
    assert_eq!(bytes(&refreshed.model), bytes(&short.model));

    let linear = TrainingParams::builder()
        .booster(BoosterKind::GbLinear)
        .build()
        .unwrap();
    let mut seen = Vec::new();
    let stopped = Trainer::new(&linear, &dtrain, 50)
        .on_round(|round| {
            seen.push(round.iteration);
            if round.iteration == 3 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .train()
        .unwrap();
    assert_eq!(seen, [0, 1, 2, 3]);
    let four = train(&linear, &dtrain, 4).unwrap();
    assert_eq!(bytes(&stopped.model), bytes(&four));
    let all = Trainer::new(&linear, &dtrain, 50)
        .on_round(|_| ControlFlow::Continue(()))
        .train()
        .unwrap();
    assert_eq!(
        bytes(&all.model),
        bytes(&train(&linear, &dtrain, 50).unwrap())
    );
}
