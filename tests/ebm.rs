//! `booster = ebm`: cyclic GA²M boosting, its shape functions
//! ([`hessboost::ebm`]), and the Boulevard EBM's bands.

use hessboost::config::{BoosterKind, GrowPolicy, TrainingParams, TrainingParamsBuilder};
use hessboost::data::FeatureType;
use hessboost::ebm::shape_functions;
use hessboost::inference::{EbmInference, KernelSolver, NoiseVariance, honest_refit};
use hessboost::prelude::*;
use std::ops::ControlFlow;

mod common;
use common::{invalid_param, labeled_dense, lcg, with_threads};

/// `n` rows of three uniform features (the third missing every seventh
/// row, which then adds `½`) with
/// `y = sin(6 x0) + (x1 − ½)² + 1[x2 > ½] + x0 x1 + noise`.
fn data(n: usize, seed: u64) -> (Vec<f32>, DMatrix) {
    let mut next = lcg(seed);
    let mut x = Vec::with_capacity(3 * n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let (a, b, c) = (next(), next(), next());
        let c = if i % 7 == 0 { f32::NAN } else { c };
        x.extend_from_slice(&[a, b, c]);
        let step = if c.is_nan() {
            0.5
        } else if c > 0.5 {
            1.0
        } else {
            0.0
        };
        y.push((6.0 * a).sin() + (b - 0.5).powi(2) + step + a * b + 0.2 * (next() - 0.5));
    }
    let d = labeled_dense(&x, 3, &y);
    (x, d)
}

fn classic() -> TrainingParamsBuilder {
    TrainingParams::builder()
        .booster(BoosterKind::Ebm)
        .eta(0.1)
        .subsample(0.8)
        .grow_policy(GrowPolicy::LossGuide)
        .max_leaves(3)
        .ebm_interactions(1)
        .ebm_outer_bags(3)
        .ebm_bag_fraction(0.85)
}

fn boulevard() -> TrainingParamsBuilder {
    TrainingParams::builder()
        .booster(BoosterKind::Ebm)
        .ebm_boulevard(true)
        .eta(0.5)
        .subsample(0.8)
        .grow_policy(GrowPolicy::LossGuide)
        .max_leaves(8)
        .min_child_weight(10.0)
        .ebm_interactions(1)
}

#[test]
fn shape_functions_add_up_to_the_prediction() {
    let (x, dtrain) = data(600, 1);
    for params in [classic(), boulevard()] {
        let model = train(&params.build().unwrap(), &dtrain, 40).unwrap();
        let shapes = shape_functions(&model).unwrap();
        // One main term per feature, then the pair FAST picked: the
        // simulated interaction is between features 0 and 1.
        let features: Vec<Vec<u32>> = shapes.terms.iter().map(|t| t.features.clone()).collect();
        assert_eq!(features, [vec![0], vec![1], vec![2], vec![0, 1]]);
        let preds = model.predict(&dtrain).unwrap();
        for (row, &p) in x.chunks(3).zip(&preds) {
            let margin: f64 = shapes.intercept
                + shapes
                    .terms
                    .iter()
                    .map(|t| {
                        let v: Vec<f32> = t.features.iter().map(|&f| row[f as usize]).collect();
                        t.value(&v).unwrap()
                    })
                    .sum::<f64>();
            assert!((margin - f64::from(p)).abs() < 1e-4, "{margin} vs {p}");
        }
        // The missing cell of feature 2 (half the step) is its own value,
        // between the two sides.
        let step = &shapes.terms[2];
        let missing = step.value(&[f32::NAN]).unwrap();
        let (low, high) = (step.value(&[0.1]).unwrap(), step.value(&[0.9]).unwrap());
        assert!(low < missing && missing < high, "{low} {missing} {high}");
    }
}

#[test]
fn the_ebm_record_survives_the_native_formats_but_not_slicing() {
    let (_, dtrain) = data(300, 2);
    let model = train(&classic().build().unwrap(), &dtrain, 10).unwrap();
    let info = model.ebm().cloned().expect("an EBM");
    for m in [
        BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap(),
        BoostedModel::from_json(&model.to_json().unwrap()).unwrap(),
    ] {
        assert_eq!(m.ebm(), Some(&info));
        assert_eq!(
            shape_functions(&m).unwrap(),
            shape_functions(&model).unwrap()
        );
    }
    assert!(model.slice(..5, 1).unwrap().ebm().is_none());
    let continued = Trainer::new(&classic().build().unwrap(), &dtrain, 1)
        .init_model(&model)
        .train();
    assert_eq!(invalid_param(continued), "init_model");
}

#[test]
fn training_and_bands_are_identical_across_thread_counts() {
    let run = || {
        let (_, dtrain) = data(300, 3);
        let bagged = train(&classic().build().unwrap(), &dtrain, 15).unwrap();
        let model = train(&boulevard().build().unwrap(), &dtrain, 15).unwrap();
        let inference = EbmInference::fit(
            &model,
            &dtrain,
            NoiseVariance::TrainingResiduals,
            KernelSolver::Exact,
        )
        .unwrap();
        (
            bagged.to_bytes().unwrap(),
            model.to_bytes().unwrap(),
            inference.term_bands(0, 0.1).unwrap(),
            inference.term_bands(3, 0.1).unwrap(),
            inference.confidence_intervals(&dtrain, 0.1).unwrap(),
        )
    };
    assert_eq!(with_threads(1, run), with_threads(4, run));
}

#[test]
fn bands_narrow_with_more_data_and_nystrom_on_every_row_is_exact() {
    let mean_se = |n: usize| {
        let (_, dtrain) = data(n, 4);
        let (_, values) = data(n, 5);
        let model = train(&boulevard().build().unwrap(), &dtrain, 30).unwrap();
        let refit = honest_refit(&model, &values).unwrap();
        let fit = |solver| {
            EbmInference::fit(&refit, &values, NoiseVariance::Known(0.04), solver).unwrap()
        };
        let exact = fit(KernelSolver::Exact);
        let bands = exact.term_bands(1, 0.05).unwrap();
        if n == 200 {
            let nystrom = fit(KernelSolver::Nystrom {
                landmarks: n,
                seed: 3,
            });
            let se = nystrom.term_bands(1, 0.05).unwrap().standard_errors;
            for (e, s) in bands.standard_errors.iter().zip(&se) {
                assert!((e - s).abs() <= 1e-8 * e.max(1e-12), "{e} vs {s}");
            }
            // The pair term's errors run through the main stage too.
            let pair = exact.term_standard_errors(3, &values).unwrap();
            let pair_nystrom = nystrom.term_standard_errors(3, &values).unwrap();
            for (e, s) in pair.iter().zip(&pair_nystrom) {
                assert!((e - s).abs() <= 1e-8 * e.max(1e-12), "{e} vs {s}");
            }
        }
        bands.standard_errors.iter().sum::<f64>() / bands.standard_errors.len() as f64
    };
    let (small, large) = (mean_se(200), mean_se(1600));
    assert!(large < 0.7 * small, "{large} vs {small}");
}

#[test]
fn inference_needs_a_boulevard_ebm_and_its_training_rows() {
    let (_, dtrain) = data(300, 6);
    let solve = |model: &BoostedModel, rows: &DMatrix| {
        EbmInference::fit(model, rows, NoiseVariance::Known(1.0), KernelSolver::Exact).map(|_| ())
    };
    let bagged = train(&classic().build().unwrap(), &dtrain, 5).unwrap();
    assert_eq!(invalid_param(solve(&bagged, &dtrain)), "model");
    let model = train(&boulevard().subsample(1.0).build().unwrap(), &dtrain, 5).unwrap();
    let fewer = dtrain.select_rows(&(0..150).collect::<Vec<_>>()).unwrap();
    assert_eq!(invalid_param(solve(&model, &fewer)), "train");
    let inference = EbmInference::fit(
        &model,
        &dtrain,
        NoiseVariance::Known(1.0),
        KernelSolver::Exact,
    )
    .unwrap();
    assert_eq!(invalid_param(inference.term_bands(4, 0.1)), "term");
}

#[test]
fn unsupported_combinations_are_refused() {
    let refused = |b: TrainingParamsBuilder| match b.build() {
        Err(HessboostError::InvalidParameter { name, .. }) => name,
        other => panic!("expected a refusal, got {other:?}"),
    };
    assert_eq!(
        refused(TrainingParams::builder().ebm_interactions(2)),
        "ebm_interactions"
    );
    assert_eq!(refused(classic().colsample_bynode(0.5)), "colsample_bynode");
    assert_eq!(refused(classic().num_parallel_tree(2)), "num_parallel_tree");
    assert_eq!(
        refused(classic().interaction_constraints(vec![vec![0, 1]])),
        "interaction_constraints"
    );
    assert_eq!(
        refused(boulevard().objective("binary:logistic")),
        "objective"
    );
    assert_eq!(refused(boulevard().ebm_outer_bags(2)), "ebm_outer_bags");
    assert_eq!(refused(boulevard().base_score(0.5)), "base_score");
    assert_eq!(refused(boulevard().alpha(1.0)), "alpha");

    let (x, dtrain) = data(100, 7);
    let params = classic().build().unwrap();
    let evals = Trainer::new(&params, &dtrain, 5)
        .eval(&dtrain, "train")
        .train();
    assert_eq!(invalid_param(evals), "early_stopping_rounds");
    let too_many = classic().ebm_interactions(4).build().unwrap();
    assert_eq!(
        invalid_param(train(&too_many, &dtrain, 2)),
        "ebm_interactions"
    );
    let labels = dtrain.labels().unwrap().to_vec();
    let coded: Vec<f32> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 3 == 1 { (v * 3.0).floor() } else { v })
        .collect();
    let categorical = DMatrix::from_dense(&coded, 100, 3)
        .unwrap()
        .with_labels(&labels)
        .unwrap()
        .with_feature_types(&[
            FeatureType::Numerical,
            FeatureType::Categorical,
            FeatureType::Numerical,
        ])
        .unwrap();
    assert_eq!(
        invalid_param(train(&params, &categorical, 2)),
        "feature_types"
    );
    let weighted = dtrain.clone().with_weights(&[2.0; 100]).unwrap();
    assert_eq!(
        invalid_param(train(&boulevard().build().unwrap(), &weighted, 2)),
        "weights"
    );
}

#[test]
fn the_round_hook_sees_both_stages_and_stops_training() {
    let (_, dtrain) = data(300, 8);
    for (params, bags) in [(classic(), 3), (boulevard(), 1)] {
        let params = params.build().unwrap();
        let plain = train(&params, &dtrain, 6).unwrap();
        let mut seen = Vec::new();
        let observed = Trainer::new(&params, &dtrain, 6)
            .on_round(|round| {
                seen.push(round.iteration);
                ControlFlow::Continue(())
            })
            .train()
            .unwrap()
            .model;
        // Six main-effect rounds, then six pair rounds; observing changes
        // nothing.
        assert_eq!(seen, (0..12).collect::<Vec<_>>());
        assert_eq!(observed.to_bytes().unwrap(), plain.to_bytes().unwrap());
        let stop_at = |k: usize| {
            Trainer::new(&params, &dtrain, 6)
                .on_round(move |round| {
                    if round.iteration == k {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    }
                })
                .train()
                .unwrap()
                .model
        };
        // Stopped in the main-effect stage: three rounds of the three main
        // terms per bag, no pair term.
        let early = stop_at(2);
        assert_eq!(early.num_trees(), bags * 3 * 3);
        assert_eq!(shape_functions(&early).unwrap().terms.len(), 3);
        // Stopped in the pair stage: every main round, two pair rounds.
        let late = stop_at(7);
        assert_eq!(late.num_trees(), bags * (6 * 3 + 2));
        assert_eq!(shape_functions(&late).unwrap().terms.len(), 4);
        if params.ebm_boulevard {
            let inference = EbmInference::fit(
                &late,
                &dtrain,
                NoiseVariance::Known(0.04),
                KernelSolver::Exact,
            )
            .unwrap();
            assert!(inference.term_bands(3, 0.1).is_ok());
        }
    }
}
