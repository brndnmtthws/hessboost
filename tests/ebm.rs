//! `booster = ebm`: cyclic GA²M boosting, its shape functions
//! ([`hessboost::ebm`]), and the Boulevard EBM's bands.

use hessboost::config::{BoosterKind, GrowPolicy, TrainingParams, TrainingParamsBuilder};
use hessboost::data::FeatureType;
use hessboost::ebm::TermAxis;
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
        let features: Vec<Vec<u32>> = shapes.terms.iter().map(|t| t.features().to_vec()).collect();
        assert_eq!(features, [vec![0], vec![1], vec![2], vec![0, 1]]);
        let preds = model.predict(&dtrain).unwrap();
        for (row, &p) in x.chunks(3).zip(&preds) {
            let margin: f64 = shapes.intercept
                + shapes
                    .terms
                    .iter()
                    .map(|t| {
                        let v: Vec<f32> = t.features().iter().map(|&f| row[f as usize]).collect();
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

    assert_eq!(
        refused(boulevard().ebm_early_stopping_rounds(5)),
        "ebm_early_stopping_rounds"
    );
    assert_eq!(
        refused(classic().ebm_bag_fraction(1.0).ebm_early_stopping_rounds(5)),
        "ebm_early_stopping_rounds"
    );
    assert_eq!(
        refused(classic().ebm_early_stopping_tolerance(0.0)),
        "ebm_early_stopping_tolerance"
    );

    let (_, dtrain) = data(100, 7);
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
    // Classic EBMs start every margin at the intercept, so a base margin
    // would split training from prediction.
    let offset = dtrain.clone().with_base_margin(&[0.5; 100]).unwrap();
    assert_eq!(invalid_param(train(&params, &offset, 2)), "base_margin");
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

/// `n` rows of a five-category feature with effects `[−1, ½, 0, 2, −1½]`
/// (category 4 missing every eleventh row, which then adds 1) and a
/// uniform one with effect `sin(6 x)`, plus noise.
fn categorical_data(n: usize, seed: u64) -> (Vec<f32>, DMatrix) {
    const EFFECT: [f32; 5] = [-1.0, 0.5, 0.0, 2.0, -1.5];
    let mut next = lcg(seed);
    let mut x = Vec::with_capacity(2 * n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let c = ((next() * 5.0) as usize).min(4);
        let b = next();
        let (code, effect) = if i % 11 == 0 {
            (f32::NAN, 1.0)
        } else {
            (c as f32, EFFECT[c])
        };
        x.extend_from_slice(&[code, b]);
        y.push(effect + (6.0 * b).sin() + 0.2 * (next() - 0.5));
    }
    let d = DMatrix::from_dense(&x, n, 2)
        .unwrap()
        .with_labels(&y)
        .unwrap()
        .with_feature_types(&[FeatureType::Categorical, FeatureType::Numerical])
        .unwrap();
    (x, d)
}

#[test]
fn categorical_shapes_recover_the_per_category_effects() {
    let (x, dtrain) = categorical_data(2000, 9);
    for params in [
        classic().ebm_interactions(0),
        boulevard().ebm_interactions(0),
    ] {
        let model = train(&params.build().unwrap(), &dtrain, 60).unwrap();
        let shapes = shape_functions(&model).unwrap();
        let shape = &shapes.terms[0];
        let TermAxis::Categorical { categories, .. } = &shape.axes()[0] else {
            panic!("expected a categorical axis, got {:?}", shape.axes()[0]);
        };
        // The categories some split sends left; one no split names (here
        // possibly the largest effect, always sent right) shares the
        // "other" cell.
        assert!(categories.len() >= 4 && categories.iter().all(|&c| c < 5));
        // Every category (and missing) against category 2, whose effect is
        // 0: the differences are the simulated effects.
        let zero = shape.value(&[2.0]).unwrap();
        for (code, effect) in [
            (0.0, -1.0),
            (1.0, 0.5),
            (3.0, 2.0),
            (4.0, -1.5),
            (f32::NAN, 1.0),
        ] {
            let got = shape.value(&[code]).unwrap() - zero;
            assert!(
                (got - effect).abs() < 0.2,
                "category {code}: {got} vs {effect}"
            );
        }
        let preds = model.predict(&dtrain).unwrap();
        for (row, &p) in x.chunks(2).zip(&preds) {
            let margin = shapes.intercept
                + shapes.terms[0].value(&row[..1]).unwrap()
                + shapes.terms[1].value(&row[1..]).unwrap();
            assert!((margin - f64::from(p)).abs() < 1e-4, "{margin} vs {p}");
        }
    }
    // The Boulevard EBM's bands cover categorical terms too.
    let model = train(&boulevard().build().unwrap(), &dtrain, 20).unwrap();
    let inference = EbmInference::fit(
        &model,
        &dtrain,
        NoiseVariance::Known(0.01),
        KernelSolver::Exact,
    )
    .unwrap();
    let bands = inference.term_bands(0, 0.05).unwrap();
    assert!(
        bands
            .standard_errors
            .iter()
            .all(|s| s.is_finite() && *s > 0.0)
    );
}

#[test]
fn categorical_shapes_reconstruct_the_margins_for_codes_past_2_pow_24() {
    // Codes 2^24 + 2c (all exact in f32): "one past the largest" would
    // round onto a listed code there.
    let (x, _) = categorical_data(1500, 12);
    let shifted: Vec<f32> = x
        .chunks(2)
        .flat_map(|r| {
            let code = if r[0].is_nan() {
                f32::NAN
            } else {
                16_777_216.0 + 2.0 * r[0]
            };
            [code, r[1]]
        })
        .collect();
    let (_, labelled) = categorical_data(1500, 12);
    let dtrain = DMatrix::from_dense(&shifted, 1500, 2)
        .unwrap()
        .with_labels(labelled.labels().unwrap())
        .unwrap()
        .with_feature_types(&[FeatureType::Categorical, FeatureType::Numerical])
        .unwrap();
    let model = train(&classic().ebm_interactions(0).build().unwrap(), &dtrain, 40).unwrap();
    let shapes = shape_functions(&model).unwrap();
    let preds = model.predict(&dtrain).unwrap();
    for (row, &p) in shifted.chunks(2).zip(&preds) {
        let margin = shapes.intercept
            + shapes.terms[0].value(&row[..1]).unwrap()
            + shapes.terms[1].value(&row[1..]).unwrap();
        assert!((margin - f64::from(p)).abs() < 1e-4, "{margin} vs {p}");
    }
}

#[test]
fn shape_lookups_refuse_or_absorb_malformed_points() {
    let (_, dtrain) = categorical_data(300, 10);
    let model = train(&classic().ebm_interactions(1).build().unwrap(), &dtrain, 10).unwrap();
    let shapes = shape_functions(&model).unwrap();
    let (categorical, pair) = (&shapes.terms[0], &shapes.terms[2]);
    for bad in [&[][..], &[1.0, 2.0, 3.0][..]] {
        assert!(matches!(
            categorical.value(bad),
            Err(HessboostError::DimensionMismatch { .. })
        ));
        assert!(pair.value(bad).is_err());
    }
    // An unknown category is the "other" cell, whatever its code; NaN is
    // missing; infinities are ordinary values.
    let other = categorical.cell(&[99.0]).unwrap();
    assert_eq!(other, categorical.axes()[0].cells() - 2);
    for odd in [-3.0, 1e30, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 2.5] {
        assert!(categorical.value(&[odd]).is_ok());
        assert!(pair.value(&[odd, odd]).is_ok());
    }
}

#[test]
fn early_stopping_ends_every_bag_at_its_best_round() {
    let (_, dtrain) = data(600, 11);
    let params = classic()
        .eta(0.3)
        .ebm_early_stopping_rounds(5)
        .build()
        .unwrap();
    let rounds = |max: usize| {
        let mut seen = 0;
        let model = Trainer::new(&params, &dtrain, max)
            .on_round(|_| {
                seen += 1;
                ControlFlow::Continue(())
            })
            .train()
            .unwrap()
            .model;
        (model, seen)
    };
    let (stopped, seen) = rounds(2000);
    // Every bag stopped long before the limit, and a larger limit changes
    // nothing: the stopping rule, not the round count, ended training.
    assert!(seen < 2 * 2000, "{seen}");
    assert!(
        stopped.num_trees() < 3 * seen * 3,
        "{}",
        stopped.num_trees()
    );
    let (again, _) = rounds(4000);
    assert_eq!(again.to_bytes().unwrap(), stopped.to_bytes().unwrap());
    assert_eq!(shape_functions(&stopped).unwrap().terms.len(), 4);
}

#[test]
fn a_boulevard_ebm_with_a_broken_round_layout_does_not_load() {
    let (_, dtrain) = data(200, 12);
    let model = train(
        &boulevard().ebm_interactions(0).build().unwrap(),
        &dtrain,
        3,
    )
    .unwrap();
    let mut json: serde_json::Value = serde_json::from_str(&model.to_json().unwrap()).unwrap();
    assert!(BoostedModel::from_json(&json.to_string()).is_ok());
    // One more main-effect tree: the stage no longer holds whole rounds.
    let first = json["trees"][0].clone();
    json["trees"].as_array_mut().unwrap().push(first);
    if let Some(weights) = json["tree_weights"].as_array_mut()
        && !weights.is_empty()
    {
        weights.push(serde_json::json!(1.0));
    }
    json["ebm"]["tree_terms"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!(0));
    assert!(matches!(
        BoostedModel::from_json(&json.to_string()),
        Err(HessboostError::ModelFormat(_))
    ));
}
