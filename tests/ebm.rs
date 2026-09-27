//! `booster = ebm`: cyclic GA²M boosting, its shape functions
//! ([`hessboost::ebm`]), and the Boulevard EBM's bands.

use hessboost::config::{
    BalancedBagging, BoosterKind, Ebm, EbmBuilder, EbmEarlyStopping, GrowPolicy, QueryBagging,
    TrainingParams, TrainingParamsBuilder,
};
use hessboost::data::FeatureType;
use hessboost::ebm::TermAxis;
use hessboost::ebm::shape_functions;
use hessboost::inference::{EbmInference, KernelSolver, NoiseVariance, honest_refit};
use hessboost::objective::{LambdaRank, Logistic, Objective};
use hessboost::prelude::*;
use std::ops::ControlFlow;

mod common;
use common::{invalid_param, labeled_dense, lcg, with_threads};

/// Early stopping after `rounds` rounds at the default tolerance.
fn stopping(rounds: usize) -> EbmEarlyStopping {
    EbmEarlyStopping::new(
        std::num::NonZeroUsize::new(rounds).unwrap(),
        EbmEarlyStopping::DEFAULT_TOLERANCE,
    )
    .unwrap()
}

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

/// The classic EBM settings of [`classic`].
fn classic_ebm() -> EbmBuilder {
    Ebm::builder()
        .interactions(1)
        .outer_bags(3)
        .bag_fraction(0.85)
}

/// A classic cyclic EBM with the EBM settings `ebm`.
fn classic_with(ebm: EbmBuilder) -> TrainingParamsBuilder {
    TrainingParams::builder()
        .booster(BoosterKind::Ebm(ebm.build().unwrap()))
        .eta(0.1)
        .subsample(0.8)
        .grow_policy(GrowPolicy::LossGuide)
        .max_leaves(3)
}

fn classic() -> TrainingParamsBuilder {
    classic_with(classic_ebm())
}

/// The Boulevard EBM settings of [`boulevard`].
fn boulevard_ebm() -> EbmBuilder {
    Ebm::builder().boulevard(true).interactions(1)
}

/// A Boulevard EBM with the EBM settings `ebm`.
fn boulevard_with(ebm: EbmBuilder) -> TrainingParamsBuilder {
    TrainingParams::builder()
        .booster(BoosterKind::Ebm(ebm.build().unwrap()))
        .eta(0.5)
        .subsample(0.8)
        .grow_policy(GrowPolicy::LossGuide)
        .max_leaves(8)
        .min_child_weight(10.0)
}

fn boulevard() -> TrainingParamsBuilder {
    boulevard_with(boulevard_ebm())
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
        for (row, &p) in x.chunks(3).zip(preds.as_slice()) {
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
            for (e, s) in pair.as_slice().iter().zip(pair_nystrom.as_slice()) {
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
    let serde_json::Value::Object(flat) = serde_json::json!({"ebm_interactions": 2}) else {
        unreachable!()
    };
    assert_eq!(
        invalid_param(TrainingParams::from_xgboost(flat)),
        "ebm_interactions"
    );
    assert_eq!(refused(classic().colsample_bynode(0.5)), "colsample_bynode");
    assert_eq!(refused(classic().num_parallel_tree(2)), "num_parallel_tree");
    assert_eq!(
        refused(classic().interaction_constraints(vec![vec![0, 1]])),
        "interaction_constraints"
    );
    assert_eq!(
        refused(boulevard().objective(Objective::SquaredLogError)),
        "objective"
    );
    assert_eq!(
        invalid_param(boulevard_ebm().outer_bags(2).build()),
        "ebm_outer_bags"
    );
    assert_eq!(refused(boulevard().base_score(0.5)), "base_score");
    assert_eq!(refused(boulevard().alpha(1.0)), "alpha");
    // Class-balanced bagging draws rows by their labels, which the Boulevard
    // EBM's kernel cannot represent; refused under its own key whatever the
    // objective (it needs a `binary:*` one, which `ebm_boulevard` refuses).
    let balanced = BalancedBagging::new(1.0, 0.2).unwrap();
    for objective in [
        Objective::SquaredError,
        Objective::BinaryLogistic(Logistic::default()),
    ] {
        let bagged = boulevard()
            .subsample(1.0)
            .objective(objective)
            .balanced_bagging(balanced);
        assert_eq!(refused(bagged), "pos_bagging_fraction");
    }
    // Query bagging needs a `rank:*` objective, which `ebm_boulevard`
    // refuses: classic EBMs only.
    let by_query = boulevard()
        .subsample(1.0)
        .objective(Objective::RankNdcg(LambdaRank::default()))
        .bagging_by_query(QueryBagging::new(0.5).unwrap());
    assert_eq!(refused(by_query), "objective");

    assert_eq!(
        invalid_param(boulevard_ebm().early_stopping(stopping(5)).build()),
        "ebm_early_stopping_rounds"
    );
    assert_eq!(
        invalid_param(
            classic_ebm()
                .bag_fraction(1.0)
                .early_stopping(stopping(5))
                .build()
        ),
        "ebm_early_stopping_rounds"
    );
    // No early stopping is `None`: the flat `0` rounds is off, and a
    // tolerance without early stopping is refused there.
    let flat =
        |pairs: &[(&str, serde_json::Value)]| TrainingParams::from_xgboost(pairs.iter().cloned());
    let ebm = serde_json::json!("ebm");
    let off = flat(&[
        ("booster", ebm.clone()),
        ("ebm_early_stopping_rounds", serde_json::json!(0)),
    ])
    .unwrap();
    assert_eq!(off, flat(&[("booster", ebm.clone())]).unwrap());
    // Even the default tolerance (which the `0`-rounds form used to accept)
    // is refused without early stopping, rounds absent or `0`.
    for tolerance in [0.0, EbmEarlyStopping::DEFAULT_TOLERANCE] {
        let tolerance = ("ebm_early_stopping_tolerance", serde_json::json!(tolerance));
        let zero = ("ebm_early_stopping_rounds", serde_json::json!(0));
        for pairs in [
            vec![("booster", ebm.clone()), tolerance.clone()],
            vec![("booster", ebm.clone()), zero, tolerance],
        ] {
            assert_eq!(invalid_param(flat(&pairs)), "ebm_early_stopping_tolerance");
        }
    }
    let on = flat(&[
        ("booster", ebm.clone()),
        ("ebm_bag_fraction", serde_json::json!(0.8)),
        ("ebm_early_stopping_rounds", serde_json::json!(7)),
        ("ebm_early_stopping_tolerance", serde_json::json!(0.01)),
    ])
    .unwrap();
    let BoosterKind::Ebm(settings) = on.booster else {
        panic!("an EBM booster");
    };
    let expected = EbmEarlyStopping::new(std::num::NonZeroUsize::new(7).unwrap(), 0.01).unwrap();
    assert_eq!(settings.early_stopping(), Some(expected));
    for params in [&off, &on] {
        assert_eq!(
            &TrainingParams::from_xgboost(params.to_xgboost().unwrap()).unwrap(),
            params
        );
    }
    assert_eq!(
        invalid_param(EbmEarlyStopping::new(
            std::num::NonZeroUsize::MIN,
            f64::INFINITY
        )),
        "ebm_early_stopping_tolerance"
    );

    let (_, dtrain) = data(100, 7);
    let params = classic().build().unwrap();
    let evals = Trainer::new(&params, &dtrain, 5)
        .eval(&dtrain, "train")
        .train();
    assert_eq!(invalid_param(evals), "early_stopping_rounds");
    let too_many = classic_with(classic_ebm().interactions(4)).build().unwrap();
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

/// A classic EBM draws each tree's rows from its outer bag by class under
/// class-balanced bagging: keeping a tenth of the negatives pulls every
/// tree toward the positives, so the predicted probabilities rise, and the
/// labels must be 0 or 1.
#[test]
fn classic_ebms_bag_rows_by_class() {
    let n = 600;
    let mut next = lcg(21);
    let mut x = Vec::with_capacity(2 * n);
    let mut labels = Vec::with_capacity(n);
    for _ in 0..n {
        let (a, b) = (next(), next());
        x.extend_from_slice(&[a, b]);
        let label = (6.0 * a).sin() + b + 0.5 * (next() - 0.5) > 0.5;
        labels.push(f32::from(u8::from(label)));
    }
    let dtrain = labeled_dense(&x, 2, &labels);
    let base = || {
        classic()
            .subsample(1.0)
            .objective(Objective::BinaryLogistic(Logistic::default()))
    };
    let bagged = base()
        .balanced_bagging(BalancedBagging::new(1.0, 0.1).unwrap())
        .build()
        .unwrap();
    let mean = |params: &TrainingParams| {
        let model = train(params, &dtrain, 40).unwrap();
        let preds = model.predict(&dtrain).unwrap();
        preds.as_slice().iter().map(|&p| f64::from(p)).sum::<f64>() / n as f64
    };
    let (plain, balanced) = (mean(&base().build().unwrap()), mean(&bagged));
    assert!(
        balanced > plain + 0.05,
        "mean probability {balanced} with balanced bagging vs {plain} without"
    );
    let soft = labeled_dense(&x, 2, &vec![0.5; n]);
    assert_eq!(invalid_param(train(&bagged, &soft, 2)), "labels");
}

/// A classic ranking EBM draws each tree's rows from its outer bag by query
/// under query bagging: the model changes (it used to be the unbagged one),
/// stays seed-deterministic, and needs query groups.
#[test]
fn classic_ebms_bag_whole_queries() {
    let (queries, size) = (40, 10);
    let n = queries * size;
    let mut next = lcg(23);
    let mut x = Vec::with_capacity(2 * n);
    let mut relevance = Vec::with_capacity(n);
    for _ in 0..n {
        let (a, b) = (next(), next());
        x.extend_from_slice(&[a, b]);
        relevance.push((3.0 * (a + 0.5 * b + 0.3 * next())).floor().min(3.0));
    }
    let dtrain = labeled_dense(&x, 2, &relevance)
        .with_group_sizes(&vec![size; queries])
        .unwrap();
    let base = || {
        classic()
            .subsample(1.0)
            .objective(Objective::RankNdcg(LambdaRank::default()))
    };
    let bagged = base()
        .bagging_by_query(QueryBagging::new(0.3).unwrap())
        .build()
        .unwrap();
    let margins = |params: &TrainingParams| {
        let model = train(params, &dtrain, 10).unwrap();
        model.predict_margin(&dtrain).unwrap().into_vec()
    };
    let with_queries = margins(&bagged);
    assert_ne!(with_queries, margins(&base().build().unwrap()));
    assert_eq!(with_queries, margins(&bagged));
    // Early stopping scores held-out rows, which would split the queries.
    let stopping = base()
        .booster(BoosterKind::Ebm(
            classic_ebm().early_stopping(stopping(5)).build().unwrap(),
        ))
        .build()
        .unwrap();
    assert_eq!(
        invalid_param(train(&stopping, &dtrain, 2)),
        "ebm_early_stopping_rounds"
    );
    let ungrouped = labeled_dense(&x, 2, &relevance);
    assert_eq!(
        invalid_param(train(&bagged, &ungrouped, 2)),
        "bagging_by_query"
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
        if matches!(params.booster, BoosterKind::Ebm(e) if e.boulevard()) {
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
        classic_with(classic_ebm().interactions(0)),
        boulevard_with(boulevard_ebm().interactions(0)),
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
        for (row, &p) in x.chunks(2).zip(preds.as_slice()) {
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
    let model = train(
        &classic_with(classic_ebm().interactions(0)).build().unwrap(),
        &dtrain,
        40,
    )
    .unwrap();
    let shapes = shape_functions(&model).unwrap();
    let preds = model.predict(&dtrain).unwrap();
    for (row, &p) in shifted.chunks(2).zip(preds.as_slice()) {
        let margin = shapes.intercept
            + shapes.terms[0].value(&row[..1]).unwrap()
            + shapes.terms[1].value(&row[1..]).unwrap();
        assert!((margin - f64::from(p)).abs() < 1e-4, "{margin} vs {p}");
    }
}

#[test]
fn shape_lookups_refuse_or_absorb_malformed_points() {
    let (_, dtrain) = categorical_data(300, 10);
    let model = train(
        &classic_with(classic_ebm().interactions(1)).build().unwrap(),
        &dtrain,
        10,
    )
    .unwrap();
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
fn early_stopping_scores_bags_with_the_custom_metric() {
    use hessboost::metric::CustomMetric;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let (_, dtrain) = data(400, 13);
    let params = classic_with(classic_ebm().early_stopping(stopping(3)))
        .eta(0.3)
        .build()
        .unwrap();
    let metric = CustomMetric::new("mae", false, |preds, labels, _| {
        CALLS.fetch_add(1, Ordering::Relaxed);
        preds
            .iter()
            .zip(labels)
            .map(|(p, y)| f64::from((p - y).abs()))
            .sum::<f64>()
            / preds.len() as f64
    });
    Trainer::new(&params, &dtrain, 200)
        .custom_metric(Box::new(metric))
        .train()
        .unwrap();
    assert!(CALLS.load(Ordering::Relaxed) > 0);
}

#[test]
fn an_oversized_early_stopping_patience_just_never_stops() {
    let (_, dtrain) = data(200, 14);
    let params = classic_with(classic_ebm().early_stopping(stopping(usize::MAX)))
        .build()
        .unwrap();
    let model = train(&params, &dtrain, 2).unwrap();
    assert!(model.num_trees() > 0);
}

#[test]
fn early_stopping_ends_every_bag_at_its_best_round() {
    let (_, dtrain) = data(600, 11);
    let params = classic_with(classic_ebm().early_stopping(stopping(5)))
        .eta(0.3)
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
        &boulevard_with(boulevard_ebm().interactions(0))
            .build()
            .unwrap(),
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

/// In-place updates regrow trees on every feature, which would break the
/// one-term-per-tree structure the shape functions read.
#[test]
fn online_updates_refuse_an_ebm() {
    use hessboost::training::online::{OnlineModel, OnlineParams};
    let (_, dtrain) = data(200, 16);
    let model = train(&classic().build().unwrap(), &dtrain, 3).unwrap();
    let gbtree = TrainingParams::builder()
        .tree_method(TreeMethod::Hist)
        .build()
        .unwrap();
    for online in [
        OnlineParams::approximate(0.1).unwrap(),
        OnlineParams::exact(),
    ] {
        assert_eq!(
            invalid_param(OnlineModel::from_model(
                model.clone(),
                &gbtree,
                &dtrain,
                online
            )),
            "model"
        );
    }
}

/// SGLB runs in the gbtree loop only, so EBMs refuse it; and an EBM's tree
/// prefixes are not the models of fewer rounds, so virtual ensembles of one
/// are refused.
#[test]
fn sglb_and_virtual_ensembles_are_refused() {
    use hessboost::config::{Langevin, ModelShrink, ModelShrinkMode};
    let refused = |b: TrainingParamsBuilder| invalid_param(b.build());
    let shrink = ModelShrink::new(0.01, ModelShrinkMode::Constant).unwrap();
    for ebm in [classic, boulevard] {
        assert_eq!(refused(ebm().langevin(Langevin::default())), "langevin");
        assert_eq!(refused(ebm().posterior_sampling(true)), "langevin");
        assert_eq!(refused(ebm().model_shrink(shrink)), "model_shrink_rate");
    }
    let (_, dtrain) = data(200, 17);
    let model = train(&classic().build().unwrap(), &dtrain, 10).unwrap();
    assert_eq!(
        invalid_param(model.predict_virtual_ensembles(&dtrain, 2)),
        "model"
    );
}

/// Each classic tree's gradients come from its own boosting iteration,
/// counted per bag over both stages, so losses that draw per round (e.g.
/// `rank:xendcg`'s targets) advance instead of repeating round 0.
#[test]
fn classic_trees_see_advancing_gradient_iterations() {
    use hessboost::data::MetaInfo;
    use hessboost::objective::{GradPair, Loss};
    use std::sync::{Arc, Mutex};
    struct Recording(Arc<Mutex<Vec<usize>>>);
    impl Loss for Recording {
        fn name(&self) -> &'static str {
            "custom:recording"
        }
        fn gradient(&self, preds: &[f32], labels: &[f32], _: Option<&[f32]>, out: &mut [GradPair]) {
            for ((g, p), y) in out.iter_mut().zip(preds).zip(labels) {
                *g = GradPair::new(p - y, 1.0);
            }
        }
        fn gradient_info_at(
            &self,
            preds: &[f32],
            info: &MetaInfo,
            out: &mut [GradPair],
            iteration: usize,
        ) {
            self.0.lock().unwrap().push(iteration);
            self.gradient_info(preds, info, out);
        }
        fn default_metric(&self) -> EvalMetric {
            EvalMetric::Rmse
        }
    }
    let seen = Arc::new(Mutex::new(Vec::new()));
    let params = classic_with(Ebm::builder().outer_bags(1).interactions(1))
        .objective(Objective::custom(Recording(Arc::clone(&seen))))
        .build()
        .unwrap();
    let (_, dtrain) = data(200, 24);
    let rounds = 4;
    train(&params, &dtrain, rounds).unwrap();
    // Three main terms, one FAST ranking at iteration `rounds`, one pair.
    let mut expected: Vec<usize> = (0..3 * rounds).collect();
    expected.push(rounds);
    expected.extend(3 * rounds..4 * rounds);
    assert_eq!(*seen.lock().unwrap(), expected);
}

/// Two label columns are refused under `labels`, not as a wrong objective.
#[test]
fn label_matrices_are_refused_as_labels() {
    let x: Vec<f32> = (0..40).map(|i| i as f32 / 40.0).collect();
    let y: Vec<f32> = x.iter().flat_map(|&v| [v, 1.0 - v]).collect();
    let dtrain = DMatrix::from_dense(&x, 40, 1)
        .unwrap()
        .with_label_matrix(&y, 2)
        .unwrap();
    let params = classic().build().unwrap();
    assert_eq!(invalid_param(train(&params, &dtrain, 2)), "labels");
}
