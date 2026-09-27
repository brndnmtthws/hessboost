//! `booster = boulevard` and its statistical inference ([`hessboost::inference`]).

use hessboost::config::{
    BalancedBagging, BoosterKind, Boulevard, Langevin, ModelShrink, ModelShrinkMode, QueryBagging,
    TrainingParams, TrainingParamsBuilder,
};
use hessboost::inference::{BoulevardInference, KernelSolver, NoiseVariance, honest_refit};
use hessboost::objective::{LambdaRank, Objective, RegLoss};
use hessboost::prelude::*;
use std::num::NonZeroUsize;

mod common;
use common::{invalid_param, labeled_dense, lcg, with_threads};

/// `n` rows of two uniform features with `y = sin(6 x0) + x1 / 2 + noise`.
fn data(n: usize, seed: u64) -> DMatrix {
    let mut next = lcg(seed);
    let mut x = Vec::with_capacity(2 * n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let (a, b) = (next(), next());
        x.extend_from_slice(&[a, b]);
        y.push((6.0 * a).sin() + 0.5 * b + 0.3 * (next() - 0.5));
    }
    labeled_dense(&x, 2, &y)
}

fn boulevard(dropout: f64) -> BoosterKind {
    BoosterKind::Boulevard(Boulevard::builder().dropout(dropout).build().unwrap())
}

fn builder() -> TrainingParamsBuilder {
    TrainingParams::builder()
        .booster(boulevard(0.5))
        .eta(0.8)
        .subsample(0.8)
        .max_depth(4)
        .min_child_weight(5.0)
}

/// Mean standard error over `points` of a model trained on `n` rows and
/// refitted on `n` others.
fn mean_standard_error(n: usize, points: &DMatrix) -> f64 {
    let params = builder().build().unwrap();
    let values = data(n, 2);
    let model = honest_refit(&train(&params, &data(n, 1), 60).unwrap(), &values).unwrap();
    let inference = BoulevardInference::fit(
        &model,
        &values,
        NoiseVariance::Known(1.0),
        KernelSolver::Exact,
    )
    .unwrap();
    let se = inference.standard_errors(points).unwrap().into_vec();
    assert!(se.iter().all(|&s| s.is_finite() && s > 0.0));
    se.iter().sum::<f64>() / se.len() as f64
}

#[test]
fn standard_errors_shrink_as_the_training_set_grows() {
    let points = data(50, 9);
    let (small, large) = (
        mean_standard_error(200, &points),
        mean_standard_error(1600, &points),
    );
    // Leaves of a fixed minimum size hold more rows, and the kernel spreads
    // each point's weight over more of them: ‖w(x)‖ falls with n.
    assert!(large < 0.7 * small, "{large} vs {small}");
}

#[test]
fn nystrom_on_every_row_reproduces_the_exact_solver() {
    for parallel in [1, 3] {
        let mut b = builder();
        if parallel > 1 {
            b = b
                .booster(boulevard(0.0))
                .eta(1.0)
                .num_parallel_tree(parallel);
        }
        let dtrain = data(300, 3);
        let model = train(&b.build().unwrap(), &dtrain, 20).unwrap();
        let points = data(40, 4);
        let se = |solver| {
            BoulevardInference::fit(&model, &dtrain, NoiseVariance::Known(0.5), solver)
                .unwrap()
                .standard_errors(&points)
                .unwrap()
        };
        let exact = se(KernelSolver::Exact);
        let nystrom = se(KernelSolver::Nystrom {
            landmarks: 300,
            seed: 7,
        });
        for (e, n) in exact.as_slice().iter().zip(nystrom.as_slice()) {
            assert!((e - n).abs() <= 1e-9 * e, "{parallel}: {e} vs {n}");
        }
    }
}

#[test]
fn training_and_inference_are_identical_across_thread_counts() {
    let run = || {
        let params = builder().build().unwrap();
        let (dtrain, points) = (data(400, 5), data(30, 6));
        let model = train(&params, &dtrain, 30).unwrap();
        let inference = BoulevardInference::fit(
            &model,
            &dtrain,
            NoiseVariance::TrainingResiduals,
            KernelSolver::Nystrom {
                landmarks: 100,
                seed: 1,
            },
        )
        .unwrap();
        (
            model.predict(&points).unwrap(),
            inference.confidence_intervals(&points, 0.1).unwrap(),
        )
    };
    assert_eq!(with_threads(1, run), with_threads(4, run));
}

#[test]
fn the_boulevard_record_survives_the_native_formats_but_not_slicing() {
    let dtrain = data(200, 8);
    let model = train(&builder().build().unwrap(), &dtrain, 10).unwrap();
    let info = model.boulevard().copied().expect("a Boulevard fit");
    let reloaded = [
        BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap(),
        BoostedModel::from_json(&model.to_json().unwrap()).unwrap(),
    ];
    for m in &reloaded {
        assert_eq!(m.boulevard(), Some(&info));
        assert_eq!(m.predict(&dtrain).unwrap(), model.predict(&dtrain).unwrap());
    }
    // A prefix of a Boulevard average is not one, and neither is a
    // continuation of it.
    let sliced = model.slice(..5, 1).unwrap();
    assert!(sliced.boulevard().is_none());
    let gbtree = TrainingParams::builder().build().unwrap();
    let continued = Trainer::new(&gbtree, &dtrain, 1).init_model(&model).train();
    assert_eq!(invalid_param(continued), "init_model");
}

#[test]
fn inference_refuses_rows_the_model_was_not_trained_on() {
    let params = builder().subsample(1.0).build().unwrap();
    let dtrain = data(300, 10);
    let model = train(&params, &dtrain, 10).unwrap();
    let fewer = dtrain.select_rows(&(0..150).collect::<Vec<_>>()).unwrap();
    let fit = BoulevardInference::fit(
        &model,
        &fewer,
        NoiseVariance::Known(1.0),
        KernelSolver::Exact,
    );
    assert_eq!(invalid_param(fit), "train");
    // After an honest refit the kernel rows are the refit's.
    let values = data(300, 11);
    let refit = honest_refit(&model, &values).unwrap();
    assert!(
        BoulevardInference::fit(
            &refit,
            &values,
            NoiseVariance::Known(1.0),
            KernelSolver::Exact
        )
        .is_ok()
    );
    let plain = train(&TrainingParams::builder().build().unwrap(), &dtrain, 5).unwrap();
    let fit = BoulevardInference::fit(
        &plain,
        &dtrain,
        NoiseVariance::Known(1.0),
        KernelSolver::Exact,
    );
    assert_eq!(invalid_param(fit), "model");
}

/// A Boulevard model trained for 0 rounds has no trees and so no kernel
/// rows: inference on it is refused (it used to fit and then panic), and
/// empty inputs elsewhere give empty results or errors, never panics.
#[test]
fn inference_on_empty_inputs_is_refused_or_empty() {
    let dtrain = data(100, 12);
    let empty = train(&builder().build().unwrap(), &dtrain, 0).unwrap();
    for solver in [
        KernelSolver::Exact,
        KernelSolver::Nystrom {
            landmarks: 10,
            seed: 1,
        },
    ] {
        let fit = BoulevardInference::fit(&empty, &dtrain, NoiseVariance::Known(1.0), solver);
        assert_eq!(invalid_param(fit), "model");
    }
    let model = train(&builder().build().unwrap(), &dtrain, 5).unwrap();
    let inference = BoulevardInference::fit(
        &model,
        &dtrain,
        NoiseVariance::Known(1.0),
        KernelSolver::Nystrom {
            landmarks: 1,
            seed: 3,
        },
    )
    .unwrap();
    assert!(inference.standard_errors(&data(3, 13)).is_ok());
    if let Ok(none) = dtrain.select_rows(&[]) {
        assert!(
            inference
                .standard_errors(&none)
                .is_ok_and(|se| se.n_rows() == 0)
        );
    }
}

/// No truncation is `None`, never a level of 0: the builder refuses `0`,
/// the flat and native `0` read as none, JSON writes none as `null` and
/// reads an older file's `0` as none.
#[test]
fn no_truncation_is_none() {
    assert_eq!(Boulevard::default().truncation(), None);
    for level in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let built = Boulevard::builder().truncation(level).build();
        assert_eq!(invalid_param(built), "boulevard_truncation");
    }
    let flat = |truncation: f64| {
        TrainingParams::from_xgboost([
            ("booster", serde_json::json!("boulevard")),
            ("boulevard_truncation", serde_json::json!(truncation)),
        ])
        .unwrap()
    };
    for (level, expected) in [(0.0, None), (2.5, Some(2.5))] {
        let params = flat(level);
        let BoosterKind::Boulevard(settings) = params.booster else {
            panic!("a Boulevard booster");
        };
        assert_eq!(settings.truncation(), expected);
        assert_eq!(
            TrainingParams::from_xgboost(params.to_xgboost().unwrap()).unwrap(),
            params
        );
    }
    let dtrain = data(150, 14);
    let model = train(&builder().build().unwrap(), &dtrain, 4).unwrap();
    assert_eq!(model.boulevard().unwrap().truncation, None);
    let json = model.to_json().unwrap();
    let mut doc: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(doc["boulevard"]["truncation"], serde_json::Value::Null);
    doc["boulevard"]["truncation"] = serde_json::json!(0.0);
    let older = BoostedModel::from_json(&doc.to_string()).unwrap();
    assert_eq!(older.boulevard(), model.boulevard());
    let native = BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
    assert_eq!(native.boulevard(), model.boulevard());
    let truncated = builder()
        .booster(BoosterKind::Boulevard(
            Boulevard::builder()
                .dropout(0.5)
                .truncation(0.2)
                .build()
                .unwrap(),
        ))
        .build()
        .unwrap();
    let clipped = train(&truncated, &dtrain, 4).unwrap();
    assert_eq!(clipped.boulevard().unwrap().truncation, Some(0.2));
    assert_ne!(
        clipped.predict(&dtrain).unwrap(),
        model.predict(&dtrain).unwrap()
    );
}

/// Standard errors are one value per row (`Predictions<f64>`) and every
/// interval a `conformal::Interval<f64>` centred on the prediction, the
/// prediction interval enclosing the confidence interval.
#[test]
fn inference_outputs_are_typed_per_row() {
    let dtrain = data(200, 15);
    let model = train(&builder().build().unwrap(), &dtrain, 10).unwrap();
    let inference = BoulevardInference::fit(
        &model,
        &dtrain,
        NoiseVariance::Known(0.25),
        KernelSolver::Exact,
    )
    .unwrap();
    let points = data(7, 16);
    let se = inference.standard_errors(&points).unwrap();
    assert_eq!((se.n_rows(), se.width()), (7, 1));
    let ci: Vec<hessboost::conformal::Interval<f64>> =
        inference.confidence_intervals(&points, 0.1).unwrap();
    let pi = inference.prediction_intervals(&points, 0.1).unwrap();
    let preds = model.predict(&points).unwrap();
    assert_eq!((ci.len(), pi.len()), (7, 7));
    for ((c, p), &y) in ci.iter().zip(&pi).zip(preds.as_slice()) {
        let center = f64::midpoint(c.lower, c.upper);
        assert!((center - f64::from(y)).abs() < 1e-9, "{c:?} around {y}");
        assert!(p.lower < c.lower && c.lower < c.upper && c.upper < p.upper);
    }
}

fn refused(b: TrainingParamsBuilder) -> &'static str {
    invalid_param(b.build())
}
#[test]
fn settings_that_break_the_linear_smoother_are_refused() {
    assert_eq!(
        refused(builder().objective(Objective::SquaredLogError)),
        "objective"
    );
    // `scale_pos_weight` reweights rows like the sample weights it refuses.
    let reweighted = Objective::SquaredError(RegLoss::new(2.0).unwrap());
    assert_eq!(refused(builder().objective(reweighted)), "objective");
    assert_eq!(refused(builder().alpha(1.0)), "alpha");
    assert_eq!(refused(builder().eta(1.5)), "eta");
    assert_eq!(refused(builder().num_parallel_tree(2)), "boulevard_dropout");
    assert_eq!(
        refused(builder().booster(boulevard(0.0)).num_parallel_tree(2)),
        "eta"
    );
    // Class-balanced bagging draws rows by their labels; it is refused under
    // its own key whatever the objective (it needs a `binary:*` one, which
    // Boulevard refuses in turn).
    let balanced = BalancedBagging::new(1.0, 0.2).unwrap();
    for objective in [
        Objective::SquaredError(RegLoss::default()),
        Objective::BinaryLogistic(RegLoss::default()),
    ] {
        let bagged = builder()
            .subsample(1.0)
            .objective(objective)
            .balanced_bagging(balanced);
        assert_eq!(refused(bagged), "pos_bagging_fraction");
    }
    // Query bagging needs a `rank:*` objective, which Boulevard refuses.
    let by_query = builder()
        .subsample(1.0)
        .objective(Objective::RankNdcg(LambdaRank::default()))
        .bagging_by_query(QueryBagging::new(0.5).unwrap());
    assert_eq!(refused(by_query), "objective");
    // A dropout of 1 keeps no tree; flat keys need their booster.
    assert_eq!(
        invalid_param(Boulevard::builder().dropout(1.0).build()),
        "boulevard_dropout"
    );
    let flat = serde_json::json!({"booster": "gbtree", "boulevard_dropout": 0.2});
    let serde_json::Value::Object(flat) = flat else {
        unreachable!()
    };
    assert_eq!(
        invalid_param(TrainingParams::from_xgboost(flat)),
        "boulevard_dropout"
    );
    let dtrain = data(100, 12);
    let params = builder().build().unwrap();
    let weighted = dtrain.clone().with_weights(&[2.0; 100]).unwrap();
    assert_eq!(invalid_param(train(&params, &weighted, 2)), "weights");
    let stopping = Trainer::new(&params, &dtrain, 5)
        .eval(&dtrain, "train")
        .early_stopping_rounds(NonZeroUsize::new(2).unwrap())
        .train();
    assert_eq!(invalid_param(stopping), "early_stopping_rounds");
}

#[test]
fn the_round_hook_sees_every_round_and_break_keeps_a_boulevard_average() {
    let params = builder().build().unwrap();
    let dtrain = data(200, 13);
    let mut seen = Vec::new();
    let stopped = Trainer::new(&params, &dtrain, 50)
        .on_round(|round| {
            seen.push(round.iteration());
            if round.iteration() == 9 {
                std::ops::ControlFlow::Break(())
            } else {
                std::ops::ControlFlow::Continue(())
            }
        })
        .train()
        .unwrap()
        .model;
    assert_eq!(seen, (0..10).collect::<Vec<_>>());
    // Stopping after 10 rounds gives the 10-round Boulevard average, not a
    // prefix of a longer run's leaves.
    let ten = train(&params, &dtrain, 10).unwrap();
    assert_eq!(stopped.num_boost_rounds(), 10);
    assert_eq!(
        stopped.predict(&dtrain).unwrap(),
        ten.predict(&dtrain).unwrap()
    );
    assert!(stopped.boulevard().is_some());
}

/// In-place updates regrow trees as ordinary boosting would, which would
/// leave a model that claims to be a Boulevard average of trees it no
/// longer is.
#[test]
fn online_updates_refuse_a_boulevard_model() {
    use hessboost::training::online::{OnlineModel, OnlineParams};
    let dtrain = data(200, 15);
    let model = train(&builder().build().unwrap(), &dtrain, 5).unwrap();
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

/// SGLB runs in the gbtree loop (Langevin noise would also add variance the
/// kernel does not model, and model shrinkage would rescale the average), so
/// it is refused; and a Boulevard model's iteration prefixes are not the
/// models of fewer rounds (every leaf carries the full run's `1/B`), so
/// virtual ensembles of one are refused.
#[test]
fn sglb_and_virtual_ensembles_are_refused() {
    let refused = |b: TrainingParamsBuilder| invalid_param(b.build());
    assert_eq!(refused(builder().langevin(Langevin::default())), "langevin");
    assert_eq!(refused(builder().posterior_sampling(true)), "langevin");
    let shrink = ModelShrink::new(0.01, ModelShrinkMode::Constant).unwrap();
    assert_eq!(refused(builder().model_shrink(shrink)), "model_shrink_rate");
    let dtrain = data(200, 17);
    let model = train(&builder().build().unwrap(), &dtrain, 20).unwrap();
    assert_eq!(
        invalid_param(model.predict_virtual_ensembles(&dtrain, 2)),
        "model"
    );
    assert_eq!(
        invalid_param(model.predict_uncertainty(&dtrain, 2)),
        "model"
    );
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
    let params = builder().build().unwrap();
    assert_eq!(invalid_param(train(&params, &dtrain, 2)), "labels");
}
