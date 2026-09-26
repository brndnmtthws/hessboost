//! `booster = boulevard` and its statistical inference ([`hessboost::inference`]).

use hessboost::config::{BoosterKind, TrainingParams, TrainingParamsBuilder};
use hessboost::inference::{BoulevardInference, KernelSolver, NoiseVariance, honest_refit};
use hessboost::prelude::*;

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

fn builder() -> TrainingParamsBuilder {
    TrainingParams::builder()
        .booster(BoosterKind::Boulevard)
        .eta(0.8)
        .boulevard_dropout(0.5)
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
    let se = inference.standard_errors(points).unwrap();
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
                .boulevard_dropout(0.0)
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
        for (e, n) in exact.iter().zip(&nystrom) {
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

#[test]
fn settings_that_break_the_linear_smoother_are_refused() {
    let refused = |b: TrainingParamsBuilder| match b.build() {
        Err(HessboostError::InvalidParameter { name, .. }) => name,
        other => panic!("expected a refusal, got {other:?}"),
    };
    assert_eq!(
        refused(builder().objective("reg:pseudohubererror")),
        "objective"
    );
    assert_eq!(refused(builder().alpha(1.0)), "alpha");
    assert_eq!(refused(builder().eta(1.5)), "eta");
    assert_eq!(refused(builder().num_parallel_tree(2)), "boulevard_dropout");
    assert_eq!(
        refused(builder().boulevard_dropout(0.0).num_parallel_tree(2)),
        "eta"
    );
    assert_eq!(
        refused(TrainingParams::builder().boulevard_dropout(0.2)),
        "boulevard_dropout"
    );
    let dtrain = data(100, 12);
    let params = builder().build().unwrap();
    let weighted = dtrain.clone().with_weights(&[2.0; 100]).unwrap();
    assert_eq!(invalid_param(train(&params, &weighted, 2)), "weights");
    let stopping = Trainer::new(&params, &dtrain, 5)
        .eval(&dtrain, "train")
        .early_stopping_rounds(2)
        .train();
    assert_eq!(invalid_param(stopping), "early_stopping_rounds");
}
