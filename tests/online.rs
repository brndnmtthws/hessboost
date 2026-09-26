//! In-place data updates (`hessboost::training::online`): the exact mode
//! reproduces retraining bit for bit, the approximate mode stays close to
//! it, updates are deterministic and atomic, and unsound configurations are
//! refused.

use std::ops::ControlFlow;

use hessboost::config::{BoosterKind, Dart, GrowPolicy};
use hessboost::data::FeatureType;
use hessboost::objective::{CustomLoss, GradPair, Logistic, Objective};
use hessboost::prelude::*;
use hessboost::training::RoundEval;
use hessboost::training::online::{OnlineModel, OnlineParams};

mod common;
use common::{invalid_param, lcg, rmse, with_threads};

const COLS: usize = 4;

fn logistic() -> Objective {
    Objective::BinaryLogistic(Logistic::default())
}

/// `y = 3 x0 - 2 x1 + x2 x3 + noise`, or its sign for classification.
fn data(n: usize, seed: u64, binary: bool) -> DMatrix {
    let mut next = lcg(seed);
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let row = [next(), next(), next(), next()];
        let f = 3.0 * row[0] - 2.0 * row[1] + row[2] * row[3] + 0.2 * (next() - 0.5);
        x.extend(row);
        y.push(if binary { f32::from(f > 0.6) } else { f });
    }
    DMatrix::from_dense(&x, n, COLS)
        .unwrap()
        .with_labels(&y)
        .unwrap()
}

fn params(objective: Objective) -> TrainingParams {
    TrainingParams::builder()
        .objective(objective)
        .tree_method(TreeMethod::Hist)
        .max_depth(4)
        .eta(0.3)
        .build()
        .unwrap()
}

#[test]
fn exact_updates_equal_retraining_bit_for_bit() {
    for objective in [Objective::SquaredError, logistic()] {
        let binary = matches!(objective, Objective::BinaryLogistic(_));
        let name = objective.name().to_owned();
        let p = params(objective);
        let train_data = data(400, 1, binary);
        let mut online =
            OnlineModel::train(&p, &train_data, 15, OnlineParams::with_tolerance(0.0)).unwrap();
        let added = data(30, 2, binary);
        // Several updates in a row: each equals retraining on the data so far.
        for (additions, deletions) in [
            (Some(&added), vec![0, 7, 399]),
            (None, vec![3, 4, 5]),
            (Some(&added), Vec::new()),
        ] {
            online.update(additions, &deletions).unwrap();
            let retrained = train(&p, online.data(), 15).unwrap();
            assert_eq!(
                online.model().to_json().unwrap(),
                retrained.to_json().unwrap(),
                "{name}"
            );
        }
        assert_eq!(online.data().n_rows(), 400 - 3 + 30 - 3 + 30);
    }
}

#[test]
fn approximate_updates_stay_close_to_retraining() {
    let p = params(Objective::SquaredError);
    let train_data = data(2000, 3, false);
    let test = data(1000, 4, false);
    let added = data(40, 5, false);
    let deletions: Vec<usize> = (0..40).map(|i| i * 13).collect();
    let mut online = OnlineModel::train(&p, &train_data, 30, OnlineParams::default()).unwrap();
    let report = online.update(Some(&added), &deletions).unwrap();
    assert_eq!(online.data().n_rows(), 2000);
    assert!(report.nodes_kept > 0);
    let retrained = train(&p, online.data(), 30).unwrap();
    let (updated, reference) = (rmse(online.model(), &test), rmse(&retrained, &test));
    assert!(
        (updated - reference).abs() < 0.05 * reference,
        "updated {updated} vs retrained {reference}"
    );
    // The fit reflects the new data: the added rows' error is no worse than
    // under the original model.
    let original = OnlineModel::train(&p, &train_data, 30, OnlineParams::default()).unwrap();
    assert!(rmse(online.model(), &added) <= rmse(original.model(), &added) * 1.01);

    // A higher tolerance keeps more splits: tolerance 1 regrows only nodes
    // whose split stopped being a valid candidate.
    let mut frozen =
        OnlineModel::train(&p, &train_data, 30, OnlineParams::with_tolerance(1.0)).unwrap();
    let tolerant = frozen.update(Some(&added), &deletions).unwrap();
    assert!(tolerant.subtrees_regrown <= report.subtrees_regrown);
    assert!(tolerant.nodes_kept >= report.nodes_kept);
}

#[test]
fn updates_ignore_the_thread_count() {
    let p = params(logistic());
    let train_data = data(600, 6, true);
    let added = data(20, 7, true);
    let run = |threads| {
        with_threads(threads, || {
            let mut online =
                OnlineModel::train(&p, &train_data, 10, OnlineParams::with_tolerance(0.05))
                    .unwrap();
            online.update(Some(&added), &[1, 2, 3, 50]).unwrap();
            online.model().to_json().unwrap()
        })
    };
    assert_eq!(run(1), run(4));
}

#[test]
fn an_interrupted_update_changes_nothing() {
    let p = params(Objective::SquaredError);
    let train_data = data(300, 8, false);
    for tolerance in [0.0, 0.1] {
        let mut online =
            OnlineModel::train(&p, &train_data, 10, OnlineParams::with_tolerance(tolerance))
                .unwrap();
        let before = online.model().to_json().unwrap();
        let stop = |round: &RoundEval| {
            if round.iteration == 3 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };
        assert_eq!(
            invalid_param(online.update_with(None, &[0, 1], stop)),
            "on_round"
        );
        assert_eq!(online.model().to_json().unwrap(), before);
        assert_eq!(online.data().n_rows(), 300);
        // The model still updates afterwards, as an uninterrupted one would.
        online.update(None, &[0, 1]).unwrap();
        let mut fresh =
            OnlineModel::train(&p, &train_data, 10, OnlineParams::with_tolerance(tolerance))
                .unwrap();
        fresh.update(None, &[0, 1]).unwrap();
        assert_eq!(
            online.model().to_json().unwrap(),
            fresh.model().to_json().unwrap()
        );
    }
}

/// Updates refuse what retraining on the updated data refuses, in both
/// modes and before anything changes: the approximate mode computes
/// gradients directly, so a label outside `binary:logistic`'s `[0, 1]` would
/// otherwise slip through.
#[test]
fn updates_refuse_labels_retraining_refuses() {
    let p = params(logistic());
    let train_data = data(300, 10, true);
    let bad = DMatrix::from_dense(&[0.5; COLS], 1, COLS)
        .unwrap()
        .with_labels(&[2.0])
        .unwrap();
    let good = data(5, 11, true);
    for tolerance in [0.1, 0.0] {
        let mut online =
            OnlineModel::train(&p, &train_data, 8, OnlineParams::with_tolerance(tolerance))
                .unwrap();
        let before = online.model().to_json().unwrap();
        // Retraining refuses the added row's label under the same name.
        let retrain_err = invalid_param(train(&p, &bad, 8));
        assert_eq!(
            invalid_param(online.update(Some(&bad), &[0])),
            retrain_err,
            "tolerance {tolerance}"
        );
        assert_eq!(online.model().to_json().unwrap(), before);
        assert_eq!(online.data().n_rows(), 300);
        // A later valid update still works, as on a fresh model.
        online.update(Some(&good), &[0]).unwrap();
        assert_eq!(online.data().n_rows(), 304);
        let mut fresh =
            OnlineModel::train(&p, &train_data, 8, OnlineParams::with_tolerance(tolerance))
                .unwrap();
        fresh.update(Some(&good), &[0]).unwrap();
        assert_eq!(
            online.model().to_json().unwrap(),
            fresh.model().to_json().unwrap()
        );
    }
}

#[test]
fn unsound_configurations_and_changes_are_refused() {
    let train_data = data(200, 9, false);
    let base = || {
        TrainingParams::builder()
            .tree_method(TreeMethod::Hist)
            .max_depth(3)
    };
    let online = OnlineParams::default();
    for (params, name) in [
        (base().subsample(0.8).build().unwrap(), "subsample"),
        (base().colsample_bytree(0.5).build().unwrap(), "subsample"),
        (
            base()
                .grow_policy(GrowPolicy::LossGuide)
                .max_leaves(8)
                .build()
                .unwrap(),
            "grow_policy",
        ),
        (base().max_leaves(8).build().unwrap(), "grow_policy"),
        (base().unlimited_depth().build().unwrap(), "grow_policy"),
        (
            base()
                .booster(BoosterKind::Dart(Dart::default()))
                .build()
                .unwrap(),
            "booster",
        ),
        (
            base().num_parallel_tree(2).build().unwrap(),
            "num_parallel_tree",
        ),
        (
            base().objective(Objective::AbsoluteError).build().unwrap(),
            "objective",
        ),
        (
            base()
                .objective(Objective::custom(CustomLoss::new(
                    "custom:sqerr",
                    1,
                    |p, y, _w, out| {
                        for (o, (p, y)) in out.iter_mut().zip(p.iter().zip(y)) {
                            *o = GradPair::new(p - y, 1.0);
                        }
                    },
                )))
                .build()
                .unwrap(),
            "objective",
        ),
    ] {
        assert_eq!(
            invalid_param(OnlineModel::train(&params, &train_data, 3, online)),
            name
        );
    }
    let p = base().build().unwrap();
    assert_eq!(
        invalid_param(OnlineModel::train(
            &p,
            &train_data,
            3,
            OnlineParams::with_tolerance(1.5)
        )),
        "tolerance"
    );
    let weighted = data(200, 9, false).with_weights(&[1.0; 200]).unwrap();
    assert_eq!(
        invalid_param(OnlineModel::train(&p, &weighted, 3, online)),
        "data"
    );
    let categorical = data(200, 9, false)
        .with_feature_types(&[FeatureType::Numerical; COLS])
        .unwrap();
    assert!(OnlineModel::train(&p, &categorical, 3, online).is_ok());

    let mut model = OnlineModel::train(&p, &train_data, 3, online).unwrap();
    assert_eq!(invalid_param(model.update(None, &[200])), "deletions");
    assert_eq!(invalid_param(model.update(None, &[4, 4])), "deletions");
    let all: Vec<usize> = (0..200).collect();
    assert_eq!(invalid_param(model.update(None, &all)), "deletions");
    let unlabelled = DMatrix::from_dense(&[0.0; COLS], 1, COLS).unwrap();
    assert_eq!(
        invalid_param(model.update(Some(&unlabelled), &[])),
        "additions"
    );
    let narrow = DMatrix::from_dense(&[0.0; 2], 1, 2)
        .unwrap()
        .with_labels(&[1.0])
        .unwrap();
    assert_eq!(invalid_param(model.update(Some(&narrow), &[])), "additions");
}

/// Models whose trees updates cannot replay are refused: linear leaves
/// (an imported LightGBM `linear_tree` model) have no Newton-step leaf to
/// recompute.
#[test]
fn from_model_refuses_linear_leaves() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/lightgbm-4.7.0-linear.txt"
    );
    let model = BoostedModel::load_lightgbm_text(path).unwrap();
    assert_eq!(model.n_features(), 6);
    let mut next = lcg(12);
    let x: Vec<f32> = (0..200 * 6).map(|_| next()).collect();
    let y: Vec<f32> = x.chunks(6).map(|r| r[0] - r[1]).collect();
    let data = DMatrix::from_dense(&x, 200, 6)
        .unwrap()
        .with_labels(&y)
        .unwrap();
    let p = params(Objective::SquaredError);
    for tolerance in [0.1, 0.0] {
        let online = OnlineParams::with_tolerance(tolerance);
        assert_eq!(
            invalid_param(OnlineModel::from_model(model.clone(), &p, &data, online)),
            "model"
        );
        // A model of these parameters on this data is accepted.
        let trained = train(&p, &data, 3).unwrap();
        assert!(OnlineModel::from_model(trained, &p, &data, online).is_ok());
    }
}

/// A model with splits deeper than `max_depth` is not one `params` trained:
/// regrowing below such a split would have no depth left.
#[test]
fn from_model_refuses_trees_deeper_than_max_depth() {
    let d = data(200, 14, false);
    let p = params(Objective::SquaredError);
    let mut deeper = p.clone();
    deeper.max_depth = p.max_depth.and_then(|depth| depth.checked_add(2));
    let model = train(&deeper, &d, 3).unwrap();
    for tolerance in [0.1, 0.0] {
        let online = OnlineParams::with_tolerance(tolerance);
        assert_eq!(
            invalid_param(OnlineModel::from_model(model.clone(), &p, &d, online)),
            "model"
        );
        assert!(OnlineModel::from_model(model.clone(), &deeper, &d, online).is_ok());
    }
}
