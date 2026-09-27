//! SGLB (Langevin boosting with model shrinkage, CatBoost's
//! `posterior_sampling`) and virtual ensembles: truncations of a shrunk
//! model are the shorter training runs, bit for bit; the refusals; the
//! formats; and the uncertainty decomposition's invariants.

mod common;

use std::sync::mpsc::{Sender, channel};

use hessboost::config::{
    BoosterKind, Dart, Langevin, LinearTree, ModelShrink, ModelShrinkMode, Monotone, MultiStrategy,
    QuantizedGrad, TrainingParamsBuilder,
};
use hessboost::metric::Metric;
use hessboost::model::Predictions;
use hessboost::objective::distributional::{DistFamily, Distributional};
use hessboost::objective::{Logistic, Multiclass, Objective};
use hessboost::prelude::*;
use serde_json::json;

/// `n` rows of a noisy regression target over four features (feature 3
/// has missing values).
fn regression(n: usize) -> DMatrix {
    let mut noise = common::lcg(17);
    let mut x = Vec::with_capacity(n * 4);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let row = common::four_features(i);
        x.extend_from_slice(&row);
        y.push(3.0 * row[0] - 2.0 * row[1] * row[2] + noise() - 0.5);
    }
    common::labeled_dense(&x, 4, &y)
}

/// The same features with a class label out of `classes` (`2`: `0`/`1`).
fn classification(n: usize, classes: usize) -> DMatrix {
    let mut noise = common::lcg(5);
    let mut x = Vec::with_capacity(n * 4);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let row = common::four_features(i);
        x.extend_from_slice(&row);
        let score = row[0] + 0.5 * row[1] + 0.3 * noise();
        y.push(((score * classes as f32 / 1.8) as usize).min(classes - 1) as f32);
    }
    common::labeled_dense(&x, 4, &y)
}

/// Model shrinkage at `rate` in `mode`.
fn shrink(rate: f64, mode: ModelShrinkMode) -> ModelShrink {
    ModelShrink::new(rate, mode).unwrap()
}

fn bits(values: impl AsRef<[f32]>) -> Vec<u32> {
    values.as_ref().iter().map(|v| v.to_bits()).collect()
}

/// The model after `k` iterations of a `rounds`-round shrunk model, however
/// it is reached (an iteration range, a slice, a virtual-ensemble member),
/// is the model the same run stopped after `k` rounds returns, bit for bit:
/// shrinkage reweights every earlier tree each iteration, and the stored
/// coefficients rebuild each truncation with training's own arithmetic.
#[test]
fn truncations_are_the_shorter_runs_bit_for_bit() {
    let base = || TrainingParams::builder().max_depth(3).eta(0.2).seed(9);
    let cases: Vec<(&str, TrainingParams, DMatrix)> = vec![
        (
            "hist posterior sampling",
            base()
                .tree_method(TreeMethod::Hist)
                .posterior_sampling(true)
                .subsample(0.8)
                .build()
                .unwrap(),
            regression(300),
        ),
        (
            "exact decreasing shrinkage with Langevin",
            base()
                .tree_method(TreeMethod::Exact)
                .objective(Objective::BinaryLogistic(Logistic::default()))
                .langevin(
                    Langevin::builder()
                        .diffusion_temperature(50.0)
                        .build()
                        .unwrap(),
                )
                .model_shrink(shrink(0.3, ModelShrinkMode::Decreasing))
                .build()
                .unwrap(),
            classification(300, 2),
        ),
        (
            "approx multiclass posterior sampling",
            base()
                .tree_method(TreeMethod::Approx)
                .objective(Objective::Softprob(Multiclass::new(3).unwrap()))
                .posterior_sampling(true)
                .build()
                .unwrap(),
            classification(300, 3),
        ),
        (
            "vector-leaf dist:normal posterior sampling",
            base()
                .tree_method(TreeMethod::Hist)
                .objective(Objective::Dist(Distributional::new(DistFamily::Normal)))
                .multi_strategy(MultiStrategy::MultiOutputTree)
                .posterior_sampling(true)
                .build()
                .unwrap(),
            regression(300),
        ),
        (
            "shrinkage alone, boosted forest",
            base()
                .tree_method(TreeMethod::Hist)
                .model_shrink(shrink(0.5, ModelShrinkMode::Constant))
                .num_parallel_tree(2)
                .subsample(0.7)
                .build()
                .unwrap(),
            regression(300),
        ),
    ];
    let rounds = 24;
    for (name, params, data) in &cases {
        let long = train(params, data, rounds).unwrap();
        let members = long.predict_virtual_ensembles(data, 4).unwrap();
        assert_eq!(members.iterations(), &[15, 18, 21, 24], "{name}");
        for k in [1, 7, 15, 18, 23, rounds] {
            let short = train(params, data, k).unwrap();
            let expected = bits(short.predict_margin(data).unwrap());
            assert_eq!(
                bits(long.predict_margin_range(data, ..k).unwrap()),
                expected,
                "{name}: iterations ..{k}"
            );
            let sliced = long.slice(..k, 1).unwrap();
            assert_eq!(
                sliced.to_json().unwrap(),
                short.to_json().unwrap(),
                "{name}: slice ..{k}"
            );
            if let Some(m) = members.iterations().iter().position(|&it| it == k) {
                assert_eq!(
                    bits(members.member_margins(m).unwrap()),
                    expected,
                    "{name}: member {m}"
                );
            }
        }
    }
}

/// Early stopping keeps the shrunk model as it was after the best
/// iteration (CatBoost's `use_best_model`): exactly the run that stops
/// there.
#[test]
fn early_stopping_keeps_the_best_iteration_model() {
    let data = regression(400);
    let valid = regression(120);
    let params = TrainingParams::builder()
        .max_depth(4)
        .eta(0.5)
        .posterior_sampling(true)
        .build()
        .unwrap();
    let result = Trainer::new(&params, &data, 200)
        .eval(&valid, "valid")
        .early_stopping_rounds(3)
        .train()
        .unwrap();
    let model = result.model;
    let best = model.best_iteration().unwrap();
    assert!(best + 1 < result.history.len(), "training must stop early");
    assert_eq!(model.num_boost_rounds(), best + 1);
    let short = train(&params, &data, best + 1).unwrap();
    assert_eq!(
        bits(model.predict_margin(&valid).unwrap()),
        bits(short.predict_margin(&valid).unwrap())
    );
}

/// A metric that sends out the predictions training evaluates each round.
struct Recorder(Sender<Vec<f32>>);

impl Metric for Recorder {
    fn name(&self) -> &'static str {
        "recorded"
    }

    fn eval(&self, preds: &[f32], _labels: &[f32], _weights: Option<&[f32]>) -> f64 {
        self.0.send(preds.to_vec()).unwrap();
        0.0
    }
}

/// A shrunk model predicts the margins its training evaluated, bit for
/// bit, after every round: prediction repeats training's shrink-then-add
/// arithmetic.
#[test]
fn predictions_are_the_training_margins() {
    // One row whose second tree exactly cancels the shrunk intercept:
    // training reaches margin 0, which a product-weighted sum of the trees
    // misses by rounding.
    let one = DMatrix::from_dense(&[0.0], 1, 1)
        .unwrap()
        .with_labels(&[0.0])
        .unwrap();
    let params = TrainingParams::builder()
        .base_score(100_663_296.0)
        .eta(1.0)
        .lambda(0.0)
        .model_shrink(shrink(0.7, ModelShrinkMode::Constant))
        .build()
        .unwrap();
    let result = Trainer::new(&params, &one, 2)
        .eval(&one, "train")
        .train()
        .unwrap();
    assert_eq!(result.history.last().unwrap().scores[0].value, 0.0);
    assert_eq!(
        bits(result.model.predict_margin(&one).unwrap()),
        bits([0.0])
    );

    let base = || TrainingParams::builder().max_depth(3).eta(0.3).seed(4);
    let targets: Vec<f32> = (0..300)
        .flat_map(|i| [(i % 7) as f32 * 1e3, (i % 5) as f32 - 2.0])
        .collect();
    let x: Vec<f32> = (0..300).flat_map(common::four_features).collect();
    let matrix = DMatrix::from_dense(&x, 300, 4)
        .unwrap()
        .with_label_matrix(&targets, 2)
        .unwrap();
    let cases: Vec<(&str, TrainingParams, DMatrix)> = vec![
        (
            "hist posterior sampling",
            base()
                .base_score(1234.5)
                .posterior_sampling(true)
                .build()
                .unwrap(),
            regression(300),
        ),
        (
            "exact decreasing shrinkage",
            base()
                .tree_method(TreeMethod::Exact)
                .model_shrink(shrink(0.4, ModelShrinkMode::Decreasing))
                .build()
                .unwrap(),
            regression(300),
        ),
        (
            "shrinkage alone, boosted forest",
            base()
                .model_shrink(shrink(0.5, ModelShrinkMode::Constant))
                .num_parallel_tree(2)
                .subsample(0.7)
                .build()
                .unwrap(),
            regression(300),
        ),
        (
            "shrinkage alone, linear leaves",
            base()
                .model_shrink(shrink(0.5, ModelShrinkMode::Constant))
                .linear_tree(LinearTree::default())
                .build()
                .unwrap(),
            regression(300),
        ),
        (
            "vector leaves over a label matrix",
            base()
                .multi_strategy(MultiStrategy::MultiOutputTree)
                .posterior_sampling(true)
                .build()
                .unwrap(),
            matrix.clone(),
        ),
        (
            "scalar trees over a label matrix",
            base().posterior_sampling(true).build().unwrap(),
            matrix,
        ),
    ];
    let rounds = 12;
    for (name, params, data) in cases {
        let (tx, rx) = channel();
        let model = Trainer::new(&params, &data, rounds)
            .eval(&data, "train")
            .custom_metric(Box::new(Recorder(tx)))
            .train()
            .unwrap()
            .model;
        let seen: Vec<Vec<f32>> = rx.try_iter().collect();
        assert_eq!(seen.len(), rounds, "{name}");
        for (k, margins) in seen.iter().enumerate() {
            assert_eq!(
                bits(model.predict_margin_range(&data, ..=k).unwrap()),
                bits(margins),
                "{name}: after round {k}"
            );
        }
        assert_eq!(
            bits(model.predict_margin(&data).unwrap()),
            bits(&seen[rounds - 1]),
            "{name}"
        );
    }
}

/// The Langevin draws are keyed by seed, iteration, and row or leaf, so the
/// model does not depend on the thread count; the seed does change it.
#[test]
fn langevin_training_is_thread_count_independent() {
    let data = regression(20_000);
    let params = |seed| {
        TrainingParams::builder()
            .objective(Objective::SquaredError)
            .tree_method(TreeMethod::Hist)
            .max_depth(4)
            .posterior_sampling(true)
            .seed(seed)
            .build()
            .unwrap()
    };
    let run = |threads, seed| {
        common::with_threads(threads, || {
            train(&params(seed), &data, 6).unwrap().to_bytes().unwrap()
        })
    };
    let serial = run(1, 0);
    assert_eq!(serial, run(4, 0));
    assert_ne!(serial, run(1, 1));
    let plain = TrainingParams::builder()
        .tree_method(TreeMethod::Hist)
        .max_depth(4)
        .build()
        .unwrap();
    assert_ne!(serial, train(&plain, &data, 6).unwrap().to_bytes().unwrap());
}

/// A leaf whose Hessian sum is below `min_child_weight` weighs `0`
/// (XGBoost's `CalcWeight`) also when its leaves are re-estimated with
/// Langevin noise, which it does not receive (as CatBoost skips leaves
/// without data): a root that cannot form a leaf predicts what the
/// noise-free scalar trees do, the shrunk intercepts.
#[test]
fn renewed_leaves_keep_min_child_weight() {
    let x = [0.0, 1.0, 2.0, 3.0];
    let scalar = common::labeled_dense(&x, 1, &[5.0, 6.0, 7.0, 8.0]);
    let matrix = DMatrix::from_dense(&x, 4, 1)
        .unwrap()
        .with_label_matrix(&[5.0, -1.0, 6.0, -2.0, 7.0, -3.0, 8.0, -4.0], 2)
        .unwrap();
    let params = |strategy, langevin: bool| {
        let builder = TrainingParams::builder()
            .base_score(0.5)
            .min_child_weight(10.0)
            .multi_strategy(strategy)
            .model_shrink(shrink(0.1, ModelShrinkMode::Constant));
        let builder = if langevin {
            builder.langevin(Langevin::default())
        } else {
            builder
        };
        builder.build().unwrap()
    };
    for (strategy, data) in [
        (MultiStrategy::OneOutputPerTree, &scalar),
        (MultiStrategy::OneOutputPerTree, &matrix),
        (MultiStrategy::MultiOutputTree, &matrix),
    ] {
        let noisy = train(&params(strategy, true), data, 3).unwrap();
        let plain = train(&params(MultiStrategy::OneOutputPerTree, false), data, 3).unwrap();
        assert_eq!(
            bits(noisy.predict_margin(data).unwrap()),
            bits(plain.predict_margin(data).unwrap()),
            "{strategy:?}"
        );
    }
}

/// Every format keeps a shrunk model's predictions (the native and compact
/// ones bit for bit, XGBoost's closed form within `f32` rounding), and the
/// native ones keep its truncations; an inconsistent shrinkage record is
/// refused.
#[test]
fn shrunk_models_round_trip() {
    let data = regression(200);
    let params = TrainingParams::builder()
        .max_depth(3)
        .posterior_sampling(true)
        .build()
        .unwrap();
    let model = train(&params, &data, 12).unwrap();
    let margin = model.predict_margin(&data).unwrap();
    let margins = bits(&margin);
    let prefix = bits(model.predict_margin_range(&data, ..5).unwrap());
    for restored in [
        BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap(),
        BoostedModel::from_json(&model.to_json().unwrap()).unwrap(),
    ] {
        assert_eq!(bits(restored.predict_margin(&data).unwrap()), margins);
        assert_eq!(
            bits(restored.predict_margin_range(&data, ..5).unwrap()),
            prefix
        );
    }
    let xgboost = BoostedModel::from_xgboost_json(&model.to_xgboost_json().unwrap()).unwrap();
    for (&x, &m) in xgboost
        .predict_margin(&data)
        .unwrap()
        .as_slice()
        .iter()
        .zip(margin.as_slice())
    {
        assert!((x - m).abs() <= 1e-5 * m.abs().max(1.0), "{x} vs {m}");
    }
    let compact = model.to_compact().unwrap();
    assert_eq!(bits(compact.predict_margin(&data).unwrap()), margins);
    let compact = hessboost::model::compact::CompactModel::from_bytes(&compact.to_bytes()).unwrap();
    assert_eq!(bits(compact.predict_margin(&data).unwrap()), margins);
    let offset = regression(200).with_base_margin(&[0.5; 200]).unwrap();
    assert_eq!(
        bits(compact.predict_margin(&offset).unwrap()),
        bits(model.predict_margin(&offset).unwrap())
    );
    // SHAP attributes the weighted trees: each row's contributions sum to
    // its margin.
    let contribs = model.predict_contribs(&data).unwrap();
    for (row, &m) in margin.as_slice().iter().enumerate() {
        let sum: f32 = contribs.get(row, 0).unwrap().iter().sum();
        assert!((sum - m).abs() < 1e-4, "{sum} vs {m}");
    }

    let mut doc: serde_json::Value = serde_json::from_str(&model.to_json().unwrap()).unwrap();
    doc["shrinkage"]["factors"][3] = 0.5.into();
    let err = BoostedModel::from_json(&doc.to_string())
        .unwrap_err()
        .to_string();
    assert!(err.contains("shrinkage record"), "{err}");
}

/// A shrunk model has no meaning for ranges starting after iteration 0 or
/// strided slices, and SHAP covers its whole ensemble only.
#[test]
fn shrunk_models_refuse_non_prefix_selections() {
    let data = regression(100);
    let params = TrainingParams::builder()
        .max_depth(2)
        .model_shrink(shrink(0.1, ModelShrinkMode::Constant))
        .build()
        .unwrap();
    let model = train(&params, &data, 10).unwrap();
    assert_eq!(
        common::invalid_param(model.predict_margin_range(&data, 2..5)),
        "iterations"
    );
    assert_eq!(common::invalid_param(model.slice(2..5, 1)), "slice");
    assert_eq!(common::invalid_param(model.slice(..6, 2)), "slice");
    assert_eq!(
        common::invalid_param(model.predict_contribs_range(&data, ..4)),
        "iterations"
    );
    assert!(model.predict_contribs_range(&data, ..).is_ok());
    assert!(model.predict_leaf_range(&data, ..4).is_ok());
}

/// Quantized training's full-precision leaf renewal would be overwritten
/// by the Langevin leaf re-estimation, so the combination is refused, in
/// the typed and the flat form; quantized histograms alone are accepted.
#[test]
fn langevin_refuses_quantized_leaf_renewal() {
    let renewed = QuantizedGrad::builder().renew_leaf(true).build().unwrap();
    for builder in [
        TrainingParams::builder().langevin(Langevin::default()),
        TrainingParams::builder().posterior_sampling(true),
    ] {
        let quantized = builder.clone().quantized(QuantizedGrad::default());
        assert!(quantized.build().is_ok());
        assert_eq!(
            common::invalid_param(builder.quantized(renewed).build()),
            "langevin"
        );
    }
    let flat = [
        ("posterior_sampling", json!(true)),
        ("use_quantized_grad", json!(true)),
        ("quant_train_renew_leaf", json!(true)),
    ];
    assert_eq!(
        common::invalid_param(TrainingParams::from_xgboost(flat)),
        "langevin"
    );
}

/// The Langevin noise scale `sqrt(2 / (eta * T))` must be a finite,
/// positive `f32` (the gradients' type): an underflowing `eta * T` would
/// make every noisy gradient infinite, an overflowing one would switch the
/// noise off. Posterior sampling's `T` (the row count) always gives one.
#[test]
fn langevin_noise_scale_must_be_representable() {
    let params = |eta: f64, temperature: f64| {
        let langevin = Langevin::builder()
            .diffusion_temperature(temperature)
            .build()
            .unwrap();
        TrainingParams::builder()
            .eta(eta)
            .langevin(langevin)
            .build()
    };
    for (eta, temperature) in [(1e-10, 1e-300), (1e30, 1e300), (1.0, 1e-78), (1.0, 1e92)] {
        assert_eq!(
            common::invalid_param(params(eta, temperature)),
            "diffusion_temperature",
            "eta {eta}, temperature {temperature}"
        );
    }
    assert!(params(1.0, 1e-70).is_ok() && params(1.0, 1e70).is_ok());
    // A subnormal scale (about 1.4e-39 here) is tiny, but still noise.
    assert!(params(1.0, 1e78).is_ok());
    let tiny = TrainingParams::builder()
        .eta(1e-44)
        .posterior_sampling(true)
        .build()
        .unwrap();
    assert!(train(&tiny, &regression(2), 1).is_ok());
}

/// Unsupported or conflicting SGLB settings are refused, never ignored.
#[test]
fn unsupported_combinations_are_refused() {
    let refused = |builder: TrainingParamsBuilder| common::invalid_param(builder.build());
    let base = TrainingParams::builder;
    let tempered = || {
        Langevin::builder()
            .diffusion_temperature(10.0)
            .build()
            .unwrap()
    };
    let constant = |rate| shrink(rate, ModelShrinkMode::Constant);
    // Posterior sampling derives the temperature and the shrinkage.
    assert_eq!(
        refused(base().posterior_sampling(true).langevin(tempered())),
        "diffusion_temperature"
    );
    assert_eq!(
        refused(base().posterior_sampling(true).model_shrink(constant(0.01))),
        "model_shrink_rate"
    );
    assert!(
        base()
            .posterior_sampling(true)
            .langevin(Langevin::default())
            .build()
            .is_ok()
    );
    // The constant coefficient 1 - rate * eta must stay positive.
    assert_eq!(
        refused(base().eta(0.5).model_shrink(constant(2.0))),
        "model_shrink_rate"
    );
    let dart = BoosterKind::Dart(Dart::default());
    assert_eq!(
        refused(base().langevin(Langevin::default()).booster(dart)),
        "langevin"
    );
    assert_eq!(
        refused(base().model_shrink(constant(0.1)).booster(dart)),
        "model_shrink_rate"
    );
    assert_eq!(
        refused(
            base()
                .model_shrink(constant(0.1))
                .booster(BoosterKind::GbLinear)
        ),
        "model_shrink_rate"
    );
    for builder in [
        base().num_parallel_tree(2),
        base().monotone_constraints(vec![Monotone::Increasing]),
        base().path_smooth(1.0),
        base().linear_tree(LinearTree::default()),
    ] {
        assert_eq!(refused(builder.langevin(Langevin::default())), "langevin");
    }
    // The flat (XGBoost/Python) keys: dependent keys need their switch, and
    // an explicit `langevin=false` conflicts with posterior sampling.
    let flat = |pairs: &[(&str, serde_json::Value)]| {
        common::invalid_param(TrainingParams::from_xgboost(pairs.iter().cloned()))
    };
    assert_eq!(
        flat(&[("diffusion_temperature", json!(10.0))]),
        "diffusion_temperature"
    );
    assert_eq!(
        flat(&[("model_shrink_mode", json!("decreasing"))]),
        "model_shrink_mode"
    );
    assert_eq!(
        flat(&[
            ("posterior_sampling", json!(true)),
            ("langevin", json!(false))
        ]),
        "langevin"
    );
    let round_trip = base()
        .langevin(tempered())
        .model_shrink(shrink(0.2, ModelShrinkMode::Decreasing))
        .build()
        .unwrap();
    assert_eq!(
        TrainingParams::from_xgboost(round_trip.to_xgboost().unwrap()).unwrap(),
        round_trip
    );

    let data = regression(50);
    let params = base().posterior_sampling(true).build().unwrap();
    let with_margin = regression(50).with_base_margin(&[0.5; 50]).unwrap();
    assert_eq!(
        common::invalid_param(train(&params, &with_margin, 2)),
        "model_shrink_rate"
    );
    let shrunk = train(&params, &data, 4).unwrap();
    assert_eq!(
        common::invalid_param(Trainer::new(&params, &data, 2).init_model(&shrunk).train()),
        "model_shrink_rate"
    );
    let plain_params = base().build().unwrap();
    assert_eq!(
        common::invalid_param(
            Trainer::new(&plain_params, &data, 2)
                .init_model(&shrunk)
                .train()
        ),
        "model_shrink_rate"
    );
    // Posterior sampling's constant coefficient 1 - eta / (2N) must stay
    // positive for the actual row count.
    let steep = base().posterior_sampling(true).eta(3.0).build().unwrap();
    assert_eq!(
        common::invalid_param(train(&steep, &regression(1), 1)),
        "posterior_sampling"
    );
    assert!(train(&steep, &regression(2), 1).is_ok());
}

/// No shrinkage is `model_shrink = None`, never a rate of 0: typed Langevin
/// shrinks only when asked, while the flat form keeps CatBoost's defaults
/// (`langevin=true` alone shrinks at 0.001, `model_shrink_rate=0` is none),
/// and both round-trip.
#[test]
fn no_model_shrinkage_is_none() {
    assert_eq!(
        common::invalid_param(ModelShrink::new(0.0, ModelShrinkMode::Constant)),
        "model_shrink_rate"
    );
    let data = regression(100);
    let typed = TrainingParams::builder()
        .langevin(Langevin::default())
        .build()
        .unwrap();
    // An unshrunk model can be trained further.
    let model = train(&typed, &data, 3).unwrap();
    assert!(
        Trainer::new(&typed, &data, 2)
            .init_model(&model)
            .train()
            .is_ok()
    );
    let flat =
        |pairs: &[(&str, serde_json::Value)]| TrainingParams::from_xgboost(pairs.iter().cloned());
    let catboost = flat(&[("langevin", json!(true))]).unwrap();
    assert_eq!(
        catboost.model_shrink,
        Some(shrink(0.001, ModelShrinkMode::Constant))
    );
    let off = flat(&[("langevin", json!(true)), ("model_shrink_rate", json!(0.0))]).unwrap();
    assert_eq!(off, typed);
    for params in [&typed, &catboost] {
        assert_eq!(
            &TrainingParams::from_xgboost(params.to_xgboost().unwrap()).unwrap(),
            params
        );
    }
    // A mode needs a rate other than 0, and posterior sampling derives its
    // own rate, refusing an explicit 0 too.
    assert_eq!(
        common::invalid_param(flat(&[
            ("model_shrink_rate", json!(0.0)),
            ("model_shrink_mode", json!("decreasing"))
        ])),
        "model_shrink_mode"
    );
    assert_eq!(
        common::invalid_param(flat(&[
            ("posterior_sampling", json!(true)),
            ("model_shrink_rate", json!(0.0))
        ])),
        "model_shrink_rate"
    );
}

/// Continued Langevin training (without shrinkage) grows the trees of the
/// uninterrupted run: its draws are keyed by the absolute iteration.
#[test]
fn langevin_continuation_matches_the_uninterrupted_run() {
    let data = regression(200);
    let params = TrainingParams::builder()
        .max_depth(3)
        .langevin(Langevin::default())
        .build()
        .unwrap();
    let first = train(&params, &data, 5).unwrap();
    let continued = Trainer::new(&params, &data, 4)
        .init_model(&first)
        .train()
        .unwrap()
        .model;
    let whole = train(&params, &data, 9).unwrap();
    assert_eq!(
        bits(continued.predict_margin(&data).unwrap()),
        bits(whole.predict_margin(&data).unwrap())
    );
}

/// The decomposition's identities: classification's knowledge uncertainty
/// is total minus data uncertainty with total at most `ln 2` (binary), and
/// a `dist:*` model's total is data plus knowledge uncertainty.
#[test]
fn uncertainty_decomposes_per_objective() {
    let binary = classification(300, 2);
    let params = |objective: Objective| {
        TrainingParams::builder()
            .objective(objective)
            .max_depth(3)
            .posterior_sampling(true)
            .build()
            .unwrap()
    };
    let model = train(
        &params(Objective::BinaryLogistic(Logistic::default())),
        &binary,
        40,
    )
    .unwrap();
    let u = model.predict_uncertainty(&binary, 10).unwrap();
    let (data, total) = (u.data.unwrap(), u.total.unwrap());
    let cells = |p: &Predictions<f64>| p.as_slice().to_vec();
    for ((&k, &d), &t) in cells(&u.knowledge)
        .iter()
        .zip(&cells(&data))
        .zip(&cells(&total))
    {
        assert!((k - (t - d)).abs() < 1e-15);
        assert!(k > -1e-12 && d >= 0.0 && t <= std::f64::consts::LN_2 + 1e-12);
    }
    assert!(u.mean.as_slice().iter().all(|&p| (0.0..=1.0).contains(&p)));

    let reg = regression(300);
    let dist = train(
        &params(Objective::Dist(Distributional::new(DistFamily::Normal))),
        &reg,
        40,
    )
    .unwrap();
    let u = dist.predict_uncertainty(&reg, 5).unwrap();
    let (data, total) = (u.data.unwrap(), u.total.unwrap());
    for ((&k, &d), &t) in cells(&u.knowledge)
        .iter()
        .zip(&cells(&data))
        .zip(&cells(&total))
    {
        assert!(k >= 0.0 && d > 0.0);
        assert_eq!(t, k + d);
    }

    let squared = train(&params(Objective::SquaredError), &reg, 40).unwrap();
    let u = squared.predict_uncertainty(&reg, 5).unwrap();
    assert!(u.data.is_none() && u.total.is_none());
    assert!(u.knowledge.as_slice().iter().all(|&k| k >= 0.0));
    // Too few iterations for the members, and no decomposition for ranking.
    assert_eq!(
        common::invalid_param(squared.predict_virtual_ensembles(&reg, 21)),
        "virtual_ensembles_count"
    );
    assert_eq!(
        common::invalid_param(squared.predict_virtual_ensembles(&reg, 0)),
        "virtual_ensembles_count"
    );
    assert_eq!(
        common::invalid_param(squared.predict_virtual_ensembles(&reg, usize::MAX)),
        "virtual_ensembles_count"
    );
    let members = squared.predict_virtual_ensembles(&reg, 5).unwrap();
    assert!(members.member_margins(4).is_some());
    assert!(members.member_margins(5).is_none());
    assert!(members.member_predictions(usize::MAX).is_none());
    assert!(members.member_margins(usize::MAX).is_none());
}

/// Each part of the decomposition has its own width: a multiclass model's
/// mean holds a probability per class and its uncertainties one value per
/// row; multi-label, multi-output, and distributional models keep theirs.
#[test]
fn uncertainty_parts_have_their_own_widths() {
    let params = |objective: Objective| {
        TrainingParams::builder()
            .objective(objective)
            .max_depth(2)
            .posterior_sampling(true)
            .build()
            .unwrap()
    };
    let labels = |i: usize| [(i % 2) as f32, ((i / 2) % 2) as f32];
    let x: Vec<f32> = (0..120).flat_map(common::four_features).collect();
    let multi_label = DMatrix::from_dense(&x, 120, 4)
        .unwrap()
        .with_label_matrix(&(0..120).flat_map(labels).collect::<Vec<_>>(), 2)
        .unwrap();
    let multiclass = classification(120, 3);
    let reg = regression(120);
    let cases = [
        (
            Objective::Softprob(Multiclass::new(3).unwrap()),
            &multiclass,
            3,
            1,
        ),
        (
            Objective::Softmax(Multiclass::new(3).unwrap()),
            &multiclass,
            3,
            1,
        ),
        (
            Objective::BinaryLogistic(Logistic::default()),
            &multi_label,
            2,
            2,
        ),
        (Objective::SquaredError, &multi_label, 2, 2),
        (
            Objective::Dist(Distributional::new(DistFamily::Normal)),
            &reg,
            1,
            1,
        ),
    ];
    for (objective, data, mean_width, width) in cases {
        let name = objective.name().to_string();
        let model = train(&params(objective), data, 20).unwrap();
        let u = model.predict_uncertainty(data, 5).unwrap();
        assert_eq!(
            (u.mean.n_rows(), u.mean.width()),
            (120, mean_width),
            "{name}"
        );
        let parts = [Some(&u.knowledge), u.data.as_ref(), u.total.as_ref()];
        for part in parts.into_iter().flatten() {
            assert_eq!((part.n_rows(), part.width()), (120, width), "{name}");
        }
    }
}
