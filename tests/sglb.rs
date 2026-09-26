//! SGLB (Langevin boosting with model shrinkage, CatBoost's
//! `posterior_sampling`) and virtual ensembles: truncations of a shrunk
//! model are the shorter training runs, bit for bit; the refusals; the
//! formats; and the uncertainty decomposition's invariants.

mod common;

use hessboost::config::{
    BoosterKind, ModelShrinkMode, Monotone, MultiStrategy, TrainingParamsBuilder,
};
use hessboost::prelude::*;

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

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
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
                .objective("binary:logistic")
                .langevin(true)
                .diffusion_temperature(50.0)
                .model_shrink_mode(ModelShrinkMode::Decreasing)
                .model_shrink_rate(0.3)
                .build()
                .unwrap(),
            classification(300, 2),
        ),
        (
            "approx multiclass posterior sampling",
            base()
                .tree_method(TreeMethod::Approx)
                .objective("multi:softprob")
                .num_class(3)
                .posterior_sampling(true)
                .build()
                .unwrap(),
            classification(300, 3),
        ),
        (
            "vector-leaf dist:normal posterior sampling",
            base()
                .tree_method(TreeMethod::Hist)
                .objective("dist:normal")
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
                .model_shrink_rate(0.5)
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
            let expected = bits(&short.predict_margin(data).unwrap());
            assert_eq!(
                bits(&long.predict_margin_range(data, ..k).unwrap()),
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
        bits(&model.predict_margin(&valid).unwrap()),
        bits(&short.predict_margin(&valid).unwrap())
    );
}

/// The Langevin draws are keyed by seed, iteration, and row or leaf, so the
/// model does not depend on the thread count; the seed does change it.
#[test]
fn langevin_training_is_thread_count_independent() {
    let data = regression(20_000);
    let params = |seed| {
        TrainingParams::builder()
            .objective("reg:squarederror")
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

/// Every format keeps a shrunk model's predictions, and the native ones
/// keep its truncations; an inconsistent shrinkage record is refused.
#[test]
fn shrunk_models_round_trip() {
    let data = regression(200);
    let params = TrainingParams::builder()
        .max_depth(3)
        .posterior_sampling(true)
        .build()
        .unwrap();
    let model = train(&params, &data, 12).unwrap();
    let margins = bits(&model.predict_margin(&data).unwrap());
    let prefix = bits(&model.predict_margin_range(&data, ..5).unwrap());
    for restored in [
        BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap(),
        BoostedModel::from_json(&model.to_json().unwrap()).unwrap(),
    ] {
        assert_eq!(bits(&restored.predict_margin(&data).unwrap()), margins);
        assert_eq!(
            bits(&restored.predict_margin_range(&data, ..5).unwrap()),
            prefix
        );
    }
    let xgboost = BoostedModel::from_xgboost_json(&model.to_xgboost_json().unwrap()).unwrap();
    assert_eq!(bits(&xgboost.predict_margin(&data).unwrap()), margins);
    let compact = model.to_compact().unwrap();
    assert_eq!(bits(&compact.predict_margin(&data).unwrap()), margins);
    // SHAP attributes the weighted trees: each row's contributions sum to
    // its margin.
    let contribs = model.predict_contribs(&data).unwrap();
    for (row, &m) in contribs
        .as_chunks::<5>()
        .0
        .iter()
        .zip(&model.predict_margin(&data).unwrap())
    {
        let sum: f32 = row.iter().sum();
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
        .model_shrink_rate(0.1)
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

/// Unsupported or conflicting SGLB settings are refused, never ignored.
#[test]
fn unsupported_combinations_are_refused() {
    let refused = |builder: TrainingParamsBuilder| common::invalid_param(builder.build());
    let base = TrainingParams::builder;
    assert_eq!(
        refused(base().posterior_sampling(true).diffusion_temperature(10.0)),
        "diffusion_temperature"
    );
    assert_eq!(
        refused(base().posterior_sampling(true).model_shrink_rate(0.01)),
        "model_shrink_rate"
    );
    assert_eq!(
        refused(
            base()
                .posterior_sampling(true)
                .model_shrink_mode(ModelShrinkMode::Decreasing)
        ),
        "model_shrink_mode"
    );
    assert_eq!(
        refused(base().diffusion_temperature(10.0)),
        "diffusion_temperature"
    );
    assert_eq!(
        refused(base().langevin(true).diffusion_temperature(0.0)),
        "diffusion_temperature"
    );
    assert_eq!(refused(base().model_shrink_rate(-0.1)), "model_shrink_rate");
    // The constant coefficient 1 - rate * eta must stay positive.
    assert_eq!(
        refused(base().eta(0.5).model_shrink_rate(2.0)),
        "model_shrink_rate"
    );
    assert_eq!(
        refused(base().model_shrink_mode(ModelShrinkMode::Decreasing)),
        "model_shrink_mode"
    );
    assert_eq!(
        refused(
            base()
                .model_shrink_mode(ModelShrinkMode::Decreasing)
                .model_shrink_rate(1.0)
        ),
        "model_shrink_rate"
    );
    assert_eq!(
        refused(base().langevin(true).booster(BoosterKind::Dart)),
        "langevin"
    );
    assert_eq!(
        refused(base().model_shrink_rate(0.1).booster(BoosterKind::GbLinear)),
        "model_shrink_rate"
    );
    assert_eq!(
        refused(base().langevin(true).num_parallel_tree(2)),
        "langevin"
    );
    assert_eq!(
        refused(
            base()
                .langevin(true)
                .monotone_constraints(vec![Monotone::Increasing])
        ),
        "langevin"
    );
    assert_eq!(refused(base().langevin(true).path_smooth(1.0)), "langevin");

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

/// Continued Langevin training (without shrinkage) grows the trees of the
/// uninterrupted run: its draws are keyed by the absolute iteration.
#[test]
fn langevin_continuation_matches_the_uninterrupted_run() {
    let data = regression(200);
    let params = TrainingParams::builder()
        .max_depth(3)
        .langevin(true)
        .model_shrink_rate(0.0)
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
        bits(&continued.predict_margin(&data).unwrap()),
        bits(&whole.predict_margin(&data).unwrap())
    );
}

/// The decomposition's identities: classification's knowledge uncertainty
/// is total minus data uncertainty with total at most `ln 2` (binary), and
/// a `dist:*` model's total is data plus knowledge uncertainty.
#[test]
fn uncertainty_decomposes_per_objective() {
    let binary = classification(300, 2);
    let params = |objective: &str| {
        TrainingParams::builder()
            .objective(objective)
            .max_depth(3)
            .posterior_sampling(true)
            .build()
            .unwrap()
    };
    let model = train(&params("binary:logistic"), &binary, 40).unwrap();
    let u = model.predict_uncertainty(&binary, 10).unwrap();
    let (data, total) = (u.data.unwrap(), u.total.unwrap());
    for ((&k, &d), &t) in u.knowledge.iter().zip(&data).zip(&total) {
        assert!((k - (t - d)).abs() < 1e-15);
        assert!(k > -1e-12 && d >= 0.0 && t <= std::f64::consts::LN_2 + 1e-12);
    }
    assert!(u.mean.iter().all(|&p| (0.0..=1.0).contains(&p)));

    let reg = regression(300);
    let dist = train(&params("dist:normal"), &reg, 40).unwrap();
    let u = dist.predict_uncertainty(&reg, 5).unwrap();
    let (data, total) = (u.data.unwrap(), u.total.unwrap());
    for ((&k, &d), &t) in u.knowledge.iter().zip(&data).zip(&total) {
        assert!(k >= 0.0 && d > 0.0);
        assert_eq!(t, k + d);
    }

    let squared = train(&params("reg:squarederror"), &reg, 40).unwrap();
    let u = squared.predict_uncertainty(&reg, 5).unwrap();
    assert!(u.data.is_none() && u.total.is_none());
    assert!(u.knowledge.iter().all(|&k| k >= 0.0));
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
