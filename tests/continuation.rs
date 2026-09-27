//! Continued training, `process_type=update`, `num_parallel_tree` forests,
//! model slicing and iteration-range prediction.

use hessboost::config::{
    BoosterKind, Dart, ExtraTrees, GrowPolicy, LinearTree, ProcessType, QuantizedGrad, Refresh,
    SamplingMethod, TreeMethod,
};
use hessboost::model::Iterations;
use hessboost::model::ModelFormat;
use hessboost::objective::{Multiclass, PseudoHuber};
use hessboost::prelude::{
    BoostedModel, DMatrix, HessboostError, Objective, Trainer, TrainingParams, train,
};
use std::num::NonZeroUsize;
use std::ops::Bound;

mod common;
use common::smooth::{continuation_noisy as noisy, regression};
use common::{incompatible_model, invalid_param, labeled_dense, rmse};
fn multiclass(n: usize) -> DMatrix {
    let d = regression(n, 0.0);
    let y: Vec<f32> = (0..n)
        .map(|i| ((d.get(i, 0).unwrap() * 3.0) as u32).min(2) as f32)
        .collect();
    d.with_labels(&y).unwrap()
}

fn base() -> hessboost::config::TrainingParamsBuilder {
    TrainingParams::builder()
        .tree_method(TreeMethod::Hist)
        .max_depth(3)
        .eta(0.3)
        .seed(7)
}

#[test]
fn continuing_grows_the_same_trees_as_training_in_one_run() {
    let d = regression(300, 0.0);
    let softprob = base()
        .objective(Objective::Softprob(Multiclass::new(3).unwrap()))
        .subsample(0.7)
        .colsample_bytree(0.6)
        .num_parallel_tree(2)
        .build()
        .unwrap();
    let cases = [
        (
            base().subsample(0.7).colsample_bynode(0.5).build().unwrap(),
            d.clone(),
        ),
        (
            base()
                .booster(BoosterKind::Dart(
                    Dart::builder().rate_drop(0.3).build().unwrap(),
                ))
                .build()
                .unwrap(),
            d.clone(),
        ),
        (
            base().tree_method(TreeMethod::Exact).build().unwrap(),
            d.clone(),
        ),
        (softprob, multiclass(300)),
    ];
    for (params, data) in cases {
        let whole = train(&params, &data, 12).unwrap();
        let first = train(&params, &data, 5).unwrap();
        let resumed = Trainer::new(&params, &data, 7)
            .init_model(&first)
            .train()
            .unwrap()
            .model;
        assert_eq!(resumed.num_boost_rounds(), 12);
        assert_eq!(
            resumed.encode(ModelFormat::Binary).unwrap(),
            whole.encode(ModelFormat::Binary).unwrap()
        );
        // The input model is untouched.
        assert_eq!(first.num_boost_rounds(), 5);
    }
    // Continuing an XGBoost-JSON round trip of the model works the same way.
    let params = base().build().unwrap();
    let first = train(&params, &d, 4).unwrap();
    let imported = BoostedModel::decode(
        first.encode(ModelFormat::XgboostJson).unwrap(),
        ModelFormat::XgboostJson,
    )
    .unwrap();
    let a = Trainer::new(&params, &d, 3)
        .init_model(&first)
        .train()
        .unwrap()
        .model;
    let b = Trainer::new(&params, &d, 3)
        .init_model(&imported)
        .train()
        .unwrap()
        .model;
    assert_eq!(
        a.predict(&d, Iterations::Best).unwrap(),
        b.predict(&d, Iterations::Best).unwrap()
    );
}

#[test]
fn continuation_keeps_the_intercept_unless_base_score_is_given() {
    let d = regression(200, 0.0);
    let shifted = regression(200, 5.0);
    let params = base().build().unwrap();
    let first = train(&params, &d, 3).unwrap();
    let kept = Trainer::new(&params, &shifted, 2)
        .init_model(&first)
        .train()
        .unwrap()
        .model;
    assert_eq!(kept.base_scores(), first.base_scores());
    // The new rounds fit the shifted labels from the old margins.
    let before = first.predict(&shifted, Iterations::Best).unwrap();
    let after = kept.predict(&shifted, Iterations::Best).unwrap();
    assert!(
        after
            .as_slice()
            .iter()
            .zip(before.as_slice())
            .all(|(a, b)| a > b)
    );

    let explicit = base().base_score(1.5).build().unwrap();
    let replaced = Trainer::new(&explicit, &shifted, 2)
        .init_model(&first)
        .train()
        .unwrap()
        .model;
    assert_eq!(replaced.base_scores(), [1.5]);
}

/// Early stopping records the best iteration and score even when the round
/// limit comes before patience runs out (XGBoost's `best_iteration`).
#[test]
fn early_stopping_records_the_best_round_without_stopping() {
    let d = noisy(300, 0);
    let valid = noisy(120, 6);
    let params = base().build().unwrap();
    let stopped = Trainer::new(&params, &d, 200)
        .eval(&valid, "valid")
        .early_stopping_rounds(NonZeroUsize::new(2).unwrap())
        .train()
        .unwrap();
    let best = stopped.model.best_iteration().expect("stops early");
    // Rounds `best + 1` and `best + 2` did not improve; stop at the same
    // round count with patience to spare.
    assert_eq!(stopped.history.len(), best + 3);
    let out = Trainer::new(&params, &d, best + 3)
        .eval(&valid, "valid")
        .early_stopping_rounds(NonZeroUsize::new(50).unwrap())
        .train()
        .unwrap();
    assert_eq!(out.model.num_boost_rounds(), best + 3);
    assert_eq!(out.model.best_iteration(), Some(best));
    assert_eq!(
        out.best_score,
        Some(out.history.round(best).unwrap().values()[0])
    );
    assert_eq!(stopped.best_score, out.best_score);
    assert_eq!(
        out.model.predict(&valid, Iterations::Best).unwrap(),
        out.model.predict(&valid, ..=best).unwrap()
    );
    // Without early stopping nothing is selected.
    let plain = Trainer::new(&params, &d, best + 3)
        .eval(&valid, "valid")
        .train()
        .unwrap();
    assert_eq!(plain.model.best_iteration(), None);
    assert_eq!(plain.best_score, None);
}

#[test]
fn early_stopping_after_continuation_reports_absolute_iterations() {
    let d = noisy(300, 0);
    let valid = noisy(120, 6);
    let params = base().build().unwrap();
    let first = Trainer::new(&params, &d, 200)
        .eval(&valid, "valid")
        .early_stopping_rounds(NonZeroUsize::new(2).unwrap())
        .train()
        .unwrap();
    let first = first.model;
    let best = first.best_iteration().expect("stops early");
    let resumed = Trainer::new(&params, &d, 3)
        .init_model(&first)
        .train()
        .unwrap()
        .model;
    // The stale selection is dropped so the new iterations take part.
    assert_eq!(resumed.best_iteration(), None);
    assert_eq!(resumed.num_boost_rounds(), first.num_boost_rounds() + 3);

    let start = first.num_boost_rounds();
    let out = Trainer::new(&params, &d, 200)
        .init_model(&first)
        .eval(&valid, "valid")
        .early_stopping_rounds(NonZeroUsize::new(3).unwrap())
        .train()
        .unwrap();
    assert_eq!(out.history.first_iteration(), start);
    let chosen = out.model.best_iteration().expect("stops early");
    assert!(chosen >= start, "{chosen} < {start} (earlier best {best})");
    assert_eq!(
        out.model.predict(&valid, Iterations::Best).unwrap(),
        out.model.predict(&valid, ..=chosen).unwrap()
    );
}

/// A continuation whose early-stopping metric never improves (here NaN:
/// `cox-nloglik` of an all-censored set) keeps the first continued
/// iteration, never an iteration of the initial model, so `predict` still
/// uses every tree of the resumed model.
#[test]
fn continuation_without_an_improving_metric_keeps_the_initial_model() {
    let d = regression(200, 0.0);
    let times: Vec<f32> = d.labels().unwrap().iter().map(|y| y.abs() + 1.0).collect();
    let train_set = d.clone().with_labels(&times).unwrap();
    let censored = d
        .with_labels(&times.iter().map(|t| -t).collect::<Vec<_>>())
        .unwrap();
    let params = base().objective(Objective::Cox).build().unwrap();
    let first = train(&params, &train_set, 4).unwrap();
    let out = Trainer::new(&params, &train_set, 10)
        .init_model(&first)
        .eval(&censored, "censored")
        .early_stopping_rounds(NonZeroUsize::new(2).unwrap())
        .train()
        .unwrap();
    assert!(out.history.rounds().all(|r| r.values()[0].is_nan()));
    assert_eq!(out.model.best_iteration(), Some(4));
    assert_eq!(
        out.model.predict(&train_set, Iterations::Best).unwrap(),
        out.model.predict(&train_set, ..5).unwrap()
    );
}

#[test]
fn incompatible_continuations_are_rejected() {
    let d = regression(100, 0.0);
    let first = train(&base().build().unwrap(), &d, 2).unwrap();
    let objective = base()
        .objective(Objective::PseudoHuber(PseudoHuber::default()))
        .build()
        .unwrap();
    assert_eq!(
        incompatible_model(Trainer::new(&objective, &d, 1).init_model(&first).train()),
        "objective"
    );
    let forest = base().num_parallel_tree(2).build().unwrap();
    assert_eq!(
        incompatible_model(Trainer::new(&forest, &d, 1).init_model(&first).train()),
        "num_parallel_tree"
    );
    let linear = base().booster(BoosterKind::GbLinear).build().unwrap();
    assert_eq!(
        incompatible_model(Trainer::new(&linear, &d, 1).init_model(&first).train()),
        "booster"
    );
    let narrow = labeled_dense(&[0.5; 30], 3, &[0.0; 10]);
    assert!(matches!(
        Trainer::new(&base().build().unwrap(), &narrow, 1)
            .init_model(&first)
            .train(),
        Err(HessboostError::DimensionMismatch { .. })
    ));
}

#[test]
fn gblinear_continues_from_its_weights() {
    let d = regression(200, 0.0);
    let params = base()
        .booster(BoosterKind::GbLinear)
        .eta(0.5)
        .build()
        .unwrap();
    let first = train(&params, &d, 3).unwrap();
    let resumed = Trainer::new(&params, &d, 5)
        .init_model(&first)
        .train()
        .unwrap()
        .model;
    let whole = train(&params, &d, 8).unwrap();
    // Coordinate descent resumes: agrees with the uninterrupted run up to
    // the f32 margin rounding of recomputing predictions.
    let (a, b) = (
        resumed.predict(&d, Iterations::Best).unwrap(),
        whole.predict(&d, Iterations::Best).unwrap(),
    );
    assert!(
        a.as_slice()
            .iter()
            .zip(b.as_slice())
            .all(|(x, y)| (x - y).abs() < 1e-4)
    );
    assert!(rmse(&resumed, &d) < rmse(&first, &d));
}

#[test]
fn refreshing_on_the_training_data_reproduces_the_model() {
    let d = regression(400, 0.0);
    for method in [TreeMethod::Hist, TreeMethod::Exact] {
        let params = base().tree_method(method).build().unwrap();
        let model = train(&params, &d, 6).unwrap();
        let update = base()
            .tree_method(method)
            .process_type(ProcessType::Update(Refresh::default()))
            .build()
            .unwrap();
        let refreshed = Trainer::new(&update, &d, 6)
            .init_model(&model)
            .train()
            .unwrap()
            .model;
        let (a, b) = (
            model.predict(&d, Iterations::Best).unwrap(),
            refreshed.predict(&d, Iterations::Best).unwrap(),
        );
        assert!(
            a.as_slice()
                .iter()
                .zip(b.as_slice())
                .all(|(x, y)| (x - y).abs() < 1e-5),
            "{method:?}"
        );
        for (old, new) in model.trees().iter().zip(refreshed.trees()) {
            for (o, n) in old.nodes().iter().zip(new.nodes()) {
                assert_eq!(
                    (o.split_feature, o.split_cond),
                    (n.split_feature, n.split_cond)
                );
                assert!((o.sum_hess - n.sum_hess).abs() <= 1e-3 * o.sum_hess.max(1.0));
            }
        }
    }
}

#[test]
fn refresh_on_new_data_recomputes_statistics_and_truncates() {
    let d = regression(300, 0.0);
    // The first 150 rows of `d`, with labels shifted by 3.
    let half = regression(150, 3.0);
    let model = train(&base().build().unwrap(), &d, 6).unwrap();
    let keep_leaves = base()
        .process_type(ProcessType::Update(Refresh::stats_only()))
        .build()
        .unwrap();
    let stats_only = Trainer::new(&keep_leaves, &half, 6)
        .init_model(&model)
        .train()
        .unwrap()
        .model;
    assert_eq!(
        stats_only.predict(&d, Iterations::Best).unwrap(),
        model.predict(&d, Iterations::Best).unwrap()
    );
    // Covers count the 150 refresh rows (squared error: hess 1 each).
    assert!(
        stats_only
            .trees()
            .iter()
            .all(|t| t.node(0).sum_hess == 150.0)
    );

    let update = base()
        .process_type(ProcessType::Update(Refresh::default()))
        .build()
        .unwrap();
    let partial = Trainer::new(&update, &half, 4)
        .init_model(&model)
        .train()
        .unwrap()
        .model;
    assert_eq!(partial.num_boost_rounds(), 4);
    // The refreshed leaves chase the shifted labels.
    let shifted = partial.predict(&half, Iterations::Best).unwrap();
    let original = model
        .slice(..4, 1)
        .unwrap()
        .predict(&half, Iterations::Best)
        .unwrap();
    assert!(
        shifted
            .as_slice()
            .iter()
            .zip(original.as_slice())
            .all(|(s, o)| s > o)
    );

    assert_eq!(
        incompatible_model(Trainer::new(&update, &half, 7).init_model(&model).train()),
        "num_boost_round"
    );
    assert_eq!(invalid_param(train(&update, &d, 1)), "process_type");
}

/// The refresh updater keeps the splits, sums every row, and recomputes
/// constant leaves from full-precision gradients only: a model with linear
/// leaves, or a refresh configured with anything refresh does not read
/// (linear leaves, path smoothing, quantized gradients, split-search
/// options, row or column sampling, symmetric growth, DART dropout), is
/// refused instead of silently refreshing to a different model or ignoring
/// the option. XGBoost's tree-shape settings, which describe the refreshed
/// trees, are accepted as XGBoost's refresh updater accepts them.
#[test]
fn refresh_refuses_options_it_cannot_apply() {
    let d = regression(300, 0.0);
    let refused = |params: &TrainingParams, model: &BoostedModel| {
        invalid_param(Trainer::new(params, &d, 2).init_model(model).train()) == "process_type"
    };
    let update = || base().process_type(ProcessType::Update(Refresh::default()));
    let linear = train(
        &base().linear_tree(LinearTree::default()).build().unwrap(),
        &d,
        3,
    )
    .unwrap();
    assert_eq!(
        incompatible_model(
            Trainer::new(&update().build().unwrap(), &d, 2)
                .init_model(&linear)
                .train()
        ),
        "process_type"
    );
    let plain = train(&base().build().unwrap(), &d, 3).unwrap();
    for params in [
        update().linear_tree(LinearTree::default()),
        update().path_smooth(1.0),
        // Refresh sums full-precision gradients, so quantization would be
        // silently skipped.
        update().quantized(QuantizedGrad::default()),
        // Refresh keeps the existing splits, so split-search-only options
        // would silently do nothing.
        update().extra_trees(ExtraTrees::default()),
        update().toad_penalty_feature(1.0),
        update().toad_penalty_threshold(0.5),
        // Refresh sums every row over every existing split: it samples
        // neither rows nor columns and grows nothing.
        update().subsample(0.5),
        update()
            .sampling_method(SamplingMethod::GradientBased)
            .subsample(0.25),
        update().colsample_bytree(0.5),
        update().colsample_bynode(0.5),
        update().grow_policy(GrowPolicy::Symmetric),
        update().booster(BoosterKind::Dart(
            Dart::builder().rate_drop(0.5).build().unwrap(),
        )),
    ] {
        assert!(refused(&params.build().unwrap(), &plain));
    }
    for params in [
        update(),
        update()
            .tree_method(TreeMethod::Approx)
            .max_depth(5)
            .max_leaves(8)
            .grow_policy(GrowPolicy::LossGuide)
            .min_child_weight(2.0)
            .gamma(0.5)
            .max_bin(64)
            .interaction_constraints(vec![vec![0, 1]])
            .lambda(2.0)
            .alpha(0.5)
            .eta(0.1)
            .nthread(2)
            .seed(99),
    ] {
        let params = params.build().unwrap();
        assert!(
            Trainer::new(&params, &d, 2)
                .init_model(&plain)
                .train()
                .is_ok()
        );
    }
}

#[test]
fn parallel_trees_form_one_iteration_and_share_the_learning_rate() {
    let d = multiclass(240);
    let forest = base()
        .objective(Objective::Softprob(Multiclass::new(3).unwrap()))
        .num_parallel_tree(3)
        .build()
        .unwrap();
    let model = train(&forest, &d, 4).unwrap();
    assert_eq!((model.num_trees(), model.num_boost_rounds()), (36, 4));
    assert_eq!(model.trees_per_iteration(), 9);
    // Without sampling the parallel trees of one output are identical, each
    // carrying a third of the single tree's step.
    let single = train(
        &base()
            .objective(Objective::Softprob(Multiclass::new(3).unwrap()))
            .build()
            .unwrap(),
        &d,
        1,
    )
    .unwrap();
    for k in 0..3 {
        let group = &model.trees()[3 * k..3 * k + 3];
        assert!(group.iter().all(|t| t == &group[0]));
        for (f, s) in group[0].nodes().iter().zip(single.trees()[k].nodes()) {
            if f.is_leaf() {
                assert!((f.leaf_value * 3.0 - s.leaf_value).abs() <= 1e-6 * s.leaf_value.abs());
            }
        }
    }
    // Sampled forests draw each parallel tree independently.
    let sampled = base().subsample(0.6).num_parallel_tree(3).build().unwrap();
    let rf = train(&sampled, &regression(240, 0.0), 1).unwrap();
    assert!(rf.trees()[0] != rf.trees()[1] && rf.trees()[1] != rf.trees()[2]);
}

#[test]
fn iteration_ranges_select_whole_iterations() {
    let d = multiclass(200);
    let params = base()
        .objective(Objective::Softprob(Multiclass::new(3).unwrap()))
        .num_parallel_tree(2)
        .build()
        .unwrap();
    let model = train(&params, &d, 6).unwrap();
    let prefix = train(&params, &d, 4).unwrap();
    assert_eq!(
        model.predict_margin(&d, ..4).unwrap(),
        prefix.predict_margin(&d, Iterations::Best).unwrap()
    );
    assert_eq!(
        model.predict(&d, ..).unwrap(),
        model.predict(&d, Iterations::Best).unwrap()
    );
    let contribs = model.predict_contribs(&d, ..4).unwrap();
    assert_eq!(
        contribs,
        prefix.predict_contribs(&d, Iterations::Best).unwrap()
    );
    let leaves = model.predict_leaf(&d, ..4).unwrap();
    assert_eq!(leaves, prefix.predict_leaf(&d, ..).unwrap());
    let inter = model.predict_interactions(&d, ..2).unwrap();
    assert_eq!(
        inter,
        model
            .slice(..2, 1)
            .unwrap()
            .predict_interactions(&d, Iterations::Best)
            .unwrap()
    );

    // A middle range keeps the intercept and adds only its trees.
    let middle = model.predict_margin(&d, 2..5).unwrap();
    let sliced = model
        .slice(2..5, 1)
        .unwrap()
        .predict_margin(&d, Iterations::Best)
        .unwrap();
    let diff = middle
        .as_slice()
        .iter()
        .zip(sliced.as_slice())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(diff < 1e-5, "{diff}");

    assert_eq!(
        incompatible_model(model.predict_margin(&d, ..7)),
        "iterations"
    );
    assert_eq!(
        invalid_param(model.predict_margin(&d, (Bound::Included(4), Bound::Excluded(3)))),
        "iterations"
    );
    // Attributions and leaves, as in XGBoost, only take prefixes.
    assert_eq!(
        invalid_param(model.predict_contribs(&d, 1..3)),
        "iterations"
    );
    assert_eq!(invalid_param(model.predict_leaf(&d, 1..)), "iterations");
}

#[test]
fn gblinear_accepts_only_the_whole_range() {
    let d = regression(100, 0.0);
    let params = base().booster(BoosterKind::GbLinear).build().unwrap();
    let model = train(&params, &d, 5).unwrap();
    let whole = model.predict_margin(&d, Iterations::Best).unwrap();
    assert_eq!(model.predict_margin(&d, ..).unwrap(), whole);
    assert_eq!(model.predict_margin(&d, 0..).unwrap(), whole);
    // An explicit empty range would predict the intercept alone on a tree
    // model; a linear model has no iterations to leave out, so it is refused.
    for bad in [model.predict_margin(&d, 0..0), model.predict(&d, ..0)] {
        assert_eq!(incompatible_model(bad), "iterations");
    }
    assert_eq!(incompatible_model(model.slice(.., 1)), "slice");
}

#[test]
fn slicing_selects_iterations_with_their_dart_weights() {
    let d = regression(200, 0.0);
    let params = base()
        .booster(BoosterKind::Dart(
            Dart::builder().rate_drop(0.4).build().unwrap(),
        ))
        .num_parallel_tree(2)
        .build()
        .unwrap();
    let model = train(&params, &d, 9).unwrap();
    let sliced = model.slice(1..8, 3).unwrap();
    assert_eq!(sliced.num_boost_rounds(), 3);
    assert_eq!(sliced.base_scores(), model.base_scores());
    // Iterations 1, 4, 7 contribute exactly their weighted trees.
    let pick = |m: &BoostedModel, it: usize| {
        let full = m.predict_margin(&d, it..=it).unwrap();
        full.as_slice()
            .iter()
            .map(|v| v - m.base_score())
            .collect::<Vec<_>>()
    };
    let picked = [1, 4, 7].map(|it| pick(&model, it));
    let expected: Vec<f32> = (0..d.n_rows())
        .map(|r| model.base_score() + picked.iter().map(|p| p[r]).sum::<f32>())
        .collect();
    let got = sliced.predict_margin(&d, Iterations::Best).unwrap();
    assert!(
        got.as_slice()
            .iter()
            .zip(&expected)
            .all(|(a, b)| (a - b).abs() < 1e-5)
    );
    for (s, it) in [(0, 1), (1, 4), (2, 7)] {
        assert_eq!(pick(&sliced, s), pick(&model, it));
    }

    let d = noisy(200, 0);
    let valid = noisy(80, 6);
    let stopped = Trainer::new(&base().build().unwrap(), &d, 200)
        .eval(&valid, "v")
        .early_stopping_rounds(NonZeroUsize::new(2).unwrap())
        .train()
        .unwrap()
        .model;
    let best = stopped.best_iteration().unwrap();
    let head = stopped.slice(..=best, 1).unwrap();
    assert_eq!(head.best_iteration(), None);
    assert_eq!(
        head.predict(&valid, Iterations::Best).unwrap(),
        stopped.predict(&valid, Iterations::Best).unwrap()
    );

    // A step of 0, an empty range, and an inverted range are bad arguments;
    // a range past the model's iterations does not fit the model.
    for (b, e, s) in [(0, 0, 0), (3, 3, 1), (5, 2, 1)] {
        assert_eq!(invalid_param(model.slice(b..e, s)), "slice");
    }
    assert_eq!(incompatible_model(model.slice(0..10, 1)), "slice");
}
