//! Continued training, `process_type=update`, `num_parallel_tree` forests,
//! model slicing and iteration-range prediction.

use hessboost::config::{BoosterKind, GrowPolicy, ProcessType, SamplingMethod, TreeMethod};
use hessboost::prelude::{BoostedModel, DMatrix, HessboostError, Trainer, TrainingParams, train};
use std::ops::Bound;

mod common;
use common::{invalid_param, labeled_dense, rmse};

/// Deterministic regression data: `n` rows, 4 features, a smooth target.
fn regression(n: usize, shift: f32) -> DMatrix {
    let mut x = Vec::with_capacity(n * 4);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let f: Vec<f32> = (0..4)
            .map(|j| ((i * (7 + 3 * j) + 11 * j) % 97) as f32 / 97.0)
            .collect();
        y.push(2.0 * f[0] - 3.0 * f[1] * f[1] + 0.5 * f[2] + shift);
        x.extend(f);
    }
    labeled_dense(&x, 4, &y)
}

/// [`regression`] with pseudo-random label noise, so boosting eventually
/// overfits a validation set drawn with different noise.
fn noisy(n: usize, salt: usize) -> DMatrix {
    let d = regression(n, 0.0);
    let y: Vec<f32> = d
        .labels()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, y)| y + ((i * (31 + salt)) % 23) as f32 / 23.0 - 0.5)
        .collect();
    d.with_labels(&y).unwrap()
}

/// Three-class labels on the same features.
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
        .objective("multi:softprob")
        .num_class(3)
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
                .booster(BoosterKind::Dart)
                .rate_drop(0.3)
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
        assert_eq!(resumed.to_bytes().unwrap(), whole.to_bytes().unwrap());
        // The input model is untouched.
        assert_eq!(first.num_boost_rounds(), 5);
    }
    // Continuing an XGBoost-JSON round trip of the model works the same way.
    let params = base().build().unwrap();
    let first = train(&params, &d, 4).unwrap();
    let imported = BoostedModel::from_xgboost_json(&first.to_xgboost_json().unwrap()).unwrap();
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
    assert_eq!(a.predict(&d).unwrap(), b.predict(&d).unwrap());
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
    let before = first.predict(&shifted).unwrap();
    let after = kept.predict(&shifted).unwrap();
    assert!(after.iter().zip(&before).all(|(a, b)| a > b));

    let explicit = base().base_score(1.5).build().unwrap();
    let replaced = Trainer::new(&explicit, &shifted, 2)
        .init_model(&first)
        .train()
        .unwrap()
        .model;
    assert_eq!(replaced.base_scores(), [1.5]);
}

#[test]
fn early_stopping_after_continuation_reports_absolute_iterations() {
    let d = noisy(300, 0);
    let valid = noisy(120, 6);
    let params = base().build().unwrap();
    let first = Trainer::new(&params, &d, 200)
        .eval(&valid, "valid")
        .early_stopping_rounds(2)
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
        .early_stopping_rounds(3)
        .train()
        .unwrap();
    assert_eq!(out.history[0].iteration, start);
    let chosen = out.model.best_iteration().expect("stops early");
    assert!(chosen >= start, "{chosen} < {start} (earlier best {best})");
    assert_eq!(
        out.model.predict(&valid).unwrap(),
        out.model.predict_range(&valid, ..=chosen).unwrap()
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
    let params = base().objective("survival:cox").build().unwrap();
    let first = train(&params, &train_set, 4).unwrap();
    let out = Trainer::new(&params, &train_set, 10)
        .init_model(&first)
        .eval(&censored, "censored")
        .early_stopping_rounds(2)
        .train()
        .unwrap();
    assert!(out.history.iter().all(|r| r.scores[0].2.is_nan()));
    assert_eq!(out.model.best_iteration(), Some(4));
    assert_eq!(
        out.model.predict(&train_set).unwrap(),
        out.model.predict_range(&train_set, ..5).unwrap()
    );
}

#[test]
fn incompatible_continuations_are_rejected() {
    let d = regression(100, 0.0);
    let first = train(&base().build().unwrap(), &d, 2).unwrap();
    let objective = base().objective("reg:pseudohubererror").build().unwrap();
    assert_eq!(
        invalid_param(Trainer::new(&objective, &d, 1).init_model(&first).train()),
        "objective"
    );
    let forest = base().num_parallel_tree(2).build().unwrap();
    assert_eq!(
        invalid_param(Trainer::new(&forest, &d, 1).init_model(&first).train()),
        "num_parallel_tree"
    );
    let linear = base().booster(BoosterKind::GbLinear).build().unwrap();
    assert_eq!(
        invalid_param(Trainer::new(&linear, &d, 1).init_model(&first).train()),
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
    let (a, b) = (resumed.predict(&d).unwrap(), whole.predict(&d).unwrap());
    assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-4));
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
            .process_type(ProcessType::Update)
            .build()
            .unwrap();
        let refreshed = Trainer::new(&update, &d, 6)
            .init_model(&model)
            .train()
            .unwrap()
            .model;
        let (a, b) = (model.predict(&d).unwrap(), refreshed.predict(&d).unwrap());
        assert!(
            a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-5),
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
        .process_type(ProcessType::Update)
        .refresh_leaf(false)
        .build()
        .unwrap();
    let stats_only = Trainer::new(&keep_leaves, &half, 6)
        .init_model(&model)
        .train()
        .unwrap()
        .model;
    assert_eq!(stats_only.predict(&d).unwrap(), model.predict(&d).unwrap());
    // Covers count the 150 refresh rows (squared error: hess 1 each).
    assert!(
        stats_only
            .trees()
            .iter()
            .all(|t| t.node(0).sum_hess == 150.0)
    );

    let update = base().process_type(ProcessType::Update).build().unwrap();
    let partial = Trainer::new(&update, &half, 4)
        .init_model(&model)
        .train()
        .unwrap()
        .model;
    assert_eq!(partial.num_boost_rounds(), 4);
    // The refreshed leaves chase the shifted labels.
    let shifted = partial.predict(&half).unwrap();
    let original = model.slice(..4, 1).unwrap().predict(&half).unwrap();
    assert!(shifted.iter().zip(&original).all(|(s, o)| s > o));

    assert_eq!(
        invalid_param(Trainer::new(&update, &half, 7).init_model(&model).train()),
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
    let update = || base().process_type(ProcessType::Update);
    let linear = train(&base().linear_tree(true).build().unwrap(), &d, 3).unwrap();
    assert!(refused(&update().build().unwrap(), &linear));
    let plain = train(&base().build().unwrap(), &d, 3).unwrap();
    for params in [
        update().linear_tree(true),
        update().path_smooth(1.0),
        // Refresh sums full-precision gradients, so quantization would be
        // silently skipped.
        update().use_quantized_grad(true),
        // Refresh keeps the existing splits, so split-search-only options
        // would silently do nothing.
        update().extra_trees(true),
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
        update().rate_drop(0.5),
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
        .objective("multi:softprob")
        .num_class(3)
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
            .objective("multi:softprob")
            .num_class(3)
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
        .objective("multi:softprob")
        .num_class(3)
        .num_parallel_tree(2)
        .build()
        .unwrap();
    let model = train(&params, &d, 6).unwrap();
    let prefix = train(&params, &d, 4).unwrap();
    assert_eq!(
        model.predict_margin_range(&d, ..4).unwrap(),
        prefix.predict_margin(&d).unwrap()
    );
    assert_eq!(
        model.predict_range(&d, ..).unwrap(),
        model.predict(&d).unwrap()
    );
    let contribs = model.predict_contribs_range(&d, ..4).unwrap();
    assert_eq!(contribs, prefix.predict_contribs(&d).unwrap());
    let leaves = model.predict_leaf_range(&d, ..4).unwrap();
    assert_eq!(leaves, prefix.predict_leaf(&d).unwrap());
    let inter = model.predict_interactions_range(&d, ..2).unwrap();
    assert_eq!(
        inter,
        model
            .slice(..2, 1)
            .unwrap()
            .predict_interactions(&d)
            .unwrap()
    );

    // A middle range keeps the intercept and adds only its trees.
    let middle = model.predict_margin_range(&d, 2..5).unwrap();
    let sliced = model.slice(2..5, 1).unwrap().predict_margin(&d).unwrap();
    let diff = middle
        .iter()
        .zip(&sliced)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(diff < 1e-5, "{diff}");

    let bad = |r| invalid_param::<Vec<f32>>(r) == "iterations";
    assert!(bad(model.predict_margin_range(&d, ..7)));
    assert!(bad(model.predict_margin_range(
        &d,
        (Bound::Included(4), Bound::Excluded(3))
    )));
    // Attributions and leaves, as in XGBoost, only take prefixes.
    assert!(bad(model.predict_contribs_range(&d, 1..3)));
    assert_eq!(
        invalid_param(model.predict_leaf_range(&d, 1..)),
        "iterations"
    );
}

#[test]
fn gblinear_accepts_only_the_whole_range() {
    let d = regression(100, 0.0);
    let params = base().booster(BoosterKind::GbLinear).build().unwrap();
    let model = train(&params, &d, 5).unwrap();
    let whole = model.predict_margin(&d).unwrap();
    assert_eq!(model.predict_margin_range(&d, ..).unwrap(), whole);
    assert_eq!(model.predict_margin_range(&d, 0..).unwrap(), whole);
    // An explicit empty range would predict the intercept alone on a tree
    // model; a linear model has no iterations to leave out, so it is refused.
    for bad in [
        model.predict_margin_range(&d, 0..0),
        model.predict_range(&d, ..0),
    ] {
        assert_eq!(invalid_param(bad), "iterations");
    }
    assert_eq!(invalid_param(model.slice(.., 1)), "slice");
}

#[test]
fn slicing_selects_iterations_with_their_dart_weights() {
    let d = regression(200, 0.0);
    let params = base()
        .booster(BoosterKind::Dart)
        .rate_drop(0.4)
        .num_parallel_tree(2)
        .build()
        .unwrap();
    let model = train(&params, &d, 9).unwrap();
    let sliced = model.slice(1..8, 3).unwrap();
    assert_eq!(sliced.num_boost_rounds(), 3);
    assert_eq!(sliced.base_scores(), model.base_scores());
    // Iterations 1, 4, 7 contribute exactly their weighted trees.
    let pick = |m: &BoostedModel, it: usize| {
        let full = m.predict_margin_range(&d, it..=it).unwrap();
        full.iter().map(|v| v - m.base_score()).collect::<Vec<_>>()
    };
    let picked = [1, 4, 7].map(|it| pick(&model, it));
    let expected: Vec<f32> = (0..d.n_rows())
        .map(|r| model.base_score() + picked.iter().map(|p| p[r]).sum::<f32>())
        .collect();
    let got = sliced.predict_margin(&d).unwrap();
    assert!(got.iter().zip(&expected).all(|(a, b)| (a - b).abs() < 1e-5));
    for (s, it) in [(0, 1), (1, 4), (2, 7)] {
        assert_eq!(pick(&sliced, s), pick(&model, it));
    }

    let d = noisy(200, 0);
    let valid = noisy(80, 6);
    let stopped = Trainer::new(&base().build().unwrap(), &d, 200)
        .eval(&valid, "v")
        .early_stopping_rounds(2)
        .train()
        .unwrap()
        .model;
    let best = stopped.best_iteration().unwrap();
    let head = stopped.slice(..=best, 1).unwrap();
    assert_eq!(head.best_iteration(), None);
    assert_eq!(
        head.predict(&valid).unwrap(),
        stopped.predict(&valid).unwrap()
    );

    for (b, e, s) in [(0, 0, 0), (3, 3, 1), (0, 10, 1), (5, 2, 1)] {
        assert_eq!(invalid_param(model.slice(b..e, s)), "slice");
    }
}
