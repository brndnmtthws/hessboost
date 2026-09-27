//! `multi_strategy = multi_output_tree` (vector-leaf trees): prediction
//! layout across the kernel paths, SHAP additivity, format round trips,
//! reduced split gradients, and configuration errors.

use hessboost::config::{
    BoosterKind, Dart, GrowPolicy, Monotone, MultiStrategy, ProcessType, Refresh, TreeMethod,
};
use hessboost::data::FeatureType;
use hessboost::model::{Contributions, Predictions};
use hessboost::objective::{CustomLoss, GradPair, SplitGradient};
use hessboost::prelude::*;
use std::num::NonZeroUsize;

mod common;
use common::{four_features, invalid_param, labeled_dense, rmse};

const N: usize = 300;
const COLS: usize = 4;
const K: usize = 3;

/// [`four_features`] rows and three targets that depend on different columns.
fn data() -> (Vec<f32>, Vec<f32>) {
    let mut x = Vec::with_capacity(N * COLS);
    let mut y = Vec::with_capacity(N * K);
    for i in 0..N {
        let [a, b, c, d] = four_features(i);
        x.extend([a, b, c, d]);
        y.extend([
            2.0 * a - b,
            (6.0 * c).sin(),
            if d.is_nan() { 1.0 } else { d * a },
        ]);
    }
    (x, y)
}

fn dtrain() -> DMatrix {
    let (x, y) = data();
    DMatrix::from_dense(&x, N, COLS)
        .unwrap()
        .with_label_matrix(&y, K)
        .unwrap()
}

fn vector_params() -> hessboost::config::TrainingParamsBuilder {
    TrainingParams::builder()
        .multi_strategy(MultiStrategy::MultiOutputTree)
        .max_depth(4)
        .eta(0.3)
}

fn model() -> BoostedModel {
    train(&vector_params().build().unwrap(), &dtrain(), 8).unwrap()
}

/// `base_score + Σ_t leaf_vector(t, row)` computed from the trees directly.
fn reference_margins(model: &BoostedModel, x: &[f32], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * K);
    for r in 0..n {
        let row = &x[r * COLS..(r + 1) * COLS];
        let mut m = model.base_scores().to_vec();
        for tree in model.trees() {
            let leaf = tree.leaf_id_dense(row, f32::NAN);
            for (o, v) in m.iter_mut().zip(tree.leaf_vector(leaf)) {
                *o += v;
            }
        }
        out.extend(m);
    }
    out
}

/// Every row-and-output's contributions (bias included) sum to its margin.
fn assert_contribs_sum_to(contribs: &Contributions, margins: &Predictions) {
    assert_eq!(
        (
            contribs.n_rows(),
            contribs.n_outputs(),
            contribs.n_features()
        ),
        (margins.n_rows(), margins.width(), COLS)
    );
    for (r, row) in margins.rows().enumerate() {
        for (o, m) in row.iter().enumerate() {
            let sum: f32 = contribs.get(r, o).unwrap().iter().sum();
            assert!((sum - m).abs() < 1e-4, "{sum} vs {m}");
        }
    }
}

#[test]
fn one_vector_tree_per_round_predicts_every_output() {
    let model = model();
    assert!(model.has_vector_leaves());
    assert_eq!(model.num_trees(), 8);
    assert_eq!(model.num_boost_rounds(), 8);
    assert!(model.trees().iter().all(|t| t.size_leaf_vector() == K));
    let (x, _) = data();
    let want = reference_margins(&model, &x, N);
    // Block kernels (lockstep groups plus a tail) and the small-batch path.
    let d = DMatrix::from_dense(&x, N, COLS).unwrap();
    assert_eq!(
        model
            .predict_margin(&d, Iterations::Best)
            .unwrap()
            .as_slice(),
        want
    );
    let few = DMatrix::from_dense(&x[..5 * COLS], 5, COLS).unwrap();
    assert_eq!(
        model
            .predict_margin(&few, Iterations::Best)
            .unwrap()
            .as_slice(),
        &want[..5 * K]
    );
    // CSR rows (missing entries absent) take the scratch-block path.
    let (mut indptr, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
    for r in 0..N {
        for c in 0..COLS {
            let v = x[r * COLS + c];
            if !v.is_nan() {
                indices.push(c as u32);
                values.push(v);
            }
        }
        indptr.push(indices.len());
    }
    let csr = DMatrix::from_csr(indptr, indices, values, COLS).unwrap();
    assert_eq!(
        model
            .predict_margin(&csr, Iterations::Best)
            .unwrap()
            .as_slice(),
        want
    );
    // Squared error: predictions are the margins, `[row][output]`.
    assert_eq!(
        model.predict(&d, Iterations::Best).unwrap().as_slice(),
        want
    );
}

#[test]
fn training_margins_match_the_final_model() {
    // The loss after each round comes from the incrementally updated margin
    // caches; the final eval RMSE must equal the model's own prediction.
    let dtrain = dtrain();
    let result = Trainer::new(
        &vector_params().subsample(0.7).seed(3).build().unwrap(),
        &dtrain,
        6,
    )
    .eval(&dtrain, "train")
    .train()
    .unwrap();
    let rmse = rmse(&result.model, &dtrain);
    let last = result.history.last().unwrap().values()[0];
    assert!((last - rmse).abs() < 1e-6, "history {last} vs model {rmse}");
}

#[test]
fn shap_is_additive_per_output() {
    let model = model();
    let (x, _) = data();
    let n = 40;
    let d = DMatrix::from_dense(&x[..n * COLS], n, COLS).unwrap();
    let margin = model.predict_margin(&d, Iterations::Best).unwrap();
    let width = COLS + 1;
    let contribs = model.predict_contribs(&d, Iterations::Best).unwrap();
    assert_contribs_sum_to(&contribs, &margin);
    let inter = model.predict_interactions(&d, Iterations::Best).unwrap();
    assert_eq!(
        (inter.n_rows(), inter.n_outputs(), inter.n_features()),
        (n, K, COLS)
    );
    for r in 0..n {
        for o in 0..K {
            for (i, &p) in contribs.get(r, o).unwrap().iter().enumerate() {
                let sum: f32 = (0..width).map(|j| inter.at(r, o, i, j).unwrap()).sum();
                assert!((sum - p).abs() < 1e-4, "{sum} vs {p}");
            }
        }
    }
}

#[test]
fn formats_round_trip_vector_leaves() {
    let model = model();
    let d = dtrain();
    let want = model.predict(&d, Iterations::Best).unwrap();
    let reloaded = [
        BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap(),
        BoostedModel::from_json(&model.to_json().unwrap()).unwrap(),
        BoostedModel::from_xgboost_json(&model.to_xgboost_json().unwrap()).unwrap(),
        BoostedModel::from_xgboost_ubjson(&model.to_xgboost_ubjson().unwrap()).unwrap(),
    ];
    for m in reloaded {
        assert!(m.has_vector_leaves());
        assert_eq!(m.predict(&d, Iterations::Best).unwrap(), want);
        assert_eq!(m.num_boost_rounds(), 8);
    }
}

#[test]
fn early_stopping_keeps_whole_vector_rounds() {
    let dtrain = dtrain();
    // A holdout whose labels belong to other rows stops improving quickly.
    let (x, y) = data();
    let shuffled: Vec<f32> = y
        .as_chunks::<K>()
        .0
        .iter()
        .rev()
        .flatten()
        .copied()
        .collect();
    let holdout = DMatrix::from_dense(&x, N, COLS)
        .unwrap()
        .with_label_matrix(&shuffled, K)
        .unwrap();
    let result = Trainer::new(
        &vector_params().eta(1.0).max_depth(6).build().unwrap(),
        &dtrain,
        40,
    )
    .eval(&holdout, "holdout")
    .early_stopping_rounds(NonZeroUsize::new(1).unwrap())
    .train()
    .unwrap();
    let model = result.model;
    let best = model.best_iteration().expect("early stopping triggers");
    // Patience 1: one round past the best, one vector tree per round.
    assert_eq!(model.num_trees(), best + 2);
    let margin = model.predict_margin(&dtrain, Iterations::Best).unwrap();
    assert_eq!(margin, model.predict_margin(&dtrain, ..=best).unwrap());
    assert_ne!(margin, model.predict_margin(&dtrain, ..).unwrap());
}

#[test]
fn single_output_builds_scalar_trees() {
    let (x, y) = data();
    let y0: Vec<f32> = y.iter().step_by(K).copied().collect();
    let d = labeled_dense(&x, COLS, &y0);
    let vector = train(&vector_params().build().unwrap(), &d, 5).unwrap();
    let scalar = train(
        &TrainingParams::builder()
            .max_depth(4)
            .eta(0.3)
            .build()
            .unwrap(),
        &d,
        5,
    )
    .unwrap();
    assert!(!vector.has_vector_leaves());
    assert_eq!(
        vector.predict(&d, Iterations::Best).unwrap(),
        scalar.predict(&d, Iterations::Best).unwrap()
    );
}

#[test]
fn dart_rounds_train_vector_trees() {
    let params = vector_params()
        .booster(BoosterKind::Dart(
            Dart::builder()
                .rate_drop(0.5)
                .skip_drop(0.0)
                .build()
                .unwrap(),
        ))
        .seed(7)
        .build()
        .unwrap();
    let d = dtrain();
    let model = train(&params, &d, 6).unwrap();
    assert!(model.has_vector_leaves());
    assert_eq!(model.num_trees(), 6);
    let (x, _) = data();
    // Dropout rescales earlier trees, so the margins are the weighted sum.
    let margins = model.predict_margin(&d, Iterations::Best).unwrap();
    let unweighted = reference_margins(&model, &x, N);
    assert_ne!(margins.as_slice(), unweighted);
    assert_contribs_sum_to(
        &model.predict_contribs(&d, Iterations::Best).unwrap(),
        &margins,
    );
}

fn squared_error(k: usize) -> CustomLoss {
    CustomLoss::new("custom:sqerr", k, |p, y, _w, out| {
        for (o, (p, y)) in out.iter_mut().zip(p.iter().zip(y)) {
            *o = GradPair::new(p - y, 1.0);
        }
    })
    .with_base_margin(0.5)
}

/// Mean over targets, XGBoost's `multioutput_reduced_gradient.py` sketch.
fn mean_sketch(g: &[GradPair]) -> SplitGradient {
    let gpair = g
        .as_chunks::<K>()
        .0
        .iter()
        .map(|r| {
            let (g, h) = r
                .iter()
                .fold((0.0, 0.0), |(g, h), p| (g + p.grad, h + p.hess));
            GradPair::new(g / K as f32, h / K as f32)
        })
        .collect();
    SplitGradient::new(gpair, 1)
}

#[test]
fn reduced_gradients_grow_structure_from_the_sketch() {
    let dtrain = dtrain();
    let sketched = vector_params()
        .lambda(0.0)
        .objective(Objective::custom(
            squared_error(K).with_split_gradient(|_, g| Some(mean_sketch(g))),
        ))
        .build()
        .unwrap();
    let model = train(&sketched, &dtrain, 1).unwrap();
    let tree = &model.trees()[0];
    // Leaf vectors are refit per target from the full gradients: with unit
    // Hessians and no regularization each is eta × the leaf's mean residual.
    let (x, y) = data();
    let mut sums = vec![[0.0f64; K]; tree.num_nodes()];
    let mut counts = vec![0usize; tree.num_nodes()];
    for r in 0..N {
        let leaf = tree.leaf_id_dense(&x[r * COLS..(r + 1) * COLS], f32::NAN);
        counts[leaf] += 1;
        for (sum, &label) in sums[leaf].iter_mut().zip(&y[r * K..(r + 1) * K]) {
            *sum += f64::from(label - 0.5);
        }
    }
    for (leaf, &count) in counts.iter().enumerate().filter(|(_, c)| **c > 0) {
        for (t, sum) in sums[leaf].iter().enumerate() {
            let want = 0.3 * sum / count as f64;
            let got = f64::from(tree.leaf_vector(leaf)[t]);
            assert!(
                (got - want).abs() < 1e-5,
                "leaf {leaf} target {t}: {got} vs {want}"
            );
        }
    }
    // The sketch changes the structure relative to the full gradients.
    let full_params = vector_params()
        .lambda(0.0)
        .objective(Objective::custom(squared_error(K)))
        .build()
        .unwrap();
    let full = train(&full_params, &dtrain, 1).unwrap();
    assert_ne!(full.trees()[0].nodes(), tree.nodes());
}

#[test]
fn unsupported_combinations_are_rejected() {
    let dtrain = dtrain();
    let exact = vector_params()
        .tree_method(TreeMethod::Exact)
        .build()
        .unwrap();
    assert_eq!(invalid_param(train(&exact, &dtrain, 1)), "multi_strategy");

    let sketch =
        Objective::custom(squared_error(K).with_split_gradient(|_, g| Some(mean_sketch(g))));
    let per_output = TrainingParams::builder()
        .objective(sketch.clone())
        .build()
        .unwrap();
    assert_eq!(invalid_param(train(&per_output, &dtrain, 1)), "objective");
    // The linear booster grows no trees, so it refuses the hook rather than
    // ignoring it.
    for strategy in [
        MultiStrategy::OneOutputPerTree,
        MultiStrategy::MultiOutputTree,
    ] {
        let linear = TrainingParams::builder()
            .booster(BoosterKind::GbLinear)
            .multi_strategy(strategy)
            .objective(sketch.clone())
            .build()
            .unwrap();
        assert_eq!(invalid_param(train(&linear, &dtrain, 1)), "objective");
    }
    let monotone = vector_params()
        .monotone_constraints(vec![Monotone::Increasing])
        .objective(sketch)
        .build()
        .unwrap();
    assert_eq!(
        invalid_param(train(&monotone, &dtrain, 1)),
        "monotone_constraints"
    );

    let wrong = vector_params()
        .objective(Objective::custom(squared_error(K).with_split_gradient(
            |_, g| Some(SplitGradient::new(g[1..].to_vec(), 1)),
        )))
        .build()
        .unwrap();
    assert!(matches!(
        train(&wrong, &dtrain, 1),
        Err(HessboostError::DimensionMismatch { .. })
    ));
}

#[test]
fn vector_forests_hold_num_parallel_tree_trees_per_iteration() {
    let params = vector_params().num_parallel_tree(3).build().unwrap();
    let d = dtrain();
    let model = train(&params, &d, 4).unwrap();
    assert_eq!(model.num_trees(), 12);
    assert_eq!(model.trees_per_iteration(), 3);
    assert_eq!(model.num_boost_rounds(), 4);
    let (x, _) = data();
    // Without sampling the forest's trees are identical, each shrunk by
    // eta / 3, and the iteration ranges select whole forests.
    let all = model.predict_margin(&d, Iterations::Best).unwrap();
    assert_eq!(all.as_slice(), reference_margins(&model, &x, N));
    let first_two = model.predict_margin(&d, ..2).unwrap();
    assert_eq!(
        first_two,
        model
            .slice(..2, 1)
            .unwrap()
            .predict_margin(&d, Iterations::Best)
            .unwrap()
    );
    assert_ne!(first_two, all);
    // SHAP over a prefix range stays additive.
    assert_contribs_sum_to(&model.predict_contribs(&d, ..2).unwrap(), &first_two);
}

#[test]
fn continued_vector_training_matches_one_run() {
    let dtrain = dtrain();
    let params = vector_params().subsample(0.8).seed(11).build().unwrap();
    let full = train(&params, &dtrain, 8).unwrap();
    let first = train(&params, &dtrain, 5).unwrap();
    let continued = Trainer::new(&params, &dtrain, 3)
        .init_model(&first)
        .train()
        .unwrap()
        .model;
    assert_eq!(continued.num_trees(), 8);
    assert_eq!(
        continued.predict_margin(&dtrain, Iterations::Best).unwrap(),
        full.predict_margin(&dtrain, Iterations::Best).unwrap()
    );
}

#[test]
fn unsupported_vector_layouts_are_rejected() {
    let dtrain = dtrain();
    let vector = model();
    // XGBoost's refresh updater handles single-target trees only.
    let refresh = vector_params()
        .process_type(ProcessType::Update(Refresh::default()))
        .build()
        .unwrap();
    assert_eq!(
        invalid_param(
            Trainer::new(&refresh, &dtrain, 2)
                .init_model(&vector)
                .train()
        ),
        "process_type"
    );
    // A model keeps one tree kind.
    let scalar = TrainingParams::builder().max_depth(4).build().unwrap();
    assert_eq!(
        invalid_param(
            Trainer::new(&scalar, &dtrain, 2)
                .init_model(&vector)
                .train()
        ),
        "multi_strategy"
    );
    // The opt-in growth modes that bypass the vector-leaf split search.
    // Set directly, unvalidated: `train` itself must refuse them.
    let unchecked = |edit: fn(&mut TrainingParams)| {
        let mut params = vector_params().build().unwrap();
        edit(&mut params);
        params
    };
    for (params, name) in [
        (
            unchecked(|p| p.grow_policy = GrowPolicy::Symmetric),
            "grow_policy",
        ),
        (
            unchecked(|p| p.toad_penalty_feature = 0.1),
            "toad_penalty_feature",
        ),
    ] {
        assert_eq!(invalid_param(train(&params, &dtrain, 1)), name);
    }
    // The bit-packed compact format stores scalar leaves only.
    assert!(matches!(
        vector.to_compact_bytes(),
        Err(HessboostError::ModelFormat(_))
    ));
}

/// Rows of `cols` features (column 0 categorical) with `y` duplicated across
/// two targets.
fn categorical_two_targets(x: &[f32], cols: usize, y: &[f32]) -> DMatrix {
    let mut types = vec![FeatureType::Numerical; cols];
    types[0] = FeatureType::Categorical;
    let labels: Vec<f32> = y.iter().flat_map(|&v| [v, v]).collect();
    DMatrix::from_dense(x, y.len(), cols)
        .unwrap()
        .with_label_matrix(&labels, 2)
        .unwrap()
        .with_feature_types(&types)
        .unwrap()
}

/// One unregularized, unshrunk round fits every row whose leaf holds a single
/// label value exactly.
fn one_round_margins(params: hessboost::config::TrainingParamsBuilder, d: &DMatrix) -> Predictions {
    let params = params
        .multi_strategy(MultiStrategy::MultiOutputTree)
        .lambda(0.0)
        .eta(1.0)
        .base_score(0.0)
        .build()
        .unwrap();
    train(&params, d, 1)
        .unwrap()
        .predict_margin(d, Iterations::Best)
        .unwrap()
}

/// Each row's two margins equal its label `want[row]`.
fn assert_margins(got: &Predictions, want: &[f32]) {
    assert_eq!((got.n_rows(), got.width()), (want.len(), 2));
    for (r, (row, &w)) in got.rows().zip(want).enumerate() {
        for g in row {
            assert!((g - w).abs() < 1e-4, "row {r}: {got:?} vs {want:?}");
        }
    }
}

#[test]
fn low_cardinality_categories_split_one_hot() {
    // Fewer than four categories: XGBoost enumerates each category against
    // the rest with missing values on either side. One depth-1 round fits
    // each case exactly only through a one-hot candidate.
    let nan = f32::NAN;
    for (x, y) in [
        // A single category against missing values.
        (vec![0.0, 0.0, nan, nan], vec![0.0, 0.0, 2.0, 2.0]),
        // The middle category alone.
        (
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            vec![-5.0, -5.0, 10.0, 10.0, -5.0, -5.0],
        ),
        // The middle category together with the missing values.
        (
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, nan, nan],
            vec![-10.0, -10.0, 10.0, 10.0, -10.0, -10.0, 10.0, 10.0],
        ),
    ] {
        let d = categorical_two_targets(&x, 1, &y);
        let margins = one_round_margins(TrainingParams::builder().max_depth(1), &d);
        assert_margins(&margins, &y);
    }
}

#[test]
fn categorical_children_keep_xgboost_priority() {
    // The root splits categories {0, 1} (XGBoost's right child, stored
    // first) from {2, 3}. Both children then gain equally from splitting on
    // `b`, and the one remaining leaf goes to XGBoost's left child, the
    // positive one, under either growth policy.
    let (mut x, mut y, mut want) = (Vec::new(), Vec::new(), Vec::new());
    for c in 0..4 {
        for b in 0..2 {
            let v = if c < 2 { -10.0 } else { 10.0 } + 2.0 * b as f32 - 1.0;
            x.extend([c as f32, b as f32]);
            y.push(v);
            want.push(if c < 2 { -10.0 } else { v });
        }
    }
    let d = categorical_two_targets(&x, 2, &y);
    for policy in [GrowPolicy::DepthWise, GrowPolicy::LossGuide] {
        let params = TrainingParams::builder()
            .grow_policy(policy)
            .max_depth(2)
            .max_leaves(3);
        assert_margins(&one_round_margins(params, &d), &want);
    }
}
