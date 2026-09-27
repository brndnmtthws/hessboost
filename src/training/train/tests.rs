use super::*;
use crate::config::{Dart, GrowPolicy, LinearTree, TreeMethod};
use crate::metric::{Metric, Rmse};
use crate::objective::{Aft, CustomLoss, LambdaRank, Multiclass, Objective, PseudoHuber, RegLoss};
use crate::rng::Rng;
use crate::test_support::labeled_dense;
use crate::training::dart::select_dropout;
use crate::training::{RoundEval, train};
use crate::tree::RegTree;

/// A deterministic uniform `[0, 1)` stream (a 64-bit LCG's top 31 bits).
fn lcg(mut s: u64) -> impl FnMut() -> f32 {
    move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((s >> 33) as f32) / (1u32 << 31) as f32
    }
}

/// A learnable 1-D step function: y = 0 for x<0.5, y = 1 for x>=0.5.
fn step_dataset(n: usize) -> DMatrix {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let xi = i as f32 / n as f32;
        x.push(xi);
        y.push(if xi >= 0.5 { 1.0 } else { 0.0 });
    }
    labeled_dense(&x, n, 1, &y)
}

/// Squared error around `y` with fixed, per-output Hessians: output 0
/// weights row 0 heavily, output 1 row 3, so their Hessian-weighted
/// `approx` cuts differ.
struct PerOutputHessians;

impl Loss for PerOutputHessians {
    fn name(&self) -> &'static str {
        "custom:per_output_hessians"
    }
    fn n_outputs(&self) -> usize {
        2
    }
    fn gradient(&self, preds: &[f32], labels: &[f32], _: Option<&[f32]>, out: &mut [GradPair]) {
        const HESS: [[f32; 4]; 2] = [[100.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 100.0]];
        for (i, (g, p)) in out.iter_mut().zip(preds).enumerate() {
            let (row, output) = (i / 2, i % 2);
            let h = HESS[output][row];
            *g = GradPair::new(h * (p - labels[row]), h);
        }
    }
    fn const_hess(&self) -> bool {
        true
    }
    fn default_metric(&self) -> crate::metric::EvalMetric {
        crate::metric::EvalMetric::Rmse
    }
}

#[test]
fn approx_constant_hessian_cuts_do_not_depend_on_thread_count() {
    // The parallel slot loop grows both outputs' trees at once; the
    // cached cuts must still come from output 0, as the serial loop's.
    let d = labeled_dense(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 1.0, 2.0, 3.0]);
    let fit = |nthread: usize| {
        let params = TrainingParams::builder()
            .objective(Objective::custom(PerOutputHessians))
            .tree_method(TreeMethod::Approx)
            .max_bin(2)
            .max_depth(1)
            .min_child_weight(0.0)
            .nthread(nthread)
            .build()
            .unwrap();
        let model = train(&params, &d, 3).unwrap();
        model.predict_margin(&d).unwrap()
    };
    let serial = fit(1);
    for _ in 0..32 {
        assert_eq!(fit(4), serial);
    }
}

/// Zero-weight rows are not sketched, so one whose value lies beyond the
/// last cut is binned into the last bin but routed right of a split on
/// it by the finished tree. Linear leaves must then be fitted from raw
/// routing, as without captured rows: fitting the second tree from the
/// builder's rows gave leaf 3 intercept 1.0 instead of 0.5, and row 0 a
/// prediction of 2.0 instead of 1.5.
#[test]
fn linear_leaves_route_zero_weight_rows_like_the_tree() {
    let nan = f32::NAN;
    let x = [0.0, 1.0, 0.0, nan, 1.0, 0.0, 0.0, 100.0, 0.0, 101.0];
    let data = DMatrix::from_dense(&x, 5, 2)
        .unwrap()
        .with_labels(&[2.0, 0.0, 10.0, 0.0, 0.0])
        .unwrap()
        .with_weights(&[1.0, 1.0, 1.0, 0.0, 0.0])
        .unwrap();
    let params = TrainingParams::builder()
        .tree_method(TreeMethod::Hist)
        .linear_tree(LinearTree::new(1.0).unwrap())
        .base_score(0.0)
        .eta(1.0)
        .lambda(1.0)
        .max_depth(2)
        .build()
        .unwrap();
    let model = Trainer::new(&params, &data, 2).train().unwrap().model;
    assert_eq!(
        model.predict(&data).unwrap().as_slice(),
        [1.5, 0.0, 7.5, 0.0, 0.0]
    );
    let linear = model.trees()[1].linear_leaves().unwrap();
    let intercepts: Vec<u64> = (0..5).map(|id| linear.intercept(id).to_bits()).collect();
    let routed: Vec<u64> = [0.0f64, 0.0, 2.5, 0.5, -0.0].map(f64::to_bits).to_vec();
    assert_eq!(intercepts, routed);
}

#[test]
fn approx_forests_share_one_index_per_output_across_threads() {
    // Per-round `approx` cuts (non-constant Hessians) are shared by an
    // output's parallel trees; the parallel slot loop builds them before
    // growing the trees, and every thread count grows the same forest.
    // 20,000 rows × 4 features reach the parallel cut construction
    // (65,536 cells), whose nested rayon loops deadlocked when tree
    // tasks built the shared index themselves.
    let n = 20_000;
    let x: Vec<f32> = (0..n * 4)
        .map(|i| ((i * 7919) % 1009) as f32 / 1009.0)
        .collect();
    let y: Vec<f32> = x
        .chunks(4)
        .map(|r| f32::from(r[0] + 0.5 * r[1] > 0.7))
        .collect();
    let d = labeled_dense(&x, n, 4, &y);
    let fit = |nthread: usize| {
        let params = TrainingParams::builder()
            .objective(Objective::Softprob(
                crate::objective::Multiclass::new(2).unwrap(),
            ))
            .tree_method(TreeMethod::Approx)
            .num_parallel_tree(4)
            .max_depth(3)
            .nthread(nthread)
            .build()
            .unwrap();
        train(&params, &d, 3).unwrap().predict_margin(&d).unwrap()
    };
    let serial = fit(1);
    for _ in 0..4 {
        assert_eq!(fit(8), serial);
    }
}

#[test]
fn binary_logistic_separates_classes() {
    let d = step_dataset(100);
    let params = TrainingParams::builder()
        .objective(Objective::BinaryLogistic(RegLoss::default()))
        .max_depth(3)
        .eta(0.3)
        .build()
        .unwrap();
    let model = train(&params, &d, 50).unwrap();
    let preds = model.predict(&d).unwrap().into_vec(); // probabilities, one per row
    // Low-x rows -> ~0, high-x rows -> ~1.
    assert!(preds[0] < 0.1, "expected ~0, got {}", preds[0]);
    assert!(preds[99] > 0.9, "expected ~1, got {}", preds[99]);
}

#[test]
fn base_score_only_model_predicts_mean() {
    // Zero rounds -> prediction is just the base score (label mean).
    let d = step_dataset(10);
    let params = TrainingParams::builder()
        .objective(Objective::SquaredError(RegLoss::default()))
        .build()
        .unwrap();
    let model = train(&params, &d, 0).unwrap();
    let preds = model.predict(&d).unwrap();
    let mean = d.labels().unwrap().iter().sum::<f32>() / 10.0;
    for p in preds.into_vec() {
        assert!((p - mean).abs() < 1e-6);
    }
}

#[test]
fn tree_methods_reach_similar_accuracy() {
    let d = step_dataset(120);
    let rmse = |method: TreeMethod| {
        let params = TrainingParams::builder()
            .objective(Objective::SquaredError(RegLoss::default()))
            .tree_method(method)
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 60).unwrap();
        let preds = model.predict(&d).unwrap();
        Rmse.eval(preds.as_slice(), d.labels().unwrap(), None)
    };
    let rmse_hist = rmse(TreeMethod::Hist);
    assert!(rmse_hist < 0.05, "hist rmse {rmse_hist}");
    for method in [TreeMethod::Exact, TreeMethod::Approx] {
        let other = rmse(method);
        assert!(other < 0.05, "{method:?} rmse {other}");
        // The methods land very close on this cleanly-binnable problem.
        assert!((other - rmse_hist).abs() < 0.02, "{method:?} rmse {other}");
    }
}

#[test]
fn lossguide_trains_end_to_end() {
    let d = step_dataset(120);
    let params = TrainingParams::builder()
        .objective(Objective::SquaredError(RegLoss::default()))
        .tree_method(TreeMethod::Hist)
        .grow_policy(GrowPolicy::LossGuide)
        .max_leaves(16)
        .eta(0.3)
        .build()
        .unwrap();
    let model = train(&params, &d, 60).unwrap();
    let preds = model.predict(&d).unwrap();
    let rmse = Rmse.eval(preds.as_slice(), d.labels().unwrap(), None);
    assert!(rmse < 0.06, "lossguide rmse {rmse}");
}

#[test]
fn exact_rejects_lossguide() {
    let d = step_dataset(20);
    let params = TrainingParams::builder()
        .tree_method(TreeMethod::Exact)
        .grow_policy(GrowPolicy::LossGuide)
        .max_leaves(8)
        .build()
        .unwrap();
    assert!(matches!(
        train(&params, &d, 5),
        Err(HessboostError::InvalidParameter { name, .. }) if name == "grow_policy"
    ));
}

#[test]
fn num_class_must_match_the_objective_outputs() {
    let x: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let y: Vec<f32> = (0..12).map(|i| (i % 3) as f32).collect();
    let d = labeled_dense(&x, 12, 1, &y);
    let params = |objective: Objective| {
        TrainingParams::builder()
            .objective(objective)
            .build()
            .unwrap()
    };
    // A stray num_class used to train a single-output model that
    // `from_bytes`/`from_json` then refused; the class count now lives
    // in the multiclass objective only.
    for (objective, num_class) in [
        (Objective::SquaredError(RegLoss::default()), 0),
        (Objective::Softprob(Multiclass::new(3).unwrap()), 3),
    ] {
        let model = train(&params(objective), &d, 2).unwrap();
        assert_eq!(
            model
                .objective()
                .built_in()
                .and_then(Objective::num_class)
                .unwrap_or(0),
            num_class
        );
        BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
    }
}

#[test]
fn multiclass_softprob_learns_three_classes() {
    // 1-D feature partitioned into 3 regions -> 3 classes.
    let n = 150;
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..n {
        let xi = i as f32 / n as f32; // 0..1
        x.push(xi);
        y.push(if xi < 0.33 {
            0.0
        } else if xi < 0.66 {
            1.0
        } else {
            2.0
        });
    }
    let d = labeled_dense(&x, n, 1, &y);
    let params = TrainingParams::builder()
        .objective(Objective::Softprob(
            crate::objective::Multiclass::new(3).unwrap(),
        ))
        .max_depth(3)
        .eta(0.3)
        .build()
        .unwrap();
    let model = train(&params, &d, 60).unwrap();
    assert_eq!(model.n_outputs(), 3);
    assert_eq!(model.num_trees(), 180); // 60 rounds * 3 classes
    assert_eq!(model.num_boost_rounds(), 60);

    // Probabilities: three per row, each row sums to 1.
    let probs = model.predict(&d).unwrap();
    assert_eq!((probs.n_rows(), probs.width()), (n, 3));
    for row in probs.rows() {
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-4);
    }

    // Predicted classes match the region labels on almost all rows.
    let classes = model.predict_class(&d).unwrap();
    let correct = classes
        .as_slice()
        .iter()
        .zip(&y)
        .filter(|(c, l)| **c == **l as u32)
        .count();
    assert!(correct as f32 / n as f32 > 0.95, "accuracy {correct}/{n}");
}

#[test]
fn poisson_trains_and_predicts_positive_rates() {
    let n = 200;
    let mut rng = lcg(7);
    let x: Vec<f32> = (0..n).map(|_| rng()).collect();
    // The rate increases with x; the label is a rough count.
    let y: Vec<f32> = x.iter().map(|xi| (1.0 + 5.0 * xi).round()).collect();
    let d = labeled_dense(&x, n, 1, &y);
    let params = TrainingParams::builder()
        .objective(Objective::Poisson)
        .max_depth(3)
        .eta(0.2)
        .build()
        .unwrap();
    let model = train(&params, &d, 60).unwrap();
    let preds = model.predict(&d).unwrap().into_vec(); // rates (exp transform), one per row
    assert!(preds.iter().all(|&p| p > 0.0), "rates must be positive");
    // Higher x should predict a higher rate: compare mean predicted rate for
    // low-x vs high-x rows (the feature is randomized, so bucket by value).
    let mean = |high: bool| {
        let bucket: Vec<f32> = (0..n)
            .filter(|&i| (x[i] >= 0.5) == high)
            .map(|i| preds[i])
            .collect();
        bucket.iter().sum::<f32>() / bucket.len() as f32
    };
    let (lo_mean, hi_mean) = (mean(false), mean(true));
    assert!(
        lo_mean < hi_mean,
        "rate should rise with x: {lo_mean} vs {hi_mean}"
    );
}

#[test]
fn custom_objective_matches_builtin_squared_error() {
    let d = step_dataset(80);

    let builtin = {
        let p = TrainingParams::builder()
            .objective(Objective::SquaredError(RegLoss::default()))
            .max_depth(3)
            .eta(0.3)
            .base_score(0.0)
            .build()
            .unwrap();
        train(&p, &d, 30).unwrap().predict(&d).unwrap()
    };

    let custom = {
        let obj = CustomLoss::new("custom", 1, |preds, labels, w, out| {
            for i in 0..preds.len() {
                let wi = w.map_or(1.0, |ws| ws[i]);
                out[i] = GradPair::new((preds[i] - labels[i]) * wi, wi);
            }
        });
        let p = TrainingParams::builder()
            .objective(Objective::custom(obj))
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        train(&p, &d, 30).unwrap().predict(&d).unwrap()
    };

    for (a, b) in builtin.as_slice().iter().zip(custom.as_slice()) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
}

#[test]
fn custom_multi_output_objective_trains_with_stride_and_round_trips() {
    let d = step_dataset(80);
    let n = d.n_rows();
    let rounds = 5usize;

    // Two outputs, `[row][output]` layout: output 0 fits the label, output 1
    // fits its negation. Same data with mirrored targets, so the learned
    // outputs must mirror each other.
    let obj = CustomLoss::new("custom:two", 2, |preds, labels, w, out| {
        for i in 0..labels.len() {
            let wi = w.map_or(1.0, |ws| ws[i]);
            out[2 * i] = GradPair::new((preds[2 * i] - labels[i]) * wi, wi);
            out[2 * i + 1] = GradPair::new((preds[2 * i + 1] + labels[i]) * wi, wi);
        }
    });
    let p = TrainingParams::builder()
        .objective(Objective::custom(obj))
        .max_depth(2)
        .build()
        .unwrap();
    let model = train(&p, &d, rounds).unwrap();

    assert_eq!(model.n_outputs(), 2);
    assert_eq!(model.base_scores().len(), 2);
    assert_eq!(model.num_trees(), 2 * rounds);

    let margin = model.predict_margin(&d).unwrap();
    assert_eq!((margin.n_rows(), margin.width()), (n, 2));
    // Unknown objective name: `predict` falls back to raw margins.
    assert_eq!(model.predict(&d).unwrap(), margin);

    let labels = d.labels().unwrap();
    let (mut err0, mut err1, mut err_init) = (0.0f32, 0.0f32, 0.0f32);
    for (i, (row, &y)) in margin.rows().zip(labels).enumerate() {
        let (o0, o1) = (row[0], row[1]);
        assert!((o0 + o1).abs() < 1e-5, "row {i}: {o0} vs {o1} not mirrored");
        err0 += (o0 - y).abs();
        err1 += (o1 + y).abs();
        err_init += y.abs(); // initial margin is 0.0
    }
    assert!(
        err0 < 0.5 * err_init,
        "output 0 did not learn: {err0} vs {err_init}"
    );
    assert!(
        err1 < 0.5 * err_init,
        "output 1 did not learn: {err1} vs {err_init}"
    );

    let via_json = BoostedModel::from_json(&model.to_json().unwrap()).unwrap();
    assert_eq!(via_json.n_outputs(), 2);
    assert_eq!(via_json.predict_margin(&d).unwrap(), margin);
    let via_bytes = BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
    assert_eq!(via_bytes.n_outputs(), 2);
    assert_eq!(via_bytes.predict_margin(&d).unwrap(), margin);
}

#[test]
fn ranking_ndcg_improves_over_rounds() {
    use crate::metric::Ndcg;
    // Query groups whose single feature is correlated with relevance, so a
    // ranker can learn to order documents. Docs are laid out in ascending
    // relevance (the worst initial order given zero starting margins).
    let n_groups = 30usize;
    let per = 6usize;
    let n = n_groups * per;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let sizes = vec![per; n_groups];
    let mut rng = lcg(42);
    for _ in 0..n_groups {
        for d in 0..per {
            let rel = d as f32; // relevance grade 0..per-1
            let noise = (rng() - 0.5) * 0.8;
            x.push(rel + noise);
            y.push(rel);
        }
    }
    let d = labeled_dense(&x, n, 1, &y)
        .with_group_sizes(&sizes)
        .unwrap();

    let params = TrainingParams::builder()
        .objective(Objective::RankNdcg(LambdaRank::default()))
        .max_depth(3)
        .eta(0.2)
        .build()
        .unwrap();
    let res = Trainer::new(&params, &d, 40)
        .eval(&d, "train")
        .train()
        .unwrap();
    assert!(!res.history.is_empty());

    // rank:ndcg's default metric: XGBoost's `ndcg@32`.
    let ndcg_of = |r: &RoundEval| r.score("train", "ndcg@32").unwrap();
    let first = ndcg_of(&res.history[0]);
    let last = ndcg_of(res.history.last().unwrap());

    // Baseline NDCG of the untrained (all-equal-score) ranking.
    let base = Ndcg::new(None).eval_grouped(&vec![0.0; n], &y, None, d.group());
    assert!(last >= first - 1e-9, "ndcg regressed: {first} -> {last}");
    assert!(
        last > base + 1e-3,
        "training should beat the untrained baseline: {base} -> {last}"
    );
    assert!(last > 0.9, "final ndcg should be high, got {last}");
}

#[test]
fn dart_trains_reduces_error_and_roundtrips() {
    let d = step_dataset(120);
    let params = TrainingParams::builder()
        .objective(Objective::SquaredError(RegLoss::default()))
        .booster(BoosterKind::Dart(
            Dart::builder().rate_drop(0.1).build().unwrap(),
        ))
        .max_depth(3)
        .eta(0.3)
        .build()
        .unwrap();
    let model = train(&params, &d, 60).unwrap();
    assert_eq!(model.num_trees(), 60);

    // It should learn the step: RMSE well below a constant predictor.
    let preds = model.predict(&d).unwrap();
    let rmse = Rmse.eval(preds.as_slice(), d.labels().unwrap(), None);
    assert!(rmse < 0.1, "dart rmse too high: {rmse}");

    // Native and JSON round-trips preserve predictions (weights included).
    for restored in [
        BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap(),
        BoostedModel::from_json(&model.to_json().unwrap()).unwrap(),
    ] {
        assert_eq!(restored.predict(&d).unwrap(), preds);
    }
}

/// XGBoost's `DropTrees`: DART without dropout never drops a tree, so it
/// trains exactly as gbtree (row sampling included).
#[test]
fn dart_without_dropout_trains_as_gbtree() {
    let d = step_dataset(200);
    let fit = |booster| {
        let params = TrainingParams::builder()
            .booster(booster)
            .max_depth(3)
            .subsample(0.7)
            .seed(5)
            .build()
            .unwrap();
        train(&params, &d, 20).unwrap()
    };
    let gbtree = fit(BoosterKind::GbTree);
    let dart = fit(BoosterKind::Dart(Dart::default()));
    assert_eq!(dart.trees(), gbtree.trees());
    assert_eq!(dart.predict(&d).unwrap(), gbtree.predict(&d).unwrap());
}

/// A round that drops nothing reads the training margin cache, so after
/// a dropout round (which rescales trees) that cache must match the
/// ensemble: training in one run equals training half, then continuing
/// from the saved half (whose caches are rebuilt from the model), for
/// scalar and vector-leaf trees.
#[test]
fn dart_rounds_after_a_dropout_read_current_margins() {
    let x: Vec<f32> = (0..400).map(|i| ((i * 37) % 101) as f32 / 101.0).collect();
    let y: Vec<f32> = (0..400).map(|i| ((i * 13) % 7) as f32).collect();
    let scalar = labeled_dense(&x, 200, 2, &y[..200]);
    let vector = DMatrix::from_dense(&x, 200, 2)
        .unwrap()
        .with_label_matrix(&y, 2)
        .unwrap();
    let dart = Dart::builder()
        .rate_drop(0.3)
        .skip_drop(0.5)
        .build()
        .unwrap();
    for (data, strategy) in [
        (&scalar, crate::config::MultiStrategy::OneOutputPerTree),
        (&vector, crate::config::MultiStrategy::MultiOutputTree),
    ] {
        let params = TrainingParams::builder()
            .booster(BoosterKind::Dart(dart))
            .multi_strategy(strategy)
            .max_depth(3)
            .seed(11)
            .build()
            .unwrap();
        let whole = train(&params, data, 12).unwrap();
        let half = train(&params, data, 6).unwrap();
        let resumed = Trainer::new(&params, data, 6)
            .init_model(&half)
            .train()
            .unwrap()
            .model;
        assert_eq!(
            resumed.predict_margin(data).unwrap(),
            whole.predict_margin(data).unwrap(),
            "{strategy:?}"
        );
    }
}

/// A round whose draw selects no tree drops nothing unless `one_drop`,
/// which then drops exactly one; an empty ensemble draws nothing.
#[test]
fn dart_forces_a_drop_only_under_one_drop() {
    let d = step_dataset(50);
    let params = TrainingParams::builder().max_depth(2).build().unwrap();
    let model = train(&params, &d, 5).unwrap();
    let never = Dart::builder().rate_drop(1e-300).build().unwrap();
    let forced = Dart::builder()
        .rate_drop(1e-300)
        .one_drop(true)
        .build()
        .unwrap();
    let mut rng = Rng::new(3);
    assert_eq!(select_dropout(&model, &never, &mut rng), None);
    let (mask, dropped) = select_dropout(&model, &forced, &mut rng).unwrap();
    assert_eq!((dropped.len(), mask.iter().filter(|&&m| m).count()), (1, 1));
    let empty = train(&params, &d, 0).unwrap();
    let mut before = rng.clone();
    assert_eq!(select_dropout(&empty, &forced, &mut rng), None);
    assert_eq!(rng.f64(), before.f64());
}

#[test]
fn gbtree_unchanged_by_weight_field() {
    // A default gbtree model carries all-1.0 weights, so predictions must be
    // bit-for-bit the unweighted tree sum.
    let d = step_dataset(100);
    let params = TrainingParams::builder()
        .objective(Objective::SquaredError(RegLoss::default()))
        .max_depth(3)
        .eta(0.3)
        .build()
        .unwrap();
    let model = train(&params, &d, 40).unwrap();
    // Compare weighted prediction against a manual unit-weight tree sum.
    let preds = model.predict_margin(&d).unwrap();
    let n = d.n_rows();
    let mut manual = vec![model.base_score(); n];
    for tree in model.trees() {
        for (row, m) in manual.iter_mut().enumerate() {
            *m += tree.predict_row(&d, row);
        }
    }
    assert_eq!(preds.width(), 1);
    for (a, b) in preds.as_slice().iter().zip(&manual) {
        assert_eq!(*a, *b, "gbtree weighting changed the sum");
    }
}

#[test]
fn categorical_split_beats_numeric_on_non_ordinal_pattern() {
    use crate::data::FeatureType;
    // One feature, 4 categories. Label is a NON-ordinal function of the
    // category: {0,2} -> 0, {1,3} -> 1. A single numeric `x < t` threshold
    // cannot separate {0,2} from {1,3}; a categorical set-split can.
    let cats = [0.0f32, 1.0, 2.0, 3.0];
    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..40 {
        for &c in &cats {
            x.push(c);
            y.push(if (c as u32) % 2 == 1 { 1.0 } else { 0.0 });
        }
    }
    let numeric = labeled_dense(&x, x.len(), 1, &y);
    let categorical = numeric
        .clone()
        .with_feature_types(&[FeatureType::Categorical])
        .unwrap();

    // Depth-1 stumps: the numeric model can only threshold, the categorical
    // model can partition the category set in a single node.
    for method in [TreeMethod::Auto, TreeMethod::Exact] {
        let mk = |d: &DMatrix| {
            let p = TrainingParams::builder()
                .objective(Objective::SquaredError(RegLoss::default()))
                .tree_method(method)
                .max_depth(1)
                .eta(0.3)
                .build()
                .unwrap();
            let m = train(&p, d, 40).unwrap();
            Rmse.eval(m.predict(d).unwrap().as_slice(), d.labels().unwrap(), None)
        };
        let rmse_num = mk(&numeric);
        let rmse_cat = mk(&categorical);

        // Categorical nearly fits the pattern; numeric is left far behind.
        assert!(
            rmse_cat < 0.02,
            "{method:?} categorical rmse too high: {rmse_cat}"
        );
        assert!(
            rmse_num > 0.05,
            "{method:?} numeric unexpectedly fit it: {rmse_num}"
        );
    }
}

#[test]
fn exact_monotone_increasing_predictions_nondecreasing() {
    use crate::config::Monotone;
    // A V-shaped target: the unconstrained fit dips then rises. Under an
    // increasing constraint with tree_method=exact, predictions must be
    // non-decreasing in the feature.
    let n = 80;
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..n {
        let xi = i as f32 / n as f32;
        x.push(xi);
        y.push((xi - 0.5).abs());
    }
    let d = labeled_dense(&x, n, 1, &y);
    let params = TrainingParams::builder()
        .objective(Objective::SquaredError(RegLoss::default()))
        .tree_method(TreeMethod::Exact)
        .max_depth(4)
        .eta(0.3)
        .monotone_constraints(vec![Monotone::Increasing])
        .build()
        .unwrap();
    let model = train(&params, &d, 50).unwrap();
    let preds = model.predict(&d).unwrap().into_vec(); // one value per row
    let mut prev = f32::NEG_INFINITY;
    for (i, p) in preds.iter().enumerate() {
        assert!(
            *p >= prev - 1e-4,
            "monotonicity violated at row {i}: {p} < {prev}"
        );
        prev = *p;
    }
}

/// Training starts from a per-instance base margin: with 0 rounds the
/// margin is exactly the base margin, and trees then fit the labels'
/// residuals from it (not from the intercept).
#[test]
fn training_starts_from_the_base_margin() {
    let d = step_dataset(60);
    let bm: Vec<f32> = (0..60).map(|i| i as f32 * 0.01 + 1.5).collect();
    let d_bm = d.with_base_margin(&bm).unwrap();
    let params = TrainingParams::builder()
        .objective(Objective::SquaredError(RegLoss::default()))
        .max_depth(3)
        .eta(0.3)
        .build()
        .unwrap();
    let initial = train(&params, &d_bm, 0).unwrap();
    assert_eq!(initial.predict_margin(&d_bm).unwrap().as_slice(), bm);

    let fitted = train(&params, &d_bm, 10).unwrap();
    let margin = fitted.predict_margin(&d_bm).unwrap();
    let rmse = Rmse.eval(margin.as_slice(), d_bm.labels().unwrap(), None);
    assert!(
        rmse < 0.2,
        "the trees did not fit the base-margin residuals: {rmse}"
    );
}

#[test]
fn colsample_bynode_changes_the_model() {
    // Multi-feature dataset so column sampling has features to drop.
    let (n, f) = (400usize, 8usize);
    let mut x = vec![0f32; n * f];
    let mut y = vec![0f32; n];
    let mut rng = lcg(3);
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..f {
            let v = rng();
            x[i * f + j] = v;
            acc += v * (j as f32 + 1.0);
        }
        y[i] = acc;
    }
    let d = labeled_dense(&x, n, f, &y);

    let train_with = |bynode: f64| {
        let p = TrainingParams::builder()
            .objective(Objective::SquaredError(RegLoss::default()))
            .max_depth(4)
            .eta(0.3)
            .colsample_bynode(bynode)
            .seed(1)
            .build()
            .unwrap();
        train(&p, &d, 20).unwrap().predict(&d).unwrap()
    };
    let full = train_with(1.0);
    let sampled = train_with(0.5);
    // With per-node sampling active, the fitted model must differ.
    let differs = full
        .as_slice()
        .iter()
        .zip(sampled.as_slice())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(differs, "colsample_bynode had no effect on the model");
}

/// Build a 3-feature dataset for the linear target y = 2*x0 - 3*x1 + 0.5*x2
/// with a little deterministic noise.
fn linear_dataset(n: usize) -> DMatrix {
    let f = 3usize;
    let mut x = vec![0f32; n * f];
    let mut y = vec![0f32; n];
    let mut rng = lcg(11);
    for i in 0..n {
        let x0 = rng();
        let x1 = rng();
        let x2 = rng();
        x[i * f] = x0;
        x[i * f + 1] = x1;
        x[i * f + 2] = x2;
        let noise = (rng() - 0.5) * 0.02;
        y[i] = 2.0 * x0 - 3.0 * x1 + 0.5 * x2 + noise;
    }
    labeled_dense(&x, n, f, &y)
}

#[test]
fn gblinear_fits_linear_target() {
    let d = linear_dataset(400);
    let params = TrainingParams::builder()
        .objective(Objective::SquaredError(RegLoss::default()))
        .booster(BoosterKind::GbLinear)
        .eta(0.5)
        .base_score(0.0)
        .build()
        .unwrap();
    let model = train(&params, &d, 200).unwrap();
    assert_eq!(model.num_trees(), 0);

    let preds = model.predict(&d).unwrap();
    let y = d.labels().unwrap();
    let rmse = Rmse.eval(preds.as_slice(), y, None);

    // Baseline: predicting the label mean.
    let mean = y.iter().sum::<f32>() / y.len() as f32;
    let mean_preds = vec![mean; y.len()];
    let rmse_mean = Rmse.eval(&mean_preds, y, None);

    assert!(rmse < 0.1, "gblinear rmse too high: {rmse}");
    assert!(
        rmse < rmse_mean * 0.25,
        "gblinear ({rmse}) should be far below the mean predictor ({rmse_mean})"
    );
}

#[test]
fn gblinear_roundtrips() {
    let d = linear_dataset(200);
    let params = TrainingParams::builder()
        .objective(Objective::SquaredError(RegLoss::default()))
        .booster(BoosterKind::GbLinear)
        .eta(0.5)
        .build()
        .unwrap();
    let model = train(&params, &d, 100).unwrap();
    let before = model.predict(&d).unwrap();

    for restored in [
        BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap(),
        BoostedModel::from_json(&model.to_json().unwrap()).unwrap(),
    ] {
        assert_eq!(restored.predict(&d).unwrap(), before);
    }
}

#[test]
fn custom_metric_matches_builtin_rmse_early_stopping() {
    use crate::metric::CustomMetric;
    let d = step_dataset(80);
    let params = TrainingParams::builder()
        .objective(Objective::SquaredError(RegLoss::default()))
        .max_depth(3)
        .eta(0.3)
        .build()
        .unwrap();

    // Builtin path: default `rmse` metric drives early stopping.
    let builtin = Trainer::new(&params, &d, 200)
        .eval(&d, "train")
        .early_stopping_rounds(5)
        .train()
        .unwrap();

    // Custom path: a CustomMetric reimplementing RMSE (minimize).
    let rmse_metric = CustomMetric::new("my-rmse", false, |preds, labels, weights| {
        let mut sq = 0.0f64;
        let mut wsum = 0.0f64;
        for i in 0..preds.len() {
            let w = weights.map_or(1.0, |ws: &[f32]| f64::from(ws[i]));
            let diff = f64::from(preds[i]) - f64::from(labels[i]);
            sq += w * diff * diff;
            wsum += w;
        }
        if wsum > 0.0 { (sq / wsum).sqrt() } else { 0.0 }
    });
    let custom = Trainer::new(&params, &d, 200)
        .eval(&d, "train")
        .early_stopping_rounds(5)
        .custom_metric(Box::new(rmse_metric))
        .train()
        .unwrap();

    // Early stopping fired `patience` rounds after the best iteration,
    // identically for both metrics, so the models match tree for tree.
    let best = builtin.model.best_iteration().expect("stops early");
    assert_eq!(builtin.history.len(), best + 6);
    assert_eq!(builtin.model.num_trees(), best + 6);
    assert_eq!(custom.model.best_iteration(), Some(best));
    assert_eq!(
        custom.model.predict(&d).unwrap(),
        builtin.model.predict(&d).unwrap()
    );

    // As in XGBoost's `xgb.train`, the custom metric is reported after
    // the configured (here the default) ones and, being last, drives
    // early stopping.
    let history = &custom.history[0];
    let names: Vec<&str> = history
        .scores
        .iter()
        .map(|score| score.metric.as_str())
        .collect();
    assert_eq!(names, vec!["rmse", "my-rmse"]);
    assert_eq!(
        history.score("train", "my-rmse"),
        Some(history.scores[1].value)
    );
    assert_eq!(history.score("valid", "my-rmse"), None);
}

#[test]
fn multiclass_softmax_returns_one_label_per_row() {
    let x = [0.0, 0.1, 0.5, 0.6, 0.9, 1.0];
    let y = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
    let d = labeled_dense(&x, 6, 1, &y);
    let params = TrainingParams::builder()
        .objective(Objective::Softmax(
            crate::objective::Multiclass::new(3).unwrap(),
        ))
        .max_depth(2)
        .build()
        .unwrap();
    let model = train(&params, &d, 20).unwrap();
    let predictions = model.predict(&d).unwrap();
    assert_eq!((predictions.n_rows(), predictions.width()), (d.n_rows(), 1));
    assert!(
        predictions
            .as_slice()
            .iter()
            .all(|value| value.fract() == 0.0 && *value < 3.0)
    );
    assert_eq!(
        model.predict_class(&d).unwrap().as_slice(),
        predictions
            .as_slice()
            .iter()
            .map(|value| *value as u32)
            .collect::<Vec<_>>()
    );
}

/// XGBoost's `LogisticRegression::CheckLabel` accepts any probability in
/// `[0, 1]` for both logistic objectives, and rejects anything outside.
#[test]
fn logistic_objectives_accept_probability_labels() {
    let x: Vec<f32> = (0..8).map(|i| i as f32 / 8.0).collect();
    let soft = [0.25f32, 0.75, 0.0, 1.0, 0.5, 0.9, 0.1, 0.6];
    for objective in [
        Objective::RegLogistic(RegLoss::default()),
        Objective::BinaryLogistic(RegLoss::default()),
    ] {
        let name = objective.name();
        let params = TrainingParams::builder()
            .objective(objective.clone())
            .max_depth(2)
            .build()
            .unwrap();
        let d = labeled_dense(&x, 8, 1, &soft);
        let model = train(&params, &d, 3).unwrap();
        assert_eq!(model.objective().name(), name);
        assert!(
            model
                .predict(&d)
                .unwrap()
                .as_slice()
                .iter()
                .all(|p| (0.0..=1.0).contains(p))
        );
        for bad in [1.5f32, -0.1] {
            let mut labels = soft;
            labels[0] = bad;
            let d = labeled_dense(&x, 8, 1, &labels);
            assert!(
                matches!(
                    train(&params, &d, 3),
                    Err(HessboostError::InvalidParameter { .. })
                ),
                "{name} should reject label {bad}"
            );
        }
    }
}

#[test]
fn count_base_score_is_in_reported_space() {
    let d = labeled_dense(&[0.0, 1.0], 2, 1, &[1.0, 2.0]);
    let params = TrainingParams::builder()
        .objective(Objective::Poisson)
        .base_score(0.5)
        .build()
        .unwrap();
    let model = train(&params, &d, 0).unwrap();
    assert!(
        model
            .predict(&d)
            .unwrap()
            .as_slice()
            .iter()
            .all(|prediction| (*prediction - 0.5).abs() < 1e-6)
    );
}

/// `base_score` is checked against the trained loss's output domain: a
/// logistic loss refuses a probability outside `(0, 1)`, and an
/// identity-link custom loss accepts one.
#[test]
fn base_score_domain_follows_the_trained_loss() {
    let d = labeled_dense(&[0.0, 1.0], 2, 1, &[0.0, 1.0]);
    let outside = TrainingParams::builder()
        .objective(Objective::BinaryLogistic(RegLoss::default()))
        .base_score(2.0)
        .build()
        .unwrap();
    assert!(matches!(
        train(&outside, &d, 1),
        Err(HessboostError::InvalidParameter {
            name: "base_score",
            ..
        })
    ));
    let identity = CustomLoss::new("custom:identity", 1, |p, y, _w, out| {
        for ((pair, &p), &y) in out.iter_mut().zip(p).zip(y) {
            *pair = GradPair::new(p - y, 1.0);
        }
    });
    let custom = TrainingParams::builder()
        .objective(Objective::custom(identity))
        .base_score(2.0)
        .build()
        .unwrap();
    let model = train(&custom, &d, 0).unwrap();
    assert_eq!(model.base_score(), 2.0);
}

#[test]
fn invalid_training_and_evaluation_inputs_return_errors() {
    let d = step_dataset(20);
    let params = TrainingParams::builder()
        .objective(Objective::BinaryLogistic(RegLoss::default()))
        .build()
        .unwrap();
    assert!(
        Trainer::new(&params, &d, 2)
            .early_stopping_rounds(1)
            .train()
            .is_err()
    );
    assert!(
        Trainer::new(&params, &d, 2)
            .eval(&d, "eval")
            .early_stopping_rounds(0)
            .train()
            .is_err()
    );

    let unlabeled = DMatrix::from_dense(&[0.0, 1.0], 2, 1).unwrap();
    assert!(
        Trainer::new(&params, &d, 2)
            .eval(&unlabeled, "eval")
            .train()
            .is_err()
    );
    let wrong_features = labeled_dense(&[0.0, 0.0], 1, 2, &[0.0]);
    assert!(
        Trainer::new(&params, &d, 2)
            .eval(&wrong_features, "eval")
            .train()
            .is_err()
    );
    let model = train(&params, &d, 2).unwrap();
    assert!(model.predict(&wrong_features).is_err());
    assert!(model.predict_margin(&wrong_features).is_err());
    assert!(model.predict_leaf(&wrong_features).is_err());
}

/// Label-domain checks run through `Loss::validate_info` for every
/// eval set, and the error names the offending dataset.
#[test]
fn eval_set_label_domain_errors_name_the_dataset() {
    let d = step_dataset(20);
    let params = TrainingParams::builder()
        .objective(Objective::BinaryLogistic(RegLoss::default()))
        .build()
        .unwrap();
    let holdout = labeled_dense(&[0.0, 1.0], 2, 1, &[0.0, 2.0]);
    match Trainer::new(&params, &d, 2)
        .eval(&holdout, "holdout")
        .train()
    {
        Err(HessboostError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "labels");
            assert_eq!(
                reason,
                "dataset `holdout` has labels outside the objective's valid domain"
            );
        }
        other => panic!("expected a label-domain error, got {other:?}"),
    }
}

/// Label matrices reach only the objectives and metrics that model them,
/// and every eval set must carry as many label columns as the training
/// matrix.
#[test]
fn target_count_mismatches_are_rejected() {
    let d = step_dataset(4);
    let params = TrainingParams::default();
    let x: Vec<f32> = (0..4).map(|i| i as f32).collect();
    let two_targets = DMatrix::from_dense(&x, 4, 1)
        .unwrap()
        .with_label_matrix(&[1.0; 8], 2)
        .unwrap();
    let poisson = TrainingParams::builder()
        .objective(Objective::Poisson)
        .build()
        .unwrap();
    assert!(matches!(
        train(&poisson, &two_targets, 1),
        Err(HessboostError::InvalidParameter { name, .. }) if name == "labels"
    ));
    assert!(matches!(
        Trainer::new(&params, &d, 1)
            .eval(&two_targets, "eval")
            .train(),
        Err(HessboostError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        Trainer::new(&params, &two_targets, 1)
            .eval(&d, "eval")
            .train(),
        Err(HessboostError::DimensionMismatch { .. })
    ));
    let ndcg = TrainingParams::builder()
        .eval_metric(crate::metric::EvalMetric::Ndcg(crate::metric::Cutoff::all()))
        .build()
        .unwrap();
    eval_metric_rejection(train(&ndcg, &two_targets, 1), "ndcg");
    // Three margins per row fit neither one per row nor one per target.
    let bad_margin = two_targets.clone().with_base_margin(&[0.0; 12]).unwrap();
    assert!(matches!(
        train(&params, &bad_margin, 1),
        Err(HessboostError::DimensionMismatch { .. })
    ));
}

/// 128 rows over two integer features, weighted, with a label matrix whose
/// two columns are unrelated functions of the features (probabilities
/// for the logistic objectives).
fn two_target_dataset(logistic: bool) -> (DMatrix, [Vec<f32>; 2], Vec<f32>) {
    let n = 128;
    let x: Vec<f32> = (0..n)
        .flat_map(|i| [(i % 32) as f32, ((i * 7) % 11) as f32])
        .collect();
    let (a, b): (Vec<f32>, Vec<f32>) = (0..n)
        .map(|i| {
            let (x0, x1) = (x[2 * i], x[2 * i + 1]);
            if logistic {
                (
                    f32::from(u8::from(x0 > 12.0)),
                    f32::from(u8::from(x1 < 4.0)),
                )
            } else {
                (x0 * 0.5 - 3.0, (x1 - 5.0).powi(2))
            }
        })
        .unzip();
    let matrix: Vec<f32> = a.iter().zip(&b).flat_map(|(&p, &q)| [p, q]).collect();
    let weights: Vec<f32> = (0..n).map(|i| 0.5 + (i % 4) as f32 * 0.5).collect();
    let d = DMatrix::from_dense(&x, n, 2)
        .unwrap()
        .with_label_matrix(&matrix, 2)
        .unwrap()
        .with_weights(&weights)
        .unwrap();
    (d, [a, b], weights)
}

/// With `one_output_per_tree`, output `j` of a multi-target model is
/// bit for bit the single-target model trained on label column `j`
/// (same trees, same per-target intercept), for every tree method and
/// multi-target objective.
#[test]
fn multi_target_outputs_equal_per_column_models() {
    for (objective, logistic) in [
        (Objective::SquaredError(RegLoss::default()), false),
        (Objective::PseudoHuber(PseudoHuber::default()), false),
        (Objective::BinaryLogistic(RegLoss::default()), true),
        (Objective::RegLogistic(RegLoss::default()), true),
    ] {
        let name = objective.name();
        for method in [TreeMethod::Hist, TreeMethod::Exact, TreeMethod::Approx] {
            let (d, cols, weights) = two_target_dataset(logistic);
            let params = TrainingParams::builder()
                .objective(objective.clone())
                .tree_method(method)
                .max_depth(3)
                .build()
                .unwrap();
            let model = train(&params, &d, 4).unwrap();
            assert_eq!((model.n_outputs(), model.n_targets()), (2, 2));
            let preds = model.predict(&d).unwrap();
            assert_eq!((preds.n_rows(), preds.width()), (d.n_rows(), 2));
            for (j, col) in cols.iter().enumerate() {
                let single = d
                    .clone()
                    .with_labels(col)
                    .unwrap()
                    .with_weights(&weights)
                    .unwrap();
                let reference = train(&params, &single, 4).unwrap();
                assert_eq!(
                    model.base_scores()[j].to_bits(),
                    reference.base_score().to_bits(),
                    "{name} {method:?} intercept {j}"
                );
                let expected = reference.predict(&single).unwrap();
                for (row, e) in expected.as_slice().iter().enumerate() {
                    assert_eq!(
                        preds.get(row, j).unwrap().to_bits(),
                        e.to_bits(),
                        "{name} {method:?} ({row},{j})"
                    );
                }
            }
        }
    }
}

/// A multi-target model round-trips through the native and XGBoost
/// formats with its target count, and `predict_class` thresholds each
/// label independently.
#[test]
fn multi_label_model_round_trips_and_classifies_per_label() {
    let (d, cols, _) = two_target_dataset(true);
    let params = TrainingParams::builder()
        .objective(Objective::BinaryLogistic(RegLoss::default()))
        .eta(0.5)
        .build()
        .unwrap();
    let model = train(&params, &d, 10).unwrap();
    let preds = model.predict(&d).unwrap();
    let classes = model.predict_class(&d).unwrap();
    assert_eq!((classes.n_rows(), classes.width()), (d.n_rows(), 2));
    for (i, (&c, &p)) in classes.as_slice().iter().zip(preds.as_slice()).enumerate() {
        assert_eq!(c, u32::from(p > 0.5), "cell {i}");
        assert_eq!(c as f32, cols[i % 2][i / 2], "separable cell {i}");
    }
    for restored in [
        BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap(),
        BoostedModel::from_json(&model.to_json().unwrap()).unwrap(),
        BoostedModel::from_xgboost_json(&model.to_xgboost_json().unwrap()).unwrap(),
        BoostedModel::from_xgboost_ubjson(&model.to_xgboost_ubjson().unwrap()).unwrap(),
    ] {
        assert_eq!(restored.n_targets(), 2);
        assert_eq!(restored.predict(&d).unwrap(), preds);
    }
}

/// An objective that learns from label bounds only: gradients and the
/// intercept come from `MetaInfo`, and no ordinary labels are required.
struct BoundsMidpoint;

impl BoundsMidpoint {
    fn target(info: &MetaInfo, row: usize) -> f32 {
        let lo = info.label_lower_bound.expect("validated")[row];
        let hi = info.label_upper_bound.expect("validated")[row];
        f32::midpoint(lo, hi)
    }
}

impl crate::objective::Loss for BoundsMidpoint {
    fn name(&self) -> &'static str {
        "test:bounds_midpoint"
    }

    fn gradient(&self, _: &[f32], _: &[f32], _: Option<&[f32]>, _: &mut [GradPair]) {
        unreachable!("training must call gradient_info");
    }

    fn gradient_info(&self, preds: &[f32], info: &MetaInfo, out: &mut [GradPair]) {
        for (row, (p, g)) in preds.iter().zip(out.iter_mut()).enumerate() {
            *g = GradPair::new(p - Self::target(info, row), 1.0);
        }
    }

    fn base_margins_info(&self, info: &MetaInfo) -> Vec<f32> {
        let sum: f32 = (0..info.n_rows).map(|row| Self::target(info, row)).sum();
        vec![sum / info.n_rows as f32]
    }

    fn validate_info(&self, info: &MetaInfo) -> Result<()> {
        if info.label_lower_bound.is_none() || info.label_upper_bound.is_none() {
            return Err(HessboostError::invalid_param(
                "label_lower_bound",
                "dataset has no label bounds",
            ));
        }
        Ok(())
    }

    fn requires_labels(&self) -> bool {
        false
    }

    fn default_metric(&self) -> crate::metric::EvalMetric {
        crate::metric::EvalMetric::Rmse
    }
}

#[test]
fn training_routes_through_metadata_hooks() {
    let x: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let lower: Vec<f32> = (0..32).map(|i| if i < 16 { 0.0 } else { 4.0 }).collect();
    let upper: Vec<f32> = lower.iter().map(|lo| lo + 2.0).collect();
    let d = DMatrix::from_dense(&x, 32, 1)
        .unwrap()
        .with_label_bounds(&lower, &upper)
        .unwrap();
    let params = TrainingParams::builder()
        .objective(Objective::custom(BoundsMidpoint))
        .eta(1.0)
        .lambda(0.0)
        .build()
        .unwrap();
    let model = train(&params, &d, 3).unwrap();
    // Base margin is the mean midpoint (1 and 5 → 3); one full-step tree
    // then lands every row on its own midpoint.
    assert_eq!(model.base_score(), 3.0);
    let preds = model.predict_margin(&d).unwrap().into_vec(); // one per row
    for (row, p) in preds.iter().enumerate() {
        let expected = if row < 16 { 1.0 } else { 5.0 };
        assert!((p - expected).abs() < 1e-3, "row {row}: {p}");
    }

    let unbounded = DMatrix::from_dense(&x, 32, 1).unwrap();
    assert!(matches!(
        train(&params, &unbounded, 1),
        Err(HessboostError::InvalidParameter { name, .. }) if name == "label_lower_bound"
    ));
}

/// The metric XGBoost names `name`, with default parameters.
fn named_metric(name: &str) -> crate::metric::EvalMetric {
    crate::metric::EvalMetric::from_xgboost(name, &crate::metric::DEFAULT_SOURCE).unwrap()
}

/// The reason of an `eval_metric` rejection of the `context` run,
/// panicking on any other outcome.
fn eval_metric_rejection<T: std::fmt::Debug>(result: Result<T>, context: &str) -> String {
    match result {
        Err(HessboostError::InvalidParameter {
            name: "eval_metric",
            reason,
        }) => reason,
        other => panic!("{context}: expected an `eval_metric` rejection, got {other:?}"),
    }
}

#[test]
fn label_metrics_are_refused_on_bound_only_eval_sets() {
    // `survival:aft` trains on label bounds alone, so the label slice is
    // empty: metrics reading ordinary labels must be refused, not index it.
    let x: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let lower: Vec<f32> = (0..32).map(|i| 1.0 + (i % 4) as f32).collect();
    let upper: Vec<f32> = lower.iter().map(|lo| lo + 1.0).collect();
    let d = DMatrix::from_dense(&x, 32, 1)
        .unwrap()
        .with_label_bounds(&lower, &upper)
        .unwrap();
    let params = |metric: &str| {
        TrainingParams::builder()
            .objective(Objective::Aft(Aft::default()))
            .eval_metric(named_metric(metric))
            .build()
            .unwrap()
    };
    for metric in ["aft-nloglik", "interval-regression-accuracy"] {
        let run = Trainer::new(&params(metric), &d, 2)
            .eval(&d, "eval")
            .train()
            .unwrap();
        assert!(run.history[1].scores[0].value.is_finite(), "{metric}");
    }
    for metric in ["rmse", "mae", "cox-nloglik"] {
        let run = Trainer::new(&params(metric), &d, 2)
            .eval(&d, "eval")
            .train();
        let reason = eval_metric_rejection(run, metric);
        assert!(reason.contains("`eval`"), "{reason}");
    }
}

/// A multi-output model that is not multiclass (three quantiles) gives
/// `mlogloss` / `merror` their width, but its regression labels are not
/// class indices: the metrics refuse the eval set instead of reading a
/// probability past the row.
#[test]
fn class_index_metrics_refuse_non_class_labels() {
    let x: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let y: Vec<f32> = (0..32).map(|i| 10.0 + i as f32).collect();
    let d = labeled_dense(&x, 32, 1, &y);
    for metric in ["mlogloss", "merror"] {
        let params = TrainingParams::builder()
            .objective(Objective::Quantile(
                crate::objective::Quantiles::new([0.1, 0.5, 0.9]).unwrap(),
            ))
            .eval_metric(named_metric(metric))
            .build()
            .unwrap();
        let run = Trainer::new(&params, &d, 2).eval(&d, "eval").train();
        let reason = eval_metric_rejection(run, metric);
        assert!(reason.contains("class"), "{reason}");
    }
}

#[test]
fn per_row_metrics_refuse_label_matrices() {
    // The survival metrics read one interval and weight per row, and
    // `pre@k` ranks one label per row within query groups; on a label
    // matrix they must be refused rather than index the row weights per
    // cell or rank every target's cells together.
    let n = 24;
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let y: Vec<f32> = (0..2 * n).map(|i| (i % 5) as f32).collect();
    let d = DMatrix::from_dense(&x, n, 1)
        .unwrap()
        .with_label_matrix(&y, 2)
        .unwrap()
        .with_weights(&vec![1.0; n])
        .unwrap()
        .with_group_sizes(&[12, 12])
        .unwrap();
    for metric in ["aft-nloglik", "interval-regression-accuracy", "pre@3"] {
        let params = TrainingParams::builder()
            .eval_metric(named_metric(metric))
            .build()
            .unwrap();
        eval_metric_rejection(
            Trainer::new(&params, &d, 1).eval(&d, "eval").train(),
            metric,
        );
    }
}

#[test]
fn metrics_must_match_the_prediction_width() {
    // Elementwise metrics read one prediction per label; a model with
    // more outputs than label columns must be refused before the first
    // evaluation instead of indexing past the labels.
    let n = 30;
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let y: Vec<f32> = (0..n).map(|i| 1.0 + (i % 3) as f32).collect();
    let d = labeled_dense(&x, n, 1, &y);
    let quantile = || {
        TrainingParams::builder().objective(Objective::Quantile(
            crate::objective::Quantiles::new(vec![0.2, 0.8]).unwrap(),
        ))
    };
    let expectile = || {
        TrainingParams::builder().objective(Objective::Expectile(
            crate::objective::Expectiles::new(vec![0.3, 0.5, 0.7]).unwrap(),
        ))
    };
    let normal = || {
        TrainingParams::builder().objective(Objective::Dist(
            crate::objective::distributional::Distributional::new(
                crate::objective::distributional::DistFamily::Normal,
            ),
        ))
    };
    let softprob = || {
        TrainingParams::builder().objective(Objective::Softprob(
            crate::objective::Multiclass::new(4).unwrap(),
        ))
    };
    let run = |params: TrainingParams| {
        Trainer::new(&params, &d, 2)
            .eval(&d, "eval")
            .train()
            .map(|r| r.history)
    };
    for (params, metric) in [
        (
            quantile().eval_metric(crate::metric::EvalMetric::Rmse),
            "rmse",
        ),
        (
            expectile().eval_metric(crate::metric::EvalMetric::Mae),
            "mae",
        ),
        (
            normal().eval_metric(crate::metric::EvalMetric::Rmse),
            "rmse",
        ),
        (
            softprob().eval_metric(crate::metric::EvalMetric::Rmse),
            "rmse",
        ),
        (
            softprob().eval_metric(crate::metric::EvalMetric::Auc),
            "auc",
        ),
        (
            quantile().eval_metric(crate::metric::EvalMetric::LogLoss),
            "logloss",
        ),
    ] {
        let reason = eval_metric_rejection(run(params.build().unwrap()), metric);
        assert!(
            reason.contains(metric) && reason.contains("`eval`"),
            "{reason}"
        );
    }
    // The multiclass metrics read one probability per output: a
    // single-output model is refused before any eval set is read.
    let single = TrainingParams::builder()
        .eval_metric(crate::metric::EvalMetric::MLogLoss)
        .build()
        .unwrap();
    let reason = eval_metric_rejection(run(single), "mlogloss");
    assert!(reason.contains("mlogloss"), "{reason}");
    // The defaults and matching metrics still evaluate.
    for params in [
        quantile(),
        quantile().eval_metric(crate::metric::EvalMetric::Quantile(
            crate::objective::Quantiles::new([0.2, 0.8]).unwrap(),
        )),
        expectile(),
        normal(),
        normal().eval_metric(crate::metric::EvalMetric::Crps(
            crate::objective::distributional::DistFamily::Normal,
        )),
        softprob(),
        softprob().eval_metric(crate::metric::EvalMetric::MError),
    ] {
        let history = run(params.build().unwrap()).unwrap();
        assert!(
            history[1].scores.iter().all(|s| s.value.is_finite()),
            "{history:?}"
        );
    }
}

#[test]
fn custom_objective_outputs_must_match_the_label_layout() {
    // A custom objective reads one label per row or one per output; a
    // label matrix of another width must be refused before it reaches the
    // gradient closure.
    let n = 20;
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let y: Vec<f32> = (0..2 * n).map(|i| (i % 4) as f32).collect();
    let d = DMatrix::from_dense(&x, n, 1)
        .unwrap()
        .with_label_matrix(&y, 2)
        .unwrap();
    let objective = |k: usize| {
        Objective::custom(CustomLoss::new("custom:k", k, |_p, _y, _w, out| {
            for g in out.iter_mut() {
                *g = GradPair::new(0.1, 1.0);
            }
        }))
    };
    let sums = || {
        Box::new(crate::metric::CustomMetric::new(
            "sum",
            false,
            |p, _y, _w| p.iter().map(|&v| f64::from(v)).sum(),
        ))
    };
    let run = |k: usize| {
        let params = TrainingParams::builder()
            .objective(objective(k))
            .build()
            .unwrap();
        Trainer::new(&params, &d, 1)
            .eval(&d, "eval")
            .custom_metric(sums())
            .train()
    };
    assert!(run(2).is_ok());
    for k in [1, 3] {
        assert!(matches!(
            run(k),
            Err(HessboostError::InvalidParameter { name, .. }) if name == "objective"
        ));
    }
}

#[test]
fn exact_interaction_constraints_confine_each_path() {
    fn visit(tree: &RegTree, node: usize, path: &mut Vec<u32>) {
        let current = tree.node(node);
        if current.is_leaf() {
            assert!(path.iter().all(|feature| *feature == path[0]));
            return;
        }
        path.push(current.split_feature);
        visit(tree, current.left as usize, path);
        visit(tree, current.right as usize, path);
        path.pop();
    }

    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..128 {
        let a = (i & 1) as f32;
        let b = ((i >> 1) & 1) as f32;
        x.extend_from_slice(&[a, b]);
        y.push(f32::from(a != b));
    }
    let d = labeled_dense(&x, 128, 2, &y);
    let params = TrainingParams::builder()
        .tree_method(TreeMethod::Exact)
        .max_depth(3)
        .interaction_constraints(vec![vec![0], vec![1]])
        .build()
        .unwrap();
    let model = train(&params, &d, 3).unwrap();
    for tree in model.trees() {
        visit(tree, 0, &mut Vec::new());
    }
}
