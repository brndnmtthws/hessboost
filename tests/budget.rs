//! Budget-mode training (`train_with_budget`, Perpetual's algorithm):
//! budget monotonicity, held-out quality against validation-tuned training,
//! determinism, model compatibility, the objectives' pointwise losses, and
//! the configuration contract.

use hessboost::config::MaxDeltaStep;
use hessboost::data::FeatureType;
use hessboost::objective::{GradPair, Multiclass, PseudoHuber, RegLoss, Tweedie};
use hessboost::prelude::*;
use hessboost::training::budget::{BudgetConfig, BudgetStop, train_with_budget};

mod common;
use common::{labeled_dense, lcg};

const N_FEATURES: usize = 6;

/// Friedman-#1-style rows: features, noiseless target.
fn friedman(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut next = lcg(seed);
    let mut x = Vec::with_capacity(n * N_FEATURES);
    let mut f = Vec::with_capacity(n);
    for _ in 0..n {
        let row: Vec<f32> = (0..N_FEATURES).map(|_| next()).collect();
        f.push(
            10.0 * (std::f32::consts::PI * row[0] * row[1]).sin()
                + 20.0 * (row[2] - 0.5).powi(2)
                + 10.0 * row[3]
                + 5.0 * row[4],
        );
        x.extend_from_slice(&row);
    }
    (x, f)
}

/// Regression with unit-variance noise.
fn regression(n: usize, seed: u64) -> DMatrix {
    let (x, f) = friedman(n, seed);
    let mut next = lcg(seed ^ 0x5eed);
    let y: Vec<f32> = f
        .iter()
        .map(|v| v + (0..12).map(|_| next()).sum::<f32>() - 6.0)
        .collect();
    labeled_dense(&x, N_FEATURES, &y)
}

/// Binary labels drawn with probability `σ((f − 14) / 2)`.
fn binary(n: usize, seed: u64) -> DMatrix {
    let (x, f) = friedman(n, seed);
    let mut next = lcg(seed ^ 0xb1);
    let y: Vec<f32> = f
        .iter()
        .map(|v| f32::from(next() < 1.0 / (1.0 + (-(v - 14.0) / 2.0).exp())))
        .collect();
    labeled_dense(&x, N_FEATURES, &y)
}

fn params(objective: Objective) -> TrainingParams {
    TrainingParams::builder()
        .objective(objective)
        .build()
        .unwrap()
}

/// Mean squared error (regression) or log loss (binary) of `model` on `data`.
fn loss(model: &BoostedModel, data: &DMatrix) -> f64 {
    let preds = model.predict(data).unwrap().into_vec(); // one value per row
    let labels = data.labels().unwrap();
    let total: f64 = preds
        .iter()
        .zip(labels)
        .map(|(&p, &y)| {
            let (p, y) = (f64::from(p), f64::from(y));
            if model.objective().name() == "binary:logistic" {
                let p = p.clamp(1e-15, 1.0 - 1e-15);
                -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
            } else {
                (p - y).powi(2)
            }
        })
        .sum();
    total / preds.len() as f64
}

/// A larger budget trains more trees and fits unseen data at least as well.
/// (The *training* loss is not monotone in the budget: the generalization
/// gate lets a small budget's large steps overfit small leaves. Upstream
/// Perpetual behaves the same way on this data.)
#[test]
fn larger_budgets_train_more_trees_and_fit_held_out_data_at_least_as_well() {
    for (objective, data, test) in [
        (
            Objective::SquaredError(RegLoss::default()),
            regression(1500, 1),
            regression(4000, 101),
        ),
        (
            Objective::BinaryLogistic(RegLoss::default()),
            binary(1500, 1),
            binary(4000, 101),
        ),
    ] {
        let name = objective.name();
        let mut previous: Option<(usize, f64)> = None;
        for budget in [0.5, 1.0, 1.5] {
            let result = train_with_budget(
                &params(objective.clone()),
                &data,
                &BudgetConfig::new(budget),
            )
            .unwrap();
            let trees = result.model.num_trees();
            let test_loss = loss(&result.model, &test);
            if let Some((prev_trees, prev_loss)) = previous {
                assert!(
                    trees > prev_trees,
                    "{name} budget {budget}: {trees} trees <= {prev_trees}"
                );
                assert!(
                    test_loss <= prev_loss,
                    "{name} budget {budget}: held-out loss {test_loss} > {prev_loss}"
                );
            }
            previous = Some((trees, test_loss));
        }
    }
}

/// Budget 1.0 without any tuning lands within a fixed band of a run whose
/// round count was tuned by early stopping on a validation set (small
/// learning rate, otherwise XGBoost defaults), and clearly beats untuned
/// default training.
#[test]
fn held_out_quality_is_comparable_to_validation_tuned_training() {
    for (objective, train_set, valid, test, band) in [
        (
            Objective::SquaredError(RegLoss::default()),
            regression(1500, 3),
            regression(1000, 4),
            regression(4000, 5),
            1.10,
        ),
        (
            Objective::BinaryLogistic(RegLoss::default()),
            binary(1500, 6),
            binary(1000, 7),
            binary(4000, 8),
            1.10,
        ),
    ] {
        let name = objective.name();
        let budget = train_with_budget(
            &params(objective.clone()),
            &train_set,
            &BudgetConfig::new(1.0),
        )
        .unwrap();
        let tuned_params = TrainingParams::builder()
            .objective(objective.clone())
            .eta(0.05)
            .build()
            .unwrap();
        let tuned = Trainer::new(&tuned_params, &train_set, 2000)
            .eval(&valid, "valid")
            .early_stopping_rounds(50)
            .train()
            .unwrap()
            .model;
        let untuned = train(&params(objective.clone()), &train_set, 100).unwrap();
        let (budget_loss, tuned_loss) = (loss(&budget.model, &test), loss(&tuned, &test));
        assert!(
            budget_loss <= band * tuned_loss,
            "{name}: budget {budget_loss} vs tuned {tuned_loss}"
        );
        assert!(
            budget_loss < loss(&untuned, &test),
            "{name}: budget {budget_loss} is not better than untuned default training"
        );
    }
}

#[test]
fn training_is_deterministic_and_independent_of_the_thread_count() {
    let data = binary(6000, 9);
    let run = |nthread: usize| {
        let p = TrainingParams::builder()
            .objective(Objective::BinaryLogistic(RegLoss::default()))
            .nthread(nthread)
            .build()
            .unwrap();
        let result = train_with_budget(&p, &data, &BudgetConfig::new(0.5)).unwrap();
        (result.model.to_json().unwrap(), result.stop)
    };
    let serial = run(1);
    assert_eq!(serial, run(1));
    assert_eq!(serial, run(4));
}

#[test]
fn budget_models_are_ordinary_gbtree_models() {
    let data = regression(800, 10);
    let model = train_with_budget(
        &params(Objective::SquaredError(RegLoss::default())),
        &data,
        &BudgetConfig::new(0.5),
    )
    .unwrap()
    .model;
    let preds = model.predict(&data).unwrap();
    let via_xgboost = BoostedModel::from_xgboost_json(&model.to_xgboost_json().unwrap()).unwrap();
    let via_native = BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
    for other in [via_xgboost, via_native] {
        let round_trip = other.predict(&data).unwrap();
        for (a, b) in preds.as_slice().iter().zip(round_trip.as_slice()) {
            assert!((a - b).abs() <= 1e-5 * a.abs().max(1.0), "{a} vs {b}");
        }
    }
}

/// Each supported objective's pointwise loss has the objective's gradient as
/// its margin derivative (and, except for Poisson's `max_delta_step`
/// safeguard, its Hessian as the second derivative).
#[test]
fn pointwise_losses_differentiate_to_the_objective_gradients() {
    let weighted = || RegLoss::new(2.0).unwrap();
    let cases: [(Objective, &[f32]); 7] = [
        (
            Objective::SquaredError(RegLoss::default()),
            &[-1.5, 0.0, 2.0],
        ),
        (
            Objective::PseudoHuber(PseudoHuber::default()),
            &[-1.5, 0.0, 2.0],
        ),
        (Objective::BinaryLogistic(weighted()), &[0.0, 1.0]),
        (Objective::RegLogistic(weighted()), &[0.3, 0.8]),
        (Objective::Poisson, &[0.0, 3.0]),
        (Objective::Gamma(RegLoss::default()), &[0.5, 4.0]),
        (Objective::Tweedie(Tweedie::default()), &[0.0, 2.5]),
    ];
    let margins = [-1.2f32, 0.1, 0.9];
    for (spec, labels) in cases {
        let name = spec.name();
        let objective = params(spec.clone()).loss(1).unwrap();
        let loss = objective.pointwise_loss().expect(name);
        for &y in labels {
            for &m in &margins {
                let mut out = [GradPair::default()];
                objective.gradient(&[m], &[y], None, &mut out);
                let h = 1e-3f32;
                let d1 = (loss(m + h, y) - loss(m - h, y)) / (2.0 * f64::from(h));
                let d2 = (loss(m + h, y) - 2.0 * loss(m, y) + loss(m - h, y)) / f64::from(h * h);
                let tol = |v: f64| 2e-2 * v.abs().max(1.0);
                let grad = f64::from(out[0].grad);
                assert!(
                    (d1 - grad).abs() <= tol(grad),
                    "{name} y={y} m={m}: {d1} vs {grad}"
                );
                if name != "count:poisson" {
                    let hess = f64::from(out[0].hess);
                    assert!(
                        (d2 - hess).abs() <= tol(hess),
                        "{name} y={y} m={m}: {d2} vs {hess}"
                    );
                }
            }
        }
    }
}

/// Missing values and categorical features: the label is decided by whether
/// feature 0 is missing and by which category feature 1 holds; the budget
/// trainer must learn both.
#[test]
fn learns_missing_value_directions_and_categorical_splits() {
    let n = 2000;
    let mut next = lcg(11);
    let mut x = Vec::with_capacity(2 * n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let missing = next() < 0.4;
        let category = (next() * 8.0).floor().min(7.0);
        x.push(if missing { f32::NAN } else { next() });
        x.push(category);
        let lucky = [1.0, 4.0, 6.0].contains(&category);
        y.push(f32::from(missing) * 2.0 + f32::from(lucky) + 0.1 * (next() - 0.5));
    }
    let data = labeled_dense(&x, 2, &y)
        .with_feature_types(&[FeatureType::Numerical, FeatureType::Categorical])
        .unwrap();
    let result = train_with_budget(
        &params(Objective::SquaredError(RegLoss::default())),
        &data,
        &BudgetConfig::new(1.0),
    )
    .unwrap();
    assert!(
        result
            .model
            .trees()
            .iter()
            .any(|t| t.nodes().iter().any(|node| node.is_categorical)),
        "no categorical split was grown"
    );
    let rmse = loss(&result.model, &data).sqrt();
    assert!(rmse < 0.1, "training rmse {rmse}");
}

#[test]
fn iteration_limit_caps_the_rounds() {
    let data = regression(1000, 12);
    let result = train_with_budget(
        &params(Objective::SquaredError(RegLoss::default())),
        &data,
        &BudgetConfig::new(1.5).iteration_limit(7),
    )
    .unwrap();
    assert_eq!(result.model.num_trees(), 7);
    assert_eq!(result.stop, BudgetStop::IterationLimit);
}

/// An unbounded `stopping_rounds` override is valid: the target-round check
/// must not overflow, so `usize::MAX` trains the same model as any other
/// round count far beyond the iteration limit.
#[test]
fn unbounded_stopping_rounds_override_trains() {
    let data = regression(300, 16);
    let config = BudgetConfig::new(0.5).iteration_limit(30);
    let train = |rounds| {
        let result = train_with_budget(
            &params(Objective::SquaredError(RegLoss::default())),
            &data,
            &config.stopping_rounds(rounds),
        )
        .unwrap();
        result.model.predict(&data).unwrap()
    };
    assert_eq!(train(usize::MAX), train(1 << 40));
}

/// A lone positive count among 999 zeros: an unbounded Newton leaf would step
/// its margin by about `+157`, overflowing the next round's gradients and
/// turning every prediction into NaN. The `max_delta_step` bound keeps the
/// steps, and so the predictions, finite and moving toward the labels.
#[test]
fn poisson_leaf_steps_are_bounded() {
    let n = 1000;
    let mut x = vec![0.0f32; n];
    x[n - 1] = 1.0;
    let data = labeled_dense(&x, 1, &x);
    let result =
        train_with_budget(&params(Objective::Poisson), &data, &BudgetConfig::new(0.5)).unwrap();
    let preds = result.model.predict(&data).unwrap().into_vec(); // one value per row
    assert!(preds.iter().all(|p| p.is_finite()), "{:?}", result.stop);
    assert!(preds[0] < 1e-3, "zero-count prediction {}", preds[0]);
    assert!(
        preds[n - 1] > 0.5,
        "positive-count prediction {}",
        preds[n - 1]
    );
}

/// A tree whose step overflows the training loss is not appended: a lone
/// `1e-30` Gamma label among ones gets a leaf of about `−3·10²⁹`, whose loss
/// `y/μ` is infinite (and would turn the next gradients, and every
/// prediction, into NaN).
#[test]
fn non_finite_steps_are_not_appended() {
    let n = 1000;
    let mut x = vec![0.0f32; n];
    x[n - 1] = 1.0;
    let mut y = vec![1.0f32; n];
    y[n - 1] = 1e-30;
    let data = labeled_dense(&x, 1, &y);
    let result = train_with_budget(
        &params(Objective::Gamma(RegLoss::default())),
        &data,
        &BudgetConfig::new(0.5),
    )
    .unwrap();
    assert_eq!(result.stop, BudgetStop::NonFiniteLoss);
    let preds = result.model.predict(&data).unwrap();
    assert!(preds.as_slice().iter().all(|p| p.is_finite()));
}

/// Four rows leave no fold with rows on both sides, and the root gradient
/// sums to zero, so every split's generalization ratio is `0/0`. That must
/// not rank the first threshold above the rest: the perfect split is taken.
#[test]
fn undefined_root_generalization_keeps_the_best_split() {
    let data = labeled_dense(&[0.0, 1.0, 2.0, 3.0], 1, &[0.0, 0.0, 1.0, 1.0]);
    let result = train_with_budget(
        &params(Objective::SquaredError(RegLoss::default())),
        &data,
        &BudgetConfig::default().iteration_limit(1),
    )
    .unwrap();
    let p = result.model.predict(&data).unwrap().into_vec(); // one value per row
    assert!(p[0] == p[1] && p[2] == p[3] && p[1] < p[2], "{p:?}");
}

/// One tree of `data`, which must train and load back from the native
/// format; returns its training predictions and the reason training stopped.
fn one_saved_tree(data: &DMatrix) -> (Vec<f32>, BudgetStop) {
    let result = train_with_budget(
        &params(Objective::SquaredError(RegLoss::default())),
        data,
        &BudgetConfig::default().iteration_limit(1),
    )
    .unwrap();
    let loaded = BoostedModel::from_bytes(&result.model.to_bytes().unwrap()).unwrap();
    (loaded.predict(data).unwrap().into_vec(), result.stop)
}

/// Separating the `±1e20` labels gains about `1e40`, which `f32` cannot
/// hold: that split is skipped for the representable one isolating the last
/// row, instead of storing an infinite gain the formats refuse to load.
#[test]
fn unrepresentable_split_gains_are_skipped() {
    let data = labeled_dense(&[0.0, 1.0, 2.0], 1, &[1e20, -1e20, 5.0]);
    let (p, _) = one_saved_tree(&data);
    assert!(p[0] == p[1] && p[1] < p[2], "{p:?}");

    // With no representable split the root stays a leaf.
    let data = labeled_dense(&[0.0, 1.0], 1, &[-1e20, 1e20]);
    let (p, stop) = one_saved_tree(&data);
    assert_eq!((p[0], stop), (p[1], BudgetStop::RootUnsplittable));
}

/// A root whose Hessian sum overflows `f32` cannot be stored: training fails
/// like the ordinary trainer does rather than returning an unloadable model.
#[test]
fn roots_with_overflowing_hessian_sums_are_errors() {
    let data = labeled_dense(&[0.0, 0.0], 1, &[0.0, 0.0])
        .with_weights(&[3e38, 3e38])
        .unwrap();
    let params = params(Objective::SquaredError(RegLoss::default()));
    assert!(matches!(
        train(&params, &data, 1),
        Err(HessboostError::ModelFormat(_))
    ));
    assert!(matches!(
        train_with_budget(&params, &data, &BudgetConfig::default()),
        Err(HessboostError::ModelFormat(_))
    ));
}

/// A root of at most eight rows without usable folds falls back to plain
/// positive-gain splits over the same partitions the fold-checked search
/// scores: here the gainful ones send the missing row left or split present
/// from missing values, and on a categorical feature separate the categories.
#[test]
fn tiny_roots_try_missing_directions_and_categorical_splits() {
    let data = labeled_dense(&[0.0, 1.0, f32::NAN], 1, &[0.0, 1.0, -1.0]);
    let (p, _) = one_saved_tree(&data);
    assert!(p[0] == p[2] && p[0] < p[1], "{p:?}");

    let data = labeled_dense(&[0.0, 1.0], 1, &[0.0, 1.0])
        .with_feature_types(&[FeatureType::Categorical])
        .unwrap();
    let (p, _) = one_saved_tree(&data);
    assert!(p[0] < p[1], "{p:?}");
}

/// The `(name, reason)` of the invalid-parameter error `train_with_budget`
/// refuses `params` / `config` with.
fn rejection(
    params: &TrainingParams,
    data: &DMatrix,
    config: &BudgetConfig,
) -> (&'static str, String) {
    match train_with_budget(params, data, config) {
        Err(HessboostError::InvalidParameter { name, reason }) => (name, reason),
        other => panic!("expected an invalid-parameter error, got {other:?}"),
    }
}

#[test]
fn derived_or_unused_parameters_are_rejected_by_name() {
    let data = regression(200, 13);
    let tuned = TrainingParams::builder()
        .eta(0.1)
        .max_depth(3)
        .build()
        .unwrap();
    let (name, reason) = rejection(&tuned, &data, &BudgetConfig::default());
    assert_eq!(name, "budget");
    for name in ["eta", "max_depth"] {
        assert!(reason.contains(&format!("`{name}`")), "{name}: {reason}");
    }

    // Objective parameters and the parameters budget mode reads are accepted.
    let accepted = TrainingParams::builder()
        .objective(Objective::BinaryLogistic(RegLoss::new(3.0).unwrap()))
        .max_bin(64)
        .nthread(2)
        .base_score(0.4)
        .build()
        .unwrap();
    train_with_budget(&accepted, &binary(300, 14), &BudgetConfig::default()).unwrap();
    let huber = TrainingParams::builder()
        .objective(Objective::PseudoHuber(PseudoHuber::new(2.0).unwrap()))
        .build()
        .unwrap();
    train_with_budget(&huber, &data, &BudgetConfig::default()).unwrap();

    // `max_delta_step` is read by `count:poisson` only: accepted there,
    // unused (and so refused) with any other objective.
    let bounded = |objective| {
        TrainingParams::builder()
            .objective(objective)
            .max_delta_step(MaxDeltaStep::Bounded(0.5))
            .build()
            .unwrap()
    };
    let counts = labeled_dense(&[0.0, 1.0, 2.0, 3.0], 1, &[0.0, 1.0, 2.0, 3.0]);
    train_with_budget(
        &bounded(Objective::Poisson),
        &counts,
        &BudgetConfig::default(),
    )
    .unwrap();
    let (name, reason) = rejection(
        &bounded(Objective::SquaredError(RegLoss::default())),
        &data,
        &BudgetConfig::default(),
    );
    assert_eq!(name, "budget");
    assert!(reason.contains("`max_delta_step`"), "{reason}");
}

#[test]
fn unsupported_objectives_and_budgets_are_rejected() {
    let data = binary(200, 15);
    let multiclass = TrainingParams::builder()
        .objective(Objective::Softprob(Multiclass::new(2).unwrap()))
        .build()
        .unwrap();
    let (name, _) = rejection(&multiclass, &data, &BudgetConfig::default());
    assert_eq!(name, "objective");
    for budget in [0.0, -1.0, 5.0, f64::NAN] {
        let (name, _) = rejection(
            &params(Objective::BinaryLogistic(RegLoss::default())),
            &data,
            &BudgetConfig::new(budget),
        );
        assert_eq!(name, "budget", "budget {budget}");
    }
}

/// A custom loss that supplies reduced split gradients (vector-leaf trees
/// only) is refused by budget mode, as by the other trainers, rather than
/// trained on its full gradients.
#[test]
fn custom_split_gradients_are_refused() {
    use hessboost::objective::{Loss, PointwiseLoss, SplitGradient};
    struct Reduced;
    impl Loss for Reduced {
        fn name(&self) -> &'static str {
            "custom:reduced"
        }
        fn gradient(&self, preds: &[f32], labels: &[f32], _: Option<&[f32]>, out: &mut [GradPair]) {
            for ((g, p), y) in out.iter_mut().zip(preds).zip(labels) {
                *g = GradPair::new(p - y, 1.0);
            }
        }
        fn split_gradient(&self, _: usize, gpair: &[GradPair]) -> Option<SplitGradient> {
            Some(SplitGradient::new(gpair.to_vec(), 1))
        }
        fn pointwise_loss(&self) -> Option<PointwiseLoss<'_>> {
            Some(Box::new(|margin, label| {
                f64::from((margin - label).powi(2)) / 2.0
            }))
        }
        fn default_metric(&self) -> EvalMetric {
            EvalMetric::Rmse
        }
    }
    let (x, y) = friedman(500, 3);
    let data = labeled_dense(&x, N_FEATURES, &y);
    let params = TrainingParams::builder()
        .objective(Objective::custom(Reduced))
        .build()
        .unwrap();
    match train_with_budget(&params, &data, &BudgetConfig::default()) {
        Err(HessboostError::InvalidParameter { name, .. }) => assert_eq!(name, "objective"),
        other => panic!(
            "expected an `objective` refusal, got {:?}",
            other.map(|r| r.stop)
        ),
    }
}
