//! Budget-mode training (`train_with_budget`, Perpetual's algorithm):
//! budget monotonicity, held-out quality against validation-tuned training,
//! determinism, model compatibility, the objectives' pointwise losses, and
//! the configuration contract.

use hessboost::objective::create_objective;
use hessboost::prelude::*;

/// 64-bit LCG returning uniforms in `[0, 1)`.
fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32) / (1u32 << 24) as f32
    }
}

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
    DMatrix::from_dense(&x, n, N_FEATURES)
        .unwrap()
        .with_labels(&y)
        .unwrap()
}

/// Binary labels drawn with probability `σ((f − 14) / 2)`.
fn binary(n: usize, seed: u64) -> DMatrix {
    let (x, f) = friedman(n, seed);
    let mut next = lcg(seed ^ 0xb1);
    let y: Vec<f32> = f
        .iter()
        .map(|v| f32::from(next() < 1.0 / (1.0 + (-(v - 14.0) / 2.0).exp())))
        .collect();
    DMatrix::from_dense(&x, n, N_FEATURES)
        .unwrap()
        .with_labels(&y)
        .unwrap()
}

fn params(objective: &str) -> TrainingParams {
    TrainingParams::builder()
        .objective(objective)
        .build()
        .unwrap()
}

/// Mean squared error (regression) or log loss (binary) of `model` on `data`.
fn loss(model: &BoostedModel, data: &DMatrix) -> f64 {
    let preds = model.predict(data).unwrap();
    let labels = data.labels().unwrap();
    let total: f64 = preds
        .iter()
        .zip(labels)
        .map(|(&p, &y)| {
            let (p, y) = (f64::from(p), f64::from(y));
            if model.objective() == "binary:logistic" {
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
            "reg:squarederror",
            regression(1500, 1),
            regression(4000, 101),
        ),
        ("binary:logistic", binary(1500, 1), binary(4000, 101)),
    ] {
        let mut previous: Option<(usize, f64)> = None;
        for budget in [0.5, 1.0, 1.5] {
            let result =
                train_with_budget(&params(objective), &data, &BudgetConfig::new(budget)).unwrap();
            let trees = result.model.num_trees();
            let test_loss = loss(&result.model, &test);
            if let Some((prev_trees, prev_loss)) = previous {
                assert!(
                    trees > prev_trees,
                    "{objective} budget {budget}: {trees} trees <= {prev_trees}"
                );
                assert!(
                    test_loss <= prev_loss,
                    "{objective} budget {budget}: held-out loss {test_loss} > {prev_loss}"
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
            "reg:squarederror",
            regression(1500, 3),
            regression(1000, 4),
            regression(4000, 5),
            1.10,
        ),
        (
            "binary:logistic",
            binary(1500, 6),
            binary(1000, 7),
            binary(4000, 8),
            1.10,
        ),
    ] {
        let budget =
            train_with_budget(&params(objective), &train_set, &BudgetConfig::new(1.0)).unwrap();
        let tuned_params = TrainingParams::builder()
            .objective(objective)
            .eta(0.05)
            .build()
            .unwrap();
        let tuned = train_with_eval(
            &tuned_params,
            &train_set,
            2000,
            &[(&valid, "valid")],
            Some(50),
        )
        .unwrap()
        .model;
        let untuned = train(&params(objective), &train_set, 100).unwrap();
        let (budget_loss, tuned_loss) = (loss(&budget.model, &test), loss(&tuned, &test));
        assert!(
            budget_loss <= band * tuned_loss,
            "{objective}: budget {budget_loss} vs tuned {tuned_loss}"
        );
        assert!(
            budget_loss < loss(&untuned, &test),
            "{objective}: budget {budget_loss} is not better than untuned default training"
        );
    }
}

#[test]
fn training_is_deterministic_and_independent_of_the_thread_count() {
    let data = binary(6000, 9);
    let run = |nthread: usize| {
        let p = TrainingParams::builder()
            .objective("binary:logistic")
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
    let model = train_with_budget(&params("reg:squarederror"), &data, &BudgetConfig::new(0.5))
        .unwrap()
        .model;
    let preds = model.predict(&data).unwrap();
    let via_xgboost = BoostedModel::from_xgboost_json(&model.to_xgboost_json().unwrap()).unwrap();
    let via_native = BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
    for other in [via_xgboost, via_native] {
        let round_trip = other.predict(&data).unwrap();
        for (a, b) in preds.iter().zip(&round_trip) {
            assert!((a - b).abs() <= 1e-5 * a.abs().max(1.0), "{a} vs {b}");
        }
    }
}

/// Each supported objective's pointwise loss has the objective's gradient as
/// its margin derivative (and, except for Poisson's `max_delta_step`
/// safeguard, its Hessian as the second derivative).
#[test]
fn pointwise_losses_differentiate_to_the_objective_gradients() {
    let cases: [(&str, &[f32]); 7] = [
        ("reg:squarederror", &[-1.5, 0.0, 2.0]),
        ("reg:pseudohubererror", &[-1.5, 0.0, 2.0]),
        ("binary:logistic", &[0.0, 1.0]),
        ("reg:logistic", &[0.3, 0.8]),
        ("count:poisson", &[0.0, 3.0]),
        ("reg:gamma", &[0.5, 4.0]),
        ("reg:tweedie", &[0.0, 2.5]),
    ];
    let margins = [-1.2f32, 0.1, 0.9];
    for (name, labels) in cases {
        let p = TrainingParams::builder()
            .objective(name)
            .scale_pos_weight(2.0)
            .build()
            .unwrap();
        let objective = create_objective(&p, 1).unwrap();
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
    let data = DMatrix::from_dense(&x, n, 2)
        .unwrap()
        .with_labels(&y)
        .unwrap()
        .with_feature_types(&[FeatureType::Numerical, FeatureType::Categorical])
        .unwrap();
    let result =
        train_with_budget(&params("reg:squarederror"), &data, &BudgetConfig::new(1.0)).unwrap();
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
        &params("reg:squarederror"),
        &data,
        &BudgetConfig::new(1.5).iteration_limit(7),
    )
    .unwrap();
    assert_eq!(result.model.num_trees(), 7);
    assert_eq!(result.stop, BudgetStop::IterationLimit);
}

#[test]
fn derived_or_unused_parameters_are_rejected_by_name() {
    let data = regression(200, 13);
    let tuned = TrainingParams::builder()
        .eta(0.1)
        .max_depth(3)
        .build()
        .unwrap();
    let error = train_with_budget(&tuned, &data, &BudgetConfig::default())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("`eta`") && error.contains("`max_depth`"),
        "{error}"
    );

    // Objective parameters and the parameters budget mode reads are accepted.
    let accepted = TrainingParams::builder()
        .objective("binary:logistic")
        .scale_pos_weight(3.0)
        .max_bin(64)
        .nthread(2)
        .base_score(0.4)
        .build()
        .unwrap();
    train_with_budget(&accepted, &binary(300, 14), &BudgetConfig::default()).unwrap();
}

#[test]
fn unsupported_objectives_and_budgets_are_rejected() {
    let data = binary(200, 15);
    let multiclass = TrainingParams::builder()
        .objective("multi:softprob")
        .num_class(2)
        .build()
        .unwrap();
    let error = train_with_budget(&multiclass, &data, &BudgetConfig::default()).unwrap_err();
    assert!(error.to_string().contains("multi:softprob"), "{error}");

    for budget in [0.0, -1.0, 5.0, f64::NAN] {
        assert!(
            train_with_budget(
                &params("binary:logistic"),
                &data,
                &BudgetConfig::new(budget)
            )
            .is_err(),
            "budget {budget} was accepted"
        );
    }
}
