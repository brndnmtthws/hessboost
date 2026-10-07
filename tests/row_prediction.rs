//! Single-row prediction and the prediction transform: a row's margins and
//! predictions are bit for bit what every batch holding it gives, whatever
//! the model (tree weights, forests, vector and linear leaves, model
//! shrinkage, gblinear, categorical splits, missing values, early stopping).

mod common;

use common::bits::bits;
use common::{incompatible_model, invalid_data, lcg};
use hessboost::data::FeatureType;
use hessboost::prelude::*;
use serde_json::{Value, json};

/// Feature 0 of every row is categorical (codes `0..6`).
const CATEGORICAL: usize = 0;

/// `n` rows of `n_cols` features in `[0, 4)`: every 5th value missing,
/// feature [`CATEGORICAL`] an integer code.
fn features(n: usize, n_cols: usize, seed: u64) -> Vec<f32> {
    let mut next = lcg(seed);
    (0..n * n_cols)
        .map(|i| {
            let v = next() * 4.0;
            if i % n_cols == CATEGORICAL {
                (v * 1.5).floor()
            } else if i % 5 == 3 {
                f32::NAN
            } else {
                v
            }
        })
        .collect()
}

/// A dense matrix of `x` (NaN missing) with feature [`CATEGORICAL`]
/// categorical.
fn matrix(x: &[f32], n_cols: usize) -> DMatrix {
    let mut types = vec![FeatureType::Numerical; n_cols];
    types[CATEGORICAL] = FeatureType::Categorical;
    DMatrix::from_dense(x, x.len() / n_cols, n_cols)
        .unwrap()
        .with_feature_types(&types)
        .unwrap()
}

/// One model kind: its parameters and labels from the training features.
struct Case {
    name: &'static str,
    params: Vec<(&'static str, Value)>,
    n_cols: usize,
    rounds: usize,
    /// Label columns per row.
    targets: usize,
    label: fn(&[f32]) -> f32,
    /// Train with an eval set and early stopping, so `Iterations::Best` is
    /// a strict prefix.
    early_stopping: bool,
}

fn signal(row: &[f32]) -> f32 {
    let x = |i: usize| if row[i].is_nan() { 1.5 } else { row[i] };
    x(1) * 0.8 - x(2) + (x(0) * 0.7).sin() + 0.3 * x(3)
}

fn binary(row: &[f32]) -> f32 {
    f32::from(u8::from(signal(row) > 0.4))
}

fn positive(row: &[f32]) -> f32 {
    (signal(row) * 0.5).exp()
}

fn count(row: &[f32]) -> f32 {
    positive(row).floor()
}

fn three_classes(row: &[f32]) -> f32 {
    (signal(row) * 0.8 + 1.5).floor().clamp(0.0, 2.0)
}

fn seventy_classes(row: &[f32]) -> f32 {
    ((signal(row) + 3.0).clamp(0.0, 6.99) * 10.0).floor()
}

fn cases() -> Vec<Case> {
    let case = |name, params: Vec<(&'static str, Value)>, label| Case {
        name,
        params,
        n_cols: 6,
        rounds: 12,
        targets: 1,
        label,
        early_stopping: false,
    };
    vec![
        Case {
            early_stopping: true,
            rounds: 40,
            ..case(
                "logistic, early stopping",
                vec![("objective", json!("binary:logistic")), ("eta", json!(0.9))],
                binary,
            )
        },
        case(
            "tweedie, dart",
            vec![
                ("objective", json!("reg:tweedie")),
                ("booster", json!("dart")),
                ("rate_drop", json!(0.3)),
            ],
            positive,
        ),
        case(
            "poisson, forest",
            vec![
                ("objective", json!("count:poisson")),
                ("num_parallel_tree", json!(3)),
                ("subsample", json!(0.7)),
            ],
            count,
        ),
        case("gamma", vec![("objective", json!("reg:gamma"))], positive),
        case(
            "softprob, vector leaves",
            vec![
                ("objective", json!("multi:softprob")),
                ("num_class", json!(3)),
                ("multi_strategy", json!("multi_output_tree")),
            ],
            three_classes,
        ),
        Case {
            rounds: 3,
            ..case(
                "softmax, 70 classes",
                vec![
                    ("objective", json!("multi:softmax")),
                    ("num_class", json!(70)),
                    ("max_depth", json!(2)),
                ],
                seventy_classes,
            )
        },
        case(
            "softmax, forest",
            vec![
                ("objective", json!("multi:softmax")),
                ("num_class", json!(3)),
                ("num_parallel_tree", json!(2)),
            ],
            three_classes,
        ),
        case(
            "quantiles",
            vec![
                ("objective", json!("reg:quantileerror")),
                ("quantile_alpha", json!([0.1, 0.5, 0.9])),
            ],
            signal,
        ),
        case(
            "expectiles",
            vec![
                ("objective", json!("reg:expectileerror")),
                ("expectile_alpha", json!([0.2, 0.8])),
            ],
            signal,
        ),
        case(
            "normal distribution",
            vec![("objective", json!("dist:normal"))],
            signal,
        ),
        case("hinge", vec![("objective", json!("binary:hinge"))], binary),
        case("linear leaves", vec![("linear_tree", json!(true))], signal),
        case(
            "model shrinkage",
            vec![("model_shrink_rate", json!(0.05))],
            signal,
        ),
        case("gblinear", vec![("booster", json!("gblinear"))], signal),
        Case {
            targets: 2,
            ..case("label matrix", vec![], signal)
        },
        Case {
            n_cols: 300,
            rounds: 40,
            ..case("wide rows", vec![("max_depth", json!(3))], signal)
        },
    ]
}

/// The labels of `case` for rows `x`: `targets` columns, the second a
/// shifted copy of the first.
fn labels(case: &Case, x: &[f32]) -> Vec<f32> {
    x.chunks(case.n_cols)
        .flat_map(|row| {
            let y = (case.label)(row);
            (0..case.targets).map(move |t| y + t as f32)
        })
        .collect()
}

fn train_case(case: &Case) -> (BoostedModel, Vec<f32>) {
    let x = features(400, case.n_cols, 7);
    let mut dtrain = matrix(&x, case.n_cols);
    let y = labels(case, &x);
    dtrain = if case.targets > 1 {
        dtrain.with_label_matrix(&y, case.targets).unwrap()
    } else {
        dtrain.with_labels(&y).unwrap()
    };
    let params = TrainingParams::from_xgboost(case.params.clone()).unwrap();
    let test = features(97, case.n_cols, 11);
    let model = if case.early_stopping {
        let dvalid = matrix(&test, case.n_cols)
            .with_labels(&labels(case, &test))
            .unwrap();
        Trainer::new(&params, &dtrain, case.rounds)
            .eval(&dvalid, "valid")
            .early_stopping_rounds(std::num::NonZeroUsize::new(3).unwrap())
            .train()
            .unwrap()
            .model
    } else {
        train(&params, &dtrain, case.rounds).unwrap()
    };
    (model, test)
}

/// The iteration selections each model is checked with: what its
/// predictions accept (a shrunk model only ranges from 0, gblinear only
/// the whole model).
fn selections(model: &BoostedModel) -> Vec<Iterations> {
    if model.linear().is_some() {
        return vec![Iterations::Best, (..).into()];
    }
    let mut out = vec![Iterations::Best, (..).into(), (..2).into()];
    if model.shrinkage().is_none() {
        out.push((1..3).into());
    }
    out
}

#[test]
fn rows_predict_what_every_batch_predicts() {
    for case in cases() {
        let (model, x) = train_case(&case);
        let name = case.name;
        let n_cols = case.n_cols;
        let (k, width) = (model.n_outputs(), model.prediction_width());
        let data = matrix(&x, n_cols);
        for iterations in selections(&model) {
            let batch = model.predict(&data, iterations).unwrap();
            let margins = model.predict_margin(&data, iterations).unwrap();
            assert_eq!(batch.width(), width, "{name}");

            let mut transformed = vec![0.0; batch.as_slice().len()];
            model
                .transform_margins_into(margins.as_slice(), &mut transformed)
                .unwrap();
            assert_eq!(
                bits(&transformed),
                bits(batch.as_slice()),
                "{name}: transform"
            );

            let (mut row_margins, mut row_values) = (vec![0.0; k], vec![0.0; width]);
            for (r, row) in x.chunks(n_cols).enumerate() {
                let one = model.predict(&matrix(row, n_cols), iterations).unwrap();
                let want = &batch.as_slice()[r * width..(r + 1) * width];
                assert_eq!(bits(one.as_slice()), bits(want), "{name}: row {r} alone");

                model
                    .predict_row_into(row, iterations, &mut row_values)
                    .unwrap();
                assert_eq!(bits(&row_values), bits(want), "{name}: row {r}");
                model
                    .predict_margin_row_into(row, iterations, &mut row_margins)
                    .unwrap();
                let margin = &margins.as_slice()[r * k..(r + 1) * k];
                assert_eq!(
                    bits(&row_margins),
                    bits(margin),
                    "{name}: margins of row {r}"
                );

                if width == 1 {
                    let value = model.predict_row(row, iterations).unwrap();
                    assert_eq!(value.to_bits(), want[0].to_bits(), "{name}: row {r}");
                }
                if k == 1 {
                    let m = model.predict_margin_row(row, iterations).unwrap();
                    assert_eq!(
                        m.to_bits(),
                        margin[0].to_bits(),
                        "{name}: margin of row {r}"
                    );
                    let value = model.transform_margin(m).unwrap();
                    assert_eq!(value.to_bits(), want[0].to_bits(), "{name}: row {r}");
                }
            }
        }
    }
}

/// `multi:softmax` predicts XGBoost's `FindMaxIndex`: the first of equal
/// largest margins. Without its trees, every row's margins are the (equal)
/// intercepts of balanced classes.
#[test]
fn softmax_ties_go_to_the_first_class() {
    let x = features(90, 6, 3);
    let y: Vec<f32> = (0..90).map(|i| (i % 3) as f32).collect();
    let dtrain = matrix(&x, 6).with_labels(&y).unwrap();
    let params = TrainingParams::from_xgboost([
        ("objective", json!("multi:softmax")),
        ("num_class", json!(3)),
    ])
    .unwrap();
    let model = train(&params, &dtrain, 2).unwrap();
    let margins = model.predict_margin(&dtrain, ..0).unwrap();
    assert!(margins.rows().all(|m| m[0] == m[1] && m[1] == m[2]));
    let classes = model.predict(&dtrain, ..0).unwrap();
    assert!(classes.as_slice().iter().all(|&c| c == 0.0));
    assert_eq!(model.predict_row(&x[..6], ..0).unwrap(), 0.0);
}

#[test]
fn malformed_rows_and_outputs_are_refused() {
    let case = &cases()[4];
    let (model, x) = train_case(case);
    let row = &x[..6];
    let mut out = [0.0; 3];
    let wrong = |result: Result<()>| match result {
        Err(HessboostError::DimensionMismatch { .. }) => {}
        other => panic!("expected a dimension mismatch, got {other:?}"),
    };
    wrong(model.predict_row_into(&row[..5], Iterations::Best, &mut out));
    wrong(model.predict_row_into(row, Iterations::Best, &mut out[..2]));
    wrong(model.predict_margin_row_into(row, Iterations::Best, &mut out[..1]));
    wrong(model.transform_margins_into(&[0.0; 4], &mut [0.0; 3]));
    wrong(model.transform_margins_into(&[0.0; 6], &mut [0.0; 3]));
    let mut infinite = row.to_vec();
    infinite[2] = f32::NEG_INFINITY;
    assert_eq!(
        invalid_data(model.predict_row_into(&infinite, Iterations::Best, &mut out)),
        ("row", None)
    );
    // Three probabilities per row: the scalar methods name the `_into` ones.
    assert_eq!(
        incompatible_model(model.predict_row(row, Iterations::Best)),
        "outputs"
    );
    assert_eq!(
        incompatible_model(model.predict_margin_row(row, Iterations::Best)),
        "outputs"
    );
    assert_eq!(incompatible_model(model.transform_margin(0.0)), "outputs");
    assert_eq!(
        incompatible_model(model.predict_row_into(row, ..99, &mut out)),
        "iterations"
    );

    let shrunk = cases()
        .into_iter()
        .find(|case| case.name == "model shrinkage")
        .unwrap();
    let (model, x) = train_case(&shrunk);
    assert_eq!(
        incompatible_model(model.predict_row(&x[..6], 1..3)),
        "iterations"
    );
}
