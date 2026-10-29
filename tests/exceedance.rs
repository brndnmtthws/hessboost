//! `reg:exceedance`: one model predicting `P(y ≥ k_j)` over a threshold
//! ladder. It trains under both multi strategies, every predicted row is
//! non-increasing and the sigmoid-then-sort transform of the margins, every
//! native format round-trips bit for bit, the flat key parses only where
//! something reads it, the default metric is `exceedance-logloss`, and
//! XGBoost export refuses the model.

use hessboost::config::MultiStrategy;
use hessboost::model::compact::CompactModel;
use hessboost::objective::Thresholds;
use hessboost::prelude::*;
use serde_json::json;

mod common;
use common::bits::bits;
use common::{four_features, invalid_param, labeled_dense, lcg};

const COLS: usize = 4;
const ROWS: usize = 600;

const LADDER_3: [f64; 3] = [0.25, 0.75, 1.5];
const LADDER_8: [f64; 8] = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5];

/// Four features and a right-skewed label: exponential noise scaled by the
/// features, so the upper thresholds are reached by a thin tail.
fn data() -> DMatrix {
    let mut uniform = lcg(17);
    let mut x = Vec::with_capacity(ROWS * COLS);
    let mut y = Vec::with_capacity(ROWS);
    for i in 0..ROWS {
        let row = four_features(i);
        x.extend_from_slice(&row);
        let noise = -(1.0 - uniform()).ln();
        y.push((0.2 + row[0] + 0.5 * row[1]) * noise);
    }
    labeled_dense(&x, COLS, &y)
}

fn params(ladder: &[f64], strategy: MultiStrategy) -> TrainingParams {
    TrainingParams::builder()
        .objective(Objective::Exceedance(
            Thresholds::new(ladder.iter().copied()).unwrap(),
        ))
        .multi_strategy(strategy)
        .tree_method(TreeMethod::Hist)
        .max_depth(3)
        .eta(0.3)
        .build()
        .unwrap()
}

/// `binary:logistic`'s sigmoid (XGBoost's `common::Sigmoid` in `f32`).
fn sigmoid(x: f32) -> f32 {
    1.0 / ((-x).min(88.7).exp() + 1.0)
}

/// The sigmoid of every margin, then each row sorted descending.
fn transformed(margins: &[f32], k: usize) -> Vec<f32> {
    let mut out: Vec<f32> = margins.iter().map(|&m| sigmoid(m)).collect();
    for row in out.chunks_exact_mut(k) {
        row.sort_by(|a, b| b.total_cmp(a));
    }
    out
}

#[test]
fn trains_under_both_strategies_with_non_increasing_rows() {
    let d = data();
    for ladder in [&LADDER_3[..], &LADDER_8[..]] {
        let k = ladder.len();
        for strategy in [
            MultiStrategy::OneOutputPerTree,
            MultiStrategy::MultiOutputTree,
        ] {
            let what = format!("K = {k}, {strategy:?}");
            let model = train(&params(ladder, strategy), &d, 12).unwrap();
            assert_eq!(model.n_outputs(), k, "{what}");
            assert_eq!(model.n_targets(), 1, "{what}");
            assert_eq!(
                model.has_vector_leaves(),
                strategy == MultiStrategy::MultiOutputTree,
                "{what}"
            );
            let margins = model.predict_margin(&d, Iterations::Best).unwrap();
            let preds = model.predict(&d, Iterations::Best).unwrap();
            assert_eq!(preds.width(), k, "{what}");
            assert_eq!(
                bits(preds.as_slice()),
                bits(transformed(margins.as_slice(), k)),
                "{what}: predict is the transform of predict_margin"
            );
            for row in preds.as_slice().chunks_exact(k) {
                assert!(
                    row.windows(2).all(|pair| pair[0] >= pair[1]),
                    "{what}: {row:?}"
                );
                assert!(row.iter().all(|p| (0.0..=1.0).contains(p)), "{what}");
            }
            // The fit tracks the empirical exceedance rates.
            let labels = d.labels().unwrap();
            for (j, &level) in ladder.iter().enumerate() {
                let rate =
                    labels.iter().filter(|&&y| f64::from(y) >= level).count() as f64 / ROWS as f64;
                let mean = preds
                    .as_slice()
                    .iter()
                    .skip(j)
                    .step_by(k)
                    .map(|&p| f64::from(p))
                    .sum::<f64>()
                    / ROWS as f64;
                assert!(
                    (mean - rate).abs() < 0.05,
                    "{what}: {level}: {mean} vs {rate}"
                );
            }
        }
    }
}

#[test]
fn native_formats_round_trip_bit_for_bit() {
    let d = data();
    for ladder in [&LADDER_3[..], &LADDER_8[..]] {
        for strategy in [
            MultiStrategy::OneOutputPerTree,
            MultiStrategy::MultiOutputTree,
        ] {
            let what = format!("K = {}, {strategy:?}", ladder.len());
            let model = train(&params(ladder, strategy), &d, 6).unwrap();
            let expected = bits(model.predict(&d, Iterations::Best).unwrap().as_slice());
            let margins = bits(
                model
                    .predict_margin(&d, Iterations::Best)
                    .unwrap()
                    .as_slice(),
            );
            for format in [ModelFormat::Binary, ModelFormat::Json] {
                let bytes = model.encode(format).unwrap();
                let restored = BoostedModel::decode(&bytes, format).unwrap();
                assert_eq!(restored.objective(), model.objective(), "{what} {format:?}");
                assert_eq!(
                    bits(restored.predict(&d, Iterations::Best).unwrap().as_slice()),
                    expected,
                    "{what} {format:?}"
                );
                assert_eq!(restored.encode(format).unwrap(), bytes, "{what} {format:?}");
            }
            if strategy == MultiStrategy::OneOutputPerTree {
                let compact = CompactModel::decode(model.to_compact_bytes().unwrap()).unwrap();
                assert_eq!(compact.objective(), model.objective(), "{what}");
                assert_eq!(
                    bits(compact.predict(&d).unwrap().as_slice()),
                    expected,
                    "{what} compact"
                );
                assert_eq!(
                    bits(compact.predict_margin(&d).unwrap().as_slice()),
                    margins,
                    "{what} compact"
                );
            }
        }
    }
}

/// A stored `reg:exceedance` without thresholds, or with invalid ones, is a
/// format error, in native JSON and binary alike.
#[test]
fn stored_thresholds_are_required_and_validated() {
    let d = data();
    let model = train(&params(&LADDER_3, MultiStrategy::OneOutputPerTree), &d, 2).unwrap();
    let doc: serde_json::Value =
        serde_json::from_slice(&model.encode(ModelFormat::Json).unwrap()).unwrap();
    assert_eq!(
        doc["objective_params"]["exceedance_threshold"],
        json!(LADDER_3)
    );
    for edit in [
        None,
        Some(json!([])),
        Some(json!([1.0, 0.5])),
        Some(json!([1.0, 1.0])),
    ] {
        let mut edited = doc.clone();
        let params = edited["objective_params"].as_object_mut().unwrap();
        match &edit {
            None => {
                params.remove("exceedance_threshold");
            }
            Some(value) => {
                params.insert("exceedance_threshold".into(), value.clone());
            }
        }
        assert!(
            matches!(
                BoostedModel::decode(edited.to_string(), ModelFormat::Json),
                Err(HessboostError::ModelFormat(_))
            ),
            "{edit:?}"
        );
    }
    // Models of other objectives store no threshold member at all.
    let plain = train(&TrainingParams::default(), &d, 1).unwrap();
    let plain_doc: serde_json::Value =
        serde_json::from_slice(&plain.encode(ModelFormat::Json).unwrap()).unwrap();
    assert!(
        plain_doc["objective_params"]
            .get("exceedance_threshold")
            .is_none()
    );
}

#[test]
fn flat_key_parses_only_where_it_is_read() {
    let p = TrainingParams::from_xgboost([
        ("objective", json!("reg:exceedance")),
        ("exceedance_threshold", json!([0.5, 1.0, 2.0])),
    ])
    .unwrap();
    assert_eq!(
        p.objective,
        Objective::Exceedance(Thresholds::new([0.5, 1.0, 2.0]).unwrap())
    );
    // A single number is a one-threshold ladder, as for `quantile_alpha`.
    let single = TrainingParams::from_xgboost([
        ("objective", json!("reg:exceedance")),
        ("exceedance_threshold", json!(1.5)),
    ])
    .unwrap();
    assert_eq!(
        single.objective,
        Objective::Exceedance(Thresholds::new([1.5]).unwrap())
    );
    // Back through the flat form.
    assert_eq!(
        TrainingParams::from_xgboost(p.to_xgboost().unwrap()).unwrap(),
        p
    );
    // Missing or invalid thresholds name the key.
    for value in [None, Some(json!([2.0, 1.0])), Some(json!([1.0, 1.0]))] {
        let mut flat = vec![("objective", json!("reg:exceedance"))];
        if let Some(value) = value {
            flat.push(("exceedance_threshold", value));
        }
        assert_eq!(
            invalid_param(TrainingParams::from_xgboost(flat)),
            "exceedance_threshold"
        );
    }
    // Nothing reads the key without the objective or the metric...
    assert_eq!(
        invalid_param(TrainingParams::from_xgboost([
            ("objective", json!("binary:logistic")),
            ("exceedance_threshold", json!([1.0])),
        ])),
        "exceedance_threshold"
    );
    // ...and the metric alone reads it.
    let metric = TrainingParams::from_xgboost([
        ("objective", json!("reg:squarederror")),
        ("eval_metric", json!("exceedance-logloss")),
        ("exceedance_threshold", json!([1.0])),
    ])
    .unwrap();
    assert_eq!(
        metric.eval_metric,
        [EvalMetric::ExceedanceLogLoss(
            Thresholds::new([1.0]).unwrap()
        )]
    );
}

#[test]
fn default_metric_is_exceedance_logloss() {
    let d = data();
    let result = Trainer::new(&params(&LADDER_8, MultiStrategy::OneOutputPerTree), &d, 5)
        .eval(&d, "train")
        .train()
        .unwrap();
    assert_eq!(result.history.metrics(), ["exceedance-logloss"]);
    let series: Vec<f64> = result
        .history
        .series("train", "exceedance-logloss")
        .unwrap()
        .collect();
    assert_eq!(series.len(), 5);
    assert!(
        series.iter().all(|v| v.is_finite() && *v > 0.0),
        "{series:?}"
    );
    assert!(series[4] < series[0], "{series:?}");
}

#[test]
fn xgboost_export_refuses_and_import_rejects_the_name() {
    let d = data();
    let model = train(&params(&LADDER_3, MultiStrategy::OneOutputPerTree), &d, 2).unwrap();
    for format in [ModelFormat::XgboostJson, ModelFormat::XgboostUbjson] {
        assert!(
            matches!(model.encode(format), Err(HessboostError::ModelFormat(_))),
            "{format:?}"
        );
    }
    let point = train(
        &TrainingParams::builder()
            .objective(Objective::BinaryLogistic(RegLoss::default()))
            .build()
            .unwrap(),
        &labeled_dense(&[0.0, 1.0, 2.0, 3.0], 1, &[0.0, 0.0, 1.0, 1.0]),
        1,
    )
    .unwrap();
    let doc = String::from_utf8(point.encode(ModelFormat::XgboostJson).unwrap())
        .unwrap()
        .replace("\"binary:logistic\"", "\"reg:exceedance\"");
    assert!(matches!(
        BoostedModel::decode(&doc, ModelFormat::XgboostJson),
        Err(HessboostError::ModelFormat(_))
    ));
}
