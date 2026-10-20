//! Row views: `DMatrix::select_rows` reads its source's values in place, and
//! training on it (and evaluating on it) is training on a copy of the rows,
//! bit for bit.

use hessboost::data::ViewCuts;
use hessboost::model::ModelFormat;
use hessboost::prelude::{DMatrix, Trainer, TrainingParams};
use serde_json::json;

mod common;
use common::{four_features, lcg};

const ROWS: usize = 3_000;

/// Four features (one with missing values) and a noisy label.
fn source(missing: f32) -> DMatrix {
    let mut noise = lcg(5);
    let x: Vec<f32> = (0..ROWS)
        .flat_map(four_features)
        .map(|v| if v.is_nan() { missing } else { v })
        .collect();
    let y: Vec<f32> = (0..ROWS)
        .map(|i| {
            let [a, b, c, _] = four_features(i);
            2.0 * a - b + c * c + noise()
        })
        .collect();
    let w: Vec<f32> = (0..ROWS).map(|i| 0.5 + (i % 4) as f32).collect();
    DMatrix::from_dense_with_missing(&x, ROWS, 4, missing)
        .unwrap()
        .with_labels(&y)
        .unwrap()
        .with_weights(&w)
        .unwrap()
}

/// The CSR form of `data`, its labels and weights included.
fn as_csr(data: &DMatrix) -> DMatrix {
    let (mut indptr, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
    for r in 0..data.n_rows() {
        for f in 0..data.n_cols() {
            if let Some(v) = data.get(r, f) {
                indices.push(f as u32);
                values.push(v);
            }
        }
        indptr.push(values.len());
    }
    DMatrix::from_csr(indptr, indices, values, data.n_cols())
        .unwrap()
        .with_labels(data.labels().unwrap())
        .unwrap()
        .with_weights(data.weights().unwrap())
        .unwrap()
}

/// A fresh matrix of `rows` of `data`: values read one by one, labels and
/// weights gathered.
fn copy_rows(data: &DMatrix, rows: &[usize]) -> DMatrix {
    let n_cols = data.n_cols();
    let x: Vec<f32> = rows
        .iter()
        .flat_map(|&r| (0..n_cols).map(move |f| data.get(r, f).unwrap_or(f32::NAN)))
        .collect();
    let labels = data.labels().unwrap();
    let weights = data.weights().unwrap();
    DMatrix::from_dense_vec(x, rows.len(), n_cols)
        .unwrap()
        .with_labels(&rows.iter().map(|&r| labels[r]).collect::<Vec<_>>())
        .unwrap()
        .with_weights(&rows.iter().map(|&r| weights[r]).collect::<Vec<_>>())
        .unwrap()
}

/// The rows a view selects: every third row from the end, then a run of
/// repeats.
fn selection() -> Vec<usize> {
    (0..ROWS)
        .rev()
        .step_by(3)
        .chain((100..400).flat_map(|r| [r, r]))
        .collect()
}

fn params(tree_method: &str) -> TrainingParams {
    TrainingParams::from_xgboost([
        ("tree_method", json!(tree_method)),
        ("max_depth", json!(4)),
        ("eta", json!(0.3)),
        ("subsample", json!(0.7)),
        ("colsample_bytree", json!(0.75)),
        ("objective", json!("reg:quantileerror")),
        ("quantile_alpha", json!(0.3)),
        ("seed", json!(11)),
    ])
    .unwrap()
}

/// The model and the eval history `train` gives on `dtrain` scored on
/// `deval`.
fn fit(params: &TrainingParams, dtrain: &DMatrix, deval: &DMatrix) -> (Vec<u8>, Vec<u64>) {
    let result = Trainer::new(params, dtrain, 12)
        .eval(deval, "eval")
        .train()
        .unwrap();
    let scores = result
        .history
        .rounds()
        .flat_map(|round| {
            round
                .scores()
                .map(|(_, _, score)| score.to_bits())
                .collect::<Vec<_>>()
        })
        .collect();
    (result.model.encode(ModelFormat::Binary).unwrap(), scores)
}

/// Training on a view of dense (NaN or sentinel missing) or CSR rows, and
/// scoring a view, equals training on and scoring a copy of the rows, for
/// every tree method.
#[test]
fn training_on_a_view_is_training_on_a_copy() {
    let rows = selection();
    let eval_rows: Vec<usize> = (0..ROWS).step_by(5).collect();
    for data in [source(f32::NAN), source(-1.0), as_csr(&source(f32::NAN))] {
        let (view, eval_view) = (
            data.select_rows(&rows).unwrap(),
            data.select_rows(&eval_rows).unwrap(),
        );
        let (copy, eval_copy) = (copy_rows(&data, &rows), copy_rows(&data, &eval_rows));
        for method in ["hist", "approx", "exact"] {
            let params = params(method);
            assert_eq!(
                fit(&params, &view, &eval_view),
                fit(&params, &copy, &eval_copy),
                "{method}"
            );
        }
        // A view of a view composes.
        let nested: Vec<usize> = (0..rows.len()).step_by(2).collect();
        let twice = view.select_rows(&nested).unwrap();
        let direct: Vec<usize> = nested.iter().map(|&i| rows[i]).collect();
        let params = params("hist");
        assert_eq!(
            fit(&params, &twice, &eval_view),
            fit(&params, &copy_rows(&data, &direct), &eval_copy)
        );
    }
}

/// A [`ViewCuts::Source`] view splits only on its source's cut values,
/// and every view of a source with these cuts trains as a view of it.
#[test]
fn source_cut_views_split_on_their_sources_cuts() {
    use hessboost::internals::HistCuts;
    let data = source(f32::NAN);
    let cuts = HistCuts::from_dmatrix(&data, 256);
    let view = data
        .select_rows_with_cuts(&selection(), ViewCuts::Source)
        .unwrap();
    let model = hessboost::prelude::train(&params("hist"), &view, 12).unwrap();
    for tree in model.trees() {
        for node in tree.nodes().iter().filter(|node| !node.is_leaf()) {
            let feature = node.split_feature as usize;
            let (start, end) = cuts.feature_bins(feature);
            assert!(
                (start..end).any(|bin| cuts.cut_value(bin) == node.split_cond),
                "split on feature {feature} at {} is not a source cut",
                node.split_cond
            );
        }
    }
}

/// A [`ViewCuts::Source`] view weighing a row its source weighs zero trains
/// as continuing its training does: the source's cuts never saw that row,
/// whose value beyond their last cut bins into the last bin, so the
/// training margins route it by its value, as the continued model's
/// prediction does, sampled rows or not.
#[test]
fn source_cut_views_route_rows_their_source_never_sketched_by_value() {
    let n = 300;
    // Feature 0 is missing on every third row, the only rows labeled 10,
    // so the first split isolates them at the last bin boundary; the last
    // row lies far beyond the others.
    let x: Vec<f32> = (0..=n)
        .map(|i| match i {
            _ if i == n => 1e6,
            _ if i % 3 == 0 => f32::NAN,
            _ => i as f32,
        })
        .collect();
    let y: Vec<f32> = (0..=n)
        .map(|i| if i < n && i % 3 == 0 { 10.0 } else { 0.0 })
        .collect();
    let mut weights = vec![1.0; n + 1];
    weights[n] = 0.0;
    let data = DMatrix::from_dense(&x, n + 1, 1)
        .unwrap()
        .with_labels(&y)
        .unwrap()
        .with_weights(&weights)
        .unwrap();
    let rows: Vec<usize> = (0..=n).collect();
    let view = data
        .select_rows_with_cuts(&rows, ViewCuts::Source)
        .unwrap()
        .with_weights(&vec![1.0; n + 1])
        .unwrap();
    for subsample in [1.0, 0.5] {
        let params = TrainingParams::from_xgboost([
            ("max_depth", json!(1)),
            ("eta", json!(1.0)),
            ("subsample", json!(subsample)),
            ("seed", json!(3)),
        ])
        .unwrap();
        let whole = hessboost::prelude::train(&params, &view, 2).unwrap();
        let first = hessboost::prelude::train(&params, &view, 1).unwrap();
        let resumed = Trainer::new(&params, &view, 1)
            .init_model(&first)
            .train()
            .unwrap()
            .model;
        assert_eq!(
            resumed.encode(ModelFormat::Binary).unwrap(),
            whole.encode(ModelFormat::Binary).unwrap(),
            "subsample {subsample}"
        );
    }
}
