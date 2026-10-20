//! Row views: `DMatrix::select_rows` reads its source's values in place, and
//! training on it (and evaluating on it) is training on a copy of the rows,
//! bit for bit.

use hessboost::config::Langevin;
use hessboost::data::{FeatureType, ViewCuts};
use hessboost::model::ModelFormat;
use hessboost::prelude::{DMatrix, Trainer, TrainingParams};
use hessboost::training::budget::{BudgetConfig, train_with_budget};
use serde_json::json;

mod common;
use common::{four_features, invalid_data, invalid_param, lcg};

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

/// A [`ViewCuts::Source`] view splits only on its source's cut values, not
/// on the ones its own rows (with their weights) would give: with fewer
/// bins than distinct values the two differ, and some split shows it.
#[test]
fn source_cut_views_split_on_their_sources_cuts() {
    use hessboost::internals::HistCuts;
    let data = source(f32::NAN);
    let mut params = params("hist");
    params.max_bin = 16;
    let source_cuts = HistCuts::from_dmatrix(&data, params.max_bin);
    let own_cuts = HistCuts::from_dmatrix(&copy_rows(&data, &selection()), params.max_bin);
    let is_cut = |cuts: &HistCuts, feature: usize, threshold: f32| {
        let (start, end) = cuts.feature_bins(feature);
        (start..end).any(|bin| cuts.cut_value(bin) == threshold)
    };
    let view = data
        .select_rows_with_cuts(&selection(), ViewCuts::Source)
        .unwrap();
    let model = hessboost::prelude::train(&params, &view, 12).unwrap();
    let mut off_own_cuts = 0;
    for tree in model.trees() {
        for node in tree.nodes().iter().filter(|node| !node.is_leaf()) {
            let feature = node.split_feature as usize;
            // `f32::MIN` sends every present value right, on any cuts.
            assert!(
                node.split_cond == f32::MIN || is_cut(&source_cuts, feature, node.split_cond),
                "split on feature {feature} at {} is not a source cut",
                node.split_cond
            );
            off_own_cuts += usize::from(
                node.split_cond != f32::MIN && !is_cut(&own_cuts, feature, node.split_cond),
            );
        }
    }
    assert!(
        off_own_cuts > 0,
        "no split tells the source's cuts from the view's own"
    );
}

/// Rows of a source-cut view whose source never sketched one of them: feature
/// 0 is missing on every third row, the only rows labeled 10, so a first
/// split isolates them at the last bin boundary; the last row, weighed zero
/// by the source and one by the view, lies far beyond the others, past the
/// source's last cut.
fn unsketched_view(n: usize) -> DMatrix {
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
    data.select_rows_with_cuts(&rows, ViewCuts::Source)
        .unwrap()
        .with_weights(&vec![1.0; n + 1])
        .unwrap()
}

/// A [`ViewCuts::Source`] view weighing a row its source weighs zero trains
/// as continuing its training does: its value beyond the source's last cut
/// bins into the last bin, so the training margins route it by its value,
/// as the continued model's prediction does, sampled rows or not.
#[test]
fn source_cut_views_route_rows_their_source_never_sketched_by_value() {
    let view = unsketched_view(300);
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

/// Langevin leaf re-estimation over such a view takes each leaf's rows as
/// its values route them: every leaf is the mean label of the rows
/// predicted there (zero intercept, no `lambda`, unit Hessians).
#[test]
fn langevin_leaves_of_unsketched_rows_follow_their_values() {
    let view = unsketched_view(300);
    let params = TrainingParams::builder()
        .max_depth(1)
        .eta(1.0)
        .lambda(0.0)
        .min_child_weight(0.0)
        .base_score(0.0)
        .langevin(Langevin::default())
        .build()
        .unwrap();
    let model = hessboost::prelude::train(&params, &view, 1).unwrap();
    let leaves = model.predict_leaf(&view, ..).unwrap();
    let tree = &model.trees()[0];
    let labels = view.labels().unwrap();
    for leaf in (0..tree.num_nodes()).filter(|&node| tree.node(node).is_leaf()) {
        let routed: Vec<f32> = leaves
            .as_slice()
            .iter()
            .zip(labels)
            .filter(|&(&l, _)| l as usize == leaf)
            .map(|(_, &y)| y)
            .collect();
        let mean = routed.iter().sum::<f32>() / routed.len() as f32;
        let value = tree.node(leaf).leaf_value;
        // Partitions instead of values would give the right leaf 10, the
        // missing rows' mean without the far row.
        assert!(
            (value - mean).abs() < 1e-3,
            "leaf {leaf}: {value} vs {mean}"
        );
    }
}

/// Budget mode follows the builder's partitions, so it refuses such a view.
#[test]
fn budget_mode_refuses_rows_the_source_never_sketched() {
    let refused = train_with_budget(
        &TrainingParams::default(),
        &unsketched_view(300),
        &BudgetConfig::new(0.5),
    );
    assert_eq!(invalid_data(refused), ("weights", None));
}

/// A source-cut view keeps the feature types its source's cuts were
/// computed for.
#[test]
fn source_cut_views_keep_their_sources_feature_types() {
    let view = unsketched_view(30);
    let refused = view.clone().with_feature_types(&[FeatureType::Categorical]);
    assert_eq!(invalid_param(refused), "feature_types");
    assert!(view.with_feature_types(&[FeatureType::Numerical]).is_ok());
}
