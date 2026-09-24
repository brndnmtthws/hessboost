//! Native model formats: the binary format (`SQB\0`, a version byte, then a
//! postcard payload) and native JSON round-trip every model feature bit for
//! bit, while other format versions, corrupt payloads, and inconsistent
//! layouts are refused.

use hessboost::prelude::*;
use serde_json::Value;

mod common;
use common::{four_features, labeled_dense};

const COLS: usize = 4;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn unknown_and_corrupt_native_payloads_are_refused() {
    let model = train(&base().build().unwrap(), &matrix(1), 3).unwrap();
    let bytes = model.to_bytes().unwrap();
    let with_version = |version: u8| {
        let mut bytes = bytes.clone();
        bytes[4] = version;
        bytes
    };
    // Every version but the current one is refused, and so are truncated
    // payloads and headers.
    for corrupt in [
        with_version(0),
        with_version(1),
        with_version(3),
        with_version(255),
        bytes[..bytes.len() - 3].to_vec(),
        bytes[..4].to_vec(),
    ] {
        let err = BoostedModel::from_bytes(&corrupt).unwrap_err();
        assert!(matches!(err, HessboostError::ModelFormat(_)), "{err}");
    }
}

/// The native JSON document of a tree-less single-output model; tests edit
/// its layout fields into inconsistent states.
fn empty_model_doc() -> Value {
    let model = train(&base().build().unwrap(), &matrix(1), 0).unwrap();
    serde_json::from_str(&model.to_json().unwrap()).unwrap()
}

fn load_doc(doc: &Value) -> hessboost::error::Result<BoostedModel> {
    BoostedModel::from_json(&doc.to_string())
}

/// Every stored field is required: a document missing one is refused
/// rather than filled in with a guess.
#[test]
fn incomplete_json_documents_are_refused() {
    for field in [
        "objective_params",
        "n_targets",
        "tree_weights",
        "num_parallel_tree",
    ] {
        let mut doc = empty_model_doc();
        doc.as_object_mut().unwrap().remove(field);
        assert!(load_doc(&doc).is_err(), "{field}");
    }
}

#[test]
fn overflowing_tree_layout_is_refused() {
    // Two outputs × 2^63 parallel trees overflows `usize`: the layout must be
    // refused, not panic or wrap to zero trees per iteration (after which
    // round counts would divide by zero).
    let mut doc = empty_model_doc();
    doc["n_outputs"] = 2.into();
    doc["n_targets"] = 2.into();
    doc["base_score"] = serde_json::json!([0.0, 0.0]);
    assert!(load_doc(&doc).is_ok());
    doc["num_parallel_tree"] = (1u64 << 63).into();
    for best_iteration in [Value::Null, 0.into()] {
        doc["best_iteration"] = best_iteration;
        assert!(matches!(
            load_doc(&doc),
            Err(HessboostError::ModelFormat(_))
        ));
    }
}

#[test]
fn objective_width_must_match_the_stored_outputs() {
    // A two-alpha objective needs two outputs: on a one-output layout it
    // would transform consecutive prediction rows as if they were one row.
    for (objective, key) in [
        ("reg:expectileerror", "expectile_alpha"),
        ("reg:quantileerror", "quantile_alpha"),
    ] {
        let mut doc = empty_model_doc();
        doc["objective"] = objective.into();
        doc["objective_params"][key] = serde_json::json!([0.2, 0.8]);
        let err = load_doc(&doc).unwrap_err();
        assert!(matches!(err, HessboostError::ModelFormat(_)), "{err}");
        // The matching layout loads.
        doc["n_outputs"] = 2.into();
        doc["base_score"] = serde_json::json!([0.0, 0.0]);
        let model = load_doc(&doc).unwrap();
        assert_eq!(model.n_outputs(), 2);
    }
}

/// `(x, y)` with `k` label columns: 160 [`four_features`] rows.
fn train_data(k: usize) -> (Vec<f32>, Vec<f32>) {
    const N: usize = 160;
    let mut x = Vec::with_capacity(N * COLS);
    let mut y = Vec::with_capacity(N * k);
    for i in 0..N {
        let row = four_features(i);
        let [a, b, c, _] = row;
        x.extend(row);
        for t in 0..k {
            y.push(2.0 * a - b + t as f32 * c + 0.1);
        }
    }
    (x, y)
}

fn matrix(k: usize) -> DMatrix {
    let (x, y) = train_data(k);
    let n = x.len() / COLS;
    let data = DMatrix::from_dense(&x, n, COLS).unwrap();
    if k == 1 {
        data.with_labels(&y).unwrap()
    } else {
        data.with_label_matrix(&y, k).unwrap()
    }
}

fn base() -> hessboost::config::TrainingParamsBuilder {
    TrainingParams::builder()
        .tree_method(TreeMethod::Hist)
        .max_depth(3)
        .nthread(1)
        .seed(11)
}

/// Whether a trained model exercises the feature its case names.
type Has = fn(&BoostedModel) -> bool;

/// Whether a model stores DART tree weights other than `1.0`.
fn has_dart_weights(model: &BoostedModel) -> bool {
    let doc: Value = serde_json::from_str(&model.to_json().unwrap()).unwrap();
    doc["tree_weights"]
        .as_array()
        .unwrap()
        .iter()
        .any(|w| w.as_f64() != Some(1.0))
}

#[test]
fn native_formats_round_trip_every_model_feature() {
    let (x, _) = train_data(1);
    let n = x.len() / COLS;
    let classes: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
    let lower: Vec<f32> = (0..n).map(|i| 1.0 + (i % 5) as f32).collect();
    let upper: Vec<f32> = lower
        .iter()
        .enumerate()
        .map(|(i, &l)| if i % 4 == 0 { f32::INFINITY } else { l })
        .collect();
    let aft = DMatrix::from_dense(&x, n, COLS)
        .unwrap()
        .with_label_bounds(&lower, &upper)
        .unwrap();
    let softmax = labeled_dense(&x, COLS, &classes);
    let counts: Vec<f32> = (0..n).map(|i| ((i * 7) % 9) as f32).collect();
    let negbinomial = labeled_dense(&x, COLS, &counts);
    // Column 0 holds category codes `0..5` whose effect is not monotone.
    let mut cat_x = x.clone();
    let mut cat_y = Vec::with_capacity(n);
    for i in 0..n {
        cat_x[i * COLS] = (i % 5) as f32;
        cat_y.push([0.0, 2.0, -1.0, 3.0, 1.0][i % 5] + cat_x[i * COLS + 1]);
    }
    let mut types = vec![FeatureType::Numerical; COLS];
    types[0] = FeatureType::Categorical;
    let categorical = labeled_dense(&cat_x, COLS, &cat_y)
        .with_feature_types(&types)
        .unwrap();
    let cases: Vec<(&str, TrainingParams, DMatrix, Has)> = vec![
        (
            "dart",
            base()
                .booster(BoosterKind::Dart)
                .rate_drop(0.5)
                .build()
                .unwrap(),
            matrix(1),
            has_dart_weights,
        ),
        (
            "gblinear",
            base().booster(BoosterKind::GbLinear).build().unwrap(),
            matrix(1),
            |m| m.num_trees() == 0,
        ),
        (
            "categorical splits",
            base().build().unwrap(),
            categorical,
            |m| {
                m.trees()
                    .iter()
                    .any(|t| t.nodes().iter().any(|node| node.is_categorical))
            },
        ),
        (
            "vector leaves",
            base()
                .multi_strategy(MultiStrategy::MultiOutputTree)
                .build()
                .unwrap(),
            matrix(3),
            |m| m.has_vector_leaves(),
        ),
        (
            "linear leaves",
            base().linear_tree(true).build().unwrap(),
            matrix(1),
            |m| m.trees().iter().any(|t| t.linear_leaves().is_some()),
        ),
        (
            "multiclass forest",
            base()
                .objective("multi:softprob")
                .num_class(3)
                .num_parallel_tree(3)
                .subsample(0.7)
                .colsample_bynode(0.6)
                .build()
                .unwrap(),
            softmax,
            |m| m.num_parallel_tree() == 3 && m.trees_per_iteration() == 9,
        ),
        ("multi-target", base().build().unwrap(), matrix(2), |m| {
            m.n_targets() == 2 && m.n_outputs() == 2
        }),
        (
            "quantiles",
            base()
                .objective("reg:quantileerror")
                .quantile_alpha(vec![0.1, 0.5, 0.9])
                .build()
                .unwrap(),
            matrix(1),
            |m| m.n_outputs() == 3,
        ),
        (
            "expectiles",
            base()
                .objective("reg:expectileerror")
                .expectile_alpha(vec![0.2, 0.8])
                .build()
                .unwrap(),
            matrix(1),
            |m| m.n_outputs() == 2,
        ),
        (
            "aft",
            base()
                .objective("survival:aft")
                .aft_loss_distribution(AftDistribution::Logistic)
                .aft_loss_distribution_scale(0.7)
                .build()
                .unwrap(),
            aft,
            |m| m.objective_params().aft_loss_distribution == AftDistribution::Logistic,
        ),
        (
            "dist:normal",
            base()
                .objective("dist:normal")
                .dist_gradient(DistGradient::Hessian)
                .build()
                .unwrap(),
            matrix(1),
            |m| {
                let p = m.objective_params();
                p.distribution == Some(DistFamily::Normal)
                    && p.dist_gradient == DistGradient::Hessian
                    && m.n_outputs() == 2
                    && !m.has_vector_leaves()
            },
        ),
        (
            "dist:normal vector leaves",
            base()
                .objective("dist:normal")
                .multi_strategy(MultiStrategy::MultiOutputTree)
                .dist_split_direction(DistSplitDirection::Cyclic)
                .build()
                .unwrap(),
            matrix(1),
            |m| {
                let p = m.objective_params();
                p.distribution == Some(DistFamily::Normal)
                    && p.dist_split_direction == DistSplitDirection::Cyclic
                    && m.has_vector_leaves()
            },
        ),
        (
            "dist:negbinomial",
            base().objective("dist:negbinomial").build().unwrap(),
            negbinomial,
            |m| m.objective_params().distribution == Some(DistFamily::NegativeBinomial),
        ),
    ];
    let mut models: Vec<(&str, BoostedModel, DMatrix, Has)> = cases
        .into_iter()
        .map(|(name, params, data, has)| (name, train(&params, &data, 4).unwrap(), data, has))
        .collect();
    // A stored early-stopping iteration selects the trees `predict` uses.
    let plain = train(&base().build().unwrap(), &matrix(1), 4).unwrap();
    let mut doc: Value = serde_json::from_str(&plain.to_json().unwrap()).unwrap();
    doc["best_iteration"] = 1.into();
    let stopped = BoostedModel::from_json(&doc.to_string()).unwrap();
    models.push(("early stopping", stopped, matrix(1), |m| {
        m.best_iteration() == Some(1)
    }));
    for (name, model, data, has_feature) in models {
        assert!(has_feature(&model), "{name}: feature not exercised");
        let bytes = model.to_bytes().unwrap();
        assert_eq!(&bytes[..5], b"SQB\0\x02", "{name}");
        let from_binary = BoostedModel::from_bytes(&bytes).unwrap();
        assert_eq!(from_binary.to_bytes().unwrap(), bytes, "{name}: binary");
        let from_json = BoostedModel::from_json(&model.to_json().unwrap()).unwrap();
        let expected = bits(&model.predict(&data).unwrap());
        for restored in [&from_binary, &from_json] {
            assert_eq!(bits(&restored.predict(&data).unwrap()), expected, "{name}");
            assert_eq!(restored.n_outputs(), model.n_outputs(), "{name}");
            assert_eq!(restored.best_iteration(), model.best_iteration(), "{name}");
            assert_eq!(restored.n_targets(), model.n_targets(), "{name}");
            assert_eq!(
                restored.num_parallel_tree(),
                model.num_parallel_tree(),
                "{name}"
            );
            assert_eq!(
                restored.has_vector_leaves(),
                model.has_vector_leaves(),
                "{name}"
            );
            assert_eq!(
                restored.objective_params(),
                model.objective_params(),
                "{name}"
            );
        }
    }
}
